"""019 predicts the four action groups as a chain inside one frame. These tests pin the
three properties that make that a clean A/B against the independent-head 016.

1. Only the head changed. The trunk parameters, the observation width and the hidden state
   are 016's with ``spatial_features`` off.
2. The chain starts AS 016. The conditioning tables are zero-initialized, so at step 0 the
   teacher-forced logits do not depend on the ancestors at all.
3. The totals stay comparable. Teacher forcing uses the ground-truth ids of the SAME target
   frame, so the summed per-group NLL is the chain-rule joint NLL of that frame.

Plus the deploy contract: the closed-loop path samples (never argmax), each draw conditions
the groups after it, and the decode hygiene (support mask, click=>trigger) still holds.
"""

import dataclasses
import importlib.util
import inspect
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from hal.data.feature_stats import FeatureStats
from hal.training import scoring
from hal.training.features import A_DIM
from hal.training.features import ACTION_CHANNELS
from hal.training.features import CAT_FEATURES
from hal.training.features import FLOAT_FEATURES
from hal.training.features import Context
from hal.training.features import TrainBatch

_EXP_DIR = Path(__file__).resolve().parent.parent.parent / "experiments"


def _load_experiment(filename: str):
    spec = importlib.util.spec_from_file_location(filename.split(".")[0], _EXP_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


exp016 = _load_experiment("016_spatial_features.py")
exp019 = _load_experiment("019_factored_frame.py")

_GROUPS = exp019._GROUP_NAMES
_G = exp019._GROUP_INDEX


def _tiny_cfg(exp, **kwargs):
    defaults = dict(d_model=64, n_layers=2, n_heads=2, L_ctx=16, head_offsets=(1, 2), batch_size=2, max_steps=8)
    return exp.TrainConfig(**{**defaults, **kwargs})


def _stats() -> dict[str, FeatureStats]:
    keys = (*FLOAT_FEATURES, *(f"nana_{k}" for k in FLOAT_FEATURES))
    return {k: FeatureStats(mean=0.0, std=1.0, min=-1.0, max=1.0) for k in keys}


def _features(exp, batch: int, length: int, gen: torch.Generator | None = None) -> dict[str, torch.Tensor]:
    """A synthetic observation batch: zeros, or plausible random values when ``gen`` is given."""

    def randn(*shape: int) -> torch.Tensor:
        return torch.zeros(*shape) if gen is None else torch.randn(*shape, generator=gen)

    features: dict[str, torch.Tensor] = {}
    for prefix in exp._PLAYER_PREFIXES:
        for feat in FLOAT_FEATURES:
            features[f"{prefix}_{feat}"] = randn(batch, length)
        for name, (vocab, _) in CAT_FEATURES.items():
            hi = 1 if gen is None else vocab
            features[f"{prefix}_{name}"] = torch.randint(0, hi, (batch, length), generator=gen)
    for channel in ACTION_CHANNELS:
        features[f"ego_{channel}"] = randn(batch, length)
    for key in ("ego_character", "opp_character", "stage"):
        features[key] = torch.randint(0, 1 if gen is None else 26, (batch, length), generator=gen)
    return features


def _context(exp, cfg, batch: int = 2, seed: int | None = 0) -> Context:
    gen = None if seed is None else torch.Generator().manual_seed(seed)
    pad = torch.zeros(batch, dtype=torch.long)
    return Context(features=_features(exp, batch, cfg.L_ctx, gen), ctx_pad=pad)


def _train_batch(exp, cfg, batch: int = 2, seed: int = 0) -> TrainBatch:
    gen = torch.Generator().manual_seed(seed)
    ctx = Context(features=_features(exp, batch, cfg.L_ctx, gen), ctx_pad=torch.tensor([0, 1] * (batch // 2)))
    target = torch.rand(batch, max(cfg.head_offsets), A_DIM, generator=gen) * 2 - 1
    return TrainBatch(context=ctx, target=target)


def _model(cfg, seed: int = 7):
    torch.manual_seed(seed)
    return exp019.GPT(cfg).eval()


def _gt_idx(model, shape: tuple[int, ...], seed: int) -> torch.Tensor:
    """Valid class ids per group, in ``_GROUP_NAMES`` order."""
    gen = torch.Generator().manual_seed(seed)
    cols = [torch.randint(0, vocab, shape, generator=gen) for vocab in model.group_vocabs]
    return torch.stack(cols, dim=-1)


def _randomize_conditioning(model, *, scale: float = 1.0, seed: int = 11) -> None:
    """Zero-init is the whole point at step 0; every later test needs a TRAINED-looking chain."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for head in model.heads:
            for table in head.emb.values():
                table.weight.copy_(torch.randn(table.weight.shape, generator=gen) * scale)


# --- the frozen recipe -------------------------------------------------------


def test_defaults_are_the_deployed_016_base_recipe() -> None:
    """Read back from the 016-base checkpoint cfg, not from 016's file defaults."""
    cfg = exp019.TrainConfig()
    assert (cfg.batch_size, cfg.max_steps) == (512, 16384)
    assert (cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.L_ctx) == (256, 8, 4, 256)
    assert (cfg.muon_lr, cfg.adam_lr) == (0.02, 8.5e-4)
    assert cfg.head_offsets == (1, 5, 9, 13)
    assert cfg.warmup_steps == 500 and cfg.val_every == 1024 and cfg.ckpt_every == 2048
    assert cfg.windows_per_replay == 4 and cfg.final_eval_n_matchups == 96
    assert cfg.chain_order == ("c_stick", "triggers", "main_stick", "buttons")
    assert cfg.main_stick_centers == "fine65"
    assert cfg.mds_schema_version == 5  # ranked-anonymized-1 is materialized at v5


def test_loader_kwargs_declare_the_mds_schema_version() -> None:
    """Both splits opt down to the dataset's version explicitly; the loader guard does the rest."""
    kwargs = exp019._loader_kwargs(exp019.TrainConfig(), _stats())
    assert kwargs["schema_version"] == 5


# --- only the head changed ---------------------------------------------------


def test_trunk_is_016_with_spatial_off() -> None:
    """Same input width, same trunk parameters from the same seed, same hidden state."""
    cfg16, cfg19 = _tiny_cfg(exp016, spatial_features=False), _tiny_cfg(exp019)
    torch.manual_seed(7)
    model16 = exp016.GPT(cfg16).eval()
    model19 = _model(cfg19)

    assert model19.ctx_proj.in_features == 374 == model16.ctx_proj.in_features
    trunk16 = {k: tuple(v.shape) for k, v in model16.state_dict().items() if not k.startswith("heads.")}
    trunk19 = {k: tuple(v.shape) for k, v in model19.state_dict().items() if not k.startswith("heads.")}
    assert trunk19 == trunk16
    features = _features(exp019, 2, cfg19.L_ctx)
    ctx_pad = torch.tensor([0, 1], dtype=torch.long)
    with torch.no_grad():
        torch.testing.assert_close(model16(features, ctx_pad), model19(features, ctx_pad), rtol=0, atol=0)


def test_head_keys_and_size_match_the_factored_design() -> None:
    """One projection per group (same total as 016's one 355-wide head) plus a conditioning
    table for every group except the last one in the chain."""
    cfg = _tiny_cfg(exp019)
    model = _model(cfg)
    expected = {f"heads.0.proj.{g}.{p}" for g in _GROUPS for p in ("weight", "bias")}
    expected |= {f"heads.0.emb.{g}.weight" for g in cfg.chain_order[:-1]}
    assert {k for k in model.state_dict() if k.startswith("heads.0.")} == expected

    model16 = exp016.GPT(_tiny_cfg(exp016, spatial_features=False))
    assert sum(p.numel() for p in model.heads[0].proj.parameters()) == sum(
        p.numel() for p in model16.heads[0].parameters()
    )
    assert sum(p.numel() for p in model.heads[0].emb.parameters()) == 99 * cfg.d_model  # 9 + 25 + 65


def test_main_stick_center_knob_drives_the_vocab_and_the_buffer() -> None:
    cfg = _tiny_cfg(exp019)
    model = _model(cfg)
    assert exp019.group_vocabs(cfg) == (256, 65, 9, 25) == tuple(model.group_vocabs)
    assert model.heads[0].proj["main_stick"].out_features == model.main_centers.shape[0]


# --- the chain starts as 016 -------------------------------------------------


def test_zero_init_head_ignores_its_ancestors() -> None:
    """At initialization the teacher-forced logits do not depend on ``gt_idx`` at all, and each
    group is exactly its own projection of ``h`` — i.e. this IS the independent-head model."""
    cfg = _tiny_cfg(exp019)
    model = _model(cfg)
    head = model.heads[0]
    h = torch.randn(3, 5, cfg.d_model, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        first = head.logits_tf(h, _gt_idx(model, (3, 5), seed=2))
        second = head.logits_tf(h, _gt_idx(model, (3, 5), seed=3))
        for name in _GROUPS:
            torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)
            torch.testing.assert_close(first[name], head.proj[name](h), rtol=0, atol=0)


def test_zero_init_summed_nll_equals_independent_per_group_nll() -> None:
    """The chain-rule total at init is the plain sum of four independent categorical NLLs."""
    cfg = _tiny_cfg(exp019)
    model = _model(cfg)
    head = model.heads[0]
    h = torch.randn(4, 6, cfg.d_model, generator=torch.Generator().manual_seed(4))
    tgt = _gt_idx(model, (4, 6), seed=5)
    valid = torch.ones(4, 6, dtype=torch.bool)
    with torch.no_grad():
        chained = exp019.group_nll(head.logits_tf(h, tgt), tgt, valid)
        for g, name in enumerate(_GROUPS):
            independent = F.cross_entropy(
                head.proj[name](h).reshape(-1, model.group_vocabs[g]), tgt[..., g].reshape(-1), reduction="none"
            )
            torch.testing.assert_close(chained[name], independent, rtol=0, atol=0)


# --- teacher forcing ---------------------------------------------------------


def test_perturbing_an_ancestor_moves_only_its_descendants() -> None:
    """Group order in the chain is causal: changing a ground-truth id must leave every group
    predicted BEFORE it bit-identical and move every group predicted after it."""
    cfg = _tiny_cfg(exp019)
    model = _model(cfg)
    _randomize_conditioning(model, scale=1.0)
    head = model.heads[0]
    h = torch.randn(2, 3, cfg.d_model, generator=torch.Generator().manual_seed(6))
    base_idx = _gt_idx(model, (2, 3), seed=7)
    with torch.no_grad():
        base = head.logits_tf(h, base_idx)
        for position, name in enumerate(cfg.chain_order[:-1]):
            g = _G[name]
            moved = base_idx.clone()
            moved[..., g] = (moved[..., g] + 1) % model.group_vocabs[g]
            other = head.logits_tf(h, moved)
            for ancestor in cfg.chain_order[: position + 1]:
                torch.testing.assert_close(base[ancestor], other[ancestor], rtol=0, atol=0)
            for descendant in cfg.chain_order[position + 1 :]:
                assert not torch.equal(base[descendant], other[descendant]), f"{descendant} ignored {name}"


def test_action_loss_is_bitwise_repeatable() -> None:
    """No RNG anywhere in the objective — dropout is off in eval mode and teacher forcing is
    ground truth, so two calls on one batch must agree exactly."""
    cfg = _tiny_cfg(exp019)
    model = _model(cfg)
    _randomize_conditioning(model)
    batch = _train_batch(exp019, cfg)
    first_nll, first_trans = exp019.action_loss(model, batch)
    second_nll, second_trans = exp019.action_loss(model, batch)
    assert set(first_nll) == set(second_nll)
    for key, value in first_nll.items():
        torch.testing.assert_close(value, second_nll[key], rtol=0, atol=0)
        torch.testing.assert_close(first_trans[key].float(), second_trans[key].float(), rtol=0, atol=0)


def test_summed_conditional_nll_is_the_chain_rule_joint() -> None:
    """The comparability claim: summing the four conditional NLLs (nats) reproduces −log of the
    joint probability obtained by walking the chain by hand."""
    cfg = _tiny_cfg(exp019)
    model = _model(cfg)
    _randomize_conditioning(model, scale=0.7)
    head = model.heads[0]
    h = torch.randn(1, cfg.d_model, generator=torch.Generator().manual_seed(8))
    idx = head.sample(
        h, group_temps=(1.0,) * 4, btn_dead=None, min_p=0.0, argmax=False, gen=torch.Generator().manual_seed(9)
    )
    with torch.no_grad():
        logits = head.logits_tf(h, idx)
        summed = sum(F.cross_entropy(logits[name], idx[..., _G[name]]) for name in _GROUPS).item()
        # Explicit product of conditionals along the path the sampler actually took.
        x, joint = h, 1.0
        for position, name in enumerate(head.chain_order):
            probs = F.softmax(head.proj[name](x), dim=-1)
            joint *= float(probs[0, int(idx[0, _G[name]])])
            if position + 1 < len(head.chain_order):
                x = x + head.emb[name](idx[..., _G[name]])
    assert summed == pytest.approx(-math.log(joint), rel=1e-5)


# --- decode ------------------------------------------------------------------


def test_argmax_decode_is_deterministic_and_idempotent() -> None:
    cfg = _tiny_cfg(exp019)
    model = _model(cfg)
    _randomize_conditioning(model)
    ctx = _context(exp019, cfg)
    first = exp019.decode(model, ctx, argmax=True)
    second = exp019.decode(model, ctx, argmax=True)
    assert first.shape == (2, 1, A_DIM)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_seeded_sampling_is_repeatable_and_seed_dependent() -> None:
    cfg = _tiny_cfg(exp019)
    model = _model(cfg)
    _randomize_conditioning(model)
    ctx = _context(exp019, cfg, batch=8)
    same = [exp019.decode(model, ctx, gen=torch.Generator().manual_seed(3)) for _ in range(2)]
    torch.testing.assert_close(same[0], same[1], rtol=0, atol=0)
    other = exp019.decode(model, ctx, gen=torch.Generator().manual_seed(4))
    assert not torch.equal(same[0], other), "sampling ignored the generator"


def test_button_support_mask_forces_the_only_supported_combo() -> None:
    """A support floor that leaves exactly one combo alive must be respected by the chain
    sampler, not just by a greedy pick."""
    cfg = _tiny_cfg(exp019)
    model = _model(cfg)
    _randomize_conditioning(model)
    combo = 0b0100_0011  # A + B + the digital Z bit: several bits set, so it is not the zero combo
    model.button_combo_counts.fill_(0)
    model.button_combo_counts[combo] = 1000
    expected = scoring.combo_to_buttons(torch.tensor(combo))
    ctx = _context(exp019, cfg, batch=8)
    for kwargs in ({"gen": torch.Generator().manual_seed(5)}, {"argmax": True}):
        action = exp019.decode(model, ctx, btn_support_min=1000, **kwargs)
        torch.testing.assert_close(action[..., exp019._N_CONT :], expected.expand(8, 1, -1), rtol=0, atol=0)


def test_click_trigger_fix_holds_on_sampled_output() -> None:
    """When the drawn combo sets the digital L/R bit, the analog trigger must read fully pressed."""
    cfg = _tiny_cfg(exp019)
    model = _model(cfg)
    _randomize_conditioning(model)
    l_bit, r_bit = exp019._BUTTON_L_CH - exp019._N_CONT, exp019._BUTTON_R_CH - exp019._N_CONT
    combo = (1 << l_bit) | (1 << r_bit)
    model.button_combo_counts.fill_(0)
    model.button_combo_counts[combo] = 1000
    ctx = _context(exp019, cfg, batch=8)
    action = exp019.decode(
        model, ctx, btn_support_min=1000, click_trigger_fix=True, gen=torch.Generator().manual_seed(6)
    )
    assert torch.all(action[..., exp019._BUTTON_L_CH] > 0.5) and torch.all(action[..., exp019._BUTTON_R_CH] > 0.5)
    assert torch.all(action[..., exp019._TRIGGER_L_CH] == 1.0)
    assert torch.all(action[..., exp019._TRIGGER_R_CH] == 1.0)


def test_sampled_ancestors_condition_the_groups_after_them() -> None:
    """Sharpen every projection so each conditional is effectively one-hot. The draw is then
    reproducible from the chain itself: each group's id is the argmax of its projection applied
    to the hidden state PLUS the conditioning of the ids drawn before it."""
    cfg = _tiny_cfg(exp019)
    model = _model(cfg)
    _randomize_conditioning(model, scale=1.0)
    head = model.heads[0]
    with torch.no_grad():
        for layer in head.proj.values():
            layer.weight.mul_(60.0)
    h = torch.randn(4, cfg.d_model, generator=torch.Generator().manual_seed(10))
    drawn = head.sample(
        h, group_temps=(1.0,) * 4, btn_dead=None, min_p=0.0, argmax=False, gen=torch.Generator().manual_seed(12)
    )
    with torch.no_grad():
        x = h
        for position, name in enumerate(head.chain_order):
            assert torch.equal(head.proj[name](x).argmax(-1), drawn[..., _G[name]]), name
            if position + 1 < len(head.chain_order):
                x = x + head.emb[name](drawn[..., _G[name]])
        # Independent-head decoding would have produced different descendants from this hidden state.
        tail = head.chain_order[-1]
        assert not torch.equal(head.proj[tail](h).argmax(-1), drawn[..., _G[tail]])


def test_closed_loop_decode_never_uses_argmax() -> None:
    """Greedy decode collapses the closed-loop policy to doing nothing, so the deployed path has
    no argmax knob at all and the policy's draws must move with its seed."""
    assert "argmax" not in {f.name for f in dataclasses.fields(exp019.DecodeSettings)}
    for fn in (exp019.decode, exp019.decode_chunk):
        assert inspect.signature(fn).parameters["argmax"].default is False
    # The head's own sampler takes no default at all: a caller must ask for greedy in writing.
    sample_argmax = inspect.signature(exp019.FactoredHead.sample).parameters["argmax"]
    assert sample_argmax.kind is inspect.Parameter.KEYWORD_ONLY and sample_argmax.default is inspect.Parameter.empty
    cfg = _tiny_cfg(exp019)
    model = _model(cfg)
    _randomize_conditioning(model)
    ctx = _context(exp019, cfg, batch=8)
    chunks = [
        exp019.make_policy(model, _stats(), cfg, device="cpu", decode_seed=seed).predict_chunk(ctx, None)
        for seed in (0, 1)
    ]
    assert not (chunks[0] == chunks[1]).all(), "policy decoded greedily"


# --- configuration -----------------------------------------------------------


@pytest.mark.parametrize(
    "chain_order",
    [
        ("c_stick", "triggers", "main_stick"),  # too short
        ("buttons", "buttons", "main_stick", "c_stick"),  # duplicate
        ("c_stick", "triggers", "main_stick", "shoulder"),  # unknown group
    ],
)
def test_validate_config_rejects_a_bad_chain_order(chain_order: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="permutation"):
        exp019.validate_config(_tiny_cfg(exp019, chain_order=chain_order), has_button_combo_counts=False)


def test_validate_config_rejects_an_unregistered_center_table() -> None:
    with pytest.raises(ValueError, match="main_stick_centers"):
        exp019.validate_config(_tiny_cfg(exp019, main_stick_centers="coarse21"), has_button_combo_counts=False)


def test_model_rejects_a_bad_chain_order_too() -> None:
    """The model never trusts a validated config to have run first."""
    with pytest.raises(ValueError, match="permutation"):
        exp019.GPT(_tiny_cfg(exp019, chain_order=("buttons", "main_stick", "c_stick", "c_stick")))


# --- training smoke ----------------------------------------------------------


def test_objective_decreases_on_a_fixed_batch() -> None:
    """Overfit sanity on the real optimizer: the chain trains, and the conditioning tables leave
    zero (so the run genuinely departs from the independent-head baseline)."""
    cfg = _tiny_cfg(exp019)
    torch.manual_seed(0)
    model = exp019.GPT(cfg)
    model.train()
    opt = exp019.make_optimizer(model, cfg)
    batch = _train_batch(exp019, cfg, batch=4)
    losses = []
    for _ in range(20):
        opt.zero_grad()
        nll, trans = exp019.action_loss(model, batch)
        obj = exp019.objective(nll, trans, cfg.aux_loss_weight, cfg.transition_loss_weight)
        obj.backward()
        opt.step()
        losses.append(obj.item())
    assert losses[-1] < losses[0], losses
    assert any(float(table.weight.detach().abs().sum()) > 0 for table in model.heads[0].emb.values())


# --- validation metrics ------------------------------------------------------


def _val_batch(cfg, *, trigger: float, batch: int = 2) -> TrainBatch:
    """A batch whose ego trigger history AND target both sit at ``trigger``, so every
    ground-truth trigger id in the offset-1 window is the same class."""
    gen = torch.Generator().manual_seed(13)
    features = _features(exp019, batch, cfg.L_ctx, gen)
    for channel in ("trigger_l", "trigger_r"):
        features[f"ego_{channel}"] = torch.full((batch, cfg.L_ctx), trigger)
    ctx = Context(features=features, ctx_pad=torch.zeros(batch, dtype=torch.long))
    target = torch.rand(batch, max(cfg.head_offsets), A_DIM, generator=gen) * 2 - 1
    target[..., exp019._TRIGGER_L_CH] = trigger
    target[..., exp019._TRIGGER_R_CH] = trigger
    return TrainBatch(context=ctx, target=target)


def _always_click_l_and_r(model) -> None:
    """Make the buttons conditional put all of its mass on the L+R click combo."""
    combo = (1 << (exp019._BUTTON_L_CH - exp019._N_CONT)) | (1 << (exp019._BUTTON_R_CH - exp019._N_CONT))
    with torch.no_grad():
        for head in model.heads:
            head.proj["buttons"].weight.zero_()
            head.proj["buttons"].bias.fill_(-50.0)
            head.proj["buttons"].bias[combo] = 50.0


def test_val_metrics_keep_the_016_surface_and_drop_the_spatial_probe() -> None:
    cfg = _tiny_cfg(exp019)
    model = _model(cfg)
    _randomize_conditioning(model)
    out = exp019.val_metrics(model, [_train_batch(exp019, cfg, batch=4)], cfg)
    for key in ("loss", "nll_off1", "nll_off2", "btn_logloss", "ablate_hist_kl", "ablate_hist_dnll"):
        assert key in out
    for name in _GROUPS:
        assert f"nll_{name}" in out and f"brier_{name}" in out and f"changeF1_{name}" in out
    assert not [key for key in out if "spatial" in key]
    # The offset-1 total IS the chain-rule joint, so it is the number that compares to 016.
    assert out["loss"] == pytest.approx(sum(out[f"nll_{name}"] for name in _GROUPS), rel=1e-6)


def test_click_trigger_invalid_mass_reads_the_true_trigger() -> None:
    """The conditional version of the metric: a head that always clicks L and R scores ~1 when the
    ground-truth trigger of the SAME frame is released, and ~0 when it is fully pressed."""
    cfg = _tiny_cfg(exp019)
    model = _model(cfg)
    _randomize_conditioning(model)
    _always_click_l_and_r(model)
    released = exp019.val_metrics(model, [_val_batch(cfg, trigger=0.0)], cfg)
    pressed = exp019.val_metrics(model, [_val_batch(cfg, trigger=1.0)], cfg)
    assert released["click_trigger_invalid_l_mass"] == pytest.approx(1.0, abs=1e-4)
    assert released["click_trigger_invalid_r_mass"] == pytest.approx(1.0, abs=1e-4)
    assert released["click_trigger_invalid_mass"] == pytest.approx(1.0, abs=1e-4)
    assert pressed["click_trigger_invalid_mass"] == pytest.approx(0.0, abs=1e-6)


def test_recon_metrics_run_on_both_decode_modes() -> None:
    cfg = _tiny_cfg(exp019)
    model = _model(cfg)
    _randomize_conditioning(model)
    cache = [_train_batch(exp019, cfg, batch=4)]
    for kwargs in ({"argmax": True}, {"argmax": False, "gen": torch.Generator().manual_seed(14)}):
        out = exp019.recon_metrics(model, cache, **kwargs)
        assert 0.0 <= out["recon_button_acc"] <= 1.0 and out["recon_cont_mae"] >= 0.0
