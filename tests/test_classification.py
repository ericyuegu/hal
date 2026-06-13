"""Correctness contracts for the classification action-chunk policy (experiment 005).

Mirrors ``test_multi_position.py``: the backbone invariants (finite differentiable loss
into the backbone, leak-free per-position targets, a strictly causal backbone) must hold
exactly as in 003 since the backbone is shared; plus the head-specific contracts — the
decode produces a valid raw action vector for every (button_head × continuous_head ×
decode-mode) combination, and single_label never emits more than one button. The
experiment is loaded by path since its filename starts with a digit."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

_EXP_PATH = Path(__file__).resolve().parent.parent / "experiments" / "005_classification.py"


def _load_experiment():
    spec = importlib.util.spec_from_file_location("exp005", _EXP_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


exp = _load_experiment()

_COMBOS = [(bh, ch) for bh in ("combo", "multi_label", "single_label") for ch in ("naive_bins", "stick_clusters")]


def _cfg(**overrides):
    base = dict(
        d_model=16,
        n_layers=1,
        n_heads=2,
        dim_feedforward=16,
        dropout=0.0,
        d_head=8,
        n_head_layers=1,
        head_heads=2,
        head_ff=16,
        n_bins=5,
        L_ctx=6,
        L_chunk=3,
    )
    base.update(overrides)
    return exp.TrainConfig(**base)


def _features(B: int, T: int, *, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    feats: dict[str, torch.Tensor] = {}
    for prefix in ("ego", "opp"):
        for f in exp.FLOAT_FEATURES:
            feats[f"{prefix}_{f}"] = torch.randn(B, T, generator=g)
        for cat, (vocab, _) in exp.PLAYER_CAT_FEATURES.items():
            feats[f"{prefix}_{cat}"] = torch.randint(0, vocab, (B, T), generator=g)
        feats[f"{prefix}_character"] = torch.randint(0, 26, (B, T), generator=g)
    feats["stage"] = torch.randint(0, 32, (B, T), generator=g)
    # ego controller history in the real action ranges so discretization is exercised
    for i, ch in enumerate(exp.ACTION_CHANNELS):
        if i < 4:
            feats[f"ego_{ch}"] = torch.empty(B, T).uniform_(-1, 1, generator=g)
        elif i < 6:
            feats[f"ego_{ch}"] = torch.empty(B, T).uniform_(0, 1, generator=g)
        else:
            feats[f"ego_{ch}"] = (torch.rand(B, T, generator=g) > 0.5).float()
    return feats


def _batch(B: int, cfg, *, ctx_pad: list[int] | None = None, seed: int = 0):
    feats = _features(B, cfg.L_ctx, seed=seed)
    pad = torch.zeros(B, dtype=torch.int64) if ctx_pad is None else torch.tensor(ctx_pad, dtype=torch.int64)
    ctx = exp.Context(features=feats, ctx_pad=pad)
    g = torch.Generator().manual_seed(seed + 1)
    cont = torch.cat(
        [
            torch.empty(B, cfg.L_chunk, 4).uniform_(-1, 1, generator=g),
            torch.empty(B, cfg.L_chunk, 2).uniform_(0, 1, generator=g),
        ],
        dim=-1,
    )
    btn = (torch.rand(B, cfg.L_chunk, exp.A_DIM - 6, generator=g) > 0.5).float()
    target = torch.cat([cont, btn], dim=-1)
    return ctx, exp.TrainBatch(context=ctx, target=target)


# --- acceptance #1: finite loss backprops into the backbone ------------------
@pytest.mark.parametrize("button_head,continuous_head", _COMBOS)
def test_loss_is_finite_and_backprops_to_backbone(button_head, continuous_head):
    cfg = _cfg(button_head=button_head, continuous_head=continuous_head)
    torch.manual_seed(0)
    model = exp.ClassifierPolicy(cfg)
    _, batch = _batch(2, cfg)
    comps = exp.action_loss(model, batch)
    loss = sum(comps.values()).mean()
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    grad = model.ctx_proj.weight.grad  # backbone input projection
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    enc_grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    assert enc_grads and any(g.abs().sum() > 0 for g in enc_grads)


def test_loss_components_are_the_four_modalities():
    cfg = _cfg()
    model = exp.ClassifierPolicy(cfg)
    _, batch = _batch(2, cfg)
    comps = exp.action_loss(model, batch)
    assert set(comps) == {"main_stick", "c_stick", "triggers", "buttons"}


def test_cluster_head_shapes_track_the_scoring_discretizers():
    """Cluster-mode heads size to the scoring centers, not n_bins (trig is 1D centers now)."""
    from hal.training import scoring

    cfg = _cfg(continuous_head="stick_clusters", n_bins=21)  # n_bins != len(TRIGGER_CENTERS)
    model = exp.ClassifierPolicy(cfg)
    assert model.main_head.out_features == scoring.STICK_CLUSTER_CENTERS_MAIN.shape[0]
    assert model.c_head.out_features == scoring.STICK_CLUSTER_CENTERS_C.shape[0]
    assert model.trig_head.out_features == 2 * scoring.TRIGGER_CENTERS.shape[0]
    _, batch = _batch(2, cfg)
    logits, _ = exp._select(model, batch, multi=False)
    N = logits["trig"].shape[0]
    assert logits["main"].shape == (N, cfg.L_chunk, scoring.STICK_CLUSTER_CENTERS_MAIN.shape[0])
    assert logits["c"].shape == (N, cfg.L_chunk, scoring.STICK_CLUSTER_CENTERS_C.shape[0])
    assert logits["trig"].shape == (N, cfg.L_chunk, 2, scoring.TRIGGER_CENTERS.shape[0])


# --- acceptance #2: leak-free per-position targets (shared with 003) ---------
def test_position_targets_are_the_next_H_actions_and_mask_padded_tail():
    cfg = _cfg(L_ctx=6, L_chunk=3)
    H = cfg.L_chunk
    ctx, batch = _batch(2, cfg, ctx_pad=[0, 2])
    tgt, valid = exp._position_targets(ctx, batch.target, H)
    a_full = torch.cat([exp.stack_actions(ctx.features), batch.target], dim=1)
    T = cfg.L_ctx
    assert tgt.shape == (2, T, H, exp.A_DIM)
    for i in range(T):
        assert torch.equal(tgt[:, i], a_full[:, i + 1 : i + 1 + H])
    assert torch.equal(tgt[:, -1], batch.target)
    assert valid[0].all()
    assert not valid[1, :2].any() and valid[1, 2:].all()


# --- acceptance #3: strictly causal backbone --------------------------------
def test_backbone_is_causal_autograd():
    cfg = _cfg()
    torch.manual_seed(0)
    model = exp.ClassifierPolicy(cfg).eval()
    ctx, _ = _batch(1, cfg)
    x = ctx.features["ego_main_stick_x"].clone().requires_grad_(True)
    feats = dict(ctx.features)
    feats["ego_main_stick_x"] = x
    hidden = model.encode_context(exp.Context(features=feats, ctx_pad=ctx.ctx_pad))
    i = cfg.L_ctx // 2
    hidden[0, i].sum().backward()
    g = x.grad[0]
    assert torch.all(g[i + 1 :] == 0), "hidden[i] leaked from inputs at j > i"
    assert g[: i + 1].abs().sum() > 0


def test_perturbing_future_actions_leaves_earlier_hidden_unchanged():
    cfg = _cfg()
    torch.manual_seed(0)
    model = exp.ClassifierPolicy(cfg).eval()
    ctx, _ = _batch(2, cfg)
    k = cfg.L_ctx // 2
    with torch.no_grad():
        h1 = model.encode_context(ctx)
        feats2 = {key: v.clone() for key, v in ctx.features.items()}
        for ch in exp.ACTION_CHANNELS:
            feats2[f"ego_{ch}"][:, k:] += 3.0
        h2 = model.encode_context(exp.Context(features=feats2, ctx_pad=ctx.ctx_pad))
    assert torch.allclose(h1[:, :k], h2[:, :k], atol=1e-5)


# --- acceptance #4: decode contract -----------------------------------------
@pytest.mark.parametrize("button_head,continuous_head", _COMBOS)
@pytest.mark.parametrize("mode", ["argmax", "sample"])
def test_decode_returns_valid_raw_action_vector(button_head, continuous_head, mode):
    cfg = _cfg(button_head=button_head, continuous_head=continuous_head)
    torch.manual_seed(0)
    model = exp.ClassifierPolicy(cfg).eval()
    ctx, _ = _batch(2, cfg)
    gen = torch.Generator().manual_seed(0)
    a = exp.decode(model, ctx, mode=mode, gen=gen)
    assert a.shape == (2, cfg.L_chunk, exp.A_DIM)
    assert not a.requires_grad
    assert a[..., 0:4].abs().max() <= 1.0  # sticks in [-1, 1]
    assert a[..., 4:6].min() >= 0.0 and a[..., 4:6].max() <= 1.0  # triggers in [0, 1]
    btn = a[..., 6:]
    assert torch.isin(btn, torch.tensor([0.0, 1.0])).all()  # buttons binary


@pytest.mark.parametrize(
    "bad", [dict(decode="agrmax"), dict(n_bins=0), dict(decode_temp=0.0), dict(norm_div=0.0), dict(button_head="x")]
)
def test_invalid_config_is_rejected(bad):
    with pytest.raises(ValueError):
        exp.ClassifierPolicy(_cfg(**bad))


def test_decode_rejects_unknown_mode():
    model = exp.ClassifierPolicy(_cfg()).eval()
    ctx, _ = _batch(2, _cfg())
    with pytest.raises(ValueError):
        exp.decode(model, ctx, mode="greedy")


@pytest.mark.parametrize("mode", ["argmax", "sample"])
def test_single_label_emits_at_most_one_button(mode):
    cfg = _cfg(button_head="single_label")
    torch.manual_seed(0)
    model = exp.ClassifierPolicy(cfg).eval()
    ctx, _ = _batch(4, cfg)
    gen = torch.Generator().manual_seed(0)
    a = exp.decode(model, ctx, mode=mode, gen=gen)
    assert a[..., 6:].sum(-1).max() <= 1.0


# --- combo head: bit-order agreement with the controller bridge --------------
def test_default_button_head_is_combo():
    assert exp.TrainConfig().button_head == "combo"


def test_combo_head_is_256_way():
    from hal.training import scoring

    cfg = _cfg(button_head="combo")
    model = exp.ClassifierPolicy(cfg)
    assert model.button_head.out_features == scoring.N_BUTTON_COMBOS == 256
    _, batch = _batch(2, cfg)
    logits, _ = exp._select(model, batch, multi=False)
    assert logits["buttons"].shape[-1] == 256


def test_combo_bit_order_matches_action_vec_button_slots():
    """The combo id's bit k must be action-vec slot ``6 + k`` — i.e. the ``_BUTTON_ORDER``
    button decoded by ``action_vec_to_controller``. A single press on button ``_BUTTON_ORDER[k]``
    must round-trip through pack→unpack to a one-hot at the same channel, and decode that combo
    id to a controller pressing exactly that button. Get this wrong and controller outputs scramble."""
    from hal.training import features
    from hal.training import scoring
    from hal.wire import BUTTON_BITS

    assert features._BUTTON_ORDER == ("a", "b", "x", "y", "z", "r", "l", "d_up")
    assert tuple(features.ACTION_CHANNELS[6:]) == tuple(f"button_{n}" for n in features._BUTTON_ORDER)

    for k, name in enumerate(features._BUTTON_ORDER):
        bits = torch.zeros(8)
        bits[k] = 1.0
        combo = scoring.buttons_to_combo(bits)
        assert combo.item() == (1 << k), f"bit {k} ({name}) does not map to combo id 2**{k}"
        # the full 14-dim action vec (sticks/triggers neutral, this one button bit set)
        a = torch.zeros(exp.A_DIM)
        a[6 + k] = 1.0
        ctrl = features.action_vec_to_controller(a.numpy())
        assert ctrl.buttons == BUTTON_BITS[name], f"channel {6 + k} decodes to the wrong button for {name}"
        # decode(combo id) → bits → the same single channel
        assert torch.equal(scoring.combo_to_buttons(combo), bits)


def test_combo_train_path_loss_and_decode_smoke():
    """A synthetic TrainBatch runs action_loss + decode in combo mode end-to-end (the train
    path the trainer drives)."""
    from hal.training import scoring

    cfg = _cfg(button_head="combo")
    torch.manual_seed(0)
    model = exp.ClassifierPolicy(cfg)
    _, batch = _batch(3, cfg)
    comps = exp.action_loss(model, batch)
    assert set(comps) == {"main_stick", "c_stick", "triggers", "buttons"}
    loss = sum(comps.values()).mean()
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert model.button_head.weight.grad is not None and model.button_head.weight.grad.abs().sum() > 0
    model.eval()
    gen = torch.Generator().manual_seed(0)
    a = exp.decode(model, batch.context, mode="sample", gen=gen)
    btn = a[..., 6:]
    assert torch.isin(btn, torch.tensor([0.0, 1.0])).all()
    # combo decode emits a coherent co-press set (any subset of the 8), never a partial-bit value
    assert torch.equal(scoring.combo_to_buttons(scoring.buttons_to_combo(btn)), btn)


@pytest.mark.parametrize("button_head,continuous_head", _COMBOS)
def test_val_metrics_emits_proper_scores(button_head, continuous_head):
    cfg = _cfg(button_head=button_head, continuous_head=continuous_head)
    torch.manual_seed(0)
    model = exp.ClassifierPolicy(cfg)
    val_cache = [_batch(2, cfg, seed=s)[1] for s in range(3)]
    m = exp.val_metrics(model, val_cache, cfg)
    assert m["action_nll_bits_per_frame"] > 0
    assert {"buttons/logloss_bits", "buttons/brier", "buttons/multipress_rate"} <= set(m)
    assert 0.0 <= m["buttons/multipress_rate"] <= 1.0
    if continuous_head == "naive_bins":
        assert "cont_density_bits_per_dim" in m
    else:
        assert "cont_discrete_bits" in m
    assert model.training  # restored after eval


def test_decode_argmax_is_deterministic():
    cfg = _cfg()
    torch.manual_seed(0)
    model = exp.ClassifierPolicy(cfg).eval()
    ctx, _ = _batch(2, cfg)
    a1 = exp.decode(model, ctx, mode="argmax")
    a2 = exp.decode(model, ctx, mode="argmax")
    assert torch.equal(a1, a2)


def test_default_decode_is_sampling():
    assert exp.TrainConfig().decode == "sample"


def test_make_policy_samples_by_default_and_respects_overrides():
    cfg = _cfg()  # decode defaults to "sample"
    torch.manual_seed(0)
    model = exp.ClassifierPolicy(cfg).eval()
    ctx, _ = _batch(3, cfg)
    # default policy samples → successive replans draw different actions (closed-loop stochasticity)
    pol = exp.make_policy(model, {}, cfg, device="cpu")
    a1, a2 = pol.predict_chunk(ctx, None), pol.predict_chunk(ctx, None)
    assert a1.shape == (3, cfg.L_chunk, exp.A_DIM)
    assert not np.allclose(a1, a2)
    # argmax override → deterministic
    greedy = exp.make_policy(model, {}, cfg, device="cpu", decode_mode="argmax")
    assert np.allclose(greedy.predict_chunk(ctx, None), greedy.predict_chunk(ctx, None))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
