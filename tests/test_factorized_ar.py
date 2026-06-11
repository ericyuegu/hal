"""Correctness contracts for the factorized autoregressive action-chunk policy (experiment 007).

Mirrors ``test_classification.py``: the shared backbone invariants (finite differentiable loss,
leak-free per-position targets, a strictly causal backbone) must hold exactly as in 005 since the
backbone is copied verbatim; plus the head-specific AR contracts unique to 007 —

* the group quantize/dequantize round-trips exactly on values that already sit on the center
  grids (sticks → joint clusters, triggers → joint L×R of the 1D centers, buttons → 256-way combo);
* teacher-forced loss is finite and backprops into both the backbone and the AR head (BOS,
  group input tables, output linears);
* **the autoregression property** (the whole point of 007): group-g logits at frame k are
  INVARIANT to perturbing teacher-forced sub-tokens at LATER positions in the 64-stream, and
  CHANGE when an EARLIER sub-token is perturbed — both directions;
* decode emits a valid raw action vector (sticks [-1,1], triggers [0,1], buttons {0,1}) for both
  modes, argmax is deterministic, and a checkpoint save→load reproduces an identical argmax decode.

The experiment is loaded by path since its filename starts with a digit."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

_EXP_PATH = Path(__file__).resolve().parent.parent / "experiments" / "007_factorized_ar.py"


def _load_experiment():
    spec = importlib.util.spec_from_file_location("exp007", _EXP_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


exp = _load_experiment()


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
        for cat, (vocab, _) in exp.CAT_FEATURES.items():
            feats[f"{prefix}_{cat}"] = torch.randint(0, vocab, (B, T), generator=g)
        feats[f"{prefix}_character"] = torch.randint(0, 26, (B, T), generator=g)
        # opp controller history in the real ranges so the opp_controller cheat path is exercised
        for i, ch in enumerate(exp.ACTION_CHANNELS):
            if i < 4:
                feats[f"{prefix}_{ch}"] = torch.empty(B, T).uniform_(-1, 1, generator=g)
            elif i < 6:
                feats[f"{prefix}_{ch}"] = torch.empty(B, T).uniform_(0, 1, generator=g)
            else:
                feats[f"{prefix}_{ch}"] = (torch.rand(B, T, generator=g) > 0.5).float()
    feats["stage"] = torch.randint(0, 32, (B, T), generator=g)
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


# --- group quantize/dequantize round-trip ------------------------------------
def test_group_quantize_dequantize_round_trips_on_grid_values():
    """Actions whose channels already sit on the center grids reproduce EXACTLY through
    quantize→dequantize, and the class indices are stable under a re-quantize. This is the
    target/decode contract: the supervised class and the decoded action agree byte-for-byte."""
    from hal.training import scoring

    m_c, c_c, t_c = (
        scoring.STICK_CLUSTER_CENTERS_MAIN,
        scoring.STICK_CLUSTER_CENTERS_C,
        scoring.TRIGGER_CENTERS,
    )
    g = torch.Generator().manual_seed(0)
    N = 256
    mi = torch.randint(0, m_c.shape[0], (N,), generator=g)
    ci = torch.randint(0, c_c.shape[0], (N,), generator=g)
    tli = torch.randint(0, t_c.shape[0], (N,), generator=g)
    tri = torch.randint(0, t_c.shape[0], (N,), generator=g)
    bi = torch.randint(0, scoring.N_BUTTON_COMBOS, (N,), generator=g)
    a = torch.cat(
        [
            scoring.cluster_to_xy(mi, m_c),
            scoring.cluster_to_xy(ci, c_c),
            torch.stack([scoring.center_to_value(tli, t_c), scoring.center_to_value(tri, t_c)], dim=-1),
            scoring.combo_to_buttons(bi),
        ],
        dim=-1,
    )
    idx = exp.quantize_groups(m_c, c_c, t_c, a)
    a2 = exp.dequantize_groups(m_c, c_c, t_c, idx)
    assert torch.allclose(a, a2, atol=1e-6), "on-grid action did not round-trip through quantize/dequantize"
    assert torch.equal(idx, exp.quantize_groups(m_c, c_c, t_c, a2)), "class indices not idempotent"
    # the joint trigger index unpacks correctly to the two per-shoulder centers
    assert torch.equal(idx[:, exp._TRIG_G], tli * t_c.shape[0] + tri)


def test_group_vocabs_match_the_scoring_discretizers():
    from hal.training import scoring

    assert exp._GROUP_NAMES == ("buttons", "main_stick", "c_stick", "triggers")
    expected = (
        scoring.N_BUTTON_COMBOS,
        scoring.STICK_CLUSTER_CENTERS_MAIN.shape[0],
        scoring.STICK_CLUSTER_CENTERS_C.shape[0],
        scoring.TRIGGER_CENTERS.shape[0] ** 2,
    )
    assert expected == exp._GROUP_VOCABS
    assert exp.N_GROUPS == 4


# --- acceptance #1: finite loss backprops into backbone AND the AR head -------
def test_loss_is_finite_and_backprops_to_backbone_and_head():
    cfg = _cfg()
    torch.manual_seed(0)
    model = exp.FactorizedARPolicy(cfg)
    _, batch = _batch(2, cfg)
    comps = exp.action_loss(model, batch)
    loss = sum(comps.values()).mean()
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    grad = model.ctx_proj.weight.grad  # backbone input projection
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    enc_grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    assert enc_grads and any(g.abs().sum() > 0 for g in enc_grads)
    # AR head: BOS, group input tables, and the per-group output linears all receive gradient
    assert model.bos.grad is not None and model.bos.grad.abs().sum() > 0
    assert all(emb.weight.grad is not None and emb.weight.grad.abs().sum() > 0 for emb in model.group_in_embeds)
    assert all(lin.weight.grad is not None and lin.weight.grad.abs().sum() > 0 for lin in model.group_out)


def test_loss_components_are_the_four_groups():
    cfg = _cfg()
    model = exp.FactorizedARPolicy(cfg)
    _, batch = _batch(2, cfg)
    comps = exp.action_loss(model, batch)
    assert set(comps) == {"main_stick", "c_stick", "triggers", "buttons"}


def test_opp_controller_cheat_widens_context_and_trains():
    """The roofline cheat concats the opp's 14-channel controller history → a wider ctx_proj input,
    and the model still trains end-to-end. ``-oppc`` shows up in the model tag."""
    base = exp.FactorizedARPolicy(_cfg(opp_controller=False))
    cheat = exp.FactorizedARPolicy(_cfg(opp_controller=True))
    assert cheat.ctx_proj.in_features == base.ctx_proj.in_features + exp.A_DIM
    assert "-oppc" in exp._model_tag(_cfg(opp_controller=True))
    assert "-oppc" not in exp._model_tag(_cfg(opp_controller=False))
    torch.manual_seed(0)
    _, batch = _batch(2, _cfg(opp_controller=True))
    loss = sum(exp.action_loss(cheat, batch).values()).mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert cheat.ctx_proj.weight.grad.abs().sum() > 0


# --- acceptance #2: leak-free per-position targets (shared with 005) ----------
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


# --- acceptance #3: strictly causal backbone (shared with 005) ----------------
def test_backbone_is_causal_autograd():
    cfg = _cfg()
    torch.manual_seed(0)
    model = exp.FactorizedARPolicy(cfg).eval()
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


# --- acceptance #4: THE autoregression property -------------------------------
def _logits_at(model, cond, tgt_idx, k: int, gg: int) -> torch.Tensor:
    with torch.no_grad():
        out = model.teacher_forced_logits(cond, tgt_idx)
    return out[:, k, gg, : exp._GROUP_VOCABS[gg]].clone()


def test_logits_invariant_to_later_subtokens_and_change_with_earlier():
    """Group-g logits at frame k must depend ONLY on sub-tokens earlier than (k, g) in the
    64-position stream: perturbing a LATER teacher-forced sub-token leaves them unchanged
    (the causal-AR guarantee), and perturbing an EARLIER one changes them (conditioning is real,
    not decorative). Probed at an interior stream position so both a predecessor and a successor
    exist."""
    cfg = _cfg()
    torch.manual_seed(0)
    model = exp.FactorizedARPolicy(cfg).eval()
    N, H, G = 4, cfg.L_chunk, exp.N_GROUPS
    cond = torch.randn(N, cfg.d_model)
    gen = torch.Generator().manual_seed(1)
    tgt = torch.stack([torch.randint(0, v, (N, H), generator=gen) for v in exp._GROUP_VOCABS], dim=-1)  # [N,H,G]

    k, gg = 1, 1  # main_stick at frame 1 → interior stream position k*G+gg
    flat = k * G + gg
    base = _logits_at(model, cond, tgt, k, gg)

    # perturb a strictly LATER sub-token in the stream → logits at (k,gg) unchanged
    lk, lg = divmod(flat + 2, G)
    tgt_later = tgt.clone()
    tgt_later[:, lk, lg] = (tgt_later[:, lk, lg] + 1) % exp._GROUP_VOCABS[lg]
    assert torch.allclose(base, _logits_at(model, cond, tgt_later, k, gg), atol=1e-5), (
        "logits at (k,g) leaked from a LATER sub-token — head is not causal"
    )

    # perturb a strictly EARLIER sub-token → logits at (k,gg) must change
    ek, eg = divmod(flat - 1, G)
    tgt_earlier = tgt.clone()
    tgt_earlier[:, ek, eg] = (tgt_earlier[:, ek, eg] + 1) % exp._GROUP_VOCABS[eg]
    assert not torch.allclose(base, _logits_at(model, cond, tgt_earlier, k, gg), atol=1e-5), (
        "logits at (k,g) did not respond to an EARLIER sub-token — conditioning is dead"
    )


def test_first_frame_buttons_depend_only_on_context():
    """The very first stream sub-token (frame 0, buttons) has no predecessor: its logits are a
    function of the backbone cond alone, so they are invariant to EVERY teacher-forced target."""
    cfg = _cfg()
    torch.manual_seed(0)
    model = exp.FactorizedARPolicy(cfg).eval()
    N = 3
    cond = torch.randn(N, cfg.d_model)
    gen = torch.Generator().manual_seed(2)
    tgt = torch.stack([torch.randint(0, v, (N, cfg.L_chunk), generator=gen) for v in exp._GROUP_VOCABS], dim=-1)
    base = _logits_at(model, cond, tgt, 0, exp._BUTTONS_G)
    tgt2 = (tgt + 1) % torch.tensor(exp._GROUP_VOCABS)  # perturb literally everything
    assert torch.allclose(base, _logits_at(model, cond, tgt2, 0, exp._BUTTONS_G), atol=1e-5)


# --- acceptance #5: decode contract -------------------------------------------
@pytest.mark.parametrize("mode", ["argmax", "sample"])
def test_decode_returns_valid_raw_action_vector(mode):
    cfg = _cfg()
    torch.manual_seed(0)
    model = exp.FactorizedARPolicy(cfg).eval()
    ctx, _ = _batch(2, cfg)
    gen = torch.Generator().manual_seed(0)
    a = exp.decode(model, ctx, mode=mode, gen=gen)
    assert a.shape == (2, cfg.L_chunk, exp.A_DIM)
    assert not a.requires_grad
    assert a[..., 0:4].abs().max() <= 1.0  # sticks in [-1, 1]
    assert a[..., 4:6].min() >= 0.0 and a[..., 4:6].max() <= 1.0  # triggers in [0, 1]
    btn = a[..., 6:]
    assert torch.isin(btn, torch.tensor([0.0, 1.0])).all()  # buttons binary
    # buttons are a coherent combo (any subset of the 8), never a partial-bit value
    from hal.training import scoring

    assert torch.equal(scoring.combo_to_buttons(scoring.buttons_to_combo(btn)), btn)


def test_decode_argmax_is_deterministic():
    cfg = _cfg()
    torch.manual_seed(0)
    model = exp.FactorizedARPolicy(cfg).eval()
    ctx, _ = _batch(2, cfg)
    a1 = exp.decode(model, ctx, mode="argmax")
    a2 = exp.decode(model, ctx, mode="argmax")
    assert torch.equal(a1, a2)


def test_decode_matches_teacher_forced_argmax_under_self_feeding():
    """Sanity that the sequential decoder agrees with the one-shot teacher-forced head: greedily
    feeding the head's own argmax back in must reproduce the per-position argmax of a teacher-forced
    pass on exactly those realized classes (the AR consistency the decode loop implements)."""
    cfg = _cfg()
    torch.manual_seed(0)
    model = exp.FactorizedARPolicy(cfg).eval()
    ctx, _ = _batch(2, cfg)
    a = exp.decode(model, ctx, mode="argmax")  # [B, H, 14]
    idx = exp.quantize_groups(model.main_centers, model.c_centers, model.trig_centers, a)  # [B,H,G]
    cond = model.encode_context(ctx)[:, -1, :]
    with torch.no_grad():
        tf = model.teacher_forced_logits(cond, idx)  # [B,H,G,maxV]
    for g in range(exp.N_GROUPS):
        assert torch.equal(tf[:, :, g, : exp._GROUP_VOCABS[g]].argmax(-1), idx[:, :, g])


@pytest.mark.parametrize("bad", [dict(decode="agrmax"), dict(decode_temp=0.0), dict(norm_div=0.0)])
def test_invalid_config_is_rejected(bad):
    with pytest.raises(ValueError):
        exp.FactorizedARPolicy(_cfg(**bad))


def test_decode_rejects_unknown_mode():
    model = exp.FactorizedARPolicy(_cfg()).eval()
    ctx, _ = _batch(2, _cfg())
    with pytest.raises(ValueError):
        exp.decode(model, ctx, mode="greedy")


# --- defaults + policy + metrics ----------------------------------------------
def test_default_decode_is_sampling():
    assert exp.TrainConfig().decode == "sample"


def test_make_policy_samples_by_default_and_respects_overrides():
    cfg = _cfg()  # decode defaults to "sample"
    torch.manual_seed(0)
    model = exp.FactorizedARPolicy(cfg).eval()
    ctx, _ = _batch(3, cfg)
    pol = exp.make_policy(model, {}, cfg, device="cpu")
    a1, a2 = pol.predict_chunk(ctx, None), pol.predict_chunk(ctx, None)
    assert a1.shape == (3, cfg.L_chunk, exp.A_DIM)
    assert not np.allclose(a1, a2)  # sampling → successive replans differ (closed-loop stochasticity)
    greedy = exp.make_policy(model, {}, cfg, device="cpu", decode_mode="argmax")
    assert np.allclose(greedy.predict_chunk(ctx, None), greedy.predict_chunk(ctx, None))


def test_val_metrics_emits_proper_scores():
    cfg = _cfg()
    torch.manual_seed(0)
    model = exp.FactorizedARPolicy(cfg)
    val_cache = [_batch(2, cfg, seed=s)[1] for s in range(3)]
    m = exp.val_metrics(model, val_cache, cfg)
    assert m["action_nll_bits_per_frame"] > 0
    assert {"buttons/logloss_bits", "buttons/brier", "buttons/multipress_rate"} <= set(m)
    assert 0.0 <= m["buttons/multipress_rate"] <= 1.0
    assert "cont_discrete_bits" in m
    assert set(exp._GROUP_NAMES) == {k.split("/")[-1] for k in m if k.startswith("loss/modality/")}
    assert model.training  # restored after eval


# --- checkpoint round-trip ----------------------------------------------------
def test_checkpoint_save_load_reproduces_argmax_decode(tmp_path):
    cfg = _cfg()
    torch.manual_seed(0)
    model = exp.FactorizedARPolicy(cfg).eval()
    ctx, _ = _batch(2, cfg)
    before = exp.decode(model, ctx, mode="argmax")
    from dataclasses import asdict

    path = tmp_path / "ckpt.pt"
    torch.save({"cfg": asdict(cfg), "model": model.state_dict(), "step": 0}, path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    reloaded = exp.FactorizedARPolicy(exp.TrainConfig(**state["cfg"]))
    reloaded.load_state_dict(state["model"])
    reloaded.eval()
    after = exp.decode(reloaded, ctx, mode="argmax")
    assert torch.equal(before, after)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
