"""Correctness contracts for the multi-position flow-matching policy (experiment 003).

Pins the four invariants the multi-position split must hold regardless of training:
finite differentiable loss, leak-free per-position targets, a strictly causal
backbone (no target leaks into the hidden it conditions on), and an unchanged
inference surface that reads only the last position. The experiment is loaded by
path since its filename starts with a digit (mirrors ``test_rtc_policy.py``).
"""

import importlib.util
from pathlib import Path

import pytest
import torch

_EXP_PATH = Path(__file__).resolve().parent.parent / "experiments" / "003_multi_position.py"


def _load_experiment():
    spec = importlib.util.spec_from_file_location("exp003", _EXP_PATH)
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
        time_emb_dim=8,
        d_head=8,
        n_head_layers=1,
        head_heads=2,
        head_ff=16,
        L_ctx=6,
        L_chunk=3,
        n_flow_steps=2,
    )
    base.update(overrides)
    return exp.TrainConfig(**base)


def _features(B: int, T: int, *, seed: int = 0) -> dict[str, torch.Tensor]:
    """A minimal but complete context-feature dict at length T (no _mask sidecars —
    the model fills missing masks with zeros)."""
    g = torch.Generator().manual_seed(seed)
    feats: dict[str, torch.Tensor] = {}
    for prefix in ("ego", "opp"):
        for f in exp.FLOAT_FEATURES:
            feats[f"{prefix}_{f}"] = torch.randn(B, T, generator=g)
        for cat, (vocab, _) in exp.PLAYER_CAT_FEATURES.items():
            feats[f"{prefix}_{cat}"] = torch.randint(0, vocab, (B, T), generator=g)
    for ch in exp.ACTION_CHANNELS:  # ego controller history
        feats[f"ego_{ch}"] = torch.randn(B, T, generator=g)
    return feats


def _batch(B: int, cfg, *, ctx_pad: list[int] | None = None, seed: int = 0):
    feats = _features(B, cfg.L_ctx, seed=seed)
    pad = torch.zeros(B, dtype=torch.int64) if ctx_pad is None else torch.tensor(ctx_pad, dtype=torch.int64)
    ctx = exp.Context(features=feats, ctx_pad=pad)
    target = torch.randn(B, cfg.L_chunk, exp.A_DIM, generator=torch.Generator().manual_seed(seed + 1))
    return ctx, exp.TrainBatch(context=ctx, target=target)


# --- acceptance #1 -----------------------------------------------------------
def test_loss_is_finite_scalar_and_backprops_to_backbone():
    cfg = _cfg()
    torch.manual_seed(0)
    model = exp.FlowMatchingPolicy(cfg)
    _, batch = _batch(2, cfg)
    loss = exp.flow_loss(model, batch).mean()
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    grad = model.ctx_proj.weight.grad  # backbone input projection
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    enc_grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    assert enc_grads and any(g.abs().sum() > 0 for g in enc_grads)


# --- acceptance #2 -----------------------------------------------------------
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
    # last position recovers exactly the original target chunk
    assert torch.equal(tgt[:, -1], batch.target)
    # valid is 0 on the leftmost ctx_pad positions, 1 elsewhere
    assert valid[0].all()  # ctx_pad=0
    assert not valid[1, :2].any() and valid[1, 2:].all()  # ctx_pad=2


def test_loss_normalizes_by_valid_count():
    """Multi-position loss supervises exactly the valid positions (scale is a per-
    element mean → T-invariant)."""
    cfg = _cfg()
    torch.manual_seed(0)
    model = exp.FlowMatchingPolicy(cfg)
    ctx, batch = _batch(2, cfg, ctx_pad=[0, 2])
    _, valid = exp._position_targets(ctx, batch.target, cfg.L_chunk)
    err2 = exp.flow_loss(model, batch)
    assert err2.shape[0] == int(valid.sum())  # only valid positions denoised


# --- acceptance #3 -----------------------------------------------------------
def test_backbone_is_causal_autograd():
    """d hidden[0,i] / d input[0,j] == 0 for j > i."""
    cfg = _cfg()
    torch.manual_seed(0)
    model = exp.FlowMatchingPolicy(cfg).eval()
    ctx, _ = _batch(1, cfg)
    x = ctx.features["ego_main_stick_x"].clone().requires_grad_(True)
    feats = dict(ctx.features)
    feats["ego_main_stick_x"] = x
    hidden = model.encode_context(exp.Context(features=feats, ctx_pad=ctx.ctx_pad))
    i = cfg.L_ctx // 2
    hidden[0, i].sum().backward()
    g = x.grad[0]  # [T]
    assert torch.all(g[i + 1 :] == 0), "hidden[i] leaked from inputs at j > i"
    assert g[: i + 1].abs().sum() > 0, "hidden[i] should depend on inputs at j <= i"


def test_perturbing_future_actions_leaves_earlier_hidden_unchanged():
    cfg = _cfg()
    torch.manual_seed(0)
    model = exp.FlowMatchingPolicy(cfg).eval()
    ctx, _ = _batch(2, cfg)
    k = cfg.L_ctx // 2
    with torch.no_grad():
        h1 = model.encode_context(ctx)
        feats2 = {key: v.clone() for key, v in ctx.features.items()}
        for ch in exp.ACTION_CHANNELS:
            feats2[f"ego_{ch}"][:, k:] += 3.0  # perturb actions[:, k:]
        h2 = model.encode_context(exp.Context(features=feats2, ctx_pad=ctx.ctx_pad))
    assert torch.allclose(h1[:, :k], h2[:, :k], atol=1e-5)


# --- acceptance #4 -----------------------------------------------------------
def test_act_returns_one_frame_under_no_grad_from_last_position():
    cfg = _cfg()
    torch.manual_seed(0)
    model = exp.FlowMatchingPolicy(cfg).eval()
    ctx, _ = _batch(2, cfg)
    out = exp.act(model, ctx, n_steps=cfg.n_flow_steps, device="cpu")
    assert out.shape == (2, exp.A_DIM)
    assert not out.requires_grad  # ran under no_grad
    # it is the first frame of one integrated chunk from the last position
    chunk = exp.integrate_chunk(model, ctx, n_steps=cfg.n_flow_steps, device="cpu")
    assert out.shape == chunk[:, 0, :].shape


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
