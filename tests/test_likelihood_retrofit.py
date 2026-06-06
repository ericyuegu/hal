"""Smoke contract for the PF-ODE + MC-button likelihood retrofit on the flow experiments.

The PF-ODE divergence integral is easy to get wrong (time direction, sign, grad-in-no-grad);
``test_scoring.py`` pins it against a closed form, and this exercises it through the REAL
flow models (001 unified, 003 multi-position) end-to-end on synthetic data: the metric must
run and produce finite bits/dim and a non-negative Bernoulli log-loss for both architectures."""

import importlib.util
from pathlib import Path

import pytest
import torch

from hal.training.features import A_DIM
from hal.training.features import ACTION_CHANNELS
from hal.training.features import CAT_FEATURES
from hal.training.features import FLOAT_FEATURES
from hal.training.features import Context
from hal.training.features import TrainBatch

_EXP_DIR = Path(__file__).resolve().parent.parent / "experiments"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), _EXP_DIR / name)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _batch(L_ctx: int, L_chunk: int, *, B: int = 2, seed: int = 0) -> TrainBatch:
    g = torch.Generator().manual_seed(seed)
    feats: dict[str, torch.Tensor] = {}
    for prefix in ("ego", "opp"):
        for f in FLOAT_FEATURES:
            feats[f"{prefix}_{f}"] = torch.randn(B, L_ctx, generator=g)
        for cat, (vocab, _) in CAT_FEATURES.items():
            feats[f"{prefix}_{cat}"] = torch.randint(0, vocab, (B, L_ctx), generator=g)
    for i, ch in enumerate(ACTION_CHANNELS):
        feats[f"ego_{ch}"] = (
            torch.randn(B, L_ctx, generator=g) if i < 6 else (torch.rand(B, L_ctx, generator=g) > 0.5).float()
        )
    ctx = Context(features=feats, ctx_pad=torch.zeros(B, dtype=torch.int64))
    gt = torch.Generator().manual_seed(seed + 1)
    cont = torch.randn(B, L_chunk, 6, generator=gt)  # flow handles any real-valued continuous
    btn = (torch.rand(B, L_chunk, A_DIM - 6, generator=gt) > 0.5).float()  # buttons must be {0,1}
    return TrainBatch(context=ctx, target=torch.cat([cont, btn], dim=-1))


_FLOW_CFGS = {
    "001_flow_matching_baseline.py": dict(
        d_model=16,
        n_layers=1,
        n_heads=2,
        dim_feedforward=16,
        dropout=0.0,
        time_emb_dim=8,
        L_ctx=6,
        L_chunk=3,
        n_flow_steps=2,
    ),
    "003_multi_position.py": dict(
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
    ),
}


@pytest.mark.parametrize("name", list(_FLOW_CFGS))
def test_likelihood_metrics_runs_through_real_flow_model(name):
    exp = _load(name)
    cfg = exp.TrainConfig(**_FLOW_CFGS[name])
    torch.manual_seed(0)
    model = exp.FlowMatchingPolicy(cfg).to(exp.DEVICE)
    val_cache = [_batch(cfg.L_ctx, cfg.L_chunk, seed=s).to(exp.DEVICE) for s in range(2)]
    m = exp.likelihood_metrics(model, val_cache, cfg, n_ode_steps=4, n_mc=3, max_batches=2)
    assert set(m) == {"bits_per_dim", "buttons/logloss_bits", "buttons/brier"}
    assert all(torch.isfinite(torch.tensor(v)) for v in m.values())
    assert m["buttons/logloss_bits"] >= 0.0 and 0.0 <= m["buttons/brier"] <= 1.0
    assert model.training  # eval state restored
    # fixed-seed PF-ODE probes + MC integration → metric is reproducible across val calls
    m2 = exp.likelihood_metrics(model, val_cache, cfg, n_ode_steps=4, n_mc=3, max_batches=2)
    assert m == m2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
