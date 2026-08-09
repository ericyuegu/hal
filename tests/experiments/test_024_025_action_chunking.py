"""Core contracts for the sequential MTP and detached flow experiments."""

import importlib.util
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from hal.training.features import A_DIM
from hal.training.features import ACTION_CHANNELS
from hal.training.features import CAT_FEATURES
from hal.training.features import FLOAT_FEATURES
from hal.training.features import Context
from hal.training.features import TrainBatch

_EXP_DIR = Path(__file__).resolve().parents[2] / "experiments"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _EXP_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


exp024 = _load("test_exp024", "024_temporal_mtp.py")
exp025 = _load("test_exp025", "025_mtp_flow.py")


def _tiny_cfg(exp, **overrides):
    values = dict(
        d_model=32,
        n_layers=1,
        n_heads=4,
        L_ctx=4,
        action_embed_dim=8,
        action_mlp_ratio=2,
        batch_size=2,
        grad_accum_steps=1,
        reservoir_capacity=4,
        warmup_steps=1,
        max_steps=4,
        compile_trunk=False,
        num_workers=0,
        push_to_r2=False,
    )
    if exp is exp025:
        values.update(flow_d_model=32, flow_layers=1, flow_heads=4, flow_ff_dim=64, flow_time_dim=32)
    return exp.TrainConfig(**{**values, **overrides})


def _features(exp, batch: int, length: int, gen: torch.Generator) -> dict[str, torch.Tensor]:
    features: dict[str, torch.Tensor] = {}
    for prefix in exp.mtp._PLAYER_PREFIXES if exp is exp025 else exp._PLAYER_PREFIXES:
        for name in FLOAT_FEATURES:
            features[f"{prefix}_{name}"] = torch.randn(batch, length, generator=gen)
        for name, (vocab, _) in CAT_FEATURES.items():
            features[f"{prefix}_{name}"] = torch.randint(0, vocab, (batch, length), generator=gen)
    history = _native_actions(batch, length, gen)
    for index, channel in enumerate(ACTION_CHANNELS):
        features[f"ego_{channel}"] = history[..., index]
    features["ego_character"] = torch.randint(0, 26, (batch, length), generator=gen)
    features["opp_character"] = torch.randint(0, 26, (batch, length), generator=gen)
    features["stage"] = torch.randint(0, 26, (batch, length), generator=gen)
    return features


def _native_actions(batch: int, length: int, gen: torch.Generator) -> torch.Tensor:
    actions = torch.empty(batch, length, A_DIM)
    actions[..., :4] = torch.rand(batch, length, 4, generator=gen) * 2.0 - 1.0
    actions[..., 4:6] = torch.rand(batch, length, 2, generator=gen)
    actions[..., 6:] = torch.randint(0, 2, (batch, length, 8), generator=gen).float()
    return actions


def _batch(exp, cfg, *, batch: int = 2, seed: int = 0) -> TrainBatch:
    gen = torch.Generator().manual_seed(seed)
    context = Context(
        features=_features(exp, batch, cfg.L_ctx, gen),
        ctx_pad=torch.tensor([0, 1][:batch], dtype=torch.long),
    )
    return TrainBatch(context=context, target=_native_actions(batch, cfg.L_chunk, gen))


def test_defaults_pin_the_requested_geometry() -> None:
    ar = exp024.TrainConfig()
    flow = exp025.TrainConfig()
    assert (ar.L_ctx, ar.batch_size, ar.d_model, ar.n_layers, ar.L_chunk) == (256, 512, 256, 8, 20)
    assert ar.grad_accum_steps == 4 and ar.exec_horizon == 4
    assert (flow.flow_d_model, flow.flow_layers, flow.flow_steps) == (128, 2, 10)
    assert flow.max_steps == 16_384


def test_temporal_mtp_loss_and_ancestral_decode_smoke() -> None:
    cfg = _tiny_cfg(exp024)
    model = exp024.GPT(cfg)
    batch = _batch(exp024, cfg)
    parts = exp024.action_loss(model, batch)
    assert parts.nll.shape == (7, 20, 4)  # pads leave 4 + 3 valid context positions
    loss = exp024.objective(parts)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.temporal_heads[-1].down.weight.grad is not None
    decoded = exp024.decode_chunk(model.eval(), batch.context, cfg.L_chunk, argmax=True)
    assert decoded.shape == (2, 20, A_DIM)
    assert ((decoded[..., 6:] == 0) | (decoded[..., 6:] == 1)).all()


def test_temporal_chain_uses_previous_teacher_forced_frame() -> None:
    cfg = _tiny_cfg(exp024)
    model = exp024.GPT(cfg).eval()
    hidden = torch.randn(2, cfg.d_model)
    targets = torch.zeros(2, cfg.L_chunk, exp024.N_GROUPS, dtype=torch.long)
    changed = targets.clone()
    changed[:, 0, exp024.C_G] = 1
    original_logits = model.teacher_forced_logits(hidden, targets)
    changed_logits = model.teacher_forced_logits(hidden, changed)
    # Head 1 predicts the changed frame, so its first group cannot inspect that target.
    torch.testing.assert_close(original_logits[0]["c_stick"], changed_logits[0]["c_stick"])
    # Head 2 receives the full changed frame embedding through the fixed-width chain.
    assert not torch.equal(original_logits[1]["c_stick"], changed_logits[1]["c_stick"])


def test_flow_codec_preserves_native_analog_values() -> None:
    gen = torch.Generator().manual_seed(4)
    native = _native_actions(2, 20, gen)
    decoded = exp025.flow_decode(exp025.flow_encode(native), click_trigger_fix=False)
    torch.testing.assert_close(decoded, native)


def test_flow_loss_is_stopped_at_the_ar_model() -> None:
    cfg = _tiny_cfg(exp025)
    model = exp025.GPT(cfg)
    batch = _batch(exp025, cfg)
    hidden = model(batch.context.features, batch.context.ctx_pad)
    loss = exp025.flow_matching_loss(model, batch, hidden, cfg).squared_error.mean()
    loss.backward()
    assert all(parameter.grad is None for parameter in model.ar.parameters())
    assert model.flow.action_out.weight.grad is not None
    assert model.flow.action_out.weight.grad.abs().sum() > 0


def test_flow_warm_start_strictly_copies_experiment_024(tmp_path: Path) -> None:
    ar_cfg = _tiny_cfg(exp024)
    source = exp024.GPT(ar_cfg)
    checkpoint = tmp_path / "024.pt"
    torch.save({"cfg": asdict(ar_cfg), "model": source.state_dict()}, checkpoint)
    flow_cfg = _tiny_cfg(exp025)
    destination = exp025.GPT(flow_cfg)
    exp025.warm_start_ar(destination, flow_cfg, str(checkpoint))
    for name, value in source.state_dict().items():
        torch.testing.assert_close(destination.ar.state_dict()[name], value)
    assert flow_cfg.init_ar_checkpoint == str(checkpoint.resolve())
    assert len(flow_cfg.init_ar_sha256) == 64


@pytest.mark.parametrize("steps", [1, 10])
def test_flow_integration_returns_legal_controller_chunks(steps: int) -> None:
    cfg = _tiny_cfg(exp025, flow_steps=steps)
    model = exp025.GPT(cfg).eval()
    batch = _batch(exp025, cfg)
    hidden = model(batch.context.features, batch.context.ctx_pad)
    decoded = exp025.integrate_chunk(model, hidden, cfg, gen=torch.Generator().manual_seed(8))
    assert decoded.shape == (2, 20, A_DIM)
    assert ((decoded[..., :4] >= -1) & (decoded[..., :4] <= 1)).all()
    assert ((decoded[..., 4:6] >= 0) & (decoded[..., 4:6] <= 1)).all()
    assert ((decoded[..., 6:] == 0) | (decoded[..., 6:] == 1)).all()
