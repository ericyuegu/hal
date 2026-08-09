"""Core contracts for the causal MTP and detached flow experiments."""

import importlib.util
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import cast

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
        temporal_d_model=32,
        temporal_layers=2,
        temporal_heads=4,
        temporal_ff_dim=64,
        classifier_chunk_tokens=128,
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
    assert ar.decoder_arch_version == 2
    assert ar.grad_accum_steps == 4 and ar.exec_horizon == 4
    assert (ar.temporal_d_model, ar.temporal_layers, ar.temporal_heads, ar.temporal_ff_dim) == (64, 1, 2, 128)
    assert ar.temporal_attn_chunk_sequences == 32_768
    assert ar.group_head_dim == 64
    assert ar.compile_trunk and ar.compile_temporal
    assert ar.wandb_log_code and ar.wandb_grad_every == 1024
    assert not ar.checkpoint_temporal and not ar.checkpoint_classifiers
    assert (ar.action_state_embed_dim, ar.char_dim, ar.stage_dim) == (48, 8, 4)
    assert (ar.main_stick_embed_dim, ar.c_stick_embed_dim, ar.trigger_embed_dim) == (40, 8, 8)
    assert (flow.flow_d_model, flow.flow_layers, flow.flow_steps) == (128, 2, 10)
    assert flow.max_steps == 16_384


def test_default_model_keeps_full_trunk_path_with_a_tiny_temporal_residual() -> None:
    cfg = exp024.TrainConfig()
    model = exp024.GPT(cfg)
    assert sum(parameter.numel() for parameter in model.parameters()) < 7_000_000
    assert sum(parameter.numel() for parameter in model.temporal.parameters()) < 350_000
    assert model.temporal.condition_in.weight.shape == (64, 256)
    assert model.temporal.post_cross_ff.up.weight.shape == (128, 64)
    assert model.temporal.group_condition["main_stick"].weight.shape == (128, 24)
    assert model.temporal.outputs["buttons"].up.weight.shape == (64, 64)
    assert model.temporal.outputs["buttons"].down.weight.shape == (256, 64)
    assert model.temporal.trunk_outputs["buttons"].weight.shape == (256, 256)
    assert not [
        name
        for name, _ in model.temporal.named_parameters()
        if "position" in name or "horizon" in name or "rotary" in name
    ]
    assert {name for name, _ in model.temporal.named_buffers() if "rotary" in name} == {
        "blocks.0.rotary.inv_freq",
        "trunk_cross_attention.rotary.inv_freq",
    }


def test_temporal_mtp_loss_and_ancestral_decode_smoke() -> None:
    cfg = _tiny_cfg(exp024)
    model = exp024.GPT(cfg)
    batch = _batch(exp024, cfg)
    parts = exp024.action_loss(model, batch)
    assert parts.nll.shape == (7, 20, 4)  # pads leave 4 + 3 valid context positions
    loss = exp024.objective(parts)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.temporal.blocks[-1].down.weight.grad is not None
    assert model.temporal.trunk_cross_attention.key_value.weight.grad is not None
    assert model.temporal.post_cross_ff.down.weight.grad is not None
    assert model.temporal.group_condition["main_stick"].weight.grad is not None
    assert model.temporal.outputs["main_stick"].up.weight.grad is not None
    assert model.temporal.trunk_outputs["buttons"].weight.grad is not None
    decoded = exp024.decode_chunk(model.eval(), batch.context, cfg.L_chunk, argmax=True)
    assert decoded.shape == (2, 20, A_DIM)
    assert ((decoded[..., 6:] == 0) | (decoded[..., 6:] == 1)).all()


def test_temporal_attention_chunks_only_the_independent_sequence_axis(monkeypatch) -> None:
    cfg = _tiny_cfg(exp024, temporal_attn_chunk_sequences=3)
    chunked = exp024.TemporalBlock(cfg).eval()
    reference = exp024.TemporalBlock(_tiny_cfg(exp024, temporal_attn_chunk_sequences=100)).eval()
    reference.load_state_dict(chunked.state_dict())
    x = torch.randn(7, cfg.L_chunk, cfg.temporal_d_model)
    expected = reference(x)

    original_sdpa = exp024.F.scaled_dot_product_attention
    launch_sizes: list[int] = []

    def recorded_sdpa(query, key, value, **kwargs):
        launch_sizes.append(query.shape[0])
        return original_sdpa(query, key, value, **kwargs)

    monkeypatch.setattr(exp024.F, "scaled_dot_product_attention", recorded_sdpa)
    actual = chunked(x)
    assert launch_sizes == [3, 3, 1]
    torch.testing.assert_close(actual, expected)


def test_temporal_attention_rejects_an_unsafe_cuda_launch_chunk() -> None:
    with pytest.raises(ValueError, match="65,535"):
        exp024.validate_config(_tiny_cfg(exp024, temporal_attn_chunk_sequences=65_536))


def test_temporal_chain_uses_previous_teacher_forced_frame() -> None:
    cfg = _tiny_cfg(exp024)
    model = exp024.GPT(cfg).eval()
    hidden = torch.randn(2, cfg.L_ctx, cfg.d_model)
    targets = torch.zeros(2, cfg.L_ctx, cfg.L_chunk, exp024.N_GROUPS, dtype=torch.long)
    changed = targets.clone()
    changed[:, :, 0, exp024.C_G] = 1
    original_logits = model.teacher_forced_logits(hidden, targets)
    changed_logits = model.teacher_forced_logits(hidden, changed)
    # Horizon 1 predicts the changed frame, so its first group cannot inspect that target.
    torch.testing.assert_close(original_logits[0]["c_stick"], changed_logits[0]["c_stick"])
    # The next group in the same frame is explicitly conditioned on the true C-stick.
    assert not torch.equal(original_logits[0]["triggers"], changed_logits[0]["triggers"])
    # Horizon 2 receives the full changed frame through the causal action chain.
    assert not torch.equal(original_logits[1]["c_stick"], changed_logits[1]["c_stick"])


def test_temporal_parallel_teacher_forcing_matches_cached_decode() -> None:
    cfg = _tiny_cfg(exp024)
    model = exp024.GPT(cfg).eval()
    gen = torch.Generator().manual_seed(11)
    hidden = torch.randn(2, cfg.L_ctx, cfg.d_model, generator=gen)
    targets = torch.stack(
        [torch.randint(vocab, (2, cfg.L_ctx, cfg.L_chunk), generator=gen) for vocab in exp024.GROUP_VOCABS],
        dim=-1,
    )
    parallel = model.temporal.teacher_forced_logits_by_group(hidden, targets)
    stepwise = model.temporal.forced_stepwise_logits(
        hidden,
        targets[:, -1],
        torch.zeros(2, dtype=torch.long),
    )
    for depth, logits in enumerate(stepwise):
        for name in exp024.GROUP_NAMES:
            torch.testing.assert_close(parallel[name][:, -1, depth], logits[name], atol=2e-5, rtol=2e-5)


def test_temporal_cross_attention_cannot_read_future_trunk_states() -> None:
    cfg = _tiny_cfg(exp024)
    model = exp024.GPT(cfg).eval()
    hidden = torch.randn(1, cfg.L_ctx, cfg.d_model)
    targets = torch.zeros(1, cfg.L_ctx, cfg.L_chunk, exp024.N_GROUPS, dtype=torch.long)
    changed = hidden.clone()
    changed[:, -1] += 100.0
    original = model.temporal.teacher_forced_logits_by_group(hidden, targets)
    future_changed = model.temporal.teacher_forced_logits_by_group(changed, targets)
    for name in exp024.GROUP_NAMES:
        # The penultimate prefix cannot attend the modified final trunk state.
        torch.testing.assert_close(original[name][:, -2], future_changed[name][:, -2])


def test_dense_padding_mask_matches_individually_stripped_decoder_calls() -> None:
    cfg = _tiny_cfg(exp024)
    model = exp024.GPT(cfg).eval()
    gen = torch.Generator().manual_seed(12)
    hidden = torch.randn(2, cfg.L_ctx, cfg.d_model, generator=gen)
    targets = torch.stack(
        [torch.randint(vocab, (2, cfg.L_ctx, cfg.L_chunk), generator=gen) for vocab in exp024.GROUP_VOCABS],
        dim=-1,
    )
    pads = torch.tensor([0, 2])
    positions = torch.arange(cfg.L_ctx)
    dense = model.temporal.teacher_forced_nll(hidden, targets, positions[None, :] >= pads[:, None])
    for row, pad in enumerate(pads.tolist()):
        stripped = model.temporal.teacher_forced_nll(
            hidden[row : row + 1, pad:],
            targets[row : row + 1, pad:],
        )
        torch.testing.assert_close(dense[row : row + 1, pad:], stripped, atol=2e-5, rtol=2e-5)


def test_chunk_targets_are_next_20_actions_at_every_prefix() -> None:
    cfg = _tiny_cfg(exp024)
    model = exp024.GPT(cfg)
    gen = torch.Generator().manual_seed(17)
    length = cfg.L_ctx + cfg.L_chunk
    indices = torch.stack(
        [torch.arange(length).remainder(vocab) for vocab in exp024.GROUP_VOCABS],
        dim=-1,
    )[None]
    native = exp024.dequantize(model, indices)
    features = _features(exp024, 1, cfg.L_ctx, gen)
    for channel, values in zip(ACTION_CHANNELS, native[:, : cfg.L_ctx].unbind(dim=-1), strict=True):
        features[f"ego_{channel}"] = values
    batch = TrainBatch(
        context=Context(features=features, ctx_pad=torch.zeros(1, dtype=torch.long)),
        target=native[:, cfg.L_ctx :],
    )
    targets, valid = exp024.chunk_targets(model, batch)
    assert valid.all()
    for prefix in range(cfg.L_ctx):
        torch.testing.assert_close(targets[0, prefix], indices[0, prefix + 1 : prefix + 21])


def test_device_batch_pipeline_preserves_cpu_batches_and_padding() -> None:
    cfg = _tiny_cfg(exp024)
    batches = [_batch(exp024, cfg, seed=seed) for seed in range(2)]
    outputs = list(exp024.device_batches(batches, "cpu", None))
    for actual, expected in zip(outputs, batches, strict=True):
        torch.testing.assert_close(actual.target, expected.target)
        torch.testing.assert_close(actual.context.ctx_pad, expected.context.ctx_pad)


def test_batch_geometry_rejects_variable_training_batch() -> None:
    cfg = _tiny_cfg(exp024)
    batch = _batch(exp024, cfg, batch=1)
    exp024.validate_batch_geometry(batch, cfg)
    with pytest.raises(ValueError, match="fixed training batch"):
        exp024.validate_batch_geometry(batch, cfg, expected_batch_size=2)


def test_validation_never_calls_fixed_shape_compiled_forward() -> None:
    cfg = _tiny_cfg(exp024)
    model = exp024.GPT(cfg).train()
    batch = _batch(exp024, cfg, batch=1)
    trunk_calls = temporal_calls = 0

    def fixed_training_forward(features, ctx_pad):
        nonlocal trunk_calls
        trunk_calls += 1
        assert ctx_pad.shape == (cfg.batch_size,)
        return type(model).forward(model, features, ctx_pad)

    def fixed_training_temporal(hidden, targets, valid_memory):
        nonlocal temporal_calls
        temporal_calls += 1
        assert hidden.shape[0] == cfg.batch_size
        return type(model.temporal).teacher_forced_nll(model.temporal, hidden, targets, valid_memory)

    model.forward = fixed_training_forward
    model.temporal.teacher_forced_nll = fixed_training_temporal
    metrics = exp024.val_metrics(model, [batch], cfg)
    assert trunk_calls == temporal_calls == 0
    assert model.__dict__["forward"] is fixed_training_forward
    assert model.temporal.__dict__["teacher_forced_nll"] is fixed_training_temporal
    assert model.training
    assert torch.isfinite(torch.tensor(metrics["loss"]))


def test_legacy_sequential_checkpoint_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="predates nonlinear causal decoder v2"):
        exp024.config_from_state({"d_model": 256, "action_embed_dim": 64})


def test_wandb_code_and_gradient_logging_are_bounded() -> None:
    captured: dict[str, object] = {}

    class Run:
        def log_code(self, **kwargs) -> None:
            captured.update(kwargs)

    exp024.log_wandb_code(Run())
    root = str(captured["root"])
    include = cast(Callable[[str, str], bool], captured["include_fn"])
    assert include(str(_EXP_DIR / "024_temporal_mtp.py"), root)
    assert not include(str(Path(root) / "data" / "replay.npy"), root)

    cfg = _tiny_cfg(exp024)
    model = exp024.GPT(cfg)
    loss = exp024.objective(exp024.action_loss(model, _batch(exp024, cfg)))
    loss.backward()
    payload = exp024.wandb_gradient_log(model, sample_limit=128)
    assert "gradients/decoder/post_cross_ff" in payload
    assert "gradient_norm/decoder/nonlinear_heads" in payload
    assert float(payload["gradient_norm/decoder/nonlinear_heads"]) > 0
    assert exp024.gradient_log_due(0, 0, 1024)
    assert not exp024.gradient_log_due(1, 0, 1024)
    assert exp024.gradient_log_due(1023, 0, 1024)


def test_synthetic_benchmark_context_has_production_geometry() -> None:
    cfg = _tiny_cfg(exp024)
    context = exp024.synthetic_context(cfg, 3, torch.device("cpu"))
    assert context.ctx_pad.shape == (3,)
    assert context.ctx_pad.eq(0).all()
    assert all(value.shape[:2] == (3, cfg.L_ctx) for value in context.features.values())


def test_flow_codec_preserves_native_analog_values() -> None:
    gen = torch.Generator().manual_seed(4)
    native = _native_actions(2, 20, gen)
    decoded = exp025.flow_decode(exp025.flow_encode(native), click_trigger_fix=False)
    torch.testing.assert_close(decoded, native)


def test_flow_validation_also_bypasses_compiled_ar_forward() -> None:
    cfg = _tiny_cfg(exp025, flow_steps=1)
    model = exp025.GPT(cfg).train()
    batch = _batch(exp025, cfg, batch=1)
    trunk_calls = temporal_calls = 0

    def fixed_training_forward(features, ctx_pad):
        nonlocal trunk_calls
        trunk_calls += 1
        assert ctx_pad.shape == (cfg.batch_size,)
        return type(model.ar).forward(model.ar, features, ctx_pad)

    def fixed_training_temporal(hidden, targets, valid_memory):
        nonlocal temporal_calls
        temporal_calls += 1
        assert hidden.shape[0] == cfg.batch_size
        return type(model.ar.temporal).teacher_forced_nll(model.ar.temporal, hidden, targets, valid_memory)

    model.ar.forward = fixed_training_forward
    model.ar.temporal.teacher_forced_nll = fixed_training_temporal
    metrics = exp025.val_metrics(model, [batch], cfg)
    assert trunk_calls == temporal_calls == 0
    assert model.ar.__dict__["forward"] is fixed_training_forward
    assert model.ar.temporal.__dict__["teacher_forced_nll"] is fixed_training_temporal
    assert model.training
    assert torch.isfinite(torch.tensor(metrics["ar_loss"]))


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
