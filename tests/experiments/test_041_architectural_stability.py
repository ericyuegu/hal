"""Contracts for the experiment 041 architecture stabilization treatment."""

import importlib.util
import math
import sys
import threading
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn


def _load(name: str, filename: str) -> ModuleType:
    """Load a numeric experiment module by path."""
    path = Path(__file__).resolve().parents[2] / "experiments" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


exp = _load("test_exp041", "041_architectural_stability.py")


def _cfg(**overrides) -> exp.TrainConfig:
    arch_values = {
        **asdict(exp.ARCHITECTURE),
        "d_model": 32,
        "n_layers": 1,
        "n_heads": 4,
        "L_ctx": 4,
        "temporal_d_model": 32,
        "temporal_layers": 2,
        "temporal_heads": 4,
        "temporal_ff_dim": 64,
        "group_head_dim": 64,
        "value_hidden_dim": 16,
        "item_type_dim": 6,
        "item_state_dim": 3,
        "item_hidden_dim": 8,
        "item_dim": 5,
    }
    values = {
        "batch_size": 2,
        "target_loss_positions": 16,
        "warmup_fraction": 0.5,
        "stable_fraction": 0.75,
        "compile_trunk": False,
        "compile_temporal": False,
        "num_workers": 0,
        "push_to_r2": False,
        "inference_mode": "eager",
        "wandb_log_code": False,
    }
    for name, value in overrides.items():
        if name in arch_values:
            arch_values[name] = value
        else:
            values[name] = value
    return exp.TrainConfig(
        arch=exp.Architecture(**arch_values),
        awr=exp.AWRCalibration(),
        **values,
    )


def _indices(cfg: exp.TrainConfig, *shape: int) -> torch.Tensor:
    return torch.zeros(*shape, exp.N_GROUPS, dtype=torch.long)


def test_action_heads_are_normalized_linear_readouts_without_skip() -> None:
    cfg = _cfg()
    decoder = exp.GPT(cfg).temporal

    assert not hasattr(decoder, "trunk_outputs")
    assert all(isinstance(head, exp.LinearActionHead) for head in decoder.outputs.values())
    assert all(head.output.bias is None for head in decoder.outputs.values())

    head = decoder.outputs["buttons"]
    features = torch.randn(3, 5, cfg.arch.temporal_d_model) * 100
    logits, normalized = head.forward_with_input(features)

    assert logits.shape == (3, 5, exp.GROUP_VOCABS[exp.BUTTONS_G])
    torch.testing.assert_close(
        normalized.float().square().mean(dim=-1).sqrt(),
        torch.ones(3, 5),
        atol=1e-5,
        rtol=1e-5,
    )


def test_group_conditioning_is_bounded_and_identity_initialized() -> None:
    cfg = _cfg()
    decoder = exp.GPT(cfg).temporal
    states = torch.randn(2, 3, cfg.arch.temporal_d_model)
    embedded = {name: torch.randn(2, 3, cfg.arch.action_embed_dim) for name in exp.GROUP_ORDER}

    for condition in decoder.group_condition.values():
        torch.testing.assert_close(condition.weight, torch.zeros_like(condition.weight))
        torch.testing.assert_close(condition.bias, torch.zeros_like(condition.bias))
    torch.testing.assert_close(decoder.group_features(states, "buttons", embedded), states)

    condition = decoder.group_condition["buttons"]
    with torch.no_grad():
        condition.bias[: decoder.d_model].fill_(100.0)
        condition.bias[decoder.d_model :].fill_(100.0)
    conditioned = decoder.group_features(states, "buttons", embedded)
    torch.testing.assert_close(conditioned, 1.5 * states + 1.0)


class _ZeroBranch(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(inputs)


class _OneBranch(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(inputs)


def test_temporal_mlp_residual_uses_the_depth_scale() -> None:
    block = exp.TemporalBlock(_cfg()).eval()
    block.proj = _ZeroBranch()
    block.mlp = _OneBranch()
    inputs = torch.randn(3, 4, block.d_model)

    output = block(inputs)

    torch.testing.assert_close(output, inputs + block.scale)


class _CaptureBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: torch.Tensor | None = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.inputs = inputs.detach()
        return inputs


def test_temporal_training_input_is_normalized_before_block_zero() -> None:
    cfg = _cfg(temporal_layers=1)
    decoder = exp.GPT(cfg).temporal
    capture = _CaptureBlock()
    decoder.blocks = nn.ModuleList([capture])
    hidden = torch.randn(cfg.batch_size, cfg.arch.L_ctx, cfg.arch.d_model) * 20
    observed = _indices(cfg, cfg.batch_size, cfg.arch.L_ctx)
    targets = _indices(cfg, cfg.batch_size, cfg.arch.L_ctx, len(cfg.arch.head_offsets))

    decoder.teacher_forced_states(hidden, observed, targets)

    assert capture.inputs is not None
    input_rms = capture.inputs.float().square().mean(dim=-1).sqrt()
    torch.testing.assert_close(input_rms, torch.ones_like(input_rms), atol=1e-4, rtol=1e-4)


def test_value_features_are_normalized_and_detached_from_policy_state() -> None:
    cfg = _cfg()
    model = exp.GPT(cfg)
    hidden = torch.randn(cfg.batch_size, cfg.arch.L_ctx, cfg.arch.d_model, requires_grad=True)

    value_features = exp.decoder_rmsnorm(hidden).detach()
    model.value_head(value_features.float()).sum().backward()

    assert hidden.grad is None
    assert all(parameter.grad is not None for parameter in model.value_head.parameters())


def test_button_forward_diagnostics_cover_stable_head_boundary() -> None:
    cfg = _cfg()
    model = exp.GPT(cfg)
    hidden = torch.randn(cfg.batch_size, cfg.arch.L_ctx, cfg.arch.d_model)
    observed = _indices(cfg, cfg.batch_size, cfg.arch.L_ctx)
    targets = _indices(cfg, cfg.batch_size, cfg.arch.L_ctx, len(cfg.arch.head_offsets))

    nll, metrics = model.temporal.teacher_forced_nll_with_diagnostics(hidden, observed, targets)

    assert nll.shape == (*targets.shape[:-1], exp.N_GROUPS)
    assert set(metrics) == {
        "button_activation/pre_norm_rms_mean",
        "button_activation/pre_norm_rms_min",
        "button_activation/pre_norm_rms_p01",
        "button_activation/input_rms_mean",
        "button_activation/input_rms_min",
        "button_activation/input_abs_max",
        "button_activation/input_abs_p999",
        "button_logits/max",
        "button_logits/min",
        "button_logits/span",
        "button_logits/abs_p999",
        "button_logits/target_mean",
        "button_logits/target_min",
        "button_logits/correct_margin_mean",
        "button_logits/correct_margin_min",
        "button_logits/legal_support_mean",
        "button_logits/legal_support_min",
        "button_logits/target_masked_frac",
    }
    assert all(torch.isfinite(value) for value in metrics.values())
    assert metrics["button_logits/target_masked_frac"] == 0


def test_architecture_diagnostics_cover_geometry_and_objective_gradients() -> None:
    cfg = _cfg()
    model = exp.GPT(cfg)
    batch = exp.synthetic_awr_batch(cfg, torch.device("cpu"))

    metrics = exp.architecture_diagnostics_log(model, batch, cfg, max_rows=1)

    expected = {
        "head_geometry/buttons_output/top_singular_value_estimate",
        "head_geometry/buttons_output/stable_rank_estimate",
        "head_geometry/buttons_output/common_mode_frobenius_norm",
        "head_geometry/token_projection/top_singular_value_estimate",
        "head_geometry/buttons_input_top_vector_alignment_p99",
        "multi_objective_grad/buttons_rms",
        "multi_objective_grad/main_stick_rms",
        "multi_objective_grad/c_stick_rms",
        "multi_objective_grad/triggers_rms",
        "multi_objective_grad/value_actual_rms",
        "multi_objective_grad/value_hypothetical_rms",
    }
    assert expected <= metrics.keys()
    assert all(math.isfinite(value) for value in metrics.values())
    assert metrics["multi_objective_grad/value_actual_rms"] == 0.0


def test_layer_diagnostics_separate_temporal_residual_branches() -> None:
    cfg = _cfg(temporal_layers=1)
    model = exp.GPT(cfg)
    batch = exp.synthetic_awr_batch(cfg, torch.device("cpu"))

    metrics = exp.layer_activation_rms_log(model, batch, cfg, max_rows=1)

    assert "attention_branch_rms/temporal_block_00" in metrics
    assert "mlp_branch_rms/temporal_block_00" in metrics
    assert all(math.isfinite(value) and value >= 0 for value in metrics.values())


def test_wandb_uses_global_optimizer_step_for_every_series(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    run = SimpleNamespace(summary={})
    monkeypatch.setattr(exp.wandb, "init", lambda **_kwargs: None)
    monkeypatch.setattr(exp.wandb, "run", run)
    monkeypatch.setattr(exp.wandb, "define_metric", lambda name, **kwargs: calls.append((name, kwargs)))

    exp._init_wandb(_cfg(), "run", None)

    assert calls[:2] == [
        ("global_step", {}),
        ("*", {"step_metric": "global_step"}),
    ]
    assert exp._TRAIN_METRICS_EVERY == 10
    assert [update for update in range(1, 31) if update % exp._TRAIN_METRICS_EVERY == 0] == [10, 20, 30]


def test_optimizer_hyperparameters_remain_the_040_baseline() -> None:
    cfg = exp.TrainConfig()

    assert cfg.adam_lr == 8.5e-4
    assert cfg.adam_weight_decay == 0.0071
    assert exp._ADAM_UPDATE_CLIP_THRESHOLD == 1.0
    assert cfg.grad_clip == 1.0
    assert exp._EXPERIMENT_ID == "041_architectural_stability_v3"


def test_device_prefetch_loads_next_batch_in_background() -> None:
    cfg = _cfg()
    first = exp.synthetic_awr_batch(cfg, torch.device("cpu"))
    second = exp.synthetic_awr_batch(cfg, torch.device("cpu"))
    first.returns.fill_(1.0)
    second.returns.fill_(2.0)
    requested = threading.Event()
    release = threading.Event()

    def batches():
        yield first
        requested.set()
        assert release.wait(timeout=2.0)
        yield second

    prefetcher = exp.DeviceBatchPrefetcher(batches(), cfg, "cpu")
    try:
        actual_first, _, _ = prefetcher.next()
        prefetcher.start_preload()
        assert requested.wait(timeout=2.0)
        release.set()
        prefetcher.finish_preload()
        actual_second, _, _ = prefetcher.next()
    finally:
        release.set()
        prefetcher.close()

    torch.testing.assert_close(actual_first.returns, torch.ones_like(actual_first.returns))
    torch.testing.assert_close(actual_second.returns, torch.full_like(actual_second.returns, 2.0))


def test_cosine_schedule_is_a_named_ablation() -> None:
    cfg = exp.TrainConfig()
    cosine = exp.lr_schedule(cfg)
    late_cfg = exp.replace(cfg, lr_schedule_kind="late-cosine")
    late_cosine = exp.lr_schedule(late_cfg)

    assert cfg.lr_schedule_kind == "cosine"
    assert cosine(cfg.warmup_steps) == 1.0
    assert cosine(100_000) < 1.0
    assert late_cosine(100_000) == 1.0
    assert cosine(cfg.max_steps - 1) == pytest.approx(cfg.lr_floor_ratio)
    assert late_cosine(cfg.max_steps - 1) == pytest.approx(cfg.lr_floor_ratio)
    exp.validate_production_config(late_cfg)


def test_production_source_mix_uses_native_replay_lengths() -> None:
    cfg = exp.TrainConfig()
    weights = exp.source_mixture_weights(cfg)

    assert weights == tuple(float(exp.streams.POLICY_WORLD_V7_TRAIN_REPLAYS[name]) for name in cfg.source_names)


def test_training_checkpoints_are_resumable_every_two_thousand_updates() -> None:
    cfg = exp.TrainConfig()

    assert cfg.ckpt_every == 2000
    assert exp._TRAIN_PREFETCH_FACTOR == 4
    assert cfg.download_retry == 8
    assert cfg.loader_timeout_s == 300.0


def test_training_flop_estimate_repeats_only_temporal_parameters() -> None:
    cfg = _cfg()
    counts = {
        "trunk": 10,
        "other": 20,
        "value_head": 30,
        "temporal_decoder": 40,
        "group_heads": 50,
    }

    estimate = exp.approximate_training_flops_per_update(cfg, counts)

    positions = cfg.batch_size * cfg.arch.L_ctx
    expected = 6 * positions * (60 + len(cfg.arch.head_offsets) * 90)
    assert estimate == expected
