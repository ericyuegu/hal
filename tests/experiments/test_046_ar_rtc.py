"""Contracts for O46 dense autoregressive training-time RTC."""

import importlib.util
import json
import sys
from dataclasses import asdict
from dataclasses import fields
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hal.sim.vec import Slot
from hal.training.features import A_DIM
from hal.training.features import ACTION_CHANNELS
from hal.training.features import Context
from hal.training.features import TrainBatch


def _load(name: str, filename: str) -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "experiments" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


exp043 = _load("test_exp043_for_046", "043_legacy_codec.py")
exp = _load("test_exp046", "046_ar_rtc.py")


def _cfg(**overrides) -> exp.TrainConfig:
    values = dict(
        d_model=32,
        n_layers=1,
        n_heads=4,
        L_ctx=4,
        temporal_d_model=32,
        temporal_layers=1,
        temporal_heads=4,
        temporal_ff_dim=64,
        group_head_dim=64,
        batch_size=2,
        grad_accum_steps=1,
        reservoir_capacity=4,
        warmup_steps=1,
        max_steps=2,
        compile_trunk=False,
        compile_temporal=False,
        inference_mode="eager",
        num_workers=0,
        push_to_r2=False,
    )
    return exp.TrainConfig(**{**values, **overrides})


def _native_actions(batch: int, length: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    actions = torch.empty(batch, length, A_DIM)
    actions[..., :4] = torch.rand(batch, length, 4, generator=generator) * 2 - 1
    actions[..., 4:6] = torch.rand(batch, length, 2, generator=generator)
    actions[..., 6:] = torch.randint(0, 2, (batch, length, A_DIM - 6), generator=generator).float()
    return actions


def _batch(cfg: exp.TrainConfig) -> TrainBatch:
    context = exp.synthetic_context(cfg, cfg.batch_size, torch.device("cpu"))
    features = dict(context.features)
    history = _native_actions(cfg.batch_size, cfg.L_ctx, seed=3)
    for name, values in zip(ACTION_CHANNELS, history.unbind(-1), strict=True):
        features[f"ego_{name}"] = values
    return TrainBatch(
        context=Context(features=features, ctx_pad=context.ctx_pad),
        target=_native_actions(cfg.batch_size, cfg.prediction_horizon_frames, seed=4),
    )


def _committed_indices(batch: int, delay: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(9)
    return torch.stack(
        [torch.randint(vocab, (batch, delay), generator=generator) for vocab in exp.LEGACY_GROUP_VOCABS],
        dim=-1,
    )


def test_public_config_names_dense_chain_and_paper_timing() -> None:
    cfg = exp.TrainConfig()
    names = {item.name for item in fields(cfg)}
    assert cfg.prediction_horizon_frames == 20
    assert cfg.training_delay_frames == (0, 1, 2, 3, 4)
    assert cfg.inference_delay_frames == cfg.execution_stride_frames == 4
    assert cfg.head_offsets == tuple(range(1, 21))
    assert not names.intersection({"head_offsets", "sample_chunk_length", "next_frame_loss_share", "exec_horizon"})
    assert "ar046-rtc" in exp.model_tag(cfg)
    exp.validate_config(cfg)


def test_dense_offsets_keep_o43_parameter_geometry_and_initialization() -> None:
    cfg = _cfg()
    baseline_cfg = exp043.TrainConfig(
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        L_ctx=cfg.L_ctx,
        temporal_d_model=cfg.temporal_d_model,
        temporal_layers=cfg.temporal_layers,
        temporal_heads=cfg.temporal_heads,
        temporal_ff_dim=cfg.temporal_ff_dim,
        group_head_dim=cfg.group_head_dim,
        batch_size=cfg.batch_size,
        reservoir_capacity=cfg.reservoir_capacity,
        inference_mode="eager",
    )
    torch.manual_seed(123)
    baseline = exp043.GPT(baseline_cfg)
    torch.manual_seed(123)
    treatment = exp.GPT(cfg)

    assert tuple(treatment.state_dict()) == tuple(baseline.state_dict())
    assert all(
        torch.equal(treatment.state_dict()[name], baseline.state_dict()[name]) for name in baseline.state_dict()
    )
    assert treatment.head_offsets == tuple(range(1, 21))


def test_prepared_targets_are_every_next_action_through_offset_twenty() -> None:
    cfg = _cfg()
    model = exp.GPT(cfg)
    batch = _batch(cfg)
    history, targets, valid = exp.prepared_targets(model, batch)
    full = model.codec.quantize(
        torch.cat(
            (
                torch.stack([batch.context.features[f"ego_{name}"] for name in ACTION_CHANNELS], dim=-1),
                batch.target,
            ),
            dim=1,
        )
    )
    assert history.shape == (cfg.batch_size, cfg.L_ctx, exp.N_GROUPS)
    assert targets.shape == (cfg.batch_size, cfg.L_ctx, 20, exp.N_GROUPS)
    assert valid.all()
    for depth in range(20):
        torch.testing.assert_close(targets[:, :, depth], full[:, depth + 1 : depth + 1 + cfg.L_ctx])


def test_fixed_delay_objective_masks_prefix_and_normalizes_each_postfix() -> None:
    nll = torch.ones(1, 5, exp.N_GROUPS, requires_grad=True)
    parts = exp.ActionLoss(nll=nll, targets=torch.empty(0))
    loss = exp.objective(parts, training_delay_frames=(2,))
    loss.backward()

    assert float(loss.detach()) == pytest.approx(float(exp.N_GROUPS))
    assert nll.grad is not None
    assert not nll.grad[:, :2].any()
    assert nll.grad[0, 2:, 0].tolist() == pytest.approx([1 / 3] * 3)


def test_zero_delay_objective_is_dense_joint_maximum_likelihood() -> None:
    nll = torch.rand(7, 20, exp.N_GROUPS)
    parts = exp.ActionLoss(nll=nll, targets=torch.empty(0))
    actual = exp.objective(parts, training_delay_frames=(0,))
    expected = nll.sum(dim=-1).mean()
    torch.testing.assert_close(actual, expected)


def test_decoder_forces_committed_prefix_and_samples_valid_postfix() -> None:
    cfg = _cfg()
    model = exp.GPT(cfg).eval()
    hidden = torch.randn(2, cfg.L_ctx, cfg.d_model)
    observed = torch.zeros(2, exp.N_GROUPS, dtype=torch.long)
    committed = _committed_indices(2, cfg.inference_delay_frames)
    decoded = model.temporal.sample_indices(
        hidden,
        observed,
        model.head_offsets,
        committed=committed,
        argmax=True,
    )

    assert decoded.shape == (2, 20, exp.N_GROUPS)
    assert torch.equal(decoded[:, : cfg.inference_delay_frames], committed)
    for group, vocab in enumerate(exp.LEGACY_GROUP_VOCABS):
        assert (decoded[..., group] < vocab).all()

    logits, rollout = model.temporal.rollout_conditioned_logits(hidden, observed, committed)
    assert len(logits) == 20
    assert torch.equal(rollout[:, : cfg.inference_delay_frames], committed)
    assert torch.equal(decoded, rollout)
    assert all(
        torch.isfinite(logits[cfg.inference_delay_frames][name][..., :vocab]).all()
        for name, vocab in zip(exp.GROUP_NAMES, exp.LEGACY_GROUP_VOCABS, strict=True)
    )


def test_committed_plan_round_trips_through_legacy_codec() -> None:
    cfg = _cfg()
    codec = exp.GPT(cfg).codec
    committed = _committed_indices(8, cfg.inference_delay_frames)

    reconstructed = codec.quantize(codec.dequantize(committed))

    assert torch.equal(reconstructed, committed)


def test_inference_returns_committed_native_actions_exactly() -> None:
    cfg = _cfg()
    model = exp.GPT(cfg).eval()
    ctx = exp.synthetic_context(cfg, 2, torch.device("cpu"))
    committed = model.codec.dequantize(_committed_indices(2, cfg.inference_delay_frames)).numpy()
    inference = exp.BF16Inference(model, cfg, compiled=False)
    bootstrap = inference.decode(ctx, argmax=True)
    actions = inference.decode(ctx, committed, argmax=True)

    assert bootstrap.shape == (2, 20, A_DIM)
    assert actions.shape == (2, 20, A_DIM)
    assert np.array_equal(actions[:, : cfg.inference_delay_frames].numpy(), committed)


def test_policy_exposes_rtc_timing_to_the_shared_scheduler() -> None:
    cfg = _cfg()
    model = exp.GPT(cfg)

    class Engine:
        def __init__(self) -> None:
            self.model = model
            self.calls: list[np.ndarray | None] = []

        def decode(self, ctx, committed, **_kwargs):
            self.calls.append(committed)
            return torch.zeros(ctx.ctx_pad.shape[0], cfg.prediction_horizon_frames, A_DIM)

    engine = Engine()
    policy = exp.make_policy(model, {}, cfg, inference=engine, device="cpu")
    assert policy.L_chunk == 20
    assert policy.s == policy.d == 4
    assert policy.runtime_spec.prediction_frames == 20
    assert policy.runtime_spec.execution_stride == 4
    assert policy.runtime_spec.committed_frames == 4

    slot = Slot(0, 1)
    pending = np.arange(20 * A_DIM, dtype=np.float32).reshape(20, A_DIM)
    policy._slots[slot] = SimpleNamespace(pending=pending)
    np.testing.assert_array_equal(policy._committed([slot])[0], pending[4:8])

    context = exp.synthetic_context(cfg, 2, torch.device("cpu"))
    committed = np.zeros((2, 4, A_DIM), dtype=np.float32)
    result = policy.predict_chunk(context, committed)
    assert result.shape == (2, 20, A_DIM)
    assert engine.calls == [committed]


@pytest.mark.parametrize(
    "overrides",
    [
        {"training_delay_frames": (1, 2, 3)},
        {"training_delay_frames": (0, 4, 3)},
        {"training_delay_frames": (0, 11)},
        {"inference_delay_frames": 3, "execution_stride_frames": 4},
        {"inference_delay_frames": 11, "execution_stride_frames": 11},
    ],
)
def test_validation_rejects_invalid_rtc_timing(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        exp.validate_config(_cfg(**overrides))


def test_metrics_report_each_training_delay() -> None:
    cfg = _cfg()
    mean_nll = torch.ones(20, exp.N_GROUPS)
    with exp._training_delays(cfg.training_delay_frames):
        metrics = exp.nll_mean_metrics(mean_nll, exp.ACTION_OFFSETS_FRAMES)
    assert metrics["dense_nll"] == pytest.approx(exp.N_GROUPS / np.log(2))
    for delay in cfg.training_delay_frames:
        assert metrics[f"conditional_nll_d{delay:02d}"] == pytest.approx(exp.N_GROUPS / np.log(2))
        assert metrics[f"first_postfix_nll_d{delay:02d}"] == pytest.approx(exp.N_GROUPS / np.log(2))


def test_checkpoint_records_dense_offsets_and_objective_version(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def capture(path: Path, **kwargs) -> None:
        captured["path"] = path
        captured.update(kwargs)

    monkeypatch.setattr(exp, "_O43_SAVE_CHECKPOINT", capture)
    cfg = _cfg()
    path = tmp_path / "checkpoint.pt"
    exp.save_checkpoint(path, cfg=asdict(cfg))

    assert captured["path"] == path
    saved = captured["cfg"]
    assert isinstance(saved, dict)
    assert saved["action_offsets_frames"] == exp.ACTION_OFFSETS_FRAMES
    assert saved["rtc_objective_version"] == exp.RTC_OBJECTIVE_VERSION
    assert exp.config_from_state(saved) == cfg


def test_eval_evidence_uses_explicit_rtc_timing(tmp_path: Path) -> None:
    cfg = _cfg()
    model = exp.GPT(cfg)
    protocol = exp._eval_protocol(
        cfg,
        model,
        n_matchups=2,
        exec_horizon=cfg.execution_stride_frames,
        checkpoint_sha256="a" * 64,
    )
    metrics = {"exec_horizon": 4.0, "boots": 2.0}
    exp._write_eval_evidence(tmp_path, [], metrics, protocol)
    payload = json.loads((tmp_path / "match_rows.json").read_text())
    saved_metrics = json.loads((tmp_path / "metrics.json").read_text())

    assert payload["protocol"]["training_delay_frames"] == [0, 1, 2, 3, 4]
    assert payload["protocol"]["inference_delay_frames"] == 4
    assert payload["protocol"]["execution_stride_frames"] == 4
    assert "exec_horizon" not in payload["protocol"]
    assert "exec_horizon" not in saved_metrics


def test_tiny_training_entrypoint_uses_dense_rtc_objective(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _cfg(
        max_steps=1,
        val_every=0,
        eval_every=0,
        ckpt_every=0,
        wandb_log_code=False,
        final_eval_n_matchups=1,
    )
    batch = _batch(cfg)
    observed_delays: list[tuple[int, ...] | None] = []
    real_objective = exp._joint_objective

    def recorded_objective(*args, **kwargs):
        observed_delays.append(exp._ACTIVE_TRAINING_DELAYS)
        return real_objective(*args, **kwargs)

    monkeypatch.setattr(exp._o43, "_joint_objective", recorded_objective)
    monkeypatch.setattr(exp._o43, "_make_loaders", lambda cfg, stats: ([batch], [batch]))
    monkeypatch.setattr(exp._o43, "make_run_name", lambda *args: "tiny-046")
    monkeypatch.setattr(exp._o43, "setup_run_dir", lambda name: (tmp_path, tmp_path / "replays"))
    monkeypatch.setattr(exp._o43, "save_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(exp._o43, "_checkpoint_sha256", lambda path: "a" * 64)
    monkeypatch.setattr(exp._o43, "BF16Inference", lambda model, cfg: object())
    monkeypatch.setattr(
        exp._o43,
        "eval_suites",
        lambda model, stats, cfg, **kwargs: {
            "char_matchup": {"scheduled_boots": 1.0, "completed_boots": 1.0, "boots": 1.0},
            "fox": {"scheduled_boots": 1.0, "completed_boots": 1.0, "boots": 1.0},
        },
    )

    class Run:
        id = "test"
        summary: dict[str, object] = {}

    monkeypatch.setattr(exp.wandb, "run", Run())
    monkeypatch.setattr(exp.wandb, "init", lambda **kwargs: None)
    monkeypatch.setattr(exp.wandb, "define_metric", lambda *args, **kwargs: None)
    monkeypatch.setattr(exp.wandb, "log", lambda *args, **kwargs: None)
    monkeypatch.setattr(exp.wandb, "finish", lambda **kwargs: None)

    exp.train(cfg, {}, comment="tiny")

    assert observed_delays == [cfg.training_delay_frames]
