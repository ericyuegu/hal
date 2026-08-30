"""Contracts for O44 fixed-delay prediction and scheduling."""

import importlib.util
import json
import sys
from dataclasses import asdict
from dataclasses import fields
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch

from hal.eval.self_play import DecodeTelemetry
from hal.sim.rollout import ObservationRow
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


exp043 = _load("test_exp043_for_044", "043_legacy_codec.py")
exp = _load("test_exp044", "044_fixed_prediction_delay.py")


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


def _actions(batch: int, length: int) -> torch.Tensor:
    actions = torch.zeros(batch, length, A_DIM)
    actions[..., 0] = torch.arange(length).float() / 100
    return actions


def _batch(cfg: exp.TrainConfig) -> TrainBatch:
    context = exp.synthetic_context(cfg, cfg.batch_size, torch.device("cpu"))
    features = dict(context.features)
    history = _actions(cfg.batch_size, cfg.L_ctx)
    for name, values in zip(ACTION_CHANNELS, history.unbind(-1), strict=True):
        features[f"ego_{name}"] = values
    return TrainBatch(
        context=Context(features=features, ctx_pad=context.ctx_pad),
        target=_actions(cfg.batch_size, exp.FUTURE_TARGET_BUFFER_FRAMES),
    )


def test_public_config_uses_distinct_timing_terms() -> None:
    cfg = exp.TrainConfig()
    names = {item.name for item in fields(cfg)}
    assert cfg.prediction_delay_frames == 3
    assert cfg.replan_interval_frames == 1
    assert cfg.prediction_horizon_frames == 6
    assert not names.intersection({"exec_horizon", "head_offsets", "sample_chunk_length", "next_frame_loss_share"})
    assert not set(asdict(cfg)).intersection(
        {"exec_horizon", "head_offsets", "sample_chunk_length", "next_frame_loss_share"}
    )


@pytest.mark.parametrize(
    ("delay", "expected"),
    [
        (1, (1, 2, 3, 4, 5, 6, 9, 12, 16, 20)),
        (3, (3, 4, 5, 6, 7, 8, 11, 14, 18, 22)),
        (12, (12, 13, 14, 15, 16, 17, 20, 23, 27, 31)),
        (18, (18, 19, 20, 21, 22, 23, 26, 29, 33, 37)),
    ],
)
def test_prediction_target_offsets(delay: int, expected: tuple[int, ...]) -> None:
    assert exp.prediction_target_offsets(_cfg(prediction_delay_frames=delay)) == expected


@pytest.mark.parametrize("delay", [3, 12, 18])
def test_prepared_targets_align_to_absolute_future_frames(delay: int) -> None:
    cfg = _cfg(prediction_delay_frames=delay)
    model = exp.GPT(cfg)
    history, targets, valid = exp.prepared_targets(model, _batch(cfg))
    assert history.shape == (cfg.batch_size, cfg.L_ctx, exp.N_GROUPS)
    assert targets.shape == (cfg.batch_size, cfg.L_ctx, len(exp.PREDICTION_STEPS), exp.N_GROUPS)
    assert valid.all()

    full = model.codec.quantize(
        torch.cat((_actions(cfg.batch_size, cfg.L_ctx), _actions(cfg.batch_size, exp.FUTURE_TARGET_BUFFER_FRAMES)), 1)
    )
    for index, offset in enumerate(exp.prediction_target_offsets(cfg)):
        assert torch.equal(targets[:, :, index], full[:, offset : offset + cfg.L_ctx])


def test_delay_does_not_change_initialization_or_parameter_count() -> None:
    cfg = _cfg(prediction_delay_frames=3)
    torch.manual_seed(123)
    early = exp.GPT(cfg)
    torch.manual_seed(123)
    late = exp.GPT(replace(cfg, prediction_delay_frames=18))
    assert tuple(early.state_dict()) == tuple(late.state_dict())
    assert all(torch.equal(early.state_dict()[name], late.state_dict()[name]) for name in early.state_dict())
    assert sum(parameter.numel() for parameter in early.parameters()) == sum(
        parameter.numel() for parameter in late.parameters()
    )


def test_d1_model_geometry_matches_o43() -> None:
    cfg044 = _cfg(prediction_delay_frames=1)
    cfg043 = exp043.TrainConfig(
        d_model=cfg044.d_model,
        n_layers=cfg044.n_layers,
        n_heads=cfg044.n_heads,
        L_ctx=cfg044.L_ctx,
        temporal_d_model=cfg044.temporal_d_model,
        temporal_layers=cfg044.temporal_layers,
        temporal_heads=cfg044.temporal_heads,
        temporal_ff_dim=cfg044.temporal_ff_dim,
        group_head_dim=cfg044.group_head_dim,
        batch_size=cfg044.batch_size,
        reservoir_capacity=cfg044.reservoir_capacity,
        inference_mode="eager",
    )
    torch.manual_seed(7)
    model043 = exp043.GPT(cfg043)
    torch.manual_seed(7)
    model044 = exp.GPT(cfg044)
    assert tuple(model043.state_dict()) == tuple(model044.state_dict())
    assert all(torch.equal(model043.state_dict()[name], model044.state_dict()[name]) for name in model043.state_dict())


class _FakeContextBuilder:
    def _context(self, live: list[Slot]) -> Context:
        rows = len(live)
        return Context(
            features={"dummy": torch.zeros(rows, 1)},
            ctx_pad=torch.zeros(rows, dtype=torch.long),
            slot_ids=torch.tensor([slot.match * 8 + slot.port for slot in live]),
        )

    def _ingest_row(self, slot: Slot, row: ObservationRow) -> None:
        del slot, row


@pytest.mark.parametrize("delay", [3, 12, 18])
def test_policy_releases_latest_plan_at_the_fixed_delay(delay: int) -> None:
    cfg = _cfg(prediction_delay_frames=delay)

    def predict(ctx: Context) -> np.ndarray:
        actions = np.zeros((ctx.ctx_pad.shape[0], cfg.prediction_horizon_frames, A_DIM), dtype=np.float32)
        anchor = policy.last_observation[slot]
        actions[:, :, 0] = (anchor + np.arange(cfg.prediction_horizon_frames)) / 100
        return actions

    policy = exp.FixedPredictionDelayPolicy(
        predict,
        {},
        cfg,
        telemetry=None,
        device="cpu",
        float_dtype=torch.float32,
    )
    policy.context = _FakeContextBuilder()
    slot = Slot(0, 1)
    executed = []
    for frame in range(delay + 3):
        plans = policy.plan_rows(
            {
                slot: [
                    ObservationRow(
                        frame_id=frame,
                        flat={},
                        action=np.zeros(A_DIM, dtype=np.float32),
                        reset=frame == 0,
                    )
                ]
            }
        )
        executed.append(float(plans[slot][0, 0]))
    assert executed[: delay - 1] == [0.0] * (delay - 1)
    assert executed[delay - 1 :] == pytest.approx([frame / 100 for frame in range(4)])
    assert policy.runtime_spec.prediction_frames == 1
    assert policy.runtime_spec.execution_stride == 1


def test_policy_reset_restarts_delay_warmup() -> None:
    cfg = _cfg(prediction_delay_frames=3)

    def predict(ctx: Context) -> np.ndarray:
        actions = np.zeros((ctx.ctx_pad.shape[0], cfg.prediction_horizon_frames, A_DIM), dtype=np.float32)
        actions[..., 0] = 0.5
        return actions

    policy = exp.FixedPredictionDelayPolicy(
        predict,
        {},
        cfg,
        telemetry=None,
        device="cpu",
        float_dtype=torch.float32,
    )
    policy.context = _FakeContextBuilder()
    slot = Slot(0, 1)

    def step(frame: int, reset: bool) -> float:
        result = policy.plan_rows(
            {
                slot: [
                    ObservationRow(
                        frame_id=frame,
                        flat={},
                        action=np.zeros(A_DIM, dtype=np.float32),
                        reset=reset,
                    )
                ]
            }
        )
        return float(result[slot][0, 0])

    assert [step(frame, frame == 0) for frame in range(3)] == pytest.approx([0.0, 0.0, 0.5])
    assert [step(frame, frame == 0) for frame in range(3)] == pytest.approx([0.0, 0.0, 0.5])


def test_policy_accepts_shared_self_play_telemetry() -> None:
    cfg = _cfg(prediction_delay_frames=3)
    telemetry = DecodeTelemetry()
    policy = exp.FixedPredictionDelayPolicy(
        lambda ctx: np.zeros((ctx.ctx_pad.shape[0], cfg.prediction_horizon_frames, A_DIM), dtype=np.float32),
        {},
        cfg,
        telemetry=telemetry,
        device="cpu",
        float_dtype=torch.float32,
    )
    policy.context = _FakeContextBuilder()
    slot = Slot(0, 1)

    result = policy.plan_rows(
        {
            slot: [
                ObservationRow(
                    frame_id=0,
                    flat={},
                    action=np.zeros(A_DIM, dtype=np.float32),
                    reset=True,
                )
            ]
        }
    )

    assert np.array_equal(result[slot][0], np.zeros(A_DIM, dtype=np.float32))


def test_validation_rejects_ambiguous_or_unsupported_timing() -> None:
    with pytest.raises(ValueError, match="prediction_delay_frames"):
        exp.validate_config(_cfg(prediction_delay_frames=19))
    with pytest.raises(ValueError, match="prediction_horizon_frames"):
        exp.validate_config(_cfg(prediction_horizon_frames=5))
    with pytest.raises(ValueError, match="replan_interval_frames"):
        exp.validate_config(_cfg(replan_interval_frames=7))


def test_eval_evidence_uses_new_timing_names(tmp_path: Path) -> None:
    cfg = _cfg(prediction_delay_frames=12)
    model = exp.GPT(cfg)
    protocol = exp._eval_protocol(
        cfg,
        model,
        n_matchups=2,
        exec_horizon=cfg.prediction_horizon_frames,
        checkpoint_sha256="a" * 64,
    )
    metrics = {"exec_horizon": 6.0, "boots": 2.0}
    exp._write_eval_evidence(tmp_path, [], metrics, protocol)
    payload = json.loads((tmp_path / "match_rows.json").read_text())
    saved_metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert payload["protocol"]["prediction_delay_frames"] == 12
    assert payload["protocol"]["replan_interval_frames"] == 1
    assert payload["protocol"]["prediction_horizon_frames"] == 6
    assert "exec_horizon" not in payload["protocol"]
    assert "exec_horizon" not in saved_metrics


def test_checkpoint_records_and_validates_prediction_steps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def capture(path: Path, **kwargs) -> None:
        captured["path"] = path
        captured.update(kwargs)

    monkeypatch.setattr(exp, "_O43_SAVE_CHECKPOINT", capture)
    cfg = _cfg(prediction_delay_frames=12)
    path = tmp_path / "checkpoint.pt"
    exp.save_checkpoint(path, cfg=asdict(cfg))

    assert captured["path"] == path
    saved = captured["cfg"]
    assert isinstance(saved, dict)
    assert saved["prediction_steps_frames"] == exp.PREDICTION_STEPS
    restored = exp.config_from_state(saved)
    assert restored == cfg
    saved["prediction_steps_frames"] = (0, 1, 2)
    with pytest.raises(ValueError, match="prediction steps"):
        exp.config_from_state(saved)


def test_tiny_training_entrypoint_uses_fixed_delay_targets(
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
        prediction_delay_frames=12,
    )
    batch = _batch(cfg)
    monkeypatch.setattr(exp._o43, "_make_loaders", lambda cfg, stats: ([batch], [batch]))
    monkeypatch.setattr(exp._o43, "make_run_name", lambda *args: "tiny-044")
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
