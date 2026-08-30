"""Experiment 043 with one fixed prediction delay per checkpoint.

The model observes game state through frame ``t``. Its main prediction contains
six controller frames for ``[t + d, t + d + H)``, where ``d`` is the trained
prediction delay and ``H`` is the prediction horizon. Closed-loop evaluation
replans independently every ``R`` frames and releases each plan on its absolute
target frame.

Run:
    uv run experiments/044_fixed_prediction_delay.py --cfg.prediction-delay-frames 3
    uv run experiments/044_fixed_prediction_delay.py --cfg.prediction-delay-frames 12
    uv run experiments/044_fixed_prediction_delay.py --cfg.prediction-delay-frames 18
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any
from typing import Literal

import numpy as np
import torch
import tyro
from torch import Tensor

import wandb
from hal.data.feature_stats import FeatureStats
from hal.eval.self_play import benchmark_checkpoint as benchmark_self_play
from hal.sim.inputs import ControllerInputs
from hal.sim.inputs import action_vec_to_controller
from hal.sim.rollout import ObservationRow
from hal.sim.rollout import PolicyRuntimeSpec
from hal.sim.vec import Slot
from hal.training.checkpoints import download_latest
from hal.training.checkpoints import load_for_resume
from hal.training.ego_stats import load_consolidated_stats
from hal.training.features import A_DIM
from hal.training.features import ACTION_CHANNELS
from hal.training.features import BASE_ACTION_PROJECTION
from hal.training.features import NEUTRAL_ACTION
from hal.training.features import V6_PLAYER_COLUMNS
from hal.training.features import Context
from hal.training.features import TrainBatch
from hal.training.features import stack_actions


def _load_o43() -> ModuleType:
    path = Path(__file__).with_name("043_legacy_codec.py")
    name = "_hal_experiment_043_for_044"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_o43 = _load_o43()
DEVICE = _o43.DEVICE
_O43_TRAIN_CONFIG = _o43.TrainConfig

# Step zero is the first predicted action. Adding d gives its absolute target
# offset from the newest observed frame. The first six steps are the live plan;
# the remaining four preserve O43's sparse auxiliary supervision.
PREDICTION_STEPS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 8, 11, 15, 19)
DECODER_OFFSETS: tuple[int, ...] = tuple(step + 1 for step in PREDICTION_STEPS)
FUTURE_TARGET_BUFFER_FRAMES = 37
MAX_PREDICTION_DELAY_FRAMES = FUTURE_TARGET_BUFFER_FRAMES - PREDICTION_STEPS[-1]


@dataclass
class TrainConfig:
    """O44 configuration with explicit prediction and scheduling terms."""

    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 6
    attn_window: int = 0
    require_flex: bool = False
    L_ctx: int = 128

    decoder_arch_version: int = 3
    codec_version: int = 2
    temporal_d_model: int = 128
    temporal_layers: int = 2
    temporal_heads: int = 4
    temporal_ff_dim: int = 256
    group_head_dim: int = 256
    action_embed_dim: int = 16
    offset_embed_dim: int = 16
    aux_loss_weight: float = 1.0
    first_prediction_loss_share: float = 0.5
    group_order: tuple[str, ...] = _o43.GROUP_ORDER

    action_vocab: int = 1024
    action_state_embed_dim: int = 48
    char_vocab: int = 32
    char_dim: int = 8
    stage_vocab: int = 32
    stage_dim: int = 4
    observation_bundle: str = "base"

    prediction_delay_frames: int = 3
    replan_interval_frames: int = 1
    prediction_horizon_frames: int = 6
    decode_temp: float = 1.0
    inference_mode: str = "compiled"
    inference_buckets: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    compiled_inference_bucket: int | None = None

    seed: int = 0
    eval_seed: int = 0
    batch_size: int = 512
    grad_accum_steps: int = 1
    muon_lr: float = 0.02
    adam_lr: float = 8.5e-4
    muon_weight_decay: float = 0.01
    adam_weight_decay: float = 0.01
    lr_floor_ratio: float = 1e-5 / 8.5e-4
    warmup_steps: int = 500
    max_steps: int = 16_384
    amp_dtype: str = "bfloat16"
    allow_tf32: bool = True
    compile_trunk: bool = True
    compile_temporal: bool = True

    wandb_log_code: bool = True
    wandb_grad_every: int = 1024
    val_every: int = 1024
    val_n_samples: int = 1192
    val_batch_size: int = 128
    ckpt_every: int = 1024
    eval_every: int = 4096
    eval_max_frames: int = 7200
    eval_n_matchups: int = 32
    final_eval_n_matchups: int = 96
    final_diag_n_matchups: int = 0
    eval_max_parallel: int | None = 32

    data_root: str = "data/processed/ranked-anonymized-1/mds-policy-v7"
    train_replay_paths: str | None = None
    replay_format: Literal["policy", "policy-world"] = "policy"
    val_data_root: str = "data/processed/ranked-anonymized-1/mds-policy-v7"
    val_replay_format: Literal["policy", "policy-world"] = "policy"
    compact_data: bool = True
    mds_schema_version: int = 7
    cache_limit_gb: int = 160
    shuffle_block_size: int = 2000
    predownload: int = 512
    windows_per_replay: int = 4
    reservoir_capacity: int = 4096
    val_split: str = "val"
    num_workers: int = 16
    prefetch_factor: int = 2
    prefetch_batches: int = 4
    push_to_r2: bool = True
    system_metrics_every: int = 25
    system_metrics_interval_s: float = 5.0
    process_metrics_interval_s: float = 30.0
    cache_metrics_interval_s: float = 30.0

    # Compatibility properties are deliberately absent from serialized configs
    # and the CLI. They let the frozen O43 training loop consume O44's clearer
    # public interface.
    @property
    def head_offsets(self) -> tuple[int, ...]:
        return DECODER_OFFSETS

    @property
    def sample_chunk_length(self) -> int:
        return FUTURE_TARGET_BUFFER_FRAMES

    @property
    def next_frame_loss_share(self) -> float:
        return self.first_prediction_loss_share

    @property
    def exec_horizon(self) -> int:
        return self.prediction_horizon_frames

    @property
    def final_diag_exec_horizon(self) -> int:
        return self.prediction_horizon_frames


def prediction_target_offsets(cfg: TrainConfig) -> tuple[int, ...]:
    """Absolute target offsets from the newest observation."""
    return tuple(cfg.prediction_delay_frames + step for step in PREDICTION_STEPS)


def _as_o43_config(cfg: TrainConfig, *, model_geometry: bool = False) -> Any:
    values = {item.name: getattr(cfg, item.name) for item in fields(_O43_TRAIN_CONFIG)}
    if model_geometry:
        values["sample_chunk_length"] = PREDICTION_STEPS[-1] + 1
    return _O43_TRAIN_CONFIG(**values)


_O43_VALIDATE_CONFIG = _o43.validate_config


def validate_config(cfg: TrainConfig) -> None:
    if not isinstance(cfg.prediction_delay_frames, int) or isinstance(cfg.prediction_delay_frames, bool):
        raise ValueError("prediction_delay_frames must be an integer")
    if not 1 <= cfg.prediction_delay_frames <= MAX_PREDICTION_DELAY_FRAMES:
        raise ValueError(
            f"prediction_delay_frames must be in [1, {MAX_PREDICTION_DELAY_FRAMES}], got {cfg.prediction_delay_frames}"
        )
    if cfg.prediction_horizon_frames != 6:
        raise ValueError("experiment 044 fixes prediction_horizon_frames at 6")
    if not 1 <= cfg.replan_interval_frames <= cfg.prediction_horizon_frames:
        raise ValueError("replan_interval_frames must be between 1 and prediction_horizon_frames")
    if not math.isfinite(cfg.first_prediction_loss_share) or not 0 <= cfg.first_prediction_loss_share <= 1:
        raise ValueError("first_prediction_loss_share must be finite and between zero and one")
    _O43_VALIDATE_CONFIG(_as_o43_config(cfg))


class GPT(_o43.GPT):
    """O43 architecture with delay-independent parameter geometry."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__(_as_o43_config(cfg, model_geometry=True))
        self.cfg = cfg
        self.L_chunk = FUTURE_TARGET_BUFFER_FRAMES
        self.target_offsets = prediction_target_offsets(cfg)


def prepared_targets(model: GPT, batch: TrainBatch) -> tuple[Tensor, Tensor, Tensor]:
    """Align O43's relative prediction steps to this checkpoint's fixed delay."""
    if batch.target.shape[1] < FUTURE_TARGET_BUFFER_FRAMES:
        raise ValueError(f"target contains {batch.target.shape[1]} frames, expected {FUTURE_TARGET_BUFFER_FRAMES}")
    history = stack_actions(batch.context.features)
    if history.shape[1] != model.trunk.L_ctx:
        raise ValueError(f"context length {history.shape[1]} != {model.trunk.L_ctx}")
    full = model.codec.quantize(torch.cat((history, batch.target[:, :FUTURE_TARGET_BUFFER_FRAMES]), dim=1))
    length = history.shape[1]
    targets = torch.stack([full[:, offset : offset + length] for offset in model.target_offsets], dim=2)
    valid = torch.arange(length, device=full.device)[None, :] >= batch.context.ctx_pad[:, None]
    return full[:, :length], targets, valid


def chunk_targets(model: GPT, batch: TrainBatch) -> tuple[Tensor, Tensor]:
    _, targets, valid = prepared_targets(model, batch)
    return targets, valid


class _MetricDict(dict[str, float]):
    """Public metric names with private aliases for the frozen validation loop."""

    def __init__(self, values: dict[str, float], aliases: dict[str, float]) -> None:
        super().__init__(values)
        self._aliases = aliases

    def __getitem__(self, key: str) -> float:
        if key in self._aliases:
            return self._aliases[key]
        return super().__getitem__(key)


_O43_NLL_MEAN_METRICS = _o43.nll_mean_metrics


def _rename_metric(name: str) -> str:
    for offset, step in zip(DECODER_OFFSETS, PREDICTION_STEPS, strict=True):
        name = name.replace(f"_o{offset:02d}", f"_step{step:02d}")
    return name.replace("next_frame", "first_prediction")


def nll_mean_metrics(
    mean_nll: Tensor,
    offsets: tuple[int, ...],
    *,
    aux_loss_weight: float = 1.0,
    next_frame_loss_share: float | None = 0.5,
) -> dict[str, float]:
    legacy = _O43_NLL_MEAN_METRICS(
        mean_nll,
        offsets,
        aux_loss_weight=aux_loss_weight,
        next_frame_loss_share=next_frame_loss_share,
    )
    public = {_rename_metric(name): value for name, value in legacy.items()}
    return _MetricDict(public, legacy)


def nll_metrics(
    nll: Tensor,
    offsets: tuple[int, ...],
    *,
    aux_loss_weight: float = 1.0,
    next_frame_loss_share: float | None = 0.5,
) -> dict[str, float]:
    return nll_mean_metrics(
        nll.mean(dim=0),
        offsets,
        aux_loss_weight=aux_loss_weight,
        next_frame_loss_share=next_frame_loss_share,
    )


def _transition_metrics(target: Tensor, prediction: Tensor, observed: Tensor) -> dict[str, float]:
    """Measure adjacent predicted frames without treating the d-frame jump as one transition."""
    target_change = target[:, 1:] != target[:, :-1]
    sampled_change = prediction[:, 1:] != prediction[:, :-1]
    expected = target[:, 1:]
    actual = prediction[:, 1:]
    true_positive = (target_change & sampled_change).sum().float()
    precision = true_positive / sampled_change.sum().clamp_min(1)
    recall = true_positive / target_change.sum().clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {
        "hold_acc": float(((~target_change) & (actual == expected)).sum() / (~target_change).sum().clamp_min(1)),
        "transition_acc": float((target_change & (actual == expected)).sum() / target_change.sum().clamp_min(1)),
        "change_precision": float(precision),
        "change_recall": float(recall),
        "change_f1": float(f1),
        "target_transition_rate": float(target_change.float().mean()),
        "sampled_transition_rate": float(sampled_change.float().mean()),
        "copy_previous_acc": float((expected == target[:, :-1]).float().mean()),
        "first_target_copies_observed_acc": float((target[:, 0] == observed).float().mean()),
    }


_O43_VAL_METRICS = _o43.val_metrics


def val_metrics(model: GPT, batches: list[TrainBatch], cfg: TrainConfig) -> dict[str, float]:
    values = dict(_O43_VAL_METRICS(model, batches, cfg))
    renamed = {_rename_metric(name): value for name, value in values.items()}
    if "dense_four_sequence_acc" in renamed:
        renamed["prediction_sequence_acc_h04"] = renamed.pop("dense_four_sequence_acc")
    return renamed


def objective(
    parts: Any,
    aux_loss_weight: float = 1.0,
    first_prediction_loss_share: float = 0.5,
) -> Tensor:
    return _o43.objective(
        parts,
        aux_loss_weight=aux_loss_weight,
        next_frame_loss_share=first_prediction_loss_share,
    )


@dataclass
class DelayTelemetry(_o43.DecodeTelemetry):
    observation_ages: list[int] = field(default_factory=list)
    neutral_actions: int = 0

    def metrics(self) -> dict[str, float]:
        values = super().metrics()
        values["decode_predicted_frames_per_s"] = values.pop("decode_executed_frames_per_s")
        ages = np.asarray(self.observation_ages, dtype=np.float64)
        values.update(
            {
                "observation_age_mean_frames": float(ages.mean()) if ages.size else 0.0,
                "observation_age_p95_frames": float(np.percentile(ages, 95)) if ages.size else 0.0,
                "neutral_actions": float(self.neutral_actions),
            }
        )
        return values


class PredictionStepStreams:
    """Independent per-step streams; step zero matches O43 exactly."""

    def __init__(self, seed: int, horizon: int) -> None:
        self.streams = [
            _o43.SlotGroupRandom(seed if step == 0 else _o43._splitmix64(seed ^ step)) for step in range(horizon)
        ]
        self.cursor = 0

    def begin(self, ctx: Context) -> None:
        for stream in self.streams:
            stream.begin(ctx)
        self.cursor = 0

    def uniforms(self, group: str) -> Tensor:
        step = self.cursor // _o43.N_GROUPS
        if step >= len(self.streams):
            raise RuntimeError("prediction-step random stream exhausted")
        self.cursor += 1
        return self.streams[step].uniforms(group)


@dataclass(frozen=True, slots=True)
class HeldPlan:
    observation_frame: int
    actions: np.ndarray


class FixedPredictionDelayPolicy:
    """Release fixed-delay predictions by absolute game-frame target."""

    def __init__(
        self,
        predict: Any,
        stats: dict[str, FeatureStats],
        cfg: TrainConfig,
        *,
        telemetry: DelayTelemetry | None,
        device: str,
        float_dtype: torch.dtype,
    ) -> None:
        self.predict = predict
        self.delay = cfg.prediction_delay_frames
        self.replan_interval = cfg.replan_interval_frames
        self.horizon = cfg.prediction_horizon_frames
        self.context_frames = cfg.L_ctx
        self.delay_telemetry = telemetry if isinstance(telemetry, DelayTelemetry) else None
        v6 = cfg.observation_bundle == "v6_lean"
        self.context = _o43.RecedingHorizon(
            predict_chunk=lambda ctx, committed: np.empty((ctx.ctx_pad.shape[0], 1, A_DIM), dtype=np.float32),
            stats=stats,
            L_ctx=cfg.L_ctx,
            L_chunk=1,
            s=1,
            d=0,
            device=device,
            float_dtype=float_dtype,
            extra=V6_PLAYER_COLUMNS if v6 else None,
            projection=None if v6 else BASE_ACTION_PROJECTION,
        )
        self.last_observation: dict[Slot, int] = {}
        self.last_request: dict[Slot, int] = {}
        self.plans: dict[Slot, list[HeldPlan]] = {}

    @property
    def runtime_spec(self) -> PolicyRuntimeSpec:
        # The broker exchanges one executed action every frame. Model horizon
        # and replan cadence are internal policy properties.
        return PolicyRuntimeSpec(
            context_frames=self.context_frames,
            prediction_frames=1,
            execution_stride=1,
            committed_frames=0,
            action_dim=A_DIM,
        )

    def _observe_frame(self, slot: Slot, frame: int, *, reset: bool) -> None:
        previous = self.last_observation.get(slot)
        if reset or (previous is not None and frame < previous):
            self.last_request.pop(slot, None)
            self.plans.pop(slot, None)
        self.last_observation[slot] = frame

    def _actions_for_live(self, live: list[Slot]) -> dict[Slot, np.ndarray]:
        due = [
            slot
            for slot in live
            if slot not in self.last_request
            or self.last_observation[slot] - self.last_request[slot] >= self.replan_interval
        ]
        if due:
            predicted = self.predict(self.context._context(due))
            expected = (len(due), self.horizon, A_DIM)
            if predicted.shape != expected:
                raise ValueError(f"fixed-delay policy got plan shape {predicted.shape}, expected {expected}")
            for row, slot in enumerate(due):
                anchor = self.last_observation[slot]
                self.plans.setdefault(slot, []).append(HeldPlan(anchor, predicted[row]))
                self.last_request[slot] = anchor

        actions: dict[Slot, np.ndarray] = {}
        for slot in live:
            target_frame = self.last_observation[slot] + 1
            candidates = [
                plan
                for plan in self.plans.get(slot, ())
                if plan.observation_frame + self.delay <= target_frame
                and target_frame < plan.observation_frame + self.delay + self.horizon
            ]
            if candidates:
                plan = max(candidates, key=lambda item: item.observation_frame)
                step = target_frame - plan.observation_frame - self.delay
                action = plan.actions[step]
                if self.delay_telemetry is not None:
                    self.delay_telemetry.observation_ages.append(target_frame - plan.observation_frame)
            else:
                action = NEUTRAL_ACTION
                if self.delay_telemetry is not None:
                    self.delay_telemetry.neutral_actions += 1
            actions[slot] = np.asarray(action, dtype=np.float32)
            self.plans[slot] = [
                plan
                for plan in self.plans.get(slot, ())
                if target_frame < plan.observation_frame + self.delay + self.horizon
            ]
        return actions

    def __call__(self, frame_index: int, obs: Mapping[Slot, dict]) -> Mapping[Slot, ControllerInputs]:
        del frame_index
        live = list(obs)
        for slot in live:
            self._observe_frame(slot, int(obs[slot]["id"]), reset=False)
        self.context._ingest(live, obs)
        actions = self._actions_for_live(live)
        for slot in live:
            self.context._push_ego(slot, actions[slot])
        return {slot: action_vec_to_controller(action) for slot, action in actions.items()}

    def plan_rows(self, rows: Mapping[Slot, Sequence[ObservationRow]]) -> Mapping[Slot, np.ndarray]:
        live = list(rows)
        for slot in live:
            slot_rows = rows[slot]
            if len(slot_rows) != 1:
                raise ValueError(f"fixed-delay worker must publish one frame, got {len(slot_rows)} for {slot}")
            row = slot_rows[0]
            self._observe_frame(slot, row.frame_id, reset=row.reset)
            self.context._ingest_row(slot, row)
        actions = self._actions_for_live(live)
        return {slot: action[None] for slot, action in actions.items()}


BF16Inference = _o43.BF16Inference


def make_policy(
    model: GPT,
    stats: dict[str, FeatureStats],
    cfg: TrainConfig,
    *,
    prediction_horizon_frames: int | None = None,
    decode_seed: int | None = None,
    inference: Any | None = None,
    telemetry: DelayTelemetry | None = None,
    device: str = DEVICE,
    **legacy: Any,
) -> FixedPredictionDelayPolicy:
    if "exec_horizon" in legacy:
        prediction_horizon_frames = legacy.pop("exec_horizon")
    if legacy:
        raise TypeError(f"unexpected policy arguments: {sorted(legacy)}")
    horizon = cfg.prediction_horizon_frames if prediction_horizon_frames is None else prediction_horizon_frames
    if horizon != cfg.prediction_horizon_frames:
        raise ValueError("evaluation prediction horizon must match the trained horizon")
    engine = BF16Inference(model, cfg) if inference is None else inference
    streams = None if decode_seed is None else PredictionStepStreams(decode_seed, horizon)
    generator = None if decode_seed is None else torch.Generator(device=device).manual_seed(decode_seed)

    @torch.no_grad()
    def predict(ctx: Context) -> np.ndarray:
        started = time.perf_counter()
        result = engine.decode(ctx, horizon, streams=streams, gen=generator).cpu().numpy()
        if telemetry is not None:
            telemetry.record(rows=ctx.ctx_pad.shape[0], horizon=horizon, seconds=time.perf_counter() - started)
        return result

    return FixedPredictionDelayPolicy(
        predict,
        stats,
        cfg,
        telemetry=telemetry,
        device=device,
        float_dtype=next(model.parameters()).dtype,
    )


@dataclass(frozen=True, slots=True)
class EvalProtocol:
    suite: str
    fixed_ego_character: int | None
    n_matchups: int
    allowed_cpus: int
    hardware_wave_bucket: int
    max_parallel: int
    max_frames: int
    seed: int
    cpu_level: int
    ego_port: int
    seed_stage: int
    matchup_schedule_sha256: str
    oriented_pairs: int
    ego_characters: int
    cpu_characters: int
    prediction_delay_frames: int
    replan_interval_frames: int
    prediction_horizon_frames: int
    main_target_offsets_frames: tuple[int, ...]
    training_target_offsets_frames: tuple[int, ...]
    dtype: str
    inference_mode: str
    inference_compile_mode: str
    compiled_inference_bucket: int
    checkpoint_sha256: str
    bootstrap_resamples: int = _o43.BOOTSTRAP_RESAMPLES
    start_retries: int = _o43.DEFAULT_START_RETRIES


_O43_EVAL_PROTOCOL = _o43._eval_protocol


def _eval_protocol(
    cfg: TrainConfig,
    model: GPT,
    *,
    n_matchups: int,
    exec_horizon: int,
    checkpoint_sha256: str,
    inference_compile_mode: str = "reduce-overhead",
    fixed_ego_character: Any | None = None,
) -> EvalProtocol:
    if exec_horizon != cfg.prediction_horizon_frames:
        raise ValueError("evaluation prediction horizon must match the trained horizon")
    legacy = _O43_EVAL_PROTOCOL(
        cfg,
        model,
        n_matchups=n_matchups,
        exec_horizon=exec_horizon,
        checkpoint_sha256=checkpoint_sha256,
        inference_compile_mode=inference_compile_mode,
        fixed_ego_character=fixed_ego_character,
    )
    values = asdict(legacy)
    values.pop("exec_horizon")
    values.update(
        {
            "prediction_delay_frames": cfg.prediction_delay_frames,
            "replan_interval_frames": cfg.replan_interval_frames,
            "prediction_horizon_frames": cfg.prediction_horizon_frames,
            "main_target_offsets_frames": tuple(
                range(cfg.prediction_delay_frames, cfg.prediction_delay_frames + cfg.prediction_horizon_frames)
            ),
            "training_target_offsets_frames": prediction_target_offsets(cfg),
        }
    )
    return EvalProtocol(**values)


def _rename_eval_metrics(metrics: dict[str, float], protocol: EvalProtocol) -> None:
    metrics.pop("exec_horizon", None)
    metrics.update(
        {
            "prediction_delay_frames": float(protocol.prediction_delay_frames),
            "replan_interval_frames": float(protocol.replan_interval_frames),
            "prediction_horizon_frames": float(protocol.prediction_horizon_frames),
        }
    )


def _write_eval_evidence(
    replay_dir: Path,
    rows: list[Any],
    metrics: dict[str, float],
    protocol: EvalProtocol,
) -> None:
    _rename_eval_metrics(metrics, protocol)
    replay_dir.mkdir(parents=True, exist_ok=True)
    payloads = (
        (
            replay_dir / "match_rows.json",
            {"schema_version": 7, "protocol": asdict(protocol), "rows": [row.as_dict() for row in rows]},
        ),
        (replay_dir / "metrics.json", metrics),
    )
    for path, payload in payloads:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True))
        temporary.replace(path)


_O43_EVAL_VS_CPU = _o43.eval_vs_cpu


def eval_vs_cpu(
    model: GPT,
    stats: dict[str, FeatureStats],
    cfg: TrainConfig,
    *,
    n_matchups: int,
    replay_dir: Path,
    checkpoint_sha256: str = "unavailable",
    inference: Any | None = None,
    fixed_ego_character: Any | None = None,
) -> dict[str, float]:
    return _O43_EVAL_VS_CPU(
        model,
        stats,
        cfg,
        n_matchups=n_matchups,
        replay_dir=replay_dir,
        exec_horizon=cfg.prediction_horizon_frames,
        checkpoint_sha256=checkpoint_sha256,
        inference=inference,
        fixed_ego_character=fixed_ego_character,
    )


def eval_suites(
    model: GPT,
    stats: dict[str, FeatureStats],
    cfg: TrainConfig,
    *,
    n_matchups: int,
    replay_dir: Path,
    checkpoint_sha256: str,
    inference: Any,
    **legacy: Any,
) -> dict[str, dict[str, float]]:
    if legacy and legacy != {"exec_horizon": cfg.prediction_horizon_frames}:
        raise TypeError(f"unexpected evaluation arguments: {legacy}")
    return {
        name: eval_vs_cpu(
            model,
            stats,
            cfg,
            n_matchups=n_matchups,
            replay_dir=replay_dir / name,
            checkpoint_sha256=checkpoint_sha256,
            inference=inference,
            fixed_ego_character=character,
        )
        for name, character in _o43._EVAL_SUITES
    }


def model_tag(cfg: TrainConfig) -> str:
    loss = round(100 * cfg.first_prediction_loss_share)
    return (
        f"mtp044-fixed-delay-v{cfg.codec_version}-d{cfg.d_model}-L{cfg.n_layers}-h{cfg.n_heads}-Lc{cfg.L_ctx}-"
        f"t{cfg.temporal_d_model}x{cfg.temporal_layers}-pd{cfg.prediction_delay_frames}-"
        f"r{cfg.replan_interval_frames}-ph{cfg.prediction_horizon_frames}-p0w{loss:02d}-{cfg.observation_bundle}"
    )


def _init_wandb(cfg: TrainConfig, run_name: str, resume_state: dict | None) -> None:
    wandb.init(
        project="hal",
        name=run_name,
        id=None if resume_state is None else resume_state.get("wandb_id"),
        resume="allow" if resume_state is not None else None,
        tags=["044", "gpt", "fixed-prediction-delay", "temporal-mtp", "legacy-codec"],
        config=asdict(cfg),
        settings=wandb.Settings(x_stats_sampling_interval=5.0, x_stats_track_process_tree=True),
    )
    if wandb.run is None:
        return
    wandb.define_metric("global_step")
    wandb.define_metric("*", step_metric="global_step")
    wandb.run.summary["prediction/target_interval"] = "[t+d, t+d+H)"
    wandb.run.summary["prediction/training_target_offsets_frames"] = list(prediction_target_offsets(cfg))
    wandb.run.summary["prediction/future_target_buffer_frames"] = FUTURE_TARGET_BUFFER_FRAMES
    wandb.run.summary["evaluation/suites"] = "char_matchup,fox"
    wandb.run.summary["training/updates"] = cfg.max_steps
    wandb.run.summary["data/nominal_samples"] = cfg.max_steps * cfg.batch_size
    wandb.run.summary["data/max_context_prefixes"] = cfg.max_steps * cfg.batch_size * cfg.L_ctx
    if cfg.wandb_log_code:
        _o43.log_wandb_code(wandb.run)


_CHECKPOINT_ARCH_FIELDS = {
    "codec_version",
    "decoder_arch_version",
    "prediction_delay_frames",
    "prediction_horizon_frames",
    "prediction_steps_frames",
    "temporal_d_model",
    "temporal_layers",
    "temporal_heads",
    "temporal_ff_dim",
    "group_head_dim",
    "action_embed_dim",
    "offset_embed_dim",
    "group_order",
    "observation_bundle",
}


def config_from_state(values: dict[str, Any]) -> TrainConfig:
    missing = _CHECKPOINT_ARCH_FIELDS - values.keys()
    if missing:
        raise ValueError(f"checkpoint is not an experiment-044 architecture; missing {sorted(missing)}")
    prediction_steps = tuple(values["prediction_steps_frames"])
    if prediction_steps != PREDICTION_STEPS:
        raise ValueError(
            f"checkpoint prediction steps {prediction_steps} do not match experiment 044 {PREDICTION_STEPS}"
        )
    known = {item.name for item in fields(TrainConfig)}
    return TrainConfig(**{name: value for name, value in values.items() if name in known})


_O43_SAVE_CHECKPOINT = _o43.save_checkpoint


def save_checkpoint(path: Path, **kwargs: Any) -> None:
    checkpoint_config = dict(kwargs["cfg"])
    checkpoint_config["prediction_steps_frames"] = PREDICTION_STEPS
    _O43_SAVE_CHECKPOINT(path, **{**kwargs, "cfg": checkpoint_config})


def load_checkpoint(path: str, *, device: str = DEVICE) -> tuple[GPT, TrainConfig, dict[str, FeatureStats], dict]:
    state = torch.load(path, map_location=device, weights_only=False)
    cfg = config_from_state(state["cfg"])
    validate_config(cfg)
    model = GPT(cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    stats = load_consolidated_stats(Path(cfg.data_root) / "stats.json")
    return model, cfg, stats, state


def eval_checkpoint(
    path: str,
    *,
    n_matchups: int | None = None,
    eager: bool = False,
    max_parallel: int | None = None,
    output_name: str | None = None,
    upload_run: str | None = None,
    backfill_wandb: bool = False,
) -> dict[str, dict[str, float]]:
    model, cfg, stats, state = load_checkpoint(path)
    cfg = replace(
        cfg,
        inference_mode="eager" if eager else cfg.inference_mode,
        eval_max_parallel=cfg.eval_max_parallel if max_parallel is None else max_parallel,
    )
    validate_config(cfg)
    step = int(state["step"])
    matchups = cfg.final_eval_n_matchups if n_matchups is None else n_matchups
    label = (
        f"d{cfg.prediction_delay_frames:02d}_r{cfg.replan_interval_frames:02d}_h{cfg.prediction_horizon_frames:02d}"
    )
    default_name = f"eval_backfill_step_{step:07d}_{label}" if upload_run else f"eval_replays_{label}"
    if output_name is not None and (Path(output_name).name != output_name or output_name in ("", ".", "..")):
        raise ValueError(f"evaluation output name must be one directory name, got {output_name!r}")
    replay_dir = Path(path).resolve().parent / (default_name if output_name is None else output_name)
    inference = BF16Inference(model, cfg)
    suites = eval_suites(
        model,
        stats,
        cfg,
        n_matchups=matchups,
        replay_dir=replay_dir,
        checkpoint_sha256=_o43._checkpoint_sha256(Path(path)),
        inference=inference,
    )
    for values in suites.values():
        _o43.require_complete_eval(values, matchups)
    if upload_run is not None:
        _o43._upload_eval_evidence(upload_run, replay_dir)
    if backfill_wandb:
        wandb_id = state.get("wandb_id")
        if not isinstance(wandb_id, str):
            raise RuntimeError("checkpoint has no W&B run id to backfill")
        _o43._backfill_eval_metrics(wandb_id, step, suites)
    print(f"[eval] step={step} {label}: {suites}", flush=True)
    return suites


# The O43 module is loaded under a private name, so replacing its globals does
# not mutate a separately imported O43 experiment. Its training loop then uses
# O44's model, target alignment, metrics, policy, and public metadata.
_o43.__file__ = __file__
_o43.TrainConfig = TrainConfig
_o43.GPT = GPT
_o43.validate_config = validate_config
_o43.prepared_targets = prepared_targets
_o43.chunk_targets = chunk_targets
_o43.nll_mean_metrics = nll_mean_metrics
_o43.nll_metrics = nll_metrics
_o43._transition_metrics = _transition_metrics
_o43.val_metrics = val_metrics
_o43.DecodeTelemetry = DelayTelemetry
_o43.make_policy = make_policy
_o43._eval_protocol = _eval_protocol
_o43._write_eval_evidence = _write_eval_evidence
_o43.eval_vs_cpu = eval_vs_cpu
_o43.eval_suites = eval_suites
_o43.model_tag = model_tag
_o43._init_wandb = _init_wandb
_o43.config_from_state = config_from_state
_o43.load_checkpoint = load_checkpoint
_o43.save_checkpoint = save_checkpoint

ActionLoss = _o43.ActionLoss
action_loss = _o43.action_loss
amp_context = _o43.amp_context
decode_chunk = _o43.decode_chunk
dequantize = _o43.dequantize
micro_batch_size = _o43.micro_batch_size
quantize = _o43.quantize
require_complete_eval = _o43.require_complete_eval
run_benchmark = _o43.run_benchmark
synthetic_context = _o43.synthetic_context
train = _o43.train

# Codec and model constants are part of the experiment test surface.
GROUP_NAMES = _o43.GROUP_NAMES
GROUP_ORDER = _o43.GROUP_ORDER
GROUP_INDEX = _o43.GROUP_INDEX
GROUP_VOCABS = _o43.GROUP_VOCABS
LEGACY_GROUP_VOCABS = _o43.LEGACY_GROUP_VOCABS
N_GROUPS = _o43.N_GROUPS
StructuredControllerCodec = _o43.StructuredControllerCodec


@dataclass
class Args:
    cfg: TrainConfig = field(default_factory=TrainConfig)
    comment: str = ""
    resume: str | None = None
    eval: str | None = None
    eval_run: str | None = None
    eval_n_matchups: int | None = None
    eval_eager: bool = False
    eval_max_parallel: int | None = None
    eval_output_name: str | None = None
    eval_backfill_wandb: bool = False
    self_play_eval: str | None = None
    self_play_matches: int = 12
    self_play_frames: int = 14_400
    self_play_eager: bool = False
    self_play_instant_match_restart: bool = False
    self_play_process_cohorts: int = 1
    self_play_cohort_sweep: bool = False
    benchmark: bool = False
    benchmark_iterations: int = 20


def _resolve_eval_checkpoint(checkpoint: str, run: str | None) -> Path:
    if run is None:
        return Path(checkpoint)
    path = download_latest(run, Path("runs") / run, name=checkpoint)
    if path is None:
        raise SystemExit(f"no {checkpoint!r} for run {run!r}")
    return path


def main(args: Args) -> None:
    if args.benchmark:
        if args.eval is not None or args.resume is not None or args.self_play_eval is not None:
            raise SystemExit("--benchmark cannot be combined with --eval, --self-play-eval, or --resume")
        run_benchmark(args.cfg, iterations=args.benchmark_iterations)
        return
    selected = sum(value is not None for value in (args.eval, args.self_play_eval, args.resume))
    if selected > 1:
        raise SystemExit("pass only one of --eval, --self-play-eval, or --resume")
    if args.eval is not None:
        checkpoint = _resolve_eval_checkpoint(args.eval, args.eval_run)
        eval_checkpoint(
            str(checkpoint),
            n_matchups=args.eval_n_matchups,
            eager=args.eval_eager,
            max_parallel=args.eval_max_parallel,
            output_name=args.eval_output_name,
            upload_run=args.eval_run,
            backfill_wandb=args.eval_backfill_wandb,
        )
        return
    if args.self_play_eval is not None:
        cohorts = (1, 2, 3, 4) if args.self_play_cohort_sweep else (args.self_play_process_cohorts,)
        for cohort_count in cohorts:
            benchmark_self_play(
                args.self_play_eval,
                load_checkpoint=load_checkpoint,
                make_inference=BF16Inference,
                make_policy=make_policy,
                n_matches=args.self_play_matches,
                max_frames=args.self_play_frames,
                eager=args.self_play_eager,
                instant_match_restart=args.self_play_instant_match_restart,
                process_cohorts=cohort_count,
            )
        return
    resume_run = resume_state = None
    cfg = args.cfg
    if args.resume is not None:
        resume_state = load_for_resume(args.resume, Path("runs") / args.resume, device=DEVICE)
        if resume_state is None:
            raise SystemExit(f"no latest.pt for run {args.resume!r}")
        resume_run = args.resume
        cfg = config_from_state(resume_state["cfg"])
    stats = load_consolidated_stats(Path(cfg.data_root) / "stats.json")
    train(cfg, stats, comment=args.comment, resume_run=resume_run, resume_state=resume_state)


if __name__ == "__main__":
    main(tyro.cli(Args))
