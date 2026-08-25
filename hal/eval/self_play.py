"""Shared checkpoint self-play benchmark machinery."""

import hashlib
import itertools
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hal.eval.h2h import mirrored_configs
from hal.eval.harness import default_session_cfg
from hal.eval.harness import run_matches_vec
from hal.sim.process_vec import ProcessVecTelemetry
from hal.sim.rollout import covering_power_of_two
from hal.sim.session import Matchup
from hal.sim.session import PlayerSetup
from hal.sim.vec import VecMatch
from hal.training.features import ACTION_CHANNELS
from hal.training.features import BASE_PLAYER_PREFIXES
from hal.training.features import CAT_FEATURES
from hal.training.features import FLOAT_FEATURES
from hal.training.features import ITEM_COLUMNS
from hal.training.features import SPATIAL_COLUMNS_LEAN
from hal.training.features import V6_PLAYER_COLUMNS
from hal.training.features import Context
from hal.wire import ITEM_SLOTS
from hal.wire import item_column


@dataclass
class DecodeTelemetry:
    """Timing and batch-size statistics for policy decode calls."""

    calls: int = 0
    rows: int = 0
    executed_frames: int = 0
    seconds: float = 0.0
    max_seconds: float = 0.0
    durations: list[float] = field(default_factory=list)

    def record(self, *, rows: int, horizon: int, seconds: float) -> None:
        self.calls += 1
        self.rows += rows
        self.executed_frames += rows * horizon
        self.seconds += seconds
        self.max_seconds = max(self.max_seconds, seconds)
        self.durations.append(seconds)

    def metrics(self) -> dict[str, float]:
        durations = np.asarray(self.durations, dtype=np.float64)
        return {
            "decode_calls": float(self.calls),
            "decode_rows": float(self.rows),
            "decode_seconds": self.seconds,
            "decode_max_seconds": self.max_seconds,
            "decode_p50_ms": float(np.percentile(durations, 50) * 1000) if durations.size else 0.0,
            "decode_p95_ms": float(np.percentile(durations, 95) * 1000) if durations.size else 0.0,
            "decode_p99_ms": float(np.percentile(durations, 99) * 1000) if durations.size else 0.0,
            "decode_calls_over_100ms": float(np.count_nonzero(durations > 0.1)),
            "decode_mean_rows": self.rows / max(self.calls, 1),
            "decode_replans_per_s": self.calls / max(self.seconds, 1e-12),
            "decode_executed_frames_per_s": self.executed_frames / max(self.seconds, 1e-12),
        }


def synthetic_context(cfg: Any, batch_size: int, device: torch.device) -> Context:
    """Build an all-zero context with the feature layout selected by ``cfg``."""
    features: dict[str, torch.Tensor] = {}
    v6_floats = tuple(V6_PLAYER_COLUMNS.floats)
    v6_categories = V6_PLAYER_COLUMNS.cats
    floats = FLOAT_FEATURES if cfg.observation_bundle == "base" else FLOAT_FEATURES + v6_floats
    for prefix in BASE_PLAYER_PREFIXES:
        for name in floats:
            features[f"{prefix}_{name}"] = torch.zeros(batch_size, cfg.L_ctx, device=device)
            features[f"{prefix}_{name}_mask"] = torch.zeros(batch_size, cfg.L_ctx, device=device)
        for name in CAT_FEATURES:
            features[f"{prefix}_{name}"] = torch.zeros(batch_size, cfg.L_ctx, dtype=torch.long, device=device)
        if cfg.observation_bundle == "v6_lean":
            for name in v6_categories:
                features[f"{prefix}_{name}"] = torch.zeros(
                    batch_size,
                    cfg.L_ctx,
                    dtype=torch.long,
                    device=device,
                )
    for name in ACTION_CHANNELS:
        features[f"ego_{name}"] = torch.zeros(batch_size, cfg.L_ctx, device=device)
    features["ego_character"] = torch.zeros(batch_size, cfg.L_ctx, dtype=torch.long, device=device)
    features["opp_character"] = torch.zeros(batch_size, cfg.L_ctx, dtype=torch.long, device=device)
    features["stage"] = torch.zeros(batch_size, cfg.L_ctx, dtype=torch.long, device=device)
    if cfg.observation_bundle == "v6_lean":
        for name in SPATIAL_COLUMNS_LEAN:
            features[name] = torch.zeros(batch_size, cfg.L_ctx, device=device)
    if getattr(cfg, "item_conditioning", False):
        for slot in range(ITEM_SLOTS):
            for name in ITEM_COLUMNS.cats:
                features[item_column(slot, name)] = torch.zeros(batch_size, cfg.L_ctx, dtype=torch.long, device=device)
            for name in ITEM_COLUMNS.floats:
                column = item_column(slot, name)
                features[column] = torch.zeros(batch_size, cfg.L_ctx, device=device)
                features[f"{column}_mask"] = torch.zeros(batch_size, cfg.L_ctx, device=device)
    return Context(
        features=features,
        ctx_pad=torch.zeros(batch_size, dtype=torch.long, device=device),
        slot_ids=torch.arange(batch_size, dtype=torch.long, device=device),
        reset=torch.ones(batch_size, dtype=torch.bool, device=device),
    )


def canonical_context(ctx: Context, observation_bundle: str, *, items: bool = False) -> Context:
    """Match every live decode's feature keys to the synthetic prewarm context.

    Missing mask sidecars mean zero, but Dynamo guards on dictionary membership and
    key order. Filling and sorting them prevents a live decode from compiling a
    second, untested program after prewarm.

    ``items`` extends the same fill to the projectile block's float sidecars, for the
    model that conditions on it.
    """
    floats = FLOAT_FEATURES
    if observation_bundle != "base":
        floats += tuple(V6_PLAYER_COLUMNS.floats)
    features = dict(ctx.features)
    for prefix in BASE_PLAYER_PREFIXES:
        for name in floats:
            key = f"{prefix}_{name}_mask"
            if key not in features:
                features[key] = torch.zeros_like(features[f"{prefix}_{name}"])
    if items:
        for slot in range(ITEM_SLOTS):
            for name in ITEM_COLUMNS.floats:
                column = item_column(slot, name)
                if f"{column}_mask" not in features:
                    features[f"{column}_mask"] = torch.zeros_like(features[column])
    return replace(ctx, features={name: features[name] for name in sorted(features)})


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _self_play_matches(n_matches: int) -> list[VecMatch]:
    return [
        VecMatch(
            matchup=Matchup(
                stage=config.stage,
                players=(
                    PlayerSetup(port=1, character=config.character_port_1, cpu_level=0),
                    PlayerSetup(port=2, character=config.character_port_2, cpu_level=0),
                ),
            ),
            model_ports=(1, 2),
        )
        for config in mirrored_configs(n_matches)
    ]


def _output_directory(
    checkpoint: Path,
    *,
    n_matches: int,
    max_frames: int,
    instant_match_restart: bool,
    process_cohorts: int | None,
) -> Path:
    mode = "instant_restart" if instant_match_restart else "single_match"
    base_name = f"self_play_benchmark_{n_matches}x{max_frames}_{mode}"
    if process_cohorts is not None:
        base_name += f"_c{process_cohorts}"
    run_number = 1
    while True:
        suffix = "" if run_number == 1 else f"_run{run_number:02d}"
        out_dir = checkpoint.parent / f"{base_name}{suffix}"
        try:
            out_dir.mkdir(parents=True, exist_ok=False)
            return out_dir
        except FileExistsError:
            run_number += 1


def _prewarm(inference: Any, cfg: Any, inference_bucket: int) -> float:
    # Fixed-bucket engines already own their compile lifecycle. Bucket-list
    # engines compile on decode, so the benchmark drives their exact program.
    native_prewarm = getattr(inference, "prewarm", None)
    if native_prewarm is not None:
        return native_prewarm(cfg.exec_horizon)
    device = next(inference.model.parameters()).device
    started = time.perf_counter()
    context = synthetic_context(cfg, inference_bucket, device)
    inference.decode(context, cfg.exec_horizon)
    inference.decode(context, cfg.exec_horizon)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter() - started


def benchmark_checkpoint(
    path: str,
    *,
    load_checkpoint: Callable[[str], tuple[Any, Any, Any, dict[str, Any]]],
    make_inference: Callable[..., Any],
    make_policy: Callable[..., Any],
    n_matches: int = 12,
    max_frames: int = 14_400,
    eager: bool = False,
    instant_match_restart: bool = False,
    process_cohorts: int | None = None,
) -> dict[str, float]:
    """Benchmark one fixed wave with a checkpoint controlling both ports."""
    if n_matches < 1:
        raise ValueError(f"n_matches must be >= 1, got {n_matches}")
    if max_frames < 2:
        raise ValueError(f"max_frames must be >= 2, got {max_frames}")

    model, cfg, stats, state = load_checkpoint(path)
    if eager:
        cfg = replace(cfg, inference_mode="eager")
    model.eval()
    real_rows = 2 * n_matches
    inference_bucket = covering_power_of_two(real_rows)
    inference = make_inference(model, cfg, bucket=inference_bucket, compile_mode="default")
    telemetry = DecodeTelemetry()
    compile_seconds = _prewarm(inference, cfg, inference_bucket)
    matches = _self_play_matches(n_matches)

    checkpoint = Path(path).resolve()
    out_dir = _output_directory(
        checkpoint,
        n_matches=n_matches,
        max_frames=max_frames,
        instant_match_restart=instant_match_restart,
        process_cohorts=process_cohorts,
    )
    policy_index = itertools.count()

    def policy_factory() -> Any:
        return make_policy(
            model,
            stats,
            cfg,
            decode_seed=cfg.eval_seed + next(policy_index),
            inference=inference,
            telemetry=telemetry,
        )

    rollout_started = time.perf_counter()
    process_telemetry = ProcessVecTelemetry()
    boots = run_matches_vec(
        default_session_cfg(out_dir / "replays", instant_match_restart=instant_match_restart),
        matches,
        policy_factory,
        max_frames=max_frames,
        max_parallel=covering_power_of_two(n_matches),
        start_retries=0,
        process_telemetry=process_telemetry,
        process_cohorts=1 if process_cohorts is None else process_cohorts,
    )
    rollout_seconds = time.perf_counter() - rollout_started
    completed_boots = sum(bool(boot) for boot in boots)
    captured_frames = sum(len(trajectory) for boot in boots for trajectory in boot)
    metrics = {
        "checkpoint_step": float(state["step"]),
        "workers": float(n_matches),
        "model_slots": float(real_rows),
        "max_frames_per_worker": float(max_frames),
        "completed_workers": float(completed_boots),
        "captured_frames": float(captured_frames),
        "inference_bucket": float(inference_bucket),
        "compile_seconds": compile_seconds,
        "rollout_seconds": rollout_seconds,
        "aggregate_fps": captured_frames / max(rollout_seconds, 1e-12),
        "wall_lockstep_fps": captured_frames / max(completed_boots, 1) / max(rollout_seconds, 1e-12),
        **telemetry.metrics(),
        **process_telemetry.metrics(),
    }
    payload = {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _checkpoint_sha256(checkpoint),
        "instant_match_restart": instant_match_restart,
        "metrics": metrics,
    }
    if process_cohorts is not None:
        payload["process_cohorts"] = process_cohorts
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    (out_dir / "metrics.json").write_text(rendered)
    print(rendered, flush=True)
    return metrics
