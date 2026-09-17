"""Evaluate an O52 value head over a Slippi replay for Slippilab."""

import contextlib
import hashlib
import importlib
import json
import math
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Annotated
from typing import Any

import numpy as np
import torch
import tyro
from botocore.exceptions import BotoCoreError
from botocore.exceptions import ClientError
from loguru import logger

from hal import r2
from hal.data.extract import extract_replay
from hal.inference.checkpoints import parse_r2_uri
from hal.inference.checkpoints import resolve_checkpoint
from hal.paths import REPO_DIR
from hal.scripts.slp_link import link_replay
from hal.training.checkpoints import checkpoint_sha256
from hal.training.dataloader import collate_train_batch
from hal.training.dataloader import make_window
from hal.training.features import BASE_ITEMS_PROJECTION
from hal.training.features import ITEM_PLAYER_COLUMNS
from hal.training.features import ITEM_PLAYER_PROJECTION
from hal.training.returns import frame_reward

_EXPERIMENT_MODULE = "experiments.052_adamw_temporal_awr"
_EXPERIMENT_ID = "052_adamw_temporal_awr_v1"
_EXPERIMENT_SOURCE_SHA256 = "d0e75291dc3e0c322aea0cdd8b57573c5fec20b2c843b71943b8e3777f247b69"
_SCHEMA_VERSION = 2
_DOWNLOAD_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class ResolvedReplay:
    path: Path
    source: str
    sha256: str
    size: int
    etag: str | None


@dataclass(frozen=True, slots=True)
class ValueSeries:
    p1: tuple[float, ...]
    p2: tuple[float, ...]
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class RetrospectiveTrace:
    value: tuple[float, ...]
    reward: tuple[float, ...]
    gae_lambda: float
    decay: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(_DOWNLOAD_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_cached_replay(path: Path, metadata_path: Path, expected: dict[str, object]) -> str | None:
    try:
        metadata = json.loads(metadata_path.read_text())
    except OSError, json.JSONDecodeError:
        return None
    if not isinstance(metadata, dict) or any(metadata.get(key) != value for key, value in expected.items()):
        return None
    digest = metadata.get("sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        return None
    try:
        if path.stat().st_size != expected["size"] or _sha256(path) != digest:
            return None
    except OSError:
        return None
    return digest


def resolve_replay(source: str, *, cache_root: str | Path = "runs") -> ResolvedReplay:
    """Resolve a local replay or download one exact R2 object into a verified cache."""
    if not source.startswith("r2://"):
        path = Path(source)
        if path.suffix.lower() != ".slp" or not path.is_file():
            raise FileNotFoundError(f"replay does not exist or is not a .slp file: {path}")
        path = path.resolve()
        return ResolvedReplay(path, source, _sha256(path), path.stat().st_size, None)

    bucket, key = parse_r2_uri(source)
    if not key.endswith(".slp"):
        raise ValueError(f"R2 replay must name one .slp object: {source!r}")
    try:
        with contextlib.closing(r2.client()) as client:
            remote = client.head_object(Bucket=bucket, Key=key)
            etag = remote.get("ETag")
            size = remote.get("ContentLength")
            if not isinstance(etag, str) or not isinstance(size, int) or isinstance(size, bool) or size < 1:
                raise RuntimeError(f"R2 returned invalid identity metadata for {source}")

            cache_key = hashlib.sha256(source.encode()).hexdigest()
            cache_dir = Path(cache_root) / "r2-replays" / cache_key
            path = cache_dir / Path(key).name
            metadata_path = cache_dir / f"{path.name}.metadata.json"
            expected: dict[str, object] = {"uri": source, "etag": etag, "size": size}
            digest = _valid_cached_replay(path, metadata_path, expected)
            if digest is None:
                cache_dir.mkdir(parents=True, exist_ok=True)
                partial = path.with_suffix(path.suffix + ".partial")
                partial.unlink(missing_ok=True)
                hasher = hashlib.sha256()
                downloaded = 0
                try:
                    response = client.get_object(Bucket=bucket, Key=key)
                    if response.get("ETag") not in (None, etag):
                        raise RuntimeError(f"R2 object changed while downloading {source}")
                    body = response["Body"]
                    with contextlib.closing(body), partial.open("wb") as output:
                        while chunk := body.read(_DOWNLOAD_CHUNK_BYTES):
                            output.write(chunk)
                            hasher.update(chunk)
                            downloaded += len(chunk)
                    if downloaded != size:
                        raise RuntimeError(f"R2 object size mismatch for {source}: expected {size}, got {downloaded}")
                    os.replace(partial, path)
                finally:
                    partial.unlink(missing_ok=True)
                digest = hasher.hexdigest()
                metadata_partial = metadata_path.with_suffix(metadata_path.suffix + ".partial")
                metadata_partial.write_text(
                    json.dumps({**expected, "sha256": digest}, separators=(",", ":"), sort_keys=True)
                )
                os.replace(metadata_partial, metadata_path)
    except (BotoCoreError, ClientError) as error:
        raise RuntimeError(f"failed to download {source}: {error}") from error
    return ResolvedReplay(path.resolve(), source, digest, size, etag)


def _load_experiment() -> ModuleType:
    experiment = importlib.import_module(_EXPERIMENT_MODULE)
    source_path = Path(experiment.__file__ or "")
    digest = _sha256(source_path)
    if digest != _EXPERIMENT_SOURCE_SHA256:
        raise RuntimeError(f"O52 source SHA-256 {digest} does not match checkpoint source {_EXPERIMENT_SOURCE_SHA256}")
    return experiment


def _validate_frames(sample: dict[str, np.ndarray]) -> np.ndarray:
    frames = sample.get("frame")
    if frames is None or frames.ndim != 1 or len(frames) == 0:
        raise ValueError("replay extraction produced no one-dimensional frame array")
    gaps = np.flatnonzero(np.diff(frames.astype(np.int64, copy=False)) != 1)
    if len(gaps):
        index = int(gaps[0])
        raise ValueError(f"replay frames are not contiguous at {int(frames[index])} -> {int(frames[index + 1])}")
    return frames


def _context_window(
    sample: dict[str, np.ndarray],
    *,
    ego_prefix: str,
    end_index: int,
    context_length: int,
) -> dict[str, np.ndarray | np.integer]:
    start = end_index + 1 - context_length
    pad = max(0, -start)
    window = make_window(
        sample,
        ego_prefix=ego_prefix,
        start=start,
        pad=pad,
        length=context_length,
        projection=BASE_ITEMS_PROJECTION,
    )
    window["ego_player_id"] = np.zeros(context_length, dtype=np.int64)
    window["ctx_pad"] = np.int64(pad)
    return window


def _pair_values(values: np.ndarray, frame_count: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if values.shape != (2 * frame_count,):
        raise ValueError(f"value output has shape {values.shape}, expected {(2 * frame_count,)}")
    paired = values.reshape(frame_count, 2)
    if not np.isfinite(paired).all():
        raise ValueError("value head produced a non-finite estimate")
    return tuple(float(value) for value in paired[:, 0]), tuple(float(value) for value in paired[:, 1])


def retrospective_trace(
    series: ValueSeries,
    reward: np.ndarray,
    *,
    gamma: float,
    half_life_frames: float,
) -> RetrospectiveTrace:
    """Return ``V + GAE`` for the realized replay trajectory.

    The O52 head at frame ``t`` predicts ``G_{t+1}``, so its Bellman residual
    uses the reward that first becomes visible at frame ``t+1``. The trace is
    retrospective: it intentionally carries later surprises back to earlier
    replay actions and must not be presented as a live estimate.
    """
    if not math.isfinite(half_life_frames) or half_life_frames <= 0:
        raise ValueError(f"credit half-life must be finite and positive, got {half_life_frames}")
    if not 0.0 < gamma < 1.0:
        raise ValueError(f"gamma must be in (0, 1), got {gamma}")
    values = (np.asarray(series.p1, dtype=np.float64) - np.asarray(series.p2, dtype=np.float64)) / 2.0
    rewards = np.asarray(reward, dtype=np.float64)
    if rewards.shape != values.shape:
        raise ValueError(f"reward shape {rewards.shape} does not match value shape {values.shape}")
    if not np.isfinite(values).all() or not np.isfinite(rewards).all():
        raise ValueError("retrospective trace inputs must be finite")
    decay = math.exp(-math.log(2.0) / half_life_frames)
    gae_lambda = decay / gamma
    if gae_lambda > 1.0:
        maximum = math.log(2.0) / -math.log(gamma)
        raise ValueError(f"credit half-life {half_life_frames} exceeds the gamma-limited maximum {maximum:.1f} frames")
    advantage = np.zeros(values.shape, dtype=np.float64)
    for index in range(len(values) - 2, -1, -1):
        delta = rewards[index + 1] + gamma * values[index + 1] - values[index]
        advantage[index] = delta + decay * advantage[index + 1]
    hindsight = values + advantage
    return RetrospectiveTrace(
        value=tuple(float(value) for value in hindsight),
        reward=tuple(float(value) for value in rewards),
        gae_lambda=gae_lambda,
        decay=decay,
    )


def score_replay(
    sample: dict[str, np.ndarray],
    *,
    model: Any,
    cfg: Any,
    stats: dict[str, Any],
    experiment: ModuleType,
    device: torch.device,
    batch_size: int,
) -> ValueSeries:
    """Return O52 values in frame-major P1/P2 order."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    frames = _validate_frames(sample)
    context_length = int(cfg.arch.L_ctx)
    p1: list[float] = []
    p2: list[float] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for first in range(0, len(frames), batch_size):
            stop = min(first + batch_size, len(frames))
            windows = [
                _context_window(
                    sample,
                    ego_prefix=ego_prefix,
                    end_index=frame_index,
                    context_length=context_length,
                )
                for frame_index in range(first, stop)
                for ego_prefix in ("p1", "p2")
            ]
            batch = collate_train_batch(
                windows,
                stats=stats,
                L_ctx=context_length,
                extra=ITEM_PLAYER_COLUMNS,
                projection=ITEM_PLAYER_PROJECTION,
            ).to(device)
            with experiment.amp_context(cfg, device):
                hidden = model.forward_dense(batch.context.features, batch.context.ctx_pad)
            features = experiment.decoder_rmsnorm(hidden[:, -1]).detach()
            values = model.value_head(features.float()).squeeze(-1).cpu().numpy()
            batch_p1, batch_p2 = _pair_values(values, stop - first)
            p1.extend(batch_p1)
            p2.extend(batch_p2)
    return ValueSeries(tuple(p1), tuple(p2), time.perf_counter() - started)


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def build_sidecar(
    *,
    checkpoint_source: str,
    checkpoint_path: Path,
    replay: ResolvedReplay,
    sample: dict[str, np.ndarray],
    series: ValueSeries,
    cfg: Any,
    state: dict[str, Any],
    credit_half_life_frames: float,
) -> dict[str, object]:
    frames = _validate_frames(sample)
    if len(series.p1) != len(frames) or len(series.p2) != len(frames):
        raise ValueError("value series length does not match replay frame count")
    checkpoint_config = state.get("cfg")
    if not isinstance(checkpoint_config, dict) or checkpoint_config.get("experiment_id") != _EXPERIMENT_ID:
        raise ValueError(f"checkpoint is not {_EXPERIMENT_ID}")
    step = state.get("step")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError(f"checkpoint has invalid step {step!r}")
    reward = frame_reward(
        sample,
        ego="p1",
        opp="p2",
        damage_shaping=cfg.awr.damage_shaping,
        win_reward=cfg.awr.win_reward,
        stock_value=cfg.awr.stock_value,
    )
    credit = retrospective_trace(
        series,
        reward,
        gamma=float(cfg.awr.gamma),
        half_life_frames=credit_half_life_frames,
    )
    return {
        "schema_version": _SCHEMA_VERSION,
        "provenance": {
            "hal_git_sha": _git_sha(),
            "experiment_source_sha256": _EXPERIMENT_SOURCE_SHA256,
        },
        "checkpoint": {
            "source": checkpoint_source,
            "sha256": checkpoint_sha256(checkpoint_path),
            "experiment_id": _EXPERIMENT_ID,
            "step": step,
        },
        "replay": {
            "source": replay.source,
            "sha256": replay.sha256,
            "size": replay.size,
            "etag": replay.etag,
            "first_frame": int(frames[0]),
            "frame_count": len(frames),
        },
        "value": {
            "target": "G_t+1",
            "gamma": float(cfg.awr.gamma),
            "stock_value": float(cfg.awr.stock_value),
            "damage_shaping": float(cfg.awr.damage_shaping),
            "win_reward": float(cfg.awr.win_reward),
            "identity": "masked",
            "projection": "(p1-p2)/2",
        },
        "credit": {
            "method": "retrospective_v_plus_gae",
            "half_life_frames": credit_half_life_frames,
            "lambda": credit.gae_lambda,
            "gamma_lambda": credit.decay,
            "reward_alignment": "delta_t uses reward_t+1",
        },
        "analysis": {
            "elapsed_seconds": series.elapsed_seconds,
            "frames_per_second": len(frames) / series.elapsed_seconds,
        },
        "series": {
            "p1": series.p1,
            "p2": series.p2,
            "reward_p1": credit.reward,
            "hindsight_p1": credit.value,
        },
    }


def value_meter(
    checkpoint: Annotated[str, tyro.conf.arg(help="Exact local .pt path or r2://bucket/key")],
    replay: Annotated[str, tyro.conf.arg(help="Exact local .slp path or r2://bucket/key")],
    output_dir: Annotated[str, tyro.conf.arg(help="New directory for the versioned sidecar")],
    device: str = "cuda",
    batch_size: int = 64,
    credit_half_life_frames: float = 45.0,
    slippilab_url: str = "http://127.0.0.1:5173",
) -> None:
    """Analyze one replay and print its local Slippilab URL."""
    destination = Path(output_dir)
    if destination.exists():
        raise SystemExit(f"output directory already exists: {destination}")
    if batch_size < 1:
        raise SystemExit(f"--batch-size must be positive, got {batch_size}")
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")

    experiment = _load_experiment()
    checkpoint_path = resolve_checkpoint(checkpoint)
    replay_artifact = resolve_replay(replay)
    logger.info(f"loading checkpoint {checkpoint_path}")
    model, cfg, stats, state = experiment.load_checkpoint(str(checkpoint_path), device=str(target_device))
    sample = extract_replay(str(replay_artifact.path))
    if sample is None:
        raise SystemExit(f"failed to extract a canonical 1v1 replay from {replay_artifact.path}")
    series = score_replay(
        sample,
        model=model,
        cfg=cfg,
        stats=stats,
        experiment=experiment,
        device=target_device,
        batch_size=batch_size,
    )
    sidecar = build_sidecar(
        checkpoint_source=checkpoint,
        checkpoint_path=checkpoint_path,
        replay=replay_artifact,
        sample=sample,
        series=series,
        cfg=cfg,
        state=state,
        credit_half_life_frames=credit_half_life_frames,
    )
    destination.mkdir(parents=True)
    sidecar_path = destination / "advantage.json"
    sidecar_path.write_text(json.dumps(sidecar, allow_nan=False, indent=2) + "\n")
    viewer_url = link_replay(replay_artifact.path, advantage=sidecar_path, slippilab_url=slippilab_url)
    logger.info(
        f"scored {len(series.p1)} frames in {series.elapsed_seconds:.2f}s "
        f"({len(series.p1) / series.elapsed_seconds:.1f} frames/s)"
    )
    print(viewer_url)


def main() -> None:
    tyro.cli(value_meter)


if __name__ == "__main__":
    main()
