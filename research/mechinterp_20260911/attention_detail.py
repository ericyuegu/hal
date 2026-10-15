"""Measure detailed last-query attention statistics for the fixed O50 cohort."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import subprocess
import tarfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from hal.inference import o50_model
from hal.training import features
from research.mechinterp_20260911 import o50_probe

FIXED_COHORT_SHA256 = "ec61e97e332672a4e4e2fdc438e208a281a4609e10f7eded72368eccac323437"
EXPECTED_WINDOWS = 256
EXPECTED_CONTEXT_LENGTH = 256
EXPECTED_LAYERS = 16
EXPECTED_HEADS = 16
LEGACY_COMPARISON_ATOL = 3e-5
PEAK_QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)

FloatArray = NDArray[np.float64]
IntegerArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class CheckpointSpec:
    """One verified historical model and its corrected attention artifact."""

    run_id: str
    checkpoint_step: int
    model_filename: str
    attention_filename: str

    @property
    def boundary_step(self) -> int:
        """Return the human-facing one-based checkpoint boundary."""
        return self.checkpoint_step + 1

    @property
    def label(self) -> str:
        """Return the checkpoint label used by prior reports."""
        return f"{self.run_id} step {self.boundary_step:,}"


CHECKPOINT_SPECS = (
    CheckpointSpec("p1fyyp1z", 24575, "p1fyyp1z-step-0024576.model.pt", "p1fyyp1z-step-0024576.json"),
    CheckpointSpec("p1fyyp1z", 57343, "p1fyyp1z-step-0057344.model.pt", "p1fyyp1z-step-0057344.json"),
    CheckpointSpec("p1fyyp1z", 98303, "p1fyyp1z-step-0098304.model.pt", "p1fyyp1z-step-0098304.json"),
    CheckpointSpec("p1fyyp1z", 131071, "p1fyyp1z-step-0131072.model.pt", "p1fyyp1z-step-0131072.json"),
    CheckpointSpec("tgpuo1be", 57343, "tgpuo1be-step-0057344.model.pt", "tgpuo1be-step-0057344.json"),
    CheckpointSpec("tgpuo1be", 131071, "tgpuo1be-step-0131072.model.pt", "tgpuo1be-step-0131072.json"),
)


@dataclass(frozen=True, slots=True)
class PerHeadMetrics:
    """Metrics averaged over replays for each layer and head."""

    mean_entropy_nats: FloatArray
    mean_exp_entropy: FloatArray
    mean_row_max_weight: FloatArray
    mean_weighted_lag: FloatArray
    current_frame_weight: FloatArray
    most_frequent_argmax_lag: IntegerArray
    argmax_lag_frequency: FloatArray


@dataclass(frozen=True, slots=True)
class GlobalMetrics:
    """Metrics averaged over all replay, layer, and head rows."""

    mean_entropy_nats: float
    mean_exp_entropy: float
    mean_row_max_weight: float
    mean_weighted_lag: float
    current_frame_weight: float
    most_frequent_argmax_lag: int
    argmax_lag_frequency: float
    peak_weight_quantiles: dict[str, float]
    mean_attention_by_lag: FloatArray
    argmax_lag_histogram: IntegerArray
    nearest_global_mean_lag: int
    mean_weight_at_nearest_global_mean_lag: float


@dataclass(frozen=True, slots=True)
class AttentionSummary:
    """Detailed attention metrics for one [replay, layer, head, key] tensor."""

    shape: tuple[int, int, int, int]
    per_head: PerHeadMetrics
    global_metrics: GlobalMetrics


@dataclass(frozen=True, slots=True)
class CohortPackage:
    """Validated fixed-cohort tensors and exact provenance metadata."""

    context_features: dict[str, Tensor]
    reference_config: o50_model.O50Config
    stats_payload: Mapping[str, object]
    player_codes: bytes | None
    target: Tensor
    data: dict[str, object]
    package_sha256: str
    data_json_sha256: str


@dataclass(frozen=True, slots=True)
class CheckpointMeasurement:
    """Detailed result for one checkpoint."""

    spec: CheckpointSpec
    provenance: dict[str, object]
    summary: AttentionSummary
    example_weights: NDArray[np.float32]
    example_rows: dict[str, FloatArray | IntegerArray]
    trace_max_abs_error: float
    probability_sum_max_abs_error: float
    reference_artifact_sha256: str
    reference_max_abs_error: float


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be a mapping with string keys")
    return cast(Mapping[str, object], value)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return _sha256_bytes(encoded)


def _bytes_tensor(value: object, label: str) -> bytes:
    if not isinstance(value, Tensor) or value.dtype != torch.uint8 or value.ndim != 1:
        raise TypeError(f"{label} must be a one-dimensional uint8 tensor")
    return value.detach().cpu().contiguous().numpy().tobytes()


def _load_cohort(path: Path) -> CohortPackage:
    package_bytes = path.read_bytes()
    raw = _mapping(torch.load(path, map_location="cpu", weights_only=True), "cohort package")
    expected = {
        "bundle_reference_json",
        "context_features",
        "data_json",
        "player_codes",
        "reference_config",
        "stats_payload",
        "target",
    }
    if set(raw) != expected:
        raise ValueError(f"cohort fields differ: expected={sorted(expected)}, found={sorted(raw)}")

    context_raw = _mapping(raw["context_features"], "cohort context_features")
    context_features: dict[str, Tensor] = {}
    for name, value in context_raw.items():
        if not isinstance(value, Tensor) or value.ndim != 2:
            raise TypeError(f"cohort context feature {name!r} must be a rank-2 tensor")
        if value.shape != (EXPECTED_WINDOWS, EXPECTED_CONTEXT_LENGTH):
            raise ValueError(f"cohort context feature {name!r} has shape {tuple(value.shape)}")
        context_features[name] = value

    target = raw["target"]
    if not isinstance(target, Tensor) or target.ndim != 3 or target.shape[0] != EXPECTED_WINDOWS:
        raise ValueError("cohort target must have one row per fixed-cohort replay")

    data_json = raw["data_json"]
    if not isinstance(data_json, str):
        raise TypeError("cohort data_json must be a string")
    data_bytes = data_json.encode()
    data = json.loads(data_bytes)
    data_mapping = _mapping(data, "cohort data_json")
    windows_raw = data_mapping.get("windows")
    if not isinstance(windows_raw, list) or len(windows_raw) != EXPECTED_WINDOWS:
        raise ValueError("cohort data_json must retain exactly 256 window records")
    windows: list[dict[str, object]] = []
    for index, raw_window in enumerate(windows_raw):
        window = dict(_mapping(raw_window, f"cohort window {index}"))
        if window.get("identity_source") != "rank":
            raise ValueError(f"cohort window {index} is not the recorded rank identity input")
        if not isinstance(window.get("ego_player_id"), int) or isinstance(window["ego_player_id"], bool):
            raise ValueError(f"cohort window {index} has no integer ego_player_id")
        windows.append(window)
    data_copy = dict(data_mapping)
    data_copy["windows"] = windows

    player_ids = context_features.get("ego_player_id")
    if player_ids is None:
        raise ValueError("cohort context has no ego_player_id feature")
    player_id_rows = player_ids.detach().cpu().numpy()
    for index, window in enumerate(windows):
        if not np.all(player_id_rows[index] == player_id_rows[index, 0]):
            raise ValueError(f"cohort player identity changes within window {index}")
        ego_player_id = cast(int, window["ego_player_id"])
        if int(player_id_rows[index, 0]) != ego_player_id:
            raise ValueError(f"cohort rank-derived player identity differs at window {index}")

    reference_config = o50_model.O50Config.from_json(_bytes_tensor(raw["reference_config"], "reference_config"))
    stats_payload = _mapping(json.loads(_bytes_tensor(raw["stats_payload"], "stats_payload")), "stats_payload")
    player_bytes = _bytes_tensor(raw["player_codes"], "player_codes")
    player_codes = player_bytes or None
    if reference_config.architecture.L_ctx != EXPECTED_CONTEXT_LENGTH:
        raise ValueError("cohort reference context length differs from the fixed protocol")
    return CohortPackage(
        context_features=context_features,
        reference_config=reference_config,
        stats_payload=stats_payload,
        player_codes=player_codes,
        target=target,
        data=data_copy,
        package_sha256=_sha256_bytes(package_bytes),
        data_json_sha256=_sha256_bytes(data_bytes),
    )


def _validate_weights(weights: np.ndarray) -> NDArray[np.float64]:
    if weights.ndim != 4 or min(weights.shape) < 1:
        raise ValueError(f"attention weights must have shape [replay, layer, head, key], got {weights.shape}")
    try:
        values = np.asarray(weights, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("attention weights must be numeric") from error
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("attention weights must be finite and non-negative")
    if not np.allclose(values.sum(axis=-1), 1.0, rtol=1e-5, atol=1e-5):
        raise ValueError("attention weights must sum to one over keys")
    return values


def _mode_and_frequency(values: IntegerArray, size: int) -> tuple[int, float]:
    counts = np.bincount(values, minlength=size)
    mode = int(counts.argmax())
    return mode, float(counts[mode] / values.size)


def summarize_attention(weights: np.ndarray) -> AttentionSummary:
    """Summarize a validated [replay, layer, head, key] probability tensor.

    Key index ``key_count - 1`` is the current frame and therefore has lag 0.
    The result keeps ``mean(exp(entropy))`` separate from ``exp(mean(entropy))``.

    Args:
        weights: Attention probabilities for a fixed context.

    Returns:
        Per-head and global attention metrics.

    Raises:
        ValueError: If the tensor shape or probabilities are invalid.
    """
    values = _validate_weights(weights)
    replay_count, layer_count, head_count, key_count = values.shape
    lags = np.arange(key_count - 1, -1, -1, dtype=np.float64)
    entropy = -(values * np.log(np.clip(values, 1e-30, None))).sum(axis=-1)
    effective = np.exp(entropy)
    row_max = values.max(axis=-1)
    weighted_lag = (values * lags).sum(axis=-1)
    current_weight = values[..., -1]
    argmax_lag = (key_count - 1 - values.argmax(axis=-1)).astype(np.int64)

    per_head_mode = np.zeros((layer_count, head_count), dtype=np.int64)
    per_head_frequency = np.zeros((layer_count, head_count), dtype=np.float64)
    for layer in range(layer_count):
        for head in range(head_count):
            mode, frequency = _mode_and_frequency(argmax_lag[:, layer, head], key_count)
            per_head_mode[layer, head] = mode
            per_head_frequency[layer, head] = frequency

    all_argmax = argmax_lag.reshape(-1)
    global_mode, global_frequency = _mode_and_frequency(all_argmax, key_count)
    mean_attention_by_lag = values.mean(axis=(0, 1, 2))[::-1]
    global_mean_lag = float(weighted_lag.mean())
    nearest_lag = min(key_count - 1, max(0, int(np.floor(global_mean_lag + 0.5))))
    peak_quantiles = np.quantile(row_max.reshape(-1), PEAK_QUANTILES)
    global_metrics = GlobalMetrics(
        mean_entropy_nats=float(entropy.mean()),
        mean_exp_entropy=float(effective.mean()),
        mean_row_max_weight=float(row_max.mean()),
        mean_weighted_lag=global_mean_lag,
        current_frame_weight=float(current_weight.mean()),
        most_frequent_argmax_lag=global_mode,
        argmax_lag_frequency=global_frequency,
        peak_weight_quantiles={
            f"p{int(quantile * 100):02d}": float(value)
            for quantile, value in zip(PEAK_QUANTILES, peak_quantiles, strict=True)
        },
        mean_attention_by_lag=mean_attention_by_lag,
        argmax_lag_histogram=np.bincount(all_argmax, minlength=key_count).astype(np.int64),
        nearest_global_mean_lag=nearest_lag,
        mean_weight_at_nearest_global_mean_lag=float(mean_attention_by_lag[nearest_lag]),
    )
    return AttentionSummary(
        shape=(replay_count, layer_count, head_count, key_count),
        per_head=PerHeadMetrics(
            mean_entropy_nats=entropy.mean(axis=0),
            mean_exp_entropy=effective.mean(axis=0),
            mean_row_max_weight=row_max.mean(axis=0),
            mean_weighted_lag=weighted_lag.mean(axis=0),
            current_frame_weight=current_weight.mean(axis=0),
            most_frequent_argmax_lag=per_head_mode,
            argmax_lag_frequency=per_head_frequency,
        ),
        global_metrics=global_metrics,
    )


def _example_rows(weights: NDArray[np.float32]) -> dict[str, FloatArray | IntegerArray]:
    summary = summarize_attention(weights[None, ...])
    return {
        "entropy_nats": summary.per_head.mean_entropy_nats,
        "effective_context": summary.per_head.mean_exp_entropy,
        "row_max_weight": summary.per_head.mean_row_max_weight,
        "weighted_lag": summary.per_head.mean_weighted_lag,
        "current_frame_weight": summary.per_head.current_frame_weight,
        "argmax_lag": summary.per_head.most_frequent_argmax_lag,
    }


def _summary_dict(summary: AttentionSummary) -> dict[str, object]:
    per_head = summary.per_head
    global_metrics = summary.global_metrics
    return {
        "shape": list(summary.shape),
        "lag_definition": "lag 0 is current key; lag key_count-1 is oldest key",
        "per_head": {
            "mean_entropy_nats": per_head.mean_entropy_nats.tolist(),
            "mean_exp_entropy": per_head.mean_exp_entropy.tolist(),
            "mean_row_max_weight": per_head.mean_row_max_weight.tolist(),
            "mean_weighted_lag": per_head.mean_weighted_lag.tolist(),
            "current_frame_weight": per_head.current_frame_weight.tolist(),
            "most_frequent_argmax_lag": per_head.most_frequent_argmax_lag.tolist(),
            "argmax_lag_frequency": per_head.argmax_lag_frequency.tolist(),
        },
        "global": {
            "mean_entropy_nats": global_metrics.mean_entropy_nats,
            "mean_exp_entropy": global_metrics.mean_exp_entropy,
            "mean_row_max_weight": global_metrics.mean_row_max_weight,
            "mean_weighted_lag": global_metrics.mean_weighted_lag,
            "current_frame_weight": global_metrics.current_frame_weight,
            "most_frequent_argmax_lag": global_metrics.most_frequent_argmax_lag,
            "argmax_lag_frequency": global_metrics.argmax_lag_frequency,
            "peak_weight_quantiles": global_metrics.peak_weight_quantiles,
            "mean_attention_by_lag": global_metrics.mean_attention_by_lag.tolist(),
            "argmax_lag_histogram": global_metrics.argmax_lag_histogram.tolist(),
            "nearest_global_mean_lag": global_metrics.nearest_global_mean_lag,
            "mean_weight_at_nearest_global_mean_lag": global_metrics.mean_weight_at_nearest_global_mean_lag,
        },
    }


def _matrix(value: object, label: str) -> NDArray[np.float64]:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a numeric matrix") from error
    if matrix.shape != (EXPECTED_LAYERS, EXPECTED_HEADS) or not np.isfinite(matrix).all():
        raise ValueError(f"{label} must have finite shape [16, 16], got {matrix.shape}")
    return matrix


def _validate_reference(
    summary: AttentionSummary,
    path: Path,
    spec: CheckpointSpec,
    cohort_sha256: str,
    provenance: Mapping[str, object],
) -> tuple[str, float]:
    raw = _mapping(json.loads(path.read_bytes()), str(path))
    if raw.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported corrected attention schema")
    checkpoint = _mapping(raw["checkpoint"], f"{path}.checkpoint")
    if checkpoint.get("wandb_id") != spec.run_id or checkpoint.get("checkpoint_step") != spec.checkpoint_step:
        raise ValueError(f"{path}: checkpoint identity differs from requested model")
    if checkpoint.get("model_export_sha256") != provenance.get("model_export_sha256"):
        raise ValueError(f"{path}: model export hash differs from the loaded model")
    cohort = _mapping(raw["cohort"], f"{path}.cohort")
    if cohort.get("sha256") != cohort_sha256 or cohort.get("windows") != EXPECTED_WINDOWS:
        raise ValueError(f"{path}: corrected attention cohort differs")
    precision = _mapping(raw["precision"], f"{path}.precision")
    if precision.get("qk_score_dtype") != "torch.float32" or precision.get("softmax_output_dtype") != "torch.float32":
        raise ValueError(f"{path}: corrected attention precision is not FP32")
    if precision.get("score_and_softmax_autocast_enabled") is not False:
        raise ValueError(f"{path}: corrected attention autocast was not disabled")
    attention = _mapping(raw["attention"], f"{path}.attention")
    expected = {
        "entropy_nats": summary.per_head.mean_entropy_nats,
        "effective_context": summary.per_head.mean_exp_entropy,
        "mean_lag": summary.per_head.mean_weighted_lag,
    }
    max_error = 0.0
    for name, actual in expected.items():
        reference = _matrix(attention[name], f"{path}.attention.{name}")
        error = float(np.max(np.abs(actual - reference)))
        max_error = max(max_error, error)
        if not np.allclose(actual, reference, rtol=0.0, atol=LEGACY_COMPARISON_ATOL):
            raise ValueError(f"{path}: {name} differs from the existing corrected measurement by {error}")
    return _sha256_file(path), max_error


def _configure_precision() -> dict[str, object]:
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    flags: dict[str, object] = {
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    if flags["cuda_matmul_allow_tf32"] or flags["cudnn_allow_tf32"] or flags["float32_matmul_precision"] != "highest":
        raise RuntimeError(f"failed to configure FP32 precision: {flags}")
    return flags


def _measure_checkpoint(
    spec: CheckpointSpec,
    *,
    scratch: Path,
    old_attention: Path,
    cohort: CohortPackage,
    device: torch.device,
    batch_size: int,
) -> CheckpointMeasurement:
    checkpoint_path = scratch / spec.model_filename
    reference_path = old_attention / spec.attention_filename
    if not checkpoint_path.is_file() or not reference_path.is_file():
        raise FileNotFoundError(f"missing checkpoint or corrected reference for {spec}")
    model, config, provenance = o50_probe.load_checkpoint_model(
        checkpoint_path,
        reference_config=cohort.reference_config,
        stats_payload=cohort.stats_payload,
        player_codes=cohort.player_codes,
        device=device,
    )
    if provenance.get("wandb_id") != spec.run_id or provenance.get("checkpoint_step") != spec.checkpoint_step:
        raise ValueError(f"loaded checkpoint provenance differs for {spec}")
    provenance["resolved_config"] = json.loads(config.to_json())

    attention_chunks: list[NDArray[np.float32]] = []
    trace_max_error = 0.0
    probability_sum_max_error = 0.0
    attention_dtypes: set[str] = set()
    for start in range(0, EXPECTED_WINDOWS, batch_size):
        stop = min(start + batch_size, EXPECTED_WINDOWS)
        context = {name: value[start:stop].to(device) for name, value in cohort.context_features.items()}
        context_actions = model.codec.quantize(features.stack_actions(context))
        padding = torch.zeros(stop - start, dtype=torch.long, device=device)
        with torch.inference_mode(), o50_model.amp_context(config, device):
            tokens = model.context_tokens(context, context_actions)
            traced_hidden, attention = o50_probe.trace_last_attention(model, tokens, padding)
            reference_hidden = model.forward_dense(context, padding, context_actions)
        if attention.dtype != torch.float32:
            raise RuntimeError(f"{spec}: runtime attention dtype is {attention.dtype}, not torch.float32")
        attention_dtypes.add(str(attention.dtype))
        if not torch.equal(traced_hidden, reference_hidden):
            trace_max_error = float((traced_hidden.float() - reference_hidden.float()).abs().max().item())
            raise RuntimeError(f"{spec}: traced trunk differs from public trunk by {trace_max_error}")
        if not torch.isfinite(attention).all():
            raise RuntimeError(f"{spec}: attention contains a non-finite probability")
        probability_error = float((attention.sum(dim=-1) - 1.0).abs().max().item())
        probability_sum_max_error = max(probability_sum_max_error, probability_error)
        if probability_error > 1e-5:
            raise RuntimeError(f"{spec}: attention probabilities do not sum to one")
        if attention.shape[1:] != (EXPECTED_LAYERS, EXPECTED_HEADS, EXPECTED_CONTEXT_LENGTH):
            raise ValueError(f"{spec}: unexpected attention shape {tuple(attention.shape)}")
        attention_chunks.append(np.asarray(attention.detach().cpu(), dtype=np.float32))

    if attention_dtypes != {"torch.float32"}:
        raise RuntimeError(f"{spec}: attention dtype records are {sorted(attention_dtypes)}")
    weights = np.concatenate(attention_chunks, axis=0)
    summary = summarize_attention(weights)
    reference_sha256, reference_error = _validate_reference(
        summary, reference_path, spec, FIXED_COHORT_SHA256, provenance
    )
    example_weights = np.ascontiguousarray(weights[0], dtype=np.float32)
    example_rows = _example_rows(example_weights)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return CheckpointMeasurement(
        spec=spec,
        provenance=dict(provenance),
        summary=summary,
        example_weights=example_weights,
        example_rows=example_rows,
        trace_max_abs_error=trace_max_error,
        probability_sum_max_abs_error=probability_sum_max_error,
        reference_artifact_sha256=reference_sha256,
        reference_max_abs_error=reference_error,
    )


def _write_source_archive(output: Path) -> tuple[str, dict[str, str], str, str | None]:
    archive_path = output / "attention_detail_source.tar"
    superseded_archive_sha256 = None
    if archive_path.exists():
        superseded_archive_sha256 = _sha256_file(archive_path)
        archive_path = output / "attention_detail_source_v2.tar"
    if archive_path.exists():
        raise FileExistsError(f"refusing to overwrite {archive_path}")
    source_paths = {
        "attention_detail": Path(__file__),
        "o50_probe": Path(o50_probe.__file__),
    }
    plan_path = Path(__file__).with_name("attention_detail_plan.md")
    if plan_path.is_file():
        source_paths["attention_detail_plan"] = plan_path
    source_hashes = {name: _sha256_file(path) for name, path in source_paths.items()}
    with tarfile.open(archive_path, mode="w") as archive:
        for name, path in source_paths.items():
            suffix = path.suffix
            archive.add(path, arcname=f"research/mechinterp_20260911/{name}{suffix}")
    return _sha256_file(archive_path), source_hashes, str(archive_path), superseded_archive_sha256


def _write_example_npz(output: Path, measurements: tuple[CheckpointMeasurement, ...]) -> tuple[str, dict[str, object]]:
    path = output / "attention_example0.npz"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    arrays: dict[str, NDArray[np.float32] | FloatArray | IntegerArray] = {}
    checkpoint_keys: dict[str, str] = {}
    fields = (
        "weights",
        "entropy_nats",
        "effective_context",
        "row_max_weight",
        "weighted_lag",
        "current_frame_weight",
        "argmax_lag",
    )
    for index, measurement in enumerate(measurements):
        key = f"checkpoint_{index:02d}"
        checkpoint_keys[key] = (
            f"{measurement.spec.run_id}:boundary_step={measurement.spec.boundary_step}"
            f" (checkpoint_step={measurement.spec.checkpoint_step})"
        )
        arrays[f"{key}__weights"] = measurement.example_weights
        for field in fields[1:]:
            arrays[f"{key}__{field}"] = measurement.example_rows[field]
    np.savez_compressed(path, **arrays)
    return _sha256_file(path), {
        "path": str(path),
        "sha256": _sha256_file(path),
        "checkpoint_keys": checkpoint_keys,
        "example_index": 0,
        "weights_dtype": "float32",
        "weights_shape": [EXPECTED_LAYERS, EXPECTED_HEADS, EXPECTED_CONTEXT_LENGTH],
        "weights_order": "[layer, head, key], key 255=current/lag0 and key 0=oldest/lag255",
        "row_fields": list(fields[1:]),
    }


def _environment(device: torch.device, precision: Mapping[str, object]) -> dict[str, object]:
    environment: dict[str, object] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "precision": dict(precision),
    }
    environment["sha256"] = _json_sha256(environment)
    return environment


def _research_git_sha() -> str:
    repository = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    git_sha = completed.stdout.strip()
    if len(git_sha) != 40 or any(character not in "0123456789abcdef" for character in git_sha):
        raise ValueError(f"git rev-parse returned an invalid SHA: {git_sha!r}")
    return git_sha


def _result(
    *,
    cohort_path: Path,
    cohort: CohortPackage,
    measurements: tuple[CheckpointMeasurement, ...],
    precision: Mapping[str, object],
    source_archive_sha256: str,
    source_hashes: Mapping[str, str],
    example_manifest: Mapping[str, object],
    device: torch.device,
    batch_size: int,
    research_git_sha: str,
    source_archive_path: str,
    superseded_source_archive_sha256: str | None,
) -> dict[str, object]:
    data = dict(cohort.data)
    data["package_path"] = str(cohort_path)
    data["package_sha256"] = cohort.package_sha256
    data["data_json_sha256"] = cohort.data_json_sha256
    data["context_feature_names"] = sorted(cohort.context_features)
    data["target_shape"] = list(cohort.target.shape)
    data["target_sha256"] = _sha256_bytes(cohort.target.detach().cpu().contiguous().numpy().tobytes())
    checkpoints: list[dict[str, object]] = []
    for measurement in measurements:
        checkpoints.append(
            {
                "run_id": measurement.spec.run_id,
                "checkpoint_step": measurement.spec.checkpoint_step,
                "boundary_step": measurement.spec.boundary_step,
                "label": measurement.spec.label,
                "model_filename": measurement.spec.model_filename,
                "provenance": measurement.provenance,
                "summary": _summary_dict(measurement.summary),
                "controls": {
                    "attention_dtype": "torch.float32",
                    "traced_trunk_max_abs_error": measurement.trace_max_abs_error,
                    "attention_probability_sum_max_abs_error": measurement.probability_sum_max_abs_error,
                    "existing_corrected_artifact_max_abs_error": measurement.reference_max_abs_error,
                },
                "existing_corrected_artifact": {
                    "filename": measurement.spec.attention_filename,
                    "sha256": measurement.reference_artifact_sha256,
                    "comparison_atol": LEGACY_COMPARISON_ATOL,
                },
            }
        )
    result: dict[str, object] = {
        "schema_version": 1,
        "protocol": {
            "name": "O50 detailed last-query attention",
            "windows": EXPECTED_WINDOWS,
            "batch_size": batch_size,
            "seed": cohort.data.get("seed"),
            "example_index": 0,
            "lag_definition": "lag 0 is current key; lag 255 is oldest key",
            "attention_shape": [EXPECTED_WINDOWS, EXPECTED_LAYERS, EXPECTED_HEADS, EXPECTED_CONTEXT_LENGTH],
            "no_training": True,
            "player_rank_inputs_retained": True,
        },
        "cohort": data,
        "resolved_config_reference": json.loads(cohort.reference_config.to_json()),
        "precision": dict(precision),
        "source": {
            "attention_detail_sha256": source_hashes["attention_detail"],
            "o50_probe_sha256": source_hashes["o50_probe"],
            "files": dict(source_hashes),
            "source_archive_sha256": source_archive_sha256,
            "source_archive_path": source_archive_path,
            "research_git_sha": research_git_sha,
        },
        "environment": _environment(device, precision),
        "example_npz": dict(example_manifest),
        "checkpoints": checkpoints,
    }
    if superseded_source_archive_sha256 is not None:
        result["source"]["superseded_failed_launch_archive_sha256"] = superseded_source_archive_sha256
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_scratch = Path("/home/ericgu/src/hal/runs/mechinterp-20260911-scratch")
    default_artifacts = Path(__file__).with_name("artifacts")
    parser.add_argument("--cohort", type=Path, default=default_scratch / "o50-cohort-256.pt")
    parser.add_argument("--scratch", type=Path, default=default_scratch)
    parser.add_argument("--old-attention", type=Path, default=default_artifacts / "o50_attention_fp32")
    parser.add_argument("--output", type=Path, default=default_artifacts / "attention_detail")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 64:
        raise ValueError("batch size must be in [1, 64]")
    owned_outputs = (
        args.output / "attention_example0.npz",
        args.output / "attention_detail.json",
    )
    if any(path.exists() for path in owned_outputs):
        raise FileExistsError(f"refusing to overwrite an existing measurement artifact in {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    source_archive_sha256, source_hashes, source_archive_path, superseded_archive_sha256 = _write_source_archive(
        args.output
    )
    precision = _configure_precision()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    cohort = _load_cohort(args.cohort)
    if cohort.package_sha256 != FIXED_COHORT_SHA256:
        raise ValueError(f"fixed cohort package hash differs: {cohort.package_sha256}")
    measurements = tuple(
        _measure_checkpoint(
            spec,
            scratch=args.scratch,
            old_attention=args.old_attention,
            cohort=cohort,
            device=device,
            batch_size=args.batch_size,
        )
        for spec in CHECKPOINT_SPECS
    )
    example_sha256, example_manifest = _write_example_npz(args.output, measurements)
    if example_sha256 != example_manifest["sha256"]:
        raise RuntimeError("example NPZ hash changed during manifest creation")
    result = _result(
        cohort_path=args.cohort,
        cohort=cohort,
        measurements=measurements,
        precision=precision,
        source_archive_sha256=source_archive_sha256,
        source_hashes=source_hashes,
        example_manifest=example_manifest,
        device=device,
        batch_size=args.batch_size,
        research_git_sha=_research_git_sha(),
        source_archive_path=source_archive_path,
        superseded_source_archive_sha256=superseded_archive_sha256,
    )
    result_path = args.output / "attention_detail.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"Saved {result_path}; {len(measurements)} checkpoints", flush=True)


if __name__ == "__main__":
    main()
