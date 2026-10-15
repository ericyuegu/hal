"""Validate and plot the historical O50 attention artifacts.

The source artifacts contain last-query trunk attention summarized over a fixed
256-replay cohort. This module validates their schema, cross-checks the early
and slump intervention traces, and writes a descriptive attention figure. It
does not load model checkpoints or make a gameplay claim.
"""

import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_ATTENTION_FIELDS = (
    "entropy_nats",
    "effective_context",
    "mean_lag",
    "last_4_mass",
    "last_16_mass",
    "distant_64_plus_mass",
    "current_mass",
)
_EXPECTED_RETENTION = (1, 2, 4, 8, 16, 32, 64, 128, 256)
_EXPECTED_LAYERS = 16
_EXPECTED_HEADS = 16
_EXPECTED_WINDOWS = 256
_EXPECTED_SEED = 20260911
_EXPECTED_NLL_UNITS = "natural-log nats per controller group per predicted frame"


@dataclass(frozen=True, slots=True)
class HistoryPoint:
    """One paired context-retention measurement."""

    retained_frames: int
    delta_nll: float
    bootstrap_low: float
    bootstrap_high: float


@dataclass(frozen=True, slots=True)
class AttentionArtifact:
    """Validated attention and context-retention measurements for one export."""

    optimizer: str
    run_id: str
    checkpoint_step: int
    update: int
    entropy_nats: tuple[tuple[float, ...], ...]
    effective_context: tuple[tuple[float, ...], ...]
    mean_lag: tuple[tuple[float, ...], ...]
    history: tuple[HistoryPoint, ...]
    cohort_sha256: str
    cohort_windows: int
    source_sha256: str
    source_path: Path

    @property
    def label(self) -> str:
        """Return a compact figure label."""
        return f"{self.optimizer} {self.update:,}"

    @property
    def mean_entropy_nats(self) -> float:
        """Return mean last-query entropy in natural-log nats."""
        return float(np.asarray(self.entropy_nats, dtype=np.float64).mean())

    @property
    def mean_effective_context(self) -> float:
        """Return mean effective context positions."""
        return float(np.asarray(self.effective_context, dtype=np.float64).mean())

    @property
    def mean_lag_frames(self) -> float:
        """Return mean attended lag in context frames."""
        return float(np.asarray(self.mean_lag, dtype=np.float64).mean())


@dataclass(frozen=True, slots=True)
class InterventionTrace:
    """Validated identity traces from the component-intervention artifact."""

    source_sha256: str
    cohort_windows: int
    early_max_trace_error: float
    slump_max_trace_error: float


@dataclass(frozen=True, slots=True)
class CorrectedAttention:
    """FP32 attention matrices and provenance for one re-audit result."""

    run_id: str
    checkpoint_step: int
    entropy_nats: tuple[tuple[float, ...], ...]
    effective_context: tuple[tuple[float, ...], ...]
    mean_lag: tuple[tuple[float, ...], ...]
    cohort_sha256: str
    cohort_windows: int
    source_sha256: str
    legacy_artifact_sha256: str
    source_path: Path
    trace_max_error: float
    probability_sum_max_error: float
    score_dtype: str
    softmax_dtype: str

    @property
    def mean_entropy_nats(self) -> float:
        """Return the corrected mean last-query entropy."""
        return float(np.asarray(self.entropy_nats, dtype=np.float64).mean())

    @property
    def mean_effective_context(self) -> float:
        """Return mean(exp(per-example/head entropy)) in context positions."""
        return float(np.asarray(self.effective_context, dtype=np.float64).mean())

    @property
    def mean_lag_frames(self) -> float:
        """Return the corrected mean attended lag in context frames."""
        return float(np.asarray(self.mean_lag, dtype=np.float64).mean())


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be an object with string keys")
    return cast(Mapping[str, object], value)


def _sequence(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return cast(list[object], value)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _matrix(
    value: object, label: str, *, lower: float | None = None, upper: float | None = None
) -> tuple[tuple[float, ...], ...]:
    rows = _sequence(value, label)
    if len(rows) != _EXPECTED_LAYERS:
        raise ValueError(f"{label} must have {_EXPECTED_LAYERS} layers")
    parsed: list[tuple[float, ...]] = []
    for layer, raw_row in enumerate(rows):
        row = _sequence(raw_row, f"{label}[{layer}]")
        if len(row) != _EXPECTED_HEADS:
            raise ValueError(f"{label}[{layer}] must have {_EXPECTED_HEADS} heads")
        values = tuple(_number(item, f"{label}[{layer}][{head}]") for head, item in enumerate(row))
        if lower is not None and min(values) < lower:
            raise ValueError(f"{label} contains a value below {lower}")
        if upper is not None and max(values) > upper:
            raise ValueError(f"{label} contains a value above {upper}")
        parsed.append(values)
    return tuple(parsed)


def _matrix_array(value: tuple[tuple[float, ...], ...]) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _infer_optimizer(checkpoint: Mapping[str, object]) -> str:
    backend = _mapping(checkpoint["backend_config"], "checkpoint.backend_config")
    experiment_id = _text(backend["experiment_id"], "checkpoint.backend_config.experiment_id")
    source_uri = _text(checkpoint["source_uri"], "checkpoint.source_uri")
    if experiment_id == "050_scaled_temporal_awr_v4" and "all-adamw" not in source_uri:
        return "Muon"
    if experiment_id == "050_scaled_temporal_awr_v6" or "all-adamw" in source_uri:
        return "AdamW"
    raise ValueError(f"cannot infer optimizer from experiment {experiment_id!r}")


def _load_history(value: object) -> tuple[HistoryPoint, ...]:
    records = _mapping(value, "history_retained_frames")
    expected_keys = {str(item) for item in _EXPECTED_RETENTION}
    if set(records) != expected_keys:
        raise ValueError("history_retained_frames keys do not match the fixed retention grid")
    result: list[HistoryPoint] = []
    for retained in _EXPECTED_RETENTION:
        record = _mapping(records[str(retained)], f"history_retained_frames[{retained}]")
        summary = _mapping(
            record["replay_paired_horizon_group_mean_nll_delta"],
            f"history_retained_frames[{retained}].replay_paired_horizon_group_mean_nll_delta",
        )
        interval = _sequence(summary["bootstrap_95"], f"history_retained_frames[{retained}].bootstrap_95")
        if len(interval) != 2:
            raise ValueError(f"history_retained_frames[{retained}].bootstrap_95 must have two values")
        low = _number(interval[0], "history bootstrap lower bound")
        high = _number(interval[1], "history bootstrap upper bound")
        if low > high:
            raise ValueError(f"history bootstrap interval is reversed at {retained} frames")
        result.append(
            HistoryPoint(
                retained_frames=retained,
                delta_nll=_number(summary["mean"], "history paired mean"),
                bootstrap_low=low,
                bootstrap_high=high,
            )
        )
    return tuple(result)


def _load_artifact(path: Path) -> AttentionArtifact:
    root = _mapping(json.loads(path.read_bytes()), str(path))
    if root.get("schema_version") != 2:
        raise ValueError(f"{path}: expected historical artifact schema_version 2")
    checkpoint = _mapping(root["checkpoint"], f"{path}.checkpoint")
    backend = _mapping(checkpoint["backend_config"], f"{path}.checkpoint.backend_config")
    architecture = _mapping(backend["architecture"], f"{path}.checkpoint.backend_config.architecture")
    if architecture.get("n_layers") != _EXPECTED_LAYERS or architecture.get("n_heads") != _EXPECTED_HEADS:
        raise ValueError(f"{path}: expected a 16-layer, 16-head trunk")
    if architecture.get("L_ctx") != 256 or backend.get("prediction_frames") != 4:
        raise ValueError(f"{path}: context or prediction-frame configuration is not the fixed O50 protocol")

    units = _mapping(root["nll_units"], f"{path}.nll_units")
    if not _text(units["group_nll"], f"{path}.nll_units.group_nll").startswith(_EXPECTED_NLL_UNITS):
        raise ValueError(f"{path}: NLL units are not natural-log nats per group and frame")
    cohort = _mapping(root["cohort_package"], f"{path}.cohort_package")
    cohort_sha256 = _text(cohort["sha256"], f"{path}.cohort_package.sha256")
    cohort_windows = _integer(cohort["windows"], f"{path}.cohort_package.windows")
    if cohort_windows != _EXPECTED_WINDOWS:
        raise ValueError(f"{path}: expected {_EXPECTED_WINDOWS} fixed-cohort replays")
    data = _mapping(root["data"], f"{path}.data")
    if data.get("seed") != _EXPECTED_SEED or len(_sequence(data["windows"], f"{path}.data.windows")) != cohort_windows:
        raise ValueError(f"{path}: data seed or replay-window count differs from the fixed cohort")

    attention = _mapping(root["attention"], f"{path}.attention")
    matrices = {
        field: _matrix(
            attention[field],
            f"{path}.attention.{field}",
            lower=1.0 if field == "effective_context" else 0.0,
            upper=1.0 if field.endswith("mass") or field == "current_mass" else None,
        )
        for field in _ATTENTION_FIELDS
    }
    entropy = matrices["entropy_nats"]
    effective_context = matrices["effective_context"]

    checkpoint_step = _integer(checkpoint["checkpoint_step"], f"{path}.checkpoint.checkpoint_step")
    run_id = _text(checkpoint["wandb_id"], f"{path}.checkpoint.wandb_id")
    return AttentionArtifact(
        optimizer=_infer_optimizer(checkpoint),
        run_id=run_id,
        checkpoint_step=checkpoint_step,
        update=checkpoint_step + 1,
        entropy_nats=entropy,
        effective_context=effective_context,
        mean_lag=matrices["mean_lag"],
        history=_load_history(root["history_retained_frames"]),
        cohort_sha256=cohort_sha256,
        cohort_windows=cohort_windows,
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_path=path,
    )


def load_historical(source: Path, *, expected_count: int = 6) -> tuple[AttentionArtifact, ...]:
    """Load and validate historical attention artifacts from a directory.

    Args:
        source: Directory containing one ``results.json`` below each checkpoint
            directory.
        expected_count: Number of artifacts required. The production figure
            requires six; a smaller value is useful for focused tests.

    Returns:
        Artifacts ordered by optimizer family and update count.

    Raises:
        ValueError: If the artifacts disagree on schema, cohort, or shape.
    """
    paths = sorted(source.glob("*/results.json"))
    if len(paths) != expected_count:
        raise ValueError(f"expected {expected_count} historical artifacts under {source}, found {len(paths)}")
    records = tuple(_load_artifact(path) for path in paths)
    identities = {(record.run_id, record.checkpoint_step) for record in records}
    if len(identities) != len(records):
        raise ValueError("historical artifacts contain a duplicate checkpoint")
    cohort_ids = {(record.cohort_sha256, record.cohort_windows) for record in records}
    if len(cohort_ids) != 1:
        raise ValueError("historical artifacts do not share one replay cohort")
    return tuple(sorted(records, key=lambda item: (item.optimizer != "Muon", item.update)))


def _sha256_text(value: object, label: str) -> str:
    text = _text(value, label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
    return text


def _load_corrected_result(path: Path) -> CorrectedAttention:
    """Load one explicit-FP32 attention re-audit result."""
    root = _mapping(json.loads(path.read_bytes()), str(path))
    if root.get("schema_version") != 1:
        raise ValueError(f"{path}: expected attention re-audit schema_version 1")

    checkpoint = _mapping(root["checkpoint"], f"{path}.checkpoint")
    run_id = _text(checkpoint["wandb_id"], f"{path}.checkpoint.wandb_id")
    checkpoint_step = _integer(checkpoint["checkpoint_step"], f"{path}.checkpoint.checkpoint_step")
    _sha256_text(checkpoint["model_export_sha256"], f"{path}.checkpoint.model_export_sha256")

    cohort = _mapping(root["cohort"], f"{path}.cohort")
    cohort_sha256 = _sha256_text(cohort["sha256"], f"{path}.cohort.sha256")
    cohort_windows = _integer(cohort["windows"], f"{path}.cohort.windows")
    if cohort_windows != _EXPECTED_WINDOWS:
        raise ValueError(f"{path}: expected {_EXPECTED_WINDOWS} corrected attention windows")

    raw_attention = _mapping(root["attention"], f"{path}.attention")
    matrices = {
        field: _matrix(
            raw_attention[field],
            f"{path}.attention.{field}",
            lower=1.0 if field == "effective_context" else 0.0,
            upper=1.0 if field.endswith("mass") or field == "current_mass" else None,
        )
        for field in _ATTENTION_FIELDS
    }

    precision = _mapping(root["precision"], f"{path}.precision")
    score_dtype = _text(precision["qk_score_dtype"], f"{path}.precision.qk_score_dtype")
    softmax_dtype = _text(precision["softmax_output_dtype"], f"{path}.precision.softmax_output_dtype")
    if score_dtype != "torch.float32" or softmax_dtype != "torch.float32":
        raise ValueError(f"{path}: corrected score and softmax dtypes must both be torch.float32")
    if precision.get("score_and_softmax_autocast_enabled") is not False:
        raise ValueError(f"{path}: corrected attention must disable score and softmax autocast")

    controls = _mapping(root["controls"], f"{path}.controls")
    trace_error = _number(controls["traced_trunk_max_abs_error"], f"{path}.controls.traced_trunk_max_abs_error")
    probability_error = _number(
        controls["attention_probability_sum_max_abs_error"],
        f"{path}.controls.attention_probability_sum_max_abs_error",
    )
    if trace_error > 2e-2:
        raise ValueError(f"{path}: corrected hidden-state parity error exceeds the recorded bound")
    if probability_error > 1e-5:
        raise ValueError(f"{path}: corrected attention probabilities do not sum to one")

    legacy = _mapping(root["legacy_functional_artifact"], f"{path}.legacy_functional_artifact")
    legacy_sha256 = _sha256_text(legacy["sha256"], f"{path}.legacy_functional_artifact.sha256")
    return CorrectedAttention(
        run_id=run_id,
        checkpoint_step=checkpoint_step,
        entropy_nats=matrices["entropy_nats"],
        effective_context=matrices["effective_context"],
        mean_lag=matrices["mean_lag"],
        cohort_sha256=cohort_sha256,
        cohort_windows=cohort_windows,
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        legacy_artifact_sha256=legacy_sha256,
        source_path=path,
        trace_max_error=trace_error,
        probability_sum_max_error=probability_error,
        score_dtype=score_dtype,
        softmax_dtype=softmax_dtype,
    )


def load_corrected_attention(
    source: Path, historical: tuple[AttentionArtifact, ...]
) -> tuple[CorrectedAttention, ...]:
    """Load six FP32 results and join them to the immutable old artifacts.

    ``source`` may name one result or a directory containing one JSON result per
    checkpoint. The join is intentionally keyed by run ID and raw checkpoint
    step, not by filename or display order. Each result must also identify the
    old functional result by its exact SHA-256 and use the historical cohort.
    """
    paths = (source,) if source.is_file() else tuple(sorted(source.glob("*.json")))
    if len(paths) != len(historical):
        raise ValueError(f"expected {len(historical)} corrected attention results under {source}, found {len(paths)}")
    corrected = tuple(_load_corrected_result(path) for path in paths)
    identities = {(item.run_id, item.checkpoint_step) for item in corrected}
    if len(identities) != len(corrected):
        raise ValueError("corrected attention results contain a duplicate checkpoint")

    by_identity = {(item.run_id, item.checkpoint_step): item for item in corrected}
    joined: list[CorrectedAttention] = []
    for record in historical:
        key = (record.run_id, record.checkpoint_step)
        item = by_identity.get(key)
        if item is None:
            raise ValueError(f"no corrected attention result matches {record.run_id} step {record.checkpoint_step}")
        if (item.cohort_sha256, item.cohort_windows) != (record.cohort_sha256, record.cohort_windows):
            raise ValueError(f"corrected attention cohort differs for {record.run_id} step {record.checkpoint_step}")
        if item.legacy_artifact_sha256 != record.source_sha256:
            raise ValueError(
                f"corrected attention does not identify the old artifact for {record.run_id} step {record.checkpoint_step}"
            )
        joined.append(item)
    return tuple(joined)


def _intervention_matrix(root: Mapping[str, object], name: str) -> tuple[tuple[float, ...], ...]:
    interventions = _mapping(root["interventions"], "interventions")
    treatment = _mapping(interventions[name], f"interventions.{name}")
    attention = _mapping(treatment["attention"], f"interventions.{name}.attention")
    metrics = _mapping(attention["metrics"], f"interventions.{name}.attention.metrics")
    return _matrix(metrics["entropy_nats"], f"interventions.{name}.attention.metrics.entropy_nats", lower=0.0)


def load_intervention_trace(path: Path, historical: tuple[AttentionArtifact, ...]) -> InterventionTrace:
    """Validate intervention identity traces against historical heat maps."""
    root = _mapping(json.loads(path.read_bytes()), str(path))
    if root.get("schema_version") != 1:
        raise ValueError(f"{path}: expected intervention artifact schema_version 1")
    cohort = _mapping(root["cohort"], f"{path}.cohort")
    data = _mapping(cohort["data"], f"{path}.cohort.data")
    windows = len(_sequence(data["windows"], f"{path}.cohort.data.windows"))
    if data.get("seed") != _EXPECTED_SEED or windows != _EXPECTED_WINDOWS:
        raise ValueError(f"{path}: intervention cohort differs from the fixed 256-replay cohort")
    interventions = _mapping(root["interventions"], f"{path}.interventions")
    for name, expected_kind in (("early_identity", "identity_early"), ("slump_identity", "identity_slump")):
        treatment = _mapping(interventions[name], f"{path}.interventions.{name}")
        if treatment.get("kind") != expected_kind:
            raise ValueError(f"{path}: {name} is not the expected identity treatment")
        attention = _mapping(treatment["attention"], f"{path}.interventions.{name}.attention")
        error = _number(attention["traced_trunk_max_abs_error"], f"{path}: {name} trace error")
        if error > 2e-2:
            raise ValueError(f"{path}: {name} trace error exceeds the recorded parity bound")

    expected = {
        "early_identity": ("Muon", 24576),
        "slump_identity": ("Muon", 57344),
    }
    errors: list[float] = []
    for name, (optimizer, update) in expected.items():
        trace = _matrix_array(_intervention_matrix(root, name))
        match = next(
            (record for record in historical if record.optimizer == optimizer and record.update == update), None
        )
        if match is None:
            raise ValueError(f"{path}: no historical artifact matches {name}")
        if not np.array_equal(trace, _matrix_array(match.entropy_nats)):
            raise ValueError(f"{path}: {name} entropy trace differs from its historical artifact")
        attention = _mapping(_mapping(interventions[name], f"interventions.{name}")["attention"], "attention")
        errors.append(_number(attention["traced_trunk_max_abs_error"], f"{path}: {name} trace error"))
    return InterventionTrace(
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        cohort_windows=windows,
        early_max_trace_error=errors[0],
        slump_max_trace_error=errors[1],
    )


def _plot(
    records: tuple[AttentionArtifact, ...],
    output: Path,
    corrected: tuple[CorrectedAttention, ...] | None = None,
) -> None:
    if corrected is not None and len(corrected) != len(records):
        raise ValueError("corrected attention count does not match historical artifacts")
    entropy_arrays = [
        _matrix_array(item.entropy_nats if corrected is None else corrected[index].entropy_nats)
        for index, item in enumerate(records)
    ]
    entropy_vmax = max(0.5, math.ceil(max(array.max() for array in entropy_arrays) * 2) / 2)
    colors = {
        ("Muon", 24576): "#a8382b",
        ("Muon", 57344): "#d26a52",
        ("Muon", 98304): "#8e2720",
        ("Muon", 131072): "#611a16",
        ("AdamW", 57344): "#2b78a8",
        ("AdamW", 131072): "#16496b",
    }
    figure = plt.figure(figsize=(14, 10))
    grid = figure.add_gridspec(3, 3, height_ratios=(1.0, 1.0, 0.85))
    figure.subplots_adjust(top=0.90, bottom=0.14, left=0.055, right=0.90, hspace=0.42, wspace=0.25)
    heat_axes = []
    image = None
    for index, record in enumerate(records):
        mean_entropy = record.mean_entropy_nats if corrected is None else corrected[index].mean_entropy_nats
        axis = figure.add_subplot(grid[index // 3, index % 3])
        heat_axes.append(axis)
        image = axis.imshow(
            entropy_arrays[index],
            cmap="magma",
            vmin=0.0,
            vmax=entropy_vmax,
            interpolation="nearest",
            aspect="equal",
        )
        short_label = f"{record.optimizer} {record.update // 1000}k"
        axis.set_title(
            f"{short_label}\nH̄={mean_entropy:.3f}; "
            f"E[e^H]={record.mean_effective_context if corrected is None else corrected[index].mean_effective_context:.2f}\n"
            f"laḡ={(record.mean_lag_frames if corrected is None else corrected[index].mean_lag_frames):.1f} frames",
            fontsize=8,
            pad=4,
        )
        axis.set_xticks((0, 4, 8, 12, 15))
        axis.set_yticks((0, 4, 8, 12, 15))
        if index // 3 == 1:
            axis.set_xlabel("Head")
        if index % 3 == 0:
            axis.set_ylabel("Layer")
    if image is None:
        raise ValueError("no attention matrices to plot")
    colorbar = figure.colorbar(image, ax=heat_axes, shrink=0.88, pad=0.02)
    colorbar.set_label("Last-query attention entropy (nats)")

    axis = figure.add_subplot(grid[2, :])
    for record in records:
        retained = np.asarray([point.retained_frames for point in record.history], dtype=np.float64)
        mean = np.asarray([point.delta_nll for point in record.history], dtype=np.float64)
        lower = np.asarray([point.bootstrap_low for point in record.history], dtype=np.float64)
        upper = np.asarray([point.bootstrap_high for point in record.history], dtype=np.float64)
        color = colors[(record.optimizer, record.update)]
        axis.fill_between(retained, lower, upper, color=color, alpha=0.10, linewidth=0)
        axis.plot(retained, mean, marker="o", markersize=3.5, linewidth=1.4, color=color, label=record.label)
    axis.axhline(0.0, color="0.35", linewidth=0.8)
    axis.set_xscale("log", base=2)
    axis.set_xticks(_EXPECTED_RETENTION)
    axis.set_xticklabels([str(value) for value in _EXPECTED_RETENTION])
    axis.set_xlabel("Retained context frames")
    axis.set_ylabel("ΔNLL (nats / controller group / predicted frame)")
    axis.set_title("Paired history-retention sensitivity", fontsize=10)
    axis.grid(axis="y", alpha=0.2)
    axis.legend(ncol=3, frameon=False, fontsize=8, loc="upper right")

    figure.suptitle("Historical O50 attention contrast", fontsize=14, y=0.97)
    precision_label = (
        "FP32 QK scores/softmax; score and softmax autocast disabled"
        if corrected is not None
        else "legacy attention precision"
    )
    figure.text(
        0.5,
        0.025,
        f"Last-query trunk attention; 16 layers × 16 heads; fixed 256-replay cohort; {precision_label}; "
        "descriptive teacher-forced diagnostics.",
        ha="center",
        fontsize=8,
        color="0.3",
    )
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "attention_contrast.png", dpi=220)
    figure.savefig(output / "attention_contrast.pdf")
    plt.close(figure)


def _summary(
    records: tuple[AttentionArtifact, ...],
    trace: InterventionTrace,
    script: Path,
    corrected: tuple[CorrectedAttention, ...] | None = None,
) -> dict[str, object]:
    checkpoints = []
    for index, record in enumerate(records):
        item = None if corrected is None else corrected[index]
        checkpoint: dict[str, object] = {
            "optimizer": record.optimizer,
            "run_id": record.run_id,
            "checkpoint_step": record.checkpoint_step,
            "update": record.update,
            "label": record.label,
            "mean_entropy_nats": record.mean_entropy_nats if item is None else item.mean_entropy_nats,
            "mean_effective_context": record.mean_effective_context if item is None else item.mean_effective_context,
            "mean_lag_frames": record.mean_lag_frames if item is None else item.mean_lag_frames,
            "source_sha256": record.source_sha256,
        }
        if item is not None:
            checkpoint.update(
                {
                    "legacy_mean_entropy_nats": record.mean_entropy_nats,
                    "corrected_result_sha256": item.source_sha256,
                    "corrected_score_dtype": item.score_dtype,
                    "corrected_softmax_dtype": item.softmax_dtype,
                    "corrected_traced_trunk_max_abs_error": item.trace_max_error,
                    "corrected_probability_sum_max_abs_error": item.probability_sum_max_error,
                    "legacy_functional_artifact_sha256": item.legacy_artifact_sha256,
                }
            )
        checkpoints.append(checkpoint)

    result: dict[str, object] = {
        "schema_version": 2 if corrected is not None else 1,
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "cohort": {
            "sha256": records[0].cohort_sha256,
            "windows": records[0].cohort_windows,
            "seed": _EXPECTED_SEED,
        },
        "units": {
            "attention_entropy": "natural-log nats per last-query layer/head distribution",
            "effective_context": "mean(exp(per-example/head entropy)) in context positions",
            "mean_lag": "context frames, where 0 is the current frame",
            "history_delta_nll": "natural-log nats per controller group per predicted frame, averaged over four frames",
        },
        "checkpoints": checkpoints,
        "history_retention": [
            {
                "label": record.label,
                "points": [
                    {
                        "retained_frames": point.retained_frames,
                        "delta_nll": point.delta_nll,
                        "bootstrap_95": [point.bootstrap_low, point.bootstrap_high],
                    }
                    for point in record.history
                ],
            }
            for record in records
        ],
        "intervention_identity_trace": {
            "source_sha256": trace.source_sha256,
            "cohort_windows": trace.cohort_windows,
            "early_max_trace_error": trace.early_max_trace_error,
            "slump_max_trace_error": trace.slump_max_trace_error,
            "entropy_heatmaps_match_historical": True,
        },
        "limitations": [
            "Attention entropy is a descriptive last-query statistic, not an importance map.",
            "History-retention changes are teacher-forced NLL diagnostics, not closed-loop gameplay results.",
            "The six checkpoints share one fixed replay cohort and training lineage; uncertainty bands are replay bootstraps.",
        ],
    }
    if corrected is not None:
        result["attention_provenance"] = {
            "precision": "FP32 QK scores and softmax with score/softmax autocast disabled",
            "corrected_results": [str(item.source_path) for item in corrected],
            "old_functional_artifacts_joined_by_sha256": True,
            "old_functional_artifact_sha256": [item.legacy_artifact_sha256 for item in corrected],
        }
    else:
        result["attention_provenance"] = {
            "precision": "legacy result matrices; no FP32 re-audit supplied",
            "old_functional_artifacts_joined_by_sha256": False,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).with_name("artifacts")
    parser.add_argument("--historical", type=Path, default=default_root / "o50_historical")
    parser.add_argument("--intervention", type=Path, default=default_root / "interventions" / "results.json")
    parser.add_argument("--reaudit", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=default_root / "attention")
    args = parser.parse_args()
    records = load_historical(args.historical)
    trace = load_intervention_trace(args.intervention, records)
    corrected = load_corrected_attention(args.reaudit, records)
    _plot(records, args.output, corrected)
    result = _summary(records, trace, Path(__file__), corrected)
    (args.output / "attention_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Wrote attention contrast for {len(records)} checkpoints to {args.output}.")


if __name__ == "__main__":
    main()
