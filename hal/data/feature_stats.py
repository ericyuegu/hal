"""Per-feature normalization statistics.

Stage 3 emits a ``stats.json`` sidecar next to ``manifest.jsonl`` containing
**sufficient statistics** (count, mean, M2, min, max) for every continuous
float column in the train split. Training-time consumers merge sufficient
stats across one or more dataset Streams under their sampling proportions to
produce the *mixture* distribution the model actually sees, then derive
``FeatureStats(mean, std, min, max)`` for the preprocessor.

Storing sufficient statistics rather than finalized {mean, std, min, max} is
non-negotiable: ``Stream.proportion`` lets users mix datasets at training
time, and finalized stats cannot be combined into mixture stats without going
back to the raw data.

Welford form: ``M2 = sum((x - mean) ** 2)``; population variance is
``M2 / count``. The parallel merge is associative within float ULP, so
worker-reduction order does not affect the result.

NaN-masked entries (``wire.MASK_FLOAT``) are dropped from the accumulator —
they never contribute to count / mean / M2 / min / max.
"""

import json
import math
from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import fsspec
import numpy as np

# Bump on breaking changes to the on-disk JSON schema (field add/remove/rename,
# semantics change). Independent of ``hal.data.schema.SCHEMA_VERSION``, which
# governs the MDS column set; the two versions are recorded together so a
# stats file can be paired with the MDS it was derived from.
STATS_SCHEMA_VERSION: int = 1


@dataclass(frozen=True, slots=True)
class FeatureStats:
    """Finalized per-feature stats consumed by ``transformations.normalize`` et al."""

    mean: float
    std: float
    min: float
    max: float


@dataclass(frozen=True, slots=True)
class FeatureStatsSufficient:
    """Mergeable per-feature sufficient statistics. Persisted by Stage 3.

    Welford form: ``M2 = sum((x - mean) ** 2)``; population variance is
    ``M2 / count``. NaN-masked entries do not contribute to any field.
    """

    count: int
    mean: float
    m2: float
    min: float
    max: float

    def finalize(self) -> FeatureStats:
        """Convert sufficient stats to finalized FeatureStats.

        ``count == 0`` features (e.g. ``p1_nana_*`` columns when no Ice
        Climbers were present in the train split) get a unit-Gaussian
        placeholder. ``normalize`` and ``standardize`` then produce
        well-defined output (no divide-by-zero) — and since the underlying
        column is fully NaN-masked, downstream math sees NaN regardless of
        the stats values. The placeholder is a stand-in, not a guess.
        """
        if self.count == 0:
            return FeatureStats(mean=0.0, std=1.0, min=-1.0, max=1.0)
        return FeatureStats(
            mean=self.mean,
            std=math.sqrt(self.m2 / self.count),
            min=self.min,
            max=self.max,
        )


def merge_sufficient(a: FeatureStatsSufficient, b: FeatureStatsSufficient) -> FeatureStatsSufficient:
    """Parallel Welford merge of two sufficient-stat blocks. Associative within float ULP.

    Public primitive — composes per-stream blocks into any grouping the caller wants
    (e.g. mixture across datasets, consolidation across symmetric ports).
    """
    if a.count == 0:
        return b
    if b.count == 0:
        return a
    n = a.count + b.count
    delta = b.mean - a.mean
    mean = a.mean + delta * b.count / n
    m2 = a.m2 + b.m2 + delta * delta * a.count * b.count / n
    return FeatureStatsSufficient(
        count=n,
        mean=mean,
        m2=m2,
        min=min(a.min, b.min),
        max=max(a.max, b.max),
    )


def _sufficient_from_array(values: np.ndarray) -> FeatureStatsSufficient:
    """One-shot sufficient stats over a 1-D array; drops NaN entries."""
    finite = values[~np.isnan(values)] if np.issubdtype(values.dtype, np.floating) else values
    if finite.size == 0:
        return FeatureStatsSufficient(count=0, mean=0.0, m2=0.0, min=math.inf, max=-math.inf)
    mean = float(finite.mean())
    diff = finite.astype(np.float64) - mean
    m2 = float(np.dot(diff, diff))
    return FeatureStatsSufficient(
        count=int(finite.size),
        mean=mean,
        m2=m2,
        min=float(finite.min()),
        max=float(finite.max()),
    )


class StatsAccumulator:
    """Per-feature Welford accumulator with an associative merge.

    Used both intra-job (per-worker → rank 0) and across persisted files at
    training startup (per-stream → mixture).
    """

    def __init__(self, feature_names: Iterable[str]) -> None:
        self._stats: dict[str, FeatureStatsSufficient] = {
            name: FeatureStatsSufficient(count=0, mean=0.0, m2=0.0, min=math.inf, max=-math.inf)
            for name in feature_names
        }

    @property
    def feature_names(self) -> list[str]:
        return list(self._stats.keys())

    def update(self, feature_name: str, values: np.ndarray) -> None:
        if feature_name not in self._stats:
            raise KeyError(f"feature {feature_name!r} not registered with this accumulator")
        block = _sufficient_from_array(np.asarray(values).reshape(-1))
        self._stats[feature_name] = merge_sufficient(self._stats[feature_name], block)

    def merge(self, other: StatsAccumulator) -> StatsAccumulator:
        if self.feature_names != other.feature_names:
            raise ValueError("cannot merge accumulators with different feature sets")
        merged = StatsAccumulator(self.feature_names)
        for name in self._stats:
            merged._stats[name] = merge_sufficient(self._stats[name], other._stats[name])
        return merged

    def to_sufficient(self) -> dict[str, FeatureStatsSufficient]:
        return dict(self._stats)

    def finalize(self) -> dict[str, FeatureStats]:
        return {name: s.finalize() for name, s in self._stats.items()}

    @classmethod
    def from_sufficient(cls, blocks: dict[str, FeatureStatsSufficient]) -> StatsAccumulator:
        acc = cls(blocks.keys())
        for name, block in blocks.items():
            acc._stats[name] = block
        return acc


def _sufficient_to_json(block: FeatureStatsSufficient) -> dict[str, float | int]:
    return {
        "count": block.count,
        "mean": block.mean,
        "m2": block.m2,
        "min": block.min,
        "max": block.max,
    }


def _sufficient_from_json(blob: dict[str, float | int]) -> FeatureStatsSufficient:
    return FeatureStatsSufficient(
        count=int(blob["count"]),
        mean=float(blob["mean"]),
        m2=float(blob["m2"]),
        min=float(blob["min"]),
        max=float(blob["max"]),
    )


def _finalized_to_json(block: FeatureStats) -> dict[str, float]:
    return {"mean": block.mean, "std": block.std, "min": block.min, "max": block.max}


def _finalized_from_json(blob: dict[str, float]) -> FeatureStats:
    return FeatureStats(
        mean=float(blob["mean"]),
        std=float(blob["std"]),
        min=float(blob["min"]),
        max=float(blob["max"]),
    )


def dump_sufficient_stats(
    path: str | Path,
    blocks: dict[str, FeatureStatsSufficient],
    *,
    split: str,
    mds_schema_version: int,
) -> None:
    """Write sufficient stats as ``stats.json``. Accepts a local Path or any
    fsspec URL (e.g. ``s3://``)."""
    payload = {
        "schema_version": STATS_SCHEMA_VERSION,
        "mds_schema_version": mds_schema_version,
        "split": split,
        "feature_count": len(blocks),
        "sufficient": {name: _sufficient_to_json(block) for name, block in blocks.items()},
    }
    with fsspec.open(str(path), "w") as f:
        f.write(json.dumps(payload, indent=2, sort_keys=True))


def dump_finalized_stats(
    path: str | Path,
    blocks: dict[str, FeatureStats],
    *,
    mds_schema_version: int,
) -> None:
    """Write finalized stats as ``stats.json``. Used at training launch to
    snapshot the resolved mixture next to the model checkpoint."""
    payload = {
        "schema_version": STATS_SCHEMA_VERSION,
        "mds_schema_version": mds_schema_version,
        "split": "mixture",
        "feature_count": len(blocks),
        "finalized": {name: _finalized_to_json(block) for name, block in blocks.items()},
    }
    with fsspec.open(str(path), "w") as f:
        f.write(json.dumps(payload, indent=2, sort_keys=True))


def _read_stats_file(path: Path, expected_mds_schema_version: int | None) -> dict:
    payload = json.loads(Path(path).read_text())
    if payload.get("schema_version") != STATS_SCHEMA_VERSION:
        raise ValueError(
            f"{path}: stats schema_version {payload.get('schema_version')!r} != expected {STATS_SCHEMA_VERSION}"
        )
    if expected_mds_schema_version is not None:
        seen = payload.get("mds_schema_version")
        if seen != expected_mds_schema_version:
            raise ValueError(f"{path}: mds_schema_version {seen!r} != expected {expected_mds_schema_version}")
    return payload


def load_sufficient_stats(
    path: Path, *, expected_mds_schema_version: int | None = None
) -> dict[str, FeatureStatsSufficient]:
    payload = _read_stats_file(path, expected_mds_schema_version)
    if "sufficient" not in payload:
        raise ValueError(f"{path}: missing 'sufficient' block (got keys {sorted(payload)})")
    return {name: _sufficient_from_json(blob) for name, blob in payload["sufficient"].items()}


def load_dataset_stats(path: Path, *, expected_mds_schema_version: int | None = None) -> dict[str, FeatureStats]:
    """Load a single dataset's stats and finalize. Accepts either a
    'sufficient' file (Stage 3 output) or a 'finalized' file (training-launch
    snapshot)."""
    payload = _read_stats_file(path, expected_mds_schema_version)
    if "finalized" in payload:
        return {name: _finalized_from_json(blob) for name, blob in payload["finalized"].items()}
    if "sufficient" in payload:
        blocks = {name: _sufficient_from_json(blob) for name, blob in payload["sufficient"].items()}
        return {name: block.finalize() for name, block in blocks.items()}
    raise ValueError(f"{path}: stats file has neither 'sufficient' nor 'finalized' block")


def load_and_merge_stats(
    stream_stats_paths: Sequence[Path],
    proportions: Sequence[float] | None,
    *,
    expected_mds_schema_version: int | None = None,
) -> dict[str, FeatureStats]:
    """Merge per-stream sufficient stats into a single mixture distribution.

    ``proportions=None``: Welford merge over the union (correct when streams
    are sampled proportional to their size).

    ``proportions`` given: mixture-weighted (correct when ``Stream.proportion``
    re-weights streams away from natural sizes). ``p_s`` is renormalized so
    ``sum(p) == 1``; matches Mosaic Streaming, which normalizes ``proportion``
    automatically. Strictly generalizes the unweighted case
    (reduces to it under ``p_s = n_s / sum(n)``).
    """
    if not stream_stats_paths:
        raise ValueError("load_and_merge_stats called with empty stream_stats_paths")

    per_stream = [
        load_sufficient_stats(Path(p), expected_mds_schema_version=expected_mds_schema_version)
        for p in stream_stats_paths
    ]

    feature_names = list(per_stream[0].keys())
    for path, blocks in zip(stream_stats_paths, per_stream, strict=True):
        if list(blocks.keys()) != feature_names:
            raise ValueError(f"{path}: feature set differs from {stream_stats_paths[0]}; cannot merge")

    if proportions is None:
        merged = StatsAccumulator.from_sufficient(per_stream[0])
        for blocks in per_stream[1:]:
            merged = merged.merge(StatsAccumulator.from_sufficient(blocks))
        return merged.finalize()

    if len(proportions) != len(stream_stats_paths):
        raise ValueError(f"proportions length {len(proportions)} != stream count {len(stream_stats_paths)}")
    if any(p < 0 for p in proportions):
        raise ValueError("proportions must be non-negative")
    total = sum(proportions)
    if total <= 0:
        raise ValueError("proportions must sum to a positive value")
    weights = [p / total for p in proportions]

    result: dict[str, FeatureStats] = {}
    for name in feature_names:
        per = [stream[name] for stream in per_stream]
        # Restrict mixture to streams that actually observed this feature.
        # If no stream has it, fall through to the unit-Gaussian placeholder
        # (same convention as FeatureStatsSufficient.finalize).
        active = [(w, s) for w, s in zip(weights, per, strict=True) if s.count > 0]
        if not active:
            result[name] = FeatureStats(mean=0.0, std=1.0, min=-1.0, max=1.0)
            continue
        active_total = sum(w for w, _ in active)
        active_weights = [w / active_total for w, _ in active]
        active_stats = [s for _, s in active]
        mean_mix = sum(w * s.mean for w, s in zip(active_weights, active_stats, strict=True))
        # Var_mix = sum_s w_s · (Var_s + (mean_s - mean_mix)^2)
        var_mix = sum(
            w * (s.m2 / s.count + (s.mean - mean_mix) ** 2) for w, s in zip(active_weights, active_stats, strict=True)
        )
        result[name] = FeatureStats(
            mean=mean_mix,
            std=math.sqrt(var_mix),
            min=min(s.min for s in active_stats),
            max=max(s.max for s in active_stats),
        )
    return result


def float_feature_names(mds_dtypes: dict[str, np.dtype]) -> list[str]:
    """Continuous-feature whitelist derived from the MDS schema.

    Stage 3 normalizes only floating-point columns. Integer columns (action
    state, button bits, stocks) are categorical and bypass the stats path
    via embeddings or int32 casts.
    """
    return [name for name, dtype in mds_dtypes.items() if np.issubdtype(np.dtype(dtype), np.floating)]
