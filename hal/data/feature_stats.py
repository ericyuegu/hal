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
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from typing import cast

import fsspec
import numpy as np
from numpy.typing import DTypeLike

from hal import streams
from hal.wire import ITEM_SLOTS

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


def _sufficient_from_json(blob: object, *, where: str) -> FeatureStatsSufficient:
    if not isinstance(blob, dict) or set(blob) != {"count", "mean", "m2", "min", "max"}:
        raise ValueError(f"{where}: invalid sufficient-stat fields")
    fields = cast(dict[str, object], blob)
    count = fields["count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError(f"{where}: invalid sufficient-stat count {count!r}")
    values = (fields["mean"], fields["m2"], fields["min"], fields["max"])
    if any(isinstance(value, bool) or not isinstance(value, int | float) for value in values):
        raise ValueError(f"{where}: invalid sufficient-stat numeric fields")
    mean, m2, minimum, maximum = (float(cast(int | float, value)) for value in values)
    if count == 0:
        if (mean, m2, minimum, maximum) != (0.0, 0.0, math.inf, -math.inf):
            raise ValueError(f"{where}: invalid zero-count sufficient-stat sentinel")
    elif not (all(map(math.isfinite, (mean, m2, minimum, maximum))) and m2 >= 0 and minimum <= mean <= maximum):
        raise ValueError(f"{where}: invalid sufficient-stat values for nonempty record")
    return FeatureStatsSufficient(count=count, mean=mean, m2=m2, min=minimum, max=maximum)


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
    blocks = payload.get("sufficient")
    if not isinstance(blocks, dict):
        raise ValueError(f"{path}: missing 'sufficient' block (got keys {sorted(payload)})")
    if payload.get("feature_count") != len(blocks):
        raise ValueError(f"{path}: feature_count does not match the sufficient-stat block")
    if any(not isinstance(name, str) for name in blocks):
        raise ValueError(f"{path}: sufficient-stat names must be strings")
    return {name: _sufficient_from_json(blob, where=f"{path}:{name}") for name, blob in blocks.items()}


def float_feature_names(mds_dtypes: Mapping[str, DTypeLike]) -> list[str]:
    """Continuous-feature whitelist derived from the MDS schema.

    Stage 3 normalizes only floating-point columns. Integer columns (action
    state, button bits, stocks) are categorical and bypass the stats path
    via embeddings or int32 casts.
    """
    return [name for name, dtype in mds_dtypes.items() if np.issubdtype(np.dtype(dtype), np.floating)]


_DIRECTION_STATS = FeatureStats(mean=0.0, std=1.0, min=-1.0, max=1.0)


_SCHEMA_IMPLIED_STATS = {
    "direction": _DIRECTION_STATS,
    "nana_direction": _DIRECTION_STATS,
}


_ITEM_PREFIXES: Final[tuple[str, ...]] = tuple(f"item{slot}_" for slot in range(ITEM_SLOTS))


def consolidate_key(name: str) -> str:
    """Strip ``p1_`` / ``p2_`` / ``ego_`` / ``opp_`` and fold the four item slots onto one key."""
    for pre in ("p1_", "p2_", "ego_", "opp_"):
        if name.startswith(pre):
            return name[len(pre) :]
    for pre in _ITEM_PREFIXES:
        if name.startswith(pre):
            return f"item_{name[len(pre) :]}"
    return name


def load_consolidated_mixture_stats(
    paths: Sequence[Path],
    proportions: Sequence[float],
    *,
    expected_mds_schema_version: int,
) -> dict[str, FeatureStats]:
    """Load an ego-symmetric, replay-weighted mixture of dataset statistics."""
    if not paths:
        raise ValueError("mixture statistics need at least one source")
    if len(paths) != len(proportions):
        raise ValueError(f"proportions length {len(proportions)} != source count {len(paths)}")
    if any(not math.isfinite(value) or value < 0 for value in proportions):
        raise ValueError("mixture proportions must be finite and non-negative")
    total = sum(proportions)
    if total <= 0:
        raise ValueError("mixture proportions must sum to a positive value")
    weights = [value / total for value in proportions]

    per_source: list[dict[str, FeatureStatsSufficient]] = []
    for path in paths:
        consolidated: dict[str, FeatureStatsSufficient] = {}
        selected = streams.ensure_stats(path)
        for name, block in load_sufficient_stats(
            selected, expected_mds_schema_version=expected_mds_schema_version
        ).items():
            key = consolidate_key(name)
            consolidated[key] = merge_sufficient(consolidated[key], block) if key in consolidated else block
        per_source.append(consolidated)

    feature_names = set(per_source[0])
    for path, source in zip(paths, per_source, strict=True):
        if set(source) != feature_names:
            raise ValueError(f"{path}: consolidated feature set differs from {paths[0]}; cannot merge")

    result: dict[str, FeatureStats] = {}
    for name in sorted(feature_names):
        active = [
            (weight, source[name])
            for weight, source in zip(weights, per_source, strict=True)
            if weight > 0 and source[name].count > 0
        ]
        if not active:
            result[name] = FeatureStats(mean=0.0, std=1.0, min=-1.0, max=1.0)
            continue
        active_total = sum(weight for weight, _ in active)
        normalized = [(weight / active_total, block) for weight, block in active]
        mean = sum(weight * block.mean for weight, block in normalized)
        variance = sum(weight * (block.m2 / block.count + (block.mean - mean) ** 2) for weight, block in normalized)
        result[name] = FeatureStats(
            mean=mean,
            std=math.sqrt(max(variance, 0.0)),
            min=min(block.min for _, block in normalized),
            max=max(block.max for _, block in normalized),
        )
    for name, stats in _SCHEMA_IMPLIED_STATS.items():
        result.setdefault(name, stats)
    return result
