"""Consolidate per-port dataset stats into a symmetric ego/opp view.

The raw MDS stats file has separate entries for ``p1_*`` and ``p2_*`` columns,
which is correct on disk: each port is its own observation. But at train time
every experiment relabels ``p1/p2`` → ``ego/opp`` and feeds the same model
both perspectives, so the model wants ONE distribution per feature, not two.

Welford-merge the per-port sufficient stats here before finalizing, and key
the result by the bare feature name (``position_x``, ``percent``, …).
"""

import math
from collections.abc import Sequence
from pathlib import Path

from hal import streams
from hal.data.feature_stats import FeatureStats
from hal.data.feature_stats import FeatureStatsSufficient
from hal.data.feature_stats import load_sufficient_stats
from hal.data.feature_stats import merge_sufficient

# Compact policy datasets pack facing direction into the integer player state,
# so it is intentionally absent from their float stats sidecar. Decoding
# restores exactly {-1, 0, 1, NaN}; preprocessing min-max scales it and
# therefore needs only these schema-defined bounds. Leaders and Nana
# consolidate to separate keys because their missingness distributions differ.
_DIRECTION_STATS = FeatureStats(mean=0.0, std=1.0, min=-1.0, max=1.0)
_SCHEMA_IMPLIED_STATS = {
    "direction": _DIRECTION_STATS,
    "nana_direction": _DIRECTION_STATS,
}


def consolidate_key(name: str) -> str:
    """Strip ``p1_`` / ``p2_`` / ``ego_`` / ``opp_`` so symmetric features collapse."""
    for pre in ("p1_", "p2_", "ego_", "opp_"):
        if name.startswith(pre):
            return name[len(pre) :]
    return name


def load_consolidated_stats(path: Path) -> dict[str, FeatureStats]:
    """Fetch selected stream stats, merge p1/p2 ports, and finalize."""
    path = streams.ensure_stats(path)
    merged: dict[str, FeatureStatsSufficient] = {}
    for name, block in load_sufficient_stats(path).items():
        key = consolidate_key(name)
        merged[key] = merge_sufficient(merged[key], block) if key in merged else block
    result = {k: b.finalize() for k, b in merged.items()}
    for name, stats in _SCHEMA_IMPLIED_STATS.items():
        result.setdefault(name, stats)
    return result


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
