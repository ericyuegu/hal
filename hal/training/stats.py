"""Consolidate per-port dataset stats into a symmetric ego/opp view.

The raw MDS stats file has separate entries for ``p1_*`` and ``p2_*`` columns,
which is correct on disk: each port is its own observation. But at train time
every experiment relabels ``p1/p2`` → ``ego/opp`` and feeds the same model
both perspectives, so the model wants ONE distribution per feature, not two.

Welford-merge the per-port sufficient stats here before finalizing, and key
the result by the bare feature name (``position_x``, ``percent``, …).
"""

from pathlib import Path

from hal.data.stats import FeatureStats
from hal.data.stats import FeatureStatsSufficient
from hal.data.stats import load_sufficient_stats
from hal.data.stats import merge_sufficient


def consolidate_key(name: str) -> str:
    """Strip ``p1_`` / ``p2_`` / ``ego_`` / ``opp_`` so symmetric features collapse."""
    for pre in ("p1_", "p2_", "ego_", "opp_"):
        if name.startswith(pre):
            return name[len(pre) :]
    return name


def load_consolidated_stats(path: Path) -> dict[str, FeatureStats]:
    """Welford-merge sufficient stats across p1/p2 ports, then finalize."""
    merged: dict[str, FeatureStatsSufficient] = {}
    for name, block in load_sufficient_stats(path).items():
        key = consolidate_key(name)
        merged[key] = merge_sufficient(merged[key], block) if key in merged else block
    return {k: b.finalize() for k, b in merged.items()}
