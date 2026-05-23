"""Trajectory → match summary metrics."""

from dataclasses import asdict
from dataclasses import dataclass

import numpy as np

from hal.sim.trajectory import Trajectory


def last_finite_stock(arr: np.ndarray) -> int:
    """Last in-game stock value. Trailing IN_GAME → menu transition frames
    carry NaN per-port fields; ``int(arr[-1])`` would either raise or
    silently report 0."""
    finite = arr[np.isfinite(arr)]
    return int(finite[-1]) if len(finite) > 0 else 0


@dataclass(frozen=True, slots=True)
class MatchSummary:
    frames: int
    p1_stocks_left: int
    p2_stocks_left: int
    p1_max_pct: float
    p2_max_pct: float

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


def summarize_trajectory(traj: Trajectory) -> MatchSummary:
    return MatchSummary(
        frames=len(traj),
        p1_stocks_left=last_finite_stock(traj.post[1]["stock"]),
        p2_stocks_left=last_finite_stock(traj.post[2]["stock"]),
        p1_max_pct=float(np.nanmax(traj.post[1]["percent"])),
        p2_max_pct=float(np.nanmax(traj.post[2]["percent"])),
    )
