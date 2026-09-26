"""Match results and timing summaries shared by local and netplay drivers."""

import math
from collections.abc import Collection
from dataclasses import dataclass
from typing import Protocol

from hal.sim.trajectory import Trajectory


@dataclass(frozen=True, slots=True)
class PlayResult:
    trajectory: Trajectory
    ego_port: int
    opponent_port: int
    stage: int
    wall_seconds: float
    inference_seconds: tuple[float, ...]
    frame_interval_seconds: tuple[float, ...]
    dolphin_step_seconds: tuple[float, ...]
    transport_correction_frames: int

    @property
    def inference_p95_ms(self) -> float:
        """Nearest-rank p95 of the complete policy call."""
        return p95_ms(self.inference_seconds)

    @property
    def game_fps(self) -> float:
        elapsed = sum(self.frame_interval_seconds)
        return len(self.frame_interval_seconds) / elapsed if elapsed > 0 else 0.0

    @property
    def frame_interval_p95_ms(self) -> float:
        return p95_ms(self.frame_interval_seconds)

    @property
    def dolphin_step_p95_ms(self) -> float:
        return p95_ms(self.dolphin_step_seconds)


class PlayObserver(Protocol):
    def observe_policy(self, seconds: float) -> None: ...

    def observe_frame(self, frame_id: int, dolphin_step_seconds: float) -> None: ...


def p95_ms(values: Collection[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return 1_000.0 * ordered[math.ceil(0.95 * len(ordered)) - 1]
