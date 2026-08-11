"""Measure the spawned closed-loop Session path with a zero-cost policy."""

import argparse
import time
from collections.abc import Mapping
from collections.abc import Sequence

import melee
import numpy as np

from hal.eval.harness import SessionConfig
from hal.eval.harness import automatic_parallelism
from hal.eval.harness import default_session_cfg
from hal.eval.harness import run_matches_vec
from hal.sim.rollout import ObservationRow
from hal.sim.rollout import PolicyRuntimeSpec
from hal.sim.rollout import covering_power_of_two
from hal.sim.session import Matchup
from hal.sim.session import PlayerSetup
from hal.sim.vec import Slot
from hal.sim.vec import VecMatch
from hal.wire import ACTION_DIM


class NeutralChunkPolicy:
    """A configurable action-chunk policy with no model cost."""

    def __init__(self, *, prediction_frames: int, execution_stride: int) -> None:
        self._runtime = PolicyRuntimeSpec(
            context_frames=1,
            prediction_frames=prediction_frames,
            execution_stride=execution_stride,
            committed_frames=0,
            action_dim=ACTION_DIM,
        )

    @property
    def runtime_spec(self) -> PolicyRuntimeSpec:
        return self._runtime

    def plan_rows(self, rows: Mapping[Slot, Sequence[ObservationRow]]) -> Mapping[Slot, np.ndarray]:
        shape = (self._runtime.prediction_frames, self._runtime.action_dim)
        return {slot: np.zeros(shape, dtype=np.float32) for slot in rows}


def _match() -> VecMatch:
    matchup = Matchup(
        stage=melee.Stage.FINAL_DESTINATION,
        players=(
            PlayerSetup(port=1, character=melee.Character.FOX),
            PlayerSetup(port=2, character=melee.Character.FOX, cpu_level=9),
        ),
    )
    return VecMatch(matchup=matchup, model_ports=(1,))


def benchmark(
    session_cfg: SessionConfig,
    *,
    workers: int,
    frames: int,
    prediction_frames: int,
    execution_stride: int,
) -> None:
    started = time.perf_counter()
    trajectories = run_matches_vec(
        session_cfg,
        [_match() for _ in range(workers)],
        lambda: NeutralChunkPolicy(
            prediction_frames=prediction_frames,
            execution_stride=execution_stride,
        ),
        max_frames=frames,
        max_parallel=covering_power_of_two(workers),
        start_retries=0,
    )
    elapsed = time.perf_counter() - started
    captured = sum(len(trajectory) for boot in trajectories for trajectory in boot)
    completed = sum(bool(boot) for boot in trajectories)
    print(
        f"workers={workers} completed={completed} captured={captured} "
        f"elapsed={elapsed:.3f}s aggregate_fps={captured / elapsed:.1f} "
        f"wall_lockstep_fps={captured / max(1, completed) / elapsed:.1f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=automatic_parallelism())
    parser.add_argument("--frames", type=int, default=2400)
    parser.add_argument("--prediction-frames", type=int, default=4)
    parser.add_argument("--execution-stride", type=int, default=4)
    args = parser.parse_args()
    benchmark(
        default_session_cfg(),
        workers=args.workers,
        frames=args.frames,
        prediction_frames=args.prediction_frames,
        execution_stride=args.execution_stride,
    )


if __name__ == "__main__":
    main()
