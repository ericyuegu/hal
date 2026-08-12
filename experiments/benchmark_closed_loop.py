"""Measure the spawned closed-loop Session path with a zero-cost policy."""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections.abc import Mapping
from collections.abc import Sequence

import melee
import numpy as np

from hal.eval.harness import SessionConfig
from hal.eval.harness import automatic_parallelism
from hal.eval.harness import default_session_cfg
from hal.eval.harness import run_matches_vec
from hal.sim.process_vec import ProcessVecTelemetry
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


def _match(*, self_play: bool) -> VecMatch:
    matchup = Matchup(
        stage=melee.Stage.FINAL_DESTINATION,
        players=(
            PlayerSetup(port=1, character=melee.Character.FOX),
            PlayerSetup(port=2, character=melee.Character.FOX, cpu_level=0 if self_play else 9),
        ),
    )
    return VecMatch(matchup=matchup, model_ports=(1, 2) if self_play else (1,))


def benchmark(
    session_cfg: SessionConfig,
    *,
    workers: int,
    frames: int,
    prediction_frames: int,
    execution_stride: int,
    self_play: bool,
    run: int,
) -> dict[str, object]:
    telemetry = ProcessVecTelemetry()
    started = time.perf_counter()
    trajectories = run_matches_vec(
        session_cfg,
        [_match(self_play=self_play) for _ in range(workers)],
        lambda: NeutralChunkPolicy(
            prediction_frames=prediction_frames,
            execution_stride=execution_stride,
        ),
        max_frames=frames,
        max_parallel=covering_power_of_two(workers),
        start_retries=0,
        process_telemetry=telemetry,
    )
    elapsed = time.perf_counter() - started
    captured = sum(len(trajectory) for boot in trajectories for trajectory in boot)
    completed = sum(bool(boot) for boot in trajectories)
    result: dict[str, object] = {
        "run": run,
        "workers": workers,
        "completed": completed,
        "captured": captured,
        "elapsed_seconds": elapsed,
        "aggregate_fps": captured / elapsed,
        "wall_lockstep_fps": captured / max(1, completed) / elapsed,
        "slippi_ports": [51441, 51441 + workers - 1],
        **telemetry.metrics(),
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=automatic_parallelism())
    parser.add_argument("--frames", type=int, default=2400)
    parser.add_argument("--prediction-frames", type=int, default=4)
    parser.add_argument("--execution-stride", type=int, default=4)
    parser.add_argument("--self-play", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    print(
        json.dumps(
            {
                "event": "environment",
                "pid": os.getpid(),
                "python": sys.version,
                "default_start_method": mp.get_start_method(),
                "session_worker_start_method": "spawn",
                "cpu_count": os.cpu_count(),
                "cpu_affinity": affinity,
                "requested_workers": args.workers,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    results = [
        benchmark(
            default_session_cfg(),
            workers=args.workers,
            frames=args.frames,
            prediction_frames=args.prediction_frames,
            execution_stride=args.execution_stride,
            self_play=args.self_play,
            run=run,
        )
        for run in range(1, args.repeat + 1)
    ]
    if any(result["completed"] != args.workers for result in results):
        raise SystemExit("one or more closed-loop workers failed")


if __name__ == "__main__":
    main()
