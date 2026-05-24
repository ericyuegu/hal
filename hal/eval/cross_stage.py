"""Sweep an experiment's policy across stages, in parallel.

Each sweep builds a grid of matches (``stages × replicas``) and runs them
concurrently through ``run_matches_vec`` — every live model-driven port across
all matches is fed to a single batched ``BatchPolicy`` call per frame. The
experiment passes in a ``policy_factory`` that builds a fresh ``BatchPolicy``
per wave (rolling-buffer state must reset between waves). Replicas of the same
stage diverge naturally via the policy's own per-step sampling.

Two flavors:

- ``sweep_vs_cpu`` — model on one port, in-game CPU on the other.
- ``sweep_self_play`` — both ports driven by the same batched policy.
"""

from collections.abc import Callable
from collections.abc import Sequence
from typing import Literal

import melee

from hal.eval.harness import SessionConfig
from hal.eval.harness import run_matches_vec
from hal.eval.scoring import MatchSummary
from hal.eval.scoring import summarize_trajectory
from hal.sim.session import Matchup
from hal.sim.session import PlayerSetup
from hal.sim.vec import BatchPolicy
from hal.sim.vec import VecMatch

# (stage, replica index, summary-or-None-if-crashed) per match in the grid.
SweepResult = list[tuple[melee.Stage, int, MatchSummary | None]]


def sweep_vs_cpu(
    policy_factory: Callable[[], BatchPolicy],
    *,
    session_cfg: SessionConfig,
    stages: Sequence[melee.Stage],
    max_parallel: int,
    replicas: int = 1,
    character: melee.Character = melee.Character.FOX,
    cpu_level: int = 9,
    ego_port: Literal[1, 2] = 1,
    max_frames: int = 15_000,
) -> SweepResult:
    """``replicas`` matches per stage, model on ``ego_port`` vs a level
    ``cpu_level`` CPU. All matches run concurrently in waves of ``max_parallel``."""
    cpu_port: Literal[1, 2] = 2 if ego_port == 1 else 1
    grid = [(stage, r) for stage in stages for r in range(replicas)]
    matches = [
        VecMatch(
            matchup=Matchup(
                stage=stage,
                players=(
                    PlayerSetup(port=ego_port, character=character, cpu_level=0),
                    PlayerSetup(port=cpu_port, character=character, cpu_level=cpu_level),
                ),
            ),
            model_ports=(ego_port,),
        )
        for stage, _ in grid
    ]
    trajs = run_matches_vec(session_cfg, matches, policy_factory, max_frames=max_frames, max_parallel=max_parallel)
    return [
        (stage, r, summarize_trajectory(t) if t is not None else None)
        for (stage, r), t in zip(grid, trajs, strict=True)
    ]


def sweep_self_play(
    policy_factory: Callable[[], BatchPolicy],
    *,
    session_cfg: SessionConfig,
    stages: Sequence[melee.Stage],
    max_parallel: int,
    replicas: int = 1,
    character: melee.Character = melee.Character.FOX,
    max_frames: int = 15_000,
) -> SweepResult:
    """``replicas`` matches per stage with both ports driven by the batched
    policy. All matches run concurrently in waves of ``max_parallel``."""
    grid = [(stage, r) for stage in stages for r in range(replicas)]
    matches = [
        VecMatch(
            matchup=Matchup(
                stage=stage,
                players=(
                    PlayerSetup(port=1, character=character, cpu_level=0),
                    PlayerSetup(port=2, character=character, cpu_level=0),
                ),
            ),
            model_ports=(1, 2),
        )
        for stage, _ in grid
    ]
    trajs = run_matches_vec(session_cfg, matches, policy_factory, max_frames=max_frames, max_parallel=max_parallel)
    return [
        (stage, r, summarize_trajectory(t) if t is not None else None)
        for (stage, r), t in zip(grid, trajs, strict=True)
    ]
