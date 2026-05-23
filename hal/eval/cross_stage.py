"""Sweep an experiment's policy across stages.

The experiment passes in a ``source_factory`` that builds a fresh
``ControllerSource`` per match (rolling-buffer state must reset between
matches). Two sweep flavors:

- ``sweep_stages_vs_cpu`` — model on one port, in-game CPU on the other.
- ``sweep_stages_self_play`` — both ports driven by sources from the same
  factory pair (a coordinator that owns shared inference state, exposed
  as two ``ControllerSource`` views).
"""

from collections.abc import Callable
from collections.abc import Sequence
from typing import Literal

import melee

from hal.eval.harness import SessionConfig
from hal.eval.harness import run_match
from hal.eval.scoring import MatchSummary
from hal.eval.scoring import summarize_trajectory
from hal.sim.session import Matchup
from hal.sim.session import PlayerSetup
from hal.sim.sources import ControllerSource
from hal.sim.sources import InternalControllerSource

EgoPrefix = Literal["p1", "p2"]


def sweep_stages_vs_cpu(
    source_factory: Callable[[EgoPrefix], ControllerSource],
    *,
    session_cfg: SessionConfig,
    stages: Sequence[melee.Stage],
    character: melee.Character = melee.Character.FOX,
    cpu_level: int = 9,
    ego_port: Literal[1, 2] = 1,
    max_frames: int = 15_000,
) -> list[tuple[melee.Stage, MatchSummary | None]]:
    """One match per stage. ``source_factory(ego_prefix)`` builds a fresh
    Source for each match — the factory must NOT capture rolling state
    across calls."""
    cpu_port: Literal[1, 2] = 2 if ego_port == 1 else 1
    ego_prefix: EgoPrefix = "p1" if ego_port == 1 else "p2"
    out: list[tuple[melee.Stage, MatchSummary | None]] = []
    for stage in stages:
        matchup = Matchup(
            stage=stage,
            players=(
                PlayerSetup(port=ego_port, character=character, cpu_level=0),
                PlayerSetup(port=cpu_port, character=character, cpu_level=cpu_level),
            ),
        )
        sources = {ego_port: source_factory(ego_prefix), cpu_port: InternalControllerSource()}
        traj = run_match(session_cfg, matchup, sources, max_frames=max_frames)
        out.append((stage, summarize_trajectory(traj) if traj is not None else None))
    return out


def sweep_stages_self_play(
    coord_factory: Callable[[], tuple[ControllerSource, ControllerSource]],
    *,
    session_cfg: SessionConfig,
    stages: Sequence[melee.Stage],
    character: melee.Character = melee.Character.FOX,
    max_frames: int = 15_000,
) -> list[tuple[melee.Stage, MatchSummary | None]]:
    """One match per stage with both ports driven by paired sources.
    ``coord_factory()`` returns ``(p1_source, p2_source)`` — typically two
    views over a shared coordinator that owns batched inference state."""
    out: list[tuple[melee.Stage, MatchSummary | None]] = []
    for stage in stages:
        matchup = Matchup(
            stage=stage,
            players=(
                PlayerSetup(port=1, character=character, cpu_level=0),
                PlayerSetup(port=2, character=character, cpu_level=0),
            ),
        )
        src1, src2 = coord_factory()
        sources = {1: src1, 2: src2}
        traj = run_match(session_cfg, matchup, sources, max_frames=max_frames)
        out.append((stage, summarize_trajectory(traj) if traj is not None else None))
    return out
