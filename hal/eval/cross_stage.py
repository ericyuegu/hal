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
from hal.eval.matchups import matchups_for
from hal.eval.scoring import MatchSummary
from hal.eval.scoring import summarize_trajectory
from hal.sim.session import Matchup
from hal.sim.session import PlayerSetup
from hal.sim.vec import BatchPolicy
from hal.sim.vec import VecMatch

# (stage, replica index, summary-or-None-if-crashed) per match in the grid.
SweepResult = list[tuple[melee.Stage, int, MatchSummary | None]]

STARTING_STOCKS = 4  # Melee match default
FRAMES_PER_MINUTE = 3600  # 60 fps
# Seed stage for the one menu navigation an instant-restart boot makes before the
# Gecko code takes over with random legal stages. Battlefield's cursor target sits
# near the menu origin, so it is the most reliable single nav under concurrent load.
PRIOR_SWEEP_SEED_STAGE = melee.Stage.BATTLEFIELD


def vs_cpu_metrics(result: SweepResult) -> dict[str, float]:
    """Reduce a ``sweep_vs_cpu`` grid to a flat metric dict for logging.

    Assumes the ego model is on port 1 vs a CPU on port 2 (the ``sweep_vs_cpu``
    default). Stocks/damage are reported as **per-game-minute rates**, pooled
    frame-weighted over every non-crashed match (``sum(metric) / sum(minutes)``)
    so the numbers are comparable across runs regardless of how many episodes ran
    or how long each lasted. ``crashed`` is the fraction of matches whose Session
    failed; an all-crashed (or zero-frame) sweep reports only ``{"crashed": 1.0}``.
    """
    summaries = [s for _, _, s in result if s is not None]
    total_frames = sum(s.frames for s in summaries)
    if not summaries or total_frames == 0:
        return {"crashed": 1.0}
    minutes = total_frames / FRAMES_PER_MINUTE
    return {
        "stocks_taken_per_min": sum(STARTING_STOCKS - s.p2_stocks_left for s in summaries) / minutes,
        "stocks_lost_per_min": sum(STARTING_STOCKS - s.p1_stocks_left for s in summaries) / minutes,
        "damage_dealt_per_min": sum(s.p2_damage_taken for s in summaries) / minutes,  # dealt by ego = taken by p2
        "damage_taken_per_min": sum(s.p1_damage_taken for s in summaries) / minutes,
        "frames": total_frames / len(summaries),  # mean episode length, kept as a diagnostic
        "matches": float(len(summaries)),  # completed matches pooled (many per boot under instant-restart)
        "crashed": (len(result) - len(summaries)) / len(result),
    }


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
    boots = run_matches_vec(session_cfg, matches, policy_factory, max_frames=max_frames, max_parallel=max_parallel)
    # No instant-restart on this path → each boot is one match (or empty if it crashed).
    return [
        (stage, r, summarize_trajectory(boot[0]) if boot else None)
        for (stage, r), boot in zip(grid, boots, strict=True)
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
    boots = run_matches_vec(session_cfg, matches, policy_factory, max_frames=max_frames, max_parallel=max_parallel)
    return [
        (stage, r, summarize_trajectory(boot[0]) if boot else None)
        for (stage, r), boot in zip(grid, boots, strict=True)
    ]


def sweep_vs_cpu_prior(
    policy_factory: Callable[[], BatchPolicy],
    *,
    session_cfg: SessionConfig,
    n_matchups: int,
    max_parallel: int,
    cpu_level: int = 9,
    ego_port: Literal[1, 2] = 1,
    seed_stage: melee.Stage = PRIOR_SWEEP_SEED_STAGE,
    max_frames: int = 15_000,
) -> SweepResult:
    """Prior-distribution vs-CPU sweep for instant-restart sessions.

    ``n_matchups`` deterministic ``(ego_char, opp_char)`` boots are drawn from the
    training matchup prior (``matchups_for``); each boots once to ``seed_stage`` and
    then — via the Gecko "Instant Match" code (``session_cfg.instant_match_restart``
    must be set) — plays many matches back-to-back on random legal stages within
    ``max_frames``. Every completed match becomes one ``SweepResult`` row (the
    ``stage`` label is the seed, kept only for shape compatibility with
    ``vs_cpu_metrics``); a boot that produced no match contributes one ``None`` row.
    Pool the rows with ``vs_cpu_metrics`` exactly as the fixed sweep."""
    cpu_port: Literal[1, 2] = 2 if ego_port == 1 else 1
    matches = [
        VecMatch(
            matchup=Matchup(
                stage=seed_stage,
                players=(
                    PlayerSetup(port=ego_port, character=ego_char, cpu_level=0),
                    PlayerSetup(port=cpu_port, character=opp_char, cpu_level=cpu_level),
                ),
            ),
            model_ports=(ego_port,),
        )
        for ego_char, opp_char in matchups_for(n_matchups)
    ]
    boots = run_matches_vec(session_cfg, matches, policy_factory, max_frames=max_frames, max_parallel=max_parallel)
    out: SweepResult = []
    for bi, boot in enumerate(boots):
        if not boot:
            out.append((seed_stage, bi, None))  # boot never reached IN_GAME (hung/crashed)
        else:
            out.extend((seed_stage, bi, summarize_trajectory(t)) for t in boot)
    return out
