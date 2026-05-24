"""Sim-aware, model-agnostic eval primitives.

The harness only knows about ``ControllerSource`` (single match) and
``BatchPolicy`` (N matches batched): experiments pass in their own
model-specific impl, which owns the model + preprocessing + rolling-history
state. None of this layer imports torch.

Note: ``run_match`` returns ``None`` on Session failure (e.g. Dolphin
startup race, peppi parse error) rather than raising — eval sweeps want
to log-and-continue across many stages, not abort on the first crash.
``run_matches_vec`` carries the same contract per match.
"""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from hal.sim.loop import drive
from hal.sim.session import Matchup
from hal.sim.session import Session
from hal.sim.sources import ControllerSource
from hal.sim.trajectory import Trajectory
from hal.sim.vec import BatchPolicy
from hal.sim.vec import VecMatch
from hal.sim.vec import drive_vec


@dataclass(frozen=True, slots=True)
class SessionConfig:
    """Inputs to ``Session(...)`` that don't depend on the match itself."""

    iso_path: str | Path
    dolphin_path: str | Path
    use_exi_inputs: bool = True
    enable_ffw: bool = True
    emulation_speed: float = 0.0
    blocking_input: bool = True
    replay_dir: str | Path | None = None
    step_timeout_seconds: float = 30.0
    tmp_home_directory: bool = True


def _build_session(session_cfg: SessionConfig, *, slippi_port: int, replay_dir: str | Path | None) -> Session:
    """Construct (don't enter) a Session from a SessionConfig, overriding the
    two fields that must differ per concurrent instance: ``slippi_port`` and
    ``replay_dir``."""
    return Session(
        iso_path=session_cfg.iso_path,
        dolphin_path=session_cfg.dolphin_path,
        slippi_port=slippi_port,
        blocking_input=session_cfg.blocking_input,
        tmp_home_directory=session_cfg.tmp_home_directory,
        replay_dir=replay_dir,
        step_timeout_seconds=session_cfg.step_timeout_seconds,
        use_exi_inputs=session_cfg.use_exi_inputs,
        enable_ffw=session_cfg.enable_ffw,
        emulation_speed=session_cfg.emulation_speed,
    )


def run_match(
    session_cfg: SessionConfig,
    matchup: Matchup,
    sources: Mapping[int, ControllerSource],
    *,
    max_frames: int,
) -> Trajectory | None:
    """Drive one match end-to-end. Returns the trajectory, or None if the
    Session raised (logged at WARNING)."""
    try:
        with _build_session(session_cfg, slippi_port=51441, replay_dir=session_cfg.replay_dir) as s:
            return drive(s, matchup, sources, max_frames=max_frames)
    except Exception as e:
        logger.warning(f"run_match: Session crashed: {e!r}")
        return None


def run_matches_vec(
    session_cfg: SessionConfig,
    matches: Sequence[VecMatch],
    policy_factory: Callable[[], BatchPolicy],
    *,
    max_frames: int,
    max_parallel: int,
    base_slippi_port: int = 51441,
) -> list[Trajectory | None]:
    """Run ``matches`` concurrently in waves of up to ``max_parallel`` Sessions,
    each frame batched through a single ``BatchPolicy`` call (see ``drive_vec``).

    Each wave's Sessions get distinct slippi_ports (``base_slippi_port + offset``)
    and, when ``session_cfg.replay_dir`` is set, a per-match replay subdir so
    their .slps don't collide. ``policy_factory`` builds a fresh policy per wave —
    per-slot rolling state must not leak across waves, and ``Slot.match`` indices
    restart at 0 each wave. Returns one entry per match, aligned to ``matches``;
    ``None`` where that Session failed.

    Per-match Session crashes are isolated inside ``drive_vec``. A failure in the
    shared batched policy call (e.g. CUDA OOM across the wave's slots) can't be
    attributed to one match, so the whole wave is logged and left ``None`` rather
    than aborting the sweep — the same log-and-continue contract as ``run_match``.
    """
    if max_parallel < 1:
        raise ValueError(f"max_parallel must be >= 1, got {max_parallel}")
    base_replay = Path(session_cfg.replay_dir) if session_cfg.replay_dir is not None else None
    out: list[Trajectory | None] = [None] * len(matches)
    for wave_start in range(0, len(matches), max_parallel):
        wave = list(range(wave_start, min(wave_start + max_parallel, len(matches))))
        try:
            sessions: list[Session] = []
            for offset, gi in enumerate(wave):
                replay_dir = None
                if base_replay is not None:
                    replay_dir = base_replay / f"match_{gi:03d}"
                    replay_dir.mkdir(parents=True, exist_ok=True)
                sessions.append(
                    _build_session(session_cfg, slippi_port=base_slippi_port + offset, replay_dir=replay_dir)
                )
            trajs = drive_vec(sessions, [matches[gi] for gi in wave], policy_factory(), max_frames=max_frames)
        except Exception as e:
            logger.warning(f"run_matches_vec: wave {wave[0]}..{wave[-1]} failed: {e!r}; its matches stay None")
            continue
        for gi, traj in zip(wave, trajs, strict=True):
            out[gi] = traj
    return out
