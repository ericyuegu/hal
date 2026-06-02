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

from hal.fixtures import DOLPHIN_EXIAI
from hal.fixtures import ISO
from hal.fixtures import ensure
from hal.paths import EMULATOR_PATH
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
    # Wall-clock cap on driving menus to the first in-game frame. Legit navigation
    # settles in a few seconds even under concurrent load; a stage-select flake
    # (libmelee's cursor limit-cycling) spins to the cap, so keep it tight — the
    # rarer flake then costs ~30s before run_matches_vec retries on a fresh Session,
    # not 120s.
    start_timeout_seconds: float = 30.0
    tmp_home_directory: bool = True
    # Eval sessions poll slippstream so a hung/paused match trips
    # step_timeout_seconds instead of blocking forever (see Session.polling_mode).
    polling_mode: bool = True


def default_session_cfg(replay_dir: Path | None = None) -> SessionConfig:
    """The standard headless eval Session: exi-ai Dolphin + fixture ISO, fast-
    forward, blocking input, throwaway tmp home. ``replay_dir`` (when not None)
    preserves the match .slps; else they die with the Session's tmp home."""
    ensure(DOLPHIN_EXIAI)
    return SessionConfig(
        iso_path=ensure(ISO),
        dolphin_path=EMULATOR_PATH,
        use_exi_inputs=True,
        enable_ffw=True,
        emulation_speed=0.0,
        blocking_input=True,
        step_timeout_seconds=30.0,
        tmp_home_directory=True,
        replay_dir=str(replay_dir) if replay_dir is not None else None,
    )


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
        start_timeout_seconds=session_cfg.start_timeout_seconds,
        use_exi_inputs=session_cfg.use_exi_inputs,
        enable_ffw=session_cfg.enable_ffw,
        emulation_speed=session_cfg.emulation_speed,
        polling_mode=session_cfg.polling_mode,
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


def _drive_wave(
    session_cfg: SessionConfig,
    indices: Sequence[int],
    matches: Sequence[VecMatch],
    policy_factory: Callable[[], BatchPolicy],
    *,
    max_frames: int,
    base_replay: Path | None,
    slippi_port_base: int,
) -> dict[int, Trajectory | None]:
    """Build fresh Sessions for the given global match ``indices`` and drive them
    once through ``drive_vec``. Returns ``{global_index: Trajectory | None}``.

    A wave-wide failure (Session build or the shared batched-policy call, e.g. CUDA
    OOM) can't be attributed to one match, so every index is left ``None`` and
    logged — the log-and-continue contract shared with ``run_match``."""
    try:
        sessions: list[Session] = []
        for offset, gi in enumerate(indices):
            replay_dir = None
            if base_replay is not None:
                replay_dir = base_replay / f"match_{gi:03d}"
                replay_dir.mkdir(parents=True, exist_ok=True)
            sessions.append(_build_session(session_cfg, slippi_port=slippi_port_base + offset, replay_dir=replay_dir))
        trajs = drive_vec(sessions, [matches[gi] for gi in indices], policy_factory(), max_frames=max_frames)
    except Exception as e:
        logger.warning(f"run_matches_vec: wave {list(indices)} failed: {e!r}; its matches stay None")
        return {gi: None for gi in indices}
    return dict(zip(indices, trajs, strict=True))


def run_matches_vec(
    session_cfg: SessionConfig,
    matches: Sequence[VecMatch],
    policy_factory: Callable[[], BatchPolicy],
    *,
    max_frames: int,
    max_parallel: int,
    base_slippi_port: int = 51441,
    start_retries: int = 2,
) -> list[Trajectory | None]:
    """Run ``matches`` concurrently in waves of up to ``max_parallel`` Sessions,
    each frame batched through a single ``BatchPolicy`` call (see ``drive_vec``).

    Each wave's Sessions get distinct slippi_ports (``base_slippi_port + offset``)
    and, when ``session_cfg.replay_dir`` is set, a per-match replay subdir so
    their .slps don't collide. ``policy_factory`` builds a fresh policy per wave —
    per-slot rolling state must not leak across waves, and ``Slot.match`` indices
    restart at 0 each wave. Returns one entry per match, aligned to ``matches``;
    ``None`` where that Session failed after all retries.

    libmelee's stage-select cursor navigation flakily fails to settle under
    concurrent FFW load (frame-delivery jitter starves its bang-bang controller),
    so a match intermittently never reaches IN_GAME and ``start_match`` trips its
    wall-clock cap. ``start_retries`` re-drives the still-``None`` matches of a
    wave on fresh Sessions (new Dolphin + slippi_port) to absorb that ~per-match
    flake; a wholly-dead match still ends up ``None`` and is logged.
    """
    if max_parallel < 1:
        raise ValueError(f"max_parallel must be >= 1, got {max_parallel}")
    base_replay = Path(session_cfg.replay_dir) if session_cfg.replay_dir is not None else None
    out: list[Trajectory | None] = [None] * len(matches)
    for wave_start in range(0, len(matches), max_parallel):
        pending = list(range(wave_start, min(wave_start + max_parallel, len(matches))))
        for attempt in range(start_retries + 1):
            # Fresh ports per attempt so a stuck-but-not-yet-reaped Dolphin from the
            # previous try can't collide with the retry's slippstream server.
            slippi_port_base = base_slippi_port + (attempt % 8) * max_parallel
            results = _drive_wave(
                session_cfg,
                pending,
                matches,
                policy_factory,
                max_frames=max_frames,
                base_replay=base_replay,
                slippi_port_base=slippi_port_base,
            )
            for gi, traj in results.items():
                if traj is not None:
                    out[gi] = traj
            pending = [gi for gi in pending if out[gi] is None]
            if not pending:
                break
            if attempt < start_retries:
                logger.warning(
                    f"run_matches_vec: {len(pending)} match(es) failed to reach IN_GAME; "
                    f"retrying on fresh Sessions (attempt {attempt + 2}/{start_retries + 1})"
                )
    return out
