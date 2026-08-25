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

import os
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from loguru import logger

from hal.fixtures import DOLPHIN_EXIAI
from hal.fixtures import ISO
from hal.fixtures import ensure
from hal.paths import EMULATOR_PATH
from hal.sim.loop import drive
from hal.sim.process_vec import PolicyExecutionError
from hal.sim.process_vec import ProcessVecTelemetry
from hal.sim.process_vec import SharedChunkPolicy
from hal.sim.process_vec import drive_process_vec
from hal.sim.rollout import nearest_power_of_two
from hal.sim.session import Matchup
from hal.sim.session import Session
from hal.sim.sources import ControllerSource
from hal.sim.trajectory import Trajectory
from hal.sim.vec import BatchPolicy
from hal.sim.vec import VecMatch
from hal.sim.vec import drive_vec

DEFAULT_START_RETRIES = 2


def usable_cpus() -> int:
    """CPUs this process may actually run on — the ceiling for concurrent Dolphin boots.

    ``os.cpu_count`` reports the HOST's cores, so inside a container with a smaller
    CPU quota it oversubscribes: a 16-vCPU box reported 64 and booted 64 lockstep
    emulators. The affinity mask is the real allowance; where the platform has no
    such mask, the core count is the best answer available.
    """
    getaffinity = getattr(os, "sched_getaffinity", None)
    if getaffinity is not None:
        return max(1, len(getaffinity(0)))
    return max(1, os.cpu_count() or 1)


def automatic_parallelism() -> int:
    """Nearest power-of-two worker bucket for the CPUs available to this process."""
    return nearest_power_of_two(usable_cpus())


def resolve_parallelism(n_matches: int, requested: int | None) -> int:
    """Resolve active workers without imposing a hidden hardware ceiling."""
    if n_matches < 1:
        raise ValueError(f"n_matches must be >= 1, got {n_matches}")
    parallel = automatic_parallelism() if requested is None else requested
    if parallel < 1:
        raise ValueError(f"max_parallel must be >= 1, got {parallel}")
    return min(n_matches, parallel)


@dataclass(frozen=True, slots=True)
class SessionConfig:
    """Inputs to ``Session(...)`` that don't depend on the match itself."""

    iso_path: str | Path
    dolphin_path: str | Path
    use_exi_inputs: bool = True
    enable_ffw: bool = True
    emulation_speed: float = 0.0
    blocking_input: bool = True
    disable_audio: bool = True
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
    # Boot Dolphin with the "Instant Match" Gecko code so a finished match restarts
    # directly into a new one (random legal stage), skipping the flaky stage-select
    # menu on every match after the first; drive_vec then plays many matches per boot.
    # Off by default — single-match-per-boot is the historical sweep contract; an eval
    # that wants many prior-sampled matches per boot (see hal.eval.matchups) opts in.
    instant_match_restart: bool = False


def default_session_cfg(replay_dir: Path | None = None, *, instant_match_restart: bool = False) -> SessionConfig:
    """The standard headless eval Session: exi-ai Dolphin + fixture ISO, fast-
    forward, blocking input, throwaway tmp home. ``replay_dir`` (when not None)
    preserves the match .slps; else they die with the Session's tmp home.
    ``instant_match_restart`` opts into the many-matches-per-boot eval flow."""
    ensure(DOLPHIN_EXIAI)
    return SessionConfig(
        iso_path=ensure(ISO),
        dolphin_path=EMULATOR_PATH,
        use_exi_inputs=True,
        enable_ffw=True,
        emulation_speed=0.0,
        blocking_input=True,
        disable_audio=True,
        step_timeout_seconds=30.0,
        tmp_home_directory=True,
        replay_dir=str(replay_dir) if replay_dir is not None else None,
        instant_match_restart=instant_match_restart,
    )


def _build_session(session_cfg: SessionConfig, *, slippi_port: int, replay_dir: str | Path | None) -> Session:
    """Construct (don't enter) a Session from a SessionConfig, overriding the
    two fields that must differ per concurrent instance: ``slippi_port`` and
    ``replay_dir``."""
    return Session(**_session_kwargs(session_cfg, slippi_port=slippi_port, replay_dir=replay_dir))


def _session_kwargs(
    session_cfg: SessionConfig, *, slippi_port: int, replay_dir: str | Path | None
) -> dict[str, object]:
    """Spawn-safe Session constructor values for one worker."""
    return dict(
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
        disable_audio=session_cfg.disable_audio,
        polling_mode=session_cfg.polling_mode,
        instant_match_restart=session_cfg.instant_match_restart,
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
    process_telemetry: ProcessVecTelemetry | None = None,
    process_cohorts: int = 1,
) -> dict[int, list[Trajectory]]:
    """Build fresh Sessions for the given global boot ``indices`` and drive them
    once through ``drive_vec``. Returns ``{global_index: [Trajectory, ...]}`` — the
    matches each boot played (empty if it failed to start or crashed first).

    A wave-wide failure (Session build or the shared batched-policy call, e.g. CUDA
    OOM) can't be attributed to one boot, so every index is left empty and logged —
    the log-and-continue contract shared with ``run_match``."""
    try:
        replay_dirs: list[Path | None] = []
        for gi in indices:
            replay_dir = None
            if base_replay is not None:
                replay_dir = base_replay / f"boot_{gi:03d}"
                replay_dir.mkdir(parents=True, exist_ok=True)
            replay_dirs.append(replay_dir)
        wave_matches = [matches[gi] for gi in indices]
        policy = policy_factory()
        process_capable = (
            hasattr(policy, "runtime_spec")
            and callable(getattr(policy, "plan_rows", None))
            and all(match.model_ports for match in wave_matches)
        )
        if process_capable:
            kwargs = [
                _session_kwargs(session_cfg, slippi_port=slippi_port_base + offset, replay_dir=replay_dir)
                for offset, replay_dir in enumerate(replay_dirs)
            ]
            boots = drive_process_vec(
                kwargs,
                wave_matches,
                cast(SharedChunkPolicy, policy),
                max_frames=max_frames,
                instant_restart=session_cfg.instant_match_restart,
                telemetry=process_telemetry,
                failure_dir=base_replay,
                cohort_count=min(process_cohorts, len(wave_matches)),
            )
        else:
            if process_cohorts != 1:
                raise ValueError("process_cohorts requires a spawned-driver policy with runtime_spec and plan_rows")
            sessions = [
                _build_session(session_cfg, slippi_port=slippi_port_base + offset, replay_dir=replay_dir)
                for offset, replay_dir in enumerate(replay_dirs)
            ]
            boots = drive_vec(
                sessions,
                wave_matches,
                policy,
                max_frames=max_frames,
                instant_restart=session_cfg.instant_match_restart,
            )
    except PolicyExecutionError:
        raise
    except Exception as e:
        logger.warning(f"run_matches_vec: wave {list(indices)} failed: {e!r}; its boots stay empty")
        return {gi: [] for gi in indices}
    return dict(zip(indices, boots, strict=True))


def run_matches_vec(
    session_cfg: SessionConfig,
    matches: Sequence[VecMatch],
    policy_factory: Callable[[], BatchPolicy],
    *,
    max_frames: int,
    max_parallel: int | None,
    base_slippi_port: int = 51441,
    start_retries: int = DEFAULT_START_RETRIES,
    process_telemetry: ProcessVecTelemetry | None = None,
    process_cohorts: int = 1,
) -> list[list[Trajectory]]:
    """Run ``matches`` (boots) concurrently in waves of up to ``max_parallel``
    Sessions, each frame batched through a single ``BatchPolicy`` call (see
    ``drive_vec``). With instant-restart each boot plays many matches; the returned
    inner list is that boot's matches.

    Each wave's Sessions get distinct slippi_ports (``base_slippi_port + offset``)
    and, when ``session_cfg.replay_dir`` is set, a per-boot replay subdir so their
    .slps don't collide. ``policy_factory`` builds a fresh policy per wave — per-slot
    rolling state must not leak across waves, and ``Slot.match`` indices restart at 0
    each wave. Returns one list per boot, aligned to ``matches``; empty where that
    Session produced no match after all retries.

    ``process_cohorts`` partitions spawned Session workers into independently
    dispatched inference groups. Its default of one preserves all-slot lockstep.

    libmelee's stage-select cursor navigation flakily fails to settle under
    concurrent FFW load (frame-delivery jitter starves its bang-bang controller),
    so a boot's first match intermittently never reaches IN_GAME and ``start_match``
    trips its wall-clock cap. ``start_retries`` re-drives the still-empty boots of a
    wave on fresh Sessions (new Dolphin + slippi_port) to absorb that flake; a wholly-
    dead boot still ends up empty and is logged. (Instant-restart navigates the menu
    only once per boot, so this flake now scales with boots, not matches.)
    """
    if not matches:
        return []
    if isinstance(process_cohorts, bool) or process_cohorts < 1:
        raise ValueError(f"process_cohorts must be >= 1, got {process_cohorts}")
    max_parallel = resolve_parallelism(len(matches), max_parallel)
    base_replay = Path(session_cfg.replay_dir) if session_cfg.replay_dir is not None else None
    out: list[list[Trajectory]] = [[] for _ in matches]
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
                process_telemetry=process_telemetry,
                process_cohorts=process_cohorts,
            )
            for gi, boot in results.items():
                if boot:
                    out[gi] = boot
            pending = [gi for gi in pending if not out[gi]]
            if not pending:
                break
            if attempt < start_retries:
                logger.warning(
                    f"run_matches_vec: {len(pending)} boot(s) failed to reach IN_GAME; "
                    f"retrying on fresh Sessions (attempt {attempt + 2}/{start_retries + 1})"
                )
    return out
