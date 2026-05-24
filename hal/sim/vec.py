"""Drive N matches concurrently with a single batched policy call per frame.

``loop.drive`` runs one match, calling a per-port ``ControllerSource`` each
frame; the model forward is buried inside that callback, so it can't batch
across matches. ``drive_vec`` inverts that: it owns N Sessions, collects every
live model-driven port's gamestate into one observation map, and hands the
whole map to a single ``BatchPolicy`` call — letting the implementation run one
batched forward across all matches. The returned inputs are scattered back and
all Sessions step concurrently on a thread pool.

Why threads work: ``Session.step`` blocks inside ``console.step`` on a socket
recv (libmelee releases the GIL there), so per-Session worker threads overlap
and the emulators advance in parallel. The batched forward stays on the main
thread (single GPU, shared model state).

Torch-free, like the rest of ``hal/sim``: the model lives behind ``BatchPolicy``
in the experiment.
"""

from collections.abc import Mapping
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Protocol
from typing import runtime_checkable

from loguru import logger

from hal.sim.inputs import ControllerInputs
from hal.sim.session import Matchup
from hal.sim.session import Session
from hal.sim.trajectory import Trajectory


@dataclass(frozen=True, slots=True)
class Slot:
    """One model-driven port within one match of a vectorized rollout."""

    match: int  # index into the ``matches`` passed to ``drive_vec``
    port: int  # libmelee port, 1..4


@dataclass(frozen=True, slots=True)
class VecMatch:
    """One match in a vectorized rollout: a ``Matchup`` plus which of its ports
    the ``BatchPolicy`` drives. Ports absent from ``model_ports`` are internal
    (CPU/human) and get no punched input, exactly as a ``None``-returning
    ``ControllerSource`` is skipped in ``loop.drive``."""

    matchup: Matchup
    model_ports: tuple[int, ...]


@runtime_checkable
class BatchPolicy(Protocol):
    """Map every live model slot's current gamestate to its next inputs in ONE
    call, so the implementation can batch its forward pass.

    Called once per frame with the observations of all currently-live model
    slots (matches that have ended are dropped). Must return one
    ``ControllerInputs`` per slot it was given — internal ports are never passed
    here, so ``None`` is not a valid response."""

    def __call__(self, frame_index: int, obs: Mapping[Slot, dict]) -> Mapping[Slot, ControllerInputs]: ...


def drive_vec(
    sessions: Sequence[Session],
    matches: Sequence[VecMatch],
    policy: BatchPolicy,
    *,
    max_frames: int,
) -> list[Trajectory | None]:
    """Drive ``matches`` concurrently to completion; ``sessions[i]`` runs
    ``matches[i]``.

    Returns one ``Trajectory`` per match, aligned to ``matches``. An entry is
    ``None`` if that Session failed to start or crashed mid-rollout — a single
    bad emulator never aborts the others (matching ``run_match``'s
    log-and-continue contract).
    """
    if len(sessions) != len(matches):
        raise ValueError(f"got {len(sessions)} sessions for {len(matches)} matches")
    n = len(matches)

    captured: list[list[dict]] = [[] for _ in range(n)]
    started = [False] * n
    crashed = [False] * n

    with ExitStack() as stack:
        for s in sessions:
            stack.enter_context(s)
        pool = stack.enter_context(ThreadPoolExecutor(max_workers=max(1, n)))

        # Concurrent start: each start_match boots Dolphin + navigates menus on
        # its own thread, parking at the first in-game frame. blocking_input
        # keeps a parked instance waiting for input (no free-run), so the
        # slowest one to reach IN_GAME sets the shared t=0 for the lockstep.
        start_futs = {
            i: pool.submit(s.start_match, m.matchup) for i, (s, m) in enumerate(zip(sessions, matches, strict=True))
        }
        for i, fut in start_futs.items():
            try:
                captured[i].append(fut.result())
                started[i] = True
            except Exception as e:
                logger.warning(f"drive_vec: match {i} start failed: {e!r}")
                crashed[i] = True

        done = [not started[i] for i in range(n)]
        for t in range(max_frames - 1):
            live = [i for i in range(n) if not done[i]]
            if not live:
                break
            obs = {Slot(i, p): captured[i][-1] for i in live for p in matches[i].model_ports}
            inputs = policy(t, obs)
            step_futs = {
                i: pool.submit(sessions[i].step, {p: inputs[Slot(i, p)] for p in matches[i].model_ports}) for i in live
            }
            for i, fut in step_futs.items():
                try:
                    frame, in_game = fut.result()
                except Exception as e:
                    logger.warning(f"drive_vec: match {i} step crashed: {e!r}")
                    crashed[i] = True
                    done[i] = True
                    continue
                captured[i].append(frame)
                if not in_game:
                    done[i] = True

    trajectories: list[Trajectory | None] = []
    for i, m in enumerate(matches):
        if started[i] and not crashed[i]:
            ports = tuple(p.port for p in m.matchup.players)
            trajectories.append(Trajectory.from_capture(captured[i], ports))
        else:
            trajectories.append(None)
    return trajectories
