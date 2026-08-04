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

import math
import time
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


def _frame_id(frame: Mapping, previous: int) -> int:
    """The canonical frame counter of one stepped frame, as an int.

    A non-finite id — seen in production from a torn libmelee frame — is unusable
    three times over: it never compares as a match boundary, it poisons every later
    comparison against it, and it raises ``cannot convert float NaN to integer`` when
    ``Trajectory.from_capture`` writes it into an int32 column. That last one happens
    OUTSIDE the per-boot guard below and would abort the whole wave, so the frame is
    rejected here, where the cost is one boot.
    """
    fid = frame.get("id", previous + 1)
    if isinstance(fid, int):
        return fid
    if not math.isfinite(fid):
        raise ValueError(f"canonical frame id is {fid!r}; the emulator produced a torn frame")
    return int(fid)


def drive_vec(
    sessions: Sequence[Session],
    matches: Sequence[VecMatch],
    policy: BatchPolicy,
    *,
    max_frames: int,
    instant_restart: bool = False,
    progress_every: int = 600,
) -> list[list[Trajectory]]:
    """Drive ``matches`` concurrently; ``sessions[i]`` runs ``matches[i]``.

    Returns one list of ``Trajectory`` **per boot**, aligned to ``matches`` — the
    matches that boot played, in order. The list is empty if that Session failed
    to start or crashed before completing any match (a single bad emulator never
    aborts the others, matching ``run_match``'s log-and-continue contract).

    With ``instant_restart`` (the eval default via the Gecko "Instant Match" code),
    one Dolphin boot plays many matches back-to-back: when a match ends, Dolphin
    restarts directly into a new one on a random legal stage — skipping the flaky
    stage-select menu the boot navigated only once. The restart is seamless
    (``in_game`` never drops), so the match boundary is the canonical frame ``id``
    resetting to the pre-game countdown (``id`` decreasing); ``drive_vec`` segments
    the stream into one ``Trajectory`` per match there, until ``max_frames`` is spent
    (the budget spans every match). Without it, a boot is one match (the historical
    behavior, ended by ``in_game`` going False), returned as a singleton list.

    Each frame's ``_matchup`` carries the **live** stage (``Session.step`` injects it;
    instant-restart randomizes it per match) plus the boot's fixed per-port characters,
    in the libmelee id spaces the model trained on. ``progress_every`` logs a heartbeat;
    0 disables it.
    """
    if len(sessions) != len(matches):
        raise ValueError(f"got {len(sessions)} sessions for {len(matches)} matches")
    n = len(matches)

    # Per-boot rollout state. ``seg`` accumulates the current match's frames; ``trajs``
    # collects completed matches. ``last_id`` tracks the frame counter to spot the
    # instant-restart boundary (a reset to the pre-game countdown).
    seg: list[list[dict]] = [[] for _ in range(n)]
    trajs: list[list[Trajectory]] = [[] for _ in range(n)]
    started = [False] * n
    done = [True] * n
    last_id = [0] * n
    ports_of = [tuple(p.port for p in m.matchup.players) for m in matches]
    # Every slot this call can ever hand the policy, built once: the step loop runs at
    # 60 Hz across N boots, so nothing whose value is fixed for the run belongs in it.
    slots_of = [tuple(Slot(i, p) for p in m.model_ports) for i, m in enumerate(matches)]
    # Per-match matchup metadata injected into each obs frame under ``_matchup``
    # (flatten_canonical_frame emits the columns only when present). Characters are
    # fixed for the boot; ``stage`` is refreshed each frame from the live frame.
    match_meta = [
        {
            "stage": int(m.matchup.stage.value),
            "character": {pl.port: int(pl.character.value) for pl in m.matchup.players},
        }
        for m in matches
    ]

    def _close_segment(i: int) -> None:
        if seg[i]:
            trajs[i].append(Trajectory.from_capture(seg[i], ports_of[i]))
        seg[i] = []

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
                f0 = fut.result()
                last_id[i] = _frame_id(f0, -1)
                seg[i].append(f0)
                started[i] = True
                done[i] = False
            except Exception as e:
                logger.warning(f"drive_vec: boot {i} start failed: {e!r}")

        n_live0 = n - sum(done)
        logger.info(f"drive_vec: stepping {n_live0}/{n} boots up to {max_frames} frames")
        t0 = time.monotonic()
        for t in range(max_frames - 1):
            live = [i for i in range(n) if not done[i]]
            if not live:
                break
            if progress_every and t > 0 and t % progress_every == 0:
                elapsed = time.monotonic() - t0
                logger.info(f"drive_vec: frame {t}/{max_frames} | live {len(live)}/{n} | {t / elapsed:.0f} steps/s")
            # Refresh each live slot's injected stage from the frame about to be shown
            # (instant-restart changes it); characters are fixed for the boot. One
            # observation per BOOT, shared by that boot's slots: the ports of a match see
            # the same frame, and the policy tells them apart by slot, not by dict.
            obs: dict[Slot, dict] = {}
            for i in live:
                match_meta[i]["stage"] = seg[i][-1].get("stage", match_meta[i]["stage"])
                view = {**seg[i][-1], "_matchup": match_meta[i]}
                for slot in slots_of[i]:
                    obs[slot] = view
            inputs = policy(t, obs) if obs else {}
            step_futs = {
                i: pool.submit(sessions[i].step, {slot.port: inputs[slot] for slot in slots_of[i]}) for i in live
            }
            for i, fut in step_futs.items():
                try:
                    frame, in_game = fut.result()
                    fid = _frame_id(frame, last_id[i])
                except Exception as e:
                    logger.warning(f"drive_vec: boot {i} step crashed: {e!r}")
                    seg[i] = []  # drop the partial match; keep matches already completed
                    done[i] = True
                    continue
                if instant_restart and fid < last_id[i]:
                    # Frame counter reset → Dolphin instant-restarted into a new match.
                    # Close the just-finished match and open a fresh segment.
                    _close_segment(i)
                    logger.info(f"drive_vec: boot {i} match {len(trajs[i])} ended at frame {t}")
                    seg[i] = [frame]
                else:
                    seg[i].append(frame)
                last_id[i] = fid
                if not in_game:
                    # No instant-restart (or a real drop to menu): the match ended.
                    _close_segment(i)
                    done[i] = True

        # Flush the match still in progress at the budget so a cut-off rollout (and,
        # under instant-restart, the final match) is still captured.
        for i in range(n):
            if started[i]:
                _close_segment(i)
    return trajs
