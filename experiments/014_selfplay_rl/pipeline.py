"""Generic double-buffer collect/learn runner (payload-agnostic).

The learner (main thread) trains on rollout iteration ``k`` while a collector
thread already gathers ``k+1``; a ``queue.Queue(maxsize=1)`` is the sole handoff,
so at most one finished payload waits between the two — the collector cannot run
more than one iteration ahead (bounded policy lag). ``overlap=False`` degrades to
a plain inline ``learn(collect())`` loop (lag 0) sharing the exact same call
sequence, which is what makes the sync and overlap gates comparable.

Snapshot discipline is the CALLER's job, not this module's: ``collect`` closes
over ``ema.copy_to(act_net)`` at its start and ``learn`` calls ``ema.update(...)``,
both guarded by a ``threading.Lock`` the caller shares between them. This runner
stays payload-agnostic (gym buffers now, Melee rollout iterations later) and owns
only the threading handshake, not what a payload is.

Failure on either side is loud and prompt: a collector exception is re-raised on
the main thread, and a learner exception sets a stop event the collector polls
around its (bounded-wait) ``put``, so the thread winds down and the learner's
traceback propagates instead of deadlocking against a full queue. One caveat: a
learner exception still waits out any in-flight ``collect()`` before surfacing —
there is no generic way to interrupt a running collect, so shutdown latency is
bounded by one collection iteration.
"""

import queue
import threading
from collections.abc import Callable
from typing import cast

_POLL_SECONDS = 0.05
_SENTINEL = object()


def run_pipeline[P](
    *,
    collect: Callable[[], P],
    learn: Callable[[P], None],
    iterations: int,
    overlap: bool,
) -> None:
    if iterations < 0:
        raise ValueError(f"iterations must be non-negative, got {iterations}")

    if not overlap:
        for _ in range(iterations):
            learn(collect())
        return

    handoff: queue.Queue[object] = queue.Queue(maxsize=1)
    stop = threading.Event()
    collector_error: list[BaseException] = []

    def produce() -> None:
        try:
            for _ in range(iterations):
                payload = collect()
                while not stop.is_set():
                    try:
                        handoff.put(payload, timeout=_POLL_SECONDS)
                        break
                    except queue.Full:
                        continue
                if stop.is_set():
                    return
        except BaseException as exc:  # re-raised on the main thread; never die silently
            collector_error.append(exc)

    def next_payload(collector: threading.Thread) -> object:
        while True:
            try:
                return handoff.get(timeout=_POLL_SECONDS)
            except queue.Empty:
                if collector.is_alive():
                    continue
                try:  # collector exited between our timeout and its final put — drain once
                    return handoff.get_nowait()
                except queue.Empty:
                    return _SENTINEL  # collector died with an error; end the loop

    collector = threading.Thread(target=produce, name="rl-collector")
    collector.start()
    try:
        for _ in range(iterations):
            payload = next_payload(collector)
            if payload is _SENTINEL:
                break
            learn(cast(P, payload))
    finally:
        stop.set()  # unblock a collector waiting on a full queue so join() returns promptly
        collector.join()

    if collector_error:
        raise collector_error[0]
