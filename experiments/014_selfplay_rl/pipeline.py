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

A collector exception is re-raised on the main thread (fail loud): the collector
never dies silently, and iterations always run exactly once, in order.
"""

import queue
import threading
from collections.abc import Callable

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
    collector_error: list[BaseException] = []

    def produce() -> None:
        try:
            for _ in range(iterations):
                handoff.put(collect())
        except BaseException as exc:  # propagate to the learner thread; never die silently
            collector_error.append(exc)
            handoff.put(_SENTINEL)

    collector = threading.Thread(target=produce, name="rl-collector")
    collector.start()
    try:
        for _ in range(iterations):
            payload = handoff.get()
            if payload is _SENTINEL:
                break
            learn(payload)  # type: ignore[arg-type]
    finally:
        collector.join()

    if collector_error:
        raise collector_error[0]
