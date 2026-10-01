"""One reservation on one runner slot: the reporting link and the Slippi loop."""

import threading
import time
from collections.abc import Callable
from contextlib import suppress

from loguru import logger

from hal.netplay_service.domain import REPORT_INTERVAL_SECONDS
from hal.netplay_service.domain import EndReason
from hal.netplay_service.domain import FinishedGame
from hal.netplay_service.domain import Job
from hal.netplay_service.domain import JobStatus
from hal.netplay_service.domain import Observed
from hal.netplay_service.domain import Phase
from hal.netplay_service.domain import Settings
from hal.netplay_service.domain import WindDown
from hal.netplay_service.queue_client import QueueUnavailableError
from hal.netplay_service.queue_contract import InvalidTransitionError
from hal.netplay_service.queue_contract import RunnerQueue


class ReservationLink:
    """Report snapshots in the background and keep the Worker's latest answer.

    The main loop only reads desired state and writes observations; it never waits
    on the network. A refused report means the Worker no longer gives this slot the
    job, so the link stops and the loop aborts at its next check.
    """

    def __init__(
        self,
        queue: RunnerQueue,
        job: Job,
        worker_id: str,
        *,
        interval_seconds: float = REPORT_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._queue = queue
        self._job_id = job.id
        self._worker_id = worker_id
        self._interval = interval_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._seq = 0
        self._job = job
        self._history: dict[int, Settings] = {job.settings.revision: job.settings}
        self._phase = Phase.BOOTING
        self._deadline: float | None = None
        self._bot_code: str | None = None
        self._locked: int | None = None
        self._games: list[FinishedGame] = []
        self._released = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"reservation-{job.id}", daemon=True)

    def __enter__(self) -> ReservationLink:
        self._send()
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()

    def _snapshot(self) -> Observed:
        with self._lock:
            self._seq += 1
            left = None if self._deadline is None else max(0.0, self._deadline - self._clock())
            return Observed(
                seq=self._seq,
                phase=self._phase,
                phase_seconds_left=left,
                bot_code=self._bot_code,
                seen_revision=self._job.settings.revision,
                locked_revision=self._locked,
                finished_games=tuple(self._games),
            )

    def _send(self) -> None:
        if self._released:
            return
        try:
            job = self._queue.report(self._job_id, self._worker_id, self._snapshot())
        except QueueUnavailableError as error:
            # The lease tolerates a few lost reports; the next tick tries again.
            logger.bind(job=self._job_id, event="report").warning("reservation report failed: {}", error)
            return
        except InvalidTransitionError:
            with self._lock:
                self._released = True
            return
        with self._lock:
            self._job = job
            self._history[job.settings.revision] = job.settings
            if job.status is not JobStatus.ASSIGNED:
                self._released = True

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self._send()
            if self._released:
                return

    def set_phase(self, phase: Phase, deadline: float | None) -> None:
        with self._lock:
            self._phase, self._deadline = phase, deadline

    def set_bot_code(self, code: str) -> None:
        with self._lock:
            self._bot_code = code

    def lock(self, revision: int | None) -> None:
        with self._lock:
            self._locked = revision

    def add_game(self, game: FinishedGame) -> None:
        with self._lock:
            self._games.append(game)

    def settings(self) -> Settings:
        with self._lock:
            return self._job.settings

    def settings_at(self, revision: int) -> Settings:
        with self._lock:
            return self._history[revision]

    def wind_down(self) -> WindDown | None:
        with self._lock:
            return self._job.wind_down

    def lock_requests(self) -> int:
        with self._lock:
            return self._job.lock_requests

    def released(self) -> bool:
        with self._lock:
            return self._released

    def should_abort(self) -> bool:
        with self._lock:
            return self._released or self._job.wind_down is not None

    def end(self, reason: EndReason, *, retryable: bool = False) -> None:
        """Flush the final snapshot so the last game is recorded, then end the job."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()
        self._send()
        if self.released():
            return
        with suppress(InvalidTransitionError):
            self._queue.end(self._job_id, self._worker_id, reason, retryable=retryable)
        with self._lock:
            self._released = True
