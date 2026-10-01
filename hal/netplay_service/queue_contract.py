"""Errors and the queue interface shared by the SQLite store and the remote queue client."""

from typing import Protocol

from hal.netplay_service.domain import EndReason
from hal.netplay_service.domain import Job
from hal.netplay_service.domain import Observed


class QueueError(RuntimeError):
    pass


class InvalidTransitionError(QueueError):
    """The worker does not own the job in a state that allows the operation."""


class SessionEndedError(InvalidTransitionError):
    """The runner's session has ended, so every lease it held is gone."""


class RunnerQueue(Protocol):
    """The queue operations a runner slot performs; lease durations belong to the queue."""

    def claim_next(self, worker_id: str) -> Job | None: ...

    def report(self, job_id: str, worker_id: str, observed: Observed) -> Job: ...

    def end(self, job_id: str, worker_id: str, reason: EndReason, *, retryable: bool) -> Job: ...

    def record_replay(
        self, job_id: str, worker_id: str, game_number: int, *, key: str, sha256: str, size: int, etag: str
    ) -> None: ...

    def get_worker_job(self, job_id: str, worker_id: str) -> Job: ...

    def finish_pairing(self, job_id: str, worker_id: str, attempt: int) -> None: ...
