"""Errors and the queue interface shared by the SQLite store and the remote queue client."""

from typing import Protocol

from hal.netplay_service.domain import Job
from hal.netplay_service.domain import JobStatus


class QueueError(RuntimeError):
    pass


class InvalidTransitionError(QueueError):
    """The worker does not own the job in a state that allows the operation."""


class SessionEndedError(InvalidTransitionError):
    """The runner's session has ended, so every lease it held is gone."""


class RunnerQueue(Protocol):
    """The queue operations a runner slot performs; lease durations belong to the queue."""

    def claim_next(self, worker_id: str) -> Job | None: ...

    def heartbeat(self, job_id: str, worker_id: str) -> None: ...

    def mark_connecting(self, job_id: str, worker_id: str, connect_code: str) -> None: ...

    def mark_playing(self, job_id: str, worker_id: str) -> None: ...

    def mark_no_show(self, job_id: str, worker_id: str) -> None: ...

    def mark_no_contest(self, job_id: str, worker_id: str) -> None: ...

    def finish_game(
        self, job_id: str, worker_id: str, *, game_number: int, actual_stage: str, result: str
    ) -> JobStatus: ...

    def fail(self, job_id: str, worker_id: str, error_code: str, *, retryable: bool) -> JobStatus: ...

    def forfeit_service_failure(self, job_id: str, worker_id: str) -> None: ...

    def record_replay(
        self, job_id: str, worker_id: str, game_number: int, *, key: str, sha256: str, size: int, etag: str
    ) -> None: ...

    def get_worker_job(self, job_id: str, worker_id: str) -> Job: ...
