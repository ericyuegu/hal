import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from hal.netplay_service.domain import JobStatus
from hal.netplay_service.domain import MatchChoices
from hal.netplay_service.queue import ActiveJobError
from hal.netplay_service.queue import InvalidTransitionError
from hal.netplay_service.queue import QueueStore


class Clock:
    def __init__(self) -> None:
        self.value = 1_000.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _choices(delay: int = 2) -> MatchChoices:
    return MatchChoices("FOX", "IBDW#0", delay)


def _store(tmp_path: Path, clock: Clock) -> QueueStore:
    return QueueStore(tmp_path / "queue.sqlite3", now=clock)


def test_schema_one_database_drops_obsolete_invites(tmp_path: Path) -> None:
    path = tmp_path / "queue.sqlite3"
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("CREATE TABLE invites(digest BLOB PRIMARY KEY)")
        connection.execute("PRAGMA user_version = 1")
        connection.commit()

    QueueStore(path)
    with closing(sqlite3.connect(path)) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        invite_table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'invites'"
        ).fetchone()
    assert version == 2
    assert invite_table is None


def test_queue_operations_close_every_sqlite_connection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[sqlite3.Connection] = []
    original_connect = QueueStore._connect

    def tracked_connect(store: QueueStore) -> sqlite3.Connection:
        connection = original_connect(store)
        opened.append(connection)
        return connection

    monkeypatch.setattr(QueueStore, "_connect", tracked_connect)
    store = _store(tmp_path, Clock())
    credentials = store.create_job("CRYO#610", _choices())
    claimed = store.claim_next("slot-0")
    assert claimed is not None
    for _ in range(20):
        store.get_job(credentials.job.id, credentials.token)
        store.get_worker_job(claimed.id, "slot-0")
        store.queue_depth()
        store.active_count()

    assert opened
    for connection in opened:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")


def test_only_one_active_job_is_allowed_per_player(tmp_path: Path) -> None:
    clock = Clock()
    store = _store(tmp_path, clock)
    credentials = store.create_job("CRYO#610", _choices())

    assert credentials.job.status is JobStatus.QUEUED
    assert credentials.job.queue_position == 1
    with pytest.raises(ActiveJobError):
        store.create_job("CRYO#610", _choices())
    assert store.create_job("OTHER#1", _choices()).job.queue_position == 2


def test_fifo_queue_and_retry_at_front(tmp_path: Path) -> None:
    clock = Clock()
    store = _store(tmp_path, clock)
    first = store.create_job("FIRST#1", _choices())
    second = store.create_job("SECOND#2", _choices(3))

    claimed = store.claim_next("slot-0")
    assert claimed is not None and claimed.id == first.job.id
    assert store.fail(claimed.id, "slot-0", "dolphin_crash", retryable=True) is JobStatus.QUEUED
    retried = store.claim_next("slot-1")
    assert retried is not None and retried.id == first.job.id and retried.attempt == 2
    assert store.fail(retried.id, "slot-1", "dolphin_crash", retryable=True) is JobStatus.FAILED
    assert store.claim_next("slot-0").id == second.job.id  # type: ignore[union-attr]


def test_capacity_counts_claimed_jobs_separately_from_the_queue(tmp_path: Path) -> None:
    store = _store(tmp_path, Clock())
    store.create_job("FIRST#1", _choices())
    store.create_job("SECOND#2", _choices())

    assert (store.active_count(), store.queue_depth()) == (0, 2)
    assert store.claim_next("slot-0") is not None
    assert (store.active_count(), store.queue_depth()) == (1, 1)


def test_no_show_expires_and_releases_player(tmp_path: Path) -> None:
    clock = Clock()
    store = _store(tmp_path, clock)
    credentials = store.create_job("CRYO#610", _choices())
    job = store.claim_next("slot-0")
    assert job is not None
    store.mark_connecting(job.id, "slot-0", "HAL#1")

    clock.advance(61)
    assert store.reap_expired() == 1
    assert store.get_job(job.id, credentials.token).status is JobStatus.NO_SHOW
    assert store.create_job("CRYO#610", _choices()).job.status is JobStatus.QUEUED


def test_worker_can_mark_no_show_and_release_player(tmp_path: Path) -> None:
    clock = Clock()
    store = _store(tmp_path, clock)
    credentials = store.create_job("CRYO#610", _choices())
    job = store.claim_next("slot-0")
    assert job is not None
    store.mark_connecting(job.id, "slot-0", "HAL#1")

    store.mark_no_show(job.id, "slot-0")
    assert store.get_job(job.id, credentials.token).status is JobStatus.NO_SHOW
    assert store.create_job("CRYO#610", _choices()).job.status is JobStatus.QUEUED


def test_rematch_updates_choices_but_not_delay(tmp_path: Path) -> None:
    clock = Clock()
    store = _store(tmp_path, clock)
    credentials = store.create_job("CRYO#610", _choices(3))
    job = store.claim_next("slot-0")
    assert job is not None
    store.mark_connecting(job.id, "slot-0", "HAL#1")
    store.mark_playing(job.id, "slot-0")
    assert store.finish_game(job.id, "slot-0", actual_stage="BATTLEFIELD", result="win") is JobStatus.REMATCH_WAIT

    rematch = store.request_rematch(
        job.id,
        credentials.token,
        character="MARTH",
        imitation="ZAIN#0",
        stage="FINAL_DESTINATION",
    )
    assert rematch.status is JobStatus.REMATCH_READY
    assert rematch.choices == MatchChoices("MARTH", "ZAIN#0", 3, "FINAL_DESTINATION")
    store.mark_playing(job.id, "slot-0")


def test_rematch_timeout_completes_reservation(tmp_path: Path) -> None:
    clock = Clock()
    store = _store(tmp_path, clock)
    credentials = store.create_job("CRYO#610", _choices())
    job = store.claim_next("slot-0")
    assert job is not None
    store.mark_connecting(job.id, "slot-0", "HAL#1")
    store.mark_playing(job.id, "slot-0")
    store.finish_game(job.id, "slot-0", actual_stage="BATTLEFIELD", result="loss")

    clock.advance(61)
    assert store.reap_expired() == 1
    assert store.get_job(job.id, credentials.token).status is JobStatus.COMPLETE
    with pytest.raises(InvalidTransitionError):
        store.request_rematch(
            job.id,
            credentials.token,
            character="FOX",
            imitation="IBDW#0",
            stage="BATTLEFIELD",
        )


def test_cancel_during_play_stops_after_game(tmp_path: Path) -> None:
    clock = Clock()
    store = _store(tmp_path, clock)
    credentials = store.create_job("CRYO#610", _choices())
    job = store.claim_next("slot-0")
    assert job is not None
    store.mark_connecting(job.id, "slot-0", "HAL#1")
    store.mark_playing(job.id, "slot-0")

    assert store.cancel(job.id, credentials.token).cancel_after_game
    assert store.finish_game(job.id, "slot-0", actual_stage="BATTLEFIELD", result="win") is JobStatus.COMPLETE


def test_replay_recording_is_idempotent_for_recovery(tmp_path: Path) -> None:
    clock = Clock()
    store = _store(tmp_path, clock)
    job = store.create_job("CRYO#610", _choices()).job
    claimed = store.claim_next("slot-0")
    assert claimed is not None
    store.mark_connecting(job.id, "slot-0", "HAL#1")
    store.mark_playing(job.id, "slot-0")
    store.finish_game(job.id, "slot-0", actual_stage="BATTLEFIELD", result="win")

    values = {"key": "replay.slp", "sha256": "a" * 64, "size": 10, "etag": "etag"}
    store.record_replay(job.id, 1, **values)
    store.record_replay(job.id, 1, **values)
    with pytest.raises(InvalidTransitionError, match="different replay"):
        store.record_replay(job.id, 1, key="other.slp", sha256="b" * 64, size=11, etag="other")
