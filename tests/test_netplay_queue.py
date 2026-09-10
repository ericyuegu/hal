from pathlib import Path

import pytest

from hal.netplay_service.domain import JobStatus
from hal.netplay_service.domain import MatchChoices
from hal.netplay_service.queue import ActiveJobError
from hal.netplay_service.queue import InvalidTransitionError
from hal.netplay_service.queue import InviteError
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


def test_invite_binds_to_exact_player_and_one_active_job(tmp_path: Path) -> None:
    clock = Clock()
    store = _store(tmp_path, clock)
    invite = store.create_invite("tester")
    credentials = store.create_job(invite, "CRYO#610", _choices())

    assert credentials.job.status is JobStatus.QUEUED
    assert credentials.job.queue_position == 1
    with pytest.raises(ActiveJobError):
        store.create_job(invite, "CRYO#610", _choices())
    with pytest.raises(InviteError, match="different player"):
        store.create_job(invite, "OTHER#1", _choices())


def test_fifo_queue_and_retry_at_front(tmp_path: Path) -> None:
    clock = Clock()
    store = _store(tmp_path, clock)
    first_invite = store.create_invite("first")
    second_invite = store.create_invite("second")
    first = store.create_job(first_invite, "FIRST#1", _choices())
    second = store.create_job(second_invite, "SECOND#2", _choices(3))

    claimed = store.claim_next("slot-0")
    assert claimed is not None and claimed.id == first.job.id
    assert store.fail(claimed.id, "slot-0", "dolphin_crash", retryable=True) is JobStatus.QUEUED
    retried = store.claim_next("slot-1")
    assert retried is not None and retried.id == first.job.id and retried.attempt == 2
    assert store.fail(retried.id, "slot-1", "dolphin_crash", retryable=True) is JobStatus.FAILED
    assert store.claim_next("slot-0").id == second.job.id  # type: ignore[union-attr]


def test_no_show_expires_and_releases_player(tmp_path: Path) -> None:
    clock = Clock()
    store = _store(tmp_path, clock)
    invite = store.create_invite("tester")
    credentials = store.create_job(invite, "CRYO#610", _choices())
    job = store.claim_next("slot-0")
    assert job is not None
    store.mark_connecting(job.id, "slot-0", "HAL#1")

    clock.advance(61)
    assert store.reap_expired() == 1
    assert store.get_job(job.id, credentials.token).status is JobStatus.NO_SHOW
    assert store.create_job(invite, "CRYO#610", _choices()).job.status is JobStatus.QUEUED


def test_rematch_updates_choices_but_not_delay(tmp_path: Path) -> None:
    clock = Clock()
    store = _store(tmp_path, clock)
    invite = store.create_invite("tester")
    credentials = store.create_job(invite, "CRYO#610", _choices(3))
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
    invite = store.create_invite("tester")
    credentials = store.create_job(invite, "CRYO#610", _choices())
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
    invite = store.create_invite("tester")
    credentials = store.create_job(invite, "CRYO#610", _choices())
    job = store.claim_next("slot-0")
    assert job is not None
    store.mark_connecting(job.id, "slot-0", "HAL#1")
    store.mark_playing(job.id, "slot-0")

    assert store.cancel(job.id, credentials.token).cancel_after_game
    assert store.finish_game(job.id, "slot-0", actual_stage="BATTLEFIELD", result="win") is JobStatus.COMPLETE
