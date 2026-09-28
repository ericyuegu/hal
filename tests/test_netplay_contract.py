from pathlib import Path

import pytest

from hal.netplay_service.domain import MatchChoices
from hal.netplay_service.queue import QueueStore
from hal.netplay_service.queue_client import QueueEndpoint
from hal.netplay_service.queue_client import RemoteQueue
from hal.netplay_service.queue_contract import InvalidTransitionError
from hal.netplay_service.queue_contract import RunnerQueue


def _playing(tmp_path: Path) -> tuple[QueueStore, str]:
    store = QueueStore(tmp_path / "queue.sqlite3")
    created = store.create_job("CRYO#610", MatchChoices("FOX", "IBDW#0", 2))
    store.claim_next("slot-0")
    store.mark_connecting(created.job.id, "slot-0", "HAL#1")
    store.mark_playing(created.job.id, "slot-0")
    return store, created.job.id


def test_queue_store_is_a_runner_queue(tmp_path: Path) -> None:
    store: RunnerQueue = QueueStore(tmp_path / "queue.sqlite3")
    assert store.claim_next("slot-0") is None


def test_finish_game_rejects_a_game_number_out_of_order(tmp_path: Path) -> None:
    store, job_id = _playing(tmp_path)
    with pytest.raises(InvalidTransitionError, match="game_number 2 does not follow game 0"):
        store.finish_game(job_id, "slot-0", game_number=2, actual_stage="BATTLEFIELD", result="win")
    store.finish_game(job_id, "slot-0", game_number=1, actual_stage="BATTLEFIELD", result="win")
    store.record_replay(job_id, "slot-0", 1, key="replays/a.slp", sha256="a" * 64, size=1, etag="e")


def test_claim_lease_defaults_to_twenty_seconds(tmp_path: Path) -> None:
    now = [1000.0]
    store = QueueStore(tmp_path / "queue.sqlite3", now=lambda: now[0])
    created = store.create_job("CRYO#610", MatchChoices("FOX", "IBDW#0", 2))
    store.claim_next("slot-0")
    now[0] += 19.0
    assert store.reap_expired() == 0
    now[0] += 2.0
    assert store.reap_expired() == 1
    assert created.job.id


def test_remote_queue_is_a_runner_queue() -> None:
    remote = RemoteQueue(QueueEndpoint("http://127.0.0.1:8787", "t"), "sess")
    queue: RunnerQueue = remote
    assert queue is remote
    remote.close()
