import hashlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from peppi_py.game import EndMethod

from hal import r2
from hal.eval import ranked_replays
from hal.netplay_service.replays import UploadedReplay
from hal.netplay_service.replays import replay_object_key


@pytest.fixture
def completed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "ranked" / "run-1"
    replay = root / "replays" / "game.slp"
    replay.parent.mkdir(parents=True)
    replay.write_bytes(b"complete replay")
    row = {
        "schema_version": 1,
        "replay": "replays/game.slp",
        "replay_sha256": hashlib.sha256(replay.read_bytes()).hexdigest(),
        "end_method": "GAME",
        "ego_port": 2,
        "stage": 25,
        "at": 1790797484.8,
    }
    record = root / "game-0001.json"
    record.write_text(json.dumps(row))
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bundle_sha256": "a" * 64,
                "source_revision": "d4822adb",
            }
        )
    )
    match = SimpleNamespace(id="mode.ranked-set-1", game=1, tiebreaker=0)
    game = SimpleNamespace(
        start=SimpleNamespace(
            match=match, stage=32, players=[SimpleNamespace(port=1, netplay=SimpleNamespace(code="HAL＃647"))]
        ),
        end=SimpleNamespace(method=EndMethod.GAME, players=[SimpleNamespace(port=1, placement=0)]),
        metadata={"startAt": "2026-09-30T19:42:41Z"},
    )
    monkeypatch.setattr(ranked_replays.peppi_py, "read_slippi", Mock(return_value=game))
    return record, replay, row, game


def test_upload_is_stable_verified_and_keeps_local_replay(completed) -> None:
    record, replay, row, _game = completed

    def send(path, metadata):
        assert path == replay
        assert metadata.player_code == "HAL#647"
        assert metadata.result == "win"
        assert metadata.game_number == 1
        assert metadata.actual_stage == "FINAL_DESTINATION"
        assert metadata.policy_sha256 == "a" * 64
        key = replay_object_key(metadata)
        assert key.startswith("netplay/v1/replays/2026/09/30/HAL#647/ranked-")
        return UploadedReplay(
            key, key.removesuffix(".slp") + ".json", row["replay_sha256"], replay.stat().st_size, "etag"
        )

    upload = Mock(side_effect=send)
    assert ranked_replays.upload_game(record, bucket="hal", upload=upload)
    assert not ranked_replays.upload_game(record, bucket="hal", upload=upload)
    upload.assert_called_once()
    assert replay.exists()
    receipt = json.loads((record.parent / "uploads" / record.name).read_text())
    assert receipt["sha256"] == row["replay_sha256"]


@pytest.mark.parametrize("change", ["hash", "schema", "path", "incomplete", "unranked", "end", "stage"])
def test_invalid_record_never_uploads(completed, change: str) -> None:
    record, replay, row, game = completed
    if change == "hash":
        replay.write_bytes(b"changed")
    elif change == "schema":
        row["schema_version"] = 2
    elif change == "path":
        row["replay"] = "../elsewhere.slp"
    elif change == "incomplete":
        game.end = None
    elif change == "unranked":
        game.start.match.id = "mode.direct-set"
    elif change == "end":
        row["end_method"] = "NO_CONTEST"
    else:
        row["stage"] = 24
    record.write_text(json.dumps(row))
    upload = Mock()
    with pytest.raises(ValueError):
        ranked_replays.upload_game(record, bucket="hal", upload=upload)
    upload.assert_not_called()
    assert not (record.parent / "uploads").exists()


def test_tiebreaker_has_its_own_key(completed) -> None:
    record, replay, row, game = completed
    keys = []

    def send(_path, metadata):
        key = replay_object_key(metadata)
        keys.append(key)
        return UploadedReplay(key, key + ".json", row["replay_sha256"], replay.stat().st_size, "etag")

    ranked_replays.upload_game(record, bucket="hal", upload=send)
    (record.parent / "uploads" / record.name).unlink()
    game.start.match.tiebreaker = 1
    ranked_replays.upload_game(record, bucket="hal", upload=send)
    assert keys[0] != keys[1]


def test_changed_receipt_is_rejected_without_overwriting_remote(completed) -> None:
    record, _replay, row, _game = completed
    receipts = record.parent / "uploads"
    receipts.mkdir()
    (receipts / record.name).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bucket": "hal",
                "record_sha256": "wrong",
                "sha256": row["replay_sha256"],
            }
        )
    )
    upload = Mock()
    with pytest.raises(ValueError, match="receipt differs"):
        ranked_replays.upload_game(record, bucket="hal", upload=upload)
    upload.assert_not_called()


def test_failed_upload_retries_from_disk_and_skips_unfinished_replays(
    completed, monkeypatch: pytest.MonkeyPatch
) -> None:
    record, replay, row, _game = completed
    unfinished = replay.with_name("still-playing.slp")
    unfinished.write_bytes(b"in progress")
    remote = Mock()
    monkeypatch.setattr(r2, "client", lambda **_kwargs: remote)
    monkeypatch.setattr(r2, "bucket", lambda: "hal")
    uploaded = UploadedReplay("key", "metadata", row["replay_sha256"], replay.stat().st_size, "etag")
    send = Mock(side_effect=[RuntimeError("network unavailable"), uploaded])
    monkeypatch.setattr(ranked_replays, "upload_replay", send)

    assert ranked_replays.upload_pending(record.parent.parent) == 0
    assert not (record.parent / "uploads" / record.name).exists()
    assert "network unavailable" in (record.parent / "upload-error.json").read_text()
    assert ranked_replays.upload_pending(record.parent.parent) == 1
    assert ranked_replays.upload_pending(record.parent.parent) == 0
    assert send.call_count == 2
    assert all(call.args[0] == replay for call in send.call_args_list)
    assert remote.close.call_count == 3


def test_worker_survives_failed_pass_and_wakes_for_new_game(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    started, recovered = threading.Event(), threading.Event()
    attempts = []

    def upload(_root, **_kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            started.set()
            raise RuntimeError("R2 unavailable")
        recovered.set()
        return 0

    monkeypatch.setattr(r2, "missing_credentials", lambda: [])
    monkeypatch.setattr(r2, "bucket", lambda: "hal")
    monkeypatch.setattr(ranked_replays, "upload_pending", upload)
    with ranked_replays.RankedReplayUploads(tmp_path) as worker:
        assert started.wait(2)
        worker.notify()
        assert recovered.wait(2)
    assert not worker._thread.is_alive()


def test_r2_upload_timeout_has_one_bounded_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("AWS_ENDPOINT_URL", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.setenv(name, "test-value")
    factory = Mock()
    monkeypatch.setattr(r2.boto3, "client", factory)
    r2.client(timeout_seconds=5)
    config = factory.call_args.kwargs["config"]
    assert config.connect_timeout == config.read_timeout == 5
    assert config.retries == {"total_max_attempts": 1}
    r2.client()
    config = factory.call_args.kwargs["config"]
    assert config.connect_timeout == config.read_timeout == 60
    assert config.retries is None
