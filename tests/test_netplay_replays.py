import hashlib
from datetime import UTC
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError

from hal.netplay_service.replays import REPLAY_LIFECYCLE_ID
from hal.netplay_service.replays import REPLAY_PREFIX
from hal.netplay_service.replays import ReplayMetadata
from hal.netplay_service.replays import ensure_replay_lifecycle
from hal.netplay_service.replays import replay_object_key
from hal.netplay_service.replays import soak_replay_directory
from hal.netplay_service.replays import upload_and_delete


def _metadata() -> ReplayMetadata:
    return ReplayMetadata(
        reservation_id="reservation-1",
        player_code="CRYO#610",
        game_number=2,
        actual_stage="BATTLEFIELD",
        result="win",
        policy_sha256="a" * 64,
        git_sha="deadbeef",
        started_at=datetime(2026, 9, 10, 23, 30, tzinfo=UTC),
        ended_at=datetime(2026, 9, 10, 23, 34, tzinfo=UTC),
    )


class FakeR2:
    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, dict[str, str], str]] = {}
        self.lifecycle: list[dict] | None = None
        self.exceptions = SimpleNamespace(ClientError=ClientError)

    def put_object(self, *, Key: str, Body, Metadata: dict[str, str], **_kwargs):
        value = Body.read() if hasattr(Body, "read") else bytes(Body)
        etag = hashlib.md5(value).hexdigest()  # noqa: S324 - test-only ETag
        self.objects[Key] = (value, Metadata, etag)
        return {"ETag": f'"{etag}"'}

    def head_object(self, *, Key: str, **_kwargs):
        value, metadata, etag = self.objects[Key]
        return {"ContentLength": len(value), "Metadata": metadata, "ETag": f'"{etag}"'}

    def get_bucket_lifecycle_configuration(self, **_kwargs):
        if self.lifecycle is None:
            raise ClientError({"Error": {"Code": "NoSuchLifecycleConfiguration"}}, "GetBucketLifecycle")
        return {"Rules": self.lifecycle}

    def put_bucket_lifecycle_configuration(self, *, LifecycleConfiguration, **_kwargs):
        self.lifecycle = LifecycleConfiguration["Rules"]


def test_replay_key_indexes_exact_player_under_utc_date() -> None:
    assert replay_object_key(_metadata()) == ("netplay/v1/replays/2026/09/10/CRYO#610/reservation-1/game-02.slp")


def test_upload_validates_r2_and_deletes_local_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_BUCKET", "hal")
    source = tmp_path / "game.slp"
    source.write_bytes(b"valid replay bytes")
    remote = FakeR2()

    uploaded = upload_and_delete(source, _metadata(), client=remote)

    assert not source.exists()
    assert uploaded.key in remote.objects
    assert uploaded.metadata_key in remote.objects
    assert remote.objects[uploaded.key][1]["sha256"] == hashlib.sha256(b"valid replay bytes").hexdigest()


def test_failed_validation_keeps_local_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_BUCKET", "hal")
    source = tmp_path / "game.slp"
    source.write_bytes(b"replay")
    remote = FakeR2()

    def wrong_head(**_kwargs):
        return {"ContentLength": 0, "Metadata": {}, "ETag": '"wrong"'}

    remote.head_object = wrong_head  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="size differs"):
        upload_and_delete(source, _metadata(), client=remote)
    assert source.exists()


def test_lifecycle_preserves_unrelated_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_BUCKET", "hal")
    remote = FakeR2()
    remote.lifecycle = [{"ID": "keep", "Status": "Enabled", "Filter": {"Prefix": "runs/"}}]
    ensure_replay_lifecycle(client=remote)

    assert remote.lifecycle is not None
    assert remote.lifecycle[0]["ID"] == "keep"
    replay_rule = next(rule for rule in remote.lifecycle if rule["ID"] == REPLAY_LIFECYCLE_ID)
    assert replay_rule["Filter"]["Prefix"] == REPLAY_PREFIX
    assert replay_rule["Expiration"]["Days"] == 30


def test_soak_replays_are_always_removed() -> None:
    with soak_replay_directory() as directory:
        replay = directory / "soak.slp"
        replay.write_bytes(b"temporary")
    assert not directory.exists()
