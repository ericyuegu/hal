import hashlib
import json
import os
import zipfile
from io import BytesIO
from pathlib import Path

import pytest

import hal.inference.checkpoints as checkpoints
from hal.inference.bundle import BundleDescription
from hal.inference.bundle import BundleMember
from hal.inference.bundle import PolicyBundleManifest
from hal.inference.bundle import extract_policy_bundle
from hal.inference.bundle import read_policy_manifest
from hal.inference.bundle import write_policy_bundle


class _R2Client:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.etag = '"version-1"'
        self.downloads = 0
        self.closes = 0

    def close(self) -> None:
        self.closes += 1

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        assert (Bucket, Key) == ("hal", "runs/o50/checkpoint.pt")
        return {"ETag": self.etag, "ContentLength": len(self.payload)}

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        assert (Bucket, Key) == ("hal", "runs/o50/checkpoint.pt")
        self.downloads += 1
        return {"ETag": self.etag, "Body": BytesIO(self.payload)}


def _description() -> BundleDescription:
    return BundleDescription(
        policy_name="test policy",
        backend="tests.fake",
        backend_version=1,
        required_observation_fields=("p1_position_x",),
        supported_transport_delays=(0, 2, 3),
        requires_player_code=False,
        source_sha256="1" * 64,
    )


def test_bundle_round_trip_validates_all_assets(tmp_path: Path) -> None:
    backend = tmp_path / "backend.json"
    weights = tmp_path / "weights.bin"
    backend.write_text('{"kind":"fake"}')
    weights.write_bytes(b"weights")
    bundle = tmp_path / "policy.halpolicy"
    written = write_policy_bundle(bundle, _description(), {"backend.json": backend, "weights.bin": weights})
    assert read_policy_manifest(bundle) == written
    with extract_policy_bundle(bundle) as (loaded, root):
        assert loaded == written
        assert (root / "weights.bin").read_bytes() == b"weights"


def test_bundle_bytes_do_not_depend_on_source_mtime(tmp_path: Path) -> None:
    backend = tmp_path / "backend.json"
    backend.write_text("{}")
    first = tmp_path / "first.halpolicy"
    second = tmp_path / "second.halpolicy"
    write_policy_bundle(first, _description(), {"backend.json": backend})
    os.utime(backend, (2_000_000_000, 2_000_000_000))
    write_policy_bundle(second, _description(), {"backend.json": backend})
    assert first.read_bytes() == second.read_bytes()


def test_bundle_rejects_corrupt_member(tmp_path: Path) -> None:
    backend = tmp_path / "backend.json"
    backend.write_text("{}")
    bundle = tmp_path / "policy.halpolicy"
    write_policy_bundle(bundle, _description(), {"backend.json": backend})
    rewritten = tmp_path / "corrupt.halpolicy"
    with zipfile.ZipFile(bundle) as source, zipfile.ZipFile(rewritten, "w") as destination:
        for info in source.infolist():
            value = source.read(info.filename)
            destination.writestr(info, b"corrupt" if info.filename == "backend.json" else value)
    with pytest.raises(ValueError, match="failed size or SHA-256"), extract_policy_bundle(rewritten):
        pass


def test_bundle_manifest_rejects_boolean_format_version() -> None:
    payload = json.loads(
        PolicyBundleManifest(
            policy_name="test",
            backend="tests.fake",
            backend_version=1,
            action_schema="hal.controller.v1",
            observation_schema="hal.flat.numeric.v1",
            required_observation_fields=(),
            supported_transport_delays=(0,),
            requires_player_code=False,
            source_sha256="1" * 64,
            backend_config="backend.json",
            members=(BundleMember("backend.json", 2, "2" * 64),),
        ).to_json()
    )
    payload["format_version"] = True
    with pytest.raises(ValueError, match="format"):
        PolicyBundleManifest.from_json(json.dumps(payload).encode())


def test_checkpoint_resolver_validates_local_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _R2Client(b"checkpoint")
    monkeypatch.setattr(checkpoints.r2, "client", lambda: client)
    uri = "r2://hal/runs/o50/checkpoint.pt"
    resolved = checkpoints.resolve_checkpoint(uri, cache_root=tmp_path)
    assert resolved.read_bytes() == client.payload
    assert client.downloads == 1
    metadata = json.loads(resolved.with_name("checkpoint.pt.metadata.json").read_text())
    assert metadata["sha256"] == hashlib.sha256(client.payload).hexdigest()

    assert checkpoints.resolve_checkpoint(uri, cache_root=tmp_path) == resolved
    assert client.downloads == 1
    resolved.write_bytes(b"bad")
    assert checkpoints.resolve_checkpoint(uri, cache_root=tmp_path).read_bytes() == client.payload
    assert client.downloads == 2
    assert client.closes == 3


@pytest.mark.parametrize("uri", ["r2://hal", "r2:///key", "r2://hal/path/"])
def test_checkpoint_resolver_rejects_non_objects(uri: str) -> None:
    with pytest.raises(ValueError, match="R2"):
        checkpoints.parse_r2_uri(uri)
