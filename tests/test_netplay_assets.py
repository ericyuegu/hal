import hashlib
import json
from pathlib import Path
from typing import BinaryIO

import pytest
from botocore.exceptions import ClientError

from hal.netplay_service.assets import AssetCache
from hal.netplay_service.assets import AssetError
from hal.netplay_service.assets import AssetManifest
from hal.netplay_service.assets import LocalSource
from hal.netplay_service.assets import PinnedAsset
from hal.netplay_service.assets import ensure_uploaded
from hal.netplay_service.assets import pinned_asset_key


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _mirror(tmp_path: Path, key: str, data: bytes) -> LocalSource:
    path = tmp_path / "mirror" / key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return LocalSource(tmp_path / "mirror")


class _Bucket:
    """An in-memory stand-in for the two S3 calls `ensure_uploaded` makes."""

    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, dict[str, str]]] = {}
        self.puts = 0

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        data, metadata = self.objects[Key]
        return {"ContentLength": len(data), "Metadata": metadata}

    def put_object(
        self, *, Bucket: str, Key: str, Body: BinaryIO, ContentType: str, Metadata: dict[str, str]
    ) -> dict[str, str]:
        self.puts += 1
        self.objects[Key] = (Body.read(), dict(Metadata))
        return {"ETag": '"etag"'}


def test_manifest_round_trips_and_rejects_drift(tmp_path: Path) -> None:
    manifest = AssetManifest(
        PinnedAsset(pinned_asset_key("a" * 64, "ssbm.ciso"), "a" * 64),
        PinnedAsset(pinned_asset_key("b" * 64, "Slippi_Online-x86_64.AppImage"), "b" * 64, executable=True),
    )
    path = tmp_path / "assets.json"
    manifest.write(path)
    assert AssetManifest.read(path) == manifest
    payload = json.loads(path.read_text())
    payload["extra"] = 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="fields changed"):
        AssetManifest.read(path)


@pytest.mark.parametrize("key", ["", "/abs", "a/../b"])
def test_pinned_asset_rejects_unsafe_keys(key: str) -> None:
    with pytest.raises(ValueError, match="key"):
        PinnedAsset(key, "a" * 64)


def test_cache_fetches_verifies_and_marks_executables(tmp_path: Path) -> None:
    data = b"emulator"
    asset = PinnedAsset(pinned_asset_key(_digest(data), "Slippi.AppImage"), _digest(data), executable=True)
    cache = AssetCache(tmp_path / "cache", _mirror(tmp_path, asset.key, data))
    path = cache.get(asset)
    assert path == tmp_path / "cache" / asset.sha256 / "Slippi.AppImage"
    assert path.read_bytes() == data
    assert path.stat().st_mode & 0o111


def test_cache_rejects_a_wrong_hash_and_leaves_nothing(tmp_path: Path) -> None:
    asset = PinnedAsset("netplay/assets/x/ssbm.ciso", "c" * 64)
    cache = AssetCache(tmp_path / "cache", _mirror(tmp_path, asset.key, b"not the iso"))
    with pytest.raises(AssetError, match="expected " + "c" * 64):
        cache.get(asset)
    assert list((tmp_path / "cache" / asset.sha256).iterdir()) == []


def test_cache_replaces_a_corrupted_cached_file(tmp_path: Path) -> None:
    data = b"iso bytes"
    asset = PinnedAsset(pinned_asset_key(_digest(data), "ssbm.ciso"), _digest(data))
    cache = AssetCache(tmp_path / "cache", _mirror(tmp_path, asset.key, data))
    cached = cache.get(asset)
    cached.write_bytes(b"iso")
    assert cache.get(asset).read_bytes() == data


def test_local_source_names_a_missing_key(tmp_path: Path) -> None:
    cache = AssetCache(tmp_path / "cache", LocalSource(tmp_path / "empty"))
    with pytest.raises(AssetError, match="local asset is missing"):
        cache.get(PinnedAsset("netplay/accounts/a.json", "a" * 64))


def test_ensure_uploaded_is_idempotent_and_refuses_a_different_object(tmp_path: Path) -> None:
    path = tmp_path / "bundle.halpolicy"
    path.write_bytes(b"bundle")
    bucket = _Bucket()
    assert ensure_uploaded(bucket, "hal", path, "netplay/policies/x.halpolicy", _digest(b"bundle"))
    assert not ensure_uploaded(bucket, "hal", path, "netplay/policies/x.halpolicy", _digest(b"bundle"))
    assert bucket.puts == 1
    with pytest.raises(AssetError, match="different hash"):
        ensure_uploaded(bucket, "hal", path, "netplay/policies/x.halpolicy", "f" * 64)


def test_ensure_uploaded_refuses_a_truncated_object_with_matching_metadata(tmp_path: Path) -> None:
    path = tmp_path / "bundle.halpolicy"
    path.write_bytes(b"bundle")
    bucket = _Bucket()
    bucket.objects["netplay/policies/x.halpolicy"] = (b"bun", {"sha256": _digest(b"bundle")})
    with pytest.raises(AssetError, match="3 bytes; expected 6"):
        ensure_uploaded(bucket, "hal", path, "netplay/policies/x.halpolicy", _digest(b"bundle"))
    assert bucket.puts == 0


def test_manifest_rejects_a_key_that_is_not_content_addressed(tmp_path: Path) -> None:
    manifest = AssetManifest(
        PinnedAsset(pinned_asset_key("a" * 64, "ssbm.ciso"), "a" * 64),
        PinnedAsset(pinned_asset_key("b" * 64, "Slippi.AppImage"), "b" * 64, executable=True),
    )
    path = tmp_path / "assets.json"
    manifest.write(path)
    payload = json.loads(path.read_text())
    payload["iso"]["key"] = pinned_asset_key("b" * 64, "ssbm.ciso")
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="iso key must be"):
        AssetManifest.read(path)
