import hashlib
import json
from pathlib import Path
from typing import BinaryIO
from unittest.mock import Mock

import pytest
from botocore.exceptions import ClientError

from hal.netplay_service.admin import check_imitations
from hal.netplay_service.admin import parse_since
from hal.netplay_service.admin import pin_assets
from hal.netplay_service.admin import policy_config
from hal.netplay_service.admin import publish_policy
from hal.netplay_service.admin import upload_accounts
from hal.netplay_service.assets import AssetManifest
from hal.netplay_service.domain import IMITATIONS
from hal.netplay_service.queue_client import Account
from hal.netplay_service.queue_client import AdminClient


class _Bucket:
    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, dict[str, str]]] = {}

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        data, metadata = self.objects[Key]
        return {"ContentLength": len(data), "Metadata": metadata}

    def put_object(
        self, *, Bucket: str, Key: str, Body: BinaryIO, ContentType: str, Metadata: dict[str, str]
    ) -> dict[str, str]:
        self.objects[Key] = (Body.read(), dict(Metadata))
        return {"ETag": '"etag"'}


def _account(path: Path, code: str) -> Path:
    path.write_text(json.dumps({"connectCode": code, "playKey": "k"}))
    return path


def test_policy_config_requires_both_netplay_delays() -> None:
    config = policy_config("a" * 64, "b" * 64, 2, (0, 2, 3))
    assert config.bundle_r2_key == f"netplay/policies/{'a' * 64}.halpolicy"
    assert config.online_delays == (2, 3)
    assert not config.masked_identity
    assert config.imitations[0].value == "MASKED"
    with pytest.raises(ValueError, match="capability-v2"):
        policy_config("a" * 64, "b" * 64, 1, (2,))


def test_imitations_must_be_in_the_bundle_vocabulary() -> None:
    codes = tuple(sorted(choice.value for choice in IMITATIONS if "#" in choice.value))
    check_imitations(IMITATIONS, codes)
    missing = tuple(code for code in codes if code not in ("ZAIN#0", "HBOX#1"))
    with pytest.raises(ValueError, match=r"HBOX#1, ZAIN#0"):
        check_imitations(IMITATIONS, missing)


def test_publish_policy_uploads_then_publishes(tmp_path: Path) -> None:
    bundle = tmp_path / "policy.halpolicy"
    bundle.write_bytes(b"bundle")
    digest = hashlib.sha256(b"bundle").hexdigest()
    config = policy_config(digest, "b" * 64, 2, (0, 2, 3))
    bucket, admin = _Bucket(), Mock(spec=AdminClient)
    assert publish_policy(bundle, config, admin, bucket, "hal")
    assert bucket.objects[config.bundle_r2_key][1] == {"sha256": digest}
    admin.put_policy.assert_called_once_with(config)


def test_accounts_upload_is_content_addressed(tmp_path: Path) -> None:
    first = _account(tmp_path / "a.json", "BOT0#1")
    second = _account(tmp_path / "b.json", "BOT1#1")
    bucket, admin = _Bucket(), Mock(spec=AdminClient)
    accounts = upload_accounts((first, second), admin, bucket, "hal")
    digest = hashlib.sha256(first.read_bytes()).hexdigest()
    assert accounts[0] == Account("BOT0#1", f"netplay/accounts/{digest}.json", digest)
    admin.put_accounts.assert_called_once_with(accounts)


def test_accounts_upload_refuses_duplicate_codes_before_any_upload(tmp_path: Path) -> None:
    first = _account(tmp_path / "a.json", "BOT0#1")
    second = _account(tmp_path / "b.json", "BOT0#1")
    bucket, admin = _Bucket(), Mock(spec=AdminClient)
    with pytest.raises(ValueError, match="BOT0#1 appears twice"):
        upload_accounts((first, second), admin, bucket, "hal")
    assert bucket.objects == {}
    admin.put_accounts.assert_not_called()


def test_accounts_upload_refuses_an_empty_list() -> None:
    admin = Mock(spec=AdminClient)
    with pytest.raises(ValueError, match="at least one account"):
        upload_accounts((), admin, _Bucket(), "hal")
    admin.put_accounts.assert_not_called()


def test_pin_assets_uploads_and_writes_the_manifest(tmp_path: Path) -> None:
    iso = tmp_path / "ssbm.ciso"
    iso.write_bytes(b"iso")
    emulator = tmp_path / "Slippi_Online-x86_64.AppImage"
    emulator.write_bytes(b"emulator")
    manifest_path = tmp_path / "assets.json"
    manifest = pin_assets(iso, emulator, manifest_path, _Bucket(), "hal")
    assert AssetManifest.read(manifest_path) == manifest
    assert manifest.emulator.executable
    assert manifest.iso.key == f"netplay/assets/{hashlib.sha256(b'iso').hexdigest()}/ssbm.ciso"


@pytest.mark.parametrize(("value", "expected"), [("90s", 910.0), ("15m", 100.0), ("1h", -2600.0), ("2d", -171800.0)])
def test_parse_since(value: str, expected: float) -> None:
    assert parse_since(value, 1000.0) == expected


def test_parse_since_rejects_other_forms() -> None:
    with pytest.raises(ValueError, match="like 30m"):
        parse_since("yesterday", 0.0)
