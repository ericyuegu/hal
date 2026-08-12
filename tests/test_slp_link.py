"""Staging rules for the slippilab link CLI: one served path per source, no collisions."""

import struct
import urllib.parse
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from hal.scripts import slp_link

_HEADER = b"{U\x03raw[$U#l"
_EVENTS = bytes([0x35, 0x04, 0x38, 0x00, 0x04]) + bytes([0x38, 1, 2, 3, 4])


def _slp(path, *, finalized: bool) -> None:
    """Write a minimal Slippi raw file; unfinalized means rawLength == 0."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = _HEADER + struct.pack(">i", len(_EVENTS) if finalized else 0) + _EVENTS
    path.write_bytes(body + (b"U\x08metadata{}}" if finalized else b""))


@pytest.fixture
def serve_dir(tmp_path, monkeypatch):
    served = tmp_path / "served"
    served.mkdir()
    monkeypatch.setattr(slp_link, "SERVE_DIR", served)
    return served


def test_same_basename_in_different_dirs_stays_distinct(tmp_path, serve_dir):
    """The collision that made every `boot_*/A-c0000-o0.slp` serve boot_000's replay."""
    sources = [tmp_path / "boot_000" / "Game.slp", tmp_path / "boot_001" / "Game.slp"]
    for src in sources:
        _slp(src, finalized=True)

    urls = [slp_link._link(src) for src in sources]
    staged = [slp_link._staged_path(src) for src in sources]

    assert urls[0] != urls[1]
    assert staged[0] != staged[1]
    assert [s.resolve() for s in staged] == sources  # each link points at its own source


def test_underscores_in_dir_names_do_not_collide(tmp_path, serve_dir):
    """`a/b__c/x.slp` vs `a/b/c/x.slp` — the flattened-name scheme mapped both to one file."""
    sources = [tmp_path / "b__c" / "x.slp", tmp_path / "b" / "c" / "x.slp"]
    for src in sources:
        _slp(src, finalized=True)

    assert slp_link._staged_path(sources[0]) != slp_link._staged_path(sources[1])
    assert slp_link._link(sources[0]) != slp_link._link(sources[1])


def test_restages_dangling_link(tmp_path, serve_dir):
    """`Path.exists()` follows symlinks, so a dangling staged link used to raise FileExistsError."""
    src = tmp_path / "Game.slp"
    _slp(src, finalized=True)
    staged = slp_link._staged_path(src)
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.symlink_to(tmp_path / "gone.slp")
    assert staged.is_symlink() and not staged.exists()

    slp_link._link(src)

    assert staged.resolve() == src


def test_link_is_idempotent(tmp_path, serve_dir):
    src = tmp_path / "Game.slp"
    _slp(src, finalized=True)
    assert slp_link._link(src) == slp_link._link(src)
    assert slp_link._staged_path(src).resolve() == src


def test_unfinalized_is_staged_as_a_finalized_copy(tmp_path, serve_dir):
    """A match killed mid-game leaves rawLength == 0; slippilab needs the repaired bytes."""
    src = tmp_path / "Game.slp"
    _slp(src, finalized=False)

    slp_link._link(src)
    staged = slp_link._staged_path(src)

    assert not staged.is_symlink()  # a copy, so the source stays untouched on disk
    assert src.read_bytes() != staged.read_bytes()
    assert struct.unpack(">i", staged.read_bytes()[11:15])[0] == len(_EVENTS)
    assert staged.read_bytes().endswith(b"U\x08metadata{}}")


def test_restages_copy_once_the_source_finalizes(tmp_path, serve_dir):
    """Linking a live match stages a repaired copy; re-linking after it closes must not serve that."""
    src = tmp_path / "Game.slp"
    _slp(src, finalized=False)
    slp_link._link(src)
    assert not slp_link._staged_path(src).is_symlink()

    _slp(src, finalized=True)  # the match ended and Dolphin closed the file
    slp_link._link(src)

    assert slp_link._staged_path(src).resolve() == src  # now a symlink to the real thing


def test_non_slp_fails_loud(tmp_path, serve_dir):
    src = tmp_path / "Game.slp"
    src.write_bytes(b"not a slippi file at all")
    with pytest.raises(SystemExit, match="not a Slippi"):
        slp_link._link(src)


def test_collect_walks_dirs_and_dedupes(tmp_path, monkeypatch):
    monkeypatch.setattr(slp_link, "REPO_DIR", str(tmp_path))
    for name in ["boot_000/Game.slp", "boot_001/Game.slp", "notes.txt"]:
        _slp(tmp_path / name, finalized=True)

    found = slp_link._collect([tmp_path])

    assert found == [tmp_path / "boot_000" / "Game.slp", tmp_path / "boot_001" / "Game.slp"]
    assert slp_link._collect([tmp_path, tmp_path / "boot_000" / "Game.slp"]) == found  # deduped
    assert slp_link._collect([tmp_path / "boot_000"]) == [tmp_path / "boot_000" / "Game.slp"]


def test_collect_resolves_relative_paths_against_the_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(slp_link, "REPO_DIR", str(tmp_path))
    _slp(tmp_path / "runs" / "Game.slp", finalized=True)

    assert slp_link._collect([Path("runs")]) == [tmp_path / "runs" / "Game.slp"]


def test_collect_rejects_missing_path(tmp_path):
    with pytest.raises(SystemExit, match="no such file or directory"):
        slp_link._collect([tmp_path / "nope"])


def test_parse_r2_locator_preserves_bucket_and_key():
    locator = slp_link._parse_r2("r2:hal/runs/a directory/Game.slp")

    assert locator == slp_link.R2Object(bucket="hal", key="runs/a directory/Game.slp")
    assert str(locator) == "r2:hal/runs/a directory/Game.slp"


@pytest.mark.parametrize("value", ["r2:", "r2:hal"])
def test_parse_r2_locator_rejects_malformed_values(value):
    with pytest.raises(SystemExit, match="expected r2:<bucket>"):
        slp_link._parse_r2(value)


def test_collect_r2_prefix_paginates_filters_and_sorts():
    client = MagicMock()
    paginator = client.get_paginator.return_value
    paginator.paginate.return_value = [
        {"Contents": [{"Key": "runs/z.slp"}, {"Key": "runs/notes.txt"}]},
        {"Contents": [{"Key": "runs/a.slp"}]},
    ]

    found = slp_link._collect_r2(slp_link.R2Object("hal", "runs/"), client)

    assert found == [slp_link.R2Object("hal", "runs/a.slp"), slp_link.R2Object("hal", "runs/z.slp")]
    client.get_paginator.assert_called_once_with("list_objects_v2")
    paginator.paginate.assert_called_once_with(Bucket="hal", Prefix="runs/")


def test_collect_r2_exact_object_uses_head_instead_of_listing():
    client = MagicMock()
    locator = slp_link.R2Object("hal", "runs/Game.slp")

    assert slp_link._collect_r2(locator, client) == [locator]
    client.head_object.assert_called_once_with(Bucket="hal", Key="runs/Game.slp")
    client.get_paginator.assert_not_called()


def test_collect_r2_reports_access_failure():
    client = MagicMock()
    client.head_object.side_effect = ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")

    with pytest.raises(SystemExit, match="failed to inspect r2:hal/missing.slp"):
        slp_link._collect_r2(slp_link.R2Object("hal", "missing.slp"), client)


@pytest.mark.parametrize("allowed_origin", ["http://localhost:5173", "http://localhost:*"])
def test_validate_r2_cors_accepts_exact_or_wildcard_origin(allowed_origin):
    client = MagicMock()
    client.get_bucket_cors.return_value = {
        "CORSRules": [{"AllowedOrigins": [allowed_origin], "AllowedMethods": ["GET"]}]
    }

    slp_link._validate_r2_cors("hal", client)


def test_validate_r2_cors_rejects_missing_browser_get_rule():
    client = MagicMock()
    client.get_bucket_cors.return_value = {
        "CORSRules": [{"AllowedOrigins": ["https://example.com"], "AllowedMethods": ["GET", "HEAD"]}]
    }

    with pytest.raises(SystemExit, match="does not allow browser GETs from http://localhost:5173"):
        slp_link._validate_r2_cors("hal", client)


def test_validate_r2_cors_reports_unverifiable_policy():
    client = MagicMock()
    client.get_bucket_cors.side_effect = ClientError(
        {"Error": {"Code": "NoSuchCORSConfiguration", "Message": "No CORS"}}, "GetBucketCors"
    )

    with pytest.raises(SystemExit, match="cannot verify CORS for R2 bucket 'hal'"):
        slp_link._validate_r2_cors("hal", client)


def test_r2_link_embeds_complete_presigned_url_and_expiry():
    client = MagicMock()
    signed = "https://example.r2/runs/Game%20One.slp?X-Amz-Signature=abc&x-id=GetObject"
    client.generate_presigned_url.return_value = signed
    obj = slp_link.R2Object("hal", "runs/Game One.slp")

    link = slp_link._r2_link(obj, client, expires_in=7200)

    replay_url = urllib.parse.parse_qs(urllib.parse.urlparse(link).query)["replayUrl"][0]
    assert replay_url == signed
    client.generate_presigned_url.assert_called_once_with(
        "get_object", Params={"Bucket": "hal", "Key": "runs/Game One.slp"}, ExpiresIn=7200
    )


def test_shared_r2_client_generates_r2_compatible_sigv4_url(monkeypatch):
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://example.r2.cloudflarestorage.com")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")

    signed = slp_link.r2.client().generate_presigned_url(
        "get_object", Params={"Bucket": "hal", "Key": "runs/Game.slp"}, ExpiresIn=3600
    )
    query = urllib.parse.parse_qs(urllib.parse.urlparse(signed).query)

    assert query["X-Amz-Algorithm"] == ["AWS4-HMAC-SHA256"]
    assert query["X-Amz-Expires"] == ["3600"]


def test_remote_only_cli_is_lazy_deduped_and_does_not_mount(monkeypatch, capsys):
    client = MagicMock()
    client.get_paginator.return_value.paginate.side_effect = [
        [{"Contents": [{"Key": "runs/a.slp"}, {"Key": "runs/b.slp"}]}],
        [{"Contents": [{"Key": "runs/a.slp"}]}],
    ]
    client.get_bucket_cors.return_value = {
        "CORSRules": [{"AllowedOrigins": [slp_link.SLIPPILAB_URL], "AllowedMethods": ["GET"]}]
    }
    client.generate_presigned_url.side_effect = lambda _operation, Params, ExpiresIn: (
        f"https://r2.example/{Params['Key']}?expires={ExpiresIn}"
    )
    monkeypatch.setattr(slp_link.r2, "client", lambda: client)
    ensure_mount = MagicMock()
    monkeypatch.setattr(slp_link, "_ensure_mount", ensure_mount)

    slp_link.slp_link(["r2:hal/runs/", "r2:hal/runs/a.slp"], expires_in=123)

    output = capsys.readouterr().out
    assert output.count("r2:hal/runs/a.slp\n") == 1
    assert output.count("r2:hal/runs/b.slp\n") == 1
    assert client.generate_presigned_url.call_count == 2
    client.get_bucket_cors.assert_called_once_with(Bucket="hal")
    ensure_mount.assert_not_called()


@pytest.mark.parametrize("expires_in", [0, 604_801])
def test_cli_rejects_expiry_outside_r2_range(expires_in):
    with pytest.raises(SystemExit, match="--expires-in must be between 1 and 604800"):
        slp_link.slp_link(["r2:hal/runs/"], expires_in=expires_in)
