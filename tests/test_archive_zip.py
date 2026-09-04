"""Synthetic tests for the zip-of-``.slp.gz`` reader path in hal.data.archive.

Builds fixture archives inline (no on-disk dependency) so the suite runs on
fresh checkouts. End-to-end coverage on real public-dump chunks (peppi parse,
full index pipeline) is exercised manually via
``python -m hal.scripts.build_index --archive ranked-anonymized-N-*.7z``.
"""

import gzip
import io
import shutil
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from hal.data.archive import _slpz_binary
from hal.data.archive import archive_member_path
from hal.data.archive import iter_archive_members
from hal.data.archive import iter_replay_work
from hal.data.archive import list_archive_slps
from hal.data.archive import read_archive_member_to_file

TMPFS_ROOT: Path = Path("/dev/shm/hal_archive_zip_test")


@pytest.fixture
def tmpfs() -> Iterator[Path]:
    shutil.rmtree(TMPFS_ROOT, ignore_errors=True)
    TMPFS_ROOT.mkdir(parents=True)
    yield TMPFS_ROOT
    shutil.rmtree(TMPFS_ROOT, ignore_errors=True)


def _build_zip_of_gz(path: Path, payloads: dict[str, bytes]) -> None:
    """Build a zip whose members are ``<name>.slp.gz`` wrapping the raw payload."""
    with zipfile.ZipFile(path, "w") as z:
        for name, raw in payloads.items():
            buf = io.BytesIO()
            with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
                gz.write(raw)
            z.writestr(f"{name}.slp.gz", buf.getvalue())


def test_list_archive_slps_zip(tmp_path: Path) -> None:
    p = tmp_path / "a.zip"
    _build_zip_of_gz(p, {"hash-001": b"x", "hash-002": b"y"})
    assert sorted(list_archive_slps(p)) == ["hash-001.slp.gz", "hash-002.slp.gz"]


def test_iter_archive_members_zip_strips_gz(tmp_path: Path, tmpfs: Path) -> None:
    payloads = {
        "hash-001": b"raw-slp-bytes-one",
        "hash-002": b"raw-slp-bytes-two",
    }
    p = tmp_path / "a.zip"
    _build_zip_of_gz(p, payloads)

    seen: dict[str, bytes] = {}
    for syn, tmpfs_path in iter_archive_members(p, tmpfs_root=tmpfs):
        seen[syn] = tmpfs_path.read_bytes()
        tmpfs_path.unlink()

    expected = {archive_member_path(p, f"{n}.slp.gz"): payloads[n] for n in payloads}
    assert seen == expected


def test_iter_archive_members_zip_filter(tmp_path: Path, tmpfs: Path) -> None:
    p = tmp_path / "a.zip"
    _build_zip_of_gz(p, {"hash-001": b"a", "hash-002": b"b"})

    seen: list[tuple[str, bytes]] = []
    for syn, tmpfs_path in iter_archive_members(p, tmpfs_root=tmpfs, filter_paths={"hash-002.slp.gz"}):
        seen.append((syn, tmpfs_path.read_bytes()))
        tmpfs_path.unlink()
    assert seen == [(archive_member_path(p, "hash-002.slp.gz"), b"b")]


def test_iter_archive_members_zip_rejects_missing_filter_member(tmp_path: Path, tmpfs: Path) -> None:
    archive = tmp_path / "a.zip"
    _build_zip_of_gz(archive, {"present": b"a"})

    with pytest.raises(FileNotFoundError, match="missing.slp.gz"):
        list(iter_archive_members(archive, tmpfs_root=tmpfs, filter_paths={"missing.slp.gz"}))


def test_read_archive_member_to_file_zip(tmp_path: Path) -> None:
    p = tmp_path / "a.zip"
    _build_zip_of_gz(p, {"hash-001": b"hello world"})
    dest = tmp_path / "out"
    dest.mkdir()
    out = read_archive_member_to_file(p, "hash-001.slp.gz", dest)
    assert out.name == "hash-001.slp"
    assert out.read_bytes() == b"hello world"


def test_read_archive_member_to_file_zip_missing(tmp_path: Path) -> None:
    p = tmp_path / "a.zip"
    _build_zip_of_gz(p, {"hash-001": b"hi"})
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(FileNotFoundError):
        read_archive_member_to_file(p, "missing.slp.gz", dest)


def test_unknown_magic_raises(tmp_path: Path) -> None:
    p = tmp_path / "junk.bin"
    p.write_bytes(b"not-an-archive-format")
    with pytest.raises(ValueError, match="unrecognized archive magic"):
        list_archive_slps(p)


def test_slpz_binary_rejects_an_unvalidated_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binary = tmp_path / "slpz-1.4.0" / "bin" / "slpz"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\nprintf 'slpz 1.4.0\\n'\n")
    binary.chmod(0o755)
    (binary.parent.parent / ".crates2.json").write_text(
        '{"installs":{"slpz 1.4.0 (registry+example)":{"bins":["slpz"]}}}'
    )
    monkeypatch.setenv("HAL_SLPZ_BIN", str(binary))
    _slpz_binary.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="must be version 1.3.0"):
            _slpz_binary()
    finally:
        _slpz_binary.cache_clear()


def test_slpz_binary_rejects_a_missing_configured_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing = tmp_path / "missing-slpz"
    monkeypatch.setenv("HAL_SLPZ_BIN", str(missing))
    _slpz_binary.cache_clear()
    try:
        with pytest.raises(FileNotFoundError, match="does not name an executable"):
            _slpz_binary()
    finally:
        _slpz_binary.cache_clear()


def test_slpz_binary_accepts_validated_cargo_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binary = tmp_path / "slpz-1.3.0" / "bin" / "slpz"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\ncat\n")
    binary.chmod(0o755)
    (binary.parent.parent / ".crates2.json").write_text(
        '{"installs":{"slpz 1.3.0 (registry+example)":{"bins":["slpz"]}}}'
    )
    monkeypatch.setenv("HAL_SLPZ_BIN", str(binary))
    _slpz_binary.cache_clear()
    try:
        assert _slpz_binary() == binary
    finally:
        _slpz_binary.cache_clear()


def test_read_archive_member_to_file_slpz_uses_validated_decoder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary = tmp_path / "slpz-1.3.0" / "bin" / "slpz"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\ncat\n")
    binary.chmod(0o755)
    (binary.parent.parent / ".crates2.json").write_text(
        '{"installs":{"slpz 1.3.0 (registry+example)":{"bins":["slpz"]}}}'
    )
    monkeypatch.setenv("HAL_SLPZ_BIN", str(binary))
    _slpz_binary.cache_clear()
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("game.slpz", b"decoded replay")
    output = tmp_path / "out"
    output.mkdir()
    try:
        decoded = read_archive_member_to_file(archive, "game.slpz", output)
    finally:
        _slpz_binary.cache_clear()
    assert decoded.name == "game.slp"
    assert decoded.read_bytes() == b"decoded replay"


def test_iter_replay_work_surfaces_bad_zip_crc(tmp_path: Path, tmpfs: Path) -> None:
    p = tmp_path / "corrupt.zip"
    payload = b"unique-replay-payload-for-crc-test"
    with zipfile.ZipFile(p, "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr("bad.slp", payload)

    raw = bytearray(p.read_bytes())
    payload_offset = raw.index(payload)
    raw[payload_offset] ^= 0x01
    p.write_bytes(raw)

    work = list(
        iter_replay_work(
            archive_members={p: ["bad.slp"]},
            tmpfs_root=tmpfs,
        )
    )
    assert len(work) == 1
    assert work[0].manifest_key == archive_member_path(p, "bad.slp")
    assert work[0].open_error is not None
    assert "Bad CRC-32" in work[0].open_error


def test_iter_replay_work_keeps_paths_unique_across_archives(tmp_path: Path, tmpfs: Path) -> None:
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    _build_zip_of_gz(first, {"game": b"first-replay"})
    _build_zip_of_gz(second, {"game": b"second-replay"})

    # Materialization can have files from both archives in flight at once.
    # Do not unlink the first file before asking the iterator for the second;
    # this reproduces the process-pool prefetch collision this guards against.
    work = list(
        iter_replay_work(
            archive_members={first: ["game.slp.gz"], second: ["game.slp.gz"]},
            tmpfs_root=tmpfs,
        )
    )

    assert work[0].open_path != work[1].open_path
    assert [item.open_path.read_bytes() for item in work] == [b"first-replay", b"second-replay"]
    for item in work:
        item.open_path.unlink()
