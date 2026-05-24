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

from hal.data.archive import archive_member_path
from hal.data.archive import iter_archive_members
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
