"""End-to-end smoke tests for the .7z streaming pipeline.

Exercises hal.data.archive, build_index --archive, and process_replays
against ``$HAL_DEV_ARCHIVE`` (the small archive we use as a fixture; default
``~/data/raw/dev.7z``) and verifies parity with on-disk extraction.

Skipped when the dev archive isn't present so the suite still runs on CI /
fresh checkouts that don't have the fixture downloaded.
"""

import json
import os
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import py7zr
import pytest
from streaming import StreamingDataset

from hal.data.archive import archive_member_path
from hal.data.archive import iter_archive_members
from hal.data.archive import parse_archive_member_path
from hal.paths import DEV_ARCHIVE_PATH
from hal.paths import repo_relative

DEV_ARCHIVE: Path = Path(DEV_ARCHIVE_PATH)
TMPFS_ROOT: Path = Path("/dev/shm/hal_archive_streaming_test")

pytestmark = pytest.mark.skipif(
    not DEV_ARCHIVE.is_file(),
    reason=f"dev archive missing at {DEV_ARCHIVE}; run `python -m hal.scripts.fetch --name dev.7z`",
)


@pytest.fixture
def tmpfs() -> Iterator[Path]:
    """Per-test tmpfs scratch dir that's wiped before and after."""
    shutil.rmtree(TMPFS_ROOT, ignore_errors=True)
    TMPFS_ROOT.mkdir(parents=True)
    yield TMPFS_ROOT
    shutil.rmtree(TMPFS_ROOT, ignore_errors=True)


def _run_module(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "run", "python", "-m", *args],
        capture_output=True,
        text=True,
        check=True,
    )


def _by_sha(jsonl: Path) -> dict[str, dict]:
    return {json.loads(line)["sha1"]: json.loads(line) for line in jsonl.open()}


def _by_basename(jsonl: Path) -> dict[str, dict]:
    """Key entries by the .slp basename. sha1 is a full-file hash so it would
    also work as a key, but the basename is stable across the
    archive-vs-disk paths and easier to debug from."""
    out: dict[str, dict] = {}
    for line in jsonl.open():
        d = json.loads(line)
        out[Path(d["path"].split("!")[-1] if d["path"].startswith("archive://") else d["path"]).name] = d
    return out


def _col_equal(x: object, y: object) -> bool:
    if isinstance(x, np.ndarray):
        if not isinstance(y, np.ndarray) or x.shape != y.shape:
            return False
        if np.issubdtype(x.dtype, np.floating):
            return np.array_equal(x, y, equal_nan=True)
        return np.array_equal(x, y)
    return x == y


def test_synthetic_path_round_trip() -> None:
    member = "dev/Game_20230614T211840.slp"
    synthetic = archive_member_path(DEV_ARCHIVE, member)
    assert parse_archive_member_path(synthetic) == (repo_relative(DEV_ARCHIVE), member)
    assert parse_archive_member_path("/tmp/foo.slp") is None
    with pytest.raises(ValueError, match="malformed"):
        parse_archive_member_path("archive:///tmp/foo.7z")


def test_iter_archive_members_streams_and_cleans(tmpfs: Path) -> None:
    """Every .slp comes through, /dev/shm stays bounded, ring drains on exit."""
    queue_size = 8
    n = 0
    total_bytes = 0
    max_inflight = 0
    run_dir = tmpfs / str(os.getpid())
    for _, p in iter_archive_members(DEV_ARCHIVE, tmpfs_root=tmpfs, queue_size=queue_size):
        max_inflight = max(max_inflight, len(list(run_dir.iterdir())))
        total_bytes += p.stat().st_size
        p.unlink()
        n += 1
    assert n > 100
    # +2 slack for the brief race between producer fill and consumer yield.
    assert max_inflight <= queue_size + 2, max_inflight
    # No .slp leaked into any subdir of tmpfs. The per-run subdir itself stays
    # until atexit (or until the next run's startup sweep) — that's by design.
    assert not list(run_dir.iterdir())
    assert total_bytes > 100_000_000


def test_iter_archive_members_filter(tmpfs: Path) -> None:
    target = "dev/Game_20230614T211840.slp"
    seen = []
    for syn, p in iter_archive_members(DEV_ARCHIVE, tmpfs_root=tmpfs, filter_paths={target}):
        seen.append(syn)
        p.unlink()
    assert seen == [archive_member_path(DEV_ARCHIVE, target)]


def test_build_index_archive_matches_root(tmp_path: Path, tmpfs: Path) -> None:
    """build_index --archive yields the same metadata as build_index --root on
    the same files, modulo the path field."""
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with py7zr.SevenZipFile(DEV_ARCHIVE, "r") as z:
        z.extractall(path=extracted)

    idx_arc = tmp_path / "idx_arc.jsonl"
    idx_disk = tmp_path / "idx_disk.jsonl"
    _run_module("hal.scripts.build_index", "--archive", str(DEV_ARCHIVE), "--output", str(idx_arc), "--workers", "4")
    _run_module("hal.scripts.build_index", "--root", str(extracted), "--output", str(idx_disk), "--workers", "4")

    arc = _by_basename(idx_arc)
    disk = _by_basename(idx_disk)
    assert set(arc) == set(disk)
    checked = (
        "slp_version",
        "stage",
        "frame_count",
        "timestamp",
        "played_on",
        "outcome",
        "rank_filename",
        "players",
        "sha1",
    )
    for name, a in arc.items():
        d = disk[name]
        for f in checked:
            assert a[f] == d[f], (name, f)


def test_process_replays_archive_byte_equal(tmp_path: Path, tmpfs: Path) -> None:
    """process_replays infers archives from paths.txt and writes byte-identical
    MDS rows to the on-disk variant."""
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with py7zr.SevenZipFile(DEV_ARCHIVE, "r") as z:
        z.extractall(path=extracted)

    idx_arc = tmp_path / "idx_arc.jsonl"
    idx_disk = tmp_path / "idx_disk.jsonl"
    _run_module("hal.scripts.build_index", "--archive", str(DEV_ARCHIVE), "--output", str(idx_arc), "--workers", "4")
    _run_module("hal.scripts.build_index", "--root", str(extracted), "--output", str(idx_disk), "--workers", "4")

    paths_arc = tmp_path / "paths_arc.txt"
    paths_disk = tmp_path / "paths_disk.txt"
    _run_module(
        "hal.scripts.filter",
        "--index",
        str(idx_arc),
        "--output",
        str(paths_arc),
        "--min-frames",
        "0",
        "--no-completed-only",
        "--stages",
    )
    _run_module(
        "hal.scripts.filter",
        "--index",
        str(idx_disk),
        "--output",
        str(paths_disk),
        "--min-frames",
        "0",
        "--no-completed-only",
        "--stages",
    )

    mds_arc = tmp_path / "mds_arc"
    mds_disk = tmp_path / "mds_disk"
    _run_module(
        "hal.scripts.materialize",
        "--paths-file",
        str(paths_arc),
        "--index",
        str(idx_arc),
        "--output",
        str(mds_arc),
        "--workers",
        "4",
    )
    _run_module(
        "hal.scripts.materialize",
        "--paths-file",
        str(paths_disk),
        "--index",
        str(idx_disk),
        "--output",
        str(mds_disk),
        "--workers",
        "4",
    )

    m_arc = _by_basename(mds_arc / "manifest.jsonl")
    m_disk = _by_basename(mds_disk / "manifest.jsonl")
    assert set(m_arc) == set(m_disk)

    # Note: synthetic path != filesystem path → different replay_uuid →
    # the same .slp can land in different splits across the two modes.
    # Open all splits on each side and look up by manifest annotation.
    def _open_splits(mds_dir: Path, manifest: dict[str, dict]) -> dict[str, StreamingDataset]:
        used = {e["annotation"]["split"] for e in manifest.values()}
        return {
            split: StreamingDataset(local=str(mds_dir / split), shuffle=False, batch_size=1, predownload=8)
            for split in used
        }

    ds_arc = _open_splits(mds_arc, m_arc)
    ds_disk = _open_splits(mds_disk, m_disk)
    for name, a_entry in m_arc.items():
        a = ds_arc[a_entry["annotation"]["split"]][a_entry["annotation"]["mds_row_idx"]]
        d_entry = m_disk[name]
        d = ds_disk[d_entry["annotation"]["split"]][d_entry["annotation"]["mds_row_idx"]]
        for col in set(a) | set(d):
            assert _col_equal(a.get(col), d.get(col)), f"{name} {col}"


def test_process_replays_fails_fast_on_missing_archive(tmp_path: Path) -> None:
    """A non-existent archive in paths.txt fails before any shard is written."""
    paths = tmp_path / "paths.txt"
    paths.write_text("archive:///nonexistent/path/foo.7z!dev/bar.slp\n")
    # Need an index file to satisfy the existence check before the archive check.
    index = tmp_path / "idx.jsonl"
    index.write_text("")
    output = tmp_path / "mds_out"

    proc = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "hal.scripts.materialize",
            "--paths-file",
            str(paths),
            "--index",
            str(index),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "not found on disk" in proc.stderr
    assert not output.exists()
