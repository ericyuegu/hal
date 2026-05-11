"""Stage 1: walk a replay tree (or a .7z archive) and build `index.jsonl`.

One `ReplayIndexEntry` per .slp, populated from peppi's start/end/metadata
blocks (no frame iteration). Parallelized via `mp.Pool`.

Usage:
    # Loose .slp files on disk
    python -m hal.data.build_index --root /path/to/replays --output index.jsonl

    # .slp members streamed directly from a solid .7z archive (no extraction)
    python -m hal.data.build_index --archive /path/to/archive.7z --output index.jsonl

`--root` and `--archive` are mutually exclusive; exactly one is required.
Archive mode materializes each member to a tmpfs file (default `/dev/shm`)
just long enough for peppi-py to read it, then unlinks. Synthetic paths
(`archive://<abs-archive>!<member>`) are recorded in `entry.path`.

Incremental mode reads the existing `index.jsonl`, collects every path already
recorded, and only indexes new entries. Failed parses are logged and counted
but never halt the run.
"""

import dataclasses
import multiprocessing as mp
from collections.abc import Iterator
from pathlib import Path

import py7zr
import tyro
from loguru import logger
from tqdm import tqdm

from hal.data.archive_iter import archive_member_path
from hal.data.archive_iter import iter_archive_members
from hal.data.manifest import ReplayIndexEntry
from hal.data.manifest import extract_index_entry
from hal.data.manifest import read_jsonl
from hal.data.manifest import write_jsonl

_DEFAULT_TMPFS: Path = Path("/dev/shm/hal_build_index")


def _index_one(args: tuple[Path, bool, str | None]) -> ReplayIndexEntry | None:
    """Worker: parse one .slp into a ReplayIndexEntry.

    `synthetic_path`, when set, replaces the on-disk path written into the
    entry — used so the index records `archive://...!member` instead of the
    transient tmpfs path. The tmpfs file is unlinked on the way out
    (success or failure) so workers don't leak ring-buffer slots.
    """
    path, compute_sha1, synthetic_path = args
    try:
        entry = extract_index_entry(path, compute_sha1=compute_sha1)
    except Exception as e:
        logger.debug(f"unhandled error indexing {path}: {e}")
        entry = None
    finally:
        if synthetic_path is not None:
            path.unlink(missing_ok=True)
    if entry is not None and synthetic_path is not None:
        entry = dataclasses.replace(entry, path=synthetic_path)
    return entry


def _existing_paths(index_path: Path) -> set[str]:
    if not index_path.exists():
        return set()
    return {entry.path for entry in read_jsonl(index_path)}


def _filesystem_work(root: Path, seen: set[str]) -> tuple[list[tuple[Path, bool, None]], int]:
    all_paths = sorted(root.rglob("*.slp"))
    new = [p for p in all_paths if str(p.resolve()) not in seen]
    logger.info(f"found {len(all_paths)} slps under {root}; {len(new)} to index")
    return [(p, True, None) for p in new], len(new)


def _list_archive_slps(archive: Path) -> list[str]:
    """Cheap (header-only) list of .slp member names, in archive order."""
    with py7zr.SevenZipFile(str(archive), "r") as z:
        return [name for name in z.getnames() if name.endswith(".slp")]


def _archive_work(
    archive: Path,
    seen: set[str],
    *,
    tmpfs_root: Path,
    queue_size: int,
    compute_sha1: bool,
) -> tuple[Iterator[tuple[Path, bool, str]], int]:
    members = _list_archive_slps(archive)
    new_members = [m for m in members if archive_member_path(archive, m) not in seen]
    logger.info(f"archive {archive.name}: {len(members)} slps, {len(new_members)} to index")
    new_set = set(new_members)

    def _gen() -> Iterator[tuple[Path, bool, str]]:
        for synthetic, tmpfs_path in iter_archive_members(
            archive,
            tmpfs_root=tmpfs_root,
            filter_paths=new_set,
            queue_size=queue_size,
        ):
            yield tmpfs_path, compute_sha1, synthetic

    return _gen(), len(new_members)


def build_index(
    output: Path,
    *,
    root: Path | None = None,
    archive: Path | None = None,
    incremental: bool = False,
    compute_sha1: bool = True,
    workers: int = max(1, (mp.cpu_count() or 2) - 1),
    tmpfs_root: Path = _DEFAULT_TMPFS,
    queue_size: int = 64,
) -> None:
    if (root is None) == (archive is None):
        raise ValueError("pass exactly one of --root or --archive")
    if root is not None and not root.is_dir():
        raise NotADirectoryError(f"--root must be a directory; got {root}")
    if archive is not None and not archive.is_file():
        raise FileNotFoundError(f"--archive not found: {archive}")

    seen: set[str] = _existing_paths(output) if incremental else set()
    if incremental:
        logger.info(f"incremental: {len(seen)} paths already in {output}")
    elif output.exists():
        raise FileExistsError(f"{output} already exists; pass --incremental to append, or delete it first")

    output.parent.mkdir(parents=True, exist_ok=True)

    if archive is not None:
        work_iter, total = _archive_work(
            archive,
            seen,
            tmpfs_root=tmpfs_root,
            queue_size=queue_size,
            compute_sha1=compute_sha1,
        )
    else:
        assert root is not None  # narrowed by the mutual-exclusion check above
        work_list, total = _filesystem_work(root, seen)
        work_iter = iter(work_list)

    if total == 0:
        return

    written = 0
    failed = 0
    batch: list[ReplayIndexEntry] = []
    BATCH = 256

    # Use fork explicitly: py3.14 defaults to forkserver on Linux, which
    # re-imports the caller's module in each worker — that breaks when the
    # caller is a VSCode interactive cell or any script that runs work at
    # import time. Fork is safe here because workers run a pure function.
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        results = pool.imap_unordered(_index_one, work_iter, chunksize=8)
        for entry in tqdm(results, total=total, desc="indexing", unit="slp"):
            if entry is None:
                failed += 1
                continue
            batch.append(entry)
            if len(batch) >= BATCH:
                write_jsonl(output, batch, append=True)
                written += len(batch)
                batch.clear()
        if batch:
            write_jsonl(output, batch, append=True)
            written += len(batch)

    logger.info(f"wrote {written} entries to {output}; {failed} failures ({failed / max(1, total) * 100:.2f}%)")


if __name__ == "__main__":
    tyro.cli(build_index)
