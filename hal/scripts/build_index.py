"""Stage 1: walk a replay tree (or a .7z archive) and build `index.jsonl`.

One `ReplayIndexEntry` per .slp. By default a single peppi pass extracts
start/end/metadata AND per-replay aggregate stats (damage/stocks/inputs/SDs)
— see `hal.data.replay_stats`. Pass `--no-with-stats` for the metadata-only
fast path (~5-10x faster, no `entry.stats`).

Usage:
    # Loose .slp files on disk
    python -m hal.scripts.build_index --root /path/to/replays --output index.jsonl

    # .slp members streamed directly from a solid .7z archive (no extraction)
    python -m hal.scripts.build_index --archive /path/to/archive.7z --output index.jsonl

`--root` and `--archive` are mutually exclusive; exactly one is required.
Archive mode materializes each member to a tmpfs file (default `/dev/shm`)
just long enough for peppi-py to read it, then unlinks. Synthetic paths
(`archive://<abs-archive>!<member>`) are recorded in `entry.path`.

Incremental mode reads the existing `index.jsonl`, collects every path already
recorded, and only indexes new entries. Failed parses are logged and counted
but never halt the run.
"""

import dataclasses
import faulthandler
import functools
import multiprocessing as mp
import signal
from pathlib import Path

import tyro
from loguru import logger
from tqdm import tqdm

from hal.data.archive import ReplayWork
from hal.data.archive import archive_member_path
from hal.data.archive import iter_replay_work
from hal.data.archive import list_archive_slps
from hal.data.index import ReplayIndexEntry
from hal.data.index import extract_index_entry
from hal.data.index import read_jsonl
from hal.data.index import write_jsonl
from hal.paths import repo_relative

_DEFAULT_TMPFS: Path = Path("/dev/shm/hal_build_index")


def _worker_init() -> None:
    # Ensures any genuine segfault in peppi-py prints a C-level traceback
    # to stderr before the worker dies, instead of vanishing silently.
    faulthandler.enable()
    # Ignore SIGINT in workers: the parent handles ctrl-c by terminating
    # the pool. Without this, every worker dumps its own KeyboardInterrupt
    # traceback before dying.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _index_one(item: ReplayWork, *, compute_sha1: bool, with_stats: bool) -> ReplayIndexEntry | None:
    name_hint = item.manifest_key if item.unlink_after else None
    try:
        entry = extract_index_entry(
            item.open_path,
            compute_sha1=compute_sha1,
            name_hint=name_hint,
            with_stats=with_stats,
        )
    except KeyboardInterrupt, SystemExit:
        raise
    except BaseException as e:
        # peppi-py is Rust/pyo3; panics surface as PanicException, which
        # subclasses BaseException, not Exception. A bare `except Exception`
        # lets one corrupt .slp kill the worker and trip BrokenProcessPool,
        # which takes the whole job (and the parent shell) down.
        logger.warning(f"unhandled error indexing {item.manifest_key}: {e!r}")
        entry = None
    finally:
        if item.unlink_after:
            item.open_path.unlink(missing_ok=True)
    if entry is not None:
        entry = dataclasses.replace(entry, path=item.manifest_key)
    return entry


def _existing_paths(index_path: Path) -> set[str]:
    if not index_path.exists():
        return set()
    return {entry.path for entry in read_jsonl(index_path)}


def _resolve_fs(root: Path, seen: set[str]) -> list[tuple[Path, str]]:
    all_paths = sorted(root.rglob("*.slp"))
    pairs = [(p, str(repo_relative(p))) for p in all_paths]
    new = [(p, key) for p, key in pairs if key not in seen]
    logger.info(f"found {len(all_paths)} slps under {root}; {len(new)} to index")
    return new


def _resolve_archive(archive: Path, seen: set[str]) -> list[str]:
    members = list_archive_slps(archive)
    new_members = [m for m in members if archive_member_path(archive, m) not in seen]
    logger.info(f"archive {archive.name}: {len(members)} slps, {len(new_members)} to index")
    return new_members


def build_index(
    output: Path,
    *,
    root: Path | None = None,
    archive: Path | None = None,
    incremental: bool = False,
    compute_sha1: bool = True,
    with_stats: bool = True,
    workers: int = max(1, (mp.cpu_count() or 2) - 1),
    tmpfs_root: Path = _DEFAULT_TMPFS,
    queue_size: int = 64,
) -> None:
    """Walk replays into `index.jsonl`.

    Default reads each .slp with `skip_frames=False` and computes per-replay
    aggregate stats (damage/stocks/inputs/SDs) via `hal.data.replay_stats`.
    Pass `--no-with-stats` for the metadata-only fast path (~5-10x faster,
    no `entry.stats`); rebuild if you later want stats on existing entries.
    """
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

    fs_paths: list[tuple[Path, str]] = []
    archive_members: dict[Path, list[str]] = {}
    if archive is not None:
        new_members = _resolve_archive(archive, seen)
        if new_members:
            archive_members[archive] = new_members
        total = len(new_members)
    else:
        assert root is not None  # narrowed by the mutual-exclusion check above
        fs_paths = _resolve_fs(root, seen)
        total = len(fs_paths)

    if total == 0:
        return

    work_iter = iter_replay_work(
        fs_paths=fs_paths,
        archive_members=archive_members,
        tmpfs_root=tmpfs_root,
        queue_size=queue_size,
    )

    worker = functools.partial(_index_one, compute_sha1=compute_sha1, with_stats=with_stats)

    written = 0
    failed = 0
    batch: list[ReplayIndexEntry] = []
    BATCH = 256

    # Use fork explicitly: py3.14 defaults to forkserver on Linux, which
    # re-imports the caller's module in each worker — that breaks when the
    # caller is a VSCode interactive cell or any script that runs work at
    # import time. Fork is safe here because workers run a pure function.
    ctx = mp.get_context("fork")
    interrupted = False
    with ctx.Pool(workers, initializer=_worker_init) as pool:
        results = pool.imap_unordered(worker, work_iter, chunksize=8)
        try:
            for entry in tqdm(results, total=total, desc="indexing", unit="slp"):
                if entry is None:
                    failed += 1
                    continue
                batch.append(entry)
                if len(batch) >= BATCH:
                    write_jsonl(output, batch, append=True)
                    written += len(batch)
                    batch.clear()
        except KeyboardInterrupt:
            # Stop the pool now so workers don't keep feeding the queue while
            # we drain. Closing the result iterator triggers the work_iter
            # generator's finally (which drains the archive producer thread).
            interrupted = True
            logger.warning("interrupted; terminating workers and draining producer")
            pool.terminate()
        finally:
            if batch and not interrupted:
                write_jsonl(output, batch, append=True)
                written += len(batch)
            # Close the work iterator explicitly so generator finally-blocks
            # (e.g. iter_archive_members draining its producer thread) run
            # now rather than whenever GC happens to collect them.
            work_iter.close()
            # On the happy path the pool is still accepting tasks; close() it
            # so join() doesn't raise "Pool is still running". On the
            # interrupted path terminate() was already called above.
            if not interrupted:
                pool.close()
            pool.join()

    if interrupted:
        logger.info(f"interrupted: wrote {written} entries to {output} before ctrl-c; {failed} failures so far")
        raise SystemExit(130)
    logger.info(
        f"wrote {written} entries to {output}; {failed} failures "
        f"({failed / max(1, total) * 100:.2f}%); with_stats={with_stats}"
    )


if __name__ == "__main__":
    tyro.cli(build_index)
