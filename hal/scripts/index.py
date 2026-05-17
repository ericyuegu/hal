"""Stage 1: walk a replay tree (or a .7z archive) and build `index.jsonl`.

One `ReplayIndexEntry` per .slp, populated from peppi's start/end/metadata
blocks (no frame iteration). Parallelized via `mp.Pool`.

Usage:
    # Loose .slp files on disk
    python -m hal.scripts.index --root /path/to/replays --output index.jsonl

    # .slp members streamed directly from a solid .7z archive (no extraction)
    python -m hal.scripts.index --archive /path/to/archive.7z --output index.jsonl

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
import multiprocessing as mp
import signal
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import py7zr
import tyro
from loguru import logger
from tqdm import tqdm

from hal.data.archive import archive_member_path
from hal.data.archive import iter_archive_members
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


@dataclass(frozen=True, slots=True)
class _WorkItem:
    """One unit of work for `_index_one`.

    `synthetic_path`, when set, replaces the on-disk path written into the
    entry — used so the index records `archive://...!member` instead of the
    transient tmpfs path. The tmpfs file is unlinked on the way out
    (success or failure) so workers don't leak ring-buffer slots.
    """

    path: Path
    compute_sha1: bool
    with_stats: bool
    synthetic_path: str | None


def _index_one(item: _WorkItem) -> ReplayIndexEntry | None:
    label = item.synthetic_path or str(item.path)
    try:
        entry = extract_index_entry(
            item.path,
            compute_sha1=item.compute_sha1,
            name_hint=item.synthetic_path,
            with_stats=item.with_stats,
        )
    except KeyboardInterrupt, SystemExit:
        raise
    except BaseException as e:
        # peppi-py is Rust/pyo3; panics surface as PanicException, which
        # subclasses BaseException, not Exception. A bare `except Exception`
        # lets one corrupt .slp kill the worker and trip BrokenProcessPool,
        # which takes the whole job (and the parent shell) down.
        logger.warning(f"unhandled error indexing {label}: {e!r}")
        entry = None
    finally:
        if item.synthetic_path is not None:
            item.path.unlink(missing_ok=True)
    if entry is not None and item.synthetic_path is not None:
        entry = dataclasses.replace(entry, path=item.synthetic_path)
    return entry


def _existing_paths(index_path: Path) -> set[str]:
    if not index_path.exists():
        return set()
    return {entry.path for entry in read_jsonl(index_path)}


def _filesystem_work(
    root: Path, seen: set[str], *, compute_sha1: bool, with_stats: bool
) -> tuple[list[_WorkItem], int]:
    all_paths = sorted(root.rglob("*.slp"))
    new = [p for p in all_paths if str(repo_relative(p)) not in seen]
    logger.info(f"found {len(all_paths)} slps under {root}; {len(new)} to index")
    items = [_WorkItem(path=p, compute_sha1=compute_sha1, with_stats=with_stats, synthetic_path=None) for p in new]
    return items, len(new)


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
    with_stats: bool,
) -> tuple[Iterator[_WorkItem], int]:
    members = _list_archive_slps(archive)
    new_members = [m for m in members if archive_member_path(archive, m) not in seen]
    logger.info(f"archive {archive.name}: {len(members)} slps, {len(new_members)} to index")
    new_set = set(new_members)

    def _gen() -> Iterator[_WorkItem]:
        for synthetic, tmpfs_path in iter_archive_members(
            archive,
            tmpfs_root=tmpfs_root,
            filter_paths=new_set,
            queue_size=queue_size,
        ):
            yield _WorkItem(
                path=tmpfs_path,
                compute_sha1=compute_sha1,
                with_stats=with_stats,
                synthetic_path=synthetic,
            )

    return _gen(), len(new_members)


def build_index(
    output: Path,
    *,
    root: Path | None = None,
    archive: Path | None = None,
    incremental: bool = False,
    compute_sha1: bool = True,
    with_stats: bool = False,
    workers: int = max(1, (mp.cpu_count() or 2) - 1),
    tmpfs_root: Path = _DEFAULT_TMPFS,
    queue_size: int = 64,
) -> None:
    """Walk replays into `index.jsonl`.

    `with_stats=True` switches peppi to `skip_frames=False` and computes
    per-replay aggregate stats (damage/stocks/inputs) per entry — see
    `hal.data.replay_stats`. ~5-10x slower than the default metadata-only
    pass; rebuild if you want stats on already-indexed entries.
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

    if archive is not None:
        work_iter, total = _archive_work(
            archive,
            seen,
            tmpfs_root=tmpfs_root,
            queue_size=queue_size,
            compute_sha1=compute_sha1,
            with_stats=with_stats,
        )
    else:
        assert root is not None  # narrowed by the mutual-exclusion check above
        work_list, total = _filesystem_work(root, seen, compute_sha1=compute_sha1, with_stats=with_stats)
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
    interrupted = False
    with ctx.Pool(workers, initializer=_worker_init) as pool:
        results = pool.imap_unordered(_index_one, work_iter, chunksize=8)
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
            close = getattr(work_iter, "close", None)
            if close is not None:
                close()
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
