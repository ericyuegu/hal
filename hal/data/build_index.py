"""Stage 1: walk a replay tree and build `index.jsonl`.

One `ReplayIndexEntry` per .slp file, populated from peppi's start/end/metadata
blocks (no frame iteration). Parallelized via `mp.Pool`.

Usage:
    python -m hal.data.build_index \\
        --root /path/to/replays \\
        --output /path/to/index.jsonl \\
        [--incremental] [--no-compute-sha1] [--workers N]

Incremental mode reads the existing `index.jsonl`, collects every path already
recorded, and only indexes new files. Failed parses are logged and counted but
never halt the run.
"""

import multiprocessing as mp
import sys
from pathlib import Path

import tyro
from loguru import logger
from tqdm import tqdm

from hal.data.manifest import ReplayIndexEntry
from hal.data.manifest import extract_index_entry
from hal.data.manifest import read_jsonl
from hal.data.manifest import write_jsonl


def _index_one(args: tuple[Path, bool]) -> ReplayIndexEntry | None:
    path, compute_sha1 = args
    try:
        return extract_index_entry(path, compute_sha1=compute_sha1)
    except Exception as e:
        logger.debug(f"unhandled error indexing {path}: {e}")
        return None


def _existing_paths(index_path: Path) -> set[str]:
    if not index_path.exists():
        return set()
    return {entry.path for entry in read_jsonl(index_path)}


def build_index(
    root: Path,
    output: Path,
    *,
    incremental: bool = False,
    compute_sha1: bool = True,
    workers: int = max(1, (mp.cpu_count() or 2) - 1),
) -> None:
    if not root.is_dir():
        raise NotADirectoryError(f"--root must be a directory; got {root}")

    seen: set[str] = _existing_paths(output) if incremental else set()
    if incremental:
        logger.info(f"incremental: {len(seen)} paths already in {output}")
    elif output.exists():
        raise FileExistsError(f"{output} already exists; pass --incremental to append, or delete it first")

    all_paths = sorted(root.rglob("*.slp"))
    new_paths = [p for p in all_paths if str(p.resolve()) not in seen]
    logger.info(f"found {len(all_paths)} slps under {root}; {len(new_paths)} to index")

    if not new_paths:
        return

    output.parent.mkdir(parents=True, exist_ok=True)

    work = [(p, compute_sha1) for p in new_paths]
    written = 0
    failed = 0
    # Append directly so a crash mid-run leaves a partial-but-valid jsonl that
    # --incremental can resume from. write_jsonl handles per-line flushing.
    batch: list[ReplayIndexEntry] = []
    BATCH = 256

    # Use fork explicitly: py3.14 defaults to forkserver on Linux, which
    # re-imports the caller's module in each worker — that breaks when the
    # caller is a VSCode interactive cell or any script that runs work at
    # import time. Fork is safe here because workers run a pure function.
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        results = pool.imap_unordered(_index_one, work, chunksize=8)
        for entry in tqdm(results, total=len(work), desc="indexing", unit="slp"):
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

    logger.info(f"wrote {written} entries to {output}; {failed} failures ({failed / max(1, len(work)) * 100:.2f}%)")


if __name__ == "__main__":
    tyro.cli(build_index)
    sys.exit(0)
