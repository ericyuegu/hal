"""Stage 3: read `paths.txt` + `index.jsonl`, write MDS shards + manifest.jsonl.

For each replay path:
  1. parse with peppi-py via `extract_replay`
  2. determine split (train/val/test) deterministically from `replay_uuid`
  3. append to the per-split `MDSWriter`
  4. record a `Stage3Annotation` on the index entry

After all writes complete, the annotated entries are flushed to
`manifest.jsonl`. The manifest is the source of truth at training time for
per-replay metadata (stage, character, slp_version, code, name) — none of
that is duplicated into per-frame columns.

Splits are by `replay_uuid` bucket, not by random shuffle, so they're
reproducible across reruns and additive when paths are added.

paths.txt is self-describing: each line is either an absolute filesystem
path (loose .slp on disk) or `archive://<abs-archive>!<member>` (synthetic
path emitted by build_index --archive / filter_replays). The two can be
mixed freely and multiple archives can appear in one paths.txt — archive
entries are bucketed by archive and each archive is streamed once
sequentially (one producer thread; consumers are the existing mp.Pool).

Usage:
    python -m hal.data.process_replays \\
        --paths-file /path/to/paths.txt \\
        --index /path/to/index.jsonl \\
        --output /path/to/mds \\
        [--workers N] [--train-split 0.98] [--val-split 0.01]
"""

import dataclasses
import multiprocessing as mp
from collections.abc import Iterable
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import tyro
from loguru import logger
from streaming import MDSWriter
from tqdm import tqdm

from hal.data.archive_iter import iter_archive_members
from hal.data.archive_iter import parse_archive_member_path
from hal.data.extract import extract_replay
from hal.data.manifest import ReplayIndexEntry
from hal.data.manifest import Stage3Annotation
from hal.data.manifest import read_jsonl
from hal.data.manifest import replay_uuid_from_path
from hal.data.manifest import write_jsonl
from hal.data.schema import MDS_DTYPE_STR_BY_COLUMN

SHARD_SIZE_LIMIT: int = 1 << 31  # 2 GiB; data is repetitive, compression is 10-20x
_DEFAULT_TMPFS: Path = Path("/dev/shm/hal_process_replays")


_INT32_SIGN_MASK: int = 0x7FFFFFFF
_INT32_RANGE: int = 1 << 31


def bucket_fraction(replay_uuid: int) -> float:
    """Map a signed int32 replay_uuid to a stable fraction in [0, 1).

    Folds the sign bit (top half of the int32 space) onto the bottom half.
    Used by ``_split_for``; readers reconstructing the split from a uuid
    must use this same function (a plain ``uuid % N`` will not agree).
    """
    return (replay_uuid & _INT32_SIGN_MASK) / _INT32_RANGE


def _split_for(replay_uuid: int, train: float, val: float) -> str:
    """Deterministic bucket from a signed int32 replay_uuid.

    Same path always lands in the same split; resilient to reordering of
    paths.txt and to incremental adds.
    """
    frac = bucket_fraction(replay_uuid)
    if frac < train:
        return "train"
    if frac < train + val:
        return "val"
    return "test"


def _process_one(args: tuple[str, str | None]) -> tuple[str, dict[str, np.ndarray] | None]:
    """Worker: parse one replay's per-frame ndarrays.

    `path` is the file peppi-py opens; `synthetic_path`, when set, is the
    archive synthetic path that gets returned to the main process for
    manifest lookup. The on-disk `path` is unlinked on the way out so
    archive workers don't leak tmpfs slots.
    """
    path, synthetic_path = args
    try:
        sample = extract_replay(path)
    except Exception as e:
        logger.debug(f"extract_replay raised on {path}: {e}")
        sample = None
    if synthetic_path is not None:
        Path(path).unlink(missing_ok=True)
    return path if synthetic_path is None else synthetic_path, sample


def _index_by_path(index: Path) -> dict[str, ReplayIndexEntry]:
    by_path: dict[str, ReplayIndexEntry] = {}
    for entry in read_jsonl(index):
        by_path[entry.path] = entry
    return by_path


def _read_paths(paths_file: Path) -> list[str]:
    return [line.strip() for line in paths_file.read_text().splitlines() if line.strip()]


def _open_writers(output: Path, splits: Iterable[str]) -> dict[str, MDSWriter]:
    return {
        split: MDSWriter(
            out=str(output / split),
            columns=MDS_DTYPE_STR_BY_COLUMN,
            compression="zstd",
            size_limit=SHARD_SIZE_LIMIT,
            exist_ok=False,
        )
        for split in splits
    }


def _bucket_paths(paths: list[str]) -> tuple[list[str], dict[Path, list[str]]]:
    """Split paths.txt into filesystem paths and per-archive member lists.

    Filesystem paths are resolved here so they match the form written into
    ``ReplayIndexEntry.path`` by ``extract_index_entry`` (which uses
    ``Path.resolve()``). A user-edited paths.txt with symlinks or ``..``
    segments would otherwise silently miss the index.

    Archive ordering is first-appearance in paths.txt; member ordering within
    an archive is preserved (currently unused — iter_archive_members yields
    in decompression order — but kept stable for reproducibility).
    """
    fs_paths: list[str] = []
    members_by_archive: dict[Path, list[str]] = {}
    for p in paths:
        parsed = parse_archive_member_path(p)
        if parsed is None:
            fs_paths.append(str(Path(p).resolve()))
            continue
        archive, member = parsed
        archive = archive.resolve()
        members_by_archive.setdefault(archive, []).append(member)
    return fs_paths, members_by_archive


def _build_work(
    members_by_archive: dict[Path, list[str]],
    fs_paths: list[str],
    *,
    tmpfs_root: Path,
    queue_size: int,
) -> Iterator[tuple[str, str | None]]:
    """Yield (peppi_input_path, synthetic_path | None) for every bucketed entry.

    Filesystem entries pass through with synthetic=None. Archive entries are
    grouped by archive and each archive is streamed once (one producer thread
    per archive at a time; archives processed sequentially).
    """
    for p in fs_paths:
        yield p, None
    for archive, members in members_by_archive.items():
        for synthetic, tmpfs_path in iter_archive_members(
            archive,
            tmpfs_root=tmpfs_root,
            filter_paths=set(members),
            queue_size=queue_size,
        ):
            yield str(tmpfs_path), synthetic


def process_replays(
    paths_file: Path,
    index: Path,
    output: Path,
    *,
    train_split: float = 0.98,
    val_split: float = 0.01,
    workers: int = max(1, (mp.cpu_count() or 2) - 1),
    tmpfs_root: Path = _DEFAULT_TMPFS,
    queue_size: int = 64,
) -> None:
    test_split = 1.0 - train_split - val_split
    if not (0.0 <= test_split <= 1.0):
        raise ValueError(f"train+val must be in [0, 1]; got train={train_split} val={val_split}")
    if not paths_file.exists():
        raise FileNotFoundError(f"--paths {paths_file} not found")
    if not index.exists():
        raise FileNotFoundError(f"--index {index} not found")
    # Per-split MDSWriter raises with exist_ok=False if its output dir already
    # exists; check the manifest sidecar here so we fail before opening writers.
    manifest_path = output / "manifest.jsonl"
    if manifest_path.exists():
        raise FileExistsError(f"{manifest_path} already exists; choose a fresh --output")

    paths = _read_paths(paths_file)
    fs_paths, members_by_archive = _bucket_paths(paths)

    # Fail loud and early on missing archives — we'd otherwise crash partway
    # through Stage 3 with shards already written and unrecoverable.
    missing = [a for a in members_by_archive if not a.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} archive(s) referenced by paths.txt not found on disk: {missing}")

    output.mkdir(parents=True, exist_ok=True)
    by_path = _index_by_path(index)
    logger.info(
        f"index: {len(by_path)}  paths: {len(paths)} "
        f"({len(fs_paths)} filesystem, {len(members_by_archive)} archive(s))  workers: {workers}"
    )

    work_iter = _build_work(
        members_by_archive,
        fs_paths,
        tmpfs_root=tmpfs_root,
        queue_size=queue_size,
    )

    splits = ("train", "val", "test")
    writers = _open_writers(output, splits)
    rows_written: dict[str, int] = dict.fromkeys(splits, 0)
    annotated: list[ReplayIndexEntry] = []
    failed = 0

    ctx = mp.get_context("fork")
    try:
        with ctx.Pool(workers) as pool:
            for path, sample in tqdm(
                pool.imap_unordered(_process_one, work_iter),
                total=len(paths),
                desc="processing",
                unit="slp",
            ):
                if sample is None:
                    failed += 1
                    continue
                entry = by_path.get(path)
                if entry is None:
                    logger.debug(f"path {path} not in index; skipping")
                    failed += 1
                    continue

                replay_uuid = replay_uuid_from_path(path)
                split = _split_for(replay_uuid, train_split, val_split)
                writer = writers[split]
                # MDSWriter assigns sample_idx in write order; capture it before writing.
                row_idx = rows_written[split]
                writer.write(sample)
                rows_written[split] += 1

                annotated.append(
                    dataclasses.replace(
                        entry,
                        annotation=Stage3Annotation(
                            replay_uuid=replay_uuid,
                            split=split,
                            mds_row_idx=row_idx,
                            frame_count_actual=int(sample["frame"].shape[0]),
                        ),
                    )
                )
    finally:
        for w in writers.values():
            w.finish()

    write_jsonl(manifest_path, annotated)
    logger.info(
        "wrote {tr} train, {v} val, {te} test ({f} failures); manifest -> {m}",
        tr=rows_written["train"],
        v=rows_written["val"],
        te=rows_written["test"],
        f=failed,
        m=manifest_path,
    )


if __name__ == "__main__":
    tyro.cli(process_replays)
