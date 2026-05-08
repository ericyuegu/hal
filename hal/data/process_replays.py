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

Usage:
    python -m hal.data.process_replays \\
        --paths /path/to/paths.txt \\
        --index /path/to/index.jsonl \\
        --output /path/to/mds \\
        [--workers N] \\
        [--train-split 0.98] [--val-split 0.01]
"""

import multiprocessing as mp
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import tyro
from loguru import logger
from streaming import MDSWriter
from tqdm import tqdm

from hal.data.extract import extract_replay
from hal.data.manifest import ReplayIndexEntry
from hal.data.manifest import Stage3Annotation
from hal.data.manifest import read_jsonl
from hal.data.manifest import replay_uuid_from_path
from hal.data.manifest import write_jsonl
from hal.data.schema import MDS_DTYPE_STR_BY_COLUMN

SHARD_SIZE_LIMIT: int = 1 << 31  # 2 GiB; data is repetitive, compression is 10-20x


def _split_for(replay_uuid: int, train: float, val: float) -> str:
    """Deterministic bucket from a signed int32 replay_uuid.

    Same path always lands in the same split; resilient to reordering of
    paths.txt and to incremental adds.
    """
    frac = (replay_uuid & 0x7FFFFFFF) / 0x80000000
    if frac < train:
        return "train"
    if frac < train + val:
        return "val"
    return "test"


def _process_one(path: str) -> tuple[str, dict[str, np.ndarray] | None]:
    """Worker: extract one replay's per-frame ndarrays. Returns (path, sample)."""
    try:
        sample = extract_replay(path)
    except Exception as e:
        logger.debug(f"extract_replay raised on {path}: {e}")
        return path, None
    return path, sample


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


def process_replays(
    paths_file: Path,
    index: Path,
    output: Path,
    *,
    train_split: float = 0.98,
    val_split: float = 0.01,
    workers: int = max(1, (mp.cpu_count() or 2) - 1),
) -> None:
    test_split = 1.0 - train_split - val_split
    if not (0.0 <= test_split <= 1.0):
        raise ValueError(f"train+val must be in [0, 1]; got train={train_split} val={val_split}")
    if not paths_file.exists():
        raise FileNotFoundError(f"--paths {paths_file} not found")
    if not index.exists():
        raise FileNotFoundError(f"--index {index} not found")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"--output {output} is non-empty; choose a fresh directory")

    output.mkdir(parents=True, exist_ok=True)
    by_path = _index_by_path(index)
    paths = _read_paths(paths_file)
    logger.info(f"index: {len(by_path)}  paths: {len(paths)}  workers: {workers}")

    splits = ("train", "val", "test")
    writers = _open_writers(output, splits)
    rows_written: dict[str, int] = dict.fromkeys(splits, 0)
    annotated: list[ReplayIndexEntry] = []
    failed = 0

    ctx = mp.get_context("fork")
    try:
        with ctx.Pool(workers) as pool:
            for path, sample in tqdm(
                pool.imap_unordered(_process_one, paths),
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
                writer.write({k: v for k, v in sample.items()})
                rows_written[split] += 1

                annotated.append(
                    ReplayIndexEntry(
                        path=entry.path,
                        slp_version=entry.slp_version,
                        stage=entry.stage,
                        players=entry.players,
                        frame_count=entry.frame_count,
                        timestamp=entry.timestamp,
                        played_on=entry.played_on,
                        outcome=entry.outcome,
                        rank_filename=entry.rank_filename,
                        sha1_partial=entry.sha1_partial,
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

    manifest_path = output / "manifest.jsonl"
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
    sys.exit(0)
