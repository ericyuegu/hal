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
    python -m hal.scripts.materialize \\
        --paths-file /path/to/paths.txt \\
        --index /path/to/index.jsonl \\
        --output /path/to/mds \\
        [--workers N] [--train-split 0.98] [--val-split 0.01]
"""

import dataclasses
import multiprocessing as mp
import os
import urllib.parse
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from loguru import logger
from streaming import MDSWriter
from tqdm import tqdm

from hal.data.archive import ReplayWork
from hal.data.archive import iter_replay_work
from hal.data.archive import parse_archive_member_path
from hal.data.extract import extract_replay
from hal.data.index import ReplayIndexEntry
from hal.data.index import Split
from hal.data.index import Stage3Annotation
from hal.data.index import read_jsonl
from hal.data.index import replay_uuid_from_path
from hal.data.index import write_jsonl
from hal.data.schema import MDS_DTYPE_STR_BY_COLUMN
from hal.data.schema import MDS_PER_FRAME_DTYPES
from hal.data.schema import SCHEMA_VERSION
from hal.data.stats import StatsAccumulator
from hal.data.stats import dump_sufficient_stats
from hal.data.stats import float_feature_names
from hal.paths import REPO_DIR
from hal.paths import repo_relative

SHARD_SIZE_LIMIT: int = 1 << 31  # 2 GiB; data is repetitive, compression is 10-20x
_DEFAULT_TMPFS: Path = Path("/dev/shm/hal_process_replays")


_INT32_SIGN_MASK: int = 0x7FFFFFFF
_INT32_RANGE: int = 1 << 31


@dataclass(frozen=True, slots=True)
class ExtractResult:
    """Typed return for `_process_one`; sample is None on parse failure."""

    manifest_key: str
    sample: dict[str, np.ndarray] | None


def bucket_fraction(replay_uuid: int) -> float:
    """Map a signed int32 replay_uuid to a stable fraction in [0, 1).

    ``replay_uuid`` is derived from the replay PATH (md5 of the absolute or
    synthetic path), not the file content. The same .slp copied to two
    locations — e.g. ``archive://X.7z!Game.slp`` vs. ``/tmp/Game.slp`` —
    therefore lands in different splits. Don't mix the on-disk and
    archive-streaming variants of the same corpus in one training run.

    Folds the sign bit (top half of the int32 space) onto the bottom half.
    Readers reconstructing the split from a uuid must use this same function
    (a plain ``uuid % N`` will not agree).
    """
    return (replay_uuid & _INT32_SIGN_MASK) / _INT32_RANGE


def _split_for(replay_uuid: int, train: float, val: float) -> Split:
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


def _process_one(item: ReplayWork) -> ExtractResult:
    """Worker: parse one replay's per-frame ndarrays."""
    try:
        sample = extract_replay(str(item.open_path))
    except KeyboardInterrupt, SystemExit:
        raise
    except BaseException as e:
        # peppi-py is Rust/pyo3; panics surface as PanicException, which
        # subclasses BaseException. A bare `except Exception` lets one corrupt
        # .slp kill the worker and trip BrokenProcessPool.
        logger.debug(f"extract_replay raised on {item.open_path}: {e!r}")
        sample = None
    if item.unlink_after:
        item.open_path.unlink(missing_ok=True)
    return ExtractResult(manifest_key=item.manifest_key, sample=sample)


def _index_by_path(index: Path) -> dict[str, ReplayIndexEntry]:
    by_path: dict[str, ReplayIndexEntry] = {}
    for entry in read_jsonl(index):
        by_path[entry.path] = entry
    return by_path


def _read_paths(paths_file: Path) -> list[str]:
    return [line.strip() for line in paths_file.read_text().splitlines() if line.strip()]


def _is_remote(output: str) -> bool:
    return urllib.parse.urlparse(output).scheme not in ("", "file")


def _join(base: str, name: str) -> str | Path:
    """Append ``name`` to ``base``. Returns a ``Path`` for local outputs and a
    plain string for remote (``s3://``, ...) URIs so each downstream consumer
    (``MDSWriter``, ``fsspec.open``) gets the form it expects."""
    if _is_remote(base):
        return f"{base.rstrip('/')}/{name}"
    return Path(base) / name


def _bridge_streaming_env() -> None:
    """``mosaicml-streaming`` reads its own ``S3_ENDPOINT_URL`` instead of the
    standard ``AWS_ENDPOINT_URL`` that botocore/s3fs use. Bridge so callers
    only need to set the idiomatic one. Idempotent; an explicit
    ``S3_ENDPOINT_URL`` wins."""
    endpoint = os.environ.get("AWS_ENDPOINT_URL")
    if not endpoint:
        raise RuntimeError(
            "remote --output requires AWS_ENDPOINT_URL to be set "
            "(used by s3fs + bridged to S3_ENDPOINT_URL for mosaicml-streaming)"
        )
    os.environ.setdefault("S3_ENDPOINT_URL", endpoint)


def _open_writers(output: str, splits: Iterable[str]) -> dict[str, MDSWriter]:
    return {
        split: MDSWriter(
            out=str(_join(output, split)),
            columns=MDS_DTYPE_STR_BY_COLUMN,
            compression="zstd",
            size_limit=SHARD_SIZE_LIMIT,
            exist_ok=False,
        )
        for split in splits
    }


def _bucket_paths(paths: list[str]) -> tuple[list[tuple[Path, str]], dict[Path, list[str]]]:
    """Split paths.txt into (open_path, manifest_key) pairs and per-archive member lists.

    Uses ``os.path.abspath`` (not ``resolve``) so symlinked-in-place fixtures
    keep their declared path; this matches ``repo_relative`` and ensures the
    manifest_key reconstructed downstream matches ``entry.path`` in the index.

    Member ordering within an archive is preserved for reproducibility, even
    though ``iter_archive_members`` currently yields in decompression order.
    """
    fs_pairs: list[tuple[Path, str]] = []
    members_by_archive: dict[Path, list[str]] = {}
    for p in paths:
        parsed = parse_archive_member_path(p)
        if parsed is None:
            abs_path = Path(os.path.abspath(p))
            manifest_key = str(repo_relative(abs_path))
            fs_pairs.append((abs_path, manifest_key))
            continue
        archive, member = parsed
        if not archive.is_absolute():
            archive = Path(REPO_DIR) / archive
        members_by_archive.setdefault(archive, []).append(member)
    return fs_pairs, members_by_archive


def process_replays(
    paths_file: Path,
    index: Path,
    output: str,
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

    remote = _is_remote(output)
    if remote:
        _bridge_streaming_env()
    manifest_path = _join(output, "manifest.jsonl")
    # Per-split MDSWriter raises with exist_ok=False if its output dir already
    # exists; check the manifest sidecar here so we fail before opening writers.
    # Skipped for remote: object stores have no cheap directory-exists check
    # and the MDSWriter collision guard still fires per-split.
    if not remote and isinstance(manifest_path, Path) and manifest_path.exists():
        raise FileExistsError(f"{manifest_path} already exists; choose a fresh --output")

    paths = _read_paths(paths_file)
    fs_pairs, members_by_archive = _bucket_paths(paths)

    # Fail loud and early on missing archives — we'd otherwise crash partway
    # through Stage 3 with shards already written and unrecoverable.
    missing = [a for a in members_by_archive if not a.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} archive(s) referenced by paths.txt not found on disk: {missing}")

    if not remote:
        Path(output).mkdir(parents=True, exist_ok=True)
    by_path = _index_by_path(index)
    logger.info(
        f"index: {len(by_path)}  paths: {len(paths)} "
        f"({len(fs_pairs)} filesystem, {len(members_by_archive)} archive(s))  workers: {workers}"
    )

    work_iter = iter_replay_work(
        fs_paths=fs_pairs,
        archive_members=members_by_archive,
        tmpfs_root=tmpfs_root,
        queue_size=queue_size,
    )

    splits = ("train", "val", "test")
    writers = _open_writers(output, splits)
    rows_written: dict[str, int] = dict.fromkeys(splits, 0)
    annotated: list[ReplayIndexEntry] = []
    failed = 0

    # Train-split normalization stats: feed every continuous column from each
    # written train sample into a Welford accumulator. Categorical columns
    # (action ids, button bits, stocks) are skipped by dtype.
    stat_features = float_feature_names(MDS_PER_FRAME_DTYPES)
    stats = StatsAccumulator(stat_features)

    ctx = mp.get_context("fork")
    try:
        with ctx.Pool(workers) as pool:
            for result in tqdm(
                pool.imap_unordered(_process_one, work_iter),
                total=len(paths),
                desc="processing",
                unit="slp",
            ):
                if result.sample is None:
                    failed += 1
                    continue
                entry = by_path.get(result.manifest_key)
                if entry is None:
                    logger.debug(f"path {result.manifest_key} not in index; skipping")
                    failed += 1
                    continue

                replay_uuid = replay_uuid_from_path(result.manifest_key)
                split = _split_for(replay_uuid, train_split, val_split)
                writer = writers[split]
                # MDSWriter assigns sample_idx in write order; capture it before writing.
                row_idx = rows_written[split]
                writer.write(result.sample)
                rows_written[split] += 1

                if split == "train":
                    for name in stat_features:
                        stats.update(name, result.sample[name])

                annotated.append(
                    dataclasses.replace(
                        entry,
                        annotation=Stage3Annotation(
                            replay_uuid=replay_uuid,
                            split=split,
                            mds_row_idx=row_idx,
                            frame_count_actual=int(result.sample["frame"].shape[0]),
                            schema_version=SCHEMA_VERSION,
                        ),
                    )
                )
    finally:
        # Close the work iterator explicitly so `iter_archive_members`'
        # finally-block (drain producer, release sem slots) runs deterministically
        # rather than whenever GC happens to collect the generator.
        work_iter.close()
        for w in writers.values():
            w.finish()

    write_jsonl(manifest_path, annotated)

    stats_path = _join(output, "stats.json")
    dump_sufficient_stats(
        stats_path,
        stats.to_sufficient(),
        split="train",
        mds_schema_version=SCHEMA_VERSION,
    )

    logger.info(
        "wrote {tr} train, {v} val, {te} test ({f} failures); manifest -> {m}; stats -> {s}",
        tr=rows_written["train"],
        v=rows_written["val"],
        te=rows_written["test"],
        f=failed,
        m=manifest_path,
        s=stats_path,
    )


if __name__ == "__main__":
    tyro.cli(process_replays)
