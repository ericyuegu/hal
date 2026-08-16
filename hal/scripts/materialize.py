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
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import tempfile
import urllib.parse
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import fsspec
import numpy as np
import tyro
from loguru import logger
from streaming import MDSWriter
from tqdm import tqdm

from hal.data.archive import ReplayWork
from hal.data.archive import iter_replay_work
from hal.data.archive import parse_archive_member_path
from hal.data.bounded_writer import BoundedMDSWriter
from hal.data.bounded_writer import RcloneMDSWriter
from hal.data.bounded_writer import rclone_copyto
from hal.data.extract import extract_policy_world_replay
from hal.data.extract import extract_replay
from hal.data.feature_stats import FeatureStatsSufficient
from hal.data.feature_stats import StatsAccumulator
from hal.data.feature_stats import dump_sufficient_stats
from hal.data.feature_stats import float_feature_names
from hal.data.index import ReplayIndexEntry
from hal.data.index import Split
from hal.data.index import Stage3Annotation
from hal.data.index import read_jsonl
from hal.data.index import replay_uuid_from_path
from hal.data.index import write_jsonl
from hal.data.mds import FULL_MDS_SHARD_SIZE_LIMIT
from hal.data.policy_world_schema import POLICY_WORLD_FLOAT_COLUMNS
from hal.data.policy_world_schema import POLICY_WORLD_MDS_COLUMNS
from hal.data.policy_world_schema import POLICY_WORLD_SCHEMA_VERSION
from hal.data.policy_world_schema import assert_policy_world_replay_equal
from hal.data.policy_world_schema import encode_policy_world_replay
from hal.data.schema import MDS_COLUMNS
from hal.data.schema import MDS_PER_FRAME_DTYPES
from hal.data.schema import SCHEMA_VERSION
from hal.data.schema import Rank
from hal.paths import REPO_DIR
from hal.paths import repo_relative

_DEFAULT_TMPFS: Path = Path("/dev/shm/hal_process_replays")


_INT32_SIGN_MASK: int = 0x7FFFFFFF
_INT32_RANGE: int = 1 << 31


@dataclass(frozen=True, slots=True)
class ExtractResult:
    """Typed return for `_process_one`; sample is None on parse failure."""

    manifest_key: str
    sample: dict[str, np.ndarray] | None
    error: str | None = None
    frame_count: int | None = None
    stats: dict[str, FeatureStatsSufficient] | None = None


_WORKER_REPLAY_FORMAT: Literal["full", "policy-world"] = "full"
_WORKER_RANK_OVERRIDE: Rank | None = None
_WORKER_RANK_OVERRIDES: dict[str, tuple[int, int]] = {}


def _worker_init(
    replay_format: Literal["full", "policy-world"],
    rank_override: Rank | None,
    rank_overrides: dict[str, tuple[int, int]],
) -> None:
    global _WORKER_RANK_OVERRIDE, _WORKER_RANK_OVERRIDES, _WORKER_REPLAY_FORMAT
    _WORKER_REPLAY_FORMAT = replay_format
    _WORKER_RANK_OVERRIDE = rank_override
    _WORKER_RANK_OVERRIDES = rank_overrides


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
        extractor = extract_policy_world_replay if _WORKER_REPLAY_FORMAT == "policy-world" else extract_replay
        sample = extractor(str(item.open_path))
        error = "extract_replay returned None" if sample is None else None
        frame_count = None if sample is None else int(sample["frame"].shape[0])
        sufficient = None
        if sample is not None:
            sample["schema_version"] = SCHEMA_VERSION
        if sample is not None and _WORKER_REPLAY_FORMAT == "policy-world":
            ranks = _WORKER_RANK_OVERRIDES.get(item.manifest_key)
            if ranks is None and _WORKER_RANK_OVERRIDE is not None:
                ranks = (int(_WORKER_RANK_OVERRIDE), int(_WORKER_RANK_OVERRIDE))
            if ranks is not None:
                sample["p1_rank"] = np.full(frame_count, ranks[0], dtype=np.uint8)
                sample["p2_rank"] = np.full(frame_count, ranks[1], dtype=np.uint8)
            encoded = encode_policy_world_replay(sample, _replay_identity(item.manifest_key))
            assert_policy_world_replay_equal(sample, encoded, item.manifest_key)
            replay_stats = StatsAccumulator(POLICY_WORLD_FLOAT_COLUMNS)
            for name in POLICY_WORLD_FLOAT_COLUMNS:
                replay_stats.update(name, sample[name])
            sufficient = replay_stats.to_sufficient()
            sample = encoded
    except KeyboardInterrupt, SystemExit:
        raise
    except BaseException as e:
        # peppi-py is Rust/pyo3; panics surface as PanicException, which
        # subclasses BaseException. A bare `except Exception` lets one corrupt
        # .slp kill the worker and trip BrokenProcessPool.
        logger.debug(f"extract_replay raised on {item.open_path}: {e!r}")
        sample = None
        error = repr(e)
        frame_count = None
        sufficient = None
    if item.unlink_after:
        item.open_path.unlink(missing_ok=True)
    return ExtractResult(
        manifest_key=item.manifest_key,
        sample=sample,
        error=error,
        frame_count=frame_count,
        stats=sufficient,
    )


def _index_by_path(index: Path) -> dict[str, ReplayIndexEntry]:
    by_path: dict[str, ReplayIndexEntry] = {}
    for entry in read_jsonl(index):
        by_path[entry.path] = entry
    return by_path


def _read_paths(paths_file: Path) -> list[str]:
    return [line.strip() for line in paths_file.read_text().splitlines() if line.strip()]


def _is_remote(output: str) -> bool:
    return urllib.parse.urlparse(output).scheme not in ("", "file")


def _is_rclone(output: str) -> bool:
    return output.startswith("r2:")


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


def _open_writers(
    output: str,
    splits: Iterable[str],
    *,
    replay_format: Literal["full", "policy-world"],
    tmpfs_root: Path,
) -> tuple[dict[str, MDSWriter], Path | None]:
    remote = _is_remote(output)
    upload_root: Path | None = None
    if remote:
        tmpfs_root.mkdir(parents=True, exist_ok=True)
        upload_root = Path(tempfile.mkdtemp(dir=tmpfs_root, prefix="mds-upload-"))
    columns = MDS_COLUMNS if replay_format == "full" else POLICY_WORLD_MDS_COLUMNS
    size_limit = FULL_MDS_SHARD_SIZE_LIMIT if replay_format == "full" else 256 * 2**20
    writers: dict[str, MDSWriter] = {}
    try:
        for split in splits:
            destination = str(_join(output, split))
            out: str | tuple[str, str] = (
                (str(upload_root / split), destination) if upload_root is not None else destination
            )
            common = {
                "columns": columns,
                "compression": "zstd",
                "hashes": ["md5", "sha256"],
                "size_limit": size_limit,
                "exist_ok": False,
                "max_workers": 2,
                "max_pending_uploads": 2,
            }
            writers[split] = (
                RcloneMDSWriter(local=upload_root / split, remote=destination, **common)
                if upload_root is not None and _is_rclone(output)
                else BoundedMDSWriter(out=out, **common)
            )
    except BaseException:
        for writer in writers.values():
            writer.finish()
        if upload_root is not None:
            shutil.rmtree(upload_root, ignore_errors=True)
        raise
    return writers, upload_root


def _replay_identity(path: str) -> str:
    return hashlib.blake2b(path.encode("utf-8"), digest_size=16, person=b"hal-policy-id-v1").hexdigest()


def _read_rank_overrides(path: Path | None) -> dict[str, tuple[int, int]]:
    if path is None:
        return {}
    out: dict[str, tuple[int, int]] = {}
    with path.open() as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row["path"])
            if key in out:
                raise ValueError(f"{path}:{line_no}: duplicate rank override for {key}")
            ranks = (int(row["p1_rank"]), int(row["p2_rank"]))
            for rank in ranks:
                Rank(rank)
            out[key] = ranks
    return out


def _write_remote_text(path: str | Path, text: str, tmpfs_root: Path) -> None:
    destination = str(path)
    if not _is_rclone(destination):
        with fsspec.open(destination, "w") as handle:
            handle.write(text)
        return
    tmpfs_root.mkdir(parents=True, exist_ok=True)
    fd, local_name = tempfile.mkstemp(dir=tmpfs_root, prefix="sidecar-")
    local = Path(local_name)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
        rclone_copyto(local, destination)
    finally:
        local.unlink(missing_ok=True)


def _write_json(path: str | Path, payload: object, tmpfs_root: Path) -> None:
    _write_remote_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n", tmpfs_root)


def _write_failures(path: str | Path, failures: list[dict[str, object]], tmpfs_root: Path) -> None:
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in failures)
    _write_remote_text(path, text, tmpfs_root)


def _write_manifest(path: str | Path, entries: list[ReplayIndexEntry], tmpfs_root: Path) -> None:
    destination = str(path)
    if not _is_rclone(destination):
        write_jsonl(destination, entries)
        return
    rows = "".join(json.dumps(entry.to_dict()) + "\n" for entry in entries)
    _write_remote_text(destination, rows, tmpfs_root)


def _dump_stats(
    path: str | Path,
    stats: StatsAccumulator,
    tmpfs_root: Path,
) -> None:
    destination = str(path)
    if not _is_rclone(destination):
        dump_sufficient_stats(
            destination,
            stats.to_sufficient(),
            split="train",
            mds_schema_version=SCHEMA_VERSION,
        )
        return
    fd, local_name = tempfile.mkstemp(dir=tmpfs_root, prefix="stats-", suffix=".json")
    os.close(fd)
    local = Path(local_name)
    try:
        dump_sufficient_stats(
            local,
            stats.to_sufficient(),
            split="train",
            mds_schema_version=SCHEMA_VERSION,
        )
        rclone_copyto(local, destination)
    finally:
        local.unlink(missing_ok=True)


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
    replay_format: Literal["full", "policy-world"] = "full",
    rank_override: Rank | None = None,
    rank_overrides: Path | None = None,
) -> None:
    test_split = 1.0 - train_split - val_split
    if not (0.0 <= test_split <= 1.0):
        raise ValueError(f"train+val must be in [0, 1]; got train={train_split} val={val_split}")
    if not paths_file.exists():
        raise FileNotFoundError(f"--paths {paths_file} not found")
    if not index.exists():
        raise FileNotFoundError(f"--index {index} not found")

    remote = _is_remote(output)
    if remote and not _is_rclone(output):
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
    per_replay_ranks = _read_rank_overrides(rank_overrides)
    unknown_overrides = sorted(set(per_replay_ranks) - set(paths))
    if unknown_overrides:
        raise ValueError(f"rank override file contains {len(unknown_overrides)} paths not in paths.txt")
    if replay_format == "full" and (rank_override is not None or per_replay_ranks):
        raise ValueError("rank overrides are supported only for replay_format='policy-world'")

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
    writers, upload_root = _open_writers(
        output,
        splits,
        replay_format=replay_format,
        tmpfs_root=tmpfs_root,
    )
    rows_written: dict[str, int] = dict.fromkeys(splits, 0)
    annotated: list[ReplayIndexEntry] = []
    failed = 0
    failures: list[dict[str, object]] = []

    # Train-split normalization stats: feed every continuous column from each
    # written train sample into a Welford accumulator. Categorical columns
    # (action ids, button bits, stocks) are skipped by dtype.
    stat_features = (
        float_feature_names(MDS_PER_FRAME_DTYPES) if replay_format == "full" else list(POLICY_WORLD_FLOAT_COLUMNS)
    )
    stats = StatsAccumulator(stat_features)

    ctx = mp.get_context("fork")
    try:
        with ctx.Pool(
            workers,
            initializer=_worker_init,
            initargs=(replay_format, rank_override, per_replay_ranks),
        ) as pool:
            for result in tqdm(
                pool.imap_unordered(_process_one, work_iter),
                total=len(paths),
                desc="processing",
                unit="slp",
            ):
                if result.sample is None:
                    failed += 1
                    failures.append(
                        {"path": result.manifest_key, "phase": "materialize", "error": result.error or "unknown"}
                    )
                    continue
                entry = by_path.get(result.manifest_key)
                if entry is None:
                    logger.debug(f"path {result.manifest_key} not in index; skipping")
                    failed += 1
                    failures.append(
                        {"path": result.manifest_key, "phase": "manifest_join", "error": "path not in index"}
                    )
                    continue

                replay_uuid = replay_uuid_from_path(result.manifest_key)
                split = _split_for(replay_uuid, train_split, val_split)
                writer = writers[split]
                # MDSWriter assigns sample_idx in write order; capture it before writing.
                row_idx = rows_written[split]
                if replay_format == "policy-world":
                    writer.write(result.sample)
                else:
                    writer.write(result.sample)
                rows_written[split] += 1

                if split == "train":
                    if replay_format == "policy-world":
                        if result.stats is None:
                            raise RuntimeError(f"{result.manifest_key}: worker returned no policy-world stats")
                        stats = stats.merge(StatsAccumulator.from_sufficient(result.stats))
                    else:
                        for name in stat_features:
                            stats.update(name, result.sample[name])

                annotated.append(
                    dataclasses.replace(
                        entry,
                        annotation=Stage3Annotation(
                            replay_uuid=replay_uuid,
                            split=split,
                            mds_row_idx=row_idx,
                            frame_count_actual=int(result.frame_count),
                            schema_version=SCHEMA_VERSION,
                        ),
                    )
                )
    finally:
        # Close the work iterator explicitly so `iter_archive_members`'
        # finally-block (drain producer, release sem slots) runs deterministically
        # rather than whenever GC happens to collect the generator.
        work_iter.close()
        try:
            for w in writers.values():
                w.finish()
        finally:
            if upload_root is not None:
                shutil.rmtree(upload_root, ignore_errors=True)

    _write_manifest(manifest_path, annotated, tmpfs_root)

    stats_path = _join(output, "stats.json")
    _dump_stats(stats_path, stats, tmpfs_root)

    failures_path = _join(output, "failures.materialize.jsonl")
    _write_failures(failures_path, failures, tmpfs_root)
    if replay_format == "policy-world":
        _write_json(
            _join(output, "projection.json"),
            {
                "policy_world_schema_version": POLICY_WORLD_SCHEMA_VERSION,
                "source_schema_version": SCHEMA_VERSION,
                "rows": rows_written,
                "failures": failed,
                "columns": POLICY_WORLD_MDS_COLUMNS,
            },
            tmpfs_root,
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
