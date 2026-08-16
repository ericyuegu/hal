"""Project a local canonical-v7 MDS dataset directly into policy-world MDS."""

import json
import os
import shutil
import tempfile
import urllib.parse
from pathlib import Path

import fsspec
import tyro
from loguru import logger
from tqdm import tqdm

from hal.data.bounded_writer import BoundedMDSWriter
from hal.data.bounded_writer import RcloneMDSWriter
from hal.data.bounded_writer import rclone_copyto
from hal.data.feature_stats import StatsAccumulator
from hal.data.feature_stats import dump_sufficient_stats
from hal.data.mds import open_shard
from hal.data.mds import read_shard_index
from hal.data.policy_world_schema import POLICY_WORLD_FLOAT_COLUMNS
from hal.data.policy_world_schema import POLICY_WORLD_MDS_COLUMNS
from hal.data.policy_world_schema import POLICY_WORLD_SCHEMA_VERSION
from hal.data.policy_world_schema import assert_policy_world_replay_equal
from hal.data.policy_world_schema import encode_policy_world_replay
from hal.data.schema import SCHEMA_VERSION
from hal.scripts.project_policy_mds import _replay_ids

DEFAULT_SCRATCH = Path("/dev/shm/hal_policy_world_projection")
DEFAULT_SHARD_SIZE = 256 * 2**20


def _is_remote(path: str) -> bool:
    return urllib.parse.urlparse(path).scheme not in ("", "file")


def _join(base: str, name: str) -> str:
    return f"{base.rstrip('/')}/{name}"


def _is_rclone(path: str) -> bool:
    return path.startswith("r2:")


def _bridge_streaming_env() -> None:
    endpoint = os.environ.get("AWS_ENDPOINT_URL")
    if not endpoint:
        raise RuntimeError("remote output requires AWS_ENDPOINT_URL")
    os.environ.setdefault("S3_ENDPOINT_URL", endpoint)


def _copy_file(source: Path, destination: str) -> None:
    if _is_rclone(destination):
        rclone_copyto(source, destination)
        return
    with source.open("rb") as src, fsspec.open(destination, "wb") as dst:
        shutil.copyfileobj(src, dst, length=1 << 20)


def _put_text(destination: str, text: str, scratch: Path) -> None:
    if not _is_rclone(destination):
        with fsspec.open(destination, "w") as handle:
            handle.write(text)
        return
    fd, name = tempfile.mkstemp(dir=scratch, prefix="sidecar-")
    local = Path(name)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
        rclone_copyto(local, destination)
    finally:
        local.unlink(missing_ok=True)


def _write_split(
    src: Path,
    out: str,
    split: str,
    ids: list[str],
    scratch: Path,
    upload_root: Path | None,
    stats: StatsAccumulator,
    shard_size: int,
    max_rows: int | None,
) -> int:
    shards = read_shard_index(src, split)
    expected = sum(int(info["samples"]) for info in shards)
    if len(ids) != expected:
        raise ValueError(f"{split}: manifest has {len(ids)} rows but MDS has {expected}")
    limit = expected if max_rows is None else min(expected, max_rows)
    destination = _join(out, split)
    writer_out: str | tuple[str, str] = (
        (str(upload_root / split), destination) if upload_root is not None else destination
    )
    common = {
        "columns": POLICY_WORLD_MDS_COLUMNS,
        "compression": "zstd",
        "hashes": ["md5", "sha256"],
        "size_limit": shard_size,
        "max_workers": 2,
        "max_pending_uploads": 2,
        "exist_ok": False,
    }
    writer = (
        RcloneMDSWriter(local=upload_root / split, remote=destination, **common)
        if upload_root is not None and _is_rclone(out)
        else BoundedMDSWriter(out=writer_out, **common)
    )
    row = 0
    try:
        with tqdm(total=limit, desc=f"project world {split}", unit="replay") as bar:
            for info in shards:
                if row >= limit:
                    break
                with open_shard(src, split, info, scratch) as reader:
                    for shard_row, source in enumerate(reader):
                        if row >= limit:
                            break
                        where = f"{split} row {row}, shard row {shard_row}"
                        compact = encode_policy_world_replay(source, ids[row])
                        assert_policy_world_replay_equal(source, compact, where)
                        writer.write(compact)
                        if split == "train":
                            for name in POLICY_WORLD_FLOAT_COLUMNS:
                                stats.update(name, source[name])
                        row += 1
                        bar.update(1)
    finally:
        writer.finish()
    return row


def project_policy_world_mds(
    src: Path,
    out: str,
    *,
    splits: tuple[str, ...] = ("train", "val", "test"),
    scratch: Path = DEFAULT_SCRATCH,
    max_rows: int | None = None,
    shard_size: int = DEFAULT_SHARD_SIZE,
) -> None:
    if max_rows is not None and max_rows < 1:
        raise ValueError(f"max_rows must be positive, got {max_rows}")
    if shard_size < 1:
        raise ValueError(f"shard_size must be positive, got {shard_size}")
    missing = [split for split in splits if not (src / split / "index.json").is_file()]
    if missing:
        raise FileNotFoundError(f"source is missing splits {missing}")
    if not (src / "manifest.jsonl").is_file():
        raise FileNotFoundError(f"{src / 'manifest.jsonl'} is required for stable replay IDs")

    remote = _is_remote(out)
    if remote and not _is_rclone(out):
        _bridge_streaming_env()
    elif Path(out).exists():
        raise FileExistsError(f"{out} already exists")

    ids = _replay_ids(src)
    scratch.mkdir(parents=True, exist_ok=True)
    upload_root = Path(tempfile.mkdtemp(dir=scratch, prefix="mds-upload-")) if remote else None
    stats = StatsAccumulator(POLICY_WORLD_FLOAT_COLUMNS)
    counts: dict[str, int] = {}
    try:
        for split in splits:
            counts[split] = _write_split(
                src,
                out,
                split,
                ids[split],
                scratch,
                upload_root,
                stats,
                shard_size,
                max_rows,
            )
        _copy_file(src / "manifest.jsonl", _join(out, "manifest.jsonl"))
        stats_destination = _join(out, "stats.json")
        if _is_rclone(stats_destination):
            stats_local = scratch / f"stats-{os.getpid()}.json"
            try:
                dump_sufficient_stats(
                    stats_local,
                    stats.to_sufficient(),
                    split="train",
                    mds_schema_version=SCHEMA_VERSION,
                )
                rclone_copyto(stats_local, stats_destination)
            finally:
                stats_local.unlink(missing_ok=True)
        else:
            dump_sufficient_stats(
                stats_destination,
                stats.to_sufficient(),
                split="train",
                mds_schema_version=SCHEMA_VERSION,
            )
        _put_text(_join(out, "failures.materialize.jsonl"), "", scratch)
        metadata = {
            "policy_world_schema_version": POLICY_WORLD_SCHEMA_VERSION,
            "source_schema_version": SCHEMA_VERSION,
            "source": str(src.resolve()),
            "rows": counts,
            "failures": 0,
            "columns": POLICY_WORLD_MDS_COLUMNS,
        }
        _put_text(_join(out, "projection.json"), json.dumps(metadata, indent=2, sort_keys=True) + "\n", scratch)
    finally:
        if upload_root is not None:
            shutil.rmtree(upload_root, ignore_errors=True)
    logger.info(f"projected policy-world MDS: {src} -> {out}; rows={counts}")


if __name__ == "__main__":
    tyro.cli(project_policy_world_mds)
