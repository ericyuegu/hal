"""Build the compact replay dataset used by the base action model."""

import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import tyro
from loguru import logger
from streaming import MDSWriter
from tqdm import tqdm

from hal.data.index import read_jsonl
from hal.data.mds import open_shard
from hal.data.mds import read_shard_index
from hal.data.policy_schema import POLICY_MDS_COLUMNS
from hal.data.policy_schema import POLICY_SCHEMA_VERSION
from hal.data.policy_schema import assert_policy_replay_equal
from hal.data.policy_schema import encode_policy_replay
from hal.data.policy_schema import policy_replay_identity

DEFAULT_SCRATCH = Path("/dev/shm/hal_policy_projection")
DEFAULT_SHARD_SIZE = 256 * 2**20


def load_replay_ids_by_split(src: Path) -> dict[str, list[str]]:
    rows: dict[str, dict[int, str]] = {}
    identity_paths: dict[str, str] = {}
    for entry in read_jsonl(src / "manifest.jsonl", verify_schema_version=False):
        if entry.annotation is None:
            continue
        annotation = entry.annotation
        split = rows.setdefault(annotation.split, {})
        if annotation.mds_row_idx in split:
            raise ValueError(f"duplicate {annotation.split} row {annotation.mds_row_idx} in manifest")
        replay_id = policy_replay_identity(entry.path)
        previous_path = identity_paths.get(replay_id)
        if previous_path is not None:
            raise ValueError(
                f"duplicate replay identity {replay_id}: manifest paths {previous_path!r} and {entry.path!r}"
            )
        identity_paths[replay_id] = entry.path
        split[annotation.mds_row_idx] = replay_id
    out = {}
    for name, split in rows.items():
        if sorted(split) != list(range(len(split))):
            raise ValueError(f"manifest rows for {name} are not contiguous")
        out[name] = [split[i] for i in range(len(split))]
    return out


def _write_split(
    src: Path,
    out: Path,
    split: str,
    ids: list[str] | None,
    scratch: Path,
    *,
    allow_row_ids: bool,
    max_rows: int | None,
    shard_size: int,
) -> int:
    shards = read_shard_index(src, split)
    expected = sum(int(info["samples"]) for info in shards)
    if ids is not None and len(ids) != expected:
        raise ValueError(f"{split}: manifest has {len(ids)} rows but MDS has {expected}")
    if ids is None and not allow_row_ids:
        raise FileNotFoundError(f"{src / 'manifest.jsonl'} is required for stable replay IDs")

    limit = expected if max_rows is None else min(expected, max_rows)
    row = 0
    writer = MDSWriter(
        out=str(out / split),
        columns=POLICY_MDS_COLUMNS,
        compression="zstd",
        size_limit=shard_size,
        exist_ok=False,
    )
    try:
        with tqdm(total=limit, desc=f"project {split}", unit="replay") as bar:
            for info in shards:
                if row >= limit:
                    break
                with open_shard(src, split, info, scratch) as reader:
                    for shard_row, source in enumerate(reader):
                        if row >= limit:
                            break
                        replay_id = ids[row] if ids is not None else f"{split}:{row}"
                        shard = info["raw_data"]["basename"]
                        where = f"{split} row {row}, {shard} row {shard_row}"
                        try:
                            compact = encode_policy_replay(source, replay_id)
                            assert_policy_replay_equal(source, compact, where)
                        except (TypeError, ValueError) as error:
                            raise ValueError(f"{where}: {error}") from error
                        writer.write(compact)
                        row += 1
                        bar.update(1)
    finally:
        writer.finish()
    return row


def project_policy_mds(
    src: Path,
    out: Path,
    *,
    splits: tuple[str, ...] = ("train", "val", "test"),
    scratch: Path = DEFAULT_SCRATCH,
    allow_row_ids: bool = False,
    max_rows: int | None = None,
    shard_size: int = DEFAULT_SHARD_SIZE,
) -> None:
    if out.exists():
        raise FileExistsError(f"{out} already exists")
    if max_rows is not None and max_rows < 1:
        raise ValueError(f"max_rows must be positive, got {max_rows}")
    if shard_size < 1:
        raise ValueError(f"shard_size must be positive, got {shard_size}")
    missing = [split for split in splits if not (src / split / "index.json").is_file()]
    if missing:
        raise FileNotFoundError(f"source is missing splits {missing}")

    ids = load_replay_ids_by_split(src) if (src / "manifest.jsonl").is_file() else {}
    scratch.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent, prefix=f".{out.name}.") as staging_name:
        staging = Path(staging_name)
        counts = {}
        with TemporaryDirectory(dir=scratch) as run_scratch:
            for split in splits:
                counts[split] = _write_split(
                    src,
                    staging,
                    split,
                    ids.get(split),
                    Path(run_scratch),
                    allow_row_ids=allow_row_ids,
                    max_rows=max_rows,
                    shard_size=shard_size,
                )
        stats = src / "stats.json"
        if stats.is_file():
            shutil.copy2(stats, staging / "stats.json")
        metadata = {
            "policy_schema_version": POLICY_SCHEMA_VERSION,
            "source": str(src.resolve()),
            "rows": counts,
            "columns": POLICY_MDS_COLUMNS,
        }
        (staging / "projection.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        if out.exists():
            raise FileExistsError(f"{out} appeared while projection was running")
        staging.rename(out)
    logger.info(f"projected policy MDS: {src} -> {out}; rows={counts}")


if __name__ == "__main__":
    tyro.cli(project_policy_mds)
