"""Upgrade an MDS dataset from SCHEMA_VERSION 6 to 7: add the ``p{1,2}_rank`` columns.

Rank is already in the dataset's own ``manifest.jsonl`` — each row is the
``ReplayIndexEntry`` the shard row was materialized from, netplay display names
included — so this never re-parses a .slp. It re-encodes the existing shards
with two extra per-frame columns, in the same order, so every
``Stage3Annotation.mds_row_idx`` (and therefore the manifest) stays valid.

The manifest is also the join: annotations carry ``(split, mds_row_idx)``, which
addresses a shard row exactly. Row ``i`` of a split gets the ranks of that
split's annotation ``i``, with the two players ordered by ascending port — the
same p1/p2 order ``extract_replay`` writes.

After writing, the shard sample counts are re-checked and a deterministic random
sample of rows is compared against the source on every shared column.

Usage:
    python -m hal.scripts.upgrade_mds --src <mds-v6> --out <mds-v7>
"""

import dataclasses
import json
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from itertools import groupby
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import tyro
from loguru import logger
from streaming import MDSWriter
from streaming.base.compression import decompress
from streaming.base.format import reader_from_json
from streaming.base.format.base.reader import Reader
from tqdm import tqdm

from hal.data.index import ReplayIndexEntry
from hal.data.index import read_jsonl
from hal.data.index import write_jsonl
from hal.data.schema import MDS_COLUMNS
from hal.data.schema import SCHEMA_VERSION
from hal.data.schema import Rank
from hal.data.schema import rank_from_player_name
from hal.scripts.materialize import SHARD_SIZE_LIMIT
from hal.wire import PLAYER_PREFIXES

# The one version this upgrade reads. A different source version means a
# different column set, so there is nothing to reuse — re-materialize instead.
SRC_SCHEMA_VERSION: int = 6

# Decompressed shards are staged here. tmpfs by default, as in ``materialize``:
# the shards are ~20x compressed, so staging on disk would push the dataset's
# whole decompressed size through it and evict the page cache a concurrent job needs.
DEFAULT_SCRATCH: Path = Path("/dev/shm/hal_upgrade_mds")

# Rows compared against the source after the write, and the cap on how many
# source shards they are drawn from. Each shard costs a full decompress, so
# spreading 1000 rows over every shard would double the job's runtime for no
# extra signal — the rows within a shard already cover the whole column set.
VERIFY_ROWS: int = 1000
VERIFY_SHARDS: int = 16


def _match_ranks(entry: ReplayIndexEntry) -> tuple[Rank, ...]:
    """Ranks of one replay's players, ordered by ascending port.

    ``extract_replay`` maps p1/p2 to the two lowest occupied libmelee ports in
    that order and drops anything that is not 1v1, so a shard row always has
    exactly two players and this order is the column order.
    """
    players = sorted(entry.players, key=lambda p: p.port)
    if len(players) != len(PLAYER_PREFIXES):
        raise ValueError(f"{entry.path}: {len(players)} players in the manifest; expected {len(PLAYER_PREFIXES)}")
    return tuple(rank_from_player_name(p.name) for p in players)


def ranks_by_split_row(manifest: Path, *, require_ranked: bool = True) -> dict[str, list[tuple[Rank, ...]]]:
    """Per-split, per-row player ranks, indexed by ``Stage3Annotation.mds_row_idx``.

    Raises on a manifest row that is unannotated, at the wrong schema version,
    duplicated, or (with ``require_ranked``) whose netplay name is not one of the
    three ranked-ladder names.
    """
    by_split: dict[str, dict[int, tuple[Rank, ...]]] = {}
    for entry in read_jsonl(manifest, verify_schema_version=False):
        if entry.schema_version != SRC_SCHEMA_VERSION:
            raise ValueError(
                f"{manifest}: entry schema_version={entry.schema_version} != {SRC_SCHEMA_VERSION}; "
                "this upgrade only reads a v6 dataset"
            )
        annotation = entry.annotation
        if annotation is None:
            raise ValueError(f"{manifest}: {entry.path} has no Stage3Annotation, so it addresses no shard row")
        ranks = _match_ranks(entry)
        if require_ranked and Rank.UNKNOWN in ranks:
            names = [p.name for p in sorted(entry.players, key=lambda p: p.port)]
            raise ValueError(
                f"{entry.path}: netplay name(s) {names} are not ranked-ladder names, so the tier is UNKNOWN. "
                "Pass --no-require-ranked to upgrade a corpus that is not from the ranked ladder."
            )
        rows = by_split.setdefault(annotation.split, {})
        if annotation.mds_row_idx in rows:
            raise ValueError(f"{manifest}: two entries claim {annotation.split} row {annotation.mds_row_idx}")
        rows[annotation.mds_row_idx] = ranks

    out: dict[str, list[tuple[Rank, ...]]] = {}
    for split, rows in by_split.items():
        if sorted(rows) != list(range(len(rows))):
            raise ValueError(f"{manifest}: {split} mds_row_idx values are not contiguous 0..{len(rows) - 1}")
        out[split] = [rows[i] for i in range(len(rows))]
    return out


def _shard_index(root: Path, split: str) -> list[dict[str, Any]]:
    index = root / split / "index.json"
    if not index.is_file():
        raise FileNotFoundError(f"{index} not found")
    return json.loads(index.read_text())["shards"]


@contextmanager
def _shard_reader(root: Path, split: str, info: dict[str, Any], scratch: Path) -> Iterator[Reader]:
    """Random-access reader over one shard.

    ``MDSReader`` only reads the uncompressed ``.mds``, so a shard that is on
    disk compressed-only is decompressed for the duration and removed after —
    one shard's worth of scratch at a time, never the dataset's. Each open gets
    its own scratch directory: source and destination shards share basenames, so
    a shared one would have them overwrite each other.
    """
    raw_basename = info["raw_data"]["basename"]
    if (root / split / raw_basename).is_file():
        yield reader_from_json(str(root), split, info)
        return

    zip_data = info.get("zip_data")
    if zip_data is None:
        raise FileNotFoundError(f"{root / split / raw_basename} is missing and the shard has no compressed copy")
    with TemporaryDirectory(dir=scratch) as tmp_root:
        tmp_dir = Path(tmp_root) / split
        tmp_dir.mkdir(parents=True)
        (tmp_dir / raw_basename).write_bytes(
            decompress(info["compression"], (root / split / zip_data["basename"]).read_bytes())
        )
        yield reader_from_json(tmp_root, split, info)


def _with_ranks(sample: dict[str, Any], ranks: tuple[Rank, ...]) -> dict[str, Any]:
    """One v6 row -> the v7 row: every column verbatim, plus the broadcast ranks."""
    if sample.get("schema_version") != SRC_SCHEMA_VERSION:
        raise ValueError(f"shard row schema_version={sample.get('schema_version')!r} != {SRC_SCHEMA_VERSION}")
    n_frames = sample["frame"].shape[0]
    out = dict(sample)
    out["schema_version"] = SCHEMA_VERSION
    for prefix, rank in zip(PLAYER_PREFIXES, ranks, strict=True):
        out[f"{prefix}_rank"] = np.full(n_frames, int(rank), dtype=np.uint8)
    return out


def _upgrade_split(src: Path, out: Path, split: str, ranks: list[tuple[Rank, ...]], scratch: Path) -> None:
    shards = _shard_index(src, split)
    expected = sum(int(s["samples"]) for s in shards)
    if expected != len(ranks):
        raise ValueError(f"{split}: {expected} shard rows but {len(ranks)} manifest rows")

    row = 0
    writer = MDSWriter(
        out=str(out / split),
        columns=MDS_COLUMNS,
        compression="zstd",
        size_limit=SHARD_SIZE_LIMIT,
        exist_ok=False,
    )
    try:
        with tqdm(total=expected, desc=f"upgrade {split}", unit="replay") as bar:
            for info in shards:
                with _shard_reader(src, split, info, scratch) as reader:
                    for sample in reader:
                        writer.write(_with_ranks(sample, ranks[row]))
                        row += 1
                        bar.update(1)
    finally:
        writer.finish()

    written = sum(int(s["samples"]) for s in _shard_index(out, split))
    if written != expected:
        raise ValueError(f"{split}: wrote {written} rows, expected {expected}")


def _row_offsets(shards: list[dict[str, Any]]) -> np.ndarray:
    """Global index of each shard's first row, plus the total as a final entry."""
    return np.concatenate(([0], np.cumsum([int(s["samples"]) for s in shards])))


def _columns_equal(a: np.ndarray, b: np.ndarray) -> bool:
    if a.dtype.kind == "f":
        return np.array_equal(a, b, equal_nan=True)
    return np.array_equal(a, b)


def _verify_split(
    src: Path,
    out: Path,
    split: str,
    ranks: list[tuple[Rank, ...]],
    scratch: Path,
    *,
    n_rows: int,
    seed: int,
) -> int:
    """Compare a deterministic random sample of rows against the source.

    Every column the two versions share must be bit-identical, the rank columns
    must hold the manifest's tier, and the row version must be the new one.
    """
    src_shards = _shard_index(src, split)
    out_shards = _shard_index(out, split)
    src_offsets = _row_offsets(src_shards)
    out_offsets = _row_offsets(out_shards)

    rng = np.random.default_rng(seed)
    shard_ids = rng.choice(len(src_shards), size=min(VERIFY_SHARDS, len(src_shards)), replace=False)
    per_shard = max(1, round(n_rows / len(shard_ids)))
    checked = 0
    for shard_id in sorted(int(i) for i in shard_ids):
        samples = int(src_shards[shard_id]["samples"])
        local = sorted(int(i) for i in rng.choice(samples, size=min(per_shard, samples), replace=False))
        rows = [(int(src_offsets[shard_id]) + i, i) for i in local]
        with _shard_reader(src, split, src_shards[shard_id], scratch) as src_reader:
            # Rows ascend, so do the out shards holding them: one decompress each.
            for out_shard_id, group in groupby(
                rows, key=lambda r: int(np.searchsorted(out_offsets, r[0], "right") - 1)
            ):
                with _shard_reader(out, split, out_shards[out_shard_id], scratch) as out_reader:
                    for row, local_idx in group:
                        _verify_row(
                            src_reader[local_idx],
                            out_reader[row - int(out_offsets[out_shard_id])],
                            ranks[row],
                            where=f"{split} row {row}",
                        )
                        checked += 1
    return checked


def _verify_row(src_row: dict[str, Any], out_row: dict[str, Any], ranks: tuple[Rank, ...], *, where: str) -> None:
    if out_row["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"{where}: schema_version={out_row['schema_version']!r} != {SCHEMA_VERSION}")
    added = {f"{prefix}_rank" for prefix in PLAYER_PREFIXES}
    if set(out_row) - set(src_row) != added:
        raise ValueError(f"{where}: added columns {sorted(set(out_row) - set(src_row))} != {sorted(added)}")
    for prefix, rank in zip(PLAYER_PREFIXES, ranks, strict=True):
        column = out_row[f"{prefix}_rank"]
        if column.dtype != np.uint8 or column.shape != src_row["frame"].shape:
            raise ValueError(f"{where}: {prefix}_rank is {column.dtype} {column.shape}, expected uint8 per frame")
        if not np.all(column == int(rank)):
            raise ValueError(f"{where}: {prefix}_rank does not hold the manifest tier {rank.name}")
    for name, value in src_row.items():
        if name == "schema_version":
            continue
        if not _columns_equal(value, out_row[name]):
            raise ValueError(f"{where}: column {name} changed")


def _restamp(entry: ReplayIndexEntry) -> ReplayIndexEntry:
    """The same entry at the new schema version. Row indices are untouched — the
    upgrade preserved order — so only the two version stamps move."""
    annotation = entry.annotation
    if annotation is not None:
        annotation = dataclasses.replace(annotation, schema_version=SCHEMA_VERSION)
    return dataclasses.replace(entry, schema_version=SCHEMA_VERSION, annotation=annotation)


def _upgrade_manifest(src: Path, out: Path, *, batch_rows: int = 10_000) -> None:
    """Rewrite the manifest at the new version, in batches so a 100k-row manifest
    never sits in memory whole."""
    dst = out / "manifest.jsonl"
    batch: list[ReplayIndexEntry] = []
    append = False
    for entry in read_jsonl(src / "manifest.jsonl", verify_schema_version=False):
        batch.append(_restamp(entry))
        if len(batch) >= batch_rows:
            write_jsonl(dst, batch, append=append)
            batch.clear()
            append = True
    write_jsonl(dst, batch, append=append)


def _upgrade_stats(src: Path, out: Path) -> None:
    """Copy the root ``stats.json`` with its version stamp moved.

    The rank columns are uint8, so they are not float features and the
    normalization statistics are unchanged.
    """
    stats = src / "stats.json"
    if not stats.is_file():
        logger.warning(f"{stats} not found; the upgraded dataset has no stats.json")
        return
    payload = json.loads(stats.read_text())
    if payload.get("mds_schema_version") != SRC_SCHEMA_VERSION:
        raise ValueError(f"{stats}: mds_schema_version={payload.get('mds_schema_version')!r} != {SRC_SCHEMA_VERSION}")
    payload["mds_schema_version"] = SCHEMA_VERSION
    (out / "stats.json").write_text(json.dumps(payload))


def _check_scratch_room(src: Path, splits: list[str], scratch: Path) -> None:
    """Refuse to start when the scratch filesystem cannot hold two shards.

    Verification stages a source and a destination shard at the same time. A
    tmpfs scratch that is too small would otherwise fail hours in, with the
    write pass already done.
    """
    largest = max(int(s["raw_data"]["bytes"]) for split in splits for s in _shard_index(src, split))
    free = shutil.disk_usage(scratch).free
    if free < 2 * largest:
        raise OSError(
            f"{scratch} has {free / 1e9:.1f} GB free but staging needs {2 * largest / 1e9:.1f} GB "
            f"(two decompressed shards). Enlarge it or pass --scratch <dir on disk>."
        )


def _log_tier_shares(split: str, ranks: list[tuple[Rank, ...]]) -> None:
    flat = [r for pair in ranks for r in pair]
    shares = {rank.name.lower(): sum(r == rank for r in flat) / len(flat) for rank in Rank}
    logger.info(f"{split}: {len(ranks)} rows; per-slot tier shares {shares}")


def upgrade_mds(
    src: Path,
    out: Path,
    *,
    require_ranked: bool = True,
    verify_rows: int = VERIFY_ROWS,
    seed: int = 0,
    scratch: Path = DEFAULT_SCRATCH,
) -> None:
    """Rewrite a v6 MDS dataset as v7, adding the two broadcast rank columns.

    ``scratch`` stages the decompressed shards the readers work from — two at a
    time at most (one source, one destination, during verification), so it needs
    room for two of the source's largest shards. It defaults to tmpfs because the
    whole job pushes the dataset's decompressed size through it, and on disk that
    traffic evicts the page cache a concurrent job depends on.
    """
    if out.exists():
        raise FileExistsError(f"{out} already exists; choose a fresh --out")
    manifest = src / "manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"{manifest} not found; the upgrade joins ranks through the manifest")

    ranks = ranks_by_split_row(manifest, require_ranked=require_ranked)
    splits = [s for s in ranks if (src / s / "index.json").is_file()]
    missing = sorted(set(ranks) - set(splits))
    if missing:
        raise FileNotFoundError(f"manifest annotates splits {missing} but {src} has no shards for them")

    scratch.mkdir(parents=True, exist_ok=True)
    _check_scratch_room(src, splits, scratch)
    out.mkdir(parents=True)
    with TemporaryDirectory(dir=scratch) as run_scratch:
        for split in splits:
            _log_tier_shares(split, ranks[split])
            _upgrade_split(src, out, split, ranks[split], Path(run_scratch))
        _upgrade_manifest(src, out)
        _upgrade_stats(src, out)
        for split in splits:
            checked = _verify_split(src, out, split, ranks[split], Path(run_scratch), n_rows=verify_rows, seed=seed)
            logger.info(f"{split}: {checked} random rows verified against {src / split}")

    logger.info(f"v{SRC_SCHEMA_VERSION} -> v{SCHEMA_VERSION}: {src} -> {out} ({', '.join(splits)})")


if __name__ == "__main__":
    tyro.cli(upgrade_mds)
