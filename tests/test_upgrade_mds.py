"""v6 -> v7 MDS upgrade: the rank join, row-order preservation, and the guards.

Every case builds a tiny multi-shard v6 dataset (shards + manifest) in tmp, runs
the upgrade, and reads the result back through ``StreamingDataset`` — the same
consumer training uses.
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
from streaming import MDSWriter
from streaming import StreamingDataset

from hal.data.index import PlayerEntry
from hal.data.index import ReplayIndexEntry
from hal.data.index import Split
from hal.data.index import Stage3Annotation
from hal.data.index import write_jsonl
from hal.data.schema import MDS_COLUMNS
from hal.data.schema import MDS_PER_FRAME_DTYPES
from hal.data.schema import SCHEMA_VERSION
from hal.data.schema import Rank
from hal.data.schema import rank_from_player_name
from hal.scripts.upgrade_mds import SRC_SCHEMA_VERSION
from hal.scripts.upgrade_mds import upgrade_mds

RANK_COLUMNS = ("p1_rank", "p2_rank")
V6_COLUMNS = {name: encoding for name, encoding in MDS_COLUMNS.items() if name not in RANK_COLUMNS}

N_FRAMES = 6
# Small enough that a handful of rows spill into a second shard, so the tests
# exercise the multi-shard read/write path and the shard-boundary arithmetic.
SHARD_BYTES = 1 << 14

LADDER_NAMES = ("Platinum Player", "Diamond Player", "Master Player")


def _v6_sample(row: int) -> dict[str, np.ndarray]:
    """One v6 shard row. ``frame`` counts from ``row * N_FRAMES`` so a written row
    identifies itself — that is how row order is checked after the upgrade."""
    rng = np.random.default_rng(row)
    sample: dict[str, np.ndarray] = {"schema_version": SRC_SCHEMA_VERSION}
    for name, dtype in MDS_PER_FRAME_DTYPES.items():
        if name in RANK_COLUMNS:
            continue
        np_dtype = np.dtype(dtype)
        if np_dtype.kind == "f":
            sample[name] = rng.standard_normal(N_FRAMES).astype(np_dtype)
        else:
            sample[name] = rng.integers(0, 64, N_FRAMES).astype(np_dtype)
    sample["frame"] = np.arange(row * N_FRAMES, (row + 1) * N_FRAMES, dtype=np.int32)
    # A masked float, so the column comparison has to treat NaN as equal to itself.
    sample["p1_hitlag_left"][0] = np.nan
    return sample


def _entry(
    row: int,
    split: Split,
    names: tuple[str | None, str | None],
    *,
    ports: tuple[int, int] = (1, 2),
) -> ReplayIndexEntry:
    return ReplayIndexEntry(
        path=f"data/raw/{split}-{row}.slp",
        slp_version=(3, 15, 0),
        stage=32,
        players=[
            PlayerEntry(port=port, character=1, costume=0, player_type="HUMAN", code=None, name=name)
            for port, name in zip(ports, names, strict=True)
        ],
        frame_count=N_FRAMES,
        timestamp=None,
        played_on=None,
        outcome=None,
        rank_filename=None,
        sha1=None,
        schema_version=SRC_SCHEMA_VERSION,
        annotation=Stage3Annotation(
            replay_uuid=row,
            split=split,
            mds_row_idx=row,
            frame_count_actual=N_FRAMES,
            schema_version=SRC_SCHEMA_VERSION,
        ),
    )


def _ranked_rows(n_train: int = 9, n_val: int = 3) -> dict[Split, list[ReplayIndexEntry]]:
    """Rows whose tiers cycle, so no row's tier can land right by accident."""
    return {
        "train": [_entry(i, "train", (LADDER_NAMES[i % 3], LADDER_NAMES[(i + 1) % 3])) for i in range(n_train)],
        "val": [_entry(i, "val", (LADDER_NAMES[i % 3], LADDER_NAMES[(i + 2) % 3])) for i in range(n_val)],
    }


def _build_v6(root: Path, rows: dict[Split, list[ReplayIndexEntry]]) -> Path:
    """Write a v6 dataset: shards per split in annotation order, plus manifest and stats."""
    root.mkdir(parents=True, exist_ok=True)
    for split, entries in rows.items():
        with MDSWriter(
            out=str(root / split), columns=V6_COLUMNS, compression="zstd", size_limit=SHARD_BYTES, exist_ok=False
        ) as writer:
            for entry in entries:
                assert entry.annotation is not None
                writer.write(_v6_sample(entry.annotation.mds_row_idx))
    write_jsonl(root / "manifest.jsonl", [e for entries in rows.values() for e in entries])
    (root / "stats.json").write_text(json.dumps({"mds_schema_version": SRC_SCHEMA_VERSION, "sufficient": {}}))
    return root


def _shards(root: Path, split: str) -> list[dict]:
    return json.loads((root / split / "index.json").read_text())["shards"]


def _read(root: Path, split: str) -> list[dict]:
    ds = StreamingDataset(local=str(root / split), batch_size=1, shuffle=False)
    return [ds[i] for i in range(ds.num_samples)]


def test_upgrade_preserves_every_column_and_row_order(tmp_path: Path) -> None:
    rows = _ranked_rows()
    src = _build_v6(tmp_path / "v6", rows)
    out = tmp_path / "v7"
    upgrade_mds(src, out)

    for split, entries in rows.items():
        assert len(_shards(src, split)) > 1, f"{split} needs more than one shard to exercise the boundary"
        upgraded = _read(out, split)
        assert len(upgraded) == len(entries)
        for row, sample in enumerate(upgraded):
            original = _v6_sample(row)
            assert sample["schema_version"] == SCHEMA_VERSION
            assert set(sample) - set(original) == set(RANK_COLUMNS)
            for name, value in original.items():
                if name == "schema_version":
                    continue
                assert np.array_equal(value, sample[name], equal_nan=value.dtype.kind == "f"), name


def test_upgrade_writes_the_manifest_tier_per_row(tmp_path: Path) -> None:
    rows = _ranked_rows()
    src = _build_v6(tmp_path / "v6", rows)
    out = tmp_path / "v7"
    upgrade_mds(src, out)

    for split, entries in rows.items():
        for sample, entry in zip(_read(out, split), entries, strict=True):
            expected = [rank_from_player_name(p.name) for p in entry.players]
            for column, rank in zip(RANK_COLUMNS, expected, strict=True):
                assert sample[column].dtype == np.uint8
                assert sample[column].shape == sample["frame"].shape
                assert np.all(sample[column] == int(rank)), f"{split} {column}"


def test_rank_columns_follow_port_order_not_manifest_order(tmp_path: Path) -> None:
    """A mixed-tier game: p1 is the LOWEST port whatever order the manifest lists
    the players in — the same rule ``extract_replay`` writes the columns by."""
    src = _build_v6(
        tmp_path / "v6", {"train": [_entry(0, "train", ("Master Player", "Platinum Player"), ports=(2, 1))]}
    )
    out = tmp_path / "v7"
    upgrade_mds(src, out)

    sample = _read(out, "train")[0]
    assert np.all(sample["p1_rank"] == Rank.PLATINUM)
    assert np.all(sample["p2_rank"] == Rank.MASTER)


def test_unknown_player_name_raises(tmp_path: Path) -> None:
    rows = _ranked_rows(n_train=4, n_val=1)
    rows["train"][2] = _entry(2, "train", ("ZAIN", "Master Player"))
    src = _build_v6(tmp_path / "v6", rows)

    with pytest.raises(ValueError, match="UNKNOWN"):
        upgrade_mds(src, tmp_path / "v7")


def test_unknown_player_name_is_allowed_when_not_ranked(tmp_path: Path) -> None:
    rows = _ranked_rows(n_train=4, n_val=1)
    rows["train"][2] = _entry(2, "train", ("ZAIN", "Master Player"))
    src = _build_v6(tmp_path / "v6", rows)
    out = tmp_path / "v7"

    upgrade_mds(src, out, require_ranked=False)
    sample = _read(out, "train")[2]
    assert np.all(sample["p1_rank"] == Rank.UNKNOWN)
    assert np.all(sample["p2_rank"] == Rank.MASTER)


def test_row_count_mismatch_raises(tmp_path: Path) -> None:
    """A manifest that annotates fewer rows than the shards hold cannot be a
    row-by-row join, so the upgrade refuses it instead of shifting the tiers."""
    rows = _ranked_rows(n_train=6, n_val=2)
    src = _build_v6(tmp_path / "v6", rows)
    write_jsonl(src / "manifest.jsonl", [e for entries in rows.values() for e in entries][:-3])

    with pytest.raises(ValueError, match="contiguous|manifest rows"):
        upgrade_mds(src, tmp_path / "v7")


def test_src_at_the_wrong_version_raises(tmp_path: Path) -> None:
    rows = _ranked_rows(n_train=3, n_val=1)
    src = _build_v6(tmp_path / "v6", rows)
    bumped = [dataclasses.replace(e, schema_version=SCHEMA_VERSION) for entries in rows.values() for e in entries]
    write_jsonl(src / "manifest.jsonl", bumped)

    with pytest.raises(ValueError, match="only reads a v6 dataset"):
        upgrade_mds(src, tmp_path / "v7")


def test_out_must_not_exist(tmp_path: Path) -> None:
    src = _build_v6(tmp_path / "v6", _ranked_rows(n_train=3, n_val=1))
    out = tmp_path / "v7"
    out.mkdir()
    with pytest.raises(FileExistsError):
        upgrade_mds(src, out)


def test_missing_manifest_raises(tmp_path: Path) -> None:
    src = _build_v6(tmp_path / "v6", _ranked_rows(n_train=3, n_val=1))
    (src / "manifest.jsonl").unlink()
    with pytest.raises(FileNotFoundError):
        upgrade_mds(src, tmp_path / "v7")


def test_manifest_and_stats_are_restamped(tmp_path: Path) -> None:
    rows = _ranked_rows(n_train=4, n_val=1)
    src = _build_v6(tmp_path / "v6", rows)
    out = tmp_path / "v7"
    upgrade_mds(src, out)

    before = [e.to_dict() for entries in rows.values() for e in entries]
    after = [json.loads(line) for line in (out / "manifest.jsonl").read_text().splitlines()]
    assert len(after) == len(before)
    for was, now in zip(before, after, strict=True):
        assert now["schema_version"] == SCHEMA_VERSION
        assert now["annotation"]["schema_version"] == SCHEMA_VERSION
        assert now["annotation"]["mds_row_idx"] == was["annotation"]["mds_row_idx"]
        assert now["path"] == was["path"]
    assert json.loads((out / "stats.json").read_text())["mds_schema_version"] == SCHEMA_VERSION
