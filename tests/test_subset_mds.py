"""Shard-level MDS subsetting: which shards survive, and that the result still reads."""

import json
from pathlib import Path

import numpy as np
import pytest
from streaming import MDSWriter
from streaming import StreamingDataset

from hal.data.schema import MDS_COLUMNS
from hal.data.schema import MDS_PER_FRAME_DTYPES
from hal.data.schema import SCHEMA_VERSION
from hal.scripts.subset_mds import subset_mds

N_FRAMES = 6
# Small enough that the fixtures span several shards, which is the whole point.
SHARD_BYTES = 1 << 13


def _sample(row: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(row)
    sample: dict[str, np.ndarray] = {"schema_version": SCHEMA_VERSION}
    for name, dtype in MDS_PER_FRAME_DTYPES.items():
        np_dtype = np.dtype(dtype)
        sample[name] = (
            rng.standard_normal(N_FRAMES).astype(np_dtype)
            if np_dtype.kind == "f"
            else rng.integers(0, 64, N_FRAMES).astype(np_dtype)
        )
    sample["frame"] = np.arange(row * N_FRAMES, (row + 1) * N_FRAMES, dtype=np.int32)
    return sample


def _build(root: Path, rows_per_split: dict[str, int]) -> Path:
    root.mkdir(parents=True)
    for split, n_rows in rows_per_split.items():
        with MDSWriter(
            out=str(root / split), columns=MDS_COLUMNS, compression="zstd", size_limit=SHARD_BYTES, exist_ok=False
        ) as writer:
            for row in range(n_rows):
                writer.write(_sample(row))
    (root / "stats.json").write_text(json.dumps({"mds_schema_version": SCHEMA_VERSION, "sufficient": {}}))
    return root


def _shards(root: Path, split: str) -> list[dict]:
    return json.loads((root / split / "index.json").read_text())["shards"]


def _basenames(root: Path, split: str) -> list[str]:
    return [s["zip_data"]["basename"] if s["zip_data"] else s["raw_data"]["basename"] for s in _shards(root, split)]


def _kept_rows(root: Path, split: str, every: int) -> list[int]:
    """Global row indices of the shards an ``--every`` subset keeps."""
    rows: list[int] = []
    offset = 0
    for i, shard in enumerate(_shards(root, split)):
        if i % every == 0:
            rows.extend(range(offset, offset + int(shard["samples"])))
        offset += int(shard["samples"])
    return rows


def test_subset_keeps_every_nth_train_shard_and_all_of_the_others(tmp_path: Path) -> None:
    src = _build(tmp_path / "mds", {"train": 8, "val": 2})
    assert len(_shards(src, "train")) >= 4, "the fixture must span several train shards"
    out = tmp_path / "sub2"

    subset_mds(src, out, every=2)

    assert _basenames(out, "train") == _basenames(src, "train")[::2], "basenames must survive"
    assert _basenames(out, "val") == _basenames(src, "val"), "val is never thinned"
    assert sum(s["samples"] for s in _shards(out, "train")) == len(_kept_rows(src, "train", 2))


def test_subset_hardlinks_rather_than_copies(tmp_path: Path) -> None:
    src = _build(tmp_path / "mds", {"train": 8})
    out = tmp_path / "sub2"

    subset_mds(src, out, every=2)

    for basename in _basenames(out, "train"):
        assert (out / "train" / basename).stat().st_ino == (src / "train" / basename).stat().st_ino
    assert (out / "stats.json").stat().st_ino == (src / "stats.json").stat().st_ino


def test_subset_is_readable_and_holds_the_kept_rows(tmp_path: Path) -> None:
    src = _build(tmp_path / "mds", {"train": 8})
    out = tmp_path / "sub2"
    expected = _kept_rows(src, "train", 2)

    subset_mds(src, out, every=2)

    ds = StreamingDataset(local=str(out / "train"), batch_size=1, shuffle=False)
    assert ds.num_samples == len(expected)
    for position, row in enumerate(expected):
        assert np.array_equal(ds[position]["frame"], _sample(row)["frame"])


def test_subset_every_one_keeps_everything(tmp_path: Path) -> None:
    src = _build(tmp_path / "mds", {"train": 8, "val": 2})
    out = tmp_path / "sub1"

    subset_mds(src, out, every=1)

    for split in ("train", "val"):
        assert _basenames(out, split) == _basenames(src, split)


def test_subset_refuses_an_existing_out(tmp_path: Path) -> None:
    src = _build(tmp_path / "mds", {"train": 2})
    out = tmp_path / "sub2"
    out.mkdir()
    with pytest.raises(FileExistsError):
        subset_mds(src, out, every=2)


def test_subset_rejects_a_root_with_no_splits(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        subset_mds(empty, tmp_path / "sub2", every=2)
