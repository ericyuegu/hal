import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from streaming import MDSWriter
from streaming import StreamingDataset

from hal.data.feature_stats import dump_sufficient_stats
from hal.data.feature_stats import load_sufficient_stats
from hal.data.index import PlayerEntry
from hal.data.index import PlayerType
from hal.data.index import ReplayIndexEntry
from hal.data.index import Split
from hal.data.index import Stage3Annotation
from hal.data.index import write_jsonl
from hal.data.policy_schema import policy_replay_identity
from hal.data.policy_world_schema import POLICY_WORLD_FLOAT_COLUMNS
from hal.data.policy_world_schema import POLICY_WORLD_MDS_COLUMNS
from hal.data.policy_world_schema import POLICY_WORLD_SCHEMA_VERSION
from hal.data.policy_world_schema import encode_policy_world_replay
from hal.data.policy_world_v8 import decode_policy_world_v8_replay
from hal.data.policy_world_v8 import encode_policy_world_v8_replay
from hal.data.replay_stats import PlayerStats
from hal.data.replay_stats import ReplayStats
from hal.data.schema import MDS_PER_FRAME_DTYPES
from hal.data.schema import POLICY_WORLD_V8_MDS_COLUMNS
from hal.data.schema import SCHEMA_VERSION
from hal.data.schema import Rank
from hal.scripts.filter_policy_world_mds import audit_policy_world_v8
from hal.scripts.filter_policy_world_mds import filter_policy_world_mds
from hal.wire import MASK_INT32

N_FRAMES = 1_500
SHARD_SIZE = 256 * 2**10
SOURCE_NAME = "ranked-anonymized-1-policy-world-v7"


def _player(port: int, player_type: PlayerType = "HUMAN") -> PlayerEntry:
    return PlayerEntry(port=port, character=1, costume=0, player_type=player_type, code=None, name=None)


def _stats(port: int, damage: float = 200.0) -> PlayerStats:
    return PlayerStats(
        port=port,
        damage_dealt=damage,
        damage_taken=200.0,
        stocks_remaining=1,
        inputs=100,
        death_percents=(80.0, 90.0, 100.0),
    )


def _entry(row: int, split: Split = "train") -> ReplayIndexEntry:
    path = f"archive://source!{split}-{row}.slp"
    return ReplayIndexEntry(
        path=path,
        slp_version=(3, 18, 0),
        stage=31,
        players=[_player(1), _player(3)],
        frame_count=N_FRAMES,
        timestamp=None,
        played_on="network",
        outcome=None,
        rank_filename=None,
        sha1=hashlib.sha1(path.encode()).hexdigest(),
        schema_version=SCHEMA_VERSION,
        annotation=Stage3Annotation(
            replay_uuid=row,
            split=split,
            mds_row_idx=row,
            frame_count_actual=N_FRAMES,
            schema_version=SCHEMA_VERSION,
        ),
        stats=ReplayStats(players=(_stats(1), _stats(3))),
    )


def _canonical() -> dict[str, object]:
    sample: dict[str, object] = {"schema_version": SCHEMA_VERSION}
    for name, dtype_like in MDS_PER_FRAME_DTYPES.items():
        dtype = np.dtype(dtype_like)
        if name == "frame":
            values = np.arange(N_FRAMES, dtype=dtype)
        elif name == "stage":
            values = np.full(N_FRAMES, 31, dtype=dtype)
        elif name in ("p1_character", "p2_character"):
            values = np.full(N_FRAMES, 1, dtype=dtype)
        elif name in ("p1_rank", "p2_rank"):
            values = np.full(N_FRAMES, Rank.DIAMOND, dtype=dtype)
        elif "_nana_" in name:
            fill = np.nan if dtype.kind == "f" or name.endswith("_direction") else MASK_INT32
            values = np.full(N_FRAMES, fill, dtype=dtype)
        elif name.startswith("item"):
            fill = np.nan if dtype.kind == "f" else MASK_INT32
            values = np.full(N_FRAMES, fill, dtype=dtype)
        else:
            values = np.zeros(N_FRAMES, dtype=dtype)
        sample[name] = values
    for prefix in ("p1", "p2"):
        sample[f"{prefix}_stock"] = np.ones(N_FRAMES, dtype=np.int32)
        sample[f"{prefix}_direction"] = np.zeros(N_FRAMES, dtype=np.float32)
    return sample


def _sample(entry: ReplayIndexEntry) -> dict[str, object]:
    return encode_policy_world_replay(_canonical(), policy_replay_identity(entry.path))


def _build_source(
    root: Path,
    *,
    manifest_has_stats: bool = True,
    reject_train_rows: bool = True,
) -> tuple[dict[str, list[ReplayIndexEntry]], dict[str, list[dict[str, object]]]]:
    rows = {
        "train": [_entry(row) for row in range(5)],
        "val": [_entry(0, "val")],
        "test": [_entry(0, "test")],
    }
    if reject_train_rows:
        assert rows["train"][1].stats is not None
        rows["train"][1] = dataclasses.replace(
            rows["train"][1],
            stats=dataclasses.replace(
                rows["train"][1].stats,
                players=(dataclasses.replace(rows["train"][1].stats.players[0], damage_dealt=50.0), _stats(3)),
            ),
        )
        assert rows["train"][2].stats is not None
        rows["train"][2] = dataclasses.replace(
            rows["train"][2],
            stats=dataclasses.replace(
                rows["train"][2].stats,
                players=(dataclasses.replace(rows["train"][2].stats.players[0], stocks_remaining=0), _stats(3)),
            ),
        )
        rows["train"][3] = dataclasses.replace(rows["train"][3], players=[_player(1), _player(3, "CPU")])
        assert rows["train"][4].stats is not None
        rows["train"][4] = dataclasses.replace(
            rows["train"][4],
            stats=dataclasses.replace(
                rows["train"][4].stats,
                players=(
                    dataclasses.replace(rows["train"][4].stats.players[0], death_percents=(10.0, 10.0, 80.0)),
                    _stats(3),
                ),
            ),
        )
    samples = {split: [_sample(entry) for entry in entries] for split, entries in rows.items()}
    root.mkdir()
    for split, split_samples in samples.items():
        with MDSWriter(
            out=str(root / split),
            columns=POLICY_WORLD_MDS_COLUMNS,
            compression="zstd",
            hashes=["md5", "sha256"],
            size_limit=SHARD_SIZE,
        ) as writer:
            for sample in split_samples:
                writer.write(sample)
    manifest = [entry for entries in rows.values() for entry in entries]
    if not manifest_has_stats:
        manifest = [dataclasses.replace(entry, stats=None) for entry in manifest]
    write_jsonl(root / "manifest.jsonl", manifest)
    dump_sufficient_stats(root / "stats.json", {}, split="train", mds_schema_version=SCHEMA_VERSION)
    (root / "failures.materialize.jsonl").write_text("")
    counts = {split: len(entries) for split, entries in rows.items()}
    projection = {
        "columns": POLICY_WORLD_MDS_COLUMNS,
        "failures": 0,
        "policy_world_schema_version": POLICY_WORLD_SCHEMA_VERSION,
        "rows": counts,
        "source_schema_version": SCHEMA_VERSION,
    }
    (root / "projection.json").write_text(json.dumps(projection))
    (root / "_SUCCESS").write_text(json.dumps({"rows": counts}))
    return rows, samples


def _read(root: Path, split: str) -> list[dict[str, object]]:
    dataset = StreamingDataset(local=str(root / split), batch_size=1, shuffle=False)
    return [dict(dataset[row]) for row in range(dataset.num_samples)]


def test_materializer_encodes_rows_and_rewrites_all_metadata(tmp_path: Path) -> None:
    source = tmp_path / "v7"
    output = tmp_path / "v8"
    _, samples = _build_source(source)

    selection = filter_policy_world_mds(
        str(source),
        str(output),
        source_name=SOURCE_NAME,
        scratch=tmp_path / "scratch",
        shard_size=SHARD_SIZE,
        stats_batch_rows=1,
    )

    assert selection["source_rows"] == {"train": 5, "val": 1, "test": 1}
    assert selection["rows"] == {"train": 1, "val": 1, "test": 1}
    assert selection["rejections"] == {"train": 4, "val": 0, "test": 0}
    assert selection["disk_bounds"]["concurrent_input_shards"] == 1
    for split in ("train", "val", "test"):
        (actual,) = _read(output, split)
        assert set(actual) == set(POLICY_WORLD_V8_MDS_COLUMNS)
        decoded = decode_policy_world_v8_replay(actual)
        expected = decode_policy_world_v8_replay(
            encode_policy_world_v8_replay(samples[split][0], source_name=SOURCE_NAME, p1_port=1, p2_port=3)
        )
        for name, value in expected.items():
            observed = decoded[name]
            if isinstance(value, np.ndarray):
                assert isinstance(observed, np.ndarray)
                assert observed.tobytes() == value.tobytes(), name
            else:
                assert observed == value

    output_index = json.loads((output / "train/index.json").read_text())
    assert all(shard["zip_data"] is None for shard in output_index["shards"])
    assert all(shard["raw_data"]["bytes"] <= SHARD_SIZE for shard in output_index["shards"])
    rejections = [json.loads(line) for line in (output / "rejections.v8.jsonl").read_text().splitlines()]
    assert len(rejections) == 4 and all(row["failed_rules"] for row in rejections)
    assert (output / "ranks.json").is_file()
    assert not (output / "_SUCCESS").exists()
    stats = load_sufficient_stats(output / "stats.json", expected_mds_schema_version=SCHEMA_VERSION)
    assert set(stats) == set(POLICY_WORLD_FLOAT_COLUMNS)
    assert audit_policy_world_v8(str(output), scratch=tmp_path / "audit")["rows"] == {
        "train": 1,
        "val": 1,
        "test": 1,
    }


def test_materializer_refuses_existing_output(tmp_path: Path) -> None:
    source = tmp_path / "v7"
    output = tmp_path / "v8"
    _build_source(source, reject_train_rows=False)
    output.mkdir()

    with pytest.raises(FileExistsError):
        filter_policy_world_mds(str(source), str(output), source_name=SOURCE_NAME, scratch=tmp_path / "scratch")


def test_materializer_joins_and_pins_supplemental_statistics(tmp_path: Path) -> None:
    source = tmp_path / "v7"
    output = tmp_path / "v8"
    rows, _ = _build_source(source, manifest_has_stats=False)
    metadata_index = tmp_path / "legacy-index.jsonl"
    write_jsonl(
        metadata_index,
        [
            dataclasses.replace(entry, schema_version=0, annotation=None)
            for entries in rows.values()
            for entry in entries
        ],
    )

    selection = filter_policy_world_mds(
        str(source),
        str(output),
        source_name=SOURCE_NAME,
        scratch=tmp_path / "scratch",
        shard_size=SHARD_SIZE,
        metadata_index=metadata_index,
    )

    metadata = selection["supplemental_metadata"]
    assert metadata["sha256"] == hashlib.sha256(metadata_index.read_bytes()).hexdigest()
    assert (output / metadata["artifact_path"]).read_bytes() == metadata_index.read_bytes()


def test_pilot_limit_keeps_first_valid_rows_in_source_order(tmp_path: Path) -> None:
    source = tmp_path / "v7"
    output = tmp_path / "v8"
    _build_source(source, reject_train_rows=False)

    selection = filter_policy_world_mds(
        str(source),
        str(output),
        source_name=SOURCE_NAME,
        scratch=tmp_path / "scratch",
        shard_size=SHARD_SIZE,
        retained_limits={"val": 1, "test": 1, "train": 1},
    )

    assert selection["rows"]["train"] == 1
    assert selection["omitted_after_pilot_limit"]["train"] == 4
