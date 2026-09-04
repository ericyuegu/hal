import dataclasses
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from streaming import StreamingDataset

from hal.data.index import ReplayIndexEntry
from hal.data.index import Stage3Annotation
from hal.data.index import read_jsonl
from hal.data.index import write_jsonl
from hal.data.policy_schema import POLICY_SCHEMA_VERSION
from hal.data.policy_schema import decode_policy_replay
from hal.scripts import project_policy_mds as project_module
from hal.scripts.project_policy_mds import load_replay_ids_by_split
from hal.scripts.project_policy_mds import project_policy_mds
from hal.training.dataloader import WindowDataset
from hal.training.dataloader import choose_chunk_starts
from hal.training.dataloader import collate_train_batch
from hal.training.dataloader import make_window
from hal.training.ego_stats import load_consolidated_stats
from hal.training.features import FeatureProjection
from hal.training.replay_reservoir import PolicyReplayPackDataset
from hal.training.replay_reservoir import _stable_replay_rng
from hal.training.replay_reservoir import make_reservoir_loader

_DEV_MDS = Path(__file__).resolve().parents[1] / "data" / "processed" / "dev" / "mds"
_RANKED_MANIFEST = (
    Path(__file__).resolve().parents[1] / "data" / "processed" / "ranked-anonymized-1" / "mds-v7" / "manifest.jsonl"
)


def _full_decode_windows(
    compact: dict,
    L_ctx: int,
    L_chunk: int,
    seed: int,
    projection: FeatureProjection | None,
) -> tuple[dict, ...]:
    sample = decode_policy_replay(compact)
    replay_id = str(compact["replay_id"])
    rng = _stable_replay_rng(seed, 0, replay_id)
    windows = []
    for cs in choose_chunk_starts(len(sample["frame"]), L_ctx, L_chunk, 2, rng):
        start = int(cs) - L_ctx
        pad = max(0, -start)
        ego_prefix = "p1" if rng.random() < 0.5 else "p2"
        window = make_window(
            sample,
            ego_prefix=ego_prefix,
            start=start,
            pad=pad,
            length=L_ctx + L_chunk,
            projection=projection,
        )
        window["ctx_pad"] = np.int64(min(pad, L_ctx))
        windows.append(window)
    return tuple(windows)


def _assert_windows_exact(actual: tuple[dict, ...], expected: tuple[dict, ...]) -> None:
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected, strict=True):
        assert got.keys() == want.keys()
        for name in got:
            assert np.array_equal(got[name], want[name], equal_nan=True), name


def _manifest_entry(path: str, replay_uuid: int, split: str, row: int) -> ReplayIndexEntry:
    return ReplayIndexEntry(
        path=path,
        slp_version=(3, 16, 0),
        stage=31,
        players=[],
        frame_count=1,
        timestamp=None,
        played_on=None,
        outcome=None,
        rank_filename=None,
        sha1=None,
        schema_version=7,
        annotation=Stage3Annotation(
            replay_uuid=replay_uuid,
            split=split,
            mds_row_idx=row,
            frame_count_actual=1,
            schema_version=7,
        ),
    )


def test_replay_identity_does_not_use_colliding_32_bit_uuid(tmp_path: Path) -> None:
    src = tmp_path / "source"
    src.mkdir()
    write_jsonl(
        src / "manifest.jsonl",
        [
            _manifest_entry("archive://one!first.slp", 1234, "train", 0),
            _manifest_entry("archive://two!second.slp", 1234, "train", 1),
        ],
    )

    replay_ids = load_replay_ids_by_split(src)["train"]
    assert len(replay_ids) == len(set(replay_ids)) == 2
    assert all(len(replay_id) == 32 for replay_id in replay_ids)


def test_duplicate_manifest_path_is_rejected_across_splits(tmp_path: Path) -> None:
    src = tmp_path / "source"
    src.mkdir()
    entry = _manifest_entry("archive://same!replay.slp", 1, "train", 0)
    write_jsonl(
        src / "manifest.jsonl",
        [entry, dataclasses.replace(entry, annotation=dataclasses.replace(entry.annotation, split="val"))],
    )

    with pytest.raises(ValueError, match="duplicate replay identity"):
        load_replay_ids_by_split(src)


@pytest.mark.skipif(not _RANKED_MANIFEST.is_file(), reason="ranked v7 manifest is not available")
def test_real_manifest_32_bit_collisions_have_unique_projected_ids() -> None:
    entries = [entry for entry in read_jsonl(_RANKED_MANIFEST, verify_schema_version=False) if entry.annotation]
    uuid_counts = Counter(entry.annotation.replay_uuid for entry in entries if entry.annotation)
    assert any(count > 1 for count in uuid_counts.values())

    replay_ids = load_replay_ids_by_split(_RANKED_MANIFEST.parent)
    flattened = [replay_id for split_ids in replay_ids.values() for replay_id in split_ids]
    assert len(flattened) == len(set(flattened)) == len(entries)


def test_projection_failure_does_not_publish_partial_output(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "source"
    (src / "train").mkdir(parents=True)
    (src / "train" / "index.json").write_text("{}")
    out = tmp_path / "policy"

    def fail_write(*args, **kwargs):
        raise RuntimeError("injected projection failure")

    monkeypatch.setattr(project_module, "_write_split", fail_write)
    with pytest.raises(RuntimeError, match="injected projection failure"):
        project_policy_mds(src, out, splits=("train",), scratch=tmp_path / "scratch", allow_row_ids=True)

    assert not out.exists()
    assert not list(tmp_path.glob(".policy.*"))


@pytest.mark.skipif(not (_DEV_MDS / "train").is_dir(), reason="local dev MDS is not available")
def test_projected_mds_preserves_model_values(tmp_path) -> None:
    out = tmp_path / "policy"
    project_policy_mds(_DEV_MDS, out, splits=("train",), scratch=tmp_path, max_rows=68)
    assert list((out / "train").glob("*.mds.zstd"))
    assert not list((out / "train").glob("*.mds"))

    source = StreamingDataset(local=str(_DEV_MDS / "train"), batch_size=1, shuffle=False)
    compact = StreamingDataset(local=str(out / "train"), batch_size=1, shuffle=False)
    assert len(compact) == 68
    for row in range(2):
        assert compact[row]["policy_schema_version"] == POLICY_SCHEMA_VERSION
        decoded = decode_policy_replay(compact[row])
        assert decoded["schema_version"] == source[row]["schema_version"]

    compact_row = compact[0]
    frames = int(compact_row["num_frames"])
    projections = (
        None,
        FeatureProjection(frozenset({"stage", "ego_position_x", "opp_position_x", "ego_main_stick_x"})),
    )
    for projection in projections:
        for L_ctx, L_chunk in ((32, 2), (frames, 2)):
            expected = _full_decode_windows(compact_row, L_ctx, L_chunk, seed=11, projection=projection)
            packs = PolicyReplayPackDataset(
                [compact_row],
                L_ctx,
                L_chunk,
                seed=11,
                windows_per_replay=2,
                schema_version=5,
                projection=projection,
            )
            actual = next(iter(packs)).windows
            _assert_windows_exact(actual, expected)
            if L_ctx == frames:
                assert actual[0]["ctx_pad"] > 0
    assert (out / "stats.json").is_file()
    assert (out / "projection.json").is_file()

    stats = load_consolidated_stats(out / "stats.json")
    full_rows = [source[row] for row in range(2)]
    compact_rows = [decode_policy_replay(compact[row]) for row in range(2)]
    window_kwargs = dict(L_ctx=32, L_chunk=2, seed=0, windows_per_replay=1, schema_version=5)
    full_windows = list(WindowDataset(full_rows, **window_kwargs))
    compact_windows = list(WindowDataset(compact_rows, **window_kwargs))
    full = collate_train_batch(full_windows, stats=stats, L_ctx=32)
    projected = collate_train_batch(compact_windows, stats=stats, L_ctx=32)
    assert projected.context.ctx_pad.equal(full.context.ctx_pad)
    assert projected.target.equal(full.target)
    for name, value in projected.context.features.items():
        assert value.equal(full.context.features[name]), name

    def reservoir_batches(prefetch_batches: int):
        reservoir = make_reservoir_loader(
            data_root=str(out),
            split="train",
            stats=stats,
            L_ctx=32,
            L_chunk=2,
            batch_size=2,
            seed=0,
            reservoir_capacity=4,
            num_workers=0,
            windows_per_replay=1,
            prefetch_batches=prefetch_batches,
            schema_version=5,
        )
        batches = list(reservoir)
        return reservoir, batches

    plain_reservoir, plain_batches = reservoir_batches(0)
    prefetched_reservoir, prefetched_batches = reservoir_batches(1)
    assert len(plain_batches) == len(prefetched_batches) == 34
    for plain, prefetched in zip(plain_batches, prefetched_batches, strict=True):
        assert plain.replay_ids == prefetched.replay_ids
        assert plain.context.ctx_pad.equal(prefetched.context.ctx_pad)
        assert plain.target.equal(prefetched.target)
        assert plain.context.features.keys() == prefetched.context.features.keys()
        for name in plain.context.features:
            assert plain.context.features[name].equal(prefetched.context.features[name]), name
    for batch in prefetched_batches:
        assert batch.replay_ids is not None
        assert len(batch.replay_ids) == len(set(batch.replay_ids)) == 2
    expected_stats = {"emitted_windows": 68, "dropped_windows": 0, "dropped_replays": 0}
    assert plain_reservoir.last_epoch_stats == expected_stats
    assert prefetched_reservoir.last_epoch_stats == expected_stats
