from collections.abc import Iterator
from pathlib import Path
from threading import Event

import numpy as np
import pytest
import torch
from streaming import MDSWriter

import hal.training.replay_reservoir as replay_reservoir
from hal.data.policy_schema import POLICY_MDS_COLUMNS
from hal.data.policy_schema import POLICY_SCHEMA_VERSION
from hal.training.ego_stats import load_consolidated_stats
from hal.training.replay_reservoir import OneBatchPrefetch
from hal.training.replay_reservoir import PolicyReplayPackDataset
from hal.training.replay_reservoir import ReplayPack
from hal.training.replay_reservoir import ReplayReservoir
from hal.training.replay_reservoir import _stable_replay_rng
from hal.training.replay_reservoir import make_reservoir_loader

_DEV_STATS = Path("data/processed/dev/mds/stats.json")


def _packs(n_replays: int, windows: int) -> Iterator[ReplayPack]:
    for replay in range(n_replays):
        values = tuple({"value": np.array([replay, window])} for window in range(windows))
        yield ReplayPack(str(replay), values)


def test_reservoir_preserves_uniqueness_cooldown_and_metrics() -> None:
    reservoir = ReplayReservoir(_packs(10, 3), batch_size=4, capacity=8, seed=4)
    batches = list(reservoir)

    seen: set[tuple[int, int]] = set()
    for batch in batches:
        assert len(batch.replay_ids) == len(set(batch.replay_ids)) == 4
        values = {tuple(window["value"]) for window in batch.windows}
        assert seen.isdisjoint(values)
        seen.update(values)
    for previous, current in zip(batches, batches[1:], strict=False):
        assert set(previous.replay_ids).isdisjoint(current.replay_ids)
    assert reservoir.emitted_windows == len(seen)
    assert reservoir.emitted_windows + reservoir.dropped_windows == 30
    assert reservoir.dropped_replays > 0
    assert reservoir.active_replays == 0


def test_reservoir_rng_is_repeatable() -> None:
    def run() -> list[tuple[str, ...]]:
        return [batch.replay_ids for batch in ReplayReservoir(_packs(20, 3), batch_size=4, capacity=8, seed=2)]

    assert run() == run()


class _CountedPacks:
    def __init__(self, packs: list[ReplayPack], start: int = 0) -> None:
        self.packs = packs
        self.at = start

    def __iter__(self) -> _CountedPacks:
        return self

    def __next__(self) -> ReplayPack:
        if self.at == len(self.packs):
            raise StopIteration
        value = self.packs[self.at]
        self.at += 1
        return value


def test_reservoir_state_resumes_exact_next_batches() -> None:
    packs = list(_packs(30, 3))
    source = _CountedPacks(packs)
    uninterrupted = ReplayReservoir(iter(source), batch_size=4, capacity=12, seed=9)
    for _ in range(5):
        next(uninterrupted)
    state = uninterrupted.state_dict()
    source_position = source.at
    expected = [next(uninterrupted) for _ in range(6)]

    resumed_source = _CountedPacks(packs, source_position)
    resumed = ReplayReservoir(iter(resumed_source), batch_size=4, capacity=12, seed=9)
    resumed.load_state_dict(state)
    actual = [next(resumed) for _ in range(6)]
    assert [batch.replay_ids for batch in actual] == [batch.replay_ids for batch in expected]
    for got, want in zip(actual, expected, strict=True):
        for got_window, want_window in zip(got.windows, want.windows, strict=True):
            np.testing.assert_array_equal(got_window["value"], want_window["value"])


@pytest.mark.parametrize("num_workers", [0, 2])
def test_real_streaming_reservoir_resume_is_tensor_exact(tmp_path: Path, num_workers: int) -> None:
    frames = 40
    with MDSWriter(out=str(tmp_path / "train"), columns=POLICY_MDS_COLUMNS, compression="zstd") as writer:
        for replay in range(20):
            sample: dict[str, object] = {
                "policy_schema_version": POLICY_SCHEMA_VERSION,
                "source_schema_version": 7,
                "replay_id": f"replay-{replay:02d}",
                "num_frames": frames,
                "stage": 25,
                "p1_character": 1,
                "p2_character": 22,
                "p1_nana_present": 0,
                "p2_nana_present": 0,
            }
            for name, encoding in POLICY_MDS_COLUMNS.items():
                if name in sample:
                    continue
                dtype = np.dtype(encoding.removeprefix("ndarray:"))
                sample[name] = np.zeros(1 if "nana" in name else frames, dtype=dtype)
            positions = np.arange(frames, dtype=np.float32) + replay * frames
            sticks = (np.arange(frames) % 5).astype(np.int8)
            sample["p1_position_x"] = positions
            sample["p2_position_x"] = -positions
            sample["p1_main_stick_x"] = sticks
            sample["p2_main_stick_x"] = -sticks
            writer.write(sample)
    stats = load_consolidated_stats(_DEV_STATS)

    def loader(workers: int = num_workers):
        return make_reservoir_loader(
            str(tmp_path),
            "train",
            stats=stats,
            L_ctx=4,
            L_chunk=2,
            batch_size=2,
            seed=23,
            reservoir_capacity=4,
            remote=None,
            shuffle_block_size=32,
            predownload=1,
            windows_per_replay=4,
            prefetch_batches=0,
            num_workers=workers,
            pin_memory=False,
            schema_version=7,
        )

    source = loader()
    source_iterator = iter(source)
    for _ in range(3):
        next(source_iterator)
    state = source.state_dict()
    assert state["schema"] == 2
    expected = [next(source_iterator) for _ in range(3)]

    for resume_state in (state, {**state, "schema": 1}):
        resumed = loader()
        resumed.load_state_dict(resume_state)
        resumed_iterator = iter(resumed)
        actual = [next(resumed_iterator) for _ in range(3)]
        for got, want in zip(actual, expected, strict=True):
            assert got.replay_ids == want.replay_ids
            assert torch.equal(got.context.ctx_pad, want.context.ctx_pad)
            assert torch.equal(got.target, want.target)
            assert got.context.features.keys() == want.context.features.keys()
            for name in got.context.features:
                assert torch.equal(got.context.features[name], want.context.features[name])

    if num_workers == 2:

        def replay_sequence(workers: int) -> list[tuple[str, ...]]:
            candidate = loader(workers)
            iterator = iter(candidate)
            sequence = []
            while len(sequence) < 45:
                try:
                    sequence.append(next(iterator).replay_ids)
                except StopIteration:
                    iterator = iter(candidate)
            return sequence

        assert replay_sequence(2) == replay_sequence(0)


def test_stable_replay_rng_depends_on_identity_and_epoch() -> None:
    first = _stable_replay_rng(7, 2, "replay-a").integers(0, 2**31, size=8)
    assert np.array_equal(first, _stable_replay_rng(7, 2, "replay-a").integers(0, 2**31, size=8))
    assert not np.array_equal(first, _stable_replay_rng(7, 3, "replay-a").integers(0, 2**31, size=8))
    assert not np.array_equal(first, _stable_replay_rng(7, 2, "replay-b").integers(0, 2**31, size=8))


def test_compact_replay_transform_sees_full_episode_before_windowing(monkeypatch) -> None:
    frames = 12
    decoded = {
        "schema_version": 7,
        "frame": np.arange(frames, dtype=np.int32),
        "p1_value": np.arange(frames, dtype=np.float32),
        "p2_value": -np.arange(frames, dtype=np.float32),
    }
    compact = {"replay_id": "r", "source_schema_version": 7, "num_frames": frames}
    monkeypatch.setattr(replay_reservoir, "decode_policy_replay", lambda row: decoded)
    seen: list[int] = []

    def transform(sample):
        seen.append(len(sample["frame"]))
        return {
            **sample,
            "p1_label": np.arange(frames, dtype=np.float32) + 100,
            "p2_label": np.arange(frames, dtype=np.float32) + 100,
        }

    packs = list(
        PolicyReplayPackDataset(
            [compact],
            L_ctx=3,
            L_chunk=2,
            seed=4,
            windows_per_replay=1,
            schema_version=7,
            projection=None,
            replay_transform=transform,
        )
    )
    assert seen == [frames]
    assert len(packs) == len(packs[0].windows) == 1
    window = packs[0].windows[0]
    pad = int(window["ctx_pad"])
    np.testing.assert_array_equal(window["ego_label"][pad:], window["frame"][pad:] + 100)


def test_compact_replay_labels_are_computed_once_then_sliced(monkeypatch) -> None:
    frames = 20
    compact = {"replay_id": "r", "source_schema_version": 7, "num_frames": frames}

    def decode_slices(row, ranges):
        del row
        return tuple(
            {
                "schema_version": 7,
                "frame": np.arange(start, stop, dtype=np.int32),
                "p1_value": np.arange(start, stop, dtype=np.float32),
                "p2_value": -np.arange(start, stop, dtype=np.float32),
            }
            for start, stop in ranges
        )

    monkeypatch.setattr(replay_reservoir, "decode_policy_replay_slices", decode_slices)
    calls = 0

    def replay_labels(row):
        nonlocal calls
        assert row is compact
        calls += 1
        values = np.arange(frames, dtype=np.float32) + 100
        return {"p1_label": values, "p2_label": values}

    packs = list(
        PolicyReplayPackDataset(
            [compact],
            L_ctx=3,
            L_chunk=2,
            seed=4,
            windows_per_replay=2,
            schema_version=7,
            projection=None,
            replay_labels=replay_labels,
        )
    )

    assert calls == 1
    assert len(packs) == 1
    for window in packs[0].windows:
        pad = int(window["ctx_pad"])
        np.testing.assert_array_equal(window["ego_label"][pad:], window["frame"][pad:] + 100)


def test_replay_labels_reject_transform_and_wrong_length(monkeypatch) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        PolicyReplayPackDataset(
            [],
            L_ctx=3,
            L_chunk=2,
            seed=0,
            windows_per_replay=1,
            schema_version=7,
            projection=None,
            replay_transform=lambda replay: replay,
            replay_labels=lambda replay: {},
        )

    frames = 12
    compact = {"replay_id": "r", "source_schema_version": 7, "num_frames": frames}
    monkeypatch.setattr(
        replay_reservoir,
        "decode_policy_replay_slices",
        lambda row, ranges: tuple(
            {
                "schema_version": 7,
                "frame": np.arange(start, stop, dtype=np.int32),
                "p1_value": np.arange(start, stop, dtype=np.float32),
                "p2_value": -np.arange(start, stop, dtype=np.float32),
            }
            for start, stop in ranges
        ),
    )
    packs = PolicyReplayPackDataset(
        [compact],
        L_ctx=3,
        L_chunk=2,
        seed=0,
        windows_per_replay=1,
        schema_version=7,
        projection=None,
        replay_labels=lambda replay: {"p1_label": np.zeros(frames - 1)},
    )

    with pytest.raises(ValueError, match="invalid shapes"):
        list(packs)


def test_prefetch_preserves_order_at_requested_depth() -> None:
    prepared = Event()

    def source() -> Iterator[int]:
        for index in range(5):
            if index == 2:
                prepared.set()
            yield index

    iterator = OneBatchPrefetch(source(), depth=3)
    assert prepared.wait(timeout=1)
    assert list(iterator) == list(range(5))


def test_prefetch_propagates_error_and_closes_source() -> None:
    closed = Event()

    def source() -> Iterator[int]:
        try:
            yield 1
            raise RuntimeError("source failed")
        finally:
            closed.set()

    iterator = OneBatchPrefetch(source(), depth=2)
    assert next(iterator) == 1
    with pytest.raises(RuntimeError, match="source failed"):
        next(iterator)
    assert closed.wait(timeout=1)


def test_prefetch_close_propagates_pending_producer_error() -> None:
    def source() -> Iterator[int]:
        yield 1
        raise RuntimeError("source failed")

    iterator = OneBatchPrefetch(source(), depth=2)
    assert next(iterator) == 1
    with pytest.raises(RuntimeError, match="source failed"):
        iterator.close()


def test_prefetch_close_is_idempotent_and_stops_iteration() -> None:
    closed = Event()

    def source() -> Iterator[int]:
        try:
            yield from range(4)
        finally:
            closed.set()

    iterator = OneBatchPrefetch(source(), depth=2)
    assert next(iterator) == 0
    iterator.close()
    iterator.close()
    assert closed.wait(timeout=1)
    with pytest.raises(StopIteration):
        next(iterator)


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_factory_rejects_invalid_prefetch_batches(value: object) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        make_reservoir_loader(
            "/unused",
            "train",
            stats={},
            L_ctx=1,
            L_chunk=1,
            batch_size=2,
            seed=0,
            reservoir_capacity=4,
            prefetch_batches=value,  # type: ignore[arg-type]
        )
