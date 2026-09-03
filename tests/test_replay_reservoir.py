from collections import Counter
from collections.abc import Iterator
from itertools import islice
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
from hal.training.replay_reservoir import ReplayPackBatchIterator
from hal.training.replay_reservoir import ReplayReservoir
from hal.training.replay_reservoir import _stable_replay_rng
from hal.training.replay_reservoir import make_reservoir_loader

_DEV_STATS = Path("data/processed/dev/mds/stats.json")


def _packs(n_replays: int, windows: int) -> Iterator[ReplayPack]:
    for replay in range(n_replays):
        values = tuple({"value": np.array([replay, window])} for window in range(windows))
        yield ReplayPack(str(replay), values)


def _write_policy_mds(path: Path, *, replays: int, frames: int) -> None:
    with MDSWriter(out=str(path / "train"), columns=POLICY_MDS_COLUMNS, compression="zstd") as writer:
        for replay in range(replays):
            sample: dict[str, object] = {
                "policy_schema_version": POLICY_SCHEMA_VERSION,
                "source_schema_version": 7,
                "replay_id": f"replay-{replay:03d}",
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
            writer.write(sample)


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


def test_real_streaming_reservoir_resume_is_tensor_exact(tmp_path: Path) -> None:
    frames = 12
    _write_policy_mds(tmp_path, replays=20, frames=frames)
    stats = load_consolidated_stats(_DEV_STATS)

    def loader():
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
            windows_per_replay=1,
            prefetch_batches=0,
            num_workers=0,
            pin_memory=False,
            schema_version=7,
        )

    source = loader()
    source_iterator = iter(source)
    for _ in range(3):
        next(source_iterator)
    state = source.state_dict()
    expected = [next(source_iterator) for _ in range(3)]

    resumed = loader()
    resumed.load_state_dict(state)
    resumed_iterator = iter(resumed)
    actual = [next(resumed_iterator) for _ in range(3)]
    for got, want in zip(actual, expected, strict=True):
        assert got.replay_ids == want.replay_ids
        assert torch.equal(got.context.ctx_pad, want.context.ctx_pad)
        assert torch.equal(got.target, want.target)
        assert got.context.features.keys() == want.context.features.keys()
        for name in got.context.features:
            assert torch.equal(got.context.features[name], want.context.features[name])


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


def test_scalar_replay_labels_expand_only_inside_selected_slices(monkeypatch) -> None:
    frames = 40
    compact = {"replay_id": "r", "source_schema_version": 7, "num_frames": frames}

    def decode_slices(row, ranges):
        del row
        return tuple(
            {
                "schema_version": 7,
                "frame": np.arange(start, stop, dtype=np.int32),
                "p1_player_id": np.zeros(stop - start, dtype=np.int32),
                "p2_player_id": np.zeros(stop - start, dtype=np.int32),
            }
            for start, stop in ranges
        )

    monkeypatch.setattr(replay_reservoir, "decode_policy_replay_slices", decode_slices)
    packs = list(
        PolicyReplayPackDataset(
            [compact],
            L_ctx=4,
            L_chunk=2,
            seed=4,
            windows_per_replay=2,
            schema_version=7,
            projection=None,
            replay_labels=lambda replay: {
                "p1_identity": np.asarray(7, dtype=np.int32),
                "p2_identity": np.asarray(11, dtype=np.int32),
            },
            require_full_context=True,
        )
    )

    assert len(packs) == 1
    for window in packs[0].windows:
        identity = window["ego_identity"]
        assert identity.shape == (6,)
        assert np.unique(identity).item() in (7, 11)


def test_four_windows_are_deterministic_and_non_overlapping(monkeypatch) -> None:
    frames = 80
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

    def run() -> ReplayPack:
        return next(
            iter(
                PolicyReplayPackDataset(
                    [compact],
                    L_ctx=4,
                    L_chunk=2,
                    seed=17,
                    windows_per_replay=4,
                    schema_version=7,
                    projection=None,
                    require_full_context=True,
                )
            )
        )

    first = run()
    second = run()
    assert len(first.windows) == 4
    first_ranges = [(int(window["frame"][0]), int(window["frame"][-1]) + 1) for window in first.windows]
    second_ranges = [(int(window["frame"][0]), int(window["frame"][-1]) + 1) for window in second.windows]
    assert first_ranges == second_ranges
    ordered = sorted(first_ranges)
    for (_, stop), (start, _) in zip(ordered[:-1], ordered[1:], strict=True):
        assert stop <= start


def test_required_pack_rejects_a_replay_that_cannot_supply_four_windows() -> None:
    compact = {"replay_id": "short", "source_schema_version": 7, "num_frames": 12}
    packs = PolicyReplayPackDataset(
        [compact],
        L_ctx=4,
        L_chunk=2,
        seed=17,
        windows_per_replay=4,
        schema_version=7,
        projection=None,
        require_pack=True,
        require_full_context=True,
    )

    with pytest.raises(ValueError, match="emits 1 of the required 4 windows"):
        next(iter(packs))


def test_replay_pack_batch_iterator_persists_worker_lookahead_and_visits() -> None:
    visits: Counter[str] = Counter()
    source = iter([(ReplayPack("a", ({"x": np.zeros(1)},)), ReplayPack("b", ({"x": np.ones(1)},)))])
    packs = ReplayPackBatchIterator(source, visits)

    assert next(packs).replay_id == "a"
    state = packs.state_dict()
    assert visits == {"a": 1, "b": 1}
    resumed = ReplayPackBatchIterator(iter(()), visits)
    resumed.load_state_dict(state)
    assert next(resumed).replay_id == "b"


def test_worker_collation_round_trips_replay_packs_through_shared_tensors() -> None:
    original = tuple(_packs(3, 2))
    collated = replay_reservoir._collate_replay_packs(list(original))
    visits: Counter[str] = Counter()
    restored = ReplayPackBatchIterator(iter((collated,)), visits)

    for expected in original:
        actual = next(restored)
        assert actual.replay_id == expected.replay_id
        for actual_window, expected_window in zip(actual.windows, expected.windows, strict=True):
            np.testing.assert_array_equal(actual_window["value"], expected_window["value"])
    assert visits == {"0": 1, "1": 1, "2": 1}


def _o51_style_loader(path: Path, *, workers: int, pack_batch: int):
    return make_reservoir_loader(
        str(path),
        "train",
        stats=load_consolidated_stats(_DEV_STATS),
        L_ctx=4,
        L_chunk=2,
        batch_size=2,
        seed=23,
        reservoir_capacity=8,
        remote=None,
        shuffle=True,
        shuffle_seed=23,
        shuffle_algo="py1s",
        shuffle_block_size=32,
        predownload=8 * pack_batch,
        windows_per_replay=4,
        prefetch_batches=0,
        num_workers=workers,
        prefetch_factor=1,
        pin_memory=False,
        schema_version=7,
        replay_pack_batch_size=pack_batch,
        worker_independent_resume=True,
        limit_worker_threads=True,
        require_full_context=True,
    )


def _batch_signature(batch) -> tuple[tuple[str, ...], torch.Tensor, torch.Tensor]:
    return batch.replay_ids, batch.context.ctx_pad.clone(), batch.target.clone()


def test_pack_batch_is_passed_to_mosaic_and_dataloader_without_changing_draws(tmp_path: Path) -> None:
    _write_policy_mds(tmp_path, replays=128, frames=80)
    small = _o51_style_loader(tmp_path, workers=0, pack_batch=2)
    large = _o51_style_loader(tmp_path, workers=0, pack_batch=4)
    assert small._pack_loader.batch_size == small._dataset.batch_size == 2
    assert large._pack_loader.batch_size == large._dataset.batch_size == 4

    small_batches = [_batch_signature(batch) for batch in islice(small, 8)]
    large_batches = [_batch_signature(batch) for batch in islice(large, 8)]
    for got, want in zip(small_batches, large_batches, strict=True):
        assert got[0] == want[0]
        assert torch.equal(got[1], want[1])
        assert torch.equal(got[2], want[2])


def test_resumption_is_tensor_exact_when_worker_count_changes(tmp_path: Path) -> None:
    _write_policy_mds(tmp_path, replays=64, frames=80)
    uninterrupted = _o51_style_loader(tmp_path, workers=2, pack_batch=4)
    iterator = iter(uninterrupted)
    for _ in range(5):
        next(iterator)
    state = uninterrupted.state_dict()
    expected = [_batch_signature(next(iterator)) for _ in range(8)]

    resumed = _o51_style_loader(tmp_path, workers=0, pack_batch=4)
    resumed.load_state_dict(state)
    resumed_iterator = iter(resumed)
    actual = [_batch_signature(next(resumed_iterator)) for _ in range(8)]

    for got, want in zip(actual, expected, strict=True):
        assert got[0] == want[0]
        assert torch.equal(got[1], want[1])
        assert torch.equal(got[2], want[2])
    assert state["visit_counters"]

    chained_state = resumed.state_dict()
    chained_expected = [_batch_signature(next(iterator)) for _ in range(5)]
    chained = _o51_style_loader(tmp_path, workers=1, pack_batch=4)
    chained.load_state_dict(chained_state)
    chained_iterator = iter(chained)
    chained_actual = [_batch_signature(next(chained_iterator)) for _ in range(5)]
    for got, want in zip(chained_actual, chained_expected, strict=True):
        assert got[0] == want[0]
        assert torch.equal(got[1], want[1])
        assert torch.equal(got[2], want[2])


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
