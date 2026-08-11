from collections.abc import Iterator
from threading import Event

import numpy as np
import pytest

import hal.training.replay_reservoir as replay_reservoir
from hal.training.replay_reservoir import OneBatchPrefetch
from hal.training.replay_reservoir import PolicyReplayPackDataset
from hal.training.replay_reservoir import ReplayPack
from hal.training.replay_reservoir import ReplayReservoir
from hal.training.replay_reservoir import _stable_replay_rng
from hal.training.replay_reservoir import make_reservoir_loader


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
