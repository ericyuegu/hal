from collections.abc import Iterator
from threading import Event

import numpy as np
import pytest

from hal.training.replay_reservoir import OneBatchPrefetch
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


# %% Annotation and batch-transform seams


def _compact_row(replay_id: str, frames: int) -> dict[str, object]:
    from hal.data.policy_schema import POLICY_MDS_COLUMNS
    from hal.data.policy_schema import POLICY_SCHEMA_VERSION

    row: dict[str, object] = {
        "policy_schema_version": POLICY_SCHEMA_VERSION,
        "source_schema_version": 7,
        "replay_id": replay_id,
        "num_frames": frames,
        "stage": 2,
        "p1_character": 1,
        "p2_character": 22,
        "p1_nana_present": 0,
        "p2_nana_present": 0,
    }
    for name, encoding in POLICY_MDS_COLUMNS.items():
        if name in row:
            continue
        dtype = np.dtype(encoding.removeprefix("ndarray:"))
        row[name] = np.zeros(1 if "nana" in name else frames, dtype=dtype)
    row["p1_position_x"] = np.arange(frames, dtype=np.float32)
    row["p2_position_x"] = np.arange(frames, dtype=np.float32) + 1000.0
    return row


def _pack_windows(rows: list[dict[str, object]], annotate) -> list[tuple[str, tuple[dict, ...]]]:
    from hal.training.replay_reservoir import PolicyReplayPackDataset

    dataset = PolicyReplayPackDataset(
        rows,  # type: ignore[arg-type]  # duck-typed iterable of compact rows
        16,
        4,
        seed=3,
        windows_per_replay=2,
        schema_version=7,
        projection=None,
        annotate_replay=annotate,
    )
    return [(pack.replay_id, pack.windows) for pack in dataset]


def test_annotation_seam_changes_no_sampling_and_slices_like_source_columns() -> None:
    def annotate(compact: dict) -> dict[str, np.ndarray]:
        frames = int(compact["num_frames"])
        return {
            "p1_marker": np.arange(frames, dtype=np.float32),
            "p2_marker": np.arange(frames, dtype=np.float32) + 1000.0,
        }

    rows = [_compact_row("a", 64), _compact_row("b", 96)]
    plain = _pack_windows(rows, None)
    labeled = _pack_windows(rows, annotate)

    assert [(replay_id, len(windows)) for replay_id, windows in plain] == [
        (replay_id, len(windows)) for replay_id, windows in labeled
    ]
    for (_, plain_windows), (_, labeled_windows) in zip(plain, labeled, strict=True):
        for before, after in zip(plain_windows, labeled_windows, strict=True):
            assert set(after) == set(before) | {"ego_marker", "opp_marker"}
            for key, value in before.items():
                np.testing.assert_array_equal(np.asarray(after[key]), np.asarray(value), err_msg=key)
            np.testing.assert_array_equal(after["ego_marker"], after["ego_position_x"])
            np.testing.assert_array_equal(after["opp_marker"], after["opp_position_x"])


def test_annotation_rejects_wrong_length_columns() -> None:
    def annotate(compact: dict) -> dict[str, np.ndarray]:
        return {"p1_marker": np.zeros(int(compact["num_frames"]) - 1, dtype=np.float32)}

    with pytest.raises(ValueError, match="annotation"):
        _pack_windows([_compact_row("a", 64)], annotate)


def _action_windows(replay: int, count: int, length: int = 20) -> tuple[dict, ...]:
    from hal.training.features import ACTION_CHANNELS

    windows = []
    for index in range(count):
        window = {f"ego_{name}": np.zeros(length, dtype=np.float32) for name in ACTION_CHANNELS}
        window["ctx_pad"] = np.int64(0)
        window["marker"] = np.full(length, replay * 10 + index, dtype=np.float32)
        windows.append(window)
    return tuple(windows)


def test_batch_transform_wraps_batches_and_receives_source_windows() -> None:
    from hal.training.replay_reservoir import ReservoirLoader

    packs = [ReplayPack(str(replay), _action_windows(replay, 2)) for replay in range(4)]

    def transform(windows, batch):
        return {"windows": windows, "batch": batch}

    loader = ReservoirLoader(
        packs,  # type: ignore[arg-type]  # duck-typed in place of a DataLoader
        stats={},
        L_ctx=16,
        batch_size=2,
        capacity=4,
        seed=0,
        extra=None,
        projection=None,
        prefetch_batches=0,
        pin_memory=False,
        batch_transform=transform,
    )
    batches = list(iter(loader))
    assert batches
    for item in batches:
        assert set(item) == {"windows", "batch"}
        assert len(item["windows"]) == 2
        markers = {int(window["marker"][0]) for window in item["windows"]}
        assert len(markers) == 2
        assert item["batch"].target.shape[0] == 2


def test_default_seams_leave_loader_output_unchanged() -> None:
    from hal.training.features import TrainBatch
    from hal.training.replay_reservoir import ReservoirLoader

    packs = [ReplayPack(str(replay), _action_windows(replay, 2)) for replay in range(4)]
    loader = ReservoirLoader(
        packs,  # type: ignore[arg-type]
        stats={},
        L_ctx=16,
        batch_size=2,
        capacity=4,
        seed=0,
        extra=None,
        projection=None,
        prefetch_batches=0,
        pin_memory=False,
    )
    for batch in iter(loader):
        assert isinstance(batch, TrainBatch)
