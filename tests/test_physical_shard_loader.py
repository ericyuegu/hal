"""Focused tests for deterministic physical-shard replay loading."""

from __future__ import annotations

import pickle
import threading
import time
from collections.abc import Mapping
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

import hal.training.physical_shard_loader as physical_shard_loader
from hal.training.physical_shard_loader import DecodedShard
from hal.training.physical_shard_loader import PhysicalRow
from hal.training.physical_shard_loader import PhysicalShardReplayLoader
from hal.training.physical_shard_loader import PhysicalShardSelection
from hal.training.physical_shard_loader import ReplayBuffer
from hal.training.physical_shard_loader import ShardTask
from hal.training.physical_shard_loader import SourceManifest
from hal.training.physical_shard_loader import SourceRowSelection
from hal.training.physical_shard_loader import _BalancedReplaySchedule
from hal.training.physical_shard_loader import _decode_generation
from hal.training.physical_shard_loader import _DecodedRows
from hal.training.physical_shard_loader import _shutdown_data_loader_workers
from hal.training.physical_shard_loader import _stack_window_rows
from hal.training.physical_shard_loader import build_shard_plan
from hal.training.physical_shard_loader import choose_generation_window_starts
from hal.training.physical_shard_loader import disk_requirement_bytes
from hal.training.physical_shard_loader import estimate_host_memory
from hal.training.physical_shard_loader import permute_shard_tasks
from hal.wire import ACTION_CHANNELS


def _selection(rows: int = 19) -> PhysicalShardSelection:
    return PhysicalShardSelection(
        sources=(SourceRowSelection("source", rows, (3, 11) if rows > 11 else ()),),
        sha256="a" * 64,
    )


def test_shard_plan_covers_prefix_once_and_excludes_only_sidecar_rows() -> None:
    selection = _selection()
    tasks = build_shard_plan(selection, {"source": SourceManifest("source", (5, 7, 11, 13))})

    exposed = []
    source_offset = 0
    for task in tasks:
        exposed.extend(source_offset + row for row in task.selected_rows)
        source_offset += (5, 7, 11, 13)[task.shard]

    assert exposed == [row for row in range(19) if row not in (3, 11)]
    assert tasks[-1].row_stop == 7
    assert sum(task.row_count for task in tasks) == selection.row_count


def test_shard_permutation_is_deterministic_and_stable_across_epochs() -> None:
    tasks = tuple(ShardTask("source", shard, 0, 2, global_shard=shard) for shard in range(32))

    first = permute_shard_tasks(tasks, seed=7, epoch=3)

    assert first == permute_shard_tasks(tasks, seed=7, epoch=3)
    assert first == permute_shard_tasks(tasks, seed=7, epoch=4)
    assert first != permute_shard_tasks(tasks, seed=8, epoch=3)
    assert sorted(first) == list(range(len(tasks)))


def test_disk_requirement_counts_valid_raw_shards_as_already_used(tmp_path: Path) -> None:
    present = tmp_path / "present.mds"
    present.write_bytes(b"x" * 100)
    manifest = SourceManifest(
        "source",
        (2, 2, 2),
        raw_bytes_per_shard=(100, 200, 300),
        zip_bytes_per_shard=(10, 20, 30),
        raw_paths=(present, tmp_path / "missing-1.mds", tmp_path / "missing-2.mds"),
    )
    tasks = tuple(ShardTask("source", shard, 0, 2) for shard in range(3))

    required = disk_requirement_bytes(tasks, {"source": manifest}, workers=1, reserved_bytes=1_000)

    assert required == 200 + 300 + 30 + 1_000


def test_host_memory_model_includes_every_concurrent_copy() -> None:
    estimate = estimate_host_memory(
        central_buffer_bytes=100,
        decoded_shard_bytes=20,
        replay_workspace_bytes=3,
        pinned_batch_bytes=7,
        validation_cache_bytes=11,
        compiler_and_process_bytes=13,
        workers=4,
    )

    assert estimate.queued_shards == 160
    assert estimate.worker_outputs_and_workspaces == 92
    assert estimate.ipc_copies == 160
    assert estimate.parent_result == 20
    assert estimate.pinned_batches == 14
    assert estimate.peak_bytes == 570


def test_window_starts_are_full_and_distinct() -> None:
    length = 266
    starts = choose_generation_window_starts(273, 256, 10, 8, np.random.default_rng(4))

    assert len(starts) == 8
    assert len(set(map(int, starts))) == 8
    assert all(0 <= int(start) <= 273 - length for start in starts)

    with pytest.raises(ValueError, match="8 windows require"):
        choose_generation_window_starts(272, 256, 10, 8, np.random.default_rng(4))


def test_window_start_selection_is_uniform() -> None:
    frames = 60
    context_length = 8
    chunk_length = 2
    valid_starts = frames - context_length - chunk_length + 1
    counts = np.zeros(valid_starts, dtype=np.int64)
    rng = np.random.default_rng(51)

    for _ in range(20_000):
        starts = choose_generation_window_starts(frames, context_length, chunk_length, 4, rng)
        np.add.at(counts, starts, 1)

    expected = counts.sum() / valid_starts
    chi_square = float(np.sum((counts - expected) ** 2 / expected))
    z_score = (chi_square - (valid_starts - 1)) / np.sqrt(2 * (valid_starts - 1))
    assert abs(z_score) < 5


def test_short_replay_error_identifies_the_physical_row() -> None:
    task = ShardTask("source", 7, 0, 1)

    with pytest.raises(
        ValueError,
        match=("short replay 'replay-9' at source=source shard=7 row=0: frame_count=272, required_count=273"),
    ):
        _decode_generation(
            {"replay_id": "replay-9", "num_frames": 272},
            task=task,
            row=0,
            epoch=0,
            seed=1,
            context_length=256,
            chunk_length=10,
            windows_per_generation=8,
            schema_version=7,
            labels=_no_labels,
            projection=None,
        )


def _decoded(ids: tuple[str, ...], *, epoch: int = 0) -> DecodedShard:
    task = ShardTask("source", 0, 0, len(ids), global_shard=0)
    values = np.arange(len(ids) * 4 * 3, dtype=np.int32).reshape(len(ids), 4, 3)
    return DecodedShard(
        task=task,
        epoch=epoch,
        task_offset=0,
        replay_ids=ids,
        locators=tuple(PhysicalRow("source", 0, row) for row in range(len(ids))),
        columns={"value": values},
        windows_per_generation=4,
    )


def test_decoded_shard_requires_fixed_four_window_columns() -> None:
    task = ShardTask("source", 0, 0, 2)
    with pytest.raises(ValueError, match=r"\[R, 4\]"):
        DecodedShard(
            task,
            0,
            0,
            ("a", "b"),
            (PhysicalRow("source", 0, 0), PhysicalRow("source", 0, 1)),
            {"value": np.zeros((2, 3, 5), dtype=np.float32)},
            4,
        )


def test_window_column_order_does_not_change_schema() -> None:
    windows = (
        (
            {"ego_x": np.ones(2), "opp_x": np.zeros(2)},
            {"opp_x": np.zeros(2), "ego_x": np.ones(2)},
            {"ego_x": np.ones(2), "opp_x": np.zeros(2)},
            {"opp_x": np.zeros(2), "ego_x": np.ones(2)},
        ),
    )

    columns = _stack_window_rows(windows, windows_per_generation=4)

    assert tuple(columns) == ("ego_x", "opp_x")
    assert columns["ego_x"].shape == (1, 4, 2)


def test_decoded_ctx_pad_remains_scalar(monkeypatch: pytest.MonkeyPatch) -> None:
    def decode_slices(
        _compact: Mapping[str, object], ranges: Sequence[tuple[int, int]]
    ) -> tuple[dict[str, np.ndarray], ...]:
        return tuple({"value": np.zeros(stop - start, dtype=np.float32)} for start, stop in ranges)

    def make_test_window(
        sample: dict[str, object],
        *,
        ego_prefix: str,
        start: int,
        pad: int,
        length: int,
        projection: object,
    ) -> dict[str, np.ndarray]:
        del ego_prefix, start, pad, length, projection
        return {"value": np.asarray(sample["value"])}

    monkeypatch.setattr(
        physical_shard_loader,
        "decode_policy_world_replay_slices",
        decode_slices,
    )
    monkeypatch.setattr(physical_shard_loader, "make_window", make_test_window)
    _replay_id, windows = _decode_generation(
        {
            "replay_id": "replay-1",
            "num_frames": 8,
            "source_schema_version": 7,
        },
        task=ShardTask("source", 0, 0, 1),
        row=0,
        epoch=0,
        seed=1,
        context_length=2,
        chunk_length=1,
        windows_per_generation=4,
        schema_version=7,
        labels=_no_labels,
        projection=None,
    )

    columns = _stack_window_rows((windows,), windows_per_generation=4)

    assert columns["ctx_pad"].shape == (1, 4)


def test_replay_buffer_rejects_duplicate_active_identities() -> None:
    buffer = ReplayBuffer(capacity=4, batch_size=4, windows_per_generation=4, seed=9)
    shard = _decoded(("a", "a", "b", "c"))

    with pytest.raises(ValueError, match="repeat a replay identity"):
        buffer.append_rows(_DecodedRows(shard, 0, 4))


def test_full_size_batch_contains_512_distinct_replay_ids() -> None:
    replay_ids = tuple(f"replay-{index}" for index in range(1024))
    buffer = ReplayBuffer(capacity=len(replay_ids), batch_size=512, windows_per_generation=4, seed=10)
    shard = _decoded(replay_ids)
    buffer.append_rows(_DecodedRows(shard, 0, len(replay_ids)))

    sampled, columns, _exhausted = buffer.sample()

    assert len(sampled) == len(set(sampled)) == 512
    assert len(buffer.last_sampled_identity_ranks) == len(set(buffer.last_sampled_identity_ranks)) == 512
    assert min(buffer.last_sampled_identity_ranks) >= 0
    assert max(buffer.last_sampled_identity_ranks) < buffer.last_sampled_identity_count == buffer.active_identities
    slots = np.asarray(buffer.last_sampled_identity_ranks)
    np.testing.assert_array_equal(columns["value"], shard.columns["value"][slots, slots % 4])


def test_replay_age_quantiles_are_absent_until_a_replay_is_reused() -> None:
    replay_ids = tuple(f"replay-{index}" for index in range(8))
    buffer = ReplayBuffer(capacity=8, batch_size=4, windows_per_generation=4, seed=51)
    buffer.append_rows(_DecodedRows(_decoded(replay_ids), 0, len(replay_ids)))
    buffer.next_windows[:] = 0

    buffer.sample()

    assert set(buffer.last_metrics).isdisjoint(
        {
            "data/replay_age_p01",
            "data/replay_age_p05",
            "data/replay_age_p50",
            "data/replay_age_p95",
        }
    )
    buffer.sample()
    buffer.sample()
    assert all(
        np.isfinite(buffer.last_metrics[name])
        for name in (
            "data/replay_age_p01",
            "data/replay_age_p05",
            "data/replay_age_p50",
            "data/replay_age_p95",
        )
    )


def test_count_active_replay_ids_ignores_replaced_generations() -> None:
    loader = object.__new__(PhysicalShardReplayLoader)
    loader._buffer = ReplayBuffer(capacity=512, batch_size=4, windows_per_generation=4, seed=51)
    loader._buffer.identity_slots = {"active": 0, "replacement": 1}

    assert loader.count_active_replay_ids(("active", "retired", "replacement")) == 2


def test_balanced_schedule_has_uniform_exposure_and_a_225_batch_reuse_floor() -> None:
    capacity = 131_072
    batch_size = 512
    schedule = _BalancedReplaySchedule(capacity, batch_size, phases=8, seed=51)
    last_seen = np.full(capacity, -1, dtype=np.int64)
    counts = np.zeros(capacity, dtype=np.int64)
    first_pass: list[frozenset[int]] = []
    second_pass: list[frozenset[int]] = []

    for batch_index in range(2 * schedule.pass_batches):
        slots = schedule.next()
        assert len(slots) == len(np.unique(slots)) == batch_size
        assert np.bincount(slots % 8, minlength=8).tolist() == [64] * 8
        previous = last_seen[slots]
        if np.any(previous >= 0):
            assert np.min(batch_index - previous[previous >= 0]) >= 225
        last_seen[slots] = batch_index
        counts[slots] += 1
        batches = first_pass if batch_index < schedule.pass_batches else second_pass
        batches.append(frozenset(map(int, slots)))

    np.testing.assert_array_equal(counts, np.full(capacity, 2))
    assert all(first != second for first, second in zip(first_pass, second_pass, strict=True))


def test_balanced_schedule_turns_over_exactly_one_phase_per_batch() -> None:
    schedule = _BalancedReplaySchedule(131_072, 512, phases=8, seed=52)
    next_windows = np.arange(schedule.capacity, dtype=np.uint16) % 8

    for _ in range(8 * schedule.pass_batches):
        slots = schedule.next()
        next_windows[slots] += 1
        exhausted = slots[next_windows[slots] == 8]
        assert len(exhausted) == 64
        next_windows[exhausted] = 0


class _FakeAdapter:
    def __init__(self, rows: int, length: int) -> None:
        self.rows = rows
        self.length = length
        self.manifests: Mapping[str, SourceManifest] = {"source": SourceManifest("source", (rows,))}

    def _generation(self, row: int, epoch: int) -> tuple[str, tuple[dict[str, np.ndarray], ...]]:
        replay_id = f"replay-{row}"
        windows = []
        for ordinal in range(4):
            value = np.float32(epoch * 100 + row * 4 + ordinal)
            window = {f"ego_{channel}": np.full(self.length, value, dtype=np.float32) for channel in ACTION_CHANNELS}
            window["ctx_pad"] = np.asarray(0, dtype=np.int64)
            windows.append(window)
        return replay_id, tuple(windows)

    @staticmethod
    def _physical_row(task: ShardTask, row: int) -> int:
        return task.shard * 16 + row

    def decode_task(self, task: ShardTask, epoch: int, task_offset: int, **_kwargs: object) -> DecodedShard:
        generations = [self._generation(self._physical_row(task, row), epoch) for row in task.selected_rows]
        names = tuple(generations[0][1][0])
        columns = {
            name: np.stack([[window[name] for window in generation[1]] for generation in generations])
            for name in names
        }
        return DecodedShard(
            task,
            epoch,
            task_offset,
            tuple(generation[0] for generation in generations),
            tuple(PhysicalRow(task.source, task.shard, row) for row in task.selected_rows),
            columns,
            4,
        )

    def decode_generations(
        self, task: ShardTask, requests: Sequence[tuple[int, int]], **_kwargs: object
    ) -> Mapping[tuple[int, int], tuple[str, tuple[dict[str, np.ndarray], ...]]]:
        return {request: self._generation(self._physical_row(task, request[0]), request[1]) for request in requests}


class _DelayedFakeAdapter(_FakeAdapter):
    def __init__(self, rows: int, length: int) -> None:
        super().__init__(rows, length)
        self.manifests = {"source": SourceManifest("source", (16, 16, 16, 16))}

    def decode_task(self, task: ShardTask, epoch: int, task_offset: int, **kwargs: object) -> DecodedShard:
        time.sleep(0.01 * (3 - task.shard))
        return super().decode_task(task, epoch, task_offset, **kwargs)


class _BlockingFakeAdapter(_FakeAdapter):
    def __init__(self, rows: int, length: int, started: threading.Event, release: threading.Event) -> None:
        super().__init__(rows, length)
        self.started = started
        self.release = release

    def decode_task(self, task: ShardTask, epoch: int, task_offset: int, **kwargs: object) -> DecodedShard:
        self.started.set()
        if not self.release.wait(timeout=5):
            raise TimeoutError("test did not release the blocking adapter")
        return super().decode_task(task, epoch, task_offset, **kwargs)


def _no_labels(_row: Mapping[str, object]) -> dict[str, np.ndarray]:
    return {}


@dataclass(frozen=True, slots=True)
class _Batch:
    replay_ids: tuple[str, ...]
    values: torch.Tensor

    def pin_memory(self) -> _Batch:
        return _Batch(self.replay_ids, self.values.pin_memory())


def _collate_batch(replay_ids: tuple[str, ...], columns: Mapping[str, np.ndarray]) -> _Batch:
    return _Batch(replay_ids, torch.from_numpy(columns["ego_main_stick_x"].copy()))


def _loader(seed: int, *, workers: int = 0, delayed: bool = False) -> PhysicalShardReplayLoader[_Batch]:
    rows = 64
    selection = PhysicalShardSelection(
        sources=(SourceRowSelection("source", rows),),
        sha256="b" * 64,
    )
    tasks = (
        tuple(ShardTask("source", shard, 0, 16, global_shard=shard) for shard in range(4))
        if delayed
        else (ShardTask("source", 0, 0, rows, global_shard=0),)
    )
    adapter_type = _DelayedFakeAdapter if delayed else _FakeAdapter
    adapter = adapter_type(rows, 5)
    return PhysicalShardReplayLoader[_Batch](
        selection=selection,
        adapter=adapter,
        tasks=tasks,
        data_protocol="test-physical-shard-v1",
        source_manifest_sha256={"source": "c" * 64},
        labels=_no_labels,
        projection=None,
        batch_transform=_collate_batch,
        batch_size=4,
        replay_slots=16,
        seed=seed,
        num_workers=workers,
        context_length=3,
        chunk_length=2,
        windows_per_generation=4,
        schema_version=7,
        reserved_disk_bytes=0,
        pin_memory=False,
    )


def test_exact_resume_reproduces_identity_sequences_and_tensors() -> None:
    original = _loader(seed=17)
    original_iterator = iter(original)
    for _ in range(7):
        next(original_iterator)
    state = original.state_dict()
    assert set(state) == {
        "schema",
        "data_protocol",
        "source_selection_sha256",
        "source_manifest_sha256",
        "cursor",
        "batch_sampler_state",
        "slots",
        "buffer_geometry",
    }
    assert not any(isinstance(value, np.ndarray) for value in state.values())
    expected = [next(original_iterator) for _ in range(32)]

    restored = _loader(seed=17)
    restored.load_state_dict(state)
    restored_iterator = iter(restored)
    actual = [next(restored_iterator) for _ in range(32)]

    for left, right in zip(expected, actual, strict=True):
        assert left.replay_ids == right.replay_ids
        torch.testing.assert_close(left.values, right.values)


def test_every_batch_contains_distinct_replay_ids() -> None:
    loader = _loader(seed=31)
    iterator = iter(loader)

    for _ in range(40):
        batch = next(iterator)
        assert batch.replay_ids is not None
        assert len(batch.replay_ids) == len(set(batch.replay_ids)) == 4


def test_delayed_workers_and_worker_count_change_preserve_exact_resume() -> None:
    original = _loader(seed=43, workers=2, delayed=True)
    original_iterator = iter(original)
    for _ in range(5):
        next(original_iterator)
    state = original.state_dict()
    expected = [next(original_iterator) for _ in range(8)]
    original.close()

    restored = _loader(seed=43, workers=0, delayed=True)
    restored.load_state_dict(state)
    restored_iterator = iter(restored)
    actual = [next(restored_iterator) for _ in range(8)]

    for left, right in zip(expected, actual, strict=True):
        assert left.replay_ids == right.replay_ids
        torch.testing.assert_close(left.values, right.values)


def test_iterator_must_start_on_main_thread() -> None:
    loader = _loader(seed=1)
    errors: list[BaseException] = []

    def start() -> None:
        try:
            iter(loader)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=start)
    thread.start()
    thread.join()

    assert len(errors) == 1
    assert "main thread" in str(errors[0])


def test_zero_worker_iterator_does_not_query_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    loader = _loader(seed=2)

    def fail() -> bool:
        raise AssertionError("loader queried CUDA state")

    monkeypatch.setattr(torch.cuda, "is_available", fail)
    monkeypatch.setattr(torch.cuda, "is_initialized", fail)

    next(iter(loader))


def test_spawn_workers_can_start_after_cuda_initialization(monkeypatch: pytest.MonkeyPatch) -> None:
    loader = _loader(seed=2, workers=2, delayed=True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)

    next(iter(loader))
    loader.close()


def test_checkpoint_rejects_an_active_parent_next() -> None:
    started = threading.Event()
    release = threading.Event()
    loader = _loader(seed=3)
    loader.adapter = _BlockingFakeAdapter(8, 5, started, release)
    iterator = iter(loader)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(next, iterator)
        assert started.wait(timeout=5)
        with pytest.raises(RuntimeError, match=r"parent-side next\(\) is active"):
            loader.state_dict()
        release.set()
        future.result(timeout=5)


def test_old_loader_schema_is_rejected() -> None:
    loader = _loader(seed=4)

    with pytest.raises(ValueError, match="unsupported physical-shard loader schema"):
        loader.load_state_dict({"schema": 0})


def test_context_manager_closes_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object | None] = []

    def record_shutdown(iterator: object | None) -> None:
        calls.append(iterator)

    monkeypatch.setattr(
        "hal.training.physical_shard_loader._shutdown_data_loader_workers",
        record_shutdown,
    )
    loader = _loader(seed=5)
    with loader:
        next(iter(loader))

    loader.close()
    assert len(calls) == 1
    with pytest.raises(RuntimeError, match="closed"):
        iter(loader)


def test_private_worker_shutdown_isolated_in_one_function() -> None:
    calls = 0

    class IteratorWithWorkers:
        def _shutdown_workers(self) -> None:
            nonlocal calls
            calls += 1

    _shutdown_data_loader_workers(IteratorWithWorkers())
    _shutdown_data_loader_workers(None)

    assert calls == 1


def test_legacy_pickle_resolves_physical_row_through_shim() -> None:
    payload = b"chal.training.o51_replay_loader\nPhysicalRow\n(Vsource\nI2\nI3\ntR."

    row = pickle.loads(payload)

    assert row == PhysicalRow("source", 2, 3)
