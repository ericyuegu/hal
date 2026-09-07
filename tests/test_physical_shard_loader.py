"""Focused tests for deterministic physical-shard replay loading."""

from __future__ import annotations

import pickle
import threading
import time
from collections import deque
from collections.abc import Mapping
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import numpy as np
import pytest
import torch

import hal.training.physical_shard_loader as physical_shard_loader
from hal.training.physical_shard_loader import MAX_DECODE_CHUNK_ROWS
from hal.training.physical_shard_loader import DecodedChunk
from hal.training.physical_shard_loader import PhysicalRow
from hal.training.physical_shard_loader import PhysicalShardReplayLoader
from hal.training.physical_shard_loader import PhysicalShardSelection
from hal.training.physical_shard_loader import ShardTask
from hal.training.physical_shard_loader import SourceManifest
from hal.training.physical_shard_loader import SourceRowSelection
from hal.training.physical_shard_loader import _ChunkSampler
from hal.training.physical_shard_loader import _decode_generation
from hal.training.physical_shard_loader import _DecodeChunkRequest
from hal.training.physical_shard_loader import _OrderedChunks
from hal.training.physical_shard_loader import _ReplayRing
from hal.training.physical_shard_loader import _ReplayRingSchedule
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
        decoded_chunk_bytes=20,
        replay_workspace_bytes=3,
        pinned_batch_bytes=7,
        pinned_batch_count=4,
        validation_cache_bytes=11,
        compiler_and_process_bytes=13,
        workers=4,
    )

    assert estimate.queued_chunks == 160
    assert estimate.worker_outputs_and_workspaces == 92
    assert estimate.ipc_copies == 160
    assert estimate.parent_result == 20
    assert estimate.pinned_batches == 28
    assert estimate.peak_bytes == 584


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


def _decoded(
    ids: tuple[str, ...],
    *,
    epoch: int = 0,
    sequence: int = 0,
    row_start: int = 0,
) -> DecodedChunk:
    task = ShardTask("source", 0, row_start, row_start + len(ids), global_shard=0)
    request = _DecodeChunkRequest(
        sequence=sequence,
        epoch=epoch,
        task_offset=0,
        task_index=0,
        row_offset=0,
        rows=task.selected_rows,
    )
    values = np.arange(len(ids) * 4 * 3, dtype=np.int32).reshape(len(ids), 4, 3)
    return DecodedChunk(
        request=request,
        task=task,
        replay_ids=ids,
        locators=tuple(PhysicalRow("source", 0, row) for row in task.selected_rows),
        columns={"value": values},
        windows_per_generation=4,
    )


def test_decoded_chunk_requires_fixed_four_window_columns() -> None:
    task = ShardTask("source", 0, 0, 2)
    request = _DecodeChunkRequest(0, 0, 0, 0, 0, (0, 1))
    with pytest.raises(ValueError, match=r"\[R, 4\]"):
        DecodedChunk(
            request,
            task,
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


def test_replay_ring_uses_the_derived_slots_and_window_ordinals() -> None:
    ring = _ReplayRing(
        capacity=100,
        batch_size=4,
        windows_per_generation=4,
        phase_block_batches=25,
        seed=10,
    )
    ring.append_chunk(_decoded(tuple(f"replay-{index}" for index in range(64))))
    ring.append_chunk(_decoded(tuple(f"replay-{index}" for index in range(64, 100)), row_start=64))

    replay_ids, columns = ring.sample()
    slots = ring.schedule.selected_slots(0)

    assert len(replay_ids) == len(set(replay_ids)) == 4
    np.testing.assert_array_equal(
        columns["value"],
        ring.columns["value"][slots, ring.schedule.window_ordinals],
    )
    assert min(ring.schedule.reuse_gaps) >= 1
    assert max(ring.schedule.reuse_gaps) <= 49


@pytest.mark.parametrize("seed", [0, 51])
def test_u1_ring_schedule_has_randomized_reuse_and_zero_period_overlap(seed: int) -> None:
    schedule = _ReplayRingSchedule(114_688, 512, 8, 25, seed)
    replay_by_slot = np.arange(schedule.capacity, dtype=np.int64)
    recent: deque[set[int]] = deque()
    previous: set[int] = set()
    first_replacements = set(range(schedule.capacity, schedule.capacity + schedule.replay_lanes))
    appearances = {replay_id: [] for replay_id in first_replacements}
    fifo_head = 0
    next_replay = schedule.capacity

    assert schedule.replay_lanes == 64
    assert schedule.period_batches == 224
    assert schedule.minimum_gap_batches == 200
    assert schedule.maximum_gap_batches == 248
    assert all(len(set(map(int, offsets))) == 8 for offsets in schedule.phase_offsets)

    for batch_index in range(schedule.cohort_count):
        slots = schedule.selected_slots(fifo_head)
        replay_ids = replay_by_slot[slots]
        current = set(map(int, replay_ids))
        assert len(current) == 512
        assert not current & previous
        if len(recent) == schedule.period_batches:
            assert not current & recent.popleft()
        recent.append(current)
        previous = current
        for replay_id in current & first_replacements:
            appearances[replay_id].append(batch_index)

        start = fifo_head * schedule.replay_lanes
        stop = start + schedule.replay_lanes
        replay_by_slot[start:stop] = np.arange(next_replay, next_replay + schedule.replay_lanes)
        next_replay += schedule.replay_lanes
        fifo_head = (fifo_head + 1) % schedule.cohort_count

    for batches in appearances.values():
        assert len(batches) == 8
        assert all(200 <= gap <= 248 for gap in np.diff(batches))
    assert next_replay - schedule.capacity == schedule.cohort_count * 64


def test_chunk_sampler_splits_cohorts_at_shard_and_corpus_boundaries() -> None:
    tasks = (
        ShardTask("source", 0, 0, 50),
        ShardTask("source", 1, 0, 50),
    )
    sampler = _ChunkSampler(tasks, seed=3, cursor=(0, 0, 0), chunk_rows=64)

    requests = list(islice(sampler, 4))

    assert [len(request.rows) for request in requests] == [50, 14, 36, 28]
    assert [request.sequence for request in requests] == [0, 1, 2, 3]
    assert max(len(request.rows) for request in requests) <= MAX_DECODE_CHUNK_ROWS
    assert requests[3].epoch == 1

    large_cohort = _ChunkSampler(
        (ShardTask("source", 0, 0, 256),),
        seed=3,
        cursor=(0, 0, 0),
        chunk_rows=128,
    )
    assert [len(request.rows) for request in islice(large_cohort, 2)] == [64, 64]


def test_ordered_chunks_restore_delayed_worker_order() -> None:
    chunks = [_decoded((f"replay-{sequence}",), sequence=sequence) for sequence in (2, 0, 1)]

    ordered = list(islice(_OrderedChunks(iter(chunks)), 3))

    assert [chunk.request.sequence for chunk in ordered] == [0, 1, 2]


class _FakeAdapter:
    def __init__(self, rows: int, length: int) -> None:
        self.rows = rows
        self.length = length
        self.manifests: Mapping[str, SourceManifest] = {"source": SourceManifest("source", (rows,))}

    def _generation(
        self,
        task: ShardTask,
        row: int,
        epoch: int,
    ) -> tuple[str, tuple[dict[str, np.ndarray], ...]]:
        replay_id = f"replay-{task.shard}-{row}"
        windows = []
        for ordinal in range(4):
            value = np.float32(epoch * 10_000 + task.shard * 1_000 + row * 4 + ordinal)
            window = {f"ego_{channel}": np.full(self.length, value, dtype=np.float32) for channel in ACTION_CHANNELS}
            window["ctx_pad"] = np.asarray(0, dtype=np.int64)
            windows.append(window)
        return replay_id, tuple(windows)

    def decode_chunk(
        self,
        request: _DecodeChunkRequest,
        task: ShardTask,
        **_kwargs: object,
    ) -> DecodedChunk:
        generations = [self._generation(task, row, request.epoch) for row in request.rows]
        names = tuple(generations[0][1][0])
        columns = {
            name: np.stack([[window[name] for window in generation[1]] for generation in generations])
            for name in names
        }
        return DecodedChunk(
            request=request,
            task=task,
            replay_ids=tuple(generation[0] for generation in generations),
            locators=tuple(PhysicalRow(task.source, task.shard, row) for row in request.rows),
            columns=columns,
            windows_per_generation=4,
        )

    def decode_generations(
        self, task: ShardTask, requests: Sequence[tuple[int, int]], **_kwargs: object
    ) -> Mapping[tuple[int, int], tuple[str, tuple[dict[str, np.ndarray], ...]]]:
        return {request: self._generation(task, request[0], request[1]) for request in requests}


class _DelayedFakeAdapter(_FakeAdapter):
    def __init__(self, rows: int, length: int) -> None:
        super().__init__(rows, length)
        base, remainder = divmod(rows, 4)
        sizes = tuple(base + (shard < remainder) for shard in range(4))
        self.manifests = {"source": SourceManifest("source", sizes)}

    def decode_chunk(
        self,
        request: _DecodeChunkRequest,
        task: ShardTask,
        **kwargs: object,
    ) -> DecodedChunk:
        time.sleep(0.005 * (3 - task.shard))
        return super().decode_chunk(request, task, **kwargs)


class _BlockingFakeAdapter(_FakeAdapter):
    def __init__(self, rows: int, length: int, started: threading.Event, release: threading.Event) -> None:
        super().__init__(rows, length)
        self.started = started
        self.release = release

    def decode_chunk(
        self,
        request: _DecodeChunkRequest,
        task: ShardTask,
        **kwargs: object,
    ) -> DecodedChunk:
        self.started.set()
        if not self.release.wait(timeout=5):
            raise TimeoutError("test did not release the blocking adapter")
        return super().decode_chunk(request, task, **kwargs)


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


def _loader(
    seed: int,
    *,
    rows: int = 113,
    workers: int = 0,
    delayed: bool = False,
) -> PhysicalShardReplayLoader[_Batch]:
    selection = PhysicalShardSelection(
        sources=(SourceRowSelection("source", rows),),
        sha256="b" * 64,
    )
    adapter_type = _DelayedFakeAdapter if delayed else _FakeAdapter
    adapter = adapter_type(rows, 5)
    sizes = adapter.manifests["source"].samples_per_shard
    tasks = tuple(ShardTask("source", shard, 0, size, global_shard=shard) for shard, size in enumerate(sizes))
    return PhysicalShardReplayLoader[_Batch](
        selection=selection,
        adapter=adapter,
        tasks=tasks,
        data_protocol="test-physical-shard-v2",
        source_manifest_sha256={"source": "c" * 64},
        labels=_no_labels,
        projection=None,
        batch_transform=_collate_batch,
        batch_size=4,
        replay_slots=100,
        seed=seed,
        num_workers=workers,
        context_length=3,
        chunk_length=2,
        windows_per_generation=4,
        replay_phase_block_batches=25,
        schema_version=7,
        reserved_disk_bytes=0,
        pin_memory=False,
    )


def test_exact_resume_reproduces_identity_sequences_and_tensors() -> None:
    original = _loader(seed=17)
    original_iterator = iter(original)
    for _ in range(15):
        next(original_iterator)
    state = original.state_dict()
    assert set(state) == {
        "schema",
        "data_protocol",
        "source_selection_sha256",
        "source_manifest_sha256",
        "cursor",
        "fifo_head",
        "batch_index",
        "slots",
        "buffer_geometry",
    }
    assert not any(isinstance(value, np.ndarray) for value in state.values())
    old_protocol = {**state, "data_protocol": "test-physical-shard-v0"}
    with pytest.raises(ValueError, match="data protocol"):
        _loader(seed=17).load_state_dict(old_protocol)
    expected = [next(original_iterator) for _ in range(32)]

    restored = _loader(seed=17)
    restored.load_state_dict(state)
    restored_iterator = iter(restored)
    actual = [next(restored_iterator) for _ in range(32)]

    for left, right in zip(expected, actual, strict=True):
        assert left.replay_ids == right.replay_ids
        torch.testing.assert_close(left.values, right.values)


def test_resume_reproduces_the_next_optimizer_update() -> None:
    original = _loader(seed=19)
    iterator = iter(original)
    for _ in range(37):
        next(iterator)
    state = original.state_dict()
    expected_batch = next(iterator)

    restored = _loader(seed=19)
    restored.load_state_dict(state)
    actual_batch = next(iter(restored))

    def update(batch: _Batch) -> torch.Tensor:
        weight = torch.nn.Parameter(torch.tensor(0.25))
        optimizer = torch.optim.SGD([weight], lr=0.01)
        loss = (weight * batch.values.float().mean()).square()
        loss.backward()
        optimizer.step()
        return weight.detach()

    torch.testing.assert_close(update(actual_batch), update(expected_batch))


def test_every_batch_contains_distinct_replay_ids() -> None:
    loader = _loader(seed=31)
    iterator = iter(loader)

    for _ in range(300):
        batch = next(iterator)
        assert batch.replay_ids is not None
        assert len(batch.replay_ids) == len(set(batch.replay_ids)) == 4

    assert loader.metrics["data/decoded_generations"] == loader.decoded_generations == 400
    assert loader.max_decoded_chunk_size == 1


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
    loader.adapter = _BlockingFakeAdapter(113, 5, started, release)
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
