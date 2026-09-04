"""Focused tests for O51's shard-owned replay loader."""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import torch

from hal.training.o51_data import SourceSlice
from hal.training.o51_data import TierSelection
from hal.training.o51_data import corpus_selection
from hal.training.o51_replay_loader import DecodedShard
from hal.training.o51_replay_loader import O51ReplayLoader
from hal.training.o51_replay_loader import PhysicalRow
from hal.training.o51_replay_loader import ReplayBuffer
from hal.training.o51_replay_loader import ShardTask
from hal.training.o51_replay_loader import SourceManifest
from hal.training.o51_replay_loader import _decode_generation
from hal.training.o51_replay_loader import _stack_window_rows
from hal.training.o51_replay_loader import build_shard_plan
from hal.training.o51_replay_loader import choose_o51_window_starts
from hal.training.o51_replay_loader import disk_requirement_bytes
from hal.training.o51_replay_loader import estimate_host_memory
from hal.training.o51_replay_loader import permute_shard_tasks
from hal.wire import ACTION_CHANNELS


def _tier(rows: int = 19) -> TierSelection:
    return TierSelection(
        scale=1,
        sources=(SourceSlice("source", rows, (3, 11) if rows > 11 else ()),),
        potential_targets=2,
        sha256="a" * 64,
    )


def test_shard_plan_covers_prefix_once_and_excludes_only_sidecar_rows() -> None:
    tier = _tier()
    tasks = build_shard_plan(tier, {"source": SourceManifest("source", (5, 7, 11, 13))})

    exposed = []
    source_offset = 0
    for task in tasks:
        exposed.extend(source_offset + row for row in task.selected_rows)
        source_offset += (5, 7, 11, 13)[task.shard]

    assert exposed == [row for row in range(19) if row not in (3, 11)]
    assert tasks[-1].row_stop == 7
    assert sum(task.row_count for task in tasks) == tier.unique_replays


def test_full_corpus_plan_contains_exactly_the_two_pinned_exclusions() -> None:
    tier = corpus_selection().tier(8)
    manifests = {view.source: SourceManifest(view.source, (view.stop,)) for view in tier.sources}

    tasks = build_shard_plan(tier, manifests)

    assert sum(len(task.excluded_rows) for task in tasks) == 2
    assert sum(task.row_count for task in tasks) == tier.unique_replays


def test_shard_permutation_is_deterministic_and_changes_by_epoch() -> None:
    tasks = tuple(ShardTask("source", shard, 0, 2, global_shard=shard) for shard in range(32))

    first = permute_shard_tasks(tasks, seed=7, epoch=3)

    assert first == permute_shard_tasks(tasks, seed=7, epoch=3)
    assert first != permute_shard_tasks(tasks, seed=7, epoch=4)
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

    assert estimate.queued_shards == 80
    assert estimate.worker_outputs_and_workspaces == 92
    assert estimate.ipc_copies == 80
    assert estimate.parent_result == 20
    assert estimate.pinned_batches == 14
    assert estimate.peak_bytes == 410


def test_circular_windows_are_full_and_pairwise_non_overlapping() -> None:
    length = 266
    starts = choose_o51_window_starts(1329, 256, 10, np.random.default_rng(4))

    assert len(starts) == 4
    assert len(set(map(int, starts))) == 4
    assert all(0 <= int(start) <= 1329 - length for start in starts)
    intervals = sorted((int(start), int(start) + length) for start in starts)
    assert all(left[1] <= right[0] for left, right in zip(intervals, intervals[1:], strict=False))

    with pytest.raises(ValueError, match="four windows require"):
        choose_o51_window_starts(1328, 256, 10, np.random.default_rng(4))


def test_short_replay_error_identifies_the_physical_row() -> None:
    task = ShardTask("source", 7, 0, 1)

    with pytest.raises(
        ValueError,
        match=("short replay 'replay-9' at source=source shard=7 row=0: frame_count=1328, required_count=1329"),
    ):
        _decode_generation(
            {"replay_id": "replay-9", "num_frames": 1328},
            task=task,
            row=0,
            epoch=0,
            seed=1,
            L_ctx=256,
            L_chunk=10,
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

    columns = _stack_window_rows(windows)

    assert tuple(columns) == ("ego_x", "opp_x")
    assert columns["ego_x"].shape == (1, 4, 2)


def test_duplicate_generations_do_not_change_identity_sampling_weight() -> None:
    buffer = ReplayBuffer(capacity=4, batch_size=1, seed=9)
    shard = _decoded(("a", "a", "b", "c"))
    for row in range(4):
        buffer.admit(shard, row)

    counts = {"a": 0, "b": 0, "c": 0}
    for _ in range(30_000):
        replay_ids, _windows, _exhausted = buffer.sample()
        counts[replay_ids[0]] += 1
        buffer.next_windows[:] = 0

    expected = 10_000
    assert all(abs(count - expected) < 500 for count in counts.values())


def test_full_size_batch_contains_512_distinct_replay_ids() -> None:
    replay_ids = tuple(f"replay-{index}" for index in range(1024))
    buffer = ReplayBuffer(capacity=len(replay_ids), batch_size=512, seed=10)
    shard = _decoded(replay_ids)
    for row in range(len(replay_ids)):
        buffer.admit(shard, row)

    sampled, _windows, _exhausted = buffer.sample()

    assert len(sampled) == len(set(sampled)) == 512
    assert len(buffer.last_sampled_identity_ranks) == len(set(buffer.last_sampled_identity_ranks)) == 512
    assert min(buffer.last_sampled_identity_ranks) >= 0
    assert max(buffer.last_sampled_identity_ranks) < buffer.last_sampled_identity_count == buffer.active_identities


def test_count_active_replay_ids_ignores_replaced_generations() -> None:
    loader = object.__new__(O51ReplayLoader)
    loader._buffer = ReplayBuffer(capacity=512, batch_size=1, seed=51)
    loader._buffer.identity_slots = {"active": {0}, "duplicate": {1, 2}}

    assert loader.count_active_replay_ids(("active", "retired", "duplicate")) == 2


class _FakeAdapter:
    def __init__(self, rows: int, length: int) -> None:
        self.rows = rows
        self.length = length

    def _generation(self, row: int, epoch: int) -> tuple[str, tuple[dict[str, np.ndarray], ...]]:
        replay_id = f"replay-{row}"
        windows = []
        for ordinal in range(4):
            value = np.float32(epoch * 100 + row * 4 + ordinal)
            window = {f"ego_{channel}": np.full(self.length, value, dtype=np.float32) for channel in ACTION_CHANNELS}
            window["ctx_pad"] = np.asarray(0, dtype=np.int64)
            windows.append(window)
        return replay_id, tuple(windows)

    def decode_task(self, task: ShardTask, epoch: int, task_offset: int, **_kwargs: object) -> DecodedShard:
        generations = [self._generation(row, epoch) for row in task.selected_rows]
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
        )

    def decode_generations(
        self, task: ShardTask, requests: list[tuple[int, int]], **_kwargs: object
    ) -> Mapping[tuple[int, int], tuple[str, tuple[dict[str, np.ndarray], ...]]]:
        del task
        return {request: self._generation(*request) for request in requests}


class _DelayedFakeAdapter(_FakeAdapter):
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


def _loader(seed: int, *, workers: int = 0, delayed: bool = False) -> O51ReplayLoader:
    rows = 8
    tier = TierSelection(
        scale=1,
        sources=(SourceSlice("source", rows),),
        potential_targets=2,
        sha256="b" * 64,
    )
    tasks = (
        tuple(ShardTask("source", shard, 0, 2, global_shard=shard) for shard in range(4))
        if delayed
        else (ShardTask("source", 0, 0, rows, global_shard=0),)
    )
    adapter_type = _DelayedFakeAdapter if delayed else _FakeAdapter
    return O51ReplayLoader(
        tier=tier,
        adapter=adapter_type(rows, 5),  # type: ignore[arg-type]
        tasks=tasks,
        stats={},
        labels=_no_labels,
        projection=None,
        batch_size=2,
        replay_slots=rows,
        seed=seed,
        num_workers=workers,
        L_ctx=3,
        L_chunk=2,
        schema_version=7,
        extra=None,
        batch_transform=None,
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
        "batch_sampler_rng_state",
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
        torch.testing.assert_close(left.target, right.target)
        assert left.context.ctx_pad.equal(right.context.ctx_pad)


def test_every_batch_contains_distinct_replay_ids() -> None:
    loader = _loader(seed=31)
    iterator = iter(loader)

    for _ in range(40):
        batch = next(iterator)
        assert batch.replay_ids is not None
        assert len(batch.replay_ids) == len(set(batch.replay_ids)) == 2


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
        torch.testing.assert_close(left.target, right.target)


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


def test_iterator_rejects_cuda_initialized_before_worker_start(monkeypatch) -> None:
    loader = _loader(seed=2)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)

    with pytest.raises(RuntimeError, match="before CUDA initialization"):
        iter(loader)


def test_checkpoint_rejects_an_active_parent_next() -> None:
    started = threading.Event()
    release = threading.Event()
    loader = _loader(seed=3)
    loader.adapter = _BlockingFakeAdapter(8, 5, started, release)  # type: ignore[assignment]
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

    with pytest.raises(ValueError, match="unsupported O51 loader schema"):
        loader.load_state_dict({"schema": 0})
