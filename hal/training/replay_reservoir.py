"""Replay-aware batching for compact policy datasets."""

import hashlib
import os
from collections import Counter
from collections import deque
from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
from concurrent.futures import CancelledError
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Condition
from threading import Lock
from threading import Thread
from typing import Any

import numpy as np
import torch
from streaming import StreamingDataset
from streaming.base.batching import generate_work
from streaming.base.world import World
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset
from torch.utils.data import Sampler

from hal.data.feature_stats import FeatureStats
from hal.data.policy_schema import decode_policy_replay
from hal.data.policy_schema import decode_policy_replay_slices
from hal.data.policy_world_schema import decode_policy_world_replay
from hal.data.policy_world_schema import decode_policy_world_replay_slices
from hal.data.schema import SCHEMA_VERSION
from hal.data.schema import check_schema_version
from hal.streams import StreamSource
from hal.training.dataloader import ReplayFormat
from hal.training.dataloader import StreamSamplePrefix
from hal.training.dataloader import _make_streaming_dataset
from hal.training.features import ExtraColumns
from hal.training.features import FeatureProjection
from hal.training.features import TrainBatch

type WindowTransform = Callable[[str, str, dict[str, np.ndarray]], None]
type ReplayFilter = Callable[[str], bool]
type ReplayLabels = Callable[[Mapping[str, object]], dict[str, np.ndarray]]


def _next_item[T](source: Iterator[T]) -> T:
    return next(source)


def _identity[T](value: T) -> T:
    return value


class OneBatchPrefetch[T](Iterator[T]):
    """Advance an iterator on one background thread."""

    def __init__(self, source: Iterator[T], *, depth: int = 1) -> None:
        if not isinstance(depth, int) or isinstance(depth, bool) or depth <= 0:
            raise ValueError(f"prefetch depth must be a positive integer, got {depth!r}")
        self._source = source
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="train-batch")
        # One worker preserves source order. Queued work fills the requested depth.
        self._futures = deque(self._pool.submit(_next_item, source) for _ in range(depth))
        self._closed = False

    def __iter__(self) -> OneBatchPrefetch[T]:
        return self

    def __next__(self) -> T:
        if self._closed or not self._futures:
            raise StopIteration
        future = self._futures.popleft()
        try:
            item: Any = future.result()
        except BaseException:
            self.close(wait=False)
            raise
        self._futures.append(self._pool.submit(_next_item, self._source))
        return item

    def close(self, *, wait: bool = True) -> None:
        if self._closed:
            return
        self._closed = True
        futures = tuple(self._futures)
        self._futures.clear()
        error: BaseException | None = None
        try:
            if wait:
                for future in futures:
                    try:
                        future.result()
                    except StopIteration, CancelledError:
                        pass
                    except BaseException as caught:
                        if error is None:
                            error = caught
            else:
                for future in futures:
                    future.cancel()
        finally:
            self._pool.shutdown(wait=wait, cancel_futures=True)
            if wait or all(future.done() for future in futures):
                close = getattr(self._source, "close", None)
                if close is not None:
                    close()
        if error is not None:
            raise error


@dataclass(frozen=True, slots=True)
class ReplayPack:
    replay_id: str
    windows: tuple[dict[str, np.ndarray], ...]
    sample_id: int | None = None
    epoch: int | None = None

    def __post_init__(self) -> None:
        if not self.windows:
            raise ValueError("a replay pack must contain at least one window")
        if (self.sample_id is None) != (self.epoch is None):
            raise ValueError("replay-pack sample ID and epoch must be present together")


@dataclass(frozen=True, slots=True)
class _ReplayWindowSpec:
    """Enough information to reconstruct one deterministic decoded window."""

    sample_id: int
    epoch: int
    window_index: int


@dataclass(frozen=True, slots=True)
class _ReplayDecodeTask:
    """One deterministic worker task over explicit MDS sample IDs."""

    ordinal: int
    epoch: int
    sample_offset: int
    sample_ids: tuple[int, ...]
    cursor_after: tuple[int, int]


@dataclass(frozen=True, slots=True)
class _TensorColumnGroup:
    """Columns with one dtype and shape, coalesced into one shared tensor."""

    names: tuple[str, ...]
    values: torch.Tensor


@dataclass(frozen=True, slots=True)
class _CollatedReplayPackBatch:
    """Worker-stacked replay windows transferred in coalesced shared tensors."""

    replay_ids: tuple[str, ...]
    pack_sizes: tuple[int, ...]
    column_groups: tuple[_TensorColumnGroup, ...]
    sample_ids: tuple[int | None, ...]
    epochs: tuple[int | None, ...]
    task_ordinal: int | None = None
    cursor_after: tuple[int, int] | None = None

    def unpack(self) -> tuple[ReplayPack, ...]:
        columns: dict[str, np.ndarray] = {}
        for group in self.column_groups:
            values = group.values.numpy().copy()
            columns.update({name: values[index] for index, name in enumerate(group.names)})
        packs: list[ReplayPack] = []
        offset = 0
        for replay_id, size, sample_id, epoch in zip(
            self.replay_ids,
            self.pack_sizes,
            self.sample_ids,
            self.epochs,
            strict=True,
        ):
            windows = tuple(
                {name: values[index] for name, values in columns.items()} for index in range(offset, offset + size)
            )
            packs.append(ReplayPack(replay_id, windows, sample_id=sample_id, epoch=epoch))
            offset += size
        if offset != next(iter(columns.values())).shape[0]:
            raise RuntimeError("collated replay-pack row count changed during transfer")
        return tuple(packs)


def _collate_replay_packs(
    packs: list[ReplayPack],
    task: _ReplayDecodeTask | None = None,
) -> _CollatedReplayPackBatch:
    """Stack worker results to avoid pickling each small window array."""
    if not packs:
        raise ValueError("cannot collate an empty replay-pack batch")
    windows = [window for pack in packs for window in pack.windows]
    keys = tuple(windows[0])
    if not keys:
        raise ValueError("cannot collate replay windows without columns")
    if any(set(window) != set(keys) for window in windows[1:]):
        raise ValueError("replay windows have inconsistent columns")
    grouped_names: dict[tuple[str, tuple[int, ...]], list[str]] = {}
    for name in keys:
        values = np.asarray(windows[0][name])
        grouped_names.setdefault((values.dtype.str, values.shape), []).append(name)
    column_groups = []
    for (_dtype, shape), names in grouped_names.items():
        values = np.stack([np.asarray(window[name]) for name in names for window in windows])
        values = values.reshape(len(names), len(windows), *shape)
        column_groups.append(_TensorColumnGroup(tuple(names), torch.from_numpy(values)))
    return _CollatedReplayPackBatch(
        replay_ids=tuple(pack.replay_id for pack in packs),
        pack_sizes=tuple(len(pack.windows) for pack in packs),
        column_groups=tuple(column_groups),
        sample_ids=tuple(pack.sample_id for pack in packs),
        epochs=tuple(pack.epoch for pack in packs),
        task_ordinal=None if task is None else task.ordinal,
        cursor_after=None if task is None else task.cursor_after,
    )


class _ReplayTaskSampler(Sampler[_ReplayDecodeTask]):
    """Issue deterministic MDS sample-ID groups independently of worker timing."""

    def __init__(self, dataset: StreamingDataset, task_size: int) -> None:
        self._dataset = dataset
        self._task_size = task_size
        self._cursor = (0, 0)
        self._world = World(1, 1, 1, 0)
        if self._dataset.num_canonical_nodes is None:
            self._dataset.num_canonical_nodes = 64 if self._dataset.shuffle_algo in ("py1s", "py2s") else 1
        self._dataset._set_shuffle_block_size(self._world)  # pyright: ignore[reportPrivateUsage]

    @property
    def cursor(self) -> tuple[int, int]:
        return self._cursor

    def resume(self, cursor: tuple[int, int]) -> None:
        epoch, sample_offset = cursor
        if epoch < 0 or sample_offset < 0:
            raise ValueError(f"invalid replay-task cursor {cursor}")
        self._cursor = (epoch, sample_offset)

    def _sample_ids(self, epoch: int) -> np.ndarray:
        work = generate_work(
            self._dataset.batching_method,
            self._dataset,
            self._world,
            epoch,
            0,
        )
        sample_ids = work.reshape(-1)
        return sample_ids[sample_ids >= 0]

    def __iter__(self) -> Iterator[_ReplayDecodeTask]:
        epoch, sample_offset = self._cursor
        ordinal = 0
        while True:
            sample_ids = self._sample_ids(epoch)
            if sample_offset > len(sample_ids):
                raise ValueError(f"replay-task offset {sample_offset} exceeds epoch {epoch} size {len(sample_ids)}")
            while sample_offset < len(sample_ids):
                stop = min(sample_offset + self._task_size, len(sample_ids))
                at_epoch_end = stop == len(sample_ids)
                cursor_after = (epoch + 1, 0) if at_epoch_end else (epoch, stop)
                yield _ReplayDecodeTask(
                    ordinal=ordinal,
                    epoch=epoch,
                    sample_offset=sample_offset,
                    sample_ids=tuple(int(value) for value in sample_ids[sample_offset:stop]),
                    cursor_after=cursor_after,
                )
                ordinal += 1
                sample_offset = stop
            epoch += 1
            sample_offset = 0


class _ReplayDecodeDataset(Dataset[_CollatedReplayPackBatch]):
    """Decode one explicit replay task inside a DataLoader worker."""

    def __init__(self, packs: PolicyReplayPackDataset) -> None:
        self._packs = packs

    def __getitem__(self, task: _ReplayDecodeTask) -> _CollatedReplayPackBatch:
        packs = [self._packs.decode_sample(sample_id, task.epoch) for sample_id in task.sample_ids]
        if any(pack is None for pack in packs):
            raise RuntimeError("a deterministic replay task did not produce one pack per sample ID")
        return _collate_replay_packs(
            [pack for pack in packs if pack is not None],
            task,
        )


class _OrderedReplayTasks(Iterator[_CollatedReplayPackBatch]):
    """Release completed worker tasks in their predetermined order."""

    def __init__(self, source: Iterator[object]) -> None:
        self._source = source
        self._next_ordinal = 0
        self._ready: dict[int, _CollatedReplayPackBatch] = {}

    def __iter__(self) -> _OrderedReplayTasks:
        return self

    def __next__(self) -> _CollatedReplayPackBatch:
        while self._next_ordinal not in self._ready:
            try:
                value = next(self._source)
            except StopIteration:
                if self._ready:
                    raise RuntimeError("replay worker results ended with a task-order gap") from None
                raise
            if not isinstance(value, _CollatedReplayPackBatch) or value.task_ordinal is None:
                raise TypeError("deterministic replay worker returned an untagged result")
            if value.task_ordinal < self._next_ordinal or value.task_ordinal in self._ready:
                raise RuntimeError(f"duplicate replay task ordinal {value.task_ordinal}")
            self._ready[value.task_ordinal] = value
        value = self._ready.pop(self._next_ordinal)
        self._next_ordinal += 1
        return value


@dataclass(frozen=True, slots=True)
class ReservoirBatch:
    replay_ids: tuple[str, ...]
    windows: tuple[dict[str, np.ndarray], ...]


class ReplayPackBatchIterator(Iterator[ReplayPack]):
    """Flatten worker-side pack batches while retaining checkpointable lookahead."""

    def __init__(self, source: Iterator[object], visits: Counter[str] | None) -> None:
        self._source = source
        self._visits = visits
        self._pending: deque[ReplayPack] = deque()
        self.received_packs = 0
        self.source_cursor: tuple[int, int] | None = None

    def __iter__(self) -> ReplayPackBatchIterator:
        return self

    def __next__(self) -> ReplayPack:
        while not self._pending:
            value = next(self._source)
            if isinstance(value, ReplayPack):
                batch = (value,)
            elif isinstance(value, _CollatedReplayPackBatch):
                batch = value.unpack()
                if value.cursor_after is not None:
                    self.source_cursor = value.cursor_after
            elif isinstance(value, list | tuple) and all(isinstance(pack, ReplayPack) for pack in value):
                batch = tuple(value)
            else:
                raise TypeError(f"pack loader yielded {type(value).__name__}, expected ReplayPack batch")
            if not batch:
                continue
            self.received_packs += len(batch)
            if self._visits is not None:
                for pack in batch:
                    self._visits[pack.replay_id] += 1
            self._pending.extend(batch)
        return self._pending.popleft()

    def state_dict(self) -> dict[str, object]:
        """Return packs fetched by a worker batch but not yet read by the reservoir."""
        state: dict[str, object] = {
            "received_packs": self.received_packs,
            "source_cursor": self.source_cursor,
        }
        if self.source_cursor is not None and all(pack.sample_id is not None for pack in self._pending):
            state["pending_specs"] = tuple((pack.replay_id, pack.sample_id, pack.epoch) for pack in self._pending)
        else:
            state["pending"] = tuple(self._pending)
        return state

    def load_state_dict(
        self,
        state: Mapping[str, object],
        replay_packs: Mapping[tuple[int, int], ReplayPack] | None = None,
    ) -> None:
        pending_specs = state.get("pending_specs")
        if pending_specs is not None:
            if replay_packs is None or not isinstance(pending_specs, tuple):
                raise ValueError("pending replay-pack descriptors cannot be restored")
            pending: list[ReplayPack] = []
            for value in pending_specs:
                if (
                    not isinstance(value, tuple)
                    or len(value) != 3
                    or not isinstance(value[0], str)
                    or not isinstance(value[1], int)
                    or not isinstance(value[2], int)
                ):
                    raise ValueError("pending replay-pack descriptor is invalid")
                replay_id, sample_id, epoch = value
                pack = replay_packs.get((sample_id, epoch))
                if pack is None or pack.replay_id != replay_id:
                    raise ValueError(f"could not reconstruct pending replay {replay_id!r}")
                pending.append(pack)
        else:
            stored = state.get("pending", ())
            if not isinstance(stored, tuple) or any(not isinstance(pack, ReplayPack) for pack in stored):
                raise ValueError("pending replay-pack state is invalid")
            pending = list(stored)
        cursor = state.get("source_cursor")
        if cursor is not None and (
            not isinstance(cursor, tuple)
            or len(cursor) != 2
            or any(not isinstance(value, int) or value < 0 for value in cursor)
        ):
            raise ValueError("replay source cursor is invalid")
        self._pending = deque(pending)
        self.received_packs = int(state.get("received_packs", 0))
        self.source_cursor = cursor


class ReplayPackPrefetch(Iterator[ReplayPack]):
    """Keep one deterministic replay cohort decoded in normal process memory."""

    def __init__(self, source: ReplayPackBatchIterator, *, capacity: int) -> None:
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 1:
            raise ValueError(f"replay prefetch capacity must be positive, got {capacity!r}")
        self._source = source
        self._capacity = capacity
        self._buffer: deque[ReplayPack] = deque()
        self._condition = Condition()
        self._source_lock = Lock()
        self._thread: Thread | None = None
        self._source_done = False
        self._closed = False
        self._error: BaseException | None = None

    @property
    def buffered_packs(self) -> int:
        with self._condition:
            return len(self._buffer)

    def start(self) -> None:
        """Start the sole producer after any checkpoint state is restored."""
        with self._condition:
            if self._thread is not None:
                raise RuntimeError("replay prefetch has already started")
            self._thread = Thread(target=self._produce, name="replay-cohort", daemon=True)
            self._thread.start()

    def __iter__(self) -> ReplayPackPrefetch:
        return self

    def __next__(self) -> ReplayPack:
        with self._condition:
            while not self._buffer and not self._source_done and not self._closed and self._error is None:
                self._condition.wait()
            if self._closed:
                raise StopIteration
            if self._buffer:
                pack = self._buffer.popleft()
                self._condition.notify_all()
                return pack
            if self._error is not None:
                raise self._error
            raise StopIteration

    def state_dict(self) -> dict[str, object]:
        """Return a coherent source cursor and descriptor-only cohort snapshot."""
        with self._source_lock, self._condition:
            if self._error is not None:
                raise RuntimeError("replay prefetch producer failed") from self._error
            if any(pack.sample_id is None or pack.epoch is None for pack in self._buffer):
                raise RuntimeError("replay prefetch contains an anonymous pack")
            return {
                "capacity": self._capacity,
                "source_done": self._source_done,
                "buffer_specs": tuple((pack.replay_id, pack.sample_id, pack.epoch) for pack in self._buffer),
                "source": self._source.state_dict(),
            }

    def load_state_dict(
        self,
        state: Mapping[str, object],
        replay_packs: Mapping[tuple[int, int], ReplayPack],
    ) -> None:
        """Restore a descriptor-only cohort before starting the producer."""
        if self._thread is not None:
            raise RuntimeError("restore replay prefetch state before starting it")
        if state.get("capacity") != self._capacity:
            raise ValueError("replay prefetch capacity changed across resume")
        stored = state.get("buffer_specs")
        if not isinstance(stored, tuple):
            raise ValueError("replay prefetch descriptors are invalid")
        restored: deque[ReplayPack] = deque()
        for value in stored:
            if (
                not isinstance(value, tuple)
                or len(value) != 3
                or not isinstance(value[0], str)
                or not isinstance(value[1], int)
                or not isinstance(value[2], int)
            ):
                raise ValueError("replay prefetch descriptor is invalid")
            replay_id, sample_id, epoch = value
            pack = replay_packs.get((sample_id, epoch))
            if pack is None or pack.replay_id != replay_id:
                raise ValueError(f"could not reconstruct prefetched replay {replay_id!r}")
            restored.append(pack)
        if len(restored) > self._capacity:
            raise ValueError("restored replay prefetch exceeds its capacity")
        self._buffer = restored
        self._source_done = bool(state.get("source_done", False))

    def close(self) -> None:
        """Stop accepting or yielding packs without waiting on a blocked read."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def _produce(self) -> None:
        try:
            while True:
                with self._condition:
                    while len(self._buffer) >= self._capacity and not self._closed:
                        self._condition.wait()
                    if self._closed:
                        return
                with self._source_lock:
                    pack = next(self._source)
                    with self._condition:
                        if self._closed:
                            return
                        self._buffer.append(pack)
                        self._condition.notify_all()
        except StopIteration:
            with self._condition:
                self._source_done = True
                self._condition.notify_all()
        except BaseException as error:
            with self._condition:
                self._error = error
                self._condition.notify_all()


class ReplayReservoir:
    """Emit batches with unique replay IDs and a replay cooldown."""

    def __init__(
        self,
        packs: Iterator[ReplayPack],
        *,
        batch_size: int,
        capacity: int,
        seed: int,
        cooldown_batches: int = 1,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if capacity < batch_size:
            raise ValueError(f"capacity={capacity} must be at least batch_size={batch_size}")
        if cooldown_batches < 0:
            raise ValueError(f"cooldown_batches must be non-negative, got {cooldown_batches}")
        minimum_capacity = batch_size * (cooldown_batches + 1)
        if capacity < minimum_capacity:
            raise ValueError(
                f"capacity={capacity} cannot enforce cooldown_batches={cooldown_batches} "
                f"with batch_size={batch_size}; need at least {minimum_capacity}"
            )
        self._source = iter(packs)
        self._batch_size = batch_size
        self._capacity = capacity
        self._rng = np.random.default_rng(seed)
        self._active: dict[str, deque[dict[str, np.ndarray]]] = {}
        self._active_specs: dict[str, deque[_ReplayWindowSpec]] = {}
        self._descriptor_backed: bool | None = None
        self._cooldown: deque[set[str]] = deque(maxlen=cooldown_batches)
        self._source_done = False
        self._finished = False
        self._emitted_windows = 0
        self._dropped_windows = 0
        self._dropped_replays = 0

    @property
    def active_replays(self) -> int:
        return len(self._active)

    @property
    def emitted_windows(self) -> int:
        return self._emitted_windows

    @property
    def dropped_windows(self) -> int:
        return self._dropped_windows

    @property
    def dropped_replays(self) -> int:
        return self._dropped_replays

    def __iter__(self) -> ReplayReservoir:
        return self

    def state_dict(self) -> dict[str, Any]:
        """Return the exact between-batch reservoir state."""
        state = {
            "batch_size": self._batch_size,
            "capacity": self._capacity,
            "rng": self._rng.bit_generator.state,
            "cooldown": tuple(tuple(sorted(replays)) for replays in self._cooldown),
            "cooldown_batches": self._cooldown.maxlen,
            "source_done": self._source_done,
            "finished": self._finished,
            "emitted_windows": self._emitted_windows,
            "dropped_windows": self._dropped_windows,
            "dropped_replays": self._dropped_replays,
        }
        if self._descriptor_backed:
            state["active_specs"] = tuple(
                (
                    replay_id,
                    tuple((spec.sample_id, spec.epoch, spec.window_index) for spec in specs),
                )
                for replay_id, specs in self._active_specs.items()
            )
        else:
            state["active"] = tuple((replay_id, tuple(windows)) for replay_id, windows in self._active.items())
        return state

    def load_state_dict(
        self,
        state: dict[str, Any],
        replay_packs: Mapping[tuple[int, int], ReplayPack] | None = None,
    ) -> None:
        """Restore a state produced after a complete batch was emitted."""
        if state["batch_size"] != self._batch_size or state["capacity"] != self._capacity:
            raise ValueError("reservoir geometry changed across resume")
        if state["cooldown_batches"] != self._cooldown.maxlen:
            raise ValueError("reservoir cooldown changed across resume")
        self._rng.bit_generator.state = state["rng"]
        active_specs = state.get("active_specs")
        if active_specs is not None:
            if replay_packs is None or not isinstance(active_specs, tuple):
                raise ValueError("active replay-window descriptors cannot be restored")
            self._active = {}
            self._active_specs = {}
            self._descriptor_backed = True
            for value in active_specs:
                if not isinstance(value, tuple) or len(value) != 2 or not isinstance(value[0], str):
                    raise ValueError("active replay descriptor is invalid")
                replay_id, stored_specs = value
                if not isinstance(stored_specs, tuple):
                    raise ValueError("active replay window descriptors are invalid")
                windows: deque[dict[str, np.ndarray]] = deque()
                specs: deque[_ReplayWindowSpec] = deque()
                for stored_spec in stored_specs:
                    if (
                        not isinstance(stored_spec, tuple)
                        or len(stored_spec) != 3
                        or any(not isinstance(part, int) or part < 0 for part in stored_spec)
                    ):
                        raise ValueError("active replay window descriptor is invalid")
                    spec = _ReplayWindowSpec(*stored_spec)
                    pack = replay_packs.get((spec.sample_id, spec.epoch))
                    if pack is None or pack.replay_id != replay_id:
                        raise ValueError(f"could not reconstruct active replay {replay_id!r}")
                    if spec.window_index >= len(pack.windows):
                        raise ValueError(f"window index changed for active replay {replay_id!r}")
                    windows.append(pack.windows[spec.window_index])
                    specs.append(spec)
                self._active[replay_id] = windows
                self._active_specs[replay_id] = specs
        else:
            self._active = {replay_id: deque(windows) for replay_id, windows in state["active"]}
            self._active_specs = {}
            self._descriptor_backed = False
        self._cooldown = deque((set(replays) for replays in state["cooldown"]), maxlen=self._cooldown.maxlen)
        self._source_done = bool(state["source_done"])
        self._finished = bool(state["finished"])
        self._emitted_windows = int(state["emitted_windows"])
        self._dropped_windows = int(state["dropped_windows"])
        self._dropped_replays = int(state["dropped_replays"])

    def _fill(self) -> None:
        while len(self._active) < self._capacity and not self._source_done:
            try:
                pack = next(self._source)
            except StopIteration:
                self._source_done = True
                break
            descriptor_backed = pack.sample_id is not None
            if self._descriptor_backed is None:
                self._descriptor_backed = descriptor_backed
            elif descriptor_backed != self._descriptor_backed:
                raise RuntimeError("replay source mixed descriptor-backed and anonymous packs")
            order = tuple(int(index) for index in self._rng.permutation(len(pack.windows)))
            windows = (pack.windows[index] for index in order)
            specs = (
                ()
                if pack.sample_id is None or pack.epoch is None
                else tuple(_ReplayWindowSpec(pack.sample_id, pack.epoch, index) for index in order)
            )
            if pack.replay_id in self._active:
                # Weighted streams can repeat a small source within one epoch.
                # Keep its repeated windows, but retain one active replay key so
                # a batch never contains the same replay more than once.
                self._active[pack.replay_id].extend(windows)
                if self._descriptor_backed:
                    self._active_specs[pack.replay_id].extend(specs)
            else:
                self._active[pack.replay_id] = deque(windows)
                if self._descriptor_backed:
                    self._active_specs[pack.replay_id] = deque(specs)

    def _finish(self) -> None:
        if self._finished:
            return
        self._dropped_windows = sum(len(windows) for windows in self._active.values())
        self._dropped_replays = len(self._active)
        self._active.clear()
        self._active_specs.clear()
        self._finished = True

    def __next__(self) -> ReservoirBatch:
        if self._finished:
            raise StopIteration
        self._fill()
        if len(self._active) < self._batch_size:
            self._finish()
            raise StopIteration

        blocked = set().union(*self._cooldown) if self._cooldown else set()
        candidates = [replay_id for replay_id in self._active if replay_id not in blocked]
        if len(candidates) < self._batch_size:
            if not self._source_done:
                raise RuntimeError("reservoir capacity did not provide enough replay IDs after cooldown")
            self._finish()
            raise StopIteration
        selected = self._rng.choice(len(candidates), size=self._batch_size, replace=False)
        replay_ids = tuple(candidates[int(index)] for index in selected)
        windows = tuple(self._active[replay_id].popleft() for replay_id in replay_ids)
        for replay_id in replay_ids:
            if self._descriptor_backed:
                self._active_specs[replay_id].popleft()
            if not self._active[replay_id]:
                del self._active[replay_id]
                if self._descriptor_backed:
                    del self._active_specs[replay_id]
        if self._cooldown.maxlen:
            self._cooldown.append(set(replay_ids))
        self._emitted_windows += self._batch_size
        return ReservoirBatch(replay_ids=replay_ids, windows=windows)


def _stable_replay_rng(seed: int, epoch: int, replay_id: str) -> np.random.Generator:
    digest = hashlib.blake2b(replay_id.encode(), digest_size=8).digest()
    identity = int.from_bytes(digest, "little")
    return np.random.default_rng((seed, epoch, identity & 0xFFFFFFFF, identity >> 32))


class PolicyReplayPackDataset(IterableDataset):
    def __init__(
        self,
        dataset: StreamingDataset,
        L_ctx: int,
        L_chunk: int,
        *,
        seed: int,
        windows_per_replay: int,
        schema_version: int,
        projection: FeatureProjection | None,
        replay_format: ReplayFormat = "policy",
        replay_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        replay_labels: ReplayLabels | None = None,
        window_transform: WindowTransform | None = None,
        replay_filter: ReplayFilter | None = None,
        require_pack: bool = False,
        require_full_context: bool = False,
    ) -> None:
        if replay_format not in ("policy", "policy-world"):
            raise ValueError(f"replay reservoir requires a compact format, got {replay_format!r}")
        if replay_transform is not None and replay_labels is not None:
            raise ValueError("replay_transform and replay_labels are mutually exclusive")
        if not isinstance(windows_per_replay, int) or isinstance(windows_per_replay, bool) or windows_per_replay < 1:
            raise ValueError("windows_per_replay must be a positive integer")
        self._dataset = dataset
        self._L_ctx = L_ctx
        self._L_chunk = L_chunk
        self._seed = seed
        self._K = windows_per_replay
        self._schema_version = schema_version
        self._projection = projection
        self._decode = decode_policy_replay if replay_format == "policy" else decode_policy_world_replay
        self._decode_slices = (
            decode_policy_replay_slices if replay_format == "policy" else decode_policy_world_replay_slices
        )
        self._replay_transform = replay_transform
        self._replay_labels = replay_labels
        self._window_transform = window_transform
        self._replay_filter = replay_filter
        self._require_pack = require_pack
        self._require_full_context = require_full_context
        self._epoch = 0
        self._current_epoch: int | None = None
        self._source_samples = 0

    @property
    def current_epoch(self) -> int | None:
        return self._current_epoch

    @property
    def source_samples(self) -> int:
        return self._source_samples

    def resume_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def decode_sample(self, sample_id: int, epoch: int) -> ReplayPack | None:
        """Decode one stable global MDS sample ID for a specified shuffle epoch."""
        compact = self._dataset[sample_id]
        return self._decode_compact(compact, epoch, sample_id=sample_id)

    def _decode_compact(
        self,
        compact: Mapping[str, object],
        epoch: int,
        *,
        sample_id: int | None,
    ) -> ReplayPack | None:
        # Defer shared helpers so dataloader can re-export this module safely.
        from hal.training.dataloader import _choose_chunk_starts
        from hal.training.dataloader import _make_window

        replay_id = str(compact["replay_id"])
        if self._replay_filter is not None and not self._replay_filter(replay_id):
            return None
        check_schema_version({"schema_version": int(compact["source_schema_version"])}, expected=self._schema_version)
        frames = int(compact["num_frames"])
        rng = _stable_replay_rng(self._seed, epoch, replay_id)
        starts = [
            int(chunk_start) - self._L_ctx
            for chunk_start in _choose_chunk_starts(
                frames,
                self._L_ctx,
                self._L_chunk,
                self._K,
                rng,
                require_full_context=self._require_full_context,
            )
        ]
        if self._require_pack and len(starts) != self._K:
            raise ValueError(
                f"replay {replay_id!r} with {frames} frames emits {len(starts)} of the required {self._K} windows"
            )
        if not starts:
            return None
        ego_prefixes = ["p1" if rng.random() < 0.5 else "p2" for _ in starts]
        ranges = tuple((max(0, start), start + self._L_ctx + self._L_chunk) for start in starts)
        windows = []
        if self._replay_transform is None:
            samples = self._decode_slices(compact, ranges)
            labels = (
                {}
                if self._replay_labels is None
                else {name: np.asarray(value) for name, value in self._replay_labels(compact).items()}
            )
            wrong_labels = {name: value.shape for name, value in labels.items() if value.shape not in ((), (frames,))}
            if wrong_labels:
                raise ValueError(f"replay labels have invalid shapes {wrong_labels}; expected scalar or {(frames,)}")
            for start, ego_prefix, frame_range, sample in zip(
                starts,
                ego_prefixes,
                ranges,
                samples,
                strict=True,
            ):
                range_start, range_stop = frame_range
                sample.update(
                    {
                        name: (
                            np.full(range_stop - range_start, value.item(), dtype=value.dtype)
                            if value.shape == ()
                            else value[range_start:range_stop]
                        )
                        for name, value in labels.items()
                    }
                )
                pad = max(0, -start)
                window = _make_window(
                    sample,
                    ego_prefix=ego_prefix,
                    start=0,
                    pad=pad,
                    length=self._L_ctx + self._L_chunk,
                    projection=self._projection,
                )
                window["ctx_pad"] = np.int64(min(pad, self._L_ctx))
                if self._window_transform is not None:
                    self._window_transform(replay_id, ego_prefix, window)
                windows.append(window)
        else:
            sample = self._replay_transform(self._decode(compact))
            for start, ego_prefix in zip(starts, ego_prefixes, strict=True):
                pad = max(0, -start)
                window = _make_window(
                    sample,
                    ego_prefix=ego_prefix,
                    start=start,
                    pad=pad,
                    length=self._L_ctx + self._L_chunk,
                    projection=self._projection,
                )
                window["ctx_pad"] = np.int64(min(pad, self._L_ctx))
                if self._window_transform is not None:
                    self._window_transform(replay_id, ego_prefix, window)
                windows.append(window)
        if not windows:
            return None
        return ReplayPack(
            replay_id,
            tuple(windows),
            sample_id=sample_id,
            epoch=epoch if sample_id is not None else None,
        )

    def __iter__(self) -> Iterator[ReplayPack]:
        epoch = self._epoch
        self._epoch += 1
        self._current_epoch = epoch
        self._source_samples = 0
        for compact in self._dataset:
            self._source_samples += 1
            pack = self._decode_compact(compact, epoch, sample_id=None)
            if pack is not None:
                yield pack


class ReservoirLoader:
    def __init__(
        self,
        pack_loader: DataLoader,
        *,
        stats: dict[str, FeatureStats],
        L_ctx: int,
        batch_size: int,
        capacity: int,
        seed: int,
        extra: ExtraColumns | None,
        projection: FeatureProjection | None,
        batch_transform: Callable[[list[dict[str, np.ndarray]], TrainBatch], object] | None,
        prefetch_batches: int,
        pin_memory: bool,
        dataset: StreamingDataset,
        source_names: tuple[str, ...],
        packs: PolicyReplayPackDataset,
        worker_independent_resume: bool,
        cooldown_batches: int,
        task_sampler: _ReplayTaskSampler | None,
        replay_prefetch_capacity: int,
    ) -> None:
        if not isinstance(prefetch_batches, int) or isinstance(prefetch_batches, bool) or prefetch_batches < 0:
            raise ValueError(f"prefetch_batches must be a non-negative integer, got {prefetch_batches!r}")
        self._pack_loader = pack_loader
        self._stats = stats
        self._L_ctx = L_ctx
        self._batch_size = batch_size
        self._capacity = capacity
        self._seed = seed
        self._extra = extra
        self._projection = projection
        self._batch_transform = batch_transform
        self._prefetch_batches = prefetch_batches
        self._pin_memory = pin_memory
        self._dataset = dataset
        self._source_names = source_names
        self._packs = packs
        self._worker_independent_resume = worker_independent_resume
        self._cooldown_batches = cooldown_batches
        self._task_sampler = task_sampler
        self._replay_prefetch_capacity = replay_prefetch_capacity
        self._epoch = 0
        self._reservoir: ReplayReservoir | None = None
        self._pack_batches: ReplayPackBatchIterator | None = None
        self._replay_prefetch: ReplayPackPrefetch | None = None
        self._resume_state: dict[str, Any] | None = None
        self._visits: Counter[str] | None = None if task_sampler is not None else Counter()
        self.last_epoch_stats: dict[str, int] | None = None

    @property
    def source_sample_counts(self) -> dict[str, int]:
        """Per-stream sample counts in configured source order."""
        if not self._source_names:
            return {}
        counts = getattr(self._dataset, "selected_samples_per_stream", self._dataset.samples_per_stream)
        return {name: int(count) for name, count in zip(self._source_names, counts, strict=True)}

    def __iter__(self) -> Iterator[object]:
        resume_state = self._resume_state
        replay_packs = self._rehydrate(resume_state) if resume_state is not None else None
        source: Iterator[object] = iter(self._pack_loader)
        if self._task_sampler is not None:
            source = _OrderedReplayTasks(source)
        pack_batches = ReplayPackBatchIterator(source, self._visits)
        if resume_state is not None:
            pack_batches.load_state_dict(resume_state.get("pack_batches", {}), replay_packs)
        reservoir_source: Iterator[ReplayPack] = pack_batches
        replay_prefetch: ReplayPackPrefetch | None = None
        if self._replay_prefetch_capacity:
            replay_prefetch = ReplayPackPrefetch(pack_batches, capacity=self._replay_prefetch_capacity)
            if resume_state is not None and resume_state.get("schema") == 4:
                prefetch_state = resume_state.get("replay_prefetch")
                if not isinstance(prefetch_state, Mapping) or replay_packs is None:
                    raise ValueError("replay prefetch checkpoint state is incomplete")
                replay_prefetch.load_state_dict(prefetch_state, replay_packs)
            replay_prefetch.start()
            reservoir_source = replay_prefetch
        reservoir = ReplayReservoir(
            reservoir_source,
            batch_size=self._batch_size,
            capacity=self._capacity,
            seed=self._seed + self._epoch,
            cooldown_batches=self._cooldown_batches,
        )
        if resume_state is not None:
            reservoir.load_state_dict(resume_state["reservoir"], replay_packs)
            self._resume_state = None
        self._reservoir = reservoir
        self._pack_batches = pack_batches
        self._replay_prefetch = replay_prefetch
        self._epoch += 1
        batches = self._batches(reservoir)
        if self._prefetch_batches:
            return OneBatchPrefetch(batches, depth=self._prefetch_batches)
        return batches

    def state_dict(self) -> dict[str, Any]:
        """Return the Mosaic cursor, pack lookahead, reservoir, visits, and RNG state."""
        if self._prefetch_batches:
            raise RuntimeError("exact reservoir checkpoints require prefetch_batches=0")
        if self._pack_loader.num_workers != 0 and not self._worker_independent_resume:
            raise RuntimeError("exact worker resume must be enabled when num_workers is nonzero")
        if self._reservoir is None or self._pack_batches is None:
            raise RuntimeError("the reservoir loader has not emitted a batch")
        if self._task_sampler is not None:
            prefetch_state = None
            if self._replay_prefetch is not None:
                prefetch_snapshot = self._replay_prefetch.state_dict()
                pack_state = prefetch_snapshot["source"]
                if not isinstance(pack_state, dict):
                    raise RuntimeError("replay prefetch source state is invalid")
                prefetch_state = {name: value for name, value in prefetch_snapshot.items() if name != "source"}
            else:
                pack_state = self._pack_batches.state_dict()
            reservoir_state = self._reservoir.state_dict()
            source_cursor = pack_state.get("source_cursor")
            if source_cursor is None:
                raise RuntimeError("deterministic replay source has no committed cursor")
            if "pending_specs" not in pack_state or "active_specs" not in reservoir_state:
                raise RuntimeError("deterministic replay state contains decoded arrays")
            state = {
                "schema": 4 if prefetch_state is not None else 3,
                "loader_epoch": self._epoch,
                "source_cursor": source_cursor,
                "sample_selection_sha256": getattr(self._dataset, "sample_selection_sha256", None),
                "pack_batches": pack_state,
                "reservoir": reservoir_state,
            }
            if prefetch_state is not None:
                state["replay_prefetch"] = prefetch_state
            return state
        if self._worker_independent_resume:
            source_samples = self._pack_batches.received_packs
        else:
            source_samples = self._packs.source_samples
        return {
            "schema": 2,
            "loader_epoch": self._epoch,
            "pack_epoch": self._epoch - 1,
            "mds": self._dataset.state_dict(source_samples, from_beginning=False),
            "pack_batches": self._pack_batches.state_dict(),
            "reservoir": self._reservoir.state_dict(),
            "visit_counters": dict(self._visits or ()),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Schedule an exact restore before the next iterator is created."""
        schema = state.get("schema")
        if schema not in (1, 2, 3, 4):
            raise ValueError(f"unsupported reservoir-loader state schema {state.get('schema')!r}")
        if self._prefetch_batches:
            raise RuntimeError("exact reservoir checkpoints require prefetch_batches=0")
        if self._pack_loader.num_workers != 0 and not self._worker_independent_resume:
            raise RuntimeError("exact worker resume must be enabled when num_workers is nonzero")
        if self._reservoir is not None:
            raise RuntimeError("load reservoir state before creating its iterator")
        if schema in (3, 4):
            if self._task_sampler is None:
                raise ValueError("deterministic replay state requires the explicit sample-ID loader")
            if schema == 4 and not self._replay_prefetch_capacity:
                raise ValueError("replay prefetch checkpoint requires replay prefetch to be enabled")
            selection_hash = getattr(self._dataset, "sample_selection_sha256", None)
            if state.get("sample_selection_sha256") != selection_hash:
                raise ValueError("stream prefix selection changed across resume")
            cursor = state.get("source_cursor")
            if (
                not isinstance(cursor, tuple)
                or len(cursor) != 2
                or any(not isinstance(value, int) or value < 0 for value in cursor)
            ):
                raise ValueError("deterministic replay source cursor is invalid")
            self._task_sampler.resume((cursor[0], cursor[1]))
        else:
            if self._task_sampler is not None:
                raise ValueError("legacy replay checkpoints require the ordered iterable loader")
            self._dataset.load_state_dict(state["mds"])
            self._packs.resume_epoch(int(state["pack_epoch"]))
        self._epoch = int(state["loader_epoch"]) - 1
        if self._visits is not None:
            self._visits = Counter({str(name): int(count) for name, count in state.get("visit_counters", {}).items()})
        self._resume_state = {
            **state,
            "pack_batches": state.get("pack_batches", {"pending": ()}),
        }

    def _rehydrate(self, state: Mapping[str, object]) -> dict[tuple[int, int], ReplayPack] | None:
        """Rebuild compact checkpoint descriptors from their original MDS rows."""
        if state.get("schema") not in (3, 4):
            return None
        keys: set[tuple[int, int]] = set()
        pack_state = state.get("pack_batches")
        reservoir_state = state.get("reservoir")
        if not isinstance(pack_state, Mapping) or not isinstance(reservoir_state, Mapping):
            raise ValueError("deterministic replay checkpoint state is incomplete")
        pending_specs = pack_state.get("pending_specs", ())
        active_specs = reservoir_state.get("active_specs", ())
        if not isinstance(pending_specs, tuple) or not isinstance(active_specs, tuple):
            raise ValueError("deterministic replay descriptors are invalid")
        for value in pending_specs:
            if not isinstance(value, tuple) or len(value) != 3:
                raise ValueError("pending replay descriptor is invalid")
            _, sample_id, epoch = value
            if not isinstance(sample_id, int) or not isinstance(epoch, int):
                raise ValueError("pending replay descriptor is invalid")
            keys.add((sample_id, epoch))
        for active in active_specs:
            if not isinstance(active, tuple) or len(active) != 2 or not isinstance(active[1], tuple):
                raise ValueError("active replay descriptor is invalid")
            for value in active[1]:
                if not isinstance(value, tuple) or len(value) != 3:
                    raise ValueError("active replay window descriptor is invalid")
                sample_id, epoch, _ = value
                if not isinstance(sample_id, int) or not isinstance(epoch, int):
                    raise ValueError("active replay window descriptor is invalid")
                keys.add((sample_id, epoch))
        if state.get("schema") == 4:
            prefetch_state = state.get("replay_prefetch")
            if not isinstance(prefetch_state, Mapping):
                raise ValueError("replay prefetch checkpoint state is incomplete")
            buffer_specs = prefetch_state.get("buffer_specs")
            if not isinstance(buffer_specs, tuple):
                raise ValueError("replay prefetch descriptors are invalid")
            for value in buffer_specs:
                if not isinstance(value, tuple) or len(value) != 3:
                    raise ValueError("replay prefetch descriptor is invalid")
                _, sample_id, epoch = value
                if not isinstance(sample_id, int) or not isinstance(epoch, int):
                    raise ValueError("replay prefetch descriptor is invalid")
                keys.add((sample_id, epoch))
        replay_packs: dict[tuple[int, int], ReplayPack] = {}
        for sample_id, epoch in sorted(keys, key=lambda value: (value[1], value[0])):
            pack = self._packs.decode_sample(sample_id, epoch)
            if pack is None:
                raise ValueError(f"MDS sample {sample_id} no longer produces a replay pack")
            replay_packs[(sample_id, epoch)] = pack
        return replay_packs

    def _batches(self, reservoir: ReplayReservoir) -> Iterator[object]:
        from hal.training.dataloader import collate_train_batch

        try:
            for item in reservoir:
                batch = collate_train_batch(
                    list(item.windows),
                    stats=self._stats,
                    L_ctx=self._L_ctx,
                    extra=self._extra,
                    projection=self._projection,
                )
                batch = TrainBatch(context=batch.context, target=batch.target, replay_ids=item.replay_ids)
                transformed = (
                    self._batch_transform(list(item.windows), batch) if self._batch_transform is not None else batch
                )
                yield transformed.pin_memory() if self._pin_memory else transformed
        finally:
            self.last_epoch_stats = {
                "emitted_windows": reservoir.emitted_windows,
                "dropped_windows": reservoir.dropped_windows,
                "dropped_replays": reservoir.dropped_replays,
            }
            if self._replay_prefetch is not None:
                self._replay_prefetch.close()


def _limit_worker_threads(_worker_id: int) -> None:
    """Keep one loader worker from creating a nested Torch/BLAS thread pool."""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    torch.set_num_threads(1)


def make_reservoir_loader(
    data_root: str | None,
    split: str,
    *,
    stats: dict[str, FeatureStats],
    L_ctx: int,
    L_chunk: int,
    batch_size: int,
    seed: int,
    reservoir_capacity: int,
    remote: str | None = None,
    sources: Sequence[StreamSource] | None = None,
    source_weights: Sequence[float] | None = None,
    source_prefixes: Sequence[StreamSamplePrefix] | None = None,
    cache_limit: str | int | None = None,
    shuffle_block_size: int | None = None,
    shuffle: bool | None = None,
    shuffle_seed: int | None = None,
    shuffle_algo: str | None = None,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    predownload: int = 512,
    download_retry: int = 2,
    windows_per_replay: int = 4,
    prefetch_batches: int = 0,
    pin_memory: bool | None = None,
    schema_version: int = SCHEMA_VERSION,
    extra: ExtraColumns | None = None,
    projection: FeatureProjection | None = None,
    replay_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    replay_labels: ReplayLabels | None = None,
    window_transform: WindowTransform | None = None,
    batch_transform: Callable[[list[dict[str, np.ndarray]], TrainBatch], object] | None = None,
    replay_format: ReplayFormat = "policy",
    replay_filter: ReplayFilter | None = None,
    replay_pack_batch_size: int = 1,
    worker_independent_resume: bool = False,
    deterministic_out_of_order: bool = False,
    cooldown_batches: int = 1,
    replay_prefetch_capacity: int = 0,
    limit_worker_threads: bool = False,
    require_full_context: bool = False,
) -> ReservoirLoader:
    """Build a compact replay loader with replay-aware batches.

    ``window_transform`` may attach target-side metadata after the replay ID,
    ego port, window start, and padding have already been sampled. It must not
    consume randomness or mutate model-input columns.
    """
    if predownload < 1:
        raise ValueError(f"predownload must be positive, got {predownload}")
    if not isinstance(prefetch_batches, int) or isinstance(prefetch_batches, bool) or prefetch_batches < 0:
        raise ValueError(f"prefetch_batches must be a non-negative integer, got {prefetch_batches!r}")
    if (
        not isinstance(replay_pack_batch_size, int)
        or isinstance(replay_pack_batch_size, bool)
        or replay_pack_batch_size < 1
    ):
        raise ValueError(f"replay_pack_batch_size must be a positive integer, got {replay_pack_batch_size!r}")
    if worker_independent_resume and replay_filter is not None:
        raise ValueError(
            "worker-independent resume requires row selection before worker dispatch; "
            "replay_filter runs inside workers"
        )
    if deterministic_out_of_order and not worker_independent_resume:
        raise ValueError("deterministic out-of-order loading requires worker-independent resume")
    if cooldown_batches < 0:
        raise ValueError(f"cooldown_batches must be non-negative, got {cooldown_batches}")
    if (
        not isinstance(replay_prefetch_capacity, int)
        or isinstance(replay_prefetch_capacity, bool)
        or replay_prefetch_capacity < 0
    ):
        raise ValueError(f"replay_prefetch_capacity must be a non-negative integer, got {replay_prefetch_capacity!r}")
    if replay_prefetch_capacity and not deterministic_out_of_order:
        raise ValueError("replay prefetch requires deterministic out-of-order loading")
    minimum_capacity = batch_size * (cooldown_batches + 1)
    if reservoir_capacity < minimum_capacity:
        raise ValueError(
            f"reservoir_capacity={reservoir_capacity} cannot enforce cooldown_batches={cooldown_batches}; "
            f"need at least {minimum_capacity}"
        )
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    dataset, source_names = _make_streaming_dataset(
        data_root,
        split,
        sources=sources,
        source_weights=source_weights,
        source_prefixes=source_prefixes,
        remote=remote,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
        cache_limit=cache_limit,
        shuffle_block_size=shuffle_block_size,
        predownload=predownload if remote or sources is not None else None,
        batch_size=replay_pack_batch_size,
        shuffle_algo=shuffle_algo,
        download_retry=download_retry,
    )
    packs = PolicyReplayPackDataset(
        dataset,
        L_ctx,
        L_chunk,
        seed=seed,
        windows_per_replay=windows_per_replay,
        schema_version=schema_version,
        projection=projection,
        replay_format=replay_format,
        replay_transform=replay_transform,
        replay_labels=replay_labels,
        window_transform=window_transform,
        replay_filter=replay_filter,
        require_pack=worker_independent_resume,
        require_full_context=require_full_context,
    )
    if num_workers > 0:
        torch.multiprocessing.set_sharing_strategy("file_system")
    generator = torch.Generator().manual_seed(seed)
    task_sampler: _ReplayTaskSampler | None = None
    if deterministic_out_of_order:
        task_sampler = _ReplayTaskSampler(dataset, replay_pack_batch_size)
        pack_loader = DataLoader(
            _ReplayDecodeDataset(packs),
            batch_size=None,
            sampler=task_sampler,
            num_workers=num_workers,
            collate_fn=_identity,
            persistent_workers=(num_workers > 0),
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            pin_memory=False,
            generator=generator,
            worker_init_fn=_limit_worker_threads if limit_worker_threads else None,
            in_order=(num_workers == 0),
        )
    else:
        pack_loader = DataLoader(
            packs,
            batch_size=replay_pack_batch_size,
            num_workers=num_workers,
            collate_fn=_collate_replay_packs,
            persistent_workers=(num_workers > 0),
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            pin_memory=False,
            generator=generator,
            worker_init_fn=_limit_worker_threads if limit_worker_threads else None,
            in_order=True,
        )
    return ReservoirLoader(
        pack_loader,
        stats=stats,
        L_ctx=L_ctx,
        batch_size=batch_size,
        capacity=reservoir_capacity,
        seed=seed,
        extra=extra,
        projection=projection,
        batch_transform=batch_transform,
        prefetch_batches=prefetch_batches,
        pin_memory=pin_memory,
        dataset=dataset,
        source_names=source_names,
        packs=packs,
        worker_independent_resume=worker_independent_resume,
        cooldown_batches=cooldown_batches,
        task_sampler=task_sampler,
        replay_prefetch_capacity=replay_prefetch_capacity,
    )
