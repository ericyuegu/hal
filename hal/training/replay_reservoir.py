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
from typing import Any

import numpy as np
import torch
from streaming import StreamingDataset
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset

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

    def __post_init__(self) -> None:
        if not self.windows:
            raise ValueError("a replay pack must contain at least one window")


@dataclass(frozen=True, slots=True)
class _CollatedReplayPackBatch:
    """Worker-stacked replay windows transferred through shared tensors."""

    replay_ids: tuple[str, ...]
    pack_sizes: tuple[int, ...]
    columns: dict[str, torch.Tensor]

    def unpack(self) -> tuple[ReplayPack, ...]:
        packs: list[ReplayPack] = []
        offset = 0
        for replay_id, size in zip(self.replay_ids, self.pack_sizes, strict=True):
            windows = tuple(
                {name: values[index].numpy() for name, values in self.columns.items()}
                for index in range(offset, offset + size)
            )
            packs.append(ReplayPack(replay_id, windows))
            offset += size
        if offset != next(iter(self.columns.values())).shape[0]:
            raise RuntimeError("collated replay-pack row count changed during transfer")
        return tuple(packs)


def _collate_replay_packs(packs: list[ReplayPack]) -> _CollatedReplayPackBatch:
    """Stack worker results so Torch shares storage instead of pickling each array."""
    if not packs:
        raise ValueError("cannot collate an empty replay-pack batch")
    windows = [window for pack in packs for window in pack.windows]
    keys = tuple(windows[0])
    if not keys:
        raise ValueError("cannot collate replay windows without columns")
    if any(set(window) != set(keys) for window in windows[1:]):
        raise ValueError("replay windows have inconsistent columns")
    columns = {name: torch.from_numpy(np.stack([np.asarray(window[name]) for window in windows])) for name in keys}
    return _CollatedReplayPackBatch(
        replay_ids=tuple(pack.replay_id for pack in packs),
        pack_sizes=tuple(len(pack.windows) for pack in packs),
        columns=columns,
    )


@dataclass(frozen=True, slots=True)
class ReservoirBatch:
    replay_ids: tuple[str, ...]
    windows: tuple[dict[str, np.ndarray], ...]


class ReplayPackBatchIterator(Iterator[ReplayPack]):
    """Flatten worker-side pack batches while retaining checkpointable lookahead."""

    def __init__(self, source: Iterator[object], visits: Counter[str]) -> None:
        self._source = source
        self._visits = visits
        self._pending: deque[ReplayPack] = deque()
        self.received_packs = 0

    def __iter__(self) -> ReplayPackBatchIterator:
        return self

    def __next__(self) -> ReplayPack:
        while not self._pending:
            value = next(self._source)
            if isinstance(value, ReplayPack):
                batch = (value,)
            elif isinstance(value, _CollatedReplayPackBatch):
                batch = value.unpack()
            elif isinstance(value, list | tuple) and all(isinstance(pack, ReplayPack) for pack in value):
                batch = tuple(value)
            else:
                raise TypeError(f"pack loader yielded {type(value).__name__}, expected ReplayPack batch")
            if not batch:
                continue
            self.received_packs += len(batch)
            for pack in batch:
                self._visits[pack.replay_id] += 1
            self._pending.extend(batch)
        return self._pending.popleft()

    def state_dict(self) -> dict[str, object]:
        """Return packs fetched by a worker batch but not yet read by the reservoir."""
        return {"pending": tuple(self._pending)}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        pending = state.get("pending", ())
        if not isinstance(pending, tuple) or any(not isinstance(pack, ReplayPack) for pack in pending):
            raise ValueError("pending replay-pack state is invalid")
        self._pending = deque(pending)


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
        return {
            "batch_size": self._batch_size,
            "capacity": self._capacity,
            "rng": self._rng.bit_generator.state,
            "active": tuple((replay_id, tuple(windows)) for replay_id, windows in self._active.items()),
            "cooldown": tuple(tuple(sorted(replays)) for replays in self._cooldown),
            "cooldown_batches": self._cooldown.maxlen,
            "source_done": self._source_done,
            "finished": self._finished,
            "emitted_windows": self._emitted_windows,
            "dropped_windows": self._dropped_windows,
            "dropped_replays": self._dropped_replays,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a state produced after a complete batch was emitted."""
        if state["batch_size"] != self._batch_size or state["capacity"] != self._capacity:
            raise ValueError("reservoir geometry changed across resume")
        if state["cooldown_batches"] != self._cooldown.maxlen:
            raise ValueError("reservoir cooldown changed across resume")
        self._rng.bit_generator.state = state["rng"]
        self._active = {replay_id: deque(windows) for replay_id, windows in state["active"]}
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
            order = self._rng.permutation(len(pack.windows))
            windows = (pack.windows[int(index)] for index in order)
            if pack.replay_id in self._active:
                # Weighted streams can repeat a small source within one epoch.
                # Keep its repeated windows, but retain one active replay key so
                # a batch never contains the same replay more than once.
                self._active[pack.replay_id].extend(windows)
            else:
                self._active[pack.replay_id] = deque(windows)

    def _finish(self) -> None:
        if self._finished:
            return
        self._dropped_windows = sum(len(windows) for windows in self._active.values())
        self._dropped_replays = len(self._active)
        self._active.clear()
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
            if not self._active[replay_id]:
                del self._active[replay_id]
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

    def __iter__(self) -> Iterator[ReplayPack]:
        # Defer shared helpers so dataloader can re-export this module safely.
        from hal.training.dataloader import _choose_chunk_starts
        from hal.training.dataloader import _make_window

        epoch = self._epoch
        self._epoch += 1
        self._current_epoch = epoch
        self._source_samples = 0
        for compact in self._dataset:
            self._source_samples += 1
            replay_id = str(compact["replay_id"])
            if self._replay_filter is not None and not self._replay_filter(replay_id):
                continue
            check_schema_version(
                {"schema_version": int(compact["source_schema_version"])}, expected=self._schema_version
            )
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
                continue
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
                wrong_labels = {
                    name: value.shape for name, value in labels.items() if value.shape not in ((), (frames,))
                }
                if wrong_labels:
                    raise ValueError(
                        f"replay labels have invalid shapes {wrong_labels}; expected scalar or {(frames,)}"
                    )
                for start, ego_prefix, frame_range, sample in zip(starts, ego_prefixes, ranges, samples, strict=True):
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
            if windows:
                yield ReplayPack(replay_id, tuple(windows))


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
        self._epoch = 0
        self._reservoir: ReplayReservoir | None = None
        self._pack_batches: ReplayPackBatchIterator | None = None
        self._resume_state: dict[str, Any] | None = None
        self._visits: Counter[str] = Counter()
        self.last_epoch_stats: dict[str, int] | None = None

    @property
    def source_sample_counts(self) -> dict[str, int]:
        """Per-stream sample counts in configured source order."""
        if not self._source_names:
            return {}
        counts = getattr(self._dataset, "selected_samples_per_stream", self._dataset.samples_per_stream)
        return {name: int(count) for name, count in zip(self._source_names, counts, strict=True)}

    def __iter__(self) -> Iterator[object]:
        pack_batches = ReplayPackBatchIterator(iter(self._pack_loader), self._visits)
        if self._resume_state is not None:
            pack_batches.load_state_dict(self._resume_state.get("pack_batches", {}))
        reservoir = ReplayReservoir(
            pack_batches,
            batch_size=self._batch_size,
            capacity=self._capacity,
            seed=self._seed + self._epoch,
        )
        if self._resume_state is not None:
            reservoir.load_state_dict(self._resume_state["reservoir"])
            self._resume_state = None
        self._reservoir = reservoir
        self._pack_batches = pack_batches
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
            "visit_counters": dict(self._visits),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Schedule an exact restore before the next iterator is created."""
        if state.get("schema") not in (1, 2):
            raise ValueError(f"unsupported reservoir-loader state schema {state.get('schema')!r}")
        if self._prefetch_batches:
            raise RuntimeError("exact reservoir checkpoints require prefetch_batches=0")
        if self._pack_loader.num_workers != 0 and not self._worker_independent_resume:
            raise RuntimeError("exact worker resume must be enabled when num_workers is nonzero")
        if self._reservoir is not None:
            raise RuntimeError("load reservoir state before creating its iterator")
        self._dataset.load_state_dict(state["mds"])
        self._packs.resume_epoch(int(state["pack_epoch"]))
        self._epoch = int(state["loader_epoch"]) - 1
        self._visits = Counter({str(name): int(count) for name, count in state.get("visit_counters", {}).items()})
        self._resume_state = {
            **state,
            "pack_batches": state.get("pack_batches", {"pending": ()}),
        }

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
    )
