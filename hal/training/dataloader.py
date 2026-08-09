"""MDS loaders shared by training experiments."""

import contextlib
import functools
import hashlib
from collections import deque
from collections.abc import Iterable
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from streaming import StreamingDataset
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
from torch.utils.data import get_worker_info

from hal.data.feature_stats import FeatureStats
from hal.data.policy_schema import decode_policy_replay
from hal.data.policy_schema import decode_policy_replay_slices
from hal.data.schema import SCHEMA_VERSION
from hal.data.schema import check_schema_version
from hal.data.streaming_compat import patch_streaming_resource_tracker
from hal.training.features import Context
from hal.training.features import ExtraColumns
from hal.training.features import FeatureProjection
from hal.training.features import TrainBatch
from hal.training.features import preprocess
from hal.training.features import stack_actions

patch_streaming_resource_tracker()

# Frozen val-window geometry shared by every experiment: the val loader is always built with this
# ``L_chunk`` so val windows — hence val NLLs — are comparable across experiments regardless of each
# run's train-time ``L_chunk``. ``_choose_chunk_starts`` draws windows in a way that depends on
# ``L_chunk`` (its valid chunk-start support ``[1, T - L_chunk]`` and the RNG stream it consumes), so
# a val loader built with each run's own ``L_chunk`` samples different windows and makes val losses
# incomparable. Wide enough to cover multi-frame target horizons (e.g. 012's farthest auxiliary head);
# each experiment slices the target down to the frames it scores.
VAL_L_CHUNK = 16


def _next_item[T](source: Iterator[T]) -> T:
    return next(source)


class OneBatchPrefetch[T](Iterator[T]):
    def __init__(self, source: Iterator[T], *, depth: int = 1) -> None:
        if not isinstance(depth, int) or isinstance(depth, bool) or depth <= 0:
            raise ValueError(f"prefetch depth must be a positive integer, got {depth!r}")
        self._source = source
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="train-batch")
        # A single worker advances the generator safely and serially, while
        # several queued futures let it prepare a complete accumulation window
        # during the preceding GPU step.
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
        if wait:
            for future in futures:
                with contextlib.suppress(BaseException):
                    future.result()
        else:
            for future in futures:
                future.cancel()
        self._pool.shutdown(wait=wait, cancel_futures=True)
        if wait or all(future.done() for future in futures):
            close = getattr(self._source, "close", None)
            if close is not None:
                close()


def _loader_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


@dataclass(frozen=True, slots=True)
class ReplayPack:
    replay_id: str
    windows: tuple[dict[str, np.ndarray], ...]

    def __post_init__(self) -> None:
        if not self.windows:
            raise ValueError("a replay pack must contain at least one window")


@dataclass(frozen=True, slots=True)
class ReservoirBatch:
    replay_ids: tuple[str, ...]
    windows: tuple[dict[str, np.ndarray], ...]


class ReplayReservoir:
    """Mix replay packs while emitting at most one window per replay in a batch.

    A replay cannot return until ``cooldown_batches`` complete batches have
    passed. The capacity check makes that rule possible while the source can
    still fill the reservoir. At the end of an epoch, only complete batches
    that satisfy the cooldown are emitted. All other active windows are counted
    as the dropped tail.
    """

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

    def _fill(self) -> None:
        while len(self._active) < self._capacity and not self._source_done:
            try:
                pack = next(self._source)
            except StopIteration:
                self._source_done = True
                break
            if pack.replay_id in self._active:
                raise ValueError(f"replay {pack.replay_id!r} is already active")
            order = self._rng.permutation(len(pack.windows))
            self._active[pack.replay_id] = deque(pack.windows[int(i)] for i in order)

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
        selected_at = self._rng.choice(len(candidates), size=self._batch_size, replace=False)
        replay_ids = tuple(candidates[int(i)] for i in selected_at)
        windows = tuple(self._active[replay_id].popleft() for replay_id in replay_ids)
        for replay_id in replay_ids:
            if not self._active[replay_id]:
                del self._active[replay_id]
        if self._cooldown.maxlen:
            self._cooldown.append(set(replay_ids))
        self._emitted_windows += self._batch_size
        return ReservoirBatch(replay_ids=replay_ids, windows=windows)


class PolicyReplayDataset(IterableDataset):
    def __init__(self, dataset: StreamingDataset) -> None:
        self._dataset = dataset

    def __iter__(self) -> Iterator[dict[str, np.ndarray | int]]:
        for sample in self._dataset:
            yield decode_policy_replay(sample)


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
    ) -> None:
        self._dataset = dataset
        self._L_ctx = L_ctx
        self._L_chunk = L_chunk
        self._seed = seed
        self._K = windows_per_replay
        self._schema_version = schema_version
        self._projection = projection
        self._epoch = 0

    def __iter__(self) -> Iterator[ReplayPack]:
        epoch = self._epoch
        self._epoch += 1
        for compact in self._dataset:
            replay_id = str(compact["replay_id"])
            check_schema_version(
                {"schema_version": int(compact["source_schema_version"])}, expected=self._schema_version
            )
            frames = int(compact["num_frames"])
            rng = _stable_replay_rng(self._seed, epoch, replay_id)
            starts = [
                int(cs) - self._L_ctx for cs in _choose_chunk_starts(frames, self._L_ctx, self._L_chunk, self._K, rng)
            ]
            ego_prefixes = ["p1" if rng.random() < 0.5 else "p2" for _ in starts]
            ranges = tuple((max(0, start), start + self._L_ctx + self._L_chunk) for start in starts)
            samples = decode_policy_replay_slices(compact, ranges)
            windows = []
            for start, ego_prefix, sample in zip(starts, ego_prefixes, samples, strict=True):
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
        batch_prefetch: bool,
        batch_prefetch_depth: int,
        pin_memory: bool,
    ) -> None:
        self._pack_loader = pack_loader
        self._stats = stats
        self._L_ctx = L_ctx
        self._batch_size = batch_size
        self._capacity = capacity
        self._seed = seed
        self._extra = extra
        self._projection = projection
        self._batch_prefetch = batch_prefetch
        self._batch_prefetch_depth = batch_prefetch_depth
        self._pin_memory = pin_memory
        self._epoch = 0
        self.last_epoch_stats: dict[str, int] | None = None

    def __iter__(self) -> Iterator[TrainBatch]:
        reservoir = ReplayReservoir(
            iter(self._pack_loader),
            batch_size=self._batch_size,
            capacity=self._capacity,
            seed=self._seed + self._epoch,
        )
        self._epoch += 1
        batches = self._batches(reservoir)
        return OneBatchPrefetch(batches, depth=self._batch_prefetch_depth) if self._batch_prefetch else batches

    def _batches(self, reservoir: ReplayReservoir) -> Iterator[TrainBatch]:
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
                yield batch.pin_memory() if self._pin_memory else batch
        finally:
            self.last_epoch_stats = {
                "emitted_windows": reservoir.emitted_windows,
                "dropped_windows": reservoir.dropped_windows,
                "dropped_replays": reservoir.dropped_replays,
            }


def _identity(value: Any) -> Any:
    return value


def relabel_ego(window: dict[str, np.ndarray], ego_prefix: str) -> dict[str, np.ndarray]:
    """Rename p1_*/p2_* keys to ego_*/opp_* based on `ego_prefix`."""
    opp_prefix = "p2" if ego_prefix == "p1" else "p1"
    rel: dict[str, np.ndarray] = {}
    for k, v in window.items():
        if k.startswith(f"{ego_prefix}_"):
            rel[f"ego_{k[3:]}"] = v
        elif k.startswith(f"{opp_prefix}_"):
            rel[f"opp_{k[3:]}"] = v
        else:
            rel[k] = v
    return rel


def _make_window(
    sample: dict,
    *,
    ego_prefix: str,
    start: int,
    pad: int,
    length: int,
    projection: FeatureProjection | None,
) -> dict[str, np.ndarray]:
    relative = relabel_ego(sample, ego_prefix)
    if projection is not None:
        relative = {k: v for k, v in relative.items() if k in projection.columns}
    else:
        relative.pop("schema_version", None)
    stop = start + length
    out = {}
    for name, values in relative.items():
        real = values[max(0, start) : stop]
        if pad:
            front = np.zeros((pad, *values.shape[1:]), dtype=values.dtype)
            real = np.concatenate([front, real], axis=0)
        out[name] = real
    return out


def _choose_chunk_starts(T: int, L_ctx: int, L_chunk: int, K: int, rng: np.random.Generator) -> np.ndarray:
    """Up to ``K`` chunk-start positions in ``[1, T - L_chunk]`` whose windows
    ``[cs - L_ctx, cs + L_chunk)`` are pairwise non-overlapping.

    Reading a whole replay off disk to emit a single ~``L_ctx+L_chunk`` window
    wastes ~99% of the bytes read; emitting ``K`` windows amortizes that read.
    The positions are stratified into ``k`` equal lanes (one window per lane,
    randomly placed within it) so the windows spread across the episode and stay
    distinct rather than clustering. ``k`` clamps to what the episode can fit, so
    short replays yield fewer than ``K``. ``K=1`` reduces to a single window drawn
    uniformly over the full range (the historical behavior)."""
    L = L_ctx + L_chunk
    cs_lo, cs_hi = 1, T - L_chunk
    span = cs_hi - cs_lo + 1
    if span < 1:
        return np.empty(0, dtype=np.int64)
    k = min(K, max(1, span // L))
    stride = span // k
    lane_lo = cs_lo + np.arange(k) * stride
    # Each lane's window must end ``L`` before the next lane starts (non-overlap);
    # the last lane has no successor, so it can range to the end of the episode.
    lane_hi = lane_lo + (stride - L)
    lane_hi[-1] = cs_hi
    lane_hi = np.maximum(lane_hi, lane_lo)
    return lane_lo + rng.integers(0, lane_hi - lane_lo + 1)


class WindowDataset(IterableDataset):
    """Wrap a StreamingDataset: pick a random ego port and ``windows_per_replay``
    non-overlapping length-``L_ctx + L_chunk`` windows from each replay, laid out
    as ``[ctx | chunk]``. Relabel p1/p2 → ego/opp before yielding.

    The window is anchored by its *chunk* position, drawn uniformly over the
    whole episode — including the opening frames. When the chunk sits near the
    start, the context runs off the front of the episode; those missing frames
    are zero-padded on the left and reported as ``ctx_pad`` so the model masks
    them from attention. This makes the episode's first frames real prediction
    targets (no skipping), matching the closed-loop cold start where the rolling
    buffer fills from empty. Each emitted window carries an int ``ctx_pad``.

    This is a neutral obs→action-chunk window: it knows nothing about latency or
    real-time chunking. An RTC experiment that conditions on already-committed
    actions slices that prefix out of the chunk itself (its first frames).
    """

    def __init__(
        self,
        mds: Iterable[dict],
        L_ctx: int,
        L_chunk: int,
        *,
        seed: int,
        windows_per_replay: int = 1,
        schema_version: int = SCHEMA_VERSION,
        projection: FeatureProjection | None = None,
    ) -> None:
        self._mds = mds
        self.L_ctx = L_ctx
        self.L_chunk = L_chunk
        self._L = L_ctx + L_chunk
        self._seed = seed
        self._K = windows_per_replay
        self._schema_version = schema_version
        self._projection = projection
        self._epoch = 0

    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        # Seed per (seed, worker, epoch): reproducible across runs (fixed seed),
        # distinct per worker, and still varying each epoch so a fixed seed
        # doesn't freeze train to one window per replay. Persistent workers keep
        # _epoch advancing across epochs.
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        rng = np.random.default_rng((self._seed, worker_id, self._epoch))
        self._epoch += 1
        for sample in self._mds:
            check_schema_version(sample, expected=self._schema_version)
            T = len(sample["frame"])
            # chunk[0] targets episode frame ``cs``; context is the L_ctx frames
            # before it. The K chunk-starts keep >=1 real context frame (the
            # cold-start floor: inference always has the just-observed frame), the
            # L_chunk-long chunk inside the episode, and their windows disjoint.
            for cs in _choose_chunk_starts(T, self.L_ctx, self.L_chunk, self._K, rng):
                cs = int(cs)
                start = cs - self.L_ctx  # virtual window start; < 0 ⇒ left-pad
                pad = max(0, -start)
                ego_prefix = "p1" if rng.random() < 0.5 else "p2"
                window = _make_window(
                    sample,
                    ego_prefix=ego_prefix,
                    start=start,
                    pad=pad,
                    length=self._L,
                    projection=self._projection,
                )
                window["ctx_pad"] = np.int64(min(pad, self.L_ctx))
                yield window


def collate_windows(batch: list[dict]) -> dict[str, np.ndarray]:
    """Stack a list of ``[seq]`` per-sample windows into ``[B, seq]`` columns."""
    keys = batch[0].keys()
    return {k: np.stack([s[k] for s in batch]) for k in keys}


def collate_train_batch(
    batch: list[dict],
    *,
    stats: dict[str, FeatureStats],
    L_ctx: int,
    extra: ExtraColumns | None = None,
    projection: FeatureProjection | None = None,
) -> TrainBatch:
    """Worker-side collate: stack → ``preprocess`` → split ``[ctx | chunk]``.

    The window the sampler yields is laid out ``[ctx | chunk]`` over
    ``seq = L_ctx + L_chunk`` frames. Context features are the first ``L_ctx``
    frames; the target action chunk is the remaining frames sliced off the
    stacked ego-action channels at ``[L_ctx :]``. Returns a fully-tensorized
    ``TrainBatch`` so the training loop does no reshaping — just ``.to(device)``.

    ``extra`` is the experiment's column routing beyond the built-in feature
    tables (see ``features.ExtraColumns``); the closed-loop policy must carry the
    same one so both observation paths build the same token.
    """
    stacked = collate_windows(batch)
    ctx_pad = torch.from_numpy(stacked["ctx_pad"].astype(np.int64))
    feats = preprocess(stacked, stats, extra=extra, projection=projection)
    actions = stack_actions(feats)
    context_features = {k: v[:, :L_ctx] for k, v in feats.items()}
    target = actions[:, L_ctx:]
    return TrainBatch(Context(features=context_features, ctx_pad=ctx_pad), target=target)


def make_loader(
    data_root: str,
    split: str,
    *,
    stats: dict[str, FeatureStats],
    L_ctx: int,
    L_chunk: int,
    batch_size: int,
    seed: int,
    remote: str | None = None,
    cache_limit: str | int | None = None,
    shuffle_block_size: int | None = None,
    num_workers: int = 4,
    prefetch_factor: int = 4,
    predownload: int | None = None,
    pin_memory: bool | None = None,
    windows_per_replay: int = 1,
    schema_version: int = SCHEMA_VERSION,
    extra: ExtraColumns | None = None,
    projection: FeatureProjection | None = None,
    compact: bool = False,
) -> DataLoader:
    """Build the (StreamingDataset → WindowDataset → DataLoader) chain. The
    DataLoader yields ``TrainBatch`` (preprocessing runs in the workers).

    ``remote`` is the dataset's R2 root URI; when set, StreamingDataset pulls the
    split's shards on demand into the ``data_root`` cache (cloud training). When
    None, ``data_root`` must already hold the shards (local dev/overfit).

    ``cache_limit`` bounds that local shard cache (e.g. ``"100gb"``) so a dataset
    far larger than disk streams without filling it — StreamingDataset evicts
    least-recently-used shards past the limit. Only meaningful with ``remote`` set:
    a local-only dataset has nowhere to re-download an evicted shard from, so it's
    ignored when ``remote`` is None.

    ``shuffle_block_size`` is the py1e shuffle unit (samples mixed together). It
    governs *startup* download: py1e must buffer a block before yielding, so the
    default (``max(4e6 // num_canonical_nodes, 2**18)`` ≈ 4M samples) buffers the
    whole dataset when it has fewer samples than that — downloading everything
    before the first batch. Set it to a few shards' worth of samples to start fast;
    smaller trades global-shuffle quality for a lighter startup.

    A plain ``DataLoader`` rather than ``StreamingDataLoader``: the latter's
    mid-epoch resumption only engages when its dataset *is* a StreamingDataset,
    but here that dataset is wrapped by ``WindowDataset``, so the wrapper's only
    live behavior would be a per-batch ``len(batch[0])`` sample count — which a
    ``TrainBatch`` (not dict/Tensor) can't satisfy. StreamingDataset still owns
    sharding/shuffle; it's iterated inside the sampler."""
    # ``predownload`` is how many samples each worker fetches ahead — the shard-prefetch
    # depth that pipelines remote (R2) downloads. StreamingDataset ties its default to
    # batch_size (``8 * batch_size``) and we pass batch_size=1, so it was only 8: the fast
    # GPU stalled on every shard miss. Set it explicitly for the *remote* path. For a
    # local-only dataset there's no download latency to hide — and over-prefetching a
    # partial local cache would try to fetch shards that aren't there — so keep streaming's
    # conservative default there.
    if predownload is None:
        predownload = 8 * batch_size if remote else None
    mds = StreamingDataset(
        remote=f"{remote}/{split}" if remote else None,
        local=str(Path(data_root) / split),
        batch_size=1,
        shuffle=(split == "train"),
        cache_limit=cache_limit if remote else None,
        shuffle_block_size=shuffle_block_size,
        predownload=predownload,
    )
    rows = PolicyReplayDataset(mds) if compact else mds
    sampler = WindowDataset(
        rows,
        L_ctx,
        L_chunk,
        seed=seed,
        windows_per_replay=windows_per_replay,
        schema_version=schema_version,
        projection=projection,
    )
    collate = functools.partial(
        collate_train_batch,
        stats=stats,
        L_ctx=L_ctx,
        extra=extra,
        projection=projection,
    )
    # Pin by default only when there's a GPU to copy to (page-locking host memory is
    # wasted on a CPU run). ``TrainBatch.pin_memory`` makes the custom batch poolable.
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    # Workers hand batch tensors to the main process via shared memory. The default
    # 'file_descriptor' strategy backs that with /dev/shm, whose size is host/container-fixed
    # (64MB on a stock vast box, and an on-start remount can fail or be undersized); at high
    # worker x prefetch x batch the in-flight tensors hit several GB and overrun it, killing
    # workers ("exited unexpectedly"). 'file_system' backs the handoff with TMPDIR files (the
    # overlay disk, page-cached) instead, so IPC capacity doesn't depend on /dev/shm size. Set
    # once in the main process before workers spawn (module stays import-clean for workers).
    if num_workers > 0:
        torch.multiprocessing.set_sharing_strategy("file_system")
    return DataLoader(
        sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=pin_memory,
        generator=_loader_generator(seed),
    )


def make_replay_reservoir_loader(
    data_root: str,
    split: str,
    *,
    stats: dict[str, FeatureStats],
    L_ctx: int,
    L_chunk: int,
    batch_size: int,
    seed: int,
    reservoir_capacity: int,
    remote: str | None = None,
    cache_limit: str | int | None = None,
    shuffle_block_size: int | None = None,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    predownload: int = 512,
    windows_per_replay: int = 4,
    batch_prefetch: bool = False,
    batch_prefetch_depth: int = 1,
    pin_memory: bool | None = None,
    schema_version: int = SCHEMA_VERSION,
    extra: ExtraColumns | None = None,
    projection: FeatureProjection | None = None,
) -> ReservoirLoader:
    if predownload < 1:
        raise ValueError(f"predownload must be positive, got {predownload}")
    if (
        not isinstance(batch_prefetch_depth, int)
        or isinstance(batch_prefetch_depth, bool)
        or batch_prefetch_depth <= 0
    ):
        raise ValueError(f"batch_prefetch_depth must be a positive integer, got {batch_prefetch_depth!r}")
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    mds = StreamingDataset(
        remote=f"{remote}/{split}" if remote else None,
        local=str(Path(data_root) / split),
        batch_size=1,
        shuffle=(split == "train"),
        cache_limit=cache_limit if remote else None,
        shuffle_block_size=shuffle_block_size,
        predownload=predownload if remote else None,
    )
    packs = PolicyReplayPackDataset(
        mds,
        L_ctx,
        L_chunk,
        seed=seed,
        windows_per_replay=windows_per_replay,
        schema_version=schema_version,
        projection=projection,
    )
    if num_workers > 0:
        torch.multiprocessing.set_sharing_strategy("file_system")
    pack_loader = DataLoader(
        packs,
        batch_size=None,
        num_workers=num_workers,
        collate_fn=_identity,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=False,
        generator=_loader_generator(seed),
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
        batch_prefetch=batch_prefetch,
        batch_prefetch_depth=batch_prefetch_depth,
        pin_memory=pin_memory,
    )
