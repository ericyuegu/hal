"""MDS loaders shared by training experiments."""

import functools
import hashlib
import math
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
from collections.abc import Sized
from pathlib import Path
from typing import Any
from typing import Literal

import numpy as np
import torch
from streaming import Stream
from streaming import StreamingDataLoader
from streaming import StreamingDataset
from streaming.base.world import World
from torch.utils.data import IterableDataset
from torch.utils.data import get_worker_info

from hal.data.feature_stats import FeatureStats
from hal.data.policy_schema import decode_policy_replay
from hal.data.policy_world_schema import decode_policy_world_replay
from hal.data.schema import SCHEMA_VERSION
from hal.data.schema import check_schema_version
from hal.data.streaming_compat import patch_streaming
from hal.streams import StreamSource
from hal.training.features import Context
from hal.training.features import ExtraColumns
from hal.training.features import FeatureProjection
from hal.training.features import TrainBatch
from hal.training.features import preprocess
from hal.training.features import stack_actions

patch_streaming()

# Frozen val-window geometry shared by every experiment: the val loader is always built with this
# ``L_chunk`` so val windows — hence val NLLs — are comparable across experiments regardless of each
# run's train-time ``L_chunk``. ``_choose_chunk_starts`` draws windows in a way that depends on
# ``L_chunk`` (its valid chunk-start support ``[1, T - L_chunk]`` and the RNG stream it consumes), so
# a val loader built with each run's own ``L_chunk`` samples different windows and makes val losses
# incomparable. Wide enough to cover multi-frame target horizons (e.g. 012's farthest auxiliary head);
# each experiment slices the target down to the frames it scores.
VAL_L_CHUNK = 16

# Public loader seams. A replay transform receives a schema-checked full replay
# and returns the replay that the window sampler reads. A batch transform receives
# the source windows and the normal TrainBatch. It can return an experiment batch.
type ReplayRow = dict[str, np.ndarray | int]
type ReplayTransform = Callable[[ReplayRow], ReplayRow]
type Window = dict[str, np.ndarray | np.integer]
type BatchTransform = Callable[[list[Window], TrainBatch], object]
type ReplayFormat = Literal["full", "policy", "policy-world"]

_STREAMING_EPOCH = "_hal_streaming_epoch"
_REPLAY_SEED_LOW = "_hal_replay_seed_low"
_REPLAY_SEED_HIGH = "_hal_replay_seed_high"


def _loader_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


class PolicyReplayDataset(IterableDataset):
    def __init__(self, dataset: StreamingDataset, replay_format: ReplayFormat = "policy") -> None:
        if replay_format not in ("policy", "policy-world"):
            raise ValueError(f"compact replay format must be 'policy' or 'policy-world', got {replay_format!r}")
        self._dataset = dataset
        self._decode = decode_policy_replay if replay_format == "policy" else decode_policy_world_replay

    def __iter__(self) -> Iterator[dict[str, np.ndarray | int]]:
        for sample in self._dataset:
            replay_id = sample.get("replay_id")
            if not isinstance(replay_id, str):
                raise TypeError(f"policy replay_id must be a string, got {type(replay_id).__name__}")
            digest = hashlib.blake2b(replay_id.encode(), digest_size=8).digest()
            decoded = self._decode(sample)
            decoded[_STREAMING_EPOCH] = int(self._dataset.next_epoch) - 1
            decoded[_REPLAY_SEED_LOW] = int.from_bytes(digest[:4], "little")
            decoded[_REPLAY_SEED_HIGH] = int.from_bytes(digest[4:], "little")
            yield decoded


def _resolve_replay_format(replay_format: ReplayFormat | None, compact: bool) -> ReplayFormat:
    if replay_format is None:
        return "policy" if compact else "full"
    if replay_format not in ("full", "policy", "policy-world"):
        raise ValueError(f"unknown replay_format {replay_format!r}")
    if compact and replay_format != "policy":
        raise ValueError("compact=True is the compatibility spelling of replay_format='policy'; do not pass both")
    return replay_format


def _make_streaming_dataset(
    data_root: str | None,
    split: str,
    *,
    sources: Sequence[StreamSource] | None,
    remote: str | None,
    shuffle: bool | None,
    shuffle_seed: int | None,
    cache_limit: str | int | None,
    shuffle_block_size: int | None,
    predownload: int | None,
    source_weights: Sequence[float] | None = None,
) -> tuple[StreamingDataset, tuple[str, ...]]:
    """Build one single- or multi-stream MDS dataset with strict mode selection."""
    if sources is not None:
        if data_root is not None or remote is not None:
            raise ValueError("sources cannot be combined with data_root or remote")
        if not sources:
            raise ValueError("sources must not be empty")
        names = tuple(source.name for source in sources)
        if len(set(names)) != len(names):
            raise ValueError("source names must be unique")
        if source_weights is not None:
            if len(source_weights) != len(sources):
                raise ValueError(f"source_weights length {len(source_weights)} != source count {len(sources)}")
            if any(not math.isfinite(weight) or weight <= 0 for weight in source_weights):
                raise ValueError("source_weights must be finite and positive")
            weight_total = sum(source_weights)
            proportions: Sequence[float | None] = tuple(weight / weight_total for weight in source_weights)
        else:
            proportions = (None,) * len(sources)
        selected_streams = [
            Stream(
                remote=source.remote,
                local=str(source.local_root),
                split=split,
                proportion=proportion,
            )
            for source, proportion in zip(sources, proportions, strict=True)
        ]
        kwargs: dict[str, Any] = {
            "streams": selected_streams,
            "cache_limit": cache_limit,
            "predownload": predownload,
        }
    else:
        if source_weights is not None:
            raise ValueError("source_weights requires sources")
        if data_root is None:
            raise ValueError("data_root is required when sources are not provided")
        names = ()
        kwargs = {
            "remote": f"{remote}/{split}" if remote else None,
            "local": str(Path(data_root) / split),
            "cache_limit": cache_limit if remote else None,
            "predownload": predownload if remote else None,
        }
    if shuffle_seed is not None:
        kwargs["shuffle_seed"] = shuffle_seed
    dataset = StreamingDataset(
        **kwargs,
        batch_size=1,
        shuffle=(split == "train") if shuffle is None else shuffle,
        shuffle_block_size=shuffle_block_size,
    )
    return dataset, names


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
) -> Window:
    relative = relabel_ego(sample, ego_prefix)
    if projection is not None:
        relative = {k: v for k, v in relative.items() if k in projection.columns}
    else:
        relative.pop("schema_version", None)
    stop = start + length
    out: Window = {}
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
        replay_transform: ReplayTransform | None = None,
    ) -> None:
        self._mds = mds
        self.L_ctx = L_ctx
        self.L_chunk = L_chunk
        self._L = L_ctx + L_chunk
        self._seed = seed
        self._K = windows_per_replay
        self._schema_version = schema_version
        self._projection = projection
        self._replay_transform = replay_transform
        self._epoch = 0

    def __iter__(self) -> Iterator[Window]:
        # Seed per (seed, worker, epoch): reproducible across runs (fixed seed),
        # distinct per worker, and still varying each epoch so a fixed seed
        # doesn't freeze train to one window per replay. Compact streaming rows
        # replace this fallback with a replay-and-epoch seed so a resumed worker
        # chooses the same window without replaying earlier RNG draws.
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        rng = np.random.default_rng((self._seed, worker_id, self._epoch))
        self._epoch += 1
        for sample in self._mds:
            check_schema_version(sample, expected=self._schema_version)
            streaming_seed = tuple(
                sample.pop(name, None) for name in (_STREAMING_EPOCH, _REPLAY_SEED_LOW, _REPLAY_SEED_HIGH)
            )
            if any(value is not None for value in streaming_seed):
                if not all(isinstance(value, int) for value in streaming_seed):
                    raise ValueError("streaming replay metadata is incomplete")
                sample_rng = np.random.default_rng((self._seed, *streaming_seed))
            else:
                sample_rng = rng
            if self._replay_transform is not None:
                sample = self._replay_transform(sample)
            frame = sample["frame"]
            if not isinstance(frame, np.ndarray):
                raise TypeError(f"frame must be an array, got {type(frame).__name__}")
            T = len(frame)
            # chunk[0] targets episode frame ``cs``; context is the L_ctx frames
            # before it. The K chunk-starts keep >=1 real context frame (the
            # cold-start floor: inference always has the just-observed frame), the
            # L_chunk-long chunk inside the episode, and their windows disjoint.
            for cs in _choose_chunk_starts(T, self.L_ctx, self.L_chunk, self._K, sample_rng):
                cs = int(cs)
                start = cs - self.L_ctx  # virtual window start; < 0 ⇒ left-pad
                pad = max(0, -start)
                ego_prefix = "p1" if sample_rng.random() < 0.5 else "p2"
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


def _collate_with_batch_transform(
    batch: list[dict],
    *,
    stats: dict[str, FeatureStats],
    L_ctx: int,
    extra: ExtraColumns | None,
    projection: FeatureProjection | None,
    batch_transform: BatchTransform | None,
) -> object:
    train_batch = collate_train_batch(
        batch,
        stats=stats,
        L_ctx=L_ctx,
        extra=extra,
        projection=projection,
    )
    return batch_transform(batch, train_batch) if batch_transform is not None else train_batch


class ResumableStreamingDataLoader(StreamingDataLoader):
    """A transformed DataLoader with Mosaic Streaming checkpoint state."""

    def __init__(self, *args, streaming_dataset: StreamingDataset, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._streaming_dataset = streaming_dataset
        self.num_samples_yielded = 0

    def _get_batch_size(self, batch: object) -> int:
        if isinstance(batch, TrainBatch):
            return batch.target.shape[0]
        transformed = getattr(batch, "batch", None)
        if isinstance(transformed, TrainBatch):
            return transformed.target.shape[0]
        if isinstance(batch, Mapping):
            try:
                value = next(iter(batch.values()))
            except StopIteration as error:
                raise ValueError("batch is empty") from error
            if not isinstance(value, Sized):
                raise TypeError(f"batch value {type(value).__name__} has no length")
            return len(value)
        if isinstance(batch, torch.Tensor):
            return len(batch)
        if isinstance(batch, Sequence) and batch:
            value = batch[0]
            if isinstance(value, Sized):
                return len(value)
        raise TypeError(f"cannot determine batch size from {type(batch).__name__}")

    def state_dict(self) -> dict[str, Any]:
        """Return Mosaic's deterministic position for samples yielded this epoch."""
        world = World.detect()
        num_samples = self.num_samples_yielded * world.num_ranks
        if self._streaming_dataset.replication is not None:
            num_samples //= self._streaming_dataset.replication
        return self._streaming_dataset.state_dict(num_samples, False)

    def load_state_dict(self, obj: dict[str, Any]) -> None:
        """Restore Mosaic's position before workers create their next iterator."""
        self._streaming_dataset.load_state_dict(obj)


def make_loader(
    data_root: str | None,
    split: str,
    *,
    stats: dict[str, FeatureStats],
    L_ctx: int,
    L_chunk: int,
    batch_size: int,
    seed: int,
    remote: str | None = None,
    sources: Sequence[StreamSource] | None = None,
    source_weights: Sequence[float] | None = None,
    cache_limit: str | int | None = None,
    shuffle_block_size: int | None = None,
    shuffle: bool | None = None,
    shuffle_seed: int | None = None,
    num_workers: int = 4,
    prefetch_factor: int = 4,
    drop_last: bool = False,
    predownload: int | None = None,
    pin_memory: bool | None = None,
    windows_per_replay: int = 1,
    schema_version: int = SCHEMA_VERSION,
    extra: ExtraColumns | None = None,
    projection: FeatureProjection | None = None,
    compact: bool = False,
    replay_format: ReplayFormat | None = None,
    replay_transform: ReplayTransform | None = None,
    batch_transform: BatchTransform | None = None,
) -> ResumableStreamingDataLoader:
    """Build the (StreamingDataset → WindowDataset → DataLoader) chain. The
    DataLoader yields ``TrainBatch`` by default (preprocessing runs in the
    workers). ``replay_transform`` runs after schema validation and before
    window sampling. ``batch_transform`` runs after the normal TrainBatch is
    built. It receives the source windows and may return an experiment wrapper.
    Both callables run in loader workers, so they must be picklable when
    ``num_workers`` is positive.

    ``sources`` selects multi-stream mode and cannot be combined with
    ``data_root`` or ``remote``. Otherwise, ``remote`` is the dataset's R2 root
    URI; when set, StreamingDataset pulls the split's shards on demand into the
    ``data_root`` cache. Without it, ``data_root`` must already hold the shards.

    ``cache_limit`` bounds that local shard cache (e.g. ``"100gb"``) so a dataset
    far larger than disk streams without filling it — StreamingDataset evicts
    least-recently-used shards past the limit. Only meaningful with ``remote`` set:
    a local-only dataset has nowhere to re-download an evicted shard from, so it's
    ignored only for a single local-only dataset.

    ``shuffle_block_size`` is the py1e shuffle unit (samples mixed together). It
    governs *startup* download: py1e must buffer a block before yielding, so the
    default (``max(4e6 // num_canonical_nodes, 2**18)`` ≈ 4M samples) buffers the
    whole dataset when it has fewer samples than that — downloading everything
    before the first batch. Set it to a few shards' worth of samples to start fast;
    smaller trades global-shuffle quality for a lighter startup.

    ``ResumableStreamingDataLoader`` bridges the transformed ``WindowDataset``
    back to its underlying ``StreamingDataset``. Its state uses Mosaic's own
    epoch and sample offset while counting ``TrainBatch`` and experiment batch
    sizes, so checkpoints retain the exact shuffled replay position."""
    # ``predownload`` is how many samples each worker fetches ahead — the shard-prefetch
    # depth that pipelines remote (R2) downloads. StreamingDataset ties its default to
    # batch_size (``8 * batch_size``) and we pass batch_size=1, so it was only 8: the fast
    # GPU stalled on every shard miss. Set it explicitly for the *remote* path. For a
    # local-only dataset there's no download latency to hide — and over-prefetching a
    # partial local cache would try to fetch shards that aren't there — so keep streaming's
    # conservative default there.
    if predownload is None:
        predownload = 8 * batch_size if remote or sources is not None else None
    resolved_format = _resolve_replay_format(replay_format, compact)
    mds, _ = _make_streaming_dataset(
        data_root,
        split,
        sources=sources,
        source_weights=source_weights,
        remote=remote,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
        cache_limit=cache_limit,
        shuffle_block_size=shuffle_block_size,
        predownload=predownload,
    )
    rows = PolicyReplayDataset(mds, resolved_format) if resolved_format != "full" else mds
    sampler = WindowDataset(
        rows,
        L_ctx,
        L_chunk,
        seed=seed,
        windows_per_replay=windows_per_replay,
        schema_version=schema_version,
        projection=projection,
        replay_transform=replay_transform,
    )
    collate = functools.partial(
        _collate_with_batch_transform,
        stats=stats,
        L_ctx=L_ctx,
        extra=extra,
        projection=projection,
        batch_transform=batch_transform,
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
    return ResumableStreamingDataLoader(
        sampler,
        streaming_dataset=mds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=drop_last,
        pin_memory=pin_memory,
        generator=_loader_generator(seed),
    )
