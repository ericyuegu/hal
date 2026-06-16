"""MDS dataloader primitives shared across experiments.

Lives outside the experiment file so that forked / forkserver-spawned
DataLoader workers can re-import ``WindowDataset`` and the collate path
without re-running experiment-level module code. Keeping this file
side-effect-free is what makes that work — and is also why the per-feature
``preprocess`` runs here in the worker (emitting a ready-to-use ``TrainBatch``)
rather than on the training hot path.
"""

import functools
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from streaming import StreamingDataset
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
from torch.utils.data import get_worker_info

from hal.data.schema import check_schema_version
from hal.data.stats import FeatureStats
from hal.training.features import Context
from hal.training.features import TrainBatch
from hal.training.features import preprocess
from hal.training.features import stack_actions


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
        self, mds: StreamingDataset, L_ctx: int, L_chunk: int, *, seed: int, windows_per_replay: int = 1
    ) -> None:
        self._mds = mds
        self.L_ctx = L_ctx
        self.L_chunk = L_chunk
        self._L = L_ctx + L_chunk
        self._seed = seed
        self._K = windows_per_replay
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
            check_schema_version(sample)
            # Shallow copy without the row scalar; windowing slices every value.
            sample = {k: v for k, v in sample.items() if k != "schema_version"}
            T = len(sample["frame"])
            # chunk[0] targets episode frame ``cs``; context is the L_ctx frames
            # before it. The K chunk-starts keep >=1 real context frame (the
            # cold-start floor: inference always has the just-observed frame), the
            # L_chunk-long chunk inside the episode, and their windows disjoint.
            for cs in _choose_chunk_starts(T, self.L_ctx, self.L_chunk, self._K, rng):
                cs = int(cs)
                start = cs - self.L_ctx  # virtual window start; < 0 ⇒ left-pad
                pad = max(0, -start)
                window = self._padded_window(sample, start, pad)
                window["ctx_pad"] = np.int64(min(pad, self.L_ctx))
                ego_prefix = "p1" if rng.random() < 0.5 else "p2"
                yield relabel_ego(window, ego_prefix)

    def _padded_window(self, sample: dict[str, np.ndarray], start: int, pad: int) -> dict[str, np.ndarray]:
        """Length-``_L`` window beginning at virtual frame ``start`` (may be <0).
        Real frames ``[max(0,start), start+_L)`` come from ``sample``; the ``pad``
        missing front frames are zero-filled (hidden via ``ctx_pad`` downstream)."""
        stop = start + self._L
        out: dict[str, np.ndarray] = {}
        for k, v in sample.items():
            real = v[max(0, start) : stop]
            if pad > 0:
                front = np.zeros((pad, *v.shape[1:]), dtype=v.dtype)
                out[k] = np.concatenate([front, real], axis=0)
            else:
                out[k] = real
        return out


def collate_windows(batch: list[dict]) -> dict[str, np.ndarray]:
    """Stack a list of ``[seq]`` per-sample windows into ``[B, seq]`` columns."""
    keys = batch[0].keys()
    return {k: np.stack([s[k] for s in batch]) for k in keys}


def collate_train_batch(batch: list[dict], *, stats: dict[str, FeatureStats], L_ctx: int) -> TrainBatch:
    """Worker-side collate: stack → ``preprocess`` → split ``[ctx | chunk]``.

    The window the sampler yields is laid out ``[ctx | chunk]`` over
    ``seq = L_ctx + L_chunk`` frames. Context features are the first ``L_ctx``
    frames; the target action chunk is the remaining frames sliced off the
    stacked ego-action channels at ``[L_ctx :]``. Returns a fully-tensorized
    ``TrainBatch`` so the training loop does no reshaping — just ``.to(device)``.
    """
    stacked = collate_windows(batch)
    ctx_pad = torch.from_numpy(stacked["ctx_pad"].astype(np.int64))
    feats = preprocess(stacked, stats)
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
    sampler = WindowDataset(mds, L_ctx, L_chunk, seed=seed, windows_per_replay=windows_per_replay)
    collate = functools.partial(collate_train_batch, stats=stats, L_ctx=L_ctx)
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
    )
