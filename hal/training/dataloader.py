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


class WindowDataset(IterableDataset):
    """Wrap a StreamingDataset: pick a random ego port and a length
    ``L_ctx + L_chunk`` window from each replay, laid out as ``[ctx | chunk]``.
    Relabel p1/p2 → ego/opp before yielding.

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

    def __init__(self, mds: StreamingDataset, L_ctx: int, L_chunk: int, *, seed: int) -> None:
        self._mds = mds
        self.L_ctx = L_ctx
        self.L_chunk = L_chunk
        self._L = L_ctx + L_chunk
        self._seed = seed
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
            T = len(sample["frame"])
            # chunk[0] targets episode frame ``cs``; context is the L_ctx frames
            # before it. cs_min keeps >=1 real context frame (the cold-start
            # floor: inference always has the just-observed frame); cs_max keeps
            # the L_chunk-long chunk inside the episode.
            cs_min = 1
            cs_max = T - self.L_chunk
            if cs_max < cs_min:
                continue
            cs = int(rng.integers(cs_min, cs_max + 1))
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
    num_workers: int = 4,
    prefetch_factor: int = 4,
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

    A plain ``DataLoader`` rather than ``StreamingDataLoader``: the latter's
    mid-epoch resumption only engages when its dataset *is* a StreamingDataset,
    but here that dataset is wrapped by ``WindowDataset``, so the wrapper's only
    live behavior would be a per-batch ``len(batch[0])`` sample count — which a
    ``TrainBatch`` (not dict/Tensor) can't satisfy. StreamingDataset still owns
    sharding/shuffle; it's iterated inside the sampler."""
    mds = StreamingDataset(
        remote=f"{remote}/{split}" if remote else None,
        local=str(Path(data_root) / split),
        batch_size=1,
        shuffle=(split == "train"),
        cache_limit=cache_limit if remote else None,
    )
    sampler = WindowDataset(mds, L_ctx, L_chunk, seed=seed)
    collate = functools.partial(collate_train_batch, stats=stats, L_ctx=L_ctx)
    return DataLoader(
        sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
