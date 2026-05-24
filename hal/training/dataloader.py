"""MDS dataloader primitives shared across experiments.

Lives outside the experiment file so that forked / forkserver-spawned
DataLoader workers can re-import ``WindowSampler`` and ``collate_windows``
without re-running experiment-level module code. Keeping this file
side-effect-free is what makes that work.
"""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
from streaming import StreamingDataLoader
from streaming import StreamingDataset
from torch.utils.data import IterableDataset
from torch.utils.data import get_worker_info


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


class WindowSampler(IterableDataset):
    """Wrap a StreamingDataset: pick a random ego port and a random
    sub-trajectory of length ``L_ctx + n_lat + L_chunk`` from each replay.
    Relabel p1/p2 → ego/opp before yielding."""

    def __init__(self, mds: StreamingDataset, L_ctx: int, L_chunk: int, *, seed: int, n_lat: int = 0) -> None:
        self._mds = mds
        self.L_ctx = L_ctx
        self.L_chunk = L_chunk
        self.n_lat = n_lat
        self._L = L_ctx + n_lat + L_chunk
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
            if T < self._L:
                continue
            start = int(rng.integers(0, T - self._L + 1))
            window = {k: v[start : start + self._L] for k, v in sample.items()}
            ego_prefix = "p1" if rng.random() < 0.5 else "p2"
            yield relabel_ego(window, ego_prefix)


def collate_windows(batch: list[dict]) -> dict[str, np.ndarray]:
    keys = batch[0].keys()
    return {k: np.stack([s[k] for s in batch]) for k in keys}


def make_loader(
    data_root: str,
    split: str,
    *,
    L_ctx: int,
    L_chunk: int,
    batch_size: int,
    seed: int,
    n_lat: int = 0,
    num_workers: int = 4,
    prefetch_factor: int = 4,
) -> StreamingDataLoader:
    """Build the (StreamingDataset → WindowSampler → DataLoader) chain."""
    mds = StreamingDataset(local=str(Path(data_root) / split), batch_size=1, shuffle=(split == "train"))
    sampler = WindowSampler(mds, L_ctx, L_chunk, seed=seed, n_lat=n_lat)
    return StreamingDataLoader(
        sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_windows,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
