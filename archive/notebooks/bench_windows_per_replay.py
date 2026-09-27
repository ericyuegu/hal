"""Throughput vs windows_per_replay (K) on the local ranked-anonymized-1 shards.

Each replay is deserialized off disk once and yields K non-overlapping windows, so
disk reads per window drop ~1/K. This measures the CPU-side amortization (read +
slice per replay spread over K windows) and where windows/s plateaus, to pick the
smallest K that keeps the throughput win. Run: `uv run notebooks/bench_windows_per_replay.py`.
"""

# %%
import time

import numpy as np
from streaming import StreamingDataset

from hal.training.dataloader import WindowDataset
from hal.training.dataloader import _choose_chunk_starts

DATA = "data/processed/ranked-anonymized-1/mds/train"
L_CTX, L_CHUNK = 256, 13  # 012: L_ctx=256, L_chunk=max(head_offsets)=13
N_REPLAYS = 500  # fixed replay budget (within the first local shard) → same decompress work across K


class _Capped:
    """First ``n`` replays of an MDS, in shard order (shuffle=False keeps the
    set identical across K and avoids random access to not-yet-local shards)."""

    def __init__(self, mds: StreamingDataset, n: int) -> None:
        self._mds, self._n = mds, n

    def __iter__(self):
        for i, s in enumerate(self._mds):
            if i >= self._n:
                return
            yield s


def _mds() -> StreamingDataset:
    return StreamingDataset(remote=None, local=DATA, batch_size=1, shuffle=False)


# %% validate non-overlap on a real replay (real T, not the synthetic unit-test T)
sample = next(iter(_mds()))
T = len(sample["frame"])
cs = _choose_chunk_starts(T, L_CTX, L_CHUNK, K=8, rng=np.random.default_rng(0))
spans = [(c - L_CTX, c + L_CHUNK) for c in sorted(int(x) for x in cs)]
assert all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1)), spans
print(f"real replay T={T}: {len(cs)} windows, non-overlapping spans OK -> {spans}")

# %% warm up the decompressed-shard cache so timed runs measure steady state
for _ in WindowDataset(_Capped(_mds(), N_REPLAYS), L_CTX, L_CHUNK, seed=0, windows_per_replay=1):
    pass

# %% timed sweep
print(f"\n{'K':>4} {'windows':>9} {'wall_s':>8} {'win/s':>9} {'us/replay':>10} {'reads/win':>10}")
for K in (1, 4, 16, 64):
    ds = WindowDataset(_Capped(_mds(), N_REPLAYS), L_CTX, L_CHUNK, seed=0, windows_per_replay=K)
    t0 = time.perf_counter()
    n = sum(1 for _ in ds)
    dt = time.perf_counter() - t0
    print(f"{K:>4} {n:>9} {dt:>8.2f} {n / dt:>9.0f} {dt / N_REPLAYS * 1e6:>10.1f} {1 / K:>10.3f}")
