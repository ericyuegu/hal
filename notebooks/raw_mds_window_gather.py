"""Prototype: gather windows straight from a raw MDS shard by byte offset."""

import json
import mmap
import time
from pathlib import Path

import numpy as np
from streaming.base.format.mds import MDSReader

from hal.data.policy_world_schema import decode_policy_world_replay_slices

D = "data/processed/ranked-anonymized-1/mds-policy-world-v8/train"
info = json.loads(Path(f"{D}/index.json").read_text())["shards"][0]
names, encs, sizes = info["column_names"], info["column_encodings"], info["column_sizes"]
with Path(f"{D}/{info['raw_data']['basename']}").open("rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
buf = np.frombuffer(mm, dtype=np.uint8)
n = int(buf[:4].view("<u4")[0])
offsets = buf[4 : 4 + 4 * (n + 1)].view("<u4").astype(np.int64)
var = [i for i, s in enumerate(sizes) if s is None]


def layout(row):
    """Per-column (start, stop) byte ranges inside one sample."""
    base = offsets[row]
    nvar = len(var)
    var_sizes = buf[base : base + 4 * nvar].view("<u4")
    pos = base + 4 * nvar
    out, vi = {}, 0
    for i, name in enumerate(names):
        size = sizes[i] if sizes[i] is not None else int(var_sizes[vi])
        vi += sizes[i] is None
        out[name] = (pos, pos + size)
        pos += size
    assert pos == offsets[row + 1]
    return out


reader = MDSReader.from_json(D, None, info)
lay = [layout(r) for r in range(n)]
frames = []
for r in range(n):
    a, b = lay[r]["num_frames"]
    frames.append(int(reader.get_item(r)["num_frames"]))  # scalar decode via reader once
DT = {e: np.dtype(e.split(":")[1]) for e in encs if e.startswith("ndarray")}


def window(r, start, L):
    out = {}
    nf = frames[r]
    for name, enc in zip(names, encs, strict=True):
        a, b = lay[r][name]
        if enc.startswith("ndarray"):
            dt = DT[enc]
            data_start = b - nf * dt.itemsize  # strip the NDArray header
            if data_start < a:  # absent Nana columns hold one placeholder value
                out[name] = buf[b - dt.itemsize : b].view(dt)
                continue
            out[name] = buf[data_start + start * dt.itemsize : data_start + (start + L) * dt.itemsize].view(dt)
    return out


# Parity against the reader.
L = 280
rng = np.random.default_rng(0)
s = reader.get_item(7)
w = window(7, 1000, L)
for k, v in w.items():
    assert np.array_equal(
        v, np.asarray(s[k])[1000 : 1000 + L] if np.asarray(s[k]).size > 1 else np.asarray(s[k]), equal_nan=True
    ), k
print("parity ok;", len(w), "per-frame columns")

# Throughput: one 256-window batch of raw views + world decode.
rows = rng.integers(0, n, 256)
starts = [int(rng.integers(0, frames[r] - L)) for r in rows]
scalars = {r: reader.get_item(int(r)) for r in set(rows.tolist())}
t = time.perf_counter()
for _ in range(10):
    for r, st in zip(rows, starts, strict=True):
        w = window(int(r), st, L)
        src = {
            k: scalars[int(r)][k]
            for k in (
                "num_frames",
                "p1_rank",
                "p2_rank",
                "p1_character",
                "p2_character",
                "stage",
                "p1_nana_present",
                "p2_nana_present",
                "replay_id",
                "policy_schema_version",
                "policy_world_schema_version",
                "source_schema_version",
            )
        }
        src.update(w)
        src["num_frames"] = L
        decode_policy_world_replay_slices(src, ((0, L),))
dt = (time.perf_counter() - t) / 10
print(f"gather+decode 256 windows: {dt * 1e3:.1f} ms/batch")
pages = [
    len(
        {
            p // 4096
            for name, enc in zip(names, encs, strict=True)
            if enc.startswith("ndarray")
            for p in range(
                *(
                    lambda a, b, nf=frames[r], it=DT[enc].itemsize: (
                        max(a, b - nf * it + st * it),
                        max(a + 1, b - nf * it + (st + L) * it),
                    )
                )(*lay[r][name]),
                4096,
            )
        }
    )
    for r, st in zip(rows, starts, strict=True)
]
print(
    f"distinct 4KiB pages per window: mean {np.mean(pages):.0f}; bytes per window {sum(v.nbytes for v in w.values())}"
)
