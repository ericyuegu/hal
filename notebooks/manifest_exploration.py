"""Interactive exploration of `hal.data.manifest`.

Open in VSCode and run cell-by-cell with the Python interactive window.
Each `# %%` block is a runnable cell.
"""

# %%
import json
import time
from collections import Counter
from pathlib import Path

from hal.data.manifest import extract_index_entry
from hal.data.manifest import read_jsonl
from hal.data.manifest import write_jsonl

REPLAY_DIR = Path("/home/ericgu/data/ssbm/mang0")
SAMPLE_REPLAY = REPLAY_DIR / "Game_20201215T114034.slp"


# %% [1] Pretty-print a single replay's metadata
e = extract_index_entry(SAMPLE_REPLAY)
print(json.dumps(e.to_dict(), indent=2))


# %% [2] Iterate a directory, dump rows to JSONL
slps = list(REPLAY_DIR.glob("*.slp"))[:50]
entries = [e for p in slps if (e := extract_index_entry(p)) is not None]
out = Path("/tmp/idx_50.jsonl")
write_jsonl(out, entries)
print(f"wrote {len(entries)}/{len(slps)} entries -> {out} ({out.stat().st_size} bytes)")
print("first row:")
print(json.dumps(entries[0].to_dict(), indent=2))


# %% [3] Throughput on 1000 replays (single-threaded; build_index will parallelize)
slps = sorted(REPLAY_DIR.glob("*.slp"))[:1000]
t0 = time.perf_counter()
ok = sum(1 for p in slps if extract_index_entry(p, compute_sha1=False) is not None)
dt = time.perf_counter() - t0
print(f"{ok}/{len(slps)} parsed in {dt:.2f}s ({len(slps) / dt:.0f} slp/s)")


# %% [4] Cost of compute_sha1=True vs False
slps = sorted(REPLAY_DIR.glob("*.slp"))[:500]
for sha in (False, True):
    t0 = time.perf_counter()
    n = sum(1 for p in slps if extract_index_entry(p, compute_sha1=sha))
    print(f"compute_sha1={sha}: {n} parsed in {time.perf_counter() - t0:.2f}s")


# %% [5] JSONL round-trip equality
slps = list(REPLAY_DIR.glob("*.slp"))[:20]
entries = [e for p in slps if (e := extract_index_entry(p))]
rt_path = Path("/tmp/rt.jsonl")
write_jsonl(rt_path, entries)
back = list(read_jsonl(rt_path))
equal = [e.to_dict() for e in entries] == [e.to_dict() for e in back]
print(f"orig={len(entries)} read={len(back)} equal={equal}")


# %% [6] Group-by preview — what Stage 2 (filter_replays) will be slicing on
slps = sorted(REPLAY_DIR.glob("*.slp"))[:500]
entries = [e for p in slps if (e := extract_index_entry(p, compute_sha1=False))]
print("total parsed:", len(entries))
print("completed:", Counter(e.outcome.completed if e.outcome else None for e in entries))
print("end_method:", Counter(e.outcome.end_method.name if e.outcome else None for e in entries))
print("slp_version:", Counter(e.slp_version for e in entries).most_common(5))

print("\nframe_count buckets:")
buckets = [0, 600, 1500, 3600, 9000, 99999]
for lo, hi in zip(buckets, buckets[1:], strict=False):
    n = sum(1 for e in entries if lo <= e.frame_count < hi)
    print(f"  [{lo:5d}, {hi:5d}): {n}")

print("\ntop player codes:", Counter(p.code for e in entries for p in e.players).most_common(5))
print("top characters:", Counter(p.character for e in entries for p in e.players).most_common(5))


# %% [7] Failure path — non-slp / missing / empty
print("non-slp file:", extract_index_entry(Path("/etc/hostname")))
print("missing file:", extract_index_entry(Path("/tmp/does_not_exist.slp")))
print("empty file:  ", extract_index_entry(Path("/dev/null")))


# %% [8] `player_for_port` lookup + outcome inspection
e = extract_index_entry(SAMPLE_REPLAY)
print("outcome:", e.outcome)
if e.outcome:
    print("  completed?", e.outcome.completed)
    print("  end_method:", e.outcome.end_method.name)
    print("  lras_initiator:", e.outcome.lras_initiator)
print("player at port 1:", e.player_for_port(1))
print("player at port 3 (none):", e.player_for_port(3))
print("rank from filename:", e.rank_filename)
print("annotation (None until Stage 3 writes):", e.annotation)
