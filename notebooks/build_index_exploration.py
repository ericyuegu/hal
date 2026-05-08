"""Exercise `hal.data.build_index` end-to-end with different flag combos.

Run cell-by-cell in VSCode's Python interactive window. Each `# %%` block is a
runnable cell. Cells share kernel state, so [0] (env + imports) must run first.
"""

# %% [0] env + imports
import json
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path

from hal.data.build_index import build_index
from hal.data.manifest import read_jsonl

REPLAY_DIR = Path("/home/ericgu/data/ssbm/mang0")
WORK = Path("/tmp/build_index_play")
WORK.mkdir(exist_ok=True)


def fresh(name: str) -> Path:
    p = WORK / name
    p.unlink(missing_ok=True)
    return p


def stage_subdir(n: int, name: str = "small") -> Path:
    d = WORK / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir()
    for src in sorted(REPLAY_DIR.glob("*.slp"))[:n]:
        (d / src.name).symlink_to(src)
    return d


# %% [1] basic run on 50 slps, default flags (sha1 on, workers=ncpu-1)
root = stage_subdir(50)
out = fresh("idx_50.jsonl")
t0 = time.perf_counter()
build_index(root=root, output=out)
print(f"{time.perf_counter() - t0:.2f}s -> {out} ({out.stat().st_size:,} bytes)")
print("first row:")
print(json.dumps(next(read_jsonl(out)).to_dict(), indent=2))


# %% [2] --no-sha1 speedup (skip 4KB read per file)
root = stage_subdir(500, "med")
for compute_sha1 in (True, False):
    out = fresh(f"idx_500_sha1={compute_sha1}.jsonl")
    t0 = time.perf_counter()
    build_index(root=root, output=out, compute_sha1=compute_sha1, workers=8)
    print(f"compute_sha1={compute_sha1}: {time.perf_counter() - t0:.2f}s")


# %% [3] worker scaling — same 500 slps, vary pool size
root = stage_subdir(500, "med")
for w in (1, 2, 4, 8, 16):
    out = fresh(f"idx_w{w}.jsonl")
    t0 = time.perf_counter()
    build_index(root=root, output=out, workers=w, compute_sha1=False)
    print(f"workers={w:>2}: {time.perf_counter() - t0:.2f}s ({500 / (time.perf_counter() - t0):.0f} slp/s)")


# %% [4] --incremental no-op: rerun on same root, expect 0 new
root = stage_subdir(50)
out = fresh("idx_inc.jsonl")
build_index(root=root, output=out, workers=4)
print(f"after first run: {sum(1 for _ in read_jsonl(out))} rows")
build_index(root=root, output=out, incremental=True, workers=4)
print(f"after no-op rerun: {sum(1 for _ in read_jsonl(out))} rows")


# %% [5] --incremental resume: add 10 more files, only those get indexed
root = stage_subdir(50)
out = fresh("idx_resume.jsonl")
build_index(root=root, output=out, workers=4)
print(f"step 1: {sum(1 for _ in read_jsonl(out))} rows")
# add 10 more
for src in sorted(REPLAY_DIR.glob("*.slp"))[50:60]:
    (root / src.name).symlink_to(src)
build_index(root=root, output=out, incremental=True, workers=4)
print(f"step 2: {sum(1 for _ in read_jsonl(out))} rows (expect 60)")


# %% [6] refusal without --incremental: existing output not clobbered
root = stage_subdir(50)
out = fresh("idx_refuse.jsonl")
build_index(root=root, output=out, workers=4)
try:
    build_index(root=root, output=out, workers=4)
except FileExistsError as e:
    print(f"correctly refused: {e}")


# %% [7] CLI surface — same as cell [1] but via subprocess (sanity check argparse)
root = stage_subdir(20)
out = fresh("idx_cli.jsonl")
result = subprocess.run(
    [
        "uv",
        "run",
        "python",
        "-m",
        "hal.data.build_index",
        "--root",
        str(root),
        "--output",
        str(out),
        "--workers",
        "4",
        "--no-sha1",
    ],
    cwd="/home/ericgu/src/hal",
    capture_output=True,
    text=True,
)
print("stdout:", result.stdout)
print("stderr:", result.stderr[-400:])
print("rows:", sum(1 for _ in read_jsonl(out)))


# %% [8] failure rate on a larger sample — what % of slps does peppi reject?
root = stage_subdir(2000, "big")
out = fresh("idx_big.jsonl")
t0 = time.perf_counter()
build_index(root=root, output=out, compute_sha1=False, workers=12)
dt = time.perf_counter() - t0
total = len(list(root.glob("*.slp")))
ok = sum(1 for _ in read_jsonl(out))
print(f"{ok}/{total} parsed ({(total - ok) / total * 100:.2f}% failure) in {dt:.2f}s ({total / dt:.0f} slp/s)")


# %% [9] index summary stats — what does the corpus look like?
out = WORK / "idx_big.jsonl"
entries = list(read_jsonl(out))
print(f"total: {len(entries)}")
print(f"completed: {sum(1 for e in entries if e.outcome and e.outcome.completed)}")
print(f"end_method: {Counter(e.outcome.end_method.name if e.outcome else None for e in entries)}")
print(f"slp_version top-5: {Counter(e.slp_version for e in entries).most_common(5)}")
print(f"played_on: {Counter(e.played_on for e in entries)}")
print(f"top characters: {Counter(p.character for e in entries for p in e.players).most_common(5)}")

print("\nframe_count buckets:")
for lo, hi in [(0, 600), (600, 1500), (1500, 3600), (3600, 9000), (9000, 999999)]:
    n = sum(1 for e in entries if lo <= e.frame_count < hi)
    print(f"  [{lo:5d}, {hi:5d}): {n}")


# %% [10] full mang0 corpus — uncomment to run; expect minutes, not seconds
# out = fresh("idx_mang0_full.jsonl")
# build_index(root=REPLAY_DIR, output=out, compute_sha1=True, workers=12)
# print(f"rows: {sum(1 for _ in read_jsonl(out))}")
