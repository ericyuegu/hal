# %%
"""Empirical action-space audit over local val replays (one-off, no GPU).

Questions it answers, per controller modality:
* sticks  — exact-neutral mass, dead-band leakage, 21-bin occupancy, distinct-value
  concentration, 37-cluster residuals (main vs c separately)
* triggers — {0, analog band, 1.0} mass split, bin occupancy
* buttons  — per-button rates, multipress, top-combo coverage (joint-class viability)
* temporal — button hold lengths + frame-to-frame action persistence (chunk/AR/FAST input)
* categoricals — observed id ranges vs CAT_FEATURES vocab sizes
"""

import collections

import numpy as np
from streaming import StreamingDataset
from streaming.base.util import clean_stale_shared_memory

ROOT = "data/processed/ranked-anonymized-1/mds/val"
N_REPLAYS = 150
DEADZONE_STICK = 23 / 80  # 0.2875
N_BINS = 21

clean_stale_shared_memory()
ds = StreamingDataset(local=ROOT, batch_size=1, shuffle=False)

cols_per_port = [
    "main_stick_x",
    "main_stick_y",
    "c_stick_x",
    "c_stick_y",
    "trigger_l",
    "trigger_r",
    "button_a",
    "button_b",
    "button_x",
    "button_y",
    "button_z",
    "button_r",
    "button_l",
    "button_d_up",
    "action",
    "stock",
    "jumps_used",
    "hurtbox_state",
    "airborne",
]
pool: dict[str, list[np.ndarray]] = {c: [] for c in cols_per_port}
runlens: dict[str, list[int]] = {b: [] for b in ("a", "b", "x", "y", "z", "r", "l", "d_up")}
persist: list[float] = []  # per-replay P(full 14-dim action unchanged frame->frame)

for i in range(min(N_REPLAYS, ds.num_samples)):
    s = ds[i]
    for port in ("p1", "p2"):
        for c in cols_per_port:
            pool[c].append(s[f"{port}_{c}"])
        # button hold run lengths
        for b in runlens:
            x = s[f"{port}_button_{b}"].astype(bool)
            if x.any():
                edges = np.flatnonzero(np.diff(np.concatenate([[0], x.view(np.int8), [0]])))
                runlens[b].extend((edges[1::2] - edges[0::2]).tolist())
        # frame-to-frame persistence of the full action vector
        a = np.stack([s[f"{port}_{c}"] for c in cols_per_port[:14]], axis=-1)
        persist.append(float((a[1:] == a[:-1]).all(-1).mean()))

P = {c: np.concatenate(v) for c, v in pool.items()}
n = len(P["main_stick_x"])
print(f"pooled {n:,} frames from {min(N_REPLAYS, ds.num_samples)} replays x 2 ports")


# %% sticks
def stick_report(name: str, x: np.ndarray, y: np.ndarray) -> None:
    neutral = (x == 0) & (y == 0)

    def inband(v: np.ndarray) -> np.ndarray:
        return (v != 0) & (np.abs(v) < DEADZONE_STICK)

    print(f"\n== {name} ==")
    print(
        f"  exact (0,0): {neutral.mean():.3f}   per-axis dead-band leak: x {inband(x).mean():.5f}  y {inband(y).mean():.5f}"
    )
    for ax, v in (("x", x), ("y", y)):
        idx = np.clip(((v + 1.0) / (2.0 / N_BINS)).astype(int), 0, N_BINS - 1)
        occ = np.bincount(idx, minlength=N_BINS) / n
        print(
            f"  {ax}: bins<0.1% mass: {(occ < 1e-3).sum()}/{N_BINS}   bins<1%: {(occ < 1e-2).sum()}/{N_BINS}   top bin {occ.max():.3f}"
        )
    pairs, counts = np.unique(np.stack([x, y], 1), axis=0, return_counts=True)
    order = np.argsort(-counts)
    cum = np.cumsum(counts[order]) / n
    k99, k999 = np.searchsorted(cum, 0.99) + 1, np.searchsorted(cum, 0.999) + 1
    print(f"  distinct (x,y): {len(pairs):,}   top-32 mass {cum[31]:.3f}   pairs for 99%: {k99}   99.9%: {k999}")
    nz = ~neutral
    mag = np.hypot(x[nz], y[nz])
    print(
        f"  non-neutral magnitude: p25 {np.percentile(mag, 25):.2f}  p50 {np.percentile(mag, 50):.2f}  p75 {np.percentile(mag, 75):.2f}  frac>=0.95: {(mag >= 0.95).mean():.3f}"
    )


stick_report("main_stick", P["main_stick_x"], P["main_stick_y"])
stick_report("c_stick", P["c_stick_x"], P["c_stick_y"])

# %% cluster residuals (the 37 hand-tuned centers, applied to both sticks)
import torch

from hal.training import scoring

centers = scoring.STICK_CLUSTER_CENTERS_MAIN.numpy()
for name in ("main_stick", "c_stick"):
    xy = np.stack([P[f"{name}_x"], P[f"{name}_y"]], 1).astype(np.float32)
    sub = xy[np.random.default_rng(0).choice(len(xy), 500_000, replace=False)]
    d2 = ((sub[:, None, :] - centers[None]) ** 2).sum(-1)
    nearest = centers[d2.argmin(1)]
    err = np.abs(nearest - sub)
    far = (err.max(1) > 0.1).mean()
    print(f"{name}: 37-cluster MAE {err.mean():.4f}   frac>0.1 off {far:.4f}")
    # how much mass per cluster (occupancy)
    occ = np.bincount(d2.argmin(1), minlength=len(centers)) / len(sub)
    print(f"   clusters <0.1% mass: {(occ < 1e-3).sum()}/{len(centers)}   top5 {np.sort(occ)[-5:][::-1].round(3)}")

# %% triggers
for t in ("trigger_l", "trigger_r"):
    v = P[t]
    zero, full = (v == 0).mean(), (v == 1.0).mean()
    analog = (v > 0) & (v < 1.0)
    print(f"{t}: zero {zero:.4f}  full(1.0) {full:.4f}  analog-band {analog.mean():.4f}", end="")
    if analog.any():
        av = v[analog]
        print(
            f"   analog values: p10 {np.percentile(av, 10):.3f} p50 {np.percentile(av, 50):.3f} p90 {np.percentile(av, 90):.3f}  distinct {len(np.unique(av))}"
        )
    else:
        print()
idx = np.clip((P["trigger_l"] / (1.0 / N_BINS)).astype(int), 0, N_BINS - 1)
occ = np.bincount(idx, minlength=N_BINS) / n
print(
    f"trigger_l 21-bin occupancy: bins<0.1% {(occ < 1e-3).sum()}/{N_BINS}  (deadzone bins 1..6 mass {occ[1:7].sum():.5f})"
)

# %% buttons
B_ORDER = ("a", "b", "x", "y", "z", "r", "l", "d_up")
btn = np.stack([P[f"button_{b}"] for b in B_ORDER], 1).astype(bool)
print("press rates:", {b: round(float(btn[:, i].mean()), 4) for i, b in enumerate(B_ORDER)})
npress = btn.sum(1)
print(
    f"any-press {float((npress >= 1).mean()):.3f}   multipress(>=2) {float((npress >= 2).mean()):.4f}   (>=3) {float((npress >= 3).mean()):.5f}"
)
combo = collections.Counter(map(tuple, btn.astype(np.uint8)))
top = combo.most_common()
cum = np.cumsum([c for _, c in top]) / n
print(
    f"distinct combos: {len(combo)}   combos for 99%: {int(np.searchsorted(cum, 0.99) + 1)}   99.9%: {int(np.searchsorted(cum, 0.999) + 1)}   99.99%: {int(np.searchsorted(cum, 0.9999) + 1)}"
)


def names(t: tuple) -> str:
    return "+".join(b for b, on in zip(B_ORDER, t) if on) or "none"


print("top 15 combos:", [(names(t), round(c / n, 4)) for t, c in top[:15]])
# digital L/R click vs analog: click => analog == 1?
for b, t in (("l", "trigger_l"), ("r", "trigger_r")):
    click = P[f"button_{b}"].astype(bool)
    if click.any():
        print(
            f"button_{b}: P(trigger==1 | click) {float((P[t][click] == 1.0).mean()):.4f}   P(click) {click.mean():.4f}"
        )

# %% temporal structure
print("\nbutton hold lengths (frames):")
for b in B_ORDER:
    r = np.array(runlens[b])
    if len(r):
        print(
            f"  {b}: n {len(r):,}  p25 {np.percentile(r, 25):.0f}  p50 {np.percentile(r, 50):.0f}  p75 {np.percentile(r, 75):.0f}  mean {r.mean():.1f}"
        )
print(f"\nfull 14-dim action persistence frame->frame: mean {np.mean(persist):.3f}")

# %% categorical id ranges vs CAT_FEATURES vocabs
from hal.training.features import PLAYER_CAT_FEATURES

for c in ("action", "stock", "jumps_used", "hurtbox_state", "airborne"):
    v = P[c]
    vocab, dim = PLAYER_CAT_FEATURES[c]
    u = np.unique(v)
    print(f"{c}: vocab={vocab} dim={dim}   observed [{u.min()}, {u.max()}]  distinct {len(u)}")

# %% chord-vocab viability: quantize (main, c, trig, buttons) jointly per frame
main_q = d2 = None  # free memory from above if re-running cells
TRIG_CENTERS = np.array([0.0, 0.35, 0.6, 0.85, 1.0], dtype=np.float32)
C9 = np.array(
    [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [0.7, 0.7], [-0.7, 0.7], [0.7, -0.7], [-0.7, -0.7]], dtype=np.float32
)


def q_stick(x, y, centers):
    xy = np.stack([x, y], 1).astype(np.float32)
    out = np.empty(len(xy), dtype=np.int32)
    for i in range(0, len(xy), 1_000_000):
        blk = xy[i : i + 1_000_000]
        out[i : i + 1_000_000] = ((blk[:, None, :] - centers[None]) ** 2).sum(-1).argmin(1)
    return out


mq = q_stick(P["main_stick_x"], P["main_stick_y"], centers)  # 37
cq = q_stick(P["c_stick_x"], P["c_stick_y"], C9)  # 9
tlq = np.abs(P["trigger_l"][:, None] - TRIG_CENTERS[None]).argmin(1)  # 5
trq = np.abs(P["trigger_r"][:, None] - TRIG_CENTERS[None]).argmin(1)  # 5
bq = (btn.astype(np.uint32) * (1 << np.arange(8, dtype=np.uint32))).sum(1)

chord = (((mq.astype(np.int64) * 9 + cq) * 5 + tlq) * 5 + trq) * 256 + bq
u, cts = np.unique(chord, return_counts=True)
order = np.argsort(-cts)
cum = np.cumsum(cts[order]) / len(chord)
print(
    f"chord vocab: distinct {len(u):,}   for 99%: {int(np.searchsorted(cum, 0.99) + 1)}   99.9%: {int(np.searchsorted(cum, 0.999) + 1)}   99.99%: {int(np.searchsorted(cum, 0.9999) + 1)}"
)

# changes per 16-frame chunk under this quantization (event-coding density)
ch = chord[1:] != chord[:-1]
print(f"quantized chord change rate frame->frame: {ch.mean():.3f}  (~{ch.mean() * 15:.1f} changes per 16-frame chunk)")
