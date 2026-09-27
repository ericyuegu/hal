# %%
"""Offline audit of learned action-chunk tokenizations, before a GPU run picks one.

The action stream is 14 channels/frame (4 stick axes on the 1/80 grid, 2 triggers on the
1/140 grid, 8 button bits) and is overwhelmingly piecewise-constant: buttons hold ~93% of
frames, the main stick is 34% exact-neutral with most of the rest rim-saturated. That is a
square wave, not a smooth trajectory, so the prior is that run-length codes win and frequency
codes ring at the edges. This notebook measures that prior instead of assuming it.

Three studies, one shared reconstruction metric:

(a) BASELINE — the per-frame discretizers in ``hal/training/scoring.py`` (256 button combos /
    65 main-stick clusters / 9 c-stick clusters / 5x5 joint triggers). Its reconstruction
    error is the quantization floor every other method is measured against, and its 4 group
    ids/frame are the token-rate reference: **compression = 4 / (tokens per frame)**, so the
    "clean reconstruction at >=4x compression" bar means <=1 token/frame at the floor.

(b) LEARNED RLE via BPE — each frame becomes one joint symbol (the 4-tuple of group ids) and
    a byte-pair encoder is trained over those symbol streams to vocab {1k, 4k, 16k}. Lossless
    by construction, so the metrics are compression, held-out generalization (tokens/frame on
    windows the merge table never saw, plus OOV rate and vocab utilization) and whether the
    learned merges are in fact run-length holds. A second symbolization (the 4 group ids
    emitted separately, base alphabet 355) covers the small-vocab regime the joint symbol
    cannot reach.

(c) FAST-style DCT — per-chunk DCT of the 6 analog channels, scalar-quantized coefficients,
    swept over chunk length H and quantization scale. Reported against the (a) floor, with a
    dedicated breakdown at TRANSITION frames (frames adjacent to an analog value change),
    since ringing concentrates at edges and edges are where Melee technique lives.

No GPU, no network. Reads the local dev MDS bundle via the standard streaming path, fitting on
one set of replays and reporting on a disjoint set. The corpus is ~100 replays, which is ample
for the reconstruction study but bounds the BPE study: merge training saturates well before a
16k vocab, so the large-vocab rows are undertrained rather than representative (see the closing
recommendation). Run end-to-end with ``uv run notebooks/tokenizer_audit.py``.
"""

# %%
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.fft import dct
from scipy.fft import idct
from streaming import StreamingDataset
from streaming.base.util import clean_stale_shared_memory
from torch import Tensor

from hal.training.dataloader import _choose_chunk_starts
from hal.training.features import ACTION_CHANNELS
from hal.training.scoring import N_BUTTON_COMBOS
from hal.training.scoring import STICK_CLUSTER_CENTERS_C
from hal.training.scoring import STICK_CLUSTER_CENTERS_MAIN
from hal.training.scoring import TRIGGER_CENTERS
from hal.training.scoring import buttons_to_combo
from hal.training.scoring import center_to_value
from hal.training.scoring import cluster_to_xy
from hal.training.scoring import combo_to_buttons
from hal.training.scoring import nearest_center
from hal.training.scoring import nearest_cluster

# %%
# The dev bundle, not ranked-anonymized-1: it is small, entirely local, and nothing else on the
# box streams it. The production split doubles as a live streaming cache for training runs
# (shards evicted and re-downloaded under ``cache_limit``), so reading it both contends with
# those runs and gives a corpus that changes between invocations.
MDS_DIR = Path("data/processed/dev/mds")
SPLIT = "train"  # the only populated dev split (101 replays; test has 4, val has 0)
SCRATCH = Path(os.environ.get("TMPDIR", "/tmp")) / "hal-tokenizer-audit"

# Fit and held-out are disjoint *replays*, which is the split that matters here: a merge table
# must generalize to controller streams from players it never saw, not merely to unseen windows
# of the same match.
N_FIT_REPLAYS = 70
L_WINDOW = 192  # divisible by every H below, so no chunk-remainder special case
WINDOWS_PER_REPLAY = 12
SEED = 0

BPE_VOCABS = (1024, 4096, 16384)
DCT_H = (16, 24)
DCT_SCALES = (0.05, 0.15, 0.40)
DCT_BPE_VOCAB = 2048  # coefficient alphabet is small; merges saturate well below this
BTN_BPE_VOCAB = 1024  # the button-only stream a DCT analog code still has to carry

BASELINE_TOKENS_PER_FRAME = 4.0  # the 4 group ids a factorized per-frame head emits

# %%
# --- data: a symlink mirror is the only safe way to point streaming at a split --------------
# ``StreamingDataset(remote=None)`` on a split whose index names a shard file it cannot find
# falls into its failed-download cleanup and EVICTS the shard — i.e. deletes the local data. A
# split materialized to bare ``.mds`` (its index still names the ``.zstd``) trips that on the
# first read. Mirroring the shards as symlinks under an index that matches what is actually on
# disk keeps the read on the standard streaming path, and any eviction unlinks a symlink
# instead of the dataset. ``remote`` stays None throughout, so nothing can be downloaded either.


def shard_mirror(split: str) -> Path:
    """Scratch dir of symlinks to ``split``'s shards plus an index that matches them: shards
    left as bare ``.mds`` are declared uncompressed, shards that kept their ``.mds.zstd`` are
    declared as-is (streaming decompresses into the mirror). Downloads nothing, and never
    opens the real split for writing.

    Only the shards actually on disk are mirrored, and the count is printed: a split that is
    also some other run's live streaming cache loses and regains shards under LRU eviction, so
    two runs would draw from different corpora. The dev bundle is nobody's cache, so the count
    should always be complete — a short count means something else is streaming this split."""
    mirror = SCRATCH / f"mds-{split}"
    if mirror.exists():
        shutil.rmtree(mirror)
    mirror.mkdir(parents=True)
    src = MDS_DIR / split
    index = json.loads((src / "index.json").read_text())
    shards = []
    for shard in index["shards"]:
        raw = src / shard["raw_data"]["basename"]
        zipped = src / shard["zip_data"]["basename"] if shard.get("zip_data") else None
        if raw.exists():
            shard["compression"] = None
            shard["zip_data"] = None
            present = raw
        elif zipped is not None and zipped.exists():
            present = zipped
        else:
            continue
        # Absolute: a relative link would resolve from the mirror dir, not the cwd.
        (mirror / present.name).symlink_to(present.resolve())
        shards.append(shard)
    assert shards, f"{split}: no shard files under {src} — a streaming eviction may have deleted them"
    print(f"  {split}: mirrored {len(shards)}/{len(index['shards'])} shards present on disk")
    index["shards"] = shards
    (mirror / "index.json").write_text(json.dumps(index))
    return mirror


def sample_actions(mds: StreamingDataset, replays: range, seed: int) -> np.ndarray:
    """``[n_windows, L_WINDOW, 14]`` ego action stream over ``replays``, placed exactly as
    training places its windows: ``_choose_chunk_starts`` picks up to ``WINDOWS_PER_REPLAY``
    non-overlapping windows per replay, with a random ego port. ``L_ctx=0`` — this audit needs
    the action channels only.

    Reads the controller columns straight off the row rather than going through
    ``WindowDataset``, whose ``check_schema_version`` pins the row to the code's current MDS
    schema. Schema bumps that widen the post-frame/item block leave the controller block —
    all this notebook reads — untouched, so pinning would reject a dataset the audit can read
    perfectly well. The columns that do matter are asserted present per row instead, and the
    materialization actually audited is printed rather than assumed."""
    rng = np.random.default_rng(seed)
    windows: list[np.ndarray] = []
    for i in replays:
        sample = mds[i]
        if not windows:
            print(f"  replays {replays.start}..{replays.stop}: MDS schema_version={sample.get('schema_version')}")
        port = "p1" if rng.random() < 0.5 else "p2"
        missing = [ch for ch in ACTION_CHANNELS if f"{port}_{ch}" not in sample]
        assert not missing, f"replay {i}: row is missing action columns {missing}"
        stream = np.stack([sample[f"{port}_{ch}"] for ch in ACTION_CHANNELS], axis=-1).astype(np.float32)
        for start in _choose_chunk_starts(len(stream), 0, L_WINDOW, WINDOWS_PER_REPLAY, rng):
            windows.append(stream[int(start) : int(start) + L_WINDOW])
    assert windows, f"replays {replays} yielded no windows"
    return np.stack(windows)


clean_stale_shared_memory()
mds = StreamingDataset(local=str(shard_mirror(SPLIT)), remote=None, shuffle=False, batch_size=1, keep_zip=True)
assert len(mds) > N_FIT_REPLAYS, f"{SPLIT}: {len(mds)} replays, need more than N_FIT_REPLAYS={N_FIT_REPLAYS}"
fit = sample_actions(mds, range(0, N_FIT_REPLAYS), SEED)
held = sample_actions(mds, range(N_FIT_REPLAYS, len(mds)), SEED + 1)
assert not np.isnan(fit).any() and not np.isnan(held).any()
print(f"fit  {fit.shape} = {fit.shape[0] * fit.shape[1]:,} frames (replays 0..{N_FIT_REPLAYS})")
print(f"held {held.shape} = {held.shape[0] * held.shape[1]:,} frames (replays {N_FIT_REPLAYS}..{len(mds)})")

# %%
# --- the baseline codec, straight off hal/training/scoring.py --------------------------
N_TRIG = len(TRIGGER_CENTERS)
GROUP_NAMES = ("buttons", "main_stick", "c_stick", "triggers")
GROUP_SIZES = (
    N_BUTTON_COMBOS,
    len(STICK_CLUSTER_CENTERS_MAIN),
    len(STICK_CLUSTER_CENTERS_C),
    N_TRIG * N_TRIG,
)
GROUP_OFFSETS = np.concatenate([[0], np.cumsum(GROUP_SIZES[:-1])])
N_GROUP_SYMBOLS = int(sum(GROUP_SIZES))


def quantize(a: Tensor) -> Tensor:
    """``[..., 14]`` action vectors → ``[..., 4]`` group ids in GROUP_NAMES order. The two
    shoulders share one joint 5x5 trigger id (they are decoded together at inference)."""
    trig_l = nearest_center(a[..., 4], TRIGGER_CENTERS)
    trig_r = nearest_center(a[..., 5], TRIGGER_CENTERS)
    return torch.stack(
        [
            buttons_to_combo(a[..., 6:]),
            nearest_cluster(a[..., 0:2], STICK_CLUSTER_CENTERS_MAIN),
            nearest_cluster(a[..., 2:4], STICK_CLUSTER_CENTERS_C),
            trig_l * N_TRIG + trig_r,
        ],
        dim=-1,
    )


def dequantize(g: Tensor) -> Tensor:
    """Inverse of ``quantize``: ``[..., 4]`` group ids → ``[..., 14]`` action vectors."""
    triggers = torch.stack(
        [
            center_to_value(g[..., 3] // N_TRIG, TRIGGER_CENTERS),
            center_to_value(g[..., 3] % N_TRIG, TRIGGER_CENTERS),
        ],
        dim=-1,
    )
    return torch.cat(
        [
            cluster_to_xy(g[..., 1], STICK_CLUSTER_CENTERS_MAIN),
            cluster_to_xy(g[..., 2], STICK_CLUSTER_CENTERS_C),
            triggers,
            combo_to_buttons(g[..., 0]),
        ],
        dim=-1,
    )


fit_groups = quantize(torch.from_numpy(fit)).numpy()
held_groups = quantize(torch.from_numpy(held)).numpy()
held_baseline = dequantize(torch.from_numpy(held_groups)).numpy()
print(f"groups {GROUP_NAMES} sizes {GROUP_SIZES} → {N_GROUP_SYMBOLS} group symbols total")


# %%
# --- shared reconstruction metric ------------------------------------------------------
def transition_frames(a: np.ndarray) -> np.ndarray:
    """``[N, L, 14]`` → ``[N, L]`` bool, True at frames adjacent to an analog value change
    (either endpoint of a frame pair whose stick/trigger values differ). These are the edges
    of the square wave — where a frequency-domain code rings and a hold-based code does not."""
    changed = (a[:, 1:, :6] != a[:, :-1, :6]).any(axis=-1)
    mask = np.zeros(a.shape[:2], dtype=bool)
    mask[:, :-1] |= changed
    mask[:, 1:] |= changed
    return mask


HELD_TRANSITION = transition_frames(held)
print(f"held transition frames: {HELD_TRANSITION.mean():.1%} of frames")


@dataclass(frozen=True, slots=True)
class Recon:
    """One row of the summary table: what a method costs and what it destroys."""

    method: str
    setting: str
    tokens_per_frame: float
    stick_mae: float
    stick_p99: float
    trig_mae: float
    trig_p99: float
    transition_p99: float
    button_exact: float

    @property
    def compression(self) -> float:
        return BASELINE_TOKENS_PER_FRAME / self.tokens_per_frame


def score(method: str, setting: str, pred: np.ndarray, tokens_per_frame: float) -> Recon:
    """Reconstruction error of ``pred`` against the raw held-out stream."""
    err = np.abs(pred - held)
    stick, trig, analog = err[..., 0:4], err[..., 4:6], err[..., 0:6]
    return Recon(
        method=method,
        setting=setting,
        tokens_per_frame=tokens_per_frame,
        stick_mae=float(stick.mean()),
        stick_p99=float(np.percentile(stick, 99)),
        trig_mae=float(trig.mean()),
        trig_p99=float(np.percentile(trig, 99)),
        transition_p99=float(np.percentile(analog[HELD_TRANSITION], 99)),
        button_exact=float(((pred[..., 6:] > 0.5) == (held[..., 6:] > 0.5)).mean()),
    )


# %%
# --- (a) baseline: the quantization floor ----------------------------------------------
rows: list[Recon] = [score("baseline", "per-frame groups", held_baseline, BASELINE_TOKENS_PER_FRAME)]
BASELINE = rows[0]
print(
    f"baseline  stick MAE {BASELINE.stick_mae:.4f} p99 {BASELINE.stick_p99:.4f} | "
    f"trigger MAE {BASELINE.trig_mae:.4f} p99 {BASELINE.trig_p99:.4f} | "
    f"transition p99 {BASELINE.transition_p99:.4f} | buttons exact {BASELINE.button_exact:.6f}"
)
for i, name in enumerate(GROUP_NAMES):
    hold = float((held_groups[:, 1:, i] == held_groups[:, :-1, i]).mean())
    used = len(np.unique(held_groups[..., i]))
    print(f"  {name:<11} hold rate {hold:.3f}  ids used {used}/{GROUP_SIZES[i]}")


# %%
# --- BPE over symbol streams -----------------------------------------------------------
def flatten(seqs: np.ndarray) -> np.ndarray:
    """``[N, T]`` per-window symbol sequences → one stream with a ``-1`` sentinel after each
    window, so no merge can span two windows."""
    sep = np.full((seqs.shape[0], 1), -1, dtype=np.int64)
    return np.concatenate([seqs.astype(np.int64), sep], axis=1).reshape(-1)


def merge_positions(stream: np.ndarray, a: int, b: int) -> np.ndarray:
    """Left-to-right non-overlapping occurrences of the pair ``(a, b)``. Distinct symbols can
    never overlap; only a self-pair (a run) needs the greedy scan."""
    hit = np.flatnonzero((stream[:-1] == a) & (stream[1:] == b))
    if a != b or hit.size < 2:
        return hit
    keep: list[int] = []
    prev = -2
    for pos in hit.tolist():
        if pos > prev + 1:
            keep.append(pos)
            prev = pos
    return np.asarray(keep, dtype=np.int64)


def apply_merge(stream: np.ndarray, a: int, b: int, new_id: int) -> np.ndarray:
    pos = merge_positions(stream, a, b)
    if pos.size == 0:
        return stream
    stream = stream.copy()
    stream[pos] = new_id
    drop = np.zeros(stream.size, dtype=bool)
    drop[pos + 1] = True
    return stream[~drop]


def bpe_train(stream: np.ndarray, n_base: int, n_merges: int) -> list[tuple[int, int, int]]:
    """Merge table as ``(a, b, new_id)`` in application order. The table is a prefix chain, so
    truncating it to ``v - n_base`` entries gives the vocab-``v`` tokenizer exactly."""
    merges: list[tuple[int, int, int]] = []
    for new_id in range(n_base, n_base + n_merges):
        lo, hi = stream[:-1], stream[1:]
        ok = (lo >= 0) & (hi >= 0)
        if not ok.any():
            break
        code, count = np.unique((lo[ok] << 32) | hi[ok], return_counts=True)
        if count.max() < 2:
            break
        top = int(code[count.argmax()])
        a, b = top >> 32, top & 0xFFFFFFFF
        merges.append((a, b, new_id))
        stream = apply_merge(stream, a, b, new_id)
    return merges


def bpe_encode(stream: np.ndarray, merges: list[tuple[int, int, int]]) -> np.ndarray:
    for a, b, new_id in merges:
        stream = apply_merge(stream, a, b, new_id)
    return stream


def n_tokens(stream: np.ndarray) -> int:
    return int((stream >= 0).sum())


def expand(sym: int, table: dict[int, tuple[int, int]], n_base: int) -> list[int]:
    """A merged symbol back to its base-symbol sequence (iterative: merge chains get deep)."""
    out: list[int] = []
    stack = [sym]
    while stack:
        cur = stack.pop()
        if cur < n_base:
            out.append(cur)
        else:
            a, b = table[cur]
            stack.extend((b, a))
    return out


# %%
# --- (b) BPE stream 1: one joint symbol per frame --------------------------------------
def frame_codes(g: np.ndarray) -> np.ndarray:
    """``[N, L, 4]`` group ids → ``[N, L]`` joint code (the 4-tuple, mixed-radix)."""
    code = g[..., 0].astype(np.int64)
    for i in range(1, 4):
        code = code * GROUP_SIZES[i] + g[..., i]
    return code


def dense_ids(codes: np.ndarray, alphabet: dict[int, int]) -> tuple[np.ndarray, float]:
    """Joint codes → dense symbol ids under ``alphabet``, extending it for codes the fit split
    never produced. Returns the ids and the out-of-alphabet frame rate."""
    flat = codes.reshape(-1).tolist()
    out = np.empty(len(flat), dtype=np.int64)
    oov = 0
    for i, code in enumerate(flat):
        sym = alphabet.get(code)
        if sym is None:
            sym = len(alphabet)
            alphabet[code] = sym
            oov += 1
        out[i] = sym
    return out.reshape(codes.shape), oov / len(flat)


joint_alphabet: dict[int, int] = {}
fit_joint, _ = dense_ids(frame_codes(fit_groups), joint_alphabet)
n_fit_alphabet = len(joint_alphabet)
held_joint, joint_oov = dense_ids(frame_codes(held_groups), joint_alphabet)
# Merge ids start above the *whole* alphabet, held-out-only symbols included: a symbol the
# tokenizer cannot name is not lossless, so those tuples are part of the base cost.
n_joint_base = len(joint_alphabet)
print(f"joint frame symbols: {n_fit_alphabet} on fit, {n_joint_base} incl. held-out-only; OOV rate {joint_oov:.4%}")
print(f"joint-symbol hold rate (held): {(held_joint[:, 1:] == held_joint[:, :-1]).mean():.3f}")

# %%
joint_merges = bpe_train(flatten(fit_joint), n_joint_base, max(BPE_VOCABS) - n_joint_base)
print(f"joint BPE: {len(joint_merges)} merges trained (base {n_joint_base})")

# %%
# --- (b) BPE stream 2: the 4 group ids emitted separately -------------------------------
# The joint symbol's alphabet alone already exceeds a 1k vocab, so it cannot be run there at
# all. The factorized stream (base 355) is what a group-headed model actually emits and it
# reaches every vocab in the sweep; BPE has to rediscover the frame tuple itself.
fit_grouped = (fit_groups + GROUP_OFFSETS).reshape(fit_groups.shape[0], -1)
held_grouped = (held_groups + GROUP_OFFSETS).reshape(held_groups.shape[0], -1)
grouped_merges = bpe_train(flatten(fit_grouped), N_GROUP_SYMBOLS, max(BPE_VOCABS) - N_GROUP_SYMBOLS)
print(f"grouped BPE: {len(grouped_merges)} merges trained (base {N_GROUP_SYMBOLS})")


# %%
@dataclass(frozen=True, slots=True)
class BPEResult:
    stream: str
    vocab: int
    fit_tokens_per_frame: float
    held_tokens_per_frame: float
    utilization: float


def bpe_sweep(
    name: str,
    merges: list[tuple[int, int, int]],
    n_base: int,
    fit_syms: np.ndarray,
    held_syms: np.ndarray,
    n_fit_frames: int,
    n_held_frames: int,
) -> list[BPEResult]:
    out = []
    fit_stream, held_stream, done = flatten(fit_syms), flatten(held_syms), 0
    for vocab in sorted(BPE_VOCABS):
        if vocab < n_base:
            print(f"  {name} vocab {vocab}: INFEASIBLE — base alphabet is {n_base} symbols")
            continue
        # Merges are a prefix chain, so each vocab continues the previous encoding.
        table = merges[: vocab - n_base]
        if len(table) < vocab - n_base:
            # Training stopped early: no pair left occurs twice. The fit corpus, not the
            # vocab budget, is the binding constraint at this point.
            print(f"  {name} vocab {vocab}: merge table saturated at {n_base + len(table)} on this corpus")
        fit_stream = bpe_encode(fit_stream, merges[done : len(table)])
        held_stream = bpe_encode(held_stream, merges[done : len(table)])
        done = len(table)
        used = len(np.unique(held_stream[held_stream >= 0]))
        out.append(
            BPEResult(
                stream=name,
                vocab=n_base + len(table),
                fit_tokens_per_frame=n_tokens(fit_stream) / n_fit_frames,
                held_tokens_per_frame=n_tokens(held_stream) / n_held_frames,
                utilization=used / (n_base + len(table)),
            )
        )
        r = out[-1]
        print(
            f"  {name} vocab {r.vocab:>5}: fit {r.fit_tokens_per_frame:.3f} tok/frame, "
            f"held {r.held_tokens_per_frame:.3f} ({BASELINE_TOKENS_PER_FRAME / r.held_tokens_per_frame:.2f}x), "
            f"utilization {r.utilization:.1%}"
        )
    return out


n_fit_frames = fit.shape[0] * fit.shape[1]
n_held_frames = held.shape[0] * held.shape[1]
bpe_results = bpe_sweep("joint", joint_merges, n_joint_base, fit_joint, held_joint, n_fit_frames, n_held_frames)
bpe_results += bpe_sweep(
    "grouped", grouped_merges, N_GROUP_SYMBOLS, fit_grouped, held_grouped, n_fit_frames, n_held_frames
)

for r in bpe_results:
    # BPE is lossless over the quantized stream: it inherits the (a) floor exactly.
    rows.append(
        Recon(
            method=f"BPE/{r.stream}",
            setting=f"vocab {r.vocab}",
            tokens_per_frame=r.held_tokens_per_frame,
            stick_mae=BASELINE.stick_mae,
            stick_p99=BASELINE.stick_p99,
            trig_mae=BASELINE.trig_mae,
            trig_p99=BASELINE.trig_p99,
            transition_p99=BASELINE.transition_p99,
            button_exact=BASELINE.button_exact,
        )
    )

# %%
# --- are the learned merges run-length holds? ------------------------------------------
joint_by_id = {new: (a, b) for a, b, new in joint_merges}
grouped_by_id = {new: (a, b) for a, b, new in grouped_merges}


def classify(base: list[int], period: int) -> str:
    """What a merge's expansion *is*, in frames. ``period`` is the number of base symbols per
    frame (1 for the joint stream, 4 for the factorized one), so a merge that spells out one
    frame is "frame" and a merge that repeats the same frame is a run-length "hold"."""
    if len(base) % period:
        return "partial-frame"
    frames = [tuple(base[i : i + period]) for i in range(0, len(base), period)]
    if len(frames) == 1:
        return "frame"
    return "hold" if len(set(frames)) == 1 else "phrase"


def merge_census(merges: list[tuple[int, int, int]], table: dict, n_base: int, period: int, k: int) -> str:
    counts: dict[str, int] = {}
    for _, _, new_id in merges[:k]:
        kind = classify(expand(new_id, table, n_base), period)
        counts[kind] = counts.get(kind, 0) + 1
    n = min(k, len(merges))
    return "  ".join(f"{kind} {c / n:.0%}" for kind, c in sorted(counts.items(), key=lambda kv: -kv[1]))


print("top joint-symbol merges (a, b → expansion):")
for a, b, new_id in joint_merges[:12]:
    base = expand(new_id, joint_by_id, n_joint_base)
    print(f"  id {new_id:>6} = ({a}, {b}) → {len(base)} frames, {classify(base, 1)}")
print("merge census (what the learned merges are):")
for k in (16, 64, 256, 1024, 4096):
    if k <= len(joint_merges):
        print(f"  joint   first {k:>5}: {merge_census(joint_merges, joint_by_id, n_joint_base, 1, k)}")
for k in (16, 64, 256, 1024, 4096):
    if k <= len(grouped_merges):
        print(f"  grouped first {k:>5}: {merge_census(grouped_merges, grouped_by_id, N_GROUP_SYMBOLS, 4, k)}")


# %%
# --- (c) FAST-style DCT ------------------------------------------------------------------
def dct_roundtrip(a: np.ndarray, H: int, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """DCT the 6 analog channels per length-``H`` chunk, scalar-quantize the coefficients, and
    invert. Returns the reconstructed ``[N, L, 14]`` stream (buttons passed through untouched —
    a frequency code has nothing to say about a bit) and the ``[N, n_chunk, 6, H]`` quantized
    coefficients in the order they would be serialized (per channel, low → high frequency, so
    the high-frequency zero tail is a run)."""
    n, L, _ = a.shape
    n_chunk = L // H
    x = a[:, : n_chunk * H, :6].reshape(n, n_chunk, H, 6)
    coef = dct(x, axis=2, norm="ortho")
    q = np.round(coef / scale)
    rec = idct(q * scale, axis=2, norm="ortho")
    out = a.copy()
    out[:, : n_chunk * H, :6] = rec.reshape(n, n_chunk * H, 6)
    out[..., 0:4] = np.clip(out[..., 0:4], -1.0, 1.0)
    out[..., 4:6] = np.clip(out[..., 4:6], 0.0, 1.0)
    return out, q.astype(np.int64).transpose(0, 1, 3, 2)


def coef_symbols(q: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Quantized coefficients → ``[N, T]`` dense symbol ids (one window per row). ``levels`` is
    the integer level set, which the quantization scale fixes — it is not learned, so taking it
    over fit ∪ held leaks nothing while keeping every base id below the first merge id."""
    return np.searchsorted(levels, q.reshape(q.shape[0], -1))


# The button channels still need their own code alongside any analog DCT stream; price it once
# so the DCT rows can report a like-for-like all-14-channel token rate.
btn_merges = bpe_train(flatten(fit_groups[..., 0]), N_BUTTON_COMBOS, BTN_BPE_VOCAB - N_BUTTON_COMBOS)
btn_tokens_per_frame = n_tokens(bpe_encode(flatten(held_groups[..., 0]), btn_merges)) / n_held_frames
print(f"button-combo stream at vocab {BTN_BPE_VOCAB}: {btn_tokens_per_frame:.3f} tok/frame")

# %%
dct_recon: dict[tuple[int, float], np.ndarray] = {}
for H in DCT_H:
    for scale in DCT_SCALES:
        _, fit_q = dct_roundtrip(fit, H, scale)
        held_rec, held_q = dct_roundtrip(held, H, scale)
        levels = np.unique(np.concatenate([fit_q.reshape(-1), held_q.reshape(-1)]))
        n_base = len(levels)
        fit_syms, held_syms = coef_symbols(fit_q, levels), coef_symbols(held_q, levels)
        merges = bpe_train(flatten(fit_syms), n_base, max(0, DCT_BPE_VOCAB - n_base))
        analog_tpf = n_tokens(bpe_encode(flatten(held_syms), merges)) / n_held_frames
        total_tpf = analog_tpf + btn_tokens_per_frame
        dct_recon[(H, scale)] = held_rec
        rows.append(score("FAST-DCT", f"H={H} q={scale}", held_rec, total_tpf))
        print(
            f"  H={H:>2} q={scale:<5}: coef alphabet {n_base}, {len(merges)} merges → "
            f"analog {analog_tpf:.3f} + buttons {btn_tokens_per_frame:.3f} = {total_tpf:.3f} tok/frame"
        )

# %%
# --- summary table ------------------------------------------------------------------------
HEADER = (
    f"{'method':<14}{'setting':<20}{'tok/frm':>9}{'compr':>8}"
    f"{'stickMAE':>10}{'stickP99':>10}{'trigMAE':>9}{'trigP99':>9}{'transP99':>10}{'btnExact':>10}"
)
print(f"\nheld-out: {n_held_frames:,} frames | compression = 4 group-ids-per-frame / tok-per-frame")
print(HEADER)
print("-" * len(HEADER))
for r in rows:
    print(
        f"{r.method:<14}{r.setting:<20}{r.tokens_per_frame:>9.3f}{r.compression:>7.2f}x"
        f"{r.stick_mae:>10.4f}{r.stick_p99:>10.4f}{r.trig_mae:>9.4f}{r.trig_p99:>9.4f}"
        f"{r.transition_p99:>10.4f}{r.button_exact:>10.5f}"
    )

# %%
# --- figures --------------------------------------------------------------------------------
SCRATCH.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for stream, marker in (("joint", "o"), ("grouped", "s")):
    pts = [r for r in bpe_results if r.stream == stream]
    if not pts:
        continue
    axes[0].plot(
        [p.vocab for p in pts],
        [p.fit_tokens_per_frame for p in pts],
        marker + "--",
        alpha=0.5,
        label=f"{stream} (fit)",
    )
    axes[0].plot(
        [p.vocab for p in pts], [p.held_tokens_per_frame for p in pts], marker + "-", label=f"{stream} (held)"
    )
    axes[1].plot([p.vocab for p in pts], [100 * p.utilization for p in pts], marker + "-", label=stream)
axes[0].axhline(BASELINE_TOKENS_PER_FRAME, color="k", ls=":", label="baseline (4 ids/frame)")
axes[0].axhline(1.0, color="r", ls=":", label="4x bar (1 tok/frame)")
axes[0].set(xscale="log", yscale="log", xlabel="BPE vocab", ylabel="tokens / frame", title="Compression vs vocab")
axes[0].legend(fontsize=8)
axes[1].set(xscale="log", xlabel="BPE vocab", ylabel="vocab used on held-out (%)", title="Vocab utilization")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(SCRATCH / "compression.png", dpi=120)
plt.show()

# %%
fig, ax = plt.subplots(figsize=(7, 5))
for method, color in (
    ("baseline", "k"),
    ("BPE/joint", "tab:blue"),
    ("BPE/grouped", "tab:cyan"),
    ("FAST-DCT", "tab:red"),
):
    pts = [r for r in rows if r.method == method]
    ax.scatter([r.tokens_per_frame for r in pts], [r.stick_p99 for r in pts], c=color, label=method, zorder=3)
    for i, r in enumerate(pts):
        # The lossless rows all land on the floor line; stagger the labels so they stay legible.
        ax.annotate(
            r.setting,
            (r.tokens_per_frame, r.stick_p99),
            fontsize=6,
            xytext=(3, 5 + 7 * (i % 3)),
            textcoords="offset points",
        )
ax.axhline(BASELINE.stick_p99, color="k", ls=":", label="quantization floor")
ax.axvline(1.0, color="r", ls=":", label="4x compression bar")
ax.set(xscale="log", yscale="log", xlabel="tokens / frame (all 14 channels)", ylabel="stick p99 |error|")
ax.set_title("Reconstruction vs compression frontier (held-out)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(SCRATCH / "frontier.png", dpi=120)
plt.show()

# %%
# Worst-case overlay at a MATCHED token budget: the DCT setting that costs about what the
# cheapest (lossless) BPE tokenizer costs, on the window it damages most.
budget = min(r.tokens_per_frame for r in rows if r.method.startswith("BPE"))
dct_rows = [r for r in rows if r.method == "FAST-DCT"]
pick = min(dct_rows, key=lambda r: abs(np.log(r.tokens_per_frame / budget)))
WORST_KEY = next(k for k in dct_recon if f"H={k[0]} q={k[1]}" == pick.setting)
print(
    f"matched-budget overlay: BPE {budget:.3f} tok/frame (lossless) vs DCT {pick.setting} {pick.tokens_per_frame:.3f}"
)
worst_rec = dct_recon[WORST_KEY]
per_window = np.abs(worst_rec[..., 0:6] - held[..., 0:6]).max(axis=(1, 2))
w = int(per_window.argmax())
ch = int(np.abs(worst_rec[w, :, 0:6] - held[w, :, 0:6]).max(axis=0).argmax())
peak = int(np.abs(worst_rec[w, :, ch] - held[w, :, ch]).argmax())
lo = min(max(0, peak - 32), L_WINDOW - 64)
sl = slice(lo, lo + 64)

fig, ax = plt.subplots(figsize=(10, 4))
ax.step(range(sl.start, sl.stop), held[w, sl, ch], where="post", color="k", lw=2, label="raw")
ax.step(
    range(sl.start, sl.stop),
    held_baseline[w, sl, ch],
    where="post",
    color="tab:green",
    ls="--",
    label="baseline quantized",
)
ax.plot(
    range(sl.start, sl.stop), worst_rec[w, sl, ch], color="tab:red", label=f"DCT H={WORST_KEY[0]} q={WORST_KEY[1]}"
)
ax.set(
    xlabel="frame",
    ylabel=ACTION_CHANNELS[ch],
    title=f"Worst DCT window at a BPE-matched token budget ({ACTION_CHANNELS[ch]}, window {w})",
)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(SCRATCH / "worst_case.png", dpi=120)
plt.show()
print(f"figures written to {SCRATCH}")

# %% [markdown]
# ## Recommendation
#
# Bar: **clean reconstruction (at the per-frame quantization floor) at >=4x compression**,
# i.e. <=1 token/frame against the baseline's 4 group ids/frame. Numbers are the run printed
# above: 155k fit frames (70 replays) and 68k held-out frames (31 disjoint replays).
#
# - **(b) BPE over the joint frame symbol — PROMOTE, at vocab 4096.** 0.290 tok/frame on
#   held-out replays = **13.8x**, at exactly the (a) floor because it is lossless over the
#   quantized stream. The hypothesis holds outright: the first 256 merges are 100% pure
#   run-length holds, still 81% at 1k, and only past ~4k do multi-frame "phrases" take over —
#   BPE rediscovers RLE unprompted, and the top merges are literal doubling chains
#   (x,x) -> (xx,xx) -> ... Two caveats for training. (i) The joint alphabet is ~2.4k symbols
#   here and grows with corpus size, so a 1k vocab is not lossy but *unrepresentable*;
#   budget >=4k. (ii) 0.50% of held-out frames use a tuple the fit replays never emitted, so
#   build the alphabet over the full train split, not a sample, or carry an escape path.
# - **(b) BPE over the factorized group stream — PROMOTE only if the head must stay narrow.**
#   0.498 tok/frame (8.0x) at vocab 1024, same floor, and the best utilization in the sweep
#   (73%). It is behind the joint stream wherever both are feasible (0.319 vs 0.290 at 4k)
#   because ~30-40% of its merges are spent re-assembling the frame tuple that the joint
#   symbolization gets for free.
# - **(c) FAST-style DCT — REJECT, but not for the expected reason.** Finely quantized
#   (H=16, q=0.05) it clears the bar and is genuinely *better* on the sticks than the
#   hand-tuned per-frame codec — stick p99 0.030 vs 0.063, transition p99 0.030 vs 0.075, at
#   4.95x. The ringing hypothesis only bites once the quantizer coarsens: transition p99 goes
#   0.030 -> 0.092 -> 0.247 across the scale sweep, overtaking the overall stick p99, and H=24
#   is worse than H=16 on both error and tokens at every scale, so there is no long-range
#   smoothness to exploit. The reject is on dominance, not on ringing: at a matched budget
#   (0.336 vs 0.245 tok/frame) BPE is lossless where DCT sits at stick p99 0.224 / transition
#   p99 0.247, and DCT cannot code the buttons at all — a separate stream (+0.073 tok/frame
#   here) rides along regardless. Its one real edge, sub-cluster stick precision, is available
#   more cheaply by adding stick centers to (a), which BPE then compresses for free.
#
# Caveats. (i) Corpus size: the dev bundle is ~100 replays, and merge training saturates on it
# (no pair occurs twice past ~6.1k merges for joint, ~8.1k for grouped), so **the 16k-vocab rows
# are undertrained rather than representative** — they report the saturated table, and the
# fit/held gap already widens with vocab (0.275 -> 0.290 at 4k, 0.200 -> 0.245 at saturation),
# i.e. large tables start memorizing. Re-fit on the full train split before committing above 4k.
# (ii) The dev replays barely touch the analog triggers (9 of 25 trigger ids occur, baseline
# trigger p99 = 0), so the trigger columns understate what a production corpus would show; the
# stick columns carry the argument. (iii) tokens/frame is a compression number, not a modelling
# number — fewer tokens can still be harder to predict, so the next step is bits/frame under an
# actual model on the promoted BPE path, not a DCT run.
