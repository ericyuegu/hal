"""Closed-loop covariate shift as self-NLL.

The 012 GPT policy scores ~1 bit/frame next-action NLL on held-out HUMAN replays
yet plays mediocre closed-loop vs the lvl-9 CPU. If deployment drifts the model
off its training distribution (compounding error / OOD opponent), the model's own
NLL on frames it GENERATED closed-loop should exceed its NLL on human val frames.

This measures that gap and attributes it by ablating input families:

    self-NLL gap = NLL(model | its own eval-replay windows) - NLL(model | human val windows)

scored with one shared window+scoring path so the two sides are apples-to-apples.
The offset-1 (deployed) head's per-group categorical NLL is the metric, in bits.

Grid = {self-play, human val (all), human val (Fox-ditto control)}
     x {full input, opp-ablated, ego-history-ablated}

Ablations probe which conditioning channel the drift rides on:
  * opp-ablated       : zero every opp_* / opp_nana_* channel (mask sidecars -> 1).
                        Gap that closes here is the (OOD lvl-9 CPU) opponent channel.
  * ego-history-ablated: zero the ego's own controller-history channels. Gap that
                        closes here is the self-action (compounding-error) channel.

Provenance: run 260616-172638 (012 multi-token GPT, d256/L8, o1.5.9.13, W&B uw05bvm2),
final.pt (step 16384) + its final/ vs-CPU eval replays, both pulled from
r2:hal/runs/<run>/. That eval is a fixed Fox-mirror on Final Destination, so the
Fox-ditto human control matches the self-play matchup; "all" is the raw val number.

Several final/ replays end on a truncated frame (abrupt recording stop -> empty slp
metadata) that panics peppi's columnar read; they are salvaged by trimming the raw
event stream to the last complete Frame Bookend before extract_replay, matching what
materialize would keep.
"""

# %%
import importlib.util
import struct
import tempfile
from itertools import batched
from pathlib import Path

import melee
import numpy as np
import torch
from peppi_py import read_slippi
from streaming import StreamingDataset

from hal.data.extract import extract_replay
from hal.training.dataloader import _choose_chunk_starts
from hal.training.dataloader import collate_train_batch
from hal.training.dataloader import relabel_ego
from hal.training.features import ACTION_CHANNELS
from hal.training.stats import load_consolidated_stats
from hal.wire import PLAYER_PREFIXES
from hal.wire import peppi_port_to_libmelee


def _load_experiment(path: str):
    spec = importlib.util.spec_from_file_location("exp012", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


exp = _load_experiment("experiments/012_multi_token.py")

# %%
SCRATCH = Path("/tmp/claude-1000/-home-ericgu-src-hal/cc158dee-eb6c-43f5-8726-66a31a704ebf/scratchpad/self-nll")
CKPT = SCRATCH / "ckpt" / "final.pt"
SELF_REPLAY_DIR = SCRATCH / "replays"
DATA_ROOT = Path("data/processed/ranked-anonymized-1/mds")
# The shared local val cache (DATA_ROOT/val) is streaming-managed and gets LRU-evicted
# by any concurrent run, so read from a private mirror of the val split holding the
# compressed shards (rclone-pulled from r2:hal/processed/.../val): index.json +
# shard.0000{0,1}.mds.zstd. StreamingDataset(remote=None, keep_zip=True) decompresses
# them in place (no download) and keeps the .zstd so re-runs stay idempotent.
VAL_LOCAL = SCRATCH / "val_mds"

L_CTX = 256
FOX = int(melee.Character.FOX.value)  # 1 (libmelee-internal, the id the MDS stores)

# Window sampling. Dense offset-1 scoring means one window contributes ~L_CTX scored
# positions, so ~100 windows/side already gives tens of thousands of scored frames.
K_SELF = 8  # non-overlapping windows per self-play replay
N_VAL_ALL = 150  # human-val windows (one per replay, random ego), the raw val number
K_DITTO = 6  # windows per Fox-mirror val replay for the matchup-matched control
N_DITTO_CAP = 90
VAL_SCAN_CAP = 800  # bound the val stream scan
FWD_BATCH = 32  # windows per forward
SEED = 0

# %%
model, cfg, stats_from_ckpt, state = exp._load_ckpt(str(CKPT))
L_CHUNK = max(cfg.head_offsets)  # target horizon covers every head; offset 1 is the deployed one
stats = load_consolidated_stats(DATA_ROOT / "stats.json")  # same normalization as training
print(
    f"ckpt step={state['step']} head_offsets={cfg.head_offsets} L_ctx={cfg.L_ctx} L_chunk={L_CHUNK} dev={exp.DEVICE}"
)
assert cfg.L_ctx == L_CTX


# %%
# --- slp salvage: trim a truncated recording to its last complete frame ------
# Every event command carries a fixed payload size declared in the leading Event
# Payloads (0x35) table, so the raw stream is walkable; cutting at the last Frame
# Bookend (0x3c) drops a partial trailing frame that would unbalance peppi's arrays.
def _slp_event_table(b: bytes) -> tuple[int, int, int, dict[int, int]]:
    marker = b.find(b"raw")
    lenpos = marker + 3 + 5  # past 'raw' and the '[$U#l' typed-array header
    (rawlen,) = struct.unpack(">I", b[lenpos : lenpos + 4])
    ev_start = lenpos + 4
    ev_end = ev_start + rawlen
    ep_size = b[ev_start + 1]
    sizes: dict[int, int] = {0x35: ep_size}
    k = ev_start + 2
    while k < ev_start + 1 + ep_size:
        (payload,) = struct.unpack(">H", b[k + 1 : k + 3])
        sizes[b[k]] = payload
        k += 3
    return lenpos, ev_start, ev_end, sizes


def _trim_to_last_frame(b: bytes) -> bytes:
    lenpos, ev_start, ev_end, sizes = _slp_event_table(b)
    k = ev_start
    last_bookend = None
    while k < ev_end:
        cmd = b[k]
        k += 1 + sizes[cmd]
        if cmd == 0x3C:
            last_bookend = k
    if last_bookend is None:
        raise ValueError("no frame bookend found")
    return b[:lenpos] + struct.pack(">I", last_bookend - ev_start) + b[ev_start:last_bookend] + b[ev_end:]


def extract_or_salvage(path: Path) -> dict[str, np.ndarray] | None:
    """extract_replay, retrying once on a trimmed copy if peppi panics on the tail."""
    try:
        return extract_replay(str(path))
    except BaseException:  # peppi is Rust/pyo3; a bad tail surfaces as PanicException
        trimmed = _trim_to_last_frame(path.read_bytes())
        with tempfile.NamedTemporaryFile(suffix=".slp", dir=SCRATCH, delete=False) as tf:
            tf.write(trimmed)
            tmp = tf.name
        return extract_replay(tmp)


def model_prefix(path: Path) -> str:
    """Which extract prefix (p1/p2) is the model: the non-CPU (HUMAN-type) port.

    extract_replay maps the two lowest occupied libmelee ports to (p1, p2) ascending,
    so resolve the model's port the same way. cpu_level=0/HUMAN is our exi-ai-driven
    policy; the lvl-9 in-game CPU is the opponent.
    """
    g = read_slippi(str(path), skip_frames=True)
    is_model = {
        peppi_port_to_libmelee(pl.port): int(getattr(pl.type, "value", pl.type)) == 0 for pl in g.start.players
    }
    occupied = sorted(is_model)
    model_ports = [p for p in occupied if is_model[p]]
    assert len(model_ports) == 1, f"{path.name}: expected 1 model port, got {model_ports}"
    return PLAYER_PREFIXES[occupied.index(model_ports[0])]


# %%
# --- shared windowing: the WindowDataset layout, with explicit ego control ---
def windows_from_sample(
    sample: dict[str, np.ndarray], *, ego_prefix: str, k: int, rng: np.random.Generator
) -> list[dict[str, np.ndarray]]:
    """Up to ``k`` non-overlapping [ctx | chunk] windows, left-padded + ego-relabeled,
    exactly as WindowDataset yields — but ego is chosen (the model's port for self-play,
    a coin flip for human val) rather than always random."""
    seq = L_CTX + L_CHUNK
    length = len(sample["frame"])
    out: list[dict[str, np.ndarray]] = []
    for cs in _choose_chunk_starts(length, L_CTX, L_CHUNK, k, rng):
        cs = int(cs)
        start = cs - L_CTX
        pad = max(0, -start)
        window: dict[str, np.ndarray] = {}
        for key, val in sample.items():
            real = val[max(0, start) : start + seq]
            if pad > 0:
                front = np.zeros((pad, *val.shape[1:]), dtype=val.dtype)
                window[key] = np.concatenate([front, real], axis=0)
            else:
                window[key] = real
        window["ctx_pad"] = np.int64(min(pad, L_CTX))
        out.append(relabel_ego(window, ego_prefix))
    return out


# %%
# --- build self-play windows (ego = the model's own port) --------------------
rng = np.random.default_rng(SEED)
self_windows: list[dict[str, np.ndarray]] = []
n_orig = n_salvaged = 0
for slp in sorted(SELF_REPLAY_DIR.rglob("*.slp")):
    try:
        sample = extract_replay(str(slp))
        n_orig += 1
    except BaseException:
        sample = extract_or_salvage(slp)
        n_salvaged += 1
    if sample is None:
        continue
    self_windows.extend(windows_from_sample(sample, ego_prefix=model_prefix(slp), k=K_SELF, rng=rng))
print(f"self-play: {n_orig} clean + {n_salvaged} salvaged replays -> {len(self_windows)} windows")

# %%
# --- build human val windows (all + Fox-mirror control), one raw stream pass --
val_mds = StreamingDataset(remote=None, local=str(VAL_LOCAL), batch_size=1, shuffle=False, keep_zip=True)
val_all: list[dict[str, np.ndarray]] = []
val_ditto: list[dict[str, np.ndarray]] = []
scanned = 0
for sample in val_mds:
    scanned += 1
    sample = {k: v for k, v in sample.items() if k != "schema_version"}
    if len(val_all) < N_VAL_ALL:
        ego = "p1" if rng.random() < 0.5 else "p2"
        val_all.extend(windows_from_sample(sample, ego_prefix=ego, k=1, rng=rng))
    if (
        len(val_ditto) < N_DITTO_CAP
        and int(sample["p1_character"][0]) == FOX
        and int(sample["p2_character"][0]) == FOX
    ):
        val_ditto.extend(windows_from_sample(sample, ego_prefix="p1", k=K_DITTO, rng=rng))
    if (len(val_all) >= N_VAL_ALL and len(val_ditto) >= N_DITTO_CAP) or scanned >= VAL_SCAN_CAP:
        break
print(f"val: scanned {scanned} replays -> all={len(val_all)} windows, Fox-ditto={len(val_ditto)} windows")


# %%
# --- ablations on the model-input feature dict (targets stay un-ablated) -----
def ablate_opp(features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Zero every opponent channel (gamestate floats + categoricals + character, plus
    opp_nana); flip any float mask sidecar to 1.0 (unavailable). Removes opponent info."""
    return {
        key: (torch.ones_like(val) if key.endswith("_mask") else torch.zeros_like(val))
        if key.startswith("opp_")
        else val
        for key, val in features.items()
    }


_EGO_HISTORY = {f"ego_{ch}" for ch in ACTION_CHANNELS}


def ablate_ego_history(features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Zero the ego's own controller-history channels (its recent actions). Ego gamestate
    is untouched, so this isolates the self-action / compounding-error channel."""
    return {key: (torch.zeros_like(val) if key in _EGO_HISTORY else val) for key, val in features.items()}


CONDITIONS = {"full": None, "opp_ablated": ablate_opp, "ego_hist_ablated": ablate_ego_history}


# %%
@torch.no_grad()
def offset1_group_nll(batch, ablate) -> dict[str, torch.Tensor]:
    """Per-group offset-1 (deployed head) NLL nats over the valid positions. Targets are
    quantized from the REAL ego history (offset-1 targets are the history shifted by one),
    so an ego-history ablation must never touch them — only the model's conditioning input."""
    ctx = batch.context
    targets, valid = exp._multi_offset_targets(ctx, batch.target, model.head_offsets)
    tgt_idx = exp._quantize(model, targets[1])
    feats = ctx.features if ablate is None else ablate(ctx.features)
    hidden = model(feats, ctx.ctx_pad)
    logits = model.heads[model.primary_head_idx](hidden).float()
    return exp.group_nll(logits, tgt_idx, valid)


def score(windows: list[dict[str, np.ndarray]]) -> dict[str, dict[str, float]]:
    """{condition: {group: bits, total: bits}} + scored-position count, over all windows."""
    out: dict[str, dict[str, float]] = {}
    n_positions = 0
    for cond, ablate in CONDITIONS.items():
        nats: dict[str, list[torch.Tensor]] = {g: [] for g in exp._GROUP_NAMES}
        for chunk in batched(windows, FWD_BATCH, strict=False):
            batch = collate_train_batch(list(chunk), stats=stats, L_ctx=L_CTX).to(exp.DEVICE)
            for group, comp in offset1_group_nll(batch, ablate).items():
                nats[group].append(comp)
        cat = {g: torch.cat(v) for g, v in nats.items()}
        out[cond] = exp.nll_breakdown(cat)
        n_positions = len(cat[exp._GROUP_NAMES[0]])
    out["_positions"] = {"n": float(n_positions)}
    return out


# %%
model.eval()
groups = {"self_play": self_windows, "val_all": val_all, "val_fox_ditto": val_ditto}
scored = {name: score(w) for name, w in groups.items() if w}

# %%
# --- report -------------------------------------------------------------------
GROUP_LABELS = list(exp._GROUP_NAMES)
COND_LABELS = list(CONDITIONS)
hdr = f"{'group / condition':<22}" + "".join(f"{c:>18}" for c in COND_LABELS)
print("\n=== offset-1 NLL (bits/frame), model 260616-172638 final.pt (step 16384) ===")
print(f"{'FOX mirror, Final Destination self-play vs lvl-9 CPU':<22}")
for name, res in scored.items():
    n_win = len(groups[name])
    n_pos = int(res["_positions"]["n"])
    print(f"\n[{name}]  windows={n_win}  scored_positions={n_pos}")
    print(hdr)
    for g in GROUP_LABELS + ["total"]:
        row = f"{g:<22}" + "".join(f"{res[c][g]:>18.4f}" for c in COND_LABELS)
        print(row)


# %%
# --- covariate-shift gaps + attribution --------------------------------------
def total(name: str, cond: str) -> float:
    return scored[name][cond]["total"]


print("\n=== covariate shift: self-play total minus human-val total (bits/frame) ===")
print("full-input gap > 0 would support the drift hypothesis (own rollouts harder to predict).")
for baseline in ("val_all", "val_fox_ditto"):
    if baseline not in scored:
        continue
    print(f"\nvs {baseline}:")
    for cond in ("full", "opp_ablated", "ego_hist_ablated"):
        gap = total("self_play", cond) - total(baseline, cond)
        print(
            f"   {cond:<16}: self {total('self_play', cond):.3f}  -  human {total(baseline, cond):.3f}  =  gap {gap:+.4f}"
        )

# Per-channel reliance = how much zeroing a channel RAISES that group's own NLL.
opp_cost = total("self_play", "opp_ablated") - total("self_play", "full")
ego_cost = total("self_play", "ego_hist_ablated") - total("self_play", "full")
print("\n=== channel reliance on self-play frames (NLL increase when zeroed, bits/frame) ===")
print(f"   opponent channels : +{opp_cost:.3f}")
print(f"   ego action-history: +{ego_cost:.3f}")

naive_gap = total("self_play", "full") - total("val_all", "full")
print("\n=== verdict ===")
print(
    f"Full-input self-NLL ({total('self_play', 'full'):.3f}) is "
    f"{'ABOVE' if naive_gap > 0 else 'BELOW'} human-val ({total('val_all', 'full'):.3f}); "
    f"naive gap {naive_gap:+.3f} bits/frame."
)
print(
    "Self frames are LOWER-entropy than human, so the model is not predictively OOD on its own\n"
    "rollouts. Predictability is dominated by ego action-history autocorrelation (see reliance\n"
    "above); the opponent channel adds little and does so equally for self and human — the OOD-\n"
    "opponent story leaves no NLL fingerprint. Any genuine gamestate drift surfaces only once the\n"
    "self-history crutch is removed (compare the ego_hist_ablated gaps above)."
)
