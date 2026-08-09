"""How much closed-loop capability does the policy's action discretization destroy?

The policy predicts DISCRETIZED actions: main stick -> 65 hand-tuned cluster centers,
c-stick -> 9, trigger pair -> 5x5, buttons -> exact 256-combo (lossless). This notebook
bounds how much of any closed-loop gap could be blamed on that encoding alone, by asking:
does a discretized human still wavedash / shorthop / airdodge / recover, or does snapping
to centers break frame-precise technique?

Paired-replay design (controls for emulator build drift):
    Replaying a recorded input stream on our Dolphin build diverges from the ORIGINAL
    recording (build/era drift), so we never compare against the source .slp. Instead we
    drive TWO same-build rollouts of the same human input stream and compare them:
      RAW    = inputs as stored in the MDS.
      QUANT  = dequantize(quantize(RAW)) built from hal/training/scoring.py primitives
               (buttons pass through exactly; only the 6 analog stick/trigger channels snap).
    Both are fixed input sequences fed through the round-trip ControllerSource path
    (hal/sim: MDSControllerSource -> drive -> Trajectory), identical to tests/test_roundtrip.

    Determinism guard: Melee cannot be seeded through stock Dolphin (diff's seed tripwire),
    so we run RAW twice and audit the floor per replay. Empirically raw-vs-raw is bit-exact
    for hundreds of frames on every replay (and for the full 2400-frame window on about half;
    late breaks cluster in Nana-AI / stage-hazard / fast-faller matchups). Every quant onset
    lands before its replay's floor break, so onset stats are pure discretization; long-window
    technique drift is reported for the clean-floor subset separately from the full set.

    Note on technique counts: the button channels are lossless, so the discretizer can never
    directly suppress a jump/airdodge press. Any change in per-rollout technique COUNT is
    therefore downstream game-state (butterfly) divergence seeded by the tiny analog snap, not
    the discretizer refusing to execute. To isolate the discretizer's DIRECT information loss
    (free of butterfly), we add an offline, game-state-independent pass: per-frame snap
    displacement and whether discretization erases sub-cell stick motion.

Headless local Dolphin (default GCPad pipe path, real-time-but-uncapped ~200 fps). No GPU.
"""

# %%
import multiprocessing as mp

# libmelee's slippstream client spawns a child via mp.Process; on Python 3.14 the default
# "forkserver" re-imports this module and is flaky alongside torch/streaming. Force fork,
# matching tests/test_roundtrip.py and the other notebooks.
if mp.get_start_method(allow_none=True) != "fork":
    mp.set_start_method("fork", force=True)

import time
from dataclasses import dataclass
from pathlib import Path

import melee
import numpy as np
import torch
from loguru import logger
from streaming import StreamingDataset

from hal.data.index import ReplayIndexEntry
from hal.data.index import read_jsonl
from hal.paths import EMULATOR_PATH
from hal.paths import ISO_PATH
from hal.sim.diff import diff
from hal.sim.loop import drive
from hal.sim.session import ReplayMatchup
from hal.sim.session import Session
from hal.sim.sources import ControllerSource
from hal.sim.sources import MDSControllerSource
from hal.sim.trajectory import Trajectory
from hal.training.scoring import STICK_CLUSTER_CENTERS_C
from hal.training.scoring import STICK_CLUSTER_CENTERS_MAIN
from hal.training.scoring import TRIGGER_CENTERS
from hal.training.scoring import center_to_value
from hal.training.scoring import cluster_to_xy
from hal.training.scoring import nearest_center
from hal.training.scoring import nearest_cluster
from hal.wire import CHARACTERS_BY_NAME

# %%
MDS_DIR = Path("/home/ericgu/src/hal/data/processed/ranked-anonymized-1/mds")
TRAIN_SPLIT = MDS_DIR / "train"

# Round-trip fixture filter (tests/test_roundtrip.py): RNG-stable stages so raw-vs-raw is
# bit-exact, and characters libmelee can menu-select whose physics don't RNG-desync.
RNG_STABLE_STAGES = {31, 32, 28, 8}  # slp-native BF, FD, Dreamland, Yoshi's Story
EXCLUDED_CHARACTERS = {
    CHARACTERS_BY_NAME["PEACH"],
    CHARACTERS_BY_NAME["GAMEANDWATCH"],
    CHARACTERS_BY_NAME["SHEIK"],
}

# libmelee Action enum ids used by the technique detectors (verified against melee.Action).
KNEE_BEND = 24  # jumpsquat (grounded; precedes every jump AND every grounded wavedash)
AIRDODGE = 236
LANDING = 42
LANDING_SPECIAL = 43  # air-dodge / special-fall landing lag == the wavedash/waveland landing
WAVEDASH_AIRDODGE_WINDOW = 5  # frames from jumpsquat to the airdodge
WAVEDASH_LAND_WINDOW = 16  # frames from jumpsquat to LANDING_SPECIAL
FULLHOP_APEX_GAIN = 25.0  # position_y rise (units) above the jumpsquat that calls a jump "full"

K_ROLLOUT = 12  # replays driven through the emulator (raw x2 + quant each)
N_FRAMES = 2400  # ~40 s per rollout — enough techniques, before butterfly saturates
# Early comparison window: game states still overlap heavily here (quant divergence has begun
# but not compounded), so a technique-count delta in this window IS the discretizer failing to
# execute, not butterfly. Full-window counts measure drift between two different games instead.
N_EARLY = 600
N_OFFLINE = 40  # replays for the cheap game-state-independent snap analysis

TRIGGER_CHANNELS = ("trigger_l", "trigger_r")


# %%
# --- quantize / dequantize (thin wrappers over hal/training/scoring primitives) ----------
def quantize_prefix(row: dict[str, np.ndarray], prefix: str) -> dict[str, np.ndarray]:
    """dequantize(quantize(raw)) for one port's 6 analog channels, exactly as the policy's
    decode would render its own predictions. Buttons are lossless and pass through, so they
    are not touched here."""
    out: dict[str, np.ndarray] = {}
    main = torch.tensor(
        np.stack([row[f"{prefix}_main_stick_x"], row[f"{prefix}_main_stick_y"]], axis=-1), dtype=torch.float32
    )
    main_q = cluster_to_xy(nearest_cluster(main, STICK_CLUSTER_CENTERS_MAIN), STICK_CLUSTER_CENTERS_MAIN)
    out[f"{prefix}_main_stick_x"] = main_q[:, 0].numpy()
    out[f"{prefix}_main_stick_y"] = main_q[:, 1].numpy()
    c = torch.tensor(np.stack([row[f"{prefix}_c_stick_x"], row[f"{prefix}_c_stick_y"]], axis=-1), dtype=torch.float32)
    c_q = cluster_to_xy(nearest_cluster(c, STICK_CLUSTER_CENTERS_C), STICK_CLUSTER_CENTERS_C)
    out[f"{prefix}_c_stick_x"] = c_q[:, 0].numpy()
    out[f"{prefix}_c_stick_y"] = c_q[:, 1].numpy()
    for shoulder in TRIGGER_CHANNELS:
        t = torch.tensor(row[f"{prefix}_{shoulder}"], dtype=torch.float32)
        out[f"{prefix}_{shoulder}"] = center_to_value(nearest_center(t, TRIGGER_CENTERS), TRIGGER_CENTERS).numpy()
    return out


def make_quant_row(row: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Copy of an MDS row with both ports' analog channels snapped to the policy's grids."""
    q = dict(row)
    for prefix in ("p1", "p2"):
        q.update(quantize_prefix(row, prefix))
    return q


# %%
# --- technique detectors (eval-side Melee analysis over a rollout's action-state stream) ---
@dataclass(frozen=True, slots=True)
class TechniqueCounts:
    jumpsquats: int  # KNEE_BEND onsets (every jump + every grounded wavedash starts here)
    airdodges: int  # AIRDODGE onsets (recovery / wavedash / defensive)
    wavedashes: int  # jumpsquat -> airdodge -> LANDING_SPECIAL: the frame-precise ground tech
    shorthops: int  # non-wavedash jumps with a low apex
    fullhops: int  # non-wavedash jumps with a high apex


def _rising_edges(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    return np.flatnonzero(m & ~np.concatenate([[False], m[:-1]]))


def detect_techniques(traj: Trajectory, port: int, n: int) -> TechniqueCounts:
    """Count movement techniques from one port's post-frame action-state + height streams.

    Wavedash = KNEE_BEND (jumpsquat) then AIRDODGE within WAVEDASH_AIRDODGE_WINDOW then
    LANDING_SPECIAL within WAVEDASH_LAND_WINDOW. Remaining jumpsquats that leave the ground
    are split shorthop/fullhop by apex height gain (button-hold-driven, so a discretization-
    insensitive control: it must stay ~constant if buttons are truly lossless)."""
    act = np.nan_to_num(traj.post[port]["action"][:n], nan=-1.0).astype(int)
    pos_y = np.asarray(traj.post[port]["position_y"][:n], dtype=np.float64)
    m = len(act)
    jumpsquats = _rising_edges(act == KNEE_BEND)
    airdodge_onsets = set(_rising_edges(act == AIRDODGE).tolist())
    n_wave = short = full = 0
    for t0 in jumpsquats:
        airdodged = any((t0 <= a < min(t0 + WAVEDASH_AIRDODGE_WINDOW + 1, m)) for a in airdodge_onsets)
        if airdodged and np.any(act[t0 : min(t0 + WAVEDASH_LAND_WINDOW, m)] == LANDING_SPECIAL):
            n_wave += 1
            continue
        end = t0 + 1
        while end < m and act[end] not in (LANDING, LANDING_SPECIAL) and end < t0 + 90:
            end += 1
        gain = float(np.nanmax(pos_y[t0:end]) - pos_y[t0]) if end > t0 else 0.0
        if gain >= FULLHOP_APEX_GAIN:
            full += 1
        else:
            short += 1
    return TechniqueCounts(len(jumpsquats), len(airdodge_onsets), n_wave, short, full)


def action_match_fraction(a: Trajectory, b: Trajectory, port: int, n: int) -> float:
    """Fraction of frames where the two rollouts share the same action-state id for a port."""
    aa = np.nan_to_num(a.post[port]["action"][:n], nan=-1.0).astype(int)
    bb = np.nan_to_num(b.post[port]["action"][:n], nan=-1.0).astype(int)
    return float(np.mean(aa == bb))


def position_gap(a: Trajectory, b: Trajectory, n: int) -> np.ndarray:
    """Per-frame max over ports of the L2 gap between the two rollouts' positions (units)."""
    gaps = []
    for port in a.post:
        dx = np.asarray(a.post[port]["position_x"][:n], float) - np.asarray(b.post[port]["position_x"][:n], float)
        dy = np.asarray(a.post[port]["position_y"][:n], float) - np.asarray(b.post[port]["position_y"][:n], float)
        gaps.append(np.hypot(np.nan_to_num(dx), np.nan_to_num(dy)))
    return np.max(np.stack(gaps, axis=0), axis=0)


def first_action_divergence(a: Trajectory, b: Trajectory, n: int) -> int | None:
    """Earliest frame where any port's action-state id differs between the rollouts."""
    first: int | None = None
    for port in a.post:
        aa = np.nan_to_num(a.post[port]["action"][:n], nan=-1.0).astype(int)
        bb = np.nan_to_num(b.post[port]["action"][:n], nan=-1.0).astype(int)
        idx = np.flatnonzero(aa != bb)
        if idx.size:
            first = int(idx[0]) if first is None else min(first, int(idx[0]))
    return first


# %%
# --- pick round-trip-safe train replays --------------------------------------------------
def pick_safe_entries(n: int) -> list[ReplayIndexEntry]:
    entries: list[ReplayIndexEntry] = []
    for e in read_jsonl(MDS_DIR / "manifest.jsonl"):
        if (
            e.annotation is not None
            and e.annotation.split == "train"
            and e.stage in RNG_STABLE_STAGES
            and len(e.players) == 2
            and not any(p.character in EXCLUDED_CHARACTERS for p in e.players)
        ):
            entries.append(e)
            if len(entries) >= n:
                break
    return entries


N_SAFE = max(K_ROLLOUT, N_OFFLINE)
safe_entries = pick_safe_entries(N_SAFE)
logger.info(f"selected {len(safe_entries)} round-trip-safe train replays")
dataset = StreamingDataset(local=str(TRAIN_SPLIT), remote=None, shuffle=False, batch_size=1, allow_unsafe_types=False)


# %%
# --- OFFLINE pass: game-state-independent discretizer displacement + motion erasure -------
# Isolates the discretizer's DIRECT information loss with no emulator and no butterfly. For
# every stored analog frame we measure how far the snap moves the value, split by regime, and
# how much sub-cell stick motion (fine DI/angle adjustment) discretization throws away.
def _rad_to_deg(x: float) -> float:
    return float(x * 180.0 / np.pi)


rim_disp: list[np.ndarray] = []  # main-stick L2 snap error on the rim (|v|>0.95): DI/wavedash
mid_disp: list[np.ndarray] = []  # main-stick L2 snap error in the mid band (0.3<|v|<=0.95)
rim_angle_deg: list[np.ndarray] = []  # main-stick angular snap error on the rim (degrees)
cstick_disp: list[np.ndarray] = []  # c-stick L2 snap error on active (non-neutral) frames
trig_disp: list[np.ndarray] = []  # trigger snap error on engaged (>0) frames
main_moved = main_move_erased = 0  # sub-cell erasure: raw main-stick moved but class did not
n_offline_frames = 0

for e in safe_entries[:N_OFFLINE]:
    row = dataset[e.annotation.mds_row_idx]
    for prefix in ("p1", "p2"):
        mx = np.asarray(row[f"{prefix}_main_stick_x"], np.float32)
        my = np.asarray(row[f"{prefix}_main_stick_y"], np.float32)
        n_offline_frames += len(mx)
        main = torch.tensor(np.stack([mx, my], axis=-1), dtype=torch.float32)
        cls = nearest_cluster(main, STICK_CLUSTER_CENTERS_MAIN)
        snap = cluster_to_xy(cls, STICK_CLUSTER_CENTERS_MAIN)
        disp = torch.linalg.vector_norm(main - snap, dim=-1).numpy()
        mag = np.hypot(mx, my)
        rim = mag > 0.95
        mid = (mag > 0.3) & (mag <= 0.95)
        rim_disp.append(disp[rim])
        mid_disp.append(disp[mid])
        # angular error on the rim: angle between raw and snapped stick vector
        if rim.any():
            raw_ang = np.arctan2(my[rim], mx[rim])
            snp = snap.numpy()[rim]
            snap_ang = np.arctan2(snp[:, 1], snp[:, 0])
            d = np.abs(np.arctan2(np.sin(raw_ang - snap_ang), np.cos(raw_ang - snap_ang)))
            rim_angle_deg.append(np.array([_rad_to_deg(v) for v in d]))
        # sub-cell motion erasure: of frames where the raw stick moved, how many kept the same
        # cluster class (the movement is discarded by discretization).
        moved = (np.diff(mx) != 0) | (np.diff(my) != 0)
        class_changed = np.diff(cls.numpy()) != 0
        main_moved += int(moved.sum())
        main_move_erased += int((moved & ~class_changed).sum())

        cx = np.asarray(row[f"{prefix}_c_stick_x"], np.float32)
        cy = np.asarray(row[f"{prefix}_c_stick_y"], np.float32)
        c = torch.tensor(np.stack([cx, cy], axis=-1), dtype=torch.float32)
        c_snap = cluster_to_xy(nearest_cluster(c, STICK_CLUSTER_CENTERS_C), STICK_CLUSTER_CENTERS_C)
        c_disp = torch.linalg.vector_norm(c - c_snap, dim=-1).numpy()
        cstick_disp.append(c_disp[np.hypot(cx, cy) > 1e-6])

        for shoulder in TRIGGER_CHANNELS:
            tv = np.asarray(row[f"{prefix}_{shoulder}"], np.float32)
            t = torch.tensor(tv, dtype=torch.float32)
            t_snap = center_to_value(nearest_center(t, TRIGGER_CENTERS), TRIGGER_CENTERS).numpy()
            trig_disp.append(np.abs(tv - t_snap)[tv > 1e-6])


def _pct(a: list[np.ndarray], qs=(50, 90, 99, 100)) -> dict[str, float]:
    x = np.concatenate(a) if a else np.array([0.0])
    return {"n": int(x.size), "mean": float(x.mean()), **{f"p{q}": float(np.percentile(x, q)) for q in qs}}


print("\n==================== OFFLINE discretizer displacement ====================")
print(f"frames analyzed: {n_offline_frames:,} over {min(N_OFFLINE, len(safe_entries))} replays x 2 ports")
print("main-stick L2 snap error [rim |v|>0.95, where DI/wavedash/firefox angles live]:", _pct(rim_disp))
print("main-stick L2 snap error [mid  0.3<|v|<=0.95]:", _pct(mid_disp))
print("main-stick ANGULAR snap error on the rim (degrees):", _pct(rim_angle_deg))
print("c-stick  L2 snap error [active frames]:", _pct(cstick_disp))
print("trigger  snap error [engaged frames]:", _pct(trig_disp))
print(
    f"sub-cell motion erasure: {main_move_erased:,}/{main_moved:,} raw main-stick moves "
    f"({100.0 * main_move_erased / max(main_moved, 1):.1f}%) stayed in the same cluster (movement discarded)"
)


# %%
# --- ROLLOUT pass: paired raw / raw / quant through the emulator --------------------------
def make_sources(row: dict[str, np.ndarray]) -> dict[int, ControllerSource]:
    return {
        1: MDSControllerSource(columns=row, port_prefix="p1"),
        2: MDSControllerSource(columns=row, port_prefix="p2"),
    }


def run_rollout(matchup: ReplayMatchup, row: dict[str, np.ndarray], slippi_port: int, n: int) -> Trajectory:
    """One same-build rollout of a fixed input stream via the proven bit-exact GCPad path
    (default Session settings, as in tests/test_roundtrip.py)."""
    s = Session(iso_path=ISO_PATH, dolphin_path=EMULATOR_PATH, slippi_port=slippi_port)
    with s:
        return drive(s, matchup, make_sources(row), max_frames=n)


@dataclass(frozen=True, slots=True)
class ReplayResult:
    stage: int
    chars: dict[int, str]
    n: int
    floor_frame: int | None  # first raw-vs-raw divergence (None => bit-exact over the window)
    quant_first_any: int | None  # first raw-vs-quant divergence, any post-field
    quant_first_action: int | None  # first raw-vs-quant action-state divergence
    pos_gap: np.ndarray  # per-frame max-over-ports position L2 gap, raw-vs-quant
    raw_counts: dict[int, TechniqueCounts]  # full window
    quant_counts: dict[int, TechniqueCounts]
    raw_counts_early: dict[int, TechniqueCounts]  # first N_EARLY frames (states still overlap)
    quant_counts_early: dict[int, TechniqueCounts]
    action_match: dict[int, float]


results: list[ReplayResult] = []
port_base = 51470
t_start = time.monotonic()
for i, e in enumerate(safe_entries[:K_ROLLOUT]):
    row = dataset[e.annotation.mds_row_idx]
    n = min(N_FRAMES, len(row["p1_main_stick_x"]) - 2)
    matchup = ReplayMatchup.from_replay(e)
    chars = {p.port: melee.Character(p.character).name for p in e.players}
    try:
        raw_a = run_rollout(matchup, row, port_base + i * 4 + 0, n)
        raw_b = run_rollout(matchup, row, port_base + i * 4 + 1, n)
        quant = run_rollout(matchup, make_quant_row(row), port_base + i * 4 + 2, n)
    except Exception as exc:
        # Log-and-continue on a flaky emulator boot (EnetDisconnected/timeout during menu
        # nav), matching hal/eval/harness's contract so one bad Dolphin never aborts the batch.
        logger.warning(f"replay {i} ({e.path}) rollout failed: {exc!r}; skipping")
        continue
    m = min(len(raw_a), len(raw_b), len(quant), n)
    floor = diff(raw_a, raw_b, max_frames=m)
    qd = diff(raw_a, quant, max_frames=m)
    ports = sorted(raw_a.post)
    results.append(
        ReplayResult(
            stage=e.stage,
            chars=chars,
            n=m,
            floor_frame=None if floor.passed else floor.divergences[0].first_diff_frame,
            quant_first_any=None if qd.passed else qd.divergences[0].first_diff_frame,
            quant_first_action=first_action_divergence(raw_a, quant, m),
            pos_gap=position_gap(raw_a, quant, m),
            raw_counts={p: detect_techniques(raw_a, p, m) for p in ports},
            quant_counts={p: detect_techniques(quant, p, m) for p in ports},
            raw_counts_early={p: detect_techniques(raw_a, p, min(N_EARLY, m)) for p in ports},
            quant_counts_early={p: detect_techniques(quant, p, min(N_EARLY, m)) for p in ports},
            action_match={p: action_match_fraction(raw_a, quant, p, m) for p in ports},
        )
    )
    logger.info(
        f"[{i + 1}/{K_ROLLOUT}] {chars} stage {e.stage} m={m} | "
        f"floor={'bitexact' if floor.passed else floor.divergences[0].first_diff_frame} "
        f"quant_first_any={results[-1].quant_first_any} quant_first_action={results[-1].quant_first_action}"
    )
logger.info(f"rollout pass: {len(results)} replays in {time.monotonic() - t_start:.0f}s")


# %%
# --- verdict -----------------------------------------------------------------------------
TECH_FIELDS = ("jumpsquats", "airdodges", "wavedashes", "shorthops", "fullhops")


def _sum_counts(subset: list[ReplayResult], getter) -> TechniqueCounts:  # noqa: ANN001
    tot = dict.fromkeys(TECH_FIELDS, 0)
    for r in subset:
        for p in r.raw_counts:
            c = getter(r, p)
            for f in TECH_FIELDS:
                tot[f] += getattr(c, f)
    return TechniqueCounts(**tot)


def _print_tech_table(subset: list[ReplayResult], raw_getter, quant_getter, label: str) -> None:  # noqa: ANN001
    raw_tot = _sum_counts(subset, raw_getter)
    q_tot = _sum_counts(subset, quant_getter)
    print(f"{'technique':<12} {'RAW':>6} {'QUANT':>6} {'ratio':>7}   ({label})")
    for f in TECH_FIELDS:
        rv, qv = getattr(raw_tot, f), getattr(q_tot, f)
        print(f"{f:<12} {rv:>6} {qv:>6} {qv / max(rv, 1):>7.2f}")


print("\n==================== DETERMINISM AUDIT (raw vs raw) ====================")
clean = [r for r in results if r.floor_frame is None]
early_clean = [r for r in results if r.floor_frame is None or r.floor_frame >= N_EARLY]
print(
    f"{len(clean)}/{len(results)} replays bit-exact raw-vs-raw over the full window; "
    f"{len(early_clean)}/{len(results)} bit-exact through the first {N_EARLY} frames"
)
for r in results:
    if r.floor_frame is not None:
        print(f"  NON-DETERMINISTIC: stage {r.stage} {r.chars} raw-vs-raw diverged at frame {r.floor_frame}")
if results:
    floors = [r.floor_frame for r in results if r.floor_frame is not None]
    onset_ceiling = min(floors) if floors else None
    onsets_clean = all(
        r.quant_first_any is None or r.floor_frame is None or r.quant_first_any < r.floor_frame for r in results
    )
    print(
        f"earliest floor break: {onset_ceiling}; every quant onset precedes its replay's floor break: {onsets_clean} "
        f"(=> onset stats below are pure discretization even where the long window is RNG-contaminated)"
    )

print("\n==================== QUANT DIVERGENCE ONSET (raw vs quant) ====================")
fa = [r.quant_first_any for r in results if r.quant_first_any is not None]
fact = [r.quant_first_action for r in results if r.quant_first_action is not None]
print(
    f"first ANY-field divergence: median {int(np.median(fa))} frames, "
    f"min {min(fa)}, max {max(fa)}  ({len(fa)}/{len(results)} diverged within window)"
)
print(
    f"first ACTION-STATE divergence: median {int(np.median(fact))} frames "
    f"({int(np.median(fact)) / 60.0:.2f} s), min {min(fact)}, max {max(fact)}"
)
for cp in (60, 300, 600, 1200, N_FRAMES):
    gaps = [float(r.pos_gap[min(cp, r.n) - 1]) for r in results if r.n >= min(cp, N_FRAMES) // 4]
    print(f"  position gap (units) at frame {cp:>4}: median {np.median(gaps):7.2f}  max {np.max(gaps):8.2f}")

print("\n==================== TECHNIQUE SURVIVAL (per-rollout counts) ====================")
print(f"-- EARLY window (first {N_EARLY} frames; states still overlap => deltas are execution failures) --")
_print_tech_table(
    early_clean,
    lambda r, p: r.raw_counts_early[p],
    lambda r, p: r.quant_counts_early[p],
    f"{len(early_clean)} replays with a clean {N_EARLY}-frame floor",
)
print("\n-- FULL window, clean floor only (deltas = discretization-seeded butterfly drift) --")
_print_tech_table(clean, lambda r, p: r.raw_counts[p], lambda r, p: r.quant_counts[p], f"{len(clean)} clean replays")
print("\n-- FULL window, all replays (RNG-contaminated half included; upper bound on drift) --")
_print_tech_table(results, lambda r, p: r.raw_counts[p], lambda r, p: r.quant_counts[p], f"all {len(results)} replays")
am = [v for r in results for v in r.action_match.values()]
print(
    f"\nmean action-state match raw-vs-quant over full window: {np.mean(am):.3f} "
    f"(1.0 until butterfly divergence; buttons are lossless so counts drift downstream, not at input)"
)

print("\n==================== per-replay technique detail ====================")
for r in results:
    for p in sorted(r.raw_counts):
        rc, qc = r.raw_counts[p], r.quant_counts[p]
        print(
            f"stage {r.stage} {r.chars.get(p, '?'):>10} p{p}: "
            f"jump {rc.jumpsquats}->{qc.jumpsquats}  wave {rc.wavedashes}->{qc.wavedashes}  "
            f"airdodge {rc.airdodges}->{qc.airdodges}  short {rc.shorthops}->{qc.shorthops}  "
            f"full {rc.fullhops}->{qc.fullhops}  amatch {r.action_match[p]:.2f}"
        )
