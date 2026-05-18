# %% [markdown]
# Trigger logical vs physical: empirical test.
#
# Theory: peppi's `pre.triggers` (logical) fuses L+R into one smoothed float, while
# `pre.triggers_physical.{l,r}` preserves each shoulder's slp-native byte. If you
# read logical and tried to feed it back as both shoulders, you'd lose per-shoulder
# state and the value itself would be a lossy aggregate.
#
# We drive Dolphin with a scripted shoulder sequence (libmelee `press_shoulder`),
# let Slippi record an .slp, then re-read with peppi-py and print logical vs
# physical side by side.

# %%
import multiprocessing as _mp

# Python 3.14 defaults to "forkserver", which re-imports __main__ to spawn workers.
# libmelee's slippstream starts a multiprocessing.Process inside Console.connect(),
# so under forkserver the child re-runs the top-level run() and recurses until the
# socket is reset. Pin to "fork" (the pre-3.14 default) before anything imports mp.
_mp.set_start_method("fork", force=True)

import tempfile
from pathlib import Path

import melee
import numpy as np
import peppi_py

from hal.paths import EMULATOR_PATH
from hal.paths import ISO_PATH
from hal.sim.session import Matchup
from hal.sim.session import PlayerSetup
from hal.sim.session import Session

ISO = Path(ISO_PATH)
DOLPHIN = Path(EMULATOR_PATH)
HOLD_FRAMES = 4  # repeat each (L, R) target this many frames so the slp records cleanly
STEPS = 11  # 0.0, 0.1, ..., 1.0
LEAD_NEUTRAL_FRAMES = 30  # let the match settle before we touch shoulders
TAIL_NEUTRAL_FRAMES = 10

# %% [markdown]
# ## Build the (L, R) program
#
# - Phase A: L sweeps 0→1, R held at 0
# - Phase B: R sweeps 0→1, L held at 0
# - Phase C: crossing — L sweeps 0→1 while R sweeps 1→0
# - Phase D: both at 1.0 simultaneously
# - Phase E: both at 0.5 simultaneously (mid value, both engaged)


# %%
def build_program() -> list[tuple[float, float]]:
    levels = np.linspace(0.0, 1.0, STEPS)
    prog: list[tuple[float, float]] = []
    for _ in range(LEAD_NEUTRAL_FRAMES):
        prog.append((0.0, 0.0))
    # Phase A
    for v in levels:
        for _ in range(HOLD_FRAMES):
            prog.append((float(v), 0.0))
    # Phase B
    for v in levels:
        for _ in range(HOLD_FRAMES):
            prog.append((0.0, float(v)))
    # Phase C: crossing
    for v in levels:
        for _ in range(HOLD_FRAMES):
            prog.append((float(v), float(1.0 - v)))
    # Phase D: both full
    for _ in range(HOLD_FRAMES * 3):
        prog.append((1.0, 1.0))
    # Phase E: both mid
    for _ in range(HOLD_FRAMES * 3):
        prog.append((0.5, 0.5))
    for _ in range(TAIL_NEUTRAL_FRAMES):
        prog.append((0.0, 0.0))
    return prog


PROGRAM = build_program()
print(f"program: {len(PROGRAM)} frames")

# %% [markdown]
# ## Run the session
#
# We bypass `apply_inputs` and call `press_shoulder` directly so the wire amount
# we punch is exactly what we asked for (no extra normalization), and so we don't
# accidentally drive sticks/buttons. The libmelee `Controller` is constructed
# with `fix_analog_inputs=False`, but `press_shoulder` itself doesn't go through
# that path — it writes the value into the named pipe directly. (Compare
# `apply_inputs`, which wraps with `melee.controller.fix_analog_trigger`.)
#
# We use Fox vs Fox on Final Destination — irrelevant for the analog-shoulder
# input plumbing.


# %%
def run() -> Path:
    replay_dir = Path(tempfile.mkdtemp(prefix="hal_trigger_test_"))
    print(f"replay_dir: {replay_dir}")

    matchup = Matchup(
        stage=melee.Stage.FINAL_DESTINATION,
        players=(
            PlayerSetup(port=1, character=melee.Character.FOX),
            PlayerSetup(port=2, character=melee.Character.FOX, cpu_level=1),
        ),
    )
    with Session(
        iso_path=ISO,
        dolphin_path=DOLPHIN,
        tmp_home_directory=False,
        replay_dir=str(replay_dir),
    ) as s:
        s.start_match(matchup)
        p1 = s._controllers[1]
        p2 = s._controllers[2]
        for trig_l, trig_r in PROGRAM:
            # Drive p2 with pure neutral; we only care about p1.
            p2.press_shoulder(melee.enums.Button.BUTTON_L, 0.0)
            p2.press_shoulder(melee.enums.Button.BUTTON_R, 0.0)
            p1.press_shoulder(melee.enums.Button.BUTTON_L, trig_l)
            p1.press_shoulder(melee.enums.Button.BUTTON_R, trig_r)
            # Skip apply_inputs entirely; step the console directly.
            _, in_game = s.step({})
            if not in_game:
                print("match ended early")
                break

    # Slippi writes Game_YYYYMMDDTHHMMSS.slp into <replay_dir>/Slippi/
    candidates = sorted(replay_dir.rglob("*.slp"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no .slp written under {replay_dir}")
    slp_path = candidates[-1]
    print(f"recorded: {slp_path}")
    return slp_path


SLP_PATH = run()


# %% [markdown]
# ## Read back with peppi


# %%
def read_back(slp_path: Path) -> dict[str, np.ndarray]:
    game = peppi_py.read_slippi(str(slp_path))
    pre = game.frames.ports[0].leader.pre  # port 1, peppi_idx 0
    logical = np.array(pre.triggers.to_pylist(), dtype=np.float32)
    phys_l = np.array(pre.triggers_physical.l.to_pylist(), dtype=np.float32)
    phys_r = np.array(pre.triggers_physical.r.to_pylist(), dtype=np.float32)
    frame_ids = np.array(game.frames.id.to_pylist(), dtype=np.int32)
    return {"frame_id": frame_ids, "logical": logical, "phys_l": phys_l, "phys_r": phys_r}


DATA = read_back(SLP_PATH)
print(f"frames in slp: {len(DATA['frame_id'])}  first id: {DATA['frame_id'][0]}")

# %% [markdown]
# ## Align program → slp frames
#
# Our program starts punching at the first in-game frame. peppi's frame ids
# include pre-game (negative) entries; the first in-game frame is id `-123`.


# %%
def in_game_slice(frame_ids: np.ndarray) -> slice:
    idx = int(np.argmax(frame_ids >= -123))
    return slice(idx, None)


SL = in_game_slice(DATA["frame_id"])
slp_logical = DATA["logical"][SL]
slp_phys_l = DATA["phys_l"][SL]
slp_phys_r = DATA["phys_r"][SL]
n = min(len(PROGRAM), len(slp_logical))
print(f"comparing {n} frames")

# %% [markdown]
# ## Compare: what we wrote vs. what peppi sees


# %%
def fmt_phase(name: str, indices: range) -> None:
    print(f"\n=== {name} ===")
    print(f"{'fr':>5} {'wrote_L':>8} {'wrote_R':>8} | {'phys_L':>8} {'phys_R':>8} | {'logical':>8}")
    for i in indices:
        if i >= n:
            break
        wl, wr = PROGRAM[i]
        pl, pr, lg = float(slp_phys_l[i]), float(slp_phys_r[i]), float(slp_logical[i])
        print(f"{i:>5} {wl:>8.3f} {wr:>8.3f} | {pl:>8.3f} {pr:>8.3f} | {lg:>8.3f}")


# Phase boundaries
a_start = LEAD_NEUTRAL_FRAMES
a_end = a_start + STEPS * HOLD_FRAMES
b_end = a_end + STEPS * HOLD_FRAMES
c_end = b_end + STEPS * HOLD_FRAMES
d_end = c_end + HOLD_FRAMES * 3
e_end = d_end + HOLD_FRAMES * 3


# Show the last (steady-state) frame of each level. The first 1–2 frames after
# each transition are settling; pick the last frame of each (HOLD_FRAMES) chunk.
def steady_indices(start: int, n_steps: int) -> range:
    return range(start + HOLD_FRAMES - 1, start + n_steps * HOLD_FRAMES, HOLD_FRAMES)


fmt_phase("Phase A — L sweep, R = 0", steady_indices(a_start, STEPS))
fmt_phase("Phase B — R sweep, L = 0", steady_indices(a_end, STEPS))
fmt_phase("Phase C — L↑ / R↓ crossing", steady_indices(b_end, STEPS))
fmt_phase("Phase D — both at 1.0", range(c_end, d_end))
fmt_phase("Phase E — both at 0.5", range(d_end, e_end))

# %% [markdown]
# ## What to look for
#
# - **Phase A / B**: only the touched shoulder shows a non-zero physical value.
#   `logical` should track the active shoulder. This confirms physical is
#   per-shoulder and is the slp-native ground truth.
# - **Phase C (crossing)**: physical.l and physical.r both vary independently.
#   `logical` collapses to a single scalar — observing what it equals at the
#   crossover (L≈R≈0.5) tells you peppi's fusion rule (likely `max(l, r)`).
# - **Phase D (both 1.0)**: physical.l == physical.r == ~1.0; logical == ~1.0.
#   Indistinguishable from "only one shoulder fully pressed" via logical alone
#   — that's the information loss.
# - **Phase E (both 0.5)**: physical.l == physical.r == ~0.5; logical likely
#   == ~0.5. Again indistinguishable from "only one shoulder at 0.5".
#
# Conclusion: if logical is what we feed back through the wire, Phases D/E
# replay correctly only because we splat one scalar to both shoulders — but
# we've lost the ability to express "only L pressed" vs "only R pressed" vs
# "both pressed". For asymmetric inputs, the round trip is lossy.
