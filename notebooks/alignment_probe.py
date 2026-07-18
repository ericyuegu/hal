# %%
"""Empirically settle the intra-row (post, controller) frame alignment.

Two comments in the repo make contradictory claims about how a stored MDS row
pairs its post-frame gamestate with its controller channels:

* ``hal/data/schema.py`` (``_controller_columns``): "Action[t] -> state[t+1]"
  — i.e. the controller stored at row t is the input that produces the state at
  row t+1.
* ``hal/training/closed_loop.py`` (``_live_batch_from_rolling``): pairs
  ``(post_i, pre_i)`` where ``pre_i`` is "the gamestate it produced" — i.e. the
  controller at row i is the input active during frame i that produced post_i.

Both cannot be true. If the extract's intra-row pairing does not match the
deploy-side rolling-buffer pairing, the model's ego-history input is shifted one
frame between train and deploy (the "jump held one frame too long" symptom).

The probe drives a real headless Dolphin as Fox, presses Y (jump) for EXACTLY
one observed frame at a known observation index, records the .slp, runs the
repo's extractor, and reads back:

  k       the frame counter observed at the instant Y was injected
  r_btn   the MDS row where the stored ego ``button_y`` == 1
  r_knee  the MDS row where the ego action-state first becomes KNEE_BEND (24)

From (k, r_btn, r_knee) we state the empirical recording latency and the
in-slp input->action-state effect delay, then walk the train vs deploy indices
concretely to reach a CONSISTENT / MISALIGNED verdict.
"""

import multiprocessing as mp

# libmelee's slippstream client spawns a child via mp.Process; py3.14 defaults to
# "forkserver", which re-imports the worker module and is flaky alongside heavy
# imports. Match tests/notebooks and force plain fork before anything spawns.
if mp.get_start_method(allow_none=True) != "fork":
    mp.set_start_method("fork", force=True)

import tempfile
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import melee
import numpy as np

from hal.data.extract import extract_replay
from hal.paths import EMULATOR_PATH
from hal.paths import ISO_PATH
from hal.sim.inputs import ControllerInputs
from hal.sim.inputs import ControllerInputsValue
from hal.sim.loop import drive
from hal.sim.session import Matchup
from hal.sim.session import PlayerSetup
from hal.sim.session import Session
from hal.sim.sources import ControllerSource
from hal.sim.sources import ScriptedControllerSource
from hal.wire import BUTTON_BITS

KNEE_BEND: int = int(melee.Action.KNEE_BEND.value)  # 24: Fox 3-frame jumpsquat
Y_BIT: int = BUTTON_BITS["y"]
_NEUTRAL: ControllerInputs = ControllerInputsValue(
    main_x=0.0, main_y=0.0, c_x=0.0, c_y=0.0, trigger_l=0.0, trigger_r=0.0, buttons=0
)


# %%
@dataclass(slots=True)
class JumpProbeSource:
    """Hold neutral; press Y for exactly one frame at each drive-iteration in
    ``inject_at``. Logs ``(drive_iter, observed_frame_id)`` at each injection so
    we know which gamestate the recorder saw when it chose to jump.

    ``drive`` calls ``src(t, last)`` with ``last == captured[t]`` — the most
    recent observed gamestate — so ``last["id"]`` is exactly the frame counter
    the policy would condition on before submitting this frame's input.
    """

    inject_at: frozenset[int]
    observed: list[tuple[int, int]] = field(default_factory=list)

    def __call__(self, frame_index: int, last_gamestate: dict | None) -> ControllerInputs | None:
        if frame_index in self.inject_at:
            observed_id = -10_000 if last_gamestate is None else int(last_gamestate["id"])
            self.observed.append((frame_index, observed_id))
            return ControllerInputsValue(
                main_x=0.0, main_y=0.0, c_x=0.0, c_y=0.0, trigger_l=0.0, trigger_r=0.0, buttons=Y_BIT
            )
        return _NEUTRAL


def record_probe(inject_at: tuple[int, ...], max_frames: int, slippi_port: int) -> tuple[Path, list[tuple[int, int]]]:
    """Boot headless Dolphin, Fox(port1, injects Y) vs neutral Fox(port2) on FD,
    record a .slp, and return (slp_path, injection log)."""
    matchup = Matchup(
        stage=melee.Stage.FINAL_DESTINATION,
        players=(
            PlayerSetup(port=1, character=melee.Character.FOX, costume=0),
            PlayerSetup(port=2, character=melee.Character.FOX, costume=1),
        ),
    )
    probe = JumpProbeSource(inject_at=frozenset(inject_at))
    sources: dict[int, ControllerSource] = {
        1: probe,
        2: ScriptedControllerSource(sequence=[]),  # neutral throughout
    }
    replay_dir = Path(tempfile.mkdtemp(prefix="hal_align_"))
    with Session(
        iso_path=ISO_PATH,
        dolphin_path=EMULATOR_PATH,
        slippi_port=slippi_port,
        tmp_home_directory=False,
        replay_dir=str(replay_dir),
    ) as s:
        drive(s, matchup, sources, max_frames=max_frames)
    slps = sorted(replay_dir.rglob("*.slp"))
    if not slps:
        raise RuntimeError(f"Slippi wrote no .slp under {replay_dir}")
    return slps[-1], probe.observed


# %%
# A jumpsquat starts within a couple frames of the Y poll if the character is
# actionable; a wider search would spuriously match a *later* press. Presses made
# during the pre-"GO!" entry lockout can't jump at all -> reported as None.
_KNEE_WINDOW: int = 4


@dataclass(frozen=True, slots=True)
class PressResult:
    drive_iter: int
    k: int  # observed frame counter at injection
    k_row: int  # MDS row whose frame == k
    r_btn: int  # MDS row where ego button_y == 1
    btn_minus_k: int  # r_btn - k_row  (recording latency)
    r_knee: int | None  # MDS row where ego action first == KNEE_BEND within the window (None if not actionable)
    knee_minus_btn: int | None  # r_knee - r_btn (in-slp input->action effect delay)


def analyze(
    slp_path: Path, observed: list[tuple[int, int]]
) -> tuple[list[PressResult], str, np.ndarray, np.ndarray, np.ndarray]:
    rows = extract_replay(str(slp_path))
    if rows is None:
        raise RuntimeError(f"extract_replay returned None for {slp_path}")
    frame = rows["frame"]
    # The injecting port is the lowest occupied libmelee port -> MDS prefix p1.
    prefix = "p1" if int(rows["p1_button_y"].sum()) >= int(rows["p2_button_y"].sum()) else "p2"
    btn = rows[f"{prefix}_button_y"]
    act = rows[f"{prefix}_action"]

    frame_to_row = {int(f): i for i, f in enumerate(frame)}
    results: list[PressResult] = []
    for drive_iter, k in observed:
        if k not in frame_to_row or (k + 1) not in frame_to_row:
            raise RuntimeError(f"observed frame {k} (or {k + 1}) not in extracted rows [{frame[0]}..{frame[-1]}]")
        k_row = frame_to_row[k]
        r_btn = frame_to_row[k + 1]
        if btn[r_btn] != 1:
            # Loud, not silent, if the +1 recording-latency assumption were wrong.
            raise RuntimeError(
                f"expected button_y==1 at row {r_btn} (frame {k + 1}); "
                f"button_y rows are {np.where(btn == 1)[0].tolist()}"
            )
        knee_after = np.where(act[r_btn : r_btn + _KNEE_WINDOW] == KNEE_BEND)[0]
        r_knee = int(r_btn + knee_after[0]) if knee_after.size else None
        results.append(
            PressResult(
                drive_iter=drive_iter,
                k=k,
                k_row=k_row,
                r_btn=r_btn,
                btn_minus_k=r_btn - k_row,
                r_knee=r_knee,
                knee_minus_btn=None if r_knee is None else r_knee - r_btn,
            )
        )
    return results, prefix, btn, act, frame


def dump_context(prefix: str, btn: np.ndarray, act: np.ndarray, frame: np.ndarray, r_btn: int) -> None:
    lo, hi = max(0, r_btn - 3), min(len(frame), r_btn + 6)
    print(f"    row   frame  {prefix}_button_y  {prefix}_action")
    for i in range(lo, hi):
        tag = " <- button_y" if btn[i] == 1 else (" <- KNEE_BEND" if act[i] == KNEE_BEND else "")
        print(f"    {i:4d}  {int(frame[i]):5d}       {int(btn[i]):2d}        {int(act[i]):5d}{tag}")


# %%
# Two independent boots, two presses each -> four data points. Injects are all
# post-"GO!" (character actionable, resting STANDING) and >50 frames apart so each
# short-hop lands before the next jump.
RUNS = (
    dict(inject_at=(150, 210), max_frames=270, slippi_port=51455),
    dict(inject_at=(180, 240), max_frames=300, slippi_port=51455),
)

all_results: list[PressResult] = []
for cfg in RUNS:
    slp_path, observed = record_probe(**cfg)
    print(f"\n=== run inject_at={cfg['inject_at']} -> {slp_path.name} ; observed={observed} ===")
    results, prefix, btn, act, frame = analyze(slp_path, observed)
    print(f"ego prefix = {prefix}; button_y rows = {np.where(btn == 1)[0].tolist()}")
    for r in results:
        print(
            f"  k={r.k} (row {r.k_row}) | r_btn={r.r_btn} (r_btn-k_row={r.btn_minus_k}) | "
            f"r_knee={r.r_knee} (r_knee-r_btn={r.knee_minus_btn})"
        )
        dump_context(prefix, btn, act, frame, r.r_btn)
    all_results.extend(results)


# %%
# --- verdict ---------------------------------------------------------------
btn_deltas = sorted({r.btn_minus_k for r in all_results})
knee_deltas = sorted({r.knee_minus_btn for r in all_results if r.knee_minus_btn is not None})
n_knee = sum(r.knee_minus_btn is not None for r in all_results)
print("\n================ SUMMARY ================")
print(f"presses measured   : {len(all_results)}  (actionable jumps: {n_knee})")
print(f"r_btn  - k_row     : {btn_deltas}   (recording latency; +1 == deploy pairs post_i with prev-step action)")
print(f"r_knee - r_btn     : {knee_deltas}   (in-slp input -> action-state effect delay)")

# Train pairs MDS row i = (post_i, pre_i) from the same slp frame (extract._extract_player
# indexes leader.pre and leader.post by the same keep_idx). Deploy (_live_batch_from_rolling)
# pairs post_i with ego a_{i-1}: the action chosen after observing frame i-1, which produced
# frame i. They match iff the stored pre_i equals a_{i-1} -- i.e. the input submitted after
# observing frame k lands in stored controller row k+1 (recording latency == +1).
consistent = btn_deltas == [1]
print("\nVERDICT:", "CONSISTENT" if consistent else "MISALIGNED")
if consistent:
    print(
        "  Input submitted after observing frame k appears in stored controller row r_btn = k+1.\n"
        "  So stored pre_i is the input active during frame i (== deploy's a_{i-1}), and\n"
        "  train row i (post_i, pre_i) == deploy position i (post_i, a_{i-1}). No train/deploy shift.\n"
        "  The closed_loop.py comment ((post_i, pre_i), 'the gamestate it produced') is CORRECT.\n"
        "  schema.py:67 'Action[t] -> state[t+1]' is the WRONG comment: pre_t is the input that\n"
        "  produced state[t], not state[t+1]."
    )
else:
    print(f"  recording latency r_btn-k_row = {btn_deltas} (expected [1] for consistency) -- investigate.")
