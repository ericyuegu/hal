"""Multishine overfit sanity-check, end to end.

Drives libmelee's ``techskill.multishine`` (ported to read the canonical
gamestate dict instead of a libmelee ``GameState``) on BOTH Fox ports, records
a ~30s .slp, pushes it through the offline pipeline (build_index -> filter ->
materialize) into an MDS dataset, overfits ``experiments/001`` on it, and then
confirms in a real-Dolphin closed-loop run that the trained model reproduces
multishine frame-for-frame.

The multishine logic is a pure function of one port's (action, action_frame,
on_ground) — it ignores the opponent — so it is the cleanest possible target
for an overfit: a deterministic, short-period input pattern. We reuse the SAME
pure function as (a) the recording ControllerSource and (b) the per-frame
reference in the closed-loop frame-diff, so there is one source of truth for
"what multishine does".

Run as VSCode ``# %%`` cells from the repo root.
"""

# %%
import importlib.util
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import melee
import numpy as np
from loguru import logger

from hal.eval.harness import SessionConfig
from hal.fixtures import DOLPHIN_EXIAI
from hal.fixtures import ISO
from hal.fixtures import ensure
from hal.paths import EMULATOR_PATH
from hal.scripts.build_index import build_index
from hal.scripts.filter import filter_index
from hal.scripts.materialize import process_replays
from hal.sim.inputs import ControllerInputsValue
from hal.sim.loop import drive
from hal.sim.session import Matchup
from hal.sim.session import PlayerSetup
from hal.sim.session import Session
from hal.sim.sources import ControllerSource
from hal.sim.vec import Slot
from hal.wire import BUTTON_BITS

REPO = Path(__file__).resolve().parents[1]
SCRATCH = REPO / "data" / "scratch" / "multishine"
RECORDING_DIR = SCRATCH / "recording"
MDS_DIR = SCRATCH / "mds"
INDEX_PATH = SCRATCH / "index.jsonl"
PATHS_PATH = SCRATCH / "paths.txt"
EXP001 = REPO / "experiments" / "001_flow_matching_rtc_baseline.py"

# ~30s @ 60Hz of clean multishine training data, then both Fox walk off-stage
# until a player loses all 4 stocks. Slippi only writes the .slp footer on an
# in-game GAME_END, so an endless multishine that we just terminate yields a
# truncated, unparseable file — the match must actually end. END_CAP bounds the
# walk-off phase; drive() stops as soon as the game ends.
CLEAN_FRAMES = 1800
END_CAP = 1600
STAGE = melee.Stage.FINAL_DESTINATION


# %%
# libmelee 1-indexes action frames by adding +1 to the raw slp ``state_age``
# for the per-character "zero-indexed" actions (Console.__fixframeindexing).
# multishine's frame checks (KNEE_BEND==3, shine>=4) are calibrated to that
# fixed value, so we replicate it off the same actiondata.csv libmelee ships.
def _load_zero_indices() -> dict[int, set[int]]:
    import csv

    path = Path(melee.console.__file__).parent / "actiondata.csv"
    out: dict[int, set[int]] = {}
    for row in csv.DictReader(path.open()):
        if row["zeroindex"] == "True":
            out.setdefault(int(row["character"]), set()).add(int(row["action"]))
    return out


_ZERO_INDICES = _load_zero_indices()


def fixed_action_frame(post: dict) -> int:
    raw = int(post["state_age"]) if post.get("state_age") is not None else 0
    return raw + 1 if post["action"] in _ZERO_INDICES.get(post["character"], ()) else raw


# %%
# ---- multishine, ported to the canonical post dict --------------------------
# Faithful port of melee.techskill.multishine. The original mutates a libmelee
# Controller via press_button / tilt_analog / release_all; here each branch
# instead returns the FULL controller state for the frame. That is equivalent
# because libmelee's controller is sticky and the original calls release_all()
# on every non-matched frame — so the live byte after any branch is exactly the
# explicit value object we return (apply_inputs writes every button + stick each
# frame). libmelee's ``tilt_analog(MAIN, .5, 0)`` is center-x / full-down, i.e.
# logical (main_x=0, main_y=-1): shine = stick-down + B.

_A = melee.enums.Action

_SHINE = ControllerInputsValue(
    main_x=0.0, main_y=-1.0, c_x=0.0, c_y=0.0, trigger_l=0.0, trigger_r=0.0, buttons=BUTTON_BITS["b"]
)
_JUMP = ControllerInputsValue(
    main_x=0.0, main_y=0.0, c_x=0.0, c_y=0.0, trigger_l=0.0, trigger_r=0.0, buttons=BUTTON_BITS["y"]
)
_NEUTRAL = ControllerInputsValue(main_x=0.0, main_y=0.0, c_x=0.0, c_y=0.0, trigger_l=0.0, trigger_r=0.0, buttons=0)


def multishine_inputs(action: int, action_frame: int, on_ground: bool) -> ControllerInputsValue:
    """Pure multishine: (action, action_frame, on_ground) -> one frame of inputs."""
    if action == _A.STANDING.value:
        return _SHINE
    if action == _A.KNEE_BEND.value:
        # Shine on frame 3 of jumpsquat, else nothing (release_all).
        return _SHINE if action_frame == 3 else _NEUTRAL
    shine_start = action in (_A.DOWN_B_STUN.value, _A.DOWN_B_GROUND_START.value)
    if shine_start and action_frame >= 4 and on_ground:
        return _JUMP
    if action == _A.DOWN_B_GROUND.value:
        return _JUMP
    return _NEUTRAL


def _post(gamestate: dict, port: int) -> dict:
    return gamestate["ports"][port]["leader"]["post"]


def multishine_from_canonical(gamestate: dict, port: int) -> ControllerInputsValue:
    """Read one port's post-frame state out of the canonical dict and run
    multishine. libmelee derives ``action_frame`` as ``int(post.state_age)``
    (console.py:1303); ``airborne`` is 0 on the ground, 1 in the air."""
    post = _post(gamestate, port)
    on_ground = post.get("airborne") == 0
    return multishine_inputs(post["action"], fixed_action_frame(post), on_ground)


def action_name(a: int) -> str:
    try:
        return _A(a).name
    except ValueError:
        return f"UNKNOWN({a})"


# Full-left, no recovery: walks off the side blast zone and dies — used only to
# bring the match to a clean GAME_END so Slippi finalizes the .slp.
_WALKOFF = ControllerInputsValue(main_x=-1.0, main_y=0.0, c_x=0.0, c_y=0.0, trigger_l=0.0, trigger_r=0.0, buttons=0)


@dataclass(slots=True)
class MultishineSource:
    """Multishine one Fox port for ``clean_frames``, then walk off to end the
    match (so the recorded .slp gets a finalized footer)."""

    port: int
    clean_frames: int = CLEAN_FRAMES

    def __call__(self, frame_index: int, last_gamestate: dict | None) -> ControllerInputsValue:
        if last_gamestate is None:
            return _NEUTRAL
        if frame_index >= self.clean_frames:
            return _WALKOFF
        return multishine_from_canonical(last_gamestate, self.port)


# %%
# ---- 1. record ~30s of double-Fox multishine to a .slp ----------------------
def record_multishine() -> Path:
    RECORDING_DIR.mkdir(parents=True, exist_ok=True)
    for slp in RECORDING_DIR.glob("*.slp"):
        slp.unlink()
    ensure(DOLPHIN_EXIAI)
    matchup = Matchup(
        stage=STAGE,
        players=(
            PlayerSetup(port=1, character=melee.Character.FOX),
            PlayerSetup(port=2, character=melee.Character.FOX),
        ),
    )
    sources: dict[int, ControllerSource] = {1: MultishineSource(1), 2: MultishineSource(2)}
    with Session(
        iso_path=ensure(ISO),
        dolphin_path=EMULATOR_PATH,
        use_exi_inputs=True,
        enable_ffw=True,
        emulation_speed=0.0,
        blocking_input=True,
        step_timeout_seconds=30.0,
        tmp_home_directory=True,
        replay_dir=str(RECORDING_DIR),
    ) as s:
        traj = drive(s, matchup, sources, max_frames=CLEAN_FRAMES + END_CAP)

    n_frames = len(traj)
    if n_frames >= CLEAN_FRAMES + END_CAP:
        raise RuntimeError(f"match never ended in {n_frames} frames — .slp will be truncated; lengthen END_CAP")
    # Sanity over the CLEAN region: the shine cycle must dominate.
    clean = traj.post[1]["action"][:CLEAN_FRAMES]
    cycle = {
        _A.STANDING.value,
        _A.KNEE_BEND.value,
        _A.DOWN_B_STUN.value,
        _A.DOWN_B_GROUND_START.value,
        _A.DOWN_B_GROUND.value,
    }
    n_shine = int(np.isin(clean, list(cycle)).sum())
    logger.info(f"recorded {n_frames} frames (ended in-game); {n_shine}/{len(clean)} clean frames in the shine cycle")
    uniq, counts = np.unique(clean, return_counts=True)
    for a, c in sorted(zip(uniq.tolist(), counts.tolist()), key=lambda kc: -kc[1])[:8]:
        logger.info(f"  {action_name(int(a)):22s} x{c}")
    if n_shine < 0.5 * len(clean):
        raise RuntimeError("multishine did not engage — <50% of clean frames in the shine cycle")

    slps = sorted(RECORDING_DIR.glob("*.slp"))
    if not slps:
        raise RuntimeError(f"no .slp written to {RECORDING_DIR}")
    # Fail loud if Slippi didn't finalize the footer (peppi must be able to read it).
    import peppi_py

    peppi_py.read_slippi(str(slps[-1]))
    logger.info(f"wrote + verified {slps[-1]}")
    return slps[-1]


# slp_path = record_multishine()


# %%
# ---- 2. offline pipeline: build_index -> filter -> materialize --------------
def build_dataset(n_copies: int = 64) -> Path:
    # whole-file selection: build_index over the slp dir, keep-all filter (no
    # predicates), materialize everything into the train split so stats.json
    # (computed over train only) is populated.
    #
    # WindowSampler yields ONE random window per replay per epoch, so a single
    # replay starves batches down to ~num_workers samples. We replicate the one
    # finalized clip into n_copies distinct-named files: replay_uuid hashes the
    # PATH (not content), so each copy is its own train row, and the loader can
    # fill batch_size-wide batches of random windows over the same clip.
    import shutil

    src_slp = sorted(RECORDING_DIR.glob("*.slp"))[-1]
    slp_dir = SCRATCH / "dataset_slps"
    if slp_dir.exists():
        shutil.rmtree(slp_dir)
    slp_dir.mkdir(parents=True)
    for i in range(n_copies):
        shutil.copy(src_slp, slp_dir / f"clip_{i:03d}.slp")

    if MDS_DIR.exists():
        shutil.rmtree(MDS_DIR)
    INDEX_PATH.unlink(missing_ok=True)
    build_index(output=INDEX_PATH, root=slp_dir, with_stats=False, compute_sha1=False, workers=2)
    filter_index(index=INDEX_PATH, output=PATHS_PATH)  # no predicates -> keep all
    process_replays(
        paths_file=PATHS_PATH,
        index=INDEX_PATH,
        output=str(MDS_DIR),
        train_split=1.0,
        val_split=0.0,
        workers=2,
    )
    _trim_train_mds(CLEAN_FRAMES)
    logger.info(f"MDS at {MDS_DIR} ({n_copies} rows, trimmed to {CLEAN_FRAMES} clean frames)")
    return MDS_DIR


def _trim_train_mds(n_frames: int) -> None:
    """Re-pack the train shard with each row truncated to the clean-multishine
    region, dropping the walk-off frames (recorded only to finalize the .slp).
    Walk-off taught a 'multishine -> walk-off' transition the closed-loop model
    drifts into on any stochastic glitch (Fox parks at the ledge); excluding
    those frames leaves multishine the only learned behavior. stats.json
    (position/etc.) stays valid and is reused."""
    import shutil

    from streaming import MDSWriter
    from streaming import StreamingDataset

    from hal.data.schema import MDS_DTYPE_STR_BY_COLUMN

    train_dir = MDS_DIR / "train"
    ds = StreamingDataset(local=str(train_dir), shuffle=False, batch_size=1)
    rows = [{k: np.asarray(v)[:n_frames] for k, v in ds[i].items()} for i in range(len(ds))]
    shutil.rmtree(train_dir)
    with MDSWriter(out=str(train_dir), columns=MDS_DTYPE_STR_BY_COLUMN) as w:
        for row in rows:
            w.write(row)
    logger.info(f"trimmed {len(rows)} train rows -> {n_frames} frames each (walk-off dropped)")


# build_dataset()


# %%
# ---- 3. overfit experiment 001 (open-loop n_lat=0) --------------------------
def overfit(
    *,
    max_steps: int = 3000,
    batch_size: int = 32,
    l_ctx: int = 256,
    l_chunk: int = 16,
    d_model: int = 256,
    n_layers: int = 6,
    comment: str = "multishine-overfit",
) -> Path:
    """Run 001 as a subprocess (keeps the kernel clean of its Dolphin tail-eval
    + wandb). Returns the run dir parsed from its stdout."""
    env = {**os.environ, "WANDB_MODE": "offline"}
    cmd = [
        sys.executable,
        str(EXP001),
        "--cfg.data-root",
        str(MDS_DIR),
        "--cfg.val-split",
        "train",
        "--cfg.latency-frames",
        "0",
        "--cfg.max-steps",
        str(max_steps),
        "--cfg.batch-size",
        str(batch_size),
        "--cfg.L-ctx",
        str(l_ctx),
        "--cfg.L-chunk",
        str(l_chunk),
        "--cfg.d-model",
        str(d_model),
        "--cfg.n-layers",
        str(n_layers),
        "--cfg.eval-every",
        "0",
        "--cfg.eval-max-frames",
        "300",
        "--cfg.val-every",
        str(max(1, max_steps // 6)),
        "--cfg.val-n-batches",
        "4",
        "--cfg.ckpt-every",
        str(max(1, max_steps // 3)),
        "--cfg.no-push-to-r2",
        "--cfg.num-workers",
        "2",
        "--comment",
        comment,
    ]
    logger.info("training: " + " ".join(cmd))
    run_dir: Path | None = None
    proc = subprocess.Popen(cmd, cwd=str(REPO), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        if "writing checkpoints to" in line:
            run_dir = Path(line.split("writing checkpoints to", 1)[1].strip())
    if proc.wait() != 0:
        raise RuntimeError(f"training exited {proc.returncode}")
    if run_dir is None:
        raise RuntimeError("could not parse run dir from training stdout")
    ckpt = (REPO / run_dir / "final.pt") if not run_dir.is_absolute() else (run_dir / "final.pt")
    logger.info(f"trained checkpoint: {ckpt}")
    return ckpt


# ckpt_path = overfit()


# %%
# ---- 4. closed-loop verification: model vs reference multishine, frame by frame
def _load_exp001():
    spec = importlib.util.spec_from_file_location("exp001", EXP001)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@dataclass(slots=True)
class _Rec:
    frame: int
    port: int
    action: int
    ref: ControllerInputsValue
    model: ControllerInputsValue


def _same_input(model: ControllerInputsValue, ref: ControllerInputsValue) -> bool:
    """Behavioral equality: identical buttons, and stick bucketed to the
    cardinal multishine targets (down vs neutral). Stick floats differ by the
    int8 wire quantization, so an exact compare would spuriously fail."""
    if model.buttons != ref.buttons:
        return False
    if abs(model.main_x - ref.main_x) > 0.25:
        return False
    return (model.main_y < -0.5) == (ref.main_y < -0.5)


def verify(ckpt_path: Path, *, measure_frames: int = 1800, n_flow_steps: int | None = None) -> float:
    """Drive the model in closed loop FROM FRAME 0 (cold start, no priming) and
    frame-diff its action against reference multishine on the same observed state
    for ``measure_frames`` (>=30s) in real Dolphin.

    No priming: the model's context masking (``ctx_pad``) hides the still-filling
    rolling buffer, so the policy is in-distribution from the very first frame.
    This is the real test that masking — not reference-data priming — carries the
    cold start."""
    exp = _load_exp001()
    import torch

    from hal.training.stats import load_consolidated_stats

    state = torch.load(ckpt_path, map_location=exp.DEVICE, weights_only=False)
    cfg = exp.TrainConfig(**state["cfg"])
    if n_flow_steps is not None:
        cfg.n_flow_steps = n_flow_steps  # inference-only; more steps = crisper integration
    model = exp.FlowMatchingPolicy(cfg).to(exp.DEVICE)
    model.load_state_dict(state["model"])
    model.eval()
    stats = load_consolidated_stats(Path(cfg.data_root) / "stats.json")
    policy = exp.make_policy(model, stats, cfg)
    ports = (1, 2)

    replay_dir = SCRATCH / "verify_replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    for slp in replay_dir.glob("*.slp"):
        slp.unlink()

    matchup = Matchup(
        stage=STAGE,
        players=(
            PlayerSetup(port=1, character=melee.Character.FOX),
            PlayerSetup(port=2, character=melee.Character.FOX),
        ),
    )
    cfg_s = SessionConfig(iso_path=ensure(ISO), dolphin_path=EMULATOR_PATH)
    session = Session(
        iso_path=cfg_s.iso_path,
        dolphin_path=cfg_s.dolphin_path,
        use_exi_inputs=cfg_s.use_exi_inputs,
        enable_ffw=cfg_s.enable_ffw,
        emulation_speed=cfg_s.emulation_speed,
        blocking_input=cfg_s.blocking_input,
        step_timeout_seconds=cfg_s.step_timeout_seconds,
        tmp_home_directory=True,
        replay_dir=str(replay_dir),
    )
    ensure(DOLPHIN_EXIAI)
    recs: list[_Rec] = []
    with session as s:
        obs = s.start_match(matchup)
        # The model drives from frame 0; frame-diff its action vs reference
        # multishine on the same observed state. The policy fills its own rolling
        # buffer and masks the empty prefix — no priming, no warm-up.
        for t in range(measure_frames):
            out = policy(t, {Slot(0, p): obs for p in ports})
            for p in ports:
                ref = multishine_from_canonical(obs, p)
                recs.append(_Rec(t, p, _post(obs, p)["action"], ref, out[Slot(0, p)]))
            obs, in_game = s.step({p: out[Slot(0, p)] for p in ports})
            if not in_game:
                logger.warning(f"match ended at frame {t}/{measure_frames}")
                break

    p1 = [r for r in recs if r.port == 1]
    if not p1:
        raise RuntimeError("no handover frames captured")
    matches = [r for r in p1 if _same_input(r.model, r.ref)]
    rate = len(matches) / len(p1)
    first_bad = next((r for r in p1 if not _same_input(r.model, r.ref)), None)
    logger.info(f"frame-diff: {len(matches)}/{len(p1)} match ({rate:.4%}) over {len(p1)} handover frames")
    # Diagnostics: why do frames mismatch, and is the model even cycling cleanly?
    btn_bad = sum(1 for r in p1 if r.model.buttons != r.ref.buttons)
    mx_bad = sum(1 for r in p1 if r.model.buttons == r.ref.buttons and abs(r.model.main_x - r.ref.main_x) > 0.25)
    my_bad = sum(
        1
        for r in p1
        if r.model.buttons == r.ref.buttons
        and abs(r.model.main_x - r.ref.main_x) <= 0.25
        and (r.model.main_y < -0.5) != (r.ref.main_y < -0.5)
    )
    logger.info(f"  mismatch by cause: buttons={btn_bad} main_x={mx_bad} main_y_bucket={my_bad}")
    uniq, counts = np.unique([r.action for r in p1], return_counts=True)
    hist = ", ".join(
        f"{action_name(int(a))}:{c}" for a, c in sorted(zip(uniq.tolist(), counts.tolist()), key=lambda kc: -kc[1])[:6]
    )
    logger.info(f"  model closed-loop action states: {hist}")
    half = len(p1) // 2
    early = sum(1 for r in p1[:half] if _same_input(r.model, r.ref)) / max(1, half)
    late = sum(1 for r in p1[half:] if _same_input(r.model, r.ref)) / max(1, len(p1) - half)
    logger.info(f"  match rate first-half={early:.2%} second-half={late:.2%}")
    if first_bad is not None:
        logger.warning(
            f"first mismatch @ handover frame {first_bad.frame} action={action_name(first_bad.action)} "
            f"model(btn={first_bad.model.buttons},my={first_bad.model.main_y:.2f}) "
            f"ref(btn={first_bad.ref.buttons},my={first_bad.ref.main_y:.2f})"
        )
    else:
        logger.info(f"FRAME-PERFECT: model reproduced multishine for all {len(p1)} handover frames")
    return rate


# rate = verify(ckpt_path)


# %%
# ---- full run --------------------------------------------------------------
if __name__ == "__main__":
    slp_path = record_multishine()
    build_dataset()
    ckpt_path = overfit()
    verify(ckpt_path)
