"""Wire-format and bit-exact round-trip tests for the hal/sim stack.

Two integration tests, both launching a real Dolphin:

* ``test_controller_wire_format_faithful`` — replays MDS controller bytes for
  the first 5 (pre-input) frames of a dev-set .slp and asserts the captured
  gamestate matches ``Trajectory.from_slp``. Plumbing check on the input path;
  bypasses input-driven physics so it does not depend on era-matched physics.

* ``test_fresh_recording_roundtrip_bit_exact`` — records a fresh .slp through
  the current Slippi build, extracts MDS columns, replays them, and asserts
  bit-exact diff for 200 frames. This is the H1 property: same-build records
  and replays round-trip exactly through libmelee → Dolphin → gamestate.

Era-mismatch gameplay-reproduction limits on the 2020 dev replays are
documented in memory ``project_roundtrip_limits``.
"""

import multiprocessing as mp

# libmelee's slippstream client spawns a child via mp.Process; on Python 3.14
# the default start method is "forkserver" which re-imports the worker script
# and is flaky alongside torch / streaming imports. Match notebooks and
# hal/data: force plain fork.
if mp.get_start_method(allow_none=True) != "fork":
    mp.set_start_method("fork", force=True)

from pathlib import Path

import melee
import numpy as np
import peppi_py
import pytest
from streaming import StreamingDataset

from hal.data.extract import extract_replay
from hal.data.index import read_jsonl
from hal.data.index import resolve_replay_path
from hal.paths import DEV_MDS_DIR as _DEV_MDS_DIR
from hal.paths import EMULATOR_PATH
from hal.paths import ISO_PATH as _ISO_PATH
from hal.sim.diff import diff
from hal.sim.inputs import ControllerInputs
from hal.sim.inputs import ControllerInputsValue
from hal.sim.loop import drive
from hal.sim.session import PlayerSetup
from hal.sim.session import ReplayMatchup
from hal.sim.session import Session
from hal.sim.sources import ControllerSource
from hal.sim.sources import InternalControllerSource
from hal.sim.sources import MdsControllerSource
from hal.sim.sources import ScriptedControllerSource
from hal.sim.sources import demo_sequence
from hal.sim.trajectory import Trajectory
from hal.wire import GAME_START_FRAME
from hal.wire import TRIGGER_DEADZONE

ISO_PATH = Path(_ISO_PATH)
DOLPHIN_PATH = Path(EMULATOR_PATH)
DEV_MDS_DIR = Path(_DEV_MDS_DIR)

# Stage filter — peppi/slp-native ids for BF, FD, DL, YS — and no Peach
# (turnips) or G&W (hammer). Picked for stability across runs.
_RNG_STABLE_STAGES = {31, 32, 28, 8}
_RNG_HEAVY_CHARACTERS = {9, 24}


def _check_prereqs() -> None:
    if not ISO_PATH.is_file():
        pytest.skip(f"ISO missing at {ISO_PATH}; run `python -m hal.scripts.fetch --name ssbm.ciso`")
    if not DOLPHIN_PATH.is_file():
        pytest.skip(
            f"Dolphin AppRun missing at {DOLPHIN_PATH}; run `python -m hal.scripts.fetch --name dolphin-exiai`"
        )
    if not (DEV_MDS_DIR / "manifest.jsonl").is_file():
        pytest.skip(f"dev MDS missing at {DEV_MDS_DIR}; run `python -m hal.scripts.fetch --name dev-mds`")


def _pick_safe_entry():
    for entry in read_jsonl(DEV_MDS_DIR / "manifest.jsonl"):
        if (
            entry.stage in _RNG_STABLE_STAGES
            and not any(p.character in _RNG_HEAVY_CHARACTERS for p in entry.players)
            and len(entry.players) == 2
            and entry.annotation is not None
        ):
            return entry
    pytest.skip("no RNG-stable replay in dev MDS")


@pytest.mark.integration
def test_controller_wire_format_faithful() -> None:
    """MDS controller bytes round-trip bit-exactly through libmelee → Dolphin.

    Frames 0..5 are pre-input (state 322 ENTRY); they exercise the
    libmelee→Dolphin wire path without any input-driven physics. A passing
    diff here proves frame alignment, port mapping, controller pipe, and
    gamestate capture are all correct.
    """
    _check_prereqs()
    entry = _pick_safe_entry()

    ds = StreamingDataset(
        local=str(DEV_MDS_DIR / entry.annotation.split),
        remote=None,
        shuffle=False,
        batch_size=1,
        allow_unsafe_types=False,
    )
    row = ds[entry.annotation.mds_row_idx]
    matchup = ReplayMatchup.from_replay(entry)

    sources: dict[int, ControllerSource] = {}
    for port, prefix in matchup.port_to_mds_prefix.items():
        sources[port] = MdsControllerSource(columns=row, port_prefix=prefix)
    for player in matchup.players:
        sources.setdefault(player.port, InternalControllerSource())

    with Session(iso_path=ISO_PATH, dolphin_path=DOLPHIN_PATH) as s:
        live = drive(s, matchup, sources, max_frames=5)

    truth = Trajectory.from_slp(resolve_replay_path(entry)).take(5)
    report = diff(live, truth, max_frames=5)
    assert report.passed, f"wire-format divergence: {report.summary()}\n" + "\n".join(
        f"  {d}" for d in report.divergences[:5]
    )


@pytest.mark.integration
def test_fresh_recording_roundtrip_bit_exact(tmp_path: Path) -> None:
    """Same-build record→replay is bit-exact across 200 frames and all post-fields.

    Records a fresh .slp with neutral inputs, extracts MDS columns from it,
    replays those bytes through a second Session, and asserts the live
    trajectory diffs bit-exact against ``Trajectory.from_slp(fresh_slp)``.
    """
    _check_prereqs()
    n_frames = 200
    matchup = ReplayMatchup(
        stage=melee.Stage.FINAL_DESTINATION,
        players=(
            PlayerSetup(port=1, character=melee.Character.FOX, costume=0),
            PlayerSetup(port=2, character=melee.Character.FOX, costume=1),
        ),
        port_to_mds_prefix={1: "p1", 2: "p2"},
    )

    record_sources: dict[int, ControllerSource] = {
        1: ScriptedControllerSource(sequence=demo_sequence(n_frames, port="p1")),
        2: ScriptedControllerSource(sequence=demo_sequence(n_frames, port="p2")),
    }
    with Session(
        iso_path=ISO_PATH,
        dolphin_path=DOLPHIN_PATH,
        slippi_port=51449,
        tmp_home_directory=False,
        replay_dir=str(tmp_path),
    ) as s:
        drive(s, matchup, record_sources, max_frames=n_frames)

    new_slps = sorted(tmp_path.rglob("*.slp"))
    assert new_slps, f"Slippi wrote no .slp under {tmp_path} — check replay_dir handling"
    fresh_slp = new_slps[-1]

    rows = extract_replay(str(fresh_slp))
    assert rows is not None, f"extract_replay returned None for fresh slp {fresh_slp}"

    replay_sources: dict[int, ControllerSource] = {
        1: MdsControllerSource(columns=rows, port_prefix="p1"),
        2: MdsControllerSource(columns=rows, port_prefix="p2"),
    }
    with Session(
        iso_path=ISO_PATH,
        dolphin_path=DOLPHIN_PATH,
        slippi_port=51449,
        tmp_home_directory=False,
        replay_dir=str(tmp_path),
    ) as s:
        live = drive(s, matchup, replay_sources, max_frames=n_frames)

    truth = Trajectory.from_slp(str(fresh_slp)).take(n_frames)
    report = diff(live, truth, max_frames=n_frames)
    assert report.passed, f"fresh-recording round-trip divergence: {report.summary()}\n" + "\n".join(
        f"  {d}" for d in report.divergences[:5]
    )


# --- analog wire sweep -------------------------------------------------------

_SWEEP_LEAD, _SWEEP_HOLD = 40, 3
# Stick bytes probing the deadzone boundary (gate at 23/80 = 0.2875), mid-range
# and clamp extremes. Triggers sweep the full 0..140 grid.
_SWEEP_STICK_BYTES = (-80, -79, -40, -24, -23, -22, -21, 0, 21, 22, 23, 24, 40, 79, 80)


def _sweep_program() -> tuple[list[ControllerInputs], list[tuple[str, float, int]]]:
    """Build (per-frame inputs, [(channel, fed_value, punch_idx), ...])."""

    def value(**kw: float) -> ControllerInputsValue:
        base = dict(main_x=0.0, main_y=0.0, c_x=0.0, c_y=0.0, trigger_l=0.0, trigger_r=0.0, buttons=0)
        base.update(kw)
        return ControllerInputsValue(**base)  # type: ignore[arg-type]

    punches: list[ControllerInputs] = [value()] * _SWEEP_LEAD
    reads: list[tuple[str, float, int]] = []
    for channel, values in (
        ("trigger_l", [k / 140.0 for k in range(141)]),
        ("trigger_r", [0.0, 43 / 140.0, 1.0]),
        ("main_x", [b / 80.0 for b in _SWEEP_STICK_BYTES]),
        ("c_x", [-1.0, -0.5, 0.5, 1.0]),
    ):
        for v in values:
            punches.extend([value(**{channel: v})] * _SWEEP_HOLD)
            reads.append((channel, v, len(punches) - 1))
    punches.extend([value()] * 10)
    return punches, reads


@pytest.mark.integration
def test_analog_sweep_reads_back_grid_exact(tmp_path: Path) -> None:
    """Every game-distinguishable analog value survives the wire bit-exactly.

    Drives the full trigger grid (0..140) plus deadzone-probing stick bytes
    through ``apply_inputs`` and asserts the recorded .slp returns the exact
    grid value. This is the regression that catches a trigger/stick wire
    protocol mismatch (the GCPad-path trigger bug was invisible to the other
    two tests because broken triggers replay consistently broken). Also pins
    the trigger deadzone: the logical channel must engage at exactly
    ``wire.TRIGGER_DEADZONE`` (byte 43).
    """
    _check_prereqs()
    punches, reads = _sweep_program()
    matchup = ReplayMatchup(
        stage=melee.Stage.FINAL_DESTINATION,
        players=(
            PlayerSetup(port=1, character=melee.Character.FOX, costume=0),
            PlayerSetup(port=2, character=melee.Character.FOX, costume=1),
        ),
        port_to_mds_prefix={1: "p1", 2: "p2"},
    )
    sources: dict[int, ControllerSource] = {
        1: ScriptedControllerSource(sequence=punches),
        2: ScriptedControllerSource(sequence=[]),  # neutral throughout
    }
    with Session(
        iso_path=ISO_PATH,
        dolphin_path=DOLPHIN_PATH,
        slippi_port=51449,
        tmp_home_directory=False,
        replay_dir=str(tmp_path),
    ) as s:
        drive(s, matchup, sources, max_frames=len(punches))

    slps = sorted(tmp_path.rglob("*.slp"))
    assert slps, f"Slippi wrote no .slp under {tmp_path}"
    game = peppi_py.read_slippi(str(slps[-1]))
    pre = game.frames.ports[0].leader.pre
    frame_ids = np.asarray(game.frames.id.to_pylist(), dtype=np.int64)
    ig0 = int(np.argmax(frame_ids >= GAME_START_FRAME))
    col = {
        "trigger_l": np.asarray(pre.triggers_physical.l.to_pylist(), dtype=np.float32)[ig0:],
        "trigger_r": np.asarray(pre.triggers_physical.r.to_pylist(), dtype=np.float32)[ig0:],
        "trigger_logical": np.asarray(pre.triggers.to_pylist(), dtype=np.float32)[ig0:],
        "main_x": np.asarray(pre.joystick.x.to_pylist(), dtype=np.float32)[ig0:],
        "c_x": np.asarray(pre.cstick.x.to_pylist(), dtype=np.float32)[ig0:],
    }

    failures: list[str] = []
    for channel, fed, punch_idx in reads:
        i = punch_idx + 1  # punches at iteration t land in slp pre[t + 1]
        rec = float(col[channel][i])
        if channel.startswith("trigger"):
            byte = round(fed * 140.0)
            if round(rec * 140.0) != byte:
                failures.append(f"{channel} fed {fed:.5f} (byte {byte}) read {rec:.5f}")
            logical = float(col["trigger_logical"][i])
            want_logical = 0.0 if fed < TRIGGER_DEADZONE else rec
            if abs(logical - want_logical) > 1e-6:
                failures.append(f"trigger_logical fed {fed:.5f} read {logical:.5f} want {want_logical:.5f}")
        else:
            byte = round(fed * 80.0)
            rec_byte = round(rec * 80.0)
            if abs(byte) <= 22:
                ok = rec == 0.0
            elif abs(byte) == 23:
                ok = rec_byte in (0, byte)  # gate boundary; either side is on-grid
            else:
                ok = rec_byte == byte
            if not ok:
                failures.append(f"{channel} fed {fed:.5f} (byte {byte}) read {rec:.5f}")
    assert not failures, f"{len(failures)} analog wire mismatches:\n" + "\n".join(failures[:20])
