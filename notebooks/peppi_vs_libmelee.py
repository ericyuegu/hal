"""Sanity check: peppi-py vs libmelee on the same slp.

For each replay we parse with both libraries, frame-align by signed frame id,
and compare every per-frame field that the new MDS schema cares about. We
print a per-field mismatch summary plus the first few example mismatches per
field so we can tell whether differences are systematic (mapping bugs) or
floating-point dust (acceptable).

Run:
    /home/ericgu/src/hal/.venv/bin/python notebooks/peppi_vs_libmelee.py \\
        /home/ericgu/data/ssbm/mang0/Game_20201215T114034.slp [more.slp ...]

If no paths are given, picks 5 slps from ~/data/ssbm/mang0/ at random.
"""

from __future__ import annotations

import math
import os
import random
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import melee
import peppi_py

# Slippi physical button bitmask (slp pre-frame block, "buttons_physical").
PHYSICAL_BUTTON_BITS: dict[melee.Button, int] = {
    melee.Button.BUTTON_A: 0x0100,
    melee.Button.BUTTON_B: 0x0200,
    melee.Button.BUTTON_X: 0x0400,
    melee.Button.BUTTON_Y: 0x0800,
    melee.Button.BUTTON_Z: 0x0010,
    melee.Button.BUTTON_R: 0x0020,
    melee.Button.BUTTON_L: 0x0040,
    melee.Button.BUTTON_START: 0x1000,
    melee.Button.BUTTON_D_UP: 0x0008,
}

FLOAT_TOL = 1e-4


@dataclass
class FieldStat:
    name: str
    compared: int = 0
    mismatches: int = 0
    examples: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.examples is None:
            self.examples = []

    def record(self, ok: bool, detail: str) -> None:
        self.compared += 1
        if not ok:
            self.mismatches += 1
            if len(self.examples) < 3:
                self.examples.append(detail)


def values_equal(a: Any, b: Any, *, tol: float = FLOAT_TOL) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, float) or isinstance(b, float):
        af = float(a)
        bf = float(b)
        if math.isnan(af) and math.isnan(bf):
            return True
        return abs(af - bf) <= tol
    return a == b


def peppi_scalar(arr: Any, i: int) -> Any:
    """Read a scalar from a pyarrow array or `None` array gracefully."""
    if arr is None:
        return None
    val = arr[i]
    # pyarrow Scalar -> python primitive
    if hasattr(val, "as_py"):
        return val.as_py()
    return val


def libmelee_frames(replay_path: Path) -> list[melee.GameState]:
    console = melee.Console(path=str(replay_path), is_dolphin=False, allow_old_version=True)
    if not console.connect():
        raise RuntimeError(f"libmelee failed to connect: {replay_path}")
    states: list[melee.GameState] = []
    try:
        s = console.step()
        while s is not None:
            if s.menu_state in (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH):
                states.append(s)
            s = console.step()
    finally:
        console.stop()
    return states


def compare_replay(replay_path: Path) -> dict[str, FieldStat]:
    stats: dict[str, FieldStat] = defaultdict(lambda: FieldStat(name="?"))

    def stat(name: str) -> FieldStat:
        s = stats[name]
        s.name = name
        return s

    g = peppi_py.read_slippi(str(replay_path), skip_frames=False)
    peppi_id = [int(x.as_py() if hasattr(x, "as_py") else x) for x in g.frames.id]
    peppi_id_to_idx = {fid: i for i, fid in enumerate(peppi_id)}

    # Derive libmelee-style action_frame (1-indexed integer counter that resets when state changes)
    # from peppi's post.state column per port.
    derived_action_frame: dict[int, list[int]] = {}
    for ppi in range(len(g.frames.ports)):
        states = [int(s.as_py()) for s in g.frames.ports[ppi].leader.post.state]
        af = []
        prev: int | None = None
        counter = 0
        for s in states:
            if s != prev:
                counter = 1
                prev = s
            else:
                counter += 1
            af.append(counter)
        derived_action_frame[ppi] = af

    melee_states = libmelee_frames(replay_path)
    if not melee_states:
        return stats

    # Map libmelee port (1-indexed) -> peppi port index (0-indexed) by Start.players[i].port
    peppi_port_to_idx: dict[int, int] = {}
    for i, pl in enumerate(g.start.players):
        # pl.port is melee.Port enum-like in peppi (P1=0,P2=1,P3=2,P4=3)
        port_value = int(getattr(pl.port, "value", pl.port))
        peppi_port_to_idx[port_value + 1] = i  # libmelee uses 1..4

    for ms in melee_states:
        pi = peppi_id_to_idx.get(int(ms.frame))
        if pi is None:
            stat("frame_alignment").record(False, f"libmelee frame {ms.frame} not in peppi")
            continue
        stat("frame_alignment").record(True, "")

        # Stage is per-replay (manifest field), and libmelee's enum value differs from
        # peppi's raw slp stage id. Skip comparison here; manifest will store both.

        for libmelee_port, libmelee_player in ms.players.items():
            ppi = peppi_port_to_idx.get(int(libmelee_port))
            if ppi is None:
                stat("port_alignment").record(False, f"libmelee port {libmelee_port} missing in peppi")
                continue
            data = g.frames.ports[ppi].leader
            pre = data.pre
            post = data.post

            def cmp(
                field: str,
                lib_val: Any,
                peppi_val: Any,
                *,
                tol: float = FLOAT_TOL,
                _frame: int = ms.frame,
                _port: int = libmelee_port,
            ) -> None:
                ok = values_equal(lib_val, peppi_val, tol=tol)
                stat(field).record(ok, f"frame {_frame} port {_port}: libmelee={lib_val!r} peppi={peppi_val!r}")

            # Post-frame state
            cmp("character", int(libmelee_player.character.value), peppi_scalar(post.character, pi))
            cmp("action", int(libmelee_player.action.value), peppi_scalar(post.state, pi))
            # Peppi `state_age` is fractional (sub-frame) and not directly comparable to
            # libmelee's integer `action_frame`. Instead, derive libmelee's convention
            # from peppi's post.state column (count consecutive identical states).
            lib_action_frame = float(libmelee_player.action_frame)
            if lib_action_frame >= 1:  # libmelee reports -1 on first frame of an action transition
                cmp("action_frame_derived", int(lib_action_frame), derived_action_frame[ppi][pi])
            cmp("position_x", float(libmelee_player.position.x), peppi_scalar(post.position.x, pi))
            cmp("position_y", float(libmelee_player.position.y), peppi_scalar(post.position.y, pi))
            cmp("percent", float(libmelee_player.percent), peppi_scalar(post.percent, pi))
            cmp("shield", float(libmelee_player.shield_strength), peppi_scalar(post.shield, pi))
            cmp("stock", int(libmelee_player.stock), peppi_scalar(post.stocks, pi))
            # libmelee facing: True=right (1.0), False=left (-1.0). Peppi direction: signed float scalar.
            peppi_dir = peppi_scalar(post.direction, pi)
            lib_dir = 1.0 if libmelee_player.facing else -1.0
            cmp("facing", lib_dir, float(peppi_dir) if peppi_dir is not None else None)
            cmp("on_ground", bool(libmelee_player.on_ground), not bool(peppi_scalar(post.airborne, pi)))
            cmp("jumps_used", int(libmelee_player.jumps_left), peppi_scalar(post.jumps, pi))
            # peppi.post.hitlag is None for slp <~3.8 (peppi doesn't parse it). Skip when None.
            peppi_hitlag = peppi_scalar(post.hitlag, pi) if post.hitlag is not None else None
            if peppi_hitlag is not None:
                cmp("hitlag_left", float(libmelee_player.hitlag_left), peppi_hitlag)

            # Velocities -> libmelee speed_* fields
            v = post.velocities
            cmp("speed_air_x_self", float(libmelee_player.speed_air_x_self), peppi_scalar(v.self_x_air, pi))
            cmp("speed_y_self", float(libmelee_player.speed_y_self), peppi_scalar(v.self_y, pi))
            cmp("speed_x_attack", float(libmelee_player.speed_x_attack), peppi_scalar(v.knockback_x, pi))
            cmp("speed_y_attack", float(libmelee_player.speed_y_attack), peppi_scalar(v.knockback_y, pi))
            cmp("speed_ground_x_self", float(libmelee_player.speed_ground_x_self), peppi_scalar(v.self_x_ground, pi))

            # Controller (input read at frame start = libmelee.controller_state at this same frame).
            # Peppi sticks are [-1, 1] with neutral=0; libmelee is [0, 1] with neutral=0.5.
            # Convert peppi -> libmelee for comparison.
            cs = libmelee_player.controller_state

            def _stick_to_lib(v: Any) -> Any:
                return None if v is None else float(v) / 2.0 + 0.5

            cmp("ctrl_main_x", float(cs.main_stick[0]), _stick_to_lib(peppi_scalar(pre.joystick.x, pi)))
            cmp("ctrl_main_y", float(cs.main_stick[1]), _stick_to_lib(peppi_scalar(pre.joystick.y, pi)))
            cmp("ctrl_c_x", float(cs.c_stick[0]), _stick_to_lib(peppi_scalar(pre.cstick.x, pi)))
            cmp("ctrl_c_y", float(cs.c_stick[1]), _stick_to_lib(peppi_scalar(pre.cstick.y, pi)))
            # KNOWN libmelee BUG: libmelee copies the slp single "logical trigger" value to
            # BOTH l_shoulder and r_shoulder, so it can't distinguish which physical trigger
            # was pressed. Peppi reads the separate physical L/R bytes correctly. We compare
            # against `pre.triggers` (logical) to validate that libmelee is at least
            # consistent with the logical aggregate; the *correct* values are physical.l/r.
            peppi_logical = peppi_scalar(pre.triggers, pi)
            cmp("ctrl_l_shoulder_vs_logical", float(cs.l_shoulder), peppi_logical)
            cmp("ctrl_r_shoulder_vs_logical", float(cs.r_shoulder), peppi_logical)

            # Buttons: decode peppi buttons_physical bitmask, compare to libmelee's button dict.
            phys_bits = int(peppi_scalar(pre.buttons_physical, pi) or 0)
            for btn, mask in PHYSICAL_BUTTON_BITS.items():
                lib_pressed = bool(cs.button.get(btn, False))
                peppi_pressed = bool(phys_bits & mask)
                cmp(f"btn_{btn.name.removeprefix('BUTTON_')}", lib_pressed, peppi_pressed)

            # Raw analog x (slp >= 1.2.0) and y (slp >= 3.15.0)
            raw_x = peppi_scalar(pre.raw_analog_x, pi) if pre.raw_analog_x is not None else None
            lib_raw = getattr(cs, "raw_main_stick", (None, None))
            if lib_raw[0] is not None and raw_x is not None:
                cmp("raw_main_stick_x", int(lib_raw[0]), int(raw_x))
            raw_y = peppi_scalar(pre.raw_analog_y, pi) if pre.raw_analog_y is not None else None
            if lib_raw[1] is not None and raw_y is not None:
                cmp("raw_main_stick_y", int(lib_raw[1]), int(raw_y))

    return stats


def summarize(label: str, stats: dict[str, FieldStat]) -> None:
    print(f"\n=== {label} ===")
    rows = sorted(stats.values(), key=lambda s: (-s.mismatches, s.name))
    width = max(len(s.name) for s in rows) if rows else 4
    for s in rows:
        rate = (s.mismatches / s.compared * 100.0) if s.compared else 0.0
        marker = "  " if s.mismatches == 0 else "!!"
        print(f"  {marker} {s.name:<{width}}  cmp={s.compared:>6}  miss={s.mismatches:>6}  ({rate:5.2f}%)")
        for ex in s.examples:
            print(f"        {ex}")


def merge_stats(into: dict[str, FieldStat], delta: dict[str, FieldStat]) -> None:
    for k, v in delta.items():
        if k not in into:
            into[k] = FieldStat(name=v.name)
        into[k].compared += v.compared
        into[k].mismatches += v.mismatches
        for ex in v.examples:
            if len(into[k].examples) < 3:
                into[k].examples.append(ex)


def pick_replays(args: list[str]) -> list[Path]:
    if args:
        return [Path(a) for a in args]
    d = Path("/home/ericgu/data/ssbm/mang0")
    all_slps = sorted(d.glob("*.slp"))
    random.seed(0)
    random.shuffle(all_slps)
    # filter out trivially short ones (fast peek via filesize > 200KB)
    picks: list[Path] = []
    for p in all_slps:
        if p.stat().st_size > 200_000:
            picks.append(p)
        if len(picks) >= 10:
            break
    return picks


def main(argv: list[str]) -> int:
    replays = pick_replays(argv[1:])
    print(f"Comparing {len(replays)} replays:")
    for p in replays:
        print(f"  {p}")

    overall: dict[str, FieldStat] = {}
    for p in replays:
        try:
            per = compare_replay(p)
        except Exception as e:
            print(f"  ! {p}: {type(e).__name__}: {e}")
            continue
        summarize(p.name, per)
        merge_stats(overall, per)

    summarize("OVERALL", overall)
    return 0 if all(s.mismatches == 0 for s in overall.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
