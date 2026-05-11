"""Replay reproduction sanity harness.

Reads a `.slp`, extracts both players' controller inputs, replays them
into a live Dolphin via libmelee's named pipes, and verifies that observed
game states match the source replay frame-by-frame.

The pipe → padBuf → game → slp chain is fully detailed in
`notebooks/replay_reproduction_sanity.README.md`. **Read it first** if you
are touching `ReplayControllerSender` or the analog conversion helpers; the
correctness of bit-exact reproduction depends on submitting pipe values
that pass through Dolphin's `floor((v-0.5)*254)` (sticks) and `u8(v*255)`
(triggers) quantization to land on padBuf bytes that the game's stick
processor turns back into the slp's recorded values. Naive values cause
double-quantization and divergence — see the README's "Why" sections for
concrete numeric traces.

Status (2026-05-01): both `normal` and `ffw` modes reproduce
`Game_20201215T165952.slp` with 4-21 `hitlag_left` mismatches per
prefix (residual drift on hit moments) — all other comparison fields
match bit-exactly. See the README + `replay_reproduction_sanity.repro_log.md`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import socket
import sys
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import melee
from loguru import logger
from melee.controller import fix_analog_stick

from hal.emulator.controller_io import ControllerSink
from hal.local_paths import EMULATOR_PATH
from hal.local_paths import ISO_PATH

SLP_RAW_MAIN_X_VERSION: tuple[int, int, int] = (1, 2, 0)
SLP_RAW_MAIN_Y_VERSION: tuple[int, int, int] = (3, 15, 0)
SLP_RAW_C_STICK_VERSION: tuple[int, int, int] = (3, 17, 0)

TRIGGER_DEADZONE_RAW: int = 0x2A
TRIGGER_MAX_RAW: int = 0x8C
TRIGGER_RANGE_RAW: int = TRIGGER_MAX_RAW - TRIGGER_DEADZONE_RAW
STICK_UNIT_RADIUS: int = 80

DIGITAL_BUTTONS: tuple[melee.Button, ...] = (
    melee.Button.BUTTON_A,
    melee.Button.BUTTON_B,
    melee.Button.BUTTON_X,
    melee.Button.BUTTON_Y,
    melee.Button.BUTTON_Z,
    melee.Button.BUTTON_L,
    melee.Button.BUTTON_R,
    melee.Button.BUTTON_START,
    melee.Button.BUTTON_D_UP,
)

PLAYER_FIELDS: tuple[str, ...] = (
    "character",
    "costume",
    "action",
    "action_frame",
    "position.x",
    "position.y",
    "percent",
    "stock",
    "facing",
    "on_ground",
    "jumps_left",
    "shield_strength",
    "hitlag_left",
    "hitstun_frames_left",
    "speed_air_x_self",
    "speed_y_self",
    "speed_x_attack",
    "speed_y_attack",
    "speed_ground_x_self",
)

CONTROLLER_FIELDS: tuple[str, ...] = (
    "button.BUTTON_A",
    "button.BUTTON_B",
    "button.BUTTON_X",
    "button.BUTTON_Y",
    "button.BUTTON_Z",
    "button.BUTTON_L",
    "button.BUTTON_R",
    "button.BUTTON_START",
    "button.BUTTON_D_UP",
    "main_stick.0",
    "main_stick.1",
    "c_stick.0",
    "c_stick.1",
    "l_shoulder",
    "r_shoulder",
)

FLOAT_TOLERANCE = 1e-4

CHARACTER_COSTUME_COUNTS: dict[melee.Character, int] = {
    melee.Character.MARIO: 5,
    melee.Character.FOX: 4,
    melee.Character.CPTFALCON: 6,
    melee.Character.DK: 5,
    melee.Character.KIRBY: 6,
    melee.Character.BOWSER: 4,
    melee.Character.LINK: 5,
    melee.Character.SHEIK: 5,
    melee.Character.NESS: 4,
    melee.Character.PEACH: 5,
    melee.Character.POPO: 4,
    melee.Character.NANA: 4,
    melee.Character.PIKACHU: 4,
    melee.Character.SAMUS: 5,
    melee.Character.YOSHI: 6,
    melee.Character.JIGGLYPUFF: 5,
    melee.Character.MEWTWO: 4,
    melee.Character.LUIGI: 4,
    melee.Character.MARTH: 5,
    melee.Character.ZELDA: 5,
    melee.Character.YLINK: 5,
    melee.Character.DOC: 5,
    melee.Character.FALCO: 4,
    melee.Character.PICHU: 4,
    melee.Character.GAMEANDWATCH: 4,
    melee.Character.GANONDORF: 5,
    melee.Character.ROY: 5,
}


@dataclass(frozen=True)
class TimeoutConfig:
    startup_s: float = 30.0
    connect_s: float = 30.0
    controller_connect_s: float = 30.0
    menu_s: float = 90.0
    first_ingame_s: float = 30.0
    frame_s: float = 5.0
    poll_s: float = 0.25


@dataclass(frozen=True)
class ReplayMetadata:
    replay_path: Path
    stage: melee.Stage
    ports: tuple[int, ...]
    characters: dict[int, melee.Character]
    costumes: dict[int, int]
    first_frame: int
    last_frame: int
    slp_version: tuple[int, int, int] = (0, 0, 0)


@dataclass(frozen=True)
class FrameRecord:
    frame: int
    state: melee.GameState
    controllers: dict[int, dict[str, Any]]


@dataclass
class ReproductionResult:
    mode: str
    compared_frames: int = 0
    mismatches: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    elapsed_s: float = 0.0


@dataclass
class CostumeCorrectionState:
    remaining_taps: int | None = None
    button: melee.Button | None = None
    button_down: bool = False
    cooldown_frames: int = 0


class ReplayControllerSender:
    """Submit slp-recorded controller states through libmelee/Dolphin's pipe.

    Default mapping (works on every slp >= 0.1.0 since it only relies on the
    `processed` floats at slp 0x19/0x1D/0x21/0x25/0x29):
        Main stick X/Y, c-stick X/Y: pipe = fix_analog_stick(processed).
            Exact inverse of `processed = clamp(raw, -80, 80) / 80` inside
            the unit circle, naturally saturating outside.
        Triggers (L + R analog): raw = round(processed * 0x8C);
            pipe = (raw + 0.5) / 255. slp 0x29 records `raw / 0x8C`.
        Buttons: physical-bit transitions → press_button / release_button.

    Optional opt-in: `use_raw_main_stick=True` injects the slp's raw stick
    bytes (0x3B / 0x40) directly into Dolphin's padBuf via
    `pipe_value_for_axis_raw`. **This does not currently reproduce
    bit-exactly on real online replays** because the slp's recorded
    `processed` (0x19) is not a naive `clamp(raw, -80, 80) / 80` of the
    recorded `raw` (0x3B) — see the empirical table in the README. Kept
    here as plumbing for future investigations (UCF + per-stick deadzone
    calibration). When `use_raw_main_stick` is True, falls back to the
    default for any axis whose raw byte isn't actually in the source slp
    (slp_version < 1.2.0 for X, < 3.15.0 for Y).

    Construct the underlying `melee.Controller` with
    `fix_analog_inputs=False` so libmelee passes the pre-quantized pipe
    value through verbatim.
    """

    def __init__(
        self,
        controllers: dict[int, ControllerSink],
        slp_version: tuple[int, int, int] = (0, 0, 0),
        *,
        use_raw_main_stick: bool = False,
        use_exi_inputs: bool = False,
    ) -> None:
        self.controllers = controllers
        self.slp_version = slp_version
        self.use_raw_main_stick = use_raw_main_stick
        self.use_exi_inputs = use_exi_inputs
        self.has_raw_main_x = use_raw_main_stick and slp_version >= SLP_RAW_MAIN_X_VERSION
        self.has_raw_main_y = use_raw_main_stick and slp_version >= SLP_RAW_MAIN_Y_VERSION
        self.previous_buttons: dict[int, dict[melee.Button, bool]] = {
            port: {button: False for button in DIGITAL_BUTTONS} for port in controllers
        }

    def send_frame(self, states_by_port: dict[int, melee.ControllerState]) -> dict[int, dict[str, Any]]:
        commands: dict[int, dict[str, Any]] = {}
        for port, controller_state in states_by_port.items():
            controller = self.controllers[port]
            previous = self.previous_buttons[port]
            port_commands: list[dict[str, Any]] = []

            for button in DIGITAL_BUTTONS:
                pressed = bool(controller_state.button.get(button, False))
                was_pressed = previous[button]
                if pressed and not was_pressed:
                    controller.press_button(button)
                    port_commands.append({"command": "press_button", "button": button_name(button)})
                elif was_pressed and not pressed:
                    controller.release_button(button)
                    port_commands.append({"command": "release_button", "button": button_name(button)})
                previous[button] = pressed

            main_x = float(controller_state.main_stick[0])
            main_y = float(controller_state.main_stick[1])
            raw_x_src, raw_y_src = controller_state.raw_main_stick
            if self.has_raw_main_x:
                main_pipe_x = pipe_value_for_axis_raw(int(raw_x_src))
            else:
                main_pipe_x = fix_analog_stick(main_x)
            if self.has_raw_main_y:
                main_pipe_y = pipe_value_for_axis_raw(int(raw_y_src))
            else:
                main_pipe_y = fix_analog_stick(main_y)

            c_x = float(controller_state.c_stick[0])
            c_y = float(controller_state.c_stick[1])
            c_pipe_x = fix_analog_stick(c_x)
            c_pipe_y = fix_analog_stick(c_y)

            l_processed = float(controller_state.l_shoulder)
            r_processed = float(controller_state.r_shoulder)
            l_raw = trigger_raw_from_processed(l_processed)
            r_raw = trigger_raw_from_processed(r_processed)
            if self.use_exi_inputs:
                l_pipe = pipe_value_for_trigger_raw(l_raw)
                r_pipe = pipe_value_for_trigger_raw(r_raw)
            else:
                l_pipe = pipe_value_for_trigger_amount_via_axis(l_processed)
                r_pipe = pipe_value_for_trigger_amount_via_axis(r_processed)

            controller.tilt_analog(melee.Button.BUTTON_MAIN, main_pipe_x, main_pipe_y)
            controller.tilt_analog(melee.Button.BUTTON_C, c_pipe_x, c_pipe_y)
            controller.press_shoulder(melee.Button.BUTTON_L, l_pipe)
            controller.press_shoulder(melee.Button.BUTTON_R, r_pipe)
            controller.flush()

            commands[port] = {
                "digital": {button_name(button): previous[button] for button in DIGITAL_BUTTONS},
                "main_stick_processed": [main_x, main_y],
                "main_stick_raw_src": [int(raw_x_src), int(raw_y_src)],
                "main_stick_raw_used_x": self.has_raw_main_x,
                "main_stick_raw_used_y": self.has_raw_main_y,
                "main_stick_pipe": [main_pipe_x, main_pipe_y],
                "c_stick_processed": [c_x, c_y],
                "c_stick_pipe": [c_pipe_x, c_pipe_y],
                "l_shoulder_processed": l_processed,
                "l_shoulder_raw": l_raw,
                "l_shoulder_pipe": l_pipe,
                "r_shoulder_processed": r_processed,
                "r_shoulder_raw": r_raw,
                "r_shoulder_pipe": r_pipe,
                "transitions": port_commands,
                "flushes": 1,
            }

        return commands

    def snapshot(self) -> dict[int, dict[str, Any]]:
        return {
            port: {button_name(button): pressed for button, pressed in state.items()}
            for port, state in self.previous_buttons.items()
        }


class JsonlDebugWriter:
    def __init__(self, debug_dir: Path | None, mode: str) -> None:
        self.frame_file = None
        self.command_file = None
        if debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)
            self.frame_file = (debug_dir / f"{mode}_frame_comparisons.jsonl").open("w")
            self.command_file = (debug_dir / f"{mode}_controller_commands.jsonl").open("w")

    def write_frame(self, payload: dict[str, Any]) -> None:
        if self.frame_file is not None:
            self.frame_file.write(json.dumps(json_ready(payload), default=json_default) + "\n")
            self.frame_file.flush()

    def write_commands(self, payload: dict[str, Any]) -> None:
        if self.command_file is not None:
            self.command_file.write(json.dumps(json_ready(payload), default=json_default) + "\n")
            self.command_file.flush()

    def close(self) -> None:
        for file in (self.frame_file, self.command_file):
            if file is not None:
                file.close()


def pipe_value_for_axis_raw(raw: int) -> float:
    """Pipe float v that yields Dolphin padBuf = raw via floor((v-0.5)*254).

    Picks the bin midpoint to avoid floating-point edge effects.
    Clamps `raw` to [-127, 127] since Dolphin's mapping cannot produce ±128.
    """
    raw_clamped = max(-127, min(127, int(raw)))
    return min(1.0, max(0.0, (raw_clamped + 0.5) / 254.0 + 0.5))


def pipe_value_for_trigger_raw(raw: int) -> float:
    """Pipe float v that yields Dolphin padBuf = raw via u8(v*255).

    Use with FFW / EXI mode where the "Allow Bot Input Overrides" gecko
    code reads `padBuf[6/7]` directly via `prepareOverwriteInputs`.
    """
    raw_clamped = max(0, min(255, int(raw)))
    return min(1.0, max(0.0, (raw_clamped + 0.5) / 255.0))


def pipe_value_for_trigger_amount_via_axis(amount: float) -> float:
    """Pipe float v for normal mode (no EXI override).

    GCPadEmu's `pad.triggerLeft = u8(triggers[0] * 0xFF)`, where triggers[0]
    comes from `MixedTriggers::GetState` which reads the analog "Axis L +"
    state. `Pipes::SetAxis("L", v)` sets `Axis L + = max(0, v-0.5)*2`.
    To make `pad.triggerLeft = round(amount*0x8C)` (so slp 0x29 logs
    `amount`), we need `Axis L + = round(amount*0x8C) / 0xFF`, i.e.,
    pipe `v = (Axis L +)/2 + 0.5`.

    The previous trigger mapping (`pipe_value_for_trigger_raw`) put the raw
    byte into `padBuf[6]`, but normal mode doesn't read padBuf — it routes
    through ControllerInterface bindings. With pipe < 0.5 the +/- split
    leaves `Axis L + = 0`, so the trigger reads 0. That's the source of
    the L_shoulder=0 mismatch starting at frame -4 in the dev replay.
    """
    if amount <= 0:
        return 0.5
    raw_target = max(0, min(255, round(amount * TRIGGER_MAX_RAW)))
    # Pick the bin midpoint so that u8(axis_state * 0xFF) == raw_target despite
    # C++ truncation in `pad.triggerLeft = u8(triggers[0] * 0xFF)`.
    axis_state = (raw_target + 0.5) / 255.0
    return min(1.0, max(0.0, axis_state / 2 + 0.5))


def axis_raw_from_processed(processed_libmelee: float) -> int:
    """Approximate raw int8 byte from a libmelee-scale processed stick value.

    Libmelee scale: 0=full-left, 0.5=neutral, 1=full-right (after game maps
    raw int8 → [-1,1] and libmelee remaps to [0,1]). Inside the unit circle,
    the game uses processed = raw/80 per axis, so raw ≈ round((p-0.5)*160).
    """
    raw = round((processed_libmelee - 0.5) * (STICK_UNIT_RADIUS * 2))
    return max(-STICK_UNIT_RADIUS, min(STICK_UNIT_RADIUS, int(raw)))


def trigger_raw_from_processed(processed: float) -> int:
    """Inverse of slp's recorded shoulder value.

    Empirically, slp 0x29 records `raw / 0x8C` (no deadzone applied to the
    recorded value; deadzone affects only state transitions in-game). Submit
    `raw = round(processed * 0x8C)` so the live game records the same value.
    """
    if processed <= 0:
        return 0
    raw = round(processed * TRIGGER_MAX_RAW)
    return max(0, min(255, int(raw)))


def button_name(button: melee.Button) -> str:
    return button.name.replace("BUTTON_", "")


def enum_json(value: Any) -> Any:
    if hasattr(value, "name") and hasattr(value, "value"):
        return {"name": value.name, "value": value.value}
    return value


def json_default(value: Any) -> Any:
    value = enum_json(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    return str(value)


def json_ready(value: Any) -> Any:
    value = enum_json(value)
    if isinstance(value, Mapping):
        return {str(json_ready(key)): json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return json_ready(dataclasses.asdict(value))
    if hasattr(value, "item"):
        return value.item()
    return value


def get_attr_path(obj: Any, path: str) -> Any:
    value = obj
    for part in path.split("."):
        if part == "button":
            continue
        if part.startswith("BUTTON_"):
            value = value.button[getattr(melee.Button, part)]
        elif part.isdigit():
            value = value[int(part)]
        else:
            value = getattr(value, part)
    return value


def normalized(value: Any) -> Any:
    if hasattr(value, "value") and hasattr(value, "name"):
        return value.value
    return value


def comparable_value(value: Any) -> Any:
    value = normalized(value)
    if isinstance(value, float):
        return round(value, 6)
    return value


def controller_snapshot(controller_state: melee.ControllerState) -> dict[str, Any]:
    return {
        "buttons": {
            button_name(button): bool(controller_state.button.get(button, False)) for button in DIGITAL_BUTTONS
        },
        "main_stick": [float(controller_state.main_stick[0]), float(controller_state.main_stick[1])],
        "c_stick": [float(controller_state.c_stick[0]), float(controller_state.c_stick[1])],
        "l_shoulder": float(controller_state.l_shoulder),
        "r_shoulder": float(controller_state.r_shoulder),
    }


def source_controller_states(state: melee.GameState, ports: tuple[int, ...]) -> dict[int, melee.ControllerState]:
    return {port: state.players[port].controller_state for port in ports if port in state.players}


def find_open_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def read_source_replay(replay_path: Path) -> tuple[ReplayMetadata, dict[int, FrameRecord]]:
    console = melee.Console(path=str(replay_path), is_dolphin=False, allow_old_version=True)
    if not console.connect():
        raise RuntimeError(f"Failed to connect to source replay: {replay_path}")

    frames: dict[int, FrameRecord] = {}
    metadata: ReplayMetadata | None = None
    try:
        while True:
            state = console.step()
            if state is None:
                break
            if state.menu_state not in (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH):
                continue

            ports = tuple(sorted(int(port) for port in state.players))
            controllers = {port: controller_snapshot(state.players[port].controller_state) for port in ports}
            frames[int(state.frame)] = FrameRecord(frame=int(state.frame), state=state, controllers=controllers)

            if metadata is None:
                metadata = ReplayMetadata(
                    replay_path=replay_path,
                    stage=state.stage,
                    ports=ports,
                    characters={port: state.players[port].character for port in ports},
                    costumes={port: int(state.players[port].costume) for port in ports},
                    first_frame=int(state.frame),
                    last_frame=int(state.frame),
                    slp_version=tuple(console.slp_version_tuple or (0, 0, 0)),  # type: ignore[arg-type]
                )
    finally:
        console.stop()

    if metadata is None or not frames:
        raise RuntimeError(f"No in-game frames found in source replay: {replay_path}")

    last_frame = int(max(frames))
    metadata = dataclasses.replace(metadata, last_frame=last_frame)
    return metadata, frames


def mode_console_kwargs(
    mode: str, emulator_path: Path, replay_dir: Path, port: int, timeouts: TimeoutConfig
) -> dict[str, Any]:
    common: dict[str, Any] = {
        "path": str(emulator_path),
        "is_dolphin": True,
        "tmp_home_directory": True,
        "copy_home_directory": False,
        "replay_dir": str(replay_dir),
        "blocking_input": True,
        "polling_mode": True,
        "polling_timeout": timeouts.poll_s,
        "slippi_port": port,
        "online_delay": 0,
        "save_replays": False,
        "fullscreen": False,
        "setup_gecko_codes": True,
    }
    if mode == "normal":
        return common | {"gfx_backend": "", "disable_audio": False, "use_exi_inputs": False, "enable_ffw": False}
    if mode == "ffw":
        return common | {"gfx_backend": "Null", "disable_audio": True, "use_exi_inputs": True, "enable_ffw": True}
    raise ValueError(f"Unknown mode: {mode}")


def step_until(console: melee.Console, deadline: float, label: str) -> melee.GameState:
    while time.monotonic() < deadline:
        state = console.step()
        if state is not None:
            return state
    raise TimeoutError(f"Timed out waiting for {label}")


def connect_controller_with_timeout(controller: melee.Controller, timeout_s: float) -> bool:
    # libmelee's POSIX named pipe open blocks, so this is a best-effort deadline around
    # implementations that return normally.
    deadline = time.monotonic() + timeout_s
    result = controller.connect()
    if time.monotonic() > deadline:
        raise TimeoutError(f"Timed out connecting controller on port {controller.port}")
    return result


def setup_match(
    console: melee.Console,
    controllers: dict[int, melee.Controller],
    metadata: ReplayMetadata,
    timeouts: TimeoutConfig,
) -> melee.GameState:
    helpers = {port: melee.MenuHelper() for port in controllers}
    costume_states = {port: CostumeCorrectionState() for port in controllers}
    deadline = time.monotonic() + timeouts.menu_s
    last_log_s = 0.0
    while time.monotonic() < deadline:
        state = step_until(console, min(deadline, time.monotonic() + timeouts.frame_s), "menu frame")
        if state.menu_state in (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH):
            logger.info("Reached in-game menu state at live frame {}", state.frame)
            return state

        now = time.monotonic()
        if now - last_log_s > 5:
            logger.info(
                "Menu progress: state={} submenu={} ready={} players={}",
                state.menu_state,
                state.submenu,
                state.ready_to_start,
                css_player_summary(state, metadata.ports),
            )
            last_log_s = now

        if state.menu_state == melee.Menu.CHARACTER_SELECT:
            selected = drive_character_select(state, controllers, helpers, metadata, costume_states)
            if selected and metadata.ports[0] in controllers:
                starter = controllers[metadata.ports[0]]
                if state.frame % 2 == 0:
                    starter.press_button(melee.Button.BUTTON_START)
                else:
                    starter.release_button(melee.Button.BUTTON_START)
        else:
            for port, controller in controllers.items():
                helpers[port].menu_helper_simple(
                    state,
                    controller,
                    metadata.characters[port],
                    metadata.stage,
                    costume=metadata.costumes.get(port, 0),
                    cpu_level=0,
                    autostart=port == metadata.ports[0],
                    swag=False,
                )

    raise TimeoutError(f"Timed out setting up match after {timeouts.menu_s:.1f}s")


def drive_character_select(
    state: melee.GameState,
    controllers: dict[int, melee.Controller],
    helpers: dict[int, melee.MenuHelper],
    metadata: ReplayMetadata,
    costume_states: dict[int, CostumeCorrectionState],
) -> bool:
    all_selected = True
    for port, controller in controllers.items():
        player = state.players.get(port)
        if player is None:
            controller.release_all()
            all_selected = False
            continue

        target_character = metadata.characters[port]
        target_costume = metadata.costumes.get(port, 0)
        character_ready = player.character is target_character and player.coin_down
        costume_ready = int(player.costume) == target_costume

        if not character_ready:
            all_selected = False
            helpers[port].choose_character(
                character=target_character,
                gamestate=state,
                controller=controller,
                cpu_level=0,
                costume=target_costume,
                swag=False,
                start=False,
            )
        elif not costume_ready:
            costume_state = costume_states[port]
            if costume_state.remaining_taps is None:
                costume_state.button, costume_state.remaining_taps = costume_cycle_plan(
                    target_character,
                    observed_costume=int(player.costume),
                    target_costume=target_costume,
                )
                logger.info(
                    "Costume correction for port {}: character={} observed={} target={} button={} taps={}",
                    port,
                    target_character,
                    int(player.costume),
                    target_costume,
                    costume_state.button,
                    costume_state.remaining_taps,
                )

            if costume_state.cooldown_frames > 0:
                all_selected = False
                costume_state.cooldown_frames -= 1
                controller.release_all()
            elif costume_state.button_down:
                all_selected = False
                costume_state.button_down = False
                costume_state.cooldown_frames = 3
                controller.release_all()
            elif costume_state.remaining_taps > 0:
                all_selected = False
                costume_state.remaining_taps -= 1
                costume_state.button_down = True
                controller.press_button(costume_state.button or melee.Button.BUTTON_X)
            else:
                controller.release_all()
        else:
            controller.release_all()

    return all_selected


def costume_cycle_plan(
    character: melee.Character,
    observed_costume: int,
    target_costume: int,
) -> tuple[melee.Button, int]:
    costume_count = CHARACTER_COSTUME_COUNTS.get(character)
    if costume_count is None:
        return melee.Button.BUTTON_X, max(target_costume - observed_costume, 0)

    forward_taps = (target_costume - observed_costume) % costume_count
    backward_taps = (observed_costume - target_costume) % costume_count
    if backward_taps < forward_taps:
        return melee.Button.BUTTON_Y, backward_taps
    return melee.Button.BUTTON_X, forward_taps


def css_player_summary(state: melee.GameState, ports: tuple[int, ...]) -> dict[int, dict[str, Any]]:
    summary = {}
    for port in ports:
        player = state.players.get(port)
        if player is None:
            continue
        summary[port] = {
            "character": getattr(player.character, "name", str(player.character)),
            "selected": getattr(player.character_selected, "name", str(player.character_selected)),
            "costume": int(player.costume),
            "coin_down": bool(player.coin_down),
            "cursor": [round(float(player.cursor.x), 2), round(float(player.cursor.y), 2)],
        }
    return summary


def compare_states(
    source: melee.GameState,
    live: melee.GameState,
    ports: tuple[int, ...],
    *,
    compare_frame_number: bool,
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []

    global_fields = ("stage",) if not compare_frame_number else ("frame", "stage")
    for field in global_fields:
        source_value = get_attr_path(source, field)
        live_value = get_attr_path(live, field)
        if normalized(source_value) != normalized(live_value):
            mismatches.append(mismatch_payload(None, field, source_value, live_value))

    for port in ports:
        if port not in source.players or port not in live.players:
            mismatches.append(
                {
                    "port": port,
                    "field": "players",
                    "expected": port in source.players,
                    "observed": port in live.players,
                }
            )
            continue
        source_player = source.players[port]
        live_player = live.players[port]
        for field in PLAYER_FIELDS:
            source_value = get_attr_path(source_player, field)
            live_value = get_attr_path(live_player, field)
            if values_differ(source_value, live_value):
                mismatches.append(mismatch_payload(port, field, source_value, live_value))

        for field in CONTROLLER_FIELDS:
            source_value = get_attr_path(source_player.controller_state, field)
            live_value = get_attr_path(live_player.controller_state, field)
            if values_differ(source_value, live_value):
                mismatches.append(mismatch_payload(port, f"controller.{field}", source_value, live_value))

    return mismatches


def values_differ(expected: Any, observed: Any) -> bool:
    expected = normalized(expected)
    observed = normalized(observed)
    if isinstance(expected, float) or isinstance(observed, float):
        return abs(float(expected) - float(observed)) > FLOAT_TOLERANCE
    return expected != observed


def mismatch_payload(port: int | None, field: str, expected: Any, observed: Any) -> dict[str, Any]:
    payload = {
        "port": port,
        "field": field,
        "expected": comparable_value(expected),
        "observed": comparable_value(observed),
    }
    if isinstance(normalized(expected), float) or isinstance(normalized(observed), float):
        payload["abs_diff"] = abs(float(normalized(expected)) - float(normalized(observed)))
    return payload


def run_mode(
    *,
    mode: str,
    metadata: ReplayMetadata,
    source_frames: dict[int, FrameRecord],
    emulator_path: Path,
    iso_path: Path,
    prefix_frames: int,
    start_frame: int,
    stop_on_mismatch: bool,
    debug_dir: Path | None,
    timeouts: TimeoutConfig,
    use_raw_main_stick: bool = False,
) -> ReproductionResult:
    run_start = time.monotonic()
    replay_dir = (debug_dir or Path("/tmp/replay_reproduction_sanity")) / mode / "replays"
    replay_dir.mkdir(parents=True, exist_ok=True)
    udp_port = find_open_udp_port()
    console = melee.Console(**mode_console_kwargs(mode, emulator_path, replay_dir, udp_port, timeouts))
    controllers = {
        port: melee.Controller(console=console, port=port, type=melee.ControllerType.STANDARD, fix_analog_inputs=False)
        for port in metadata.ports
    }
    sender = ReplayControllerSender(
        controllers,
        slp_version=metadata.slp_version,
        use_raw_main_stick=use_raw_main_stick,
        use_exi_inputs=(mode == "ffw"),
    )
    debug = JsonlDebugWriter(debug_dir, mode)
    result = ReproductionResult(mode=mode)
    history: deque[dict[str, Any]] = deque(maxlen=12)

    logger.info(
        "Starting mode={} replay={} emulator={} iso={} udp_port={} stage={} ports={} characters={} costumes={}",
        mode,
        metadata.replay_path,
        emulator_path,
        iso_path,
        udp_port,
        metadata.stage,
        metadata.ports,
        metadata.characters,
        metadata.costumes,
    )

    try:
        console.run(iso_path=str(iso_path))
        connect_deadline = time.monotonic() + timeouts.connect_s
        while time.monotonic() < connect_deadline:
            if console.connect():
                break
            time.sleep(0.25)
        else:
            raise TimeoutError(f"Timed out connecting to Dolphin for mode={mode}")

        for controller in controllers.values():
            if not connect_controller_with_timeout(controller, timeouts.controller_connect_s):
                raise RuntimeError(f"Failed to connect controller on port {controller.port}")

        live_state = setup_match(console, controllers, metadata, timeouts)
        console.controllers.clear()
        first_ingame_deadline = time.monotonic() + timeouts.first_ingame_s
        while live_state.frame < metadata.first_frame and time.monotonic() < first_ingame_deadline:
            source_state = source_frames.get(live_state.frame + 1)
            if source_state is not None:
                commands = sender.send_frame(source_controller_states(source_state.state, metadata.ports))
                debug.write_commands(
                    {
                        "mode": mode,
                        "live_frame": live_state.frame,
                        "source_frame": source_state.frame,
                        "commands": commands,
                    }
                )
            live_state = step_until(
                console, min(first_ingame_deadline, time.monotonic() + timeouts.frame_s), "first in-game frame"
            )

        logger.info("Aligned mode={} at live_frame={} source_start={}", mode, live_state.frame, start_frame)

        target_end = start_frame + prefix_frames
        while result.compared_frames < prefix_frames:
            source_frame = live_state.frame
            if source_frame < start_frame:
                pass
            elif source_frame >= target_end:
                break
            elif source_frame in source_frames:
                frame_record = source_frames[source_frame]
                mismatches = compare_states(frame_record.state, live_state, metadata.ports, compare_frame_number=True)
                comparison = {
                    "mode": mode,
                    "source_frame": source_frame,
                    "live_frame": live_state.frame,
                    "mismatch_count": len(mismatches),
                    "mismatches": mismatches[:10],
                }
                debug.write_frame(comparison)
                history.append(comparison)
                result.compared_frames += 1
                if mismatches:
                    previous_snapshot = sender.snapshot()
                    commands_this_frame = {
                        port: controller_snapshot(source_frames[source_frame].state.players[port].controller_state)
                        for port in metadata.ports
                    }
                    enriched = mismatches[0] | {
                        "mode": mode,
                        "source_frame": source_frame,
                        "live_frame": live_state.frame,
                        "submitted_inputs": commands_this_frame,
                        "previous_controller_state": previous_snapshot,
                        "recent_context": list(history),
                    }
                    result.mismatches.append(enriched)
                    logger.error("Mismatch: {}", json.dumps(enriched, default=json_default))
                    if stop_on_mismatch:
                        break

            next_source = source_frames.get(live_state.frame + 1)
            if next_source is not None:
                previous_snapshot = sender.snapshot()
                commands = sender.send_frame(source_controller_states(next_source.state, metadata.ports))
                debug.write_commands(
                    {
                        "mode": mode,
                        "live_frame": live_state.frame,
                        "source_frame": next_source.frame,
                        "previous_controller_state": previous_snapshot,
                        "commands": commands,
                    }
                )
            live_state = step_until(console, time.monotonic() + timeouts.frame_s, "live frame")

    finally:
        debug.close()
        for controller in controllers.values():
            controller.disconnect()
        console.stop()

    result.elapsed_s = time.monotonic() - run_start
    throughput = result.compared_frames / result.elapsed_s if result.elapsed_s else 0.0
    logger.info(
        "Finished mode={} compared_frames={} mismatches={} elapsed_s={:.3f} throughput_fps={:.2f}",
        mode,
        result.compared_frames,
        len(result.mismatches),
        result.elapsed_s,
        throughput,
    )
    return result


def run_reproduction(
    replay_path: Path,
    emulator_path: Path = Path(EMULATOR_PATH),
    iso_path: Path = Path(ISO_PATH),
    modes: tuple[str, ...] = ("normal", "ffw"),
    prefix_frames: int = 300,
    start_frame: int = 0,
    stop_on_mismatch: bool = True,
    debug_dir: Path | None = None,
    timeouts: TimeoutConfig | None = None,
    use_raw_main_stick: bool = False,
) -> list[ReproductionResult]:
    if timeouts is None:
        timeouts = TimeoutConfig()

    metadata, source_frames = read_source_replay(replay_path)
    if start_frame < metadata.first_frame:
        start_frame = metadata.first_frame
    if start_frame + prefix_frames > metadata.last_frame:
        raise ValueError(
            f"Requested frames [{start_frame}, {start_frame + prefix_frames}) exceed replay last frame {metadata.last_frame}"
        )

    logger.info(
        "Loaded replay metadata: stage={} ports={} characters={} costumes={} frames=[{}, {}]",
        metadata.stage,
        metadata.ports,
        metadata.characters,
        metadata.costumes,
        metadata.first_frame,
        metadata.last_frame,
    )
    return [
        run_mode(
            mode=mode,
            metadata=metadata,
            source_frames=source_frames,
            emulator_path=emulator_path,
            iso_path=iso_path,
            prefix_frames=prefix_frames,
            start_frame=start_frame,
            stop_on_mismatch=stop_on_mismatch,
            debug_dir=debug_dir,
            timeouts=timeouts,
            use_raw_main_stick=use_raw_main_stick,
        )
        for mode in modes
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a .slp through libmelee controllers and compare game states.")
    parser.add_argument("replay", type=Path, help="Path to the source .slp replay")
    parser.add_argument("--emulator", type=Path, default=Path(EMULATOR_PATH), help="Path to Dolphin/AppRun")
    parser.add_argument("--iso", type=Path, default=Path(ISO_PATH), help="Path to the Melee ISO")
    parser.add_argument("--mode", choices=("normal", "ffw", "both"), default="both")
    parser.add_argument("--prefix-frames", type=int, default=300)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--debug-dir", type=Path, default=None)
    parser.add_argument("--stop-on-mismatch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-mismatch", dest="stop_on_mismatch", action="store_false")
    parser.add_argument(
        "--use-raw-main-stick",
        action="store_true",
        help="Inject slp 0x3B/0x40 directly (experimental; see README).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    modes = ("normal", "ffw") if args.mode == "both" else (args.mode,)
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    results = run_reproduction(
        replay_path=args.replay,
        emulator_path=args.emulator,
        iso_path=args.iso,
        modes=modes,
        prefix_frames=args.prefix_frames,
        start_frame=args.start_frame,
        stop_on_mismatch=args.stop_on_mismatch,
        debug_dir=args.debug_dir,
        use_raw_main_stick=args.use_raw_main_stick,
    )
    return 1 if any(result.mismatches for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
