"""Unit tests for the controller-input wire path (no Dolphin).

The ``_dolphin_*_byte`` helpers mirror the emulator side of the pipe protocol
(``Pipes.cpp::ParseCommand``/``SetAxis`` + the GCPad byte conversion) so the
libmelee converters are pinned against the exact math Dolphin runs. If either
side changes, these tests are the first tripwire before the emulator sweep.
"""

import math

import melee
import numpy as np

from hal.sim.inputs import ControllerInputsValue
from hal.sim.inputs import MdsControllerView
from hal.sim.inputs import apply_inputs
from hal.wire import BUTTON_BITS


def _dolphin_trigger_byte(wire: float) -> int:
    """``SET {L,R} <wire>`` → trigger byte: ParseCommand recenters to
    wire/2 + 0.5, SetAxis takes the positive half-axis, GCPad scales by 255."""
    value = min(max(wire / 2.0 + 0.5, 0.0), 1.0)
    hi = max(0.0, value - 0.5) * 2.0
    return int(hi * 255)


def _dolphin_stick_byte(wire: float) -> int:
    """``SET {MAIN,C} <wire> <wire>`` → int8 stick byte: floor((wire-0.5)*254)."""
    return math.floor((min(max(wire, 0.0), 1.0) - 0.5) * 254)


def test_trigger_wire_recovers_every_byte_on_the_140_grid() -> None:
    """Melee saturates triggers at byte 140 (physical = byte/140), so a stored
    trigger k/140 must come back as byte k bit-exactly."""
    for byte in range(141):
        wire = melee.controller.fix_analog_trigger(byte / 140.0)
        assert _dolphin_trigger_byte(wire) == byte, f"trigger byte {byte} mangled"


def test_stick_wire_recovers_every_byte_on_the_80_grid() -> None:
    """Melee's logical stick is byte/80 (clamped at ±80), so a stored logical
    value b/80 must come back as byte b bit-exactly."""
    for byte in range(-80, 81):
        wire = melee.controller.fix_analog_stick_signed(byte / 80.0)
        assert _dolphin_stick_byte(wire) == byte, f"stick byte {byte} mangled"


def _minimal_columns(prefix: str) -> dict[str, np.ndarray]:
    """Single-frame column dict satisfying MdsControllerView property reads."""
    cols: dict[str, np.ndarray] = {
        f"{prefix}_main_stick_x": np.array([0.5], dtype=np.float32),
        f"{prefix}_main_stick_y": np.array([-0.25], dtype=np.float32),
        f"{prefix}_c_stick_x": np.array([0.0], dtype=np.float32),
        f"{prefix}_c_stick_y": np.array([0.0], dtype=np.float32),
        f"{prefix}_trigger_l": np.array([0.5], dtype=np.float32),
        f"{prefix}_trigger_r": np.array([0.0], dtype=np.float32),
    }
    for b in BUTTON_BITS:
        cols[f"{prefix}_button_{b}"] = np.array([0], dtype=np.int32)
    cols[f"{prefix}_button_a"] = np.array([1], dtype=np.int32)
    return cols


def test_view_reads_logical_columns() -> None:
    view = MdsControllerView(columns=_minimal_columns("p1"), port_prefix="p1", frame_idx=0)
    assert view.main_x == 0.5
    assert view.main_y == -0.25
    assert view.trigger_l == 0.5
    assert view.trigger_r == 0.0
    assert view.buttons == BUTTON_BITS["a"]


class _RecordingSink:
    """ControllerSink double recording every setter call."""

    def __init__(self) -> None:
        self.pressed: list[melee.enums.Button] = []
        self.released: list[melee.enums.Button] = []
        self.tilts: dict[melee.enums.Button, tuple[float, float]] = {}
        self.shoulders: dict[melee.enums.Button, float] = {}

    def press_button(self, button: melee.enums.Button) -> None:
        self.pressed.append(button)

    def release_button(self, button: melee.enums.Button) -> None:
        self.released.append(button)

    def tilt_analog(self, button: melee.enums.Button, x: float, y: float) -> None:
        self.tilts[button] = (x, y)

    def press_shoulder(self, button: melee.enums.Button, amount: float) -> None:
        self.shoulders[button] = amount


def test_apply_inputs_converts_logical_to_wire_and_dispatches_buttons() -> None:
    sink = _RecordingSink()
    src = ControllerInputsValue(
        main_x=0.5,
        main_y=-0.25,
        c_x=1.0,
        c_y=0.0,
        trigger_l=1.0,
        trigger_r=0.0,
        buttons=BUTTON_BITS["a"] | BUTTON_BITS["l"],
    )
    apply_inputs(sink, src)  # type: ignore[arg-type]  # ControllerSink double

    fix = melee.controller.fix_analog_stick_signed
    assert sink.tilts[melee.enums.Button.BUTTON_MAIN] == (fix(0.5), fix(-0.25))
    assert sink.tilts[melee.enums.Button.BUTTON_C] == (fix(1.0), fix(0.0))
    assert sink.shoulders[melee.enums.Button.BUTTON_L] == melee.controller.fix_analog_trigger(1.0)
    assert sink.shoulders[melee.enums.Button.BUTTON_R] == melee.controller.fix_analog_trigger(0.0)
    assert set(sink.pressed) == {melee.enums.Button.BUTTON_A, melee.enums.Button.BUTTON_L}
    # Every non-pressed button is explicitly released (no stale carry-over).
    assert len(sink.pressed) + len(sink.released) == len(BUTTON_BITS)
