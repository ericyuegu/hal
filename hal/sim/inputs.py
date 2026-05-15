"""Per-frame controller-input value objects + libmelee setter dispatch.

``ControllerInputs`` (Protocol) is the structural contract; ``MdsControllerView``
(zero-copy view over MDS columns) and ``ControllerInputsValue`` (frozen
dataclass for synthesized inputs) both satisfy it. ``apply_inputs`` is
duck-typed against the protocol.

Stick path: prefer ``raw`` int8 bytes per-axis when the slp recorded them;
otherwise fall back to the ``logical`` float via libmelee's
``fix_analog_stick``. The Controller is constructed with
``fix_analog_inputs=False`` so the composed wire byte reaches Dolphin
unmodified. Slp-version gating and the fallback-path physics drift are
documented in CLAUDE.md (Architecture → Footguns).
"""

from dataclasses import dataclass
from typing import Literal
from typing import Protocol
from typing import runtime_checkable

import melee
import numpy as np

from hal.wire import BUTTON_BITS
from hal.wire import MASK_INT8
from hal.wire import raw_byte_to_wire
from hal.wire import slp_button_to_melee

# Pre-resolved (bit, libmelee enum) pairs for the per-frame press/release
# dispatch. Derived from wire.BUTTON_BITS so MDS columns and live punches
# share one canonical bit layout. Order matters only for diagnostics;
# press_button / release_button are commutative within a frame.
_BUTTON_DISPATCH: tuple[tuple[int, melee.enums.Button], ...] = tuple(
    (bit, slp_button_to_melee(name)) for name, bit in BUTTON_BITS.items()
)


@runtime_checkable
class ControllerInputs(Protocol):
    """Structural protocol for one frame of controller state for one port."""

    main_x: float
    main_y: float
    c_x: float
    c_y: float
    trigger_l: float
    trigger_r: float
    buttons: int  # uint16 bitmask matching wire.BUTTON_BITS
    raw_main_x: int  # int8 from slp (>= 1.2.0), or wire.MASK_INT8 if unrecorded
    raw_main_y: int  # int8 from slp (>= 3.15.0), or wire.MASK_INT8 if unrecorded
    raw_c_x: int  # int8 from slp (>= 3.17.0), or wire.MASK_INT8 if unrecorded
    raw_c_y: int  # int8 from slp (>= 3.17.0), or wire.MASK_INT8 if unrecorded


@runtime_checkable
class ControllerSink(Protocol):
    """Structural protocol for the subset of ``melee.Controller`` that
    ``apply_inputs`` and ``ReplayControllerSender`` invoke. Lets test doubles
    stand in without inheriting libmelee's Console-bound base class."""

    def press_button(self, button: melee.enums.Button) -> None: ...
    def release_button(self, button: melee.enums.Button) -> None: ...
    def tilt_analog(self, button: melee.enums.Button, x: float, y: float) -> None: ...
    def press_shoulder(self, button: melee.enums.Button, amount: float) -> None: ...


@dataclass(frozen=True, slots=True)
class ControllerInputsValue:
    """Concrete value object satisfying ControllerInputs.

    Used by sources that produce inputs from scratch (model output, scripted
    sequences, .slp random-access). For MDS playback prefer ``MdsControllerView``
    — it aliases the underlying NumPy arrays without copying.
    """

    main_x: float
    main_y: float
    c_x: float
    c_y: float
    trigger_l: float
    trigger_r: float
    buttons: int
    raw_main_x: int = MASK_INT8
    raw_main_y: int = MASK_INT8
    raw_c_x: int = MASK_INT8
    raw_c_y: int = MASK_INT8


# Frozen: one instance per (port, frame) is fine. ``dataclass(frozen=True,
# slots=True)`` construction is sub-microsecond; per-frame view allocation is
# dominated by the libmelee pipe write and Dolphin frame budget.
@dataclass(frozen=True, slots=True)
class MdsControllerView:
    """Zero-copy view over MDS columns at a given frame index.

    Field access reads ``columns[f"{port_prefix}_{name}"][frame_idx]`` — no per-
    field copy beyond the NumPy 0-d scalar Python wraps it in. ``buttons`` is
    re-derived from the 9 single-bit columns each access; this is cheap (9
    indexes + bit-or ≈ ns) and keeps the schema unchanged.
    """

    columns: dict[str, np.ndarray]
    port_prefix: Literal["p1", "p2"]
    frame_idx: int

    @property
    def main_x(self) -> float:
        return float(self.columns[f"{self.port_prefix}_main_stick_x"][self.frame_idx])

    @property
    def main_y(self) -> float:
        return float(self.columns[f"{self.port_prefix}_main_stick_y"][self.frame_idx])

    @property
    def c_x(self) -> float:
        return float(self.columns[f"{self.port_prefix}_c_stick_x"][self.frame_idx])

    @property
    def c_y(self) -> float:
        return float(self.columns[f"{self.port_prefix}_c_stick_y"][self.frame_idx])

    @property
    def trigger_l(self) -> float:
        return float(self.columns[f"{self.port_prefix}_trigger_l_physical"][self.frame_idx])

    @property
    def trigger_r(self) -> float:
        return float(self.columns[f"{self.port_prefix}_trigger_r_physical"][self.frame_idx])

    @property
    def buttons(self) -> int:
        mask = 0
        for name, bit in BUTTON_BITS.items():
            if self.columns[f"{self.port_prefix}_button_{name}"][self.frame_idx]:
                mask |= bit
        return mask

    @property
    def raw_main_x(self) -> int:
        col = self.columns.get(f"{self.port_prefix}_main_stick_raw_x")
        return int(col[self.frame_idx]) if col is not None else MASK_INT8

    @property
    def raw_main_y(self) -> int:
        col = self.columns.get(f"{self.port_prefix}_main_stick_raw_y")
        return int(col[self.frame_idx]) if col is not None else MASK_INT8

    @property
    def raw_c_x(self) -> int:
        col = self.columns.get(f"{self.port_prefix}_c_stick_raw_x")
        return int(col[self.frame_idx]) if col is not None else MASK_INT8

    @property
    def raw_c_y(self) -> int:
        col = self.columns.get(f"{self.port_prefix}_c_stick_raw_y")
        return int(col[self.frame_idx]) if col is not None else MASK_INT8


def apply_inputs(controller: melee.Controller, src: ControllerInputs) -> None:
    """Punch one frame of inputs into a libmelee Controller.

    Setters write directly to the named pipe; ``Console.step()`` flushes — do
    not call ``flush()`` here. The button loop unconditionally presses or
    releases every button so we don't carry stale state from a previous source.
    """
    main_x_wire = _stick_axis_wire(src.raw_main_x, src.main_x)
    main_y_wire = _stick_axis_wire(src.raw_main_y, src.main_y)
    controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, main_x_wire, main_y_wire)

    c_x_wire = _stick_axis_wire(src.raw_c_x, src.c_x)
    c_y_wire = _stick_axis_wire(src.raw_c_y, src.c_y)
    controller.tilt_analog(melee.enums.Button.BUTTON_C, c_x_wire, c_y_wire)
    controller.press_shoulder(melee.enums.Button.BUTTON_L, melee.controller.fix_analog_trigger(src.trigger_l))
    controller.press_shoulder(melee.enums.Button.BUTTON_R, melee.controller.fix_analog_trigger(src.trigger_r))

    buttons = src.buttons
    for bit, button in _BUTTON_DISPATCH:
        if buttons & bit:
            controller.press_button(button)
        else:
            controller.release_button(button)


def _stick_axis_wire(raw_byte: int, logical: float) -> float:
    """Pick the per-axis wire float: raw byte if recorded, else logical fallback.

    Both paths delegate to libmelee — raw to ``raw_byte_to_wire``, logical to
    ``fix_analog_stick_signed`` (the signed [-1, 1] entry point matching peppi's
    logical stick range).
    """
    if raw_byte != MASK_INT8:
        return raw_byte_to_wire(raw_byte)
    return melee.controller.fix_analog_stick_signed(logical)
