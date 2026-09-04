"""Per-frame controller-input value objects + libmelee setter dispatch.

``ControllerInputs`` is the structural contract. ``ControllerInputsValue``
and the stateful MDS source both satisfy it. ``apply_inputs`` uses the protocol.

All values are logical (game-causal): sticks in [-1, 1], triggers in [0, 1],
buttons a wire-bitmask. ``fix_analog_stick_signed`` / ``fix_analog_trigger``
in libmelee own the wire conversion; the Controller is constructed with
``fix_analog_inputs=False`` so the converted value reaches Dolphin unmodified.
See ``hal.wire`` for the controller data model.
"""

import math
from dataclasses import dataclass
from typing import Protocol
from typing import runtime_checkable

import melee
import numpy as np

from hal.wire import ACTION_CHANNELS
from hal.wire import ACTION_DIM
from hal.wire import BUTTON_BITS
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

    Used by sources that produce inputs from scratch, such as models and scripts.
    """

    main_x: float
    main_y: float
    c_x: float
    c_y: float
    trigger_l: float
    trigger_r: float
    buttons: int


def action_vec_to_controller(action: np.ndarray) -> ControllerInputsValue:
    """Convert one canonical policy action vector to logical controller inputs.

    This codec is simulator-side and Torch-free so spawned Session workers do
    not import the training program.  ``hal.wire.ACTION_CHANNELS`` defines the
    channel order; START is intentionally absent from that wire.
    """
    values = np.asarray(action).reshape(-1)
    if values.shape != (ACTION_DIM,):
        raise ValueError(f"action has shape {values.shape}, expected {(ACTION_DIM,)}")
    buttons = 0
    for offset, channel in enumerate(ACTION_CHANNELS[6:]):
        name = channel.removeprefix("button_")
        if values[6 + offset] > 0.5:
            buttons |= BUTTON_BITS[name]
    return ControllerInputsValue(
        main_x=float(np.clip(values[0], -1.0, 1.0)),
        main_y=float(np.clip(values[1], -1.0, 1.0)),
        c_x=float(np.clip(values[2], -1.0, 1.0)),
        c_y=float(np.clip(values[3], -1.0, 1.0)),
        trigger_l=float(np.clip(values[4], 0.0, 1.0)),
        trigger_r=float(np.clip(values[5], 0.0, 1.0)),
        buttons=int(buttons),
    )


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"controller input {name} must be finite, got {value!r}")


def apply_inputs(controller: ControllerSink, src: ControllerInputs) -> None:
    """Punch one frame of inputs into a libmelee Controller.

    Setters write directly to the named pipe; ``Console.step()`` flushes — do
    not call ``flush()`` here. The button loop unconditionally presses or
    releases every button so we don't carry stale state from a previous source.
    """
    main_x = src.main_x
    main_y = src.main_y
    c_x = src.c_x
    c_y = src.c_y
    trigger_l = src.trigger_l
    trigger_r = src.trigger_r
    _require_finite("main_x", main_x)
    _require_finite("main_y", main_y)
    _require_finite("c_x", c_x)
    _require_finite("c_y", c_y)
    _require_finite("trigger_l", trigger_l)
    _require_finite("trigger_r", trigger_r)

    controller.tilt_analog(
        melee.enums.Button.BUTTON_MAIN,
        melee.controller.fix_analog_stick_signed(main_x),
        melee.controller.fix_analog_stick_signed(main_y),
    )
    controller.tilt_analog(
        melee.enums.Button.BUTTON_C,
        melee.controller.fix_analog_stick_signed(c_x),
        melee.controller.fix_analog_stick_signed(c_y),
    )
    controller.press_shoulder(melee.enums.Button.BUTTON_L, melee.controller.fix_analog_trigger(trigger_l))
    controller.press_shoulder(melee.enums.Button.BUTTON_R, melee.controller.fix_analog_trigger(trigger_r))

    buttons = src.buttons
    for bit, button in _BUTTON_DISPATCH:
        if buttons & bit:
            controller.press_button(button)
        else:
            controller.release_button(button)
