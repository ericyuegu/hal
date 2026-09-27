"""Controller values and pure wire conversions shared by policies and simulators."""

import math
from dataclasses import dataclass
from numbers import Real
from typing import Final
from typing import Protocol
from typing import runtime_checkable

import numpy as np
from numpy.typing import DTypeLike

from hal.wire import ACTION_CHANNELS
from hal.wire import ACTION_DIM
from hal.wire import BUTTON_BITS

# Buttons represented by ``hal.controller.v1``. START is intentionally absent.
POLICY_BUTTON_MASK: Final[int] = 0x0F78


@runtime_checkable
class ControllerInputs(Protocol):
    """One frame of logical GameCube controller state."""

    main_x: float
    main_y: float
    c_x: float
    c_y: float
    trigger_l: float
    trigger_r: float
    buttons: int


@dataclass(frozen=True, slots=True)
class ControllerAction:
    """Concrete semantic controller action at the public policy boundary."""

    main_x: float
    main_y: float
    c_x: float
    c_y: float
    trigger_l: float
    trigger_r: float
    buttons: int


NEUTRAL_CONTROLLER_ACTION = ControllerAction(
    main_x=0.0,
    main_y=0.0,
    c_x=0.0,
    c_y=0.0,
    trigger_l=0.0,
    trigger_r=0.0,
    buttons=0,
)


def validate_controller_action(action: ControllerAction) -> None:
    """Reject values that cannot represent a logical GameCube controller."""
    if not isinstance(action, ControllerAction):
        raise ValueError("controller action must be a ControllerAction record")
    analog = {
        "main_x": (action.main_x, -1.0, 1.0),
        "main_y": (action.main_y, -1.0, 1.0),
        "c_x": (action.c_x, -1.0, 1.0),
        "c_y": (action.c_y, -1.0, 1.0),
        "trigger_l": (action.trigger_l, 0.0, 1.0),
        "trigger_r": (action.trigger_r, 0.0, 1.0),
    }
    for name, (value, lower, upper) in analog.items():
        if (
            not isinstance(value, Real)
            or isinstance(value, bool)
            or not math.isfinite(value)
            or not lower <= value <= upper
        ):
            raise ValueError(f"controller {name} must be finite and in [{lower}, {upper}], got {value!r}")
    if not isinstance(action.buttons, int) or isinstance(action.buttons, bool) or not 0 <= action.buttons <= 0xFFFF:
        raise ValueError(f"controller buttons must be a uint16 bitmask, got {action.buttons!r}")
    if action.buttons & ~POLICY_BUTTON_MASK:
        raise ValueError(f"controller buttons contain unsupported bits 0x{action.buttons & ~POLICY_BUTTON_MASK:04x}")


def controller_to_action_vec(action: ControllerInputs, *, dtype: DTypeLike = np.float64) -> np.ndarray:
    """Encode logical inputs in canonical policy channel order."""
    return np.asarray(
        [
            action.main_x,
            action.main_y,
            action.c_x,
            action.c_y,
            action.trigger_l,
            action.trigger_r,
            *(float(bool(action.buttons & BUTTON_BITS[name.removeprefix("button_")])) for name in ACTION_CHANNELS[6:]),
        ],
        dtype=dtype,
    )


def action_vec_to_controller(action: np.ndarray) -> ControllerAction:
    """Convert one canonical policy action vector to logical controller inputs.

    This codec is Torch-free so spawned Session workers do not import the model.  ``hal.wire.ACTION_CHANNELS`` defines the
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
    return ControllerAction(
        main_x=float(np.clip(values[0], -1.0, 1.0)),
        main_y=float(np.clip(values[1], -1.0, 1.0)),
        c_x=float(np.clip(values[2], -1.0, 1.0)),
        c_y=float(np.clip(values[3], -1.0, 1.0)),
        trigger_l=float(np.clip(values[4], 0.0, 1.0)),
        trigger_r=float(np.clip(values[5], 0.0, 1.0)),
        buttons=int(buttons),
    )


def controller_action_wire_values(action: ControllerInputs) -> tuple[int, int, int, int, int, int, int]:
    """Quantize commands using the pinned libmelee 0.47.0+hal.realtime.1 map.

    This compares commanded bytes. Slippi observations may have passed through
    Melee's deadzones and cannot always recover the original commanded bytes.
    """
    return (
        round(((float(action.main_x) + 1.0) / 2.0 - 0.5) * 160),
        round(((float(action.main_y) + 1.0) / 2.0 - 0.5) * 160),
        round(((float(action.c_x) + 1.0) / 2.0 - 0.5) * 160),
        round(((float(action.c_y) + 1.0) / 2.0 - 0.5) * 160),
        round(float(action.trigger_l) * 140),
        round(float(action.trigger_r) * 140),
        action.buttons,
    )


def controller_actions_equal_at_wire_precision(expected: ControllerInputs, actual: ControllerInputs) -> bool:
    return controller_action_wire_values(expected) == controller_action_wire_values(actual)
