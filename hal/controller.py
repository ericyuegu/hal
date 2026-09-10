"""Dependency-free controller values shared by policies and simulators."""

from dataclasses import dataclass
from typing import Final
from typing import Protocol
from typing import runtime_checkable

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
