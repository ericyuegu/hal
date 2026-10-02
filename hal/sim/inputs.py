"""Per-frame controller-input value objects + libmelee setter dispatch.

``ControllerAction`` and the stateful MDS source both satisfy the
``ControllerInputs`` contract used by ``apply_inputs``.

All values are logical (game-causal): sticks in [-1, 1], triggers in [0, 1],
buttons a wire-bitmask. ``fix_analog_stick_signed`` / ``fix_analog_trigger``
in libmelee own the wire conversion; the Controller is constructed with
``fix_analog_inputs=False`` so the converted value reaches Dolphin unmodified.
See ``hal.wire`` for the controller data model.
"""

import math
from collections import deque
from collections.abc import Iterable
from collections.abc import Mapping
from numbers import Integral
from numbers import Real
from typing import Protocol
from typing import cast
from typing import runtime_checkable

import melee

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.controller import POLICY_BUTTON_MASK
from hal.controller import ControllerAction
from hal.controller import ControllerInputs
from hal.controller import controller_action_wire_values
from hal.wire import ACTION_CHANNELS
from hal.wire import BUTTON_BITS

# Pre-resolved (bit, libmelee enum) pairs for the per-frame press/release
# dispatch. Derived from wire.BUTTON_BITS so MDS columns and live punches
# share one canonical bit layout. Order matters only for diagnostics;
# press_button / release_button are commutative within a frame.
_POLICY_BUTTON_NAMES = tuple(channel.removeprefix("button_") for channel in ACTION_CHANNELS[6:])
_BUTTON_DISPATCH: tuple[tuple[int, melee.enums.Button], ...] = tuple(
    (BUTTON_BITS[name], melee.enums.Button[f"BUTTON_{name.upper()}"]) for name in _POLICY_BUTTON_NAMES
)
if sum(bit for bit, _button in _BUTTON_DISPATCH) != POLICY_BUTTON_MASK:
    raise RuntimeError("controller action schema and Slippi button wire disagree")


@runtime_checkable
class ControllerSink(Protocol):
    """Structural protocol for the subset of ``melee.Controller`` that
    ``apply_inputs`` and ``ReplayControllerSender`` invoke. Lets test doubles
    stand in without inheriting libmelee's Console-bound base class."""

    def press_button(self, button: melee.enums.Button) -> None: ...
    def release_button(self, button: melee.enums.Button) -> None: ...
    def tilt_analog(self, button: melee.enums.Button, x: float, y: float) -> None: ...
    def press_shoulder(self, button: melee.enums.Button, amount: float) -> None: ...


class ActionTransport:
    """Apply a physical controller delay to submitted logical actions."""

    def __init__(self, delay_frames: int) -> None:
        if not isinstance(delay_frames, int) or isinstance(delay_frames, bool) or delay_frames < 0:
            raise ValueError("delay_frames must be a non-negative integer")
        self.delay_frames = delay_frames
        self._pending: deque[ControllerAction] = deque()
        self.reset()

    @property
    def pending(self) -> tuple[ControllerAction, ...]:
        return tuple(self._pending)

    def reset(self, actions: Iterable[ControllerAction] | None = None) -> None:
        values = tuple(actions) if actions is not None else (NEUTRAL_CONTROLLER_ACTION,) * self.delay_frames
        if len(values) != self.delay_frames:
            raise ValueError(f"transport reset needs {self.delay_frames} actions, got {len(values)}")
        self._pending = deque(values)

    def submit(self, action: ControllerAction) -> ControllerAction:
        """Commit one output and return the action due in the next state."""
        if not self.delay_frames:
            return action
        due = self._pending.popleft()
        self._pending.append(action)
        return due


def _read_canonical_pair(pre: Mapping[str, object], name: str, first: str, second: str) -> tuple[float, float]:
    value = pre[name]
    if not isinstance(value, Mapping):
        raise ValueError(f"canonical pre.{name} must be an object")
    fields = cast(Mapping[str, object], value)
    try:
        first_value = fields[first]
        second_value = fields[second]
    except KeyError as error:
        raise ValueError(f"canonical pre.{name} must contain numeric {first} and {second}") from error
    if (
        not isinstance(first_value, Real)
        or isinstance(first_value, bool)
        or not isinstance(second_value, Real)
        or isinstance(second_value, bool)
    ):
        raise ValueError(f"canonical pre.{name} must contain numeric {first} and {second}")
    return float(first_value), float(second_value)


def canonical_pre_to_action(pre: Mapping[str, object]) -> ControllerAction:
    """Read the controller state recorded on one canonical Slippi frame."""
    main_x, main_y = _read_canonical_pair(pre, "joystick", "x", "y")
    c_x, c_y = _read_canonical_pair(pre, "cstick", "x", "y")
    trigger_l, trigger_r = _read_canonical_pair(pre, "triggers_physical", "l", "r")
    try:
        buttons_value = pre["buttons_physical"]
    except KeyError as error:
        raise ValueError("canonical pre.buttons_physical must be an integer") from error
    if not isinstance(buttons_value, Integral) or isinstance(buttons_value, bool):
        raise ValueError("canonical pre.buttons_physical must be an integer")
    buttons = int(buttons_value)
    return ControllerAction(main_x, main_y, c_x, c_y, trigger_l, trigger_r, buttons)


def controller_action_readback(action: ControllerInputs) -> ControllerAction:
    """Predict Slippi's processed sticks for the pinned libmelee input path.

    Melee clamps each stick to radius 80, truncates to signed bytes, then
    zeros axes below byte 23. Commands and forced prefixes must retain their
    original values; this conversion is only for checking observed inputs.
    """
    main_x, main_y, c_x, c_y, _left, _right, _buttons = controller_action_wire_values(action)
    sticks: list[float] = []
    for x, y in ((main_x, main_y), (c_x, c_y)):
        radius = math.hypot(x, y)
        if radius > 80:
            x, y = int(x * 80 / radius), int(y * 80 / radius)
        sticks.extend((0.0 if abs(x) < 23 else x / 80, 0.0 if abs(y) < 23 else y / 80))
    return ControllerAction(
        sticks[0], sticks[1], sticks[2], sticks[3], action.trigger_l, action.trigger_r, action.buttons
    )


def controller_actions_match(
    expected: ControllerInputs,
    actual: ControllerInputs,
    *,
    analog_tolerance: float = 0.02,
) -> bool:
    """Compare a punched action with its quantized Slippi representation."""
    return (
        abs(expected.main_x - actual.main_x) <= analog_tolerance
        and abs(expected.main_y - actual.main_y) <= analog_tolerance
        and abs(expected.c_x - actual.c_x) <= analog_tolerance
        and abs(expected.c_y - actual.c_y) <= analog_tolerance
        and abs(expected.trigger_l - actual.trigger_l) <= analog_tolerance
        and abs(expected.trigger_r - actual.trigger_r) <= analog_tolerance
        and expected.buttons == actual.buttons
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
    if not isinstance(src.buttons, Integral) or isinstance(src.buttons, bool):
        raise ValueError(f"controller input buttons must be an integer, got {src.buttons!r}")
    buttons = int(src.buttons)
    unknown_buttons = buttons & ~POLICY_BUTTON_MASK
    if unknown_buttons:
        raise ValueError(f"controller input has unsupported button bits 0x{unknown_buttons:04x}")

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

    for bit, button in _BUTTON_DISPATCH:
        if buttons & bit:
            controller.press_button(button)
        else:
            controller.release_button(button)
