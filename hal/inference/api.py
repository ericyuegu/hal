"""Model-independent input and output contract for live HAL policies."""

import math
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from numbers import Real
from typing import Protocol
from typing import runtime_checkable

from hal.controller import POLICY_BUTTON_MASK
from hal.controller import ControllerAction

ObservationScalar = float | int


@dataclass(frozen=True, slots=True)
class PolicySpec:
    """Static requirements declared by one loaded policy."""

    name: str
    backend: str
    required_observation_fields: tuple[str, ...]
    supported_transport_delays: tuple[int, ...]
    requires_player_code: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("policy name must be non-empty")
        if not self.backend:
            raise ValueError("policy backend must be non-empty")
        if len(set(self.required_observation_fields)) != len(self.required_observation_fields):
            raise ValueError("required observation fields must be unique")
        if not self.supported_transport_delays:
            raise ValueError("policy must support at least one transport delay")
        if any(
            not isinstance(delay, int) or isinstance(delay, bool) or delay < 0
            for delay in self.supported_transport_delays
        ):
            raise ValueError("supported transport delays must be non-negative integers")
        if tuple(sorted(set(self.supported_transport_delays))) != self.supported_transport_delays:
            raise ValueError("supported transport delays must be sorted and unique")


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Run shape fixed before a policy starts compiling or accumulating state."""

    max_batch_size: int
    transport_delay_frames: int
    replan_interval_frames: int | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_batch_size, int)
            or isinstance(self.max_batch_size, bool)
            or self.max_batch_size < 1
        ):
            raise ValueError("max_batch_size must be a positive integer")
        if (
            not isinstance(self.transport_delay_frames, int)
            or isinstance(self.transport_delay_frames, bool)
            or self.transport_delay_frames < 0
        ):
            raise ValueError("transport_delay_frames must be a non-negative integer")
        if self.replan_interval_frames is not None and (
            not isinstance(self.replan_interval_frames, int)
            or isinstance(self.replan_interval_frames, bool)
            or self.replan_interval_frames < 1
        ):
            raise ValueError("replan_interval_frames must be a positive integer or None")


@dataclass(frozen=True, slots=True)
class PolicyInput:
    """One observed stream and the exact controller actions around it.

    ``applied_action`` produced this observation. ``pending_actions`` are in
    execution order for the next transport-delayed frames.
    """

    stream_id: int
    frame_id: int
    controlled_port: int
    observation: Mapping[str, ObservationScalar]
    applied_action: ControllerAction
    pending_actions: tuple[ControllerAction, ...]
    player_code: str | None = None
    reset: bool = False


@dataclass(frozen=True, slots=True)
class PolicyOutput:
    """The newly submitted controller action for one policy stream."""

    stream_id: int
    action: ControllerAction


@runtime_checkable
class Policy(Protocol):
    """A batched stateful policy with explicit preparation."""

    @property
    def spec(self) -> PolicySpec: ...

    def prepare(self, config: RuntimeConfig) -> None: ...

    def step(self, inputs: Sequence[PolicyInput]) -> Sequence[PolicyOutput]: ...


def validate_controller_action(action: ControllerAction) -> None:
    """Reject values that cannot represent a logical GameCube controller."""
    analog = {
        "main_x": (action.main_x, -1.0, 1.0),
        "main_y": (action.main_y, -1.0, 1.0),
        "c_x": (action.c_x, -1.0, 1.0),
        "c_y": (action.c_y, -1.0, 1.0),
        "trigger_l": (action.trigger_l, 0.0, 1.0),
        "trigger_r": (action.trigger_r, 0.0, 1.0),
    }
    for name, (value, lower, upper) in analog.items():
        if not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"controller {name} must be finite and in [{lower}, {upper}], got {value!r}")
    if not isinstance(action.buttons, int) or isinstance(action.buttons, bool) or not 0 <= action.buttons <= 0xFFFF:
        raise ValueError(f"controller buttons must be a uint16 bitmask, got {action.buttons!r}")
    if action.buttons & ~POLICY_BUTTON_MASK:
        raise ValueError(f"controller buttons contain unsupported bits 0x{action.buttons & ~POLICY_BUTTON_MASK:04x}")


def validate_policy_inputs(spec: PolicySpec, config: RuntimeConfig, inputs: Sequence[PolicyInput]) -> None:
    """Validate one batch before it crosses a backend boundary."""
    if config.transport_delay_frames not in spec.supported_transport_delays:
        raise ValueError(
            f"policy {spec.name!r} does not support transport delay {config.transport_delay_frames}; "
            f"supported delays are {spec.supported_transport_delays}"
        )
    if not inputs or len(inputs) > config.max_batch_size:
        raise ValueError(f"policy batch size must be in [1, {config.max_batch_size}], got {len(inputs)}")
    stream_ids = [item.stream_id for item in inputs]
    if len(set(stream_ids)) != len(stream_ids):
        raise ValueError("policy batch contains duplicate stream IDs")
    required = set(spec.required_observation_fields)
    for item in inputs:
        if not isinstance(item.stream_id, int) or isinstance(item.stream_id, bool):
            raise ValueError(f"policy stream_id must be an integer, got {item.stream_id!r}")
        if not isinstance(item.frame_id, int) or isinstance(item.frame_id, bool):
            raise ValueError(f"stream {item.stream_id} frame_id must be an integer, got {item.frame_id!r}")
        if item.controlled_port not in (1, 2):
            raise ValueError(f"stream {item.stream_id} controls unsupported port {item.controlled_port}")
        if not isinstance(item.reset, bool):
            raise ValueError(f"stream {item.stream_id} reset must be a boolean")
        if len(item.pending_actions) != config.transport_delay_frames:
            raise ValueError(
                f"stream {item.stream_id} has {len(item.pending_actions)} pending actions; "
                f"expected {config.transport_delay_frames}"
            )
        missing = required - item.observation.keys()
        if missing:
            raise ValueError(f"stream {item.stream_id} is missing observation fields {sorted(missing)}")
        for name, value in item.observation.items():
            if not isinstance(name, str):
                raise ValueError(f"stream {item.stream_id} has a non-string observation field")
            if not isinstance(value, (Real, Integral)) or isinstance(value, bool):
                raise ValueError(f"stream {item.stream_id} observation {name!r} must be numeric, got {value!r}")
            if isinstance(value, Real) and math.isinf(float(value)):
                raise ValueError(f"stream {item.stream_id} observation {name!r} must not be infinite")
        if spec.requires_player_code and not item.player_code:
            raise ValueError(f"stream {item.stream_id} requires a player code")
        validate_controller_action(item.applied_action)
        for action in item.pending_actions:
            validate_controller_action(action)


def validate_policy_outputs(
    inputs: Sequence[PolicyInput],
    outputs: Sequence[PolicyOutput],
) -> dict[int, ControllerAction]:
    """Return outputs by stream after checking exact batch correspondence."""
    expected = {item.stream_id for item in inputs}
    actual = [item.stream_id for item in outputs]
    if len(set(actual)) != len(actual):
        raise ValueError("policy returned duplicate stream IDs")
    if set(actual) != expected:
        raise ValueError(f"policy returned streams {sorted(actual)}, expected {sorted(expected)}")
    result = {}
    for output in outputs:
        validate_controller_action(output.action)
        result[output.stream_id] = output.action
    return result
