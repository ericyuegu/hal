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
    requires_player_identity: bool = False

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
    transport_delays: tuple[int, ...]
    replan_interval_frames: int | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_batch_size, int)
            or isinstance(self.max_batch_size, bool)
            or self.max_batch_size < 1
        ):
            raise ValueError("max_batch_size must be a positive integer")
        if not self.transport_delays:
            raise ValueError("transport_delays must be non-empty")
        if any(not isinstance(delay, int) or isinstance(delay, bool) or delay < 0 for delay in self.transport_delays):
            raise ValueError("transport_delays must contain non-negative integers")
        if tuple(sorted(set(self.transport_delays))) != self.transport_delays:
            raise ValueError("transport_delays must be sorted and unique")
        if self.replan_interval_frames is not None and (
            not isinstance(self.replan_interval_frames, int)
            or isinstance(self.replan_interval_frames, bool)
            or self.replan_interval_frames < 1
        ):
            raise ValueError("replan_interval_frames must be a positive integer or None")

    def require_single_delay(self) -> int:
        """Return the configured delay for a path that cannot mix delays."""
        if len(self.transport_delays) != 1:
            raise ValueError(f"this path requires one transport delay, got {self.transport_delays}")
        return self.transport_delays[0]


@dataclass(frozen=True, slots=True)
class PolicyInput:
    """One observed stream and the exact controller actions around it.

    ``applied_action`` produced this observation. ``pending_actions`` are in
    execution order for the next transport-delayed frames. ``player_identity``
    selects the behavior to imitate; it does not identify the live opponent.
    """

    stream_id: int
    frame_id: int
    controlled_port: int
    observation: Mapping[str, ObservationScalar]
    applied_action: ControllerAction
    pending_actions: tuple[ControllerAction, ...]
    player_identity: str | None = None
    desired_return: float | None = 20.0
    temperature: float = 1.0
    reset: bool = False


@dataclass(frozen=True, slots=True)
class PredictionRequest:
    """Advance one model stream and predict actions after its fixed prefix."""

    stream_id: int
    generation: int
    sequence: int
    source_frame: int
    observations: tuple[PolicyInput, ...]
    fixed_actions: tuple[ControllerAction, ...]

    def __post_init__(self) -> None:
        for name in ("stream_id", "generation", "sequence", "source_frame"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"prediction {name} must be an integer")
        if self.generation < 0 or self.sequence < 0:
            raise ValueError("prediction generation and sequence must be non-negative")
        if not self.observations or self.observations[-1].frame_id != self.source_frame:
            raise ValueError("prediction observations must end at the source frame")
        for index, item in enumerate(self.observations):
            if item.stream_id != self.stream_id:
                raise ValueError("prediction observations contain another stream")
            if index and item.frame_id != self.observations[index - 1].frame_id + 1:
                raise ValueError("prediction observations must be contiguous")
        for action in self.fixed_actions:
            validate_controller_action(action)


@dataclass(frozen=True, slots=True)
class FrameAction:
    target_frame: int
    action: ControllerAction


@dataclass(frozen=True, slots=True)
class ActionPlan:
    """New predictions only; fixed actions remain on the request."""

    stream_id: int
    generation: int
    sequence: int
    source_frame: int
    actions: tuple[FrameAction, ...]


def action_plan(request: PredictionRequest, tail: Sequence[ControllerAction]) -> ActionPlan:
    first = request.source_frame + len(request.fixed_actions) + 1
    return ActionPlan(
        request.stream_id,
        request.generation,
        request.sequence,
        request.source_frame,
        tuple(FrameAction(first + index, action) for index, action in enumerate(tail)),
    )


def contiguous_horizons(offsets: tuple[int, ...]) -> tuple[int, ...]:
    """Return horizons whose prediction heads start at frame one without gaps."""
    count = 0
    for expected, actual in enumerate(offsets, start=1):
        if actual != expected:
            break
        count += 1
    return tuple(range(1, count + 1))


def validate_prediction_request(
    spec: PolicySpec,
    runtime: RuntimeConfig,
    request: PredictionRequest,
    *,
    context_frames: int,
    prefix_frames: int,
) -> None:
    """Validate a complete or incremental observation request."""
    if len(request.observations) > context_frames or len(request.fixed_actions) != prefix_frames:
        raise ValueError("prediction observations or fixed actions differ from the prepared shape")
    for item in request.observations:
        validate_policy_inputs(spec, runtime, (item,))


def validate_action_plan(request: PredictionRequest, plan: ActionPlan, horizon: int) -> None:
    if (plan.stream_id, plan.generation, plan.sequence, plan.source_frame) != (
        request.stream_id,
        request.generation,
        request.sequence,
        request.source_frame,
    ):
        raise ValueError("action plan does not match its request")
    if len(plan.actions) != horizon - len(request.fixed_actions):
        raise ValueError("action plan has the wrong prediction horizon")
    first = request.source_frame + len(request.fixed_actions) + 1
    for index, item in enumerate(plan.actions):
        if item.target_frame != first + index:
            raise ValueError("action plan target frames are not contiguous")
        validate_controller_action(item.action)


@runtime_checkable
class PredictionPolicy(Protocol):
    """A stateful model that predicts actions for one or more streams."""

    @property
    def spec(self) -> PolicySpec: ...

    @property
    def sampling_seed(self) -> int: ...

    @property
    def context_frames(self) -> int: ...

    @property
    def supported_horizons(self) -> tuple[int, ...]: ...

    @property
    def prediction_horizon(self) -> int: ...

    def prepare_prediction(self, runtime: RuntimeConfig, horizon: int, prefix_frames: int) -> None: ...

    def reset_prediction(self) -> None: ...

    def predict(self, requests: Sequence[PredictionRequest]) -> Sequence[ActionPlan]: ...


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
    unsupported = set(config.transport_delays) - set(spec.supported_transport_delays)
    if unsupported:
        raise ValueError(
            f"policy {spec.name!r} does not support transport delays {sorted(unsupported)}; "
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
        delay = len(item.pending_actions)
        if delay not in config.transport_delays:
            raise ValueError(
                f"stream {item.stream_id} has {delay} pending actions; prepared transport delays are "
                f"{config.transport_delays}"
            )
        missing = required - item.observation.keys()
        if missing:
            raise ValueError(f"stream {item.stream_id} is missing observation fields {sorted(missing)}")
        for name, value in item.observation.items():
            if not isinstance(name, str):
                raise ValueError(f"stream {item.stream_id} has a non-string observation field")
            # Live frames use built-in scalars; avoid numeric ABC dispatch for each field.
            native = type(value) in (int, float)
            if not native and (not isinstance(value, (Real, Integral)) or isinstance(value, bool)):
                raise ValueError(f"stream {item.stream_id} observation {name!r} must be numeric, got {value!r}")
            if (native or isinstance(value, Real)) and math.isinf(float(value)):
                raise ValueError(f"stream {item.stream_id} observation {name!r} must not be infinite")
        if item.player_identity is not None and (
            not isinstance(item.player_identity, str) or not item.player_identity
        ):
            raise ValueError(f"stream {item.stream_id} player identity must be a non-empty string")
        if spec.requires_player_identity and item.player_identity is None:
            raise ValueError(f"stream {item.stream_id} requires a player identity")
        if item.desired_return is not None and (
            not isinstance(item.desired_return, Real)
            or isinstance(item.desired_return, bool)
            or not math.isfinite(float(item.desired_return))
            or not 0.0 <= item.desired_return <= 40.0
        ):
            raise ValueError(f"stream {item.stream_id} desired return must be in [0, 40] or null")
        if (
            not isinstance(item.temperature, Real)
            or isinstance(item.temperature, bool)
            or not math.isfinite(float(item.temperature))
            or not 0.8 <= item.temperature <= 1.1
        ):
            raise ValueError(f"stream {item.stream_id} temperature must be in [0.8, 1.1]")
        validate_controller_action(item.applied_action)
        for action in item.pending_actions:
            validate_controller_action(action)
