"""Model-independent input and output contract for live HAL policies."""

import math
import re
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from numbers import Real
from typing import Final
from typing import Literal
from typing import Protocol
from typing import runtime_checkable

from hal.controller import ControllerAction
from hal.controller import validate_controller_action

ObservationScalar = float | int
DESIRED_RETURN_RANGE: Final[tuple[float, float]] = (-20.0, 140.0)


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
class PreparedInferenceProfile:
    name: str
    checkpoint_sha256: str
    execution_mode: Literal["window", "kv_cache"]
    prediction_horizon_frames: int
    fixed_prefix_frames: int
    update_shapes: tuple[int, ...]
    capacity: int

    def __post_init__(self) -> None:
        if (
            type(self.name) is not str
            or not self.name
            or type(self.checkpoint_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", self.checkpoint_sha256) is None
        ):
            raise ValueError("prepared inference profile needs a name and checkpoint hash")
        if type(self.execution_mode) is not str or self.execution_mode not in ("window", "kv_cache"):
            raise ValueError("prepared inference execution mode is unsupported")
        if (
            any(
                type(value) is not int
                for value in (self.prediction_horizon_frames, self.fixed_prefix_frames, self.capacity)
            )
            or self.prediction_horizon_frames < 1
            or not 0 <= self.fixed_prefix_frames < self.prediction_horizon_frames
            or self.capacity < 1
        ):
            raise ValueError("prepared inference horizon, prefix, or capacity is invalid")
        if (
            type(self.update_shapes) is not tuple
            or not self.update_shapes
            or any(type(shape) is not int or shape < 1 for shape in self.update_shapes)
            or tuple(sorted(set(self.update_shapes))) != self.update_shapes
        ):
            raise ValueError("prepared inference update shapes must be sorted positive values")


@dataclass(frozen=True, slots=True)
class PolicyInput:
    """One observed frame and the controller action that produced it.

    ``player_identity`` selects the behavior to imitate; it does not identify
    the live opponent. Future committed actions belong to the request prefix.
    """

    stream_id: int
    frame_id: int
    controlled_port: int
    observation: Mapping[str, ObservationScalar]
    applied_action: ControllerAction
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
    deadline_monotonic: float | None = None

    def __post_init__(self) -> None:
        for name in ("stream_id", "generation", "sequence", "source_frame"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"prediction {name} must be an integer")
        if self.generation < 1 or self.sequence < 0:
            raise ValueError("prediction generation must be positive and sequence non-negative")
        if self.deadline_monotonic is not None and (
            not isinstance(self.deadline_monotonic, (float, int))
            or isinstance(self.deadline_monotonic, bool)
            or not math.isfinite(self.deadline_monotonic)
            or self.deadline_monotonic < 0
        ):
            raise ValueError("prediction deadline must be finite and non-negative")
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
    """Predicted actions and ego state value at source_frame; fixed actions stay on the request."""

    stream_id: int
    generation: int
    sequence: int
    source_frame: int
    actions: tuple[FrameAction, ...]
    state_value: float


def action_plan(request: PredictionRequest, tail: Sequence[ControllerAction], *, state_value: float) -> ActionPlan:
    first = request.source_frame + len(request.fixed_actions) + 1
    return ActionPlan(
        request.stream_id,
        request.generation,
        request.sequence,
        request.source_frame,
        tuple(FrameAction(first + index, action) for index, action in enumerate(tail)),
        state_value,
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
    if (
        not isinstance(plan, ActionPlan)
        or type(plan.actions) is not tuple
        or any(type(value) is not int for value in (plan.stream_id, plan.generation, plan.sequence, plan.source_frame))
    ):
        raise ValueError("action plan identity must contain exact integers")
    if (plan.stream_id, plan.generation, plan.sequence, plan.source_frame) != (
        request.stream_id,
        request.generation,
        request.sequence,
        request.source_frame,
    ):
        raise ValueError("action plan does not match its request")
    if type(plan.state_value) is not float or not math.isfinite(plan.state_value):
        raise ValueError("action plan state value must be a finite float")
    if len(plan.actions) != horizon - len(request.fixed_actions):
        raise ValueError("action plan has the wrong prediction horizon")
    first = request.source_frame + len(request.fixed_actions) + 1
    for index, item in enumerate(plan.actions):
        if not isinstance(item, FrameAction) or type(item.target_frame) is not int:
            raise ValueError("action plan target frames must contain exact integers")
        if item.target_frame != first + index:
            raise ValueError("action plan target frames are not contiguous")
        validate_controller_action(item.action)


@runtime_checkable
class PredictionPolicy(Protocol):
    """A stateful model that predicts actions for one or more streams."""

    @property
    def checkpoint_sha256(self) -> str | None: ...

    @property
    def history_mode(self) -> Literal["window", "kv_cache"]: ...

    @property
    def prepared_update_shapes(self) -> tuple[int, ...]: ...

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

    def validate_prepared_profile(self, profile: PreparedInferenceProfile) -> None: ...

    def reset_prediction(self) -> None: ...

    def predict(self, requests: Sequence[PredictionRequest]) -> Sequence[ActionPlan]: ...

    def release_stream(self, stream_id: int) -> None: ...


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
        if type(item.controlled_port) is not int or item.controlled_port not in (1, 2):
            raise ValueError(f"stream {item.stream_id} controls unsupported port {item.controlled_port}")
        if not isinstance(item.reset, bool):
            raise ValueError(f"stream {item.stream_id} reset must be a boolean")
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
            or not DESIRED_RETURN_RANGE[0] <= item.desired_return <= DESIRED_RETURN_RANGE[1]
        ):
            raise ValueError(f"stream {item.stream_id} desired return must be in [-20, 140] or null")
        if (
            not isinstance(item.temperature, Real)
            or isinstance(item.temperature, bool)
            or not math.isfinite(float(item.temperature))
            or not 0.8 <= item.temperature <= 1.1
        ):
            raise ValueError(f"stream {item.stream_id} temperature must be in [0.8, 1.1]")
        validate_controller_action(item.applied_action)
