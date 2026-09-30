"""Use the public policy interface in HAL's local Dolphin evaluator."""

import math
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from typing import cast

import numpy as np

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.controller import ControllerAction
from hal.controller import action_vec_to_controller
from hal.controller import controller_to_action_vec
from hal.eval.observations import applied_action_from_frame
from hal.eval.scheduling import ActionScheduler
from hal.eval.scheduling import FrameTiming
from hal.inference.api import DESIRED_RETURN_RANGE
from hal.inference.api import PolicyInput
from hal.inference.api import PredictionPolicy
from hal.inference.api import PredictionRequest
from hal.inference.api import RuntimeConfig
from hal.representation.observations import flatten_canonical_frame
from hal.sim.inputs import ActionTransport
from hal.sim.inputs import ControllerInputs
from hal.sim.inputs import controller_actions_match
from hal.sim.rollout import ObservationRow
from hal.sim.rollout import PolicyRuntimeSpec
from hal.sim.rollout import Slot
from hal.wire import ACTION_DIM


@dataclass(frozen=True, slots=True)
class PolicySettings:
    player_identity: str | None = None
    desired_return: float | None = 20.0
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.player_identity is not None and not self.player_identity:
            raise ValueError("player identity must be non-empty")
        if self.desired_return is not None and (
            not math.isfinite(self.desired_return)
            or not DESIRED_RETURN_RANGE[0] <= self.desired_return <= DESIRED_RETURN_RANGE[1]
        ):
            raise ValueError("desired return must be in [-20, 140] or null")
        if not math.isfinite(self.temperature) or not 0.8 <= self.temperature <= 1.1:
            raise ValueError("temperature must be in [0.8, 1.1]")


@dataclass(slots=True)
class _StreamState:
    transport: ActionTransport
    scheduler: ActionScheduler
    last_frame_id: int
    expected_applied: dict[int, ControllerAction] = field(default_factory=dict)


class PolicyBatchAdapter:
    """Run planned model actions through the vector evaluator's frame seam."""

    def __init__(
        self,
        policy: PredictionPolicy,
        runtime: RuntimeConfig,
        timing: FrameTiming,
        *,
        player_identity: str | None = None,
        desired_return: float | None = 20.0,
        temperature: float = 1.0,
        observed_actions: bool = True,
        settings_by_port: Mapping[int, PolicySettings] | None = None,
    ) -> None:
        default_settings = PolicySettings(player_identity, desired_return, temperature)
        port_settings = {} if settings_by_port is None else dict(settings_by_port)
        if any(port not in (1, 2, 3, 4) for port in port_settings):
            raise ValueError("policy settings refer to an unsupported controller port")
        if runtime.require_single_delay() != timing.physical_delay_frames:
            raise ValueError("adapter timing input delay must match the prepared runtime")
        self.policy = policy
        self.runtime = runtime
        self.timing = timing
        self.default_settings = default_settings
        self.settings_by_port = port_settings
        self.observed_actions = observed_actions
        self._states: dict[Slot, _StreamState] = {}
        self.neutral_actions = 0
        self.total_actions = 0
        self.policy.reset_prediction()

    @property
    def runtime_spec(self) -> PolicyRuntimeSpec:
        return PolicyRuntimeSpec(
            self.policy.context_frames,
            self.timing.prediction_horizon_frames,
            self.timing.replan_interval_frames,
            self.timing.fixed_prefix_frames,
            ACTION_DIM,
            observed_actions=self.observed_actions,
        )

    def fault_snapshot(self) -> tuple[dict[str, object], dict[str, np.ndarray]]:
        snapshot = getattr(self.policy, "fault_snapshot", None)
        if not callable(snapshot):
            return {}, {}
        return cast(Callable[[], tuple[dict[str, object], dict[str, np.ndarray]]], snapshot)()

    def plan_rows(self, rows: Mapping[Slot, Sequence[ObservationRow]]) -> Mapping[Slot, np.ndarray]:
        if not rows:
            return {}
        latest: dict[Slot, PolicyInput] = {}
        for slot, values in rows.items():
            if not values or len(values) > self.timing.replan_interval_frames:
                raise ValueError("worker request must contain one to replan-interval observations")
            for row in values:
                latest[slot] = self._ingest_one(slot, row)
        self._predict(latest)
        chunks: dict[Slot, np.ndarray] = {}
        for slot, item in latest.items():
            state = self._states[slot]
            actions: list[ControllerAction] = []
            for offset in range(self.timing.prediction_horizon_frames):
                target = item.frame_id + offset + 1
                if offset < self.timing.replan_interval_frames:
                    submitted = state.scheduler.action_to_submit(item.frame_id + offset)
                    action = state.transport.submit(submitted)
                    state.expected_applied[target] = action
                    self.total_actions += 1
                    self.neutral_actions += int(action == NEUTRAL_CONTROLLER_ACTION)
                else:
                    action = state.scheduler.planned.get(target, NEUTRAL_CONTROLLER_ACTION)
                actions.append(action)
            chunks[slot] = np.stack([controller_to_action_vec(action) for action in actions])
        return chunks

    def __call__(
        self,
        frame_index: int,
        obs: Mapping[Slot, dict],
    ) -> Mapping[Slot, ControllerInputs]:
        del frame_index
        rows = {}
        for slot, frame in obs.items():
            applied = applied_action_from_frame(frame, slot.port)
            action = controller_to_action_vec(applied)
            rows[slot] = ObservationRow(int(frame["id"]), flatten_canonical_frame(frame), action)
        return self._step(rows)

    def _step(self, obs: Mapping[Slot, ObservationRow]) -> Mapping[Slot, ControllerAction]:
        if not obs:
            return {}
        latest = {slot: self._ingest_one(slot, row) for slot, row in obs.items()}
        self._predict(latest)
        result = {}
        for slot, item in latest.items():
            state = self._states[slot]
            scheduled = state.scheduler.action_to_submit(item.frame_id)
            due = state.transport.submit(scheduled)
            state.expected_applied[item.frame_id + 1] = due
            result[slot] = due
            self.total_actions += 1
            self.neutral_actions += int(due == NEUTRAL_CONTROLLER_ACTION)
        return result

    def _ingest_one(self, slot: Slot, frame: ObservationRow) -> PolicyInput:
        delay = self.runtime.require_single_delay()
        frame_id = frame.frame_id
        state = self._states.get(slot)
        if state is not None and not frame.reset and frame_id >= state.last_frame_id:
            expected_frame = state.last_frame_id + 1
            if frame_id != expected_frame:
                raise ValueError(
                    f"local stream {slot} expected frame {expected_frame}, got {frame_id}; "
                    "start a new episode with reset=True"
                )
        reset = frame.reset or state is None or frame_id < state.last_frame_id
        if reset:
            generation = 1 if state is None else state.scheduler.generation + 1
            state = _StreamState(
                ActionTransport(delay),
                ActionScheduler(self.timing, self.policy.context_frames, generation),
                frame_id,
            )
            self._states[slot] = state
        applied = action_vec_to_controller(frame.action)
        expected = state.expected_applied.pop(frame_id, None)
        if expected is not None and not reset and not controller_actions_match(expected, applied):
            raise RuntimeError(
                f"local controller alignment failed for {slot} at frame {frame_id}: "
                f"expected {expected!r}, observed {applied!r}"
            )
        state.last_frame_id = frame_id
        settings = self.settings_by_port.get(slot.port, self.default_settings)
        item = PolicyInput(
            stream_id=slot.match * 8 + slot.port,
            frame_id=frame_id,
            controlled_port=slot.port,
            observation={name: frame.flat[name] for name in self.policy.spec.required_observation_fields},
            applied_action=applied,
            player_identity=settings.player_identity,
            desired_return=settings.desired_return,
            temperature=settings.temperature,
            reset=reset,
        )
        state.scheduler.observe(item)
        return item

    def _predict(self, latest: Mapping[Slot, PolicyInput]) -> None:
        requests: list[PredictionRequest] = []
        for slot in latest:
            state = self._states[slot]
            request = state.scheduler.request_plan()
            if request is not None:
                requests.append(request)
        if requests:
            if len(requests) > self.runtime.max_batch_size:
                raise ValueError("prediction batch exceeds the prepared maximum")
            plans = tuple(self.policy.predict(requests))
            by_stream = {plan.stream_id: plan for plan in plans}
            if len(plans) != len(requests) or set(by_stream) != {request.stream_id for request in requests}:
                raise ValueError("model returned plans for the wrong local streams")
            for slot, item in latest.items():
                if item.stream_id in by_stream:
                    self._states[slot].scheduler.accept_plan(by_stream[item.stream_id], item.frame_id)
