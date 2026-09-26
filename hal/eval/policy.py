"""Use the public policy interface in HAL's local Dolphin evaluator."""

import math
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.controller import ControllerAction
from hal.eval.observations import applied_action_from_frame
from hal.eval.scheduling import ActionScheduler
from hal.eval.scheduling import FrameTiming
from hal.inference.api import PolicyInput
from hal.inference.api import PredictionPolicy
from hal.inference.api import PredictionRequest
from hal.inference.api import RuntimeConfig
from hal.inference.transport import ActionTransport
from hal.sim.inputs import ControllerInputs
from hal.sim.inputs import action_vec_to_controller
from hal.sim.inputs import controller_actions_match
from hal.sim.inputs import controller_to_action_vec
from hal.sim.rollout import ObservationRow
from hal.sim.rollout import PolicyRuntimeSpec
from hal.sim.vec import Slot
from hal.training.canonical import flatten_canonical_frame
from hal.wire import ACTION_DIM


@dataclass(slots=True)
class _StreamState:
    transport: ActionTransport
    scheduler: ActionScheduler
    last_frame_id: int
    expected_applied: ControllerAction | None = None


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
    ) -> None:
        if desired_return is not None and (not math.isfinite(desired_return) or not 0.0 <= desired_return <= 40.0):
            raise ValueError("desired return must be in [0, 40] or null")
        if not math.isfinite(temperature) or not 0.8 <= temperature <= 1.1:
            raise ValueError("temperature must be in [0.8, 1.1]")
        if runtime.require_single_delay() != timing.input_delay_frames:
            raise ValueError("adapter timing input delay must match the prepared runtime")
        self.policy = policy
        self.runtime = runtime
        self.timing = timing
        self.player_identity = player_identity
        self.desired_return = desired_return
        self.temperature = temperature
        self._states: dict[Slot, _StreamState] = {}
        self.neutral_actions = 0
        self.total_actions = 0
        self.policy.reset_prediction()

    @property
    def runtime_spec(self) -> PolicyRuntimeSpec:
        # The adapter owns transport and its scheduler owns replanning. Workers
        # exchange one frame at a time and must not add a second delay queue.
        return PolicyRuntimeSpec(1, 1, 1, 0, ACTION_DIM, observed_actions=True)

    def plan_rows(self, rows: Mapping[Slot, Sequence[ObservationRow]]) -> Mapping[Slot, np.ndarray]:
        observations: dict[Slot, ObservationRow] = {}
        for slot, values in rows.items():
            if len(values) != 1:
                raise ValueError("frame policy requires exactly one observation per worker request")
            observations[slot] = values[0]
        outputs = self._step(observations)
        return {slot: controller_to_action_vec(action)[None, :] for slot, action in outputs.items()}

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
        delay = self.runtime.require_single_delay()
        items: list[PolicyInput] = []
        requests: list[PredictionRequest] = []
        for slot, frame in obs.items():
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
            if (
                state.expected_applied is not None
                and not reset
                and not controller_actions_match(state.expected_applied, applied)
            ):
                raise RuntimeError(
                    f"local controller alignment failed for {slot} at frame {frame_id}: "
                    f"expected {state.expected_applied!r}, observed {applied!r}"
                )
            state.last_frame_id = frame_id
            item = PolicyInput(
                stream_id=slot.match * 8 + slot.port,
                frame_id=frame_id,
                controlled_port=slot.port,
                observation={name: frame.flat[name] for name in self.policy.spec.required_observation_fields},
                applied_action=applied,
                pending_actions=state.transport.pending,
                player_identity=self.player_identity,
                desired_return=self.desired_return,
                temperature=self.temperature,
                reset=reset,
            )
            items.append(item)
            state.scheduler.observe(item)
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
            for slot, item in zip(obs, items, strict=True):
                if item.stream_id in by_stream:
                    self._states[slot].scheduler.accept_plan(by_stream[item.stream_id])
        result = {}
        for slot, item in zip(obs, items, strict=True):
            state = self._states[slot]
            state.scheduler.apply_ready_plan(item.frame_id)
            scheduled = state.scheduler.action_to_submit(item.frame_id)
            due = state.transport.submit(scheduled)
            state.expected_applied = due
            result[slot] = due
            self.total_actions += 1
            self.neutral_actions += int(due == NEUTRAL_CONTROLLER_ACTION)
        return result
