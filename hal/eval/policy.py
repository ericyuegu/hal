"""Use the public policy interface in HAL's local Dolphin evaluator."""

import math
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.controller import ControllerAction
from hal.inference.api import Policy
from hal.inference.api import PolicyInput
from hal.inference.api import RuntimeConfig
from hal.inference.api import step_policy
from hal.inference.api import validate_policy_outputs
from hal.inference.transport import ActionTransport
from hal.sim.inputs import ControllerInputs
from hal.sim.inputs import action_vec_to_controller
from hal.sim.inputs import canonical_pre_to_action
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
    last_frame_id: int
    expected_applied: ControllerAction | None = None


class PolicyBatchAdapter:
    """Apply explicit transport delay while retaining HAL's BatchPolicy seam."""

    def __init__(
        self,
        policy: Policy,
        runtime: RuntimeConfig,
        *,
        player_identity: str | None = None,
        desired_return: float | None = 20.0,
        temperature: float = 1.0,
    ) -> None:
        if desired_return is not None and (not math.isfinite(desired_return) or not 0.0 <= desired_return <= 40.0):
            raise ValueError("desired return must be in [0, 40] or null")
        if not math.isfinite(temperature) or not 0.8 <= temperature <= 1.1:
            raise ValueError("temperature must be in [0.8, 1.1]")
        self.policy = policy
        self.runtime = runtime
        self.player_identity = player_identity
        self.desired_return = desired_return
        self.temperature = temperature
        self._states: dict[Slot, _StreamState] = {}
        self.neutral_actions = 0
        self.total_actions = 0

    @property
    def runtime_spec(self) -> PolicyRuntimeSpec:
        # The adapter owns transport and the policy owns replanning. Workers
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
            applied = canonical_pre_to_action(frame["ports"][slot.port]["leader"]["pre"])
            action = controller_to_action_vec(applied)
            rows[slot] = ObservationRow(int(frame["id"]), flatten_canonical_frame(frame), action)
        return self._step(rows)

    def _step(self, obs: Mapping[Slot, ObservationRow]) -> Mapping[Slot, ControllerAction]:
        if not obs:
            return {}
        delay = self.runtime.require_single_delay()
        items = []
        for slot, frame in obs.items():
            frame_id = frame.frame_id
            state = self._states.get(slot)
            reset = frame.reset or state is None or frame_id != state.last_frame_id + 1
            if reset:
                state = _StreamState(ActionTransport(delay), frame_id)
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
            items.append(
                PolicyInput(
                    stream_id=slot.match * 8 + slot.port,
                    frame_id=frame_id,
                    controlled_port=slot.port,
                    observation=frame.flat,
                    applied_action=applied,
                    pending_actions=state.transport.pending,
                    player_identity=self.player_identity,
                    desired_return=self.desired_return,
                    temperature=self.temperature,
                    reset=reset,
                )
            )
        outputs = validate_policy_outputs(items, tuple(step_policy(self.policy, self.runtime, items)))
        result = {}
        for slot, item in zip(obs, items, strict=True):
            state = self._states[slot]
            due = state.transport.submit(outputs[item.stream_id])
            state.expected_applied = due
            result[slot] = due
            self.total_actions += 1
            self.neutral_actions += int(due == NEUTRAL_CONTROLLER_ACTION)
        return result
