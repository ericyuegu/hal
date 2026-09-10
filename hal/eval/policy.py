"""Use the public policy interface in HAL's local Dolphin evaluator."""

from collections.abc import Mapping
from dataclasses import dataclass

from hal.controller import ControllerAction
from hal.inference.api import Policy
from hal.inference.api import PolicyInput
from hal.inference.api import RuntimeConfig
from hal.inference.api import validate_policy_inputs
from hal.inference.api import validate_policy_outputs
from hal.inference.transport import ActionTransport
from hal.sim.inputs import ControllerInputs
from hal.sim.inputs import canonical_pre_to_action
from hal.sim.inputs import controller_actions_match
from hal.sim.vec import Slot
from hal.training.canonical import flatten_canonical_frame


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
        player_code: str | None = None,
    ) -> None:
        self.policy = policy
        self.runtime = runtime
        self.player_code = player_code
        self._states: dict[Slot, _StreamState] = {}

    def __call__(
        self,
        frame_index: int,
        obs: Mapping[Slot, dict],
    ) -> Mapping[Slot, ControllerInputs]:
        del frame_index
        if not obs:
            return {}
        items = []
        for slot, frame in obs.items():
            frame_id = int(frame["id"])
            state = self._states.get(slot)
            reset = state is None or frame_id != state.last_frame_id + 1
            if reset:
                state = _StreamState(ActionTransport(self.runtime.transport_delay_frames), frame_id)
                self._states[slot] = state
            applied = canonical_pre_to_action(frame["ports"][slot.port]["leader"]["pre"])
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
                    observation=flatten_canonical_frame(frame),
                    applied_action=applied,
                    pending_actions=state.transport.pending,
                    player_code=self.player_code,
                    reset=reset,
                )
            )
        validate_policy_inputs(self.policy.spec, self.runtime, items)
        outputs = validate_policy_outputs(items, tuple(self.policy.step(items)))
        result = {}
        for slot, item in zip(obs, items, strict=True):
            state = self._states[slot]
            due = state.transport.submit(outputs[item.stream_id])
            state.expected_applied = due
            result[slot] = due
        return result
