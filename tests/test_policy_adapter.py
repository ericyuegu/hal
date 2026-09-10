import pytest

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.controller import ControllerAction
from hal.eval.policy import PolicyBatchAdapter
from hal.inference.api import PolicyInput
from hal.inference.api import PolicyOutput
from hal.inference.api import PolicySpec
from hal.inference.api import RuntimeConfig
from hal.sim.vec import Slot


def _pre(action: ControllerAction) -> dict:
    return {
        "joystick": {"x": action.main_x, "y": action.main_y},
        "cstick": {"x": action.c_x, "y": action.c_y},
        "triggers_physical": {"l": action.trigger_l, "r": action.trigger_r},
        "buttons_physical": action.buttons,
    }


def _frame(frame_id: int, action: ControllerAction) -> dict:
    return {
        "id": frame_id,
        "ports": {1: {"leader": {"pre": _pre(action)}}},
        "flat": frame_id,
    }


class _TaggedPolicy:
    spec = PolicySpec("tagged", "tests.tagged", ("flat",), (2, 3))

    def __init__(self) -> None:
        self.inputs: list[PolicyInput] = []

    def prepare(self, _config: RuntimeConfig) -> None:
        pass

    def step(self, inputs) -> tuple[PolicyOutput, ...]:
        self.inputs.extend(inputs)
        return tuple(
            PolicyOutput(item.stream_id, ControllerAction(len(self.inputs) / 10, 0, 0, 0, 0, 0, 0)) for item in inputs
        )


@pytest.mark.parametrize("delay", [2, 3])
def test_adapter_pairs_actual_actions_and_orders_pending_queue(
    monkeypatch: pytest.MonkeyPatch,
    delay: int,
) -> None:
    monkeypatch.setattr("hal.eval.policy.flatten_canonical_frame", lambda frame: {"flat": frame["flat"]})
    policy = _TaggedPolicy()
    adapter = PolicyBatchAdapter(policy, RuntimeConfig(1, delay))
    slot = Slot(0, 1)
    applied = NEUTRAL_CONTROLLER_ACTION
    returned = []
    for frame_id in range(delay + 3):
        due = adapter(frame_id, {slot: _frame(frame_id, applied)})[slot]
        returned.append(due)
        applied = due

    tags = [ControllerAction(index / 10, 0, 0, 0, 0, 0, 0) for index in range(1, delay + 4)]
    assert returned == [NEUTRAL_CONTROLLER_ACTION] * delay + tags[:3]
    assert policy.inputs[delay].pending_actions == tuple(tags[:delay])
    assert policy.inputs[delay + 1].applied_action == tags[0]


def test_adapter_resets_transport_on_frame_discontinuity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hal.eval.policy.flatten_canonical_frame", lambda frame: {"flat": frame["flat"]})
    policy = _TaggedPolicy()
    adapter = PolicyBatchAdapter(policy, RuntimeConfig(1, 2))
    slot = Slot(0, 1)
    adapter(0, {slot: _frame(10, NEUTRAL_CONTROLLER_ACTION)})
    result = adapter(1, {slot: _frame(-123, NEUTRAL_CONTROLLER_ACTION)})
    assert result[slot] == NEUTRAL_CONTROLLER_ACTION
    assert policy.inputs[-1].reset
    assert policy.inputs[-1].pending_actions == (NEUTRAL_CONTROLLER_ACTION,) * 2
