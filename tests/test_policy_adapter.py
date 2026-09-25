import melee
import numpy as np
import pytest

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.controller import ControllerAction
from hal.eval.harness import SessionConfig
from hal.eval.harness import default_session_cfg
from hal.eval.harness import run_matches_vec
from hal.eval.policy import PolicyBatchAdapter
from hal.inference.api import PolicyInput
from hal.inference.api import PolicyOutput
from hal.inference.api import PolicySpec
from hal.inference.api import RuntimeConfig
from hal.sim.inputs import action_vec_to_controller
from hal.sim.inputs import controller_to_action_vec
from hal.sim.rollout import ObservationRow
from hal.sim.session import Matchup
from hal.sim.session import PlayerSetup
from hal.sim.vec import Slot
from hal.sim.vec import VecMatch


def test_conditioned_adapter_routes_to_process_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = PolicyBatchAdapter(_TaggedPolicy(), RuntimeConfig(1, (2,)))
    called = []

    def process_driver(kwargs, matches, policy, **options):
        called.append(policy)
        assert policy.runtime_spec.observed_actions
        return [[] for _ in matches]

    def thread_driver(*args, **kwargs):
        pytest.fail("conditioned evaluation must use process workers")

    monkeypatch.setattr("hal.eval.harness.drive_process_vec", process_driver)
    monkeypatch.setattr("hal.eval.harness.drive_vec", thread_driver)
    match = VecMatch(
        Matchup(
            stage=melee.Stage.FINAL_DESTINATION,
            players=(
                PlayerSetup(port=1, character=melee.Character.FOX),
                PlayerSetup(port=2, character=melee.Character.FALCO),
            ),
        ),
        (1,),
    )
    run_matches_vec(
        SessionConfig("unused", "unused"), [match], lambda: adapter, max_frames=10, max_parallel=1, start_retries=0
    )
    assert called == [adapter]


@pytest.mark.parametrize("delay", [2, 3])
def test_process_adapter_matches_frame_adapter_through_reset(monkeypatch, delay: int) -> None:
    monkeypatch.setattr("hal.eval.policy.flatten_canonical_frame", lambda frame: {"flat": frame["flat"]})
    left_policy, right_policy = _TaggedPolicy(), _TaggedPolicy()
    left = PolicyBatchAdapter(left_policy, RuntimeConfig(2, (delay,)))
    right = PolicyBatchAdapter(right_policy, RuntimeConfig(2, (delay,)))
    slots = (Slot(0, 1), Slot(1, 1))
    applied = dict.fromkeys(slots, NEUTRAL_CONTROLLER_ACTION)
    for tick in range(12):
        frames = {slot: _frame(tick if slot.match else tick % 7, applied[slot]) for slot in slots}
        expected = left(tick, frames)
        rows = {
            slot: [ObservationRow(frame["id"], {"flat": frame["flat"]}, controller_to_action_vec(applied[slot]))]
            for slot, frame in frames.items()
        }
        actual = right.plan_rows(rows)
        for slot in slots:
            np.testing.assert_array_equal(actual[slot][0], controller_to_action_vec(expected[slot]))
            applied[slot] = action_vec_to_controller(actual[slot][0])
        assert left_policy.inputs == right_policy.inputs
    assert right.runtime_spec.execution_stride == 1
    assert right.runtime_spec.committed_frames == 0
    assert right.runtime_spec.observed_actions


def test_process_adapter_rejects_multiple_observations() -> None:
    adapter = PolicyBatchAdapter(_TaggedPolicy(), RuntimeConfig(1, (2,)))
    with pytest.raises(ValueError, match="exactly one"):
        adapter.plan_rows({Slot(0, 1): []})


@pytest.mark.integration
def test_process_adapter_observes_real_dolphin_inputs(tmp_path) -> None:
    class MovingPolicy:
        spec = PolicySpec("moving", "test", (), (2,))

        def step(self, inputs):
            return tuple(PolicyOutput(item.stream_id, ControllerAction(0.5, 0, 0, 0, 0, 0, 0)) for item in inputs)

        def prepare(self, config):
            pass

    adapter = PolicyBatchAdapter(MovingPolicy(), RuntimeConfig(1, (2,)))
    match = VecMatch(
        Matchup(
            stage=melee.Stage.FINAL_DESTINATION,
            players=(
                PlayerSetup(port=1, character=melee.Character.FOX),
                PlayerSetup(port=2, character=melee.Character.FALCO, cpu_level=9),
            ),
        ),
        (1,),
    )
    boots = run_matches_vec(
        default_session_cfg(tmp_path), [match], lambda: adapter, max_frames=40, max_parallel=1, start_retries=0
    )
    assert len(boots[0]) == 1
    assert len(boots[0][0]) == 40
    assert adapter.total_actions == 39
    assert adapter.neutral_actions == 2


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
            PolicyOutput(item.stream_id, ControllerAction((len(self.inputs) % 10) / 10, 0, 0, 0, 0, 0, 0))
            for item in inputs
        )


def test_adapter_passes_return_target_and_temperature() -> None:
    policy = _TaggedPolicy()
    adapter = PolicyBatchAdapter(policy, RuntimeConfig(1, (2,)), desired_return=19.976, temperature=0.9)
    adapter.plan_rows(
        {Slot(0, 1): [ObservationRow(0, {"flat": 0}, controller_to_action_vec(NEUTRAL_CONTROLLER_ACTION))]}
    )
    assert policy.inputs[0].desired_return == 19.976
    assert policy.inputs[0].temperature == 0.9


@pytest.mark.parametrize("delay", [2, 3])
def test_adapter_pairs_actual_actions_and_orders_pending_queue(
    monkeypatch: pytest.MonkeyPatch,
    delay: int,
) -> None:
    monkeypatch.setattr("hal.eval.policy.flatten_canonical_frame", lambda frame: {"flat": frame["flat"]})
    policy = _TaggedPolicy()
    adapter = PolicyBatchAdapter(policy, RuntimeConfig(1, (delay,)))
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
    assert adapter.total_actions == delay + 3
    assert adapter.neutral_actions == delay


def test_adapter_resets_transport_on_frame_discontinuity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hal.eval.policy.flatten_canonical_frame", lambda frame: {"flat": frame["flat"]})
    policy = _TaggedPolicy()
    adapter = PolicyBatchAdapter(policy, RuntimeConfig(1, (2,)))
    slot = Slot(0, 1)
    adapter(0, {slot: _frame(10, NEUTRAL_CONTROLLER_ACTION)})
    result = adapter(1, {slot: _frame(-123, NEUTRAL_CONTROLLER_ACTION)})
    assert result[slot] == NEUTRAL_CONTROLLER_ACTION
    assert policy.inputs[-1].reset
    assert policy.inputs[-1].pending_actions == (NEUTRAL_CONTROLLER_ACTION,) * 2
