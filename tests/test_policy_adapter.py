import melee
import numpy as np
import pytest

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.controller import ControllerAction
from hal.controller import action_vec_to_controller
from hal.controller import controller_to_action_vec
from hal.eval.harness import SessionConfig
from hal.eval.harness import default_session_cfg
from hal.eval.harness import run_matches_vec
from hal.eval.policy import PolicyBatchAdapter
from hal.eval.scheduling import FrameTiming
from hal.inference.api import PolicyInput
from hal.inference.api import PolicySpec
from hal.inference.api import PredictionRequest
from hal.inference.api import RuntimeConfig
from hal.inference.api import action_plan
from hal.sim.rollout import ObservationRow
from hal.sim.rollout import Slot
from hal.sim.rollout import VecMatch
from hal.sim.session import Matchup
from hal.sim.session import PlayerSetup


def _timing(delay: int = 2) -> FrameTiming:
    return FrameTiming(delay, 0, delay, 1, delay + 1)


def test_conditioned_adapter_routes_to_process_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = PolicyBatchAdapter(_TaggedPolicy(), RuntimeConfig(1, (2,)), _timing())
    called = []

    def process_driver(kwargs, matches, policy, **options):
        called.append(policy)
        assert policy.runtime_spec.observed_actions
        return [[] for _ in matches]

    monkeypatch.setattr("hal.eval.harness.drive_process_vec", process_driver)
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
    left = PolicyBatchAdapter(left_policy, RuntimeConfig(2, (delay,)), _timing(delay))
    right = PolicyBatchAdapter(right_policy, RuntimeConfig(2, (delay,)), _timing(delay))
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
    assert right.runtime_spec.committed_frames == delay
    assert right.runtime_spec.observed_actions


def test_process_adapter_rejects_observations_outside_declared_stride() -> None:
    adapter = PolicyBatchAdapter(_TaggedPolicy(), RuntimeConfig(1, (2,)), _timing())
    with pytest.raises(ValueError, match="one to replan-interval"):
        adapter.plan_rows({Slot(0, 1): []})
    neutral = controller_to_action_vec(NEUTRAL_CONTROLLER_ACTION)
    with pytest.raises(ValueError, match="one to replan-interval"):
        adapter.plan_rows({Slot(0, 1): [ObservationRow(frame, {"flat": frame}, neutral) for frame in range(2)]})


def test_official_dense_adapter_keeps_two_frame_process_stride_and_intended_actions() -> None:
    class ChunkPolicy:
        spec = PolicySpec("chunk", "test", ("flat",), (0,))
        context_frames = 8

        def __init__(self) -> None:
            self.batches: list[tuple[PredictionRequest, ...]] = []

        def reset_prediction(self) -> None:
            pass

        def predict(self, requests: tuple[PredictionRequest, ...]):
            self.batches.append(tuple(requests))
            return tuple(action_plan(request, (ControllerAction(0.3, 0, 0, 0, 0, 0, 0),) * 2) for request in requests)

    policy = ChunkPolicy()
    adapter = PolicyBatchAdapter(
        policy,
        RuntimeConfig(2, (0,), replan_interval_frames=2),
        FrameTiming(0, 0, 2, 2, 4),
        observed_actions=False,
    )
    slots = (Slot(0, 1), Slot(1, 1))
    neutral = controller_to_action_vec(NEUTRAL_CONTROLLER_ACTION)
    first = adapter.plan_rows({slot: [ObservationRow(0, {"flat": 0}, neutral, reset=True)] for slot in slots})
    assert adapter.runtime_spec.execution_stride == 2
    assert adapter.runtime_spec.prediction_frames == 4
    assert not adapter.runtime_spec.observed_actions
    assert len(policy.batches) == 1 and len(policy.batches[0]) == 2
    for chunk in first.values():
        assert chunk.shape == (4, 14)
        np.testing.assert_array_equal(chunk[:2], np.stack((neutral, neutral)))
    second = adapter.plan_rows(
        {slot: [ObservationRow(frame, {"flat": frame}, first[slot][frame - 1]) for frame in (1, 2)] for slot in slots}
    )
    assert len(policy.batches) == 2 and len(policy.batches[1]) == 2
    assert all(
        tuple(request.fixed_actions) == (ControllerAction(0.3, 0, 0, 0, 0, 0, 0),) * 2 for request in policy.batches[1]
    )
    for chunk in second.values():
        np.testing.assert_array_equal(chunk[:2], first[slots[0]][2:4])


@pytest.mark.integration
def test_process_adapter_observes_real_dolphin_inputs(tmp_path) -> None:
    class MovingPolicy:
        spec = PolicySpec("moving", "test", (), (2,))
        context_frames = 4

        def predict(self, requests):
            return tuple(action_plan(request, (ControllerAction(0.5, 0, 0, 0, 0, 0, 0),)) for request in requests)

        def reset_prediction(self):
            pass

    adapter = PolicyBatchAdapter(MovingPolicy(), RuntimeConfig(1, (2,)), _timing())
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
    context_frames = 8

    def __init__(self) -> None:
        self.inputs: list[PolicyInput] = []
        self.requests: list[PredictionRequest] = []

    def reset_prediction(self) -> None:
        pass

    def predict(self, requests: tuple[PredictionRequest, ...]):
        self.requests.extend(requests)
        self.inputs.extend(request.observations[-1] for request in requests)
        return tuple(
            action_plan(request, (ControllerAction((len(self.inputs) % 10) / 10, 0, 0, 0, 0, 0, 0),))
            for request in requests
        )


def test_adapter_passes_return_target_and_temperature() -> None:
    policy = _TaggedPolicy()
    adapter = PolicyBatchAdapter(policy, RuntimeConfig(1, (2,)), _timing(), desired_return=19.976, temperature=0.9)
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
    adapter = PolicyBatchAdapter(policy, RuntimeConfig(1, (delay,)), _timing(delay))
    slot = Slot(0, 1)
    applied = NEUTRAL_CONTROLLER_ACTION
    returned = []
    for frame_id in range(delay + 3):
        due = adapter(frame_id, {slot: _frame(frame_id, applied)})[slot]
        returned.append(due)
        applied = due

    tags = [ControllerAction(index / 10, 0, 0, 0, 0, 0, 0) for index in range(1, delay + 4)]
    assert returned == [NEUTRAL_CONTROLLER_ACTION] * delay + tags[:3]
    assert policy.requests[delay].fixed_actions == tuple(tags[:delay])
    assert policy.inputs[delay + 1].applied_action == tags[0]
    assert adapter.total_actions == delay + 3
    assert adapter.neutral_actions == delay


def test_adapter_resets_transport_on_backward_episode_restart(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hal.eval.policy.flatten_canonical_frame", lambda frame: {"flat": frame["flat"]})
    policy = _TaggedPolicy()
    adapter = PolicyBatchAdapter(policy, RuntimeConfig(1, (2,)), _timing())
    slot = Slot(0, 1)
    adapter(0, {slot: _frame(10, NEUTRAL_CONTROLLER_ACTION)})
    result = adapter(1, {slot: _frame(-123, NEUTRAL_CONTROLLER_ACTION)})
    assert result[slot] == NEUTRAL_CONTROLLER_ACTION
    assert policy.inputs[-1].reset
    assert policy.requests[-1].fixed_actions == (NEUTRAL_CONTROLLER_ACTION,) * 2


@pytest.mark.parametrize("next_frame", [10, 12])
def test_adapter_rejects_missing_or_repeated_frame(next_frame: int) -> None:
    policy = _TaggedPolicy()
    adapter = PolicyBatchAdapter(policy, RuntimeConfig(1, (2,)), _timing())
    slot = Slot(0, 1)
    neutral = controller_to_action_vec(NEUTRAL_CONTROLLER_ACTION)
    adapter.plan_rows({slot: [ObservationRow(10, {"flat": 10}, neutral)]})

    with pytest.raises(ValueError, match=f"expected frame 11, got {next_frame}"):
        adapter.plan_rows({slot: [ObservationRow(next_frame, {"flat": next_frame}, neutral)]})

    adapter.plan_rows({slot: [ObservationRow(11, {"flat": 11}, neutral)]})
    assert policy.inputs[-1].frame_id == 11
    assert not policy.inputs[-1].reset


def test_adapter_accepts_explicit_reset_after_forward_frame_jump() -> None:
    policy = _TaggedPolicy()
    adapter = PolicyBatchAdapter(policy, RuntimeConfig(1, (2,)), _timing())
    slot = Slot(0, 1)
    neutral = controller_to_action_vec(NEUTRAL_CONTROLLER_ACTION)
    adapter.plan_rows({slot: [ObservationRow(10, {"flat": 10}, neutral)]})
    adapter.plan_rows({slot: [ObservationRow(20, {"flat": 20}, neutral, reset=True)]})

    assert policy.inputs[-1].reset
    assert policy.requests[-1].fixed_actions == (NEUTRAL_CONTROLLER_ACTION,) * 2
