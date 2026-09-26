from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.controller import ControllerAction
from hal.inference.api import PolicyInput
from hal.inference.api import PolicySpec
from hal.inference.api import PredictionRequest
from hal.inference.api import RuntimeConfig
from hal.inference.api import action_plan
from hal.inference.api import validate_action_plan
from hal.inference.api import validate_policy_inputs
from hal.inference.api import validate_prediction_request
from hal.inference.transport import ActionTransport


def _input(stream_id: int, *, delay: int = 2) -> PolicyInput:
    return PolicyInput(
        stream_id=stream_id,
        frame_id=10,
        controlled_port=1,
        observation={"position": 1.0},
        applied_action=NEUTRAL_CONTROLLER_ACTION,
        pending_actions=(NEUTRAL_CONTROLLER_ACTION,) * delay,
        player_identity="IBDW#0",
    )


def test_policy_contract_is_model_independent_and_batched() -> None:
    spec = PolicySpec(
        name="fake",
        backend="tests.fake.v1",
        required_observation_fields=("position",),
        supported_transport_delays=(0, 2, 3),
        requires_player_identity=True,
    )
    config = RuntimeConfig(max_batch_size=2, transport_delays=(2, 3))
    inputs = [_input(4), _input(9)]
    validate_policy_inputs(spec, config, inputs)
    for item in inputs:
        request = PredictionRequest(item.stream_id, 1, 0, item.frame_id, (item,), item.pending_actions)
        validate_prediction_request(spec, config, request, context_frames=8, prefix_frames=2)
        plan = action_plan(request, (NEUTRAL_CONTROLLER_ACTION,) * 4)
        validate_action_plan(request, plan, horizon=6)
        assert [action.target_frame for action in plan.actions] == [13, 14, 15, 16]


def test_prediction_rejects_a_plan_for_another_stream() -> None:
    item = _input(0)
    request = PredictionRequest(0, 1, 0, item.frame_id, (item,), item.pending_actions)
    plan = action_plan(request, (NEUTRAL_CONTROLLER_ACTION,))
    with pytest.raises(ValueError, match="does not match"):
        validate_action_plan(request, replace(plan, stream_id=1), horizon=3)


def test_runtime_delays_are_explicit_and_canonical() -> None:
    assert RuntimeConfig(2, (2,)).require_single_delay() == 2
    with pytest.raises(ValueError, match="requires one"):
        RuntimeConfig(2, (2, 3)).require_single_delay()
    for delays in ((), (3, 2), (2, 2), (-1,)):
        with pytest.raises(ValueError, match="transport_delays"):
            RuntimeConfig(2, delays)


def test_policy_contract_rejects_missing_fields_and_wrong_queue_length() -> None:
    spec = PolicySpec("fake", "tests.fake.v1", ("missing",), (2,))
    config = RuntimeConfig(1, (2,))
    with pytest.raises(ValueError, match="missing observation fields"):
        validate_policy_inputs(spec, config, [_input(0)])
    with pytest.raises(ValueError, match="pending actions"):
        validate_policy_inputs(
            PolicySpec("fake", "tests.fake.v1", ("position",), (2,)),
            config,
            [_input(0, delay=1)],
        )


def test_policy_contract_requires_a_nonempty_player_identity() -> None:
    spec = PolicySpec("fake", "tests.fake.v1", ("position",), (2,), requires_player_identity=True)
    config = RuntimeConfig(1, (2,))
    item = _input(0)
    for identity in (None, ""):
        invalid = PolicyInput(
            stream_id=item.stream_id,
            frame_id=item.frame_id,
            controlled_port=item.controlled_port,
            observation=item.observation,
            applied_action=item.applied_action,
            pending_actions=item.pending_actions,
            player_identity=identity,
        )
        with pytest.raises(ValueError, match="player identity"):
            validate_policy_inputs(spec, config, [invalid])


@pytest.mark.parametrize("value", ["1", True, float("inf"), -float("inf"), np.float32("inf"), np.bool_(True)])
def test_policy_contract_rejects_non_numeric_or_infinite_observations(value: object) -> None:
    item = _input(0)
    invalid = PolicyInput(
        stream_id=item.stream_id,
        frame_id=item.frame_id,
        controlled_port=item.controlled_port,
        observation={"position": value},
        applied_action=item.applied_action,
        pending_actions=item.pending_actions,
        player_identity=item.player_identity,
    )
    with pytest.raises(ValueError, match="observation"):
        validate_policy_inputs(
            PolicySpec("fake", "tests.fake.v1", ("position",), (2,)),
            RuntimeConfig(1, (2,)),
            [invalid],
        )


@pytest.mark.parametrize(
    "value", [0, 1.5, float("nan"), np.int64(2), np.float32(1.5), np.float64("nan"), Fraction(1, 3)]
)
def test_policy_contract_accepts_native_and_extended_numeric_observations(value: object) -> None:
    item = replace(_input(0), observation={"position": value})
    validate_policy_inputs(PolicySpec("fake", "tests.fake.v1", ("position",), (2,)), RuntimeConfig(1, (2,)), [item])


def test_policy_contract_rejects_pause_button() -> None:
    item = _input(0)
    request = PredictionRequest(item.stream_id, 1, 0, item.frame_id, (item,), item.pending_actions)
    plan = action_plan(request, (ControllerAction(0, 0, 0, 0, 0, 0, 0x1000),))
    with pytest.raises(ValueError, match="unsupported bits"):
        validate_action_plan(request, plan, horizon=3)


@pytest.mark.parametrize("delay", [0, 2, 3])
def test_transport_returns_actions_for_the_corresponding_future_state(delay: int) -> None:
    transport = ActionTransport(delay)
    submitted = [ControllerAction(index / 10, 0.0, 0.0, 0.0, 0.0, 0.0, 0) for index in range(1, 7)]
    due = [transport.submit(action) for action in submitted]
    expected = [NEUTRAL_CONTROLLER_ACTION] * delay + submitted[: len(submitted) - delay]
    assert due == expected
    expected_pending = tuple(submitted[-delay:]) if delay else ()
    assert transport.pending == expected_pending
