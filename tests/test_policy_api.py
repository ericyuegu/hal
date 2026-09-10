import pytest

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.controller import ControllerAction
from hal.inference.api import PolicyInput
from hal.inference.api import PolicyOutput
from hal.inference.api import PolicySpec
from hal.inference.api import RuntimeConfig
from hal.inference.api import validate_policy_inputs
from hal.inference.api import validate_policy_outputs
from hal.inference.transport import ActionTransport


def _input(stream_id: int, *, delay: int = 2) -> PolicyInput:
    return PolicyInput(
        stream_id=stream_id,
        frame_id=10,
        controlled_port=1,
        observation={"position": 1.0},
        applied_action=NEUTRAL_CONTROLLER_ACTION,
        pending_actions=(NEUTRAL_CONTROLLER_ACTION,) * delay,
        player_code="IBDW#0",
    )


def test_policy_contract_is_model_independent_and_batched() -> None:
    spec = PolicySpec(
        name="fake",
        backend="tests.fake.v1",
        required_observation_fields=("position",),
        supported_transport_delays=(0, 2, 3),
        requires_player_code=True,
    )
    config = RuntimeConfig(max_batch_size=2, transport_delay_frames=2)
    inputs = [_input(4), _input(9)]
    validate_policy_inputs(spec, config, inputs)
    actions = validate_policy_outputs(
        inputs,
        [
            PolicyOutput(9, ControllerAction(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)),
            PolicyOutput(4, NEUTRAL_CONTROLLER_ACTION),
        ],
    )
    assert set(actions) == {4, 9}


def test_policy_contract_rejects_missing_fields_and_wrong_queue_length() -> None:
    spec = PolicySpec("fake", "tests.fake.v1", ("missing",), (2,))
    config = RuntimeConfig(1, 2)
    with pytest.raises(ValueError, match="missing observation fields"):
        validate_policy_inputs(spec, config, [_input(0)])
    with pytest.raises(ValueError, match="pending actions"):
        validate_policy_inputs(
            PolicySpec("fake", "tests.fake.v1", ("position",), (2,)),
            config,
            [_input(0, delay=1)],
        )


@pytest.mark.parametrize("value", ["1", True, float("inf")])
def test_policy_contract_rejects_non_numeric_or_infinite_observations(value: object) -> None:
    item = _input(0)
    invalid = PolicyInput(
        stream_id=item.stream_id,
        frame_id=item.frame_id,
        controlled_port=item.controlled_port,
        observation={"position": value},
        applied_action=item.applied_action,
        pending_actions=item.pending_actions,
        player_code=item.player_code,
    )
    with pytest.raises(ValueError, match="observation"):
        validate_policy_inputs(
            PolicySpec("fake", "tests.fake.v1", ("position",), (2,)),
            RuntimeConfig(1, 2),
            [invalid],
        )


def test_policy_contract_rejects_pause_button() -> None:
    item = _input(0)
    with pytest.raises(ValueError, match="unsupported bits"):
        validate_policy_outputs(
            [item],
            [PolicyOutput(item.stream_id, ControllerAction(0, 0, 0, 0, 0, 0, 0x1000))],
        )


@pytest.mark.parametrize("delay", [0, 2, 3])
def test_transport_returns_actions_for_the_corresponding_future_state(delay: int) -> None:
    transport = ActionTransport(delay)
    submitted = [ControllerAction(index / 10, 0.0, 0.0, 0.0, 0.0, 0.0, 0) for index in range(1, 7)]
    due = [transport.submit(action) for action in submitted]
    expected = [NEUTRAL_CONTROLLER_ACTION] * delay + submitted[: len(submitted) - delay]
    assert due == expected
    expected_pending = tuple(submitted[-delay:]) if delay else ()
    assert transport.pending == expected_pending
