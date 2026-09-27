"""Shared live observation construction for both play drivers."""

import numpy as np
import pytest

from hal.controller import ControllerAction
from hal.eval import observations
from hal.inference.action_sequence_artifact import REQUIRED_OBSERVATION_FIELDS
from hal.inference.api import PolicySpec
from hal.representation.observations import project_observation_columns
from hal.sim.ipc import ArenaSpec
from hal.sim.ipc import RolloutArena
from hal.wire import ACTION_DIM
from hal.wire import BUTTON_BITS


def _frame(frame_id: int, *, stage: int, character: int, buttons: int = 0) -> dict:
    pre = {
        "joystick": {"x": 0.25, "y": 0.0},
        "cstick": {"x": 0.0, "y": 0.0},
        "triggers_physical": {"l": 0.0, "r": 0.0},
        "buttons_physical": buttons,
    }
    return {
        "id": frame_id,
        "stage": stage,
        "ports": {
            1: {"leader": {"pre": pre, "post": {"character": character}}},
            2: {"leader": {"pre": pre, "post": {"character": 22}}},
        },
    }


def test_live_observation_projects_required_fields_and_masks_start(monkeypatch: pytest.MonkeyPatch) -> None:
    def flatten(frame: dict) -> dict[str, float | int]:
        matchup = frame["_matchup"]
        return {
            "stage": matchup["stage"],
            "p1_character": matchup["character"][1],
            "unused": 77,
        }

    monkeypatch.setattr(observations, "flatten_canonical_frame", flatten)
    spec = PolicySpec("fake", "fake", ("stage", "p1_character"), (2,))
    frame = _frame(-44, stage=31, character=7, buttons=BUTTON_BITS["a"] | BUTTON_BITS["start"])
    item = observations.policy_input_from_frame(
        frame,
        spec=spec,
        stream_id=8,
        controlled_port=1,
        player_identity="MASTER",
        reset=True,
        matchup_characters={1: 7, 2: 22},
    )

    assert item.frame_id == -44
    assert item.observation == {"stage": 31, "p1_character": 7}
    assert item.applied_action == ControllerAction(0.25, 0, 0, 0, 0, 0, BUTTON_BITS["a"])
    assert item.player_identity == "MASTER"
    assert item.reset
    transformed = _frame(-43, stage=31, character=19)
    later = observations.policy_input_from_frame(
        transformed,
        spec=spec,
        stream_id=8,
        controlled_port=1,
        matchup_characters={1: 7, 2: 22},
    )
    assert later.observation["p1_character"] == 7


def test_live_observation_rejects_missing_required_field(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(observations, "flatten_canonical_frame", lambda _: {"stage": 31})
    spec = PolicySpec("fake", "fake", ("missing",), (2,))
    with pytest.raises(KeyError, match="missing"):
        observations.policy_input_from_frame(
            _frame(0, stage=31, character=1),
            spec=spec,
            stream_id=1,
            controlled_port=1,
            matchup_characters={1: 1, 2: 22},
        )


def test_direct_policy_observations_match_local_shared_memory_types_and_masks() -> None:
    frame = _frame(0, stage=31, character=1)
    frame["ports"][1]["leader"]["post"].update({"stock": 4, "action": 14, "position": {"x": 1.25, "y": -2.5}})
    frame["items"] = [{"id": 7, "type": 1, "state": 3, "owner": 0}]
    flat = observations.flatten_live_frame(frame, {1: 1, 2: 22})
    spec = PolicySpec("059", "hal.action_sequence.test", REQUIRED_OBSERVATION_FIELDS, (2,))
    direct = observations.policy_input_from_frame(
        frame, spec=spec, stream_id=7, controlled_port=1, flat=flat
    ).observation
    with RolloutArena.create(ArenaSpec(1, 8, 4, ACTION_DIM)) as arena:
        arena.write_observation(0, 1, 0, flat, np.zeros(ACTION_DIM, dtype=np.float32), reset=True)
        local, _, _ = arena.observation(0, 1)
        for name, value in direct.items():
            assert type(value) is type(local[name])
            np.testing.assert_equal(value, local[name])


@pytest.mark.parametrize("value", [0.5, float("inf"), float("-inf"), 1 << 31, -(1 << 31) - 1])
def test_observation_projection_rejects_invalid_integer_categories(value: float | int) -> None:
    with pytest.raises(ValueError, match="not an int32 value"):
        project_observation_columns({"item0_state": value}, ("item0_state",))
