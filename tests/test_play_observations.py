"""Shared live observation construction for both play drivers."""

import pytest

from hal.controller import ControllerAction
from hal.eval import observations
from hal.inference.api import PolicySpec
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
