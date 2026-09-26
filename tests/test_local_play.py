"""Synchronous local match planning with fake Dolphin sessions."""

import melee
import pytest

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.controller import ControllerAction
from hal.eval import local
from hal.eval.scheduling import FrameTiming
from hal.inference.api import PolicySpec
from hal.inference.api import PredictionRequest
from hal.inference.api import RuntimeConfig
from hal.inference.api import action_plan
from hal.sim.session import Matchup
from hal.sim.session import PlayerSetup


def _pre(action: ControllerAction) -> dict:
    return {
        "joystick": {"x": action.main_x, "y": action.main_y},
        "cstick": {"x": action.c_x, "y": action.c_y},
        "triggers_physical": {"l": action.trigger_l, "r": action.trigger_r},
        "buttons_physical": action.buttons,
    }


def _frame(frame_id: int, applied: dict[int, ControllerAction]) -> dict:
    return {
        "id": frame_id,
        "stage": 31,
        "ports": {
            port: {"leader": {"pre": _pre(action), "post": {"character": port}}} for port, action in applied.items()
        },
    }


class _Session:
    def __init__(self, *, end_frame: int = 4) -> None:
        self.frame_id = -2
        self.end_frame = end_frame
        self.submitted: list[dict[int, ControllerAction]] = []

    def start_match(self, _matchup: Matchup) -> dict:
        return _frame(self.frame_id, {1: NEUTRAL_CONTROLLER_ACTION, 2: NEUTRAL_CONTROLLER_ACTION})

    def step(self, inputs: dict[int, ControllerAction]) -> tuple[dict, bool]:
        self.submitted.append(inputs.copy())
        self.frame_id += 1
        applied = {1: inputs.get(1, NEUTRAL_CONTROLLER_ACTION), 2: inputs.get(2, NEUTRAL_CONTROLLER_ACTION)}
        return _frame(self.frame_id, applied), self.frame_id < self.end_frame


class _Policy:
    spec = PolicySpec("fake", "fake", ("stage",), (2,))
    context_frames = 8

    def __init__(self) -> None:
        self.batches: list[tuple[PredictionRequest, ...]] = []
        self.resets = 0

    def reset_prediction(self) -> None:
        self.resets += 1

    def predict(self, requests: tuple[PredictionRequest, ...]):
        self.batches.append(requests)
        return tuple(
            action_plan(request, (ControllerAction(request.stream_id / 10, 0, 0, 0, 0, 0, 0),) * 2)
            for request in requests
        )


def _matchup(*, cpu: bool) -> Matchup:
    return Matchup(
        melee.Stage.FINAL_DESTINATION,
        (
            PlayerSetup(port=1, character=melee.Character.FOX),
            PlayerSetup(port=2, character=melee.Character.FALCO, cpu_level=9 if cpu else 0),
        ),
    )


def test_two_models_share_a_prediction_batch_and_apply_absolute_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(local, "flatten_live_frame", lambda frame, _characters: {"stage": frame["stage"]})
    monkeypatch.setattr(local.Trajectory, "from_capture", lambda frames, _ports: frames)
    session = _Session()
    policy = _Policy()

    result = local.run_local_match(
        session,
        _matchup(cpu=False),
        {1: policy, 2: policy},
        RuntimeConfig(2, (2,)),
        FrameTiming(2, 0, 2, 4),
        max_frames=8,
    )

    assert policy.resets == 1
    assert all({request.stream_id for request in batch} == {1, 2} for batch in policy.batches)
    assert policy.batches[0][0].source_frame == -2
    assert policy.batches[0][0].fixed_actions == (NEUTRAL_CONTROLLER_ACTION,) * 2
    assert [inputs[1].main_x for inputs in session.submitted[:4]] == [0, 0, 0.1, 0.1]
    assert [inputs[2].main_x for inputs in session.submitted[:4]] == [0, 0, 0.2, 0.2]
    assert result.ego_port == 1
    assert result.opponent_port == 2
    assert len(result.trajectory) == 7


def test_cpu_port_stays_internal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(local, "flatten_live_frame", lambda frame, _characters: {"stage": frame["stage"]})
    monkeypatch.setattr(local.Trajectory, "from_capture", lambda frames, _ports: frames)
    session = _Session()
    policy = _Policy()

    local.run_local_match(
        session,
        _matchup(cpu=True),
        {1: policy},
        RuntimeConfig(1, (2,)),
        FrameTiming(2, 0, 2, 4),
        max_frames=8,
    )

    assert all(tuple(inputs) == (1,) for inputs in session.submitted)
    assert all(len(batch) == 1 for batch in policy.batches)


def test_local_match_stops_before_instant_restart_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(local, "flatten_live_frame", lambda frame, _characters: {"stage": frame["stage"]})
    monkeypatch.setattr(local.Trajectory, "from_capture", lambda frames, _ports: frames)

    class RestartingSession(_Session):
        def step(self, inputs: dict[int, ControllerAction]) -> tuple[dict, bool]:
            frame, in_game = super().step(inputs)
            if self.frame_id == 1:
                return _frame(-123, {1: NEUTRAL_CONTROLLER_ACTION, 2: NEUTRAL_CONTROLLER_ACTION}), True
            return frame, in_game

    result = local.run_local_match(
        RestartingSession(),
        _matchup(cpu=True),
        {1: _Policy()},
        RuntimeConfig(1, (2,)),
        FrameTiming(2, 0, 2, 4),
        max_frames=8,
    )

    assert [frame["id"] for frame in result.trajectory] == [-2, -1, 0]


def test_local_match_rejects_missing_observation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(local, "flatten_live_frame", lambda frame, _characters: {"stage": frame["stage"]})

    class SkippingSession(_Session):
        def step(self, inputs: dict[int, ControllerAction]) -> tuple[dict, bool]:
            super().step(inputs)
            self.frame_id += 1
            return _frame(self.frame_id, {1: NEUTRAL_CONTROLLER_ACTION, 2: NEUTRAL_CONTROLLER_ACTION}), True

    with pytest.raises(RuntimeError, match="skipped"):
        local.run_local_match(
            SkippingSession(),
            _matchup(cpu=True),
            {1: _Policy()},
            RuntimeConfig(1, (2,)),
            FrameTiming(2, 0, 2, 4),
            max_frames=8,
        )
