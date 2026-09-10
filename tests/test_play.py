from collections import deque
from pathlib import Path
from types import SimpleNamespace

import melee
import pytest
from peppi_py.game import EndMethod

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.controller import ControllerAction
from hal.eval.play import require_completed_replay
from hal.eval.play import run_netplay_match
from hal.inference.api import PolicyInput
from hal.inference.api import PolicyOutput
from hal.inference.api import PolicySpec
from hal.inference.api import RuntimeConfig
from hal.sim.netplay import NetplaySetup


def _pre(action: ControllerAction) -> dict:
    return {
        "joystick": {"x": action.main_x, "y": action.main_y},
        "cstick": {"x": action.c_x, "y": action.c_y},
        "triggers_physical": {"l": action.trigger_l, "r": action.trigger_r},
        "buttons_physical": action.buttons,
    }


def _frame(frame_id: int, action: ControllerAction, *, ended: bool = False) -> dict:
    frame = {
        "id": frame_id,
        "stage": 32 if frame_id == 0 else 8,
        "ports": {
            1: {"leader": {"pre": _pre(action), "post": {"character": 1, "stock": 0 if ended else 4}}},
            2: {
                "leader": {
                    "pre": _pre(NEUTRAL_CONTROLLER_ACTION),
                    "post": {"character": 22, "stock": 4},
                }
            },
        },
    }
    if ended:
        frame["end"] = {"method": 3}
    return frame


class _Session:
    online_delay = 2
    ego_port = 1
    opponent_port = 2

    def __init__(
        self,
        *,
        corrupt_frame: int | None = None,
        replay_frame: int | None = None,
        replay_age: int = 1,
    ) -> None:
        self.frame_id = 0
        self.queue = deque([NEUTRAL_CONTROLLER_ACTION] * self.online_delay)
        self.submitted: list[ControllerAction] = []
        self.corrupt_frame = corrupt_frame
        self.replay_frame = replay_frame
        self.replay_age = replay_age
        self.applied = [NEUTRAL_CONTROLLER_ACTION]

    def start_match(self, _setup: NetplaySetup) -> dict:
        return _frame(0, NEUTRAL_CONTROLLER_ACTION)

    def step(self, action: ControllerAction) -> tuple[dict, bool]:
        self.submitted.append(action)
        due = self.queue.popleft()
        self.queue.append(action)
        self.frame_id += 1
        if self.frame_id == self.corrupt_frame:
            due = ControllerAction(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
        elif self.frame_id == self.replay_frame:
            due = self.applied[-self.replay_age]
        self.applied.append(due)
        in_game = self.frame_id < 9
        return _frame(self.frame_id, due, ended=not in_game), in_game


class _Policy:
    spec = PolicySpec(
        name="fake",
        backend="tests.fake",
        required_observation_fields=("stage", "p1_character", "p2_character"),
        supported_transport_delays=(2,),
    )

    def __init__(self) -> None:
        self.inputs: list[PolicyInput] = []

    def prepare(self, _config: RuntimeConfig) -> None:
        pass

    def step(self, inputs: tuple[PolicyInput, ...]) -> tuple[PolicyOutput, ...]:
        self.inputs.extend(inputs)
        action = ControllerAction(len(self.inputs) / 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
        return (PolicyOutput(inputs[0].stream_id, action),)


def _flatten(frame: dict) -> dict[str, int]:
    matchup = frame["_matchup"]
    return {
        "stage": matchup["stage"],
        "p1_character": matchup["character"][1],
        "p2_character": matchup["character"][2],
    }


def test_match_loop_flushes_delay_and_passes_real_conditioning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hal.eval.play.flatten_canonical_frame", _flatten)
    monkeypatch.setattr("hal.eval.play.Trajectory.from_capture", lambda frames, _ports: frames)
    policy = _Policy()
    session = _Session()
    runtime = RuntimeConfig(1, 2, 2)
    result = run_netplay_match(
        session,
        NetplaySetup(character=melee.Character.FOX, opponent_code="A#1"),
        policy,
        runtime,
        max_frames=10,
    )
    first = policy.inputs[0]
    assert first.reset
    assert first.frame_id == 3
    assert first.applied_action == NEUTRAL_CONTROLLER_ACTION
    assert first.pending_actions == (NEUTRAL_CONTROLLER_ACTION,) * 2
    assert first.observation == {"stage": 8, "p1_character": 1, "p2_character": 22}
    assert session.submitted[:3] == [NEUTRAL_CONTROLLER_ACTION] * 3
    assert session.submitted[3].main_x == 0.1
    assert len(result.trajectory) == 10
    assert result.inference_p95_ms >= 0.0
    assert result.transport_correction_frames == 0


def test_match_loop_accepts_a_recent_slippi_time_sync_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hal.eval.play.flatten_canonical_frame", _flatten)
    monkeypatch.setattr("hal.eval.play.Trajectory.from_capture", lambda frames, _ports: frames)
    result = run_netplay_match(
        _Session(replay_frame=8, replay_age=2),
        NetplaySetup(character=melee.Character.FOX, opponent_code="A#1"),
        _Policy(),
        RuntimeConfig(1, 2, 2),
        max_frames=10,
    )
    assert result.transport_correction_frames == 1


def test_match_loop_detects_controller_transport_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hal.eval.play.flatten_canonical_frame", _flatten)
    session = _Session(corrupt_frame=4)
    with pytest.raises(RuntimeError, match="controller transport mismatch"):
        run_netplay_match(
            session,
            NetplaySetup(character=melee.Character.FOX, opponent_code="A#1"),
            _Policy(),
            RuntimeConfig(1, 2, 2),
            max_frames=10,
        )


def test_match_loop_rejects_delay_mismatch() -> None:
    with pytest.raises(ValueError, match="differs from policy delay"):
        run_netplay_match(
            _Session(),
            NetplaySetup(character=melee.Character.FOX, opponent_code="A#1"),
            _Policy(),
            RuntimeConfig(1, 3),
        )


@pytest.mark.parametrize("method", [EndMethod.TIME, EndMethod.GAME, EndMethod.RESOLVED])
def test_completed_replay_accepts_definitive_game_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: EndMethod,
) -> None:
    replay = tmp_path / "game.slp"
    replay.touch()
    monkeypatch.setattr(
        "hal.eval.play.peppi_py.read_slippi",
        lambda *_args, **_kwargs: SimpleNamespace(end=SimpleNamespace(method=method)),
    )
    assert require_completed_replay(tmp_path, ()) == replay


def test_completed_replay_rejects_no_contest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    replay = tmp_path / "game.slp"
    replay.touch()
    monkeypatch.setattr(
        "hal.eval.play.peppi_py.read_slippi",
        lambda *_args, **_kwargs: SimpleNamespace(end=SimpleNamespace(method=EndMethod.NO_CONTEST)),
    )
    with pytest.raises(RuntimeError, match="NO_CONTEST"):
        require_completed_replay(tmp_path, ())
