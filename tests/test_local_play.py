"""Local play: a person on a GameCube adapter port against a policy pipe port."""

import multiprocessing as mp
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import melee
import numpy as np
import pytest

import hal.eval.local_play as local_play
import hal.sim.session as session_module
import hal.sim.worker as worker_module
from hal.scripts.policy import _policy_parser
from hal.sim.ipc import ArenaSpec
from hal.sim.ipc import RolloutArena
from hal.sim.rollout import PolicyRuntimeSpec
from hal.sim.session import Matchup
from hal.sim.session import PlayerSetup
from hal.sim.session import Session

PERSON = PlayerSetup(1, melee.Character.FOX, controller_type=melee.ControllerType.GCN_ADAPTER)
POLICY = PlayerSetup(2, melee.Character.FALCO)
MATCHUP = Matchup(melee.Stage.BATTLEFIELD, (PERSON, POLICY))


class _Console:
    def __init__(self) -> None:
        self.adapter_ports: list[tuple[int, melee.ControllerType]] = []

    def setup_dolphin_controller(self, port: int, controller_type: melee.ControllerType) -> None:
        self.adapter_ports.append((port, controller_type))

    def run(self, iso_path: str) -> None:
        pass

    def connect(self) -> bool:
        return True


class _Controller:
    ports: list[int] = []

    def __init__(self, *, console: object, port: int, type: melee.ControllerType, fix_analog_inputs: bool) -> None:
        assert type is melee.ControllerType.STANDARD
        _Controller.ports.append(port)

    def connect(self) -> bool:
        return True


class _MenuHelper:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.stage_selected = False

    def menu_helper_simple(self, **kwargs: object) -> None:
        self.calls.append(kwargs)


class _PlayerState:
    def __init__(self, character: melee.Character) -> None:
        self.character = character


class _GameState:
    def __init__(self, menu_state: melee.Menu, players: dict[int, _PlayerState]) -> None:
        self.menu_state = menu_state
        self.players = players


def test_start_match_gives_the_person_an_adapter_and_the_policy_a_pipe(monkeypatch: pytest.MonkeyPatch) -> None:
    batches: list[bool] = []

    @contextmanager
    def popen(*, dolphin_batch: bool = False) -> Iterator[None]:
        batches.append(dolphin_batch)
        yield

    _Controller.ports = []
    monkeypatch.setattr(session_module.melee, "Controller", _Controller)
    monkeypatch.setattr(session_module, "popen_with_pdeathsig", popen)
    monkeypatch.setattr(session_module, "fix_dolphin_ini_case", lambda _console: None)
    session = Session(iso_path="unused.iso", dolphin_path="unused", dolphin_batch=True)
    console = _Console()
    session._console = console  # type: ignore[assignment]
    session._instrument_controller_flush = lambda _port, _controller: None  # type: ignore[method-assign]
    session._navigate_to_live = lambda: {"id": -123}  # type: ignore[method-assign]

    session.start_match(MATCHUP)

    assert console.adapter_ports == [(1, melee.ControllerType.GCN_ADAPTER)]
    assert _Controller.ports == [2]
    assert set(session._menu_helpers) == {2}
    assert batches == [True]


def test_menus_leave_the_person_alone_and_the_policy_starts_the_match() -> None:
    session = Session(iso_path="unused.iso", dolphin_path="unused")
    helper = _MenuHelper()
    session._matchup = MATCHUP
    session._menu_helpers = {2: helper}  # type: ignore[dict-item]
    session._controllers = {2: object()}  # type: ignore[dict-item]

    session._drive_menus(_GameState(melee.Menu.CHARACTER_SELECT, {}))  # type: ignore[arg-type]

    assert len(helper.calls) == 1
    assert helper.calls[0]["character_selected"] is melee.Character.FALCO
    assert helper.calls[0]["autostart"] is True


def test_live_character_check_accepts_the_persons_choice() -> None:
    session = Session(iso_path="unused.iso", dolphin_path="unused")
    session._matchup = MATCHUP
    players = {1: _PlayerState(melee.Character.MARTH), 2: _PlayerState(melee.Character.FALCO)}

    session._validate_live_characters(_GameState(melee.Menu.IN_GAME, players))  # type: ignore[arg-type]

    players[2] = _PlayerState(melee.Character.FOX)
    with pytest.raises(RuntimeError, match="p2: expected FALCO, got FOX"):
        session._validate_live_characters(_GameState(melee.Menu.IN_GAME, players))  # type: ignore[arg-type]


def _frame(characters: dict[int, melee.Character]) -> dict:
    def post(character: melee.Character) -> dict:
        return {
            "character": int(character.value),
            "position": {"x": 1.0, "y": 2.0},
            "percent": 0.0,
            "shield": 60.0,
            "stock": 4,
            "direction": 1.0,
            "action": 14,
            "jumps_used": 0,
            "airborne": 0,
            "hurtbox_state": 0,
            "hitlag_left": 0.0,
        }

    return {
        "id": -123,
        "start": {"random_seed": 0},
        "ports": {port: {"leader": {"post": post(character)}} for port, character in characters.items()},
        "items": [],
        "stage": int(melee.Stage.BATTLEFIELD.value),
    }


def test_worker_reads_the_persons_character_from_the_game() -> None:
    runtime = PolicyRuntimeSpec(
        context_frames=8, prediction_frames=4, execution_stride=2, committed_frames=0, action_dim=14
    )
    spec = ArenaSpec(workers=1, ring_capacity=runtime.raw_ring_capacity, prediction_frames=4, action_dim=14)
    parent, child = mp.Pipe(duplex=True)
    with RolloutArena.create(spec) as arena:
        state = worker_module._SessionWorkerState(0, child, arena, MATCHUP, (2,), (0,), runtime)
        state.publish(
            _frame({1: melee.Character.MARTH, 2: melee.Character.FALCO}),
            {2: np.zeros(14, dtype=np.float32)},
            reset=True,
        )
    parent.close()
    child.close()

    assert state.metadata["character"] == {1: int(melee.Character.MARTH.value), 2: int(melee.Character.FALCO.value)}


def test_local_session_waits_for_policy_actions_in_real_time(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(local_play, "tested_dolphin_version", lambda _path: None)

    options = local_play.local_session_options("ssbm.iso", "slippi.AppImage", tmp_path, 51441)
    match = local_play.local_matchup(melee.Character.FALCO, melee.Stage.BATTLEFIELD)

    assert options["blocking_input"] is True
    assert options["emulation_speed"] == 1.0
    assert options["enable_ffw"] is False
    assert options["use_exi_inputs"] is False
    assert match.model_ports == (local_play.POLICY_PORT,)
    assert match.matchup.players[0].uses_adapter
    assert not match.matchup.players[1].uses_adapter


def test_play_command_defaults() -> None:
    args = _policy_parser().parse_args(["play", "policy.hal", "--character", "falco", "--stage", "battlefield"])

    assert args.command == "play"
    assert args.character is melee.Character.FALCO
    assert args.stage is melee.Stage.BATTLEFIELD
