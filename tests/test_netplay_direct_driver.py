from dataclasses import dataclass
from dataclasses import field

import melee
import pytest

from hal.sim.netplay import DirectMenuDriver
from hal.sim.netplay import DirectSelection
from hal.sim.netplay import PlayerDisconnected
from hal.sim.netplay import PlayerIdle
from hal.sim.netplay import PlayerNoShow


@dataclass
class _Cursor:
    x: float = 0.0
    y: float = 0.0


@dataclass
class _Player:
    character: melee.Character = melee.Character.FOX
    costume: int = 0
    coin_down: bool = False
    cursor: _Cursor = field(default_factory=_Cursor)
    cpu_level: int = 0
    is_holding_cpu_slider: bool = False
    controller_status: melee.ControllerStatus = melee.ControllerStatus.CONTROLLER_HUMAN


@dataclass
class _State:
    menu_state: melee.Menu
    submenu: melee.SubMenu = melee.SubMenu.ONLINE_CSS
    frame: int = 0
    menu_selection: int = 0
    ready_to_start: int = 0
    players: dict[int, _Player] = field(default_factory=lambda: {1: _Player(), 2: _Player()})


class _Controller:
    port = 1

    def __init__(self) -> None:
        self.pressed: list[melee.Button] = []
        self.prev = melee.ControllerState()

    def press_button(self, button: melee.Button) -> None:
        self.pressed.append(button)
        self.prev.button[button] = True

    def release_button(self, button: melee.Button) -> None:
        self.prev.button[button] = False

    def release_all(self) -> None:
        for button in list(self.prev.button):
            self.prev.button[button] = False

    def tilt_analog(self, *_args: object) -> None:
        pass


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _driver(selection: list[DirectSelection], locks: list[int], clock: _Clock) -> DirectMenuDriver:
    return DirectMenuDriver(
        opponent_code="CRYO#610",
        selection=lambda: selection[-1],
        lock_requests=lambda: locks[-1],
        connect_timeout_seconds=60,
        idle_timeout_seconds=300,
        hold_seconds=5,
        hold_cap_seconds=30,
        probe_interval_seconds=2,
        clock=clock,
    )


FOX = DirectSelection(1, melee.Character.FOX, 0, melee.Stage.RANDOM_STAGE, "IBDW#0")
FALCO = DirectSelection(2, melee.Character.FALCO, 0, melee.Stage.POKEMON_STADIUM, "IBDW#0")


def _starts(controller: _Controller) -> int:
    return controller.pressed.count(melee.Button.BUTTON_START)


def _between_games(driver: DirectMenuDriver) -> None:
    driver.connected = True
    driver.begin_game()


def test_hold_restarts_on_change_and_locks_latest() -> None:
    clock, selection = _Clock(), [FOX]
    driver = _driver(selection, [0], clock)
    _between_games(driver)
    controller = _Controller()
    css = _State(melee.Menu.SLIPPI_ONLINE_CSS)
    driver(css, controller)  # type: ignore[arg-type]
    assert driver.phase == "character_select" and driver.deadline == pytest.approx(105.0)
    clock.now = 103.0
    selection.append(FALCO)
    css.players[1].character = melee.Character.FALCO
    driver(css, controller)  # type: ignore[arg-type]
    assert driver.deadline == pytest.approx(108.0)
    clock.now = 107.9
    driver(css, controller)  # type: ignore[arg-type]
    assert driver.locked is None and _starts(controller) == 0
    clock.now = 108.1
    css.frame = 1
    driver(css, controller)  # type: ignore[arg-type]
    assert driver.locked == FALCO and _starts(controller) == 1


def test_hold_is_capped() -> None:
    clock, selection = _Clock(), [FOX]
    driver = _driver(selection, [0], clock)
    _between_games(driver)
    css = _State(melee.Menu.SLIPPI_ONLINE_CSS, frame=1)
    controller = _Controller()
    for step in range(40):
        clock.now = 100.0 + step
        selection.append(DirectSelection(step + 2, melee.Character.FOX, 0, melee.Stage.RANDOM_STAGE, f"P{step}#1"))
        driver(css, controller)  # type: ignore[arg-type]
        if driver.locked is not None:
            break
    assert driver.locked is not None and clock.now <= 131.0


def test_lock_request_locks_now() -> None:
    clock, locks = _Clock(), [0]
    driver = _driver([FOX], locks, clock)
    _between_games(driver)
    css = _State(melee.Menu.SLIPPI_ONLINE_CSS, frame=1)
    controller = _Controller()
    driver(css, controller)  # type: ignore[arg-type]
    locks.append(1)
    driver(css, controller)  # type: ignore[arg-type]
    assert driver.locked == FOX


def test_probe_presses_start_every_interval_after_lock() -> None:
    clock, locks = _Clock(), [0]
    driver = _driver([FOX], locks, clock)
    _between_games(driver)
    locks.append(1)
    css = _State(melee.Menu.SLIPPI_ONLINE_CSS, frame=1)
    controller = _Controller()
    driver(css, controller)  # type: ignore[arg-type]
    assert _starts(controller) == 1
    clock.now = 101.0
    driver(css, controller)  # type: ignore[arg-type]
    assert _starts(controller) == 1
    clock.now = 102.1
    driver(css, controller)  # type: ignore[arg-type]
    assert _starts(controller) == 2


def test_code_entry_held_after_connection_means_disconnected() -> None:
    clock = _Clock()
    driver = _driver([FOX], [0], clock)
    _between_games(driver)
    entry = _State(melee.Menu.SLIPPI_ONLINE_CSS, submenu=melee.SubMenu.NAME_ENTRY_SUBMENU)
    driver(entry, _Controller())  # type: ignore[arg-type]
    clock.now += 1.9
    driver(entry, _Controller())  # type: ignore[arg-type]
    clock.now += 0.2
    with pytest.raises(PlayerDisconnected):
        driver(entry, _Controller())  # type: ignore[arg-type]


def test_a_starting_match_is_not_a_disconnect() -> None:
    # Slippi reports the code-entry submenu for about a second while a match starts;
    # the ready-to-fight banner tells it apart from the real code-entry screen.
    clock = _Clock()
    driver = _driver([FOX], [0], clock)
    _between_games(driver)
    starting = _State(melee.Menu.SLIPPI_ONLINE_CSS, submenu=melee.SubMenu.NAME_ENTRY_SUBMENU, ready_to_start=255)
    for _ in range(10):
        clock.now += 1.0
        controller = _Controller()
        driver(starting, controller)  # type: ignore[arg-type]
        assert controller.pressed == []


def test_idle_timeout_at_character_select() -> None:
    clock = _Clock()
    driver = _driver([FOX], [0], clock)
    _between_games(driver)
    css = _State(melee.Menu.SLIPPI_ONLINE_CSS)
    driver(css, _Controller())  # type: ignore[arg-type]
    clock.now = 400.1
    with pytest.raises(PlayerIdle):
        driver(css, _Controller())  # type: ignore[arg-type]


def test_no_show_while_waiting_for_the_player() -> None:
    clock = _Clock()
    driver = _driver([FOX], [0], clock)
    driver.begin_game()
    driver(  # type: ignore[arg-type]
        _State(melee.Menu.SLIPPI_ONLINE_CSS, submenu=melee.SubMenu.NAME_ENTRY_SUBMENU),
        _Controller(),
    )
    assert driver.phase == "waiting_for_player"
    clock.now = 160.1
    with pytest.raises(PlayerNoShow):
        driver(_State(melee.Menu.SLIPPI_ONLINE_CSS), _Controller())  # type: ignore[arg-type]


def test_frozen_stadium_toggle_is_pressed_once_per_dolphin_session() -> None:
    clock = _Clock()
    selection = [DirectSelection(1, melee.Character.FOX, 0, melee.Stage.POKEMON_STADIUM, "IBDW#0")]
    driver = _driver(selection, [0], clock)
    for game in range(2):
        _between_games(driver)
        controller = _Controller()
        stage_select = _State(melee.Menu.STAGE_SELECT, frame=30)
        stage_select.players[1].cursor = _Cursor(15, 3.5)
        for frame in range(30, 120):
            stage_select.frame = frame
            driver(stage_select, controller)  # type: ignore[arg-type]
        toggles = controller.pressed.count(melee.Button.BUTTON_Z)
        assert toggles == (1 if game == 0 else 0)


def test_on_change_reports_deadline_move_and_lock() -> None:
    clock = _Clock()
    changes: list[None] = []
    driver = DirectMenuDriver(
        opponent_code="CRYO#610",
        selection=lambda: FOX,
        lock_requests=lambda: 0,
        connect_timeout_seconds=60,
        idle_timeout_seconds=300,
        hold_seconds=5,
        hold_cap_seconds=30,
        probe_interval_seconds=2,
        on_change=lambda: changes.append(None),
        clock=clock,
    )
    _between_games(driver)
    changes.clear()
    css = _State(melee.Menu.SLIPPI_ONLINE_CSS, frame=1)
    controller = _Controller()
    driver(css, controller)  # type: ignore[arg-type]
    assert len(changes) == 1
    clock.now = 105.1
    driver(css, controller)  # type: ignore[arg-type]
    assert len(changes) == 2


def test_a_lock_request_from_an_earlier_game_does_not_skip_the_hold() -> None:
    clock, locks = _Clock(), [0]
    driver = _driver([FOX], locks, clock)
    _between_games(driver)
    css = _State(melee.Menu.SLIPPI_ONLINE_CSS, frame=1)
    controller = _Controller()
    locks.append(1)
    driver(css, controller)  # type: ignore[arg-type]
    assert driver.locked == FOX
    locks.append(2)  # pressed after HAL already locked in
    _between_games(driver)
    driver(css, controller)  # type: ignore[arg-type]
    assert driver.locked is None
    clock.now += 5.1
    driver(css, controller)  # type: ignore[arg-type]
    assert driver.locked == FOX


def test_polls_without_a_new_state_leave_presses_alone() -> None:
    # The session polls far faster than frames arrive; flushing a release between
    # frames would cancel every press before the game samples input.
    driver = _driver([FOX], [0], _Clock())
    controller = _Controller()
    controller.press_button(melee.Button.BUTTON_A)
    assert driver(None, controller) is False  # type: ignore[arg-type]
    assert controller.prev.button[melee.Button.BUTTON_A] is True
