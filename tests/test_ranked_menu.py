import io
from pathlib import Path
from unittest.mock import Mock

import melee
import pytest
from PIL import Image

from hal.sim.ranked import RankedMenu
from hal.sim.ranked import RankedScreen
from hal.sim.ranked import ScreenKind
from hal.sim.ranked import read_screen
from hal.sim.ranked import screen_kind


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("Select two stages you do not want to play", ScreenKind.TWO_STRIKES),
        ("Select a stage you do not want to play", ScreenKind.STRIKE),
        ("Select a stage you want to play", ScreenKind.PICK),
        ("Waiting for opponent to strike", ScreenKind.WAIT),
        ("Press START to search", ScreenKind.READY),
        ("Searching for opponent", ScreenKind.WAIT),
        ("Opponent disconnected. Press A to continue", ScreenKind.RETURN),
        ("Please restart Slippi Launcher to verify email and accept rules", ScreenKind.ERROR),
    ],
)
def test_recognizes_observed_prompts(text: str, kind: ScreenKind) -> None:
    assert screen_kind(text) == kind


def test_two_bans_move_to_a_different_stage_and_confirm() -> None:
    now = [10.0]
    screen = [RankedScreen(now[0], ScreenKind.TWO_STRIKES, 311)]
    menu = RankedMenu(lambda: screen[0], clock=lambda: now[0])
    state = melee.GameState()
    state.menu_state = melee.Menu.UNKNOWN_MENU
    controller = Mock()

    def act(view: RankedScreen, expected: melee.Button) -> None:
        screen[0] = view
        assert menu(state, controller)
        assert controller.press_button.call_args.args == (expected,)
        now[0] += 0.1
        assert menu(None, controller)
        now[0] += 0.4

    act(screen[0], melee.Button.BUTTON_A)
    act(RankedScreen(now[0], ScreenKind.TWO_STRIKES, 311), melee.Button.BUTTON_D_RIGHT)
    act(RankedScreen(now[0], ScreenKind.TWO_STRIKES, 395), melee.Button.BUTTON_A)
    act(RankedScreen(now[0], ScreenKind.TWO_STRIKES, confirmation=True), melee.Button.BUTTON_A)


def test_does_not_act_twice_on_one_screenshot_or_on_opponents_turn() -> None:
    now = [10.0]
    screen = [RankedScreen(10, ScreenKind.TWO_STRIKES, 311)]
    menu = RankedMenu(lambda: screen[0], clock=lambda: now[0])
    state = melee.GameState()
    state.menu_state = melee.Menu.UNKNOWN_MENU
    controller = Mock()
    assert menu(state, controller)
    now[0] = 10.1
    assert menu(None, controller)
    now[0] = 10.5
    assert not menu(None, controller)
    screen[0] = RankedScreen(10.5, ScreenKind.WAIT)
    assert not menu(None, controller)
    screen[0] = RankedScreen(0, ScreenKind.TWO_STRIKES, 311)
    assert not menu(None, controller)
    assert controller.press_button.call_count == 1


def test_missing_cursor_does_not_mean_ok() -> None:
    state = melee.GameState()
    state.menu_state = melee.Menu.UNKNOWN_MENU
    menu = RankedMenu(lambda: RankedScreen(10, ScreenKind.TWO_STRIKES), clock=lambda: 10)
    controller = Mock()
    assert not menu(state, controller)
    controller.press_button.assert_not_called()


def test_online_search_uses_prompt_when_coin_flag_is_false() -> None:
    state = melee.GameState()
    state.menu_state = melee.Menu.SLIPPI_ONLINE_CSS
    state.players[1] = melee.PlayerState()
    state.players[1].character = melee.Character.FOX
    state.players[1].coin_down = False
    now = [10.0]
    screen = [ScreenKind.CHARACTER]
    menu = RankedMenu(lambda: RankedScreen(now[0], screen[0]), clock=lambda: now[0])
    controller = Mock()
    assert menu(state, controller)
    assert controller.press_button.call_args.args == (melee.Button.BUTTON_A,)
    now[0] = 10.1
    assert menu(None, controller)
    now[0] = 10.5
    # libmelee reports false even after the online character is locked in.
    screen[0] = ScreenKind.READY
    assert menu(state, controller)
    assert controller.press_button.call_args.args == (melee.Button.BUTTON_START,)


def test_menu_driver_never_sends_inputs_in_game() -> None:
    state = melee.GameState()
    state.menu_state = melee.Menu.IN_GAME
    controller = Mock()
    menu = RankedMenu(lambda: RankedScreen(10, ScreenKind.TWO_STRIKES, 311), clock=lambda: 10)
    assert not menu(state, controller)
    controller.assert_not_called()
    controller.press_button.assert_not_called()


def test_recorded_dim_cursor_is_detected() -> None:
    path = Path(__file__).parent / "fixtures/ranked/two-strikes.png"
    screen = read_screen(
        path.read_bytes(), 10, lambda image: "Select two stages you do not want to play" if image.width == 550 else ""
    )
    assert screen.kind == ScreenKind.TWO_STRIKES
    assert screen.cursor_x == 311
    assert not screen.confirmation


def test_rejects_changed_capture_dimensions() -> None:
    data = io.BytesIO()
    Image.new("RGB", (640, 480)).save(data, format="PNG")
    with pytest.raises(ValueError, match="960 by 720"):
        read_screen(data.getvalue(), 10)


@pytest.mark.parametrize("name", ["confirm.png", "confirm-dim.png", "confirm-character.png"])
def test_confirmation_replaces_the_caption_and_uses_the_gold_ok_border(name: str) -> None:
    png = (Path(__file__).parent / "fixtures/ranked" / name).read_bytes()
    text = Mock(side_effect=AssertionError("confirmation does not need OCR"))
    screen = read_screen(png, 10, text)
    assert screen.kind == ScreenKind.CONFIRM
    assert screen.confirmation
    state = melee.GameState()
    state.menu_state = melee.Menu.UNKNOWN_MENU
    controller = Mock()
    assert RankedMenu(lambda: screen, clock=lambda: 10)(state, controller)
    assert controller.press_button.call_args.args == (melee.Button.BUTTON_A,)


def test_ban_confirmation_and_character_confirmation_each_press_ok() -> None:
    now = [10.0]
    screen = [RankedScreen(10, ScreenKind.CONFIRM, confirmation=True)]
    menu = RankedMenu(lambda: screen[0], clock=lambda: now[0])
    state = melee.GameState()
    state.menu_state = melee.Menu.UNKNOWN_MENU
    controller = Mock()
    assert menu(state, controller)
    now[0] = 10.1
    assert menu(None, controller)
    now[0] = 10.5
    screen[0] = RankedScreen(now[0], ScreenKind.WAIT)
    assert not menu(state, controller)
    now[0] = 11
    screen[0] = RankedScreen(now[0], ScreenKind.CONFIRM, confirmation=True)
    assert menu(state, controller)
    assert [call.args for call in controller.press_button.call_args_list] == [
        (melee.Button.BUTTON_A,),
        (melee.Button.BUTTON_A,),
    ]


def test_delayed_screenshot_cannot_repeat_a_confirmation() -> None:
    now = [10.0]
    screen = [RankedScreen(10, ScreenKind.CONFIRM, confirmation=True)]
    menu = RankedMenu(lambda: screen[0], clock=lambda: now[0])
    state = melee.GameState()
    state.menu_state = melee.Menu.UNKNOWN_MENU
    controller = Mock()
    assert menu(state, controller)
    now[0] = 10.1
    assert menu(None, controller)
    now[0] = 10.5
    screen[0] = RankedScreen(10.05, ScreenKind.CONFIRM, confirmation=True)
    assert not menu(None, controller)
    assert controller.press_button.call_count == 1


def test_search_screen_is_not_a_confirmation() -> None:
    png = (Path(__file__).parent / "fixtures/ranked/searching.png").read_bytes()
    screen = read_screen(png, 10, lambda _image: "Searching for opponent")
    assert screen.kind == ScreenKind.WAIT
    assert not screen.confirmation
