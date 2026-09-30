"""Ranked menu decisions for the tested Slippi 3.6.4 display."""

import io
import re
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import melee
import numpy as np
from PIL import Image

from hal.sim.session import LIVE_MENU_STATES


class ScreenKind(StrEnum):
    UNKNOWN = "unknown"
    WAIT = "wait"
    READY = "ready"
    CONFIRM = "confirm"
    STRIKE = "strike"
    TWO_STRIKES = "two_strikes"
    PICK = "pick"
    CHARACTER = "character"
    COLOR = "color"
    RETURN = "return"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class RankedScreen:
    captured_at: float
    kind: ScreenKind
    cursor_x: int | None = None
    text: str = ""
    confirmation: bool = False


def screen_kind(text: str) -> ScreenKind:
    text = " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())
    if "error" in text or "verify email" in text or "accept rules" in text:
        return ScreenKind.ERROR
    if "disconnect" in text or ("press" in text and ("continue" in text or "return" in text)):
        return ScreenKind.RETURN
    if "waiting" in text or "searching" in text or "connecting" in text or "get ready" in text:
        return ScreenKind.WAIT
    if "start" in text and "search" in text:
        return ScreenKind.READY
    if "select" in text or "choose" in text:
        if "two stages" in text or "2 stages" in text:
            return ScreenKind.TWO_STRIKES
        if "stage" in text:
            return ScreenKind.STRIKE if "not" in text or "ban" in text else ScreenKind.PICK
        if "color" in text:
            return ScreenKind.COLOR
        if "character" in text:
            return ScreenKind.CHARACTER
    return ScreenKind.UNKNOWN


def ocr(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.resize((image.width * 2, image.height * 2)).save(buffer, format="PNG")
    result = subprocess.run(
        ["tesseract", "stdin", "stdout", "--psm", "6"],
        input=buffer.getvalue(),
        capture_output=True,
        check=True,
        timeout=3,
    )
    return result.stdout.decode("utf-8")


def read_screen(png: bytes, captured_at: float, read_text: Callable[[Image.Image], str] = ocr) -> RankedScreen:
    with Image.open(io.BytesIO(png)) as source:
        if source.size != (960, 720):
            raise ValueError("ranked screenshots must be 960 by 720")
        image = source.convert("RGB")
    pixels = np.asarray(image).astype(np.int16)
    red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    yellow = (red > 25) & (green > 20) & (blue < green * 0.6) & (red > green * 0.9)
    # OK replaces the caption. Its tall gold border is outside the stage row.
    left_edge = yellow[355:455, 385:400].sum(axis=0).max() > 40
    right_edge = yellow[355:455, 460:475].sum(axis=0).max() > 40
    bottom_edge = yellow[430:450, 390:470].sum(axis=1).max() > 50
    confirmation = bool(left_edge and right_edge and bottom_edge)
    if confirmation:
        return RankedScreen(captured_at, ScreenKind.CONFIRM, confirmation=True)
    # Exclude player names, connect codes, the timer, and stage icons from OCR.
    caption = read_text(image.crop((210, 385, 760, 443)))
    kind = screen_kind(caption)
    if kind == ScreenKind.UNKNOWN:
        caption = read_text(image.crop((540, 404, 798, 585)))
        kind = screen_kind(caption)
    columns = np.flatnonzero(yellow[320:335, 235:724].sum(axis=0) >= 2)
    cursor_x = int(np.median(columns)) + 235 if len(columns) >= 15 else None
    return RankedScreen(captured_at, kind, cursor_x, caption.strip())


class RankedMenu:
    """Release every press and require a fresh screenshot for ranked choices."""

    def __init__(
        self,
        latest_screen: Callable[[], RankedScreen | None],
        *,
        clock: Callable[[], float] = time.monotonic,
        on_action: Callable[[str], None] = lambda _action: None,
    ) -> None:
        self._latest_screen = latest_screen
        self._clock = clock
        self._on_action = on_action
        self._state: melee.GameState | None = None
        self._release_at = 0.0
        self._next_action = 0.0
        self._used_screen = -1.0
        self._selected_cursor: int | None = None
        self._last_kind = ScreenKind.UNKNOWN

    def __call__(self, state: melee.GameState | None, controller: melee.Controller) -> bool:
        now = self._clock()
        if state is not None:
            self._state = state
        if self._state is None or self._state.menu_state in LIVE_MENU_STATES:
            self._release_at = 0
            return False
        if self._release_at:
            if now < self._release_at:
                return False
            controller.release_all()
            self._release_at = 0
            return True
        if now < self._next_action:
            return False
        state = self._state
        screen = self._latest_screen()
        fresh = (
            screen is not None
            and 0 <= now - screen.captured_at < 2
            and screen.captured_at > self._used_screen
            and screen.captured_at >= self._next_action
        )
        button: melee.Button | None = None
        direction: tuple[float, float] | None = None
        if state.menu_state == melee.Menu.PRESS_START:
            button = melee.Button.BUTTON_START
        elif state.menu_state == melee.Menu.MAIN_MENU:
            target = {
                melee.SubMenu.MAIN_MENU_SUBMENU: 0,
                melee.SubMenu.ONEP_MODE_SUBMENU: 2,
                melee.SubMenu.ONLINE_PLAY_SUBMENU: 0,
            }.get(state.submenu)
            if target is not None:
                if state.menu_selection == target:
                    button = melee.Button.BUTTON_A
                else:
                    direction = (0.5, 0.0 if state.menu_selection < target else 1.0)
        elif state.menu_state == melee.Menu.SLIPPI_ONLINE_CSS:
            player = state.players.get(1)
            if player is None:
                return False
            if player.character != melee.Character.FOX:
                if player.coin_down:
                    button = melee.Button.BUTTON_B
                elif player.cursor.y < 10:
                    direction = (0.5, 1.0)
                elif player.cursor.y > 13:
                    direction = (0.5, 0.0)
                elif player.cursor.x < -23.5:
                    direction = (1.0, 0.5)
                elif player.cursor.x > -20.5:
                    direction = (0.0, 0.5)
                else:
                    button = melee.Button.BUTTON_A
            elif fresh and screen is not None:
                if screen.kind == ScreenKind.READY:
                    button = melee.Button.BUTTON_START
                elif screen.kind == ScreenKind.CHARACTER:
                    button = melee.Button.BUTTON_A
                elif screen.kind == ScreenKind.RETURN:
                    button = melee.Button.BUTTON_Z
                self._used_screen = screen.captured_at
        elif state.menu_state == melee.Menu.UNKNOWN_MENU and fresh and screen is not None:
            self._used_screen = screen.captured_at
            if screen.kind != self._last_kind:
                self._selected_cursor = None
            self._last_kind = screen.kind
            if screen.kind in (ScreenKind.STRIKE, ScreenKind.TWO_STRIKES, ScreenKind.PICK):
                # A toggles a stage. Move to another stage before a second A.
                if screen.confirmation:
                    button = melee.Button.BUTTON_A
                    self._selected_cursor = None
                elif screen.cursor_x is None:
                    return False
                elif self._selected_cursor is not None and abs(screen.cursor_x - self._selected_cursor) < 20:
                    button = melee.Button.BUTTON_D_RIGHT
                else:
                    button = melee.Button.BUTTON_A
                    self._selected_cursor = screen.cursor_x
            elif screen.kind in (ScreenKind.CONFIRM, ScreenKind.CHARACTER, ScreenKind.RETURN):
                button = melee.Button.BUTTON_A
            elif screen.kind == ScreenKind.COLOR:
                # The color-collision dialog requires X once before OK appears.
                button = melee.Button.BUTTON_X if self._selected_cursor is None else melee.Button.BUTTON_A
                self._selected_cursor = 0
        if button is None and direction is None:
            return False
        controller.release_all()
        if button is not None:
            controller.press_button(button)
            self._on_action(button.name)
        elif direction is not None:
            controller.tilt_analog(melee.Button.BUTTON_MAIN, *direction)
            self._on_action(f"stick {direction}")
        self._release_at = now + 0.065
        self._next_action = now + 0.25
        return True
