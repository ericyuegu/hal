"""Compose the stream card and controller from packaged static layers."""

from collections import deque
from pathlib import Path

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.controller import ControllerAction
from hal.netplay_service.domain import CHARACTERS
from hal.netplay_service.domain import IMITATIONS
from hal.netplay_service.stream import GameStreamState
from hal.netplay_service.stream import StreamState
from hal.netplay_service.stream import difficulty_label
from hal.wire import BUTTON_BITS

ASSETS = Path(__file__).with_name("overlay_assets")
PANEL_SIZE = (456, 608)
CONTROLLER_POSITION = (58, 342)
CONTROLLER_SIZE = (340, 256)
_BUTTONS = {
    "aButton": "a",
    "bButton": "b",
    "xButton": "x",
    "yButton": "y",
    "zButton": "z",
    "lTrigger": "l",
    "rTrigger": "r",
    "startButton": "start",
}
_CHARACTER_LABELS = {choice.value: choice.label for choice in CHARACTERS}
_IMITATION_LABELS = {choice.value: choice.label for choice in IMITATIONS}


def bar_endpoint(value: float) -> int:
    return 228 + round(max(-1, min(1, value / 120)) * 196)


class PanelRenderer:
    def __init__(self) -> None:
        self._sans = {
            size: ImageFont.truetype(str(ASSETS / "fonts/SchibstedGrotesk-Regular.ttf"), size)
            for size in (17, 20, 24, 28)
        }
        self._bold = {
            size: ImageFont.truetype(str(ASSETS / "fonts/SchibstedGrotesk-SemiBold.ttf"), size)
            for size in (32, 36, 46)
        }
        self._mono = ImageFont.truetype(str(ASSETS / "fonts/IBMPlexMono-Regular.ttf"), 13)
        self._controller = Image.open(ASSETS / "controller/base.png").convert("RGBA")
        self._buttons = {name: Image.open(ASSETS / f"controller/{name}.png").convert("RGBA") for name in _BUTTONS}
        self._sprites = {
            choice.value: Image.open(ASSETS / f"characters/{choice.value}.png").convert("RGBA")
            for choice in CHARACTERS
        }
        self._state: StreamState | None = None
        self._activity = ""
        self._base = Image.new("RGBA", PANEL_SIZE, "white")
        self._trails: deque[ControllerAction] = deque(maxlen=5)
        self._game_id: str | None = None
        self._value_key: tuple[str, int] | None = None
        self._value_layer = Image.new("RGBA", PANEL_SIZE)

    def _static(self, state: StreamState, activity: str) -> Image.Image:
        if state == self._state and activity == self._activity:
            return self._base.copy()
        self._state, self._activity = state, activity
        image = Image.new("RGBA", PANEL_SIZE, "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 455, 279), fill="#FAFBFB", outline="#E4E7EA")
        draw.rectangle((0, 0, 455, 2), fill="#0E6E63")
        draw.ellipse((32, 33, 40, 41), fill="#0E6E63")
        label = "SELF-PLAY" if isinstance(state, GameStreamState) and state.mode == "self-play" else "LIVE"
        draw.text((52, 26), label, font=self._mono, fill="#0E6E63")
        if isinstance(state, GameStreamState):
            draw.text((394, 26), f"GAME {state.game_number}", anchor="ra", font=self._mono, fill="#5C646C")
            sprite = self._sprites[state.character]
            sprite = sprite.resize((48, 48), Image.Resampling.NEAREST)
            image.alpha_composite(sprite, (32, 85))
            draw.text((98, 61), "PLAYING AS", font=self._mono, fill="#5C646C")
            name = _CHARACTER_LABELS[state.character]
            font = self._bold[46] if self._bold[46].getlength(name) <= 326 else self._bold[32]
            draw.text((96, 76), name, font=font, fill="#0D1013")
            draw.text((32, 158), "IMITATING", font=self._mono, fill="#5C646C")
            difficulty = f"Difficulty: {difficulty_label(state.desired_return)}"
            width = self._sans[17].getlength(difficulty) + 24
            imitation = _IMITATION_LABELS[state.imitation]
            font = next(
                self._sans[size]
                for size in (28, 24, 20, 17)
                if self._sans[size].getlength(imitation) <= 424 - width - 32 - 16
            )
            draw.text((32, 198), imitation, anchor="lm", font=font, fill="#0D1013")
            draw.rounded_rectangle((424 - width, 180, 424, 215), radius=18, fill="#E6F0EE")
            draw.text((412, 185), difficulty, anchor="ra", font=self._sans[17], fill="#0E6E63")
            controller_label = f"HAL INPUTS · PORT {state.port}" if state.mode == "self-play" else "HAL INPUTS"
        else:
            draw.text((32, 82), "Waiting for a challenger", font=self._sans[24], fill="#0D1013")
            draw.text((32, 128), activity, font=self._sans[20], fill="#5C646C")
            controller_label = "HAL INPUTS"
        draw.text((228, 308), controller_label, anchor="ma", font=self._mono, fill="#5C646C")
        self._base = image
        return image.copy()

    def render(self, state: StreamState, inputs: ControllerAction, value: float | None, activity: str) -> Image.Image:
        image = self._static(state, activity)
        identity = state.game_id if isinstance(state, GameStreamState) else None
        if identity != self._game_id or inputs == NEUTRAL_CONTROLLER_ACTION:
            self._trails.clear()
            self._game_id = identity
        self._trails.append(inputs)
        value_key = None if value is None else (f"{value:+.1f}", bar_endpoint(value))
        if value_key != self._value_key:
            self._value_key = value_key
            self._value_layer = Image.new("RGBA", PANEL_SIZE)
        if value_key is not None and self._value_layer.getbbox() is None:
            draw = ImageDraw.Draw(self._value_layer)
            draw.text((32, 228), "STATE VALUE", font=self._mono, fill="#5C646C")
            draw.text((424, 221), value_key[0], anchor="ra", font=self._sans[20], fill="#0E6E63")
            draw.rounded_rectangle((32, 258, 424, 264), radius=3, fill="#DDE7E3")
            endpoint = value_key[1]
            draw.rounded_rectangle(
                (min(228, endpoint), 258, max(228, endpoint), 264),
                radius=3,
                fill="#0E6E63" if endpoint >= 228 else "#BA5964",
            )
            draw.line((228, 254, 228, 268), fill="#8A9997")
        image.alpha_composite(self._value_layer)
        controller = self._controller.copy()
        for name, button in _BUTTONS.items():
            if inputs.buttons & BUTTON_BITS[button]:
                controller.alpha_composite(self._buttons[name])
        # Partial shoulder travel is visible separately from the physical digital switch.
        for x, amount in ((40, inputs.trigger_l), (267, inputs.trigger_r)):
            if amount > 0:
                ImageDraw.Draw(controller).rounded_rectangle(
                    (x, 9, x + round(32 * amount), 13), radius=2, fill="#0E6E63"
                )
        draw_controller = ImageDraw.Draw(controller)
        for index, action in enumerate(self._trails):
            last = index == len(self._trails) - 1
            for x, y, travel, radius, sx, sy, color in (
                (104.7, 274.5, 34, 28, action.main_x, action.main_y, "#879D99"),
                (395.2, 411.6, 19.5, 20, action.c_x, action.c_y, "#E2BE3C"),
            ):
                px, py = (x + sx * travel + 4) * 340 / 594, (y - sy * travel - 112) * 256 / 448
                r = radius * 340 / 594 * (1 if last else 0.35)
                draw_controller.ellipse((px - r, py - r, px + r, py + r), fill=color if last else "#B5C7C2")
        image.alpha_composite(controller, CONTROLLER_POSITION)
        return image
