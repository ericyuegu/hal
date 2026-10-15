"""Re-compose the post's return-conditioning clips (90th vs. 99th percentile) in the reel's theme.

Each clip is a Dolphin frame dump of HAL against the level 9 CPU, starting 60 frames before the
CPU leaves the stage. Under the game, the dark band from ``highlight_reel`` shows each player's
character and controller below that player's percentage, and the example number and stage. The
post plays the two arms side by side, so every example lasts as long as its longer arm in both.

Inputs in ``--sources``: ``windows.json`` (one entry per example, with each arm's replay path and
start frame) and ``clips/p{example}-{arm}.avi``, the dumps. Outputs: ``return-20.webm`` and
``return-120.webm`` with WebP posters.
"""

import json
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import peppi_py
import tyro
from PIL import Image
from PIL import ImageDraw

from scripts.highlight_reel import BG
from scripts.highlight_reel import CARD_WIDTH
from scripts.highlight_reel import FAINT
from scripts.highlight_reel import INK
from scripts.highlight_reel import NOTE_X
from scripts.highlight_reel import PORT_COLORS
from scripts.highlight_reel import RULE
from scripts.highlight_reel import RULE_X
from scripts.highlight_reel import TEAL
from scripts.highlight_reel import Args as ReelArgs
from scripts.highlight_reel import Overlay
from scripts.highlight_reel import Player
from scripts.highlight_reel import decoded_frames
from scripts.highlight_reel import faded
from scripts.highlight_reel import ffmpeg
from scripts.highlight_reel import tracked

FPS = 60
DUMP = (1284, 1056)
FEED = (1208, 993)
BAND = 315
CANVAS = (FEED[0], FEED[1] + BAND)
CONTROLLER_Y = FEED[1] + 88
# P1 and P2 percentage centres in the dump's HUD, as fractions of its width.
HUD = (0.175, 0.41)
# Dump frame k shows game frame start + k + 1 (checked against the in-game timer).
DUMP_LAG = 1
FADE = 12
ARMS = (("rc20", "return-20"), ("rc120", "return-120"))
STAGES = {
    31: "Battlefield",
    32: "Final Destination",
    2: "Fountain of Dreams",
    3: "Pokémon Stadium",
    8: "Yoshi's Story",
    28: "Dream Land",
}
POSTER_FRAME = 30


@dataclass(frozen=True, slots=True)
class Args:
    sources: Path = Path("outputs/return-clips-sources")
    # Examples in playback order, as indices into windows.json.
    picks: tuple[int, ...] = (2, 1, 4, 0)
    output: Path = Path("outputs/return-clips")


@dataclass(frozen=True, slots=True)
class Example:
    stage: str
    players: tuple[Player, ...]
    start: int
    frame_ids: np.ndarray
    dump: Path
    length: int


def load_example(sources: Path, window: dict, n: int, arm: str) -> Example:
    entry = window[arm]
    game = peppi_py.read_slippi(str(sources / entry["path"]))
    if game.frames is None:
        raise ValueError(f"{entry['path']} has no frames")
    ids = np.asarray(game.frames.id)
    # Offline games against the CPU have no rollback, so frame ids must strictly increase.
    if np.any(np.diff(ids) != 1):
        raise ValueError(f"{entry['path']} has repeated or skipped frames")
    kinds = [player.type.name for player in game.start.players]
    if kinds != ["HUMAN", "CPU"]:
        raise ValueError(f"{entry['path']} must have HAL in port 1 and the CPU in port 2, not {kinds}")
    players = []
    for port, (data, label, character) in enumerate(
        zip(game.frames.ports, ("HAL", "Level 9 CPU"), (entry["ego"], entry["opp"]), strict=True)
    ):
        pre = data.leader.pre
        inputs = {
            "buttons": pre.buttons,
            "physical": pre.buttons_physical,
            "jx": pre.joystick.x,
            "jy": pre.joystick.y,
            "cx": pre.cstick.x,
            "cy": pre.cstick.y,
        }
        # These clips keep the port colours: the post shows two runs of the same matchup side by side.
        players.append(
            Player(port, label, character, PORT_COLORS[port], {k: np.asarray(v) for k, v in inputs.items()})
        )
    dump = sources / "clips" / f"p{n}-{arm}.avi"
    frames = imageio_ffmpeg.count_frames_and_secs(str(dump))[0]
    length = min(frames, int(ids[-1]) - entry["start"] - DUMP_LAG + 1)
    return Example(STAGES[entry["stage"]], tuple(players), entry["start"], ids, dump, length)


class Band:
    def __init__(self, overlay: Overlay) -> None:
        self.overlay = overlay

    def card_x(self, port: int) -> int:
        return round(FEED[0] * HUD[port]) - CARD_WIDTH // 2

    @cache  # noqa: B019 - one Band lives for the whole render.
    def panel(self, players: tuple[tuple[str, str], ...], kicker: str) -> Image.Image:
        image = Image.new("RGBA", CANVAS)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, FEED[1], CANVAS[0], CANVAS[1]), fill=BG)
        name = self.overlay.sans(24)
        for port, (label, character) in enumerate(players):
            x = self.card_x(port)
            tracked(draw, (x, FEED[1] + 40), character.upper(), self.overlay.mono(14), FAINT, 2.2)
            draw.text((x, FEED[1] + 70), label, font=name, fill=INK, anchor="ls")
            if label == "HAL":
                draw.rectangle((x, FEED[1] + 77, x + name.getlength(label), FEED[1] + 80), fill=TEAL)
        draw.line((RULE_X, FEED[1] + 28, RULE_X, CANVAS[1] - 28), fill=RULE, width=2)
        tracked(draw, (NOTE_X, FEED[1] + 40), kicker, self.overlay.mono(15), TEAL, 2.2)
        return image

    def frame(self, example: Example, feed: Image.Image, k: int, kicker: str, alpha: float) -> Image.Image:
        index = min(int(np.searchsorted(example.frame_ids, example.start + k + DUMP_LAG)), len(example.frame_ids) - 1)
        image = Image.new("RGBA", CANVAS, BG)
        image.paste(feed.resize(FEED, Image.Resampling.LANCZOS), (0, 0))
        labels = tuple((player.label, player.character) for player in example.players)
        image.alpha_composite(self.panel(labels, kicker))
        for player in example.players:
            image.alpha_composite(self.overlay.controller(player, index), (self.card_x(player.port), CONTROLLER_Y))
        if alpha < 1:
            image = Image.alpha_composite(Image.new("RGBA", CANVAS, BG), faded(image, alpha))
        return image.convert("RGB")


def dump_frames(path: Path, length: int) -> Iterator[Image.Image]:
    """The dump's first ``length`` frames, then its last frame held."""
    last = None
    for k, image in enumerate(decoded_frames(path)):
        if k >= length:
            break
        if image.size != DUMP:
            raise ValueError(f"{path} is {image.size}, not {DUMP}")
        last = image
        yield image
    if last is None:
        raise ValueError(f"{path} has no frames")
    while True:
        yield last


def main(args: Args) -> None:
    windows = json.loads((args.sources / "windows.json").read_text())
    band = Band(Overlay(ReelArgs(), (), (), ()))
    args.output.mkdir(parents=True, exist_ok=True)
    examples = {arm: [load_example(args.sources, windows[n], n, arm) for n in args.picks] for arm, _ in ARMS}
    lengths = [max(examples[arm][i].length for arm, _ in ARMS) for i in range(len(args.picks))]
    for arm, name in ARMS:
        webm = args.output / f"{name}.webm"
        cmd = [
            imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-y",
            "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{CANVAS[0]}x{CANVAS[1]}", "-framerate", str(FPS), "-i", "pipe:0",
            "-c:v", "libvpx-vp9", "-crf", "33", "-b:v", "0", "-row-mt", "1", "-deadline", "good", "-cpu-used", "2",
            "-pix_fmt", "yuv420p", "-an", str(webm),
        ]  # fmt: skip
        with subprocess.Popen(cmd, stdin=subprocess.PIPE) as process:
            assert process.stdin is not None
            for i, (example, length) in enumerate(zip(examples[arm], lengths, strict=True)):
                kicker = f"EXAMPLE {i + 1} OF {len(args.picks)}  ·  {example.stage.upper()}"
                for k, feed in zip(range(length), dump_frames(example.dump, example.length), strict=False):
                    alpha = min(1, (k + 1) / FADE, (length - k) / FADE)
                    process.stdin.write(band.frame(example, feed, k, kicker, alpha).tobytes())
                print(name, i, length, flush=True)
            process.stdin.close()
            if process.wait():
                raise RuntimeError(f"{webm} encode failed")
        poster = args.output / f"{name}_poster.webp"
        ffmpeg(
            "-ss",
            str(POSTER_FRAME / FPS),
            "-i",
            str(webm),
            "-frames:v",
            "1",
            "-c:v",
            "libwebp",
            "-quality",
            "82",
            str(poster),
        )
        print(webm, webm.stat().st_size, flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
