"""Render the ranked-ladder highlight reel for the Hillclimbing Melee post.

The reel cuts Twitch VOD captures of HAL's ranked games into one edit. Under the game feed, each
player's controller sits directly below that player's in-game percentage, and a timeline names
each step of the highlighted exchange as it happens. Each slowed moment plays once at full
speed and then once as a half-speed replay.

Inputs:
- ``--inputs``: per-game controller inputs read from the Slippi replays (JSON, see ``load_games``).
- ``--sources``: the VOD captures, in the order that ``CLIPS`` indexes them.
- ``--controller-svg``: SlippiLab's ``Inputs.tsx``, which holds the GameCube controller SVG.
- ``--blog``: the blog repository, for its fonts (Schibsted Grotesk and IBM Plex Mono).

Outputs in ``--output``: the MP4 master, the post's WebM and WebP poster, the edit timeline,
a contact sheet, and a manifest with input and output hashes.
"""

import hashlib
import io
import json
import re
import subprocess
from collections.abc import Iterator
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass
from functools import cache
from functools import lru_cache
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import tyro
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

FPS = 60
# Slippi frames run at 60000/1001 Hz; VOD timestamps are seconds.
REPLAY_HZ = 60000 / 1001
# The game image inside every source capture, and where it lands in the reel.
SOURCE_CROP = (304, 0, 1312, 1006)
CANVAS = (1288, 1310)
FEED = (40, 40, 1208, 926)

# Tokens from ericyuegu2/src/styles/index-rail.css and figures.css, which the post renders with.
PAPER = "#FFFFFF"
INK = "#0D1013"
GHOST = "#A3ABB2"
FAINT = "#69737C"
IDLE = "#B4BCC2"
DOTTED = "#D5DCE0"
ACCENT = "#0E6E63"
CARD = "#F1F3F5"
SANS_FONT = "node_modules/@fontsource-variable/schibsted-grotesk/files/schibsted-grotesk-latin-wght-normal.woff2"
MONO_FONT = "node_modules/@fontsource/ibm-plex-mono/files/ibm-plex-mono-latin-400-normal.woff2"


@dataclass(frozen=True, slots=True)
class Clip:
    source: int
    game: int
    start: float
    end: float
    speed: float
    replay: bool
    # Replay frame = round(VOD seconds * REPLAY_HZ + offset).
    offset: float


@dataclass(frozen=True, slots=True)
class Segment:
    clip: Clip
    frames: int
    output_frame: int


@dataclass(frozen=True, slots=True)
class Cue:
    start: int
    end: int
    steps: tuple[str, ...]
    # Replay frame where each step after the first begins; the first runs from ``start``.
    step_frames: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class Player:
    port: int
    label: str
    character: str
    inputs: dict[str, np.ndarray]


@dataclass(frozen=True, slots=True)
class Game:
    first_frame: int
    players: tuple[Player, ...]

    @property
    def last_frame(self) -> int:
        return self.first_frame + len(self.players[0].inputs["buttons"]) - 1


# Game 2's live clip ends at 39.55 s because its source crossfades into game 3 from 39.57 s.
CLIPS = (
    Clip(1, 0, 54.4, 80.7, 1, False, -544),
    Clip(1, 1, 105.4, 111.2, 1, False, -6281),
    Clip(1, 1, 106.95, 108.75, 0.5, True, -6281),
    Clip(0, 2, 34.3, 39.55, 1, False, 18),
    Clip(0, 2, 36.55, 38.75, 0.5, True, 18),
    Clip(0, 3, 40.4, 51.5, 1, False, -2423),
    Clip(2, 4, 58.7, 62.1, 1, False, -426),
    Clip(2, 4, 59.75, 61.2, 0.5, True, -426),
    Clip(2, 4, 62.1, 71.5, 1, False, -426),
    Clip(3, 5, 92, 101.6, 1, False, -397.7),
    Clip(4, 6, 177.5, 191.3, 1, False, -9854),
)
# Card labels; "YL" is Young Link, whose full name does not fit beside "Grandmaster".
OPPONENTS = ("Falco", "Falco", "Sheik", "Marth", "Ganondorf", "YL", "Fox")
# Step frames are HAL's action changes in the replays (the bair, shine, up-smash, ... onsets).
CUES = (
    (
        Cue(2798, 2898, ("Shine out of shield",)),
        Cue(3018, 3110, ("Late bair", "shine knockdown"), (3062,)),
        Cue(3110, 3230, ("Roll catch", "up-smash", "dash attack"), (3118, 3171)),
        Cue(3230, 3320, ("Up-tilt", "up-air"), (3254,)),
        Cue(3320, 3440, ("Soft reverse bair", "hard bair"), (3353,)),
        Cue(3455, 3540, ("SHFFL nair",)),
        Cue(3550, 3640, ("Laser during recovery",)),
        Cue(3805, 3896, ("Tech-roll away",)),
        Cue(3896, 4050, ("Dair on shield", "grab punish"), (3916,)),
    ),
    (
        Cue(150, 220, ("Double jump", "Armada shine"), (184,)),
        Cue(227, 357, ("Up-B back to stage",)),
    ),
    (
        Cue(2090, 2190, ("DBZ trade",)),
        Cue(2220, 2285, ("Side-B", "platform slide-off"), (2257,)),
        Cue(2300, 2380, ("Shine turnaround", "wavedash to ledge"), (2311,)),
    ),
    (Cue(85, 270, ("Up-air juggle",)),),
    (
        Cue(3160, 3260, ("Ledgedash", "turnaround up-tilt"), (3189,)),
        Cue(3425, 3575, ("Back-air wall",)),
        Cue(3680, 3810, ("Running up-air catches ledge jump",)),
    ),
    (
        Cue(5310, 5400, ("Shine", "drill"), (5336,)),
        Cue(5400, 5510, ("Double up-air on the platform",)),
        Cue(5605, 5686, ("Shine spike",)),
    ),
    (
        Cue(960, 1060, ("Side-B back to stage",)),
        Cue(1080, 1230, ("Up-smash", "up-air"), (1139,)),
        Cue(1410, 1520, ("Shine", "wavedash to ledge"), (1486,)),
    ),
)
# A clip may run this far past its replay's last frame onto the game-end screen.
END_SCREEN_FRAMES = 20
# P1 and P2 percentage centres in the in-game HUD, as fractions of the feed width.
HUD = (0.189, 0.41)
CARD_WIDTH = 200
BAND_Y = FEED[1] + FEED[3] + 50
NOTE_X = FEED[0] + round(FEED[2] * 0.58)
# Fade through white on the feed at each cut into or out of a replay, peaking on the cut.
DIP_OUT, DIP_IN, DIP_PEAK = 8, 12, 0.9
POSTER = (0, 3150)


@dataclass(frozen=True, slots=True)
class Args:
    inputs: Path = Path("outputs/hal-highlights-v2/inputs-private.json")
    sources: tuple[Path, ...] = (
        Path.home() / "src/ericyuegu2/HAL-highlights.webm",
        Path("outputs/hal-highlights-v3/sources/v2890236934-normalized.mp4"),
        Path("outputs/hal-highlights-v3/sources/v2890236932-normalized.mp4"),
        Path("outputs/hal-highlights-v2/sources/v2890236933.mp4"),
        Path("outputs/hal-highlights-v2/sources/v2890236935.mp4"),
    )
    controller_svg: Path = Path.home() / "src/slippilab/src/components/panels/Inputs.tsx"
    blog: Path = Path.home() / "src/ericyuegu2"
    output: Path = Path("outputs/hal-highlights")
    # Write still previews of representative frames instead of rendering the reel.
    preview: bool = False


def plan(clips: Sequence[Clip]) -> tuple[Segment, ...]:
    segments: list[Segment] = []
    cursor = 0
    for clip in clips:
        if clip.end <= clip.start or clip.speed <= 0:
            raise ValueError(f"invalid clip {clip}")
        frames = round((clip.end - clip.start) / clip.speed * FPS)
        segments.append(Segment(clip, frames, cursor))
        cursor += frames
    return tuple(segments)


def locate(segments: Sequence[Segment], n: int) -> tuple[Segment, float, int]:
    """Return the segment, source seconds, and replay frame shown at output frame ``n``."""
    for segment in segments:
        if segment.output_frame <= n < segment.output_frame + segment.frames:
            clip = segment.clip
            t = clip.start + (n - segment.output_frame) / FPS * clip.speed
            return segment, t, round(t * REPLAY_HZ + clip.offset)
    raise ValueError(f"output frame {n} is outside the edit")


def current_step(cue: Cue, frame: int) -> int:
    return sum(frame >= start for start in cue.step_frames)


def replay_cuts(segments: Sequence[Segment]) -> tuple[int, ...]:
    return tuple(
        segment.output_frame
        for previous, segment in zip(segments, segments[1:], strict=False)
        if segment.clip.replay or previous.clip.replay
    )


def replay_dip(cuts: Sequence[int], n: int) -> float:
    level = 0.0
    for cut in cuts:
        if cut - DIP_OUT <= n < cut:
            level = max(level, (n - cut + DIP_OUT + 1) / DIP_OUT)
        elif cut <= n < cut + DIP_IN:
            level = max(level, 1 - (n - cut) / DIP_IN)
    return level * level * (3 - 2 * level) * DIP_PEAK


def validate_cues(cues: Sequence[Sequence[Cue]]) -> None:
    for game in cues:
        for cue in game:
            if len(cue.step_frames) != len(cue.steps) - 1:
                raise ValueError(f"{cue.steps} needs {len(cue.steps) - 1} step frames")
            if list(cue.step_frames) != sorted(cue.step_frames) or not all(
                cue.start < frame < cue.end for frame in cue.step_frames
            ):
                raise ValueError(f"step frames for {cue.steps} must increase inside the cue")


def confirmed_rows(frames: np.ndarray) -> np.ndarray:
    """Row of each frame id's last occurrence, ordered by frame id.

    Netplay replays store rollbacks: a frame id repeats, or the stream rewinds, and the last
    occurrence is the confirmed frame. Indexing rows by ``frame - first_frame`` drifts by one
    for every rolled-back row before it.
    """
    if frames.size == 0:
        raise ValueError("replay has no frames")
    first = int(frames.min())
    rows = np.full(int(frames.max()) - first + 1, -1)
    rows[frames - first] = np.arange(frames.size)
    if np.any(rows < 0):
        raise ValueError("replay skips a frame id")
    return rows


def load_games(path: Path) -> tuple[Game, ...]:
    """Read the replay-derived input manifest: one entry per game, two players each."""
    raw = json.loads(path.read_text())
    if len(raw) != len(OPPONENTS):
        raise ValueError(f"{path} has {len(raw)} games; the edit uses {len(OPPONENTS)}")
    games: list[Game] = []
    for index, game in enumerate(raw):
        rows = confirmed_rows(np.asarray(game["frames"]))
        players: list[Player] = []
        for player in game["players"]:
            inputs = {key: np.asarray(player["data"][key]) for key in ("buttons", "physical", "jx", "jy", "cx", "cy")}
            if any(len(values) != len(game["frames"]) for values in inputs.values()):
                raise ValueError(f"game {index} port {player['port']} inputs do not cover its frames")
            character = "Fox" if player["label"] == "HAL" else OPPONENTS[index]
            confirmed = {key: values[rows] for key, values in inputs.items()}
            players.append(Player(int(player["port"]), str(player["label"]), character, confirmed))
        if sorted(player.port for player in players) != [0, 1]:
            raise ValueError(f"game {index} must use ports 1 and 2")
        games.append(Game(int(min(game["frames"])), tuple(sorted(players, key=lambda player: player.port))))
    return tuple(games)


def ffmpeg(*args: str) -> None:
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-y", *args], check=True)


def frame_count(path: Path) -> int:
    return imageio_ffmpeg.count_frames_and_secs(str(path))[0]


def feed_filter() -> str:
    x, y, w, h = FEED
    cx, cy, cw, ch = SOURCE_CROP
    return f"crop={cw}:{ch}:{cx}:{cy},scale={w}:{h}:flags=lanczos,pad={CANVAS[0]}:{CANVAS[1]}:{x}:{y}:color=white,setsar=1"


def encode_segment(segment: Segment, sources: Sequence[Path], path: Path) -> None:
    clip = segment.clip
    span = clip.end - clip.start
    duration = segment.frames / FPS
    # Replays keep the game audio, slowed and ducked, so the cut back to live play keeps its level.
    gain = ",volume=0.45" if clip.replay else ""
    video = f"[0:v]trim=duration={span},setpts=(PTS-STARTPTS)/{clip.speed},tpad=stop_mode=clone:stop_duration=0.1,fps={FPS},{feed_filter()}[v]"
    audio = (
        f"[0:a]atrim=duration={span},asetpts=PTS-STARTPTS,atempo={clip.speed}{gain},apad,atrim=duration={duration}[a]"
    )
    ffmpeg(
        "-ss", str(clip.start), "-i", str(sources[clip.source]), "-filter_complex", f"{video};{audio}",
        "-map", "[v]", "-map", "[a]", "-frames:v", str(segment.frames), "-t", str(duration),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", str(path),
    )  # fmt: skip
    if frame_count(path) != segment.frames:
        raise RuntimeError(f"{path} does not have {segment.frames} frames")


def encode_base(segments: Sequence[Segment], sources: Sequence[Path], work: Path) -> Path:
    parts = [work / f"{index:02d}.mp4" for index in range(len(segments))]
    for segment, part in zip(segments, parts, strict=True):
        encode_segment(segment, sources, part)
    total = sum(segment.frames for segment in segments)
    inputs = [arg for part in parts for arg in ("-i", str(part))]
    chains = [
        f"[{i}:v]setpts=PTS-STARTPTS[v{i}];[{i}:a]atrim=duration={s.frames / FPS},asetpts=PTS-STARTPTS[a{i}]"
        for i, s in enumerate(segments)
    ]
    labels = "".join(f"[v{i}][a{i}]" for i in range(len(segments)))
    base = work / "base.mp4"
    ffmpeg(
        *inputs, "-filter_complex", ";".join(chains) + f";{labels}concat=n={len(segments)}:v=1:a=1[v][a]",
        "-map", "[v]", "-map", "[a]", "-r", str(FPS), "-frames:v", str(total), "-t", str(total / FPS),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "aac", "-b:a", "192k", str(base),
    )  # fmt: skip
    return base


class Overlay:
    """Draws the band under the feed: player cards, controllers, and the move timeline."""

    def __init__(self, args: Args, games: Sequence[Game], segments: Sequence[Segment]) -> None:
        self.games = games
        self.segments = segments
        self.cuts = replay_cuts(segments)
        self.sans_path = args.blog / SANS_FONT
        self.mono_path = args.blog / MONO_FONT
        for path in (self.sans_path, self.mono_path, args.controller_svg):
            if not path.is_file():
                raise FileNotFoundError(path)
        self.shapes = controller_shapes(args.controller_svg.read_text())
        self.box = self.shell(0, 0).getbbox()
        if self.box is None:
            raise ValueError("controller SVG has no shell")
        bw, bh = self.box[2] - self.box[0], self.box[3] - self.box[1]
        self.controller_height = round(bh * CARD_WIDTH / bw)

    @cache  # noqa: B019 - one Overlay lives for the whole render.
    def sans(self, size: int) -> ImageFont.FreeTypeFont:
        font = ImageFont.truetype(str(self.sans_path), size)
        font.set_variation_by_axes([600])
        return font

    @cache  # noqa: B019
    def mono(self, size: int) -> ImageFont.FreeTypeFont:
        return ImageFont.truetype(str(self.mono_path), size)

    @lru_cache(maxsize=1024)  # noqa: B019
    def shell(self, buttons: int, physical: int) -> Image.Image:
        """The controller body with pressed buttons filled; sticks are drawn per frame."""
        elements = []
        for key in self.shapes:
            if key in ("controlStick", "cStick"):
                continue
            if key == "shell":
                elements.append(svg_element(self.shapes[key], PAPER, INK, 2.4))
            elif key in BUTTON_BITS:
                pressed = (physical if key == "aButton" else buttons) & BUTTON_BITS[key]
                color = ACCENT if pressed else IDLE
                elements.append(svg_element(self.shapes[key], ACCENT if pressed else PAPER, color, 2.4))
            else:
                elements.append(svg_element(self.shapes[key], PAPER, IDLE, 2.4))
        return rasterize(elements)

    def controller(self, player: Player, index: int) -> Image.Image:
        inputs = player.inputs
        image = self.shell(int(inputs["buttons"][index]), int(inputs["physical"][index])).copy()
        # SlippiLab's stick geometry: gate centres and travel in SVG units.
        sticks = Image.new("RGBA", (SVG_PX * 4, SVG_PX * 4))
        draw = ImageDraw.Draw(sticks)
        scale = 4 * SVG_PX / SVG_SIZE
        for x, y, r in (
            (104.7 + inputs["jx"][index] * 34, 274.5 - inputs["jy"][index] * 34, 30),
            (395.2 + inputs["cx"][index] * 19.5, 411.6 - inputs["cy"][index] * 19.5, 21),
        ):
            draw.ellipse(((x - r) * scale, (y - r) * scale, (x + r) * scale, (y + r) * scale), fill=INK)
        image.alpha_composite(sticks.resize(image.size, Image.Resampling.LANCZOS))
        assert self.box is not None
        return image.crop(self.box).resize((CARD_WIDTH, self.controller_height), Image.Resampling.LANCZOS)

    def card_x(self, player: Player) -> int:
        return FEED[0] + round(FEED[2] * HUD[player.port]) - CARD_WIDTH // 2

    @lru_cache(maxsize=16)  # noqa: B019
    def panel(self, game: int) -> Image.Image:
        image = Image.new("RGBA", CANVAS)
        draw = ImageDraw.Draw(image)
        # No border: the post frames the embedded video itself.
        for player in self.games[game].players:
            x = self.card_x(player)
            draw.rectangle(
                (x - 18, BAND_Y - 18, x + CARD_WIDTH + 18, BAND_Y + 40 + self.controller_height + 24), fill=CARD
            )
            name = self.sans(24)
            label_x = x + name.getlength(player.label) + 14
            character = player.character.upper()
            if label_x + tracked_width(character, self.mono(14), 2.2) > x + CARD_WIDTH:
                raise ValueError(f"{player.label} {character} does not fit its card")
            draw.text((x, BAND_Y + 26), player.label, font=name, fill=INK, anchor="ls")
            tracked(draw, (label_x, BAND_Y + 26), character, self.mono(14), FAINT, 2.2)
        return image

    @lru_cache(maxsize=128)  # noqa: B019
    def timeline(self, cue: Cue, current: int) -> Image.Image:
        """One node per step on a vertical rule. Played steps are filled, the current step is in
        full ink, and steps still to come are grey, so the list follows the play."""
        font = self.sans(30)
        width = FEED[0] + FEED[2] - NOTE_X
        rows: list[tuple[int, list[str]]] = []
        for i, step in enumerate(cue.steps):
            rows.append((i, wrap(step[0].upper() + step[1:], font, width - 28)))
        height = sum(46 + 38 * (len(lines) - 1) for _, lines in rows) + 28
        image = Image.new("RGBA", (width, height))
        draw = ImageDraw.Draw(image)
        baselines = []
        baseline = 28
        for i, lines in rows:
            baselines.append(baseline)
            color = INK if i <= current else GHOST
            for j, line in enumerate(lines):
                draw.text((28, baseline + 38 * j), line, font=font, fill=color, anchor="ls")
            baseline += 46 + 38 * (len(lines) - 1)
        if len(baselines) > 1:
            draw.line((5, baselines[0] - 9, 5, baselines[-1] - 9), fill=DOTTED, width=2)
        for i, b in enumerate(baselines):
            color = INK if i <= current else GHOST
            draw.ellipse((0, b - 14, 10, b - 4), fill=color if i <= current else PAPER, outline=color, width=2)
        return image

    def frame(self, n: int) -> Image.Image:
        segment, _, replay_frame = locate(self.segments, n)
        index = segment.clip.game
        game = self.games[index]
        if replay_frame > game.last_frame:
            if replay_frame - game.last_frame >= END_SCREEN_FRAMES:
                raise ValueError(f"output frame {n} is {replay_frame - game.last_frame} frames past game {index}")
            replay_frame = game.last_frame
        image = self.panel(index).copy()
        for player in game.players:
            image.alpha_composite(
                self.controller(player, replay_frame - game.first_frame), (self.card_x(player), BAND_Y + 40)
            )
        for cue in CUES[index]:
            if cue.start <= replay_frame < cue.end:
                ease = min(1, (replay_frame - cue.start + 1) / 9, (cue.end - replay_frame) / 9)
                ease = ease * ease * (3 - 2 * ease)
                note = self.timeline(cue, current_step(cue, replay_frame)).copy()
                note.putalpha(note.getchannel("A").point([round(a * ease) for a in range(256)]))
                image.alpha_composite(note, (NOTE_X, round(BAND_Y - 4 + (1 - ease) * 8)))
                break
        dip = replay_dip(self.cuts, n)
        if dip:
            image.alpha_composite(Image.new("RGBA", FEED[2:], (255, 255, 255, round(255 * dip))), FEED[:2])
        return image


SVG_SIZE = 585.781
SVG_PX = 282
BUTTON_BITS = {
    "startButton": 4096,
    "aButton": 256,
    "bButton": 512,
    "xButton": 1024,
    "yButton": 2048,
    "lTrigger": 64,
    "rTrigger": 32,
    "zButton": 16,
}


def controller_shapes(source: str) -> dict[str, tuple[str, str]]:
    """Extract the controller's SVG elements from SlippiLab's JSX, keyed by element id."""
    source = re.sub(r"//[^\n]*", "", source)
    shapes = {}
    for tag, body in re.findall(r"<(path|circle)\s+(.*?)\s*/>", source, re.S):
        match = re.search(r'id="([^"]+)"', body)
        if match and not match[1].endswith("Cutout"):
            shapes[match[1]] = (tag, body)
    if "shell" not in shapes or not set(BUTTON_BITS) <= set(shapes):
        raise ValueError("controller SVG is missing the shell or a button")
    return shapes


def svg_element(shape: tuple[str, str], fill: str, stroke: str, width: float) -> str:
    tag, body = shape
    body = re.sub(r'\s(fill|stroke|stroke-width)=(?:\{[^}]*\}|"[^"]*")', "", body)
    # Static gates use JSX numeric literals; the moving sticks are drawn separately.
    body = re.sub(r"(c[xy]|r)=\{([0-9.]+)\}", r'\1="\2"', body)
    return f'<{tag} {body} fill="{fill}" stroke="{stroke}" stroke-width="{width}" stroke-linejoin="round"/>'


def rasterize(elements: Sequence[str]) -> Image.Image:
    # Imported here because cairosvg needs the system cairo library at import time.
    import cairosvg

    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SVG_SIZE} {SVG_SIZE}">{"".join(elements)}</svg>'
    png = cairosvg.svg2png(bytestring=svg.encode(), output_width=SVG_PX * 2, output_height=SVG_PX * 2)
    assert isinstance(png, bytes)
    return Image.open(io.BytesIO(png)).convert("RGBA").resize((SVG_PX, SVG_PX), Image.Resampling.LANCZOS)


def tracked(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: str,
    spacing: float,
) -> None:
    """Draw letter-spaced mono text from the left end of its baseline, like the post's labels."""
    x, y = xy
    for char in text:
        draw.text((x, y), char, font=font, fill=fill, anchor="ls")
        x += font.getlength(char) + spacing


def tracked_width(text: str, font: ImageFont.FreeTypeFont, spacing: float) -> float:
    return sum(font.getlength(char) for char in text) + spacing * (len(text) - 1)


def wrap(text: str, font: ImageFont.FreeTypeFont, width: float) -> list[str]:
    lines = [""]
    for word in text.split():
        candidate = f"{lines[-1]} {word}".strip()
        if lines[-1] and font.getlength(candidate) > width:
            lines.append(word)
        else:
            lines[-1] = candidate
    return lines


def source_frame(path: Path, t: float) -> Image.Image:
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-ss", str(t), "-i", str(path), "-frames:v", "1"]
    data = subprocess.run([*cmd, "-f", "image2pipe", "-c:v", "png", "-"], check=True, capture_output=True).stdout
    return Image.open(io.BytesIO(data)).convert("RGB")


def render(overlay: Overlay, base: Path, out: Path, total: int) -> None:
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-v", "warning", "-y", "-i", str(base),
        "-f", "rawvideo", "-pixel_format", "rgba", "-video_size", f"{CANVAS[0]}x{CANVAS[1]}", "-framerate", str(FPS), "-i", "pipe:0",
        "-filter_complex", "[0:v][1:v]overlay=0:0:format=auto,format=yuv420p[v]", "-map", "[v]", "-map", "0:a",
        "-frames:v", str(total), "-t", str(total / FPS), "-r", str(FPS),
        "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out),
    ]  # fmt: skip
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as process:
        assert process.stdin is not None
        for n in range(total):
            process.stdin.write(overlay.frame(n).tobytes())
            if n % 600 == 0:
                print(f"{n}/{total}", flush=True)
        process.stdin.close()
        if process.wait():
            raise RuntimeError("overlay encode failed")


def decoded_frames(path: Path) -> Iterator[Image.Image]:
    reader = imageio_ffmpeg.read_frames(str(path))
    meta = next(reader)
    assert isinstance(meta, dict)
    width, height = meta["size"]
    for data in reader:
        assert isinstance(data, bytes)
        yield Image.frombytes("RGB", (width, height), data)


def contact_sheet(path: Path, segments: Sequence[Segment], out: Path) -> int:
    """One frame two seconds into each segment, so every clip and replay is on the sheet."""
    shots = {s.output_frame + min(120, s.frames // 2): s for s in segments}
    cols, tw, th = 3, 480, 500
    sheet = Image.new("RGB", (cols * tw, th * ((len(shots) + cols - 1) // cols)), PAPER)
    draw = ImageDraw.Draw(sheet)
    k = count = 0
    for count, image in enumerate(decoded_frames(path), start=1):
        segment = shots.get(count - 1)
        if segment is None:
            continue
        image.thumbnail((tw, th - 26))
        x, y = k % cols * tw, k // cols * th
        sheet.paste(image, (x, y))
        label = f"{(count - 1) / FPS:.2f}s game {segment.clip.game}{' replay' if segment.clip.replay else ''}"
        draw.text((x + 8, y + th - 22), label, fill=INK)
        k += 1
    sheet.save(out, quality=88)
    return count


def publish(master: Path, segments: Sequence[Segment], output: Path) -> tuple[Path, Path]:
    """The post's WebM (VP9 + Opus at the bitrate of its earlier reel) and WebP poster."""
    webm = output / "highlights.webm"
    poster = output / "highlights_poster.webp"
    vp9 = ["-c:v", "libvpx-vp9", "-b:v", "2000k", "-maxrate", "3000k", "-bufsize", "4000k", "-row-mt", "1"]
    vp9 += ["-tile-columns", "2", "-deadline", "good", "-cpu-used", "2", "-pix_fmt", "yuv420p"]
    log = str(output / "work/vp9")
    ffmpeg("-i", str(master), *vp9, "-pass", "1", "-passlogfile", log, "-an", "-f", "null", "-")
    ffmpeg("-i", str(master), *vp9, "-pass", "2", "-passlogfile", log, "-c:a", "libopus", "-b:a", "128k", str(webm))
    game, replay_frame = POSTER
    n = min(
        (
            n
            for s in segments
            if s.clip.game == game and not s.clip.replay
            for n in range(s.output_frame, s.output_frame + s.frames)
        ),
        key=lambda n: abs(locate(segments, n)[2] - replay_frame),
    )
    ffmpeg("-ss", str(n / FPS), "-i", str(master), "-frames:v", "1", "-c:v", "libwebp", "-quality", "82", str(poster))
    return webm, poster


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(args: Args) -> None:
    if len(args.sources) <= max(clip.source for clip in CLIPS):
        raise ValueError(f"the edit uses {max(clip.source for clip in CLIPS) + 1} sources")
    for path in (args.inputs, *args.sources):
        if not path.is_file():
            raise FileNotFoundError(path)
    validate_cues(CUES)
    games = load_games(args.inputs)
    segments = plan(CLIPS)
    total = sum(segment.frames for segment in segments)
    overlay = Overlay(args, games, segments)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.preview:
        for game, target in ((0, 3150), (2, 2257), (4, 3196), (4, 3680)):
            n = min(
                (n for s in segments if s.clip.game == game for n in range(s.output_frame, s.output_frame + s.frames)),
                key=lambda n: abs(locate(segments, n)[2] - target),
            )
            segment, t, _ = locate(segments, n)
            image = Image.new("RGBA", CANVAS, PAPER)
            feed = source_frame(args.sources[segment.clip.source], t).crop(
                (SOURCE_CROP[0], SOURCE_CROP[1], SOURCE_CROP[0] + SOURCE_CROP[2], SOURCE_CROP[1] + SOURCE_CROP[3])
            )
            image.paste(feed.resize(FEED[2:], Image.Resampling.LANCZOS), FEED[:2])
            image.alpha_composite(overlay.frame(n))
            image.convert("RGB").save(args.output / f"preview-{game}-{target}.jpg", quality=90)
        return
    work = args.output / "work"
    work.mkdir(exist_ok=True)
    base = encode_base(segments, args.sources, work)
    master = args.output / "highlights.mp4"
    render(overlay, base, master, total)
    decoded = contact_sheet(master, segments, args.output / "contact-sheet.jpg")
    if decoded != total:
        raise RuntimeError(f"{master} decodes to {decoded} frames, expected {total}")
    webm, poster = publish(master, segments, args.output)
    git = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    manifest = {
        "git_sha": git,
        "fps": FPS,
        "canvas": CANVAS,
        "frames": total,
        "segments": [asdict(segment) for segment in segments],
        "inputs": {str(path): sha256(path) for path in (args.inputs, *args.sources, args.controller_svg)},
        "outputs": {path.name: sha256(path) for path in (master, webm, poster)},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{total} frames; webm {webm.stat().st_size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
