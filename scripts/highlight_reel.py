"""Render the ranked-ladder highlight reel for the Hillclimbing Melee post.

The reel cuts Twitch VOD captures of HAL's ranked games into one edit after a short title card.
A dark band under the game feed holds each player's rank, character, and controller directly
below that player's in-game percentage, plus the game, stage, and a caption for selected moments.
The title card climbs the opponents' ranks. ``first-cut``, the earlier edit, also replays its
decisive moments at half speed, where the caption's move names light up as they happen.

Inputs:
- ``--edit``: which entry of ``EDITS`` to render. An edit names the Slippi replays that its
  controllers and captions follow, and its footage: clips recorded from those replays by Slippi
  Playback Dolphin (``--playback``, with the disc at ``--iso``), or VOD captures.
- ``--controller-svg``: SlippiLab's ``Inputs.tsx``, which holds the GameCube controller SVG.
- ``--ranks``: Slippi's rank icon SVGs (from slippi.gg), named as in ``RANK_ICONS``.
- ``--blog``: the blog repository, for its fonts (Schibsted Grotesk and IBM Plex Mono).

Outputs in ``--output``: the MP4 master, the post's WebM and WebP poster, a contact sheet, and a
manifest with input and output hashes. ``--preview`` writes a storyboard instead.
"""

import hashlib
import io
import json
import os
import re
import signal
import struct
import subprocess
import tempfile
import time
from collections.abc import Iterator
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import replace
from functools import cache
from functools import lru_cache
from pathlib import Path
from typing import Literal

import imageio_ffmpeg
import numpy as np
import peppi_py
import tyro
from PIL import Image
from PIL import ImageDraw
from PIL import ImageEnhance
from PIL import ImageFilter
from PIL import ImageFont

FPS = 60
# Slippi frames run at 60000/1001 Hz; VOD timestamps are seconds.
REPLAY_HZ = 60000 / 1001
# The game image inside every source capture, and where it lands in the reel.
VOD_CROP = (304, 0, 1312, 1006)
FEED = (0, 0, 1208, 926)
# Recordings fill the feed: Dolphin's 4:3 picture scaled to the feed height, trimmed at the sides.
RECORDING_CROP = (0, 0, FEED[2], FEED[3])
# Dolphin stamps one frame dump per game frame at this rate.
DUMP_FPS = 60
# Tested with Slippi Playback Dolphin 3.5.2 (playback-3.5.2-Linux.zip, sha256 c7fcba0b6114eeab5aa20367d89a669daeebd639ddc43aa9969883b18c49cf1c).
# Its OpenGL backend renders under Xvfb, on Mesa's llvmpipe when no GPU driver is usable.
PLAYBACK_INI = """[Core]
CPUThread = True
DSPHLE = True
FastDiscSpeed = True
EmulationSpeed = 0
[DSP]
Backend = No audio output
DumpAudio = True
DumpAudioSilent = True
[Movie]
DumpFrames = True
DumpFramesSilent = True
[Display]
RenderToMain = True
[Interface]
ShowLaunchMessages = False
OnScreenDisplayMessages = False
"""
PLAYBACK_GFX_INI = """[Settings]
InternalResolution = 2
InternalResolutionFrameDumps = True
AspectRatio = 2
UseFFV1 = True
ShowFPS = False
"""
BAND = 316
CANVAS = (FEED[2], FEED[3] + BAND)

# A dark band, so the game image runs into it and pressed buttons read in their real colours.
# The accent is the post's teal (index-rail.css), lightened for a dark ground.
BG = "#0D1013"
INK = "#F2F4F5"
FAINT = "#8A939B"
GHOST = "#4A535B"
RULE = "#262C32"
TEAL = "#3FB8A6"
# GameCube button colours, and the in-game HUD colours for ports 1 and 2.
BUTTON_COLORS = {
    "aButton": "#1FB383",
    "bButton": "#E5484D",
    "xButton": "#B9C0C7",
    "yButton": "#B9C0C7",
    "zButton": "#7C5CD6",
    "startButton": "#B9C0C7",
    "lTrigger": "#B9C0C7",
    "rTrigger": "#B9C0C7",
}
STICK = "#D6DADE"
GATE = "#B9C0C7"
C_STICK = "#F2B705"
PORT_COLORS = ("#E5484D", "#4C7DF0")
# Each player's controller takes the colour of their costume. Black costumes use a mid grey so the
# controller stays visible on the dark band.
COSTUME_COLORS = {
    ("Fox", 0): "#C98A4B",
    ("Fox", 1): "#E5484D",
    ("Fox", 2): "#4C7DF0",
    ("Fox", 3): "#3FAE5A",
    ("Falco", 0): "#5B7FE0",
    ("Ganondorf", 1): "#C8423F",
    ("Sheik", 4): "#E6E8EA",
    ("Marth", 3): "#6E757C",
    ("Young Link", 4): "#6E757C",
}
# Slippi's external character and stage ids.
CHARACTERS = {2: "Fox", 9: "Marth", 19: "Sheik", 20: "Falco", 21: "Young Link", 25: "Ganondorf"}
STAGES = {2: "Fountain of Dreams", 8: "Yoshi's Story", 28: "Dream Land", 31: "Battlefield", 32: "Final Destination"}
SANS_FONT = "node_modules/@fontsource-variable/schibsted-grotesk/files/schibsted-grotesk-latin-wght-normal.woff2"
MONO_FONT = "node_modules/@fontsource/ibm-plex-mono/files/ibm-plex-mono-latin-400-normal.woff2"


@dataclass(frozen=True, slots=True)
class Recording:
    """Replay frames ``first`` through ``last`` rendered from the replay by Slippi Playback Dolphin."""

    replay: Path
    first: int
    last: int


@dataclass(frozen=True, slots=True)
class Clip:
    # A VOD capture, or a recording until main() renders it to a file.
    source: Path | Recording
    # Index into the edit's games.
    game: int
    start: float
    end: float
    speed: float
    replay: bool
    # Replay frame = round(source seconds * REPLAY_HZ + offset).
    offset: float
    # The game picture in the source, as (x, y, width, height).
    crop: tuple[int, int, int, int] = VOD_CROP


@dataclass(frozen=True, slots=True)
class Segment:
    clip: Clip
    frames: int
    output_frame: int


@dataclass(frozen=True, slots=True)
class Cue:
    start: int
    end: int
    text: str
    # Move names shown under the caption during a replay.
    steps: tuple[str, ...] = ()
    # Replay frame where each step after the first begins; the first runs from ``start``.
    step_frames: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class GameSpec:
    replay: Path
    # The opponent's Slippi rank. Replays do not record it.
    rank: str


@dataclass(frozen=True, slots=True)
class Edit:
    games: tuple[GameSpec, ...]
    clips: tuple[Clip, ...]
    # One tuple of cues per game, in replay order.
    cues: tuple[tuple[Cue, ...], ...]


@dataclass(frozen=True, slots=True)
class Player:
    port: int
    label: str
    character: str
    color: str
    inputs: dict[str, np.ndarray]


@dataclass(frozen=True, slots=True)
class Game:
    first_frame: int
    stage: str
    players: tuple[Player, ...]

    @property
    def last_frame(self) -> int:
        return self.first_frame + len(self.players[0].inputs["buttons"]) - 1


# Twitch VOD captures. The earlier edit crossfades between games, so its Sheik clip cannot run
# past 39.55 s and its Marth clip must end before about 52.7 s.
EARLIER_EDIT = Path.home() / "src/HAL-highlights.webm"
FALCO_VOD = Path("outputs/hal-highlights-v3/sources/v2890236934-normalized.mp4")
GANONDORF_VOD = Path("outputs/hal-highlights-v3/sources/v2890236932-normalized.mp4")
YOUNG_LINK_VOD = Path("outputs/hal-highlights-v2/sources/v2890236933.mp4")
ZOEY_VOD = Path("outputs/hal-highlights-v2/sources/v2890236935.mp4")

# Ganondorf's rank is the one HAL played against; the others are the slippi.gg tiers of each
# opponent's rating when the reel was made.
REPLAYS = Path("outputs/hal-highlights-v2/replays")
FALCO_1 = GameSpec(REPLAYS / "game-01.slp", "Diamond 1")
FALCO_2 = GameSpec(REPLAYS / "game-02.slp", "Diamond 1")
SHEIK = GameSpec(REPLAYS / "game-03.slp", "Diamond 2")
MARTH = GameSpec(REPLAYS / "game-04.slp", "Diamond 2")
GANONDORF = GameSpec(REPLAYS / "game-05.slp", "Grandmaster")
YOUNG_LINK = GameSpec(REPLAYS / "game-06.slp", "Grandmaster")
ZOEY_2 = GameSpec(REPLAYS / "game-07.slp", "Grandmaster")
CRUSH = GameSpec(Path("outputs/hal-highlights-v4/replays/crush-g1.slp"), "Grandmaster")

RANK_ICONS = {
    "Diamond 1": "Diamond_I.svg",
    "Diamond 2": "Diamond_II.svg",
    "Grandmaster": "Grand_Master.svg",
}
# The title climbs these, one step each, the way the reel's opponents do.
TITLE_RANKS = ("Diamond 1", "Diamond 2", "Grandmaster")
# Title frames at which each rank appears; the climb finishes before the poster frame.
TITLE_STEPS = (8, 22, 36)
TITLE_STEP_FRAMES = 14
# Each replay covers only the decisive exchange. In game 2 it cuts in as soon as the shine lands,
# then live play resumes where it stopped. Live clips run about a second past each stock so it
# can land. Cue frames come from the replays: action changes, hits, and stock losses.
FIRST_CUT = Edit(
    (FALCO_1, FALCO_2, SHEIK, MARTH, GANONDORF, YOUNG_LINK, ZOEY_2),
    (
        Clip(FALCO_VOD, 0, 54.4, 81.7, 1, False, -544),
        Clip(FALCO_VOD, 1, 105.4, 106.87, 1, False, -6281),
        Clip(FALCO_VOD, 1, 106.15, 106.87, 0.5, True, -6281),
        Clip(FALCO_VOD, 1, 106.87, 109.95, 1, False, -6281),
        Clip(EARLIER_EDIT, 2, 34.3, 39.55, 1, False, 18),
        Clip(EARLIER_EDIT, 3, 40.4, 52.5, 1, False, -2423),
        Clip(GANONDORF_VOD, 4, 58.7, 70.85, 1, False, -426),
        Clip(GANONDORF_VOD, 4, 67.65, 69.4, 0.5, True, -426),
        Clip(YOUNG_LINK_VOD, 5, 92, 102.6, 1, False, -397.7),
        Clip(YOUNG_LINK_VOD, 5, 99.9, 101.55, 0.5, True, -397.7),
        Clip(ZOEY_VOD, 6, 177.5, 192.3, 1, False, -9854),
    ),
    (
        (
            Cue(
                2770,
                2995,
                "Falco keeps approaching with jumping down-airs. HAL shields the first one and shines out of shield.",
            ),
            Cue(2995, 3150, "On the next one, HAL jumps first and meets Falco with a back-air."),
            Cue(3150, 3360, "Falco lands one more hit for the rest of the stock."),
            Cue(3505, 3660, "HAL backs off, respecting the ledge."),
        ),
        (
            Cue(
                40,
                150,
                "HAL corners Falco at the ledge and catches them off guard with a shine spike.",
                ("Back-air", "shine spike"),
                (105,),
            ),
            Cue(150, 250, "HAL follows Falco off stage with an Armada shine."),
        ),
        (
            Cue(
                2150,
                2389,
                "After trading blows, HAL gets to ledge with shine turnaround wavedash much faster than Sheik expected.",
            ),
        ),
        (Cue(110, 330, "Marth lands one hit this stock."),),
        (
            Cue(3120, 3270, "HAL ledgedashes back onto the stage and up-tilts under Ganondorf's back-air."),
            Cue(
                3600,
                3760,
                "Ganondorf is on the ledge at 101%. HAL waits at mid-stage, then catches the ledge jump with a running "
                "up-air.",
                ("Wait at mid-stage", "running up-air"),
                (3686,),
            ),
        ),
        (
            Cue(
                5400,
                5700,
                "HAL sharks with two up-airs on platform, then drops down and shine spikes Young Link's recovery.",
                ("Drop down", "shine spike"),
                (5638,),
            ),
        ),
        (
            Cue(960, 1090, "HAL is at 116% to Fox's 20%. It recovers with a side-B through Fox at the ledge."),
            Cue(1440, 1560, "HAL takes the ledge again before Fox's up-B reaches it."),
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class Moment:
    """Replay frames ``first`` through ``last`` of a game, recorded from its replay, with one caption."""

    game: GameSpec
    first: int
    last: int
    caption: str


def edit_from(moments: Sequence[Moment]) -> Edit:
    """An edit that plays the moments in order. Games are numbered by first appearance. Moments of
    one game that follow each other with the same caption share one cue, so it stays up across the cut."""
    games = list(dict.fromkeys(moment.game for moment in moments))
    clips = tuple(
        Clip(
            Recording(moment.game.replay, moment.first, moment.last),
            games.index(moment.game),
            0,
            (moment.last - moment.first) / REPLAY_HZ,
            1,
            False,
            moment.first,
            RECORDING_CROP,
        )
        for moment in moments
    )
    cues: list[list[Cue]] = [[] for _ in games]
    for previous, moment in zip((None, *moments), moments, strict=False):
        game_cues = cues[games.index(moment.game)]
        if previous is not None and previous.game == moment.game and previous.caption == moment.caption:
            game_cues[-1] = Cue(game_cues[-1].start, moment.last, moment.caption)
        else:
            game_cues.append(Cue(moment.first, moment.last, moment.caption))
    return Edit(tuple(games), clips, tuple(tuple(sorted(c, key=lambda cue: cue.start)) for c in cues))


# Each clip starts about a second before the first decisive action and ends about a second after the stock.
MARTH_COMBO = Moment(MARTH, 25, 700, "HAL juggles Marth on-stage mercilessly after baiting out their double jump.")
GANONDORF_LEDGE_JUMP = Moment(
    GANONDORF, 3420, 3821, "HAL finds kill confirmations and reversals in all sorts of scenarios."
)
FALCO_SHINE_SPIKE = Moment(FALCO_2, 37, 309, "HAL catches Falco off guard with a shine spike.")
SHEIK_LEDGE = Moment(
    SHEIK, 2074, 2388, "HAL edge cancels side-B, then shine turnaround grabs ledge before Sheik gets there."
)
YOUNG_LINK_SHINE_SPIKE = Moment(
    YOUNG_LINK,
    5280,
    5752,
    "HAL sharks with two up-airs on platform, then drops down and shine spikes Young Link's recovery.",
)
ZOEY_REVERSAL = Moment(
    ZOEY_2, 900, 1630, "At a disadvantage, HAL side-Bs past to recover, then steals ledge for the stock."
)
CRUSH_CAPTION = "HAL finds unexpected reversals, but sometimes makes crucial blunders in edge guarding."
CRUSH_TOO_DEEP = Moment(CRUSH, 6110, 6614, CRUSH_CAPTION)
CRUSH_LAST_STOCK = Moment(CRUSH, 8160, 8920, CRUSH_CAPTION)
EDITS = {
    "first-cut": FIRST_CUT,
    # Combos and recovery, then gimps and reversals, then the edge guards HAL drops.
    "reel": edit_from(
        (
            MARTH_COMBO,
            GANONDORF_LEDGE_JUMP,
            FALCO_SHINE_SPIKE,
            SHEIK_LEDGE,
            YOUNG_LINK_SHINE_SPIKE,
            ZOEY_REVERSAL,
            CRUSH_TOO_DEEP,
            CRUSH_LAST_STOCK,
        )
    ),
}
# A clip may run this far past its replay's last frame onto the game-end screen.
END_SCREEN_FRAMES = 90
# P1 and P2 percentage centres in the in-game HUD, as fractions of the feed width.
HUD = (0.189, 0.41)
CARD_WIDTH = 220
BAND_Y = FEED[1] + FEED[3]
CONTROLLER_Y = BAND_Y + 88
RULE_X = 640
NOTE_X = 668
NOTE_WIDTH = CANVAS[0] - NOTE_X - 36
CAPTION_SIZE = 28
CAPTION_LEADING = 37
# Each stick leaves a fading trail of its positions over this many frames.
TRAIL = 10
# Fade through white on the feed at each cut into or out of a replay, peaking on the cut.
DIP_OUT, DIP_IN, DIP_PEAK = 8, 12, 0.9
# The title holds over the blurred first frame, then the frame comes into focus.
TITLE_FRAMES, TITLE_FOCUS = 150, 30
POSTER_FRAME = 60


@dataclass(frozen=True, slots=True)
class Args:
    edit: Literal["reel", "first-cut"] = "reel"
    controller_svg: Path = Path.home() / "src/slippilab/src/components/panels/Inputs.tsx"
    ranks: Path = Path("outputs/hal-highlights-v2/ranks")
    # Slippi Playback Dolphin and the NTSC 1.02 disc it plays recordings on.
    playback: Path = Path.home() / "data/dolphin/playback/Slippi_Playback-x86_64.AppImage"
    iso: Path = Path.home() / "data/dolphin/ssbm.ciso"
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
    """Return the segment, source seconds, and replay frame shown at edit frame ``n``."""
    for segment in segments:
        if segment.output_frame <= n < segment.output_frame + segment.frames:
            clip = segment.clip
            t = clip.start + (n - segment.output_frame) / FPS * clip.speed
            return segment, t, round(t * REPLAY_HZ + clip.offset)
    raise ValueError(f"edit frame {n} is outside the edit")


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
    return smoothstep(level) * DIP_PEAK


def validate_cues(cues: Sequence[Sequence[Cue]]) -> None:
    for game in cues:
        for previous, cue in zip(game, game[1:], strict=False):
            if cue.start < previous.end:
                raise ValueError(f"cue at {cue.start} overlaps the cue before it")
        for cue in game:
            if cue.end <= cue.start:
                raise ValueError(f"cue at {cue.start} is empty")
            if cue.steps and len(cue.step_frames) != len(cue.steps) - 1:
                raise ValueError(f"{cue.steps} needs {len(cue.steps) - 1} step frames")
            if not cue.steps and cue.step_frames:
                raise ValueError(f"cue at {cue.start} has step frames but no steps")
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


def load_game(spec: GameSpec) -> Game:
    """Each player's confirmed controller inputs, character, and costume colour from one replay.

    The opponent is the player whose netplay name is not HAL; replays do not record ranks.
    """
    replay = peppi_py.read_slippi(str(spec.replay))
    if replay.frames is None:
        raise ValueError(f"{spec.replay} has no frames")
    ids = np.asarray(replay.frames.id)
    rows = confirmed_rows(ids)
    if [player.port.name for player in replay.start.players] != ["P1", "P2"]:
        raise ValueError(f"{spec.replay} must use ports 1 and 2")
    names = [player.netplay.name if player.netplay else None for player in replay.start.players]
    if names.count("HAL") != 1:
        raise ValueError(f"{spec.replay} must have exactly one player named HAL, not {names}")
    if spec.rank not in RANK_ICONS:
        raise ValueError(f"{spec.replay} has rank {spec.rank!r}, which has no icon")
    players: list[Player] = []
    for port, (start, data) in enumerate(zip(replay.start.players, replay.frames.ports, strict=True)):
        character = CHARACTERS[start.character]
        color = COSTUME_COLORS[(character, start.costume)]
        pre = data.leader.pre
        inputs = {
            "buttons": pre.buttons,
            "physical": pre.buttons_physical,
            "jx": pre.joystick.x,
            "jy": pre.joystick.y,
            "cx": pre.cstick.x,
            "cy": pre.cstick.y,
        }
        label = "HAL" if names[port] == "HAL" else spec.rank
        players.append(Player(port, label, character, color, {k: np.asarray(v)[rows] for k, v in inputs.items()}))
    return Game(int(ids.min()), STAGES[replay.start.stage], tuple(players))


def ffmpeg(*args: str) -> None:
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-y", *args], check=True)


def frame_count(path: Path) -> int:
    return imageio_ffmpeg.count_frames_and_secs(str(path))[0]


def feed_filter(crop: tuple[int, int, int, int]) -> str:
    x, y, w, h = FEED
    cx, cy, cw, ch = crop
    return (
        f"crop={cw}:{ch}:{cx}:{cy},scale={w}:{h}:flags=lanczos,pad={CANVAS[0]}:{CANVAS[1]}:{x}:{y}:color={BG},setsar=1"
    )


def without_names(replay: bytes) -> bytes:
    """The replay with its players' name tags, display names, and connect codes zeroed. Slippi
    Playback draws display names under each player's percentage."""
    data = bytearray(replay)
    raw = data.index(b"raw[$U#l") + 12
    if data[raw] != 0x35:
        raise ValueError("the replay does not start with an event payloads event")
    payloads = data[raw + 1] - 1
    sizes = {data[raw + 2 + i]: struct.unpack(">H", data[raw + 3 + i : raw + 5 + i])[0] for i in range(0, payloads, 3)}
    start = raw + 1 + data[raw + 1]
    if data[start] != 0x36:
        raise ValueError("the replay does not start with a game start event")
    # Game Start offsets from the Slippi replay spec: name tags, display names, connect codes.
    for offset, width in ((0x161, 16), (0x1A5, 31), (0x221, 10)):
        if offset + 4 * width > sizes[0x36] + 1:
            raise ValueError(f"the game start event is too short for the field at {offset:#x}")
        data[start + offset : start + offset + 4 * width] = bytes(4 * width)
    return bytes(data)


def dumped_frames(path: Path) -> int:
    """Frames in a dump that Dolphin may still be writing. Dolphin writes an empty packet for a
    repeated frame, which a decoder skips, so count timestamps rather than decoded frames."""
    if not path.is_file():
        return 0
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-i", str(path), "-map", "0:v", "-c", "copy"]
    result = subprocess.run([*cmd, "-f", "null", "-", "-progress", "pipe:1"], capture_output=True, text=True)
    times = re.findall(r"^out_time_us=(\d+)$", result.stdout, re.MULTILINE)
    return round(int(times[-1]) * DUMP_FPS / 1e6) if times else 0


def await_dump(process: subprocess.Popen[bytes], log: Path, dump: Path, frames: int) -> None:
    """Wait until Dolphin has dumped ``frames`` frames. Its playback log runs ahead of rendering, and
    after the game's last frame it keeps dumping the end screen."""
    count, changed = 0, time.monotonic()
    while count < frames:
        time.sleep(3)
        if process.poll() is not None:
            raise RuntimeError(f"Dolphin exited with {process.returncode}: {log.read_text(errors='replace')[-2000:]}")
        latest = dumped_frames(dump)
        if latest != count:
            count, changed = latest, time.monotonic()
        elif time.monotonic() - changed > 120:
            raise RuntimeError(f"Dolphin stopped dumping at {count} of {frames} frames")


def record(recording: Recording, args: Args, path: Path) -> None:
    """Render the recording at the feed size with its game audio."""
    frames = recording.last - recording.first + 1
    with tempfile.TemporaryDirectory(prefix="playback-") as tmp:
        root = Path(tmp)
        config = root / "User" / "Config"
        config.mkdir(parents=True)
        (config / "Dolphin.ini").write_text(PLAYBACK_INI)
        (config / "GFX.ini").write_text(PLAYBACK_GFX_INI)
        replay = root / "replay.slp"
        replay.write_bytes(without_names(recording.replay.read_bytes()))
        # Dump frame k shows replay frame startFrame + 1 + k.
        comm = {
            "mode": "normal",
            "replay": str(replay),
            "startFrame": recording.first - 1,
            "endFrame": recording.last + 2,
            "commandId": f"{recording.first}-{recording.last}",
            "isRealTimeMode": False,
            "outputOverlayFiles": False,
        }
        (root / "comm.json").write_text(json.dumps(comm))
        cmd = [
            "xvfb-run", "-a", "-s", "-screen 0 1920x1080x24", str(args.playback), "-u", str(root / "User"),
            "-i", str(root / "comm.json"), "-e", str(args.iso), "-b", "-v", "OGL", "--cout", "--hide-seekbar",
            "--output-directory", str(root), "-o", "dump",
        ]  # fmt: skip
        env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
        env["APPIMAGE_EXTRACT_AND_RUN"] = "1"
        # Dolphin keeps running after playback ends, so stop its whole process group once the frames are dumped.
        with (
            (root / "dolphin.log").open("wb") as log,
            subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True) as process,
        ):
            try:
                await_dump(process, root / "dolphin.log", root / "dump.avi", frames)
            finally:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait()
        width = round(FEED[3] * 4 / 3)
        video = (
            # The fps filter repeats the previous frame for each empty packet, keeping frames on their timestamps.
            f"[0:v]fps={DUMP_FPS},trim=end_frame={frames},setpts=N/(60000/1001)/TB,"
            f"scale={width}:{FEED[3]}:flags=lanczos,crop={FEED[2]}:{FEED[3]},setsar=1,format=yuv420p[v]"
        )
        audio = f"[1:a]atrim=duration={frames / REPLAY_HZ},aresample=48000[a]"
        ffmpeg(
            "-i", str(root / "dump.avi"), "-i", str(root / "dump.wav"), "-filter_complex", f"{video};{audio}",
            "-map", "[v]", "-map", "[a]", "-r", "60000/1001",
            "-c:v", "libx264", "-preset", "slow", "-crf", "12", "-c:a", "aac", "-b:a", "256k", str(path),
        )  # fmt: skip


def recorded(clip: Clip, args: Args) -> Clip:
    """The clip with its recording rendered to a file, which is kept and reused."""
    if not isinstance(clip.source, Recording):
        return clip
    recording = clip.source
    if not recording.replay.is_file():
        raise FileNotFoundError(recording.replay)
    path = args.output / "recordings" / f"{sha256(recording.replay)[:12]}-{recording.first}-{recording.last}.mp4"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"recording {recording.replay.name} frames {recording.first}-{recording.last}", flush=True)
        partial = path.with_suffix(".partial.mp4")
        record(recording, args, partial)
        partial.rename(path)
    return replace(clip, source=path)


def encode_segment(segment: Segment, path: Path) -> None:
    clip = segment.clip
    span = clip.end - clip.start
    duration = segment.frames / FPS
    # Replays keep the game audio, slowed and ducked, so the cut back to live play keeps its level.
    gain = ",volume=0.45" if clip.replay else ""
    video = f"[0:v]trim=duration={span},setpts=(PTS-STARTPTS)/{clip.speed},tpad=stop_mode=clone:stop_duration=0.1,fps={FPS},{feed_filter(clip.crop)}[v]"
    audio = (
        f"[0:a]atrim=duration={span},asetpts=PTS-STARTPTS,atempo={clip.speed}{gain},apad,atrim=duration={duration}[a]"
    )
    ffmpeg(
        "-ss", str(clip.start), "-i", str(clip.source), "-filter_complex", f"{video};{audio}",
        "-map", "[v]", "-map", "[a]", "-frames:v", str(segment.frames), "-t", str(duration),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", str(path),
    )  # fmt: skip
    if frame_count(path) != segment.frames:
        raise RuntimeError(f"{path} does not have {segment.frames} frames")


def encode_title(overlay: Overlay, backdrop: Image.Image, path: Path) -> None:
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-y",
        "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{CANVAS[0]}x{CANVAS[1]}", "-framerate", str(FPS), "-i", "pipe:0",
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
        "-map", "0:v", "-map", "1:a", "-frames:v", str(TITLE_FRAMES), "-t", str(TITLE_FRAMES / FPS),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", str(path),
    ]  # fmt: skip
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as process:
        assert process.stdin is not None
        for k in range(TITLE_FRAMES):
            process.stdin.write(overlay.title(backdrop, k).tobytes())
        process.stdin.close()
        if process.wait():
            raise RuntimeError("title encode failed")
    if frame_count(path) != TITLE_FRAMES:
        raise RuntimeError(f"{path} does not have {TITLE_FRAMES} frames")


def encode_base(title: Path, segments: Sequence[Segment], work: Path) -> Path:
    parts = [work / f"{index:02d}.mp4" for index in range(len(segments))]
    for segment, part in zip(segments, parts, strict=True):
        encode_segment(segment, part)
    durations = [TITLE_FRAMES / FPS, *(segment.frames / FPS for segment in segments)]
    files = [title, *parts]
    total = TITLE_FRAMES + sum(segment.frames for segment in segments)
    inputs = [arg for part in files for arg in ("-i", str(part))]
    chains = [
        f"[{i}:v]setpts=PTS-STARTPTS[v{i}];[{i}:a]atrim=duration={d},asetpts=PTS-STARTPTS[a{i}]"
        for i, d in enumerate(durations)
    ]
    labels = "".join(f"[v{i}][a{i}]" for i in range(len(files)))
    base = work / "base.mp4"
    ffmpeg(
        *inputs, "-filter_complex", ";".join(chains) + f";{labels}concat=n={len(files)}:v=1:a=1[v][a]",
        "-map", "[v]", "-map", "[a]", "-r", str(FPS), "-frames:v", str(total), "-t", str(total / FPS),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "aac", "-b:a", "192k", str(base),
    )  # fmt: skip
    return base


def smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def mix(a: str, b: str, t: float) -> str:
    """The colour ``t`` of the way from ``a`` to ``b``."""
    ca = [int(a[i : i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02X}" for x, y in zip(ca, cb, strict=True))


def faded(image: Image.Image, alpha: float) -> Image.Image:
    out = image.copy()
    out.putalpha(out.getchannel("A").point([round(a * alpha) for a in range(256)]))
    return out


class Overlay:
    """Draws the band under the feed (player labels, controllers, captions) and the title card."""

    def __init__(
        self, args: Args, games: Sequence[Game], segments: Sequence[Segment], cues: Sequence[Sequence[Cue]]
    ) -> None:
        self.games = games
        self.segments = segments
        self.cues = cues
        self.cuts = replay_cuts(segments)
        self.sans_path = args.blog / SANS_FONT
        self.mono_path = args.blog / MONO_FONT
        self.rank_paths = {rank: args.ranks / name for rank, name in RANK_ICONS.items()}
        for path in (self.sans_path, self.mono_path, args.controller_svg, *self.rank_paths.values()):
            if not path.is_file():
                raise FileNotFoundError(path)
        self.shapes = controller_shapes(args.controller_svg.read_text())
        self.box = self.shell(0, 0, PORT_COLORS[0]).getbbox()
        if self.box is None:
            raise ValueError("controller SVG has no shell")
        bw, bh = self.box[2] - self.box[0], self.box[3] - self.box[1]
        self.controller_height = round(bh * CARD_WIDTH / bw)
        if CONTROLLER_Y + self.controller_height > CANVAS[1] - 16:
            raise ValueError("controllers do not fit in the band")
        # Draw every label and caption once up front, so one that does not fit fails before a render.
        for game in range(len(games)):
            self.panel(game)
        for game_cues in cues:
            for cue in game_cues:
                self.caption(cue, len(cue.steps) - 1, True)

    @cache  # noqa: B019 - one Overlay lives for the whole render.
    def sans(self, size: int, weight: int = 600) -> ImageFont.FreeTypeFont:
        font = ImageFont.truetype(str(self.sans_path), size)
        font.set_variation_by_axes([weight])
        return font

    @cache  # noqa: B019
    def mono(self, size: int) -> ImageFont.FreeTypeFont:
        return ImageFont.truetype(str(self.mono_path), size)

    @cache  # noqa: B019
    def rank_icon(self, rank: str, height: int) -> Image.Image:
        import cairosvg

        png = cairosvg.svg2png(url=str(self.rank_paths[rank]), output_height=height * 4)
        assert isinstance(png, bytes)
        image = Image.open(io.BytesIO(png)).convert("RGBA")
        image = image.crop(image.getbbox())
        return image.resize((round(image.width * height / image.height), height), Image.Resampling.LANCZOS)

    @lru_cache(maxsize=2048)  # noqa: B019
    def shell(self, buttons: int, physical: int, tint: str) -> Image.Image:
        """The controller body in the player's colour, with pressed buttons filled in their own
        colour; sticks are drawn per frame."""
        body = mix(tint, BG, 0.78)
        outline = mix(tint, BG, 0.55)
        elements = []
        for key, shape in self.shapes.items():
            if key in ("controlStick", "cStick"):
                continue
            if key == "shell":
                elements.append(svg_element(shape, body, outline, 2.4))
            elif key in BUTTON_BITS:
                color = BUTTON_COLORS[key]
                pressed = (physical if key == "aButton" else buttons) & BUTTON_BITS[key]
                if pressed:
                    elements.append(svg_element(shape, color, color, 2.4))
                else:
                    elements.append(svg_element(shape, body, mix(color, BG, 0.55), 2.4))
            elif key == "cStickGate":
                elements.append(svg_element(shape, body, mix(C_STICK, BG, 0.6), 2.4))
            else:
                elements.append(svg_element(shape, body, mix(GATE, BG, 0.6), 2.4))
        return rasterize(elements)

    def controller(self, player: Player, index: int) -> Image.Image:
        inputs = player.inputs
        image = self.shell(int(inputs["buttons"][index]), int(inputs["physical"][index]), player.color).copy()
        body = mix(player.color, BG, 0.78)
        # SlippiLab's stick geometry: gate centres and travel in SVG units.
        sticks = Image.new("RGBA", (SVG_PX * 4, SVG_PX * 4))
        draw = ImageDraw.Draw(sticks)
        scale = 4 * SVG_PX / SVG_SIZE
        for back in range(TRAIL, -1, -1):
            j = max(0, index - back)
            for x, y, r, color in (
                (104.7 + inputs["jx"][j] * 34, 274.5 - inputs["jy"][j] * 34, 30.0, STICK),
                (395.2 + inputs["cx"][j] * 19.5, 411.6 - inputs["cy"][j] * 19.5, 21.0, C_STICK),
            ):
                if back:
                    r *= 0.45
                    color = mix(color, body, 0.4 + 0.6 * back / (TRAIL + 1))
                draw.ellipse(((x - r) * scale, (y - r) * scale, (x + r) * scale, (y + r) * scale), fill=color)
        image.alpha_composite(sticks.resize(image.size, Image.Resampling.LANCZOS))
        assert self.box is not None
        return image.crop(self.box).resize((CARD_WIDTH, self.controller_height), Image.Resampling.LANCZOS)

    def card_x(self, player: Player) -> int:
        return FEED[0] + round(FEED[2] * HUD[player.port]) - CARD_WIDTH // 2

    @lru_cache(maxsize=16)  # noqa: B019
    def panel(self, game: int) -> Image.Image:
        """The band for one game: each player's character and rank over their controller, and the
        game and stage over the caption."""
        image = Image.new("RGBA", CANVAS)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, BAND_Y, CANVAS[0], CANVAS[1]), fill=BG)
        name = self.sans(24)
        spans = []
        for player in self.games[game].players:
            x = self.card_x(player)
            tracked(draw, (x, BAND_Y + 40), player.character.upper(), self.mono(14), FAINT, 2.2)
            label_x = x
            if player.label == "HAL":
                draw.rectangle((x, BAND_Y + 77, x + name.getlength("HAL"), BAND_Y + 80), fill=TEAL)
            else:
                icon = self.rank_icon(player.label, 34)
                image.alpha_composite(icon, (x, BAND_Y + 46))
                label_x = x + icon.width + 10
            draw.text((label_x, BAND_Y + 70), player.label, font=name, fill=INK, anchor="ls")
            spans.append((x, label_x + name.getlength(player.label)))
        (_, left_end), (right_start, right_end) = sorted(spans)
        if left_end + 16 > right_start or right_end + 16 > RULE_X:
            raise ValueError(f"game {game} player labels overlap")
        draw.line((RULE_X, BAND_Y + 28, RULE_X, CANVAS[1] - 28), fill=RULE, width=2)
        kicker = f"GAME {game + 1}  ·  {self.games[game].stage.upper()}"
        tracked(draw, (NOTE_X, BAND_Y + 40), kicker, self.mono(15), TEAL, 2.2)
        return image

    @lru_cache(maxsize=128)  # noqa: B019
    def caption(self, cue: Cue, current: int, replay: bool) -> Image.Image:
        """The caption under the game line. In a replay, the move names follow it: played and
        current moves in full ink, moves still to come in grey."""
        image = Image.new("RGBA", (NOTE_WIDTH, BAND - 60))
        draw = ImageDraw.Draw(image)
        font = self.sans(CAPTION_SIZE, 450)
        baseline = 0
        for line in wrap(cue.text, font, NOTE_WIDTH):
            baseline += CAPTION_LEADING
            draw.text((0, baseline), line, font=font, fill=INK, anchor="ls")
        if replay and cue.steps:
            baseline += 40
            x = 0.0
            step_font = self.mono(15)
            for i, step in enumerate(cue.steps):
                text = step.upper()
                tracked(draw, (x, baseline), text, step_font, INK if i <= current else GHOST, 1.6)
                x += tracked_width(text, step_font, 1.6) + 14
                if i < len(cue.steps) - 1:
                    draw.line((x, baseline - 6, x + 22, baseline - 6), fill=FAINT, width=2)
                    draw.polygon(((x + 22, baseline - 11), (x + 30, baseline - 6), (x + 22, baseline - 1)), fill=FAINT)
                    x += 44
            if x > NOTE_WIDTH:
                raise ValueError(f"move names for {cue.steps} do not fit")
        if baseline + 8 > image.height:
            raise ValueError(f"caption at {cue.start} does not fit the band")
        return image

    @cache  # noqa: B019
    def title_text(self, k: int) -> Image.Image:
        """Title text at title frame ``k``. The rank icons climb a staircase from left to right,
        each step drawn in as the line from the previous one reaches it."""
        image = Image.new("RGBA", CANVAS)
        draw = ImageDraw.Draw(image)
        cx, cy = CANVAS[0] // 2, 440
        draw.text((cx, cy), "HAL", font=self.sans(120, 650), fill=INK, anchor="ms")
        draw.rectangle((cx - 60, cy + 26, cx + 60, cy + 30), fill=TEAL)
        draw.text((cx, cy + 96), "Fox vs. humans on Slippi ranked", font=self.sans(32, 450), fill=INK, anchor="ms")
        label_font = self.mono(15)
        pitch, rise, height = 230, 44, 52
        base = cy + 290
        centers = [(cx + (i - (len(TITLE_RANKS) - 1) / 2) * pitch, base - i * rise) for i in range(len(TITLE_RANKS))]
        for i, (rank, start) in enumerate(zip(TITLE_RANKS, TITLE_STEPS, strict=True)):
            shown = smoothstep((k - start) / TITLE_STEP_FRAMES)
            if shown <= 0:
                continue
            x, y = centers[i]
            if i:
                # The line runs from under the previous icon to under this one.
                (px, py), reach = centers[i - 1], shown
                x0, y0, x1, y1 = px + 48, py + 6, x - 48, y + 6
                draw.line((x0, y0, x0 + (x1 - x0) * reach, y0 + (y1 - y0) * reach), fill=TEAL, width=2)
            layer = Image.new("RGBA", CANVAS)
            icon = self.rank_icon(rank, height)
            layer.alpha_composite(icon, (round(x - icon.width / 2), round(y - height + (1 - shown) * 10)))
            label = rank.upper()
            spacing = 2.5
            tracked(
                ImageDraw.Draw(layer),
                (x - tracked_width(label, label_font, spacing) / 2, y + 34),
                label,
                label_font,
                INK if i == len(TITLE_RANKS) - 1 else FAINT,
                spacing,
            )
            image.alpha_composite(faded(layer, shown))
        return image

    def title(self, backdrop: Image.Image, k: int) -> Image.Image:
        """Title frame ``k``: the text over the blurred, darkened first frame of the edit, which
        comes into focus over the last ``TITLE_FOCUS`` frames as the text fades out."""
        focus = smoothstep((k - (TITLE_FRAMES - TITLE_FOCUS)) / TITLE_FOCUS)
        feed = backdrop.filter(ImageFilter.GaussianBlur(10 * (1 - focus))) if focus < 1 else backdrop
        feed = ImageEnhance.Brightness(feed).enhance(0.28 + 0.72 * focus)
        image = Image.new("RGBA", CANVAS, BG)
        image.paste(feed, FEED[:2])
        fade = Image.new("RGBA", (FEED[2], 220), BG)
        fade.putalpha(
            Image.linear_gradient("L").resize((FEED[2], 220)).point([round(a * (1 - focus)) for a in range(256)])
        )
        image.alpha_composite(fade, (FEED[0], FEED[1] + FEED[3] - 220))
        image.alpha_composite(faded(self.title_text(k), 1 - focus))
        return image.convert("RGB")

    def frame(self, n: int) -> Image.Image:
        """The overlay at reel frame ``n``; the title card is part of the base video."""
        if n < TITLE_FRAMES:
            return Image.new("RGBA", CANVAS)
        n -= TITLE_FRAMES
        segment, _, replay_frame = locate(self.segments, n)
        index = segment.clip.game
        game = self.games[index]
        if replay_frame > game.last_frame:
            if replay_frame - game.last_frame >= END_SCREEN_FRAMES:
                raise ValueError(f"edit frame {n} is {replay_frame - game.last_frame} frames past game {index}")
            replay_frame = game.last_frame
        image = self.panel(index).copy()
        for player in game.players:
            controller = self.controller(player, replay_frame - game.first_frame)
            image.alpha_composite(controller, (self.card_x(player), CONTROLLER_Y))
        for cue in self.cues[index]:
            if cue.start <= replay_frame < cue.end:
                ease = smoothstep(min((replay_frame - cue.start + 1) / 9, (cue.end - replay_frame) / 9))
                note = faded(self.caption(cue, current_step(cue, replay_frame), segment.clip.replay), ease)
                image.alpha_composite(note, (NOTE_X, round(BAND_Y + 48 + (1 - ease) * 6)))
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
    """Greedy word wrap that breaks only at plain spaces, so a no-break space holds."""
    lines = [""]
    for word in text.split(" "):
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


def feed_frame(clip: Clip, t: float) -> Image.Image:
    assert isinstance(clip.source, Path)
    cx, cy, cw, ch = clip.crop
    image = source_frame(clip.source, t).crop((cx, cy, cx + cw, cy + ch))
    return image.resize(FEED[2:], Image.Resampling.LANCZOS)


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
    """The title card and one frame two seconds into each segment, so every clip and replay is on
    the sheet."""
    shots = {POSTER_FRAME: "title"}
    for s in segments:
        replay = " replay" if s.clip.replay else ""
        shots[TITLE_FRAMES + s.output_frame + min(120, s.frames // 2)] = f"game {s.clip.game + 1}{replay}"
    cols, tw, th = 3, 480, 520
    sheet = Image.new("RGB", (cols * tw, th * ((len(shots) + cols - 1) // cols)), "white")
    draw = ImageDraw.Draw(sheet)
    k = count = 0
    for count, image in enumerate(decoded_frames(path), start=1):
        label = shots.get(count - 1)
        if label is None:
            continue
        image.thumbnail((tw, th - 26))
        x, y = k % cols * tw, k // cols * th
        sheet.paste(image, (x, y))
        draw.text((x + 8, y + th - 22), f"{(count - 1) / FPS:.2f}s {label}", fill="black")
        k += 1
    sheet.save(out, quality=88)
    return count


def publish(master: Path, output: Path) -> tuple[Path, Path]:
    """The post's WebM (VP9 + Opus at the bitrate of its earlier reel) and the title card as its
    WebP poster."""
    webm = output / "highlights.webm"
    poster = output / "highlights_poster.webp"
    vp9 = ["-c:v", "libvpx-vp9", "-b:v", "2000k", "-maxrate", "3000k", "-bufsize", "4000k", "-row-mt", "1"]
    vp9 += ["-tile-columns", "2", "-deadline", "good", "-cpu-used", "2", "-pix_fmt", "yuv420p"]
    log = str(output / "work/vp9")
    ffmpeg("-i", str(master), *vp9, "-pass", "1", "-passlogfile", log, "-an", "-f", "null", "-")
    ffmpeg("-i", str(master), *vp9, "-pass", "2", "-passlogfile", log, "-c:a", "libopus", "-b:a", "128k", str(webm))
    poster_at = str(POSTER_FRAME / FPS)
    ffmpeg("-ss", poster_at, "-i", str(master), "-frames:v", "1", "-c:v", "libwebp", "-quality", "82", str(poster))
    return webm, poster


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def game_clock(frame: int) -> str:
    """The in-game timer at a replay frame, counting down from 8:00 at frame 0."""
    seconds = int(480 - frame / 60)
    return f"{seconds // 60}:{seconds % 60:02d}"


def storyboard(overlay: Overlay, segments: Sequence[Segment], backdrop: Image.Image, out: Path) -> None:
    """The title card, then one row per clip: where it sits in the edit, its game and timer span,
    its captions, and four frames from it with the band drawn."""
    shots, thumb, label_width, pad = 4, (300, 308), 440, 16
    total = TITLE_FRAMES + sum(segment.frames for segment in segments)
    row_height = thumb[1] + pad
    sheet = Image.new("RGB", (label_width + shots * (thumb[0] + pad), 120 + (len(segments) + 1) * row_height), BG)
    draw = ImageDraw.Draw(sheet)
    small, body = overlay.mono(15), overlay.sans(19, 450)
    # The timeline: each clip's share of the edit.
    bar_width = sheet.width - 2 * pad
    x = pad + bar_width * TITLE_FRAMES / total
    draw.rectangle((pad, 40, x, 72), fill=GHOST)
    for index, segment in enumerate(segments):
        end = x + bar_width * segment.frames / total
        draw.rectangle((x + 1, 40, end - 1, 72), fill=TEAL if index % 2 else mix(TEAL, BG, 0.4))
        draw.text((x + 6, 64), str(index + 1), font=small, fill=INK, anchor="ls")
        x = end
    tracked(draw, (pad, 28), f"EDIT  {total / FPS:.1f} S  ·  {len(segments)} CLIPS", small, FAINT, 1.6)
    title = overlay.title(backdrop, POSTER_FRAME).resize(thumb, Image.Resampling.LANCZOS)
    sheet.paste(title, (label_width, 104))
    tracked(draw, (pad, 128), f"TITLE  0:00-{TITLE_FRAMES / FPS:.1f} S", small, FAINT, 1.6)
    for index, segment in enumerate(segments):
        y = 104 + (index + 1) * row_height
        clip = segment.clip
        game = overlay.games[clip.game]
        start = (TITLE_FRAMES + segment.output_frame) / FPS
        first, last = (
            locate(segments, segment.output_frame)[2],
            locate(segments, segment.output_frame + segment.frames - 1)[2],
        )
        opponent = next(player for player in game.players if player.label != "HAL")
        lines = [
            f"{index + 1}  ·  {start:.1f}-{start + segment.frames / FPS:.1f} S  ({segment.frames / FPS:.1f} S)",
            f"GAME {clip.game + 1}  ·  {opponent.label.upper()} {opponent.character.upper()}",
            f"{game.stage.upper()}  ·  TIMER {game_clock(first)}-{game_clock(last)}",
        ]
        if clip.replay:
            lines.append("HALF-SPEED REPLAY")
        for k, line in enumerate(lines):
            tracked(draw, (pad, y + 24 + 24 * k), line, small, TEAL if k == 0 else FAINT, 1.6)
        baseline = y + 24 + 24 * len(lines) + 12
        for cue in overlay.cues[clip.game]:
            if cue.start < last and first < cue.end:
                for line in wrap(cue.text, body, label_width - 2 * pad):
                    baseline += 26
                    draw.text((pad, baseline), line, font=body, fill=INK, anchor="ls")
                baseline += 10
        for k in range(shots):
            n = segment.output_frame + round((k + 0.5) * (segment.frames - 1) / shots)
            _, t, frame = locate(segments, n)
            image = Image.new("RGBA", CANVAS, BG)
            image.paste(feed_frame(clip, t), FEED[:2])
            image.alpha_composite(overlay.frame(TITLE_FRAMES + n))
            x = label_width + k * (thumb[0] + pad)
            sheet.paste(image.convert("RGB").resize(thumb, Image.Resampling.LANCZOS), (x, y))
            draw.text((x + 6, y + 20), f"{(TITLE_FRAMES + n) / FPS:.1f}s  {game_clock(frame)}", font=small, fill=INK)
    sheet.save(out, quality=88)


def main(args: Args) -> None:
    edit = EDITS[args.edit]
    replays = tuple(spec.replay for spec in edit.games)
    for path in (*replays, args.playback, args.iso):
        if not path.is_file():
            raise FileNotFoundError(path)
    clips = tuple(recorded(clip, args) for clip in edit.clips)
    sources: list[Path] = []
    for clip in clips:
        if not isinstance(clip.source, Path) or not clip.source.is_file():
            raise FileNotFoundError(clip.source)
        if clip.source not in sources:
            sources.append(clip.source)
    if len(edit.cues) != len(edit.games):
        raise ValueError(f"the edit has cues for {len(edit.cues)} games and {len(edit.games)} games")
    validate_cues(edit.cues)
    games = tuple(load_game(spec) for spec in edit.games)
    segments = plan(clips)
    total = TITLE_FRAMES + sum(segment.frames for segment in segments)
    overlay = Overlay(args, games, segments, edit.cues)
    args.output.mkdir(parents=True, exist_ok=True)
    first = segments[0].clip
    backdrop = feed_frame(first, first.start)
    if args.preview:
        storyboard(overlay, segments, backdrop, args.output / f"storyboard-{args.edit}.jpg")
        return
    work = args.output / "work"
    work.mkdir(exist_ok=True)
    title = work / "title.mp4"
    encode_title(overlay, backdrop, title)
    base = encode_base(title, segments, work)
    master = args.output / "highlights.mp4"
    render(overlay, base, master, total)
    decoded = contact_sheet(master, segments, args.output / "contact-sheet.jpg")
    if decoded != total:
        raise RuntimeError(f"{master} decodes to {decoded} frames, expected {total}")
    webm, poster = publish(master, args.output)
    git = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    rank_icons = sorted(args.ranks / name for name in RANK_ICONS.values())
    inputs = (*sources, *replays, args.playback, args.controller_svg, *rank_icons)
    manifest = {
        "git_sha": git,
        "edit": args.edit,
        "fps": FPS,
        "canvas": CANVAS,
        "frames": total,
        "title_frames": TITLE_FRAMES,
        "segments": [asdict(segment) for segment in segments],
        "inputs": {str(path): sha256(path) for path in inputs},
        "outputs": {path.name: sha256(path) for path in (master, webm, poster)},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(f"{total} frames; webm {webm.stat().st_size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
