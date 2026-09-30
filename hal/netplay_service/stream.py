"""Display, audio, overlay, and OBS lifecycle for the leased stream slot."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from loguru import logger
from websockets.exceptions import WebSocketException

from hal.netplay_service import obs
from hal.netplay_service.domain import CHARACTERS
from hal.netplay_service.domain import IMITATIONS
from hal.netplay_service.health import SLOT_HEARTBEAT_MAX_AGE_SECONDS
from hal.netplay_service.health import SlotState
from hal.netplay_service.health import read_slot_status
from hal.netplay_service.queue_client import StreamGrant
from hal.netplay_service.queue_contract import QueueError

_STATE_VERSION: Final[int] = 1
_CHARACTER_LABELS: Final[dict[str, str]] = {choice.value: choice.label for choice in CHARACTERS}
_IMITATION_LABELS: Final[dict[str, str]] = {choice.value: choice.label for choice in IMITATIONS}


@dataclass(frozen=True, slots=True)
class IdleStreamState:
    state: str = "idle"
    schema_version: int = _STATE_VERSION


@dataclass(frozen=True, slots=True)
class GameStreamState:
    character: str
    imitation: str
    desired_return: float | None
    game_number: int
    state: str = "game"
    schema_version: int = _STATE_VERSION

    def __post_init__(self) -> None:
        if self.character not in _CHARACTER_LABELS:
            raise ValueError(f"unsupported stream character {self.character!r}")
        if self.imitation not in _IMITATION_LABELS:
            raise ValueError(f"unsupported stream imitation {self.imitation!r}")
        if self.game_number < 1:
            raise ValueError("stream game number must be positive")


StreamState = IdleStreamState | GameStreamState


def write_stream_state(path: Path, state: StreamState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(asdict(state), allow_nan=False, separators=(",", ":"), sort_keys=True))
    temporary.replace(path)


def read_stream_state(path: Path) -> StreamState:
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read stream state {path}") from error
    if not isinstance(raw, dict) or raw.get("schema_version") != _STATE_VERSION:
        raise ValueError("stream state has an unsupported schema")
    if raw.get("state") == "idle" and set(raw) == {"schema_version", "state"}:
        return IdleStreamState()
    expected = {"schema_version", "state", "character", "imitation", "desired_return", "game_number"}
    if raw.get("state") != "game" or set(raw) != expected:
        raise ValueError("stream state fields changed")
    character = raw["character"]
    imitation = raw["imitation"]
    desired_return = raw["desired_return"]
    game_number = raw["game_number"]
    if not isinstance(character, str) or not isinstance(imitation, str):
        raise ValueError("stream game labels must be strings")
    if desired_return is not None and (
        isinstance(desired_return, bool) or not isinstance(desired_return, int | float)
    ):
        raise ValueError("stream desired return must be numeric or null")
    if isinstance(game_number, bool) or not isinstance(game_number, int):
        raise ValueError("stream game number must be an integer")
    return GameStreamState(
        character, imitation, None if desired_return is None else float(desired_return), game_number
    )


def overlay_text(state: StreamState, queue_depth: int) -> str:
    if queue_depth < 0:
        raise ValueError("queue depth must be non-negative")
    if isinstance(state, IdleStreamState):
        noun = "player" if queue_depth == 1 else "players"
        return f"Play HAL at 20xx.xyz · {queue_depth} {noun} waiting"
    character = _CHARACTER_LABELS[state.character]
    imitation = _IMITATION_LABELS[state.imitation]
    difficulty = "auto" if state.desired_return is None else f"{state.desired_return:g}"
    return (
        f"HAL · {imitation} {character} · difficulty {difficulty} · Game {state.game_number} of 5 · play at 20xx.xyz"
    )


def write_overlay(path: Path, text: str) -> None:
    if "\n" in text or "\r" in text:
        raise ValueError("stream overlay must be one line")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(text + "\n")
    temporary.replace(path)


def _stop_process(process: subprocess.Popen[bytes], timeout: float = 2.0) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout)


def _display_ready(number: int) -> bool:
    return Path(f"/tmp/.X11-unix/X{number}").exists()


class DisplayGroup:
    """Use a GPU display for the stream and isolated Xvfb for other slots."""

    def __init__(
        self,
        slots: int,
        base: int,
        stream_capable: bool,
        *,
        stream_display: str | None = None,
        popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
        run: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
        ready: Callable[[int], bool] = _display_ready,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if slots < 1 or base < 0 or base + slots > 65_535:
            raise ValueError("Xvfb display range is invalid")
        if stream_capable and not stream_display:
            raise ValueError("streaming requires a dedicated NVIDIA Xorg display")
        if stream_display in {f":{base + slot}" for slot in range(slots)}:
            raise ValueError("stream display overlaps the managed Xvfb range")
        self.displays = tuple(
            stream_display if slot == 0 and stream_capable and stream_display else f":{base + slot}"
            for slot in range(slots)
        )
        self._numbers = tuple(base + slot for slot in range(slots))
        self._stream_capable = stream_capable
        self._popen = popen
        self._run = run
        self._ready = ready
        self._sleep = sleep
        self._processes: list[subprocess.Popen[bytes]] = []
        self._window_config: tempfile.TemporaryDirectory[str] | None = None

    def __enter__(self) -> DisplayGroup:
        try:
            for slot, number in enumerate(self._numbers):
                if slot == 0 and self._stream_capable:
                    environment = os.environ | {"DISPLAY": self.displays[slot]}
                    result = self._run(["glxinfo", "-B"], env=environment, check=True, capture_output=True)
                    if b"OpenGL vendor string: NVIDIA Corporation" not in result.stdout:
                        raise RuntimeError("stream display must use NVIDIA hardware rendering")
                    self._window_config = tempfile.TemporaryDirectory(prefix="hal-openbox-")
                    window_config = Path(self._window_config.name) / "rc.xml"
                    window_config.write_text(
                        '<openbox_config xmlns="http://openbox.org/3.4/rc"><applications>'
                        '<application class="Apprun" title="Dolphin">'
                        "<decor>no</decor><fullscreen>yes</fullscreen>"
                        "</application></applications></openbox_config>"
                    )
                    self._processes.append(
                        self._popen(
                            ["openbox", "--sm-disable", "--config-file", str(window_config)],
                            env=environment,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                    )
                    continue
                process = self._popen(
                    ["Xvfb", f":{number}", "-screen", "0", "640x480x24", "-nolisten", "tcp"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self._processes.append(process)
                deadline = time.monotonic() + 10.0
                while not self._ready(number):
                    if process.poll() is not None:
                        raise RuntimeError(f"Xvfb :{number} exited with {process.returncode}")
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Xvfb :{number} did not become ready")
                    self._sleep(0.05)
                self._run(
                    ["xsetroot", "-display", f":{number}", "-solid", "#10131a"],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_: object) -> None:
        for process in reversed(self._processes):
            _stop_process(process)
        self._processes.clear()
        if self._window_config is not None:
            self._window_config.cleanup()
            self._window_config = None


class PulseAudio:
    """Own the one null-sink PulseAudio daemon used by the stream slot."""

    def __init__(
        self,
        directory: Path,
        *,
        popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
        ready: Callable[[Path], bool] = Path.exists,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.directory = directory
        self.socket = directory / "native"
        self.environment = {"PULSE_SERVER": f"unix:{self.socket}", "PULSE_SINK": "hal_stream"}
        self._popen = popen
        self._ready = ready
        self._sleep = sleep
        self._process: subprocess.Popen[bytes] | None = None

    def command(self) -> list[str]:
        command = ["pulseaudio", "--daemonize=no", "--exit-idle-time=-1"]
        if os.geteuid() == 0:
            command.extend(("--system", "--disallow-exit", "--disable-shm"))
        command.extend(
            (
                f"--load=module-native-protocol-unix socket={self.socket} auth-anonymous=1",
                "--load=module-null-sink sink_name=hal_stream sink_properties=device.description=HAL_Stream",
            )
        )
        return command

    def __enter__(self) -> PulseAudio:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.directory.chmod(0o777)
        self.socket.unlink(missing_ok=True)
        self._process = self._popen(self.command(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + 10.0
        while not self._ready(self.socket):
            if self._process.poll() is not None:
                raise RuntimeError(f"PulseAudio exited with {self._process.returncode}")
            if time.monotonic() >= deadline:
                self.__exit__(None, None, None)
                raise TimeoutError("PulseAudio did not create its socket")
            self._sleep(0.05)
        return self

    def __exit__(self, *_: object) -> None:
        if self._process is not None:
            _stop_process(self._process)
            self._process = None
        self.socket.unlink(missing_ok=True)


class StreamSupervisor:
    """Run OBS while a stream grant exists and keep its overlay current."""

    def __init__(
        self,
        display: str,
        state_path: Path,
        overlay_path: Path,
        queue_depth: Callable[[], int],
        pulse_environment: dict[str, str],
        *,
        bandwidth_test: bool,
        slot_status_path: Path,
        obs_factory: Callable[[], obs.ObsStudio] | None = None,
    ) -> None:
        self._display = display
        self._state_path = state_path
        self._overlay_path = overlay_path
        self._queue_depth = queue_depth
        self._environment = os.environ | pulse_environment
        self._bandwidth_test = bandwidth_test
        self._slot_status_path = slot_status_path
        self._obs_factory = obs_factory or (lambda: obs.ObsStudio(display, overlay_path.parent, self._environment))
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._lock = threading.Lock()
        self._grant: StreamGrant | None = None
        self._thread = threading.Thread(target=self._run, name="hal-netplay-stream", daemon=True)

    def __enter__(self) -> StreamSupervisor:
        if not self._state_path.exists():
            write_stream_state(self._state_path, IdleStreamState())
        self._refresh_overlay(0)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread.is_alive():
            self._thread.join()

    def set_grant(self, grant: StreamGrant | None) -> None:
        with self._lock:
            if self._grant == grant:
                return
            self._grant = grant
        self._wake.set()

    def _current_grant(self) -> StreamGrant | None:
        with self._lock:
            return self._grant

    def _wait(self, seconds: float) -> None:
        self._wake.wait(seconds)
        self._wake.clear()

    def _read_queue_depth(self, previous_depth: int) -> int:
        depth = previous_depth
        with suppress(QueueError):
            depth = self._queue_depth()
        return depth

    def _refresh_overlay(self, depth: int) -> str:
        try:
            state = read_stream_state(self._state_path)
        except ValueError:
            state = IdleStreamState()
        playing = False
        try:
            status = read_slot_status(self._slot_status_path)
            playing = (
                isinstance(state, GameStreamState)
                and status.state in (SlotState.PLAYING, SlotState.DEGRADED)
                and 0 <= time.time() - status.updated_at <= SLOT_HEARTBEAT_MAX_AGE_SECONDS
            )
        except OSError, ValueError:
            pass
        text = overlay_text(state if playing else IdleStreamState(), depth)
        if not self._overlay_path.exists() or self._overlay_path.read_text() != text + "\n":
            write_overlay(self._overlay_path, text)
        return text

    def _run(self) -> None:
        studio: obs.ObsStudio | None = None
        running_key: str | None = None
        backoff = 1.0
        depth = 0
        next_depth = 0.0
        next_stats = 0.0
        healthy_since = time.monotonic()
        try:
            while not self._stop.is_set():
                grant = self._current_grant()
                if studio is not None and (grant is None or grant.key != running_key):
                    studio.close()
                    studio = None
                    running_key = None
                    backoff = 1.0
                if grant is None:
                    self._wait(0.25)
                    continue
                try:
                    if studio is None:
                        studio = self._obs_factory()
                        studio.start(grant.key, bandwidth_test=self._bandwidth_test)
                        running_key = grant.key
                        healthy_since = time.monotonic()
                    now = time.monotonic()
                    if now >= next_depth:
                        depth = self._read_queue_depth(depth)
                        next_depth = now + 5.0
                    studio.update(self._refresh_overlay(depth))
                    if now >= next_stats:
                        stats = studio.stats() | {"updated_at": time.time()}
                        stream_stats = stats.get("stream")
                        if (
                            now - healthy_since > 10
                            and isinstance(stream_stats, dict)
                            and not stream_stats.get("outputActive")
                            and not stream_stats.get("outputReconnecting")
                        ):
                            raise RuntimeError("OBS stopped sending video")
                        path = self._overlay_path.with_suffix(".obs.json")
                        temporary = path.with_suffix(".partial")
                        temporary.write_text(json.dumps(stats, allow_nan=False))
                        temporary.replace(path)
                        next_stats = now + 5.0
                    if now - healthy_since >= 60:
                        backoff = 1.0
                except (OSError, RuntimeError, ValueError, WebSocketException) as error:
                    logger.error("OBS stream failed: {}; restarting after {} seconds", error, backoff)
                    if studio is not None:
                        studio.close()
                    studio = None
                    self._wait(backoff)
                    backoff = min(backoff * 2, 30.0)
                    continue
                self._wait(0.25)
        finally:
            if studio is not None:
                studio.close()
