"""Read existing Ranked telemetry without contacting or controlling the player."""

import fcntl
import json
import logging
import math
import os
import threading
import time
from collections import deque
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import replace
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import cast


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("expected a JSON object")
    return cast(dict[str, object], value)


def _number(data: dict[str, object], key: str) -> float:
    value = data.get(key)
    if type(value) not in (int, float):
        raise ValueError(f"invalid number: {key}")
    number = float(cast(int | float, value))
    if not math.isfinite(number):
        raise ValueError(f"non-finite number: {key}")
    return number


def _integer(data: dict[str, object], key: str) -> int:
    value = data.get(key)
    if type(value) is not int:
        raise ValueError(f"invalid integer: {key}")
    return value


def _boolean(data: dict[str, object], key: str) -> bool:
    value = data.get(key)
    if type(value) is not bool:
        raise ValueError(f"invalid boolean: {key}")
    return value


@dataclass(frozen=True, slots=True)
class ObsSample:
    at: float
    fps: float
    render_ms: float
    render_skipped: int
    encoder_skipped: int
    network_skipped: int
    output_bytes: int
    duration_ms: float
    active: bool
    reconnecting: bool
    congestion: float


def read_obs(path: Path) -> ObsSample:
    # This is the existing OBS 30.2.3 GetStats/GetStreamStatus snapshot.
    data = _object(json.loads(path.read_text()))
    obs, stream = _object(data.get("obs")), _object(data.get("stream"))
    result = ObsSample(
        _number(data, "at"),
        _number(obs, "activeFps"),
        _number(obs, "averageFrameRenderTime"),
        _integer(obs, "renderSkippedFrames"),
        _integer(obs, "outputSkippedFrames"),
        _integer(stream, "outputSkippedFrames"),
        _integer(stream, "outputBytes"),
        _number(stream, "outputDuration"),
        _boolean(stream, "outputActive"),
        _boolean(stream, "outputReconnecting"),
        _number(stream, "outputCongestion"),
    )
    if min(result.at, result.fps, result.render_ms, result.duration_ms, *counters(result)) < 0:
        raise ValueError("negative OBS counter")
    if not 0 <= result.congestion <= 1:
        raise ValueError("invalid congestion")
    return result


def counters(sample: ObsSample) -> tuple[int, int, int, int]:
    return sample.render_skipped, sample.encoder_skipped, sample.network_skipped, sample.output_bytes


@dataclass(frozen=True, slots=True)
class GameProgress:
    stream_id: int
    generation: int
    frame: int
    received_at: float


def read_progress(path: Path) -> GameProgress | None:
    data = _object(json.loads(path.read_text()))
    if _integer(data, "schema_version") != 1:
        raise ValueError("unsupported meter schema")
    playing = _boolean(data, "playing")
    value = data.get("sample")
    if value is None:
        return None
    sample = _object(value)
    result = GameProgress(
        _integer(sample, "stream_id"),
        _integer(sample, "generation"),
        _integer(sample, "source_frame"),
        _number(sample, "received_at"),
    )
    if min(result.stream_id, result.generation, result.received_at) < 0:
        raise ValueError("invalid game identity or timestamp")
    return result if playing and result.frame >= 0 else None


@dataclass(frozen=True, slots=True)
class ProcessUsage:
    pid: int
    start_ticks: int
    name: str
    rss_kib: int
    anonymous_kib: int
    cpu_seconds: float
    threads: int


def read_processes(proc: Path = Path("/proc")) -> tuple[ProcessUsage, ...]:
    """Read only names and counters from the current PID namespace."""
    ticks = os.sysconf("SC_CLK_TCK")
    result: list[ProcessUsage] = []
    for path in proc.iterdir():
        if not path.name.isdecimal() or int(path.name) == os.getpid():
            continue
        try:
            stat = (path / "stat").read_text().rsplit(")", 1)[1].split()
            status = dict(line.split(":", 1) for line in (path / "status").read_text().splitlines())
        except FileNotFoundError, ProcessLookupError:
            continue
        if "VmRSS" not in status:  # Exited processes can remain as zombies until reaped.
            continue
        result.append(
            ProcessUsage(
                int(path.name),
                int(stat[19]),
                status["Name"].strip(),
                int(status["VmRSS"].split()[0]),
                int(status["RssAnon"].split()[0]),
                (int(stat[11]) + int(stat[12])) / ticks,
                int(status["Threads"]),
            )
        )
    return tuple(sorted(result, key=lambda item: item.pid))


@dataclass(frozen=True, slots=True)
class StreamHealth:
    schema_version: int
    at: float
    obs: ObsSample | None
    game: GameProgress | None
    game_fps_5s: float | None
    prediction_age_seconds: float | None
    render_skips: int | None
    encoder_skips: int | None
    network_skips: int | None
    bitrate_mbps: float | None
    obs_counter_reset: bool
    alerts: tuple[str, ...]
    processes: tuple[ProcessUsage, ...] = ()


class StreamMonitor:
    def __init__(self) -> None:
        self._obs: ObsSample | None = None
        self._game: GameProgress | None = None
        self._frames: deque[tuple[float, int]] = deque(maxlen=16)

    def observe(
        self,
        obs: ObsSample | None,
        game: GameProgress | None,
        *,
        now: float,
        wall_time: float,
        errors: tuple[str, ...] = (),
    ) -> StreamHealth:
        alerts = list(errors)
        render = encoder = network = None
        bitrate = None
        reset = False
        if obs is not None:
            if not 0 <= wall_time - obs.at <= 3:
                alerts.append("obs_stats_stale")
            if not obs.active:
                alerts.append("stream_inactive")
            if obs.reconnecting:
                alerts.append("stream_reconnecting")
            if obs.congestion > 0:
                alerts.append("network_congestion")
            if obs.fps < 58:
                alerts.append("obs_fps_low")
            old = self._obs
            if old is not None and obs.at > old.at:
                reset = obs.duration_ms < old.duration_ms or any(
                    new < previous for new, previous in zip(counters(obs), counters(old), strict=True)
                )
                if not reset:
                    render = obs.render_skipped - old.render_skipped
                    encoder = obs.encoder_skipped - old.encoder_skipped
                    network = obs.network_skipped - old.network_skipped
                    bitrate = (obs.output_bytes - old.output_bytes) * 8 / (obs.at - old.at) / 1_000_000
                    for count, alert in (
                        (render, "render_drops"),
                        (encoder, "encoder_drops"),
                        (network, "network_drops"),
                    ):
                        if count:
                            alerts.append(alert)
            self._obs = obs
        else:
            self._obs = None

        fps = age = None
        previous = self._game
        if game is None or game.received_at > now:
            self._frames.clear()
            if game is not None:
                alerts.append("game_clock_invalid")
        else:
            age = now - game.received_at
            if age > 2:
                alerts.append("game_progress_stale")
            if (
                previous is None
                or (game.stream_id, game.generation) != (previous.stream_id, previous.generation)
                or game.frame < previous.frame
            ):
                self._frames.clear()
            # Poll time includes stalls; prediction time would hide a stopped producer.
            self._frames.append((now, game.frame))
            while len(self._frames) > 2 and now - self._frames[1][0] >= 5:
                self._frames.popleft()
            first_time, first_frame = self._frames[0]
            elapsed = now - first_time
            if elapsed >= 5:
                fps = (game.frame - first_frame) / elapsed
                if fps < 55:
                    alerts.append("game_fps_low")
        self._game = game
        return StreamHealth(1, wall_time, obs, game, fps, age, render, encoder, network, bitrate, reset, tuple(alerts))


def poll(run_dir: Path, monitor: StreamMonitor, *, processes: bool = False) -> StreamHealth:
    errors: list[str] = []
    obs = None
    game = None
    try:
        obs = read_obs(run_dir / "obs-stats.json")
    except (OSError, ValueError) as error:
        errors.append(f"obs_stats_unavailable:{type(error).__name__}")
    try:
        game = read_progress(run_dir / "value.json")
    except (OSError, ValueError) as error:
        errors.append(f"game_progress_unavailable:{type(error).__name__}")
    usage: tuple[ProcessUsage, ...] = ()
    if processes:
        try:
            usage = read_processes()
        except (OSError, ValueError, KeyError, IndexError) as error:
            errors.append(f"process_stats_unavailable:{type(error).__name__}")
    health = monitor.observe(obs, game, now=time.monotonic(), wall_time=time.time(), errors=tuple(errors))
    return replace(health, processes=usage)


def run_monitor(run_dir: Path, output: Path, stop: threading.Event, *, processes: bool = False) -> None:
    if not run_dir.is_dir():
        raise ValueError("run directory does not exist")
    output.mkdir(parents=True, exist_ok=True)
    # Duplicate writers must not race on rotation.
    with (output / "monitor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monitor = StreamMonitor()
        history = RotatingFileHandler(output / "history.jsonl", maxBytes=8 * 1024 * 1024, backupCount=2)
        history.setFormatter(logging.Formatter("%(message)s"))
        last_alerts: tuple[str, ...] | None = None
        try:
            while not stop.is_set():
                health = poll(run_dir, monitor, processes=processes)
                payload = json.dumps(asdict(health), allow_nan=False)
                temporary = output / "status.partial"
                temporary.write_text(payload + "\n")
                temporary.replace(output / "status.json")
                history.handle(logging.LogRecord(__name__, logging.INFO, __file__, 0, payload, (), None))
                if health.alerts != last_alerts:
                    print(json.dumps({"at": health.at, "alerts": health.alerts}), flush=True)
                    last_alerts = health.alerts
                stop.wait(1)
        finally:
            history.close()
