"""Frame-based value smoothing and a separately restartable OBS overlay."""

import json
import math
import os
import subprocess
import sys
import threading
import time
from contextlib import suppress
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from websockets.exceptions import WebSocketException

from hal.inference.api import ActionPlan
from hal.netplay_service.obs import Json
from hal.netplay_service.obs import ObsConnection


@dataclass(frozen=True, slots=True)
class ValueSample:
    stream_id: int
    generation: int
    sequence: int
    source_frame: int
    state_value: float
    ema: float
    received_at: float


class ValueMeter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sample: ValueSample | None = None

    def clear(self) -> None:
        with self._lock:
            self._sample = None

    def observe(self, plan: ActionPlan, *, now: float | None = None) -> None:
        if type(plan.state_value) is not float or not math.isfinite(plan.state_value):
            raise ValueError("state value must be a finite float")
        received_at = time.monotonic() if now is None else now
        with self._lock:
            old = self._sample
            ema = plan.state_value
            if old is not None and old.stream_id == plan.stream_id:
                if plan.generation < old.generation:
                    return
                if plan.generation == old.generation:
                    if plan.sequence <= old.sequence or plan.source_frame <= old.source_frame:
                        return
                    alpha = 1 - 2 ** (-(plan.source_frame - old.source_frame) / 6)
                    ema = old.ema + alpha * (plan.state_value - old.ema)
            self._sample = ValueSample(
                plan.stream_id, plan.generation, plan.sequence, plan.source_frame, plan.state_value, ema, received_at
            )

    def snapshot(self) -> ValueSample | None:
        with self._lock:
            return self._sample


def write_value(path: Path, sample: ValueSample | None, *, playing: bool) -> None:
    temporary = path.with_suffix(".partial")
    temporary.write_text(
        json.dumps(
            {"schema_version": 1, "playing": playing, "sample": None if sample is None else asdict(sample)},
            allow_nan=False,
        )
    )
    temporary.replace(path)


def read_value(path: Path, *, now: float) -> float | None:
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError("unsupported meter schema")
    if type(data.get("playing")) is not bool:
        raise ValueError("invalid meter playing state")
    sample = data.get("sample")
    if sample is None:
        return None
    if not isinstance(sample, dict):
        raise ValueError("invalid meter sample")
    for key in ("stream_id", "generation", "sequence", "source_frame"):
        if type(sample.get(key)) is not int:
            raise ValueError("invalid meter identity")
    for key in ("state_value", "ema", "received_at"):
        if type(sample.get(key)) not in (int, float) or not math.isfinite(sample[key]):
            raise ValueError("invalid meter value")
    if not data["playing"] or sample["source_frame"] < 0 or not 0 <= now - sample["received_at"] <= 1:
        return None
    return float(sample["ema"])


def read_control(path: Path) -> str:
    data = json.loads(path.read_text())
    if (
        not isinstance(data, dict)
        or type(data.get("schema_version")) is not int
        or data["schema_version"] != 1
        or not isinstance(data.get("password"), str)
        or not data["password"]
    ):
        raise ValueError("invalid OBS control file")
    return data["password"]


class MeterDisplay:
    """Only these four sources belong to the meter process."""

    def __init__(self, connection: ObsConnection) -> None:
        self.connection = connection
        self.items: dict[str, int] = {}
        self._shown = False
        self._value: float | None = None

    def configure(self) -> None:
        sources = self.connection.request("GetSceneItemList", {"sceneName": "HAL"}).get("sceneItems")
        if not isinstance(sources, list):
            raise ValueError("invalid OBS scene items")
        existing: dict[str, int] = {}
        for item in sources:
            if isinstance(item, dict):
                name, identity = item.get("sourceName"), item.get("sceneItemId")
                if isinstance(name, str) and type(identity) is int:
                    existing[name] = identity
        specifications: tuple[tuple[str, str, dict[str, Json], int, int], ...] = (
            ("Value background", "color_source_v3", {"width": 36, "height": 360, "color": 0xFF303030}, 1780, 350),
            ("Value fill", "color_source_v3", {"width": 36, "height": 1, "color": 0xFF80D060}, 1780, 530),
            ("Value zero", "color_source_v3", {"width": 48, "height": 2, "color": 0xFFFFFFFF}, 1774, 529),
            (
                "Model value",
                "text_ft2_source_v2",
                {
                    "text": "Model value\n+0.0",
                    "font": {"face": "DejaVu Sans", "size": 26},
                    "color1": 0xFFFFFFFF,
                    "color2": 0xFFFFFFFF,
                    "outline": True,
                },
                1700,
                745,
            ),
        )
        for name, kind, settings, x, y in specifications:
            identity = existing.get(name)
            if identity is None:
                value = self.connection.request(
                    "CreateInput",
                    {
                        "sceneName": "HAL",
                        "inputName": name,
                        "inputKind": kind,
                        "inputSettings": settings,
                        "sceneItemEnabled": False,
                    },
                ).get("sceneItemId")
                if type(value) is not int:
                    raise ValueError("invalid OBS meter item")
                identity = value
            else:
                self.connection.request("SetInputSettings", {"inputName": name, "inputSettings": settings})
            self.items[name] = identity
            self.connection.request(
                "SetSceneItemTransform",
                {
                    "sceneName": "HAL",
                    "sceneItemId": identity,
                    "sceneItemTransform": {"positionX": x, "positionY": y},
                },
            )
        self._visibility(False)

    def _visibility(self, visible: bool) -> None:
        for identity in self.items.values():
            self.connection.request(
                "SetSceneItemEnabled",
                {
                    "sceneName": "HAL",
                    "sceneItemId": identity,
                    "sceneItemEnabled": visible,
                },
            )
        self._shown = visible

    def update(self, value: float | None) -> None:
        if value is not None and not math.isfinite(value):
            raise ValueError("meter value must be finite")
        if value is None:
            if self._shown:
                self._visibility(False)
            self._value = None
            return
        value = round(value, 1)
        if value == self._value and self._shown:
            return
        height = max(1, round(min(abs(value) / 120, 1) * 180))
        self.connection.request(
            "SetInputSettings",
            {
                "inputName": "Model value",
                "inputSettings": {
                    "text": f"Model value\n{value:+.1f}",
                },
            },
        )
        self.connection.request(
            "SetInputSettings",
            {
                "inputName": "Value fill",
                "inputSettings": {
                    "height": height,
                    "color": 0xFF80D060 if value >= 0 else 0xFF6060E0,
                },
            },
        )
        self.connection.request(
            "SetSceneItemTransform",
            {
                "sceneName": "HAL",
                "sceneItemId": self.items["Value fill"],
                "sceneItemTransform": {"positionX": 1780, "positionY": 530 - height if value >= 0 else 530},
            },
        )
        if not self._shown:
            self._visibility(True)
        self._value = value


class OverlayProcess:
    """Restarting this child reloads presentation code without touching play or OBS."""

    def __init__(self, output: Path) -> None:
        self.output = output
        self.process: subprocess.Popen[bytes] | None = None
        self._next_start = 0.0
        self._backoff = 1.0
        self._started = 0.0

    def ensure_running(self, now: float) -> None:
        if self.process is not None and self.process.poll() is None:
            if now - self._started >= 30:
                self._backoff = 1
            return
        if self.process is not None:
            self.process.wait()
            self.process = None
            self._next_start = now + self._backoff
            self._backoff = min(30, self._backoff * 2)
        if now < self._next_start:
            return
        try:
            with (self.output / "overlay.log").open("ab") as log:
                self.process = subprocess.Popen(
                    [sys.executable, "-m", "hal.scripts.ranked_overlay", "--run-dir", str(self.output)],
                    stdout=log,
                    stderr=log,
                )
            self._started = now
            (self.output / "overlay.pid").write_text(str(self.process.pid))

        except OSError as error:
            self.close()
            self._next_start = now + self._backoff
            self._backoff = min(30, self._backoff * 2)
            logger.warning("Value overlay could not start: {}", type(error).__name__)

    def close(self) -> None:
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
            self.process = None


def run_overlay(run_dir: Path, stop: threading.Event) -> None:
    control = run_dir / "obs-control.json"
    while not stop.is_set():
        connection = ObsConnection()
        display = MeterDisplay(connection)
        try:
            password = read_control(control)
            connection.open(password)
            display.configure()
            last_status = 0.0
            while not stop.wait(1 / 30):
                if read_control(control) != password:
                    break
                now = time.monotonic()
                value = read_value(run_dir / "value.json", now=now)
                display.update(value)
                if now - last_status >= 1:
                    path = run_dir / "overlay-status.json"
                    temporary = path.with_suffix(".partial")
                    temporary.write_text(
                        json.dumps({"schema_version": 1, "pid": os.getpid(), "at": time.time(), "value": value})
                    )
                    temporary.replace(path)
                    last_status = now
        except (OSError, ValueError, RuntimeError, WebSocketException) as error:
            logger.warning("Value overlay reconnecting: {}", type(error).__name__)
        finally:
            with suppress(OSError, ValueError, RuntimeError, WebSocketException):
                display.update(None)
            connection.close()
        stop.wait(1)
