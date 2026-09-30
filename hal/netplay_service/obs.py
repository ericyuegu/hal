"""OBS 30.2.3 window capture controlled through obs-websocket v5."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import subprocess
import tempfile
import time
from pathlib import Path
from typing import cast

from websockets.sync.client import ClientConnection
from websockets.sync.client import connect

type Json = None | bool | int | float | str | list[Json] | dict[str, Json]

VERSION = "30.2.3"
PACKAGE_VERSION = "30.2.3.1-3~bpo24.04.1"


class ObsNotReady(RuntimeError):
    """The control socket opened before OBS finished loading its scene."""


def _object(value: object) -> dict[str, Json]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("OBS response must contain an object")
    return cast(dict[str, Json], value)


def _string(value: Json) -> str:
    if not isinstance(value, str):
        raise ValueError("OBS response must contain a string")
    return value


def authentication(password: str, salt: str, challenge: str) -> str:
    secret = base64.b64encode(hashlib.sha256((password + salt).encode()).digest()).decode()
    return base64.b64encode(hashlib.sha256((secret + challenge).encode()).digest()).decode()


def write_configuration(directory: Path, password: str) -> None:
    """Pin the on-disk profile format to the tested OBS release."""
    root = directory / "obs-studio"
    profile = root / "basic/profiles/HAL"
    profile.mkdir(parents=True)
    (root / "global.ini").write_text(
        "[General]\nFirstRun=true\nPre19Defaults=false\nPre21Defaults=false\nPre23Defaults=false\nPre24.1Defaults=false\n"
        "[Basic]\nProfile=HAL\nProfileDir=HAL\nSceneCollection=HAL\nSceneCollectionFile=HAL\n"
        "[BasicWindow]\nRunWizard=false\nPreviewEnabled=false\n"
    )
    (profile / "basic.ini").write_text(
        "[General]\nName=HAL\n"
        "[Video]\nBaseCX=1920\nBaseCY=1080\nOutputCX=1920\nOutputCY=1080\n"
        "FPSType=0\nFPSCommon=60\nColorFormat=NV12\nColorSpace=709\nColorRange=Partial\n"
        "[Output]\nMode=Advanced\nReconnect=true\nRetryDelay=2\nMaxRetries=10000\n"
        "[AdvOut]\nEncoder=jim_nvenc\nAudioEncoder=ffmpeg_aac\nTrackIndex=1\n"
        "Track1Bitrate=160\nApplyServiceSettings=true\nRescale=false\n"
        "[Audio]\nSampleRate=48000\nChannelSetup=Stereo\n"
    )
    (profile / "streamEncoder.json").write_text(
        json.dumps(
            {
                "rate_control": "CBR",
                "bitrate": 6000,
                "keyint_sec": 2,
                "preset2": "p5",
                "tune": "hq",
                "multipass": "qres",
                "profile": "high",
                "bf": 2,
                "lookahead": False,
                "psycho_aq": True,
            }
        )
    )
    scenes = root / "basic/scenes"
    scenes.mkdir()
    (scenes / "HAL.json").write_text(
        json.dumps(
            {
                "name": "HAL",
                "current_scene": "HAL",
                "current_program_scene": "HAL",
                "scene_order": [{"name": "HAL"}],
                "sources": [{"name": "HAL", "id": "scene", "settings": {"items": []}}],
            }
        )
    )
    plugin = root / "plugin_config/obs-websocket"
    plugin.mkdir(parents=True)
    (plugin / "config.json").write_text(
        json.dumps(
            {
                "server_enabled": True,
                "server_port": 4455,
                "auth_required": True,
                "server_password": password,
                "first_load": False,
                "alerts_enabled": False,
            }
        )
    )


def dolphin_window(items: Json) -> str | None:
    """Accept only the Slippi render window, never the desktop or launcher."""
    if not isinstance(items, list):
        raise ValueError("OBS window list must be an array")
    matches: list[str] = []
    for item in items:
        value = _object(item).get("itemValue")
        if not isinstance(value, str):
            continue
        parts = value.split("\r\n")
        if len(parts) == 3 and parts[1] == "Dolphin" and parts[2] == "AppRun.wrapped":
            matches.append(value)
    if len(matches) > 1:
        raise ValueError("multiple Dolphin render windows on the stream display")
    return matches[0] if matches else None


class ObsStudio:
    """Own OBS, its private profile, and its authenticated control socket."""

    def __init__(self, display: str, directory: Path, environment: dict[str, str]) -> None:
        self._environment = environment | {
            "DISPLAY": display,
            "QT_QPA_PLATFORM": "xcb",
            "__EGL_VENDOR_LIBRARY_FILENAMES": "/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
        }
        self._directory = directory
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._socket: ClientConnection | None = None
        self._request_id = 0
        self._window: str | None = None
        self._visible = False
        self._text = ""
        self._capture_id: Json = None

    def request(self, method: str, data: dict[str, Json] | None = None) -> dict[str, Json]:
        if self._socket is None:
            raise RuntimeError("OBS is not connected")
        self._request_id += 1
        request_id = str(self._request_id)
        self._socket.send(
            json.dumps(
                {
                    "op": 6,
                    "d": {
                        "requestType": method,
                        "requestId": request_id,
                        "requestData": data or {},
                    },
                }
            )
        )
        response = _object(json.loads(self._socket.recv(timeout=3)))
        body = _object(response.get("d"))
        if response.get("op") != 7 or body.get("requestId") != request_id:
            raise ValueError("unexpected OBS response")
        status = _object(body.get("requestStatus"))
        if status.get("code") == 207:
            raise ObsNotReady("OBS is still loading")
        if status.get("result") is not True:
            # OBS error comments can repeat request data, including the stream key.
            raise RuntimeError(f"OBS {method} failed with code {status.get('code')}")
        return _object(body.get("responseData", {}))

    def start(self, key: str, *, bandwidth_test: bool) -> None:
        if not key:
            raise ValueError("Twitch key must be non-empty")
        version = subprocess.check_output(["obs", "--version"], env=self._environment, text=True).strip()
        if version != f"OBS Studio - {PACKAGE_VERSION}":
            raise RuntimeError(f"unsupported OBS release: {version}; expected {PACKAGE_VERSION}")
        self._directory.mkdir(parents=True, exist_ok=True)
        self._temporary = tempfile.TemporaryDirectory(prefix="obs-", dir=self._directory)
        directory = Path(self._temporary.name)
        password = secrets.token_urlsafe(32)
        try:
            write_configuration(directory, password)
            with (self._directory / "obs.log").open("wb") as log:
                os.chmod(log.name, 0o600)
                self._process = subprocess.Popen(
                    [
                        "obs",
                        "--profile",
                        "HAL",
                        "--collection",
                        "HAL",
                        "--multi",
                        "--disable-shutdown-check",
                        "--disable-missing-files-check",
                        "--minimize-to-tray",
                    ],
                    env=self._environment | {"XDG_CONFIG_HOME": str(directory)},
                    stdout=log,
                    stderr=log,
                )
            deadline = time.monotonic() + 15
            while True:
                if self._process.poll() is not None:
                    raise RuntimeError("OBS exited during startup; see obs.log")
                try:
                    self._socket = connect(
                        "ws://127.0.0.1:4455", open_timeout=1, close_timeout=1, proxy=None, max_size=8 * 1024 * 1024
                    )
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("OBS control socket did not open") from None
                    time.sleep(0.1)
            hello = _object(json.loads(self._socket.recv(timeout=3)))
            auth = _object(_object(hello.get("d")).get("authentication"))
            self._socket.send(
                json.dumps(
                    {
                        "op": 1,
                        "d": {
                            "rpcVersion": 1,
                            "eventSubscriptions": 0,
                            "authentication": authentication(
                                password, _string(auth["salt"]), _string(auth["challenge"])
                            ),
                        },
                    }
                )
            )
            if _object(json.loads(self._socket.recv(timeout=3))).get("op") != 2:
                raise RuntimeError("OBS authentication failed")
            while True:
                try:
                    version_info = self.request("GetVersion")
                    break
                except ObsNotReady:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("OBS did not finish loading its scene") from None
                    time.sleep(0.1)
            if version_info.get("obsVersion") != VERSION:
                raise RuntimeError("OBS control socket belongs to a different release")
            self._configure_scene()
            self.request(
                "SetStreamServiceSettings",
                {
                    "streamServiceType": "rtmp_custom",
                    "streamServiceSettings": {
                        "server": "rtmp://live.twitch.tv/app",
                        "key": key + ("?bandwidthtest=true" if bandwidth_test else ""),
                        "use_auth": False,
                    },
                },
            )
            self.request("StartStream")
        except BaseException:
            self.close()
            raise

    def _create_input(self, name: str, kind: str, settings: dict[str, Json], *, enabled: bool = True) -> Json:
        return self.request(
            "CreateInput",
            {
                "sceneName": "HAL",
                "inputName": name,
                "inputKind": kind,
                "inputSettings": settings,
                "sceneItemEnabled": enabled,
            },
        )["sceneItemId"]

    def _configure_scene(self) -> None:
        self._capture_id = self._create_input(
            "Dolphin",
            "xcomposite_input",
            {
                "capture_window": "",
                "show_cursor": False,
                "include_border": False,
            },
            enabled=False,
        )
        self.request(
            "SetSceneItemTransform",
            {
                "sceneName": "HAL",
                "sceneItemId": self._capture_id,
                "sceneItemTransform": {
                    "boundsType": "OBS_BOUNDS_SCALE_INNER",
                    "boundsWidth": 1920,
                    "boundsHeight": 1080,
                    "positionX": 0,
                    "positionY": 0,
                },
            },
        )
        overlay_id = self._create_input(
            "Overlay",
            "text_ft2_source_v2",
            {
                "text": "Play HAL at 20xx.xyz",
                "font": {"face": "DejaVu Sans", "size": 32},
                "color1": 0xFFFFFFFF,
                "color2": 0xFFFFFFFF,
                "outline": True,
            },
        )
        self.request(
            "SetSceneItemTransform",
            {"sceneName": "HAL", "sceneItemId": overlay_id, "sceneItemTransform": {"positionX": 36, "positionY": 36}},
        )
        self._create_input("Game audio", "pulse_output_capture", {"device_id": "hal_stream.monitor"})
        self.request("SetCurrentProgramScene", {"sceneName": "HAL"})

    def update(self, text: str) -> None:
        if self._process is None or self._process.poll() is not None:
            raise RuntimeError("OBS exited; see obs.log")
        if text != self._text:
            self.request("SetInputSettings", {"inputName": "Overlay", "inputSettings": {"text": text}})
            self._text = text
        window = dolphin_window(
            self.request(
                "GetInputPropertiesListPropertyItems",
                {
                    "inputName": "Dolphin",
                    "propertyName": "capture_window",
                },
            ).get("propertyItems")
        )
        if window is not None and (self._window is None or window.split("\r\n")[0] != self._window.split("\r\n")[0]):
            self.request("SetInputSettings", {"inputName": "Dolphin", "inputSettings": {"capture_window": window}})
            self._window = window
        visible = window is not None
        if visible != self._visible:
            self.request(
                "SetSceneItemEnabled",
                {"sceneName": "HAL", "sceneItemId": self._capture_id, "sceneItemEnabled": visible},
            )
            self._visible = visible

    def screenshot(self) -> bytes | None:
        """Capture only the selected Dolphin window, never an unset source."""
        if not self._visible:
            return None
        result = self.request(
            "GetSourceScreenshot",
            {"sourceName": "Dolphin", "imageFormat": "png", "imageWidth": 960, "imageHeight": 720},
        )
        data = _string(result.get("imageData"))
        prefix = "data:image/png;base64,"
        if not data.startswith(prefix):
            raise ValueError("OBS screenshot must be a PNG data URL")
        return base64.b64decode(data[len(prefix) :], validate=True)

    def stats(self) -> dict[str, Json]:
        return {"obs": self.request("GetStats"), "stream": self.request("GetStreamStatus")}

    def close(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=3)
        self._process = None
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None
