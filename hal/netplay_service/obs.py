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


def _rgb(value: str) -> int:
    """OBS stores colors as 0xAABBGGRR."""
    red, green, blue = (int(value[index : index + 2], 16) for index in (1, 3, 5))
    return 0xFF000000 | blue << 16 | green << 8 | red


# Palette and type of the hillclimbing-melee post on ericyuegu.com.
_SANS = "Schibsted Grotesk"
_MONO = "IBM Plex Mono"
_PAPER = _rgb("#FFFFFF")
_INK = _rgb("#0D1013")
_INK_SOFT = _rgb("#5C646C")
_RULE = _rgb("#E4E7EA")
_ASIDE = _rgb("#FAFBFB")
_ACCENT = _rgb("#0E6E63")
# Match the visible header top (43) and footer bottom (1068) with the pinned
# fonts and OBS release. Scale Melee's native image without stretching it.
_VIDEO_X, _VIDEO_Y, _VIDEO_W, _VIDEO_H = 36, 43, 1344, 1025
_SIDE_X = _VIDEO_X + _VIDEO_W + 48
_SIDE_W = 1920 - 36 - _SIDE_X
# Slippi 3.6.4 rounds its 73:60 viewport to 1312x1080 in our 1920x1080
# stream window. Remove only its pillarboxes before proportional scaling.
_DOLPHIN_PILLARBOX = 304


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


class ObsConnection:
    """One thread owns each authenticated OBS connection."""

    def __init__(self) -> None:
        self._socket: ClientConnection | None = None
        self._request_id = 0

    def open(self, password: str) -> None:
        self._socket = connect(
            "ws://127.0.0.1:4455", open_timeout=1, close_timeout=1, proxy=None, max_size=8 * 1024 * 1024
        )
        try:
            self.authenticate(password)
        except BaseException:
            self.close()
            raise

    def authenticate(self, password: str) -> None:
        if self._socket is None:
            raise RuntimeError("OBS is not connected")
        hello = _object(json.loads(self._socket.recv(timeout=3)))
        auth = _object(_object(hello.get("d")).get("authentication"))
        self._socket.send(
            json.dumps(
                {
                    "op": 1,
                    "d": {
                        "rpcVersion": 1,
                        "eventSubscriptions": 0,
                        "authentication": authentication(password, _string(auth["salt"]), _string(auth["challenge"])),
                    },
                }
            )
        )
        if _object(json.loads(self._socket.recv(timeout=3))).get("op") != 2:
            raise RuntimeError("OBS authentication failed")

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

    def screenshot(self) -> bytes | None:
        """Capture only the selected Dolphin window, never an unset source."""
        settings = _object(self.request("GetInputSettings", {"inputName": "Dolphin"}).get("inputSettings"))
        if not settings.get("capture_window"):
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

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None


class ObsStudio(ObsConnection):
    """Own OBS, its private profile, and its authenticated control socket."""

    def __init__(self, display: str, directory: Path, environment: dict[str, str]) -> None:
        super().__init__()
        self._environment = environment | {
            "DISPLAY": display,
            "QT_QPA_PLATFORM": "xcb",
            "__EGL_VENDOR_LIBRARY_FILENAMES": "/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
        }
        self._directory = directory
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._password: str | None = None
        self._window: str | None = None
        self._visible = False
        self._text = ""
        self._capture_id: Json = None

    def start(self, key: str, *, bandwidth_test: bool, ranked: bool = False) -> None:
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
            self.authenticate(password)
            self._password = password
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
            self._configure_scene(ranked=ranked)
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

    def _configure_scene(self, *, ranked: bool = False) -> None:
        if not ranked:
            self._color_panel("Background", _PAPER, 1920, 1080, 0, 0)
            self._color_panel("Video frame", _RULE, _VIDEO_W + 2, _VIDEO_H + 2, _VIDEO_X - 1, _VIDEO_Y - 1)
            self._color_panel("Video well", _ASIDE, _VIDEO_W, _VIDEO_H, _VIDEO_X, _VIDEO_Y)
            center_x = _VIDEO_X + _VIDEO_W // 2
            center_y = _VIDEO_Y + _VIDEO_H // 2
            self._text_panel(
                "Video idle",
                "Waiting for a challenger",
                center_x,
                center_y - 24,
                44,
                _INK,
                style="SemiBold",
                centered=True,
            )
            self._text_panel(
                "Video idle hint",
                "Queue up at 20xx.xyz",
                center_x,
                center_y + 40,
                22,
                _INK_SOFT,
                face=_MONO,
                centered=True,
            )

            x = _SIDE_X
            self._color_panel("Brand mark", _ACCENT, 12, 12, x, 46)
            self._text_panel(
                "Brand", "HAL / SUPER SMASH BROS. MELEE", x + 26, 40, 17, _INK_SOFT, face=_MONO, style="Medium"
            )
            self._text_panel("Title", "Play HAL", x - 4, 90, 84, _INK, style="Bold")
            self._text_panel(
                "Description", "A neural network that learned\nMelee from human Slippi replays.", x, 206, 24, _INK_SOFT
            )

            # Same card as the post's release links: hairline frame, aside fill, accent top rule.
            card_y, card_h = 326, 272
            self._color_panel("Status frame", _RULE, _SIDE_W, card_h, x, card_y)
            self._color_panel("Status card", _ASIDE, _SIDE_W - 2, card_h - 2, x + 1, card_y + 1)
            self._color_panel("Status rule", _ACCENT, _SIDE_W, 3, x, card_y)
            self._color_panel("Live dot", _ACCENT, 10, 10, x + 32, card_y + 41)
            self._text_panel("Status label", "LIVE", x + 52, card_y + 34, 17, _ACCENT, face=_MONO, style="Medium")
            self._text_panel(
                "Overlay",
                "Waiting for a challenger",
                x + 32,
                card_y + 78,
                34,
                _INK,
                style="SemiBold",
                width=_SIDE_W - 60,
            )
            self._text_panel("Overlay detail", "", x + 32, card_y + 140, 24, _INK_SOFT, width=_SIDE_W - 60)

            panel = self._create_input("Live panel", "image_source", {"file": ""})
            self.request(
                "SetSceneItemTransform",
                {
                    "sceneName": "HAL",
                    "sceneItemId": panel,
                    "sceneItemTransform": {"positionX": x, "positionY": 326},
                },
            )
            self._color_panel("Join rule", _RULE, _SIDE_W, 1, x, 950)
            self._text_panel("Join label", "PLAY HAL YOURSELF", x, 974, 17, _ACCENT, face=_MONO, style="Medium")
            self._text_panel("Join", "20xx.xyz", x - 3, 992, 64, _INK, style="Bold")
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
                    "cropLeft": 0 if ranked else _DOLPHIN_PILLARBOX,
                    "cropRight": 0 if ranked else _DOLPHIN_PILLARBOX,
                    "boundsType": "OBS_BOUNDS_SCALE_INNER",
                    "boundsWidth": 1920 if ranked else _VIDEO_W,
                    "boundsHeight": 1080 if ranked else _VIDEO_H,
                    "positionX": 0 if ranked else _VIDEO_X,
                    "positionY": 0 if ranked else _VIDEO_Y,
                },
            },
        )
        self._create_input("Game audio", "pulse_output_capture", {"device_id": "hal_stream.monitor"})
        self.request("SetCurrentProgramScene", {"sceneName": "HAL"})

    def _color_panel(self, name: str, color: int, width: int, height: int, x: int, y: int) -> None:
        item = self._create_input(name, "color_source_v3", {"color": color, "width": width, "height": height})
        self.request(
            "SetSceneItemTransform",
            {"sceneName": "HAL", "sceneItemId": item, "sceneItemTransform": {"positionX": x, "positionY": y}},
        )

    def _text_panel(
        self,
        name: str,
        text: str,
        x: int,
        y: int,
        size: int,
        color: int,
        *,
        face: str = _SANS,
        style: str = "Regular",
        width: int | None = None,
        centered: bool = False,
    ) -> None:
        settings: dict[str, Json] = {
            "text": text,
            "font": {"face": face, "style": style, "size": size, "flags": 0},
            "color1": color,
            "color2": color,
            "antialiasing": True,
        }
        if width is not None:
            settings |= {"word_wrap": True, "custom_width": width}
        item = self._create_input(name, "text_ft2_source_v2", settings)
        transform: dict[str, Json] = {"positionX": x, "positionY": y}
        if centered:
            transform["alignment"] = 0
        self.request(
            "SetSceneItemTransform", {"sceneName": "HAL", "sceneItemId": item, "sceneItemTransform": transform}
        )

    def update(self, text: str) -> None:
        if self._process is None or self._process.poll() is not None:
            raise RuntimeError("OBS exited; see obs.log")
        if text != self._text:
            headline, *details = text.split(" · ")
            self.request("SetInputSettings", {"inputName": "Overlay", "inputSettings": {"text": headline}})
            self.request(
                "SetInputSettings", {"inputName": "Overlay detail", "inputSettings": {"text": "\n".join(details)}}
            )
            self._text = text
        self.refresh_window()

    def refresh_window(self) -> None:
        if self._process is None or self._process.poll() is not None:
            raise RuntimeError("OBS exited; see obs.log")
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
        if not self._visible:
            return None
        return super().screenshot()

    def capture_connection(self) -> ObsConnection:
        if self._password is None:
            raise RuntimeError("OBS is not authenticated")
        connection = ObsConnection()
        connection.open(self._password)
        return connection

    def write_control(self, path: Path) -> None:
        if self._password is None:
            raise RuntimeError("OBS is not authenticated")
        temporary = path.with_suffix(".partial")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w") as destination:
            json.dump({"schema_version": 1, "password": self._password}, destination)
        temporary.replace(path)

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
        super().close()
        self._password = None
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None

    def update_panel(self, path: Path) -> None:
        self.request("SetInputSettings", {"inputName": "Live panel", "inputSettings": {"file": str(path)}})
