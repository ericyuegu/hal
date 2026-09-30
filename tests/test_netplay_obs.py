import base64
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from hal.netplay_service import obs


def test_obs_authentication_matches_v5_challenge() -> None:
    secret = base64.b64encode(hashlib.sha256(b"passwordsalt").digest())
    expected = base64.b64encode(hashlib.sha256(secret + b"challenge").digest()).decode()
    assert obs.authentication("password", "salt", "challenge") == expected


def test_obs_configuration_uses_texture_nvenc_and_1080p60(tmp_path: Path) -> None:
    obs.write_configuration(tmp_path, "control-secret")
    root = tmp_path / "obs-studio"
    assert "FirstRun=true" in (root / "global.ini").read_text()
    profile = root / "basic/profiles/HAL"
    video = (profile / "basic.ini").read_text()
    assert "Encoder=jim_nvenc" in video
    assert "OutputCX=1920\nOutputCY=1080" in video
    assert "FPSCommon=60" in video
    encoder = json.loads((profile / "streamEncoder.json").read_text())
    assert (encoder["rate_control"], encoder["bitrate"], encoder["keyint_sec"], encoder["bf"]) == ("CBR", 6000, 2, 2)
    assert not encoder["lookahead"]
    plugin = json.loads((root / "plugin_config/obs-websocket/config.json").read_text())
    assert plugin["auth_required"] and plugin["server_password"] == "control-secret"
    assert not any("live_secret" in p.read_text() for p in root.rglob("*") if p.is_file())


def test_obs_requires_nvidia_egl_even_if_the_parent_selected_mesa(tmp_path: Path) -> None:
    studio = obs.ObsStudio(":90", tmp_path, {"__EGL_VENDOR_LIBRARY_FILENAMES": "50_mesa.json"})
    assert studio._environment["__EGL_VENDOR_LIBRARY_FILENAMES"] == "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"


def test_window_selection_excludes_launcher_and_other_apps() -> None:
    windows = [
        {"itemValue": "1\r\nFaster Melee - Slippi (3.6.4)\r\nAppRun.wrapped"},
        {"itemValue": "2\r\nDolphin\r\nFileManager"},
        {"itemValue": "3\r\nDolphin\r\nAppRun.wrapped"},
        {"itemValue": "4\r\nOBS Studio\r\nobs"},
    ]
    assert obs.dolphin_window(windows) == "3\r\nDolphin\r\nAppRun.wrapped"
    assert obs.dolphin_window(windows[:2]) is None
    with pytest.raises(ValueError, match="multiple Dolphin"):
        obs.dolphin_window(windows + [{"itemValue": "5\r\nDolphin\r\nAppRun.wrapped"}])


def test_obs_request_checks_identity_and_redacts_errors(tmp_path: Path) -> None:
    studio = obs.ObsStudio(":90", tmp_path, {})
    socket = Mock()
    studio._socket = socket
    socket.recv.return_value = json.dumps(
        {
            "op": 7,
            "d": {
                "requestId": "1",
                "requestStatus": {"result": False, "code": 400, "comment": "live_secret"},
            },
        }
    )
    with pytest.raises(RuntimeError, match="code 400") as error:
        studio.request("SetStreamServiceSettings", {"key": "live_secret"})
    assert "live_secret" not in str(error.value)
    socket.recv.return_value = json.dumps(
        {
            "op": 7,
            "d": {
                "requestId": "other",
                "requestStatus": {"result": True},
            },
        }
    )
    with pytest.raises(ValueError, match="unexpected"):
        studio.request("GetStats")


def test_capture_disables_when_window_disappears_and_does_not_rehook_each_update(tmp_path: Path) -> None:
    studio = obs.ObsStudio(":90", tmp_path, {})
    studio._process = Mock()
    studio._process.poll.return_value = None
    studio._capture_id = 1
    request = Mock(return_value={"propertyItems": [{"itemValue": "3\r\nDolphin\r\nAppRun.wrapped"}]})
    studio.request = request
    studio.update("HAL", playing=True)
    assert request.call_args.args == (
        "SetSceneItemEnabled",
        {"sceneName": "HAL", "sceneItemId": 1, "sceneItemEnabled": True},
    )
    request.reset_mock()
    studio.update("HAL", playing=True)
    assert request.call_count == 1
    assert request.call_args.args[0] == "GetInputPropertiesListPropertyItems"
    request.return_value = {"propertyItems": []}
    studio.update("HAL", playing=True)
    assert request.call_args.args[1]["sceneItemEnabled"] is False
    request.reset_mock()
    studio.update("Play HAL at 20xx.xyz", playing=False)
    assert all(call.args[0] != "GetInputPropertiesListPropertyItems" for call in request.call_args_list)


def test_waiting_card_is_built_off_air_then_inserted_below_game(tmp_path: Path) -> None:
    studio = obs.ObsStudio(":90", tmp_path, {})
    request = Mock(return_value={"sceneItemId": 7})
    studio.request = request

    studio.configure_waiting_card()

    calls = [call.args for call in request.call_args_list]
    inputs = [data for method, data in calls if method == "CreateInput"]
    assert all(data["sceneName"] == "Waiting card" for data in inputs)
    assert {data["inputKind"] for data in inputs} == {"color_source_v3", "text_ft2_source_v2"}
    background = next(data["inputSettings"] for data in inputs if data["inputName"] == "Waiting background")
    assert (background["width"], background["height"]) == (1920, 1080)
    assert background["color"] & 0xFFFFFF != 0
    text = " ".join(data["inputSettings"].get("text", "") for data in inputs)
    assert "NEXT MATCH" in text
    assert "20xx.xyz" in text
    assert "#" not in text
    assert calls[-3:] == [
        ("CreateSceneItem", {"sceneName": "HAL", "sourceName": "Waiting card", "sceneItemEnabled": False}),
        ("SetSceneItemIndex", {"sceneName": "HAL", "sceneItemId": 7, "sceneItemIndex": 0}),
        ("SetSceneItemEnabled", {"sceneName": "HAL", "sceneItemId": 7, "sceneItemEnabled": True}),
    ]
    assert all(method != "SetCurrentProgramScene" for method, _ in calls)


def test_obs_startup_places_waiting_card_under_capture_and_overlay(tmp_path: Path) -> None:
    studio = obs.ObsStudio(":90", tmp_path, {})
    request = Mock(return_value={"sceneItemId": 7})
    studio.request = request

    studio._configure_scene()

    main_sources = [
        data["inputName"] if method == "CreateInput" else data["sourceName"]
        for method, data in (call.args for call in request.call_args_list)
        if method in ("CreateInput", "CreateSceneItem") and data["sceneName"] == "HAL"
    ]
    assert main_sources == ["Waiting card", "Dolphin", "Overlay", "Game audio"]
