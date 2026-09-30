import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from hal.inference.api import ActionPlan
from hal.netplay_service.obs import ObsStudio
from hal.netplay_service.value_meter import MeterDisplay
from hal.netplay_service.value_meter import OverlayProcess
from hal.netplay_service.value_meter import ValueMeter
from hal.netplay_service.value_meter import read_control
from hal.netplay_service.value_meter import read_value
from hal.netplay_service.value_meter import write_value


def plan(frame: int, value: float, *, sequence: int = 0, generation: int = 1) -> ActionPlan:
    return ActionPlan(0, generation, sequence, frame, (), value)


def test_ema_uses_game_frames_and_rejects_repeated_or_old_samples() -> None:
    meter = ValueMeter()
    meter.observe(plan(0, 20.0), now=1)
    meter.observe(plan(4, 100.0, sequence=1), now=2)
    expected = 20 + (1 - 2 ** (-4 / 6)) * 80
    assert meter.snapshot().ema == pytest.approx(expected)
    before = meter.snapshot()
    meter.observe(plan(4, -100.0, sequence=1), now=3)
    meter.observe(plan(3, -100.0, sequence=2), now=3)
    assert meter.snapshot() == before
    meter.observe(plan(10, 100.0, sequence=2), now=4)
    assert meter.snapshot().ema == pytest.approx((expected + 100) / 2)


def test_meter_resets_for_games_and_generations() -> None:
    meter = ValueMeter()
    meter.observe(plan(120, 30.0), now=1)
    meter.observe(plan(-10, -20.0, generation=2), now=2)
    assert meter.snapshot().ema == -20
    meter.observe(plan(121, 100.0, sequence=1), now=3)
    assert meter.snapshot().ema == -20
    meter.clear()
    assert meter.snapshot() is None
    meter.observe(plan(0, 10.0), now=4)
    assert meter.snapshot().ema == 10


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), 3, True])
def test_invalid_values_never_replace_the_current_sample(invalid) -> None:
    meter = ValueMeter()
    meter.observe(plan(0, 2.0), now=1)
    with pytest.raises(ValueError, match="finite"):
        meter.observe(plan(4, invalid, sequence=1), now=2)
    assert meter.snapshot().ema == 2


def test_meter_file_hides_menus_countdown_and_stale_values(tmp_path: Path) -> None:
    path = tmp_path / "value.json"
    meter = ValueMeter()
    meter.observe(plan(0, 40.0), now=5)
    sample = meter.snapshot()
    write_value(path, sample, playing=True)
    assert read_value(path, now=5.5) == 40
    assert read_value(path, now=6.1) is None
    assert read_value(path, now=4.9) is None
    write_value(path, sample, playing=False)
    assert read_value(path, now=5.5) is None
    write_value(path, replace(sample, source_frame=-1), playing=True)
    assert read_value(path, now=5.5) is None
    write_value(path, None, playing=False)
    assert read_value(path, now=5.5) is None
    assert not path.with_suffix(".partial").exists()


@pytest.mark.parametrize("field,value", [("schema_version", 2), ("schema_version", True), ("playing", 1)])
def test_meter_file_rejects_invalid_schema(tmp_path: Path, field, value) -> None:
    path = tmp_path / "value.json"
    data = {"schema_version": 1, "playing": False, "sample": None}
    data[field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        read_value(path, now=1)


def display() -> tuple[MeterDisplay, Mock]:
    connection = Mock()
    connection.request.side_effect = lambda method, data=None: (
        {"sceneItems": []} if method == "GetSceneItemList" else {"sceneItemId": connection.request.call_count}
    )
    meter = MeterDisplay(connection)
    meter.configure()
    return meter, connection


def test_display_uses_only_meter_sources_and_clamps_only_the_bar() -> None:
    meter, connection = display()
    for value in (240.0, -240.0, 0.0):
        connection.request.reset_mock()
        meter.update(value)
        calls = [call.args for call in connection.request.call_args_list]
        settings = {data["inputName"]: data["inputSettings"] for method, data in calls if method == "SetInputSettings"}
        assert settings["Model value"]["text"] == f"Model value\n{value:+.1f}"
        assert settings["Value fill"]["height"] == (1 if value == 0 else 180)
        assert all(method not in ("StartStream", "StopStream", "SetCurrentProgramScene") for method, _data in calls)
        assert all("Dolphin" not in str(data) for _method, data in calls)
        assert "#" not in json.dumps(calls)
        assert "Cody" not in json.dumps(calls)
        assert "advantage 120" not in json.dumps(calls)
    connection.request.reset_mock()
    meter.update(0.0)
    connection.request.assert_not_called()
    meter.update(None)
    assert all(call.args[1]["sceneItemEnabled"] is False for call in connection.request.call_args_list)


def test_overlay_restart_reuses_sources_instead_of_duplicating_them() -> None:
    meter, connection = display()
    known = meter.items.copy()
    connection.request.reset_mock()
    connection.request.side_effect = lambda method, data=None: {
        "sceneItems": [{"sourceName": name, "sceneItemId": identity} for name, identity in known.items()]
    }
    restarted = MeterDisplay(connection)
    restarted.configure()
    assert restarted.items == known
    assert "CreateInput" not in [call.args[0] for call in connection.request.call_args_list]


def test_ranked_scene_has_no_title_and_private_control_file(tmp_path: Path) -> None:
    studio = ObsStudio(":90", tmp_path, {})
    studio.request = Mock(return_value={"sceneItemId": 1})
    studio._configure_scene(ranked=True)
    names = [call.args[1]["inputName"] for call in studio.request.call_args_list if call.args[0] == "CreateInput"]
    assert names == ["Dolphin", "Game audio"]
    studio._password = "private-local-control"
    path = tmp_path / "obs-control.json"
    studio.write_control(path)
    assert path.stat().st_mode & 0o777 == 0o600
    assert read_control(path) == "private-local-control"


def test_failed_overlay_process_restarts_without_owning_player_or_obs(tmp_path: Path, monkeypatch) -> None:
    from hal.netplay_service import value_meter

    processes = [Mock(pid=100), Mock(pid=101)]
    for process in processes:
        process.poll.return_value = None
    launch = Mock(side_effect=processes)
    monkeypatch.setattr(value_meter.subprocess, "Popen", launch)
    worker = OverlayProcess(tmp_path)
    worker.ensure_running(1)
    worker.ensure_running(2)
    assert launch.call_count == 1
    processes[0].poll.return_value = 1
    worker.ensure_running(3)
    assert launch.call_count == 1
    worker.ensure_running(4)
    assert launch.call_count == 2
    assert launch.call_args.args[0][1:3] == ["-m", "hal.scripts.ranked_overlay"]
    assert (tmp_path / "overlay.pid").read_text() == "101"
    worker.close()
    processes[1].terminate.assert_called_once()


def test_overlay_spawn_failure_retries_without_raising_into_the_player(tmp_path: Path, monkeypatch) -> None:
    from hal.netplay_service import value_meter

    child = Mock(pid=77)
    child.poll.return_value = None
    spawn = Mock(side_effect=[OSError("cannot spawn"), child])
    monkeypatch.setattr(value_meter.subprocess, "Popen", spawn)
    worker = OverlayProcess(tmp_path)
    worker.ensure_running(1)
    worker.ensure_running(1.5)
    assert spawn.call_count == 1
    worker.ensure_running(2)
    assert spawn.call_count == 2
    worker.close()
