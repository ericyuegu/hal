import json
import subprocess
import time
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from hal.netplay_service.queue_client import StreamGrant
from hal.netplay_service.stream import DisplayGroup
from hal.netplay_service.stream import GameStreamState
from hal.netplay_service.stream import IdleStreamState
from hal.netplay_service.stream import PulseAudio
from hal.netplay_service.stream import StreamSupervisor
from hal.netplay_service.stream import overlay_text
from hal.netplay_service.stream import write_stream_state


class _Process:
    def __init__(self, returncode: int | None = None) -> None:
        self.returncode = returncode
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


def _wait_for(condition: Any) -> None:
    deadline = time.monotonic() + 2.0
    while not condition():
        assert time.monotonic() < deadline
        time.sleep(0.005)


def test_xvfb_group_starts_one_sized_display_per_slot_and_cleans_up() -> None:
    commands: list[list[str]] = []
    roots: list[list[str]] = []
    processes: list[_Process] = []

    def popen(command: list[str], **_kwargs: object) -> Any:
        commands.append(command)
        process = _Process()
        processes.append(process)
        return process

    def run(command: list[str], **_kwargs: object) -> Any:
        roots.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=b"OpenGL vendor string: NVIDIA Corporation")

    with DisplayGroup(3, 100, True, stream_display=":90", popen=popen, run=run, ready=lambda _number: True) as group:
        assert group.displays == (":90", ":101", ":102")
        assert commands[0][:3] == ["openbox", "--sm-disable", "--config-file"]
        assert "<fullscreen>yes</fullscreen>" in Path(commands[0][3]).read_text()
        assert commands[1:] == [
            ["Xvfb", ":101", "-screen", "0", "640x480x24", "-nolisten", "tcp"],
            ["Xvfb", ":102", "-screen", "0", "640x480x24", "-nolisten", "tcp"],
        ]
        assert roots[0] == ["glxinfo", "-B"]
        assert [command[2] for command in roots[1:]] == [":101", ":102"]
    assert all(process.terminated for process in processes)


def test_pulseaudio_owns_one_private_null_sink(tmp_path: Path) -> None:
    commands: list[list[str]] = []
    process = _Process()

    def popen(command: list[str], **_kwargs: object) -> Any:
        commands.append(command)
        return process

    pulse = PulseAudio(tmp_path / "pulse", popen=popen, ready=lambda _path: True)
    with pulse:
        assert pulse.environment == {
            "PULSE_SERVER": f"unix:{tmp_path}/pulse/native",
            "PULSE_SINK": "hal_stream",
        }
        command = commands[0]
        assert command.count("pulseaudio") == 1
        assert "--exit-idle-time=-1" in command
        assert any("module-null-sink sink_name=hal_stream" in argument for argument in command)
        assert any(f"socket={tmp_path}/pulse/native" in argument for argument in command)
    assert process.terminated


def test_overlay_state_has_no_connect_code(tmp_path: Path) -> None:
    player_code = "CRYO#610"
    state = GameStreamState("FALCO", "MASTER", 25.0, 12)
    path = tmp_path / "state.json"
    write_stream_state(path, state)
    game = overlay_text(state, 3)
    idle = overlay_text(IdleStreamState(), 3)

    assert game == "HAL · Master rank Falco · difficulty 25 · Game 12 · play at 20xx.xyz"
    assert idle == "Play HAL at 20xx.xyz · 3 players waiting"
    assert player_code not in path.read_text()
    assert player_code not in game
    assert set(json.loads(path.read_text())) == {
        "schema_version",
        "state",
        "character",
        "imitation",
        "desired_return",
        "game_number",
    }


def test_stream_display_rejects_software_renderer() -> None:
    run = Mock(return_value=subprocess.CompletedProcess([], 0, stdout=b"OpenGL renderer string: llvmpipe"))
    popen = Mock()
    with (
        pytest.raises(RuntimeError, match="NVIDIA hardware"),
        DisplayGroup(1, 100, True, stream_display=":90", run=run, popen=popen),
    ):
        pass
    popen.assert_not_called()


def test_stream_display_rejects_missing_or_overlapping_display() -> None:
    with pytest.raises(ValueError, match="dedicated"):
        DisplayGroup(1, 100, True)
    with pytest.raises(ValueError, match="overlaps"):
        DisplayGroup(2, 100, True, stream_display=":101")


def test_stream_supervisor_starts_once_and_stops_on_lease_loss(tmp_path: Path) -> None:
    studio = Mock()
    studio.stats.return_value = {}
    supervisor = StreamSupervisor(
        ":90",
        tmp_path / "state.json",
        tmp_path / "overlay.txt",
        lambda: 4,
        {},
        bandwidth_test=True,
        slot_status_path=tmp_path / "slot.json",
        obs_factory=lambda: studio,
    )
    with supervisor:
        grant = StreamGrant(0, "live_secret")
        supervisor.set_grant(grant)
        _wait_for(lambda: studio.start.call_count == 1)
        supervisor.set_grant(grant)
        time.sleep(0.05)
        assert studio.start.call_count == 1
        supervisor.set_grant(None)
        _wait_for(lambda: studio.close.call_count == 1)
    studio.start.assert_called_once_with("live_secret", bandwidth_test=True)
    studio.close.assert_called_once()


def test_stream_supervisor_caps_restart_backoff_at_thirty_seconds(tmp_path: Path) -> None:
    delays: list[float] = []
    studio = Mock()
    studio.start.side_effect = RuntimeError("OBS exited")
    supervisor = StreamSupervisor(
        ":90",
        tmp_path / "state.json",
        tmp_path / "overlay.txt",
        lambda: 0,
        {},
        bandwidth_test=True,
        slot_status_path=tmp_path / "slot.json",
        obs_factory=lambda: studio,
    )

    def wait(seconds: float) -> None:
        if seconds >= 1:
            delays.append(seconds)
            if len(delays) == 7:
                supervisor._stop.set()

    supervisor._wait = wait  # type: ignore[method-assign]
    supervisor.set_grant(StreamGrant(0, "live_secret"))
    with supervisor:
        _wait_for(lambda: len(delays) == 7)
    assert delays == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0]
    assert studio.close.call_count == 7


def test_overlay_hides_connecting_and_stale_game_labels(tmp_path: Path) -> None:
    from hal.netplay_service.health import SlotState
    from hal.netplay_service.health import SlotStatus
    from hal.netplay_service.health import write_slot_status

    state = tmp_path / "state.json"
    slot = tmp_path / "slot.json"
    write_stream_state(state, GameStreamState("FOX", "IBDW#0", 20, 1))
    supervisor = StreamSupervisor(
        ":90",
        state,
        tmp_path / "overlay.txt",
        lambda: 0,
        {},
        bandwidth_test=True,
        slot_status_path=slot,
    )
    for status, age, visible in (
        (SlotState.CONNECTING, 0, False),
        (SlotState.PLAYING, 0, True),
        (SlotState.PLAYING, 10, False),
        (SlotState.IDLE, 0, False),
    ):
        write_slot_status(slot, SlotStatus(0, status, None, None, None, None, None, 0, time.time() - age))
        text = supervisor._refresh_overlay(2)
        assert ("iBDW" in text) is visible
        assert "CRYO#610" not in text
