import json
import time
from pathlib import Path
from typing import Any

from hal.netplay_service.queue_client import StreamGrant
from hal.netplay_service.stream import GameStreamState
from hal.netplay_service.stream import IdleStreamState
from hal.netplay_service.stream import PulseAudio
from hal.netplay_service.stream import StreamSupervisor
from hal.netplay_service.stream import XvfbGroup
from hal.netplay_service.stream import ffmpeg_command
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
        return object()

    with XvfbGroup(3, 100, True, popen=popen, run=run, ready=lambda _number: True) as group:
        assert group.displays == (":100", ":101", ":102")
        assert commands == [
            ["Xvfb", ":100", "-screen", "0", "1920x1080x24", "-nolisten", "tcp"],
            ["Xvfb", ":101", "-screen", "0", "640x480x24", "-nolisten", "tcp"],
            ["Xvfb", ":102", "-screen", "0", "640x480x24", "-nolisten", "tcp"],
        ]
        assert [command[2] for command in roots] == [":100", ":101", ":102"]
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
    state = GameStreamState("FALCO", "MASTER", 25.0, 2)
    path = tmp_path / "state.json"
    write_stream_state(path, state)
    game = overlay_text(state, 3)
    idle = overlay_text(IdleStreamState(), 3)

    assert game == "HAL · Master rank Falco · difficulty 25 · Game 2 of 5 · play at 20xx.xyz"
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


def test_ffmpeg_uses_nvenc_cbr_audio_and_reloadable_overlay(tmp_path: Path) -> None:
    command = ffmpeg_command(":100", tmp_path / "overlay.txt", "live_secret", bandwidth_test=True)
    joined = " ".join(command)
    assert "-f x11grab -framerate 60 -draw_mouse 0 -video_size 1920x1080 -i :100.0" in joined
    assert "-f pulse -i hal_stream.monitor" in joined
    assert "drawtext=" in joined and "reload=1" in joined
    assert "-c:v h264_nvenc" in joined
    assert "-preset p5 -tune ll -profile:v high" in joined
    assert "-rc-lookahead 0 -bf 0 -zerolatency 1" in joined
    assert "-spatial-aq 1 -aq-strength 8 -pix_fmt yuv420p" in joined
    assert "-rc cbr" in joined
    assert "-b:v 6M -maxrate 6M -minrate 6M" in joined
    assert "-g 120" in joined
    assert "-forced-idr 1" in joined
    assert "-c:a aac -b:a 160k" in joined
    assert command[-1] == "rtmp://live.twitch.tv/app/live_secret?bandwidthtest=true"


def test_stream_supervisor_starts_once_and_stops_on_lease_loss(tmp_path: Path) -> None:
    processes: list[_Process] = []
    commands: list[list[str]] = []

    def popen(command: list[str], **_kwargs: object) -> Any:
        commands.append(command)
        process = _Process()
        processes.append(process)
        return process

    state = tmp_path / "state.json"
    write_stream_state(state, IdleStreamState())
    supervisor = StreamSupervisor(
        ":100",
        state,
        tmp_path / "overlay.txt",
        lambda: 4,
        {"PULSE_SERVER": "unix:/tmp/pulse", "PULSE_SINK": "hal_stream"},
        bandwidth_test=True,
        popen=popen,
    )
    with supervisor:
        grant = StreamGrant(0, "live_secret")
        supervisor.set_grant(grant)
        _wait_for(lambda: len(processes) == 1)
        supervisor.set_grant(grant)
        time.sleep(0.05)
        assert len(processes) == 1
        supervisor.set_grant(None)
        _wait_for(lambda: processes[0].terminated)
    assert len(commands) == 1
    assert "live_secret" in commands[0][-1]


def test_stream_supervisor_caps_restart_backoff_at_thirty_seconds(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    write_stream_state(state, IdleStreamState())
    delays: list[float] = []

    def popen(_command: list[str], **_kwargs: object) -> Any:
        return _Process(returncode=1)

    supervisor = StreamSupervisor(
        ":100",
        state,
        tmp_path / "overlay.txt",
        lambda: 0,
        {},
        bandwidth_test=True,
        popen=popen,
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
