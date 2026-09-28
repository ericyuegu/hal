import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

from hal.netplay_service.health import RunnerState
from hal.netplay_service.health import RunnerStatus
from hal.netplay_service.health import SlotState
from hal.netplay_service.health import SlotStatus
from hal.netplay_service.health import write_runner_status
from hal.netplay_service.health import write_slot_status

_ROOT = Path(__file__).parents[1]
_DEPLOY = _ROOT / "deploy" / "netplay"
_RUN_HOST = _DEPLOY / "run-host.sh"
_RUN_LOCAL = _DEPLOY / "run-local.sh"


def _write_executable(path: Path, body: str) -> None:
    path.write_text(f"#!/usr/bin/env bash\nset -eu\n{body}")
    path.chmod(0o755)


def test_shell_launchers_parse() -> None:
    for path in (_RUN_HOST, _RUN_LOCAL, _DEPLOY / "deploy-web.sh"):
        subprocess.run(["bash", "-n", path], check=True)


def test_host_launcher_runs_only_the_remote_runner(tmp_path: Path) -> None:
    commands = tmp_path / "commands"
    commands.mkdir()
    log = tmp_path / "log"
    _write_executable(commands / "uv", 'printf "%s\\n" "$*" >> "$HAL_DEPLOY_TEST_LOG"\n')
    environment_file = tmp_path / "netplay.env"
    environment_file.write_text(
        "\n".join(
            (
                "HAL_GIT_SHA=" + "a" * 40,
                "HAL_NETPLAY_API_URL=http://127.0.0.1:8787",
                "HAL_NETPLAY_RUNNER_TOKEN=token",
                "HAL_NETPLAY_STREAM=0",
                "AWS_ENDPOINT_URL=https://example.invalid",
                "AWS_ACCESS_KEY_ID=test",
                "AWS_SECRET_ACCESS_KEY=test",
                "AWS_BUCKET=test",
            )
        )
    )
    environment = os.environ | {
        "PATH": f"{commands}:/usr/bin:/bin",
        "HAL_DEPLOY_TEST_LOG": str(log),
        "HAL_NETPLAY_STATE_DIR": str(tmp_path / "state"),
    }

    result = subprocess.run([_RUN_HOST, environment_file], capture_output=True, env=environment, text=True)

    assert result.returncode == 0, result.stderr
    recorded = log.read_text()
    assert "sync --locked" in recorded
    assert "run hal-netplay-runner --slots 1" in recorded
    assert "--display-base 100" in recorded
    assert "hal-netplay-api" not in recorded
    assert "cloudflared" not in _RUN_HOST.read_text()
    assert "xvfb-run" not in _RUN_HOST.read_text()


def test_local_launcher_has_one_command_interface() -> None:
    result = subprocess.run([_RUN_LOCAL, "--help"], check=True, capture_output=True, text=True)
    assert result.stdout == "usage: deploy/netplay/run-local.sh [environment-file]\n"


def test_local_launcher_rejects_a_second_instance(tmp_path: Path) -> None:
    environment_file = tmp_path / "netplay.env"
    environment_file.touch()
    lock_path = tmp_path / "run-local.lock"
    environment = os.environ | {"HAL_NETPLAY_LOCAL_LOCK": str(lock_path)}

    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(
            [_RUN_LOCAL, environment_file], check=False, capture_output=True, env=environment, text=True
        )

    assert result.returncode == 2
    assert result.stderr == "local netplay is already running\n"


def test_local_launcher_orders_worker_page_then_runner() -> None:
    source = _RUN_LOCAL.read_text()
    worker = source.index("wrangler dev")
    page = source.index("npm run dev")
    runner = source.index("hal-netplay-runner")
    assert worker < page < runner
    assert "HAL_NETPLAY_API_URL=http://127.0.0.1:8787" in source
    assert "--local-assets" in source


def test_page_uses_same_origin_api_and_local_proxy() -> None:
    client = (_ROOT / "web" / "netplay" / "lib" / "netplay-api.ts").read_text()
    vite = (_ROOT / "web" / "netplay" / "vite.config.ts").read_text()
    assert "NEXT_PUBLIC_HAL_API_URL" not in client
    assert "fetch(path" in client
    assert "'/v1': { target: 'http://127.0.0.1:8787' }" in vite


def test_compose_has_only_a_runner() -> None:
    compose = (_DEPLOY / "compose.yaml").read_text()
    assert "  runner:" in compose
    assert "  api:" not in compose
    assert "  tunnel:" not in compose
    assert "NVIDIA_DRIVER_CAPABILITIES: compute,graphics,utility,video" in compose


def test_runner_image_contains_stream_runtime_without_an_entrypoint_display() -> None:
    dockerfile = (_DEPLOY / "Dockerfile").read_text()
    entrypoint = (_ROOT / "docker" / "entrypoint.sh").read_text()
    for package in ("ffmpeg", "fonts-dejavu-core", "procps", "pulseaudio", "x11-xserver-utils", "xvfb"):
        assert package in dockerfile
    assert "Xvfb" not in entrypoint
    assert 'exec "$@"' in entrypoint


def test_stream_measurement_captures_each_active_slot(tmp_path: Path) -> None:
    status_path = tmp_path / "runner.json"
    write_runner_status(
        status_path,
        RunnerStatus(
            RunnerState.READY,
            "ready",
            "a" * 64,
            2,
            2,
            60.0,
            59.9,
            17.0,
            5.0,
            10.0,
            8.0,
            0.2,
            0,
            100.0,
        ),
    )
    for slot in range(2):
        write_slot_status(
            status_path.with_name(f"{status_path.name}.slot-{slot}.json"),
            SlotStatus(slot, SlotState.PLAYING, 59.9, 17.0, 5.0, 10.0, None, 0, 100.0),
        )
    output = tmp_path / "measurement.json"

    subprocess.run(
        [
            sys.executable,
            _DEPLOY / "capture-stream-metrics.py",
            "--label",
            "test",
            "--hardware",
            "test-gpu",
            "--streaming",
            "on",
            "--status-path",
            status_path,
            "--output",
            output,
        ],
        cwd=_ROOT,
        check=True,
    )

    payload = json.loads(output.read_text())
    assert [(slot["slot"], slot["role"]) for slot in payload["slots"]] == [(0, "stream"), (1, "headless")]
    assert payload["slots"][0]["game_fps"] == 59.9
    assert payload["git_sha"]
