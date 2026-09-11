import fcntl
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parents[1]
_RUN_HOST = _REPO_ROOT / "deploy" / "netplay" / "run-host.sh"
_RUN_LOCAL = _REPO_ROOT / "deploy" / "netplay" / "run-local.sh"


def _write_executable(path: Path, body: str) -> None:
    path.write_text(f"#!/usr/bin/env bash\nset -eu\n{body}")
    path.chmod(0o755)


def test_host_launcher_skips_cloudflare_for_local_mode(tmp_path: Path) -> None:
    command_dir = tmp_path / "bin"
    command_dir.mkdir()
    log_path = tmp_path / "commands.log"
    _write_executable(
        command_dir / "uv",
        "printf 'uv|%s|%s|%s\\n' \"${HAL_NETPLAY_ALLOWED_ORIGINS:-}\" "
        '"${HAL_NETPLAY_ALLOWED_HOSTS:-}" "$*" >> "$HAL_DEPLOY_TEST_LOG"\n',
    )
    _write_executable(command_dir / "xvfb-run", "exit 0\n")
    _write_executable(
        command_dir / "cloudflared",
        "printf 'cloudflared\\n' >> \"$HAL_DEPLOY_TEST_LOG\"\n",
    )

    environment_file = tmp_path / "netplay.env"
    environment_file.write_text(
        "\n".join(
            (
                "HAL_GIT_SHA=" + "a" * 40,
                "HAL_NETPLAY_POLICY=/policy.halpolicy",
                "HAL_NETPLAY_USER_JSON_A=/user.json",
                "HAL_ISO_PATH=/ssbm.ciso",
                "HAL_NETPLAY_EMULATOR_PATH=/Slippi.AppImage",
                "CLOUDFLARE_TUNNEL_TOKEN=",
                "AWS_ENDPOINT_URL=https://example.invalid",
                "AWS_ACCESS_KEY_ID=test",
                "AWS_SECRET_ACCESS_KEY=test",
                "AWS_BUCKET=test",
                f"HAL_NETPLAY_STATE_DIR={tmp_path / 'state'}",
                "",
            )
        )
    )
    environment = os.environ.copy()
    environment["PATH"] = f"{command_dir}:/usr/bin:/bin"
    environment["HAL_DEPLOY_TEST_LOG"] = str(log_path)

    result = subprocess.run(
        [_RUN_HOST, environment_file],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
        timeout=5,
    )

    assert result.returncode == 0, result.stderr
    command_log = log_path.read_text()
    assert "uv|http://127.0.0.1:3000,http://localhost:3000|127.0.0.1,localhost|sync" in command_log
    assert "cloudflared" not in command_log


def test_host_launcher_stops_runner_when_api_fails(tmp_path: Path) -> None:
    command_dir = tmp_path / "bin"
    command_dir.mkdir()
    runner_pid_path = tmp_path / "runner.pid"
    _write_executable(
        command_dir / "uv",
        "if [[ $* == *hal-netplay-api* ]]; then sleep 0.2; exit 1; fi\n"
        "if [[ $* == *hal-netplay-runner* ]]; then\n"
        '  echo $$ > "$HAL_RUNNER_PID_PATH"\n'
        "  trap 'exit 0' TERM\n"
        "  while true; do sleep 1; done\n"
        "fi\n",
    )
    _write_executable(command_dir / "xvfb-run", 'shift\nexec "$@"\n')

    environment_file = tmp_path / "netplay.env"
    environment_file.write_text(
        "\n".join(
            (
                "HAL_GIT_SHA=" + "a" * 40,
                "HAL_NETPLAY_POLICY=/policy.halpolicy",
                "HAL_NETPLAY_USER_JSON_A=/user.json",
                "HAL_ISO_PATH=/ssbm.ciso",
                "HAL_NETPLAY_EMULATOR_PATH=/Slippi.AppImage",
                "AWS_ENDPOINT_URL=https://example.invalid",
                "AWS_ACCESS_KEY_ID=test",
                "AWS_SECRET_ACCESS_KEY=test",
                "AWS_BUCKET=test",
                f"HAL_NETPLAY_STATE_DIR={tmp_path / 'state'}",
                "",
            )
        )
    )
    environment = os.environ.copy()
    environment["PATH"] = f"{command_dir}:/usr/bin:/bin"
    environment["HAL_RUNNER_PID_PATH"] = str(runner_pid_path)

    result = subprocess.run(
        [_RUN_HOST, environment_file],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
        timeout=5,
    )

    assert result.returncode == 1
    runner_pid = int(runner_pid_path.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(runner_pid, 0)


def test_local_launcher_has_one_command_interface() -> None:
    result = subprocess.run(
        [_RUN_LOCAL, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == "usage: deploy/netplay/run-local.sh [environment-file]\n"


def test_local_launcher_rejects_a_second_instance(tmp_path: Path) -> None:
    environment_file = tmp_path / "netplay.env"
    environment_file.touch()
    lock_path = tmp_path / "run-local.lock"
    environment = os.environ.copy()
    environment["HAL_NETPLAY_LOCAL_LOCK"] = str(lock_path)

    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(
            [_RUN_LOCAL, environment_file],
            check=False,
            capture_output=True,
            env=environment,
            text=True,
        )

    assert result.returncode == 2
    assert result.stderr == "local netplay is already running\n"


def test_local_launcher_waits_for_host_cleanup(tmp_path: Path) -> None:
    command_dir = tmp_path / "bin"
    command_dir.mkdir()
    runner_pid_path = tmp_path / "runner.pid"
    _write_executable(
        command_dir / "uv",
        "if [[ $* == *hal-netplay-api* ]]; then trap 'exit 0' TERM; while true; do sleep 1; done; fi\n"
        "if [[ $* == *hal-netplay-runner* ]]; then\n"
        '  echo $$ > "$HAL_RUNNER_PID_PATH"\n'
        "  trap '' TERM\n"
        "  while true; do sleep 1; done\n"
        "fi\n",
    )
    _write_executable(command_dir / "xvfb-run", 'shift\nexec "$@"\n')
    _write_executable(command_dir / "npm", "trap 'exit 0' TERM\nwhile true; do sleep 1; done\n")

    environment_file = tmp_path / "netplay.env"
    environment_file.write_text(
        "\n".join(
            (
                "HAL_GIT_SHA=" + "a" * 40,
                "HAL_NETPLAY_POLICY=/policy.halpolicy",
                "HAL_NETPLAY_USER_JSON_A=/user.json",
                "HAL_ISO_PATH=/ssbm.ciso",
                "HAL_NETPLAY_EMULATOR_PATH=/Slippi.AppImage",
                "AWS_ENDPOINT_URL=https://example.invalid",
                "AWS_ACCESS_KEY_ID=test",
                "AWS_SECRET_ACCESS_KEY=test",
                "AWS_BUCKET=test",
                f"HAL_NETPLAY_STATE_DIR={tmp_path / 'state'}",
                "",
            )
        )
    )
    environment = os.environ.copy()
    environment["PATH"] = f"{command_dir}:/usr/bin:/bin"
    environment["HAL_NETPLAY_LOCAL_LOCK"] = str(tmp_path / "run-local.lock")
    environment["HAL_RUNNER_PID_PATH"] = str(runner_pid_path)

    process = subprocess.Popen(
        [_RUN_LOCAL, environment_file],
        env=environment,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    for _ in range(100):
        if runner_pid_path.exists():
            break
        time.sleep(0.01)
    assert runner_pid_path.exists()

    process.send_signal(signal.SIGINT)
    process.communicate(timeout=7)

    runner_pid = int(runner_pid_path.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(runner_pid, 0)
