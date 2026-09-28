import fcntl
import os
import subprocess
from pathlib import Path

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
