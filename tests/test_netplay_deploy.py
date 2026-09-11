import os
import subprocess
from pathlib import Path

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
    assert "uv|http://127.0.0.1:3000|127.0.0.1,localhost|sync" in command_log
    assert "cloudflared" not in command_log


def test_local_launcher_has_one_command_interface() -> None:
    result = subprocess.run(
        [_RUN_LOCAL, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == "usage: deploy/netplay/run-local.sh [environment-file]\n"
