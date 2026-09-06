import subprocess

import pytest

from hal import r2


def test_rclone_uses_aws_credentials_without_a_config_file(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://account.r2.cloudflarestorage.com")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret-key")

    def run(
        command: tuple[str, ...],
        *,
        capture_output: bool,
        text: bool,
        check: bool,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        assert command == ("rclone", "lsf", "r2:hal")
        assert capture_output and text and not check
        captured.update(env)
        return subprocess.CompletedProcess(command, 0, "object\n", "")

    monkeypatch.setattr(r2.subprocess, "run", run)

    assert r2.run_rclone("lsf", "r2:hal") == "object\n"
    assert captured["RCLONE_CONFIG_R2_TYPE"] == "s3"
    assert captured["RCLONE_CONFIG_R2_PROVIDER"] == "Cloudflare"
    assert captured["RCLONE_CONFIG_R2_ACCESS_KEY_ID"] == "access-key"
    assert captured["RCLONE_CONFIG_R2_SECRET_ACCESS_KEY"] == "secret-key"
    assert captured["RCLONE_CONFIG_R2_ENDPOINT"] == "https://account.r2.cloudflarestorage.com"
