import os
import subprocess
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from hal.netplay_service.health import RunnerState
from hal.netplay_service.health import RunnerStatus
from hal.netplay_service.health import write_runner_status
from hal.netplay_service.host_health import _handler
from hal.netplay_service.host_health import runner_is_healthy

_ROOT = Path(__file__).parents[1]
_DEPLOY = _ROOT / "deploy" / "netplay"
_SHA = "a" * 40


def _status(updated_at: float, healthy_slots: int = 1) -> RunnerStatus:
    return RunnerStatus(
        state=RunnerState.READY if healthy_slots else RunnerState.RECOVERING,
        message="ready" if healthy_slots else "recovering",
        policy_sha256="b" * 64,
        slots=1,
        healthy_slots=healthy_slots,
        target_fps=60.0,
        game_fps=60.0 if healthy_slots else None,
        frame_interval_p95_ms=16.7 if healthy_slots else None,
        dolphin_step_p95_ms=5.0 if healthy_slots else None,
        policy_round_trip_p95_ms=10.0 if healthy_slots else None,
        model_inference_p95_ms=8.0 if healthy_slots else None,
        batch_wait_p95_ms=0.2 if healthy_slots else None,
        recoveries=0,
        updated_at=updated_at,
    )


def _stub_gcloud(tmp_path: Path, *, fail_ssh: bool = False, instances: str = "") -> tuple[Path, dict[str, str]]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "gcloud.log"
    script = bin_dir / "gcloud"
    script.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$*" >> "$GCLOUD_LOG"\n'
        + ('if [[ $* == *"compute ssh"* ]]; then exit 1; fi\n' if fail_ssh else "")
        + 'if [[ $* == *"list-instances"* ]]; then printf "%s\\n" "$GCLOUD_INSTANCES"; fi\n'
    )
    script.chmod(0o755)
    timeout = bin_dir / "timeout"
    timeout.write_text('#!/usr/bin/env bash\nshift\nexec "$@"\n')
    timeout.chmod(0o755)
    environment = os.environ | {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "GCLOUD_LOG": str(log),
        "GCLOUD_INSTANCES": instances,
    }
    return log, environment


def _up_args(*extra: str) -> list[str]:
    return [
        str(_DEPLOY / "gce-up.sh"),
        "hal-netplay-test",
        "--project",
        "hal-project",
        "--image",
        f"us-docker.pkg.dev/hal-project/hal/hal-netplay-runner:{_SHA}",
        "--secret",
        "hal-netplay-runner-env",
        "--service-account",
        "runner@hal-project.iam.gserviceaccount.com",
        *extra,
    ]


def test_runner_dockerfile_is_sha_pinned_and_contains_no_assets() -> None:
    source = (_DEPLOY / "Dockerfile").read_text()
    assert "ARG HAL_GIT_SHA" in source
    assert 'LABEL org.opencontainers.image.revision="$HAL_GIT_SHA"' in source
    assert 'HAL_GIT_SHA="$HAL_GIT_SHA"' in source
    assert 'CMD ["hal-netplay-runner"' in source
    assert "COPY data" not in source
    assert "COPY fixtures" not in source
    assert ".halpolicy" not in source
    assert "account.json" not in source


def test_host_health_rejects_missing_invalid_stale_and_unhealthy_status(tmp_path: Path) -> None:
    path = tmp_path / "runner.json"
    assert runner_is_healthy(path, 100.0) is False
    path.write_text("not json")
    assert runner_is_healthy(path, 100.0) is False
    write_runner_status(path, _status(80.0))
    assert runner_is_healthy(path, 100.0) is False
    write_runner_status(path, _status(100.0, healthy_slots=0))
    assert runner_is_healthy(path, 100.0) is False
    write_runner_status(path, _status(100.0))
    assert runner_is_healthy(path, 100.0) is True


def test_host_health_serves_only_healthz(tmp_path: Path) -> None:
    path = tmp_path / "runner.json"
    write_runner_status(path, _status(100.0))
    from http.server import ThreadingHTTPServer

    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(path, 10.0, lambda: 100.0))
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        host, port = server.server_address
        with urllib.request.urlopen(f"http://{host}:{port}/healthz") as response:
            assert response.status == 200
            assert response.headers["Cache-Control"] == "no-store"
        try:
            urllib.request.urlopen(f"http://{host}:{port}/other")
        except urllib.error.HTTPError as error:
            assert error.code == 404
        else:
            raise AssertionError("unknown health path succeeded")
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_gce_shell_scripts_parse_and_startup_has_required_boundaries() -> None:
    paths = (_DEPLOY / "gce-startup.sh", _DEPLOY / "gce-up.sh", _DEPLOY / "gce-down.sh")
    subprocess.run(["bash", "-n", *paths], check=True)
    source = paths[0].read_text()
    for text in (
        "gcloud secrets versions access latest",
        "chmod 0600",
        "nvidia-smi",
        'docker pull "$image"',
        "--gpus all --ipc=host",
        "APPIMAGE_EXTRACT_AND_RUN=1",
        "compute,graphics,utility,video",
        "--graphics-backend OGL",
        "hal-netplay-runner.service",
        "hal-netplay-health.service",
        "/var/cache/hal-netplay:/root/.cache/hal-netplay",
        "/var/lib/hal-netplay:/var/lib/hal-netplay",
    ):
        assert text in source


@pytest.mark.parametrize("docker_installed", [False, True])
def test_startup_installs_missing_docker_and_configures_gpu_runtime(tmp_path: Path, docker_installed: bool) -> None:
    commands = tmp_path / "bin"
    commands.mkdir()
    log = tmp_path / "startup.log"

    def command(name: str, body: str) -> Path:
        path = commands / name
        path.write_text(f'#!/bin/bash\nprintf "%s\\n" "{name} $*" >> "$STARTUP_LOG"\n{body}')
        path.chmod(0o755)
        return path

    command(
        "curl",
        'case "${@: -1}" in\n'
        '*/hal-netplay-git-sha) printf "%s\\n" "$STARTUP_SHA" ;;\n'
        "*/hal-netplay-slots) echo 16 ;;\n"
        "*/hal-netplay-drain-timeout) echo 900 ;;\n"
        "*) echo test ;;\nesac\n",
    )
    docker = command("docker", "exit 0\n")
    template = tmp_path / "docker-template"
    template.write_bytes(docker.read_bytes())
    if not docker_installed:
        docker.unlink()
    command(
        "apt-get",
        'if [[ $1 == install ]]; then /bin/cp "$DOCKER_TEMPLATE" "$MOCK_BIN/docker"; '
        '/bin/chmod +x "$MOCK_BIN/docker"; fi\n',
    )
    for name in ("nvidia-ctk", "nvidia-smi", "systemctl"):
        command(name, "exit 0\n")
    # Stop before filesystem, display, or credential setup on the test host.
    command("install", "exit 91\n")
    environment = os.environ | {
        "PATH": str(commands),
        "STARTUP_LOG": str(log),
        "STARTUP_SHA": _SHA,
        "DOCKER_TEMPLATE": str(template),
        "MOCK_BIN": str(commands),
    }
    result = subprocess.run(
        ["/bin/bash", str(_DEPLOY / "gce-startup.sh")], env=environment, capture_output=True, text=True
    )
    assert result.returncode == 91, result.stderr
    calls = log.read_text().splitlines()
    installs = [call for call in calls if call.startswith("apt-get ")]
    assert installs == (
        [] if docker_installed else ["apt-get update", "apt-get install -y --no-install-recommends docker.io"]
    )
    configure = calls.index("nvidia-ctk runtime configure --runtime=docker")
    restart = calls.index("systemctl restart docker")
    assert configure < restart < calls.index("docker info") < calls.index("nvidia-smi ")


def test_gce_up_renders_standalone_g4_command(tmp_path: Path) -> None:
    log, environment = _stub_gcloud(tmp_path)
    subprocess.run(_up_args(), check=True, env=environment, capture_output=True, text=True)
    command = log.read_text()
    assert "compute instances create hal-netplay-test" in command
    assert "--machine-type=g4-standard-48" in command
    assert "--image-family=common-cu129-ubuntu-2404-nvidia-580" in command
    assert "--boot-disk-type=hyperdisk-balanced" in command
    assert "--maintenance-policy=TERMINATE --restart-on-failure" in command
    assert "--scopes=cloud-platform" in command
    assert "--service-account=runner@hal-project.iam.gserviceaccount.com" in command
    assert "hal-netplay-secret=hal-netplay-runner-env" in command
    assert "startup-script=" in command


@pytest.mark.parametrize("slots", [1, 8, 9, 16])
def test_gce_up_accepts_capacity_through_sixteen(tmp_path: Path, slots: int) -> None:
    log, environment = _stub_gcloud(tmp_path)
    subprocess.run(_up_args("--slots", str(slots)), check=True, env=environment, capture_output=True, text=True)
    assert f"hal-netplay-slots={slots}," in log.read_text()


@pytest.mark.parametrize("slots", ["0", "17", "-1", "1.5", "08"])
def test_gce_up_rejects_unsupported_capacity_before_creating_resources(tmp_path: Path, slots: str) -> None:
    log, environment = _stub_gcloud(tmp_path)
    result = subprocess.run(_up_args("--slots", slots), env=environment, capture_output=True, text=True)
    assert result.returncode == 2
    assert "--slots must be between 1 and 16" in result.stderr
    assert not log.exists()


def test_gce_down_drains_before_delete(tmp_path: Path) -> None:
    log, environment = _stub_gcloud(tmp_path)
    subprocess.run(
        [str(_DEPLOY / "gce-down.sh"), "hal-netplay-test", "--project", "hal-project"],
        check=True,
        env=environment,
    )
    commands = log.read_text().splitlines()
    assert "compute ssh hal-netplay-test" in commands[0]
    assert "systemctl stop hal-netplay-runner.service" in commands[0]
    assert "compute instances delete hal-netplay-test" in commands[1]


def test_gce_down_refuses_delete_after_failed_drain_without_force(tmp_path: Path) -> None:
    log, environment = _stub_gcloud(tmp_path, fail_ssh=True)
    result = subprocess.run(
        [str(_DEPLOY / "gce-down.sh"), "hal-netplay-test", "--project", "hal-project"],
        check=False,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "refusing deletion" in result.stderr
    assert "compute instances delete" not in log.read_text()


def test_gce_managed_up_and_down(tmp_path: Path) -> None:
    log, environment = _stub_gcloud(tmp_path, instances="zones/us-central1-a/instances/hal-netplay-test-abcd")
    subprocess.run(_up_args("--managed"), check=True, env=environment, capture_output=True, text=True)
    up = log.read_text()
    assert "compute health-checks create http hal-netplay-test-health" in up
    assert "compute firewall-rules create hal-netplay-test-health" in up
    assert "compute instance-templates create hal-netplay-test-template" in up
    assert "compute instance-groups managed create hal-netplay-test" in up
    assert "--size=1" in up
    assert "--initial-delay=2400" in up

    log.write_text("")
    subprocess.run(
        [str(_DEPLOY / "gce-down.sh"), "hal-netplay-test", "--project", "hal-project", "--managed"],
        check=True,
        env=environment,
    )
    down = log.read_text()
    assert "compute ssh hal-netplay-test-abcd" in down
    assert "managed resize hal-netplay-test" in down
    assert "--size=0" in down
    assert "managed delete hal-netplay-test" in down
    assert "instance-templates delete hal-netplay-test-template" in down
