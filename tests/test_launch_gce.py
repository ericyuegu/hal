import base64
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "hal_launch_gce", Path(__file__).parents[1] / "scripts" / "launch_gce.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

Args = _MODULE.Args
create_command = _MODULE.create_command
parse_secrets = _MODULE.parse_secrets
startup_script = _MODULE.startup_script


def test_parse_secrets_rejects_shell_unsafe_values() -> None:
    assert parse_secrets(["WANDB_API_KEY=wandb-key"]) == [("WANDB_API_KEY", "wandb-key")]
    with pytest.raises(ValueError):
        parse_secrets(["BAD-NAME=secret"])
    with pytest.raises(ValueError):
        parse_secrets(["SAFE=$(touch nope)"])


def test_startup_script_contains_no_secret_values() -> None:
    script = startup_script(
        sha="abc123",
        train_cmd="python train.py --name 'two words'",
        project="my-project",
        secrets=[("WANDB_API_KEY", "wandb-key")],
        image="example/image:tag",
        keep_alive=False,
    )
    assert base64.b64encode(b"WANDB_API_KEY=wandb-key\n").decode() in script
    assert "secret-value" not in script
    assert base64.b64encode(b"python train.py --name 'two words'").decode() in script
    assert "export HAL_KEEP_ALIVE=0" in script
    assert "apt-get install --yes --no-install-recommends docker.io" in script
    assert "nvidia-ctk runtime configure --runtime=docker" in script


def test_startup_secret_records_include_the_final_credential() -> None:
    secrets = parse_secrets(_MODULE.DEFAULT_SECRETS)
    script = startup_script(
        sha="a" * 40,
        train_cmd="true",
        project="project",
        secrets=secrets,
        image="example/image:tag",
        keep_alive=False,
    )
    line = next(line for line in script.splitlines() if line.startswith("export HAL_SECRET_SPECS_B64="))
    records = base64.b64decode(line.split("=", 1)[1]).decode()
    result = subprocess.run(
        ["bash", "-c", 'while IFS="=" read -r env_name secret_id; do printf "%s\\n" "$env_name"; done'],
        input=records,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.splitlines() == [name for name, _ in secrets]
    assert result.stdout.splitlines()[-1] == "WANDB_API_KEY"


def test_create_command_spot_keyless_service_account() -> None:
    args = Args(cmd=["python", "train.py"], service_account="jobs@example.iam.gserviceaccount.com")
    command = create_command(args, project="project", name="hal-job", startup_file="/tmp/start.sh")
    assert "--provisioning-model=SPOT" in command
    assert "--instance-termination-action=DELETE" in command
    assert "--no-restart-on-failure" in command
    assert "--scopes=cloud-platform" in command
    assert "--service-account=jobs@example.iam.gserviceaccount.com" in command
    assert all("secret-value" not in argument for argument in command)


def test_create_command_g4_uses_integrated_gpus_and_hyperdisk() -> None:
    args = Args(
        cmd=["torchrun", "train.py"],
        zone="us-central1-b",
        machine_type="g4-standard-96",
        gpu_count=2,
        disk=3000,
        disk_type="hyperdisk-balanced",
        spot=False,
    )
    command = create_command(args, project="project", name="hal-o60", startup_file="/tmp/start.sh")
    assert "--machine-type=g4-standard-96" in command
    assert "--boot-disk-size=3000GB" in command
    assert "--boot-disk-type=hyperdisk-balanced" in command
    assert not any(argument.startswith("--accelerator=") for argument in command)
    assert "--provisioning-model=SPOT" not in command


def test_create_command_rejects_g4_gpu_count_mismatch() -> None:
    args = Args(cmd=["python", "train.py"], machine_type="g4-standard-96", gpu_count=1)
    with pytest.raises(ValueError, match="has 2 integrated GPUs"):
        create_command(args, project="project", name="hal-o60", startup_file="/tmp/start.sh")


def test_dry_run_of_unpublished_commit_needs_no_cloud_cli(tmp_path: Path) -> None:
    for command in (
        ["git", "init", "--quiet"],
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "--quiet",
            "--allow-empty",
            "-m",
            "Initial",
        ],
    ):
        subprocess.run(command, cwd=tmp_path, check=True)
    result = subprocess.run(
        [
            sys.executable,
            str(_MODULE.ROOT / "scripts/launch_gce.py"),
            "--dry-run",
            "--project",
            "project",
            "--max-run-hours",
            "24",
            "--no-spot",
            "--",
            "true",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--max-run-duration=24h" in result.stdout
    assert "--instance-termination-action=STOP" in result.stdout
    assert "remote SHA, auth, image access, quota and secrets remain unchecked" in result.stderr
    assert "pushing" not in result.stderr


@pytest.mark.parametrize("hours", [0, -1, 2881])
def test_invalid_wall_clock_cap(hours: int) -> None:
    with pytest.raises(ValueError, match="max run hours"):
        create_command(Args(max_run_hours=hours), project="project", name="job", startup_file="start.sh")


def test_startup_preserves_results_and_supports_targeted_fixtures() -> None:
    script = startup_script(
        sha="a" * 40,
        train_cmd="true",
        project="project",
        secrets=[],
        image="image",
        keep_alive=False,
        skip_fixtures=True,
    )
    assert "export HAL_SKIP_FIXTURES=1" in script
    assert "-v /var/lib/hal:/hal-output" in script
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    runtime = script.split("<<'HAL_RUNTIME'\n", 1)[1].split("\nHAL_RUNTIME", 1)[0]
    subprocess.run(["bash", "-n"], input=runtime, text=True, check=True)


@pytest.mark.parametrize("stage,code", [("ok", 0), ("training", 3), ("pull", 4), ("upload", 1), ("late-term", 0)])
def test_vm_lifecycle_uploads_before_shutdown(tmp_path: Path, stage: str, code: int) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    events = tmp_path / "events"
    for command, body in {
        "gcloud": 'printf "test-credential"\n',
        "nvidia-smi": "true\n",
        "systemd-run": 'echo "timer $*" >> "$EVENTS"\n',
        "systemctl": (
            'echo "cancel $*" >> "$EVENTS"\nif [[ "$FAIL_STAGE" == late-term ]]; then kill -TERM "$PPID"; fi\n'
        ),
        "shutdown": 'echo shutdown >> "$EVENTS"\n',
        "docker": (
            'echo "docker $*" >> "$EVENTS"\n'
            'if [[ "$1" == pull && "$FAIL_STAGE" == pull ]]; then exit 4; fi\n'
            'if [[ "$1" == run ]]; then\n'
            ' if [[ "$*" == *results.py* ]]; then\n'
            '  echo upload >> "$EVENTS"\n'
            '  if [[ "$FAIL_STAGE" == upload ]]; then exit 5; fi\n'
            ' elif [[ "$FAIL_STAGE" == training ]]; then exit 3; fi\n'
            "fi\n"
        ),
    }.items():
        path = bin_dir / command
        path.write_text("#!/bin/bash\n" + body)
        path.chmod(0o755)
    for directory in ("run", "output", "log"):
        (tmp_path / directory).mkdir()
    script = startup_script(
        sha="a" * 40,
        train_cmd="true",
        project="project",
        secrets=[("AWS_BUCKET", "bucket")],
        image="test/image@sha256:" + "b" * 64,
        keep_alive=False,
        max_run_hours=24,
        results_prefix="o65/pilot-results/test",
    )
    script = script.replace("/run/hal", str(tmp_path / "run"))
    script = script.replace("/var/lib/hal", str(tmp_path / "output"))
    script = script.replace("/var/log", str(tmp_path / "log"))
    result = subprocess.run(
        ["bash"],
        input=script,
        text=True,
        capture_output=True,
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "EVENTS": str(events), "FAIL_STAGE": stage},
    )
    assert result.returncode == code, result.stdout + result.stderr
    lines = events.read_text().splitlines()
    assert any("hal-hard-stop --on-boot=86400s" in line for line in lines)
    assert any("hal-job-deadline --on-boot=85500s" in line for line in lines)
    assert lines.index("upload") < lines.index("shutdown")
    assert "test-credential" not in result.stdout + result.stderr
    assert not (tmp_path / "run/job.env").exists()
    assert (tmp_path / "output/upload-exit-code").read_text().strip() == ("5" if stage == "upload" else "0")


@pytest.mark.parametrize("prefix", ["elsewhere/run", "o65/pilot-results/../run", "o65/pilot-results/a//b"])
def test_results_prefix_is_restricted(prefix: str) -> None:
    with pytest.raises(ValueError, match="results upload"):
        create_command(Args(results_prefix=prefix, max_run_hours=24), project="project", name="job", startup_file="s")


def test_upload_requires_cap_and_auto_shutdown() -> None:
    for args in (
        Args(results_prefix="o65/pilot-results/job"),
        Args(results_prefix="o65/pilot-results/job", max_run_hours=24, keep_alive=True),
    ):
        with pytest.raises(ValueError, match="wall-clock cap, and auto-shutdown"):
            create_command(args, project="project", name="job", startup_file="s")
