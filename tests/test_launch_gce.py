import base64
import importlib.util
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
        machine_type="g4-standard-96",
        keep_alive=False,
    )
    assert base64.b64encode(b"WANDB_API_KEY=wandb-key\n").decode() in script
    assert "secret-value" not in script
    assert base64.b64encode(b"python train.py --name 'two words'").decode() in script
    assert "export HAL_KEEP_ALIVE=0" in script
    assert "apt-get install --yes --no-install-recommends docker.io" in script
    assert "nvidia-ctk runtime configure --runtime=docker" in script
    assert "export HAL_NCCL_P2P_LEVEL=PHB" in script
    assert '-e "NCCL_P2P_LEVEL=${HAL_NCCL_P2P_LEVEL}"' in script


def test_startup_script_preserves_legacy_nccl_defaults() -> None:
    script = startup_script(
        sha="abc123",
        train_cmd="python train.py",
        project="my-project",
        secrets=[],
        image="example/image:tag",
        machine_type="a2-highgpu-1g",
        keep_alive=False,
    )
    assert "export HAL_NCCL_P2P_LEVEL=''" in script


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
