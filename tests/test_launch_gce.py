import base64
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).parents[1]
_SPEC = importlib.util.spec_from_file_location("hal_launch_gce", _ROOT / "scripts" / "launch_gce.py")
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

Args = _MODULE.Args
create_command = _MODULE.create_command
parse_secrets = _MODULE.parse_secrets
startup_script = _MODULE.startup_script
validate_shape = _MODULE.validate_shape

_SECRETS = [("AWS_ENDPOINT_URL", "endpoint"), ("AWS_BUCKET", "bucket"), ("WANDB_API_KEY", "wandb-key")]


def _render(**overrides: object) -> str:
    kwargs: dict[str, object] = {
        "sha": "abc123",
        "train_cmd": "python train.py --name 'two words'",
        "project": "my-project",
        "secrets": _SECRETS,
        "image": "example/image:tag",
        "keep_alive": False,
        "local_ssd_count": 1,
    }
    return startup_script(**{**kwargs, **overrides})  # type: ignore[arg-type]


def test_parse_secrets_rejects_shell_unsafe_values() -> None:
    assert parse_secrets(["WANDB_API_KEY=wandb-key"]) == [("WANDB_API_KEY", "wandb-key")]
    with pytest.raises(ValueError):
        parse_secrets(["BAD-NAME=secret"])
    with pytest.raises(ValueError):
        parse_secrets(["SAFE=$(touch nope)"])


def test_startup_script_contains_no_secret_values() -> None:
    script = _render()
    assert (
        base64.b64encode(b"AWS_ENDPOINT_URL=endpoint\nAWS_BUCKET=bucket\nWANDB_API_KEY=wandb-key\n").decode() in script
    )
    assert "secret-value" not in script
    assert base64.b64encode(b"python train.py --name 'two words'").decode() in script
    assert "export HAL_KEEP_ALIVE=0" in script
    assert "export HAL_LOCAL_SSD_COUNT=1" in script


def test_every_secret_survives_an_unguarded_boot_read_loop() -> None:
    """`while read` returns non-zero on a final line with no trailing newline and skips it,
    so a newline-*separated* spec blob silently dropped the last secret (WANDB_API_KEY) on
    every launch. Feed the real blob through the bare loop — no `|| [ -n ...]` rescue — so
    this fails if the launcher ever stops terminating its lines."""
    blob = re.search(r"export HAL_SECRET_SPECS_B64='?([A-Za-z0-9+/=]+)'?", _render()).group(1)  # type: ignore[union-attr]
    loop = 'while IFS="=" read -r name _; do [ -n "$name" ] || continue; echo "$name"; done < <(printf "%s" "$1" | base64 -d)'
    result = subprocess.run(["bash", "-c", loop, "_", blob], check=True, capture_output=True, text=True)
    assert result.stdout.split() == [name for name, _ in _SECRETS]


def test_boot_script_tolerates_an_unterminated_final_spec() -> None:
    source = (_ROOT / "docker" / "on-start-gce.sh").read_text()
    assert 'read -r env_name secret_id || [ -n "$env_name" ]' in source


def test_create_command_spot_keyless_service_account() -> None:
    args = Args(cmd=["python", "train.py"], service_account="jobs@example.iam.gserviceaccount.com")
    command = create_command(args, project="project", name="hal-job", startup_file="/tmp/start.sh")
    assert "--provisioning-model=SPOT" in command
    assert "--instance-termination-action=DELETE" in command
    assert "--no-restart-on-failure" in command
    assert "--scopes=cloud-platform" in command
    assert "--service-account=jobs@example.iam.gserviceaccount.com" in command
    assert all("secret-value" not in argument for argument in command)


def test_create_command_omits_accelerator_for_gpu_bundled_machine_types() -> None:
    """a2/a3/a4/g2/g4 carry their GPUs in the machine type and the API rejects --accelerator,
    so the flag being unconditional made every L4/A100 shape uncreatable."""
    bundled = create_command(
        Args(cmd=["python", "train.py"], machine_type="g2-standard-32", accelerator="", local_ssd_count=2),
        project="project",
        name="hal-job",
        startup_file="/tmp/start.sh",
    )
    assert not any(argument.startswith("--accelerator") for argument in bundled)
    assert bundled.count("--local-ssd=interface=NVME") == 2

    explicit = create_command(
        Args(cmd=["python", "train.py"], machine_type="n1-standard-8", accelerator="nvidia-tesla-t4"),
        project="project",
        name="hal-job",
        startup_file="/tmp/start.sh",
    )
    assert "--accelerator=type=nvidia-tesla-t4,count=1" in explicit


def test_validate_shape_rejects_both_mismatches() -> None:
    validate_shape("g2-standard-32", "")
    validate_shape("n1-standard-8", "nvidia-tesla-t4")
    with pytest.raises(SystemExit):
        validate_shape("g2-standard-32", "nvidia-l4")
    with pytest.raises(SystemExit):
        validate_shape("n1-standard-8", "")
