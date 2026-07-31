"""Launch one HAL training command on a Google Compute Engine GPU VM.

This is a fire-and-forget counterpart to ``launch_vast.py``.  It uses the
locally authenticated ``gcloud`` CLI, so a human can launch with
``gcloud auth login``; no service-account key file is required.  The VM reads
R2/W&B credentials from Secret Manager through its attached service account.
Secret *values* are never placed in instance metadata.

One-time setup (an administrator may need to do the IAM steps)::

    gcloud auth login
    gcloud config set project MY_PROJECT
    # Create one Secret Manager secret for each default --secret below, then grant
    # the VM service account roles/secretmanager.secretAccessor on those secrets.

Launch examples::

    uv run scripts/launch_gce.py --dry-run --zone us-central1-a -- \
        uv run experiments/001_flow_matching_baseline.py
    uv run scripts/launch_gce.py --zone us-central1-a --service-account hal-jobs@MY_PROJECT.iam.gserviceaccount.com -- \
        uv run experiments/001_flow_matching_baseline.py --cfg.max-steps 100000

The startup log is available with ``gcloud compute instances get-serial-port-output``.
On completion the VM shuts down, stopping compute charges but retaining its boot
disk for inspection. Delete the instance when it is no longer needed.
"""

from __future__ import annotations

import base64
import re
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import datetime
from pathlib import Path

import tyro
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
STARTUP_PATH = ROOT / "docker" / "on-start-gce.sh"
DEFAULT_SECRETS = (
    "AWS_ENDPOINT_URL=AWS_ENDPOINT_URL",
    "AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY",
    "AWS_BUCKET=AWS_BUCKET",
    "WANDB_API_KEY=WANDB_API_KEY",
)
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RESOURCE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


def _run_gcloud(*args: str, capture: bool = True) -> str:
    command = ["gcloud", *args, "--quiet"]
    result = subprocess.run(command, check=True, capture_output=capture, text=True)
    return result.stdout.strip() if capture else ""


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def parse_secrets(specs: list[str] | tuple[str, ...]) -> list[tuple[str, str]]:
    """Parse ENV=secret-id mappings, rejecting strings unsafe for the boot shell."""
    parsed: list[tuple[str, str]] = []
    seen: set[str] = set()
    for spec in specs:
        env_name, separator, secret_id = spec.partition("=")
        if not separator or not _ENV_NAME.fullmatch(env_name) or not _RESOURCE_NAME.fullmatch(secret_id):
            raise ValueError(f"invalid secret mapping {spec!r}; expected ENV_NAME=secret-id")
        if env_name in seen:
            raise ValueError(f"duplicate secret environment name {env_name!r}")
        seen.add(env_name)
        parsed.append((env_name, secret_id))
    return parsed


def startup_script(
    *, sha: str, train_cmd: str, project: str, secrets: list[tuple[str, str]], image: str, keep_alive: bool
) -> str:
    """Render the metadata startup script. It contains secret names, never values."""
    secret_specs = "\n".join(f"{env_name}={secret_id}" for env_name, secret_id in secrets)
    values = {
        "HAL_GIT_SHA": sha,
        "HAL_TRAIN_CMD_B64": base64.b64encode(train_cmd.encode()).decode(),
        "HAL_GCP_PROJECT": project,
        "HAL_SECRET_SPECS_B64": base64.b64encode(secret_specs.encode()).decode(),
        "HAL_IMAGE": image,
        "HAL_KEEP_ALIVE": "1" if keep_alive else "0",
    }
    exports = "\n".join(f"export {name}={shlex.quote(value)}" for name, value in values.items())
    return f"#!/usr/bin/env bash\n{exports}\n\n{STARTUP_PATH.read_text()}"


def create_command(args: Args, *, project: str, name: str, startup_file: str) -> list[str]:
    command = [
        "gcloud",
        "compute",
        "instances",
        "create",
        name,
        f"--project={project}",
        f"--zone={args.zone}",
        f"--machine-type={args.machine_type}",
        f"--accelerator=type={args.accelerator},count={args.gpu_count}",
        "--maintenance-policy=TERMINATE",
        "--no-restart-on-failure",
        f"--image-family={args.image_family}",
        "--image-project=deeplearning-platform-release",
        f"--boot-disk-size={args.disk}GB",
        f"--boot-disk-type={args.disk_type}",
        "--scopes=cloud-platform",
        f"--metadata-from-file=startup-script={startup_file}",
        "--labels=app=hal,workload=training",
        "--quiet",
    ]
    if args.service_account:
        command.append(f"--service-account={args.service_account}")
    if args.spot:
        command.extend(("--provisioning-model=SPOT", "--instance-termination-action=DELETE"))
    if args.no_external_ip:
        command.append("--no-address")
    return command


def _preflight(args: Args, secrets: list[tuple[str, str]]) -> tuple[str, str]:
    if shutil.which("gcloud") is None:
        raise SystemExit("gcloud is not installed; install the Google Cloud CLI, then run `gcloud auth login`.")
    account = _run_gcloud("auth", "list", "--filter=status:ACTIVE", "--format=value(account)")
    if not account:
        raise SystemExit("no active gcloud account; run `gcloud auth login` (no key file is needed).")
    project = args.project or _run_gcloud("config", "get-value", "project")
    if not project or project == "(unset)":
        raise SystemExit("no project selected; pass --project or run `gcloud config set project PROJECT_ID`.")

    if _git("status", "--porcelain"):
        raise SystemExit("working tree is dirty — commit before launching (the VM runs the pushed SHA).")
    sha = _git("rev-parse", "HEAD")
    if not _git("branch", "-r", "--contains", sha):
        branch = _git("rev-parse", "--abbrev-ref", "HEAD")
        logger.info(f"{sha[:10]} not on a remote branch; pushing {branch}")
        subprocess.run(["git", "push", "origin", branch], check=True)

    for _env_name, secret_id in secrets:
        try:
            _run_gcloud("secrets", "describe", secret_id, f"--project={project}", "--format=value(name)")
        except subprocess.CalledProcessError as exc:
            raise SystemExit(
                f"Secret Manager secret {secret_id!r} is missing or inaccessible in {project}. "
                "Create it and grant the VM service account roles/secretmanager.secretAccessor."
            ) from exc
    return sha, project


@dataclass(frozen=True)
class Args:
    cmd: tyro.conf.Positional[list[str]] = field(default_factory=list)
    """Training command after `--`."""
    project: str | None = None
    """GCP project ID. Defaults to the active gcloud project."""
    zone: str = "us-central1-a"
    """Compute zone. GPU models and quota are zone-specific."""
    name: str | None = None
    """Instance name. Defaults to hal-<UTC timestamp>-<git SHA>."""
    machine_type: str = "n1-standard-8"
    """Machine type hosting the attached GPU."""
    accelerator: str = "nvidia-tesla-t4"
    """GCE accelerator type, for example nvidia-tesla-t4 or nvidia-tesla-a100."""
    gpu_count: int = 1
    """Number of GPUs."""
    disk: int = 500
    """Boot disk size in GB; retained when a normally completed VM shuts down."""
    disk_type: str = "pd-balanced"
    """Boot disk type."""
    image_family: str = "common-cu129-ubuntu-2404-nvidia-580"
    """Deep Learning VM image family with an NVIDIA driver."""
    image: str = "ghcr.io/ericyuegu/hal:cuda13"
    """Training container image."""
    service_account: str | None = None
    """Keyless VM identity. It needs Secret Accessor on every --secret; defaults to the project's Compute SA."""
    secret: list[str] = field(default_factory=lambda: list(DEFAULT_SECRETS))
    """Secret mapping ENV_NAME=secret-id. Repeat to replace the defaults."""
    spot: bool = True
    """Use a cheaper interruptible Spot VM; preemption deletes the instance and boot disk."""
    no_external_ip: bool = False
    """Do not assign an external IP. The subnet must provide NAT for GitHub, GHCR, R2, and W&B."""
    keep_alive: bool = False
    """Leave the VM running after training for debugging instead of shutting it down."""
    dry_run: bool = False
    """Run local/auth/git/secret preflight and print the redacted create request without creating a VM."""


def main(args: Args) -> None:
    if not args.cmd:
        raise SystemExit("pass a training command after `--` (use --dry-run to validate without creating a VM).")
    try:
        secrets = parse_secrets(args.secret)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    sha, project = _preflight(args, secrets)
    train_cmd = shlex.join(args.cmd)
    name = args.name or f"hal-{datetime.now(UTC):%Y%m%d-%H%M%S}-{sha[:7]}"
    rendered = startup_script(
        sha=sha, train_cmd=train_cmd, project=project, secrets=secrets, image=args.image, keep_alive=args.keep_alive
    )

    # gcloud requires a file for a multiline startup script. The file only contains
    # non-secret identifiers and is removed as soon as create returns.
    with tempfile.NamedTemporaryFile("w", prefix="hal-gce-startup-", suffix=".sh") as startup:
        startup.write(rendered)
        startup.flush()
        command = create_command(args, project=project, name=name, startup_file=startup.name)
        if args.dry_run:
            printable = ["<startup-script>" if item.startswith("--metadata-from-file=") else item for item in command]
            logger.info(
                f"[dry-run] authenticated as {_run_gcloud('auth', 'list', '--filter=status:ACTIVE', '--format=value(account)')}"
            )
            logger.info(f"[dry-run] secret mappings (names only): {dict(secrets)}")
            logger.info(f"[dry-run] startup script: {len(rendered)} bytes; SHA={sha}; command={train_cmd}")
            print(shlex.join(printable))
            return
        subprocess.run(command, check=True)

    logger.success(f"created {name} in {args.zone} on SHA {sha[:10]}")
    logger.info(f"logs: gcloud compute instances get-serial-port-output {name} --zone={args.zone} --project={project}")
    logger.info(f"ssh:  gcloud compute ssh {name} --zone={args.zone} --project={project}")
    if args.keep_alive:
        logger.warning("--keep-alive is set: the VM will continue billing until stopped or deleted manually.")
    else:
        logger.info(
            f"the VM shuts down after training; delete its retained disk with: "
            f"gcloud compute instances delete {name} --zone={args.zone} --project={project}"
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
