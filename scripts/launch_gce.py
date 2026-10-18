"""Launch a HAL training command on a Google Compute Engine GPU VM.

Authenticate gcloud locally and grant the VM service account access to the
declared Secret Manager entries. The VM reads R2 and W&B credentials through
that account. Instance metadata contains secret identifiers.

Use --dry-run with --project to check local inputs and print the create
request. Publish the source commit before a real launch. Set --max-run-hours
to bound VM lifetime. Read startup output with gcloud compute instances
get-serial-port-output. After training, the VM shuts down and retains its
boot disk for inspection.
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
from functools import partial
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
_G4_GPU_COUNTS = {
    "g4-standard-48": 1,
    "g4-standard-96": 2,
    "g4-standard-192": 4,
    "g4-standard-384": 8,
}


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
    *,
    sha: str,
    train_cmd: str,
    project: str,
    secrets: list[tuple[str, str]],
    image: str,
    keep_alive: bool,
    skip_fixtures: bool = False,
) -> str:
    """Render startup metadata with secret identifiers."""
    # Bash read skips an unterminated final record; the W&B key is last by default.
    secret_specs = "".join(f"{env_name}={secret_id}\n" for env_name, secret_id in secrets)
    values = {
        "HAL_GIT_SHA": sha,
        "HAL_TRAIN_CMD_B64": base64.b64encode(train_cmd.encode()).decode(),
        "HAL_GCP_PROJECT": project,
        "HAL_SECRET_SPECS_B64": base64.b64encode(secret_specs.encode()).decode(),
        "HAL_IMAGE": image,
        "HAL_KEEP_ALIVE": "1" if keep_alive else "0",
        "HAL_SKIP_FIXTURES": "1" if skip_fixtures else "0",
    }
    exports = "\n".join(f"export {name}={shlex.quote(value)}" for name, value in values.items())
    return f"#!/usr/bin/env bash\n{exports}\n\n{STARTUP_PATH.read_text()}"


def create_command(args: Args, *, project: str, name: str, startup_file: str) -> list[str]:
    if args.max_run_hours is not None and not 1 <= args.max_run_hours <= 2880:
        raise ValueError("max run hours must be between 1 and 2880")
    if args.machine_type.startswith("g4-"):
        expected_gpus = _G4_GPU_COUNTS.get(args.machine_type)
        if expected_gpus is None:
            raise ValueError(f"unsupported G4 machine type {args.machine_type!r}")
        if args.gpu_count != expected_gpus:
            raise ValueError(
                f"{args.machine_type} has {expected_gpus} integrated GPUs; got --gpu-count={args.gpu_count}"
            )
    elif args.gpu_count < 1:
        raise ValueError("attached-GPU machine types require a positive --gpu-count")
    command = [
        "gcloud",
        "compute",
        "instances",
        "create",
        name,
        f"--project={project}",
        f"--zone={args.zone}",
        f"--machine-type={args.machine_type}",
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
    if args.machine_type not in _G4_GPU_COUNTS:
        command.insert(7, f"--accelerator=type={args.accelerator},count={args.gpu_count}")
    if args.service_account:
        command.append(f"--service-account={args.service_account}")
    if args.spot:
        command.extend(("--provisioning-model=SPOT", "--instance-termination-action=DELETE"))
    if args.max_run_hours is not None:
        command.append(f"--max-run-duration={args.max_run_hours}h")
        if not args.spot:
            command.append("--instance-termination-action=STOP")
    if args.no_external_ip:
        command.append("--no-address")
    return command


def _preflight(args: Args, secrets: list[tuple[str, str]]) -> tuple[str, str]:
    if _git("status", "--porcelain"):
        raise SystemExit("working tree is dirty; commit before launching (the VM runs the published SHA).")
    sha = _git("rev-parse", "HEAD")
    if args.dry_run:
        if not args.project:
            raise SystemExit("--dry-run requires --project; it performs local checks only.")
        logger.info(
            "[dry-run] local rendering only; remote SHA, auth, image access, quota and secrets remain unchecked"
        )
        return sha, args.project
    if shutil.which("gcloud") is None:
        raise SystemExit("gcloud is not installed; install the Google Cloud CLI, then run `gcloud auth login`.")
    account = _run_gcloud("auth", "list", "--filter=status:ACTIVE", "--format=value(account)")
    if not account:
        raise SystemExit("no active gcloud account; run `gcloud auth login` (no key file is needed).")
    project = args.project or _run_gcloud("config", "get-value", "project")
    if not project or project == "(unset)":
        raise SystemExit("no project selected; pass --project or run `gcloud config set project PROJECT_ID`.")

    # Query GitHub to check publication; local tracking refs can be stale.
    _git("fetch", "--quiet", "--prune", "https://github.com/ericyuegu/hal.git", "+refs/heads/*:refs/hal-launch/*")
    published = _git("for-each-ref", "--contains", sha, "--format=%(refname)", "refs/hal-launch/")
    if not published:
        raise SystemExit(f"SHA {sha} is unpublished; publish it explicitly before launching.")

    for _env_name, secret_id in secrets:
        try:
            _run_gcloud("secrets", "describe", secret_id, f"--project={project}", "--format=value(name)")
        except subprocess.CalledProcessError as exc:
            raise SystemExit(
                f"Secret Manager secret {secret_id!r} is missing or inaccessible in {project}. "
                "Create it and grant the VM service account roles/secretmanager.secretAccessor."
            ) from exc
    return sha, project


@dataclass(frozen=True, slots=True)
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
    secret: list[str] = field(default_factory=partial(list, DEFAULT_SECRETS))
    """Secret mapping ENV_NAME=secret-id. Repeat to replace the defaults."""
    spot: bool = True
    """Use a cheaper interruptible Spot VM; preemption deletes the instance and boot disk."""
    no_external_ip: bool = False
    """Do not assign an external IP. The subnet must provide NAT for GitHub, GHCR, R2, and W&B."""
    keep_alive: bool = False
    """Leave the VM running after training for debugging instead of shutting it down."""
    dry_run: bool = False
    """Check local inputs and print the create request."""
    max_run_hours: int | None = None
    """Stop an on-demand VM at this wall-clock limit, even if startup or training hangs."""
    skip_fixtures: bool = False
    """Let the job fetch its specific fixtures instead of fetching every development fixture."""


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
        sha=sha,
        train_cmd=train_cmd,
        project=project,
        secrets=secrets,
        image=args.image,
        keep_alive=args.keep_alive,
        skip_fixtures=args.skip_fixtures,
    )

    # gcloud requires a file for a multiline startup script. The file only contains
    # non-secret identifiers and is removed as soon as create returns.
    with tempfile.NamedTemporaryFile("w", prefix="hal-gce-startup-", suffix=".sh") as startup:
        startup.write(rendered)
        startup.flush()
        try:
            command = create_command(args, project=project, name=name, startup_file=startup.name)
        except ValueError as error:
            raise SystemExit(str(error)) from error
        if args.dry_run:
            printable = ["<startup-script>" if item.startswith("--metadata-from-file=") else item for item in command]
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
