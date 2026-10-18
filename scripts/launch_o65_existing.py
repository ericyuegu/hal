"""Copy the fixed O65 pilot to the lent VM and run it under systemd."""

import base64
import dataclasses
import re
import shlex
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Literal

import tyro

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "centering-star-502613-k3"
ZONE = "us-west4-a"
INSTANCE = "hal-netplay-g4-1015-us-west4-a"
SECRETS = ("AWS_ENDPOINT_URL", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_BUCKET", "WANDB_API_KEY")
IMAGE = (
    "us-west1-docker.pkg.dev/centering-star-502613-k3/hal-netplay/hal-netplay-runner"
    "@sha256:93b281d12c0614516eca8b5a508dae2723da80e53933e9616bb08d6e926f3016"
)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def validate_name(name: str) -> None:
    if re.fullmatch(r"hal-o65-[a-z0-9][a-z0-9-]{0,50}", name) is None:
        raise ValueError("run name must start with hal-o65- and contain lowercase letters, digits or hyphens")


def job_script(sha: str, name: str) -> str:
    validate_name(name)
    if re.fullmatch(r"[0-9a-f]{40}", sha) is None:
        raise ValueError("expected a full HAL commit SHA")
    secrets = "".join(f"{name}={name}\n" for name in SECRETS)
    values = {
        "HAL_GIT_SHA": sha,
        "HAL_RUN_NAME": name,
        "HAL_GCP_PROJECT": PROJECT,
        "HAL_IMAGE": IMAGE,
        "HAL_SECRET_SPECS_B64": base64.b64encode(secrets.encode()).decode(),
        "HAL_RESULTS_HELPER_B64": base64.b64encode((ROOT / "scripts/o65_results.py").read_bytes()).decode(),
        "HAL_R2_HELPER_B64": base64.b64encode((ROOT / "hal/r2.py").read_bytes()).decode(),
    }
    exports = "\n".join(f"export {key}={shlex.quote(value)}" for key, value in values.items())
    return f"#!/usr/bin/env bash\n{exports}\n\n{(ROOT / 'docker/o65-existing.sh').read_text()}"


def ssh_command(command: str) -> list[str]:
    return [
        "gcloud",
        "compute",
        "ssh",
        INSTANCE,
        f"--project={PROJECT}",
        f"--zone={ZONE}",
        "--quiet",
        "--command",
        command,
    ]


def start_command(name: str, staged: str) -> str:
    validate_name(name)
    job = f"/var/lib/hal-o65/{name}"
    script = f"{job}/job.sh"
    # ExecStopPost also runs after failed startup and after SIGKILL of the main process.
    service = shlex.join(
        [
            "sudo",
            "systemd-run",
            f"--unit={name}",
            "--service-type=exec",
            "--property=RuntimeMaxSec=85500s",
            "--property=TimeoutStopSec=900s",
            "--property=KillMode=control-group",
            "--property=KillSignal=SIGKILL",
            "--property=Restart=no",
            f"--property=ExecStopPost=/bin/bash {script} finish",
            "/bin/bash",
            script,
            "run",
        ]
    )
    return " && ".join(
        (
            "sudo install -d -m 0700 /var/lib/hal-o65",
            shlex.join(["sudo", "mkdir", "-m", "0700", job]),
            shlex.join(["sudo", "install", "-m", "0700", staged, script]),
            shlex.join(["rm", "--", staged]),
            service,
        )
    )


@dataclasses.dataclass(frozen=True, slots=True)
class Args:
    action: Literal["start", "stop"] = "start"
    run_name: str | None = None
    dry_run: bool = False
    render: Path | None = None


def main(args: Args) -> None:
    if args.action == "stop":
        if args.run_name is None or args.render is not None:
            raise ValueError("stop requires --run-name and does not accept --render")
        validate_name(args.run_name)
        command = ssh_command(shlex.join(["sudo", "systemctl", "stop", f"{args.run_name}.service"]))
        if args.dry_run:
            print(shlex.join(command))
        else:
            subprocess.run(command, check=True)
        return
    if git("status", "--porcelain"):
        raise SystemExit("working tree is dirty; commit before rendering or starting the published HAL SHA")
    sha = git("rev-parse", "HEAD")
    name = args.run_name or f"hal-o65-proxy-{sha[:8]}"
    rendered = job_script(sha, name)
    if args.render is not None:
        args.render.parent.mkdir(parents=True, exist_ok=True)
        args.render.write_text(rendered)
    staged = f".hal-o65-{uuid.uuid4().hex}.sh"
    start = ssh_command(start_command(name, staged))
    with tempfile.NamedTemporaryFile("w", prefix="hal-o65-", suffix=".sh") as job:
        job.write(rendered)
        job.flush()
        copy = [
            "gcloud",
            "compute",
            "scp",
            job.name,
            f"{INSTANCE}:{staged}",
            f"--project={PROJECT}",
            f"--zone={ZONE}",
            "--quiet",
        ]
        if args.dry_run:
            print(f"Local render only; publication, R2, secrets and VM readiness remain unchecked. HAL SHA: {sha}")
            print(shlex.join(copy))
            print(shlex.join(start))
        else:
            from scripts.o65_delivery import check_remote

            check_remote(ROOT / "experiments/o65/pilot-inputs.json")
            git("fetch", "--quiet", "https://github.com/ericyuegu/hal.git", "+refs/heads/*:refs/hal-launch/*")
            if not git("for-each-ref", "--contains", sha, "--format=%(refname)", "refs/hal-launch/"):
                raise SystemExit(f"HAL SHA {sha} is unpublished; publish it before starting the pilot")
            for secret in SECRETS:
                subprocess.run(
                    ["gcloud", "secrets", "describe", secret, f"--project={PROJECT}", "--format=value(name)"],
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
            subprocess.run(copy, check=True)
            subprocess.run(start, check=True)
    print(f"Run: {name}; unit: {name}.service; output: /var/lib/hal-o65/{name}/output")
    print(f"Results: o65/pilot-results/{name}/")


if __name__ == "__main__":
    main(tyro.cli(Args))
