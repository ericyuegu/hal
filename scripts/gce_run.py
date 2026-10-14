"""Run a pushed commit on a kept-alive HAL GCE VM and follow its log.

The VM must come from ``launch_gce.py --keep-alive``. Its boot disk holds the
repository, replay shards, fixtures, and compiler caches, so a run skips VM boot,
image pull, shard download, and cold compilation. Jobs run under systemd on the
VM; Ctrl-C or a dropped connection detaches without stopping the job.

Examples::

    uv run scripts/gce_run.py --name hal-o62-dev --zone us-east1-b -- \
        torchrun --standalone --nproc-per-node=8 \
        experiments/062_production_bc_action_sequence.py train --stop-after-update 450
    uv run scripts/gce_run.py --name hal-o62-dev --zone us-east1-b --attach
    uv run scripts/gce_run.py --name hal-o62-dev --zone us-east1-b --stop
"""

from __future__ import annotations

import base64
import shlex
import subprocess
from dataclasses import dataclass
from dataclasses import field

import tyro
from loguru import logger

UNIT = "hal-job"
LOG = "/var/lib/hal/logs/job.log"
EXIT = "/var/lib/hal/logs/job.exit"

# Prints the log until the job exits, then exits with the job's status.
FOLLOW = f"""
pid=$(systemctl show -p MainPID --value {UNIT} 2>/dev/null || echo 0)
if [ "${{pid:-0}}" != 0 ]; then
  sudo tail -n +1 --pid="$pid" -F {LOG} 2>/dev/null
else
  sudo cat {LOG}
fi
[ -f {EXIT} ] || {{ echo "[gce-run] job has no exit status yet"; exit 1; }}
code=$(cat {EXIT}); echo "[gce-run] job exited $code"; exit "$code"
"""


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def pushed_head() -> str:
    """Return HEAD after checking that the VM can fetch exactly this tree."""
    if _git("status", "--porcelain"):
        raise SystemExit("working tree is dirty; commit before running (the VM checks out the pushed SHA)")
    sha = _git("rev-parse", "HEAD")
    if not _git("branch", "-r", "--contains", sha):
        branch = _git("rev-parse", "--abbrev-ref", "HEAD")
        logger.info(f"{sha[:10]} is not on a remote branch; pushing {branch}")
        subprocess.run(["git", "push", "origin", branch], check=True)
    return sha


def start_script(sha: str, command: str, *, follow: bool) -> str:
    """Start one job unless another is active, then optionally follow it."""
    encoded = base64.b64encode(command.encode()).decode()
    then = FOLLOW if follow else f"echo '[gce-run] started {UNIT} on {sha[:10]}'"
    return f"""set -euo pipefail
if systemctl is-active --quiet {UNIT}; then echo "[gce-run] {UNIT} is already running; use --attach or --stop"; exit 75; fi
sudo rm -f {LOG} {EXIT}
sudo systemd-run --quiet --collect --unit={UNIT} /usr/local/sbin/hal-run {shlex.quote(sha)} {encoded}
{then}"""


@dataclass(frozen=True)
class Args:
    cmd: tyro.conf.Positional[list[str]] = field(default_factory=list)
    """Command to run inside the HAL container, after `--`."""
    name: str = "hal-o62-dev"
    """Kept-alive VM name."""
    zone: str = "us-east1-b"
    project: str = "centering-star-502613-k3"
    attach: bool = False
    """Follow the current or last job instead of starting one."""
    stop: bool = False
    """Stop the current job; its container receives SIGTERM, then SIGKILL after 60 s."""
    detach: bool = False
    """Start the job and return without following its log."""


def main(args: Args) -> None:
    if sum((bool(args.cmd), args.attach, args.stop)) != 1:
        raise SystemExit("pass exactly one of a command after `--`, --attach, or --stop")
    if args.stop:
        script = f"sudo systemctl stop {UNIT}; echo '[gce-run] stopped {UNIT}'"
    elif args.attach:
        script = FOLLOW
    else:
        sha = pushed_head()
        script = start_script(sha, shlex.join(args.cmd), follow=not args.detach)
        logger.info(f"running {sha[:10]} on {args.name}: {shlex.join(args.cmd)}")
    ssh = [
        "gcloud",
        "compute",
        "ssh",
        args.name,
        f"--zone={args.zone}",
        f"--project={args.project}",
        "--quiet",
        f"--command=bash -c {shlex.quote(script)}",
    ]
    raise SystemExit(subprocess.run(ssh, check=False).returncode)


if __name__ == "__main__":
    main(tyro.cli(Args))
