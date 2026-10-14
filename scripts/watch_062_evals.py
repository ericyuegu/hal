"""Evaluate O62 milestone checkpoints at extra deployment delays as training uploads them.

Training on GCE has no Modal broker, so this watcher runs on its own one-GPU VM.
It polls R2 for ``runs/<run>/checkpoints/step-*.pt`` and runs the experiment's
``eval`` command once for each milestone and delay. An evaluation is complete
when its ``metrics.json`` is in R2, so a restarted watcher skips finished work.
The watcher exits after the final milestone, or when training stops uploading.

    uv run scripts/launch_gce.py --project centering-star-502613-k3 --zone us-east1-b \
        --machine-type g4-standard-48 --gpu-count 1 --disk 200 --disk-type hyperdisk-balanced --no-spot -- \
        uv run scripts/watch_062_evals.py --run <run>
"""

from __future__ import annotations

import re
import subprocess
import time
from collections.abc import Iterable
from dataclasses import dataclass

import tyro
from loguru import logger

from hal import r2

EXPERIMENT = "experiments/062_production_bc_action_sequence.py"
CHECKPOINT = re.compile(r"checkpoints/step-([0-9]{7})\.pt")


@dataclass(frozen=True, slots=True)
class Args:
    run: str
    """Training run name in R2."""
    final_update: int = 204_800
    delay_frames: tuple[int, ...] = (2, 18)
    replan_interval_frames: int = 2
    n_matchups: int = 96
    max_parallel: int = 32
    poll_seconds: float = 300.0
    max_attempts: int = 2
    """Attempts per evaluation before the watcher records it as failed."""
    idle_hours: float = 6.0
    """Exit when no new milestone appears for this long; training uploads one about every two hours."""


def output_name(args: Args, update: int, delay_frames: int) -> str:
    return f"eval{args.n_matchups}-step-{update:07d}-d{delay_frames}r{args.replan_interval_frames}"


def wandb_namespace(args: Args, delay_frames: int) -> str:
    return f"eval_d{delay_frames}r{args.replan_interval_frames}"


def milestone_updates(relative_keys: Iterable[str]) -> list[int]:
    return sorted(int(match.group(1)) for key in relative_keys if (match := CHECKPOINT.fullmatch(key)))


def pending_evaluations(args: Args, relative_keys: Iterable[str]) -> list[tuple[int, int]]:
    """Return (update, delay) pairs whose checkpoint is uploaded and whose metrics are not."""
    keys = set(relative_keys)
    return [
        (update, delay)
        for update in milestone_updates(keys)
        for delay in args.delay_frames
        if f"{output_name(args, update, delay)}/metrics.json" not in keys
    ]


def eval_command(args: Args, update: int, delay_frames: int) -> list[str]:
    return [
        "uv",
        "run",
        EXPERIMENT,
        "eval",
        "--run",
        args.run,
        "--checkpoint",
        f"checkpoints/step-{update:07d}.pt",
        "--n-matchups",
        str(args.n_matchups),
        "--max-parallel",
        str(args.max_parallel),
        "--delay-frames",
        str(delay_frames),
        "--replan-interval-frames",
        str(args.replan_interval_frames),
        "--output-name",
        output_name(args, update, delay_frames),
        "--shared-wandb",
        "--wandb-namespace",
        wandb_namespace(args, delay_frames),
    ]


def run_keys(run: str) -> set[str]:
    """Return the run's R2 object keys relative to ``runs/<run>/``."""
    prefix = f"runs/{run}/"
    pages = r2.client().get_paginator("list_objects_v2").paginate(Bucket=r2.bucket(), Prefix=prefix)
    return {item["Key"].removeprefix(prefix) for page in pages for item in page.get("Contents", ())}


def main(args: Args) -> None:
    if args.final_update <= 0 or not args.delay_frames:
        raise SystemExit("final update must be positive and at least one delay is required")
    failures: dict[tuple[int, int], int] = {}
    last_milestone = None
    last_progress = time.monotonic()
    while True:
        keys = run_keys(args.run)
        milestones = milestone_updates(keys)
        if milestones and milestones[-1] != last_milestone:
            last_milestone = milestones[-1]
            last_progress = time.monotonic()
        todo = [item for item in pending_evaluations(args, keys) if failures.get(item, 0) < args.max_attempts]
        for update, delay in todo:
            command = eval_command(args, update, delay)
            logger.info(f"evaluating update {update} at delay {delay}: {' '.join(command)}")
            if subprocess.run(command, check=False).returncode != 0:
                failures[(update, delay)] = failures.get((update, delay), 0) + 1
                logger.error(f"evaluation of update {update} at delay {delay} failed")
        if todo:
            continue
        failed = sorted(item for item, count in failures.items() if count >= args.max_attempts)
        if last_milestone == args.final_update:
            if failed:
                raise SystemExit(f"evaluations failed after {args.max_attempts} attempts: {failed}")
            logger.success(f"evaluated every milestone through update {args.final_update}")
            return
        if time.monotonic() - last_progress > args.idle_hours * 3600:
            raise SystemExit(f"no new milestone after update {last_milestone} for {args.idle_hours} h")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main(tyro.cli(Args))
