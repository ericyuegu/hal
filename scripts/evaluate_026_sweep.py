"""Repair one experiment-026 sweep run with a complete 96-boot evaluation.

The final checkpoint is downloaded from R2, evaluated at stride four with only
eight concurrent Dolphin workers, and uploaded under a distinct replay path.
The original W&B summary is updated only after all 96 boots reach active play.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Final

import tyro

import wandb
from hal.training.checkpoints import BackgroundUploader
from hal.training.checkpoints import download_latest

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
EXPERIMENT: Final[Path] = ROOT / "experiments" / "026_temporal_mtp.py"
WANDB_PATH: Final[str] = "ericyuegu/hal"
WANDB_ID: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9]+$")


@dataclass(frozen=True, slots=True)
class Args:
    run_name: str
    """Exact checkpoint run name under the R2 runs/ prefix."""
    wandb_id: str
    """Original W&B run ID whose final eval summary should be repaired."""
    max_parallel: int = 8
    """Concurrent Dolphin boots; eight matches the L40S job's requested CPUs."""
    n_matchups: int = 96
    """Number of deterministic matchup boots required for a trusted result."""
    output_name: str = "eval_replays_s4_complete96"
    """Distinct artifact directory that preserves the built-in final eval."""


def _load_experiment() -> ModuleType:
    spec = importlib.util.spec_from_file_location("evaluate_exp026", EXPERIMENT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {EXPERIMENT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_args(args: Args) -> None:
    if Path(args.run_name).name != args.run_name or args.run_name in ("", ".", ".."):
        raise SystemExit("--run-name must be one directory-safe name")
    if not WANDB_ID.fullmatch(args.wandb_id):
        raise SystemExit("--wandb-id must be alphanumeric")
    if args.n_matchups != 96:
        raise SystemExit("this repair requires exactly 96 matchup boots")
    if args.max_parallel < 1 or args.max_parallel & (args.max_parallel - 1):
        raise SystemExit("--max-parallel must be a positive power of two")
    if Path(args.output_name).name != args.output_name or args.output_name in ("", ".", ".."):
        raise SystemExit("--output-name must be one directory-safe name")


def _archive_builtin_eval(run: wandb.apis.public.Run) -> None:
    if "audit/builtin_eval/archived_at" in run.summary:
        return
    values = {
        f"audit/builtin_eval/{name.removeprefix('eval/')}": value
        for name, value in dict(run.summary).items()
        if name.startswith("eval/")
    }
    values["audit/builtin_eval/archived_at"] = datetime.now(UTC).isoformat()
    for name, value in values.items():
        run.summary[name] = value
    run.summary.update()


def _publish_eval(
    run: wandb.apis.public.Run,
    metrics: dict[str, float],
    *,
    args: Args,
    checkpoint_sha256: str,
) -> None:
    _archive_builtin_eval(run)
    values: dict[str, str | int | float] = {f"eval/{name}": value for name, value in metrics.items()}
    values.update(
        {
            "audit/selected_eval/r2_metrics_key": f"replays/{args.output_name}/metrics.json",
            "audit/selected_eval/checkpoint_sha256": checkpoint_sha256,
            "audit/selected_eval/max_parallel": args.max_parallel,
            "audit/selected_eval/required_boots": args.n_matchups,
            "audit/selected_eval/completed_at": datetime.now(UTC).isoformat(),
        }
    )
    for name, value in values.items():
        run.summary[name] = value
    run.summary.update()


def main(args: Args) -> None:
    _validate_args(args)
    exp = _load_experiment()
    api = wandb.Api(timeout=90)
    run = api.run(f"{WANDB_PATH}/{args.wandb_id}")
    if run.name != args.run_name:
        raise AssertionError(f"W&B run name {run.name!r} != {args.run_name!r}")

    with tempfile.TemporaryDirectory(prefix=f"eval-026-{args.wandb_id}-") as directory:
        run_dir = Path(directory) / args.run_name
        checkpoint = download_latest(args.run_name, run_dir, name="final.pt")
        if checkpoint is None:
            raise FileNotFoundError(f"R2 final checkpoint not found for {args.run_name}")
        checkpoint_sha = _sha256(checkpoint)
        replay_dir = run_dir / args.output_name
        uploader = BackgroundUploader(args.run_name)
        try:
            metrics = exp.eval_checkpoint(
                str(checkpoint),
                exec_horizon=4,
                n_matchups=args.n_matchups,
                max_parallel=args.max_parallel,
                output_name=args.output_name,
            )
        finally:
            if replay_dir.is_dir():
                uploader.upload_tree(replay_dir, base=run_dir)
            uploader.close()

        _publish_eval(run, metrics, args=args, checkpoint_sha256=checkpoint_sha)
        print(f"[eval-repair] {args.run_name}: {metrics}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
