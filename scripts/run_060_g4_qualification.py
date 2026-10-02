"""Run every experiment 060 qualification phase on one G4 host."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import tyro

from hal.training.runs import source_git_sha

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "060_compute_optimal_action_sequence.py"
BENCHMARK = ROOT / "scripts" / "benchmark_060_ddp.py"
COMPARE = ROOT / "scripts" / "compare_060_resume.py"
FINALIZE = ROOT / "scripts" / "finalize_060_qualification.py"


@dataclass(frozen=True, slots=True)
class Args:
    output_dir: Annotated[Path, tyro.conf.Positional]
    warm_updates: int = 50
    measured_updates: int = 200


def _torchrun(processes: int, script: Path, *args: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={processes}",
        str(script),
        *args,
    ]


def _run(command: list[str]) -> None:
    print(f"[qualification] running: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _new_run(command: list[str]) -> Path:
    run_root = ROOT / "runs"
    before = set(run_root.iterdir()) if run_root.exists() else set()
    _run(command)
    after = set(run_root.iterdir())
    created = sorted(after - before)
    if len(created) != 1:
        raise RuntimeError(f"qualification phase created {len(created)} run directories; expected one")
    return created[0]


def _train_command(*args: str) -> list[str]:
    return _torchrun(2, EXPERIMENT, "train", *args)


def main(args: Args) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"immutable qualification directory already exists: {args.output_dir}")
    if args.warm_updates < 1 or args.measured_updates < 20:
        raise ValueError("qualification requires at least one warm and twenty measured updates")
    args.output_dir.mkdir(parents=True)

    baseline_path = args.output_dir / "baseline.json"
    ddp_path = args.output_dir / "ddp.json"
    smoke_path = args.output_dir / "smoke.json"
    resume_path = args.output_dir / "resume.json"
    record_path = args.output_dir / "record.json"

    benchmark_args = (
        "--warm-updates",
        str(args.warm_updates),
        "--measured-updates",
        str(args.measured_updates),
    )
    _run(_torchrun(1, BENCHMARK, str(baseline_path), *benchmark_args))
    _run(_torchrun(2, BENCHMARK, str(ddp_path), *benchmark_args))

    _new_run(
        _train_command(
            "--comment",
            "g4-qualification-smoke",
            "--smoke",
            "--stop-after-update",
            "512",
            "--qualification-output",
            str(smoke_path),
        )
    )

    control_run = _new_run(
        _train_command(
            "--comment",
            "g4-resume-control",
            "--smoke",
            "--stop-after-update",
            "2",
        )
    )
    interrupted_run = _new_run(
        _train_command(
            "--comment",
            "g4-resume-interrupted",
            "--smoke",
            "--stop-after-update",
            "1",
        )
    )
    _run(
        _train_command(
            "--resume",
            interrupted_run.name,
            "--resume-checkpoint",
            "smoke-final.pt",
            "--smoke",
            "--stop-after-update",
            "2",
        )
    )
    _run(
        [
            sys.executable,
            str(COMPARE),
            str(control_run / "smoke-final.pt"),
            str(interrupted_run / "smoke-final.pt"),
            str(resume_path),
        ]
    )

    sha = source_git_sha(ROOT)
    upload_run = f"060-g4-{sha[:12]}"
    _run(
        [
            sys.executable,
            str(FINALIZE),
            str(baseline_path),
            str(ddp_path),
            str(smoke_path),
            str(resume_path),
            str(record_path),
            "--upload-run",
            upload_run,
        ]
    )
    record = json.loads(record_path.read_text())
    if record.get("status") != "passed":
        raise RuntimeError(f"qualification did not pass: {record.get('failures')}")
    print(f"[qualification] passed; record={record_path}; R2={upload_run}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
