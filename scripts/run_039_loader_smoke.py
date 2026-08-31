"""Run the compact O39 loader gate as one short Modal allocation."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "039_capacity_scaling.py"


@dataclass(frozen=True, slots=True)
class Probe:
    level: int
    source_run: str
    checkpoint: str
    target_positions: int


PROBES = (
    Probe(3, "cap-L3-d192-2M-U1-prefix-D350981584-tauPL", "branch_D350981584.pt", 701_963_168),
    Probe(4, "cap-L4-d256-4M-U1-prefix-D312145120-tauPL", "branch_D312145120.pt", 624_290_248),
    Probe(5, "cap-L5-d320-7M-U1-prefix-D2p30-tauPL", "branch_D2p30.pt", 1_621_761_184),
    Probe(7, "cap-L7-d448-18M-U1-prefix-D2p30-tauPL", "branch_D2p30.pt", 3_803_727_888),
    Probe(10, "cap-L10-d640-51M-U1-prefix-D2p30-tauPL", "branch_D2p30.pt", 1_493_689_000),
)


@dataclass(frozen=True, slots=True)
class Args:
    worker_counts: tuple[int, ...] = (4, 8, 16)
    warmup_updates: int = 4
    measured_updates: int = 16
    prefetch_updates: int = 1
    fallback_prefetch_updates: int = 2
    output: str = "results/039_loader_smoke/modal_l40s.json"


def _probe_path(probe: Probe, workers: int, prefetch_updates: int) -> Path:
    name = f"{probe.source_run}-loader-probe-w{workers}-p{prefetch_updates}"
    return ROOT / "runs" / name / "loader_probe.json"


def run_probe(probe: Probe, workers: int, prefetch_updates: int, args: Args) -> dict[str, Any]:
    path = _probe_path(probe, workers, prefetch_updates)
    previous_mtime = path.stat().st_mtime_ns if path.exists() else None
    command = [
        sys.executable,
        str(EXPERIMENT),
        "--model-l",
        str(probe.level),
        "--phase",
        "prefix",
        "--target-positions",
        str(probe.target_positions),
        "--throughput-probe-from-run",
        probe.source_run,
        "--throughput-probe-checkpoint",
        probe.checkpoint,
        "--loader-workers",
        str(workers),
        "--loader-prefetch-updates",
        str(prefetch_updates),
        "--throughput-probe-warmup",
        str(args.warmup_updates),
        "--throughput-probe-updates",
        str(args.measured_updates),
    ]
    print(f"[suite] L{probe.level} workers={workers} prefetch={prefetch_updates}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    if not path.is_file() or path.stat().st_mtime_ns == previous_mtime:
        raise RuntimeError(f"probe did not refresh {path}")
    result = json.loads(path.read_text())
    result["level"] = probe.level
    result["source_run"] = probe.source_run
    result["checkpoint"] = probe.checkpoint
    return result


def _stable(result: dict[str, Any]) -> bool:
    mean = float(result["mean_update_wall_s"])
    std = float(result["std_update_wall_s"])
    return math.isfinite(mean) and math.isfinite(std) and mean > 0 and std / mean <= 0.25


def select_workers(results: list[dict[str, Any]]) -> int:
    stable = [result for result in results if _stable(result)]
    if not stable:
        raise RuntimeError("no worker count had finite, stable update times")
    fastest = min(float(result["mean_update_wall_s"]) for result in stable)
    close = [result for result in stable if float(result["mean_update_wall_s"]) <= 1.05 * fastest]
    return min(int(result["loader_workers"]) for result in close)


def main(args: Args) -> None:
    if not args.worker_counts or any(workers <= 0 for workers in args.worker_counts):
        raise SystemExit("worker counts must be positive")
    if args.warmup_updates < 0 or args.measured_updates < 1:
        raise SystemExit("warmup must be non-negative and measured updates must be positive")
    if args.prefetch_updates not in (1, 2) or args.fallback_prefetch_updates not in (1, 2):
        raise SystemExit("prefetch depths must be 1 or 2")

    representative = next(probe for probe in PROBES if probe.level == 7)
    sweep = [run_probe(representative, workers, args.prefetch_updates, args) for workers in args.worker_counts]
    selected_workers = select_workers(sweep)
    selected_l7 = next(result for result in sweep if int(result["loader_workers"]) == selected_workers)

    selected_results = [selected_l7]
    for probe in PROBES:
        if probe is representative:
            continue
        selected_results.append(run_probe(probe, selected_workers, args.prefetch_updates, args))

    fallback_results = []
    if args.fallback_prefetch_updates != args.prefetch_updates:
        failed_levels = {int(result["level"]) for result in selected_results if not result["loader_gate_pass"]}
        for probe in PROBES:
            if probe.level in failed_levels:
                fallback_results.append(run_probe(probe, selected_workers, args.fallback_prefetch_updates, args))

    effective = {int(result["level"]): result for result in selected_results}
    effective.update({int(result["level"]): result for result in fallback_results})
    payload = {
        "schema_version": 1,
        "hardware_required": "NVIDIA L40S",
        "warmup_updates": args.warmup_updates,
        "measured_updates": args.measured_updates,
        "selected_workers": selected_workers,
        "selection_rule": "fastest stable mean; smaller worker count within 5%",
        "worker_sweep": sweep,
        "selected_worker_results": selected_results,
        "fallback_results": fallback_results,
        "all_short_loader_gates_pass": all(bool(result["loader_gate_pass"]) for result in effective.values()),
        "all_short_forecasts_pass": all(bool(result["forecast_gate_pass"]) for result in effective.values()),
        "full_gate_required": args.warmup_updates < 32 or args.measured_updates < 256,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(output)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
