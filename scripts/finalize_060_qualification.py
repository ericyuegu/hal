"""Validate and upload the experiment 060 G4 qualification record."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated
from typing import Any
from typing import cast

import tyro

from hal.training.checkpoints import BackgroundUploader


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


@dataclass(frozen=True, slots=True)
class Args:
    baseline: Annotated[Path, tyro.conf.Positional]
    ddp: Annotated[Path, tyro.conf.Positional]
    smoke: Annotated[Path, tyro.conf.Positional]
    resume: Annotated[Path, tyro.conf.Positional]
    output: Annotated[Path, tyro.conf.Positional]
    upload_run: str | None = None


def _benchmark_memory_failures(name: str, record: dict[str, Any]) -> list[str]:
    totals = cast(list[int], record["hardware"]["gpu_total_bytes"])
    failures = []
    for rank in cast(list[dict[str, int]], record["rank_memory"]):
        peak = rank["gpu_peak_reserved_bytes"]
        if peak >= 0.9 * totals[rank["rank"]]:
            failures.append(f"{name} rank {rank['rank']} GPU peak is not below 90%")
    host_peak = sum(item["process_peak_rss_bytes"] for item in record["rank_memory"])
    if host_peak >= 0.9 * int(record["hardware"]["host_total_bytes"]):
        failures.append(f"{name} host peak is not below 90%")
    return failures


def main(args: Args) -> None:
    if args.output.exists():
        raise FileExistsError(f"immutable qualification record already exists: {args.output}")
    paths = {
        "baseline": args.baseline,
        "ddp": args.ddp,
        "smoke": args.smoke,
        "resume": args.resume,
    }
    records = {name: _read(path) for name, path in paths.items()}
    baseline = records["baseline"]
    ddp = records["ddp"]
    smoke = records["smoke"]
    resume = records["resume"]
    experiment_id = "060_compute_optimal_action_sequence_v1"
    failures: list[str] = []
    for name, record in records.items():
        if record.get("experiment_id") != experiment_id:
            failures.append(f"{name} experiment identity differs")
    git_shas = {record.get("git_sha") for record in records.values()}
    if len(git_shas) != 1:
        failures.append("benchmark and smoke Git SHAs differ")
    if baseline.get("world_size") != 1 or ddp.get("world_size") != 2:
        failures.append("throughput records are not matched one- and two-process runs")
    if baseline.get("global_samples_per_update") != 256 or ddp.get("global_samples_per_update") != 512:
        failures.append("throughput records do not use local batch 256")
    baseline_throughput = float(baseline["samples_per_second"])
    ddp_throughput = float(ddp["samples_per_second"])
    throughput_ratio = ddp_throughput / baseline_throughput
    if throughput_ratio < 1.6:
        failures.append(f"two-GPU throughput ratio {throughput_ratio:.3f} is below 1.6")
    failures.extend(_benchmark_memory_failures("baseline", baseline))
    failures.extend(_benchmark_memory_failures("DDP", ddp))
    if smoke.get("updates") != 512:
        failures.append("production-shape smoke did not run 512 updates")
    for field in ("loss_and_gradients_finite", "both_gpus_work"):
        if smoke.get(field) is not True:
            failures.append(f"smoke {field} gate failed")
    if smoke.get("nccl_errors") != 0 or smoke.get("recompilation_errors") != 0:
        failures.append("smoke reported NCCL or recompilation errors")
    if float(smoke.get("loader_wait_mean", 1.0)) > 0.05:
        failures.append("smoke mean loader wait exceeds 5%")
    if float(smoke.get("loader_wait_p95", 1.0)) > 0.10:
        failures.append("smoke p95 loader wait exceeds 10%")
    gpu_capacity_gb = min(cast(list[int], smoke["gpu_total_bytes"])) / 2**30
    if float(smoke.get("gpu_peak_gb", gpu_capacity_gb)) >= 0.9 * gpu_capacity_gb:
        failures.append("smoke GPU peak is not below 90%")
    if float(smoke.get("host_sampled_peak_gb", float("inf"))) * 2**30 >= 0.9 * int(smoke["host_total_bytes"]):
        failures.append("smoke host peak is not below 90%")
    if resume.get("passed") is not True:
        failures.append("resume comparison failed")
    if resume.get("next_rank_local_batch_exact") is not True:
        failures.append("resume did not reproduce rank-local data")
    if resume.get("next_optimizer_update_exact") is not True:
        failures.append("resume did not reproduce the next optimizer update")

    artifact_hashes = {name: _sha256(path) for name, path in paths.items()}
    qualification = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "git_sha": next(iter(git_shas)) if len(git_shas) == 1 else None,
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "throughput_ratio": throughput_ratio,
        "resolved_config": smoke.get("resolved_config"),
        "partition_sha256": smoke.get("partition_sha256"),
        "commands": {name: record.get("command") for name, record in records.items()},
        "hardware": ddp.get("hardware"),
        "baseline": baseline,
        "ddp": ddp,
        "smoke": smoke,
        "resume": resume,
        "artifact_sha256": artifact_hashes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(qualification, indent=2, sort_keys=True) + "\n")
    if failures:
        raise RuntimeError(f"experiment 060 qualification failed: {failures}")
    if args.upload_run is not None:
        uploader = BackgroundUploader(args.upload_run, prefix="qualifications")
        uploader.upload(args.output, key="record.json")
        uploader.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
