"""Record raw prepared netplay-profile timing without launching Dolphin."""

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import asdict
from multiprocessing import Pipe
from pathlib import Path

import torch

from hal.inference.engine import configure_inference_process
from hal.netplay_service.runner import _InferenceProcessConfig
from hal.netplay_service.runner import _prepare_netplay_engine

_SOURCE_FILES = (
    "hal/controller.py",
    "hal/eval/qualification.py",
    "hal/eval/netplay.py",
    "hal/inference/action_sequence_artifact.py",
    "hal/inference/action_sequence_policy.py",
    "hal/inference/api.py",
    "hal/inference/benchmark.py",
    "hal/inference/client.py",
    "hal/inference/cuda_graph.py",
    "hal/inference/engine.py",
    "hal/inference/gpu_observations.py",
    "hal/inference/kv_cache.py",
    "hal/models/action_sequence.py",
    "hal/models/controller_codec.py",
    "hal/netplay_service/runner.py",
    "tests/fixtures/o59/capture_netplay_ready.py",
)


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _nvidia_query(*fields: str, compute_apps: bool = False) -> tuple[str, ...]:
    query = "--query-compute-apps" if compute_apps else "--query-gpu"
    output = subprocess.check_output(
        ["nvidia-smi", f"{query}={','.join(fields)}", "--format=csv,noheader,nounits"], text=True
    )
    return tuple(line.strip() for line in output.splitlines() if line.strip())


def _ranked_ms(seconds: tuple[float, ...], percentile: float) -> float:
    values = sorted(seconds)
    return 1000 * values[math.ceil(percentile * len(values)) - 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--capacity", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.capacity < 1:
        raise ValueError("stream capacity must be positive")
    if args.output.exists():
        raise FileExistsError(f"immutable qualification record already exists: {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA hardware qualification requires a GPU")
    root = Path(__file__).resolve().parents[3]
    source_before = {name: _sha256(root / name) for name in _SOURCE_FILES}
    compute_before = _nvidia_query("pid", "used_gpu_memory", compute_apps=True)
    gpu_before = _nvidia_query("utilization.gpu", "utilization.memory", "memory.used")
    configure_inference_process()
    connections = [Pipe() for _ in range(args.capacity)]
    try:
        torch.cuda.reset_peak_memory_stats()
        preparation_started = time.perf_counter()
        _, ready = _prepare_netplay_engine(
            _InferenceProcessConfig(args.bundle, "cuda", args.seed, True, args.capacity, 0.0005),
            {slot: parent for slot, (parent, _) in enumerate(connections)},
        )
        torch.cuda.synchronize()
        preparation_seconds = time.perf_counter() - preparation_started
        budgets = []
        for check in ready.budgets:
            measurements = []
            for measurement in check.measurements:
                samples = measurement.seconds
                measurements.append(
                    {
                        "horizon": measurement.prediction_horizon_frames,
                        "fixed_prefix": measurement.fixed_prefix_frames,
                        "count": len(samples),
                        "p50_ms": _ranked_ms(samples, 0.50),
                        "p95_ms": _ranked_ms(samples, 0.95),
                        "p99_ms": _ranked_ms(samples, 0.99),
                        "seconds": samples,
                    }
                )
            budgets.append({"timings": [asdict(value) for value in check.timings], "measurements": measurements})
        result = {
            "schema_version": 1,
            "bundle_sha256": _sha256(args.bundle),
            "checkpoint_sha256": ready.checkpoint_sha256,
            "capability_version": ready.capability_version,
            "capacity": args.capacity,
            "hardware": ready.hardware,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "process_pid": os.getpid(),
            "preparation_seconds": preparation_seconds,
            "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "profiles": [asdict(profile) for profile in ready.profiles],
            "budgets": budgets,
            "gpu_residency_before": {"compute_apps": compute_before, "utilization": gpu_before},
            "gpu_residency_after": {
                "compute_apps": _nvidia_query("pid", "used_gpu_memory", compute_apps=True),
                "utilization": _nvidia_query("utilization.gpu", "utilization.memory", "memory.used"),
            },
            "source_sha256_before": source_before,
            "source_sha256_after": {name: _sha256(root / name) for name in _SOURCE_FILES},
        }
    finally:
        for parent, child in connections:
            parent.close()
            child.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str))
    if result["source_sha256_before"] != result["source_sha256_after"]:
        raise RuntimeError("qualification source changed during capture")
    for check in budgets:
        measurement = check["measurements"][0]
        print(
            f"prefix={measurement['fixed_prefix']} count={measurement['count']} "
            f"p95={measurement['p95_ms']:.3f}ms p99={measurement['p99_ms']:.3f}ms",
            flush=True,
        )


if __name__ == "__main__":
    main()
