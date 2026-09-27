"""Measure the existing training step and prefetch path after a resume capture."""

import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from hal.training.system_metrics import process_tree_pids
from hal.training.system_metrics import read_cgroup_memory
from hal.training.system_metrics import read_process_tree_memory


def _process_counters() -> dict[str, float]:
    import psutil

    counters = {"cpu_seconds": 0.0, "disk_read_bytes": 0.0, "disk_write_bytes": 0.0}
    for pid in process_tree_pids(os.getpid()):
        try:
            process = psutil.Process(pid)
            cpu = process.cpu_times()
            io = process.io_counters()
        except psutil.NoSuchProcess:
            continue
        counters["cpu_seconds"] += cpu.user + cpu.system
        counters["disk_read_bytes"] += io.read_bytes
        counters["disk_write_bytes"] += io.write_bytes
    return counters


def _peak_process_tree_rss(proc_root: Path = Path("/proc")) -> int | None:
    peak_bytes = 0
    for pid in process_tree_pids(os.getpid(), proc_root):
        try:
            lines = (proc_root / str(pid) / "status").read_text().splitlines()
        except FileNotFoundError, ProcessLookupError:
            return None
        for line in lines:
            if line.startswith("VmHWM:"):
                process_peak_bytes = int(line.split()[1]) * 1024
                if process_peak_bytes <= 0:
                    return None
                peak_bytes += process_peak_bytes
                break
        else:
            # gVisor exposes current RSS but omits the high-water counter.
            return None
    return peak_bytes or None


def _run_updates(
    prefetch: Any,
    train_step: Callable,
    *,
    start_step: int,
    count: int,
) -> list[torch.Tensor]:
    if count < 1 or not prefetch.drained:
        raise ValueError("each measured window must start drained and contain at least one update")
    prefetch.fill_lookahead(count)
    prefetch.stage_next()
    losses = []
    for index in range(count):
        batch, valid_prefixes = prefetch.next()
        remaining = count - index - 1
        prefetch.fill_lookahead(remaining)
        step = start_step + index
        result = train_step(
            batch=batch,
            step=step,
            update=step + 1,
            valid_prefixes=valid_prefixes,
            prefix_validated_on_cpu=True,
        )
        losses.append(result.nll_sum.detach())
        if remaining:
            prefetch.stage_next()
    if not prefetch.drained:
        raise RuntimeError("training measurement crossed its final loader boundary")
    return losses


def measure_training_updates(
    prefetch: Any,
    train_step: Callable,
    *,
    next_step: int,
    batch_size: int,
    warm_updates: int,
    measured_updates: int,
) -> dict[str, object]:
    if warm_updates < 100 or measured_updates < 200:
        raise ValueError("production throughput requires at least 100 warm and 200 measured updates")
    started = time.perf_counter()
    warm_losses = _run_updates(prefetch, train_step, start_step=next_step, count=warm_updates)
    torch.cuda.synchronize()
    if not bool(torch.isfinite(torch.stack(warm_losses)).all()):
        raise RuntimeError("training warmup produced a non-finite loss")
    warm_seconds = time.perf_counter() - started
    counters_before = _process_counters()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.compiler.set_stance("fail_on_recompile"):
        losses = _run_updates(prefetch, train_step, start_step=next_step + warm_updates, count=measured_updates)
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    counters_after = _process_counters()
    if not bool(torch.isfinite(torch.stack(losses)).all()):
        raise RuntimeError("training measurement produced a non-finite loss")
    return {
        "protocol": "o59-production-resume-prefetch-training",
        "first_measured_update": next_step + warm_updates + 1,
        "last_measured_update": next_step + warm_updates + measured_updates,
        "batch_size": batch_size,
        "warm_updates": warm_updates,
        "measured_updates": measured_updates,
        "warm_seconds": warm_seconds,
        "measurement_seconds": elapsed,
        "samples_per_second": batch_size * measured_updates / elapsed,
        "updates_per_second": measured_updates / elapsed,
        "measurement_counters": {key: value - counters_before[key] for key, value in counters_after.items()},
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
        "process_tree_memory": read_process_tree_memory(os.getpid()),
        "process_tree_peak_rss_upper_bound_bytes": _peak_process_tree_rss(),
        "cgroup_memory": read_cgroup_memory(),
        "nll_sums": torch.stack(losses).cpu().tolist(),
    }
