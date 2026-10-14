"""Measure O62 production training updates on synthetic batches.

The benchmark runs the experiment's own compiled ``train_step`` without the
replay loader, so it isolates model, optimizer, and gradient all-reduce cost
from data loading. Run one process on one GPU, or one process per GPU::

    CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc-per-node=1 scripts/benchmark_062_step.py
    torchrun --standalone --nproc-per-node=8 scripts/benchmark_062_step.py --profile
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch._functorch.config as functorch_config
import torch.distributed as dist
import tyro

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_PATH = ROOT / "experiments" / "062_production_bc_action_sequence.py"


def _load_experiment() -> Any:
    spec = importlib.util.spec_from_file_location("hal_benchmark_experiment_062", EXPERIMENT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {EXPERIMENT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def gpu_busy_fraction(trace_path: Path) -> float:
    """Return the fraction of the traced span covered by at least one GPU kernel."""
    with gzip.open(trace_path, "rt") as trace:
        events = json.load(trace)["traceEvents"]
    kernels = sorted((e["ts"], e["ts"] + e["dur"]) for e in events if e.get("cat") == "kernel" and e.get("ph") == "X")
    if not kernels:
        raise ValueError(f"{trace_path} contains no GPU kernels")
    busy = 0.0
    start, stop = kernels[0]
    for kernel_start, kernel_stop in kernels[1:]:
        if kernel_start > stop:
            busy += stop - start
            start = kernel_start
        stop = max(stop, kernel_stop)
    busy += stop - start
    return busy / (stop - kernels[0][0])


@dataclass(frozen=True, slots=True)
class Args:
    local_batch: int = 256
    """Windows per GPU."""
    warm_updates: int = 5
    measured_updates: int = 20
    activation_memory_budget: float | None = None
    """AOTAutograd's recompute budget in (0, 1]; None keeps PyTorch's default partitioner."""
    profile: bool = False
    """Trace the measured updates and report GPU busy time and top kernels."""


def main(args: Args) -> None:
    if args.warm_updates < 1 or args.measured_updates < 1:
        raise ValueError("benchmark requires at least one warm and one measured update")
    if args.activation_memory_budget is not None:
        if not 0 < args.activation_memory_budget <= 1:
            raise ValueError("activation memory budget must be in (0, 1]")
        functorch_config.activation_memory_budget = float(args.activation_memory_budget)
    experiment = _load_experiment()
    world_size = int(os.environ["WORLD_SIZE"])
    cfg = replace(
        experiment.production_config(),
        world_size=world_size,
        batch_size=args.local_batch * world_size,
        local_batch_size=args.local_batch,
        microbatch_size=args.local_batch,
        push_to_r2=False,
        wandb_log_code=False,
    )
    distributed = experiment.init_distributed(cfg)
    try:
        process_seed = experiment.rank_seed(cfg.seed, distributed.rank)
        torch.manual_seed(process_seed)
        torch.set_float32_matmul_precision("high")
        sidecar = experiment.load_identity_sidecar(cfg)
        model = experiment.make_model(cfg, sidecar.vocabulary).to(distributed.device)
        flops_per_update = experiment.approximate_training_flops_per_update(
            cfg, experiment.subsystem_parameter_counts(model)
        )
        optimizer = experiment.make_optimizer(model, cfg)
        scheduler = experiment.LearningRateScheduler(optimizer, experiment.lr_schedule(cfg))
        trunk_fn, temporal_fn = experiment._training_functions(model, cfg)
        compile_started = time.monotonic()
        experiment._compile_synthetic_forward_backward(model, cfg, step=0, trunk_fn=trunk_fn, temporal_fn=temporal_fn)
        compile_seconds = time.monotonic() - compile_started
        owner = experiment.wrap_training_owner(model, cfg, distributed, trunk_fn, temporal_fn)
        batch = experiment.synthetic_awr_batch(cfg, distributed.device)
        prefix_sampler = experiment.PrefixSampler(process_seed ^ 0x060A11, distributed.device)
        valid_prefixes = cfg.local_batch_size * experiment.POLICY_PREFIXES_PER_WINDOW

        def update(step: int) -> None:
            experiment.train_step(
                model,
                owner,
                batch,
                cfg,
                step=step,
                update=step + 1,
                valid_prefixes=valid_prefixes,
                optimizer=optimizer,
                scheduler=scheduler,
                prefix_sampler=prefix_sampler,
                prefix_validated_on_cpu=True,
            )

        for step in range(args.warm_updates):
            update(step)
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.reset_peak_memory_stats()
        profiler = None
        if args.profile:
            profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            )
            profiler.start()
        update_seconds = []
        with torch.compiler.set_stance("fail_on_recompile"):
            for step in range(args.warm_updates, args.warm_updates + args.measured_updates):
                started = time.perf_counter()
                update(step)
                torch.cuda.synchronize()
                update_seconds.append(time.perf_counter() - started)
        profile_dir = ROOT / "runs" / "benchmark-062-step" / f"b{args.local_batch}-w{world_size}"
        if profiler is not None:
            experiment.export_update_profile(profiler, profile_dir, distributed)
        median_s = float(np.median(update_seconds))
        summary = {
            "rank": distributed.rank,
            "world_size": world_size,
            "local_batch": args.local_batch,
            "activation_memory_budget": args.activation_memory_budget,
            "compile_s": round(compile_seconds, 1),
            "update_s_median": round(median_s, 4),
            "update_s_p95": round(float(np.percentile(update_seconds, 95)), 4),
            "tflops_per_gpu": round(flops_per_update * args.local_batch / cfg.batch_size / median_s / 1e12, 1),
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        }
        if profiler is not None:
            summary["gpu_busy_fraction"] = round(gpu_busy_fraction(profile_dir / f"rank{distributed.rank}.json.gz"), 4)
        print(f"[benchmark] {json.dumps(summary)}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main(tyro.cli(Args))
