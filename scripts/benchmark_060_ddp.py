"""Measure matched one- and two-process experiment 060 training throughput.

Run this on the G4 qualification host with one process for the local-batch-256
baseline, then with two processes for DDP::

    torchrun --standalone --nproc-per-node=1 scripts/benchmark_060_ddp.py baseline.json
    torchrun --standalone --nproc-per-node=2 scripts/benchmark_060_ddp.py ddp.json
"""

from __future__ import annotations

import importlib.util
import json
import os
import resource
import subprocess
import sys
import time
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated
from typing import Any
from typing import cast

import torch
import torch.distributed as dist
import tyro

from hal.training.runs import source_git_sha

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_PATH = ROOT / "experiments" / "060_compute_optimal_action_sequence.py"


def _load_experiment() -> Any:
    spec = importlib.util.spec_from_file_location("hal_benchmark_experiment_060", EXPERIMENT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {EXPERIMENT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _host_total_bytes() -> int:
    return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")


def _hardware_preflight() -> dict[str, object]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError(f"G4 qualification requires exactly two visible GPUs, got {torch.cuda.device_count()}")
    properties = [torch.cuda.get_device_properties(index) for index in range(2)]
    if any("RTX PRO 6000 Blackwell" not in properties[index].name for index in range(2)):
        raise RuntimeError(f"unexpected G4 GPUs: {[item.name for item in properties]}")
    if any(item.total_memory < 90 * 2**30 for item in properties):
        raise RuntimeError("each G4 GPU must expose at least 90 GiB")
    peer_access = [
        [source == target or torch.cuda.can_device_access_peer(source, target) for target in range(2)]
        for source in range(2)
    ]
    if not all(peer_access[source][target] for source in range(2) for target in range(2)):
        raise RuntimeError(f"direct GPU peer access is unavailable: {peer_access}")
    host_total = _host_total_bytes()
    if host_total < 340 * 2**30:
        raise RuntimeError(f"g4-standard-96 must expose about 360 GiB of host RAM, got {host_total / 2**30:.1f}")
    topology = subprocess.run(
        ["nvidia-smi", "topo", "-m"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "gpu_names": [item.name for item in properties],
        "gpu_total_bytes": [item.total_memory for item in properties],
        "host_total_bytes": host_total,
        "peer_access": peer_access,
        "nvidia_smi_topology": topology,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


@dataclass(frozen=True, slots=True)
class Args:
    output: Annotated[Path, tyro.conf.Positional]
    warm_updates: int = 50
    measured_updates: int = 200


def _validate_args(args: Args) -> None:
    if args.output.exists():
        raise FileExistsError(f"immutable benchmark output already exists: {args.output}")
    if args.warm_updates < 1 or args.measured_updates < 20:
        raise ValueError("benchmark requires at least one warm and twenty measured updates")


def main(args: Args) -> None:
    _validate_args(args)
    experiment = _load_experiment()
    hardware = _hardware_preflight()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size not in (1, 2) or not 0 <= rank < world_size:
        raise RuntimeError("benchmark world size must be one or two")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        cfg = experiment.TrainConfig(push_to_r2=False, wandb_log_code=False)
        experiment.validate_config(cfg)
        process_seed = experiment.rank_seed(cfg.seed, rank)
        torch.manual_seed(process_seed)
        torch.set_float32_matmul_precision("high")
        model = experiment.make_model(cfg).to(device)
        counts = experiment.subsystem_parameter_counts(model)
        optimizer = experiment.make_optimizer(model, cfg)
        scheduler = experiment.LearningRateScheduler(optimizer, experiment.lr_schedule(cfg))
        trunk_fn, temporal_fn = experiment._training_functions(model, cfg)
        experiment._compile_synthetic_forward_backward(
            model,
            cfg,
            step=0,
            trunk_fn=trunk_fn,
            temporal_fn=temporal_fn,
        )
        context = experiment.DistributedContext(rank, local_rank, world_size, device)
        owner = experiment.wrap_training_owner(model, cfg, context, trunk_fn, temporal_fn)
        batch = experiment.synthetic_awr_batch(cfg, device)
        prefix = experiment.PrefixSampler(process_seed ^ 0x060A11, device)
        valid_prefixes = cfg.local_batch_size * experiment.POLICY_PREFIXES_PER_WINDOW

        def update(step: int) -> Any:
            return experiment.train_step(
                model,
                owner,
                batch,
                cfg,
                step=step,
                update=step + 1,
                valid_prefixes=valid_prefixes,
                optimizer=optimizer,
                scheduler=scheduler,
                prefix_sampler=prefix,
                prefix_validated_on_cpu=True,
            )

        for step in range(args.warm_updates):
            update(step)
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        results = []
        with torch.compiler.set_stance("fail_on_recompile"):
            for step in range(args.warm_updates, args.warm_updates + args.measured_updates):
                results.append(update(step))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        elapsed_tensor = torch.tensor(elapsed, dtype=torch.float64, device=device)
        dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
        elapsed = float(elapsed_tensor.cpu())
        if not all(bool(torch.isfinite(result.nll_sum).all()) for result in results):
            raise RuntimeError("benchmark produced a non-finite NLL")
        local_record = {
            "rank": rank,
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        }
        gathered: list[object] | None = [None] * world_size if rank == 0 else None
        dist.gather_object(local_record, gathered, dst=0)
        if rank != 0:
            return
        assert gathered is not None
        rank_records = cast(list[dict[str, int]], gathered)
        samples = cfg.local_batch_size * world_size * args.measured_updates
        record = {
            "schema_version": 1,
            "experiment_id": experiment._EXPERIMENT_ID,
            "git_sha": source_git_sha(ROOT),
            "command": sys.argv,
            "world_size": world_size,
            "global_samples_per_update": cfg.local_batch_size * world_size,
            "warm_updates": args.warm_updates,
            "measured_updates": args.measured_updates,
            "measurement_seconds": elapsed,
            "samples_per_second": samples / elapsed,
            "updates_per_second": args.measured_updates / elapsed,
            "hardware": hardware,
            "rank_memory": rank_records,
            "parameter_counts": counts,
            "resolved_config": asdict(cfg),
            "partition_sha256": experiment.data_partition_hashes(cfg),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main(tyro.cli(Args))
