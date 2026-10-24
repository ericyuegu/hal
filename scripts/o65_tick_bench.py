"""Capture and profile one O65 student tick without a simulator.

Run on the tick devbox with the O64 checkpoint and a warm Inductor cache. Each
replay starts at the same cache position and queue length; weights, synthetic
observations and counter-based sampling uniforms remain fixed.
"""

import dataclasses
import hashlib
import importlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import torch
import tyro

from hal.eval.self_play_checkpoint import load_evaluation_checkpoint
from hal.inference.kv_cache import BlockRestartCache
from hal.training import mfu
from scripts import o65_trace

o65 = importlib.import_module("experiments.065_selfplay_ppo")


@dataclasses.dataclass(frozen=True, slots=True)
class Args:
    checkpoint: Path = Path("/o65-results/o64-inputs/step-0098304.pt")
    output: Path = Path("/cache/tick-bench")
    rows: int = 1024
    # A non-boundary tick; 0 mod 4 decodes four offsets, 2 mod 4 decodes two.
    clock: int = 134
    repeats: int = 100
    seed: int = 65

    def __post_init__(self) -> None:
        if self.rows < 1 or self.repeats < 1:
            raise ValueError("rows and repeats must be positive")
        if self.clock < 2 or self.clock % 2 or self.clock % 128 == 0:
            raise ValueError("clock must select an even, non-boundary tick")


@torch.no_grad()
def main(args: Args) -> None:
    root = Path(__file__).resolve().parents[1]
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
    source_sha256 = {
        path: hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in (
            "scripts/o65_tick_bench.py",
            "experiments/065_selfplay_ppo.py",
            "hal/models/action_sequence.py",
            "hal/models/attention.py",
            "hal/inference/kv_cache.py",
            "hal/inference/decision_decoder.py",
            "hal/inference/cuda_graph.py",
        )
    }
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(False)
    device = torch.device("cuda", 0)
    checkpoint = load_evaluation_checkpoint(args.checkpoint, device=device, decoder_precision="bf16")
    checkpoint.model.requires_grad_(True)
    checkpoint.model.percentile_head.requires_grad_(False)
    policy = o65.TransformerPolicy(
        checkpoint.model, o65.production_config(checkpoint.model.head_offsets), checkpoint.statistics
    )
    matches = (args.rows + 1) // 2
    store = o65.FrameStore(132, matches, device)
    table = o65.DenseRollout(64, store)
    table.allocate_evaluation(policy.model)
    ids = torch.arange(args.rows, device=device)
    meta = table.integers[: args.rows]
    meta[:, o65.STREAM] = ids
    meta[:, o65.LANE] = ids // 2
    meta[:, o65.SIDE] = ids % 2
    meta[:, o65.CLOCK] = args.clock
    meta[:, o65.LENGTH] = 130
    configurations = policy.schedule.configurations
    for row, config in enumerate(configurations):
        meta[row :: len(configurations), o65.DELAY] = config.delay
        meta[row :: len(configurations), o65.REPLAN] = config.replan
        meta[row :: len(configurations), o65.OPP_DELAY] = config.delay
        meta[row :: len(configurations), o65.OPP_REPLAN] = config.replan
        meta[row :: len(configurations), o65.OPP_LENGTH] = config.delay
    table.conditions[: args.rows, :2] = -5
    table.conditions[: args.rows, 2:] = 0.5
    table.active[: args.rows] = True
    ring = o65.ActionRing(matches, 32, device)
    layout = o65.TickLayout(tuple(range(args.rows)), tuple(row // 2 for row in range(args.rows)))
    key = ring, (layout.buffer, args.rows)
    cache = BlockRestartCache(policy.model, args.rows, device, dtype=torch.bfloat16)
    policy.tick_caches[key] = cache
    for value in (*cache.cache.layers, cache.cache.history):
        value.normal_(std=0.1)
    cache.cache.positions.copy_(torch.arange(cache.cache.capacity, device=device)[None])
    count = 4 if args.clock % 4 == 0 else 2
    decoder_rows = tuple(
        row for row in range(args.rows) if args.clock % configurations[row % len(configurations)].replan == 0
    )
    bucket = next(size for size in o65.tick_decoder_buckets(args.rows, count) if size >= len(decoder_rows))
    states = o65.TickInputs(table, ids, decoder_rows=decoder_rows)
    rng = o65.CounterRng(args.seed)

    def reset() -> None:
        cache.cache.next_position.fill_(128)
        cache.frames_in_block.zero_()
        ring.length_device.flatten()[: args.rows].copy_(meta[:, o65.DELAY])

    reset()
    started = time.perf_counter()
    policy.sample_tick(states, rng, layout, ring, clock=args.clock)
    torch.cuda.synchronize()
    capture_seconds = time.perf_counter() - started
    captured = policy.tick_graphs[key, 2, count, bucket]
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(args.repeats)
    ]
    for start, end in events:
        reset()
        start.record()
        captured.graph.replay()
        end.record()
    torch.cuda.synchronize()
    timings = [start.elapsed_time(end) for start, end in events]
    args.output.mkdir(parents=True, exist_ok=True)
    trace = args.output / "trace.json"
    reset()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        with torch.profiler.record_function(f"o65_graph/tick/rows{args.rows}"):
            captured.graph.replay()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(trace))
    name = torch.cuda.get_device_name(device)
    peak = mfu.bf16_dense_peak_flops(name)
    tick_ms = statistics.median(timings)
    flops = policy.work_flops["rollout"]
    report = {
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "checkpoint_provenance": checkpoint.provenance,
        "source_sha256": source_sha256,
        "args": {**dataclasses.asdict(args), "checkpoint": str(args.checkpoint), "output": str(args.output)},
        "device": name,
        "torch": str(torch.__version__),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "inference_mode": torch.is_inference_mode_enabled(),
        "capture_seconds": capture_seconds,
        "captured_tick_graphs": len(policy.tick_graphs),
        "timing_scope": "captured_graph_only",
        "decoder_rows": len(decoder_rows),
        "decoder_bucket": bucket,
        "tick_ms": tick_ms,
        "min_ms": min(timings),
        "max_ms": max(timings),
        "flops": flops,
        "mfu": None if peak is None else flops / (tick_ms / 1000 * peak),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "trace": str(trace),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    o65_trace.main(o65_trace.Args(trace))


if __name__ == "__main__":
    main(tyro.cli(Args))
