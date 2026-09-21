"""Local synthetic O59 cache benchmark; run from the repository root with uv."""

import argparse
import hashlib
import json
import platform
import subprocess
import time
from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import asdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import triton
from o59_shifting_reference import shifting_trunk_forward
from test_059_muon_history_decoder import exp


class ShiftingInference(exp.BF16Inference):
    def _rolling_trunk(self, bucket: int, steps: int) -> Any:
        key = (bucket, steps)
        if key not in self._rolling_trunks:

            def forward(
                features: dict[str, torch.Tensor],
                actions: torch.Tensor,
                keys: torch.Tensor,
                values: torch.Tensor,
                hidden: torch.Tensor,
                pad: torch.Tensor,
                active: torch.Tensor,
                positions: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                return shifting_trunk_forward(
                    self.model,
                    self.model.context_tokens(features, actions),
                    keys,
                    values,
                    hidden,
                    pad,
                    active,
                    positions,
                )

            self._rolling_trunks[key] = torch.compile(
                forward,
                dynamic=False,
                fullgraph=True,
                mode=self.compile_mode,
            )
        return self._rolling_trunks[key]


def decode_projected(
    temporal: Any,
    hidden: torch.Tensor,
    observed: torch.Tensor,
    uniforms: torch.Tensor,
    forced_prefix: torch.Tensor,
    return_value: torch.Tensor,
    condition_present: torch.Tensor,
    history: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Time temporal decoding with memory projection supplied by the caller.

    This test-only copy is checked against the production sampler before timing.
    """
    raw_trunk = hidden[:, -1]
    state_bias = temporal._state_bias(exp.decoder_rmsnorm(raw_trunk))
    trunk_logits = temporal._trunk_skip_logits(raw_trunk)
    previous = observed
    caches: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(temporal.blocks)
    conditioning = temporal.return_conditioner(return_value, condition_present)
    frames = []
    for depth, offset in enumerate(temporal.head_offsets[:4]):
        state, caches = temporal._decode_step(previous, offset, state_bias, caches, history, conditioning)
        embedded: dict[str, torch.Tensor] = {}
        picks: dict[str, torch.Tensor] = {}
        for name in exp.CONTROLLER_DECODE_ORDER:
            logits = temporal._center(
                temporal.outputs[name].project(temporal.group_features(state, name, embedded)) + trunk_logits[name]
            )
            if name == "buttons":
                logits = logits.masked_fill(temporal.codec.button_mask(picks["triggers"]), float("-inf"))
            group = exp.CONTROLLER_GROUP_INDEX[name]
            pick = (
                forced_prefix[:, depth, group]
                if depth < forced_prefix.shape[1]
                else exp.sample_categorical(
                    logits,
                    argmax=False,
                    uniform=uniforms[depth, group],
                    temperature=1.0,
                )
            )
            picks[name] = pick
            embedded[name] = temporal.codec.group_embedding(name, pick)
        previous = torch.stack([picks[name] for name in exp.CONTROLLER_GROUP_NAMES], dim=-1)
        frames.append(previous)
    return torch.stack(frames, dim=1)


def tensor_digest(tensors: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def distribution(milliseconds: list[float], batch: int = 32) -> dict[str, float]:
    return {
        "mean_ms": float(np.mean(milliseconds)),
        "p50_ms": float(np.percentile(milliseconds, 50)),
        "p95_ms": float(np.percentile(milliseconds, 95)),
        "p99_ms": float(np.percentile(milliseconds, 99)),
        "replans_per_second": 1000 / float(np.mean(milliseconds)),
        "committed_frames_per_second": batch * 2 * 1000 / float(np.mean(milliseconds)),
    }


def gpu_times(fn: Callable[[], Any], iterations: int) -> list[float]:
    pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iterations)]
    for start, end in pairs:
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in pairs]


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", choices=("recompute", "shifting", "circular"), required=True)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--mode", default="default", choices=("default", "reduce-overhead"))
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--trunk-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Required CUDA benchmark coverage is unavailable")
    torch.manual_seed(59)
    np.random.seed(59)
    torch.set_num_threads(4)
    arch = exp.Architecture(
        d_model=256,
        n_layers=12,
        n_heads=4,
        L_ctx=args.context,
        temporal_d_model=256,
        temporal_layers=6,
        temporal_heads=4,
        temporal_ff_dim=1024,
        group_head_dim=256,
        value_hidden_dim=128,
    )
    cfg = exp.TrainConfig(arch=arch, batch_size=32, inference_mode="compiled")
    model = exp.GPT(cfg).cuda().eval()
    engine_type = ShiftingInference if args.cache == "shifting" else exp.BF16Inference
    engine = engine_type(
        model,
        cfg,
        compiled=True,
        bucket=32,
        compile_mode=args.mode,
        cache_mode="recompute" if args.cache == "recompute" else "rolling",
    )
    context = exp.synthetic_context(cfg, 32, torch.device("cuda"))
    context = replace(
        context,
        ctx_pad=torch.zeros_like(context.ctx_pad),
        observation_counts=torch.full((32,), args.context, dtype=torch.long, device="cuda"),
        slot_ids=torch.arange(32, device="cuda"),
        reset=torch.ones(32, dtype=torch.bool, device="cuda"),
    )
    committed = torch.as_tensor(exp.NEUTRAL_ACTION, device="cuda").expand(32, 2, -1)
    gen = torch.Generator(device="cuda").manual_seed(59)
    # Both controls and treatment see identical weights, inputs, and uniform draws.
    report: dict[str, Any] = {
        "schema_version": 1,
        "weights_sha256": tensor_digest(model.state_dict()),
        "inputs_sha256": tensor_digest(context.features),
        "uv_lock_sha256": hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest(),
        "arguments": vars(args) | {"output": str(args.output)},
        "configuration": asdict(cfg),
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": hashlib.sha256(Path(exp.__file__).read_bytes()).hexdigest(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "shifting_reference_sha256": hashlib.sha256(
            Path(__file__).with_name("o59_shifting_reference.py").read_bytes()
        ).hexdigest(),
        "triton": triton.__version__,
        "working_tree_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff"])).hexdigest(),
        "seed": 59,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "python": platform.python_version(),
        "driver": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).strip(),
    }
    start = time.perf_counter()
    # Fill using two-token updates, avoiding a context-sized compiled unroll.
    if args.cache != "recompute":
        for count in range(2, args.context + 1, 2):
            initial = replace(
                context,
                reset=torch.full_like(context.reset, count == 2),
                ctx_pad=torch.full_like(context.ctx_pad, args.context - count),
                observation_counts=torch.full_like(context.observation_counts, count),
            )
            engine._prepare_decode(initial, 4, streams=None, gen=gen, committed=committed)
    current_count = args.context

    def decode() -> Any:
        nonlocal current_count
        current_count += 2
        continuing = replace(
            context,
            reset=torch.zeros_like(context.reset),
            observation_counts=torch.full_like(context.observation_counts, current_count),
        )
        return engine.decode(continuing, 4, gen=gen, committed=committed)

    if not args.trunk_only:
        for _ in range(10):
            decode()
    torch.cuda.synchronize()
    report["compile_and_fill_seconds"] = time.perf_counter() - start
    report["cache_bytes"] = engine.rolling_cache_bytes
    torch.cuda.reset_peak_memory_stats()
    print(f"Compiled and filled in {report['compile_and_fill_seconds']:.2f}s", flush=True)
    graphs_before_timing = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    if not args.trunk_only:
        repetitions = []
        with torch.compiler.set_stance("fail_on_recompile"):
            for _ in range(args.repetitions):
                latencies = []
                for _ in range(args.iterations):
                    start = time.perf_counter()
                    decode()
                    torch.cuda.synchronize()
                    latencies.append((time.perf_counter() - start) * 1000)
                repetitions.append(distribution(latencies) | {"samples_ms": latencies})
                print(f"Complete decode repetition {len(repetitions)}: {np.mean(latencies):.3f} ms", flush=True)
        report["complete_decode"] = repetitions
    report["timing_recompilations"] = torch._dynamo.utils.counters["stats"]["unique_graphs"] - graphs_before_timing
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    # GPU-only components exclude Python context preparation and synchronization.
    current_count += 2
    next_context = replace(
        context,
        reset=torch.zeros_like(context.reset),
        observation_counts=torch.full_like(context.observation_counts, current_count),
    )
    prepared = engine._prepare_decode(next_context, 4, streams=None, gen=gen, committed=committed)
    canonical = exp.canonical_context(exp._condition_ego_player(context, exp.MASKED_PLAYER_ID), "base", items=True)
    observed = model.codec.quantize(exp.stack_actions(canonical.features))
    features = canonical.features
    if args.cache == "recompute":

        def trunk() -> Any:
            return engine._trunk(32)(features, context.ctx_pad, observed)
    else:
        features = {name: value[:, -2:] for name, value in features.items()}
        observed = observed[:, -2:]
        state = engine._rolling_state
        active = torch.ones(2, 32, dtype=torch.bool, device="cuda")
        positions = torch.full((2, 32), current_count, dtype=torch.long, device="cuda")
        positions[1] += 1

        def trunk() -> Any:
            positions.add_(2)
            state.keys, state.values, state.hidden, state.ctx_pad = engine._rolling_trunk(32, 2)(
                features,
                observed,
                state.keys,
                state.values,
                state.hidden,
                state.ctx_pad,
                active,
                positions,
            )
            return state.hidden

    projection = torch.compile(model.temporal.history_attention.project_memory, fullgraph=True, mode=args.mode)

    def decoder() -> Any:
        return engine._decoder(32, 4, 2)(
            prepared.hidden,
            prepared.ctx_pad,
            prepared.observed,
            prepared.uniforms,
            prepared.forced_prefix,
            prepared.return_value,
            prepared.condition_present,
        )

    with exp.amp_context(cfg, "cuda"):
        component_start = time.perf_counter()
        if not args.trunk_only:
            history = model.temporal._live_history(prepared.hidden, prepared.ctx_pad)
            projected = torch.compile(decode_projected, fullgraph=True, mode=args.mode)

            def temporal_only() -> torch.Tensor:
                return projected(
                    model.temporal,
                    prepared.hidden,
                    prepared.observed,
                    prepared.uniforms,
                    prepared.forced_prefix,
                    prepared.return_value,
                    prepared.condition_present,
                    history,
                )

            # A separately compiled sampler must preserve the sampled actions.
            torch.testing.assert_close(temporal_only(), decoder(), rtol=0, atol=0)
        for _ in range(10):
            trunk()
            if not args.trunk_only:
                projection(prepared.hidden)
                decoder()
                temporal_only()
        torch.cuda.synchronize()
        report["component_warm_seconds"] = time.perf_counter() - component_start
        with torch.compiler.set_stance("fail_on_recompile"):
            report["trunk_gpu"] = [distribution(gpu_times(trunk, args.iterations)) for _ in range(args.repetitions)]
            if not args.trunk_only:
                report["history_projection_gpu"] = distribution(
                    gpu_times(lambda: projection(prepared.hidden), args.iterations)
                )
                report["temporal_including_projection_gpu"] = distribution(gpu_times(decoder, args.iterations))
                report["temporal_without_projection_gpu"] = distribution(gpu_times(temporal_only, args.iterations))
        # Profiling is separate from all timing runs.
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
        ) as profile:
            for _ in range(3):
                trunk()
                if not args.trunk_only:
                    decoder()
                torch.cuda.synchronize()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        profile.export_chrome_trace(str(args.output.with_suffix(".trace.json")))
        args.output.with_suffix(".profile.txt").write_text(
            profile.key_averages().table(sort_by="self_device_time_total", row_limit=100)
        )
        report["cuda_graph_launches_in_profile"] = sum(
            event.count for event in profile.key_averages() if "cudaGraphLaunch" in event.key
        )
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    report["compiler_counters"] = {name: dict(values) for name, values in torch._dynamo.utils.counters.items()}
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key not in ("configuration", "complete_decode")},
            indent=2,
            default=str,
        ),
        flush=True,
    )
    if "complete_decode" in report:
        print(
            [{key: value for key, value in rep.items() if key != "samples_ms"} for rep in report["complete_decode"]],
            flush=True,
        )


if __name__ == "__main__":
    main()
