"""Measure O65 model calls and hardware rates without simulation or PPO."""

import dataclasses
import gc
import json
import math
import statistics
import subprocess
import time
from collections import Counter
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Literal
from typing import TypedDict
from typing import cast

import torch
import triton
import triton.language as tl
import tyro
from torch import Tensor

from hal.eval.self_play_checkpoint import PROXY_SHA256
from hal.eval.self_play_checkpoint import load_evaluation_checkpoint
from hal.eval.self_play_checkpoint import sha256
from hal.inference.cuda_graph import CapturedCall
from hal.inference.cuda_graph import prepared_shape_compilation
from hal.inference.decision_decoder import DecisionDecoder
from hal.inference.kv_cache import KVCache
from hal.inference.kv_cache import forward_tokens_with_kv_cache
from hal.models.action_sequence import DecisionOutput
from hal.models.self_play import SelfPlayTransformer
from hal.models.window_replay import ReplayOutput
from hal.models.window_replay import ReplaySlots
from hal.models.window_replay import replay_windows
from hal.models.window_replay import validate_replay_inputs
from hal.training.muon import SingleDeviceMuonWithAuxAdam


class TickLayout(TypedDict):
    buffer: int
    anchor: int | None
    batch: int


class OptimizerGroup(TypedDict):
    use_muon: bool
    logical_splits: int
    shapes: list[list[int]]


class FloorLayout(TypedDict):
    format: str
    tick_layouts: list[TickLayout]
    actor_steps: int
    actor_windows: int
    actor_slots: int
    reference_windows: int
    reference_slots: int
    frames: int
    decisions: int
    matches: int
    critic_parameters: list[list[int]]
    optimizer_groups: list[OptimizerGroup]


def read_layout(path: Path) -> FloorLayout:
    raw = json.loads(path.read_text())
    if (
        not isinstance(raw, dict)
        or set(raw) != set(FloorLayout.__annotations__)
        or raw["format"] != "hal.o65.floor-layout.v1"
    ):
        raise ValueError("unsupported floor layout")
    for name in (
        "actor_steps",
        "actor_windows",
        "actor_slots",
        "reference_windows",
        "reference_slots",
        "frames",
        "decisions",
        "matches",
    ):
        if type(raw[name]) is not int or raw[name] < 1:
            raise ValueError(f"floor layout {name} must be a positive integer")
    if (raw["actor_windows"], raw["actor_slots"], raw["reference_slots"]) != (32, 16, 64):
        raise ValueError("floor probe requires the trainer's qualified 32/16/64 window shapes")
    layouts = raw["tick_layouts"]
    if not isinstance(layouts, list) or not layouts:
        raise ValueError("floor layout needs policy batches")
    for layout in layouts:
        if (
            not isinstance(layout, dict)
            or set(layout) != {"buffer", "anchor", "batch"}
            or type(layout["buffer"]) is not int
            or layout["buffer"] not in (0, 1)
            or type(layout["batch"]) is not int
            or layout["batch"] < 1
            or (layout["anchor"] is not None and (type(layout["anchor"]) is not int or layout["anchor"] < 0))
        ):
            raise ValueError("invalid floor policy batch")
    groups = raw["optimizer_groups"]
    if not isinstance(groups, list) or not groups:
        raise ValueError("floor layout needs optimizer groups")
    for group in groups:
        if (
            not isinstance(group, dict)
            or set(group) != {"use_muon", "logical_splits", "shapes"}
            or type(group["use_muon"]) is not bool
            or type(group["logical_splits"]) is not int
            or group["logical_splits"] < 1
        ):
            raise ValueError("invalid floor optimizer group")
        validate_shapes(group["shapes"])
    validate_shapes(raw["critic_parameters"])
    return cast(FloorLayout, raw)


def validate_shapes(value: object) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError("floor tensor shapes must be a nonempty list")
    for shape in value:
        if not isinstance(shape, list) or any(type(size) is not int or size < 1 for size in shape):
            raise ValueError("invalid floor tensor shape")


@triton.jit
def empty_kernel():
    return


@triton.jit
def copy_kernel(source, target, size: tl.constexpr, BLOCK: tl.constexpr):
    at = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(target + at, tl.load(source + at, at < size, other=0), at < size)


@dataclasses.dataclass(frozen=True, slots=True)
class Latency:
    wall_seconds: float
    cuda_seconds: float
    samples: int


@dataclasses.dataclass(frozen=True, slots=True, order=True)
class Gemm:
    batch: int
    m: int
    n: int
    k: int
    precision: Literal["bf16", "fp32"] = "bf16"
    left_stride: tuple[int, ...] = ()
    right_stride: tuple[int, ...] = ()

    @property
    def flops(self) -> int:
        return 2 * self.batch * self.m * self.n * self.k

    @property
    def bytes(self) -> int:
        return (
            (2 if self.precision == "bf16" else 4) * self.batch * (self.m * self.k + self.k * self.n + self.m * self.n)
        )


def gemm_shape(shapes: list[list[int]]) -> Gemm:
    if len(shapes) < 2 or len(shapes[0]) not in (2, 3) or len(shapes[0]) != len(shapes[1]):
        raise ValueError("expected mm or bmm profiler dimensions")
    left, right = shapes[:2]
    if left[-1] != right[-2] or (len(left) == 3 and left[0] != right[0]):
        raise ValueError("incompatible profiler GEMM dimensions")
    if any(type(value) is not int or value < 1 for value in (*left, *right)):
        raise ValueError("profiler GEMM dimensions must be positive integers")
    return Gemm(left[0] if len(left) == 3 else 1, left[-2], right[-1], left[-1])


def measure(operation: Callable[[], object], *, repeats: int = 7, reset: Callable[[], None] | None = None) -> Latency:
    walls, device = [], []
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        if reset is not None:
            reset()
        torch.cuda.synchronize()
        began = time.perf_counter()
        start.record()
        operation()
        stop.record()
        stop.synchronize()
        walls.append(time.perf_counter() - began)
        device.append(start.elapsed_time(stop) / 1000)
    return Latency(statistics.median(walls), statistics.median(device), repeats)


def graph_call(operation: Callable[[], object]) -> Callable[[], object]:
    captured = CapturedCall(operation, (), device=torch.device("cuda", torch.cuda.current_device()))

    def replay() -> None:
        captured.graph.replay()

    return replay


def profile_shapes(operation: Callable[[], object], output: Path) -> Counter[Gemm]:
    operation()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_flops=True,
    ) as profiler:
        operation()
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(output))
    return read_gemm_shapes(output)


def read_gemm_shapes(path: Path) -> Counter[Gemm]:
    shapes: Counter[Gemm] = Counter()
    for event in json.loads(path.read_text())["traceEvents"]:
        if event.get("cat") != "cpu_op" or event["name"] not in ("aten::mm", "aten::bmm", "aten::addmm"):
            continue
        metadata = event["args"]
        start = 1 if event["name"] == "aten::addmm" else 0
        dimensions = metadata["Input Dims"][start : start + 2]
        types = metadata["Input type"][start : start + 2]
        precision = {"c10::BFloat16": "bf16", "float": "fp32"}
        if len(set(types)) != 1 or types[0] not in precision:
            raise ValueError(f"unqualified model GEMM types: {types}")
        strides = metadata["Input Strides"][start : start + 2]
        shape = dataclasses.replace(
            gemm_shape(dimensions),
            precision=cast(Literal["bf16", "fp32"], precision[types[0]]),
            left_stride=tuple(strides[0]),
            right_stride=tuple(strides[1]),
        )
        shapes[shape] += 1
    if not shapes:
        raise ValueError("model profiler did not record GEMM shapes")
    return shapes


def hardware_launches() -> dict[str, object]:
    result: dict[str, object] = {}
    source = torch.ones(1, device="cuda")
    target = torch.zeros_like(source)
    for name, kernel in (
        ("empty", lambda: empty_kernel[(1,)]()),
        ("tiny", lambda: copy_kernel[(1,)](source, target, cast(tl.constexpr, 1), BLOCK=cast(tl.constexpr, 32))),
    ):
        kernel()
        for count in (1, 32, 256, 1024):

            def series(count: int = count, kernel: Callable[[], object] = kernel) -> None:
                for _ in range(count):
                    kernel()

            eager = measure(series)
            replay = measure(graph_call(series))
            result[f"{name}/{count}"] = {
                "eager": dataclasses.asdict(eager),
                "graph": dataclasses.asdict(replay),
                "eager_wall_seconds_per_kernel": eager.wall_seconds / count,
                "graph_cuda_seconds_per_kernel": replay.cuda_seconds / count,
            }
    return result


def hardware_memory() -> dict[str, object]:
    # Two 256 MiB buffers exceed the PRO 6000's L2; count one read and one write.
    size = 64 * 1024 * 1024
    source = torch.randn(size, device="cuda")
    target = torch.empty_like(source)
    operation = graph_call(
        lambda: copy_kernel[(triton.cdiv(size, 1024),)](
            source, target, cast(tl.constexpr, size), BLOCK=cast(tl.constexpr, 1024)
        )
    )
    latency = measure(operation, repeats=21)
    torch.testing.assert_close(source, target, rtol=0, atol=0)
    return {
        "latency": dataclasses.asdict(latency),
        "bytes": size * 8,
        "bytes_per_second": size * 8 / latency.cuda_seconds,
    }


def hardware_gemms(shapes: set[Gemm]) -> list[dict[str, object]]:
    results = []
    for shape in sorted(shapes):
        dtype = torch.bfloat16 if shape.precision == "bf16" else torch.float32
        batched = len(shape.left_stride) == 3
        left_shape = (shape.batch, shape.m, shape.k) if batched else (shape.m, shape.k)
        right_shape = (shape.batch, shape.k, shape.n) if batched else (shape.k, shape.n)
        left = strided_random(left_shape, shape.left_stride, dtype)
        right = strided_random(right_shape, shape.right_stride, dtype)
        target = torch.empty((*left_shape[:-1], shape.n), device="cuda", dtype=dtype)
        if not batched:
            operation = partial(torch.mm, left, right, out=target)
        else:
            operation = partial(torch.bmm, left, right, out=target)
        latency = measure(graph_call(partial(repeat, operation, 32)))
        seconds = latency.cuda_seconds / 32
        results.append(
            {
                **dataclasses.asdict(shape),
                "flops": shape.flops,
                "bytes": shape.bytes,
                "latency": dataclasses.asdict(latency),
                "graph_nodes": 32,
                "seconds_per_gemm": seconds,
                "tflops_per_second": shape.flops / seconds / 1e12,
            }
        )
    return results


def strided_random(shape: tuple[int, ...], stride: tuple[int, ...], dtype: torch.dtype) -> Tensor:
    if len(shape) != len(stride) or any(value < 0 for value in stride):
        raise ValueError("profiler matrix strides must be nonnegative and match the rank")
    size = 1 + sum((length - 1) * step for length, step in zip(shape, stride, strict=True))
    storage = torch.randn(size, device="cuda", dtype=dtype)
    return storage.as_strided(shape, stride)


def repeat(operation: Callable[[], object], count: int) -> None:
    for _ in range(count):
        operation()


def tick_benchmark(
    model: SelfPlayTransformer, batch: int, count: int, directory: Path
) -> tuple[dict[str, object], Counter[Gemm]]:
    cache = KVCache(model, 4, torch.device("cuda"), batch_size=batch, dtype=torch.bfloat16)
    cache.positions.copy_(torch.arange(cache.capacity, device="cuda")[None])
    tokens = torch.randn(batch, 2, model.cfg.d_model, device="cuda", dtype=torch.bfloat16)
    observed = torch.zeros(batch, 4, device="cuda", dtype=torch.long)
    prefix = torch.zeros(batch, 10, 4, device="cuda", dtype=torch.long)
    delay = torch.full((batch,), 8 if count == 4 else 10, device="cuda", dtype=torch.long)
    replan = torch.full_like(delay, count)
    logs = torch.zeros(batch, device="cuda")
    uniforms = torch.full((count, 4, batch), 0.5, device="cuda", dtype=torch.float64)
    decoder = DecisionDecoder(model, ((8, 4), (10, 2)), compiled=False)

    def reset() -> None:
        cache.next_position.fill_(192)

    @torch.no_grad()
    def tick() -> DecisionOutput:
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            hidden = forward_tokens_with_kv_cache(model, tokens, cache)
        return decoder(
            hidden[:, -1:],
            observed,
            prefix,
            logs,
            delay,
            replan,
            uniforms,
            sampled_offsets=count,
            history=cache.memory(),
            validate=False,
        )

    reset()
    tick()
    shapes = profile_shapes(tick, directory / f"tick-b{batch}-r{count}.json")
    eager = measure(tick, reset=reset)
    compiled = torch.compile(tick, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
    with prepared_shape_compilation(64):
        reset()
        compiled()
        compiled_latency = measure(compiled, reset=reset)
        graphed = graph_call(compiled)
    graphed_latency = measure(graphed, reset=reset)
    result: dict[str, object] = {
        "batch": batch,
        "sampled_offsets": count,
        "appended_frames": 2,
        "query_position": 192,
        "eager": dataclasses.asdict(eager),
        "compiled": dataclasses.asdict(compiled_latency),
        "graphed": dataclasses.asdict(graphed_latency),
        "gemms": [{**dataclasses.asdict(shape), "calls": calls} for shape, calls in sorted(shapes.items())],
    }
    return result, shapes


def window_benchmark(
    model: SelfPlayTransformer, features: dict[str, Tensor], observed: Tensor, slots: ReplaySlots, directory: Path
) -> tuple[dict[str, object], Counter[Gemm]]:
    if observed.shape != (32, 256, 4) or slots.positions_WK.shape != (32, 16):
        raise ValueError("bare window probe requires recorded [32,256] histories and [32,16] actor slots")
    ctx_pad = torch.zeros(len(observed), device="cuda", dtype=torch.long)
    compiled_replay = torch.compile(
        replay_windows,
        fullgraph=True,
        dynamic=False,
        options={"triton.cudagraphs": False, "emulate_precision_casts": True},
    )

    def forward(*, shapes: bool = False) -> ReplayOutput:
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            hidden = model(features, ctx_pad, observed)
        operation = replay_windows if shapes else compiled_replay
        return operation(model, hidden, observed, slots, grad=True)

    def backward(*, shapes: bool = False) -> None:
        for parameter in model.actor_parameters():
            if parameter.grad is not None:
                parameter.grad.zero_()
        output = forward(shapes=shapes)
        # Fixed cotangents exercise the log-probability and distribution paths.
        loss = output.log_probs_WKF.sum() / output.log_probs_WKF.numel()
        loss = loss + sum(p.square().sum() / p.numel() for p in output.distributions)
        loss.backward()

    with prepared_shape_compilation(64):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            hidden = model(features, ctx_pad, observed)
            validate_replay_inputs(model, hidden, observed, slots)
        del hidden
        backward()
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.synchronize()
        actor = graph_call(backward)
        actor_latency = measure(actor)
        actor_shapes = profile_shapes(partial(backward, shapes=True), directory / "actor-step.json")
        # Reference matches the actor's autograd metadata and reuses its trunk.
        reference_slots = ReplaySlots(
            *(
                getattr(slots, field.name)
                if field.name == "history_starts_WL"
                else getattr(slots, field.name).repeat(1, 4, *([1] * (getattr(slots, field.name).ndim - 2)))
                for field in dataclasses.fields(slots)
            )
        )

        def reference(*, shapes: bool = False) -> None:
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                hidden = model(features, ctx_pad, observed)
            for start in range(0, 64, 16):
                part = ReplaySlots(
                    *(
                        getattr(reference_slots, field.name)
                        if field.name == "history_starts_WL"
                        else getattr(reference_slots, field.name)[:, start : start + 16].contiguous()
                        for field in dataclasses.fields(reference_slots)
                    )
                )
                operation = replay_windows if shapes else compiled_replay
                output = operation(model, hidden, observed, part, grad=True)
                del output

        reference_shapes = profile_shapes(partial(reference, shapes=True), directory / "reference-chunk.json")
        ref = graph_call(reference)
        ref_latency = measure(ref)
    return {
        "actor": dataclasses.asdict(actor_latency),
        "reference_chunk": dataclasses.asdict(ref_latency),
        "actor_gemms": [{**dataclasses.asdict(s), "calls": n} for s, n in sorted(actor_shapes.items())],
        "reference_gemms": [{**dataclasses.asdict(s), "calls": n} for s, n in sorted(reference_shapes.items())],
    }, actor_shapes + reference_shapes


def critic_benchmark(
    model: SelfPlayTransformer, count: int, directory: Path
) -> tuple[dict[str, object], Counter[Gemm]]:
    trunk = torch.randn(count, model.cfg.d_model, device="cuda")
    decoder = torch.randn(count, model.cfg.temporal_d_model, device="cuda")
    condition = torch.zeros(count, 3, device="cuda")
    queue = torch.zeros(count, model.critic_residual.max_queue, 4, device="cuda", dtype=torch.long)
    lengths = torch.full((count,), 8, device="cuda", dtype=torch.long)

    def backward() -> None:
        for parameter in model.critic_parameters():
            if parameter.grad is not None:
                parameter.grad.zero_()
        values = model.self_play_value(trunk, decoder, condition, condition, queue, lengths, validate=False)
        (values / 150).square().mean().backward()

    backward()
    shapes = profile_shapes(backward, directory / "critic-epoch.json")
    model.zero_grad(set_to_none=True)
    gc.collect()
    operation = graph_call(partial(repeat, backward, 4))
    return {
        "rows": count,
        "epochs": 4,
        "latency": dataclasses.asdict(measure(operation)),
        "gemms": [{**dataclasses.asdict(s), "calls": n * 4} for s, n in sorted(shapes.items())],
    }, shapes


def muon_benchmark(groups: list[OptimizerGroup], directory: Path) -> tuple[dict[str, object], Counter[Gemm]]:
    parameters = []
    for group in groups:
        tensors = [torch.nn.Parameter(torch.randn(shape, device="cuda") * 0.01) for shape in group["shapes"]]
        for tensor in tensors:
            tensor.grad = torch.randn_like(tensor) * 0.001
        entry: dict[str, object] = {"params": tensors, "lr": 1e-5, "weight_decay": 0, "use_muon": group["use_muon"]}
        if group["use_muon"]:
            entry.update(momentum=0.95, logical_splits=group["logical_splits"], muon_scale_clamp_min_one=False)
        else:
            entry.update(betas=(0.9, 0.95), eps=1e-8)
        parameters.append(entry)
    optimizer = SingleDeviceMuonWithAuxAdam(parameters)
    optimizer.step()
    shapes = profile_shapes(optimizer.step, directory / "muon-step.json")
    eager = measure(optimizer.step)
    captured = measure(graph_call(optimizer.step))
    return {
        "eager": dataclasses.asdict(eager),
        "fixed_step_graph": dataclasses.asdict(captured),
        "gemms": [{**dataclasses.asdict(s), "calls": n} for s, n in sorted(shapes.items())],
    }, shapes


@dataclasses.dataclass(frozen=True, slots=True)
class Args:
    checkpoint: Path
    inputs: Path
    layout: Path
    output: Path


def main(args: Args) -> None:
    if sha256(args.checkpoint) != PROXY_SHA256:
        raise ValueError("floor measurements require the recorded O63 proxy")
    args.output.mkdir(parents=True, exist_ok=False)
    layout = read_layout(args.layout)
    raw = torch.load(args.inputs, weights_only=False, map_location="cuda")
    if (
        not isinstance(raw, dict)
        or set(raw) != {"format", "features", "observed", "slots", "model"}
        or raw["format"] != "hal.o65.floor-inputs.v1"
    ):
        raise ValueError("unsupported floor model inputs")
    torch.set_num_threads(2)
    torch.manual_seed(65)
    torch.backends.cuda.matmul.allow_tf32 = False
    loaded = load_evaluation_checkpoint(args.checkpoint, device=torch.device("cuda"), decoder_precision="bf16")
    model = loaded.model.eval().requires_grad_(True)
    model.percentile_head.requires_grad_(False)
    model.load_state_dict(raw["model"], strict=True)
    result: dict[str, object] = {
        "format": "hal.o65.floor-model.v1",
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "source_sha256": sha256(Path(__file__)),
        "checkpoint_sha256": PROXY_SHA256,
        "inputs_sha256": sha256(args.inputs),
        "layout_sha256": sha256(args.layout),
        "config": dataclasses.asdict(model.cfg),
        "launch": hardware_launches(),
        "memory": hardware_memory(),
        "tick_input": "BF16 trunk tokens; KV at position 192; queue and draws synthetic; shapes from warmed trainer",
        "actor_backward": "fixed scalar cotangents on log-probabilities and full distributions; no PPO or optimizer",
        "muon_graph": "fixed bias-correction step; performance probe only, never used for training",
    }
    (args.output / "rates.json").write_text(json.dumps(result, indent=2))
    shapes: set[Gemm] = set()
    ticks = []
    for batch in sorted({int(row["batch"]) for row in layout["tick_layouts"]}):
        for count in (2, 4):
            print(f"bare tick batch={batch} R={count}", flush=True)
            row, observed_shapes = tick_benchmark(model, batch, count, args.output)
            ticks.append(row)
            shapes.update(observed_shapes)
            result["ticks"] = ticks
            (args.output / "rates.json").write_text(json.dumps(result, indent=2))
            gc.collect()
            torch.cuda.empty_cache()
    windows, window_shapes = window_benchmark(
        model, cast(dict[str, Tensor], raw["features"]), raw["observed"], ReplaySlots(**raw["slots"]), args.output
    )
    shapes.update(window_shapes)
    result["windows"] = windows
    critic, critic_shapes = critic_benchmark(model, int(layout["decisions"]), args.output)
    result["critic"] = critic
    shapes.update(critic_shapes)
    muon, muon_shapes = muon_benchmark(layout["optimizer_groups"], args.output)
    result["muon"] = muon
    shapes.update(muon_shapes)
    result["gemms"] = hardware_gemms(shapes)
    if not all(math.isfinite(row["tflops_per_second"]) for row in cast(list[dict[str, float]], result["gemms"])):
        raise ValueError("non-finite measured hardware rate")
    (args.output / "rates.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print("bare model and hardware complete", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
