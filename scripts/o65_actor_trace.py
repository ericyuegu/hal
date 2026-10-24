"""Trace replay trunk and decoder work on one real, compact actor chunk."""

import dataclasses
import json
import os
from pathlib import Path

import torch
import tyro
from torch.profiler import ProfilerActivity
from torch.profiler import record_function

from hal.eval.self_play_checkpoint import load_evaluation_checkpoint
from hal.eval.self_play_checkpoint import sha256
from hal.inference.column_context import PackedColumnContextBuilder
from hal.inference.graphed_actor import GraphedActorStep
from hal.models.attention import Trunk
from hal.models.replay_actor import ActorTargets
from hal.models.replay_actor import actor_loss
from hal.models.window_replay import ReplayOutput
from hal.models.window_replay import ReplaySlots
from hal.models.window_replay import replay_windows
from hal.sim import native
from hal.wire import ACTION_DIM
from scripts.o65_replay_probe import load_cohort


@dataclasses.dataclass(frozen=True, slots=True)
class Args:
    inputs: Path
    checkpoint: Path = Path("/o65-results/o64-inputs/step-0098304.pt")
    cohort: int = 0
    windows: int = 64
    compiled_trunk: bool = True
    output: Path = Path("/cache/train-actor-split")


def actor_history(trunk: Trunk, tokens: torch.Tensor) -> torch.Tensor:
    """Match the trainer's compiled history boundary, including zero padding."""
    with torch.autocast(tokens.device.type, dtype=torch.bfloat16, enabled=tokens.is_cuda, cache_enabled=False):
        return trunk(tokens, torch.zeros(len(tokens), dtype=torch.long, device=tokens.device))


def stage_seconds(path: Path) -> dict[str, float]:
    events = json.loads(path.read_text())["traceEvents"]
    stages = [
        event
        for event in events
        if event.get("cat") == "user_annotation" and event.get("name", "").startswith("o65_actor/")
    ]
    cpu = {event.get("args", {}).get("External id"): event for event in events if event.get("cat") == "cpu_op"}
    totals: dict[str, float] = {}
    for event in events:
        if event.get("cat") != "kernel":
            continue
        launch = cpu.get(event.get("args", {}).get("External id"))
        if launch is None:
            raise ValueError("actor trace kernel has no CPU launch attribution")
        owners = [stage for stage in stages if stage["ts"] <= launch["ts"] <= stage["ts"] + stage["dur"]]
        if len(owners) != 1:
            raise ValueError("actor trace kernel has ambiguous phase attribution")
        name = owners[0]["name"].removeprefix("o65_actor/")
        totals[name] = totals.get(name, 0.0) + event["dur"] / 1e6
    return totals


def main(args: Args) -> None:
    if args.windows < 1:
        raise ValueError("actor trace requires a positive window count")
    torch.manual_seed(65)
    device = torch.device("cuda", torch.cuda.current_device())
    checkpoint = load_evaluation_checkpoint(args.checkpoint, device=device)
    model = checkpoint.model.eval().requires_grad_(True)
    model.percentile_head.requires_grad_(False)
    model.trunk.resolve_attention("cuda")
    saved = load_cohort(args.inputs, checkpoint.checkpoint_sha256, args.cohort)
    original = ReplaySlots(*(value.to(device) for value in saved["inputs"][5:]))
    count = min(args.windows, len(original.valid_WK))
    ids = torch.nn.functional.pad(torch.arange(count, device=device), (0, args.windows - count))
    capacity = original.valid_WK.shape[1]
    fields = []
    for field in dataclasses.fields(original):
        value = getattr(original, field.name)[ids]
        if field.name == "valid_WK":
            value = value & (torch.arange(args.windows, device=device) < count)[:, None]
        fields.append(value)
    slots = ReplaySlots(*fields)
    if not bool(slots.valid_WK.any()):
        raise ValueError("actor trace needs valid recorded decisions")
    configurations = sorted(
        set(zip(slots.delay_WK[slots.valid_WK].tolist(), slots.replan_WK[slots.valid_WK].tolist(), strict=True))
    )
    context = {name: value.to(device)[ids] for name, value in saved["context"].items()}
    observed = saved["observed"].to(device)[ids]
    selected = saved["selected"].to(device)[ids, :capacity] & slots.valid_WK
    mean = saved["advantage_mean"]
    sigma = saved["advantage_sigma"]
    trunk = (
        torch.compile(
            actor_history,
            fullgraph=True,
            dynamic=False,
            options={"triton.cudagraphs": False, "emulate_precision_casts": True},
        )
        if args.compiled_trunk
        else actor_history
    )
    decoder = torch.compile(
        replay_windows,
        fullgraph=True,
        dynamic=False,
        options={"triton.cudagraphs": False, "emulate_precision_casts": True},
    )
    parameters = model.actor_parameters()
    with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        tokens = model.context_tokens(context, observed)
        hidden = trunk(model.trunk, tokens)
    reference = decoder(model, hidden, observed, slots, grad=True)
    targets = ActorTargets(
        saved["behavior"].to(device)[ids, :capacity],
        saved["advantages"].to(device)[ids, :capacity],
        selected,
        saved["coefficients"].to(device)[ids, :capacity],
        tuple(p.detach().clone() for p in reference.distributions),
        mean.to(device).float(),
        sigma.to(device).float(),
        saved["denominator"].to(device),
    )
    del reference, hidden, tokens

    def step() -> None:
        with (
            record_function("o65_actor/observation"),
            torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False),
        ):
            tokens = model.context_tokens(context, observed)
        with record_function("o65_actor/trunk_forward"):
            hidden = trunk(model.trunk, tokens)
        # Split backward at the history boundary to attribute GPU work. Shared
        # codec parameters receive contributions from both passes in training.
        leaf = hidden.detach().requires_grad_(True)
        with record_function("o65_actor/decoder_forward"):
            output = decoder(model, leaf, observed, slots, grad=True)
        with record_function("o65_actor/loss"):
            loss = actor_loss(output, targets)
        with record_function("o65_actor/decoder_backward"):
            decoder_gradients = torch.autograd.grad(loss.loss, (*parameters, leaf), allow_unused=True)
        with record_function("o65_actor/trunk_backward"):
            trunk_gradients = torch.autograd.grad(hidden, parameters, decoder_gradients[-1], allow_unused=True)
        with record_function("o65_actor/validation"):
            for value in (*decoder_gradients, *trunk_gradients):
                if value is not None and not bool(torch.isfinite(value).all()):
                    raise ValueError("actor trace gradient contains NaN or infinity")

    # Compile before profiling; the trace measures steady state kernels.
    step()
    torch.cuda.synchronize()
    trace = args.output.with_suffix(".json")
    with torch.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profile:
        step()
        torch.cuda.synchronize()
    profile.export_chrome_trace(str(trace))
    seconds = stage_seconds(trace)
    # External event nodes survive replay. Hooks mark the history gradient
    # boundary in the actor's single backward, including shared codec weights.
    events = tuple(torch.cuda.Event(enable_timing=True, external=True) for _ in range(6))

    def decoder_backward_done(gradient: torch.Tensor) -> torch.Tensor:
        events[4].record()
        return gradient

    def trunk_backward_done(gradient: torch.Tensor) -> torch.Tensor:
        events[5].record()
        return gradient

    builder = PackedColumnContextBuilder(checkpoint.statistics, native.OBSERVATION_FIELDS, device)
    inputs = (
        *(value.to(device)[ids] for value in saved["inputs"][:5]),
        *(getattr(slots, field.name) for field in dataclasses.fields(slots)),
    )

    def replay_forward(values: tuple[torch.Tensor, ...]) -> ReplayOutput:
        events[0].record()
        records, actions, sides, identities, lengths = values[:5]
        current_context = builder(records, actions, sides, identities, lengths)
        length = records.shape[1]
        ego = actions.gather(2, sides[:, None, None, None].expand(-1, length, 1, ACTION_DIM)).squeeze(2)
        valid = torch.arange(length, device=device)[None] < lengths[:, None]
        current_observed = torch.where(valid[..., None], model.codec.quantize(ego), 0)
        with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            tokens = model.context_tokens(current_context, current_observed)
        events[1].record()
        with torch.enable_grad():
            hidden = trunk(model.trunk, tokens)
            hidden.register_hook(decoder_backward_done)
            tokens.register_hook(trunk_backward_done)
            events[2].record()
            output = decoder(model, hidden, current_observed, ReplaySlots(*values[5:]), grad=True)
        events[3].record()
        return output

    graph = GraphedActorStep(replay_forward, inputs, targets, parameters)
    graph.captured.graph.replay()
    torch.cuda.synchronize()
    start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    replay_samples: list[dict[str, float]] = []
    for _ in range(10):
        start.record()
        graph.captured.graph.replay()
        stop.record()
        stop.synchronize()
        spans = {
            name: events[first].elapsed_time(events[last]) / 1e3
            for name, first, last in (
                ("observation_forward", 0, 1),
                ("trunk_forward", 1, 2),
                ("decoder_forward", 2, 3),
                ("loss_and_decoder_backward", 3, 4),
                ("trunk_backward", 4, 5),
            )
        }
        total = start.elapsed_time(stop) / 1e3
        replay_samples.append({**spans, "other": total - sum(spans.values()), "total": total})
    replay_trace = args.output.with_suffix(".replay.json")
    with torch.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profile:
        graph.captured.graph.replay()
        torch.cuda.synchronize()
    profile.export_chrome_trace(str(replay_trace))
    if any(not bool(torch.isfinite(value).all()) for value in graph.gradients):
        raise ValueError("actor graph gradient contains NaN or infinity")
    report = {
        "format": "hal.o65.actor-split.v2",
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "inputs_sha256": sha256(args.inputs),
        "probe_sha256": sha256(Path(__file__)),
        "cohort": args.cohort,
        "windows": args.windows,
        "live_windows": int(slots.valid_WK.any(-1).sum()),
        "decisions": capacity,
        "queue_capacity": slots.forced_queue_WKD4.shape[2],
        "configurations": configurations,
        "compiled_trunk": args.compiled_trunk,
        "seed": 65,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "inductor_pdl": os.environ.get("TORCHINDUCTOR_ENABLE_PDL"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "gpu": torch.cuda.get_device_name(device),
        "kernel_seconds": seconds,
        "trace": str(trace),
        "graph_replay_seconds": {
            name: sum(sample[name] for sample in replay_samples) / len(replay_samples) for name in replay_samples[0]
        },
        "graph_replay_samples": replay_samples,
        "graph_replay_trace": str(replay_trace),
        "graph_replay_boundaries": "CUDA events and history gradient hooks in one actor backward; observation includes packed context construction; decoder backward includes loss; other includes gradient zeroing and observation backward; leaf accumulation can occur in either backward span or other",
        "measurement_scope": "one actor graph with private gradients/pool; partial windows repeat the first context with invalid slots; the diagnostic trace includes observation backward in trunk_backward, while graph timing puts it in other",
    }
    args.output.with_suffix(".report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
