"""Compare fixed-weight actor precision with disjoint minibatch gradients."""

import dataclasses
import json
import math
import os
from pathlib import Path
from typing import Literal
from typing import TypedDict
from typing import cast

import torch
import tyro
from torch import Tensor

from hal.eval.self_play_checkpoint import load_evaluation_checkpoint
from hal.eval.self_play_checkpoint import sha256
from hal.models.replay_actor import ActorTargets
from hal.models.replay_actor import actor_loss
from hal.models.window_replay import ReplaySlots
from hal.models.window_replay import replay_windows
from scripts.o65_replay_probe import fp32_copy
from scripts.o65_replay_probe import history


@dataclasses.dataclass(frozen=True, slots=True)
class Args:
    inputs: Path
    checkpoint: Path = Path("/o65-results/o64-inputs/step-0098304.pt")
    output: Path = Path("/cache/o65-gradient-noise.json")
    modes: tuple[Literal["fp32", "eager", "compiled"], ...] = ("fp32", "eager", "compiled")


type TargetValues = tuple[Tensor, Tensor, Tensor, Tensor, tuple[Tensor, ...], Tensor, Tensor, Tensor]


class GradientChunk(TypedDict):
    format: str
    step: int
    context: dict[str, Tensor]
    observed: Tensor
    slots: tuple[Tensor, ...]
    targets: TargetValues


class GradientManifest(TypedDict):
    format: str
    provenance: dict[str, str]
    trainer_source_sha256: str
    training: dict[str, object]
    steps: list[list[int]]
    chunks: dict[str, str]
    advantage_mean: float
    advantage_sigma: float
    denominator: float


def load_manifest(directory: Path, checkpoint_sha256: str) -> GradientManifest:
    saved = json.loads((directory / "manifest.json").read_text())
    if (
        not isinstance(saved, dict)
        or set(saved) != set(GradientManifest.__annotations__)
        or saved["format"] != "hal.o65.update-gradient-inputs.v1"
        or saved["provenance"]["checkpoint_sha256"] != checkpoint_sha256
    ):
        raise ValueError("update gradient input schema or checkpoint mismatch")
    steps = saved["steps"]
    if len(steps) < 2 or not all(steps) or not saved["chunks"]:
        raise ValueError("gradient noise requires two occupied optimizer batches")
    windows = [window for step in steps for window in step]
    if any(type(window) is not int or window < 0 for window in windows) or len(windows) != len(set(windows)):
        raise ValueError("gradient noise requires disjoint window batches")
    if saved["training"]["clip_epsilon"] != 0.2:
        raise ValueError("gradient noise requires clip_epsilon=0.2")
    if not all(math.isfinite(saved[name]) for name in ("advantage_mean", "advantage_sigma", "denominator")) or any(
        saved[name] <= 0 for name in ("advantage_sigma", "denominator")
    ):
        raise ValueError("invalid update advantage statistics")
    for name, digest in saved["chunks"].items():
        if Path(name).name != name or sha256(directory / name) != digest:
            raise ValueError("gradient chunk identity mismatch")
    return cast(GradientManifest, saved)


def load_chunk(path: Path, steps: int) -> GradientChunk:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(saved, dict)
        or set(saved) != set(GradientChunk.__annotations__)
        or saved["format"] != "hal.o65.update-gradient-chunk.v1"
        or type(saved["step"]) is not int
        or not 0 <= saved["step"] < steps
        or len(saved["slots"]) != len(dataclasses.fields(ReplaySlots))
        or len(saved["targets"]) != len(dataclasses.fields(ActorTargets))
    ):
        raise ValueError("gradient chunk schema mismatch")
    return cast(GradientChunk, saved)


def comparison(first: Tensor, second: Tensor) -> dict[str, float | None]:
    a, b = first.double(), second.double()
    if not bool(torch.isfinite(a).all() & torch.isfinite(b).all()):
        raise ValueError("gradient contains NaN or infinity")
    an, bn = float(a.norm()), float(b.norm())
    return {
        "cosine": float(torch.dot(a, b)) / (an * bn) if an and bn else None,
        "relative_l2": float((a - b).norm()) / bn if bn else None,
        "first_norm": an,
        "second_norm": bn,
    }


def accumulate(
    objective: Tensor, parameters: tuple[torch.nn.Parameter, ...], total: Tensor, *, retain_graph: bool
) -> None:
    values = torch.autograd.grad(objective, parameters, allow_unused=True, retain_graph=retain_graph)
    start = 0
    for parameter, value in zip(parameters, values, strict=True):
        if value is not None:
            total[start : start + parameter.numel()].add_(value.flatten())
        start += parameter.numel()


def main(args: Args) -> None:
    torch.set_num_threads(2)
    torch.manual_seed(65)
    device = torch.device("cuda", torch.cuda.current_device())
    checkpoint = load_evaluation_checkpoint(args.checkpoint, device=device)
    manifest = load_manifest(args.inputs, checkpoint.checkpoint_sha256)
    model = checkpoint.model.eval().requires_grad_(True)
    model.percentile_head.requires_grad_(False)
    model.trunk.resolve_attention("cuda")
    high = fp32_copy(model).requires_grad_(True)
    high.percentile_head.requires_grad_(False)
    compiled_history = torch.compile(
        history, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False, "emulate_precision_casts": True}
    )
    compiled_replay = torch.compile(
        replay_windows,
        fullgraph=True,
        dynamic=False,
        options={"triton.cudagraphs": False, "emulate_precision_casts": True},
    )
    parameters = model.actor_parameters()
    size = sum(parameter.numel() for parameter in parameters)
    gradients: dict[str, Tensor] = {}
    counts = [0] * len(manifest["steps"])
    for mode in args.modes:
        current = high if mode == "fp32" else model
        current_parameters = current.actor_parameters()
        batch_gradients = [torch.zeros(size, device=device) for _ in range(2)]
        whole = torch.zeros(size, device=device) if mode == "fp32" else None
        for index, name in enumerate(manifest["chunks"]):
            saved = load_chunk(args.inputs / name, len(counts))
            step = saved["step"]
            if mode != "fp32" and step >= 2:
                continue
            context = {key: value.to(device) for key, value in saved["context"].items()}
            observed = saved["observed"].to(device)
            slots = ReplaySlots(*(value.to(device) for value in saved["slots"]))
            target_values = tuple(
                tuple(item.to(device) for item in value) if isinstance(value, tuple) else value.to(device)
                for value in saved["targets"]
            )
            targets = ActorTargets(*cast(TargetValues, target_values))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=mode != "fp32", cache_enabled=False):
                tokens = current.context_tokens(context, observed)
            if mode == "fp32":
                hidden = current.trunk.forward_dense(
                    tokens, torch.zeros(len(tokens), dtype=torch.long, device=device), force_dense=True
                )
            else:
                hidden = (compiled_history if mode == "compiled" else history)(current.trunk, tokens)
            operation = compiled_replay if mode == "compiled" else replay_windows
            output = operation(current, hidden, observed, slots, grad=True)

            if step < 2:
                accumulate(
                    actor_loss(output, targets).loss,
                    current_parameters,
                    batch_gradients[step],
                    retain_graph=whole is not None,
                )
            if whole is not None:
                whole_targets = dataclasses.replace(
                    targets,
                    advantage_mean=torch.tensor(manifest["advantage_mean"], device=device),
                    advantage_sigma=torch.tensor(manifest["advantage_sigma"], device=device),
                    denominator=torch.tensor(manifest["denominator"], device=device),
                )
                accumulate(actor_loss(output, whole_targets).loss, current_parameters, whole, retain_graph=False)
                counts[step] += int(targets.selected_WK.sum())
            del output, hidden, tokens
            if index % 8 == 0:
                print(f"gradient mode={mode} chunk={index + 1}/{len(manifest['chunks'])}", flush=True)
        for index, gradient in enumerate(batch_gradients):
            gradients[f"{mode}_batch{index}"] = gradient.cpu()
        if whole is not None:
            gradients["fp32_whole"] = whole.cpu()
    if "fp32" not in args.modes:
        raise ValueError("gradient noise report requires FP32")
    pairs = [("fp32_batch0", "fp32_batch1"), ("fp32_batch0", "fp32_whole")]
    for mode in ("eager", "compiled"):
        if mode in args.modes:
            pairs.extend((f"{mode}_batch{step}", f"fp32_batch{step}") for step in range(2))
    groups: dict[str, list[tuple[int, int]]] = {}
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    start = 0
    for parameter in parameters:
        path = names[id(parameter)].split(".")
        group = ".".join(path[:3]) if path[0] in ("trunk", "temporal") else path[0]
        groups.setdefault(group, []).append((start, start + parameter.numel()))
        start += parameter.numel()
    report = {
        "format": "hal.o65.gradient-noise.v1",
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "manifest_sha256": sha256(args.inputs / "manifest.json"),
        "probe_sha256": sha256(Path(__file__)),
        "environment": {
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "tf32": torch.backends.cuda.matmul.allow_tf32,
            "gpu": torch.cuda.get_device_name(),
            "inductor_cache": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        },
        "scope": "Checkpoint weights, stored behavior and reference; no optimizer, no gradient clipping. "
        "Each minibatch uses its production advantage moments; the whole update uses global moments.",
        "windows_per_batch": [len(step) for step in manifest["steps"]],
        "decisions_per_batch": counts,
        "comparisons": {f"{a}_vs_{b}": comparison(gradients[a], gradients[b]) for a, b in pairs},
        "groups": {
            group: {
                f"{a}_vs_{b}": comparison(
                    torch.cat([gradients[a][first:last] for first, last in spans]),
                    torch.cat([gradients[b][first:last] for first, last in spans]),
                )
                for a, b in pairs
            }
            for group, spans in groups.items()
        },
    }
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["comparisons"], indent=2), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
