"""Compare eager and compiled O64 replay trunks on identical synthetic tokens."""

import dataclasses
import gc
import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import torch
import tyro
from torch import Tensor

from hal.eval.self_play_checkpoint import load_evaluation_checkpoint
from hal.inference.cuda_graph import CapturedCall


@dataclasses.dataclass(frozen=True, slots=True)
class Args:
    checkpoint: Path = Path("/o65-results/o64-inputs/step-0098304.pt")
    windows: int = 64
    output: Path = Path("/cache/train-trunk-probe.json")


def errors(candidate: Tensor, control: Tensor) -> dict[str, float]:
    reference = control.detach().float()
    delta = candidate.detach().float() - reference
    return {
        "max": float(delta.abs().max()),
        "rmse": float(delta.square().mean().sqrt()),
        "relative_l2": float(delta.norm() / reference.norm().clamp_min(1e-30)),
    }


def main(args: Args) -> None:
    torch.manual_seed(65)
    device = torch.device("cuda", torch.cuda.current_device())
    checkpoint = load_evaluation_checkpoint(args.checkpoint, device=device)
    model = checkpoint.model.eval()
    model.requires_grad_(True)
    model.percentile_head.requires_grad_(False)
    trunk = model.trunk
    trunk.resolve_attention("cuda")
    tokens = torch.randn(args.windows, 256, model.cfg.d_model, dtype=torch.bfloat16, device=device)
    padding = torch.zeros(args.windows, dtype=torch.long, device=device)
    upstream = torch.randn_like(tokens)
    compiled = torch.compile(
        trunk,
        fullgraph=True,
        dynamic=False,
        options={"triton.cudagraphs": False, "emulate_precision_casts": True},
    )
    outputs = []
    gradients = []
    timings = {}
    for name, operation in (("eager", trunk), ("compiled", compiled)):

        def backward(operation: Callable[[Tensor, Tensor], Tensor] = operation) -> Tensor:
            model.zero_grad(set_to_none=True)
            with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                output = operation(tokens, padding)
                (output.float() * upstream.float()).mean().backward()
            return output

        print(f"prepare {name}", flush=True)
        graph = CapturedCall(backward, (tokens, padding, upstream), device=device)
        graph((tokens, padding, upstream))
        start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(20):
            graph.graph.replay()
        stop.record()
        stop.synchronize()
        timings[name] = start.elapsed_time(stop) / 20
        del graph
        model.zero_grad(set_to_none=True)
        with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            output = operation(tokens, padding)
            gradient = torch.autograd.grad((output.float() * upstream.float()).mean(), tuple(trunk.parameters()))
        outputs.append(output.detach().clone())
        gradients.append(torch.cat([value.flatten() for value in gradient]))
        del output, gradient
        gc.collect()
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        high = trunk.forward_dense(tokens.float(), padding, force_dense=True)
        (high * upstream.float()).mean().backward()
    high_gradient = torch.cat([p.grad.flatten() for p in trunk.parameters() if p.grad is not None])
    for tensor in (*outputs, *gradients, high, high_gradient):
        if not bool(torch.isfinite(tensor).all()) or not bool(tensor.norm() > 0):
            raise ValueError("trunk qualification requires finite, nonzero outputs and gradients")
    report = {
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "seed": 65,
        "windows": args.windows,
        "milliseconds": timings,
        "hidden": errors(outputs[1], outputs[0]),
        "gradients": errors(gradients[1], gradients[0]),
        "eager_fp32_hidden": errors(outputs[0], high),
        "compiled_fp32_hidden": errors(outputs[1], high),
        "eager_fp32_gradients": errors(gradients[0], high_gradient),
        "compiled_fp32_gradients": errors(gradients[1], high_gradient),
    }
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
