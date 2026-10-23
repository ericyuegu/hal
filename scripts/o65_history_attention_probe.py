"""Compare shared FlexAttention with repeated variable-length history memory."""

import dataclasses
import gc
import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import torch
import tyro
from torch import Tensor
from torch.nn.attention.flex_attention import flex_attention
from torch.nn.attention.varlen import varlen_attn

from hal.inference.cuda_graph import CapturedCall


@dataclasses.dataclass(frozen=True, slots=True)
class Args:
    windows: int = 64
    slots: int = 16
    output: Path = Path("/cache/train-history-attention-probe.json")


def shared(query: Tensor, key: Tensor, value: Tensor, starts: Tensor, positions: Tensor) -> Tensor:
    offsets = query.shape[2] // positions.shape[1]
    lower = starts.repeat_interleave(offsets, 1)
    upper = positions.repeat_interleave(offsets, 1)

    def mask(score: Tensor, b: Tensor, h: Tensor, q: Tensor, kv: Tensor) -> Tensor:
        return torch.where((kv >= lower[b, q]) & (kv <= upper[b, q]), score, -torch.inf)

    return cast(Tensor, flex_attention(query, key, value, score_mod=mask))


def repeated(query: Tensor, key: Tensor, value: Tensor, starts: Tensor, positions: Tensor) -> Tensor:
    windows, heads, queries, width = query.shape
    slots = positions.shape[1]
    offsets = queries // slots
    length = key.shape[2]
    rows = windows * slots
    packed_query = (
        query.reshape(windows, heads, slots, offsets, width).permute(0, 2, 3, 1, 4).reshape(-1, heads, width)
    )
    packed_key = key.transpose(1, 2)[:, None].expand(-1, slots, -1, -1, -1).reshape(-1, heads, width)
    packed_value = value.transpose(1, 2)[:, None].expand(-1, slots, -1, -1, -1).reshape(-1, heads, width)
    key_base = torch.arange(rows, device=query.device, dtype=torch.int32) * length
    query_base = torch.arange(rows, device=query.device, dtype=torch.int32) * offsets
    # Each memory row has an ignored prefix, one visible interval and an ignored
    # suffix. Only the visible interval gets queries; no mask or data-dependent
    # packing is needed.
    cu_key = torch.cat(
        (
            torch.stack((key_base, key_base + starts.flatten(), key_base + positions.flatten() + 1), -1).flatten(),
            key_base.new_full((1,), rows * length),
        )
    ).to(torch.int32)
    cu_query = torch.cat(
        (
            torch.stack((query_base, query_base, query_base + offsets), -1).flatten(),
            query_base.new_full((1,), rows * offsets),
        )
    )
    result = cast(Tensor, varlen_attn(packed_query, packed_key, packed_value, cu_query, cu_key, offsets, length))
    return result.reshape(windows, slots, offsets, heads, width).permute(0, 3, 1, 2, 4).reshape_as(query)


def main(args: Args) -> None:
    torch.manual_seed(65)
    device = torch.device("cuda", torch.cuda.current_device())
    shape = (args.windows, 16, args.slots * 25, 64)
    query = torch.randn(shape, dtype=torch.bfloat16, device=device, requires_grad=True)
    key = torch.randn(args.windows, 16, 256, 64, dtype=torch.bfloat16, device=device, requires_grad=True)
    value = torch.randn_like(key, requires_grad=True)
    positions = torch.randint(0, 256, (args.windows, args.slots), device=device)
    starts = (positions - torch.randint(0, 256, positions.shape, device=device)).clamp_min(0)
    upstream = torch.randn_like(query).float()
    results = []
    timings = {}
    for name, implementation in (("shared", shared), ("repeated", repeated)):
        compiled = torch.compile(
            implementation,
            fullgraph=True,
            dynamic=False,
            options={"triton.cudagraphs": False, "emulate_precision_casts": True},
        )

        def backward(operation: Callable[..., Tensor] = compiled) -> tuple[Tensor, ...]:
            with torch.enable_grad():
                output = operation(query, key, value, starts, positions)
                gradients = torch.autograd.grad((output.float() * upstream).mean(), (query, key, value))
            return output, *gradients

        print(f"prepare {name}", flush=True)
        graph = CapturedCall(backward, (query, key, value, starts, positions, upstream), device=device)
        graph.graph.replay()
        results.append(tuple(t.detach().clone() for t in graph.output))
        start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(20):
            graph.graph.replay()
        stop.record()
        stop.synchronize()
        timings[name] = start.elapsed_time(stop) / 20
        del graph
        gc.collect()
    report = {"milliseconds": timings, "windows": args.windows, "slots": args.slots, "errors": []}
    for candidate, baseline in zip(results[1], results[0], strict=True):
        delta = candidate.float() - baseline.float()
        report["errors"].append(
            {
                "max": float(delta.abs().max()),
                "rmse": float(delta.square().mean().sqrt()),
                "relative_l2": float(delta.norm() / baseline.float().norm().clamp_min(1e-30)),
            }
        )
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
