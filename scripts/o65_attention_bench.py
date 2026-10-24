"""Compare cached trunk attention tiles on fixed BF16 inputs and CUDA graphs."""

import dataclasses
import functools
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F
import tyro
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask
from torch.nn.attention.flex_attention import FlexKernelOptions
from torch.nn.attention.flex_attention import flex_attention

from hal.inference.cuda_graph import CapturedCall
from hal.models.attention import prefix_attention


@dataclasses.dataclass(frozen=True, slots=True)
class Args:
    output: Path = Path("/tmp/o65-attention-bench.json")
    rows: tuple[int, ...] = (1024, 128, 64)
    repeats: int = 100

    def __post_init__(self) -> None:
        if not self.rows or any(rows < 1 for rows in self.rows) or self.repeats < 1:
            raise ValueError("row counts and repeats must be positive")


def attention(query: Tensor, key: Tensor, value: Tensor, lengths: Tensor, block: int, splits: int) -> Tensor:
    batch, _, queries, _ = query.shape
    capacity = key.shape[2]

    def keep(b: Tensor, h: Tensor, q: Tensor, kv: Tensor) -> Tensor:
        return kv < lengths[b] - queries + q + 1

    counts = (lengths.clamp(0, capacity) + block - 1) // block
    indices = torch.arange((capacity + block - 1) // block, device=query.device, dtype=torch.int32)
    mask = BlockMask.from_kv_blocks(
        counts.to(torch.int32)[:, None, None],
        indices[None, None, None].expand(batch, 1, 1, -1),
        BLOCK_SIZE=(128, block),
        mask_mod=keep,
        seq_lengths=(queries, capacity),
        compute_q_blocks=False,
    )
    return cast(
        Tensor,
        flex_attention(
            query,
            key,
            value,
            block_mask=mask,
            kernel_options=cast(FlexKernelOptions, {"BLOCK_N": block, "SPLIT_KV": splits}),
        ),
    )


@torch.no_grad()
def main(args: Args) -> None:
    # SPLIT_KV is an untyped Torch 2.11 tuning option used only by this probe.
    if str(torch.__version__).split("+", 1)[0] != "2.11.0":
        raise RuntimeError("attention tuning probe requires tested Torch 2.11.0")
    torch.set_num_threads(2)
    torch.manual_seed(65)
    report: list[dict[str, int | float]] = []
    for rows in args.rows:
        # Match the trunk's interleaved Q projection and contiguous per-head KV cache.
        query = torch.randn(rows, 2, 16, 64, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
        key, value = torch.randn(2, rows, 16, 259, 64, device="cuda", dtype=torch.bfloat16)
        lengths = torch.full((rows,), 256, device="cuda", dtype=torch.long)
        for block, splits in ((64, 1), (64, 0), (128, 1), (256, 1), (64, 2), (64, 4)):
            operation = (
                functools.partial(prefix_attention, causal=True)
                if splits == 0
                else functools.partial(attention, block=block, splits=splits)
            )
            compiled = torch.compile(operation, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
            started = time.perf_counter()
            graph = CapturedCall(
                functools.partial(compiled, query, key, value, lengths),
                (query, key, value, lengths),
                device=torch.device("cuda", 0),
            )
            for test_lengths in (torch.arange(rows, device="cuda") % 260, torch.full_like(lengths, 256)):
                lengths.copy_(test_lengths)
                limits = lengths[:, None] - 2 + torch.arange(1, 3, device="cuda")[None]
                mask = torch.arange(259, device="cuda")[None, None] < limits[..., None]
                expected = F.scaled_dot_product_attention(query, key, value, attn_mask=mask[:, None])
                # CUDA SDPA can produce nonzero values for fully masked rows.
                # Match the explicit zero contract in prefix_attention's reference.
                expected = torch.where((limits > 0)[:, None, :, None], expected, 0)
                graph.graph.replay()
                print(
                    json.dumps(
                        {
                            "qualification": [rows, block, splits],
                            "max_valid_error": ((graph.output - expected).abs() * (limits > 0)[:, None, :, None])
                            .max()
                            .item(),
                            "max_empty_output": (graph.output.abs() * (limits <= 0)[:, None, :, None]).max().item(),
                        }
                    ),
                    flush=True,
                )
                torch.testing.assert_close(graph.output, expected, atol=1.6e-2, rtol=1.6e-2)
            capture_seconds = time.perf_counter() - started
            timings = []
            for _ in range(args.repeats):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                graph.graph.replay()
                end.record()
                timings.append((start, end))
            torch.cuda.synchronize()
            elapsed = statistics.median(start.elapsed_time(end) for start, end in timings)
            row = {
                "rows": rows,
                "block": block,
                "splits": splits,
                "ms": elapsed,
                "tb_s": rows * 16 * 256 * 64 * 2 * 2 / (elapsed / 1000) / 1e12,
                "capture_seconds": capture_seconds,
                "max_abs_error": (graph.output - expected).abs().max().item(),
            }
            print(json.dumps(row), flush=True)
            report.append(row)
            args.output.write_text(
                json.dumps(
                    {
                        "format": "hal.o65.attention-bench.v1",
                        "device": torch.cuda.get_device_name(),
                        "torch": str(torch.__version__),
                        "seed": 65,
                        "source_sha256": {
                            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in (Path(__file__), Path(__file__).parents[1] / "hal/models/attention.py")
                        },
                        "measurements": report,
                    },
                    indent=2,
                )
            )
            torch._dynamo.reset()


if __name__ == "__main__":
    main(tyro.cli(Args))
