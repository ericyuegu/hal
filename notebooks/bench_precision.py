"""Throughput A/B for the 001 train step under different float precisions.

The cloud A6000 runs the step pegged at 100% SM but only ~105W/275W: pure FP32
with TF32 off, so Ampere tensor cores sit idle. This isolates the *compute* cost
of one train step (forward+backward+opt) from the data pipeline — it pre-stages a
handful of real batches on the GPU and loops over them — and times it under:
fp32 (baseline), tf32, bf16-autocast, bf16+tf32, and optionally torch.compile.

    uv run notebooks/bench_precision.py --batch-size 512
    uv run notebooks/bench_precision.py --batch-size 256 --l-ctx 256   # match a local report

Run on whatever GPU is present (3060 locally, A6000 on the box) to see the win.
"""

import importlib.util
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
from loguru import logger
from torch.optim import AdamW

from hal import streams
from hal.training.dataloader import make_loader
from hal.training.stats import load_consolidated_stats

# Load the digit-prefixed experiment file as a module (can't `import 001_...`).
_EXP = Path(__file__).resolve().parents[1] / "experiments" / "001_flow_matching_baseline.py"
_spec = importlib.util.spec_from_file_location("exp001", _EXP)
exp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp)


@dataclass(frozen=True)
class Args:
    # dev is small but complete locally; a compute benchmark only needs real batch
    # *shapes*, so we stage a small real batch and tile it up to --batch-size.
    data_root: str = "data/processed/dev/mds"
    batch_size: int = 512
    l_ctx: int = 256
    l_chunk: int = 16
    n_staged: int = 3  # GPU-resident batches looped for timing (keep small: 12GB cards)
    warmup: int = 5
    iters: int = 30
    compile: bool = False  # also time torch.compile(model) (long warmup)


def _tile(batch, n: int):
    """Repeat a TrainBatch's rows (dim 0) up to batch dim `n` — compute cost depends
    only on shape, so identical rows are fine for timing."""
    src = batch.target.shape[0]
    idx = torch.arange(n, device=batch.target.device) % src
    feats = {k: v[idx] for k, v in batch.context.features.items()}
    ctx = exp.Context(features=feats, ctx_pad=batch.context.ctx_pad[idx])
    return exp.TrainBatch(ctx, target=batch.target[idx])


def _stage_batches(args: Args, stats: dict) -> list:
    loader = make_loader(
        data_root=args.data_root,
        split="train",
        remote=None,
        stats=stats,
        L_ctx=args.l_ctx,
        L_chunk=args.l_chunk,
        batch_size=32,
        seed=0,
        num_workers=2,
        prefetch_factor=2,
    )
    seed_batch = next(iter(loader)).to("cuda")
    return [_tile(seed_batch, args.batch_size) for _ in range(args.n_staged)]


def _time(model, opt, batches, args, *, autocast_dtype, matmul_precision) -> float:
    """Median samples/s over `iters` steps under the given precision."""
    torch.set_float32_matmul_precision(matmul_precision)
    torch.backends.cuda.matmul.allow_tf32 = matmul_precision != "highest"
    torch.backends.cudnn.allow_tf32 = matmul_precision != "highest"

    def step(batch):
        opt.zero_grad(set_to_none=True)
        if autocast_dtype is None:
            loss = exp.flow_loss(model, batch)
        else:
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                loss = exp.flow_loss(model, batch)
        loss.backward()
        opt.step()

    for i in range(args.warmup):
        step(batches[i % len(batches)])
    torch.cuda.synchronize()
    dts = []
    for i in range(args.iters):
        t0 = time.perf_counter()
        step(batches[i % len(batches)])
        torch.cuda.synchronize()
        dts.append(time.perf_counter() - t0)
    dts.sort()
    return args.batch_size / dts[len(dts) // 2]


def main(args: Args) -> None:
    assert torch.cuda.is_available(), "needs a GPU"
    gpu = torch.cuda.get_device_name(0)
    cfg = exp.TrainConfig(batch_size=args.batch_size, L_ctx=args.l_ctx, L_chunk=args.l_chunk)
    stats = load_consolidated_stats(Path(args.data_root) / "stats.json")
    logger.info(
        f"GPU={gpu} | model={exp._model_tag(cfg)} | batch={args.batch_size} | staging {args.n_staged} batches…"
    )
    batches = _stage_batches(args, stats)

    torch.manual_seed(0)
    model = exp.FlowMatchingPolicy(cfg).to("cuda")
    opt = AdamW(model.parameters(), lr=3e-4)

    configs = [
        ("fp32 (baseline, tf32 off)", None, "highest"),
        ("tf32", None, "high"),
        ("bf16 autocast", torch.bfloat16, "highest"),
        ("bf16 + tf32", torch.bfloat16, "high"),
    ]
    results = []
    for name, dt, prec in configs:
        sps = _time(model, opt, batches, args, autocast_dtype=dt, matmul_precision=prec)
        results.append((name, sps))
        logger.info(f"  {name:32s} {sps:8.0f} samples/s")

    if args.compile:
        torch.set_float32_matmul_precision("high")
        cmodel = torch.compile(exp.FlowMatchingPolicy(cfg).to("cuda"))
        copt = AdamW(cmodel.parameters(), lr=3e-4)
        sps = _time(
            cmodel,
            copt,
            batches,
            Args(**{**args.__dict__, "warmup": 12}),
            autocast_dtype=torch.bfloat16,
            matmul_precision="high",
        )
        results.append(("bf16 + tf32 + compile", sps))
        logger.info(f"  {'bf16 + tf32 + compile':32s} {sps:8.0f} samples/s")

    base = results[0][1]
    logger.info(f"=== {gpu} | speedups vs fp32 baseline ({base:.0f} samples/s) ===")
    for name, sps in results:
        logger.info(f"  {name:32s} {sps:8.0f} samples/s  ({sps / base:.2f}x)")


if __name__ == "__main__":
    main(tyro.cli(Args))
