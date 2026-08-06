# %%
"""How long a sequence can we train on this box, and at what micro-batch?

Sliding-window attention makes attention cost grow about linearly with the sequence length, so a
longer window of frames becomes affordable. A longer sequence also cuts disk work: the loader reads
a whole replay to emit each window, so more tokens per read means fewer reads for the same tokens.

This bench answers one question per sequence length: the largest micro-batch that fits in VRAM, and
what one step of that shape costs. The model is the 022 shape - the 374-wide input projection, the
d256/L8/h4 trunk, four action heads and the value head - under bf16 autocast, the way training runs.

Run it cell by cell, or `uv run notebooks/bench_seq_len.py`.
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"  # the training allocator setting

import time

import torch
import torch._dynamo
import torch.nn as nn

from hal.training.trunk import Trunk
from hal.training.trunk import TrunkConfig

D_IN = 374  # the 016-base observation token
A_VOCAB = 355  # the four action groups, flattened
N_HEADS_OUT = 4  # one head per predicted offset
TOKEN_BUDGET = 131_072  # 020's tokens per optimizer step (batch 512 x L_ctx 256)
SEQ_LENS = (256, 512, 1024, 2048)
ATTN_WINDOW = 128
DEVICE = "cuda"

# One FlexAttention kernel per (batch, sequence) shape. The default limit is 8, and a bench that
# passes it would quietly drop back to eager and report the wrong speed.
torch._dynamo.config.cache_size_limit = 64


# %%
class BenchModel(nn.Module):
    """The trunk plus the parts around it that also hold activations."""

    def __init__(self, L_ctx: int, attn_window: int) -> None:
        super().__init__()
        cfg = TrunkConfig(d_model=256, n_layers=8, n_heads=4, L_ctx=L_ctx, attn_window=attn_window)
        self.proj = nn.Linear(D_IN, cfg.d_model)
        self.trunk = Trunk(cfg)
        self.heads = nn.ModuleList([nn.Linear(cfg.d_model, A_VOCAB) for _ in range(N_HEADS_OUT)])
        self.value_head = nn.Linear(cfg.d_model, 1)

    def forward(self, x: torch.Tensor, ctx_pad: torch.Tensor) -> torch.Tensor:
        h = self.trunk(self.proj(x), ctx_pad)
        loss = sum(head(h).float().logsumexp(-1).mean() for head in self.heads)
        return loss + self.value_head(h).float().pow(2).mean()


def step(model: BenchModel, batch: int, L_ctx: int) -> None:
    x = torch.randn(batch, L_ctx, D_IN, device=DEVICE)
    ctx_pad = torch.zeros(batch, dtype=torch.long, device=DEVICE)
    with torch.autocast(DEVICE, dtype=torch.bfloat16):
        loss = model(x, ctx_pad)
    loss.backward()
    model.zero_grad(set_to_none=True)


def measure(L_ctx: int, batch: int, attn_window: int = ATTN_WINDOW, n_steps: int = 4) -> tuple[float, float]:
    """Peak VRAM (GiB) and mean step time (s) for one shape. Raises on OOM."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = BenchModel(L_ctx, attn_window).to(DEVICE)
    step(model, batch, L_ctx)  # warm up: compile FlexAttention, grow the allocator
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        step(model, batch, L_ctx)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n_steps
    peak = torch.cuda.max_memory_allocated() / 2**30
    del model
    return peak, dt


# %%
def largest_batch(L_ctx: int, candidates: tuple[int, ...]) -> tuple[int, float, float]:
    """The largest candidate micro-batch that fits, with its peak VRAM and step time."""
    best = (0, 0.0, 0.0)
    for batch in candidates:
        try:
            peak, dt = measure(L_ctx, batch)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            break
        best = (batch, peak, dt)
    return best


rows = []
for L_ctx in SEQ_LENS:
    # Powers of two only, so the micro-step token count always divides the budget and the grad-accum
    # count is exact. A micro-batch past the whole budget has no use.
    candidates = tuple(b for b in (8, 16, 32, 64, 128, 256, 512) if b * L_ctx <= TOKEN_BUDGET)
    batch, peak, dt = largest_batch(L_ctx, candidates)
    tokens = batch * L_ctx
    rows.append(
        dict(
            L_ctx=L_ctx,
            micro_batch=batch,
            peak_gib=round(peak, 2),
            step_s=round(dt, 3),
            tokens_per_micro_step=tokens,
            tokens_per_s=round(tokens / dt),
            grad_accum=TOKEN_BUDGET // tokens,
        )
    )
    print(rows[-1], flush=True)

# %%
# What the window buys: the same shapes with full attention, for comparison.
for r in rows:
    _, dt_full = measure(r["L_ctx"], r["micro_batch"], attn_window=0)
    r["full_attn_step_s"] = round(dt_full, 3)
    r["swa_speedup"] = round(dt_full / r["step_s"], 2)
    print(r, flush=True)

# %%
print("L_ctx  micro  peak GiB  step s  tok/micro-step  tok/s   accum for 131k  full-attn s  SWA speedup")
for r in rows:
    print(
        f"{r['L_ctx']:>5}  {r['micro_batch']:>5}  {r['peak_gib']:>8}  {r['step_s']:>6}  "
        f"{r['tokens_per_micro_step']:>14}  {r['tokens_per_s']:>6}  {r['grad_accum']:>14}  "
        f"{r['full_attn_step_s']:>11}  {r['swa_speedup']:>11}"
    )
