# Part 1 hand-off notes (temporary — delete after the 022 fork consumes it)

What Part 1 built, and what the `experiments/022_awr_rank.py` fork must change. Experiment 020 is
frozen: every item below is a change to make in 022, not in 020.

## What exists now

- `hal/training/trunk.py` — the shared trunk: `TrunkConfig`, `Rotary`, `apply_rotary_emb`,
  `rmsnorm`, `dense_mask`, `block_mask`, `flex_is_usable`, `CausalSelfAttention`, `MLP`, `Block`,
  `Trunk`. Parameter creation order is 020's, so seeded init draws are the same.
- `tests/test_trunk.py` — init and forward identity against 020, flex-vs-dense equality, the mask
  against a double-loop reference, incremental-vs-full under SWA, and a 20-step loss-curve match.
- `notebooks/bench_seq_len.py` — the VRAM and step-time bench used for the geometry below.

### How 022 uses it

```python
from hal.training.trunk import Trunk, TrunkConfig

# in GPT.__init__, in place of 020's `self.blocks = nn.ModuleList([Block(cfg) ...])`:
self.trunk = Trunk(TrunkConfig(cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.L_ctx, cfg.attn_window))
```

The `Trunk` builds the mask itself, so `GPT.forward` becomes `self.trunk(self._context_tokens(f), ctx_pad)`
and `GPT.forward_incremental` becomes `self.trunk.forward_incremental(x, past)` (drop 020's unused
`position` argument; `max_cache` now lives on the trunk and is `attn_window or L_ctx`). Two call-site
details: the trunk returns `[B, L, d_model]`, so 022 keeps 020's `[:, -1]` at the caller; and 022 must
keep 020's one-token guard (`020:863-864`), because the trunk's incremental attention has no causal
mask inside the given chunk and is therefore only correct for `L == 1`.

The attention path resolves at the first forward: FlexAttention when it compiles on the box, the
dense SDPA mask when it does not. `Trunk.attn_path` reports which one, and the choice is logged.
Note that the checkpoint keys gain a `trunk.` prefix (`trunk.blocks.0.attn.c_attn.weight`), which is
fine for a new experiment but means 020 checkpoints cannot be loaded into 022.

Config check to add in `validate_config`: reject `eval_incremental_kv=True` with `attn_window == 0`.
With a window the rolling cache holds exactly the trained window and RoPE is relative, so
incremental decode is exact (`tests/test_trunk.py::test_incremental_matches_full_forward_under_swa`
pins this); without a window it silently drops history. 020's comment at `020:292-294` says the
same and can be replaced by the check.

## Fix 1 — save `final.pt` before the final closed-loop eval

Today `020:2468` runs the final eval unconditionally and `020:2469` saves after it. An eval-incapable
box (the H100: glibc 2.35 against the Dolphin build's 2.39 floor) crashes at the finish line and the
weights are lost.

In 022:

```python
_save("final.pt", cfg.max_steps)
if cfg.final_eval_n_matchups > 0:
    _log_eval(cfg.max_steps, _eval_and_upload("final", n_matchups=cfg.final_eval_n_matchups))
```

Keep the `n_matchups must be > 0` raise in `_eval_protocol` (`020:1943`): an explicit `--eval` with a
zero count is still an error. Only the end-of-training call becomes conditional.

## Fix 2 — two trunk forwards per validation batch, not five

Per val batch, 020 runs the trunk five times:

| Site | 020 line | Forward |
|---|---|---|
| `val_metrics` | 1696 | `h` |
| `val_metrics` (copycat probe) | 1701 | `h_ablated` |
| `recon_metrics(argmax=True)` → `decode` | 1065 | `h[:, -1]` |
| `recon_metrics(argmax=False)` → `decode` | 1065 | `h[:, -1]` |
| `awr_val_metrics` | 1819 | `h` |

Only the first two are distinct: the other three are the same function of the same inputs, under the
same eval mode.

Design: compute the pair once in `_val_log_dict` (`020:2235-2261`) and pass it down.

1. New helper next to `val_metrics`:
   `def val_hidden(model, val_cache) -> list[tuple[Tensor, Tensor]]` — under `_evaluation_mode` and
   `torch.no_grad`, return `(h, h_ablated)` per batch, where `h_ablated` is the forward with the ego
   controller-history channels zeroed (the code at `020:1698-1701`).
2. `val_metrics(model, val_cache, cfg, hidden)` and `awr_val_metrics(model, val_cache, cfg, hidden)`
   take the list and drop their own forwards.
3. Split the last-position sampling out of `decode` (`020:1064-1075`): a `decode_from_hidden(model,
   h_last, ...)` that holds the body from `logits = model.heads[...]` onward, and `decode` keeps its
   signature and calls it with `model(...)[:, -1]`. `_recon_metrics_eval` then takes the per-batch
   `h[:, -1]` and calls `decode_from_hidden`, so the sampler and the RNG order do not move.
4. `gradient_diagnostics` (`020:1565`) keeps its own forwards: it needs gradients.

Requirement: the `val/*` dict must be bitwise identical before and after. Check it by logging the
dict on a dev-fixture run with both versions and comparing with `==` on every scalar.

Expected saving: validation drops from 5 forwards per batch to 2, so a val pass gets ~2.5x cheaper.

## Bench — sequence length on the 3060 (RTX 3060, 12 GiB)

`uv run notebooks/bench_seq_len.py`, trunk d256 / L8 / h4, window 128, the 374-wide input projection,
4 action heads and the value head, bf16 autocast, forward + backward. The micro-batch is the largest
power of two whose token count still divides the 131,072-token step budget (020's batch 512 x L 256).

| L_ctx | micro-batch | peak VRAM | step s | tokens/micro-step | tokens/s | accum for 131k | full-attn step s | SWA speed-up |
|---|---|---|---|---|---|---|---|---|
| 256 | 256 | 5.94 GiB | 0.368 | 65,536 | 178,173 | 2 | 0.367 | 1.00x |
| 512 | 128 | 5.94 GiB | 0.373 | 65,536 | 175,870 | 2 | 0.381 | 1.02x |
| 1024 | 64 | 5.94 GiB | 0.378 | 65,536 | 173,433 | 2 | 0.409 | 1.08x |
| 2048 | 32 | 5.94 GiB | 0.387 | 65,536 | 169,199 | 2 | 0.466 | 1.21x |

Readings:

- At a fixed token count per micro-step, VRAM does not move at all and step time rises 5% from
  L=256 to L=2048. Under a 128-frame window, sequence length is close to free.
- The attention block is a small part of the step at d256/L8 (the MLP dominates), so the SWA
  speed-up over full attention is only 1.21x at L=2048. The reason to take SWA is not this number:
  it is that SWA keeps the cost flat as L grows, and it makes incremental decode exact.
- The 3060 has spare memory at the budget: a mixed grid also fits 384 x 256, 192 x 512, 96 x 1024
  and 48 x 2048 (8.88 GiB, 98,304 tokens per micro-step). Cloud boxes with more VRAM should re-run
  the bench and raise the micro-batch instead of the accumulation count.

### Recommended base geometry

**L_ctx = 1024, micro-batch 64, grad-accum 2 (batch 128, 131,072 tokens per step), attn_window = 128.**

Reasons: the step cost is within 3% of L=256; the loader reads a whole ~6-10k-frame replay for each
window, so 4 windows x 1024 frames turns about half the read frames into training tokens instead of
about an eighth at L=256, which attacks the known disk bottleneck; and 8 layers x a 128-frame window
reach about 1024 frames of receptive field through multi-hop attention, so the context is used.
L=2048 buys more I/O relief for 5% more step time, but a 2048-frame window plus the offsets starts
to cut into short replays; keep it as the fallback if the loader is still the bottleneck after
Part 2. Set `windows_per_replay` to match the chosen L.

## One trap for the 022 tests

The experiment train loops call `torch.set_float32_matmul_precision("high")`, which stays set for the
whole pytest process. Under TF32 the FlexAttention and SDPA kernels differ by about 6e-4, so a
path-comparison test passes alone and fails in the full suite. `tests/test_trunk.py` pins
`"highest"` for the duration of each test and restores the previous value. Any 022 test that
compares two kernels must do the same.

## Attention path on this box

- CUDA (RTX 3060): FlexAttention compiles and runs. It is the active path.
- CPU: FlexAttention has no backward, so `flex_is_usable("cpu")` is false and the trunk falls back
  to the dense SDPA mask with a warning. CPU-only tests therefore exercise the reference path.
