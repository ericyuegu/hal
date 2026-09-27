# Attention and decode experiment index

Updated: 2026-08-07

This file is an index. It does not define launch settings.

## P0: short full-causal package

Use `docs/experiments/e0_normalized_aux_bc.md`.

P0 trains with a 256-frame raw context, full causal attention, batch 512, and full rolling-window
recomputation during closed-loop inference.

## P1: matched long-context package

Use `docs/experiments/p1_matched_attention.md`.

P1 changes context length to 1,024, batch size to 128, and the per-layer attention window to 128.
It keeps the compact data, token count, approximate attention-edge count, model, loss, optimizer,
steps, and evaluation schedule matched to P0. This is a package comparison, not a pure mask test.

The old W&B run `19sowpt8` is exploratory only. It used the old sampler and a 512-entry
action-state table.

## P2: temporal KV decode ablation

P2 uses the selected P1 checkpoint and changes only closed-loop decoding. It may run only after
full recomputation and temporal KV agree across:

- More than two complete rolling contexts.
- Mixed slot lengths and independent resets.
- Every output logit and probability.
- FP32 and the exact evaluation cast.
- Fixed-seed sampled actions.

Full causal attention with a rolling KV cache is invalid after raw-window eviction. P0 must always
recompute its complete rolling context.

## I1: optional mask isolation

If P0 versus P1 is scientifically unclear, compare full causal attention with SWA32 at fixed
context 256 and batch 512. Write a separate plan before launching it. Do not mix I1 into the main
flight unless the package result leaves the mask question unresolved.
