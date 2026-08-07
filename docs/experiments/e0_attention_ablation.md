# E0 attention and decode prelude

Status: P1/P2 exploratory run active; P0 pending

## Question

Choose a practical attention and decode package before E0 becomes the reference for E1-E7.

This is not a pure attention-mask study. P0 and P1 change context length, batch size, and attention
mask together. They keep the token budget and approximate attention work per optimizer step close.
P1 and P2 then isolate decode implementation by using one checkpoint.

## Valid packages

### P0: short full causal reference

- `L_ctx=256`
- `batch_size=512`
- `attn_window=0`
- Full rolling-window recomputation in closed loop
- `eval_incremental_kv=False`

P0 is the candidate E0 reference. It uses the established short-context full-causal package while
keeping the new normalized auxiliary BC objective and v7 data.

### P1: long SWA with full recomputation

- `L_ctx=1024`
- `batch_size=128`
- `attn_window=128`
- Full rolling-window recomputation in closed loop
- `eval_incremental_kv=False` for this evaluation

Use the current SWA checkpoint. Do not retrain it for P1. Its saved training configuration remains
unchanged; the full-recompute evaluation override must be recorded separately.

### P2: long SWA with temporal KV

- Use the exact P1 checkpoint.
- `L_ctx=1024`
- `attn_window=128`
- `eval_incremental_kv=True`
- Feed every observed frame into the cache.
- Reset each slot at every match boundary.

P2 differs from P1 only in decode implementation. It is not a training arm.

## What P0 versus P1 controls

Both packages process 131,072 frame tokens per optimizer step:

\[
512\times256=128\times1024=131072.
\]

They also expose about 16 million causal attention edges per layer and step.

For P0:

\[
512\times\frac{256\times257}{2}=16{,}842{,}752.
\]

For P1, including the shorter windows during the first 127 positions:

\[
128\times\left(\frac{128\times129}{2}+896\times128\right)
=15{,}736{,}832.
\]

This makes P0 and P1 similar in token count and attention-edge count. It does not make them the
same geometry. They differ in:

- 512 shorter samples versus 128 longer samples per step.
- Full causal attention versus a 128-frame layer window.
- Context boundaries and the number of distinct replay windows in one batch.
- Effective temporal receptive field.

Call this a practical package comparison. Do not claim that it isolates the mask, context length,
or batch diversity.

Keep the dataset, seed, number of steps, objective, optimizer, model width and depth, output heads,
decode distribution, and evaluation schedule fixed. Use the same data-stream rules, but do not claim
that the two geometries contain identical training windows.

## Optional I1 pure-mask study

Run I1 only if P0 versus P1 leaves an important mask question unresolved.

- `L_ctx=256`
- `batch_size=512`
- `attn_window=32`
- First evaluate with full recomputation.
- Reevaluate the same checkpoint with exact temporal KV after the parity gate.

Eight 32-frame layers have a maximum receptive field of:

\[
1+8(32-1)=249,
\]

which fits inside 256 frames. P0 versus I1 changes only the attention mask when every other training
and evaluation field is fixed. It is the affordable pure-mask arm. It is not required before E1 if
the practical P0/P1 decision is clear.

A later compute-matched study must have its own plan. Giving one arm more steps changes data
exposure and schedule length. Do not mix that result into the fixed-step package comparison.

## Invalid package

Do not use full causal attention with a rolling KV cache after the input buffer rolls.

Before frame 256, a growing full-causal cache can match P0 recomputation. After eviction, retained
later-layer states still contain information from removed frames. Trimming old keys and values does
not remove that information. P0 must use full recomputation, and code must reject its first invalid
cache eviction.

SWA is different. P1/P2 has a maximum eight-layer receptive field of:

\[
1+8(128-1)=1017,
\]

which fits inside 1,024 frames. A per-layer 128-frame cache can match rolling full recomputation when
relative rotary positions, frame ingestion, and resets are correct.

## Fixed model and objective

Use the linear independent heads and normalized auxiliary BC:

\[
L=L_1+\frac{L_5+L_9+L_{13}}{3}.
\]

Keep these fields fixed across trained arms unless listed in the package definition:

- `d_model=256`, `n_layers=8`, `n_heads=4`.
- `head_offsets=(1,5,9,13)`, `aux_loss_weight=1.0`.
- `transition_loss_weight=1.0`, `history_dropout_p=0.0`.
- `max_steps=16384`, `warmup_steps=500`, `seed=0`.
- `muon_lr=0.02`, `adam_lr=8.5e-4`, `weight_decay=0.01`.
- BF16 training, TF32 enabled, compiled training, and `require_flex=True` on Vast.
- `data/processed/ranked-anonymized-1/mds-v7`, schema 7.
- Per-frame execution, temperature 1, and no decode filters or repairs.
- No AWR, value head, rank weight, action-group conditioning, or temporal action conditioning.

Use `windows_per_replay=2` in clean P0 and P1 runs so it does not add another package difference.

## Intended code work

- Add a manual checkpoint-evaluation override for `eval_incremental_kv` in
  `experiments/023_mtp_heads.py` if one does not already exist.
- Test that the override changes only decode and is written into the evaluation protocol.
- Use existing trunk and closed-loop code unless parity tests prove it wrong.
- Record any shared-code change here before making it.
- Preserve the saved checkpoint configuration. Do not rewrite it to describe a manual evaluation.

## P2 parity gate

P2 may run only after these tests pass on the evaluation GPU and precision:

1. Compare the newest hidden state from P1 full recomputation and P2 at cold start and at frames
   127, 128, 129, 1,016, 1,017, 1,024, and later.
2. Continue for more than two complete 1,024-frame contexts. Compare P2 with the last token from
   each current rolling-window recomputation.
3. Test no padding, heavy left padding, and a slot that has just restarted.
4. Test mixed batches with different cache lengths and reset times.
5. Compare every offset and action-group logit.
6. Test FP32 and the exact FP16 evaluation cast. Derive tolerances from measured error.
7. Compare sampled actions over the frozen validation set with fixed random seeds. Require zero
   mismatches before closed-loop evaluation.
8. Confirm that every observed frame enters the cache, including frames between replans.
9. Confirm that reset removes keys, values, and the saved newest hidden state for one slot.
10. Confirm that full-causal incremental decode raises before invalid eviction.
11. Confirm that P1 and P2 load identical tensors and record different decode protocols.

Run the focused trunk, closed-loop ring, and 023 experiment suites before P2.

## Metrics

For P0 and P1 at equal optimizer steps, report:

- Training objective and offset-1 NLL.
- Validation total and per-group offset-1 NLL.
- Per-group argmax and exact-frame accuracy.
- Hold and transition NLL and accuracy.
- Predicted transition rate, persistence, and change-event F1.
- NLL at offsets 1, 5, 9, and 13.
- Primary and auxiliary trunk-gradient norms, cosine, ratio, and sign conflict.
- Parameter count, peak GPU memory, compile time, median and p95 step time, samples per second, and
  tokens per second.

P2 has no new training score. Report P1/P2 maximum and percentile hidden-state, logit, and
probability error, plus sampled-action mismatch count.

For each decode path, report:

- Model forward time per observed frame at real evaluation batch sizes.
- End-to-end emulator steps and active frames per second.
- Wall time for 32 and 96 CPU matchups and 128 H2H games.
- GPU memory used by weights and caches.
- Live slots and cache-length groups per forward.
- Context-build, transfer, trunk, head-sampling, and emulator time when profiling can separate them.

Use the same host for a direct latency claim. Otherwise report hardware facts and call the result
descriptive.

## Closed-loop evaluation

Use the same level-9 CPU schedule:

- 32 fixed matchups at periodic checks.
- 96 fixed matchups for each final comparison.
- `eval_max_frames=7200`, `eval_seed=0`, and identical decode seeds.
- Save match rows, replays, worker logs, and the full protocol.

Report stocks and damage taken and dealt per active minute, bootstrap intervals, crashes, completed
matches, and paired deltas from aligned CPU rows. The CPU protocol does not report full game win
rate. Label any result from terminal games as a terminal subset.

After P0 exists, run:

- P1 full recompute against P0 over 64 mirrored configurations, or 128 games.
- P2 temporal KV against P0 over the same configurations and seeds.

Report non-tied win rate, per-game and paired per-configuration stock difference, confidence
intervals, stage results, and character results. P1 and P2 are one learned policy. A difference
between them is a decode warning, not a model gain.

## Infrastructure

Use a 1 TB Vast disk for all future clean runs and set `cache_limit_gb` to about 900. The current
500 GB disk reached 92% use while about 463 GB of an approximately 800 GB materialized dataset was
present. Its 440 GB cache had to evict and refetch shards.

Keep at least 128 GB of system RAM, one experiment at a time, checkpoint uploads every 2,048 steps,
and the 3.0-3.5 hour target. Flag startup over 30 minutes, sustained steps over 0.5 seconds, GPU use
below 80% after warmup, a 32-match evaluation over 25 minutes, or a projected total over 3.5 hours.

Record provisioning, downloads, validation caching, compilation, training, evaluation, H2H,
uploads, GPU, CPU count, RAM, disk use, and network speed.

## Current P1/P2 evidence

Retain Vast instance `47034073`, W&B run `19sowpt8`, and its uploaded artifacts.

At step 8,192:

- Validation action NLL was 1.063 bits per frame; button log loss was 0.032.
- The P2 CPU evaluation took 316 seconds and produced 43 matches with no crashes.
- Stocks taken per minute were 0.672, 95% CI `[0.576, 0.769]`.
- Stocks lost per minute were 1.551, 95% CI `[1.320, 1.792]`.
- Damage dealt per minute was 119.5, 95% CI `[110.0, 129.8]`.
- Damage taken per minute was 129.2, 95% CI `[122.1, 136.7]`.
- Dead-frame fraction was 0.0230.
- Derived mean stock difference was -1.279.
- Eleven games ended before the frame limit; none were wins. This is the terminal subset, not a
  full-schedule win rate.
- `latest.pt` and `step_008192.pt` uploaded. Forty-six replays and the match rows were uploading.

The result was broadly flat against step 4,096. Treat it as exploratory P2 evidence.

At step 12,288, validation action NLL was 1.027 bits per frame. The P2 CPU evaluation took 326
seconds and produced 42 matches with no crashes. Stocks taken and lost per minute were 0.911 and
1.502. Damage dealt and taken per minute were 134.3 and 119.0. Derived mean stock difference was
-0.881. Ten games ended before the frame limit and one was a win. This was the strongest periodic
snapshot so far, but the small, mostly truncated sample does not establish a policy result.

The run trains P1 (`L_ctx=1024`, SWA128) but evaluates with P2 temporal KV. It cannot be the P0
reference. It also cannot supply a P1 closed-loop result until its exact checkpoint is reevaluated
with full recomputation. Log that reevaluation as a separate manual record and preserve the original
run configuration.

## Decision rules

- P1 beats P0 in CPU and H2H results: use the long-SWA package.
- P1 matches P0 within uncertainty and is materially better in throughput or memory: prefer P1 for
  efficiency and report no measured policy gain.
- P1 loses to P0: use P0 as E0. P2 speed does not excuse a policy regression.
- P2 matches P1 actions and policy results and is faster end to end: enable temporal KV for P1.
- P2 changes sampled validation actions or closed-loop results beyond repeatability: fix decode
  parity before using its speed result.
- P2 is not faster end to end: keep P1 full recomputation.
- CPU and H2H disagree: call the result inconclusive and inspect paired rows and replays.

Do not select a package from NLL or kernel time alone. Repeat the chosen comparison for at least
three paired seeds before making a final claim.

## Results

Pending P0 and P1 full-recompute evaluation.
