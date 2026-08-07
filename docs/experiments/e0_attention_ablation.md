# E0 attention and decode ablation

Status: planned

## Question

The current E0 run changes two things relative to the older full-attention policy:

- It trains with 128-frame sliding-window attention.
- It evaluates with a temporal KV cache.

This ablation separates the learned attention mask from the decode implementation. The new E0
reference is full causal attention with full-context recomputation.

## Arms

### A0: full causal reference

- Train with `L_ctx=1024` and `attn_window=0`.
- Evaluate by rebuilding the complete rolling 1,024-frame context at every policy step.
- Set `eval_incremental_kv=False`.

A0 is the new E0 reference for later model ablations.

### A1: sliding-window attention

- Train with `L_ctx=1024` and `attn_window=128`.
- Evaluate by rebuilding the complete rolling 1,024-frame context at every policy step.
- Set `eval_incremental_kv=False` for this evaluation.

A1 differs from A0 only in the trained attention mask.

### A2: exact temporal KV decode

- Use the exact A1 checkpoint. Do not retrain it.
- Evaluate with `attn_window=128` and `eval_incremental_kv=True`.
- Feed every observed frame into the cache, including frames between policy replans.
- Reset each slot's cache at every match boundary.

A2 differs from A1 only in how the same checkpoint is evaluated. A2 is a decode ablation, not a
training arm.

## Invalid arm

Do not run full causal attention with a rolling `L_ctx`-length KV cache after the context buffer
rolls.

Before the first 1,024 frames, a full causal cache can match a growing full-context forward. After
the buffer rolls, it cannot match recomputation. Cached hidden states in later layers still contain
information from frames that the rolling full-context input has removed. Trimming old keys and
values does not remove that information from the retained hidden states.

The full-causal A0 policy must use full recomputation. Code must continue to reject incremental
full-attention decode when adding one more frame would evict history.

SWA is different. Each layer can read at most 128 frames, and eight layers have a maximum temporal
receptive field of:

\[
1 + 8(128 - 1) = 1017\text{ frames}.
\]

This fits inside `L_ctx=1024`. A per-layer 128-frame cache can therefore match a full recomputation
over the rolling input when relative rotary positions and resets are correct.

## Fixed-data comparison

The primary A0 versus A1 comparison uses fixed data and fixed geometry:

- The same v7 dataset and validation split.
- The same sampled windows in the same order.
- `L_ctx=1024`, `batch_size=128`, and 131,072 frame tokens per optimizer step.
- The same seed, initialization, parameters, optimizer, learning-rate schedule, and 16,384 steps.
- The same output heads, offsets, objective, decode distribution, and evaluation matchups.
- The same compiled FlexAttention training path.

Only `attn_window` changes. The two models have the same parameter count. A0 does more attention
work per step because it permits every causal edge. This is intentional: the primary comparison
asks what the attention mask does at equal data exposure and model geometry.

Do not call this a compute-matched comparison.

If a compute-matched study is later useful, write a separate plan before running it. For example,
give A1 more optimizer steps until its measured training FLOPs or GPU time matches A0. That study
changes data exposure and schedule length, so it cannot replace the fixed-data result. Do not
quietly reduce A0 batch size or context length to make it faster.

## Fixed model and objective

Use the linear E0 head and normalized auxiliary BC objective:

\[
L=L_1+\frac{L_5+L_9+L_{13}}{3}.
\]

Keep:

- `d_model=256`, `n_layers=8`, `n_heads=4`.
- `head_offsets=(1,5,9,13)` and `aux_loss_weight=1.0`.
- `transition_loss_weight=1.0` and `history_dropout_p=0.0`.
- `max_steps=16384`, `warmup_steps=500`, and `seed=0`.
- `muon_lr=0.02`, `adam_lr=8.5e-4`, and `weight_decay=0.01`.
- BF16 training, TF32 enabled, and compiled training.
- `data/processed/ranked-anonymized-1/mds-v7`, schema 7, and two windows per replay.
- Per-frame execution with sampling temperature 1 and no decode filters or repairs.
- No AWR, value head, rank weight, action-group conditioning, or temporal action conditioning.

Use `require_flex=True` on Vast. A dense fallback is not a valid timing comparison.

## Intended code work

Before launching A0 or reporting A1/A2:

- Update `experiments/023_mtp_heads.py` only if manual checkpoint evaluation cannot override
  `eval_incremental_kv` while preserving every saved model field.
- Update `tests/experiments/test_023_mtp_heads.py` with the decode override and protocol-record tests.
- Use existing shared trunk and closed-loop cache code unless a parity test proves that it is wrong.
- Record any required shared-code change in this file before making it.
- Update this file with commands, commits, checkpoints, W&B IDs, timing, and results.

The saved checkpoint configuration remains the training record. A manual A1 reevaluation of a
checkpoint trained with incremental evaluation enabled must log the explicit full-recompute
override in its protocol and W&B record.

## Parity gate for A2

A2 may run only after all tests below pass on the exact evaluation device and precision.

1. Compare the newest hidden state from full A1 recomputation with temporal KV decode during cold
   start, at 127, 128, 129, 1,016, 1,017, 1,024, and later frames.
2. Continue for more than two full context lengths. At each step, compare temporal KV output with
   the last token from the current rolling 1,024-frame recomputation.
3. Test no left padding, heavy left padding, and a slot that has just restarted.
4. Test a mixed batch whose slots have different cache lengths and reset on different frames.
5. Compare every offset and action-group logit, not only the trunk state.
6. Test FP32 and the exact FP16 evaluation cast. Set tolerances from measured error. Do not loosen
   them only to pass.
7. With fixed random seeds, compare sampled actions on the full frozen validation set. Report the
   mismatch count. The required count is zero before closed-loop evaluation.
8. Confirm that every observed frame enters the cache even when `exec_horizon>1`, although this
   ablation deploys `exec_horizon=1`.
9. Confirm that a match reset removes both cached keys and values and the cached newest hidden
   state for that slot.
10. Confirm that full causal incremental decode raises before its first invalid eviction.
11. Confirm that manual A1 and A2 evaluations load identical model tensors and differ only in the
    decode path recorded in the evaluation protocol.

Also run the full focused trunk, closed-loop ring, and 023 experiment test suites before A2.

## Offline metrics

For A0 and A1, report at equal optimizer steps:

- Training objective and offset-1 NLL.
- Validation total and per-group offset-1 NLL.
- Per-group argmax and exact-frame accuracy.
- Hold and transition NLL and accuracy.
- Predicted transition rate, persistence, and change-event F1.
- NLL at offsets 1, 5, 9, and 13.
- Primary and auxiliary trunk-gradient norms, cosine, norm ratio, and sign conflict.
- Parameter count and peak GPU memory.
- Median step time, samples per second, tokens per second, compile time, and validation time.

A2 has no new training or offline model score. Its validation logits should match A1 within the
parity tolerance. Report maximum and percentile absolute logit error, probability error, and sampled
action mismatches.

## Closed-loop evaluation

Evaluate all three protocols against the same level-9 CPU schedule:

- Periodic checks use 32 fixed matchups.
- Final checks use 96 fixed matchups.
- Use `eval_max_frames=7200` and `eval_seed=0`.
- Keep matchup order, policy seed, temperature, frame budget, and concurrency fixed.
- Save `match_rows.json`, replays, worker logs, and the complete decode protocol.

For A0 and A1, report stocks and damage taken and dealt per active minute, bootstrap intervals,
crash rate, and paired deltas from aligned CPU rows.

For A1 and A2, use the same checkpoint. Any policy difference is a decode-parity warning, not a
model improvement. Compare aligned action traces when rows or outcomes differ.

Run final head-to-head evaluation as follows:

- A1 full recompute against A0: 64 mirrored configurations, or 128 games.
- A2 temporal KV against A0: the same 64 mirrored configurations and seeds.
- Report non-tied win rate, per-game stock difference, paired per-configuration stock difference,
  confidence intervals, stage results, and character results.

A1 and A2 should give the same policy result within numerical and emulator repeatability. Do not
run A1 against A2 as the main H2H result because they are the same learned policy.

## Latency and throughput

Measure training and evaluation separately.

For A0 and A1 training, record:

- Warm median and p95 optimizer-step time.
- GPU utilization, peak memory, and attention kernel path.
- Validation and checkpoint overhead.
- Total wall time through final upload.

For each decode protocol, record:

- Model forward milliseconds per observed frame at batch sizes seen in evaluation.
- End-to-end emulator steps per second and active frames per second.
- CPU evaluation wall time for 32 and 96 matchups.
- H2H wall time for 128 games.
- GPU memory used by model state and KV caches.
- Number of live slots and cache-length groups per forward.
- Time spent building contexts, copying tensors, running the trunk, sampling heads, and running the
  emulator when profiling can separate them.

Use identical host hardware for a direct latency claim. If hosts differ, report raw hardware facts
and treat the comparison as descriptive.

## Current Vast run

Retain Vast instance `47034073` and W&B run `19sowpt8` as exploratory evidence.

That run trains the A1 architecture: `L_ctx=1024` with `attn_window=128`. Its current closed-loop
path is A2 temporal KV decode. Keep its checkpoint, W&B history, match rows, replays, host facts, and
timing.

It cannot be the new E0 reference because it is not A0. Its closed-loop result also cannot be called
the A1 full-recompute result until the exact same checkpoint is reevaluated with
`eval_incremental_kv=False`. After that reevaluation exists, the checkpoint can supply both valid A1
and A2 evidence without retraining.

Do not overwrite the original run configuration. Log the A1 reevaluation as a separate manual
evaluation record with its override and source checkpoint.

## Decision rules

The ablation is invalid if data windows, initialization, training steps, objective, decode sampling,
or evaluation schedules drift between A0 and A1.

Interpret valid results as follows:

- A1 beats A0 in CPU and H2H results: keep SWA128 as the learned attention architecture.
- A1 matches A0 within uncertainty and is materially faster or smaller in memory: prefer SWA128 for
  efficiency, while reporting no measured policy gain.
- A1 loses to A0: restore full causal attention as the E0 reference. Temporal KV speed does not
  excuse a policy regression.
- A2 matches A1 policy results and is faster: enable temporal KV for SWA128 evaluation.
- A2 changes sampled validation actions or closed-loop results beyond repeatability: treat this as a
  correctness failure. Fix parity before using its speed result.
- A2 is not faster end to end: keep full recomputation. Kernel speed alone is not enough.
- CPU and H2H disagree: call the policy result inconclusive and inspect paired rows and replays.

Do not select an attention arm from NLL or throughput alone. The main policy decision uses the final
CPU and mirrored H2H evidence. Repeat the chosen comparison over at least three paired seeds before
making a final claim.

## Results

Pending.
