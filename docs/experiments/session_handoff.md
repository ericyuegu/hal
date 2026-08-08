# Action-model experiment handoff

Updated: 2026-08-07

Status: paused by user request. No Vast instance is active.

This file is the entry point for the work completed in this session. The detailed experiment plans
remain in their own files. The current cumulative branch is `exp/e2-within-frame-factorization`.

## Where to start

- Overall state and run order: `docs/experiments/sequence_status.md`.
- Scientific motivation: `docs/experiments/action_chunk_roadmap.md`.
- Data pipeline: `docs/experiments/data_pipeline.md`.
- P0: `docs/experiments/e0_normalized_aux_bc.md`.
- P1: `docs/experiments/p1_matched_attention.md`.
- P2: `docs/experiments/p2_temporal_kv_ablation.md`.
- E1: `docs/experiments/e1_output_head_capacity.md`.
- E2: `docs/experiments/e2_within_frame_factorization.md`.
- E3: `docs/experiments/e3_temporal_factorization.md`.
- E4: `docs/experiments/e4_dense_chunk_readiness.md`.
- E5: `docs/experiments/e5_primary_awr.md`.
- E6: `docs/experiments/e6_chunk_critic.md`.
- E7: `docs/experiments/e7_chunk_awr_execution.md`.

The main experiment implementation is `experiments/023_mtp_heads.py`. Shared model code is in
`hal/training/trunk.py`. Shared data, evaluation, and upload code is under `hal/training/`,
`hal/data/`, `hal/eval/`, and `hal/sim/`. Do not restart new work from `experiments/020_awr.py` or
`experiments/022_awr_rank.py`; they are reviewed historical references, not the selected base.

Historical topic branches remain available as snapshots: `exp/data-pipeline`,
`exp/p0-normalized-mtp`, `exp/p1-matched-attention`, `exp/p1-swa-recompute`,
`exp/p2-kv-diagnostics`, and `exp/e1-output-head-capacity`. The current cumulative E2 branch
contains later fixes and is the authoritative continuation point.

## Results so far

### Compact replay pipeline

The compact artifact is `r2:hal/processed/ranked-anonymized-1/mds-policy-v7`. It is a projection of
canonical schema v7, not canonical v8. It contains 114,768 replays, about 1.23 billion frames, 296
objects, and 13,339,116,151 bytes. The full audit found no value mismatch.

Decoded training arrays fell from 802.47 GB to 76.19 GB. Compressed storage fell from 29.82 GB to
13.34 GB. Batches keep 512 distinct replay IDs through a replay reservoir of 4,096 entries and a
one-batch cooldown. Keep batch size 512. Prefetch factor 32 did not beat 2. Keep prefetch factor 2.

The compact path reduced the original P0 projection from about 7.2 hours to below the 3.5-hour
end-to-end limit. It did not make the GPU compute-bound. Replay decoding and batch replacement still
consume much of each step.

### P0: normalized auxiliary BC

P0 completed 16,384 steps. W&B run `obx3o3az` is the fixed short-context reference. Its final
checkpoint SHA-256 is `5d12d010fa3acd1ec07bd86a8e85d2cbb84c584a77b9b79e90dc6fcf03c32e4b`.

Final NLL at offsets 1, 5, 9, and 13 was 1.029, 2.650, 3.379, and 3.851 bits per frame. The official
96-boot FP16, full-recompute CPU evaluation reported 0.777 stocks taken and 1.468 stocks lost per
active minute. P0 remains the deployed baseline.

### P1: long-context SWA package

Matched P1 completed 16,384 steps in W&B run `46zi7fgo`. Its checkpoint SHA-256 is
`8e9b04c91aa76d1ba49a910c82f1328bc1b0dc3ce7dabf3e9018cb556d964148`.

P1 improved offline NLL, but it did not pass the paired closed-loop gate. In 64 mirrored H2H pairs,
its paired mean stock difference was -0.062. It was ahead on 24 pairs, behind on 21, and tied on 19.
Damage difference was -8.76 per active minute with a 95% interval below zero. P1 did not replace P0.
This was a package comparison, not a clean SWA-only claim.

### P2: temporal KV ablation

Temporal KV decoding failed exact parity. Across 6,195 slot-frames, it produced 14 FP32 and 32 FP16
sampled-action mismatches. KV reduced model time per row from 0.390 ms to 0.146 ms, but evaluation
wall time fell only from 672 to 507 seconds because Dolphin dominates the sweep. Keep full rolling-
window recomputation. Do not use persistent temporal KV in later experiments.

### E1: state-only MLP capacity control

E1 completed 16,384 steps in W&B run `q3aojgfm`. Its checkpoint SHA-256 is
`c175aa53f1d0f4ff80157b26a51c67cf55c4577ded656b60b97d12e10d8a560f`.

The MLP improved final offset-1 NLL only from 1.029 to 1.026. Its CPU point estimate was slightly
better than P0, but paired H2H was negative. E1 was ahead on 20 mirrored configurations, behind on
33, and tied on 11. Mean paired stock difference was -0.391. E1 did not promote. Keep it as E2's
required capacity control.

### E2-S: partial run only

E2-S is implemented and passed the complete local gate: 949 repository tests, Ruff, type checking,
format checks, and Python compilation. It launched from commit `f520ea8` on Vast instance
`47144794`, W&B run `43ppjdxc`.

The user canceled it after the last observed step 2,300. The step-2,048 checkpoint reached R2.
Validation at step 2,048 was finite: action NLL 1.134 bits per frame and button log loss 0.032.
Observed warm steps were usually about 0.4 to 0.46 seconds, slower than E1's 0.281-second median.
There is no final checkpoint, CPU result, H2H result, or promotion decision. Treat this run only as
startup evidence. Restart E2-S from step 0 when the flight resumes.

## Remaining implementation and run plan

Run every unit alone. Use 16,384 training steps unless its plan explicitly defines a critic warm-up
in the same process. Use an RTX 4090, 250 GB disk, at least 200 GB system RAM, compact policy v7,
full rolling-window recomputation, and at most 32 concurrent Dolphin boots.

### E2-S and E2-I

Implementation is complete in `experiments/023_mtp_heads.py` and
`tests/experiments/test_023_mtp_heads.py`. Shared one-batch preprocessing prefetch is in
`hal/training/dataloader.py`.

1. Restart E2-S from step 0. Use order `c_stick, triggers, buttons, main_stick` and the pinned E1
   checkpoint.
2. Complete final validation, 96-boot CPU evaluation, 64 mirrored H2H configurations against E1,
   upload closure, and the decision.
3. Only then run E2-I with order `main_stick, buttons, triggers, c_stick` against E2-S.
4. Select the order from closed-loop evidence. Do not select it from teacher-forced NLL alone.

Measure teacher-forced and ancestor-sampled NLL, exact-action accuracy, exposure gaps, per-group
accuracy, transition metrics, step time, loader wait, GPU memory, and H2H outcomes. The partial E2-S
run suggests a material compute cost. Measure it before interpreting a policy result.

### E3-C and E3-T

Detailed plan: `docs/experiments/e3_temporal_factorization.md`. Code is not implemented.

Touch only `experiments/023_mtp_heads.py`, its focused test file, and the E3 plan unless a proven
shared seam is required. Add a shared temporal residual MLP. Concatenate the normalized state and
the complete previous-action embedding along the feature dimension. Implement its state and action
blocks as one affine layer. Zero-initialize the action block and residual output so E3 starts with
exact E2 logits.

Run the parameter-matched null-condition control E3-C first. Run teacher-forced previous-action
conditioning E3-T second. Offset 1 must bypass the temporal module. Later sparse heads use true
previous actions during training and sampled previous actions only in rollout diagnostics. Keep
offsets `(1,5,9,13)` and execute only offset 1. E3-T must beat E3-C before a gain can be attributed
to temporal action conditioning.

### E4

Detailed plan: `docs/experiments/e4_dense_chunk_readiness.md`. Code is not implemented.

Add a `sample_chunk_length=13` field in `experiments/023_mtp_heads.py` and the loader seam. Change
scored offsets to `(1,2,3,4)` without changing sampled windows or RNG order. Inherit verified E3-T;
do not inherit the null-condition control. Keep execution horizon 1 and the fixed objective
`L1 + mean(L2,L3,L4)`.

Report teacher-forced and rollout-conditioned metrics, transition precision and recall, exact chunk
accuracy, run lengths, and a copy-previous-action baseline. E4 is a chunk-readiness test. It need
not beat E3 in one-frame play, but it must model transitions better than copying holds.

### E5

Detailed plan: `docs/experiments/e5_primary_awr.md`. Code is not implemented.

Add `hal/training/returns.py`; add the smallest required return-label seam to the loader; implement
the value head and primary-only AWR in experiment 023. Compute full-replay returns before window
sampling. Align the deployed prediction `a_{t+1}` with return `G_{t+1}`. Keep the value head detached
from the policy trunk in the first arm.

Run 2,048 value-only warm-up updates inside the declared 16,384-step process. Gate on finite values,
positive held-out return correlation, normalized frame and replay-window ESS of at least 0.2, and
raw clip fraction at most 0.2. If the gate passes, activate AWR at update 2,048. Weight only offset
1. Keep all future heads as unweighted BC. Normalize eligible weights to mean one with FP32
`logsumexp`. Truncated replays keep actor weight 1 and supply no value target.

### E6

Detailed plan: `docs/experiments/e6_chunk_critic.md`. Code is not implemented.

Add `hal/training/chunks.py`, shifted raw contexts in the loader, and critic-only support in
experiment 023. Build `s_t`, `s_{t+2}`, and `s_{t+4}` as complete raw rolling windows. Recompute the
full frozen transformer for each state. Never append to an old hidden state and never use temporal
KV.

Give Q2 and Q4 separate action encoders and Q heads. Encode each action frame by concatenating its
four group embeddings, then use a small two-block causal action transformer. Concatenate the chunk
vector with the state vector before the nonlinear Q MLP. Train a parameter-matched state-only
control for each horizon. Warm V for 2,048 updates, then train Q for 16,384 updates. Run three critic
seeds one at a time. Gate on held-out calibration, action ranking, action perturbation, policy-
sample coverage, ensemble disagreement, and comparison with the state-only controls.

### E7-H2 and E7-H4

Detailed plan: `docs/experiments/e7_chunk_awr_execution.md`. Code is not implemented.

Add macro-BC and macro-AWR modes to experiment 023. Reuse `hal/training/chunks.py`. Add per-slot
chunk interruption handling to `hal/training/closed_loop.py` and an execution-horizon override to
`hal/scripts/h2h.py`.

First compare the same E4 checkpoint with execution horizon 1 and 2. Then train matched H2 macro-BC
and macro-AWR arms from the same E4 checkpoint and replay order. One ensemble chunk advantage must
weight every within-frame and temporal factor of the executed joint H2 action. Keep offsets 3 and 4
as unweighted BC. If H2 passes, repeat the declared execution and training sequence for H4.

Every predicted frame has coefficient 0.5, so total head-loss scale remains 2.0. Do not divide the
training joint likelihood by H; report that only as an NLL-per-frame diagnostic. Do not apply a
chunk advantage to an action that the evaluator does not commit to.

## Hard-earned rules and footguns

### Objectives and probability models

- Independent future heads estimate marginal distributions. They are not a joint chunk policy.
- A within-frame factored action must concatenate ancestor embeddings into an MLP feature axis.
  Additive group embeddings erase which group supplied which feature.
- A zero action-conditioning matrix is useful only with a trainable residual output. Expect staged
  gradients: the output learns first; conditioning weights and embeddings learn on later updates.
- Normalize the three auxiliary MTP losses to one fixed total. This is an experimental and optimizer
  control. It prevents adding heads from silently increasing gradient scale. It is not an AWR rule.
- In ordinary one-step AWR, only the action aligned with the advantage receives the weight. Future
  auxiliary heads remain BC. Weight all future factors only after they form the executed macro-
  action scored by the same chunk critic.
- A trajectory-level weight and a macro-action weight can look identical in code. They differ in
  what the policy chose and what the critic evaluated. The macro critic must condition on the exact
  committed chunk and use an H-step target.
- Normalize AWR weights in log space. Direct exponentiation can underflow. A raw cap does not cap
  the final mean-normalized weight.
- Never compute returns inside a sampled window. The missing tail changes the target. A truncated
  replay has an unknown return, not a terminal value of zero.

### Attention and execution

- Persistent temporal KV is invalid for the intended rolling raw context after the buffer evicts a
  frame. Stored K/V values still contain contextual information from that dropped frame.
- SWA does not repair that semantic mismatch. It also creates a multi-layer receptive field larger
  than one attention window. Treat SWA as a separate package ablation.
- Dense attention at context 1,024 is memory-heavy and was not selected as the default package. P0
  uses context 256 and full causal recomputation.
- Sparse offsets `(1,5,9,13)` are not four consecutive executable actions. Do not execute them as a
  chunk. E4 first establishes dense `(1,2,3,4)` targets.
- Receding-horizon execution must clear a pending chunk on a reset, stock change, rejected boot, or
  early slot end. It must keep rebuilding the raw rolling context.

### Data and numerical safety

- Do not use `misc_as` as a model feature. It is multiplexed by action state and was not needed by
  P0. The compact projection omits it.
- One NaN crash came from libmelee converting nonfinite canonical `misc_as` directly to the integer
  legacy field `hitstun_frames_left`. The pinned fix maps missing or nonfinite `misc_as`,
  `state_age`, and `hitlag_left` to zero only in that legacy projection while preserving the raw
  canonical float. The project pins libmelee branch `exp/nonfinite-misc-as` at commit `222a399`.
  A separate guard rejects a torn nonfinite canonical frame ID per boot instead of crashing the
  whole wave.
- Reject nonfinite loss and gradient norm before `opt.step()`. Keep the previous uploaded checkpoint
  unchanged on failure.
- Stored character IDs are libmelee internal IDs. SLP start-block IDs are external and must be
  converted once at ingestion. Mixing these spaces silently conditions on the wrong character.
- A path-based 128-bit replay ID is required. The old 32-bit manifest ID had three collisions.
- Keep one replay per batch row and the 4,096-entry reservoir. Smaller batches and low diversity
  previously hurt control and increase correlation risk.
- Prefetching more work does not guarantee throughput. Prefetch 32 matched prefetch 2. Measure
  loader wait, replay diversity, and step time together.

### Evaluation and infrastructure

- Closed-loop results decide promotion. Teacher-forced NLL is diagnostic.
- Use paired H2H only when context length, decode dtype, KV mode, temperatures, and sampling filters
  match. Record the resolved protocol in every package.
- CPU boots can produce several instant-restart games. The evaluator finishes the current wave, so
  active game rows can exceed the requested boot count. This behavior is intentional.
- Bootstrap CPU uncertainty by boot, not by flattened games from one Dolphin process. Use mirrored
  configuration pairs for H2H statistics.
- A torn final replay frame is recoverable only by removing that incomplete frame and recording the
  repair. Do not silently ignore input-stat failures.
- H2H recovery scans used to upload immutable replay files two or three times. The shared uploader
  now deduplicates unchanged file versions while re-uploading changed result files.
- Run tests with a high file limit. Sandboxed W&B sockets and a 1,024-file limit can cause a false
  cascade. The real-Dolphin integration test can stall after a large threaded suite; rerun it alone,
  then require a clean full-suite pass.
- Compile, validation caching, and loader startup can overlap only with private RNG generators.
  Prove exact batch order and RNG parity before treating this as a systems-only change.
- Launch only from a clean pushed SHA. Run a no-rent audit, commit its record, repeat the audit on
  the final unchanged SHA, and confirm that no other experiment is active.
- Use 250 GB disk. A small disk can fill the compiler temp directory and make Torch compile appear
  to hang rather than fail clearly.
- Upload checkpoints before expensive evaluation. Treat upload failure as run failure. A stopped
  instance preserves local recovery data; a destroyed instance does not.

## Files touched in this session

This list is the union of paths in commits after the session base
`96d9ff31a26cae0cf247879faccd56ed04222421`. It includes files later restored or deleted.

### Repository and launch configuration

- `CLAUDE.md`
- `docker/on-start.sh`
- `pyproject.toml`
- `uv.lock`
- `scripts/launch_vast.py`
- `scripts/bench_dataloader.py`
- `notebooks/bench_seq_len.py`
- `notebooks/probe_sm120.py`
- `PART1_NOTES.md` was temporary and was deleted after its content was moved into the plans.

### Experiment code

- `experiments/020_awr.py`
- `experiments/022_awr_rank.py`
- `experiments/023_mtp_heads.py`

### Experiment records and plans

- `docs/experiments/action_chunk_roadmap.md`
- `docs/experiments/data_pipeline.md`
- `docs/experiments/e0_attention_ablation.md`
- `docs/experiments/e0_normalized_aux_bc.md`
- `docs/experiments/e1_output_head_capacity.md`
- `docs/experiments/e2_within_frame_factorization.md`
- `docs/experiments/e3_temporal_factorization.md`
- `docs/experiments/e4_dense_chunk_readiness.md`
- `docs/experiments/e5_primary_awr.md`
- `docs/experiments/e6_chunk_critic.md`
- `docs/experiments/e7_chunk_awr_execution.md`
- `docs/experiments/p1_matched_attention.md`
- `docs/experiments/p1_old_recompute_rescore.md`
- `docs/experiments/p2_temporal_kv_ablation.md`
- `docs/experiments/sequence_status.md`
- `docs/experiments/session_handoff.md`

### Data code

- `hal/data/behavior.py`
- `hal/data/extract.py`
- `hal/data/index.py`
- `hal/data/mds.py`
- `hal/data/policy_schema.py`
- `hal/data/schema.py`
- `hal/data/streaming_compat.py`
- `hal/streams.py`
- `hal/scripts/audit_policy_mds.py`
- `hal/scripts/project_policy_mds.py`
- `hal/scripts/subset_mds.py`
- `hal/scripts/upgrade_mds.py`

### Training and model code

- `hal/training/checkpoints.py`
- `hal/training/closed_loop.py`
- `hal/training/dataloader.py`
- `hal/training/features.py`
- `hal/training/trunk.py`

### Evaluation and simulator code

- `hal/eval/behavior.py`
- `hal/eval/cross_stage.py`
- `hal/eval/h2h.py`
- `hal/eval/harness.py`
- `hal/eval/matchups.py`
- `hal/eval/paired.py`
- `hal/scripts/analyze_replays.py`
- `hal/scripts/h2h.py`
- `hal/sim/inputs.py`
- `hal/sim/session.py`
- `hal/sim/vec.py`

`hal/eval/matchups.py` was touched during the session and ends with no net difference from the base.

### Tests

- `tests/experiments/test_002_flow_matching_rtc.py`
- `tests/experiments/test_016_spatial_features.py`
- `tests/experiments/test_020_awr.py`
- `tests/experiments/test_021_v6_features.py`
- `tests/experiments/test_022_awr_rank.py`
- `tests/experiments/test_023_mtp_heads.py`
- `tests/test_behavior.py`
- `tests/test_checkpoints.py`
- `tests/test_closed_loop_rings.py`
- `tests/test_cross_stage.py`
- `tests/test_dataloader.py`
- `tests/test_extract.py`
- `tests/test_h2h.py`
- `tests/test_inputs.py`
- `tests/test_launch_vast.py`
- `tests/test_matchups.py`
- `tests/test_paired.py`
- `tests/test_policy_schema.py`
- `tests/test_project_policy_mds.py`
- `tests/test_roundtrip.py`
- `tests/test_session.py`
- `tests/test_subset_mds.py`
- `tests/test_trunk.py`
- `tests/test_upgrade_mds.py`

The external libmelee fix is not stored in this repository. It is on
`github.com/ericyuegu/libmelee`, branch `exp/nonfinite-misc-as`, commit
`222a399cd1368d33458b4c36f4823f7220ca889d`. It changes `melee/console.py` and
`test_canonical.py`. `pyproject.toml` and `uv.lock` pin it here.
