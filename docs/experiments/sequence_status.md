# Action-model experiment status

Updated: 2026-08-07

## Systems prelude

| Stage | Axis | Status | Next gate |
| --- | --- | --- | --- |
| D0-D3 | Systems: storage width, reads per replay, prefetch, and replay mixing | Correctness and clean-cache GPU gate passed | Keep prefetch 2; prefetch 32 had no measurable benefit. |
| P0 | Scientific package: short full-causal context | Complete; final checkpoint and official FP16 evidence verified | Use as the short-context reference. |
| P1-old | Exploratory package: long context, SWA128, smaller batch | Complete; FP16 recompute rescore verified | Keep as exploratory decode evidence. |
| P1-match | Attention package at matched data and action vocabulary | Complete; paired result does not promote P1; audited evidence verified | Keep P0 as E0. Use P1 only for the P2 systems ablation. |
| P2 | Systems: temporal-KV decode efficiency | Complete; parity failed and both closed-loop arms are verified | Keep full rolling-window recomputation. |
| I1 | Optional scientific isolation: full causal versus SWA32 at fixed 256/512 geometry | Deferred | Run only if P0 versus P1 leaves the mask question unresolved. |
| Compute match | Optional systems and scaling study | Deferred | Write a separate plan before changing steps or data exposure. |

P0 and P1-match both process 131,072 tokens and about 16 million attention edges per layer and
step. They differ in context, batch, and mask. This is a practical package comparison, not pure
mask isolation. P1-old also differs in sampler and action vocabulary, so it is exploratory only.

P0 remains the downstream default. P1 must improve final CPU stock differential and have positive
paired H2H stock difference with more ahead than behind configurations. Mixed evidence keeps P0.
Decode speed alone cannot select the policy package.

## Scientific sequence

| Experiment | Axis | Status | Required result before the next stage |
| --- | --- | --- | --- |
| E0 | Loss scaling: offset 1 plus one fixed-total auxiliary BC mean | Complete; P0 is the fixed reference | Keep its checkpoint, data, objective, and evaluation protocol fixed. |
| E1 | Head capacity: zero-init state-only residual MLP | Active on Vast instance `47136334`, W&B `q3aojgfm` | Same-seed E0 equality, live gradients, final CPU evaluation, and H2H against E0. |
| E2 | Within-frame conditional factorization and group order | Local implementation and tests complete; GPU gate blocked on E1 | Beat the E1 capacity control in closed loop or give a clear diagnostic gain without a policy loss. |
| E3 | Temporal joint modeling and teacher-forcing exposure bias | Matched null-condition and action-condition plans ready; blocked on E2 | Beat the null-condition capacity control without harming control. |
| E4 | Dense temporal resolution and chunk readiness | Fresh bridge plan ready; blocked on E3 | Produce a correct dense `(1,2,3,4)` joint action sequence. A policy gain is not assumed. |
| E5 | Action-aligned AWR on the deployed primary policy | Primary-only plan with a fixed critic warm-up ready; blocked on policy selection | Pass the value gate, then weight only the action whose advantage is defined. |
| E6 | Chunk-conditioned value validity | Critic plan with value warm-up ready; blocked on E4/E5 infrastructure | Pass held-out calibration, ranking, perturbation, and policy-sample checks for `Q(s, chunk)`. |
| E7 | Macro-action optimization and execution | Fresh matched-control plan ready; blocked on E6 | Use one chunk advantage on the joint likelihood and execute the same `H=2`, then `H=4`, action. |

E1 reached step 9,600 without an error. At step 8,192, validation action NLL was 1.071 bits per
frame. The 32-boot CPU sweep completed in 166 seconds with no crash. Stocks taken and lost per
active minute were 0.670 and 1.388. This point is weaker than P0 at the same step and is not a final
decision. The checkpoint and complete periodic evidence uploaded to R2.

E1 also exposed a systems limit. Median loader wait was 0.136 seconds in a 0.281-second median step,
while the host was mostly idle and showed no disk-read wait. The E2 plan now includes a one-batch
background-prefetch gate with exact data-order and RNG parity. It does not change the replay
reservoir or scientific sample stream. The local gate passes, including 32 constructed batch hashes,
34 real compact-MDS batches, controlled overlap, errors, early close, and the complete 939-test
repository suite. E2's first GPU steps still need to measure the live wait reduction.

## Current evidence

P0 completed all 16,384 steps in W&B run `obx3o3az`. Final validation NLL at offsets 1, 5, 9, and
13 was 1.029, 2.650, 3.379, and 3.851 bits per frame. Button log loss was 0.0305. The verified
56,698,679-byte final checkpoint has SHA-256 `5d12d010fa3acd1ec07bd86a8e85d2cbb84c584a77b9b79e90dc6fcf03c32e4b`.
The official FP16 full-recompute evaluation completed on Vast instance `47107185`. All 96 boots
succeeded. It produced 118 active games and two countdown-only tails. Stocks taken and lost per
active minute were 0.777 and 1.468. Damage dealt and taken per active minute were 129.6 and 116.4.
The official rows and 122 replay files are in R2 under `manual_evals/p0-final-fp16`; labeled metrics
are in W&B run `obx3o3az`.

Vast instance `47034073` and W&B run `19sowpt8` train the P1 architecture and use the P2 decode
path. Keep this as exploratory evidence, not the E0 reference.

The P1-old FP16 full-recompute rescore completed on Vast instance `47110149`. All 96 boots
succeeded and produced 120 active games. Stocks taken and lost per active minute were 0.798 and
1.405. Damage dealt and taken per active minute were 132.5 and 116.1. The complete rows and 125
replay files are in `manual_evals/p1-old-final-recompute-fp16`. The sweep reached the 25-minute
evaluation warning. It used 96 concurrent boots, so it is not a matched decode comparison with P0
or the historical KV evaluation.

The matched-P1 replacement gate completed on Vast instance `47115519` and W&B run `6ydiy4kq`.
Every logged batch had 128 distinct replays and zero adjacent reuse. All numerical checks passed.
Median warm step time was 0.296 seconds, which projects full training to about 81 minutes. The
verified gate checkpoint is in R2. The full matched-P1 run is approved.

The matched-P1 run is W&B `46zi7fgo` on Vast instance `47117879`. Training completed all 16,384
batches with finite metrics and 128 distinct replays per batch. Final validation NLL at offsets 1,
5, 9, and 13 was 1.011, 2.590, 3.297, and 3.761 bits per frame. The loader had one repeatable
9-second stall at each epoch boundary, which cost about 36 seconds over the run. The verified
56,698,807-byte final checkpoint has SHA-256
`8e9b04c91aa76d1ba49a910c82f1328bc1b0dc3ce7dabf3e9018cb556d964148`; every floating model and
optimizer tensor is finite. The final CPU and H2H evaluations are complete.

The final P1 CPU sweep completed all 96 boots with no crash and produced 122 active games. Stocks
taken and lost per active minute were 0.781 and 1.461. Its only saved protocol difference from the
official P0 sweep is concurrency: P1 used 96 concurrent boots and P0 used 32. The tiny point gain is
therefore not promotion evidence. P2 pins both decode arms to 32 concurrent boots, so its recompute
arm will provide the matched systems rerun.

The first P1 H2H orientation completed 64 of 64 games. Its audit found 45 budget-cut replays with a
torn final frame, which made input-stat parsing fail. The files are recoverable by removing only
that incomplete frame. Future H2H runs now repair this case and fail on any remaining missing input
statistics. P1 promotion remains gated on a separate audited record file covering both
orientations; the original launch records and replays will be preserved.

H2H record schema 3 now stores the repair decision in `replay_trimmed` instead of leaving it only
in process logs.

The complete P1 H2H outcome has 128 of 128 games and 64 mirrored pairs. P1 had 45 stock leads, 48
deficits, and 35 ties. Its mean stock difference was -0.031 per game. The paired configuration
mean was -0.062; signs were 24 ahead, 21 behind, and 19 tied. Damage difference was -8.76 per active
minute, with a 95% interval below zero. P1 therefore fails the required positive paired-mean gate.
P0 is the selected E0 reference.

The repaired package is complete under `h2h_final_audited`: 128 records, 128 replays, 96 declared
final-frame trims, 256 input-stat blocks, and a passing input tripwire. Rclone found 134 matching
files and zero differences. W&B is finished, and the P1 Vast instance destroyed itself. P2 is no
longer blocked on P1 evidence.

P2's full command passed a no-rent launcher check at commit `cface89`. The best qualifying RTX 4090
offer had 252 GB RAM, DLPerf 125.6, and an effective price of $0.824 per hour. The command preserved
the parity, 32-way recompute, and 32-way KV order.

The final exact-SHA check passed at `335b7e7`, and P2 launched on Vast instance `47129969`. Its
strict parity gate failed with finite values. FP32 had 14 sampled-action mismatches in 6,195
comparisons; FP16 had 32. The maximum group-logit errors were `0.08052` and `0.09961`. The command
stopped before either closed-loop arm. The failed record is verified in R2 with SHA-256
`4bc514a79dde89c011c387ad19065bb05778c6613c5501cf734bf0a46278d424`. Instance `47129969` was
destroyed after verification.

The P2 diagnostic rerun launched from exact commit `4bc0f1e` on Vast instance `47132921`. Its RTX
4090 host has 251 GB RAM, an 80 GB disk, DLPerf 125.6, and an effective price of $0.714 per hour. It
became ready at 18:56:56 PDT after a 13-minute image pull.

P2 completed and destroyed its instance. The schema-2 parity record is finite but failed: 14 FP32
and 32 FP16 sampled-action mismatches in 6,195 slot-frames. Fixed-frame diagnostics show drift from
both Flex versus dense attention and dense recomputation versus incremental decoding. KV reduced
model time per row from 0.390 ms to 0.146 ms, but wall time only fell from 672 to 507 seconds. The
boot-matched KV-minus-recompute net-stock delta was `-0.094` per active minute, with 95% interval
`[-0.269, 0.077]`. R2 contains the parity record, 231 JSON rows including 228 active rows, and 240
replays. W&B contains both labeled metric sets. No Vast instance is active.

E1's state-only residual MLP is implemented on `exp/e1-output-head-capacity`. It adds 860,044
parameters and starts with exact E0 logits, objectives, and sampled actions. Tests cover its first-
and second-update gradient path, optimizer ownership, checkpoint compatibility, and the pinned P0
checkpoint SHA-256. The focused suite passed 60 tests; the full repository suite passed 916 tests
in 134.73 seconds. Its 250 GB launch payload passed a no-rent audit and found three qualifying RTX
4090 offers.

The final E1 no-rent audit passed at pushed commit `bba9b87`, then E1 launched from that unchanged
SHA on Vast instance `47136334`. Its RTX 4090 host has 251 GB RAM, 250 GB disk, DLPerf 125.6, and an
effective price of $0.755 per hour. It became ready in 43 seconds. W&B run `q3aojgfm` has the correct
state-MLP configuration. Startup verified the source SHA, `sm_89`, P0 checkpoint hash, parameter
count, and concurrent validation-cache and training-prefetch start. No other experiment is active.

The first routine check found E1 healthy at step 4,350. Warm steps were usually 0.27 to 0.30
seconds, GPU memory use was 12.0 of 49.1 GiB, and no numerical failure appeared. Step-4,096
validation NLL was 1.109 bits per frame. Its 32-boot CPU sweep finished in 166 seconds with no
crash, 41 active matches, 0.639 stocks taken, and 1.534 stocks lost per active minute. The early
control point is weak but does not decide the final gate. Its checkpoint, 44 replays, rows, metrics,
result, and worker log uploaded to R2.

The exploratory P1-old run reported this periodic snapshot at step 12,288:

- Validation action NLL: 1.027 bits per frame.
- Button log loss: 0.031.
- CPU evaluation: 326 seconds, 42 matches, no crashes.
- Stocks taken per minute: 0.911, 95% CI `[0.751, 1.081]`.
- Stocks lost per minute: 1.502, 95% CI `[1.316, 1.702]`.
- Damage dealt per minute: 134.3, 95% CI `[123.0, 146.1]`.
- Damage taken per minute: 119.0, 95% CI `[112.0, 126.1]`.
- Dead-frame fraction: 0.0221.
- Derived mean stock difference: -0.881.
- Terminal subset: 10 games and one win. This is not a full-schedule win rate.
- The step checkpoint, 44 replays, and match rows were queued for upload.

This was its strongest periodic snapshot, but most matchups reached the frame limit and the sample
is small. It is not evidence for the active matched-P1 run. The P1-old full-recompute rescore above
is the valid closed-loop result for that checkpoint.

## Immediate next actions

1. Continue E1 through step 16,384, final CPU evaluation, and H2H if no declared gate fails.
2. Verify E1's checkpoint, rows, replays, W&B history, R2 files, cost, and final decision.
3. Audit E2-S's final launch SHA and command, then run its GPU gate only after E1 is complete.

## Remaining run units

Run these units one at a time. A named stage can contain more than one unit:

1. P2 checkpoint parity, recompute evaluation, and KV evaluation in one evaluation-only job.
2. E1 state-only MLP screening run.
3. E2-S stable-prefix factorization run.
4. E2-I intent-first factorization run.
5. E3-C null-condition capacity-control run.
6. E3-T previous-action temporal-conditioning run.
7. E4 dense-offset chunk-readiness run.
8. E5 primary-only AWR run. It pauses after update 2,047 for its value gate, then continues in the
   same 16,384-step process only if the gate passes.
9. E6 chunk-critic runs for three seeds. Each includes value warm-up, Q2, Q4, and matched state-only
   controls.
10. E7-H2 execution-only evaluation, macro-BC training, and macro-AWR training. Run the paired H2H
    comparison only after both training arms finish.
11. E7-H4 repeats the execution-only, macro-BC, and macro-AWR units only if H2 passes.

The first screening run at E1 through E5 does not support a final architecture claim. A promoted
result needs the predeclared multi-seed confirmation. Do not start confirmation seeds while a later
screening stage is still needed to answer the next design question.

The fresh matched-P1 plan is `docs/experiments/p1_matched_attention.md`.
The exploratory P1 recompute rescore plan is `docs/experiments/p1_old_recompute_rescore.md`.
The P2 decode plan is `docs/experiments/p2_temporal_kv_ablation.md`.
The E2 factorization plan is `docs/experiments/e2_within_frame_factorization.md`.
The E3 temporal plan is `docs/experiments/e3_temporal_factorization.md`.
The E4 dense-chunk plan is `docs/experiments/e4_dense_chunk_readiness.md`.
The E5 AWR plan is `docs/experiments/e5_primary_awr.md`.
The E6 critic plan is `docs/experiments/e6_chunk_critic.md`.
The E7 macro-action plan is `docs/experiments/e7_chunk_awr_execution.md`.

The current branch treats any failed R2 upload as a failed process after draining the complete
upload queue. Vast will stop that instance for recovery instead of destroying the only local copy.
This applied to the official P0 evaluation and applies to later runs. P0's final objects have been
verified in R2.

Current training runs reject a nonfinite objective or gradient norm before `opt.step()`. The error
names the step and leaves the last uploaded checkpoint unchanged. P0 launched before this guard was
added; its complete finite loss and gradient history was checked separately.

Before the next launch, the complete suite passed 892 tests outside the restricted sandbox,
including real Dolphin integration tests. Six old 020/022 mini-training tests now use a no-op
uploader because they test model training and deliberately configure a dead R2 endpoint. Separate
checkpoint tests cover queue draining and production upload failure propagation. This run includes
the exact CPU evaluation protocol checks added at commit `c4ae635`.

The latest run completed in 136 seconds at commit `d062eb4`. A first sandboxed attempt failed when
W&B offline mode could not open its local socket, then exhausted the 1,024-file process limit during
the cascade. The normal unsandboxed run with the project file limit passed all 892 tests. This was a
test-host restriction, not a model or evaluator failure.

The complete suite passed again at commit `16c3e46`: 892 tests in 135 seconds. The focused P2 suite
passed 77 tests and skipped six GPU-only tests, and Ruff passed. The skipped cases require the final
P1 checkpoint and run as the P2 GPU parity gate before either decode sweep.

At commit `1d95e4f`, P2's synthetic parity schedule was corrected so one slot stays continuous
through more than 1,024 raw-window evictions while two other slots test asynchronous resets. The
focused suite passed 78 tests and skipped six GPU-only cases. The complete suite passed 893 tests in
136 seconds.

## Evaluation rules

- Closed-loop policy results decide promotion. Offline NLL and accuracy explain results.
- Use 32 deterministic character-pair CPU boots for periodic checks and 96 for final comparisons.
- Use 64 mirrored H2H configurations, or 128 games, for each challenger against its reference.
- Keep character schedules, decode seeds, temperature, frame budget, and concurrency rules fixed.
- Save match rows, replay files, worker logs, and explicit decode protocols.
- Report separate CPU estimates and uncertainty intervals. Later instant-restart stages are random,
  so boot-and-ordinal row alignment is only a diagnostic, not a paired causal estimate.
- If a character-matched CPU diagnostic is useful, pool games within each boot and bootstrap boots.
  Do not bootstrap flattened games from one Dolphin process as independent samples.
- Report paired stock differences for mirrored H2H configurations.
- The CPU protocol does not report full game win rate. Label terminal-game results as a subset.
- Treat H2H as sensitive but possibly non-transitive. Do not hide a CPU regression behind H2H.
- Use at least three paired training seeds for a claim. Use five for a final result.
- Do not select a model from auxiliary NLL, teacher-forced metrics, or throughput alone.

## Infrastructure rules

- Run one training experiment at a time.
- Use a 250 GB disk and `cache_limit_gb=128` for the compact policy dataset.
- Require at least 200 GB of system RAM and audit CPU count, network speed, reliability, and price.
- Use `require_flex=True`; a dense fallback is not a valid timing comparison.
- Target 3.0-3.5 hours through evaluation and upload. Flag a projection above 3.5 hours.
- Flag startup over 30 minutes, warm steps over 0.5 seconds, GPU use below 80%, or a periodic CPU
  evaluation over 25 minutes.
- Record provisioning, downloads, cache fill, compile, validation, training, CPU evaluation, H2H,
  upload, peak memory, and host facts.
- Upload checkpoints every 2,048 steps. Verify final checkpoints and evaluation evidence before
  allowing the instance to stop.
- Write a fresh auditable plan before implementing or launching each experiment.

## Deferred work

- Optional I1 pure-mask and compute-matched attention studies.
- Plan carry-over, temporal ensembling, and inpainting.
- Prefix critics and adaptive chunk length.
- Next-state and reward prediction.
- Latent world models and model-predictive control.
- Online data collection and DAgger-style correction.
- A correctness audit of `experiments/018_bpe_rle.py` before any action-duration or RLE revisit.
