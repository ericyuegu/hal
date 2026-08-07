# Action-model experiment status

Updated: 2026-08-07

## Systems prelude

| Stage | Axis | Status | Next gate |
| --- | --- | --- | --- |
| D0-D3 | Systems: storage width, reads per replay, prefetch, and replay mixing | Correctness and clean-cache GPU gate passed | Keep prefetch 2; prefetch 32 had no measurable benefit. |
| P0 | Scientific package: short full-causal context | Complete; final checkpoint and official FP16 evidence verified | Use as the short-context reference. |
| P1-old | Exploratory package: long context, SWA128, smaller batch | Complete; FP16 recompute rescore verified | Keep as exploratory decode evidence. |
| P1-match | Attention package at matched data and action vocabulary | Replacement 256-step systems gate running | Verify correctness and projected wall time before the full run. |
| P2 | Systems: temporal-KV decode efficiency | Fresh evaluation-only plan ready; exploratory evidence available | Compare KV and full recomputation on the same P1 checkpoint. |
| I1 | Optional scientific isolation: full causal versus SWA32 at fixed 256/512 geometry | Deferred | Run only if P0 versus P1 leaves the mask question unresolved. |
| Compute match | Optional systems and scaling study | Deferred | Write a separate plan before changing steps or data exposure. |

P0 and P1-match both process 131,072 tokens and about 16 million attention edges per layer and
step. They differ in context, batch, and mask. This is a practical package comparison, not pure
mask isolation. P1-old also differs in sampler and action vocabulary, so it is exploratory only.

## Scientific sequence

| Experiment | Axis | Status | Required result before the next stage |
| --- | --- | --- | --- |
| E0 | Loss scaling: offset 1 plus one fixed-total auxiliary BC mean | Code and tests ready; reference pending attention choice | Finish the selected package, final 96-match CPU evaluation, checkpoint, and evidence upload. |
| E1 | Head capacity: zero-init state-only residual MLP | Auditable plan ready; blocked on E0 | Same-seed E0 equality, live gradients, final CPU evaluation, and H2H against E0. |
| E2 | Within-frame conditional factorization and group order | Fresh two-arm plan ready; blocked on E1 | Beat the E1 capacity control in closed loop or give a clear diagnostic gain without a policy loss. |
| E3 | Temporal joint modeling and teacher-forcing exposure bias | Fresh sparse-joint plan ready; blocked on E2 | Show coherent teacher-forced and rollout-conditioned sparse predictions without harming control. |
| E4 | Dense temporal resolution and chunk readiness | Fresh bridge plan ready; blocked on E3 | Produce a correct dense `(1,2,3,4)` joint action sequence. A policy gain is not assumed. |
| E5 | Action-aligned AWR on the deployed primary policy | Fresh primary-only plan ready; blocked on policy selection | Weight only the action whose state advantage is defined; preserve auxiliary BC. |
| E6 | Chunk-conditioned value validity | Fresh critic plan ready; blocked on E4/E5 infrastructure | Pass held-out calibration, ranking, perturbation, and policy-sample checks for `Q(s, chunk)`. |
| E7 | Macro-action optimization and execution | Fresh matched-control plan ready; blocked on E6 | Use one chunk advantage on the joint likelihood and execute the same `H=2`, then `H=4`, action. |

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

At step 12,288:

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

This was the strongest periodic snapshot so far, but most matchups reached the frame limit and the
sample is small. P1 has no closed-loop result until the same checkpoint is reevaluated with full
recomputation.

## Immediate next actions

1. Finish the matched-P1 systems gate, then train P1-match on the accepted compact data path and action
   vocabulary.
2. Evaluate P1-match with full recomputation.
3. Pass the P1/P2 FP32 and FP16 parity gate on that checkpoint, including long rolling contexts,
   mixed resets, logits, and fixed-seed sampled actions.
4. Add and test the missing P2 decode timing counters.
5. Run the P2 temporal-KV evaluation and compare P0, P1-match recomputation, and P1-match KV.
6. Choose the practical attention package, declare the E0 reference, and copy its exact fixed
   configuration into the E1 launch audit.

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
This applies to the official P0 evaluation and later runs. The active P0 trainer uses its older
launch commit, so its final objects still require direct remote verification before teardown.

Future training runs also reject a nonfinite objective or gradient norm before `opt.step()`. The
error names the step and leaves the last uploaded checkpoint unchanged. The active P0 trainer does
not contain this later guard, so its live loss and gradient history remain part of every status
check.

Before the next launch, the complete suite passed 877 tests outside the restricted sandbox,
including real Dolphin integration tests. Six old 020/022 mini-training tests now use a no-op
uploader because they test model training and deliberately configure a dead R2 endpoint. Separate
checkpoint tests cover queue draining and production upload failure propagation. This run includes
the exact CPU evaluation protocol checks added at commit `c4ae635`.

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
