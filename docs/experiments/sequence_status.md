# Action-model experiment status

Updated: 2026-08-07

## Systems prelude

| Stage | Axis | Status | Next gate |
| --- | --- | --- | --- |
| D0-D3 | Systems: storage width, reads per replay, prefetch, and replay mixing | Correctness passed; prefetch tuning active | Reduce mean loader wait from 0.209 seconds without changing batch diversity. |
| P0 | Scientific package: short full-causal context | Fresh plan ready | Restart from step 0 after the GPU data gate. |
| P1-old | Exploratory package: long context, SWA128, smaller batch | Training complete | Reevaluate the final checkpoint with full recomputation. |
| P1-match | Attention package at matched data and action vocabulary | Pending P0 | Retrain after P0 with only context, batch, and mask changed. |
| P2 | Systems: temporal-KV decode efficiency | Exploratory evidence available | Compare KV and full recomputation on the same P1 checkpoint. |
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
| E2 | Within-frame conditional factorization and group order | Planned | Beat the E1 capacity control in closed loop or give a clear diagnostic gain without a policy loss. |
| E3 | Temporal joint modeling and teacher-forcing exposure bias | Planned | Show coherent teacher-forced and free-running sparse future predictions without harming control. |
| E4 | Dense temporal resolution and chunk readiness | Planned; prior dense result was slightly worse | Produce a correct dense `(1,2,3,4)` joint action sequence. A policy gain is not assumed. |
| E5 | Action-aligned AWR on the deployed primary policy | Planned | Weight only the action whose state advantage is defined; preserve auxiliary BC. |
| E6 | Chunk-conditioned value validity | Planned | Pass held-out calibration, ranking, perturbation, and policy-sample checks for `Q(s, chunk)`. |
| E7 | Macro-action optimization and execution | Planned | Use one chunk advantage on the joint likelihood and execute the same `H=2`, then `H=4`, action. |

## Current evidence

Vast instance `47034073` and W&B run `19sowpt8` train the P1 architecture and use the P2 decode
path. Keep this as exploratory evidence, not the E0 reference.

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

1. Run the 256-step clean-cache GPU data gate.
2. Relaunch P0 from step 0 and finish its fixed evaluations.
3. Record the final `19sowpt8` checkpoint, W&B history, rows, replays, and timing.
4. Pass the P1/P2 FP32 and FP16 parity gate, including long rolling contexts, mixed resets, logits,
   and fixed-seed sampled actions.
5. Evaluate P1-old with full recomputation over 96 CPU matchups.
6. Train P1-match on the accepted data path and action vocabulary.
7. Compare P0, P1-match full recomputation, and P1-match temporal KV.
8. Choose the practical attention package, declare the E0 reference, and copy its exact fixed
   configuration into the E1 launch audit.

## Evaluation rules

- Closed-loop policy results decide promotion. Offline NLL and accuracy explain results.
- Use 32 fixed CPU matchups for periodic checks and 96 for final comparisons.
- Use 64 mirrored H2H configurations, or 128 games, for each challenger against its reference.
- Keep matchup seeds, decode seeds, temperature, frame budget, and concurrency rules fixed.
- Save match rows, replay files, worker logs, and explicit decode protocols.
- Report paired stock differences and uncertainty intervals.
- The CPU protocol does not report full game win rate. Label terminal-game results as a subset.
- Treat H2H as sensitive but possibly non-transitive. Do not hide a CPU regression behind H2H.
- Use at least three paired training seeds for a claim. Use five for a final result.
- Do not select a model from auxiliary NLL, teacher-forced metrics, or throughput alone.

## Infrastructure rules

- Run one training experiment at a time.
- Use a 1 TB disk and set `cache_limit_gb` to about 900 for future clean runs.
- Require at least 128 GB of system RAM and audit CPU count, network speed, reliability, and price.
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
