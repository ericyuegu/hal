# O52 all-AdamW proxy sweep

Registered before the first O52 launch.

Preflight amendment, 2026-09-08: three smoke runs failed before loading a training batch. O50's 131,072-slot replay ring exceeded rank-1's 112,188 selected replays; the first correction to 65,536 slots then violated O50's 200-batch minimum reuse gap; the third run found that rank-1 has 1,189 validation samples rather than the 2,048 requested by O50. O52 uses the largest batch-aligned ring that fits rank-1, 112,128 slots and a 195-batch minimum reuse gap, and a fixed 1,024-sample validation set. No optimizer update occurred before these corrections.

## Question

Can AdamW train every O50 matrix stably, and which master learning rate is best at the 14.48M proxy scale?

The control is the O50 AdamW master rate, `4.25e-4`. The treatment rates are `1.0625e-4`, `2.125e-4`, `8.5e-4`, and `1.7e-3`.

## Invariants

- O50 architecture, initialization, AWR objective, batch size 512, cosine schedule, and global gradient clip 1.0.
- Exactly `2^30` supervised positions: 16,384 optimizer updates with 512 warm-up updates; AWR activates at update 4,097 as in O50.
- Seed 0 and AdamW betas `(0.9, 0.95)`, epsilon `1e-12`, and weight decay `1e-4` before O50 duration scaling.
- Only the immutable `ranked-anonymized-1-policy-world-v8` train and validation splits.
- A 112,128-slot replay ring with O50's generation settings and a 195-batch minimum reuse gap.
- A fixed 1,024-sample validation set.
- One Modal `RTX-PRO-6000` per arm. No automatic closed-loop evaluation during screening.

## Decision rule

Every arm must reach update 16,384 with finite loss, gradients, and final validation metrics. Rank eligible arms by final validation NLL, then far NLL, then rollout NLL, then lower learning rate. Evaluate the top two final checkpoints over the same fixed 96 closed-loop matchups before selecting a rate for larger training.

Post-training decision amendment, 2026-09-08: before any O52 gameplay result was available, the user rejected validation-based screening. Evaluate every eligible arm over the same fixed 96 closed-loop matchups. Select by net-stock cluster-bootstrap lower bound, then mean net stock per minute, then mean net damage per minute, then lower learning rate. Offline validation metrics are diagnostic only and cannot select an arm.

## Full-data follow-up

Registered 2026-09-09 before launch.

The treatment is O50 trained with AdamW on every parameter at master learning rate `8e-4`. It keeps the global pre-step gradient clip at `1.0`; it does not add per-head clipping. The control is the historical full-data O50 Muon-plus-AdamW run. No new matched Muon control will be launched.

The architecture, initialization, objective, 44-source corpus and mixture, 2,048-example validation cohort, replay-ring order, seed, identity-mask RNG, `8 * 2^30` supervised positions, 4,096-update warm-up, automatic evaluation schedule, weight decay, duration-scaled Adam betas and epsilon, semantic learning-rate roles, and readout fan-in scaling remain fixed. The AdamW hidden and input learning rate is approximately `2.828e-4` before schedule scaling.

Passive diagnostics are descriptive evidence for divergence. At each 25-update boundary they measure bounded logical-matrix samples, Adam moments, realized updates including weight decay, exact action-group gradient RMS, projection activations, centered legal logits, and the exact target-logit gradient L1 implied by each local cross-entropy term and its objective coefficient. The target-logit attribution is local to the output projection. It does not decompose gradients in the shared trunk. These diagnostics add no model pass or backward pass.

Run eligibility requires finite training and validation metrics through 131,072 updates and complete scheduled gameplay evaluations. Compare the treatment with the historical control by the registered gameplay rule: net-stock cluster-bootstrap lower bound, then mean net stock per minute, then mean net damage per minute. Offline metrics and passive diagnostics are for diagnosis only. They cannot select the treatment.
