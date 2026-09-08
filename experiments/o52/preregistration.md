# O52 all-AdamW proxy sweep

Registered before the first O52 launch.

## Question

Can AdamW train every O50 matrix stably, and which master learning rate is best at the 14.48M proxy scale?

The control is the O50 AdamW master rate, `4.25e-4`. The treatment rates are `1.0625e-4`, `2.125e-4`, `8.5e-4`, and `1.7e-3`.

## Invariants

- O50 architecture, initialization, AWR objective, batch size 512, cosine schedule, and global gradient clip 1.0.
- Exactly `2^30` supervised positions: 16,384 optimizer updates with 512 warm-up updates; AWR activates at update 4,097 as in O50.
- Seed 0 and AdamW betas `(0.9, 0.95)`, epsilon `1e-12`, and weight decay `1e-4` before O50 duration scaling.
- Only the immutable `ranked-anonymized-1-policy-world-v8` train and validation splits.
- One Modal `RTX-PRO-6000` per arm. No automatic closed-loop evaluation during screening.

## Decision rule

Every arm must reach update 16,384 with finite loss, gradients, and final validation metrics. Rank eligible arms by final validation NLL, then far NLL, then rollout NLL, then lower learning rate. Evaluate the top two final checkpoints over the same fixed 96 closed-loop matchups before selecting a rate for larger training.
