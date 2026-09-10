# O53 fixed polar controller

Registered before the first O53 launch.

## Question

Does Slippi-AI's fixed polar stick layout improve closed-loop play relative to O52's fixed Cartesian stick classes when all other inputs are fixed?

## Treatment and control

The treatment is `experiments/053_polar_controller.py` at the commit produced from this preregistration. It replaces only the stick codec:

- Main stick: origin plus three rings with 4, 16, and 64 angles, for 85 classes.
- C-stick: origin plus two rings with 4 and 8 angles, for 13 classes.
- Each stick token adds a learned projection of normalized `[radius, sin(angle), cos(angle)]` to its class embedding. The origin semantic value is `[0, 0, 0]`.

The control is [O52 run `yxy8brhx`](https://wandb.ai/ericyuegu/hal/runs/yxy8brhx), from source commit `393b94d`. W&B records source artifact `source-hal-experiments_052_adamw_temporal_awr.py:v3` with digest `7e6fc5b752aa5f4d40d33098f4a90c9b`. The final checkpoint is the immutable R2 object:

`r2://hal/runs/260908-220500_052_adamw_temporal_awr_adamw052-d256-L16-h4-Lc256-t128x4-o1-2-3-4-5-6-9-12-16-20-d2r2-nonlinear-head-trunk-skip-projectiles-v8-all-adamw-alr0.0017-awd0.0001-awr-v-near-b199.5-g0.99618-wu512__o52-lr1p7e3/checkpoints/step-0016384.pt`

Its training process is marked crashed because the job ended after upload, but its log records the update-16,384 checkpoint and its W&B run contains the complete fixed 96-matchup evaluation.

## Resolved configuration

Both arms use the proxy architecture: `d_model=256`, 16 trunk layers, 4 heads, context 256, temporal width 128, 4 temporal layers, 2 temporal heads, FF width 384, group-head width 128, action embedding width 32, and offsets `1,2,3,4,5,6,9,12,16,20`.

Both arms use AdamW for every parameter with master learning rate `0.0017`, betas `(0.9, 0.95)`, epsilon `1e-12`, weight decay `1e-4`, cosine decay, and global gradient clip 1.0. They use batch size 512, seed 0, 512 warm-up updates, and exactly `2^30` supervised positions over 16,384 optimizer updates.

Both arms use only `ranked-anonymized-1-policy-world-v8`, selection SHA-256 `ad28edec1ad37565707d1bb0fb2262d94f05617fc957d4d8c8b234979cf3a381`, source manifest SHA-256 `b97eab90e761bcf2bf03b48981f0ab6acc1ac3057157c58ae0c5a72c76c43bd8`, schema-v7 MDS, a 112,128-slot replay ring, and the fixed 1,024-example validation cohort.

The control has 14,480,922 parameters. O53 has 14,490,994 because its stick embeddings and output heads have 24 more classes and its two stick semantic projections each have one more input.

## Invariants

- Architecture, initialization, AWR objective and calibration, optimizer, schedule, batch order, seed, corpus, identity artifacts, validation cohort, and training duration.
- Four separate controller groups and O52's autoregressive order: C-stick, main stick, triggers, buttons.
- The 256-way button vocabulary, 25-way trigger-pair vocabulary, quantization, semantic projections, and trigger-dependent button legality mask.
- Closed-loop protocol: prediction 4, delay 2, replan 2, maximum 7,200 frames, and the same 96 fixed matchups.

## Treatment command

```text
uv run experiments/053_polar_controller.py train --proxy --comment o53-polar
```

## Decision rule

The treatment is eligible only if it reaches update 16,384 with finite training and validation metrics and completes all 96 fixed gameplay matchups. Compare eligible treatment and control by net-stock cluster-bootstrap lower confidence bound, then mean net stock, then mean net damage. Offline NLL and class accuracy are diagnostic only because the stick class partitions differ. Main-stick and C-stick reconstruction MSE are directly comparable codec diagnostics and cannot select the treatment.
