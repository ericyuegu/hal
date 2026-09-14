# O55 trunk-history cross-attention

Registered before the first O55 launch.

## Question

Does direct access to every past causal trunk state improve action-chunk prediction and closed-loop play relative to forcing the temporal decoder to use only the final trunk state?

## Arms

All arms retain the previous controller frame and offset embeddings, the four within-frame autoregressive controller groups, the nonlinear action heads, and the direct final-trunk-to-logit skip.

- `baseline`: O52's decoder. Broadcast `W_h h_t` into every action token, then use four causal temporal self-attention blocks.
- `add-history`: the baseline plus one cross-attention layer after temporal block 1. Its queries are action tokens and its keys and values are every real trunk state `h_1...h_t`.
- `remove-state-bias`: `add-history` without the broadcast `W_h h_t`. This isolates whether the cross-attention path makes the state bias unnecessary.
- `replace-attention`: no `W_h h_t`. Replace the self-attention branch in each of the four temporal blocks with cross-attention to `h_1...h_t`; retain each block's feed-forward branch. The previous action still enters each action token, but action tokens do not attend to earlier action tokens.

Cross-attention uses the temporal width and head count, RMS-normalized queries and memory, rotary positions, and an explicit left-padding mask. A fixed local initialization seed keeps every parameter shared by the arms byte-identical at initialization. Dormant state-projection and self-attention parameters remain registered for checkpoint comparability, but active parameter counts exclude them.

At proxy scale, stored/active parameter counts are:

| Arm | Stored | Active |
| --- | ---: | ---: |
| `baseline` | 14,480,922 | 14,480,922 |
| `add-history` | 14,579,226 | 14,579,226 |
| `remove-state-bias` | 14,579,226 | 14,546,458 |
| `replace-attention` | 14,874,138 | 14,579,226 |

## Supervision and invariants

Each sampled trajectory supervises one action chunk through frame 20 at its final context prefix. The loss predicts offsets `1,2,3,4,5,6,9,12,16,20`. This replaces O52's 128 supervised prefixes per trajectory in every arm, including the baseline.

The proxy run uses batch size 512 and exactly `2^23` sampled trajectory windows: 16,384 optimizer updates with 512 warm-up updates. AWR starts at update 4,097. All arms use seed 0, 256 context frames, the 14.48M proxy trunk, AdamW master learning rate `0.0017`, betas `(0.9, 0.95)`, epsilon `1e-12`, weight decay `1e-4`, cosine decay, and global gradient clip 1.0. AWR uses beta 199.5, cap 3.5, gamma 0.99855, value weight 1.0, and far-offset weight 0.5.

Corpus, replay-ring order, identity conditioning and dropout, feature representation, validation cohort, optimizer schedule, and all random-number seeds are invariant. Training and live evaluation use the same decoder variant and trunk-output interface.

## Commands

```text
uv run experiments/055_history_cross_attention.py train --proxy --cfg.decoder-variant baseline --comment o55-a-baseline
uv run experiments/055_history_cross_attention.py train --proxy --cfg.decoder-variant add-history --comment o55-b-add-history
uv run experiments/055_history_cross_attention.py train --proxy --cfg.decoder-variant remove-state-bias --comment o55-c-remove-state-bias
uv run experiments/055_history_cross_attention.py train --proxy --cfg.decoder-variant replace-attention --comment o55-d-replace-attention
```

## Evaluation and decision rule

An arm is eligible only if it reaches update 16,384 with finite training and validation metrics and completes all 192 fixed gameplay matchups. The schedule has 94 oriented character pairs, 17 ego characters, 18 CPU characters, and SHA-256 `c7871050cfabe18f3df054e181ba4191675a382796e18143bf92504ccdbb6eb6`. Evaluation uses prediction 4, delay 2, replan 2, and at most 7,200 frames.

The primary comparisons are:

- `add-history - baseline`: the value of adding full trunk history.
- `remove-state-bias - add-history`: the value of retaining the broadcast final-state path.
- `replace-attention - remove-state-bias`: the value of replacing temporal self-attention after the state-bias removal.
- `replace-attention - baseline`: the complete proposed replacement.

For each comparison, compute treatment minus control in net stocks per active minute. Use the saved per-match rows, aggregate within each boot, and resample the 192 matched boots together for a seeded 2,000-resample two-sided 95% percentile interval. `add-history` or `replace-attention` supersedes `baseline` only if its direct interval against `baseline` is entirely positive. If neither arm meets that criterion, retain `baseline`. Offline NLL, accuracy, and rollout metrics are diagnostic only and cannot select an arm.
