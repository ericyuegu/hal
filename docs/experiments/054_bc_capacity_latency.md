# O54: BC capacity-latency scaling

Status: implementation validation; no training launched

O54 measures compute-optimal behavior-cloning capacity under a buffered
deployment-latency constraint. It keeps O52's representation, temporal decoder,
initialization, identity conditioning, context length, and statistical batch.
It removes AWR, returns, and the value head.

## Model family

The treatment is joint trunk width and depth. The aspect ratio follows O48's
nearest-integer `depth = width / 48` family. O52's four-layer temporal decoder
stays fixed in depth and scales proportionally in width.

| Width | Trunk depth | Total BC parameters | `N_eff` |
|---:|---:|---:|---:|
| 128 | 3 | 1,692,953 | 7,025,686 |
| 256 | 5 | 5,764,505 | 23,180,566 |
| 384 | 8 | 17,094,169 | 58,373,654 |
| 512 | 11 | 39,024,281 | 119,289,622 |
| 640 | 13 | 70,178,585 | 203,175,958 |
| 768 | 16 | 121,763,737 | 332,445,974 |
| 896 | 19 | 194,172,953 | 507,886,102 |
| 1024 | 21 | 278,362,265 | 711,408,406 |

`N_eff = 2 * (trunk + inputs) + 14 * (temporal decoder + heads)`.

## Iso-compute endpoints

The three compute lines are anchored by `D=2^30` at W256, W512, and W1024.
Update counts are rounded to the nearest 32 updates. Each row is a separate
optimizer trajectory.

| Line | Width | Updates | Supervised positions | Approximate FLOPs |
|---|---:|---:|---:|---:|
| C1 | 128 | 54,048 | 3,542,089,728 | 1.4931e17 |
| C1 | 256 | 16,384 | 1,073,741,824 | 1.4934e17 |
| C1 | 384 | 6,496 | 425,721,856 | 1.4911e17 |
| C2 | 256 | 84,320 | 5,525,995,520 | 7.6857e17 |
| C2 | 384 | 33,472 | 2,193,620,992 | 7.6830e17 |
| C2 | 512 | 16,384 | 1,073,741,824 | 7.6852e17 |
| C3 | 512 | 97,696 | 6,402,605,056 | 4.5826e18 |
| C3 | 768 | 35,072 | 2,298,478,592 | 4.5847e18 |
| C3 | 1024 | 16,384 | 1,073,741,824 | 4.5832e18 |

For the fixed-`D=2^30` latency frontier, train the largest width in each
measured latency bucket. A frontier point not already present at `D=2^30`
needs a separate trajectory.

## Optimizer

All learned parameters use AdamW. The learning rate is `8e-4`, betas are
`(0.9, 0.95)`, epsilon is `1e-12`, and gradient clipping is `1.0`. Final
readouts retain O52's fan-in learning-rate scaling. The schedule is 512 updates
of linear warm-up followed by a constant learning rate.

Weight decay follows the [Power Lines](https://arxiv.org/abs/2505.13738)
AdamW-timescale rule:

```text
lambda = lambda_ref * (D_ref / D) * ((D / N) / (D_ref / N_ref))^0.52
```

The declared anchor is the O54 W512/L11 BC model at `D_ref=2^30`, with
`lambda_ref=1e-4`. `N` is the literal BC parameter count. Because weight decay
depends on the endpoint's `D` and `N`, a long trajectory cannot provide a
shorter endpoint for the scaling fit.

## Invariants

All endpoints use the same deduplicated all-44 v8 replay manifest, physical
shard order, nested replay policy, objective offsets, batch size, supervised
positions per window, seeds, and gameplay matchups. Model geometry, endpoint
length, derived weight decay, and measured buffered timing are the treatment.

The v2 timing manifest hashes the full joint architecture family. All latency
results from the earlier fixed-depth v1 family are invalid for this treatment
and must be measured again before training.
