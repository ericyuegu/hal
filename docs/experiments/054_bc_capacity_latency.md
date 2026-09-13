# O54: BC capacity-latency scaling

Status: latency preflight accepted; throughput probes not launched

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

## Buffered latency contract

The production timing contract is eager BF16 inference at batch size one on the
RTX 3060. For each width, measure the smallest `d <= 6` that has p99 latency
below `d / 60` seconds, using 50 warm-ups and 500 synchronized measurements.
Use `R=d`, `H=2d`, and execute offsets `d+1...2d` from the preceding plan.

Run the search three times in deterministic randomized width orders while one
real-time, native-resolution Vulkan Dolphin session is active on the same GPU.
The accepted delay is the largest of the three trial delays. If needed, repeat
the faster trials at that accepted delay so all three timing rows measure the
exact production configuration. A width is rejected if any accepted-delay
trial misses its deadline.

The v3 timing manifest hashes the joint architecture family, the three accepted
rows per width, the raw probe artifact, and the Dolphin executable and render
configuration. Both JSON files belong in `docs/experiments/` and must be
committed before a training launch. Earlier B32 and fixed-depth timing artifacts
are invalid.

The September 13, 2026 preflight used PyTorch 2.11.0 with CUDA 13.0 on the
12 GiB RTX 3060. The concurrent executable was the fingerprinted Slippi 3.6.4
build, running Vulkan at native EFB scale and paced at 60 Hz. The table reports
the worst p99 from the three accepted trials.

| Width | Delay | Horizon | Worst p99 | Deadline |
|---:|---:|---:|---:|---:|
| 128 | 1 | 2 | 15.004 ms | 16.667 ms |
| 256 | 1 | 2 | 15.513 ms | 16.667 ms |
| 384 | 2 | 4 | 26.253 ms | 33.333 ms |
| 512 | 2 | 4 | 28.563 ms | 33.333 ms |
| 640 | 2 | 4 | 29.722 ms | 33.333 ms |
| 768 | 2 | 4 | 31.780 ms | 33.333 ms |
| 896 | 3 | 6 | 43.355 ms | 50.000 ms |
| 1024 | 3 | 6 | 43.660 ms | 50.000 ms |

The accepted timing-manifest digest is
`28b844ba35c54137221cec7e8ab92fa6507b3ef7d0b8e27933659d04244f3cfa`.
The embedded raw-probe digest is
`b5070a742a22d4aaa69dbfd845d2d81ae1c59d2ca48dd5a46b0e6d267d198617`.

The final timing manifest determines the fixed-data frontier. Run
`study-plan` only after that manifest exists; it deduplicates fixed-`D` points
already present in the nine iso-compute trajectories and prints detached Modal
commands without starting them.

The accepted buckets select W256, W768, and W1024. W256 and W1024 are already
fixed-`D` iso-compute endpoints, so the final matrix adds only the W768
fixed-`D` trajectory: 10 runs and 18.645 EFLOPs in total.

After all endpoint evaluations finish, `analyze` accepts only completed
96-block evaluations bound to the same timing manifest. It fits a concave,
bracketed quadratic in `log10(N_eff)` for each compute line, fits BC parameter
count against compute, and writes the fixed-data gameplay-strength versus B1
p99 latency plot.

The commands printed by `study-plan` use the working `$275` allocation and
`$825` intervention threshold. They are not approval-ready until W256 and
W1024 RTX PRO 6000 smoke runs supply measured update throughput, wall-time, and
the revised cost estimate. No command printed by the planner executes Modal.
