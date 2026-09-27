# O54: latency-capacity scaling across data

Status: B200 training launched; W1024 relaunch pending after the initial
batch-512 execution OOM

O54 asks which deployed model is strongest at a given data budget when model
capacity and control latency increase together. Four independent pure-BC
trajectories each run through the same nested data endpoints:

| Endpoint | Updates | Cumulative replays | Supervised positions |
|---:|---:|---:|---:|
| `D=2^28` | 4,096 | 65,536 | 268,435,456 |
| `D=2^29` | 8,192 | 131,072 | 536,870,912 |
| `D=2^30` | 16,384 | 262,144 | 1,073,741,824 |
| `D=2^31` | 32,768 | 524,288 | 2,147,483,648 |

## Model family

Both the trunk and temporal decoder scale. The decoder width is half the trunk
width. The selected depth ratios are explicit experiment choices, not a rule
that can add architectures at launch time.

| Model | Temporal decoder | BC parameters | `N_eff` | Training through `2^31` |
|---|---:|---:|---:|---:|
| W256/L5 | W128/L2 | 5,436,825 | 18,593,046 | 0.240 EFLOPs |
| W512/L11 | W256/L4 | 39,024,281 | 119,289,622 | 1.537 EFLOPs |
| W768/L16 | W384/L6 | 124,712,857 | 373,733,654 | 4.816 EFLOPs |
| W1024/L21 | W512/L8 | 288,848,025 | 858,209,046 | 11.058 EFLOPs |

The four training trajectories total approximately 17.650 EFLOPs. Here
`N_eff = 2 * (trunk + inputs) + 14 * (temporal decoder + heads)`, which
accounts for two trunk passes and fourteen temporal-offset losses per
supervised position.

The invariant model components are O52's representation, projectile set
encoder, controller codec, trunk and temporal parameterization, readout
scaling, identity conditioning, and 256-frame context. O54 has no returns, AWR,
or value head.

Every optimizer update still contains 512 replay windows. W256, W512, and
W768 execute that batch at once. W1024 accumulates two 256-window
microbatches because a full-batch CUDA graph uses more than the B200's 178.35
GiB. Loss normalization remains over the complete optimizer batch, and AdamW,
clipping, scheduling, data endpoints, and replay admission advance once per
512-window update.

## Objective and optimizer

Each 276-frame window contains 256 context frames and 20 future frames. The
last 128 context positions are supervised at offsets
`1...12, 16, 20`. Offsets through the deployed horizon `H` have weight 1;
later offsets have weight 0.5. The loss divides by the sum of offset weights
and the number of valid supervised positions.

All learned parameters use AdamW with learning rate `8e-4`, betas
`(0.9, 0.95)`, epsilon `1e-12`, and global gradient clipping at 1.0. The
learning rate warms up linearly for 512 updates and is then constant. O52's
fan-in scaling remains on the final readouts.

Weight decay is fixed for the entire trajectory of each model:

```text
lambda = 1e-4 * (2^30 / 2^31)
         * (((2^31 / N) / (2^30 / N_W512)) ** 0.52)
```

Fixing the decay at the final `D=2^31` treatment lets the earlier checkpoints
belong to the same optimizer trajectory. It means those checkpoints do not
represent independently tuned shorter runs.

## Nested replay sampling

The source identity is the frozen ordered set of all 44 policy-world-v8 train
manifests. The union excludes Monotheon v8 rows 14,136 and 14,139, which are
copies of two Daniel replays. This leaves 1,295,368 unique replay rows. For
each cumulative endpoint, replay counts are allocated across sources in
proportion to their deduplicated row counts with deterministic
largest-remainder rounding. Each source contributes a physical row prefix.
The four cumulative selections are nested.

Training opens only the new range needed for a phase:

| Phase ending at | New replay rows |
|---:|---:|
| `D=2^28` | 65,536 |
| `D=2^29` | 65,536 |
| `D=2^30` | 131,072 |
| `D=2^31` | 262,144 |

The physical-shard loader uses 65,536 replay slots. It draws four generations
of eight distinct windows per replay. A batch contains one window from each of
512 distinct replay IDs. The 25-batch phase-block schedule gives a minimum
104-batch gap before another window from the same replay. Each phase has
exactly enough updates to consume 32 windows from every new replay once.

At a data boundary, CPU lookahead is empty before the checkpoint is written.
The current phase loader is then closed. Only after the checkpoint and
evaluation request exist does training construct the next phase loader. Thus a
`D=2^k` checkpoint cannot contain a batch from a later replay range. A
within-phase checkpoint stores the physical source cursor, ring descriptors,
per-replay window counters, optimizer and scheduler, identity-mask RNG, CPU
RNG, CUDA RNG, and W&B identity. A boundary checkpoint intentionally drops the
completed ring and records the hash of the next phase selection.

Normalization statistics and the player vocabulary remain the frozen O52
all-44 artifacts. They are representation metadata; future-phase replay rows
do not enter training batches.

## Buffered deployment

The production contract is eager inference with BF16 model parameters at batch
size one on the local RTX 3060 while one real-time Vulkan Dolphin session uses
the same GPU. Each exact architecture gets one randomized timing pass with 50
warm-ups and 500 synchronized calls per candidate delay.

For each architecture, select the smallest `d <= 6` whose p99 is less than
`d / 60` seconds. Deployment uses `R=d` and `H=2d`. One inference call
stages offsets `d+1...2d`; the policy executes the corresponding slice from
the preceding call. A new or reset slot executes neutral actions for its first
`d` frames while the first plan is staged. Reset drops context, executing
actions, and staged actions for that slot.

The timing manifest hashes the architecture family, raw samples, chosen rows,
GPU identity, Dolphin executable, and renderer configuration. Training rejects
a missing, changed, non-RTX-3060, non-B1, or non-eager manifest. Earlier O54
timing artifacts used a fixed four-layer decoder and are not valid for this
study. The preflight writes the manifest and raw probe report under `runs/`,
outside the repository.

The September 13, 2026 preflight used PyTorch 2.11.0 and CUDA 13.0 on the
12 GiB RTX 3060. It measured this deployment:

| Model | Delay | Horizon | p99 | Deadline | Margin |
|---:|---:|---:|---:|---:|---:|
| W256/L5 + decoder L2 | 1 | 2 | 13.903 ms | 16.667 ms | 2.764 ms |
| W512/L11 + decoder L4 | 2 | 4 | 29.132 ms | 33.333 ms | 4.201 ms |
| W768/L16 + decoder L6 | 3 | 6 | 48.868 ms | 50.000 ms | 1.132 ms |
| W1024/L21 + decoder L8 | 6 | 12 | 99.183 ms | 100.000 ms | 0.817 ms |

The W1024 model missed d4 and d5. Its d6 acceptance is valid under the
one-trial rule but has little timing margin. The manifest digest is
`51a31ddcd8b26c0b3817404b59aa53ce724593243352228e9194cd92c97381b7`.

## Gameplay evaluation and analysis

Every one of the 16 model-data endpoints gets one 96-block evaluation against a
level-9 CPU. The matchup schedule is deterministic and identical across
endpoints: 96 prior-weighted character-pair boots, port 1 as the model, the same
seed stage, and 7,200 frames per boot. Instant restart plays multiple matches
within a boot. Rates exclude countdown frames.

Evaluation uses the checkpoint's measured `d/R/H`, the same feature
projection and controller codec as training, masked ego identity, eager BF16
inference, and at most 32 concurrent Dolphin sessions on one RTX PRO 6000
evaluator.
It stores the checkpoint hash, full protocol, match rows, and aggregate
metrics. The primary response is pooled mean net stocks per active minute.
Uncertainty resamples the same complete 96 blocks across every endpoint.

For each deployed model, fit

```text
S(D) = S_28 + A * (1 - (D / 2^28)^(-alpha))
A >= 0, 0.01 <= alpha <= 4
```

through `D=2^30`. Compare the predicted complete model ordering with the
observed ordering at `D=2^31`. Only if that holdout ordering matches, refit
all four points and report the predicted winner among the four deployed models
at `D=2^34`. Otherwise, report the holdout failure and no `D=2^34` winner.

## Launch gate

Before paid training:

1. Run focused and repository checks.
2. Run finite-gradient checks for W256/L2 and W1024/L8 on the RTX 3060.
3. Produce the one-trial RTX 3060 timing artifacts under `runs/` and verify
   their hashes.
4. Dry-run all four B200 commands and verify the Modal Secret.
5. Verify observed throughput and memory after launch before accepting the
   projections.

The projection uses conservative update times of 0.25, 0.30, 0.40, and 1.00
seconds for W256 through W1024. These include margin over the closest
batch-512 B200 measurements in `cl0nqptn`, `k80jesjp`, `n8fm1otc`, and
`pq4g0ivv`. They project 17.7 aggregate training hours; the longest run,
W1024, takes 9.1 training hours.
[Modal's September 14 rates](https://modal.com/pricing) and the launcher's
32-core, 128-GiB training reservation total $8.7817/hour. Training therefore
costs about $156. Sixteen comparable 96-block RTX PRO 6000 evaluations add
about $22.
Allowing for startup and retries gives a $225 working estimate and
approximately 10--12 hours of elapsed study time when the four training apps
run in parallel. Warn at $450. Intervene at $675. Replace the throughput
projection with observed O54 telemetry after launch.
