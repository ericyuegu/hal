# O45: legacy sparse endpoint flow with inference-time RTC

O45 physically reduces O38's 355-dimensional categorical endpoint to O43's
57 legacy classes. It keeps O38's d256 trunk, 32 causal prefixes, ten sparse
offsets, equal-group endpoint loss, four NFEs, seed 0, ranked-anonymous data,
and 16,384 updates.

Codec version 2 uses vocabularies `(6, 37, 9, 5)`, the exact legacy centers,
the early-release button reducer, and separate digital and fused analog
shoulders. The reduced dimensions are used in every flow state, projection,
head, loss, and decoded endpoint. The default model has 8,865,957 parameters
and 15,395,888 MACs per NFE. The static estimator reports 1,824,481,664 FLOPs
per sparse replan.

## Runtime modes

`P` is the number of frames returned by one prediction, `S` is the execution
stride, and `D` is the committed prefix. Thus `P = S + D`.

- H4-D0: `(P,S,D)=(4,4,0)`. This is the clean codec-only comparison with O38.
- H4-D1: `(P,S,D)=(5,4,1)`.
- H1-D1: `(P,S,D)=(2,1,1)`.

Training is unconditioned. At RTC inference, O45 quantizes the committed raw
actions, places their categorical one-hot endpoints in the flow state before
every Euler NFE, and restores the exact raw prefix in the returned plan. The
bidirectional flow attention can read those pinned tokens. Slot-keyed flow
randomness advances by `S`, not `P`. Compiled decoders are cached separately
by hardware bucket and committed-prefix length.

Each final evaluation uses 96 boots and a separate evidence directory. H4-D0
also retains the ordinary `eval/*` W&B metrics. Evidence records `P`, `S`, `D`,
codec version, group vocabularies, and `hard_categorical_prefix_clamp`.

## Dense-16 addition

The second requested checkpoint uses `(P,S,D)=(16,12,4)`. O38's sparse target
set has only six consecutive frames, so this checkpoint trains dense offsets
`1..16`. All other training settings remain the same. It has the same parameter
count; its flow expert uses 24,146,816 MACs per NFE and the static estimator
reports 1,894,489,088 FLOPs per replan.

## Launch commands

Sparse O45:

```bash
uv run scripts/launch_modal.py --gpu L40S --cpu 16 --cpu-limit 16 \
  --memory-gib 64 --memory-limit-gib 64 --disk-gib 512 \
  --timeout-hours 8 --app-name hal-045-legacy-flow -- \
  uv run experiments/045_legacy_sparse_endpoint_flow.py
```

Dense P16-D4:

```bash
uv run scripts/launch_modal.py --gpu L40S --cpu 16 --cpu-limit 16 \
  --memory-gib 64 --memory-limit-gib 64 --disk-gib 512 \
  --timeout-hours 8 --app-name hal-045-dense16-rtc -- \
  uv run experiments/045_legacy_sparse_endpoint_flow.py --dense-16
```

## Production record

Status: complete on 2026-08-30. Both seed-0 checkpoints reached step 16,384,
all four 96-boot evaluations completed, and the requested L40S latency
measurements finished. The launch commit was
`eba01c3f695d7f3dbf0a4665e5fbecfbecc91a24`.

| Variant | W&B | R2 prefix | Final SHA-256 | Bytes | Median training throughput |
| --- | --- | --- | --- | ---: | ---: |
| Sparse-10 | [`awt7z5tp`](https://wandb.ai/ericyuegu/hal/runs/awt7z5tp) | `runs/045-legacy-endpoint-flow-sparse10-nfe4-p4-s4-d0-seed0/` | `b7a823d6913fc52cc437866e4333d6135363da3f6decc10fe24b6672dac2cc85` | 81,308,971 | 1,684 samples/s |
| Dense-16 | [`wmctbm9r`](https://wandb.ai/ericyuegu/hal/runs/wmctbm9r) | `runs/045-legacy-endpoint-flow-dense16-nfe4-p16-s12-d4-seed0/` | `a3bad194470ac6d66d1d7654284e8daf7ab3a1463578c7c5d2e688121ee4d270` | 81,309,035 | 1,475 samples/s |

The sparse final validation flow objective was 0.2482 nats, with 0.5872 exact
frame accuracy and 0.4219 four-frame sequence accuracy. The dense-16 values
were 0.2398, 0.5365, and 0.3359. Training gradients stayed finite. Sparse was
preempted once at step 600 before its first checkpoint; Modal restarted it from
step zero under the final W&B run above. The discarded partial W&B run is
`jxbb25xl`. Dense had no interruption.

### Closed-loop results

| Mode | `(P,S,D)` | Boots | Matches | Crashes | Net stock/min (LCB) | Net damage/min (LCB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| H4-D0 | `(4,4,0)` | 96 | 100 | 0 | -0.3871 (-0.5209) | -20.6324 (-29.2138) |
| H4-D1 | `(5,4,1)` | 96 | 131 | 0 | -1.0185 (-1.3201) | -36.7102 (-46.9447) |
| H1-D1 | `(2,1,1)` | 96 | 127 | 0 | -1.0391 (-1.2961) | -38.8339 (-49.5566) |
| H12-D4 dense-16 | `(16,12,4)` | 96 | 125 | 0 | -1.3162 (-1.4701) | -88.9820 (-97.8458) |

H4-D0 is weaker than O38: net stock/min changed from -0.2811 to -0.3871,
and net damage/min changed from -8.1861 to -20.6324. Both RTC modes are weaker
again. The dense P16-D4 result does not support the longer committed plan.

### Latency

These are batch-one compiled BF16 measurements from the production L40S. Each
replan uses four NFEs.

| Mode | p50 | p95 | p99 | Amortized p50/frame | FLOPs/replan |
| --- | ---: | ---: | ---: | ---: | ---: |
| H4-D0 | 11.812 ms | 12.233 ms | 12.480 ms | 2.953 ms | 1.824 GF |
| H4-D1 | 12.565 ms | 14.503 ms | 15.813 ms | 3.141 ms | 1.824 GF |
| H1-D1 | 12.449 ms | 12.792 ms | 12.955 ms | 12.449 ms | 1.824 GF |
| H12-D4 dense-16 | 12.463 ms | 13.372 ms | 14.174 ms | 1.039 ms | 1.894 GF |

### Evidence audit

A read-only audit streamed both `final.pt` objects from R2, recomputed their
SHA-256 hashes, loaded every final `metrics.json`, and compared every metric
with the corresponding W&B summary field. H4-D0 also matched its mirrored
ordinary `eval/*` namespace. All comparisons passed. The four protocols name
the verified checkpoint, codec version 2, vocabularies `[6,37,9,5]`, exact
`P`, `S`, and `D`, and the expected hard categorical prefix clamp.

| Evidence | JSON SHA-256 | Protocol SHA-256 |
| --- | --- | --- |
| `replays/final_h4_d0/metrics.json` | `125566d7486fabfdbee7a696dd4eb69aa6494b5d7b4dd97026c4d28d970eb5fb` | `4d44f6d4df8ce43a4a456eaf1a6b8598a5e546ae4d9156e233b49cf73ce77746` |
| `replays/final_h4_d1/metrics.json` | `561a0a3844d7866e29d46803f051d75204547bcbab7f74587d5e68092c8d3519` | `2d148209af479c6d4c9b6ce6b1acd1e6f6af71b8b94d4c96aea20c2b49a3e5c3` |
| `replays/final_h1_d1/metrics.json` | `b06da189e0b066f9cbad72f8651a7083b1d6a14f81133a43936b58bda00253b5` | `d88a2fe20598269e9a47ed14c1889ed11849a46f15152ff0b0e8b5dddfeda441` |
| `replays/final_h12_d4/metrics.json` | `89deb4205c28241aa09b5bae1d43ceeda5c69810363fb87a55c12cb6ed3219cb` | `9269ba16d78a8a7f889c90b2f4fd31defa0f43084af9888e77a14a97ec11c65b` |

The sparse prefix contains 380 objects and 899,085,399 bytes. The dense prefix
contains 132 objects and 404,634,874 bytes.

### Launch trace

- Sparse: Modal app `ap-Vv6iwQ9uxC2gCUIrU33Kz8`, function call
  `fc-01M18RW62XWD5MGNKRS5SP38E3`, launch
  `9e93132eba014e6fbc27d2bf63684085`.
- Dense-16: Modal app `ap-jO7YMLPdaS8K2iJ1G95sB0`, function call
  `fc-01M18RWTP06WHAY2SX6YBZNFSS`, launch
  `e269f8585c2343e4a70df375290f2527`.

Before launch, the O45, O38, and O43 suites passed with 24, 17, and 19 tests.
Ruff, `py_compile`, `git diff --check`, pre-commit formatting/type hooks, and
CPU latency smoke benchmarks for all four modes also passed.
