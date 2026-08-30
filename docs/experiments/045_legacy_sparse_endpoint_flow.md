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

Status: pending launch.
