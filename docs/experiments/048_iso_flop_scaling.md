# O48: deduplicated iso-FLOP scaling

O48 replaces O39/O47 as the capacity-scaling protocol. It keeps O43's legacy
controller codec, sparse ten-offset objective, 50% next-frame weight, base
observation bundle, and H=1 execution. The d384 member is exactly O43's
d384/L8/t128x2 architecture.

## Data contract

The training corpus is the union of all six ranked-anonymous policy-world-v7
streams and all 38 professional-player policy-world-v7 streams. The source
manifests contain 1,300,640 train rows and 13,292,693,140 frames before
deduplication.

The corpus index applies these rules once:

1. Identify the same game across archives by its Slippi file SHA-1.
2. Keep the first occurrence in the frozen 44-source order.
3. Remove a replay if its compact path-derived ID is not globally unique.
4. Assign every physical MDS shard a stable BLAKE2 priority.

An endpoint selects the smallest prefix of that shard ordering whose canonical
episodes contain at least `D` loss-bearing positions. Therefore larger data
tiers contain every episode in smaller tiers. The overshoot is at most one
physical shard. Training reads derived local MDS indexes that contain only the
selected shards, then filters noncanonical rows inside those shards.

`D` counts the two possible ego perspectives for every usable replay frame:
`2 * (frames - 1)`. It is the available unique-data budget. The processed
position count is `updates * batch * context`. O48 requires the former to be at
least the latter; it does not silently cap or repeat a smaller corpus.

## Model and compute grid

One width dial derives the rest of the capacity family:

| d_model | Trunk layers | Temporal width | Total parameters | FLOP-equivalent parameters |
|---:|---:|---:|---:|---:|
| 128 | 3 | 64 | 0.907M | 2.903M |
| 192 | 4 | 64 | 2.136M | 4.374M |
| 256 | 5 | 96 | 4.511M | 8.454M |
| 320 | 7 | 96 | 9.231M | 13.435M |
| 384 | 8 | 128 | 15.053M | 21.460M |
| 448 | 9 | 160 | 22.894M | 31.984M |
| 512 | 11 | 160 | 35.876M | 45.263M |

The five main lines have five model sizes each:

| FLOP line | Widths | Processed-position range |
|---:|---|---:|
| 1.000e16 | 128, 192, 256, 320, 384 | 0.078B–0.574B |
| 3.162e16 | 128, 192, 256, 320, 384 | 0.246B–1.815B |
| 1.000e17 | 192, 256, 320, 384, 448 | 0.521B–3.811B |
| 3.162e17 | 192, 256, 320, 384, 448 | 1.648B–12.051B |
| 1.000e18 | 256, 320, 384, 448, 512 | 3.682B–19.714B |

The one YOLO endpoint is `c3e18-d448-yolo`: 22.894M literal parameters,
31.984M FLOP-equivalent parameters, and 16.479B processed positions. We use
3.162e18 rather than 1e19 because the latter would require 52.1B positions at
this predicted model size, about twice the raw corpus ceiling. A 1e19 run would
therefore test corpus exhaustion or force a much larger model, not cleanly
extend the compute-optimal line.

The complete matrix is 26 runs and approximately 1.045e19 training FLOPs.

## Optimizer contract

- The statistical batch is fixed at 512 contexts, or 65,536 positions. Holding
  it fixed prevents batch efficiency from becoming a hidden axis of the
  capacity experiment. Gradient accumulation can change for memory only.
- Muon remains at LR 0.02. Its matrix update already has aspect-ratio scaling,
  so width does not require an LR retune. The auxiliary AdamW route remains at
  LR 0.00085.
- Both decoupled weight-decay rates use the Power Lines timescale rule, anchored
  to O43's `D`, `N`, and weight decay. The exponent is 0.52. This makes the
  regularization horizon explicit as `D/N` changes instead of applying one
  per-update decay over very different update counts.
- Each run warms up for 3% of its updates and then holds both LRs constant.
  There is no cooldown or other LR decay.

The Muon width transfer follows the optimizer's own parameterization guidance.
The decay rule follows [Power Lines](https://arxiv.org/abs/2505.13738). Its
application to this non-language domain remains a declared prior, not a hidden
claim that the LLM fit is exact here.

## Evaluation and hardware

There is no periodic validation or gameplay evaluation. The final checkpoint
gets the fixed validation cache and the standard 96-match H=1 `char_matchup`
and `fox` suites. O48 checkpoints also admit H=2, H=4, and H=6 decoding for the
later latency Pareto study; those results do not select the NLL scaling fit.

Modal currently charges per second. The supported request names are
`RTX-PRO-6000`, `H100`, and `B200`; current rates are on the
[Modal pricing page](https://modal.com/pricing). Hardware is selected from a
short measured training benchmark, not peak-FLOP specifications. The intended
routing is RTX PRO 6000 for small endpoints and B200 where its measured
dollars/FLOP or wall time is better.

## Commands

Inspect all exact integers without cloud access:

```bash
uv run experiments/048_iso_flop_scaling.py --describe
```

Build and upload the compact canonical corpus index once:

```bash
uv run experiments/048_iso_flop_scaling.py --build-corpus-index
```

Train one point:

```bash
uv run experiments/048_iso_flop_scaling.py --endpoint c1e16-d256
```

Fit terminal `val/loss` against log literal parameter count independently on
each FLOP line. An edge minimum requires one adjacent follow-up model before the
line is accepted. Use terminal H=1 closed-loop results as a second outcome, not
as the capacity selector.
