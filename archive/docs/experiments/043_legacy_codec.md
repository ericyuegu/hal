# O43: legacy-codec forensic ladder

O43 keeps the current 15,053,039-parameter model architecture. Codec version 2
reproduces the controller classes at HAL commit
`2aa752d042a824ab607bbe48546cd61d9ffc4861`:

- buttons: A, B, Jump, Z, digital L/R, None;
- main stick: 37 classes;
- C-stick: 9 classes;
- fused analog shoulder: 0, 0.35, 0.6, 0.85, 1;
- the historical early-release reducer.

X and Y map to Jump. L and R map to one digital-shoulder class. Sampling that
class emits digital L, as the old postprocessor did. The analog shoulder is a
separate class and also emits on L.

The reducer compares each raw button set with the preceding raw set. It emits
the lowest-index newly pressed button. If buttons were released and no new
button was pressed, it emits None. An unchanged set copies the preceding
reduced output. Thus:

```text
raw:     B  B  B+A  B+A  B
encoded: B  B  A    A    None
```

The previous O43 implementation was not this codec. It had five button classes,
three shoulder classes, omitted digital L/R, folded a digital click into analog
pressure, and encoded the last frame above as B. Checkpoints written without
`codec_version=2` are rejected because their logits have different meanings.

## Current treatment defaults

O43 now evaluates with execution horizon 1. The training objective assigns 50%
of its total action-loss weight to offset 1. The other nine offsets share the
remaining 50%. The objective retains the former total scale of 2, so this
change does not halve the gradient scale.

Periodic evaluation remains at H1. Final evaluation runs both H1 and H4, with
96 matchups in each of the `char_matchup` and `fox` suites.

Set `--cfg.next-frame-loss-share None` to restore the former O26 objective:
the mean of offsets 1-4 plus the mean of offsets 5-20. Set
`--cfg.exec-horizon 4` to restore H4 evaluation. These switches exist for the
ladder below.

## Architecture ablation matrix

The completed default O43 run is Arm A. Three new runs separately test the
architecture changes; no arm combines them.

| Arm | Only change from A | Comparison |
|---|---|---|
| A | None; reuse the existing O43 run | Control |
| B | Zero-initialize the existing FiLM projections | B - A: FiLM initialization |
| C | Replace nonlinear action heads with O42's normalized, bias-free linear heads | C - A: linear heads |
| D | Remove the trunk-logit skip | D - A: skip removal |

Arm B keeps O43's FiLM equation exactly:

```text
state * (1 + tanh(scale)) + shift
```

It does not add O42's bounded scale or shift. Arm C retains the FiLM behavior
and trunk skip. Arm D retains the nonlinear action heads and original FiLM
initialization.

Launch only the three treatment runs:

```bash
uv run experiments/043_legacy_codec.py \
  --cfg.ablation-arm B \
  --comment abl-b-zero-init-film

uv run experiments/043_legacy_codec.py \
  --cfg.ablation-arm C \
  --comment abl-c-linear-head

uv run experiments/043_legacy_codec.py \
  --cfg.ablation-arm D \
  --comment abl-d-no-trunk-skip
```

Arm A is W&B run `1imfy8v3` and R2 run
`260828-203353_043_legacy_codec_mtp043-legacy-v2-d384-L8-h6-Lc128-t128x2-o1-2-3-4-5-6-9-12-16-20-s1-o1w50-base_ranked-anon-1_forensic-ranked-legacy-codec-h1-next50`.
Its existing checkpoint supplies H1. Backfill only its H4 final evaluation:

```bash
uv run experiments/043_legacy_codec.py \
  --eval final.pt \
  --eval-run 260828-203353_043_legacy_codec_mtp043-legacy-v2-d384-L8-h6-Lc128-t128x2-o1-2-3-4-5-6-9-12-16-20-s1-o1w50-base_ranked-anon-1_forensic-ranked-legacy-codec-h1-next50 \
  --eval-exec-horizon 4 \
  --eval-n-matchups 96 \
  --eval-backfill-wandb
```

The backfill writes H4 metrics with `_s4`; it does not replace A's H1 metrics.
Use the saved `match_rows.json` files for the B - A, C - A, and D - A
matched comparisons at each horizon.

## Cody membership finding

`hal/scripts/filter.py` did run for the present Cody MDS. The issue is that its
default `completed_only` and `stock_zero_only` settings are both false. Its
other quality rules therefore admitted games that the old materializer rejected
when neither player ended at zero stocks.

The checked build contains 64,049 selected replays and 550,366,204 frames.
Adding only the old stock-zero predicate retains 52,155 replays and 447,617,097
frames. It removes 11,894 replays and 102,749,107 frames, or 18.67% of the
current frames. The historical parser audit has 11,960 current-only replays, so
the missing completion predicate explains 99.4% of that membership excess.

The filter now exposes `--stock-zero-only`. O43 can apply its output as an
allowlist over the existing compact MDS. No replay rematerialization is needed,
and normalization statistics remain fixed. This makes the treatment a corpus
membership test rather than a combined membership-and-normalization test.

Create the allowlist:

```bash
uv run python -m hal.scripts.filter \
  --index data/builds/policy-world-20260816/professional/cody/filterable-index.jsonl \
  --output data/builds/policy-world-20260816/professional/cody/paths-stock-zero.txt \
  --stock-zero-only
```

## Stacked experiment progression

Use one initialization seed, sampler settings, update count, validation cache,
matchup list, and evaluation seed throughout. Record net stocks per minute at
4k, 8k, 12k, and 16k updates. Each row adds one change to the row above.

| Stage | Training configuration | Evaluation | What the delta measures |
|---|---|---|---|
| L0 | Existing pre-fix O43 Cody run | H4 | Buggy-codec control |
| L1 | Exact codec; former objective; current Cody corpus | H4 | Codec correction |
| L2 | Same L1 checkpoint | H1 | Execution delay only; no retraining |
| L3 | L1 plus 50% next-frame loss | H1 | Training-objective allocation |
| L4 | L3 plus the stock-zero allowlist | H1 | The extra 18.67% of Cody frames |

L1:

```bash
uv run experiments/043_legacy_codec.py \
  --comment forensic-l1-exact-codec-h4 \
  --cfg.data-root data/processed/professional/cody/mds-policy-world-v7 \
  --cfg.replay-format policy-world \
  --cfg.exec-horizon 4 \
  --cfg.next-frame-loss-share None
```

L2, evaluated from every L1 checkpoint of interest:

```bash
uv run experiments/043_legacy_codec.py \
  --eval runs/<l1-run>/<checkpoint>.pt \
  --eval-exec-horizon 1 \
  --eval-output-name forensic-l2-s1
```

L3:

```bash
uv run experiments/043_legacy_codec.py \
  --comment forensic-l3-next-half \
  --cfg.data-root data/processed/professional/cody/mds-policy-world-v7 \
  --cfg.replay-format policy-world
```

L4:

```bash
uv run experiments/043_legacy_codec.py \
  --comment forensic-l4-stock-zero \
  --cfg.data-root data/processed/professional/cody/mds-policy-world-v7 \
  --cfg.replay-format policy-world \
  --cfg.train-replay-paths data/builds/policy-world-20260816/professional/cody/paths-stock-zero.txt
```

L1 versus L0 is the codec result. L2 versus L1 is the delay result because both
rows use the same checkpoint. L3 versus L2 is the training-objective result.
L4 versus L3 is the corpus-membership result. Use the same fixed matchup seeds
for every checkpoint evaluation; otherwise evaluation variance can be larger
than the lift being measured.
