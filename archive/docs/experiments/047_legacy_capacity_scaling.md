# O47: legacy-codec compute-optimal capacity brackets

O47 measures the compute-optimal model size for the complete O43 treatment. It
keeps codec v2, the ten sparse offsets, the 50% next-frame objective, the full
ranked-anonymous-1 corpus, batch 512, context 128, and seed 0 fixed.

The existing high-compute reference is W&B run `1imfy8v3`:

```text
260828-203353_043_legacy_codec_mtp043-legacy-v2-d384-L8-h6-Lc128-t128x2-
o1-2-3-4-5-6-9-12-16-20-s1-o1w50-base_ranked-anon-1_
forensic-ranked-legacy-codec-h1-next50
```

## Fixed matrix

| Endpoint | FLOPs | Trunk geometry | Total parameters | Effective parameters | Updates | Processed positions |
|---|---:|---|---:|---:|---:|---:|
| `c5e16-xs` | 5.000e16 | d192 L4 h3 | 2.506M | 8.078M | 15,740 | 1.032B |
| `c5e16-s` | 5.000e16 | d256 L5 h4 | 4.722M | 10.573M | 12,027 | 788M |
| `c5e16-m` | 5.000e16 | d320 L7 h5 | 9.445M | 15.574M | 8,165 | 535M |
| `cref-s` | 1.383e17 | d256 L5 h4 | 4.722M | 10.573M | 33,255 | 2.179B |
| `cref-m` | 1.383e17 | d320 L7 h5 | 9.445M | 15.574M | 22,576 | 1.480B |
| existing reference | 1.383e17 | d384 L8 h6 | 15.053M | 21.460M | 16,384 | 1.074B |

Effective parameters are the FLOP accounting quantity
`N_trunk + N_other + 10 * (N_temporal + N_group_heads)`. They match compute;
literal trainable parameters select and label model size.

Every endpoint processes less than the corpus's 2,409,583,026 loss positions.
No point enters the repeated-data regime. Warmup remains the reference's fixed
fraction of its run. Periodic gameplay is disabled. Each terminal checkpoint
gets the standard H1 96-boot `char_matchup` and `fox` evaluation. Offline
validation remains periodic so divergence is visible during training.

Inspect the exact integer quantities:

```bash
uv run experiments/047_legacy_capacity_scaling.py --describe
```

Launch each new endpoint with:

```bash
uv run experiments/047_legacy_capacity_scaling.py --endpoint c5e16-xs
uv run experiments/047_legacy_capacity_scaling.py --endpoint c5e16-s
uv run experiments/047_legacy_capacity_scaling.py --endpoint c5e16-m
uv run experiments/047_legacy_capacity_scaling.py --endpoint cref-s
uv run experiments/047_legacy_capacity_scaling.py --endpoint cref-m
```

Fit `val/loss` against log total parameters separately within each compute
budget. A valid minimum must be inside its three-model bracket. If an edge wins,
add only the adjacent model needed on that side. Use terminal closed-loop scores
as supporting evidence; select compute-optimal capacity by validation loss.
