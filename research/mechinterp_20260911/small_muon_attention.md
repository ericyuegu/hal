# Attention in earlier, smaller Muon models

Four earlier Muon models were tested at exactly 16,384 training updates.
All completed training with finite losses and stable late-training gradient
norms. Their attention is much less concentrated than in the large O50
Muon checkpoints from the previous report.

## Main result

For each replay, layer, and head, measure the attention distribution from
the most recent query token to all 128 input tokens. Take the largest
weight within that distribution, then average those largest weights.
The maximum can occur at a different key token position in each row.

| Model and run | Parameters | Heads per layer | Mean largest weight | Mean entropy, nats | Mean effective frames |
| --- | ---: | ---: | ---: | ---: | ---: |
| [O37 D1, a117chkw](https://wandb.ai/ericyuegu/hal/runs/a117chkw) | 7.1M | 4 | 26.79% | 2.482 | 18.20 |
| [O37 D3, 5wfk2esf](https://wandb.ai/ericyuegu/hal/runs/5wfk2esf) | 7.1M | 4 | 25.83% | 2.604 | 20.52 |
| [O26, cqbbbg77](https://wandb.ai/ericyuegu/hal/runs/cqbbbg77) | 15.1M | 6 | 30.51% | 2.387 | 17.38 |
| [O43, 1imfy8v3](https://wandb.ai/ericyuegu/hal/runs/1imfy8v3) | 15.1M | 6 | 36.30% | 2.113 | 13.53 |

Each model has eight transformer layers. These means cover 256 distinct
replays and all layers and heads: 8,192 attention distributions per 7.1M
model and 12,288 per 15.1M model.

The earlier large Muon checkpoints had mean largest weights of 94.97%
and 98.21%. Thus, the earlier statement about attention being concentrated
on almost one frame per head does **not** apply to these small Muon runs.
This is not evidence that size alone causes the difference: context length,
architecture, training data, objective, decoder, and training duration also
changed.

Some individual heads still have sharp attention on individual inputs.
The fractions of distributions that put at least 90% of their weight on
one token are 1.10%, 0.77%, 2.27%, and 4.31%, respectively. A maximum-position
marker alone would hide most of the distribution in these models. The
new figure therefore shows every attention weight, with each head's
maximum marked separately.

Different heads often have different maximum-weight positions, but some
share a position. In the displayed input and layer zero, O37 D1 heads 0
and 1 both have their maximum at token 123, four frames before the query.
Their weights at that token are 21.01% and 25.48%. Head 2 instead has its
maximum at token 101, with only 4.24% of its weight there. The rest is
spread over other positions.

## Position and entropy definitions

Each input token represents one observed frame. Key token position 0 is
the oldest frame; position 127 is the most recent frame. The query token
is always position 127. Frame age is `127 - key position`. These are
positions within one input, not averages of positions across replays.

The most recent token receives 2.3–2.8% of the attention weight on average
across these models. It is the maximum-weight position in only 2.5–3.2%
of their distributions. This does not say that nearby recent frames are
unimportant.

For weights `p`, entropy is `H = -sum(p * ln(p))`. A nat is the unit that
results from using the natural logarithm. Entropy is zero when all weight
goes to one frame. Equal weight on N frames gives entropy `ln(N)`.
The effective number of frames is `exp(H)`: the number of equally weighted
frames that would have the same entropy. It is not a count of frames with
nonzero weight. We calculate `exp(H)` for each distribution before taking
the mean; we do not exponentiate the mean entropy.

## Why these checkpoints

The four final checkpoints were fixed before measuring attention, using
their previous gameplay evaluations, not validation loss or attention.
O37 D1 was the strongest H1/H2 cell in the earlier factorization study;
O37 D3 was the strongest H4/H6 cell. O26 was the production baseline;
O43 was the legacy-controller-encoding baseline. Here H is the number of
frames executed between replans, not observation delay.

Their recorded net stocks per minute include D1 +0.133 at H1, D3 -0.021
at H6, O26 +0.159 at H4, and O43 +0.244 at H1 in its character-matchup
suite. These are different evaluation settings, not a fair ranking of
the four models. Each listed setting used 96 emulator starts. No new
gameplay evaluation or training was run for this attention comparison.

The training-history snapshots each contain all 16,384 distinct updates,
numbered 0 through 16,383. Each API scan also returned eight exactly
duplicated rows. We checked that the duplicate rows agree and counted each
update once for the table below. The raw snapshots are preserved.

| Run | Median loss, preceding 2,048 updates | Median loss, final 2,048 updates | Gradient norm range, final 2,048 updates |
| --- | ---: | ---: | ---: |
| O37 D1 | 5.209 | 5.176 | 0.400–0.636 |
| O37 D3 | 3.140 | 3.122 | 0.310–0.527 |
| O26 | 3.099 | 3.079 | 0.308–0.505 |
| O43 | 2.589 | 2.574 | 0.235–0.414 |

These are each run's logged training losses; their different objectives
make the absolute values unsuitable for comparison. The histories support
using these as completed, numerically stable baselines. They do not prove
that training reached the best possible solution. The optimizer setup is
Muon for its selected weight matrices and AdamW for the remaining groups.

## Inputs and measurement checks

We used the same 256 raw replay examples and final query frames as in the
previous large-model analysis. These older models require 128 context
frames, so each received the last 128 frames of the original 256-frame
context. Every original raw-window hash, data index, and manifest matched.
Inputs were rebuilt using the earlier ranked-data normalization statistics
and each checkpoint's controller encoding. No large-model-normalized
input tensors were reused.

The primary measurement matches the previous analysis: evaluation mode,
BF16 projections, FP32 attention scores and softmax, and disabled TF32.
It ran on the local RTX 3060 in batches of 16. The traced hidden state
matched the current dense-attention forward path exactly for all four
models. This check verifies the measurement path, not gameplay quality.

The interactive figure uses preselected replay example zero, not a sample
chosen after seeing the result. All four panels show the same query frame.
The layer selector starts at layer zero. Every row is a head; every column
is a key token position. Hover details give the weight of that cell and
identify the maximum-weight position for that head. A shared maximum does
not prove that two heads have the same function; different maxima do not
prove independent functions either.

These are attention weights, not measurements of causal influence on an
action. Earlier layers can mix information from several frames into one
key token. Residual connections provide another information path.

## Check against the older numerical code

The original O26 code used FP32 positional-encoding factors. O37 and O43
already used BF16 factors under mixed precision. The primary comparison
uses BF16 factors for all four, matching the earlier large-model analysis.

We also reran all four models with FP32 factors, keeping checkpoints,
inputs, batching, and the other numerical choices fixed. For O26, this
restores the historical choice; for O37 and O43 it is a sensitivity test.
The copied BF16 control matched both the primary probabilities and the
current dense hidden states exactly.

O26's mean largest weight changed from 30.5150% to 30.5176%. Across all
four models, the largest change in that mean was 0.01125 percentage points.
The broad-attention result is therefore not explained by this precision
difference.

Exact maximum positions are less stable: 3.47–5.04% of the attention
distributions changed their maximum-weight key position. The largest
individual weight change was 13.88 percentage points. Thus, stable model
averages do not mean that every individual head/input result is unchanged.
The [sensitivity results](artifacts/attention_small_muon_rope_sensitivity/sensitivity.json)
include both sets of full probability arrays and their checks.

## Reproduction and records

The [predeclared plan](small_muon_attention_plan.md) gives the selection and
measurement rules. The [measurement data](artifacts/attention_small_muon/attention.json)
records the full configuration, checkpoint hashes, source hashes, input
identity, precision, and environment. All 40,960 probability rows are saved
in `artifacts/attention_small_muon/last-query-attention.npz`.

The Git base is `52da5b2de1d402b86404fc4d24e382409649e525`; new research
sources and tests are archived with their hashes. The archive also contains
the three original experiment source files recovered from W&B. Each is
byte-identical to a recorded Git version. O43's W&B process metadata is
from a later evaluation backfill, not its original training invocation.

The only configuration differences between W&B and the checkpoint are the
O37 experiment identity field, which was absent in W&B, and a one-unit
floating-point serialization difference in O43's learning-rate floor.
Both were checked explicitly. No unsupported configuration fallback was used.

Primary result SHA-256:
`5a6768a41f8799d2ff283e6f863fc8b16840126716ac71c27a648a8bff2fbd65`.
Probability archive SHA-256:
`c384dbb42481665efc9bd209a2966f68b40daee5b2aed1f0096f468d42075741`.
Measurement source archive SHA-256:
`7e16e3d8425f6ec9f8d1b5254a0e9e9f26660b0976936af0170ac08fdf511d52`.

Reproduction commands and check results are recorded in
[the validation notes](small_muon_attention_validation.md).
