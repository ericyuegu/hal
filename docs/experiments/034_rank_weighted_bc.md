# 034 rank-weighted behavioral cloning

Status: preregistered and implemented 2026-08-19; local RTX 3060 smoke passed; production run pending.

## Aim

Test whether the ranked ladder label contains useful demonstrator-quality signal that ordinary behavioral cloning
leaves unused. Experiment 034 changes only the contribution of each demonstrated prefix to the 026 loss. It does
not add a critic, infer per-action quality, filter replays, change sampling, or warm-start a model.

The eligible reference is 026 W&B run `cqbbbg77`, trained for 16,384 optimizer steps at effective batch 512. Its
final 96-boot evaluation reported:

- net stocks per active minute: `+0.159109693`;
- ordinary bootstrap lower confidence bound: `+0.031887949`;
- boot-clustered lower confidence bound: `+0.058350794`.

The preregistered promotion target is an ordinary `net_stock_lcb > 1.031887949`, a gain of more than `+1.0` over
026. The clustered interval remains a required robustness report, but it is not substituted for the named primary
metric after seeing the result.

## Scientific hypothesis

Rank is a noisy replay-level proxy for demonstrator quality. If Master actions are more useful targets on average
than Diamond actions, and Diamond actions are more useful than Platinum actions, then a soft mixture tilt should
improve closed-loop control without discarding the state and matchup coverage supplied by lower ranks.

For prefix `i`, let `r_i` be `1`, `2`, or `4` for Platinum, Diamond, or Master. With `n_i` valid context positions in
the sampled window, the optimizer-step normalization is

\[
Z = \sum_i n_i r_i.
\]

The primary and auxiliary objectives are

\[
L_{primary} = \frac{\sum_i r_i\sum_{t\in valid_i}\sum_{o=1}^{4}\sum_g
\operatorname{NLL}_{itog}}{4Z},
\]

\[
L_{aux} = \frac{\sum_i r_i\sum_{t\in valid_i}\sum_{o\in aux}\sum_g
\operatorname{NLL}_{itog}}{|aux|Z},
\qquad L=L_{primary}+L_{aux}.
\]

`Z` covers the complete effective optimizer batch, not each gradient-accumulation microbatch. Raw weights are
detached FP32 constants. Uniform weights must be numerically identical to 026.

This treatment is confidence-weighted maximum likelihood in the same broad family as Wu et al.'s
[demonstration-confidence weighting](https://proceedings.mlr.press/v97/wu19a.html), but rank is not calibrated
per-action confidence. Soft weighting is preferred to hard filtering because the latter would also remove lower-rank
state coverage; CRR likewise found soft exponential weighting more robust than hard action filtering in its offline
control setting ([Wang et al., 2020](https://proceedings.neurips.cc/paper/2020/hash/588cb956d6bbe67078f29f8de420a13d-Abstract.html)).

## Frozen setup

- Source examples: ranked-anonymized-1, in the same replay and window order as 026.
- Storage view and row order: the exact original `mds-policy-v7` stream used by 026.
- Rank join: deterministic replay-ID-to-port-rank sidecar derived from the canonical v7 manifest. Rank is attached
  only after replay order, window starts, and ego port have been sampled; it is never a model feature.
- Seed: 0.
- Effective batch: 512 examples per optimizer step.
- Steps: `2**14 = 16,384` from a fresh initialization.
- Model: exact 026 `d384-L8-h6-Lctx128` trunk and temporal decoder.
- Targets: offsets `(1, 2, 3, 4, 5, 6, 9, 12, 16, 20)` and the 026 controller-group order.
- Objective balance, optimizer, learning-rate schedule, reservoir sampling, validation cache, decode temperature, and
  H4 deployment: exact 026 production settings.
- Final evaluation: the fixed 96-boot 026 protocol, plus the fixed 32-boot H6 diagnostic.

The local 100-step RTX 3060 smoke may split the effective batch into smaller microbatches and reduce logging,
validation, and closed-loop evaluation work, but it must retain an effective batch of 512, one normalization over
all 512 examples, and every scientific field above. The implementation rejects unrelated smoke overrides.

## Required integrity gates

The generated sidecar contains 114,768 replays and has SHA-256
`9670b77ef378e762a2d157a3b37cbce35bd05ea54b84a9218a9f0d1269173e70`. Its source manifest SHA-256 is
`a563c62603b8cfdef219cd133324cb090f7e6488fcfc4b6941d03e863a255d16`. Across the two player slots it contains
98,248 Platinum, 59,218 Diamond, and 72,070 Master labels. At weights `(1, 2, 4)`, the corpus-level prefix-frequency
approximation gives ESS fraction `0.7465` and assigns about 57.1% of gradient mass to Master examples.
Local training reads the generated sidecar cache. A cloud worker does not require a new uploaded artifact: when the
cache is absent, it verifies the already-published canonical manifest against the frozen source hash and derives the
same replay-ID lookup in memory without writing external state.

The 2026-08-19 preflight scanned the replay ID stored in every compact policy row and found exact sidecar coverage:
112,409 train, 1,192 validation, and 1,167 test rows (114,768 total), with no duplicate, missing, or unexpected ID.
The ordered replay-ID SHA-256 digests are
`32284318494ed8032ad67634f534fa2f1a85b55b8f9d2edb4b2f040e55e4e5c0` (train),
`ecf3d0817225dc845d91cf1916c3dffda8d673e58d79a450d2d29e0fd66cd098` (validation), and
`c8589fd5617e910802e021c34b7dc7ddeff0e257286e402c95595bc76040db4b` (test). The sorted all-split ID-set digest
is `7fb115c26c8d6a42ff8a9166cdd612661219a34a20925fbd77206fbebce0f79a`.

Before launch:

1. The rank sidecar covers every replay ID sampled from the original policy stream, matches its frozen hash, and
   selects the sampled ego port's rank without changing any policy column, replay ID, window, or RNG draw.
2. The default 034 configuration differs from the eligible 026 recipe only in rank weights, explicit experiment
   identity/provenance, and rank diagnostics. An 026 checkpoint must be rejected rather than accepted as a 034
   resume or evaluation.
3. `(1, 1, 1)` weights reproduce 026's scalar objective and gradients.
4. Splitting one effective batch into microbatches does not change the weighted scalar objective or gradients beyond
   normal floating-point reduction tolerance.
5. Unknown, Professional, missing, non-finite, or non-positive rank weights fail before an optimizer update. The
   ranked-anonymized-1 treatment is defined only for Platinum, Diamond, and Master.
6. Weight normalization is finite and positive. The effective sample-size fraction
   `(sum w)^2 / (N * sum w^2)` is at least `0.60`.
7. Unweighted training NLL and all validation metrics remain unweighted and directly comparable with 026.
8. A 100-step local train and closed-loop evaluation complete without a crash, NaN, data-order violation, or
   inference regression.

Log, per optimizer step, the example fraction and normalized gradient-mass fraction for every tier, raw weight
minimum/mean/maximum, and effective sample-size fraction. Production throughput should remain in the established
`1,200-3,500` samples/s range; profile before accepting a material regression.

## Decision

- Promote only if the final ordinary 96-boot `net_stock_lcb` exceeds `1.031887949` and integrity gates remain valid.
- A positive point estimate below that bound is evidence for the idea but does not satisfy the objective.
- If the run is clearly divergent, violates an integrity gate, projects past eight hours, or is decisively poor at a
  scheduled closed-loop checkpoint, stop it and preserve the evidence.
- If 034 fails cleanly, do not tune rank weights against the same final CPU schedule. Move to an orthogonal model or
  objective hypothesis.

## Evidence and result

Local smoke command: 100 steps at effective batch 512, split into eight 64-example microbatches; production
compilation was retained, while validation and closed-loop work were reduced to 128 validation examples and two
1,800-frame boots at each execution horizon. Offline W&B ID: `kd9imqzq`. Run directory:
`260819-023636_034_rank_weighted_bc_mtp026-d384-L8-h6-Lc128-t128x2-o1-2-3-4-5-6-9-12-16-20-s4-base-rank1-2-4_ranked-anon-1_034-smoke-rtx3060`.

- Assertion: the exact 026 model and stream can train with whole-step `(1,2,4)` rank weighting and retain real-time
  H4/H6 inference on an RTX 3060.
- Setup: 15,053,039 parameters, seed 0, BF16, 100 fresh-initialization updates, 51,200 examples. Checkpoint SHA-256:
  `dc036fb1d7ef4cf79fbfb19f4b5b331550a30e3b99d58ba770da09404f7c1723`.
- Data/treatment: full-step ESS never fell below `0.68087` (mean `0.73599`, gate `0.60`); mean Master example share
  was `30.52%` and mean Master gradient-mass share was `56.13%`. Sidecar and full replay-ID audit matched the frozen
  hashes above.
- Signal: steady-state training averaged `530.6` samples/s (`485.9` including loader wait), with `2.486 GiB` peak
  allocated and `2.912 GiB` peak reserved. Unweighted train loss fell `42.588 -> 7.519` bits; weighted objective fell
  `42.655 -> 7.511` bits; final unweighted validation loss was `7.207` bits.
- Test/evaluation: H4 and H6 each completed `2/2` boots with zero crashes. H4 decode p95/p99 was
  `9.19/9.89 ms`; H6 was `11.75/13.27 ms`. The tiny post-100-step score estimates are intentionally non-decisional
  (`-3.22` and `-5.37` net stocks/min); this smoke tests execution and integrity, not the promotion hypothesis.

Production source commit, launch command, W&B run ID, L40S throughput, checkpoint hash, final metrics, uncertainty
intervals, and the promote/reject decision remain pending.

## Next orthogonal candidate: recursive chunk decoder

Do not combine this with 034. Experiment 026 already uses a shared two-layer causal temporal Transformer over future
frame tokens, rather than independent temporal heads. A candidate 035 would replace those temporal attention blocks
with a small recurrent register that repeatedly reads the frozen scene embedding, previous controller frame, and
offset embedding, then emits the next controller frame. The expensive observation trunk remains amortized across the
chunk. The first candidate should be a standard 128-wide GRU cell; modern scan-based RNN/SSM and matrix-memory cells
mainly target much longer sequences than this 10-offset plan and are deferred pending evidence that the simple
recurrence is insufficient.
