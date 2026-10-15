# O39 scaling and metric audit

This audit checks the persisted Experiment 039 result slab against the endpoint, match-join,
latency, and fit logic in `notebooks/039_scaling_viz.py`. It does not load `config_json`.

Source: `/home/ericgu/src/hal/results/039_capacity_scaling/experiment_039_raw_outcomes.csv`,
SHA-256 `c372b10718b367b423c95bf0122904c797b38386a4300ee0e30f8a41eba46e4b` (63,943,713
bytes). The slab has 62 run, 315 evaluation, 46,655 match, and 63 latency-benchmark rows.
Machine-readable outputs are in [`artifacts/scaling/`](artifacts/scaling/).

## Matching and invariants

| Population | Count | Rule or invariant |
| --- | ---: | --- |
| Exact completed cooldown endpoints | 41 | `endpoint_exact`, cooldown, finished, not excluded, processed equals target, NLL present |
| Healthy scaled fit endpoints | 35 | Scaled family, excluding the one divergence below |
| Explicitly excluded run records | 7 | Failed/killed attempts; retained in the source audit, not silently substituted |
| Complete evaluations | 310 | 96/96 boots and no crash, linked to selected endpoints |
| Evaluations with active matches | 308 | `match_active_frames > 0`; two complete evaluations have no active rows |
| Active match rows | 45,489 | Restricted to the 310 selected evaluation keys; rates recompute from these rows |
| Boot clusters per active evaluation | 96 | Boot index is nested within each evaluation; match rows are not treated as independent |

All selected endpoints share one episode hash, `unique_data_divisor = 1`, and
`unique_loss_positions = 2,409,583,026`. Evaluation timing fields agree exactly:
`delay_frames = eval_control_delay = eval_replan_interval` and
`eval_decode_horizon = 2 * delay - 1`. Complete active evaluations share one matchup schedule
hash and have 40 checkpoint hashes. The two no-match evaluations are retained in
`complete_evaluations.csv` but excluded from correlations.

The notebook divergence rule is a regression of more than 0.01 bits from the best earlier
checkpoint for that model. It marks only L10 run `9h6pnh39` (D = 1,493,689,000,
NLL = 1.851101, prior best = 1.795632). The row remains in endpoint and sensitivity outputs,
but not in the healthy fit or healthy correlations.

## What the NLL measures

O39 logs bits, not nats. `experiments/039_capacity_scaling.py` divides cross-entropy nats by
`ln(2)` before logging `primary_nll`, `auxiliary_nll`, and `loss`. The persisted objective is

```text
validation_nll = validation_primary_nll + 0.5 * validation_auxiliary_nll
```

The maximum residual of this identity is `4.44e-16` bits. The weighted auxiliary term is
34.31% of total NLL on average (range 33.96–34.58%). The components are almost interchangeable
in this dataset: Spearman correlations are 0.99775 (total/primary), 0.99681 (total/auxiliary),
and 0.99325 (primary/auxiliary).

Capacity is also not a raw parameter count. The notebook uses

```text
N_eff = training_flops / (6 D) = trunk_parameters + 36 * decoder_parameters
```

For example, L3 has 2.06M total parameters but 23.74M effective parameters; L18 has 293.13M
total but 503.69M effective. `N_eff` is a compute-model axis, not an independent causal
capacity variable.

## NLL versus gameplay

The gameplay response is recomputed net stocks per active minute. The table reports Spearman
rank correlation; intervals are 90% bootstrap intervals over run IDs for evaluation-level
comparisons. The pooled row is diagnostic only.

| Scope | Evaluations / runs | Total NLL | Primary | Auxiliary | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| All delays pooled | 307 / 39 | -0.490 | -0.495 | -0.481 | Delay-confounded; not a selection metric |
| Fixed d=1 | 39 / 39 | -0.239 | -0.237 | -0.228 | Weak; total 90% CI [-0.472, 0.035] |
| Fixed d=4 | 32 / 32 | -0.721 | -0.728 | -0.705 | Total 90% CI [-0.826, -0.536] |
| Fixed d=8 | 34 / 34 | -0.743 | -0.752 | -0.736 | Similar ordering |
| Fixed d=16 | 34 / 34 | -0.821 | -0.830 | -0.809 | Strongest fixed-delay ordering |

Within a model at fixed d=4, NLL and gameplay both improve with exposure: the seven models
with enough rows have total-NLL rho from -0.8 to -1.0 over only 3–5 endpoints each. This is
an exposure trend, not evidence that a lower NLL model wins at a fixed exposure.

At fixed d=4 and fixed exposure, the model ordering is not stable:

| Processed D | Models | Total-NLL rho |
| ---: | ---: | ---: |
| 2^26 | 7 | +0.143 |
| 2^27 | 7 | -0.536 |
| 2^28 | 5 | +1.000 |
| 2^29 | 7 | -0.393 |
| 2^30 | 4 | +0.400 |

The small per-exposure samples and missing active rows make these estimates noisy, but the
positive and near-zero cases are direct counterexamples to a uniform fixed-D capacity ranking.
Primary and auxiliary correlations show the same pattern. A d=4 regression with both NLL
components, log2 N_eff, and log2 D has condition number 34.3 and R² 0.710; its standardized
primary and auxiliary coefficients are -0.744 and +0.458. The sign split is a collinearity
warning, not a component effect.

## Scaling fit and the proposed compute optimum

The healthy scaled fit has 35 points and reproduces the notebook's robust law

```text
L = E + A N_eff^(-alpha) + B D^(-beta)
```

with Huber delta 0.001 and the notebook's Nelder–Mead starts. The fit uses processed D, not
distinct coverage.

| Quantity | Observed healthy fit | Fit at C = 4.2e19 |
| --- | ---: | ---: |
| N_eff | 2.374e7 – 5.037e8 | 1.335e9 |
| Processed D | 6.711e7 – 1.622e9 | 5.242e9 |
| Training FLOPs | 1.241e16 – 2.113e18 | 4.2e19 |
| Validation NLL (bits) | — | 1.702895 |

The target compute is 19.9x above the largest observed compute. The fitted point is 2.65x
above the observed N_eff range and 3.23x above the observed D range (about 2.18 corpus
epochs). Therefore the point is an extrapolated model optimum, not an observed optimum.

The fit itself is numerically stable enough to reproduce, but its allocation is not identified
by the data. A 200-resample model-curve bootstrap gives 90% intervals of
`N_eff = [4.36e8, 6.08e9]` and `D = [1.15e9, 1.61e10]`. Fit sensitivity is similarly broad:

| Variant | N_eff at target C | D at target C | Predicted NLL bits |
| --- | ---: | ---: | ---: |
| Full healthy scaled | 1.335e9 | 5.242e9 | 1.7029 |
| Leave out one model | 0.917–2.349e9 | 2.980–7.637e9 | 1.6973–1.7101 |
| Power-of-two D only | 1.413e9 | 4.953e9 | 1.7048 |
| Include 026 baseline | 1.292e9 | 5.419e9 | 1.7015 |
| Include divergent L10 endpoint | 1.713e9 | 4.087e9 | 1.7085 |
| Distinct-coverage axis | 2.041e9 | 3.429e9 | 1.7090 |

The distinct-coverage variant uses the notebook's with-replacement approximation
`U_seen = U * (1 - exp(-D/U))`, with U equal to the shared corpus size. It shifts the proposed
allocation materially. The negative result is that O39 supports a broad extrapolative family,
not a precise capacity prescription at 4.2e19 FLOPs.

## Deployment latency is a separate gate

`eval_valid_latency_bucket` is not used to remove scientific gameplay rows; all 310 selected
evaluation rows have value 0, and it is not a substitute for a measured latency frontier. The
separate RTX 3060 frontier artifact (`latency_capacity_frontier.csv`, SHA-256
`5b44fadb52bc511424378231c490ccbe45d0dd1831d4207ac7cf65bda76ef334`) reports, for example:

| Delay | Largest measured eligible model | p95 latency | Frame budget | Deadline misses |
| ---: | --- | ---: | ---: | ---: |
| 1 | L10 | 11.665 ms | 16.667 ms | 0 |
| 2 | L13 | 21.673 ms | 33.333 ms | 0 |
| 4 | L18 | 45.782 ms | 66.667 ms | 0 |

This gate can define a deployment frontier, but applying it to the offline gameplay/NLL
correlation would answer a different question.

## Testable hypotheses for follow-up

1. At fixed delay and fixed processed D, a new matched checkpoint pair will not show a stable
   monotone relation between NLL and net stocks across model depth; the sign should vary with D.
2. Replacing total NLL with either primary or auxiliary NLL will not improve held-out
   fixed-delay gameplay ranking after clustering by run, because the three metrics are already
   nearly collinear.
3. The processed-data and coverage-aware fits should be compared on held-out loss
   predictions, not by asking whether a chosen allocation lies inside an optimum interval.
   Preselect several model/data allocations at one feasible compute budget, record distinct
   positions and replay reuse, and compare their observed NLLs with both fits' predictions.
   If the two predicted loss surfaces are too close relative to run variability, that budget
   cannot distinguish them. Keep gameplay evaluation as a separate selection endpoint.
