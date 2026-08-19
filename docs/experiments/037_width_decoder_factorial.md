# Experiment 037: width × decoder factorial

Status: preregistered; implementation and integrity gates are in progress. No
cloud run is eligible until every gate in this document passes.

## Question and hypotheses

Experiment 026 changed both the observation-trunk width and the action decoder.
Experiment 037 isolates those changes in a 2 × 2 factorial:

- **W0** is a d256, 8-layer, 4-head causal trunk.
- **W1** is the exact d384, 8-layer, 6-head 026 trunk.
- **D0** predicts every selected future offset and controller group without a
  learned edge from another future action.
- **D1** is the exact 026 temporal and controller-group autoregressive decoder.

The primary null hypotheses are zero width main effect, zero decoder main
effect, and zero width-by-decoder interaction in net stocks per active minute.
The directional alternatives are positive width and decoder effects. A positive
interaction means that the two changes are complementary rather than separable.

"Exact 026" below means exact architecture, parameter initialization, forward
path, loss, optimizer partition, and sampling path under 037's shared budget. It
does **not** claim to reproduce historical 026's 16,384-step cosine schedule or
score. W1D1 uses a new 4,096-step cosine horizon like every other cell;
comparisons with the historical +0.159 stock/min result are descriptive only.

## Source reconstruction

The descriptions below come from the executed forward paths, not the filenames.

`experiments/023_mtp_heads.py` maps each trunk state to one output per selected
offset. Its linear and state-MLP modes share no action information across
offsets or groups. Its factored mode is not an independent control because it
teacher-forces earlier groups into later groups.

`experiments/024_temporal_mtp.py` shifts complete target controller frames into
a causal future-action transformer. It also cross-attends to the causal trunk
memory and modulates later groups with earlier target groups. Its context,
offset set, codec, and decoder geometry differ from 026, so it is not a control.

`experiments/026_temporal_mtp.py` first quantizes the observed controller history
and all future targets through one structured codec. The base feature bundle,
semantic controller embeddings, and d384 trunk produce one hidden state per
context prefix. For D1, the temporal token for selected offset `k` contains the
trunk state, offset embedding, and complete controller action from the preceding
selected offset (the observed action for offset 1). Two causal d128 attention
blocks propagate this history. Within a frame, classifiers run in C-stick,
main-stick, triggers, buttons order; FiLM-style scale and shift projections give
later groups the earlier target groups in training and earlier samples in
inference. A full-width linear trunk skip is added to every group classifier.
Buttons are masked against the trigger/button legality table.

The observation path in `hal/training/features.py` supplies exactly the base
projection used by 026. `hal/training/trunk.py` supplies the full-causal rotary
pre-norm transformer. Experiment 037 changes neither path except for W.

## D0 architecture and capacity match

D0 retains the 026 structured codec, offset set, d128 decoder state, nonlinear
group classifiers, and full-width trunk skips. For each trunk state and offset,
it concatenates only normalized trunk state and the learned offset embedding,
then applies one affine projection and four pointwise residual SwiGLU-free
feed-forward blocks (d128 → d264 → d128). The same function is vectorized
over offsets. There is no attention along the offset axis, no prior-action input,
and no within-frame group modulation. Sharing the pointwise function is parameter
sharing, not statistical conditioning: for hidden state `h` and offset `o`, every
raw logit is

`z[o,g] = head[g](F(h, e[o])) + trunk_skip[g](norm(h))`.

Thus `z[o,g]` is invariant to observed actions, target actions, and samples at
all other offsets and groups. The only exception is the codec-required hard
support operation. Training masks button classes using the target trigger after
raw logits are computed. Inference samples all non-button groups from their raw
logits, then masks buttons using that frame's sampled trigger. This operation has
no learned parameters and cannot alter another raw logit.

The decoder-capacity denominator is every trainable parameter reachable below
`model.temporal`, including the shared codec reference, input/offset mapping,
dynamics blocks, group heads, and trunk skips. Expected exact counts are:

| Width | D0 decoder | D1 decoder | Relative difference | D0 whole model | D1 whole model |
|---|---:|---:|---:|---:|---:|
| W0 | 624,707 | 650,051 | -3.90% | 7,056,367 | 7,081,711 |
| W1 | 686,531 | 711,875 | -3.56% | 15,027,695 | 15,053,039 |

The implementation test must recompute these values. D0 is invalid if the
absolute decoder difference reaches 5%, if any trainable parameter is outside
the optimizer, or if any D0 parameter is unreachable from the loss.

## Cells and fixed configuration

| Cell | Trunk | Decoder | Fresh-run comment |
|---|---|---|---|
| W0D0 | d256, L8, h4 | independent pointwise offset/group logits | `037-w0d0` |
| W0D1 | d256, L8, h4 | exact 026 temporal/group path | `037-w0d1` |
| W1D0 | d384, L8, h6 | independent pointwise offset/group logits | `037-w1d0` |
| W1D1 | d384, L8, h6 | exact 026 temporal/group path | `037-w1d1` |

All cells use seed 0, ranked-anonymized-1 policy-v7 in identical reservoir order,
base features, context 128, batch 512, no gradient accumulation, target offsets
1/2/3/4/5/6/9/12/16/20, primary dense-four plus weight-1 auxiliary loss,
Muon 0.02, AdamW 8.5e-4, weight decay 0.01, warmup 500, the same 4,096-step
cosine schedule, BF16, and the same validation cache. They use the 026 semantic
codec and class vocabularies (buttons 256, main stick 65, C-stick 9, triggers
25), temperature 1, a four-frame live decode, four-frame execution, and replan
after four frames. Evaluation seed, matchup schedule, slot/group action-random
tables, frame budget, and bootstrap tables are fixed and shared.

`max_steps=4096` defines the scheduler horizon, not only an early stop on the
historical schedule. The common post-scheduler-step learning rates are:

| Update | Muon LR | AdamW LR |
|---:|---:|---:|
| 1,024 | 0.0189824505 | 0.0008067541 |
| 2,048 | 0.0122589210 | 0.0005210041 |
| 4,096 | 0.0002352941 | 0.0000100000 |

The only changed variables are W and D. D0's pointwise depth/width is a
preregistered capacity-matching consequence of D; it is not tuned per cell.

## Budget and estimates

Each cell processes `4096 × 512 × 128 = 268,435,456` context-frame
positions (2,097,152 sampled sequences). Using `C ≈ 6NT` gives:

| Cell | Parameters | Proxy train compute | Inference proxy/replan | Proxy/executed frame |
|---|---:|---:|---:|---:|
| W0D0 | 7.056M | 11.36 PFLOP | 1.81 GFLOP | 0.452 GFLOP |
| W0D1 | 7.082M | 11.41 PFLOP | 1.81 GFLOP | 0.453 GFLOP |
| W1D0 | 15.028M | 24.20 PFLOP | 3.85 GFLOP | 0.962 GFLOP |
| W1D1 | 15.053M | 24.25 PFLOP | 3.85 GFLOP | 0.963 GFLOP |

`6NT` omits the explicit ten-offset decoder work, quadratic trunk and temporal
attention geometry, optimizer updates, data movement, compilation, validation,
checkpointing, and emulator evaluation. Actual profiler FLOPs supersede this
proxy without changing the equal-data estimand.

On one L40S, expected training time is 0.3–0.6 hours for W0 and 0.45–0.8 hours
for W1, plus approximately 0.3–0.5 hours for the 96-boot final evaluation and
the frozen 32-boot horizon-6 diagnostic. Fresh production evidence from 034,
which has W1D1-equivalent batch geometry, measured 18.124 GiB trainer peak
allocation, 20.430 GiB peak reservation, and 20.969 GiB device allocation.
Therefore the prelaunch W1 estimate is about 21 GiB observed with a 28 GiB
planning margin. Scaling the activation-dominated portion by width gives a
provisional 11–15 GiB W0 range. The 45 GiB hard gate remains deliberately
conservative. Replace all wall, FLOP, and memory estimates with measured values
in the result table while retaining these entries explicitly as prelaunch
estimates.

Production-shape latency is measured at 32 rows after two warmups. Targets are:

- W1D1: p50 ≤ 15 ms, p95 ≤ 18 ms, p99 ≤ 22 ms per four-frame replan;
- W1D0: no slower than W1D1 at p95;
- W0 cells: p95 ≤ 12 ms;
- all cells: p99 < 25 ms and no compile graph break in the decode program.

These are gates, not estimates of quality. Report both replan latency and latency
divided by four executed frames.

## Checkpoints, evaluation, and diagnostics

Persist named checkpoints at 1,024, 2,048, and 4,096 optimizer updates, plus
`latest.pt` for inspection and upload only; no 037 checkpoint is resumable.
Validate at the same three boundaries. For every
checkpoint report NLL by offset and group, hold/transition accuracy, transition
precision/recall/F1, target and rollout change rates, exact-frame and dense-four
sequence accuracy, and teacher-forced versus rollout exposure gaps.

Every final checkpoint receives the exact 026 evaluation: 96 prior-sampled
character boots against level-9 CPU, 7,200 frames per boot, instant restart on a
random legal stage, and the identical boot schedule. Request and limit 32 CPU
cores so all workers can start. Persist protocol and match rows. Report net stock
and net damage per active minute with boot-clustered 95% intervals and lower
bounds; also report component stock/damage rates and intervals, crashes,
completed matches, frames, and wall time. No smoke or debug row enters the final
result.

The final-update periodic 32-boot H4 evaluation is intentionally suppressed.
Update 4,096 receives one 96-boot H4 final evaluation and the separate frozen
32-boot H6 diagnostic. Their rows live in distinct directories and are never
mixed. Intermediate periodic evaluation remains disabled because the first
`eval_every=4096` boundary is the final update.

## Estimands and paired analysis

Let `Ywd` be a cell's boot-pooled metric, with W and D coded 0/1. Apply the same
bootstrap resample indices to all four boot schedules. For every resample compute:

- width main effect: `((Y10 - Y00) + (Y11 - Y01)) / 2`;
- decoder main effect: `((Y01 - Y00) + (Y11 - Y10)) / 2`;
- interaction: `(Y11 - Y10) - (Y01 - Y00)`.

Report point estimates and percentile 95% intervals for net stocks and net
damage. The equal-data 4,096-step effects are primary. The same paired reduction
is secondary for components and diagnostic checkpoints.

The approximate isoFLOP comparisons use W1@1,024 versus W0@2,048 and W1@2,048
versus W0@4,096. Their proxy-compute mismatch is about 6%; report it rather than
interpolating policy outcomes. Decoder effects compare equal steps within width.

The practical threshold is 0.10 net stocks per active minute. The production
026 result is +0.159 with a boot-cluster fifth percentile of +0.058, implying a
rough marginal standard error near 0.061. With 96 boots, an unpaired normal
approximation has an 80%-power minimum detectable difference near 0.24; pairing
at boot correlation 0.5 or 0.75 lowers this to roughly 0.17 or 0.12. Therefore
96 boots can establish a large effect but may leave the 0.10 threshold
indeterminate. If a primary interval crosses both 0 and ±0.10, add one smallest
extension of 96 new boots per cell using a preregistered continuation of the
schedule, then recompute once at 192 boots. Do not repeatedly inspect extensions.

## Integrity and launch gates

Before cloud launch, all of these conditions must pass:

1. At the same seed and shared configuration, W1D1 has exactly the same state
   keys, parameters, logits, loss, gradients, categorical samples, and optimizer
   partition as 026.
2. Perturbing D0 observed actions, target actions, or sampled earlier actions
   does not change any raw logit. The only allowed post-logit difference is the
   trigger/button legality mask.
3. Every D0 parameter belongs to the optimizer and receives a finite gradient on
   a coverage batch. Decoder capacity differs from D1 by less than 5% at W0 and
   W1.
4. All cells have the same loader/replay sampling-contract hash and the same
   actual first two train batches (replay IDs, padding, and first target frame)
   and exact cached validation model inputs (sorted named feature tensors,
   padding, and complete targets) in the same order. If validation replay IDs
   are present they are included, but the compact validation loader does not
   expose them. W0/W1 trunk initial states match within a width across D.
5. Named checkpoints are exactly 1,024, 2,048, and 4,096. Reloading a named
   checkpoint preserves cell identity, scientific configuration, model state,
   and logits; reload is an integrity check, not permission to resume training.
6. Matchup schedule hashes and slot/group random tables are identical across
   cells. H4 and H6 each complete two local decode calls with zero crashes or
   invalid trigger/button combinations.
7. CUDA training and decode compilation have zero unexpected graph breaks.
   Peak allocation is below 45 GiB and latency gates pass.
8. A short same-seed smoke for all cells is finite. Report aggregate and every
   group/offset loss; do not select a cell from smoke NLL.

The binding production-data preflight passed for all four cells with sampling
contract `4888b73937a3fa31077b6fd3203b5c3b25f1d4070f59257a2e3bd79587fff499`,
actual first-two-train-batch hash
`d0fc497d151d8b5a09762bf0b78efb8e8d2597b3e8ea66d3011fe60e4765271c`,
and exact cached-validation-input hash
`6cb9fa4404452a5aeb1811e9fcf2087cdbf061ad10e8a05e1026b8ac348b0841`.

Each run writes and uploads `launch_manifest.json` before its first update. It
contains the complete config hash, source-file hash, actual first-two-train-batch
hash, exact cached-validation-input-order hash, action-random contract hash,
96-boot schedule hash, named checkpoint paths, and CPU parallelism. The external
launch record pins all four cells to one Git commit and records a distinct Modal
call and W&B attempt ID for each cell.

## Decision rule

Call a factor clear only when its paired 95% interval excludes zero and its
point estimate reaches 0.10 net stocks/min. If only D is clear, keep D1 in the
10× recipe and choose width by the quality/compute frontier. If only W is clear,
scale the trunk and use D0. If the interaction is clear and positive, scale only
the combined W1D1 design and make no independent-cause claim. If neither is
clear, stop the scale decision and next isolate training exposure/schedule, the
remaining known confound. If D1 clears the quality rule but violates latency,
report the D0/D1 quality-latency frontier; do not name D1 the deployment winner.

Stop a cell for non-finite loss/gradient, replay-contract mismatch, target
leakage, parameter mismatch, evaluator crash fraction above zero, or a persistent
throughput/memory failure. A stopped or recovered cell cannot be paired until it
has the same number of updates and the exact frozen evaluation schedule.

Checkpoints do not contain a verified loader/RNG cursor. An interrupted attempt
is discarded; Modal restarts that cell as a distinct fresh attempt at update 0.
`--resume` is prohibited. Never splice an interrupted prefix into the paired
factorial, and record all discarded attempt IDs.

## Minimum implementation changes

Add one experiment wrapper around 026, one independent decoder, a frozen
cell-to-config constructor, named-checkpoint preservation, and a pure factorial
analysis tool over four `match_rows.json` files. Do not modify 026, the shared
feature codec, trunk, data pipeline, evaluation protocol, or optimizer.

## Follow-ups outside this first test

Do not add a change/hold hurdle, rank conditioning, recurrent decoder, SSM,
alternative offset set, extra seed, or scale continuation to these four cells.
Those are separate experiments after this causal attribution is resolved.

## Consequence for the 10× recipe

D-only keeps the 026 decoder; W-only scales the trunk with D0; a positive
interaction scales W1D1 as one package; neither result pauses scaling for the
next confound; a latency failure selects from the measured quality-latency
frontier.
