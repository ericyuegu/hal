# Prompt: Design experiment A

You are an ML experiment designer. Work in `/home/ericgu/src/hal`.

## Objective

Design one controlled experiment that answers this question:

> Did the gain from experiment 026 come mainly from the d384 trunk, or from the temporal and controller-group decoder?

Produce an implementation-ready design. Do not change code. Do not start training. Do not create W&B runs.

## Source rules

Use source code, local evaluation artifacts, and W&B records as the source of truth.

Inspect these files first:

- `experiments/023_mtp_heads.py`
- `experiments/024_temporal_mtp.py`
- `experiments/026_temporal_mtp.py`
- `hal/training/features.py`
- `hal/training/trunk.py`
- `docs/offline_training_retrospective.html`

Reconstruct each forward path from code. Do not use filenames or comments as proof.

Use the production 026 run as the reference:

- W&B run: `cqbbbg77`
- Trunk: d384, 8 layers, 6 heads
- Context: 128 frames
- Parameters: 15.053M
- Temporal decoder: d128, 2 layers, 4 heads, FF 256
- Offsets: 1, 2, 3, 4, 5, 6, 9, 12, 16, 20
- Controller order: C-stick, main stick, triggers, buttons
- Live horizon: 4 frames
- Execution horizon: 4 frames
- Replan interval: 4 frames
- Controller codec: buttons 256, main stick 65, C-stick 9, triggers 25
- Training batch in the production run: 512 sequences

The 024-to-026 comparison is not a valid control. It changed width, context, features, codec, offsets, decoder design, and training exposure.

## Required experiment

Design a 2×2 factorial experiment.

Factor W is trunk width:

- W0: d256, 8 layers, 4 heads
- W1: d384, 8 layers, 6 heads

Factor D is action decoder structure:

- D0: an independent MTP decoder. Future offsets do not condition on prior predicted actions. Controller groups do not condition on prior predicted groups.
- D1: the exact 026 temporal and group decoder. Future offsets condition on the prior action. Groups use the 026 order.

The four cells are W0D0, W0D1, W1D0, and W1D1.

Use W1D1 as the 026 control. Make W0D0 a clean independent-head control. Do not copy the full 023 or 024 recipe.

Keep these items equal in all cells:

- input features;
- controller codec and semantic embeddings;
- context length;
- target offsets;
- primary and auxiliary loss weights;
- data split and replay order;
- batch size;
- optimizer rules;
- learning-rate schedule;
- number of processed frame positions;
- checkpoint schedule;
- action sampling temperature;
- live decode horizon;
- execution horizon;
- replan interval;
- evaluation seeds and matchup schedule.

The D0 decoder must have similar parameter capacity to D1. Target a difference of less than 5 percent within each width. The D0 decoder must still have no temporal or group conditioning. Explain the parameter-matching method. Do not hide unused parameters in the model.

Use common random numbers where possible. Use the same replay order for all cells. Use the same evaluation boots and action-sampling random tables.

## Budget

Start with 4,096 optimizer steps per cell. Save checkpoints at 1,024, 2,048, and 4,096 steps.

Estimate all of these values before implementation:

- trainable parameters;
- processed frame positions;
- proxy training FLOPs;
- expected wall time;
- peak memory;
- inference FLOPs;
- p50, p95, and p99 replan latency targets.

Use `C ≈ 6NT` as one training-compute estimate. Also report important work that this estimate omits.

Add a compute-matched comparison from intermediate checkpoints. The d384 cells use more FLOPs per step. Select checkpoints that make the width comparison approximately isoFLOP. Keep the equal-data comparison as the primary factorial result.

## Evaluation

Evaluate each final checkpoint with the exact 026 protocol:

- 96 prior-sampled character boots;
- level-9 CPU opponent;
- 7,200 frames per match;
- instant restart on a random legal stage;
- the same boot schedule for every cell;
- no debug or smoke runs in the final result.

Report these primary metrics:

- net stocks per minute;
- boot-clustered 95 percent confidence interval;
- `net_stock_lcb`;
- net damage per minute;
- boot-clustered damage confidence interval.

Also report stocks taken and lost, damage dealt and taken, crashes, completed matches, and runtime.

Use validation NLL only as a diagnostic. Report NLL by offset and by controller group. Report hold and transition metrics separately. Measure the rollout exposure gap. Do not select the winner from aggregate NLL.

Benchmark inference with the production batch shape. Report replan latency and amortized latency per executed frame.

## Analysis

Estimate these effects:

- the width main effect, averaged over D0 and D1;
- the decoder main effect, averaged over W0 and W1;
- the width-by-decoder interaction;
- equal-data effects;
- approximately isoFLOP effects.

Use paired boot differences because all cells use the same boot schedule. Give confidence intervals for each effect.

Before implementation, define a practical effect threshold. Base the threshold on the variance in existing 026-protocol evaluations. Explain the power of 96 boots. If 96 boots cannot decide the result, specify the smallest extension.

Use these decision rules:

- If D has the clear main effect, retain the 026 decoder at scale.
- If W has the clear main effect, favor trunk scale and keep the simpler decoder.
- If the interaction is large, scale only the combined W1D1 design. Do not claim an independent cause.
- If neither factor explains the gain, stop the scale decision. Identify the next confound to test.
- If D1 improves quality but breaks the latency limit, report the quality-latency frontier. Do not call D1 the deployment winner.

## Required deliverable

Write one design document. Include:

1. A source-based description of D0 and D1.
2. The exact four-cell configuration table.
3. A list of fixed variables and changed variables.
4. Parameter, FLOP, memory, wall-time, and latency estimates.
5. The checkpoint and evaluation schedule.
6. The statistical analysis plan.
7. The pre-registered decision rule.
8. The minimum code changes that implementation will require.
9. Tests that prevent train/inference leakage or target misalignment.
10. Risks and stop conditions.

End with one short answer to this question:

> If the result matches each possible outcome, what changes in the 10× scale recipe?

Do not expand the first test beyond the four factorial cells. Put later ablations in a separate follow-up section.
