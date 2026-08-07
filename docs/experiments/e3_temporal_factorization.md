# E3: sparse temporal factorization

Status: blocked on E2 evidence

Updated: 2026-08-07

## Question

Does a joint model of sparse future actions improve representation learning and primary closed-loop
control over independent future marginals?

E3 keeps offsets `(1,5,9,13)` and executes only offset 1. It factorizes the sparse future sequence:

\[
p(a_1,a_5,a_9,a_{13}\mid s)
=p(a_1\mid s)p(a_5\mid s,a_1)p(a_9\mid s,a_1,a_5)
\,p(a_{13}\mid s,a_1,a_5,a_9).
\]

Each `a_o` is itself the selected E2 within-frame conditional distribution over controller groups.
This is a joint distribution over four sparse actions. It is not yet an executable four-frame chunk.

## Design choice

Use a shared residual MLP for temporal conditioning in the first E3 arm. Do not add a transformer
block yet. Four sparse predictions do not justify a larger decoder before the conditioning idea
passes. A tiny transformer is a later capacity ablation if the MLP result is promising.

Adopt the important DeepSeek-style MTP rule: each later module receives the true previous target
action during training. At free-running evaluation, it receives its own previous sampled action.

Do not share the action classifiers across offsets in the first arm. Weight sharing is not required
for a valid joint factorization and would add another experimental axis. Keep the selected E2 heads
and change only their temporal input state.

## Files to change

- `experiments/023_mtp_heads.py`: add temporal head mode, full-action embeddings, teacher-forced
  sparse rollout, free-running sparse rollout, and exposure-gap metrics.
- `tests/experiments/test_023_mtp_heads.py`: test target shifts, causality, initialization, gradients,
  free-running samples, masks, losses, checkpoint loading, and optimizer ownership.
- This file: record the final implementation, commands, run IDs, timing, metrics, artifacts, and
  decision.

Do not edit historical experiment files or shared data code.

## Temporal module

Let `h` be the transformer state for the current raw context. The offset-1 head reads `h` directly.
It is the deployed policy and remains structurally identical to E2.

For each later depth `k`, embed the complete previous action by concatenating the four selected E2
group embeddings and projecting them to one action vector. Then compute:

\[
u_k=W_h\operatorname{RMSNorm}(z_{k-1})+W_aE(a_{o_{k-1}})+e_k,
\]

\[
z_k=h+W_2\operatorname{SiLU}(u_k).
\]

The head for offset `o_k` reads `z_k`. Share `W_h`, `W_a`, `W_2`, and normalization across the three
later depths. Use a learned depth embedding `e_k` so the shared module knows whether it predicts
offset 5, 9, or 13.

Use hidden width `2 * d_model`. Initialize `W_2`, its bias, and `W_a` to zero. Create all E2 model
parts before the temporal module. At initialization, every `z_k` equals `h`, so same-seed E2 and E3
logits must match exactly at every offset.

The offset-1 path must not call the temporal module. This avoids giving the deployed head an extra
state-only capacity path. E3 can affect it only through the shared trunk gradients from better or
worse auxiliary modeling.

All temporal-module, action-embedding, and head parameters use AdamW, never Muon.

## Teacher forcing and free running

For training:

- Offset 5 receives the true offset-1 action.
- Offset 9 receives the true offset-5 action and the hidden state produced at depth 5.
- Offset 13 receives the true offset-9 action and the hidden state produced at depth 9.

Do not feed the current target action to its own predictor. Do not use an argmax prediction in the
teacher-forced path. Apply the same valid-frame mask to the previous action, temporal state, and
current loss.

For free-running validation, sample the complete within-frame action at each depth, then feed that
sample to the next depth. Freeze the sampling seed and number of examples. Report the gap from
teacher forcing.

Closed-loop execution samples offset 1 only. It must not compute offsets 5, 9, or 13 unless an
explicit diagnostic requests them.

## Objective

Keep:

\[
L=L_1+\frac{L_5+L_9+L_{13}}{3}.
\]

Keep AWR, rank weighting, value loss, and critic training off. Do not change reduction, target
alignment, optimizer, or total auxiliary weight.

## Required tests

1. Same-seed E2 and E3 have exact shared parameters and exact initial logits at all offsets.
2. The offset-1 forward path does not invoke or depend on the temporal module.
3. Every temporal `W_2`, `W_2` bias, and `W_a` starts at zero.
4. The first backward pass trains `W_2`; later passes train `W_a` and action embeddings.
5. Changing target offset 1 can change offset-5 logits after learning, but cannot change offset-1
   logits in the same forward.
6. Changing target offset 5 can change offsets 9 and 13, but not offsets 1 or 5.
7. Changing target offset 9 can change offset 13 only.
8. Teacher forcing uses the exact quantized target classes at the previous sparse offset.
9. Free running uses the sampled complete previous action and the declared within-frame group order.
10. Padding and truncated targets cannot enter an embedding lookup or a loss.
11. Teacher-forced and free-running losses use the same target and valid-position alignment.
12. Auxiliary losses still form one fixed mean across three heads.
13. Closed-loop offset-1 samples match E2 exactly at initialization for the same decode seed.
14. Save and reload the temporal mode, dimensions, group order, logits, and samples.
15. Every new parameter appears in AdamW exactly once and never in Muon.
16. Run the small end-to-end training test and confirm finite metrics and `final.pt`.

Run focused tests, Ruff, type checking, Python compilation, and `git diff --check` before launch.

## Fixed configuration

Copy the selected E2-S configuration. Change only:

- Temporal mode from independent to sparse autoregressive MLP.
- Temporal MLP ratio and depth embeddings.
- Run labels and H2H reference.

Keep the selected attention package, within-frame group order, offsets, loss weights, 16,384 steps,
seed 0, 131,072 frames per step, compact data, replay mixing, optimizer, schedule, decode, and CPU
protocol fixed.

Report exact parameter counts and warm step time. If total parameters rise by more than 5% or warm
training time rises by more than 10% from E2, add a temporal capacity control before claiming that
conditioning caused the result.

## Evaluation

Log teacher-forced and free-running NLL and accuracy for every temporal offset and action group.
Also log exact-action accuracy, exposure gaps, transition metrics, temporal-module norms and
gradients, shared-trunk gradient interaction, parameter counts, memory, and throughput.

Run the standard periodic and final CPU protocol. Run 64 mirrored H2H configurations against E2-S.
Save rows and replays. Report stocks, damage, dead frames, terminal results, paired deltas, H2H stock
difference, non-tied win rate, confidence intervals, ties, and crashes.

Target 3.0 to 3.5 hours through evaluation and upload on an RTX 4090. Flag startup over 30 minutes,
warm steps over 0.5 seconds, a slowdown over 25% from E2, a periodic evaluation over 25 minutes, or
a projected total over 3.5 hours.

## Decision

- Continue if free-running sparse predictions are coherent and closed-loop play does not regress.
- A small teacher-forced gain with a large free-running loss is evidence of exposure bias.
- Better auxiliary NLL without a primary or closed-loop gain does not promote E3.
- If E3 regresses, E4 may still use its correct temporal code as a chunk-readiness control, but do
  not call E3 a policy improvement.

Dense offsets and multi-frame execution belong to E4 and E7. Do not execute sparse predictions as
if they were contiguous frames.

## Results

Pending E2 completion.
