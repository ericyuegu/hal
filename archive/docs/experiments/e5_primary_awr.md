# E5: primary-only AWR

Status: implemented as experiment 036 on the 026 architecture — see
[036_advantage_weighted_bc.md](036_advantage_weighted_bc.md). 036 keeps this document's reward
dose, gamma, actor weight, warm-up, and detached critic, and records four deliberate deviations:
the "primary" weight covers the deployed four-frame chunk (offsets 1..4), not offset 1 alone,
because 026 deploys a chunk per decision; the value head shares the policy optimizer's Adam
groups (no separate AdamW or separate clip); there is no in-run warm-up ESS stop gate (the
preflight audit and watch metrics replace it); and the weighting scope plus the head-offset set
are explicit arm axes.

Updated: 2026-08-19

## Question

Does advantage-weighted regression improve the deployed next-frame policy when future-action heads
remain ordinary behavior-cloning auxiliaries?

E5 applies one state-action advantage to the complete offset-1 controller action. It does not apply
that weight to offsets 5, 9, or 13, and it does not invent separate advantages for those actions.

## Reference

Start from the best validated E2-style policy. Use E3 only if temporal factorization earned its
continuation gate. Do not use the dense E4 offsets merely because E4 ran later; E4 is a chunk bridge,
not an automatic policy promotion.

The reference and E5 must have the same architecture, data, targets, and training budget. The only
policy-objective change is the detached AWR weight on offset 1. The value head must not backpropagate
into the shared policy trunk in the first arm.

## Files to change

- `experiments/023_mtp_heads.py`: add the value head, reward/return batch fields, primary-only AWR
  loss, finite weight computation, diagnostics, tags, and checkpoint support.
- `hal/training/dataloader.py`: add one optional replay-annotation seam if full-replay return labels
  cannot be attached without copying the loader.
- Add `hal/training/returns.py`: implement reward events, discounted returns, and compact replay
  annotation in one shared place.
- `tests/experiments/test_023_mtp_heads.py`: test action alignment, loss scope, gradient isolation,
  weights, masks, checkpoint loading, and end-to-end training.
- Add focused reward and loader tests under `tests/` if shared code changes.
- This file: record the implementation, exact reference, commands, run IDs, timing, metrics,
  artifacts, and decision.

Do not copy the old 020 or 022 experiment file. Reuse audited formulas through small shared helpers.
Do not add rank weights.

## Reward and return alignment

For state `s_t`, the deployed head predicts action `a_{t+1}`. Its return must start with reward
`r_{t+1}`, after that action can affect the game:

\[
G_{t+1}=r_{t+1}+\gamma r_{t+2}+\gamma^2r_{t+3}+\cdots.
\]

Use these declared reward units:

- `+1` when the opponent loses a stock.
- `-1` when ego loses a stock.
- `+0.01` per point of damage dealt and `-0.01` per point taken.
- `+0.5` or `-0.5` on the stock event that decides the game.
- `gamma=0.99827`.

The old 020 tests audit stock drops, percent resets, terminal-stock detection, full-replay returns,
and the one-frame target shift. They do not validate this combined dose: 020 defaulted to stock-only
reward, no terminal bonus, and `gamma=0.999`. Treat E5's settings as a new fixed treatment. Here,
100 percentage points have the same immediate magnitude as one stock, the deciding stock is worth
1.5 stocks, and the discount half-life is about 400 frames, or 6.7 seconds. Audit the combined
return distribution before the value warm-up.

Compute rewards and returns from the complete replay before window sampling. Then slice them with
the same start, padding, ego relabeling, and valid-position mask as the action targets. Never compute
a return only inside a sampled window.

Audit how many replays end in a terminal stock state. A truncated replay has an unknown return
tail. Keep its policy rows in the same sampled data stream, use actor weight 1, and mask its value
loss. Do not treat the missing tail as zero and do not remove the replay from BC. Record eligible
and truncated replay and frame counts.

The compact policy artifact already stores the stock and percent columns needed for this reward.
Derive returns while reading. Do not create another large materialized dataset unless a measured
loader failure proves it is needed.

## Critic

Add a scalar `V(s_t)` head over the transformer state at every valid prefix. Create it after all
policy modules so the same-seed policy initialization does not change.

Fit it to the Monte Carlo return:

\[
L_V=\operatorname{mean}\left(V(s_t)-G_{t+1}\right)^2.
\]

Take this mean over eligible terminal-replay rows only.

Use `stop_gradient(h_t)` as the value-head input in the first E5 arm. This lets the critic learn but
prevents value regression from becoming an extra policy representation loss. The actor weight is
also detached. The policy cannot reduce its loss by changing `V`.

Use `lambda_V=1.0`. Log value loss separately from the policy objective. Do not call this critic
`Q`; it does not read an action.

Give the value head its own AdamW optimizer, scheduler, and gradient clip. Use the declared output
head AdamW hyperparameters and schedule, but do not add value parameters to the policy optimizer.
Clip policy and value gradients separately. Otherwise a large value gradient can scale down the
policy gradient through one global clip even though `h_t` is detached. The policy optimizer's
parameter list and state must remain byte-for-byte equal to the matched BC arm during warm-up.

## Critic warm-up

Do not use a random value estimate in an actor weight. For steps 0 through 2,047, train the value
head but set every actor weight to one. The actor therefore uses the matched BC objective during
this fixed warm-up. Start AWR at step 2,048 and keep training the value head so it can track the
changing detached policy representation.

Run the gate inside the 16,384-step candidate process after update 2,047 and before update 2,048.
Configure the optimizer and scheduler for all 16,384 steps from process start. Do not train a
shortened 2,048-step job and then restart or resume it. Require finite held-out predictions,
positive held-out return correlation, both normalized ESS ratios of at least 0.2, and no more than
20% of raw weights at the clip. If the gate passes, the same process performs its first weighted
actor update at step 2,048. If it fails, upload the gate evidence and stop. Do not move the
activation step after looking at closed-loop results.

Log an explicit `awr/active` field. Save the warm-up step in the checkpoint. A resumed run must
activate weighting at the same global step.

## Actor weight

Define:

\[
A_t=G_{t+1}-V(s_t),
\qquad
\tilde w_t=\min\left(5,\exp(A_t/0.8)\right).
\]

Compute the clipped log weight in FP32:

\[
q_t=\min(A_t/\beta,\log w_{max}).
\]

Reject nonfinite advantages before this operation. Do not exponentiate all raw weights and divide by
their arithmetic mean; sufficiently negative FP32 values can all underflow to zero.

Normalize eligible weights to mean one within the effective batch:

\[
w_t=\frac{\tilde w_t}{\operatorname{mean}(\tilde w)}.
\]

Implement the same value stably over the `N` eligible rows:

\[
\log \bar w=\operatorname{logsumexp}(q)-\log N,
\qquad
w_t=\exp(q_t-\log \bar w).
\]

When `N > 0`, check that normalized eligible weights are finite with mean one. Assign weight 1 to
truncated rows. If a batch has no eligible row, use weight 1 for every actor row and skip its value
update. Raw weights may be materialized only for diagnostics; they are not the normalization
denominator. Because eligible and truncated rows each have mean weight one, the complete actor
batch also has mean weight one.

`weight_max=5` caps the raw pre-normalization weight. It does not cap the final normalized weight.
When the raw mean is below one, normalization can make the largest `w_t` exceed 5. Report the raw
clip fraction and the normalized maximum as separate values. Do not apply a second clip after
normalization; that would change the declared relative weights and their mean.

This normalization is an optimizer control, not part of the probability factorization. It preserves
relative example weights while keeping the actor gradient scale close to BC. Without it, a change in
mean weight also changes the effective learning rate. Log raw and normalized weights so a later
unnormalized sensitivity test remains possible.

The first arm requires `grad_accum_steps=1`. Normalize across every valid frame in that optimizer
batch, not once per replay window. If a later arm uses gradient accumulation, it must collect all
microbatch weights before normalization; separate microbatch means are a different objective.

Report ESS over eligible frames and eligible replay windows. For the window-level value, first
average raw weights within each eligible window, then compute ESS across those window means. Do not
let unit-weight truncated rows inflate either gate statistic. Frames from one replay are correlated,
so frame ESS alone is too optimistic. Require both normalized ESS ratios to be at least 0.2 at the
warm-up gate.

Use one `w_t` for the sum of all four conditional group losses at offset 1. Do not compute one
advantage per controller group.

## Objective

Use:

\[
L=\operatorname{mean}(w_t L_{1,t})
+\frac{L_5+L_9+L_{13}}{3}
+\lambda_VL_V.
\]

The auxiliary losses remain unweighted BC. Their fixed mean keeps total auxiliary scale independent
of the number of heads. This normalization is also an experimental control; it is not an AWR rule.

Before step 2,048, replace `w_t` with one in this objective. Do not skip value training during that
period.

Keep transition loss weight 1.0. Do not apply rank, trajectory, chunk, or future-head advantage
weights.

## Required tests

1. Use a synthetic replay with stock and damage events. Check every reward index and discounted
   return by hand.
   Swap ego and opponent ports and assert that every reward and return changes sign.
2. Assert the action at offset 1 pairs with `G_{t+1}`, not `G_t` or `G_{t+2}`.
3. Assert returns are computed over the full replay before windowing.
   With the same replay and sampler seed, assert the BC and return-labeled paths produce
   byte-identical replay IDs, window starts, context padding, model features, action targets, and
   batch order. The return label must be the only added tensor.
4. Assert padding and invalid targets receive no value, actor, or auxiliary loss.
   Assert truncated rows keep unit actor weight, receive ordinary BC, and receive no value loss.
   Assert an all-truncated batch is finite and does not update the value head.
5. With the whole AWR package off, reproduce the selected BC objective exactly and leave the value
   head gradient-free.
6. During critic warm-up, reproduce the BC actor objective exactly while the value head receives a
   finite, nonzero gradient.
   After one update, assert exact policy-parameter and policy-optimizer-state equality with a
   same-seed BC model on the same batch.
7. With zero advantages, reproduce BC primary loss and mean-one weights exactly.
8. Assert only offset-1 losses receive AWR weights. Perturb weights and show auxiliary losses do not
   change.
9. Assert the same weight multiplies buttons, main stick, C-stick, and triggers for one frame.
10. Assert advantages and weights are detached from both actor and value parameters.
11. Assert value loss trains the value head but not the policy trunk when detach is enabled.
12. Check clipping before exponentiation, mean normalization, effective sample size, and all finite
    edge cases.
    Check both frame-level and window-level ESS by hand.
    Include an all-large-negative-advantage case and prove it returns finite mean-one weights rather
    than `0/0`. Include a case where a normalized weight exceeds the raw cap and prove no second
    clip is applied.
13. Reject NaN or infinite return, value, advantage, raw weight, normalized weight, loss, gradient,
    and parameter values with a useful error.
14. Check that the gate runs after the step-2,047 update and before the step-2,048 batch. Check that
    the optimizer and scheduler were configured for 16,384 steps from process start. A passing gate
    must continue without rebuilding the loader, optimizer, scheduler, or model.
15. Check save and reload of reward, AWR, critic, warm-up, and detach settings.
16. Assert the value head is in its separate AdamW optimizer exactly once and never in the policy
    AdamW or Muon optimizer. Assert policy and value gradients are clipped separately.
17. Run the small end-to-end train test with AWR on and off.
18. Reject AWR with `grad_accum_steps != 1` in the first arm.

Run focused tests, Ruff, type checking, Python compilation, and `git diff --check` before the GPU
gate.

## Preflight audit

Before the full run, scan a fixed replay sample and record:

- Terminal and truncated replay fractions.
- Return and advantage quantiles.
- Value MSE, bias, correlation, and calibration by return bin.
- Raw and normalized weight quantiles.
- Frame-level and window-level effective sample size, plus fraction at the clip.
- Weight statistics by character, stage, rank tier, action transition, and reward proximity.
- Loader wait and decoded bytes added by return annotation.

Also hash the first 32 BC and return-labeled training batches after removing only the return tensor.
Require matching hashes before the value gate. This proves the warm-up comparison did not change
sampled policy data.

Use `beta=0.8` and `weight_max=5` for the declared first arm. Do not tune them from closed-loop
results. Stop if either normalized ESS ratio is below 0.2 or more than 20% of raw weights hit the
clip; record a new plan before changing the dose.

The gate is not a separate experiment run. Give it its own W&B step fields and R2 artifact inside
the E5 run. Upload the artifact before the pass or fail decision can terminate the process.

## Fixed configuration and evaluation

Copy the selected reference exactly. Change only AWR, value-head, run-label, and H2H-reference
fields. Keep 16,384 steps, seed 0, 131,072 frames per step, attention, action architecture, offsets,
auxiliary scale, compact data, replay mixing, optimizer, schedule, decode, and evaluation fixed.

Request 32 periodic matchups and 96 final matchups. Limit both sweeps to 32 concurrent Dolphin
boots. Run 64 mirrored H2H configurations against the matched BC reference. Save checkpoints,
match rows, replays, H2H records, and the return audit.

Report stocks, damage, dead frames, terminal results, crashes, CPU rate differences, H2H stock
difference, non-tied stock-lead rate, confidence intervals, and ties. Also report value and weight
metrics through training.

Use one RTX 4090 experiment at a time. Target 3.0 to 3.5 hours through evaluation and upload. Flag
startup over 30 minutes, warm steps over 0.5 seconds, a slowdown over 25% from the reference, a
periodic evaluation over 25 minutes, or a projected total over 3.5 hours.

## Decision

- Promote E5 only from closed-loop evidence against the matched BC policy.
- Better value calibration alone does not promote the actor.
- A gain with collapsed effective sample size is not a stable AWR result.
- A regression with a healthy critic and healthy weights is evidence against this AWR dose.
- If critic calibration fails, fix or reject the critic before changing the actor temperature.

E6 is a different problem: it validates `Q(s, action chunk)`. E5's state value cannot score or
justify a macro-action.

## Results

Experiment 036 completed this test on the 026-style architecture. The production
W&B run is `hwzv0k9a`. Use the
[audited blog experiment table](../blog_experiment_evidence.md) for the official
run, metric, lineage, and mismatch record.
