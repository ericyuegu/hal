# E5: primary-only AWR

Status: blocked on the policy architecture decision after E4

Updated: 2026-08-07

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

Use the existing audited reward units:

- `+1` when the opponent loses a stock.
- `-1` when ego loses a stock.
- `+0.01` per point of damage dealt and `-0.01` per point taken.
- `+0.5` or `-0.5` on the stock event that decides the game.
- `gamma=0.99827`.

Compute rewards and returns from the complete replay before window sampling. Then slice them with
the same start, padding, ego relabeling, and valid-position mask as the action targets. Never compute
a return only inside a sampled window.

Audit how many replays end in a terminal stock state. If a replay is truncated, do not silently
treat its missing tail as a terminal zero. Record the fraction and predeclare its mask or exclusion
before training.

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

Use `stop_gradient(h_t)` as the value-head input in the first E5 arm. This lets the critic learn but
prevents value regression from becoming an extra policy representation loss. The actor weight is
also detached. The policy cannot reduce its loss by changing `V`.

Use `lambda_V=1.0`. Log value loss separately from the policy objective. Do not call this critic
`Q`; it does not read an action.

## Actor weight

Define:

\[
A_t=G_{t+1}-V(s_t),
\qquad
\tilde w_t=\min\left(5,\exp(A_t/0.8)\right).
\]

Compute the exponential in FP32 as `exp(clamp(A / beta, max=log(weight_max)))`. Reject nonfinite
advantages before exponentiation.

Normalize valid weights to mean one within the effective batch:

\[
w_t=\frac{\tilde w_t}{\operatorname{mean}(\tilde w)}.
\]

This normalization is an optimizer control, not part of the probability factorization. It preserves
relative example weights while keeping the actor gradient scale close to BC. Without it, a change in
mean weight also changes the effective learning rate. Log raw and normalized weights so a later
unnormalized sensitivity test remains possible.

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

Keep transition loss weight 1.0. Do not apply rank, trajectory, chunk, or future-head advantage
weights.

## Required tests

1. Use a synthetic replay with stock and damage events. Check every reward index and discounted
   return by hand.
2. Assert the action at offset 1 pairs with `G_{t+1}`, not `G_t` or `G_{t+2}`.
3. Assert returns are computed over the full replay before windowing.
4. Assert padding and invalid targets receive no value, actor, or auxiliary loss.
5. With AWR off, reproduce the selected BC objective exactly and leave the value head gradient-free.
6. With zero advantages, reproduce BC primary loss and mean-one weights exactly.
7. Assert only offset-1 losses receive AWR weights. Perturb weights and show auxiliary losses do not
   change.
8. Assert the same weight multiplies buttons, main stick, C-stick, and triggers for one frame.
9. Assert advantages and weights are detached from both actor and value parameters.
10. Assert value loss trains the value head but not the policy trunk when detach is enabled.
11. Check clipping before exponentiation, mean normalization, effective sample size, and all finite
    edge cases.
12. Reject NaN or infinite return, value, advantage, raw weight, normalized weight, loss, gradient,
    and parameter values with a useful error.
13. Check save and reload of reward, AWR, critic, and detach settings.
14. Assert the value head is in AdamW exactly once and never in Muon.
15. Run the small end-to-end train test with AWR on and off.

Run focused tests, Ruff, type checking, Python compilation, and `git diff --check` before the GPU
gate.

## Preflight audit

Before the full run, scan a fixed replay sample and record:

- Terminal and truncated replay fractions.
- Return and advantage quantiles.
- Value MSE, bias, correlation, and calibration by return bin.
- Raw and normalized weight quantiles.
- Effective sample size and fraction at the clip.
- Weight statistics by character, stage, rank tier, action transition, and reward proximity.
- Loader wait and decoded bytes added by return annotation.

Use `beta=0.8` and `weight_max=5` for the declared first arm. Do not tune them from closed-loop
results. Stop if normalized effective sample size is below 0.2 or more than 20% of raw weights hit
the clip; record a new plan before changing the dose.

## Fixed configuration and evaluation

Copy the selected reference exactly. Change only AWR, value-head, run-label, and H2H-reference
fields. Keep 16,384 steps, seed 0, 131,072 frames per step, attention, action architecture, offsets,
auxiliary scale, compact data, replay mixing, optimizer, schedule, decode, and evaluation fixed.

Run the standard periodic and final CPU protocol. Run 64 mirrored H2H configurations against the
matched BC reference. Save checkpoints, match rows, replays, H2H records, and the return audit.

Report stocks, damage, dead frames, terminal results, crashes, paired CPU deltas, H2H stock
difference, non-tied win rate, confidence intervals, and ties. Also report value and weight metrics
through training.

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

Pending the post-E4 architecture decision.
