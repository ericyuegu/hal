# E6: chunk-conditioned critic validation

Status: blocked on E4 and E5 infrastructure

Updated: 2026-08-07

## Question

Can an offline critic distinguish the value of logged two-frame and four-frame action chunks from
the same state well enough to support chunk AWR?

E6 does not train or execute a new actor. It freezes the selected dense E4 policy and trains critic
probes. E7 cannot use chunk weights until E6 passes.

## Important limit

Logged data does not contain counterfactual chunks for the exact same state. A critic can fit logged
returns while ignoring its action input, or assign arbitrary values to policy chunks outside the
data support. Low held-out error alone does not prove a useful `Q(s, chunk)`.

E6 therefore needs action-ablation, perturbation, support, calibration, and seed-stability checks.
Even those checks do not create counterfactual ground truth. If the critic extrapolates strongly on
policy samples, stop. A world model or online data would be needed for a stronger claim.

## Definition

The policy state `s_t` ends at frame `t`. A dense chunk begins with `a_{t+1}`. For horizon `H`:

\[
\mathbf a_t^H=(a_{t+1},\ldots,a_{t+H}).
\]

Define the behavior-continuation value:

\[
Q_H^\mu(s_t,\mathbf a_t^H)
=\mathbb E\left[
\sum_{k=1}^{H}\gamma^{k-1}r_{t+k}
+\gamma^H V^\mu(s_{t+H})
\mid s_t,\mathbf a_t^H
\right].
\]

The continuation policy `mu` is the logged behavior distribution. This is not the value of
committing to an infinite open-loop policy.

Train and report `H=2` and `H=4`. Use separate output heads and targets. Do not average their
advantages.

## Files to change

- `experiments/023_mtp_heads.py`: add critic-only mode, frozen-policy loading, rolling future-state
  construction, Q/V modules, targets, diagnostics, checkpointing, and evaluation commands.
- `hal/training/returns.py`: reuse E5 reward and return labels.
- `hal/training/dataloader.py`: expose current and shifted rolling contexts without violating the
  fixed raw-window semantics.
- `tests/experiments/test_023_mtp_heads.py`: test chunk and reward alignment, future contexts,
  targets, frozen parameters, critic conditioning, and checkpoint loading.
- Add shared loader tests for shifted contexts.
- This file: record code, commands, source policy, run IDs, timings, calibration, perturbations,
  artifacts, and the gate decision.

Do not copy critic code from 020 or 022 without auditing its indices. Do not alter the E4 actor.

## State construction

Build three raw rolling contexts for each sampled point:

- `s_t`, ending before `a_{t+1}`.
- `s_{t+2}`, after the two logged actions.
- `s_{t+4}`, after the four logged actions.

Each context must contain only its newest `L_ctx` raw frames and must pass through every transformer
layer. Do not represent `s_{t+H}` by appending frames to the old transformer output. Do not reuse a
persistent temporal KV cache.

Freeze the E4 trunk and action policy. Run frozen state encoders without gradients. Confirm their
parameters and buffers remain byte-identical through critic training.

## Critic architecture

Quantize each chunk action into the same four canonical controller groups as the policy. Give the
critic its own action embeddings. Do not reuse trainable actor embeddings.

Use a 32-wide embedding for each action group. For each action frame, concatenate the four group
embeddings along the feature dimension and project the resulting 128 features to `d_q=128`. Add a
learned four-position embedding. Pass the `H` action tokens through two pre-norm causal transformer
blocks. Each block has four attention heads, head width 32, a 256-wide SiLU MLP, RMSNorm, and no
dropout. Apply a final RMSNorm and use the last action token as the chunk vector.

Concatenate that 128-wide chunk vector with `RMSNorm(h_t)`. Give each horizon its own
`Linear(d_model + 128, 512)`, SiLU, and `Linear(512, 1)` Q head. Share only the action encoder
between horizons.

Concatenation is important: the critic must receive state and action information in distinct feature
columns before the nonlinear MLP. Do not reduce the chunk to an additive sum of group embeddings.

Fit one state-only control for each horizon. Replace the chunk vector with a learned
horizon-specific 128-wide null vector, then use a head with the exact same shape as its Q head. The
control has no access to action classes. Record action-encoder, Q-head, state-control, and total
parameter counts separately. Verify that changing any valid action group can change the
corresponding Q output after training while leaving its state-only control unchanged.

Train a separate scalar `V(s_t)` probe on Monte Carlo `G_{t+1}`. The Q and V probes read the same
frozen state representation.

## Targets

Use a slowly updated target copy of V for the bootstrap:

\[
y_t^H=\sum_{k=1}^{H}\gamma^{k-1}r_{t+k}
+\gamma^H V_{target}(s_{t+H}).
\]

Fit Q with Huber loss and V with Huber loss to the Monte Carlo return. Detach all targets. Keep the
direct Monte Carlo return as an independent calibration label.

Use the E5 reward settings and `gamma=0.99827`. Apply one mask that requires valid `s_t`, all H
actions and rewards, and valid `s_{t+H}`. Never bootstrap across padding, a replay boundary, or a
terminal state. Set the bootstrap term to zero after a true terminal transition.

Use an EMA target update declared before launch. Record its coefficient. Do not tune it from final
critic results.

Use the same chunk-interruption rule as E7 execution. Exclude a training row when the controlled
player loses control, either player loses a stock, or the game enters a reset before all H planned
actions can execute. An event on the Hth transition is allowed because the full chunk has executed.
A true game terminal on that last transition keeps the rewards and sets the bootstrap to zero.
Record the excluded fraction by horizon and reason.

## Value warm-up

Do not train Q against a random target value. First run 2,048 value-only updates on the frozen E4
states and Monte Carlo returns. Do not update Q or its state-only controls during this phase. Require
finite held-out predictions and positive held-out return correlation. Then copy the warmed V weights
into `V_target` and begin the Q phase.

Continue updating V and its EMA target during Q training. Log the phase and global Q step
separately. A resumed run must not repeat or skip the target-network initialization. If the value
gate fails, stop before any Q update and revise the critic plan.

## Required tests

1. Use unique synthetic frame IDs to check `s_t`, actions `t+1:t+H`, rewards `t+1:t+H`, and
   `s_{t+H}` for both horizons.
2. Check the H-step discounted target by hand, including terminal and padded cases.
3. Assert each shifted context has exactly `L_ctx` raw frames and drops the correct oldest frames.
4. Assert every future context is fully recomputed and cannot use temporal KV.
5. Assert no actor parameter or buffer changes during critic optimization.
6. Assert Q receives all action frames and all four groups through concatenated feature columns.
7. Change one action group at one time. Confirm the critic input changes at the intended token only.
   Confirm the matched state-only output does not change.
8. Shuffle temporal order while preserving the action multiset. Confirm the encoded chunk changes.
9. Assert Q2 cannot read actions 3 or 4.
10. Assert Q and V targets are detached and finite.
11. Assert terminal transitions remove the bootstrap term.
    Assert a control interruption before H masks the whole Q row, while an interruption on the Hth
    transition keeps the row.
12. Assert critic parameters appear in their optimizer exactly once and actor parameters do not.
13. Save and reload critic geometry, horizons, reward settings, source-policy identity, and EMA
    state.
14. Reject source checkpoints that are not dense `(1,2,3,4)` temporal policies.
15. Run a small end-to-end critic job with finite train and validation outputs.
16. Assert each state-only ablation receives the exact target and valid mask used by its matched Q
    head.
17. Build policy-sample support thresholds from logged validation chunks without reading Q values.
18. Assert Q and both state-only controls remain byte-identical during the 2,048-step V warm-up.
19. Assert `V_target` is an exact copy of warmed V at Q step 0, then follows the declared EMA rule.
20. Check resume immediately before and after the phase boundary.
21. Assert the E6 interruption mask is byte-identical to the mask E7 uses for the same logged rows.

Run focused tests, Ruff, type checking, Python compilation, and `git diff --check` before launch.

## Training protocol

Use replay-level train and validation splits. Run 2,048 V-only warm-up steps, followed by 16,384 Q
optimizer steps for the main probes and matched state-only controls. Keep the same compact replay
mixing unless a measured systems limit requires a documented change. The actor remains frozen, so
this is not a policy data-budget comparison.

Train at least three critic seeds before passing the gate. One run may contain several bootstrap
heads, but report both within-run ensemble disagreement and across-seed stability.

Use a 256-step GPU gate first. Measure the cost of rebuilding `s_t`, `s_{t+2}`, and `s_{t+4}`. If
the projected run exceeds 3.5 hours, report it before launch. A larger critic batch is allowed only
after measuring memory and preserving replay diversity.

## Held-out checks

Report for Q2, Q4, and V:

- Huber loss, MSE, bias, correlation, and calibration by target-return bin.
- Error by character, stage, stock state, rank tier, and chunk behavior probability.
- Direct Monte Carlo calibration in addition to bootstrapped-target fit.
- Three-seed mean, interval, and rank correlation.

Then test whether Q uses actions:

- Replace the logged chunk with a chunk from another held-out state.
- Shuffle frame order within the logged chunk.
- Replace one frame and one action group at a time with an in-support alternative.
- Zero or mask all action embeddings.
- Compare Q error with a state-only predictor trained on the same `y_t^H` target.

Fit one state-only `Q_H^{state}(s_t)` ablation for each horizon. Match its target, optimizer, data,
training steps, and output-head capacity to the chunk critic. The V probe trained on Monte Carlo
returns is not this control because it has a different target. The chunk critic must improve
held-out prediction enough to exceed seed noise. Sensitivity alone is not enough; arbitrary
sensitivity can also be wrong.

Finally sample chunks from the frozen E4 policy. At each held-out state, compare the sampled chunk's
teacher-forced log likelihood with the logged chunk's likelihood under the same frozen policy.
For each horizon, divide chunk log likelihood by `4H` so it is measured per action group. Define the
support threshold as the fifth percentile of this value over held-out logged chunks. Compute and
save that threshold before joining the likelihood rows with Q outputs. Report relative likelihood,
Q value, V value, advantage, and ensemble disagreement. A high likelihood under the sampling policy
alone does not prove data support. Separate in-support and low-support policy samples. Do not use
their Q values to justify open-loop execution.

For each horizon, define critic disagreement as the population standard deviation of Q across the
three critic seeds. Set its threshold to the 95th percentile on held-out logged chunks. Save this
threshold before evaluating policy-sampled chunks. Behavior likelihood and critic disagreement
answer different questions. Low policy likelihood does not make an observed logged chunk
counterfactual. E7 may use a logged chunk's advantage when critic disagreement stays inside the
declared held-out range.

## Gate

E6 passes only if:

- For both horizons, every seed has lower held-out MSE than its matched state-only control, and a
  replay-level paired bootstrap has a 95% interval strictly below zero for
  `MSE(Q) - MSE(state-only)`. Resample replays, keep all rows and all three seed predictions for a
  selected replay together, then average across seeds. Do not treat three predictions of the same
  row as independent samples.
- Every seed has positive Pearson and Spearman correlation with its held-out target. Each pair of
  critic seeds has Spearman correlation of at least 0.8 on held-out logged chunks. For each seed,
  the linear calibration slope is between 0.5 and 1.5 and absolute mean bias is at most 10% of the
  held-out target standard deviation.
- Replacing a logged chunk with another held-out chunk increases MSE, with a replay-level paired
  95% bootstrap interval strictly above zero. Temporal shuffling must do the same for Q4. Report Q2
  temporal shuffling, but do not gate on it because a two-frame swap can be close to symmetric.
- No more than 1% of in-support single-group perturbations produce a Q value outside the held-out
  target range expanded by one target interquartile range on each side.
- At most 10% of in-support policy chunks exceed the logged-chunk 95th-percentile disagreement
  threshold.
- Advantages and proposed weights are finite. Both frame-level and replay-level normalized ESS
  ratios are at least 0.2, and no more than 20% of raw proposed weights hit their cap.

If Q ignores actions, fails calibration, or extrapolates on policy samples, do not train chunk AWR.
Record E6 as a negative result. Closed-loop execution without a trusted chunk critic may still be a
separate behavior-cloning systems test, but it is not E7.

## Results

Pending E4 and E5 infrastructure.
