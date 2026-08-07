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

For each action frame, concatenate the four group embeddings along the feature dimension. Project
that vector to `d_q`, add a temporal position embedding, and pass the `H` action tokens through two
small causal mixer blocks. Concatenate the pooled chunk vector with `RMSNorm(h_t)`, then use a
two-layer MLP to produce scalar `Q_H`.

Concatenation is important: the critic must receive state and action information in distinct feature
columns before the nonlinear MLP. Do not reduce the chunk to an additive sum of group embeddings.

Use separate final heads for `Q_2` and `Q_4`. A shared action encoder is allowed. Record exact
parameter counts and verify that changing any valid action group can change the corresponding Q
output after training.

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

## Required tests

1. Use unique synthetic frame IDs to check `s_t`, actions `t+1:t+H`, rewards `t+1:t+H`, and
   `s_{t+H}` for both horizons.
2. Check the H-step discounted target by hand, including terminal and padded cases.
3. Assert each shifted context has exactly `L_ctx` raw frames and drops the correct oldest frames.
4. Assert every future context is fully recomputed and cannot use temporal KV.
5. Assert no actor parameter or buffer changes during critic optimization.
6. Assert Q receives all action frames and all four groups through concatenated feature columns.
7. Change one action group at one time. Confirm the critic input changes at the intended token only.
8. Shuffle temporal order while preserving the action multiset. Confirm the encoded chunk changes.
9. Assert Q2 cannot read actions 3 or 4.
10. Assert Q and V targets are detached and finite.
11. Assert terminal transitions remove the bootstrap term.
12. Assert critic parameters appear in their optimizer exactly once and actor parameters do not.
13. Save and reload critic geometry, horizons, reward settings, source-policy identity, and EMA
    state.
14. Reject source checkpoints that are not dense `(1,2,3,4)` temporal policies.
15. Run a small end-to-end critic job with finite train and validation outputs.

Run focused tests, Ruff, type checking, Python compilation, and `git diff --check` before launch.

## Training protocol

Use replay-level train and validation splits. Use 16,384 critic optimizer steps for the main probe
and the same compact replay mixing unless a measured systems limit requires a documented change.
The actor remains frozen, so this is not a policy data-budget comparison.

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
- Compare Q error with a state-only V baseline.

Fit a state-only Q ablation with matched optimizer and data. The chunk critic must improve held-out
prediction enough to exceed seed noise. Sensitivity alone is not enough; arbitrary sensitivity can
also be wrong.

Finally sample chunks from the frozen E4 policy. Report their behavior-model log likelihood,
Q value, V value, advantage, and ensemble disagreement. Separate in-support and low-support samples.
Do not use low-support values for E7 weights.

## Gate

E6 passes only if:

- Q improves over the state-only ablation on held-out logged chunks.
- Calibration and ranking are stable across seeds.
- Temporal and single-action perturbations have nontrivial, bounded effects.
- Policy-sampled in-support chunks have controlled ensemble disagreement.
- Advantages and proposed weights remain finite with a useful effective sample size.

If Q ignores actions, fails calibration, or extrapolates on policy samples, do not train chunk AWR.
Record E6 as a negative result. Closed-loop execution without a trusted chunk critic may still be a
separate behavior-cloning systems test, but it is not E7.

## Results

Pending E4 and E5 infrastructure.
