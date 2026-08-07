# Action prediction and chunk control roadmap

## Goal

Build a coherent action-chunk policy in small, testable steps. At every step, the probability model,
critic, and execution protocol must describe the same decision.

Closed-loop win rate and stock difference are the main results. Offline metrics explain the result.
They do not replace it.

## Terms

The current MTP heads model separate future marginals:

\[
p(a_{t+1}\mid s_t),\quad p(a_{t+5}\mid s_t),\quad
p(a_{t+9}\mid s_t),\quad p(a_{t+13}\mid s_t).
\]

A within-frame chain models the controller groups at one frame:

\[
p(a_t\mid s)=\prod_g p(a_{t,g}\mid s,a_{t,<g}).
\]

A temporal chain models a joint future sequence:

\[
p(a_{t+1:t+H}\mid s_t)
=\prod_{k=1}^{H}\prod_g
p(a_{t+k,g}\mid s_t,a_{t+1:t+k-1},a_{t+k,<g}).
\]

Sparse offsets can define a joint distribution over sparse future actions. They do not define a
complete executable chunk. A chunk requires dense offsets such as `(1, 2, 3, 4)`.

## Fixed evaluation rules

- Use the same data, optimizer, training examples, CPU character schedule, and decode settings across an
  ablation.
- Run at least three paired seeds. Use five for a final claim.
- Report paired uncertainty intervals for win rate and stock difference.
- Report parameter count, training time, and inference latency.
- Compare fixed-data and fixed-compute budgets when an architecture adds material work.
- Keep rank weighting off for this sequence.
- Select models without using the final opponent set.
- Treat primary NLL, transition NLL, action accuracy, calibration, and gradient metrics as
  diagnostics.
- Save exact initialization tests for every zero-initialized residual branch.

## E0: normalized auxiliary BC baseline

Re-run the independent `(1, 5, 9, 13)` MTP baseline with AWR off:

\[
L=L_1+\lambda_{aux}\frac{L_5+L_9+L_{13}}{3}.
\]

Use `lambda_aux = 1.0` for the first run. The old per-head weight of `1.0` had total auxiliary
weight `3.0`; it is not the same objective. Record that old result as historical evidence, not as a
direct control.

Required metrics:

- Joint and per-group primary NLL.
- Per-group argmax accuracy.
- Hold and transition NLL and accuracy.
- Predicted transition rate and persistence.
- Auxiliary-to-primary trunk gradient norm and cosine.
- Closed-loop win rate and stock difference.

## E1: output-head capacity control

Add one shared state MLP and a zero-initialized logit-residual output for every group and offset. Do
not condition one action group on another.

This isolates extra nonlinear capacity from action factorization. Match the E2 parameter count as
closely as practical. Keep the trunk, targets, loss, and execution unchanged.

E1 must start as the same function as E0. Test this directly.

## E2: within-frame action factorization

Condition each group on the true earlier groups during training and on sampled earlier groups during
decode.

The state and condition projections form one affine preactivation:

\[
W_h h + W_c c = W[h;c].
\]

Compute the state projection once, then reuse it across groups and offsets:

\[
c_g=[E(a_{<g})],
\]

\[
u_g=\mathrm{SiLU}(W_h\,\mathrm{RMSNorm}(h)+W_{c,g}c_g),
\]

\[
\ell_{o,g}=W_{o,g}h+b_{o,g}+W_{2,o,g}u_g,
\]

with `W_2` and `W_c` initialized to zero. Keeping `W_h` and `W_c` as separate modules lets the
model reuse `W_h h`; it does not change the algebra above. This keeps the independent-head function
at initialization and gives the MLP a nonlinear interaction between state and action condition. Do
not add a direct condition-to-logit bypass unless it is an explicit ablation.

### Chain order

The true joint entropy does not depend on factorization order. Finite model capacity and sampled
ancestor errors do.

Use two predeclared orders:

1. Stable-prefix order: `c_stick, triggers, main_stick, buttons`.
2. Intent-first order: `main_stick, buttons, triggers, c_stick`.

The stable-prefix order is the main arm. Its early groups should be easier to sample correctly, so it
limits error propagation. The intent-first order may provide more useful information to later groups,
but an early stick or button error can corrupt the rest of the chain.

Do not select an order from teacher-forced NLL alone. Also measure free-running ancestral accuracy and
closed-loop play.

## E3: temporally factorized sparse MTP

Replace the independent future heads with sequential MTP modules over `(1, 5, 9, 13)`.

Each module reads the previous hidden prediction and the teacher-forced previous target action. At
decode, it reads its own previous sample. Share action embeddings and output heads. Start with a
shared temporal module across depths unless profiling shows a clear reason not to.

Keep execution at one frame. Keep AWR off. This experiment asks whether a coherent sparse future
model improves representation and primary control.

Measure both teacher-forced and free-running future action metrics. Their gap measures temporal
exposure bias.

## E4: dense chunk-enabling control

Change the temporal offsets to `(1, 2, 3, 4)`. Continue to execute only offset 1.

Prior runs found dense MTP slightly worse in closed loop. Treat this as a known negative result and a
necessary control, not as a presumed policy improvement. The purpose is to obtain a complete short
action sequence that can later be executed and scored as one chunk.

Do not use a dense-head NLL improvement to overwrite the prior closed-loop result.

## E5: primary-only AWR

Apply advantage weights only to the deployed offset-1 action:

\[
L=w_t L_1+\lambda_{aux}\,\mathrm{mean}(L_{aux})+\lambda_V L_V.
\]

Rank weights may affect all terms because they change the data mixture. Advantage weights do not
apply to auxiliary marginals.

Run this on the best E2-style policy first. Then test the temporally factorized model if E3 earned a
clear continuation.

## E6: chunk critic validation

The current critic proves that return labeling, value fitting, and AWR weighting run end to end. It
does not prove that a chunk-conditioned action value works. The current critic is `V(s)`; the new
critic is:

\[
Q_H(s_t,a_{t:t+H-1}).
\]

Train it first as a read-only probe on logged dense chunks:

\[
y_t=\sum_{k=0}^{H-1}\gamma^k r_{t+k}+\gamma^H V(s_{t+H}).
\]

Use logged actions during critic training. Do not train only on policy logits or policy-generated
chunks.

Required checks:

- Held-out Bellman error and value calibration.
- Ranking of high-return and low-return logged chunks.
- Sensitivity to shuffled temporal order.
- Sensitivity to replacing one action in a chunk.
- Value estimates for policy samples compared with logged chunks.
- Stability across seeds.

This can be a short validation stage, but it cannot be skipped.

## E7: chunk AWR and chunk execution

When E6 passes, define one chunk advantage:

\[
A_H(s_t,\mathbf a_t)=Q_H(s_t,\mathbf a_t)-V(s_t).
\]

Weight the complete joint likelihood:

\[
L_{actor}=-w_H\sum_{k=1}^{H}\sum_g
\log p(a_{t+k,g}\mid s_t,a_{t+1:t+k-1},a_{t+k,<g}).
\]

Start with `H = 2`, then test `H = 4`. Execute all `H` actions before selecting the next chunk. This
keeps the critic, actor, and deployed macro-action aligned.

Receding-horizon execution with a plan longer than the executed prefix is a later experiment. It
requires either a critic for the executed prefix or a learned dynamics model for the unexecuted tail.

## Deferred work

- Plan carry-over, temporal ensembling, and inpainting.
- Next-state and reward prediction.
- Latent world models and model-predictive control.
- Prefix critics and adaptive chunk length.
- Online data collection or DAgger-style correction.

## RLE note

`experiments/018_bpe_rle.py` previously tested run-length and BPE action tokens without a closed-loop
gain. Do not rerun it as part of E0-E7. Before revisiting it, audit:

- Target alignment and token-boundary indexing.
- Training/decode equivalence.
- Queue reset behavior at episode boundaries.
- Loss normalization against frame-based baselines.
- The executed span distribution.
- Whether variable open-loop duration, rather than the tokenizer, caused the failure.

If the audit passes, compare a simple action-plus-duration model before returning to BPE. It is easier
to interpret and gives a cleaner semi-Markov baseline.

## Decision gates

- Continue from E1 to E2 only if the conditioning result beats the capacity control in closed loop or
  gives a clear diagnostic gain without a closed-loop loss.
- Continue from E3 to dense chunks only if free-running temporal predictions are coherent enough to
  execute.
- Let E4 remain a negative representation result if closed-loop play is worse. Its chunk can still be
  used for E6 and E7.
- Do not optimize the actor against the chunk critic until E6 passes its perturbation and calibration
  checks.
- Do not introduce a world model to explain a failure that can still be isolated in action BC or the
  chunk critic.
