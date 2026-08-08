# E7: chunk AWR and chunk execution

Status: blocked on the E6 critic gate

Updated: 2026-08-07

## Question

Does advantage-weighting a joint dense action chunk improve a policy that commits to that same chunk
in closed loop?

E7 aligns all three objects:

- The policy chooses `a_{t+1:t+H}`.
- The critic scores `Q_H(s_t, a_{t+1:t+H})`.
- The evaluator executes all H actions before asking the policy again.

Start with `H=2`. Test `H=4` only after H2 passes its correctness and evaluation gates.

## Chunk weight versus trajectory weight

A chunk weight and a trajectory weight are not the same. A chunk advantage depends on the current
state and the particular H-action sequence. It changes at every chunk boundary within one replay. A
trajectory weight applies one episode-level outcome to every action in that episode.

E7 uses a chunk advantage. It does not broadcast one game return over the full replay.

## Required controls

For each horizon, use the same dense temporal policy and run:

1. Execution-only control: evaluate the E4 checkpoint with execution horizon H and no retraining.
2. Macro-BC control: train with the same equal-per-frame joint chunk objective as E7 but set every
   advantage weight to one.
3. Macro-AWR treatment: train from the same starting checkpoint and data order with E6 advantages.

The macro-BC control is required. E4 gives offset 1 more loss weight than each auxiliary head, so it
is not the weight-one version of a joint chunk likelihood.

## Files to change

- `experiments/023_mtp_heads.py`: add macro-BC and macro-AWR objectives, frozen E6 critic loading,
  chunk-weight diagnostics, and explicit H-frame execution.
- `hal/training/chunks.py`: reuse the exact E6 interruption predicates.
- `hal/training/closed_loop.py`: add an optional per-slot interruption callback that clears only
  the pending chunk and replan clock while keeping the observed rolling context.
- `tests/experiments/test_023_mtp_heads.py`: test joint likelihood, weight scope, critic alignment,
  queue behavior, reset behavior, checkpoint identity, and end-to-end training.
- `tests/test_closed_loop_rings.py`: test raw rolling contexts after multi-frame commitment and
  mixed slot resets.
- This file: record exact source actor and critic checkpoints, commands, runs, timing, metrics,
  artifacts, and decisions.

Do not change the reward, critic, data, factorization order, decode temperature, or evaluation
schedule after seeing E7 results.

## Actor objective

For logged dense chunk `a_{t+1:t+H}`, keep each frozen E6 Q paired with the V from the same critic
seed:

\[
A_{H,i}(s_t,\mathbf a_t^H)=Q_{H,i}(s_t,\mathbf a_t^H)-V_i(s_t),
\qquad
\bar A_H=\frac{1}{3}\sum_{i=1}^{3} A_{H,i}.
\]

Use the mean advantage in the actor weight. Do not mix a Q from one seed with a V from another seed.

Compute a detached, clipped exponential weight with the same finite implementation as E5. Choose
`beta_H` from the frozen E6 validation audit before E7. Test `beta` values
`(0.2, 0.4, 0.8, 1.6, 3.2)` with raw `weight_max=5`. After the disagreement fallback, choose the
smallest beta for which both normalized ESS ratios are at least 0.2 and no more than 20% of raw
weights hit the cap. If no value passes, do not train macro-AWR. Select H2 and H4 separately and
save the complete selection table before either actor run.

Normalize valid weights to mean one within the effective batch with the same FP32 `logsumexp`
calculation. Do not divide underflowed raw exponentials by their mean. Report raw and normalized
statistics. The cap applies before normalization; the final normalized maximum can exceed 5. Do
not clip it again.

Require `grad_accum_steps=1` in the first H2 and H4 arms so normalization covers the whole optimizer
batch. Report ESS over valid chunks and over replay-window mean weights. Require both ratios to pass
the E6 threshold; chunk rows within one replay window are not independent.

Apply the E6 disagreement rules before normalization. If a logged chunk exceeds either its Q or
advantage disagreement threshold, ignore its advantage and set its raw weight to one. Keep its
macro-BC loss. Do not drop the row, because that would change the behavior-data distribution. Low
behavior-policy likelihood alone is not a reason to discard an observed chunk. Log likelihood,
both disagreements, the fallback fraction, and effective sample size before and after the fallback.

The joint chunk NLL is:

\[
\ell_H=-\sum_{k=1}^{H}\sum_g
\log p(a_{t+k,g}\mid s_t,a_{t+1:t+k-1},a_{t+k,<g}).
\]

One chunk weight multiplies every temporal and within-frame factor in this sum. Divide by H in the
implemented loss to keep gradient scale per executed frame stable across H. This division is one
constant for a fixed horizon, so it does not change the optimum.

For `H=2` with four dense output frames:

\[
L=\operatorname{mean}(w_H\ell_H/H)+\operatorname{mean}(L_3,L_4).
\]

Offsets 3 and 4 are still auxiliary predictions, so they remain unweighted BC. For `H=4`, all four
predictions belong to the executed macro-action and receive the one chunk weight; there is no future
auxiliary term.

This is when future heads should receive advantage weights: only when they are factors of the action
that the critic scores and the evaluator commits to. An unused future prediction remains an
auxiliary BC target.

Apply the E6 chunk-interruption mask to the executed joint term. If control is interrupted before H
actions can execute, do not train that row as an H-step macro-action. Macro-BC and macro-AWR use the
same mask. An interruption on the Hth transition remains valid. Auxiliary offsets outside the H-step
macro-action keep their ordinary BC masks.

Freeze the complete three-seed E6 critic package, including its E4 state encoder, Q heads, V heads,
action encoders, and buffers. The three source encoders must have identical state hashes. Recompute
critic state features once from the raw context with one verified frozen encoder, then feed the same
features to all three Q/V pairs.
Do not feed the fine-tuning actor's changing hidden state into Q or V. The actor cannot change the
critic or backpropagate through the weight. Reject a checkpoint whose horizon, reward settings,
action factorization, or source actor does not match.

This adds one frozen state-encoder forward during training. Record its time and memory separately.
Do not share activations with the actor merely to improve throughput; that changes the weight as the
actor learns.

## Training protocol

Start macro-BC and macro-AWR from the same final E4 checkpoint. Give each arm 16,384 optimizer steps
with identical replay order and seed. Reset the optimizer and schedule for both arms. Record this as
fine-tuning exposure, not training from scratch.

Use the selected attention package, dense offsets `(1,2,3,4)`, selected within-frame order, temporal
teacher forcing, compact data, replay reservoir, batch token count, optimizer, and checkpoint cadence.

Macro-BC and macro-AWR may differ only in the chunk weight, run label, and H2H reference fields.

## Closed-loop execution

At a decision boundary:

1. Rebuild the newest `L_ctx` raw frames.
2. Run the complete transformer once.
3. Sample all H action frames autoregressively from the temporal and within-frame chain.
4. Queue and execute exactly those H controller states.
5. Observe every intervening emulator state and append it to the raw rolling buffer.
6. After H frames, discard the plan and rebuild the newest raw window again.

Do not use temporal KV. Do not reuse contextualized states after the raw buffer rolls. Do not
in-paint an old plan into the next inference. Plan carry-over and shorter receding horizons are later
experiments.

If either player's stock count changes, the frame ID resets, a slot ends, or the evaluator rejects a
boot, discard the queued chunk immediately. Keep ordinary hitstun, shield stun, and action animation
frames; inputs still execute on those frames. Never execute actions from the previous game or slot.

The online callback must inspect the new observation before selecting that frame's controller
output. If action `k < H` caused an interruption visible in the new observation, clear actions
`k+1:H` and replan from that observation. If action H caused it, the completed chunk remains valid
and the next call replans normally. This is the same boundary used by E6's logged mask.

## Required tests

1. Use unique target classes to prove the H2 and H4 joint likelihood includes exactly the intended
   frame and group terms.
2. Assert one scalar chunk weight multiplies every included term.
3. Assert H2 leaves offsets 3 and 4 unweighted and H4 leaves no auxiliary head.
4. With weights one, macro-AWR equals macro-BC exactly.
5. Assert division by H gives equal per-frame scale for constant losses.
6. Assert Q2 receives only actions 1 and 2; Q4 receives actions 1 through 4.
7. Assert Q, V, advantages, and weights are detached and frozen.
   Change the actor trunk while holding raw input fixed and assert critic features, Q, V, and weights
   remain exactly unchanged.
8. Reject critic and actor checkpoint identity or horizon mismatches.
9. Compute three hand-written Q/V pairs and check the exact paired mean advantage. Assert a logged
   chunk with high Q disagreement or high advantage disagreement receives raw weight one and
   remains in the macro-BC loss. Assert low behavior likelihood alone does not trigger the fallback.
10. Assert disagreement gating happens before effective-batch weight normalization.
    Check chunk-level and replay-window ESS by hand.
    Check that all-large-negative advantages still produce finite mean-one weights.
11. Assert teacher forcing uses logged earlier actions and free-running decode uses sampled earlier
   actions.
12. Assert an H-frame queue causes one policy inference followed by exactly H controller outputs.
13. Assert all intervening raw states and executed actions enter the next rebuilt context.
14. Assert a stock change, frame reset, slot end, and rejected boot clear every queued action.
    Assert ordinary hitstun does not clear it.
15. Assert mixed vector slots can sit at different chunk phases without sharing state.
16. Run longer than `L_ctx` and confirm rebuilt contexts contain no dropped raw frame information.
17. Save and reload horizon, objective, critic identity, reward settings, and decode protocol.
18. Run small macro-BC and macro-AWR jobs with finite losses, gradients, weights, and parameters.
19. Reject macro-AWR with `grad_accum_steps != 1` in the first arms.
20. Assert every frozen critic parameter and buffer remains byte-identical through training.
    Assert the three source-encoder hashes match and only one encoder forward is used per actor
    batch.
21. Assert a stock change or frame reset before H removes the macro-action row in both macro-BC and
    macro-AWR, while the same event on the Hth transition keeps it.
22. Test the fixed beta grid and selection rule, including the exact ESS and clip boundaries and
    the no-passing-beta failure. Assert H2 and H4 select independently.

Run focused tests, Ruff, type checking, Python compilation, and `git diff --check` before the GPU
gate.

## Evaluation sequence

For H2:

1. Evaluate the frozen E4 actor with H1 and H2 execution using the same checkpoint.
2. Train and evaluate macro-BC H2.
3. Train and evaluate macro-AWR H2.
4. Run 64 mirrored H2H configurations for macro-AWR H2 against macro-BC H2.

Proceed to H4 only if H2 has no queue or reset failure, passes both ESS gates and the raw clip gate,
and the execution-only H2 point estimate for stocks taken minus stocks lost per active minute is no
more than 0.25 below the same frozen actor at H1. This is a safety gate, not an improvement claim.
Repeat the same execution-only, macro-BC, macro-AWR, and H2H sequence.

Request 32 periodic matchups and 96 final matchups. Limit both sweeps to 32 concurrent Dolphin
boots. Save all rows, replays, critic and weight audits, and H2H records.

Report:

- Stocks, damage, dead frames, terminal results, crashes, and CPU rate differences.
- H2H stock difference, non-tied stock-lead rate, confidence intervals, and ties.
- Raw and normalized weight distributions, clip fraction, chunk ESS, and replay-window ESS.
- Policy calls, committed frames, interrupted chunks, and queue clears.
- Mean, median, and p95 inference time per decision and per executed frame.
- Action transition rates and divergence between teacher-forced and free-running chunks.

Use one RTX 4090 experiment at a time. Target 3.0 to 3.5 hours per training arm through evaluation
and upload. Flag startup over 30 minutes, warm steps over 0.5 seconds, a slowdown over 25% from the
macro-BC arm, a periodic evaluation over 25 minutes, or a projected total over 3.5 hours.

## Decision

- The AWR claim comes from macro-AWR versus macro-BC at the same execution horizon.
- The commitment claim comes from execution horizon H versus H1 on the same frozen actor.
- Do not credit AWR for an inference-speed gain caused by fewer policy calls.
- Do not credit chunking for a policy gain caused only by extra fine-tuning.
- A good H2 result does not imply H4 will work.
- If the critic gate fails or weights collapse, do not launch macro-AWR.

Promote macro-AWR at a fixed horizon only if it has no new crash or artifact failure, its final CPU
point estimate for stocks taken minus stocks lost per active minute is better than macro-BC, its
paired H2H mean stock difference per configuration is positive, and more configurations favor
macro-AWR than macro-BC. Report all intervals and ties. Mixed evidence is inconclusive and keeps
macro-BC as the result for that horizon.

A model that predicts four actions but executes only one is not a chunk policy. A model that uses a
trajectory return as one weight for every decision is not this chunk-AWR experiment.

## Results

Pending E6 completion.
