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
- `tests/experiments/test_023_mtp_heads.py`: test joint likelihood, weight scope, critic alignment,
  queue behavior, reset behavior, checkpoint identity, and end-to-end training.
- `tests/test_closed_loop_rings.py`: test raw rolling contexts after multi-frame commitment and
  mixed slot resets.
- This file: record exact source actor and critic checkpoints, commands, runs, timing, metrics,
  artifacts, and decisions.

Do not change the reward, critic, data, factorization order, decode temperature, or evaluation
schedule after seeing E7 results.

## Actor objective

For logged dense chunk `a_{t+1:t+H}`, use the frozen E6 probes:

\[
A_H(s_t,\mathbf a_t^H)=Q_H(s_t,\mathbf a_t^H)-V(s_t).
\]

Compute a detached, clipped exponential weight with the same finite implementation as E5. Choose
`beta_H` from the frozen E6 validation audit before E7. Normalize valid weights to mean one within
the effective batch and report raw and normalized statistics.

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

Freeze the E6 Q and V networks. The actor cannot change the critic or backpropagate through the
weight. Reject a checkpoint whose horizon, reward settings, action factorization, or source actor
does not match.

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

If a game ends, a slot resets, a player is not controllable, or the evaluator rejects a boot,
discard the queued chunk immediately. Never execute actions from the previous game or slot.

## Required tests

1. Use unique target classes to prove the H2 and H4 joint likelihood includes exactly the intended
   frame and group terms.
2. Assert one scalar chunk weight multiplies every included term.
3. Assert H2 leaves offsets 3 and 4 unweighted and H4 leaves no auxiliary head.
4. With weights one, macro-AWR equals macro-BC exactly.
5. Assert division by H gives equal per-frame scale for constant losses.
6. Assert Q2 receives only actions 1 and 2; Q4 receives actions 1 through 4.
7. Assert Q, V, advantages, and weights are detached and frozen.
8. Reject critic and actor checkpoint identity or horizon mismatches.
9. Assert teacher forcing uses logged earlier actions and free-running decode uses sampled earlier
   actions.
10. Assert an H-frame queue causes one policy inference followed by exactly H controller outputs.
11. Assert all intervening raw states and executed actions enter the next rebuilt context.
12. Assert match end, slot reset, death/reset state, and rejected boot clear every queued action.
13. Assert mixed vector slots can sit at different chunk phases without sharing state.
14. Run longer than `L_ctx` and confirm rebuilt contexts contain no dropped raw frame information.
15. Save and reload horizon, objective, critic identity, reward settings, and decode protocol.
16. Run small macro-BC and macro-AWR jobs with finite losses, gradients, weights, and parameters.

Run focused tests, Ruff, type checking, Python compilation, and `git diff --check` before the GPU
gate.

## Evaluation sequence

For H2:

1. Evaluate the frozen E4 actor with H1 and H2 execution using the same checkpoint.
2. Train and evaluate macro-BC H2.
3. Train and evaluate macro-AWR H2.
4. Run 64 mirrored H2H configurations for macro-AWR H2 against macro-BC H2.

Proceed to H4 only if H2 has no queue or reset failure, a useful critic weight distribution, and no
large closed-loop collapse. Repeat the same execution-only, macro-BC, macro-AWR, and H2H sequence.

Use the standard 32-boot periodic and 96-boot final CPU protocol. Save all rows, replays, critic and
weight audits, and H2H records.

Report:

- Stocks, damage, dead frames, terminal results, crashes, and paired CPU deltas.
- H2H stock difference, non-tied win rate, confidence intervals, and ties.
- Raw and normalized weight distributions, clip fraction, and effective sample size.
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

A model that predicts four actions but executes only one is not a chunk policy. A model that uses a
trajectory return as one weight for every decision is not this chunk-AWR experiment.

## Results

Pending E6 completion.
