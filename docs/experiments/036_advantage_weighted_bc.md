# 036 advantage-weighted behavioral cloning

Status: production run complete. The official run and metric record is in the
[audited blog experiment table](../blog_experiment_evidence.md).

## Aim

Test whether advantage-weighted regression improves the deployed 026 policy when the advantage
comes from a dense reward. Experiment 020 ran AWR on an older architecture with a sparse
stock-only reward and was null twice. This experiment keeps the exact 026 model, data stream,
optimizer, schedule, and evaluation, and changes only two things: a value head that fits
full-episode returns, and a per-position weight on the behavior-cloning loss.

The reference is 026 W&B run `cqbbbg77` (16,384 steps, effective batch 512). The primary gate is
closed-loop `net_stock_lcb` against that reference, with the mirrored H2H protocol as the
promotion check. Better value calibration alone does not promote the actor.

## Treatment

For context position `t`, the deployed head predicts the chunk that starts at frame `t+1`, so its
return target is

\[
G_{t+1} = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots
\]

computed on the complete replay before window sampling. The reward is the E5 dose: `+1` when the
opponent loses a stock, `-1` when the ego does, `+-0.01` per percent dealt or taken, `+-0.5` on
the match-deciding stock, and `gamma = 0.99827` (about a 400-frame half-life). Experiment 031
used `0.99^(1/4) ~= 0.99749` for the same dose; 036 follows the E5 document and keeps gamma a
config field.

The advantage and weight are

\[
A_t = G_{t+1} - V(s_t),
\qquad
\tilde w_t = \min\left(e^{A_t/\beta}, w_{max}\right),
\qquad
\beta = 0.8,\; w_{max} = 5,
\]

computed in log space (clip before exponentiation, `logsumexp` normalization) and normalized to
mean 1 over the eligible rows of one optimizer batch. Truncated replays (no observed terminal
stock) get NaN returns and a false eligibility mask: their rows keep weight 1 and take no value
loss. `w_max` caps the raw weight only; the normalized maximum can exceed it and is reported,
never clipped again.

The value head is a two-layer MLP (`NonlinearActionHead`, hidden 256) on **detached** trunk
states, created after every policy module so policy initialization matches a same-seed 026 model.
Its MSE runs over eligible rows only. For steps below 2,048 the actor weights are all 1 (plain
BC) while the value head trains; activation depends only on the global step.

## Arms

Three axes vary between runs; nothing else changes:

- `advantage_scope`: `primary` weights only the dense-prefix (deployed chunk) joint NLL;
  `all` also weights the auxiliary offsets.
- `head_offsets`: for example `(1..6)`, `(1..6, 12, 20)`, and the 026 default
  `(1..6, 9, 12, 16, 20)`. Every variant keeps the dense `1..6` prefix the live decoders need.
- `temporal_state_film`: FiLM the temporal-chain states on the trunk state (see below).

## Decoder conditioning

036's temporal decoder differs from 026's in one exact rewrite and one ablation flag:

- **Additive token projection (always on; same function as 026).** 026 copies the trunk state
  into all ten chain tokens and projects the concatenation. A linear layer over a concatenation
  decomposes into a sum of per-part linears, so 036 computes the trunk share once per position
  and broadcast-adds the per-step action and offset shares. The `token_projection` parameter is
  unchanged (same shape, same same-seed initialization); only the compute schedule changes. This
  removes the `[B, L, 10, 384]` copy of the trunk state and ~80% of the projection FLOPs. A test
  pins equality against the concatenating reference; the warm-up-equals-026 test therefore
  asserts near-exact (not bitwise) equality.
- **`temporal_state_film` (ablation, default off).** The concat/additive conditioning gives the
  chain no multiplicative interaction with the trunk state at the input. With the flag on, a
  zero-initialized FiLM layer (`s * (1 + tanh(scale)) + shift`, scale/shift from the trunk
  state) modulates the post-chain states in both the parallel teacher-forced path and every
  stepwise decode path. Zero initialization means the arm starts at the exact baseline function.
  Off reproduces 026's decoder behavior.

Every logged train and validation metric stays unweighted and identical to 026, so arms and the
control compare directly. Weights change the backprop objective only.

## Deviations from the E5 design

Recorded on purpose; each is deliberate:

- **The "primary" scope weights the joint NLL of offsets 1 to 4, not offset 1 alone.** E5 was
  written for an architecture that deploys one frame per decision; 026 deploys a four-frame
  chunk, so the chunk is the action the advantage credits. The `head_offsets` (1..6) arm keeps
  offsets 5 and 6 as unweighted auxiliaries under "primary" scope.
- The value head lives in the shared optimizer's Adam groups, not a separate AdamW with its own
  clip. 026's optimizer routes it there automatically, its `step` is strictly per-parameter, and
  the gradient-norm call measures without clipping, so nothing couples value and policy updates.
  Side effect: the value head's weight matrices receive the shared `0.01` weight decay.
- There is no warm-up ESS gate that stops the run. The `--audit-returns` beta sweep is the
  pre-run check, and the watch metrics below stand in for the gate. Beta is not retuned from
  closed-loop results.
- `grad_accum_steps` must be 1 (026's production value), so the mean-1 normalization covers the
  whole optimizer batch with no cross-microbatch pass.

## Watch metrics

Two failure modes are tolerated by design (per E5) rather than guarded by a crash, so they must
be watched in W&B during the first weighted steps:

- `train/weight_ess`, `train/weight_clip_frac`, `train/weight_norm_max`: `w_max` caps the raw
  weight only. After mean-1 normalization a single row can exceed the cap when most advantages
  are negative (a bimodal batch can reach ~30x), and no gradient clipping attenuates it. A
  collapsing ESS means beta is too small for the realized advantage scale.
- `train/eligible_frac`: a stream where almost every replay is truncated silently degrades the
  arm into an exact 026 re-run (all weights 1, no value learning). Near-zero eligible_frac means
  the result is not evidence about AWR.

## Files

- `experiments/036_advantage_weighted_bc.py` — one self-contained file: the full 026
  architecture and train loop (copied, per the experiments-are-single-files rule; experiments
  never import each other) plus the treatment: `AWRBatch`, `collate_awr_batch`,
  `advantage_weights`, `advantage_weighted_objective`, `microbatch_loss`, the value head, the
  loader labeling, and `return_audit`.
- `hal/training/returns.py` — shared reward events, discounted returns (`scipy.signal.lfilter`),
  terminal inference, and per-port replay labeling. Experiment 031 imports the same code.
- `hal/training/features.py` — `TrainBatch.record_stream` also records the optional `slot_ids`
  and `reset` tensors; 036's `device_batches` uses it so `AWRBatch` stages its extra tensors on
  the copy stream.
- `tests/test_returns.py`, `tests/experiments/test_036_advantage_weighted_bc.py` (the test file
  loads 026 as well, only to assert same-seed policy equality and warm-up == BC).
- `experiments/026_temporal_mtp.py` is byte-identical to its pre-036 state.

## Commands

```bash
uv run experiments/036_advantage_weighted_bc.py --audit-returns --audit-split val
uv run experiments/036_advantage_weighted_bc.py
uv run experiments/036_advantage_weighted_bc.py --cfg.advantage-scope all
uv run experiments/036_advantage_weighted_bc.py --cfg.head-offsets 1 2 3 4 5 6
uv run experiments/036_advantage_weighted_bc.py --eval runs/<run>/final.pt
```

Run the return audit before the first production run and record the terminal fraction, return
quantiles, and the beta table here. The audit shuffles the stream so it samples the corpus
rather than the first shard, and it reports per-port return means: the two ports negate each
other, so the pooled mean is zero by construction and only the per-port means can expose a sign
or asymmetry bug.

## Results

The production W&B run is `hwzv0k9a`. Use the audited blog experiment table for
the final metrics, control, lineage, and mismatch record.
