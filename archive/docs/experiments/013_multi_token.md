# 013 multi-token experiment design

This is the durable design log for `013_multi_token.py` and the ablations that follow it. Keep 013 itself a
validity-fixed version of 012: its default model, targets, loss, optimizer, and decode distribution should stay
unchanged unless a difference is explicitly listed here.

## Frozen evaluation protocol

- Periodic training evaluation: 16 deterministic prior-sampled matchups.
- Final-checkpoint evaluation: 96 deterministic prior-sampled matchups.
- Matchup count and concurrency are separate. `eval_parallel_per_cpu` controls only how many of the fixed matchups
  run simultaneously.
- The policy sampler has an explicit evaluation seed; matchup count, frame budget, seed, and concurrency are saved
  in the checkpoint configuration.
- Manual checkpoint evaluation defaults to the final 96-matchup protocol, with explicit CLI overrides available.
- Every sweep persists `match_rows.json` beside its replays. The file contains the exact trajectory-derived rows and
  the matchup count, concurrency, frame budget, seed, execution horizon, and decode settings needed to audit or pair
  it with another checkpoint.
- Interpret closed-loop comparisons through paired matchup rows and uncertainty intervals, not one pooled point
  estimate. Offline NLL and transition metrics are diagnostics, not substitutes for closed-loop performance.

## Diagnostic instrumentation added after the validity baseline

These metrics are observational: they run in eval mode on frozen validation data, use `autograd.grad` rather than
`.backward()`, do not populate parameter gradients, and do not consume the training RNG stream.

- At each validation boundary, use the first `gradient_diagnostic_batch_size` examples (default 64) from the first
  frozen validation batch to compute each horizon loss's exact gradient over shared parameters. Output-head
  parameters are excluded.
- Log `grad/head_<o>_norm` and every pairwise `grad/cos_<o>_<p>`.
- Also log the direction the configured objective actually applies,
  `aux_loss_weight * sum(auxiliary_head_gradients)`: its norm, norm relative to the primary, cosine with the primary,
  and coordinate-wise sign-conflict fraction. This catches objective-scale domination separately from semantic
  gradient conflict.
- Validation logs the argmax next-action change rate relative to the observed current action, adjacent prediction
  persistence, and A-B-A one-frame flipback rate for each action group.
- Validation logs raw factorized-model probability mass on digital L/R clicks without a full corresponding analog
  trigger. This is measured before the optional click-trigger decode repair, so it diagnoses the model rather than
  the postprocessor.
- When a checkpoint embeds validated full-dataset button counts, validation logs mass on unseen combos and combos
  below `diagnostic_rare_button_count` (default 100). Without that artifact it logs counts-unavailable and omits the
  two mass values; it never substitutes the old reference-sample table.

## Four ablations after the 013 validity baseline

All four arms must use the same data windows, number of optimizer steps, evaluation matchup schedule, policy seed,
decode settings, and final 96-matchup protocol. Auxiliary-loss comparisons should report shared-trunk gradient
norms and pairwise gradient cosine similarities in addition to losses.

### A. Next-action-only control

Configuration: `head_offsets=(1,)`; no auxiliary heads.

Hypothesis: this should give the cleanest optimization of the deployed head. It should beat multi-horizon training
if the future heads cause negative transfer, but may generalize worse if future prediction is useful representation
regularization.

Expected evidence:

- Better offset-1 train NLL is nearly guaranteed because all capacity and updates serve that task.
- Better validation NLL and closed-loop play would mean the current auxiliary heads are net harmful.
- Worse validation or closed-loop play despite better train NLL would support a genuine regularization benefit from
  multi-horizon prediction.

Unexpected-result interpretation:

- If validation improves but closed loop does not, next-action likelihood is not the limiting policy metric; inspect
  transition behavior, temporal coherence, and covariate shift.
- If both train and validation worsen, check that comparisons use equal optimizer steps and that changing the number
  of heads did not inadvertently alter learning-rate or loss normalization.

### B. Spread horizons with normalized auxiliary weight

Configuration: `head_offsets=(1,5,9,13)` and
`loss = primary + lambda_aux * mean(auxiliary_heads)`, initially `lambda_aux=0.25`. The current 013
`aux_loss_weight` is a **per-head** multiplier, so this arm must set it to `lambda_aux / 3`; setting it directly to
0.25 would give the auxiliary sum total weight 0.75 and would not implement this hypothesis.

Hypothesis: spread horizons may teach slower game-state/intent features, but their total influence should be much
smaller than the deployed task. This should retain any representation benefit while avoiding the 012 objective in
which discarded heads contribute most of the scalar loss.

Expected evidence:

- Offset-1 validation NLL at least as good as 012's unnormalized objective.
- Smaller auxiliary-vs-primary trunk gradient domination and less negative gradient cosine.
- If long-horizon representation is useful, closed-loop gains may exceed what the small NLL change suggests.

Unexpected-result interpretation:

- If it is worse than both A and C, long-horizon action targets are probably too conditionally ambiguous or conflict
  with reactive control.
- If it is worse than unnormalized 012, either strong auxiliary supervision was genuinely useful or optimization
  scale changed; use gradient diagnostics before attributing the result to horizon semantics.
- If far-head NLL improves while the primary worsens, the arm is still a failure for the deployed policy.

### C. Contiguous near horizons with normalized auxiliary weight

Configuration: `head_offsets=(1,2,3,4)` with the same normalized `lambda_aux=0.25`.

Hypothesis: nearby targets should be less multimodal and more gradient-aligned with the next-action task than offsets
5/9/13. This arm should produce the best primary NLL of the multi-head variants and also leaves heads suitable for a
later chunk-execution test. The existing exploratory contiguous run is encouraging but not conclusive because it did
not use the corrected evaluation and loss-normalization protocol.

Expected evidence:

- More positive gradient cosine with the primary head than arm B.
- Better offset-1 validation NLL than arm B.
- Chunk heads should have monotonically increasing NLL with horizon and reasonable temporal consistency, though this
  arm is evaluated with `exec_horizon=1` for the representation-learning comparison.

Unexpected-result interpretation:

- If B beats C, useful supervision likely comes from slower intent/state features rather than local action smoothing.
- If A beats C, even nearby direct forecasts cause negative transfer.
- If C improves offline metrics but not closed loop, independently decoded temporal marginals or BC covariate shift
  remain the likely bottleneck; do not infer that chunk execution will help.

### D. Proper primary NLL plus a separate transition/duration auxiliary

Configuration: preserve unweighted next-action NLL for the deployed distribution and add a separate auxiliary task
that predicts whether each action group changes (and, if implemented, its remaining hold duration). Do not use
transition-weighted primary cross-entropy.

Hypothesis: this should improve sparse change-event representation and transition F1 without biasing the sampled
action distribution toward excessive changes or sacrificing hold calibration.

Expected evidence:

- Transition/change F1 improves relative to A at similar primary NLL.
- Hold NLL does not suffer the large degradation seen with transition-weighted cross-entropy.
- Closed-loop behavior shows fewer copycat fixed points without obvious controller chatter.

Unexpected-result interpretation:

- Better auxiliary accuracy without better primary transitions means the auxiliary representation is not reaching or
  helping the action head; inspect loss weight and trunk gradient flow.
- Better transition F1 but worse closed loop means the event metric rewards changes that are not strategically useful,
  or the policy is changing at the wrong time despite the tolerance window.
- No transition improvement may mean quantized boundary changes are too noisy a target; use persistent/distance-aware
  stick events or duration targets.

## Deliberately deferred modeling changes

These are plausible improvements, but they must not enter 013 because they would confound the four ablations above.

- Autoregressive dependencies between buttons, main stick, C-stick, and triggers.
- A coherent temporal chunk decoder conditioned on earlier sampled actions.
- A persistent latent plan sampled less frequently than controller frames.
- Change/hazard/duration action factorization beyond arm D's isolated auxiliary experiment.
- Relative and facing-normalized player geometry, stage-relative features, and explicit interaction features.
- Per-player encoders or structured cross-player attention instead of one flat concatenation.
- Future-state, opponent-action, value, or outcome auxiliary targets.
- Learned or geometry-aware controller quantization and distance-aware stick losses.
- Residual-branch scaling changes, learned RMSNorm gains, gated MLPs, or other backbone changes.
- DAgger, self-play fine-tuning, offline RL, advantage-weighted BC, or other covariate-shift interventions.
- Dataset filtering or conditioning by player skill/style/identity.

## Deferred performance work

These should be implemented and benchmarked separately after the objective ablations unless resource limits block the
runs. Every optimization needs numerical-parity tests before becoming the new baseline.

- Incremental inference with a per-layer KV cache and sliding-window eviction; clear each slot's cache at match reset.
- Packed/variable-length causal attention or padding buckets to avoid the default 64 MiB per-batch boolean mask and
  recover the fastest causal SDPA kernel.
- Quantize controller group targets in data-loader workers instead of constructing large GPU nearest-center
  temporaries on every step.
- Filter unused opponent-controller and START columns before host-to-device transfer.
- Keep cached validation batches on CPU (preferably pinned) and transfer one batch at a time.
- Reuse validation hidden states/logits for proper scores, argmax reconstruction, and sampled reconstruction instead
  of performing four backbone passes per batch.
- Fuse the horizon projections into one matrix operation and reshape to `[B,L,H,A_VOCAB]`.
- Replace repeated boolean-index gathers in group cross-entropy with an ignored-target or masked unreduced loss.
- Evaluate `torch.compile` after removing or gating runtime `jaxtyped`/`beartype` checks from the training hot path.
- Avoid unconditional per-step CUDA synchronization and full gradient-norm measurement when not profiling.
- Store constant character/stage conditioning once per sample rather than duplicating it across all context frames.

## Result interpretation guardrails

- Primary validation NLL is the main offline score; auxiliary NLL is never evidence of a better deployed policy by
  itself.
- Always split primary NLL into hold and transition subsets, but remember that transition-weighted metrics are not
  proper scores.
- Report action persistence, predicted change rate, temporal chatter, and cross-group validity. Only report rare-action
  mass when rarity comes from the checkpoint's validated full-dataset count artifact; never use the old 614-replay
  reference table as if it described the current training set.
- Closed-loop estimates need the final 96-matchup protocol and uncertainty intervals. Periodic 16-matchup evaluations
  are early-warning diagnostics only.
- A statistically clear offline improvement with no closed-loop improvement is evidence against the offline metric as
  a sufficient proxy, not evidence that the closed-loop result should be ignored.
- A closed-loop improvement without an offline improvement can be real; inspect transition timing, robustness, and
  distribution shift rather than selecting solely by NLL.
