# Actor correctness repair

This work starts at `603a1f9229ca17c49d0e1c608114d85318e1ac96`, from the
2026-10-25 Astra review. The checkpoint-32 producer is
`fde1887c6f7170cd817f2497fa1582b071f62c71`; its manifest SHA-256 is
`206883a06d62321c672ed1a8141233de8d429e110947e4c83a0dfd2299508968`.
The latest reviewed Modal call, `fc-01M5R9B2E4RKTTZT164ZHE0MCS`, returned 1.
The preserved Modal volume contains the manifest and 265 checkpoint shards.
All 265 shard hashes were rechecked by the CPU preflight. The preflight also
verified the compiler-cache payload hashes before copying them to a separate
diagnostic namespace. This establishes artifact identity, not numerical validity.

## Contracts

- Logits are unnormalized scores. Probability tensors contain masses; log-probability
  tensors contain natural logarithms. FP32/BF16 specify storage and arithmetic precision,
  not these representations.
- Each full head has shape `[windows, decisions, sampled_frames, categories]`.
  The four heads use controller decode order. Their category masses sum to one on
  active frames. Illegal categories have log probability `-inf`.
- Selected-action log probabilities have shape `[windows, decisions, 16]`, in
  frame-major decode order. They must equal a gather from the corresponding full head.
  Sampled action indices themselves use codec order; the orders are distinct.
- The joint PPO log ratio sums new minus behavior selected-action logs over executed
  factors. Exponentiation occurs once after this sum. Advantage normalization uses
  the full optimizer batch's statistics, including all ranks.
- Forward KL is `sum(p * (log_p - log_reference))` over categories, then a sum over
  executed factors. It is in nats. The KL coefficient is divided by advantage sigma.
  Local loss sums are divided by global selected-decision count / world size, before
  averaged gradients. References and all targets are detached.
- Invalid slots and unexecuted factors are unspecified and excluded from reductions.
  A rank with no selected decisions remains valid when the global denominator is
  positive. Such a rank provides no evidence of live numerical correctness.
- CUDA graph outputs can be reused. A reference or diagnostic value retained across
  later calls needs owned storage. Graph scratch addresses and parameter-gradient
  addresses have separate lifetime contracts.
- A failed live actor invariant must leave actor parameters, optimizer state,
  scheduler, and actor-step count unchanged. Critic training precedes actor training;
  this is not a promise to roll back an entire partially completed update.

## Current changes and evidence

- Both diagnostic reference paths now preserve log probabilities. CPU regressions
  use nonzero KL coefficients, equal and different policies, a scalar double-precision
  objective, finite-difference gradients, and deliberately incorrect probability units.
  Reference capture is also checked against a forward function that reuses storage.
- The existing full-head KL validation runs before actor optimizer mutation. Fault
  injection checks an already populated optimizer, scheduler, and actor-step count.
- Optional failure capture coordinates ranks, saves inputs, targets, output heads,
  model/optimizer state, gradients and RNG state, then terminates. It records a failed
  actor operation; it is not an exact-resume checkpoint.
- The precision report calls its finite-value result `finite`, not `passed`; its
  format is now v3. The redundant test for a removed gradient gate was deleted.
  Direction metrics have explicit numeric expectations. Historical v2 reports are
  unchanged. The unsupported BF16 learning conclusion was corrected.
- Each continuation gets a new W&B run ID. Its lineage records the restored
  checkpoint hash, checkpoint producer, current source, and previous attempt ID.
  Original metrics are archived before rollback. Modal retains each attempt's
  files in a separate directory instead of overwriting `last-attempt`.
  The historical W&B run remains unchanged: its 44 rows end at update 43,
  while its last published checkpoint is update 32.

The full-model normalization cause, exact-source GPU qualification,
and performance comparison remain open. No production recovery is qualified by these
CPU results.

## Validation log

Commands below used `UV_CACHE_DIR=/tmp/hal-uv-cache PYTHONPATH=.` and
`uv run --no-sync --project /home/ericgu/src/hal` in the production worktree.

- Baseline: `pytest -q tests/test_o65_replay_probe.py tests/test_o65_actor_trace.py
  tests/test_window_replay_actor.py -ra`: 31 passed, 6 CUDA skips, one NVML warning.
- New diagnostic semantic tests: `pytest -q tests/test_actor_diagnostic_targets.py -ra`:
  5 passed.
- Focused diagnostic/live-invariant/resume tests: `pytest -q
  tests/test_actor_diagnostic_targets.py tests/test_o65_replay_probe.py
  tests/test_o65_actor_trace.py tests/test_065_loop.py -k 'diagnostic or probe or trace
  or live_actor_head or nonfinite_actor or exact_next_batch' -ra`:
  38 passed, 3 CUDA skips, 68 deselected. This precedes optional failure capture.
- Focused Ruff checks passed after correcting import order and explicit loop-variable
  bindings. Type checks of the two diagnostic scripts and of the modified experiment
  passed with Python 3.14 and `--error-on-warning`.
- Attempt lineage and Modal archive tests: `pytest -q tests/test_o65_ddp_launch.py
  tests/test_o65_modal.py`: 50 passed. These include two successive failed attempts,
  exact preservation of the original metrics bytes, and distinct restored-checkpoint
  and previous-attempt identities.

Full required checks and GPU checks remain pending.
