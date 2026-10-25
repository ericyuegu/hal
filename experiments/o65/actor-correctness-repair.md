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
  Live checks also compare selected-action logs to full-head gathers in codec/decode
  order and require the replay execution mask to match the recorded factors.
  Normalization errors report the failed-row count and maximum error.
- Optional failure capture coordinates ranks, saves inputs, targets, output heads,
  model/optimizer state, gradients and RNG state, then terminates. It records a failed
  actor operation; it is not an exact-resume checkpoint.
- The precision report calls its finite-value result `finite`, not `passed`; its
  format is now v3. The redundant test for a removed gradient gate was deleted.
  Direction metrics have explicit numeric expectations. Historical v2 reports are
  unchanged. The unsupported BF16 learning conclusion was corrected.
- Removed the duplicated invalid-value scan before categorical KL. Masked padding
  remains excluded, and the categorical check rejects invalid active values.
  Removed the mocked-compiler test that only checked printed stage labels and
  stopped before consuming actor targets. The semantic target tests exercise the
  repaired boundary. No speed improvement is claimed without a GPU measurement.
- Retired the 600-line historical proxy qualifier and its import helpers. The
  adapter requested log-probability fields from a recorded control that stores
  probabilities, and the loader combined that control with current library code.
  Its source and recorded comparison remain identified in
  `archive/o65-dense-proxy-qualification.md`. Current per-decision/window gradient
  parity and actor objective tests remain supported; full-model qualification uses
  the captured failing continuation.
- Actor loss and KL locals now distinguish action logs, full-head logs, full-head
  probabilities, and the joint log ratio. Empty-rank target validation is explicitly
  documented as providing no numerical coverage.
- Probability conversion is named `head_probabilities` on decision, replay and
  evaluation outputs. Decoder lists distinguish `head_log_probs` from
  `action_log_probs`. The old ambiguous `.distributions` interface is removed;
  the historical tick control retains its recorded interface in its own checkout.
- The shared KL validator reports NaN and positive infinity before support
  differences. Removing the duplicate outer scan had exposed a misleading support
  error for these values. Focused tests now check the failing policy and category.
- Each continuation gets a new W&B run ID. Its lineage records the restored
  checkpoint hash, checkpoint producer, current source, and previous attempt ID.
  Original metrics are archived before rollback. Modal retains each attempt's
  files in a separate directory instead of overwriting `last-attempt`.
  The historical W&B run remains unchanged: its 44 rows end at update 43,
  while its last published checkpoint is update 32.

## Captured full-model failure

The two-B200 diagnostic captured the first actor operation after checkpoint 32
on original ranks 0 and 1, at source `fdadcd902feeaffa792e64ae7926f1a4a85382ff`.
It preserved selected checkpoint shard bytes, local simulation state, local shapes,
model state and rank RNG. Global moments and critic gradient averaging used two
ranks. It stopped before the actor optimizer step. This is a diagnostic projection,
not an exact eight-rank continuation.

Modal call `fc-01M5RDGTCMPEQM8AGG8AKGWW2G` completed with return code 0 because
the capture completed; both ranks' live actor checks failed. The actor snapshots
are in volume `hal-o65-profile`, under
`o65-correctness-fdadcd902fee-two-rank/failures/`:

- Rank 0 SHA-256: `8cb08f4fdfbf53d4d593aee00f848adee3516f8d8164ed9750ee6411704cd795`.
- Rank 1 SHA-256: `50a7dd1d2ee728bd79598d1030d2f32ad4e7d012354e3693fa447ec5a0d3d101`.

Both local downloads match their recorded hashes. Float64 reductions find 1,307
failed main-stick rows out of 64,390 on rank 0, and 1,328 out of 64,415 on rank 1.
The worst rows have probability mass `1.5193371585e-5` and `3.5470973536e-6`.
Independent Python `math.fsum(math.exp(value) ...)` agrees on these rows. This
establishes that large errors remain; the original first-failure log was not a
bound on the error. The other actor heads and all reference heads normalize,
and captured selected-action gathers and execution masks agree exactly.

An isolated one-B200 replay uses the same rank-0 snapshot, model state, RNG and
inputs at the same source. Modal call `fc-01M5RE268844V8ZEQ67D05F0MF` completed
all modes. Reports are under `o65-correctness-fdadcd902fee-actor-replay/` in the
same volume.

| Execution | Observed full-head result |
| --- | --- |
| Eager BF16 | All 9 chunks normalize at forward, after loss and after backward. |
| Compiled BF16 | 3,653 failed rows already at forward; later snapshots preserve the failure. Some rows contain NaNs. |
| Graphed BF16 | 1,289 failed rows after the unmodified grouped actor operation. |
| Eager FP32 | All 9 chunks normalize at forward, after loss and after backward. |

The compiled failure affects the main-stick head in the five chunks with capacity
64. The four chunks with capacity 32 pass. This locates the first observed failure
before loss construction and backward, without CUDA graph capture.
Cold-cache call `fc-01M5REGEZG81CDZD52VA6M8Y9N` rebuilt the same source without
importing the portable cache. Compiled BF16 still failed at forward: 5,022
main-stick rows failed normalization across the five capacity-64 chunks. The
capacity-32 chunks passed. Cache import is therefore not necessary for the failure.
The grouped graph mode passed in this cold-cache run, although the same mode failed
in the earlier warm-cache run. This single pass is not qualification of the path.
Copies used for intermediate observation can change timing; the grouped graph run
retains the normal actor operation. The next controls change decoder compiler
buffer reuse and reduction settings while preserving the captured inputs and model.

The root cause, repaired-source GPU qualification, eight-rank continuation through
checkpoint 64, gradient/update comparison and performance measurements remain open.
The BF16 learning-effect uncertainty remains separate from this normalization defect.

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
- Actor and diagnostic tests: `pytest -q tests/test_o65_replay_probe.py
  tests/test_actor_diagnostic_targets.py tests/test_window_replay_actor.py
  tests/test_065_loop.py tests/test_window_replay_model.py
  tests/test_o65_numerical_recovery.py -k 'not worker_failure' -ra`:
  149 passed, 10 CUDA skips, 2 deselected, one NVML warning. This command used
  `TMPDIR=/tmp OMP_NUM_THREADS=2 MKL_NUM_THREADS=2`.
- Required Dolphin integration command, with `HAL_REQUIRE_INTEGRATION=1`:
  initially 1 failed, 7 passed because `data/raw/dev.7z` was absent. After installing
  both fixtures and checking their recorded hashes, 8 passed, 6 deselected,
  7 multiprocessing fork warnings. No integration case was skipped.
- Preliminary full CPU suite: 34 failed, 2780 passed, 187 skipped, 90 deselected,
  3 errors, 24 warnings. Socket restrictions caused most failures; source edits
  overlapped this run and caused two stale-import failures. This is not final-source
  validation. A frozen-source rerun with local socket access is required.
- The frozen `2acf7724` rerun had local CUDA and socket access. It reached
  2006 passed and 2 skipped, then was interrupted after the `/tmp` user quota
  caused 49 failures and 3 errors (38 warnings, 90 deselected). Its first failure
  was a compiler file write; later checkpoint and subprocess files were also
  affected. The test artifacts and log are preserved. This is incomplete validation.
  Subsequent test temporary files and a copy of the compiler cache use the workspace
  filesystem instead of the quota-limited temporary mount.
- Shared KL regression command: `pytest -q tests/test_o65_memory.py
  tests/test_o65_numerical_recovery.py -k 'kl or normalization' -ra`:
  20 passed, 2 CUDA skips, 6 deselected.
- Probability-interface and KL tests across the self-play model, memory, dense
  replay, training batches, historical tick parity, proxy checkpoint, anchor
  stacking and diagnostic targets: 150 passed, 46 CUDA skips, 1 distributed-worker
  test deselected. These checks ran on CPU; full CUDA checks remain separate.
- After retiring the proxy qualifier, collection succeeds with `TMPDIR` unset:
  3018 tests collected, 89 integration tests deselected. The removed helper had
  required that environment variable while collecting an otherwise deselected test.

Full required checks and GPU checks remain pending.
