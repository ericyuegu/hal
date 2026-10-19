# O65 BF16 decoder

## Scope and method

Treatment selects `--decoder-precision bf16`; control selects `fp32`. Both
retain FP32 parameters, probability normalization, sampling thresholds,
log-probability accumulation, and critic heads. The BF16 setting uses BF16
linear projections and attention with FP32 residual sums, conditioning scales,
readout bias addition, and norms.

Invariant inputs are the pinned 17M-parameter O63 bootstrap30 checkpoint,
feature statistics, sampler, seed 65, MSL data and library, native adapter,
reward, optimizer, and development cadences `(8, 4)` and `(10, 2)`. Executed
offset validation is unchanged. Timing trajectories can diverge when precision
changes categorical samples; the numerical check instead uses common inputs
and recorded actions.

All GPU jobs hold `/home/ericgu/data/scratch/o65-tmp/gpu.lock`, run in a user
systemd scope with `MemoryMax=10G`, and set `OMP_NUM_THREADS=2`,
`MKL_NUM_THREADS=2`, `TORCHINDUCTOR_COMPILE_THREADS=1`, and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. Each process has at most 64 resident matches.
Jobs run serially on the local RTX 3060 12 GB. No remote machine is used.

Inside that guard, timing uses:

```sh
uv run python tests/test_065_columnar.py --matches 64 \
  --output runs/bf16-decoder/timing-64-off-bf16 \
  --performance --decoder-precision bf16
```

Repeat with 16 matches, FP32, and `--warmup` for the other seven arms. Profile
uses `uv run python experiments/065_selfplay_ppo.py profile` with the pinned
checkpoint and sampler, local MSL data, `--cpus 0 1 2 3`,
`--matches-per-shard 16 --groups 1 --warm-updates 4`, and the precision flag.
The stored resolved configurations provide every argument and input hash.

## Precision and artifact contract

`SelfPlayTransformer.decoder_precision` is validated at construction and has
no public setter. Rollout sampling, both no-grad policies, actor replay,
serial and batched evaluation, and direct live action-sequence sampling use
the setting. CUDA graphs cast weights during execution, with autocast weight
caching disabled, so an actor update cannot leave a stale BF16 weight copy.

`resolved.json`, the training configuration hash, the evaluation actor, and
the policy hash bind precision. Evaluation format v2 requires the field and
rejects old v1 actors and conflicting overrides. Resume rejects both a changed
configuration and inconsistent precision in the published actor before model
mutation. Frozen anchors and snapshots retain the setting.

The disabled-autocast blocks in `SelfPlayTransformer.self_play_value` and
`ActionSequenceTransformer.estimate_value` remain: they own critic/value-head
computation. Decoder softmax, CDF sampling, log-softmax, and factor sums retain
their existing FP32 arithmetic. The non-self-play BC decoder keeps its original
arithmetic. The columnar golden fixture and independent BC/evaluation parity
controls explicitly select FP32.

## Initial direct-BF16 trial

The first trial cast decoder inputs to BF16 and accumulated residuals in BF16. It failed the
unchanged vectorize-style numerical limits. Its common-input results were:

| Metric | Maximum | p99 |
| --- | ---: | ---: |
| Training factor log-probability error, nats | 0.204725 | 0.035517 |
| Cached rollout factor log-probability error, nats | 0.176307 | 0.035290 |
| Executed decision KL error, nats | 0.007171 | 0.003182 |

Actor gradient cosine was 0.998423 and relative difference norm was 0.060652.
Critic cosine was 0.999319 and relative difference norm was 0.050112. The
limits are factor max/p99 below 0.2/0.06 nats, decision KL below 0.02/0.005
nats, gradient cosine above 0.999, and relative difference norm below 0.03.
The factor maximum and both gradient norm checks failed; actor cosine also
failed. No limit was relaxed.

All eight three-update timing arms completed for that trial. Mean updates
2 and 3 at 64 matches took 10.090 s with FP32 and 9.418 s with BF16 during
PPO; warmup took 7.281 s and 6.443 s. These are rejected-trial measurements.
The revised implementation keeps residual accumulation and normalization in
FP32 while retaining BF16 projections and attention.

FP32 residuals alone reduced training factor max/p99 error to
0.096456/0.031351 nats. Actor cosine/relative difference norm were
0.999213/0.041445; critic values were 0.999385/0.047291. Both norm limits still
failed. That focused run was interrupted after the known numerical failure
and 62 passes while compiling another test. The final change keeps conditioner
scales, readout bias addition, and the sum of temporal/trunk logits in FP32.
Readout matrix products remain BF16. A focused CPU/CUDA regression test proves
that a 0.001 bias change survives the readout.

## Common-input numerical qualification

The diagnostic uses one 16-match rollout, both cadences, fixed anchor lanes,
and common weights, observations, queues, actions, and minibatch placement.
Each temporal head's final bias receives `linspace(-0.1, 0.1)` in both arms;
the frozen references retain the proxy weights. This creates a nonzero KL and
tests its effect on GAE and critic targets. Timing runs do not use this bias
perturbation. Each precision recomputes its own KL, advantages, and targets.
Actor gradients sum all minibatches at common weights without an optimizer
step. The cached rollout comparison teacher-forces the common actions through
the sequential sampling kernels; the training comparison uses parallel replay.

| Error, BF16 versus FP32 | Maximum | p99 | Limit, max / p99 |
| --- | ---: | ---: | ---: |
| Training factor log-probability, nats | 0.074996 | 0.017076 | 0.2 / 0.06 |
| Cached rollout factor log-probability, nats | 0.072083 | 0.015030 | 0.2 / 0.06 |
| Executed decision KL, nats | 0.00007475 | 0.00004086 | 0.02 / 0.005 |
| Critic targets, reward units | 0.008518 | 0.002925 | Diagnostic |

The FP32 decision KL averages 0.00080964 nats. Values match exactly at the
zero-initialized residual critic. The critic gradient comparison includes
both changed decoder features and the recomputed targets.

| Gradient | Cosine | Relative difference norm | Norm ratio, BF16 / FP32 |
| --- | ---: | ---: | ---: |
| Actor | 0.999763 | 0.022236 | 1.004183 |
| Critic | 1.000000 | 0.00004952 | 0.999970 |

Both pass cosine >0.999 and relative difference norm <0.03. No tolerance was
changed. The final focused run also checks the historical FP32 vectorization
control and live precision selection: 6 passed, 58 deselected, no skips.
The earlier common-input run gave actor cosine 0.999766 and relative norm
0.022067; the record retains both default-algorithm runs.

A separate BF16 rollout-layout comparison gave actor cosine 0.999267 and
relative difference norm 0.038381. It exceeds the historical FP32
vectorization refactor's 0.03 limit. That test now explicitly preserves its
FP32 control; the new common-input test measures the precision treatment.
This does not establish BF16 parity across batch layouts. Trials with FP32
readout projections, FP32 heads, and alternative sequential replay layouts
gave relative differences from 0.032451 to 0.048323. They did not resolve this
limit and were reverted. The measurement record retains these failed trials.

The signed gap below is `log(mu / pi_old)` summed over executed factors.
Quantiles and maximum use its absolute value. No importance weight exceeds
the cap of 2 in either arm. The unchanged student scoring pass remains needed.

| Decoder | Signed mean | Standard deviation | p50 | p95 | p99 | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 | 0.000198 | 0.010993 | 0.001638 | 0.024868 | 0.044047 | 0.113456 |
| BF16 | 0.000424 | 0.015853 | 0.002357 | 0.036950 | 0.061112 | 0.133954 |

By cadence, FP32/BF16 p99 is 0.049837/0.066253 at `(8, 4)` and
0.040824/0.057333 at `(10, 2)`. These cached common-input diagnostics are
separate from the complete-update default tick-graph metrics in the raw timing
records. This is a numerical and systems qualification, not evidence of a
gameplay improvement.

## Complete-update timing

Each arm runs three updates on four native shards, one active group, and
256 frames per match per update. PPO has critic warmup off; WARMUP keeps it
on for all three updates and skips the reference and actor passes. The table
averages updates 2 and 3. Throughput is their total match-frames divided by
their total update seconds. GPU peaks cover all three updates, including graph
preparation. Raw records retain every update, allocated and reserved peaks,
and the first update's timing.

The measured runtime is `a71c1969`. After the additional trials were reverted,
all eight runtime source files match that commit byte for byte; the measurement
record contains their SHA-256 hashes. Later changes make historical test
controls explicit and add evidence. Timing and profiles therefore use the
final runtime arithmetic.

| Matches | Phase | Decoder | Rollout s | No-grad s | Actor s | Critic s | Update s | Match-frames/s | GPU GiB, allocated / reserved |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | PPO | FP32 | 3.743 | 0.645 | 0.521 | 0.017 | 4.970 | 824.1 | 1.734 / 2.098 |
| 16 | PPO | BF16 | 3.750 | 0.702 | 0.502 | 0.018 | 5.018 | 816.3 | 1.497 / 1.648 |
| 16 | WARMUP | FP32 | 3.788 | 0.357 | 0.000 | 0.017 | 4.206 | 973.9 | 0.710 / 1.006 |
| 16 | WARMUP | BF16 | 3.740 | 0.384 | 0.000 | 0.017 | 4.186 | 978.4 | 0.641 / 0.873 |
| 64 | PPO | FP32 | 5.723 | 2.404 | 1.910 | 0.045 | 10.203 | 1605.8 | 3.274 / 3.934 |
| 64 | PPO | BF16 | 4.965 | 2.564 | 1.755 | 0.045 | 9.450 | 1733.8 | 2.839 / 3.096 |
| 64 | WARMUP | FP32 | 5.727 | 1.273 | 0.000 | 0.045 | 7.164 | 2287.1 | 1.308 / 2.201 |
| 64 | WARMUP | BF16 | 4.938 | 1.359 | 0.000 | 0.045 | 6.462 | 2535.6 | 1.269 / 1.885 |

At 64 matches, BF16 reduces complete PPO time by 7.38% and warmup time by
9.80%. PPO reserved memory falls by 21.3%. At 16 matches, PPO takes 0.96%
longer; warmup takes 0.46% less time. The no-grad pass is slower with BF16 at
both sizes. These are single three-update runs per arm, without a confidence
interval or a production-GPU projection.

First-update FP32/BF16 times are 6.916/7.122 s for 16-match PPO,
6.130/6.178 s for 16-match warmup, 12.129/11.535 s for 64-match PPO, and
9.146/8.586 s for 64-match warmup. Graph preparation accounts for the larger
first-update cost. Phase timers do not cover every diagnostic; update time
includes them. No timed update has an episode end or idle frame.

## Bounded 64-match GPU profile

Both arms use the maintained `profile` subcommand, four initial PPO updates,
then a clean timed update and a separate traced update. Each has one group and
four shards of 16 matches: 64 resident matches. Profiling does not launch
evaluation or uploads. Source is `a71c1969`; complete identities, per-site
estimates, and raw-file hashes are in
[the measurement record](bf16-decoder-measurements.json). Raw traces remain in
`runs/bf16-decoder/profile-64-{fp32,bf16}/`.

The decoder-only rows below are kernel-duration sums from sampled no-grad
decoder chunks, scaled by the timed update's work. They exclude trunk chunks.
They are estimates from bounded samples, not whole-phase captures.

| Profile quantity | FP32 s | BF16 s |
| --- | ---: | ---: |
| Student decoder kernels, estimated | 0.697249 | 0.422229 |
| Reference decoder kernels, estimated | 0.696698 | 0.422837 |
| Bootstrap decoder kernels, estimated | 0.014534 | 0.008914 |
| All rollout kernels, estimated, including trunk | 4.518938 | 3.789924 |
| FP32 memory-efficient attention in sampled rollout ticks, estimated | 1.329212 | 0 |
| Clean rollout wall time | 5.739257 | 4.985864 |
| Clean complete update wall time | 10.118765 | 9.476547 |

Student decoder kernel time falls by 39.4%. Its estimated runtime launches
rise from 46,137 to 56,693, and kernel executions rise from 46,137 to 58,786.
The added casts and pointwise work offset some device gains. The inclusive
student no-grad wall time rises from 1.089 to 1.178 s; reference rises from
1.094 to 1.176 s. Kernel sums therefore do not establish a no-grad wall-time
gain. The BF16 rollout's leading attention kernels are BF16 FlashAttention and
BF16 memory-efficient attention. Their combined time includes trunk work and
is not reported as decoder-only time.

Timed and traced updates are consecutive policy versions. Trace export and
profiler overhead are excluded from the clean times. CUDA event spans include
host gaps and are not reported as GPU busy time. No production GPU, large-pool
memory fit, learning result, or deployment latency was measured.

## Validation and review

The required checks use the guard and environment described above. Logs are
under `runs/bf16-decoder/`; pytest basetemps are under `$TMPDIR`. The type check
adds the O65 trainer, pilot, new numerical test, and modified test modules to
the maintained HAL/059/scripts targets.

| Command or check | Result |
| --- | --- |
| `uv run ruff format --check .` | Pass, 381 files. |
| `uv run ruff check .` | Pass. |
| `uv run ty check --python-version 3.14 --error-on-warning hal experiments/059_muon_action_sequence.py scripts experiments/065_selfplay_ppo.py experiments/o65/pilot.py tests/test_065_decoder_precision.py tests/test_065_columnar.py tests/test_065_loop.py tests/test_065_vectorize.py tests/test_action_sequence_policy_batch.py tests/test_batched_self_play_integration.py tests/test_o65_evaluation.py tests/test_o65_proxy_checkpoint.py tests/test_self_play_model.py` | Pass, zero diagnostics. |
| `pytest -q -ra tests/test_self_play_model.py tests/test_o65_evaluation.py tests/test_o65_gpu_profile.py` | Initial focused run: 77 passed, one warning, no skips. |
| `pytest -q -ra -s tests/test_065_decoder_precision.py` | First native attempt: 2 passed, 1 failed because MSL was absent from `PYTHONPATH`; corrected before the numerical trials. |
| Same command with the MSL checkout on `PYTHONPATH` | Direct BF16 trial: 3 passed, 1 failed the numerical bounds. |
| `pytest -q -ra -s tests/test_065_decoder_precision.py tests/test_self_play_model.py tests/test_action_sequence_policy_batch.py` | Residual-only trial: 1 failed, 62 passed, 16 warnings; interrupted after the numerical failure while compiling another test. |
| `pytest -q -ra -x -s tests/test_065_decoder_precision.py tests/test_self_play_model.py -k 'decoder_precision_on_common_inputs or small_bias_changes or owns_compute_precision'` | Final numerical check: 7 passed, 33 deselected, one warning, no skips. |
| `uv run pytest -q -ra -m "not integration" --basetemp "$TMPDIR/handoff-unit-final"` | First full run: 7 failed, 2,557 passed, 8 skipped, 90 deselected, 27 warnings. Three old CPU parity cases needed explicit FP32; four new live tests supplied a return outside u95. Test inputs were corrected; tolerances were unchanged. |
| Same full suite with `--basetemp "$TMPDIR/handoff-unit-final-qualified"` | 8 failed, 2,556 passed, 8 skipped, 90 deselected, 29 warnings. All eight failures were socket-path failures from the longer basetemp. |
| `uv run pytest -q -ra -m "not integration" --basetemp "$TMPDIR/u"` | Final run: 2,564 passed, 8 skipped, 90 deselected, 26 warnings, 328.17 s. The skips are detailed below. No test failed. |
| `HAL_REQUIRE_INTEGRATION=1 uv run pytest -q -ra tests/test_roundtrip.py tests/test_session_cleanup.py -m integration --basetemp "$TMPDIR/required-integration"` | 8 passed, 6 deselected, 7 warnings, no skips. |
| `HAL_REQUIRE_INTEGRATION=1 HAL_O65_COMPILE_CUDA=1 uv run --with zstandard==0.25.0 pytest -q -ra -s tests/test_065_loop.py tests/test_065_columnar.py tests/test_065_vectorize.py tests/test_native_sim_integration.py tests/test_batched_self_play_integration.py -m integration --basetemp "$TMPDIR/native-integration"` | 23 passed, 1 failed, 129 deselected, no skips. The failure was the BF16 layout check above. Golden fixtures at 16/64, native parity, tiny native exact resume, and FP32 serial/batched evaluation passed. |
| `pytest -q -ra -x -s tests/test_065_decoder_precision.py tests/test_065_vectorize.py tests/test_action_sequence_policy_batch.py -k 'proxy_decoder_precision_on_common_inputs or proxy_tick_numerics_and_update_gradients or widened_live_policy_uses_saved_decoder_precision'` during four follow-up trials | Each trial: 1 passed, 1 failed, 58 deselected, no skips. Common-input precision passed; the BF16 layout gate failed. Logs: `readout-projection-fp32.log`, `heads-fp32.log`, `sequential-reference.log`, `input-fp32.log`. All four treatments were reverted. |
| `HAL_REQUIRE_INTEGRATION=1 uv run pytest -q -ra -x -s tests/test_065_decoder_precision.py tests/test_065_vectorize.py tests/test_action_sequence_policy_batch.py -k 'proxy_decoder_precision_on_common_inputs or proxy_tick_numerics_and_update_gradients or widened_live_policy_uses_saved_decoder_precision' --basetemp "$TMPDIR/qualified-final"` | 6 passed, 58 deselected, no skips. This reruns the numerical treatment, the explicit FP32 vectorization control, and both live precision modes on CPU/CUDA. |
| Socket-path probe with a long basetemp | 1 failed with `AF_UNIX path too long`. Using `$TMPDIR/s`, all 10 live-overlay tests pass. The full-suite basetemp was shortened to `$TMPDIR/u`; no product code changed. |
| `uv run python experiments/065_selfplay_ppo.py train --checkpoint runs/proxy/final.pt --sampler experiments/o65/pilot-sampler.json --output runs/bf16-decoder/resume-64 --data-dir /home/ericgu/src/melee-sim-light/data --cpus 0 1 2 3 --updates 4 --groups 4 --matches-per-shard 4 --compiled --max-frame 140 --training.critic-warmup-updates 0 --evaluation-every 0 --verify-resume` | Pass with default BF16, 64 resident matches and 16 active. Next-batch and next-optimizer hashes match exactly; maximum parameter error is 0. The next update includes 16 resets and 16 time-outs. |
| Sixteen timing arms, three updates each | All completed: eight rejected-trial arms and eight final arms. |
| `profile --cpus 0 1 2 3 --matches-per-shard 16 --groups 1 --warm-updates 4 --decoder-precision {fp32,bf16}` | Both completed with CUDA activities and all required trace sites. |
| Input identity and resident-match audit | Pass for all timing arms and both profiles. |

Initial static checks found two new typing errors and seven pre-existing test
annotations in a modified module. These were corrected; the final targets have
zero diagnostics. Queued first full-suite and profile commands were cancelled
before they started after the rejected direct-BF16 result.

The measurement record includes the final full-suite counts and log hash,
required integration and focused-run counts, and the native resume report and
resolved configuration. Resume restores the boundary optimizer hash exactly.
The resumed complete-update `mu_old_gap` p99 is 0.054015 nats, with no truncated
importance weights.

The review covers `git diff o65-selfplay-ppo-plan...HEAD`, the precision delta
from `8234dfd2`, and the plan's math and risks. It checks executed offsets,
episode and factor masks, critic detach points, conditional sampling support,
graph ownership, parameter updates, actor/reference precision, persisted
identities, and resume mode changes. The first pass found a direct live
sampling path outside `DecisionDecoder` and missing actor-precision
cross-validation during restore. Both are fixed and have focused tests.
Numerical trials also found the BF16 residual and readout-bias rounding issues
described above. The second review pass found no further implementation issue.

## Cross-stream changes

The trainer CLI, configuration identity, and profile metadata propagate
precision. Evaluation actor format v2, its loader, evaluation CLI/results, and
`hal/eval/O65.md` record it. Direct live sampling also needed the precision
boundary. Tests cover these paths. No MSL source, other worktree, remote
machine, or published branch was changed.

## Impasses and remaining qualification

The full suite skips six schema tests because the required
`data/processed/ranked-anonymized-1/mds-v7-sub4/train` fixture is absent. Under
AGENTS.md this is a validation failure, even if pytest exits successfully.
The read-only fixture symlink was not changed. Two further tests require
`HAL_REQUIRE_NETPLAY_HARDWARE_QUALIFICATION=1` on the production GPU and skip on
this machine. Both skip reasons must remain visible in the handoff.

BF16 batch-layout gradient sensitivity remains above the historical FP32
refactor bound. Production GPU performance, learning quality, large-pool
memory fit, and deployment latency remain unmeasured. These measurements do
not replace gameplay selection or the production hardware qualification.
