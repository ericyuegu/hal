# O65 training throughput, round 2 measurements

## Summary

Both milestones completed on one B200 at 1,024 matches and 512 rollout frames.
The sampled treatment reduces preparation from **6.30 to 4.17 s** and actor
training from **10.92 to 9.23 s** against the historical a0 control.
All-decision training takes **12.93 s** with the production 512-window optimizer
batch and graph chunks capped at 64 windows. It uses **15 optimizer steps** per
update; the matched 64-window setting uses 118 steps and takes 24.57 s.

The old sandbox died from **GPU preemption**, not a confirmed host OOM.
Serial reference and actor graphs now share one pool. The measured settings fit
the B200 and the existing 64 GiB host limit. Cache persistence passed a real
sandbox restart. **The 40% MFU target is not met. Trunk-gradient precision remains
unqualified.**

The checkpoint, sampler, simulator, data manifest, seed 65, reward weights
`(0.1, 120, 120)`, 11 cadences, and rollout length are fixed. All-decision
selection is the approved scientific treatment. The 512-window optimizer batch
is a separate batching treatment; its updates are not equivalent to 118 smaller
optimizer steps. [Structured evidence](training-throughput-round2-measurement.json)
records configurations, environments, provenance, artifact hashes, and diagnostic
results. The earlier [round-2 report](training-throughput-round2.md) is historical.

## Commits

The measured trainer is `e7d2fbd4`, source SHA-256
`89c86ada0fad1803b5fb9ccbf21fe4baabfc0a677e5b6a1ee4a0269f361b9701`.
Remote Git metadata names the base image, `a0ae12df`; the helper copies local
changes into that image. It does not describe the measured source by itself.

- `9c3fe53c` — use a genuinely eager decoder in the eager precision control.
- `8dac3d3c` — time stages inside actual actor graph replay.
- `4e6d335a` — trace the complete stored chunk and queue width.
- `e7d2fbd4` — merge actor shapes, share replay memory, and record startup peaks.
- `3b062bbf` — check outputs and gradients after changing shared-pool replay order.
- `3ac9832b` — match the precision probe's compiled boundary to training.
- `e17b7131` — limit the diagnostic compiler stub to its intended function.

The final evidence commit adds this report and its JSON record.

## Measurements

Untraced update timings; one measurement per setting:

| Setting | Preparation s / MFU | Actor s / MFU | Critic s | Rollout s | Update s | Actor steps |
|---|---:|---:|---:|---:|---:|---:|
| Historical a0; 16 decisions, 64 windows | 6.302 / 20.6% | 10.919 / 13.3% | 0.948 | 14.575 | 32.828 | 118 |
| Current; 16 decisions, 64 windows | 4.172 / 23.5% | 9.234 / 15.7% | 0.952 | 15.460 | 29.900 | 118 |
| Current; all decisions, 64 windows | 4.188 / 23.4% | 24.567 / 22.2% | 0.952 | 15.550 | 45.336 | 118 |
| Current; all decisions, 512 windows | 4.133 / 23.7% | 12.930 / 24.6% | 0.947 | 13.899 | 31.971 | 15 |

The sampled preparation and actor reductions are 33.8% and 15.4%. An earlier
compiled-trunk-only actor run reached **8.813 s**, faster than the current sampled
9.234 s. These single-run results do not establish a new sampled actor record.
Rollout timings also vary between Modal hosts.

Run directories under `~/data/scratch/o65-devbox/runs/`:

- a0 control: `bench-20261022-230834-m64-full/`.
- earlier compiled trunk: `bench-20261022-233947-m64-full/`.
- current sampled: `bench-20261023-190532-m64-full/`.
- current all, 64 windows: `bench-20261023-193233-m64-full/`.
- current all, 512 windows: `bench-20261023-194651-m64-full/`.

The sampled run trains 120,832 decisions per update; both all-decision runs train
370,304. Actor replay counts are 118 sampled graphs, 236 all-decision graphs at
64 optimizer windows, and 134 at 512 windows. The last count comprises 74
64-slot graphs and 60 32-slot graphs. All three runs report **zero post-startup
captures and compiles**.

Fine actor grouping padded each tested 64-window cohort to 704 windows across
11 cadences. Grouping by decision capacity reduces this to one sampled shape
or two all-decision shapes. Each slot retains its delay and replan; the actor
retains the full stored queue width. Reference replay still groups by cadence.

The BF16 model compute floors are 0.979 s for preparation, 1.449 s for sampled
actor training, 5.462 s for all-decision actor training at 64 windows, and
3.177 s at 512 windows. These counts include padded work and exclude memory,
optimizer, and host costs. The critic uses FP32; its reported 2.5% uses the
benchmark's BF16 peak denominator and is not an FP32 hardware-utilization claim.

Bench command, with `WINDOWS=64` or `512` and `DECISIONS=16` or `all`:

```sh
uv run python -m scripts.o65_devbox bench --name train --full \
  --rollout-frames 512 --matches-per-shard 64 \
  --windows-per-step "$WINDOWS" --decisions-per-window "$DECISIONS" \
  --compile-threads 2 --capture-training-inputs
```

The production benchmark child completed successfully. Its download helper
failed because I had already created the destination directory for restart
evidence. I downloaded the completed reports and traces directly and hashed
them. No timing was reconstructed or benchmark rerun used to hide that failure.

### Memory and startup

Modal's retained result for old sandbox `sb-JJlEtZWm7qGtS5ThWrBDi5` states
`Container terminated due to preemption`. Its poll code is 137. This resolves
the earlier uncertainty about host OOM. The result is retained at
`~/data/scratch/o65-train-r2-old-sandbox-result.json`.

Host memory was sampled every five seconds, including trainer and compiler
process RSS. Cgroup usage is the process-group total; forked RSS contains shared
pages and must not be added.

| Setting | Startup cgroup peak GiB | Timed cgroup peak GiB | Trainer RSS at startup peak GiB |
|---|---:|---:|---:|
| Sampled, 64 windows | 17.35 | 17.73 | 14.10 |
| All, 64 windows | 15.65 | 16.06 | 13.75 |
| All, 512 windows, restarted sandbox | 14.66 | 15.11 | 13.47 |

At the sampled startup peak, the compiler manager held 4.28 GiB RSS and its two
workers held 2.90 and 2.85 GiB. The highest observed host usage, including the
traced update, was 17.78 GiB. Cgroup historical peak counters and process PSS were
unavailable. There is no measured reason to raise the 64 GiB limit or add a host
memory option. Compiler workers remain bounded at two.

GPU values below are **allocated / reserved GiB**, for the complete trainer
during each phase. Startup rows are maxima across cadence captures. The PPO
phase rows describe the first warm update; the final row describes the later
untraced update. They are different measurement scopes.

| Phase | Sampled / 64 windows | All / 64 windows | All / 512 windows |
|---|---:|---:|---:|
| Startup rollout | 60.15 / 76.91 | 60.15 / 76.71 | 60.15 / 76.71 |
| Startup value | 57.77 / 98.65 | 57.77 / 98.46 | 57.77 / 98.46 |
| Startup reference | 68.26 / 138.64 | 68.37 / 150.67 | 68.37 / 150.67 |
| Startup actor | 61.42 / 133.78 | 80.50 / 172.57 | 80.50 / 172.52 |
| First PPO rollout | 47.85 / 121.32 | 47.96 / 130.43 | 47.96 / 130.45 |
| First PPO reference | 65.77 / 139.27 | 65.88 / 148.33 | 65.87 / 148.35 |
| First PPO preparation | 72.15 / 150.46 | 72.26 / 159.53 | 72.26 / 159.54 |
| First PPO critic | 84.95 / 161.80 | 85.06 / 170.87 | 85.06 / 170.88 |
| First PPO actor | 69.68 / 161.84 | 69.93 / 170.91 | 71.52 / 170.93 |
| Timed whole update | 85.89 / 163.25 | 86.00 / 172.32 | 86.00 / 172.34 |

Before pool sharing, reference capture failed with 177.16 GiB allocated,
including 113.83 GiB in private pools. The current all-decision startup completes
with reference peaks near 68.4 GiB allocated and actor peaks near 80.5 GiB.
The all-decision timed reserve peak leaves about 6.7 GiB below the reported
179.1 GiB device capacity. This qualifies the measured serial workload only.

Startup capture / total startup seconds:

| Cache state | Capture s | Total s |
|---|---:|---:|
| Sampled; partially populated cache | 1,360.707 | 1,375.661 |
| All, 64 windows; same sandbox | 549.697 | 564.273 |
| All, 512 windows; restarted sandbox with persisted cache | 1,189.777 | 1,209.661 |

The first empty-cache attempt, using the pre-fix runtime, was stopped after
about 886 s before startup completed. No complete cold-start latency is
available. It is not a timing baseline. The first completed sampled run was
already partly warm.
That pre-fix attempt is `bench-20261023-184547-m64-full`; I stopped its trainer
with SIGTERM (exit 143) after finding the 704-window padding. Its host samples
are retained in the run directory. It produced no throughput report.

The cache volume `hal-o65-devbox-v2` persisted across a real `down` / `up` from
`sb-u0EDvsvVq4p1qVuJ22NKGu` to `sb-5XjfT8O59AtP35WmPq7l7r`. All eight sampled
Inductor/Triton artifacts retained their hashes. The volume reported about
70,500 used inodes out of 131 million. No manual commit was needed for v2.
`TORCHINDUCTOR_CACHE_DIR=/cache/inductor` and `TRITON_CACHE_DIR=/cache/triton`
were set throughout.

Nine completed FX graph loads and nine autograd cache hits were recorded after
restart, without an FX graph miss at that sample. Cache loading remained slow:
large graph loads took roughly 70–80 s, including 45–50 s in Python code-cache
loading / async compiler waits; Dynamo tracing took roughly 10–16 s per large
variant. Rollout capture alone took 801.56 s after restart. Stack samples,
compiler timings, cache manifests, and host samples are retained with the run.
These observations prove reuse, not a fast restart. Both sandboxes are stopped.

### Actor graph split

CUDA events and gradient hooks time one actual graph backward. Values are the
mean of ten replays, in milliseconds. Each graph has 64 physical windows; all64
and all32 contain 31 and 33 live windows in the recorded optimizer step.

| Chunk | Trunk forward | Trunk backward | Decoder forward | Loss + decoder backward | Observation + other | Total |
|---|---:|---:|---:|---:|---:|---:|
| Sampled, eager trunk | 11.988 | 23.095 | 7.669 | 14.961 | 6.365 | 64.078 |
| Sampled, compiled trunk | 6.088 | 12.366 | 7.884 | 15.339 | 6.508 | 48.185 |
| All64, compiled trunk | 5.979 | 12.020 | 27.583 | 52.971 | 6.497 | 105.050 |
| All32, compiled trunk | 5.980 | 12.006 | 14.444 | 27.499 | 6.431 | 66.359 |

“Other” includes gradient clearing and observation backward. Hook boundaries can
place leaf accumulation in either backward span or in “other”. These are
isolated graphs with private pools and gradients; they exclude optimizer and
batching work. The separate diagnostic traces use split autograd calls for
kernel attribution and have different backward boundaries. They are not used
as whole-update timings.

Reports, actual replay traces, diagnostic traces, and commands are under
`~/data/scratch/o65-train-r2-diagnostics/actor-*`.

### Precision diagnostics

These use the real v2 chunks, complete-step advantage statistics and denominator,
and checkpoint `e6d6303fbf25993c768b1e86fecbb350433a2fb229235b556334e9eb6ae868cf`.
The eager mode now uses an eager decoder. Both BF16 modes create zero padding
inside the same history boundary as training. FP32 forces dense attention.
“Eager” describes the outer history and replay callables; PyTorch operators
retain their normal kernel implementations, including FlexAttention's JIT.

| Chunk | Eager–FP32 actor-gradient cosine | Compiled–FP32 cosine | Compiled–eager cosine | Eager / compiled relative L2 vs FP32 |
|---|---:|---:|---:|---:|
| Sampled | 0.8243 | 0.7756 | 0.7444 | 0.6131 / 0.7124 |
| All64, 64-window optimizer batch | 0.8211 | 0.8328 | 0.8613 | 0.5979 / 0.5759 |
| All32, 64-window optimizer batch | 0.7497 | 0.7276 | 0.8141 | 0.7090 / 0.7410 |
| All64, 512-window optimizer batch | 0.7068 | 0.6599 | 0.7163 | 0.7336 / 0.8209 |

Absolute output errors against FP32:

| Chunk | BF16 mode | Log-prob max / RMSE | Factor-KL max | Value max / RMSE |
|---|---|---:|---:|---:|
| sampled | eager | 0.1810 / 0.00708 | 0.000000 | 3.822 / 0.780 |
| sampled | compiled | 0.2238 / 0.00764 | 0.000000 | 3.610 / 0.763 |
| all64 | eager | 0.1431 / 0.00659 | 0.001819 | 5.332 / 0.771 |
| all64 | compiled | 0.1887 / 0.00648 | 0.001111 | 7.769 / 0.748 |
| all32 | eager | 0.2077 / 0.00741 | 0.001662 | 3.822 / 0.755 |
| all32 | compiled | 0.2279 / 0.00782 | 0.000488 | 3.840 / 0.767 |
| production-all64 | eager | 0.3583 / 0.00843 | 0.001996 | 5.600 / 0.793 |
| production-all64 | compiled | 0.4083 / 0.00875 | 0.001629 | 5.121 / 0.787 |

All results are finite. Eager repeats differ by only about 1e-7 relative L2 in
the actor gradient, so nondeterminism does not explain the discrepancy.
Decoder output-head gradient cosines against FP32 are about 0.997–1.000.
The first trunk block, observation encoder, and codec contain the large errors.

Giving both BF16 trunks the same FP32 upstream gradient does not remove the
error. Unit advantages and ratio 1 also do not remove it. A separate trunk-only
probe holds the BF16 input tokens and upstream gradient fixed. Against FP32,
trunk-parameter gradient cosine is **0.272 with varlen attention, 0.295 with
dense SDPA, and 0.179 with math SDPA**; hidden-state cosine stays above 0.99994.
This localizes the sensitivity to BF16 trunk computation. It does not identify
one faulty operator or establish a new precision acceptance threshold.

The final scalar reduction had already been excluded by the previous exact
FP64-reduction control. The current results also exclude decoder differences,
advantage centering, PPO ratio changes, and a FlashAttention-only explanation
as complete explanations. The isolated script is
`~/data/scratch/o65-train-r2-trunk-backward.py`; its hash and scope are recorded
in the structured evidence. Training precision was not changed.

## Tests run

Required commands:

```sh
uv run ruff format --check .
uv run ruff check .
uv run ty check --python-version 3.14 --error-on-warning \
  hal experiments/059_muon_action_sequence.py scripts
nice uv run pytest -q -m "not integration"
HAL_REQUIRE_INTEGRATION=1 nice uv run pytest -q \
  tests/test_roundtrip.py tests/test_session_cleanup.py -m integration
```

Format (419 files), lint, and type checks pass. Pytest uses the shared heavy-test
lock, `PYTHONPATH=$PWD`, `TMPDIR=~/data/scratch/o65-tmp-train`, two OMP/MKL/compiler
threads, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and `nice`.

Focused commands:

```sh
nice uv run pytest -q tests/test_o65_training_batching.py \
  tests/test_o65_training_probe.py tests/test_o65_devbox_batching.py \
  tests/test_cuda_graph_preparation.py tests/test_window_replay_actor.py
nice uv run pytest -q tests/test_o65_replay_probe.py tests/test_o65_actor_trace.py
CUDA_VISIBLE_DEVICES= nice uv run pytest -q \
  tests/test_o65_replay_probe.py::test_history_matches_the_training_boundary_and_input_gradient
```

- Focused batching, probe persistence, devbox, graph preparation, and actor
  tests: **84 passed**, 21 warnings, 441.32 s.
- Initial diagnostic tests: **10 passed**, 17 warnings, 36.07 s.
- New history-boundary CPU output and input-gradient parity: **1 passed**, 2.10 s.
- Updated diagnostic tests first found **1 failed, 10 passed**, 16 warnings,
  9.25 s. The history compiler stub also intercepted FlexAttention's internal
  seven-argument wrapper. After restricting the stub: **11 passed**, 16 warnings,
  9.27 s.
- First complete non-integration suite: **2 failed, 2,804 passed, 39 skipped,
  90 deselected**, 27 warnings, 707.29 s. Failures were the known R2-credential
  test and the diagnostic stub above.
- Complete non-integration suite: **1 failed, 2,805 passed, 39 skipped, 90 deselected**,
  27 warnings, 699.25 s, after the test-only stub fix. The sole failure is the
  known missing R2 credentials..
- Required integration: **8 failed, 6 deselected**, 2.02 s. Every failure is the
  missing `data/emulator/ssbm.ciso`: controller wire, fresh roundtrip, instant
  restart, independent restarts, analog sweep, stick vocabulary, normal-exit
  cleanup, and SIGKILL-parent cleanup. These are failures, not passing validation.

The environment failure is
`tests/test_control_timing_recovery.py::test_repeated_recovery_keeps_original_wave_seed_matchups_and_slots`:
missing `AWS_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, and `AWS_SECRET_ACCESS_KEY`.

All 39 skips, with paths under `tests/`:

| Tests and lines | Count | Missing fixture or opt-in |
|---|---:|---|
| `test_archive_streaming.py`: 84, 93, 114, 125, 134, 167, 260 | 7 | `data/raw/dev.7z` |
| `test_behavior.py`: 496 | 1 | `data/raw/dev.7z` |
| `test_behavior_rows.py`: 215, 251, 272 | 3 | `data/raw/dev.7z` |
| `test_extract.py`: 154, 174, 194, 218, 235, 262, 272, 282, 340, 351 | 10 | `data/raw/dev.7z` |
| `test_replay_stats.py`: 162 | 1 | `data/raw/dev.7z` |
| `test_policy_schema.py`: 188, 211, 227, 237, 246 | 5 | local dev MDS |
| `test_policy_world_schema.py`: 50 | 6 | local v7 subset |
| `test_o65_proxy_checkpoint.py`: 28 | 2 | `HAL_O65_PROXY_CHECKPOINT`, O63 bootstrap30 checkpoint |
| `test_netplay_hardware.py`: 52, 68 | 2 | production hardware opt-in |
| `test_o65_fixed_layout.py`: 284 | 1 | `HAL_O65_FIXED_GPU=1` |
| `test_o65_reward.py`: 84 | 1 | `HAL_O65_REWARD_GPU=1` |

Unavailable fixtures remain validation limits; these skips do not qualify them.

Logs are `~/data/scratch/o65-train-r2-{memory-final,diagnostic-boundary,handoff-full,handoff-integration}-tests.log`,
`~/data/scratch/o65-train-r2-{diagnostic-recheck,full-recheck,boundary-cpu}.log`.
All B200 replay precision
probes, actor traces, and the trunk isolation probe completed with exit 0.

## Self-review findings and fixes

The cumulative diff from `a0ae12df` was reviewed for masks, offsets, selected
decisions, queue padding, gradient normalization, optimizer/scheduler counts,
storage ownership, streams, and resume. Opus reviewed the measured runtime and
the history-boundary diagnostic change. Those reviews found no runtime
correctness defect. The full suite caught the later compiler-stub error; the
corrected stub passed all 11 focused diagnostic tests.

Findings fixed: the eager control previously used a compiled decoder; the actor
trace narrowed the stored chunk and queue; the precision probe created padding
outside the production compile boundary; full-weight update comparisons hid
small update errors. Tests now compare optimizer deltas, cover mixed delays,
check trailing queue padding, preserve OOM peak reports, and replay the shared
pool in a changed order with nonzero gradients. Outputs are copied or consumed
before another graph uses the shared pool. No tick-state writes were added.

Startup memory lines contain running maxima across cadences; repeated duration
fields contain the last invocation. The report uses them with that scope.
Remaining coverage limits are multi-rank execution and a qualified precision
contract for the full O64 trunk.

## Cross-stream changes

This pass adds startup/profile instrumentation and artifact recovery in
`experiments/065_selfplay_ppo.py` and `scripts/o65_devbox.py`. Cumulative shared
changes add optional pool/stream/gradient ownership to the graph wrappers and
explicit decision modes to production and older sampled launchers. No host RAM
setting was changed. No tick, attention, simulation, wire, or overlap loss
implementation was changed by this stream.

## Impasses

The emulator ISO and required R2 credentials are missing. A complete
empty-cache startup was not measured. BF16 trunk backward sensitivity needs
a precision decision or further diagnosis; I did not change the scientific
treatment to make its diagnostics pass.

An optional final Opus audit of this report hit the CLI session limit. The
earlier Opus code reviews completed. I checked the evidence tables against the
saved JSON reports.

## Remaining work

Resolve and qualify trunk backward precision; raise measured preparation and
actor MFU toward 40%; reduce costly cache loading at startup; supply the missing
fixtures and credentials and complete their validation. Multi-GPU and
2,048-match qualification remain with integration. No long training run is
qualified by these measurements.
