# O65 training throughput, round 2

## Summary

Compact reference replay, compiled BF16 trunks, and all-decision actor training
are implemented. **No new full O64 update completed on the B200.** Throughput,
MFU, O64 peak memory, the actor trunk/decoder time split, and the model-gradient
diagnosis remain unqualified. No long training run was launched.

The control remains O64 checkpoint
`e6d6303fbf25993c768b1e86fecbb350433a2fb229235b556334e9eb6ae868cf`, seed 65,
1,024 matches and 512 rollout frames. The sampler SHA-256 is
`1605d4c7be39cf77115a28d9ea559e041cff4f491ca4b9155c08a94c655e7daf`.
The benchmark attempts retained 64 optimizer windows and 16 sampled decisions,
so they isolate the first milestone. The 512-window config default remains
supported through chunks of at most 64 windows.

The approved scientific change is `decisions_per_window="all"`: 64 valid
choices per complete replan-2 window and 32 per replan-4 window. Partial windows
use only executed decisions. Integers 1..16 retain sampled mode. Production
records `"all"`; the older pilot, smoke and profile launchers explicitly retain
16. Window placement and optimizer/scheduler step counts are preserved.

Both actor and reference group by fixed delay/replan, cut queue padding to
`delay`, and replay 32 or 64 slots in all-decision mode. The decoder still plans
four offsets. Every valid reference decision contributes reward KL. Actor chunks
share the optimizer step's advantage statistics and denominator; gradients
accumulate before one optimizer step. Partial groups pad windows.

The reference retains its read-only autograd copy. Trunks use BF16 autocast;
observation encoding stays eager. Bootstrap value inference and the FP32 critic
keep their previous paths. The critic trains on recorded rollout features.

## Commits

Round-2 runtime: `4e578dc6`. The branch starts after `9d4eecb4`.

- `4d595e6a` Compact reference replay and compile training trunks
- `2b991817` Diagnose replay precision and trace actor work
- `605ff699` Allow a sandbox-local compiler cache
- `b7645526` Share memory across serial reference graphs
- `f15af4d4` Isolate the replay trunk backward diagnostic
- `7f51c758` Bound compiler workers and quote sandbox paths
- `4e578dc6` Train every decision in compact cadence groups

The final evidence commit updates this report. The former rejected patch was
removed when trunk compilation was adopted. Structured results record the
exact committed trainer SHA-256.

## Measurements

These are **historical round-1** full-update B200 measurements, not results for
round-2 grouping or all-decision training:

| Phase | Control seconds / MFU | Earlier compiled-trunk seconds / MFU |
|---|---:|---:|
| Preparation | 6.302 / 20.6% | 5.623 / 23.1% |
| Actor | 10.919 / 13.3% | 8.813 / 16.4% |
| Critic | 0.948 / 2.5% | 0.949 / 2.5% |

Control: `~/data/scratch/o65-devbox/runs/bench-20261022-230834-m64-full/`.
Earlier compilation: `~/data/scratch/o65-devbox/runs/bench-20261022-233947-m64-full/`.

Round-2 attempts, in order:

1. `/cache/runs/bench-20261023-121529-m64-full`: compilation exhausted the
   shared cache Volume's 500,000 inodes. No report was written.
2. `/tmp/o65-train-cache/runs/bench-20261023-123611-m64-full`: reference
   capture hit CUDA OOM. PyTorch reported **177.16 GiB allocated**, including
   **113.83 GiB in private pools**, at the error. These are error-time values,
   not actor-phase peaks. No report was written.
3. The shared-pool retry lost sandbox `sb-JJlEtZWm7qGtS5ThWrBDi5` during
   startup capture. Modal reported exit **137** and refused further exec calls.
   No timing report or trace was recovered. Host OOM is not confirmed: cgroup
   peak/events and an attributed system error were unavailable.

Logs are `~/data/scratch/o65-train-r2-reference-{bench,retry,pool}.log`.
The shared App is `ap-8Mio3cBZiOiQFKUKg0xEzC`. The sandbox remained terminated
on subsequent polls. I did not run `up`, `down`, or create another machine.

The helper now accepts `--cache-root /tmp/o65-train-cache` and
`--compile-threads 2`. The warm cache was linked onto local sandbox disk without
removing shared artifacts. These paths and compiler workers affect preparation;
training settings are still explicit. The local cache disappeared with the
sandbox and must be prepared again on its replacement.

New actor seconds, phase MFU, optimizer steps per measured update, allocated and
reserved peaks, and trunk/decoder trace times are **not available**. The 40% MFU
target has not been qualified.

## Precision diagnosis

The retained round-1 input artifact is
`~/data/scratch/o65-devbox/runs/bench-20261022-235147-m64-full/training-inputs.pt`,
SHA-256 `b52e7dd1ba8f6d5257f07c2971d1b24bb6a3e07a2f0974db7753d76a2cdc62e4`.

Its three cohorts each contain 1,024 selected decisions. Their advantage means
are -0.416, 0.493 and -1.799; standard deviations are 9.776, 9.920 and 12.674.
No selected advantage is zero. Thus a vanishing normalization scale is not the
cause of the reported 0.58–0.82 BF16/FP32 gradient cosine.

A CPU check used those exact advantages at ratio 1 and KL 0. Centered scalar
losses were about 1e-9, but the upstream gradient norm was 0.03125. Changing only
the final sum/divide to FP64 gave **relative L2 0** for all three cohorts and
cosine 1 to rounding. Scalar loss cancellation does not erase the gradient, and
the final reduction does not explain the poor cosine. This check does not
identify errors inside the model or cancellation among parameter gradients.
Its exact script, environment and hashes are retained in
[the structured results](training-throughput-round2.json).

The model-gradient cause remains unresolved. `scripts/o65_replay_probe.py` now
supports policy-only, KL-only, unit/uncentered advantages, ratio-1, FP64 scalar
reduction, and a shared FP32 gradient at the trunk output. It reports per-layer
errors, log-probs, KL and values, comparing eager and compiled BF16 with FP32.
Numerical errors remain diagnostics; nonfinite quantities and invalid inputs
fail. There is no old precision acceptance gate or new precision contract.

Training-probe v2 stores actual compact chunks from the first optimizer step,
with its complete mean, sigma and denominator. The loader explicitly supports
retained v1 whole-step artifacts. `scripts/o65_actor_trace.py` writes a bounded
trace separating observation encoding, trunk and decoder forward/backward.
It supports eager and compiled trunk controls. Both model diagnostics still
need the replacement B200 and real O64 inputs.

## Tests run

Required format, lint and type checks pass:

```sh
uv run ruff format --check .
uv run ruff check .
uv run ty check --python-version 3.14 --error-on-warning \
  hal experiments/059_muon_action_sequence.py scripts
```

Focused checks cover all 11 reference configurations, partial reference slices,
all-decision selection, unchanged placement and optimizer steps, microbatch
update parity, next-batch/optimizer resume, compact probe inputs and full-step
statistics, masks, input immutability, empty local chunks, shared pools and
shared actor gradients. Local CUDA: **11 passed**, 52 deselected, on RTX 3060.
B200 graph-wrapper checks before termination: **8 passed**.

The required full suite completed under `nice` and the shared heavy-test lock:

```sh
flock "$HOME/data/scratch/o65-heavy-tests.lock" env \
  PYTHONPATH="$PWD" TMPDIR="$HOME/data/scratch/o65-tmp-train" \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 nice uv run pytest -q -m 'not integration' \
  --basetemp="$HOME/data/scratch/o65-tmp-train/all" -rs
```

**2,785 passed, 39 skipped, 3 failed, 90 deselected**, in 650.24 s.
Two failures were `test_065_dense.py`'s `[3]` and `[16]` reference comparisons:
they required distributions for invalid slots, which the approved compaction
omits. The comparisons now cover all four planned frames of every valid
slot; full log-probs, masks and frozen-weight checks remain. The third is the
known `test_control_timing_recovery.py` R2-credential failure.

After the test-only correction, the focused command below passed **40 tests**
with four CUDA cases deselected. The complete suite was not rerun after that
assertion change. Validation remains incomplete because required fixtures are
missing. A separate CUDA graph pool identity check passed.

```sh
CUDA_VISIBLE_DEVICES= PYTHONPATH="$PWD" TMPDIR="$HOME/data/scratch/o65-tmp-train" \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice uv run pytest -q \
  tests/test_065_dense.py::test_reference_cohorts_preserve_slots_and_frozen_parameters \
  tests/test_o65_training_batching.py tests/test_o65_training_probe.py \
  tests/test_o65_replay_probe.py tests/test_o65_actor_trace.py \
  tests/test_o65_devbox_batching.py -k 'not cuda' \
  --basetemp="$HOME/data/scratch/o65-tmp-train/r2" -rs
```

Logs: `~/data/scratch/o65-train-r2-full-pytest.log` and
`~/data/scratch/o65-train-r2-final-focused.log`. An earlier partial full suite
was stopped with exit 143 while applying review fixes. Its log is
`~/data/scratch/o65-train-r2-full-aborted.log`; it is not a completed validation.

All 39 full-suite skips:

| Tests and lines | Count | Missing fixture or opt-in |
|---|---:|---|
| `test_archive_streaming.py`: 84, 93, 114, 125, 134, 167, 260 | 7 | `data/raw/dev.7z` |
| `test_behavior.py`: 496 | 1 | `data/raw/dev.7z` |
| `test_behavior_rows.py`: 215, 251, 272 | 3 | `data/raw/dev.7z` |
| `test_extract.py`: 154, 174, 194, 218, 235, 262, 272, 282, 340, 351 | 10 | `data/raw/dev.7z` |
| `test_replay_stats.py`: 162 | 1 | `data/raw/dev.7z` |
| `test_policy_schema.py`: 188, 211, 227, 237, 246 | 5 | local dev MDS |
| `test_policy_world_schema.py`: 50 | 6 | local v7 subset |
| `test_o65_proxy_checkpoint.py`: 28 | 2 | `HAL_O65_PROXY_CHECKPOINT` |
| `test_netplay_hardware.py`: 52, 68 | 2 | production GPU hardware opt-in |
| `test_o65_fixed_layout.py`: 284 | 1 | `HAL_O65_FIXED_GPU=1` |
| `test_o65_reward.py`: 84 | 1 | `HAL_O65_REWARD_GPU=1` |

Required integration command:

```sh
HAL_REQUIRE_INTEGRATION=1 TMPDIR=~/data/scratch/o65-tmp-train \
  nice uv run pytest -q tests/test_roundtrip.py tests/test_session_cleanup.py \
  -m integration --basetemp="$HOME/data/scratch/o65-tmp-train/it" -ra
```

**8 failed, 6 deselected**: all eight require the missing
`data/emulator/ssbm.ciso`. They are controller wire, fresh roundtrip,
instant restart, independent restarts, analog sweep, stick vocabulary,
normal-exit cleanup, and SIGKILL-parent cleanup.
Log: `~/data/scratch/o65-train-r2-integration.log`.

Resolved test failures: incorrect positional `argsort` arguments with
`stable=True`; a fixture using `engine.shards` instead of `engine.groups`;
script imports during focused collection without `PYTHONPATH`; and a CUDA
control forward outside the declared compiler preparation budget. Their logs
remain under `~/data/scratch/o65-train-r2-*.log`.

## Self-review findings and fixes

The entire cumulative diff is reviewed for masks, offsets, selected decisions,
gradient scaling, graph storage and streams, configuration identity, and resume.
Opus reviewed both milestones. The actual capture OOM led to serial graph pool
sharing. The actor shapes now share one FP32 gradient tuple rather than keeping
11 copies. Outputs are copied before another graph reuses its pool. Input
capture now matches training's actual compact shapes and complete-step loss
statistics. Profile decision work counts selections rather than capacity.
The all-invalid reference path and empty local actor path are covered.
A final Opus follow-up found no remaining code defect. It checked shared
gradient identity, pool lifetime, startup keys, probe v1/v2 statistics and the
reference diagnostic's 16-slot decoder shape. Its report is
`~/data/scratch/o65-train-r2-review-fixes-retry.txt`.

## Cross-stream changes

Optional pool/stream ownership in `hal/inference/cuda_graph.py`; optional shared
gradient storage and pool forwarding in `hal/inference/graphed_actor.py`;
benchmark cache/compiler controls; explicit all-decision production launch
settings; explicit sampled settings in the older pilot/smoke/profile launchers.
No tick, attention, simulator, wire, reward, or overlap loss implementation was
changed.

## Impasses

The assigned B200 sandbox exited 137. The brief prohibits me from replacing it.
The shared cache Volume has exhausted its inode limit. The required emulator
ISO and the known R2 credentials are missing locally. Missing fixtures remain
validation failures, not clean passes.

## Remaining work

Restore `train`; rerun each milestone separately with matched optimizer windows
and 64-window chunks; capture real O64 v2 inputs; run the precision controls and
actor split trace; report before/after phase time, MFU, memory and optimizer
steps; complete missing-fixture validation. Until then, these changes are not
qualified for the planned long training run.
