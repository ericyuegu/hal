# O65 training throughput

The control is `a0ae12df`, with the O64 checkpoint
`e6d6303fbf25993c768b1e86fecbb350433a2fb229235b556334e9eb6ae868cf`,
seed 65, 1,024 matches on one B200 and 512 rollout frames. Changes preserve
the policy, rewards, window placement, optimizer steps and precision contract.

## Existing profile

Source: `~/data/scratch/o65-o64-profile4/profile/`. This run uses 64 windows
per actor step, one microbatch and 16 selected decisions per window. Its
training code matches the control; its rollout code predates the current
overlap between anchor graphs and the student graph.

| Phase | Wall s | Model TFLOP | BF16 compute floor s | Achieved TFLOP/s | MFU |
|---|---:|---:|---:|---:|---:|
| Preparation | 6.258 | 2917.5 | 1.297 | 466.2 | 20.7% |
| Actor | 10.204 | 3260.6 | 1.449 | 319.5 | 14.2% |
| Critic | 0.946 | 53.9 | 0.024 | 56.9 | 2.5% |

The floor divides model FLOPs by 2,250 TFLOP/s. It excludes memory traffic,
optimizer work, nonlinearities and host work. The critic uses FP32, so its
BF16 floor is only a common normalization, not an attainable FP32 target.
FLOPs come from `timed.json` metrics. `estimates.json` retains a proxy-model
formula that understates O64 work and must not be used for O64 MFU.

The bounded traces show where to investigate. They are instrumented samples,
not replacements for the untraced timings above.

| Sample | Kernels | GEMM ms | Attention ms | Triton pointwise ms | Triton reductions ms | Other/index ms |
|---|---:|---:|---:|---:|---:|---:|
| Reference windows | 280108 | 2162.4 | 695.4 | 1209.8 | 662.6 | 1252.7 |
| Actor step 0 | 5767 | 25.1 | 6.0 | 4.8 | 4.7 | 42.4 |
| Bootstrap | 1386 | 102.0 | 35.5 | 4.0 | 2.0 | 275.1 |
| Critic fit | 727 | 717.8 | 0 | 0 | 0 | 233.0 |

The actor sample includes the actor graph, optimizer and diagnostics. The
actor graph itself accounts for about 65.2 ms of the sample's 83.0 ms summed
kernel time. Eager ATen pointwise kernels appear under Other/index, while
the pointwise column counts Triton kernels only. Eager trunk pointwise
operations dominate the graph's non-GEMM work. The reference trace includes
128 graph replays; each encodes a trunk and four groups of decoder slots.
The critic's largest kernels are FP32 SIMT GEMMs. Its low BF16-normalized MFU
does not imply the same gap against its FP32 compute ceiling. This work
preserves the critic's FP32 arithmetic.

Print the phase table and kernel totals used for the sample table with:

```sh
uv run python -m scripts.o65_training_trace ~/data/scratch/o65-o64-profile4/profile
```

## Fresh control

`~/data/scratch/o65-devbox/runs/bench-20261022-230834-m64-full/` records
the control at 64 windows per step and one microbatch. Its loaded trainer
source SHA256 is
`8b0a7d7967d17f01c37a8db1fdf95b0620e03891d62d29d9d1f8c252e620cf78`,
which matches `a0ae12df`. The checkpoint, sampler, seed, matches, rollout
length and training configuration match the existing profile above.

| Phase | Wall s | BF16 compute floor s | Achieved TFLOP/s | MFU |
|---|---:|---:|---:|---:|
| Preparation | 6.302 | 1.297 | 462.9 | 20.6% |
| Actor | 10.919 | 1.449 | 298.6 | 13.3% |
| Critic | 0.948 | 0.024 | 56.8 | 2.5% |

Rollout takes 14.575 s; the complete update takes 32.828 s. The timed update's
peak allocated memory is 91,745,788,416 bytes (85.4 GiB), and peak reserved memory is
188,986,949,632 bytes (176.0 GiB). Startup capture takes 728.9 s.

The initial default-512-window benchmark was stopped during compilation,
before an update or report. The orchestrator changed the devbox helper's
default to 64 after its 512-window OOM. A separate failed 64-window start
found that devbox sync does not revert a remote file when the local file
returns to the base commit. The control source was restored explicitly on
the sandbox and its SHA256 was checked before the successful run.

## Retained cleanup

The only retained runtime change skips unused full-model gradient buffers and
reuses the already gathered windows when `actor_microbatches=1`. Multiple
microbatches keep their original accumulation and gather path. Model forward,
PPO loss, selected frames, statistics, optimizer and scheduler are unchanged.
A CPU control comparison matched the model, both optimizers, scheduler and all
non-timing metrics exactly. The existing whole-step/microbatch parity test
also covers padding and update order.

Matched B200 run: `~/data/scratch/o65-devbox/runs/bench-20261023-001008-m64-full/`.
Loaded trainer source SHA256:
`f8a4037f3213e3a5910049816a59adc440372f02c86f432f0c7a413f76e6f326`.
The checkpoint, seed, data, sampler and resolved training settings match the
fresh control. The table uses `systems/*_seconds` consistently, as does the
MFU calculation; profiler outer phase spans are slightly different.

| Phase | Control s | Cleanup s | Control MFU | Cleanup MFU |
|---|---:|---:|---:|---:|
| Preparation | 6.302 | 6.322 | 20.6% | 20.5% |
| Actor | 10.919 | 10.637 | 13.3% | 13.6% |
| Critic | 0.948 | 0.947 | 2.5% | 2.5% |

Actor time fell 2.6%; the complete update fell from 32.828 to 32.479 s (1.1%).
These are single whole-update samples, not confidence intervals. Preparation
and critic have no targeted arithmetic change. First-update optimization peak
allocated memory is 90,736,531,968 bytes (84.5 GiB) in both runs; this is a
separate first-update metric from the timed-update peak above. First-update reserved memory is
188,970,172,416 bytes (176.0 GiB), versus 188,974,366,720 in the control.
The 40% actor/preparation MFU target is not met.

## Rejected variants

Whole-trunk compilation after eager observation encoding reduced preparation
from 6.302 to 5.623 s (20.6% to 23.1% MFU) and actor from 10.919 to 8.813 s
(13.3% to 16.4% MFU). Critic remained 0.949 s / 2.5% MFU. The complete update
was 29.724 s. This variant is **rejected**, not a qualified speedup.
Its run is `~/data/scratch/o65-devbox/runs/bench-20261022-233947-m64-full/`;
trainer source SHA256 is
`1b59469a88ca5f124cdb1436d1c4852b21dd4b2a23589ada2646cb83bb521b44`.
`trunk-compile-rejected.patch` freezes that source relative to the control.
It is evidence only and must not be applied to a training run.

The first synthetic gate compared each BF16 path's error against FP32. Large
baseline gradient error made that gate too weak. It is superseded by recorded
real first-update PPO inputs, a repeated eager control and direct gradient
checks: cosine at least 0.999 and relative L2 at most 0.05, in addition to the
FP32-relative error gate. A focused regression test rejects the case that the
old gate accepted.

Three real 64-window cohorts fail the new gate:

| Cohort | Compiled/eager gradient cosine | Relative L2 vs eager | Eager/FP32 cosine | Compiled/FP32 cosine | Repeat eager relative L2 |
|---|---:|---:|---:|---:|---:|
| 0 | 0.7386 | 0.7415 | 0.8184 | 0.7743 | 1.50e-07 |
| 1 | 0.7862 | 0.6762 | 0.5791 | 0.5884 | 1.23e-07 |
| 2 | 0.7054 | 0.8064 | 0.6195 | 0.6639 | 1.72e-07 |

Both BF16 paths also differ substantially from FP32. These diagnostics do not
establish the cause. The retained runtime keeps the eager BF16 computation.

These final qualification runs put autocast inside the compiled function and
call it outside autocast; that boundary change did not fix the gradient
failure. The root cause remains unresolved. The reports include step-zero
ratios, per-group gradients and input/probe/checkpoint hashes. The reference
is the initial O64 policy, so KL is zero here; a future accepted variant also
needs nonzero-KL coverage and the actual captured actor graph after an update.
All compiled-trunk production changes have been removed.

Inputs: `~/data/scratch/o65-devbox/runs/bench-20261022-235147-m64-full/training-inputs.pt`,
SHA256 `b52e7dd1ba8f6d5257f07c2971d1b24bb6a3e07a2f0974db7753d76a2cdc62e4`.
Reproduce the diagnostic on the devbox with:

```sh
TORCHINDUCTOR_CACHE_DIR=/cache/inductor TRITON_CACHE_DIR=/cache/triton \
  /opt/venv/bin/python -m scripts.o65_replay_probe \
  --inputs /cache/runs/bench-20261022-235147-m64-full/training-inputs.pt \
  --cohort 0 --output /cache/train-rejected-final-c0.json
```

Failure of this command is expected: it writes its report before rejecting the
compiled variant. `scripts/o65_devbox.py bench --full --capture-training-inputs`
provides a separate, unmeasured input-capture mode. It requires a fresh initial
actor and one warm update; it does not add file I/O to measured training phases.

A second probe replaced shared history FlexAttention with repeated memory and
variable-length FlashAttention at 64 windows, 16 slots, 25 offsets, 16 heads,
head width 64 and context 256. Captured forward-plus-backward time increased
from 2.408 to 4.424 ms. It is rejected for speed. The standalone diagnostic is
`scripts/o65_history_attention_probe.py`; results are at
`~/data/scratch/o65-devbox/runs/train-history-attention-probe/report.json`.
No attention implementation change is retained.
The attention microbenchmark reports numerical errors without an acceptance
gate; its slower timing was sufficient to reject it.

`training-throughput-results.json` records compact benchmark metadata and raw
qualification reports. Large traces and captured inputs remain in the paths
above. `scripts/o65_trunk_probe.py` is an isolated synthetic trunk diagnostic;
its 34.94 to 19.95 ms result is also unqualified because gradient parity fails.

## Remaining work

Resolve the compiled backward discrepancy before attempting further trunk
fusion. The reference graph still has eager trunk pointwise work. A reference
batching change needs a fresh memory check; an actor batch-size change belongs
to the orchestrator because it changes the optimizer schedule. The critic's
FP32 precision remains unchanged. No long training run was launched.

## Validation

Final required checks pass:

```sh
uv run ruff format --check .
uv run ruff check .
uv run ty check --python-version 3.14 --error-on-warning \
  hal experiments/059_muon_action_sequence.py scripts
```

Focused CPU checks: 21 passed, including capture identity, actual trainer targets,
loader/gate failures, timing floors, batching overrides, stale-source rejection,
and fresh-gather actor parity. Command:

```sh
TMPDIR=~/data/scratch/o65-tmp-train OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  nice uv run pytest -q --basetemp="$HOME/data/scratch/o65-tmp-train/f" \
  tests/test_o65_training_probe.py tests/test_o65_replay_probe.py \
  tests/test_o65_devbox_batching.py tests/test_o65_training_trace.py \
  'tests/test_o65_training_batching.py::test_actor_step_reuses_windows_without_writes_or_gradient_changes[cpu]'
```

B200: `tests/test_o65_training_batching.py` passed both CPU and CUDA cases
(2 passed, 180.54 s). It checks loss, log-probs, gradients and unchanged input
storage against a fresh gather through the actual actor graph. All three
real-input compiled-trunk qualifications fail, as expected for the rejected
variant. Their reports are preserved in `training-throughput-results.json`.

The final full suite ran under `nice` with the shared heavy-test lock,
`TMPDIR=~/data/scratch/o65-tmp-train`, two OMP/MKL/compiler threads and a short
`--basetemp` to keep Unix socket paths within the platform limit:

```sh
nice uv run pytest -q -m 'not integration' \
  --basetemp="$HOME/data/scratch/o65-tmp-train/all" -rs
```

Result: **2,755 passed, 39 skipped, 1 failed, 90 deselected** in 532.30 s.
The failure is the known
`test_control_timing_recovery.py::test_repeated_recovery_keeps_original_wave_seed_matchups_and_slots`:
R2 needs `AWS_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` to
fetch the missing ISO. Log: `~/data/scratch/o65-train-final-pytest.log`.

All 39 skips:

| Tests and lines | Count | Missing fixture or opt-in condition |
|---|---:|---|
| `test_archive_streaming.py`: 84, 93, 114, 125, 134, 167, 260 | 7 | `data/raw/dev.7z` |
| `test_behavior.py`: 496 | 1 | `data/raw/dev.7z` |
| `test_behavior_rows.py`: 215, 251, 272 | 3 | `data/raw/dev.7z` |
| `test_extract.py`: 154, 174, 194, 218, 235, 262, 272, 282, 340, 351 | 10 | `data/raw/dev.7z` |
| `test_replay_stats.py`: 162 | 1 | `data/raw/dev.7z` |
| `test_policy_schema.py`: 188, 211, 227, 237, 246 | 5 | local dev MDS |
| `test_policy_world_schema.py`: 50 | 6 | local v7 subset |
| `test_o65_proxy_checkpoint.py`: 28 | 2 | `HAL_O65_PROXY_CHECKPOINT`, O63 bootstrap30 artifact |
| `test_netplay_hardware.py`: 52, 68 | 2 | `HAL_REQUIRE_NETPLAY_HARDWARE_QUALIFICATION=1` on production GPU |
| `test_o65_fixed_layout.py`: 284 | 1 | `HAL_O65_FIXED_GPU=1` on RTX 3060 |
| `test_o65_reward.py`: 84 | 1 | `HAL_O65_REWARD_GPU=1` on RTX 3060 |

Required integration command:

```sh
HAL_REQUIRE_INTEGRATION=1 TMPDIR=~/data/scratch/o65-tmp-train \
  nice uv run pytest -q --basetemp="$HOME/data/scratch/o65-tmp-train/it" \
  tests/test_roundtrip.py tests/test_session_cleanup.py -m integration
```

Result: **8 failed, 6 deselected** in 2.16 s. All eight fail because this
worktree lacks `data/emulator/ssbm.ciso`: controller wire, fresh roundtrip,
instant-restart roundtrip, independent restarts, analog sweep, stick vocabulary,
normal-exit cleanup and SIGKILL-parent cleanup. Log:
`~/data/scratch/o65-train-integration.log`. These fixture failures leave
validation incomplete; they are not clean passes.

An earlier full suite before the rejected prototype was removed had 2,745 passes,
39 skips and four failures. Three netplay failures came from Unix socket paths
longer than 108 bytes under pytest's default nested temporary directory. All
three passed with a short `--basetemp`; the final full suite confirms that fix.
The fourth was the same missing-R2-credentials failure. A separate fixture
check, `tests/test_policy_schema.py tests/test_policy_world_schema.py`, had
18 passes and the same 11 MDS/v7 skips listed above.

## Review

The complete diff from `a0ae12df` was reviewed for numerical changes, input
aliasing, graph ownership, stream ordering, frame/offset masks, unsupported
configuration and artifact identity. The retained runtime adds no graph writes
or streams. Actor capture clones inputs into private persistent buffers;
reused gathered windows stay read-only. The multiple-microbatch path keeps its
original guarded accumulation. CPU update parity and B200 actor graph parity
cover the retained change.

Opus reviews ran at the measurement, implementation and final-review milestones.
The substantive fixes were a real-input gradient gate, repeated eager controls,
actual trainer-target tests, weights-only artifact loading, unsupported-clipping
and distributed-capture rejection, consistent timing/memory labels, and stale
trainer-source rejection before benchmark launch. The final follow-up found no
remaining correctness defect in the retained changes. Reviewer reports are
`~/data/scratch/o65-train-{milestone1-review,runtime-review,final-review,review-followup,review-resolution}.txt`.
The rejected patch was applied to a temporary control source and reproduced
its recorded SHA256 exactly. Blank context lines were normalized so the final
Git whitespace check also passes.

Cross-stream changes are the profiling-only input-capture switch and writer in
`065_selfplay_ppo.py`, and the shared devbox CLI's batching/capture options,
64-window benchmark default and trainer-source check. No `hal/` changes remain.
The source check covers the trainer; general synchronization of reverted or
deleted files remains an existing helper limitation.

## Final-commit repeat

`~/data/scratch/o65-devbox/runs/bench-20261023-004052-m64-full/` repeats the
cleanup on committed trainer source `030234fe`; the remote source hash matches
that commit exactly. The resolved training configuration and FLOP counts match
the control and first cleanup run.

| Phase | Control s / MFU | Cleanup repeat s / MFU |
|---|---:|---:|
| Preparation | 6.302 / 20.6% | 6.312 / 20.5% |
| Actor | 10.919 / 13.3% | 10.740 / 13.5% |
| Critic | 0.948 / 2.5% | 0.947 / 2.5% |

The repeat update takes 32.204 s. Across the two cleanup samples, actor time
is 1.6–2.6% below the one control sample. This is a small observed gain; more
samples would be needed to quantify run-to-run variation. The 40% MFU target
remains unmet. No production batch-size, precision or optimizer change was made.

The first cleanup's large reference trace transfer stalled in the Modal client.
A compressed read recovered the complete trace; every expected report and trace
is present locally. The completed benchmark was not rerun for this transfer.
