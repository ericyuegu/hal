# O59 BF16 throughput results

These measurements apply to `059_muon_history_decoder_v1`. The later v2 model
uses nonlinear trunk-skip heads and a 4x temporal MLP, so its throughput and MFU
must be measured separately.

Use physical batch **512**, CPU prefix validation, and `max-autotune` for long
training runs. Keep production diagnostics and eager Muon. Active-AWR throughput
rose from **1,340.3 to 1,403.7 samples/s (+4.73%)** on one B200. Each of the three
combined-treatment windows exceeded every baseline window. Approximate MFU rose
from 28.2% to 29.6%. Reserved-memory accounting leaves 28.2% of the GPU free.

Batch 1024 ran out of memory; the sweep stopped before 2048. No larger production
batch is proposed. The model, loss, optimizer settings, and training schedules
are unchanged.

The main allocation took **3,245.8 seconds (54.1 minutes, 0.902 B200-hours)**,
including data preparation, compilation, warmup, and all comparisons. Two setup
attempts preceded it: one was rejected before allocation; one exited immediately
on a missing work directory. Neither reached compilation or training. Total B200
use stayed below two hours. The allocation has exited.

## Measurements

All rows use batch 512. Throughput and update time are medians of three unprofiled
active-AWR windows, each containing 51,200 examples, through pinned CPU prefetching.

| Treatment | Samples/s | Gain | Update | Peak reserved | First update |
|---|---:|---:|---:|---:|---:|
| Baseline, `reduce-overhead` | 1,340.3 | — | 382.00 ms | 128.20 GiB | 180.65 s |
| Reuse CPU validation | 1,346.1 | +0.43% | 380.35 ms | 126.21 GiB | 24.64 s |
| `max-autotune` | 1,395.1 | +4.08% | 367.01 ms | 128.27 GiB | 1,017.02 s |
| CPU validation + `max-autotune` | **1,403.7** | **+4.73%** | **364.74 ms** | **128.13 GiB** | 134.52 s |
| Diagnostics disabled, `reduce-overhead` | 1,339.4 | −0.07% | 382.27 ms | 126.28 GiB | 79.29 s |

First-update costs exclude process startup. Cases share a compiler cache, so
these costs are not independent cold-start measurements. The first max-autotune
case spent about 17 minutes compiling; the combined case reused that cache and
started in 2.24 minutes. Max-autotune alone needs about 56,000 updates to repay
its extra cold compilation cost relative to baseline. `reduce-overhead` remains
the default for short iterations. The compiler cache is saved for later launches.
Restoring it in a fresh B200 allocation was not measured.

The three baseline prefetch windows were 1,343.5, 1,340.3, and 1,338.6 samples/s.
The combined windows were 1,406.2, 1,403.7, and 1,403.3. These are repeated windows
within one allocation, not independent training seeds or a learning-quality test.

Device-resident baseline throughput was 1,338.0 samples/s, close to pinned CPU
prefetching. The five-update baseline trace showed GPU activity during 97.18% of
the interval between its first and last GPU activities; the largest idle gap was
1.89 ms. Profiling adds overhead, so these values are diagnostic only.

| Mean profiled phase | Baseline | Combined |
|---|---:|---:|
| Target preparation and sampling | 1.97 ms | 1.19 ms |
| Trunk forward | 70.56 ms | 67.65 ms |
| Temporal forward | 52.65 ms | 43.09 ms |
| Objective | 2.39 ms | 2.74 ms |
| Backward | 239.40 ms | 238.25 ms |
| Gradient guard and clipping | 0.91 ms | 0.91 ms |
| Optimizer and scheduler | 17.16 ms | 17.34 ms |

The entire optimizer accounted for 4.46% of baseline update time. Compiled Muon
was therefore not tested: Muon alone cannot meet the planned 10% trigger. The
optional benchmark hook preserves the existing arithmetic and has update-parity
coverage. Disabling diagnostics gave no repeatable gain; production keeps them.

Batch 1024 failed while allocating a 1 GiB temporal activation. CUDA reported
177.54 GiB in use on a 178.35 GiB B200, with 814.62 MiB free. The batch rule selects
the smallest tested batch within 5% of the best throughput and with at least 10%
memory free. Batch 512 is the only completed eligible size.

## Correctness and production changes

The production training path now reuses prefix validation already performed by
`DeviceBatchPrefetcher` on the CPU. Direct sampler calls retain their validation.
GPU prefix samples and RNG advancement are exact. The successful B200 comparison
also had exact loss and updated parameters; gradient maximum absolute difference
was `7.08e-08`, and relative L2 difference was `1.35e-07`.

The first validation attempt incorrectly required bitwise-identical gradients
and stopped before measurement. Its loss and updated parameters were exact.
The corrected check allows at most eight FP32 epsilons in gradient maximum
absolute and relative L2 differences, while still requiring exact samples, RNG,
loss, and updated parameters. The failed attempt remains in the artifact store.

For the combined compiled treatment, compared with the same initial state and
first active-AWR batch, objective difference was `1.53e-05`, gradient maximum
absolute difference was `7.74e-05`, gradient relative L2 difference was `0.106%`,
and updated-parameter maximum absolute difference was `1.69e-07`. These compiled
kernels are not bitwise equivalent. All measured metrics were finite, and measured
windows rejected recompilation.

`TrainConfig.train_compile_mode` is now an explicit, validated setting. New
checkpoint configuration version 2 records it. Version 1 loads with its historical
`reduce-overhead` setting; unsupported formats and mismatched fields still fail.
For a new long training run, add:

```sh
--cfg.train-compile-mode max-autotune
```

Resume checks found an existing defect: a CUDA generator requires its serialized
RNG ByteTensor on the CPU, but O59 moved it to CUDA. The restore path now uses
CPU state. Regression tests cover CPU-loaded and GPU-loaded RNG state, and exact
next-batch and next-update resume with both AWR states on CPU and a local RTX 3060.
The update tests use the actual eager training step with a small model. Full-size
compiled B200 next-update resume was not tested.

Larger-batch benchmarks resolve Adam learning rates, betas, epsilon, and the
scheduler from the batch-512 reference. Production still rejects larger batches.
Adopting one would require a separate optimizer-scaling and learning-behavior
experiment because several schedule boundaries remain fixed in updates.

## Protocol and faster iteration

The fixed bank contains 2,048 real windows: eight deterministic windows from each
of the first 256 selected replay rows of
`ranked-anonymized-6-policy-world-v8`. Only required shards are downloaded. The
training manifest, schema, and identity artifacts are validated. Normalization
uses the production 44-source mixture's small statistics sidecars; replay data
comes from only the one selected dataset. Identity dropout is frozen in groups
of 512 before comparing paths. Example order and artifact hashes are recorded.

All cases use seed zero and identical initial model states. The bank is cycled
in recorded order. This measures the full optimization step and host-to-device
prefetch path, not sustained decoding of a full dataset. No FP8, gradient
accumulation, or activation checkpointing was used.

The baseline times three windows for each transport and each AWR state.
Treatments time three active-AWR windows per transport and exercise both AWR
states during warmup and separate five-update profiles. Compilation and warmup
are excluded from throughput windows but included in the GPU budget.

The launcher mounts an exact source archive onto the existing CUDA image. It
avoids dependency builds and Dolphin fixture downloads. Cases run in child
processes inside one allocation and share data and compiler caches. This releases
CUDA memory after an OOM without another Modal startup. There are no retries;
the suite budget is 6,900 seconds, with a 7,100-second hard function timeout.

Use the saved cache for a later benchmark:

```sh
uv run scripts/benchmark_o59_modal.py \
  --compiler-cache-from e9951c6a72f945fab312cd70fc30b9ad
```

For an existing B200 environment:

```sh
uv run experiments/059_muon_history_decoder.py benchmark --output results/o59-throughput
```

The launcher verifies the cache archive hash before restoration. Cache roundtrip,
corruption rejection, and budget validation have fast local tests. The focused
suite takes about 15 seconds on the local RTX 3060; the required broad suite takes
about two minutes. Modal is needed for B200 measurements, not ordinary regressions.

[PyTorch compile-mode documentation](https://docs.pytorch.org/docs/2.11/generated/torch.compile.html)
describes the compilation controls used here.

## Evidence and operational limits

[Machine-readable measurements](throughput-results.json) include resolved reference
configuration, environment, windows, phase timings, numerical differences,
provenance, hashes, and the original allocation exit status. Raw reports, logs,
source archives, batch bank, and all five profiler traces are in the
`hal-o59-benchmarks` Modal Volume, separate from resumable production checkpoints.

- App: `ap-eA6vTMYTDSRkJMQ5exf2ue`
- Function call: `fc-01M2YVRG78S75CP0V19BNMRFG0`
- Container: `ta-01M2YVRGFYTV5TPE9W1ES2BFSR`
- Volume directory: `e9951c6a72f945fab312cd70fc30b9ad`
- Base Git SHA: `dfca643180d361c961b84cd5982fc4814025a133`
- Source archive SHA-256: `9c8b4ab3fc872d23188bf2864b4987a06efe79bb40cd4f153a3e91ab99260f46`
- Bank SHA-256: `03fbf5997cb7ee732f57735c308cfdb66c0bf36cf4bcecbd77c0ec5d35a9f7e6`
- Cache SHA-256: `7b33c14806867747024d14acece27ccad53a2a9c8999d26cc86499a19e8b92a6`
- Environment: PyTorch `2.11.0+cu130`, CUDA 13.0, Python 3.14.3, NVIDIA B200.

Source revisions 2–5 and their hashes are archived. They added treatment timing
savings, corrected the validation tolerance, and queued the retry and combined
case inside the existing allocation. Each treatment report records its source
hash. The original baseline source is in the initial archive. The temporary
remote retry dispatch is not part of the repository launcher.

All five individual comparisons saved complete reports and traces. A later
attempt to verify the resume fix through container exec failed; the benchmark
control process exited with SIGHUP (`-1`) before final automatic aggregation.
Its original `summary.json` is therefore incomplete and still contains the first
validation failure. The comparison file linked above uses the completed individual
reports and retains that operational failure. Resume verification subsequently
passed on the local GPU. No additional B200 allocation was started.

The 349 MiB compiler archive was separately saved and committed before exit.
Its export and a small CUDA RNG diagnostic overlapped diagnostic-ablation startup
or early device timing. They did not overlap that case's active-prefetch windows
or any recommended-treatment windows. No speed claim uses the affected timing.

## Validation

Final checks:

```sh
uv run ruff format --check .
uv run ruff check .
uv run ty check --python-version 3.14 --error-on-warning \
  hal experiments/051_muon_parameterization.py \
  scripts/cache_modal_fixtures.py scripts/launch_gce.py \
  scripts/launch_modal.py scripts/launch_vast.py scripts/replay_policy_fault.py
uv run ty check --python-version 3.14 --error-on-warning \
  experiments/059_muon_history_decoder.py scripts/benchmark_o59_modal.py
uv run pytest -q tests/experiments/test_059_muon_history_decoder.py \
  tests/test_muon.py tests/test_benchmark_o59_modal.py
TMPDIR=/home/ericgu/src/hal/results/o59-test-tmp uv run pytest -q \
  --ignore=tests/experiments -m 'not integration' \
  --basetemp=results/o59-test-tmp/final-pytest
```

Formatting, lint, and both type checks passed. Focused tests: **70 passed**, no
skips, 15 warnings, 14.99 seconds. Broad tests: **1,166 passed**, 18 integration
tests deselected, 29 warnings, 122.45 seconds. Warnings include Python 3.14 Torch
Script deprecations and the eager FlexAttention path used by the small resume test.

An earlier broad attempt failed because `/tmp` hit its quota and pytest lost
stderr. A first-failure rerun confirmed `OSError: [Errno 122] Disk quota exceeded`;
using a workspace temporary directory resolved it. Earlier sandboxed focused
runs skipped CUDA tests because the sandbox hid the local GPU; the final run
executed them outside that restriction. A cache-test collection failure from a
namespace import was fixed with the repository's explicit module-loading pattern.

The changed paths concern training execution and benchmark orchestration. Wire,
controller, session, and offline/live representation code did not change, so the
additional integration suite was not required.
