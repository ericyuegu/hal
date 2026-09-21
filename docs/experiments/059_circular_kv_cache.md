# O59 circular K/V storage

Local RTX 3060 study, 2026-09-21. Base commit: `9c364c7e8a94b785d5a9447b6a06cdc74a077e4c`.

## Scope and controls

The treatment replaces chronological K/V copies with persistent circular storage in
`cache_mode="rolling"`. The controls are the saved shifting implementation and full
recomputation. The shifting reference lives only in test code. Model weights,
checkpoint format, sampling, absolute-position RoPE, and chronological final hidden
history do not change. After eviction, rolling inference deliberately retains older
layer representations. Full recomputation is therefore a cost control, not a
post-eviction correctness reference.

Each incoming token writes slot `observation_position % L_ctx`. The two new tokens
run in sequence. Startup and reset masks expose only the new sequence's valid
physical prefix. Inactive rows receive no CUDA writes. Row reordering can copy the
cache; ordinary continuation, including padded rows, keeps its K/V addresses.

Native indexed assignments in PyTorch 2.11.0 kept the input addresses but generated
full-cache temporaries and copyback kernels. The CUDA path instead uses one small
Triton slot-write operation with declared K/V mutation. Attention remains in
PyTorch. Generated code for the 12-layer proxy contains 24 slot writes, no
cache-shaped temporary allocation, and no K/V copyback. An isolated compiled writer
also records zero temporary CUDA allocation. CPU inference uses native writes.

The CUDA operation requires the tested PyTorch 2.11.0 / Triton 3.6.0 pair. Remove
this workaround only after native writes pass the compiled allocation test and
generated-code inspection without full-cache copies. The mutation declaration uses
[PyTorch's public Triton operator API](https://docs.pytorch.org/docs/2.11/library.html#torch.library.triton_op).

## Method

The proxy uses width 256, 12 trunk layers, six decoder layers, four heads, batch 32,
BF16 attention, four predicted frames, and two committed/new frames. Context 256
is the main comparison. Contexts 64 and 512 measure the trunk separately. Seed 59
fixes model initialization, synthetic observations, and sampling uniforms. No
checkpoint, replay, or Dolphin process is used.

Each complete decode run fills the cache, warms compilation, and measures three
repetitions of 1,000 two-frame replans. Each replan ends with a CUDA synchronize.
GPU events time the trunk, memory projection, and temporal decoder separately.
The separate decoder copy accepts projected memory and must produce exactly the
same sampled indices as the production decoder before timing. GPU event intervals
include any gaps in host kernel submission. Profiles are collected after timing.
All GPU runs execute outside the sandbox, one at a time.

Environment: NVIDIA RTX 3060, 12 GiB, driver 595.91.07, CUDA runtime 13.0,
PyTorch 2.11.0+cu130, Triton 3.6.0, Python 3.14.4. Compile mode is `default` unless
stated otherwise. Compile/fill times include first execution and can reuse the
local compiler disk cache; they are not cold compilation comparisons.

Raw JSON, individual latency samples, profiler traces, emitted kernels, source
snapshots, and validation logs are retained locally under `results/o59-circular/`.
The benchmark records resolved configuration, source hashes, Git SHA, seeds,
environment, and compiler counters. The final artifact manifest binds the local
files to the implementation commit.

## Results

Complete decode at context 256, with pooled percentiles from 3,000 replans:

| Storage | Mean ms | p50 ms | p95 ms | p99 ms | Replans/s | Committed frames/s | Peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| recompute | 16.733 | 16.676 | 16.974 | 17.832 | 59.76 | 3825 | 137.1 |
| shifting | 14.487 | 14.549 | 15.521 | 17.616 | 69.03 | 4418 | 397.2 |
| circular | 14.791 | 14.834 | 15.644 | 17.950 | 67.61 | 4327 | 209.9 |

The final matched pair gives a circular/shifting latency ratio of 1.021: circular
was about 2.1% slower. An earlier pair gave 14.02 ms circular versus 14.87 ms
shifting (about 5.7% faster). These runs do not establish a consistent latency gain
at context 256. Peak allocation fell about 47%, from 397.2 MiB to 209.9 MiB in
the final pair. Persistent cache size is 100 MiB for both implementations, including
final hidden history. Full recomputation has no persistent trunk K/V cache.

Trunk GPU-event mean latency, averaged over three 1,000-call repetitions:

| Context | Recompute ms | Shifting ms | Circular ms |
|---:|---:|---:|---:|
| 64 | 3.996 | 5.049 | 5.378 |
| 256 | 12.353 | 5.378 | 5.596 |
| 512 | 24.662 | 7.292 | 5.365 |

Separate component GPU-event timings at context 256 (1,000 calls each):

| Storage | History projection ms | Temporal without projection ms | Temporal including projection ms | Compile and fill s |
|---|---:|---:|---:|---:|
| recompute | 0.186 | 7.766 | 7.223 | 13.23 |
| shifting | 0.186 | 7.851 | 7.498 | 7.07 |
| circular | 0.186 | 7.869 | 7.585 | 7.15 |

Components are separately compiled and timed; their means need not add to the
complete-path latency. The initial shifting compile/fill took 39.78 s; the final
pair reused disk cache entries. All final timing loops ran under
`fail_on_recompile`: zero timing recompilations and zero graph breaks. The main
programs compiled four graphs per arm during warmup (trunk, complete temporal
decoder, memory projection, and temporal decoder with projected memory). The
default-mode profiles recorded zero CUDA graph launches.

The matched shifting/circular runs have identical model and input hashes:

- Weights: `94346bec283cc8f9c93abe2455e3ee4377998cc6da16330ddad728fcbd7847d2`.
- Inputs: `6f5dd2ebe69458feba422f79e07947b94377b2659aec1a489b26cd8d8cb94360`.

## Correctness and compiler behavior

The tests compare CPU float32 and GPU BF16 with the independent shifting reference
through more than three complete 256-slot wraps. They cover one- and two-token
updates, unequal row positions, full-cache resets, mixed resets/continuations,
reordering, removal, new streams, padded rows, fixed capacity, and stable storage.
Reset comparisons start the reference from empty storage, so stale masked entries
cannot affect the new sequence. Before eviction, tests also compare full
recomputation. Float32 checks use `rtol=2e-5, atol=2e-6`. BF16 requires hidden relative
RMS error below 1% and cosine similarity above 0.9999. Logit and sampled-action
errors are reported separately.

Measured wrap-test errors (the GPU test uses the 12-layer, width-256 trunk and
six-layer, width-256 decoder; the CPU test uses two small trunk layers):

| Dtype | Maximum hidden relative RMS | Minimum hidden cosine | Final logit RMS | Sampled disagreement |
|---|---:|---:|---:|---:|
| CPU float32 | 0.0000000790 | 0.999999642 | 0.000000103 | 0 |
| GPU BF16 | 0.004621 (0.4621%) | 0.999989331 | 0.001742 | 0 |

Both `default` and `reduce-overhead` pass repeated position/reset calls under
`fail_on_recompile`. The profiler records no CUDA graph replay in `default` mode.
In `reduce-overhead`, the decoder replays a CUDA graph, while PyTorch excludes the
trunk because its input K/V tensors are mutated. The compile-mode flag alone does
not establish that the trunk uses a CUDA graph.

## Limits

These synthetic results measure implementation costs on this RTX 3060. They do not
predict full-model RTX 6000 speed or gameplay quality. No Modal evaluation was run.
The existing rolling path still uses cache capacity as its attention window; a
custom narrower `arch.attn_window` is not applied by that path. This change does
not address that separate configuration issue.

## Reproduction and validation

Run the focused tests on a CUDA host. Required GPU tests fail if CUDA is missing:

```sh
uv run pytest -q -s tests/experiments/test_059_muon_history_decoder.py
uv run python tests/experiments/benchmark_059_cache.py --cache shifting --output results/o59-circular/shifting.json
uv run python tests/experiments/benchmark_059_cache.py --cache circular --output results/o59-circular/circular.json
uv run python tests/experiments/benchmark_059_cache.py --cache recompute --output results/o59-circular/recompute.json
```

Use `--context 64 --trunk-only` or `--context 512 --trunk-only` for each arm of the
context sweep. Defaults are 1,000 iterations and three repetitions. Output JSON
contains p50/p95/p99 for each repetition, peak allocated/reserved memory, cache
bytes, compiler counters, and the resolved configuration.

Final required checks:

- `uv run pytest -q -s tests/experiments/test_059_muon_history_decoder.py`: 70 passed,
  zero skips, 16 warnings. These include dependency deprecations, the existing
  eager flex-attention warning, and a profiler warning.
- `uv run pytest -q --ignore=tests/experiments -m "not integration"`: 1,165 passed,
  18 integration tests deselected, zero skips, 29 warnings.
- `uv run ruff format --check .`: passed, 339 files already formatted.
- `uv run ruff check .`: passed.
- `uv run ty check --python-version 3.14 --error-on-warning hal experiments/051_muon_parameterization.py scripts/cache_modal_fixtures.py scripts/launch_gce.py scripts/launch_modal.py scripts/launch_vast.py scripts/replay_policy_fault.py`:
  passed with zero diagnostics.

Format, lint, and type checks used `UV_CACHE_DIR=/tmp/hal-uv`; the default cache
was read-only in the sandbox. GPU tests and benchmarks ran through escalated
commands. No shared controller, session, extraction, or wire code changed, so the
plan's conditional integration suite was not required.

Development failures were corrected before the final checks: the first benchmark
needed explicit synthetic stream metadata; a Triton kernel pointer annotation
using `Any` failed in generated Python and was replaced with `tl.tensor`; and the
GPU stream-membership test needed production autocast instead of manually casting
all model parameters to BF16. The first expanded GPU test command had one failure
and six passes; the corrected membership test passed both devices, and the final
70-test suite passed all cases. No required GPU coverage was skipped.
