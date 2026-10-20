# O65 floor validation

The floor stream starts at `57ed1c57` on `o65-dense`. It changes measurement
instrumentation and tools. It does not change rollout/update arithmetic,
controller values, objective, reward, model, GAE, optimizer or resume formats.
Measurements and their limits are in [floor.md](floor.md).

## Commands and results

All paths below are in `/home/ericgu/src/hal-o65-dense`. Local GPU commands use
`flock /home/ericgu/data/scratch/o65-tmp/gpu.lock`. Heavy tests run serially in
a systemd user scope, initially with MemoryMax=10G and increased to 12G when
needed. No test was killed for memory pressure. Temporary files and compiler
caches use `/home/ericgu/data/scratch/o65-tmp/floor`; no basetemp uses `/tmp`.

```sh
uv run ruff format --check .
uv run ruff check .
uv run ty check --python-version 3.14 --error-on-warning \
  hal experiments/059_muon_action_sequence.py experiments/065_selfplay_ppo.py \
  scripts tests/test_o65_floor.py tests/test_o65_floor_trace.py \
  tests/test_o65_modal.py tests/test_o65_gpu_profile.py
```

All three pass. Formatting checks 397 files; type checking has zero diagnostics.
The additional targets cover the trainer and every new Python file.

The full-suite environment is:

```sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor"
export TRITON_CACHE_DIR="$TMPDIR/triton"
export PYTHONPATH=/home/ericgu/src/melee-sim-light:tests:.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HAL_O65_PROXY_CHECKPOINT=runs/proxy/final.pt
export HAL_O65_COMPILE_CUDA=1
export MSL_CORE_LIBRARY=/home/ericgu/src/melee-sim-light/build/melee_core/python/libmelee_core.so
uv run pytest -q -ra -m "not integration" --basetemp "$TMPDIR/h2"
```

The first complete run passes 2,701 tests, with eight skips, 86 deselections
and 27 warnings in 1,662.56 s (`runs/floor/unit.log`). A later complete run
passes 2,704 tests, with eight skips, 86 deselections and 26 warnings in
408.00 s (`runs/floor/handoff-unit.log`). Final source validation passes
**2,708 tests, eight skips, 86 deselections and 26 warnings**, 405.92 s.
Log: `runs/floor/handoff-final-unit.log`.

Every full-suite skip is explicit:

- One each at `tests/test_netplay_hardware.py:52` and `:68`: production hardware
  qualification is disabled on this host.
- Six at `tests/test_policy_world_schema.py:37`: the required fixture
  `data/processed/ranked-anonymized-1/mds-v7-sub4/train` is absent. These are
  **validation failures under AGENTS.md**, although pytest reports skips. The
  shared, read-only fixture tree was not changed or replaced with synthetic data.

Required integration:

```sh
HAL_REQUIRE_INTEGRATION=1 uv run pytest -q -ra \
  tests/test_roundtrip.py tests/test_session_cleanup.py \
  -m integration --basetemp "$TMPDIR/i"
```

Result: **8 passed, 6 deselected, 7 dependency warnings, no skips**, 57.88 s.
Log: `runs/floor/integration.log`. The MSL/Python environment above was present.

The real Nsight phase-containment regression uses:

```sh
HAL_NSYS_BINARY="$PWD/runs/floor/nsys/opt/nvidia/nsight-systems-cli/2026.5.1/target-linux-x64/nsys" \
HAL_REQUIRE_INTEGRATION=1 uv run pytest -q -ra \
  tests/test_o65_gpu_profile.py -m integration --basetemp "$TMPDIR/np"
```

Result: **1 passed, 17 deselected, no skips**, 7.17 s. It captures 64 actual
CUDA matrix multiplies and verifies that the reference phase ends after its
last GPU kernel. Log: `runs/floor/test-nvtx.log`.
The final-source repeat also passes: **1 passed, 18 deselected, no skips**,
7.10 s (`runs/floor/nsys-final.log`).

Focused measurement, launcher and attribution tests pass 64 tests with one
integration deselection (`runs/floor/focused-final.log`). The final combined
check passes **68 tests with one integration deselection**, 3.62 s
(`runs/floor/measurement-final.log`). Later checks pass
30 launcher tests, including exact Nsight version rejection, and 18 profiler
tests with one integration deselection, including rejected-policy cleanup.
The first floor-test collection failed because `scripts` was not importable
under pytest's path setup. Loading the tool with the repository's existing
file-loader pattern fixes it; the complete suites include that test file.
Initial lint/type diagnostics were fixed before launching cloud work.

## Development measurements

The 3060 native 64-match complete-update Nsight run exits zero. It uses eight
shards because this host has only 12 admitted CPUs. Its raw trace is
`runs/floor/local64.sqlite`; analyzer output is under `local64-analysis`.
All GPU work is attributed. This was a development run while measurement
instrumentation was being edited, so its timings are not source-qualified
performance evidence. Its pre-fix host NVTX ranges must not be used for phase
wall-time attribution.

The standalone bare-model development probe and supplementary prefill/bootstrap
probe both exit zero (`runs/floor/dev-bare.log`, `dev-extra.log`). The cache/DRAM
copy tier probe also exits zero on the 3060 (`memory-tiers-3060.json`).
These checks validate operation and shapes; PRO 6000 measurements determine
hardware rates and phase costs.

## Modal measurements

```sh
uv run python -m scripts.launch_o65_modal --mode floor --matches 1024 \
  --timeout-seconds 4200 --name floor-baseline-5d122bbb-20261020 --output runs/modal-floor
uv run python -m scripts.launch_o65_modal --mode floor-extra --matches 1024 \
  --timeout-seconds 480 --source-name floor-baseline-5d122bbb-20261020 \
  --name floor-extra-ae4eed4f-20261020 --output runs/modal-floor
uv run python -m scripts.launch_o65_modal --mode profile --matches 2048 \
  --timeout-seconds 1900 --name floor-sweep2048-aa807cce-20261020 --output runs/modal-floor
```

The first job completes the warmed clean update, full Nsight update, SQLite
export, bare-model/rate probes, and 256-match sweep arm. It then times out
during the 2,048-match warm-up: overall result `exit_code=-1`, elapsed
4,141.188 s. The failed result is preserved, downloaded and charged with the
startup/teardown margin. No OOM was recorded. The 4,096 arm was not reached.

The supplementary job succeeds in 48.478 GPU-container seconds. It checks both
source artifact hashes before CUDA work. It reuses the pinned Nsight image
layer and does not warm a trainer. No smoke job ran.

The final 2,048-match arm is separately bounded by the remaining ledger. Its
result and final accounting will be appended after completion. No further
cloud job is planned.

Monitoring initially encountered an unavailable public `modal.Container` API,
missing output from non-PTY CLI reads, and one redirected PTY read that returned
`Operation not permitted`. The cause of that error was not established.
These were read-only diagnostics. Unredirected CLI reads with both a terminal
and `--pty` work on the live container. No training process or source was
modified in a running container.

## Self-review

Review uses `git diff o65-selfplay-ppo-plan...HEAD`, with detailed checks of the
stream changes against `57ed1c57`. The review covers offset supervision,
frame/queue masks, detach boundaries, capture ownership, reference metadata,
non-finite checks and resume contracts. An AST scan checks all 73 changed
Python files: no model import of `melee`, simulator import of `torch`, or
experiment-to-experiment import. All 28 changed JSON evidence files parse.

Findings and fixes:

- End NVTX phase ranges after GPU completion. The old order assigned queued
  work outside its phase. The real Nsight regression above covers the fix.
- Keep failed/rescheduled Modal attempts inside one absolute reservation.
  Deadline tests reject an expired reservation; failed jobs retain their charge.
- Preserve profiler GEMM dtype, strides and `addmm`'s bias position. Focused
  arithmetic/parser tests cover these contracts.
- Reject changes to pinned supplementary input/layout bytes before CUDA work.
- Reject unqualified preinstalled Nsight versions and correct stale control
  metadata. Earlier raw launch strings are retained; their resolved config is
  the authority. No evidence was rewritten to hide those strings.
- Put recorder attachment inside its cleanup scope. An unsupported policy now
  clears the trainer and engine recorder before raising; a focused test covers it.
- Correct an analysis-only double multiplication of the critic epoch count
  before publication. The probe's recorded GEMM counts already include four
  epochs; the evidence and floor table use that count once.
- State benchmark limits: repeated reference slots, fixed actor cotangents,
  frozen-step optimizer graph, omitted non-GEMM floor terms, buffer overlap and
  profiler overhead. No negative trainer overhead or full floor is claimed.

The second pass checks the fixes, their callers, tests and recorded arithmetic.
It finds no change to the training math or persisted training state. The
existing FP32 gate thresholds and accepted margins are unchanged. No new
statistical equivalence claim is made; the inherited dense comparison and
timeout-bootstrap scientific question remain outside this measurement change.

## Cross-stream changes and limits

Measurement-only additions touch `experiments/065_selfplay_ppo.py`,
`scripts/launch_o65_modal.py` and their tests. New scripts implement the bare
probe and trace analysis. No `hal/` or simulator source was changed by this
stream, and no other worktree was edited. Only `o65-floor` was pushed.

The complete floor, exact full-reference benchmark, detailed compiled-submodule
provenance and subsequent trainer optimization remain unfinished. The report
states these limits instead of relaxing the prerequisite for tuning.
