# O59 v4 validation

Checks ran on 2026-09-20. Local CUDA checks used the RTX 3060. Full-size
execution comparisons used the B200 allocation in [the throughput report](throughput.md).
O58 was read as a reference and was not changed or imported.

## Required checks

All final commands passed. `UV_CACHE_DIR=/tmp/hal-uv-cache` selected a writable
cache. Pytest temporary directories used the workspace to avoid the `/tmp` quota.

```sh
uv run ruff format --check .
uv run ruff check .
uv run ty check --python-version 3.14 --error-on-warning \
  hal experiments/051_muon_parameterization.py \
  experiments/059_muon_history_decoder.py \
  scripts/cache_modal_fixtures.py scripts/launch_gce.py \
  scripts/launch_modal.py scripts/launch_vast.py scripts/replay_policy_fault.py \
  scripts/benchmark_o59_modal.py
uv run pytest -q tests/experiments/test_059_muon_history_decoder.py \
  tests/test_muon.py tests/test_benchmark_o59_modal.py \
  --basetemp=results/o59-v4-preflight/focused
uv run pytest -q --ignore=tests/experiments -m 'not integration' \
  --basetemp=results/o59-v4-preflight/unit
HAL_REQUIRE_INTEGRATION=1 uv run pytest -q \
  tests/test_roundtrip.py tests/test_session_cleanup.py -m integration \
  --basetemp=results/o59-v4-preflight/integration
```

| Check | Final result |
|---|---|
| Format | 339 files already formatted |
| Lint | Passed |
| Types, including O59 and both changed launchers | Zero diagnostics |
| O59, Muon, benchmark launcher | 101 passed, 15 warnings, 38.26 s |
| Repository unit tests | 1,173 passed, 18 deselected, 29 warnings, 120.67 s |
| Required Dolphin integration | 7 passed, 2 deselected, 7 warnings, 52.54 s |

No final test was skipped. Deselection follows the explicit integration filters.
Warnings include Python 3.14 TorchScript and multiprocessing deprecations and
the eager FlexAttention path in small-model tests.

Coverage includes production and proxy parameter counts, complete optimizer
membership, return horizons and terminal handling, current-position return
alignment versus next-frame AWR alignment, zero initialization, absence versus
valid zero, independent masking, and calibration freeze and hash validation.
Nonzero conditioning is tested across teacher forcing, stepwise decoding,
compiled decoding, tracing, and committed prefixes. Changing the desired return
does not recompile the decoder.

Resume tests cover the next loader batch, identity and return masks, prefix RNG,
calibration state, gradients, and optimizer update. CUDA and CPU tests exercise
both AWR states. Each execution candidate has forward, gradient, and update
comparisons. Old checkpoint and bank formats are rejected. Launcher tests cover
FIFO order, source and bank hash checks, child failure and process-group cleanup,
cancellation, deadlines, and compiler-cache persistence.

## Earlier failures and skips

- An initial broad run used a pytest base directory whose parent did not exist.
  It produced 286 fixture errors and 887 passes. Creating the parent directory
  resolved the setup error. The complete rerun passed.
- A launcher test assumed exactly two persistence calls. The timeout path can
  also persist at its polling boundary. The test now requires persistence after
  both jobs without asserting the incidental number of calls. The first run
  had one failure and 86 passes; the corrected tests pass.
- Two CUDA candidate tests initially used identical synthetic states and a
  full-rate first Muon update without the production warmup scheduler. Their
  update comparison failed. A repeat with per-parameter diagnostics confirmed
  the failure. The regression now uses distinct context states, varied targets,
  and the actual first warmup update. All six CPU/CUDA candidate tests pass.
  Numerical tolerances were not relaxed. B200 acceptance uses real replay
  windows and the same fixed tolerances before any throughput window.
- The new physical-loader resume fixture initially used an invalid replay-ring
  geometry. Two focused attempts failed before training. Matching the required
  batch/window divisibility and phase period fixed the fixture; its loader,
  prefetch, mask, calibration, and next-update comparison now passes.
- Early sandboxed focused runs could not see CUDA and skipped GPU tests. The
  final focused run used CUDA access and executed every test.
- Routine intermediate formatting and import-order failures were corrected
  before the final checks. A model-tag test exposed a 261-character run name;
  the shorter v4 tag now has a regression check for the filesystem limit.

Local logs are in `results/o59-v4-*.log`. The B200 worker retains immutable
source archives, manifests, logs, numerical reports, and traces in the
`hal-o59-benchmarks` Volume.
