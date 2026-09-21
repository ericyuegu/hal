# O59 v5 validation

Checks ran on 2026-09-20. Focused CUDA tests used the local RTX 3060. The full
execution comparison used the B200 allocation in
[the throughput report](throughput.md).

## Final checks

All required commands passed. `UV_CACHE_DIR=/tmp/hal-uv-cache` selected a
writable cache. Pytest temporary directories used the workspace.

```sh
uv run ruff format --check .
uv run ruff check .
uv run ty check --python-version 3.14 --error-on-warning \
  hal experiments/051_muon_parameterization.py \
  experiments/059_muon_history_decoder.py \
  scripts/cache_modal_fixtures.py scripts/launch_gce.py \
  scripts/launch_modal.py scripts/launch_vast.py scripts/replay_policy_fault.py
uv run pytest -q tests/experiments/test_059_muon_history_decoder.py \
  tests/test_muon.py --basetemp=results/o59-v5-final-clean/cuda
uv run pytest -q --ignore=tests/experiments -m 'not integration' \
  --basetemp=results/o59-v5-final-clean/unit
HAL_REQUIRE_INTEGRATION=1 uv run pytest -q \
  tests/test_roundtrip.py tests/test_session_cleanup.py -m integration \
  --basetemp=results/o59-v5-final-clean/integration
```

| Check | Final result |
|---|---|
| Format | 337 files already formatted |
| Lint | Passed |
| Types, including O59 | Zero diagnostics |
| O59 and Muon focused CUDA | 70 passed, 15 warnings, 18.07 s |
| Repository unit tests | 1,163 passed, 18 deselected, 29 warnings, 122.01 s |
| Required Dolphin integration | 7 passed, 2 deselected, 7 warnings, 53.90 s |

No final focused test was skipped. Deselection follows the required integration
filters. Warnings are dependency warnings for Python 3.14 TorchScript,
multiprocessing fork, Streaming shuffle geometry, and eager FlexAttention in a
small-model parity test.

## Coverage

The regression suite covers production and proxy parameter contracts, complete
optimizer membership, return horizons and terminal handling, current-position
return alignment, valid zero versus absent conditioning, zero initialization,
independent masking, calibration freeze, and calibration hash validation.

Adaptive RMSNorm tests verify the fixed action order, cumulative earlier-group
conditioning, normalization before modulation, no second normalization,
gradient flow to every earlier group, and unchanged optimizer roles. Nonzero
conditioning is compared across teacher forcing, stepwise decoding, compiled
inference, tracing, sampling, and committed prefixes. The trigger-to-button
legality mask remains active in every path.

Resume tests cover the next loader batch, identity and return masks, prefix RNG,
calibration state, gradients, and the next optimizer update. Checkpoint format 4
and experiment identity v5 reject all older O59 checkpoints.

The B200 gate compared each rejected execution implementation with the same
saved state and real replay bank before measurement. It checked inactive and
active AWR, every parameter, gradient, update delta, mask, prefix draw, RNG, and
conditioning state. After no implementation passed the three-window rule, the
candidate branches, embedded benchmark command, launcher, profiler hooks, and
continuation-only diagnostics were removed. The final focused and repository
checks ran against that cleaned source.

An initial sandboxed focused run could not access CUDA and reported 55 passes
and four skips. The required rerun used CUDA access and passed all 70 focused
tests. The repository unit suite also ran outside the restrictive sandbox
because FastAPI's test client had previously hung there.
