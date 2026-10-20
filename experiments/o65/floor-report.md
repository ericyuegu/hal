## Summary

Measured 1,024 matches at **11.427 s/update, 22,941 frames/s, 15,873 decisions/s**;
2,048 at **18.143 s/update, 28,898 frames/s, 20,026 decisions/s**.

Hardware rates, bare probes and the complete Nsight trace are recorded in
[floor.md](performance/floor.md). The floor remains partial. No trainer
optimization was admitted. All containers are stopped; conservative usage is
**1.853 GPU-hours and USD 11.05**, including startup/teardown and build reserves.

## Commits

Pushed only `o65-floor`: implementation and evidence commits `f1313d2e` through
`809b03cd`, plus this report. The parent is `57ed1c57`.

## Tests run

- Ruff formatting, Ruff lint and required/extended type checks: pass.
- Full non-integration suite: **2,708 passed, eight skipped**. Six missing-fixture
  skips count as validation failures; two production hardware tests are disabled.
- Required Dolphin integration: **8 passed**, no skips.
- Nsight integration regression: **1 passed**; focused measurement tests:
  **68 passed**.
- First Modal pipeline timed out during the larger sweep after completing the
  baseline, bare probes and 256 arm. Supplementary probes and the separate
  2,048 arm passed. All **79 artifact hashes** verify.

Commands, earlier failures and all skips are in
[floor-validation.md](performance/floor-validation.md).

## Self-review findings and fixes

Fixed NVTX ranges ending before GPU completion, rejected-profile cleanup,
reservation deadlines, stale metadata and unqualified Nsight acceptance.
Corrected GEMM dtype/stride accounting, critic epoch counting and cross-card
rate lookup. The second pass found no training-math change. KL thresholds remain fixed.

## Cross-stream changes

Profiling hooks in `065_selfplay_ppo.py`, the Modal launcher and their tests.
No `hal/`, MSL or other-worktree source was changed.

## Impasses

The exported payload lacks the complete reference slots. Regeneration and a
4,096-match cold warm-up exceed the remaining reservation. The required v7
fixture is absent.

## Remaining work

Measure the exact reference pass, complete non-GEMM/traffic floors and compiled
submodule attribution, then qualify and remeasure the first trainer optimization.
The current evidence does not justify a claim that any phase is within 2x of
its complete floor.
