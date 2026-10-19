# Batched evaluation validation

All commands run locally in `hal-o65-fasteval`, with `OMP_NUM_THREADS=2`.
Pytest basetemps and logs use `$TMPDIR` or `runs/fasteval/`. Heavy jobs run
sequentially. No remote machine or main-worktree file changes are made.

## Required checks

| Command | Result |
| --- | --- |
| `uv run ruff format --check .` | Passed; 377 files formatted |
| `uv run ruff check .` | Passed |
| `uv run ty check --python-version 3.14 --error-on-warning hal experiments/059_muon_action_sequence.py scripts experiments/065_selfplay_ppo.py experiments/o65/pilot.py tests/test_batched_self_play_integration.py` | Passed; zero diagnostics |
| `uv run pytest -q -m 'not integration' --basetemp "$TMPDIR/pytest-handoff"` | 2,488 passed, 11 skipped, 85 deselected, 27 warnings; 520.80 s |
| `HAL_REQUIRE_INTEGRATION=1 uv run pytest -q -rs tests/test_roundtrip.py tests/test_session_cleanup.py -m integration --basetemp "$TMPDIR/pytest-required-integration"` | 8 passed, 6 deselected, 7 warnings; 56.88 s; no skips |

The type check adds both modified experiment files and the new integration
test to the required targets. The profiler is included by the `scripts` target.
Integration and later focused commands also set
`TORCHINDUCTOR_COMPILE_THREADS=1`.

## Focused checks

The first evaluation test run passed 29 tests. The expanded run of
`tests/test_o65_evaluation.py`, `tests/test_o65_pilot.py`, and
`tests/test_065_loop.py::test_periodic_metrics_preserve_unscored_budget_matches`
passed 41 tests in 29.02 s. The two-action request trial passed 31 evaluation
tests in 27.49 s; that trial is removed after its throughput comparison.

The initial command
`HAL_REQUIRE_INTEGRATION=1 uv run pytest -q -s tests/test_batched_self_play_integration.py -m integration --basetemp "$TMPDIR/pytest-batched-integration"`
passed in 149.16 s. It compared all sampled factors, submitted actions, native
frame bytes, and seven stockouts through resets. Its maximum log-probability
gap was 0.0000038147 nats. The final test adds one-match simulator replay and
retains the separate timeout/frame-budget cases. Rerunning it with `-q -s -rs`
and basetemp `$TMPDIR/pytest-batched-integration-final` passes in 161.23 s,
with the same gap and seven identical stockouts. There are no skips.

An end-to-end `scripts.eval_o65` invocation with both proxy paths, simulator
`msl`, 512 frames, and opponent cadence 10/2 completes four matches and writes
`hal.o65.evaluation-results.v2`. A run of the final profiler repeats the
four-match MSL measurement at 482.31/s. Its eight-worker Dolphin run with the
pilot's 1,924-frame budget completes at 360.15/s. These runs have no failures
or dropped matches.

## Unit skips

The full unit command leaves `HAL_O65_PROXY_CHECKPOINT` unset. The three
proxy checks therefore skip. Rerunning all eleven skipped node ids with
`HAL_O65_PROXY_CHECKPOINT=runs/proxy/final.pt HAL_O65_COMPILE_CUDA=1`
and `pytest -q -rs` gives **3 passed, 8 skipped** in 11.46 s.

The three passing checks are:

- `test_proxy_muon_repeated_fit_on_long_history`.
- `test_real_bootstrap30_checkpoint_conditioning_and_cache_parity[dtype0]`.
- `test_real_bootstrap30_checkpoint_conditioning_and_cache_parity[dtype1]`.

The remaining skips are:

- `tests/test_netplay_hardware.py`: two opt-in 059 hardware qualification tests.
  They require `HAL_REQUIRE_NETPLAY_HARDWARE_QUALIFICATION=1` and a qualified
  059 policy bundle. This O65 change does not qualify those profiles.
- `tests/test_policy_world_schema.py`: all six tests require
  `data/processed/ranked-anonymized-1/mds-v7-sub4/train`. That fixture is absent
  from the provided read-only data tree. These tests remain unqualified.

## Failures and fixes

- Initial focused lint/type checks found an unused import, missing schema
  imports, a fixed-tuple annotation error, and nested context-manager style.
  Those checks pass after correction.
- The first commit hook failed because the local environment lacked
  `cairosvg`, and an untracked profiler referred to helpers outside that
  staged commit. Installing CairoSVG 2.9.1 in this worktree's virtual
  environment and keeping the commit dependencies coherent resolved the
  failures. No dependency manifest or lockfile changes are made. Both
  implementation commits pass all hooks.
- The first 256-match profile failed with CUDA OOM while copying the initial
  caches for graph capture. Removing that empty-state copy lets the repeat
  finish at 1,602 match-frames/s with 7.98 GiB peak reserved GPU memory.

## Review scope

The review covers `git diff o65-selfplay-ppo-plan...HEAD`, including the inherited
O65 implementation, with manual checks of the new evaluation code, the cache,
decoder, representation, and unchanged PPO mask/detach boundaries. The first
two implementation commits add 864 lines and remove 82 across ten files from
`e6d768d5`. The full range and stream delta are saved under `runs/fasteval/`.

Review fixed the capture-memory error, preserved float64 score arithmetic,
updated the two versioned-result consumers, and strengthened the simulator
reference with serial replay. The second pass found no further code defects.
Missing v7 fixture coverage and production-GPU measurements remain open.
