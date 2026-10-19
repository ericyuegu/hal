# GPU profile validation

Code tested: `137f2afe`. No shared library, model, simulator, optimizer, loss,
or persisted training format changed in this task. Only the O65 trainer,
its profiling tests, and performance evidence changed.

## Final commands

GPU-capable commands held
`flock /home/ericgu/data/scratch/o65-tmp/gpu.lock` and ran through
`systemd-run --user --scope -p MemoryMax=10G`. The final environment used:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCHINDUCTOR_COMPILE_THREADS=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH=.:/home/ericgu/src/melee-sim-light
export MSL_CORE_LIBRARY=/home/ericgu/src/melee-sim-light/build/melee_core/python/libmelee_core.so
```

| Command | Result |
| --- | --- |
| `uv run ruff format --check .` | Passed; 380 Python files. |
| `uv run ruff check .` | Passed. |
| `uv run ty check --python-version 3.14 --error-on-warning hal experiments/059_muon_action_sequence.py scripts experiments/065_selfplay_ppo.py tests/test_o65_gpu_profile.py` | Passed; zero diagnostics. |
| `uv run pytest -q -rs -m "not integration" --basetemp runs/ut`, with `HAL_O65_PROXY_CHECKPOINT=$PWD/runs/proxy/final.pt HAL_O65_COMPILE_CUDA=1` | 2,547 passed, 8 skipped, 89 deselected, 26 warnings; 323.85 s. |
| `HAL_REQUIRE_INTEGRATION=1 uv run pytest -q tests/test_roundtrip.py tests/test_session_cleanup.py -m integration --basetemp "$TMPDIR/handoff-integration"` | 8 passed, no skips, 6 deselected, 7 warnings; 56.87 s. |
| `uv run pytest -q tests/test_o65_gpu_profile.py tests/test_065_math.py --basetemp "$TMPDIR/profile-final"` | 54 passed, no skips; 7.05 s. |
| `uv run python experiments/065_selfplay_ppo.py profile --checkpoint runs/proxy/final.pt --data-dir /home/ericgu/src/melee-sim-light/data --output runs/gpu-profile-64-final --cpus 0 1 2 3 --matches-per-shard 16 --groups 1 --warm-updates 4` | Passed; 64 total matches, four initial updates, one timed update, one bounded trace update. Timed update: 10.090 s. |
| Committed-source hash against `resolved.json` | Exact match: `890ed82b42601cfa2e14d2443e58159393531cf3be2184aa541e6900de61afcb`. |
| `git diff --check o65-selfplay-ppo-plan...HEAD` | Passed. |

The eight final skips were:

- `test_declared_profiles_meet_complete_path_budget` and
  `test_prepared_profiles_do_not_compile_or_capture_during_play` in
  `test_netplay_hardware.py`: production netplay qualification was not enabled.
- All six tests in `test_policy_world_schema.py`: the local v7 subset at
  `data/processed/ranked-anonymized-1/mds-v7-sub4/train` is unavailable. These
  checks remain unvalidated; no fixture was substituted.

The supplied O63 checkpoint tests ran with the explicit path and CUDA graph
flag. Required roundtrip and cleanup fixtures were present; integration had
no skips. Warnings include Python's multithreaded-fork warning and existing
websocket deprecations. The profiling run also emits Torch's warning about
clearing profiler events between cycles; each trace intentionally contains
only its own sample.

## Earlier failures and corrections

- The first focused pytest command failed collection for two modules because an
  `experiments.o65` helper import was not available on the test import path.
  The profiler was moved into the trainer. Subsequent focused runs passed:
  50, 53, and finally 54 tests as regression coverage was added.
- Early `ruff check` runs reported SIM117 and SIM102 violations. They were
  corrected; final repository checks pass.
- The first four-active-match profiling smoke command failed on a copy event:
  Torch 2.11 uses lowercase `bytes`. The parser was corrected and tested.
  A repeat smoke command succeeded, but its synchronization summaries were
  superseded by the annotation-based exclusions in the final collector.
- The first full-suite attempt was stopped locally with exit 143 after
  Inductor spawned 12 compiler workers. The retry limited compilation to one
  worker. No VM service was stopped.
- That retry returned 4 failures, 2,539 passes, 11 skips, 89 deselections,
  and 30 warnings. The odd-reset resume test rejected missing
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`. The overlay-restart test and both stream
  supervisor tests failed because the long basetemp made the Unix socket path
  exceed its length limit. The corrected final command used `runs/ut` and
  the required CUDA environment. All four tests passed there. Three proxy
  skips were also removed by setting the available checkpoint path.
- A queued isolated resume retry with the same missing environment was
  cancelled before it started, exit 143; the corrected full suite superseded it.
- An early local qualification used 64 active matches across four groups,
  or 256 total pool matches. The final qualification uses one group and 64
  total matches. Only the latter supplies the published measurements.

Raw logs remain under `runs/gpu-profile-*.log`.

## Self-review

Reviewed the task diff and the plan-base comparison, with attention to the
math and the touched rollout, replay, critic, and actor paths. Profiling does
not change delays, offsets, masks, factor order, detach placement, selection,
or optimizer ordering. The focused parity test checks two successive batches,
model states, and both optimizer states with timing enabled on the CPU path.
Existing GPU queue, resume, checkpoint, and representation checks also passed
in the final suite.

The substantial findings were measurement defects: event spans include idle
gaps; CPU operator durations double-count device work; graph launches differ
from kernel executions; profiler cleanup adds synchronizations; fixed launch
counts must scale by calls while batch work scales by rows. The implementation
and regression tests address these distinctions. Missing kernels and missing
required trace sites are rejected. Partial trace coverage and consecutive
policy versions are explicit in the report. The second review found no further
required code changes.

The final 64-match collection uses the committed source directly. The earlier
uncommitted collector and four-group measurements are not the published
measurement. The RTX PRO 6000 remains unmeasured because netplay was active
through the final readiness check at 08:05 UTC. No container was started on
that VM, no service was changed, and no uploads or pushes occurred.
