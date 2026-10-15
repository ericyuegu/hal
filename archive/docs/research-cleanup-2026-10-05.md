# Research working-tree cleanup

The cleanup starts at `2169049a` on `cleanup/research-work`. Main and the older
research branches remain unchanged. No cloud resources were started, modified,
or deployed. Nothing was pushed.

## Commit order

1. `7422c526`: G4 training launcher validation and Docker startup setup.
2. `3e26f2ac`: model support for omitted return conditioning.
3. `3d806427`: O60 v2 training, scientific contracts, and qualification tools.
4. `cc8dfd5d`: main-stick notebook and its referenced figures.
5. `e715c4cd`: explicit controller layouts and action-trace vocabulary sizes.
6. `076a6067`: separate O61 pruned81 treatment and contract tests.
7. `bd730641`: O59 control-timing evaluation, recovery, and analysis.
8. `506f5797`: O59/O39 scaling comparison and source metric snapshots.
9. Historical evaluation archival and this handoff record.

The O60 v2 changes are scientific changes from `bb67dbdc`, not a refactor of
v1. `experiments/o60/README.md` explains the separate `o60-plain-bc` identity.
The old branches were not merged wholesale or rewritten.

## Preservation

Before edits, the 73 modified or untracked files were copied to
`/tmp/hal-research-before`, with a SHA-256 manifest. After organization, each
original file has the same bytes at its original or archived path except:

- `.gitignore` now excludes local `outputs/` state.
- `notebooks/039_scaling_viz.py` reads O59 evidence from a tracked path.

`archive/research-2026-09/sha256.json` records the original paths and hashes of
35 archived files. Local logs, bytecode, and the plain-BC launch state were not
committed or deleted. The archive README identifies historical code that uses
old imports and temporary paths. It is not a supported runtime.

The scaling evidence and its three metric snapshots are tracked under
`experiments/o59/scaling-comparison/`. Their source hashes were verified. The
existing 64 MB O39 outcome CSV remains local; its hash and the frontier CSV hash
are recorded in `local-inputs.json`. Figures remain generated local files.

## Validation

Focused checks passed:

```sh
uv run pytest -q tests/test_launch_gce.py
# 5 passed
uv run pytest -q tests/test_action_sequence_model.py tests/test_compute_optimal_action_sequence.py tests/test_compute_optimal_action_sequence_distributed.py
# 10 passed, before applying the separate O61 changes
uv run pytest -q tests/test_controller_codec.py tests/test_action_sequence_model.py tests/test_action_trace.py tests/test_compute_optimal_action_sequence.py tests/test_pruned81_action_sequence.py
# 35 passed
uv run pytest -q tests/test_control_timing.py tests/test_control_timing_analysis.py tests/test_control_timing_launch.py tests/test_control_timing_recovery.py
# 55 passed
uv run ruff format --check .
# 321 files already formatted
uv run ruff check .
# passed
uv run ty check --python-version 3.14 --error-on-warning hal experiments/059_muon_action_sequence.py scripts
# passed
MPLCONFIGDIR=/tmp/hal-matplotlib MPLBACKEND=Agg uv run --no-sync --with pandas notebooks/039_scaling_viz.py
# passed; regenerated figures; O39 fit retains 35 points and RMS residual 0.0046
git diff --check
# passed
```

The full non-integration suite passed:

```sh
uv run --no-sync pytest -q -m 'not integration'
# 1783 passed, 2 skipped, 24 deselected, 26 warnings in 162.36 seconds
```

The two skips are `tests/test_netplay_hardware.py`: profile latency and absence
of compilation during play. These tests require explicit hardware qualification
and a qualified policy bundle. Warnings include existing WebSocket and
multiprocessing deprecations. The log is `/tmp/hal-research-pytest.log`.

The required integration command is:

```sh
HAL_REQUIRE_INTEGRATION=1 uv run pytest -q tests/test_roundtrip.py tests/test_session_cleanup.py -m integration
```

The first run had 7 passes and one failure in
`test_fresh_recording_roundtrip_bit_exact`. The recorded and replayed game
diverged at frame 128, with a seed mismatch from frame zero. The log is
`/tmp/hal-research-integration.log`.

The failed test passed in isolation on the unchanged base commit `2169049a`,
extracted under `/tmp/hal-research-base-2169049a`, using the same fixtures and
Python environment. It also passed in isolation on this branch:

```sh
HAL_REQUIRE_INTEGRATION=1 uv run pytest -q tests/test_roundtrip.py::test_fresh_recording_roundtrip_bit_exact -m integration
# 1 passed in 9.29 seconds
```

The base comparison used `.venv/bin/python -m pytest` with `PYTHONPATH` set to
the extracted base, and `HAL_ISO_PATH`, `HAL_EMULATOR_PATH`, and
`HAL_DEV_MDS_DIR` pointing to the original local fixtures. It passed in 9.37
seconds. Its log is `/tmp/hal-research-baseline-integration.log`; the cleanup
branch's isolated log is `/tmp/hal-research-roundtrip-retry.log`. The relevant
test, simulation, wire, and controller-input source is unchanged from the base.
The cause of the first failure is not established.

A single repeat of the full integration command passed: **8 passed, 6
deselected, 7 warnings in 57.85 seconds**. The repeat log is
`/tmp/hal-research-integration-repeat.log`. The first failure remains recorded
above; the repeat did not reproduce a test-order effect.

Commit hooks ran format, lint, and type checks for code commits and passed.
They skipped notebook/image-only changes because no applicable Python files
were staged. Archive files are excluded by the existing hook configuration.

Additional checks used `git status`, `git diff`, `git log`, source reads, file
inventories, a credential-pattern scan, and Python SHA-256 comparisons. The
archive manifest and scaling source hashes matched. The credential-pattern
scan found no matches; it is not a general secret audit.

## Failed attempts and skips

- The first full integration run failed as described above. No assertion was
  weakened, no fixture was replaced, and no failure was converted to a skip.
- Sandboxed uv commands sometimes failed to create a lock in
  `/home/ericgu/.cache/uv`. This affected the first O60 test attempt, the first
  plot attempt, and intermediate Ruff attempts. Authorized reruns passed.
  Direct `.venv/bin/ruff` checks also passed.
- The first plot run with cache access failed because `pandas` was absent.
  The successful command above used an isolated uv dependency overlay.
- One file search named an absent archive directory, and one shell search
  used an unmatched test-file glob. Corrected source searches completed.
- No live G4 qualification, two-GPU training qualification, cloud deployment,
  network match, or training run was attempted. Local tests do not establish
  production performance or experimental success.
- Worker/frontend tests were not run because this cleanup changes neither.
