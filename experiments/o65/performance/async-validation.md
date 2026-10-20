# O65 async validation

Runtime source: `974a9ca0`. All temporary directories and logs are in
`$TMPDIR=/home/ericgu/data/scratch/o65-tmp/async` or `runs/async/`.
GPU jobs ran sequentially for the clean comparison. Initial concurrent profiles
remain diagnostic only. No cloud job was launched.

## Handoff commands

The final run sets `TORCHINDUCTOR_COMPILE_THREADS=1` and
`HAL_O65_PROXY_CHECKPOINT=$PWD/runs/proxy/final.pt`. Basetemps are explicit under
`$TMPDIR`. Raw command logs are in `runs/async/final-*.log`.

```sh
uv run ruff format --check .
uv run ruff check .
uv run ty check --python-version 3.14 --error-on-warning \
  hal experiments/059_muon_action_sequence.py scripts
uv run ty check --python-version 3.14 --error-on-warning \
  experiments/065_selfplay_ppo.py tests/test_compile_cache.py \
  tests/test_o65_async.py tests/test_o65_async_trace.py
uv run pytest -q -ra -m "not integration" \
  --basetemp "$TMPDIR/pytest-final-handoff"
HAL_REQUIRE_INTEGRATION=1 uv run pytest -q -ra \
  tests/test_roundtrip.py tests/test_session_cleanup.py -m integration \
  --basetemp "$TMPDIR/pytest-final-required"
HAL_REQUIRE_INTEGRATION=1 uv run pytest -q -ra \
  tests/test_o65_async.py tests/test_native_sim_integration.py \
  tests/test_065_loop.py::test_native_shard_close_releases_batch_after_copy_failure \
  -m integration --basetemp "$TMPDIR/pytest-final-native"
```

Formatting, lint and both type-check scopes pass with zero diagnostics. The final
non-integration run reports **2,730 passed, eight skipped, 91 deselected** in
405.55 s. Six missing-fixture skips count as validation failures even though
pytest returns zero. Required Dolphin integration reports **8 passed, 6 deselected** in 57.76 s.
Native/async/cleanup integration reports **21 passed, 3 deselected** in 88.02 s.
Neither integration invocation skipped a test. All seven command exit codes are
zero; the six v7 fixture skips still fail the project fixture requirement.

## Focused checks and earlier runs

All pytest commands use `uv run pytest -q`, the named test targets and an
explicit scratch basetemp. Integration commands set `HAL_REQUIRE_INTEGRATION=1`.
The log stem identifies each invocation under `runs/async/`.

| Log stem | Targets / mode | Result |
| --- | --- | --- |
| `baseline-tests` | `test_o65_async.py -m integration` at instrumentation commit | 1 passed |
| `loop` | `test_065_loop.py -m "not integration"` | 87 passed, 1 integration deselected; includes CPU/CUDA Muon resume |
| `cache-tests` | `test_compile_cache.py test_o65_modal.py` | 42 passed |
| `cache-tests2` | same targets after artifact-count metadata | 42 passed |
| `cache-review` | same targets with malformed-manifest cases | 46 passed |
| `transfer-cpu` | `test_o65_async.py -k "parity and cpu" -m integration` | 1 failed: compared relocated native snapshot bytes across independent batches |
| `transfer-cpu2` | same | 1 failed: cross-batch pointer bytes still differ after restore |
| `transfer-cpu3` | same, diagnostic snapshot dump | 1 failed; headers match except payload hash, raw payload contains relocated pointers |
| `transfer4` | `test_o65_async.py -m integration` | 3 passed |
| `transfer-final` | same | 3 passed, 1 unit test deselected |
| `host-release` | `test_o65_async.py -m "not integration"` | 1 passed, 3 integration tests deselected |
| `overlap-fix` | CPU overlap regression plus failed-tick cleanup | 2 passed |
| `transfer-slices` | async integration with four shards and FP32 bit test | 4 passed, 1 unit test deselected |
| `tick-trace-tests` | `test_o65_async_trace.py` | Collection failed: `scripts` is not on pytest's import path |
| `tick-trace-tests2` | same, using the repository's explicit file-loader convention | 3 passed |
| `capture-event` | `test_cuda_graph_preparation.py` | 7 passed |
| `benchmark-provenance` | Torch-free benchmark integration | 1 passed |
| `host-failure` | async unit tests after exceptional host futures | 2 passed, 5 integration deselected |
| `host-failure-integration` | async integration after exceptional host futures | 5 passed, 2 unit tests deselected |
| `host-interrupt` | async unit tests including KeyboardInterrupt | 3 passed, 5 integration deselected |
| `integration-required` | required Dolphin command | 8 passed, 6 deselected |
| `integration-native` | async and native integration before final failure-signal change | 19 passed, 1 deselected |

Initial benchmark invocation failed because `melee_sim` was not on
`PYTHONPATH`; the documented environment fixes it. One edit command used bare
`python`, which is unavailable; subsequent commands use `uv run python`.
Initial Ruff import/unused-variable findings and one capture-report union type
error were fixed; all later static checks pass.

The first broad run (`pytest-handoff`) was interrupted to constrain compiler
memory use: **366 passed, one failed, 89 deselected**. Its real failure was the
CPU overlap regression; it was fixed in `f2901c17`. Compiler workers are limited
to one in subsequent runs.

The next broad run (`pytest-final`) reported **2,722 passed, one failed, ten
skipped, 90 deselected**. The failure was an old mocked capture test that lacked
a mock for the new warmup event; `0577890f` fixes that test. Two proxy-checkpoint
skips were removed by setting the existing checkpoint path. The other skips
were the six absent v7 fixture cases and two production hardware gates.

The following broad run (`pytest-handoff3`) reported **2,728 passed, eight
skipped, 90 deselected**. Six skips are validation failures:
`tests/test_policy_world_schema.py:37`, missing local v7 subset. The two remaining
skips are `tests/test_netplay_hardware.py:52` and `:68`, disabled production GPU
qualification tests. The last handoff run repeats the suite after the copy-error
and interruption cleanup changes.

## Measurement and resume checks

All six clean/full-trace profile processes exited zero:
`before16-clean`, `before64-clean`, `after16-cold`, `after16-warm`,
`after64-new-shapes`, `after64-warm`. Each Nsight export and complete attribution
analysis passed. Tick analysis also passed on the prior floor's real SQLite
trace: 128 ticks and two boundaries. The artifact manifest records raw files,
sizes and SHA-256 hashes.

`native-resume16` exited zero. Its v5 report passes every state family. All ten
state hashes match the boundary on each of three restores; the next update
matches the existing two-control RMS tolerances. No tolerance was changed.
Raw report: `runs/async/native-resume16/resume-check.json`.

The first CPU parity assertions were corrected, not weakened numerically:
raw native arenas contain relocated pointers. Tests compare every rollout and
frame-store tensor and the pool snapshot across paths, then require exact
snapshot roundtrip within each native batch. Both CPU and CUDA pass through
staggered ends and resets. Controller transfers preserve every FP32 bit,
including negative zero. Failed transfers do not advance the affected native
batch.

Final artifact verification: **70 recorded SHA-256 hashes match**. The current
runtime compilation identity matches both persisted-cache measurement processes.
All evidence floats are finite.
