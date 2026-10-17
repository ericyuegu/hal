# O63 validation

Final commit review on 2026-10-17 reran formatting, lint, and the expanded type
target successfully. The full suite passed with **2,162 tests, eight skips,
65 deselections, and 26 warnings in 171.04 seconds**. The skips are the same two
production-GPU checks and six optional local v7-fixture checks listed below.
The sandbox attempt first failed with a temporary-disk quota error during
archive extraction; the first host rerun also hit that quota while saving a
test checkpoint. After removing disposable output from completed and failed
pytest runs, the host rerun passed with local CUDA access. No dataset was
downloaded for validation. Required integration results below remain unchanged;
the final cleanup only corrects historical source wording in the report.

Checks run on 2026-10-17 with Python 3.14.4 and PyTorch 2.11.0+cu130.
CUDA tests use the local RTX 3060. Commands use `UV_CACHE_DIR=/tmp/hal-uv-cache`
and `UV_NO_SYNC=1` to use the installed locked environment. GPU and emulator
checks run with host device access.

| Command | Result |
| --- | --- |
| `uv run ruff format --check .` | Passed; 340 files |
| `uv run ruff check .` | Passed |
| `uv run ty check --python-version 3.14 --error-on-warning hal experiments/059_muon_action_sequence.py experiments/063_percentile_conditioning.py scripts` | Passed; zero diagnostics |
| `uv run pytest -q -m "not integration" -x --tb=short -rs` | Passed; 1,961 tests, 8 skipped, 39 deselected |
| `HAL_REQUIRE_INTEGRATION=1 uv run pytest -q tests/test_roundtrip.py tests/test_session_cleanup.py -m integration` | Passed; 8 tests, 6 deselected, no skips |
| `uv run pytest -q tests/experiments/test_063_percentile_conditioning.py tests/test_launch_modal.py` | Passed; 103 tests, no skips |

Focused coverage includes return alignment against frame rewards, all three
availability masks, the warmup boundary, the actual checkpoint evaluation
input, evidence separation, offline validation inputs, detached gradients,
shared parameter initialization, optimizer membership, parameter and FLOPs
contracts, label uniformity and ties, and exact CPU/CUDA next-update resume.
The real buffered loader test checks the next batch and optimizer update in all
three arms after a drained boundary. A separate test checks the labeler state
in the production boundary checkpoint writer.

Earlier checks found and resolved these errors before launch:

- The initial production head count was low by 512 parameters. The verified
  count is 1,081,921, and the production total is 247,944,126.
- A gradient test incorrectly required equal return-conditioner gradients for
  different conditioning inputs. It now checks all shared non-conditioner
  gradients, including the value head, with equal initialized weights.
- Checkpoint validation tried to restore a CPU label RNG into a CUDA generator.
  Inference now validates the state structure without restoring a training RNG.
  Exact training resume restores it on the training device.
- The CUDA test configuration used FP32 with the required BF16 flash-attention
  path. The device test now uses the training AMP configuration.
- Review found one checkpoint evaluation call still reading the raw-return
  calibration. It now maps each target through `desired_return_for_target`.
  A regression test calls that checkpoint evaluation path for all four targets.

The first full suite ran against the pre-fix imports: 1,938 passed, 6 failed,
8 skipped, and 35 deselected. The six failures were the CPU/CUDA generator
validation and CUDA test precision errors above. The final suite passed after those fixes and the ranked-only loader adaptation.

Warnings in successful focused checks: PyTorch warns that the eager debug
FlexAttention path materializes attention scores. Integration checks also emit
the existing Python multiprocessing warning about fork in a threaded process.

The final full-suite skips are explicit:

- Two tests in `tests/test_netplay_hardware.py` require the production GPU and
  `HAL_REQUIRE_NETPLAY_HARDWARE_QUALIFICATION=1`.
- Six tests in `tests/test_policy_world_schema.py` require an optional local v7
  subset that is absent on this host.
- All eight required roundtrip and session integration tests ran. No required
  integration fixture was missing.


GCE startup found an existing launcher defect: the encoded credential list did
not end with a newline. Bash `read` omitted its last record, the W&B key. The
launcher now terminates every record. `uv run pytest -q tests/test_launch_gce.py`
passes all six tests, including a Bash reproduction of the last-record read.
Formatting, lint, and the expanded type target pass after the fix.

The first full-suite rerun after that fix exited early with a closed-stderr
error while a scratch quota error was also observed. Its failure is recorded
rather than counted as a successful test run. Completed temporary test
checkpoints and redundant cloned Git packs were removed. The clean rerun passed: 1,954 tests, 8 skips, and 35 deselections in 171.56 seconds.


The actual `eval_checkpoint` path also passed a CUDA/Dolphin smoke with an
untrained proxy checkpoint at u75: one 180-frame match, 90 decoder calls, zero
crashes, zero failed or timed-out workers. Compilation took 33.67 seconds;
evaluation took 16.31 seconds. Timing was prediction 4, delay 2, replan 2.
See [closed_loop_smoke.json](closed_loop_smoke.json) for durable metrics.

Compiled CUDA next-update resume was added after review. All three arms pass
strict equality with deterministic algorithms enabled in the test. Production
flags are unchanged. A prior strict comparison with default Inductor kernels
failed on small optimizer-state differences; the generated embedding backward
kernel uses atomic reductions. The deterministic test isolates checkpoint state
restoration; it does not establish bitwise replay for default compiled kernels.
The focused compiled command was
`uv run pytest -q tests/experiments/test_063_percentile_conditioning.py -k 'training_change_and_next_update_resume and True'`:
3 passed, 57 deselected, 14 warnings in 29.61 seconds.


After the compiled resume cases were added, all required checks passed again:
format (340 files), lint, expanded type target, and the complete non-integration
suite. The latter ran 1,961 passing tests, with eight explicit skips and
39 deselections, in 169.87 seconds. The compiled tests also passed after earlier
CUDA tests, so no test-runner environment override was needed. The integration
result above remains valid: only tests and evidence changed after that run.
