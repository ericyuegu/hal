# O63 validation

Checks run on 2026-10-17 with Python 3.14.4 and PyTorch 2.11.0+cu130.
CUDA tests use the local RTX 3060. Commands use `UV_CACHE_DIR=/tmp/hal-uv-cache`
and `UV_NO_SYNC=1` to use the installed locked environment. GPU and emulator
checks run with host device access.

| Command | Result |
| --- | --- |
| `uv run ruff format --check .` | Passed; 340 files |
| `uv run ruff check .` | Passed |
| `uv run ty check --python-version 3.14 --error-on-warning hal experiments/059_muon_action_sequence.py experiments/063_percentile_conditioning.py scripts` | Passed; zero diagnostics |
| `uv run pytest -q -m "not integration" -rs` | Final run in progress |
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
validation and CUDA test precision errors above. The final suite reruns all
checks after those fixes and the ranked-only loader adaptation.

Warnings in successful focused checks: PyTorch warns that the eager debug
FlexAttention path materializes attention scores. Integration checks also emit
the existing Python multiprocessing warning about fork in a threaded process.
