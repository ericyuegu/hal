# Higher-u evaluation validation

Validation uses Python 3.14.4 and PyTorch 2.11.0+cu130. Commands use
`UV_CACHE_DIR=/tmp/hal-uv-cache UV_NO_SYNC=1` and the installed environment.
CUDA checks run on the local RTX 3060 with host device access.

| Command | Result |
| --- | --- |
| `uv run ruff format --check .` | Passed; 342 files |
| `uv run ruff check .` | Passed |
| `uv run ty check --python-version 3.14 --error-on-warning hal experiments/059_muon_action_sequence.py experiments/063_percentile_conditioning.py scripts` | Passed; zero diagnostics |
| `uv run pytest -q -m "not integration" -x --tb=short -rs` | Passed; 2,162 tests, eight skips, 65 deselections, 26 warnings, 175.72 seconds |
| `uv run pytest -q tests/experiments/test_063_percentile_conditioning.py tests/test_launch_modal.py -k 'eval or broker or target'` | Passed; 28 tests, 87 deselected |
| `uv run pytest -q tests/test_launch_modal.py` | Passed after both fixture-scope fixes; 54 tests |

The eight skips are two production-GPU qualification tests and six tests whose
optional local v7 subset is absent. No datasets were downloaded to satisfy them.
Warnings concern existing eager FlexAttention diagnostics and multiprocessing
fork calls in a threaded process.

The first lint pass on fixture detection found a missing explicit `strict=` in
adjacent-pair `zip`. It was corrected to `strict=False`; final lint and type
checks passed. Earlier full suites also passed before the final fixture-scope
changes; the 2,162-test result above covers the final runtime changes.

Read-only review checked all new target mappings, separate evidence paths,
checkpoint compatibility, unchanged training behavior, and both image-build and
worker-startup fixture scope. It found a runtime fetch-all path after the first
image-only fix. That path was fixed before any worker was launched. Regression
tests assert that evaluation images and worker preparation use only the ISO and
Dolphin fixtures. Training commands retain their existing fixture behavior.

The first six Modal submissions requested 128 GiB scratch and were rejected
before worker creation; Modal requires at least 512 GiB. Corrected requests use
512 GiB. Failed attempt identities and log hashes are in
[failed_launches.json](failed_launches.json).


After the jobs completed, W&B chart history was missing some points from the
parallel shared writers, and the original run states were left running. The
validated raw metrics were republished through one primary writer per original
run, with metadata and system-stat collection disabled. Both runs were finished
again. This was a logging reconciliation; no gameplay was rerun. See
[wandb_reconciliation.json](wandb_reconciliation.json). Parallel manual CLI
evaluations need this collection step until shared logging has separate backend
coverage; the original inline evaluator is unchanged.
