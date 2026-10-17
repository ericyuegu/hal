# O64 validation

Validation uses Python 3.14.4 and PyTorch 2.11.0+cu130. Commands use
`UV_CACHE_DIR=/tmp/hal-uv-cache UV_NO_SYNC=1`. CUDA tests run with host access on
the local RTX 3060. Final test scratch is in
`outputs/o64-validation-tmp`, on the workspace disk.

| Command | Result |
| --- | --- |
| `uv run ruff format --check .` | Passed; 344 files |
| `uv run ruff check .` | Passed |
| `uv run ty check --python-version 3.14 --error-on-warning hal experiments/059_muon_action_sequence.py experiments/063_percentile_conditioning.py experiments/064_production_percentile.py scripts` | Passed; zero diagnostics |
| `uv run pytest -q tests/experiments/test_064_production_percentile.py` | Passed; 12 tests, including exact CPU, CUDA eager, and CUDA compiled next-update resume |
| `uv run pytest -q tests/experiments/test_064_production_percentile.py tests/test_launch_modal.py` | Sandbox CPU pass: 73 passed, two CUDA skips; separate CUDA run above covers these skips |
| `uv run pytest -q -m "not integration" -x --tb=short -rs` | Final run in progress |

Coverage checks the exact config differences from O63, the production parameter
contract, all 26 offset targets including the last available future frame, loss
coefficients and gradients, AWR application only to the first six offsets,
unchanged percentile targets and label draws, checkpoint identity rejection,
persisted offset weights, u95 defaults, and Modal target/namespace routing.

The first focused run found an incomplete test fixture: `EvalArgs` requires a
checkpoint argument. The fixture was corrected. An initial full CUDA run later
hit the host's `/tmp` quota while compiling the new backward graph; it reported
148 passed and one failure. Moving scratch to the larger workspace disk
resolved that environment failure. The focused CUDA rerun passed all 12 tests.
No dataset was downloaded for local validation.

The read-only reviewer checked source and scientific contracts. A separate
full-corpus metadata audit verified all 44 pinned train indexes, split row
identities, and actual replay lengths. It found no replay below the required
293 frames. See [corpus_length_audit.json](corpus_length_audit.json).

B200 full-batch memory and throughput qualification is pending. The preflight
source is recorded in [synthetic_preflight_source.txt](synthetic_preflight_source.txt).
