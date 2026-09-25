# Prelaunch validation

Checks ran in the isolated `/tmp/hal-o60` worktree. `PYTHONPATH=/tmp/hal-o60`
selects that source. `UV_NO_SYNC=1` uses the existing Python environment;
`UV_CACHE_DIR=/tmp/hal-uv-cache` avoids a read-only default cache.
A separate `uv sync --locked` attempt failed with disk quota exceeded while
downloading dependencies. No dependency or lockfile change was made.

Final checks:

- `uv run ruff format --check .`: 362 files passed.
- `uv run ruff check .`: passed.
- `uv run ty check --python-version 3.14 --error-on-warning hal experiments/051_muon_parameterization.py scripts/cache_modal_fixtures.py scripts/launch_gce.py scripts/launch_modal.py scripts/launch_vast.py scripts/replay_policy_fault.py`: passed, zero diagnostics.
- `uv run pytest -q --ignore=tests/experiments -m "not integration" -rs`: 1,215 passed, 23 deselected, zero skips.
- `HAL_REQUIRE_INTEGRATION=1 uv run pytest -q tests/test_roundtrip.py tests/test_session_cleanup.py -m integration`: seven passed, two deselected, zero skips.
- `uv run pytest -q tests/experiments/test_060_plain_bc.py tests/test_o50_parity.py -rs`: 18 passed, zero skips.
- Earlier focused loader and inference coverage: 57 passed, zero skips.

The first focused run had a module-name collision. Later test corrections used
the AWRBatch wrapper's replay IDs, the public portable-config signature,
floating-point tolerance for differently ordered arithmetic, and the production
optimizer/scheduler restore order. The final resume test is bit-exact.

The first repository suite had two failures and 13 skips. The layout test
rejects every importlib import; environment capture now uses `uv pip freeze`.
The local-launcher cleanup test timed out with missing node_modules in the
worktree. The existing node_modules and required data/checkpoint fixtures were
linked into the isolated worktree. The full rerun passed without skips.
Ruff initially classified wandb differently because the isolated worktree did
not have the ignored wandb directory. Restoring that directory made the existing
repository import grouping consistent; frozen experiments were not edited.

The recovered O52 loader and current loader produced identical replay IDs and
tensors over 300 deterministic fixture batches, seed 0. `loader-parity.json`
records the source hash and output digest. This exercises ring rollover and
refill on a small fixture; it is not a claim that the full corpus was loaded
locally. Batch construction, selection, masking, and validation-loader
functions are AST-identical to O52. Full allocated initialization is bit-exact
at 14,480,922 parameters. The new portable identity has offline/live model
parity coverage.

The comparison script reproduces both persisted control metrics from all 96
boot rows. A control-versus-itself check returns zero differences and zero-width
intervals. Focused tests check paired resampling, difference direction, and
rejection of incomplete or changed evaluations.
