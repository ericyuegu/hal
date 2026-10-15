# Small Muon attention: reproduction and validation

Worktree: `/home/ericgu/src/hal/runs/mechinterp-20260911-worktree`.
Git base: `52da5b2de1d402b86404fc4d24e382409649e525`.
All changes are new research files and tests. No tracked production file
or previous study result was edited. Existing main-checkout changes to
`experiments/052_adamw_temporal_awr.py` and `notebooks/awr_explorer.py`
were left untouched.

## Environment

Commands used the shared project environment without syncing dependencies:

```sh
export UV_PROJECT_ENVIRONMENT=/home/ericgu/src/hal/.venv
export UV_CACHE_DIR=/tmp/hal-mechinterp-uvcache
export PYTHONPATH=/home/ericgu/src/hal/runs/mechinterp-20260911-worktree
export TMPDIR=/home/ericgu/src/hal/runs/mechinterp-20260911-scratch
export HAL_ISO_PATH=/home/ericgu/src/hal/data/emulator/ssbm.ciso
export HAL_EMULATOR_PATH=/home/ericgu/src/hal/data/emulator/exiai/squashfs-root/AppRun
```

The measurement used Python 3.14.4, PyTorch 2.11.0+cu130, NumPy 2.1.3,
and the local NVIDIA RTX 3060. It used seed 20260911, batches of 16,
evaluation mode, no gradients, and disabled TF32. No Modal GPU was used.

## Primary measurement

This command completed successfully. It refuses to overwrite an existing
output directory; use a new directory for a repeat. The frozen input
package and checkpoint hashes are checked before use.

```sh
uv run --no-sync python research/mechinterp_20260911/small_muon_attention.py \
  --checkpoint-root /home/ericgu/src/hal/runs/mechinterp-20260911-scratch/small-muon-16k \
  --inventory-root research/mechinterp_20260911/artifacts/attention_small_muon_inventory \
  --input-root research/mechinterp_20260911/artifacts/attention_small_muon_inputs \
  --historical-source-root /home/ericgu/src/hal/runs/mechinterp-20260911-scratch/small-muon-source-audit \
  --output research/mechinterp_20260911/artifacts/attention_small_muon
```

All four raw checkpoints loaded with `weights_only=True`. They required
no extra pickle globals. Each stored update number is 16,384. Checkpoint
IDs, hashes, relevant state keys, tensor shapes, and tensor dtypes matched.
The traced hidden state matched the current dense path exactly, with
maximum absolute error zero for each checkpoint.

The 256 original raw replay-window hashes, data index, and data manifest
matched. The prepared input package has SHA-256
`7149b3ccbc81798ad2877e2d996cf14e51f4160f1c1a241685f6f8cca943ea33`.
The historical normalization statistics have SHA-256
`031dd0e35a7201b299933ec999615f8657dd3e616e0efd3d78b156b1920e0bef`.

The primary source archive contains the measurement, input preparation,
checkpoint reconstruction, inventory program, plan, four focused test
files, relevant HAL modules, and three historical experiment sources.
All 20 member-file hashes match their recorded values. The historical
source files also match their stated Git versions byte for byte.

Independent checks recomputed the scalar metrics using float64 and
`math.fsum`. The largest difference from the recorded means was
3.6e-15. All saved example arrays and maximum positions agree exactly
with the full probability archive. All weights are finite and nonnegative;
the largest row-sum error is below 2.6e-7. There are no exact tied maxima.

## Python checks

```sh
uv run --no-sync ruff format --check .
uv run --no-sync ruff check .
uv run --no-sync ruff check --output-format concise .
uv run --no-sync ruff check --config 'lint.isort.known-first-party=["hal", "wandb"]' .
uv run --no-sync ty check --python-version 3.14 --error-on-warning \
  hal experiments/051_muon_parameterization.py \
  scripts/cache_modal_fixtures.py scripts/launch_gce.py \
  scripts/launch_modal.py scripts/launch_vast.py scripts/replay_policy_fault.py
uv run --no-sync ty check --python-version 3.14 --error-on-warning \
  research/mechinterp_20260911/small_muon_attention.py \
  research/mechinterp_20260911/small_muon_inputs.py \
  research/mechinterp_20260911/small_muon_model.py \
  research/mechinterp_20260911/small_muon_inventory.py \
  tests/test_mechinterp_small_muon_attention.py \
  tests/test_mechinterp_small_muon_inputs.py \
  tests/test_mechinterp_small_muon_model.py \
  tests/test_mechinterp_small_muon_inventory.py
uv run --no-sync pytest -q \
  tests/test_mechinterp_small_muon_inventory.py \
  tests/test_mechinterp_small_muon_inputs.py \
  tests/test_mechinterp_small_muon_model.py \
  tests/test_mechinterp_small_muon_attention.py -rs
uv run --no-sync pytest -q --ignore=tests/experiments -m 'not integration' -rs
```

- Formatting passed: 375 files at the primary check and 377 files after
  adding the sensitivity program and its tests.
- Unmodified Ruff reported 51 existing import-order errors. The research
  worktree lacks the main checkout's ignored `wandb/` directory, which
  changes import classification. With that classification made explicit,
  the full check passed. No unrelated imports were rewritten. Focused
  Ruff checks on all eight new primary Python files passed without an
  override.
- Both listed type checks passed.
- The primary focused tests passed: 27 passed, no skips, one NVML warning,
  3.58 seconds. They cover strict checkpoint loading, historical controller
  encoding, input identity, attention masking and tracing, and metric
  definitions.
- The full primary suite passed: 1,237 passed, 17 integration tests
  deselected, no skips, 30 warnings, 122.80 seconds. Warnings concern
  existing dependency deprecations, NVML, shuffle settings, process
  forking, and an uncompiled attention path.
- After adding the sensitivity tests, the same full-suite command passed
  again: 1,239 passed, 17 integration tests deselected, no skips,
  30 warnings, 122.64 seconds. The required type check and full formatting
  check also passed again. Ruff still reported only the same 51 existing
  import-order errors; the explicit-classification check passed again.
- Separate emulator integration tests were not run: no production replay
  extraction, wire, controller, session, or offline/live representation
  code changed. The new programs inspect fixed research inputs only.

## Positional-encoding sensitivity check

```sh
uv run --no-sync python research/mechinterp_20260911/small_muon_rope_sensitivity.py \
  --checkpoint-root /home/ericgu/src/hal/runs/mechinterp-20260911-scratch/small-muon-16k \
  --input-root research/mechinterp_20260911/artifacts/attention_small_muon_inputs \
  --frozen-attention-root research/mechinterp_20260911/artifacts/attention_small_muon \
  --output research/mechinterp_20260911/artifacts/attention_small_muon_rope_sensitivity
uv run --no-sync pytest -q tests/test_mechinterp_small_muon_rope_sensitivity.py -rs
uv run --no-sync ty check --python-version 3.14 --error-on-warning \
  --exclude research/mechinterp_20260911/artifacts \
  research/mechinterp_20260911 tests/test_mechinterp_small_muon_rope_sensitivity.py
```

All three commands passed. The two focused tests passed in 2.77 seconds
with no skips or warnings. Focused Ruff checks also passed. The copied
BF16 control has zero maximum hidden-state error against the current dense
path and zero probability error against the primary saved arrays, for
all four models. All eight saved probability arrays are finite and
normalized. A separate read-only recomputation also checked that every
treatment hidden state is finite.

O26 alone restores its historical FP32 factor choice. O37 and O43 used
BF16 factors in their original code, so their FP32 results are explicitly
labeled as sensitivity treatments. The original experiment and trunk
sources were checked at each family's exact Git revision.

Sensitivity result SHA-256:
`98ea864f021d153205b001929ffc2c0faa66b6761199aa413fb4de0e605fe885`.
All-row archive SHA-256:
`aaef17046902ea24e7fc4e6d0b5011b4d2bb512ffcb7ef929451106312534f05`.
Sensitivity source archive SHA-256:
`71b97c8dda2aae8be2ceefafca5e334d9a9d76869c96198c63d97e67904d3662`.
Source-archive sidecar SHA-256:
`6029361fa82f136cc5063a0a99805a9a5ca583aa18cfb1f366f4a37d2e34e71d`.
Primary results and their source archive were not modified.

## Earlier failed checks

An initial focused pytest invocation omitted `PYTHONPATH` and failed
collection with `ModuleNotFoundError: research`. The correctly configured
invocation passed. Initial focused style/type checks also found two
nested-condition style issues and test-only type errors; those were
corrected before the primary source was frozen and measured.

One early full-suite invocation used the default temporary directory:

```sh
/home/ericgu/src/hal/.venv/bin/python -m pytest -q \
  --ignore=tests/experiments -m 'not integration'
```

That attempt reported 139 failed, 639 passed, one skipped, 17 deselected,
nine warnings, and 10 errors, followed by a pytest internal error:
`OSError: [Errno 122] Disk quota exceeded`. The failed attempt did not
retain a separate log, so its individual failure and skip details are
not available for audit. Its results are not treated as a successful test
run. The complete scratch-directory rerun is reported above and has no
failures or skips.

Only that attempt's verified temporary directory was moved, without
deletion, from `/tmp/pytest-of-ericgu/pytest-325` to
`/home/ericgu/src/hal/runs/mechinterp-20260911-scratch/failed-pytest-325`.
The files remain available. Pre-existing pytest directories were not
removed. All later tests use the explicit scratch `TMPDIR`.

A later sandbox setup also reported a temporary-storage quota error before
one report edit. A read-only storage check followed by a retry succeeded.
No additional files were moved or removed, and the final test run passed.

## Interactive figure

```sh
node --check research/mechinterp_20260911/small_muon_attention_browser_check.mjs
NODE_PATH=/tmp/policy-anatomy-browser/node_modules \
node research/mechinterp_20260911/small_muon_attention_browser_check.mjs \
  research/mechinterp_20260911/artifacts/attention_small_muon/small-muon-attention.html \
  /home/ericgu/.codex/plugins/cache/openai-bundled/visualize/1.0.29/skills/visualize/scripts/render.py \
  research/mechinterp_20260911/artifacts/attention_small_muon/browser_qa
```

Both checks passed with Playwright 1.63.0 and Chromium 153.0.8010.12.
All 2,560 displayed cells and 20 maximum markers were checked at all
eight layer settings, at widths 736 and 360 pixels, in light and dark
themes. Layer selection, hover values, touch pinning, and tooltip bounds
passed. No clipped labels, horizontal overflow, or browser JavaScript
errors were found. The desktop light and mobile dark screenshots were
also inspected directly.

The figure's example tensors are byte-identical to example zero in the
frozen probability archive. Its SHA-256 is
`e8247e75679cf2cb6441dc0ac37c3d7e254e0b56f3fc8ed19a94533060ec046c`.
Browser-check source SHA-256:
`6b459f34623a6789d00d0f2de90f08d3f243be2f307c04869d5988b0d77c0fc4`.
Browser-check result SHA-256:
`cd584b4c941c7acdbc3d926544d1bdad6c020f2556dd65c56c093f19a77785a3`.

`git diff --check` passed. The previous large-model measurement source and
interactive figure hashes remain unchanged.
