# Do different attention heads have their largest weights at different tokens?

Yes, for the saved input they often do, but some heads share a position.
In layer zero of the half-rate Muon continuation, the 16 heads have their
largest attention weights at 11 distinct key token positions. Heads 2, 9,
and 12 share position 69. In the original Muon checkpoint, layer zero has
14 distinct maximum-weight key positions across its 16 heads.

These are positions within the same input sequence, for the same most recent
query token. They are not positions averaged across different replays.

## What the figure shows

Each row is an attention head. Each dot marks the key token position with
the largest attention weight in that head. The horizontal axis runs from
key position 0, the oldest input token, to position 255, the most recent.
The query token is position 255. Each input token represents one frame.

Hover details report the actual attention weight and the other heads in
that layer whose maximum occurs at the same key position. A marker does
not mean all attention goes to that position. For example, the continuation's
layer-zero head 2 gives its maximum-weight position only 36.51% of its weight.
Other positions receive the rest.

The layer selector displays all 16 heads from one layer at a time. The
half-rate continuation is shown first, followed by the original Muon run.
Both checkpoints are from update 57,344. The initial layer is zero; the
replay is the same preselected example zero used in the earlier figures.

## Counts for this input

| Quantity | Half-rate Muon continuation, `7dgv85x8` | Original Muon, `p1fyyp1z` |
| --- | ---: | ---: |
| Distinct maximum-weight key positions in layer 0 | 11 | 14 |
| Head pairs with the same maximum position in layer 0, out of 120 pairs | 6 | 2 |
| Range of distinct maximum positions per layer | 11–16 | 12–16 |
| Distinct maximum positions across all 16 layers and 16 heads | 106 | 107 |

These are exact counts for one input, not estimates over the 256-replay
cohort. No pair of heads or layers is treated as an independent replay.
Different maximum-weight positions do not establish that heads perform
different functions or use independent information. Values at each key
position can already contain information from other frames through earlier
layers, and the model also has a residual path.

No row has an exact tie for its largest saved float32 attention weight.
The derivation rejects ties rather than assigning a position arbitrarily.
It also rejects invalid shapes, negative or non-finite probabilities, and
distributions that do not sum to one.

## Reproduction

No checkpoint was loaded and no new forward pass, training, or random
sampling was performed. The inputs are the previously frozen attention
probabilities and their metadata. The small derivation program and its
tests follow the repository's Python conventions.

```sh
UV_PROJECT_ENVIRONMENT=/home/ericgu/src/hal/.venv \
UV_CACHE_DIR=/tmp/hal-mechinterp-uvcache \
PYTHONPATH=/home/ericgu/src/hal/runs/mechinterp-20260911-worktree \
/home/ericgu/src/hal/.venv/bin/python \
research/mechinterp_20260911/attention_head_positions.py
```

The program refuses to overwrite its derived JSON or source archive.
Use a distinct `--output` directory for a repeat.

- [Plan](attention_head_positions_plan.md), SHA-256:
  `8a8573c3b7a24268fa7e55dc3f65179f716cb4518aa85ef6ebb7805824d67990`.
- [Derived data](artifacts/attention_head_positions/head_positions.json), SHA-256:
  `bfbd2bee9e8978316a128a06b904552e1653c0c73c4d48b03ad64ea99a00c835`.
- [Derivation source](attention_head_positions.py), SHA-256:
  `f71b48ede87d58772bec2b93c71fc8f006dd483a2b4f6b009f9e6aec5c67dc8b`.
- Source archive SHA-256:
  `3aa8514b337e0a42c373d1c305a2abd226a064357ef0c29e5b829f1576a55a4c`.
- Original probability NPZ SHA-256:
  `1518b5581f7beda9a431d2f4c9c7e22c6712c62a350a8fdf135f37c7d8ec38be`.

The derived data also records the original measurement JSON hash, model
export hashes, replay identity, precision, Git SHA, and environment.

## Validation

Commands ran in the research worktree with the shared project environment
and cache shown above. `uv run --no-sync` avoids changing that environment.
The full test run also used the existing Melee ISO and emulator paths and
the task scratch directory for temporary files.

```sh
uv run --no-sync ruff format --check .
uv run --no-sync ruff check . --output-format concise
uv run --no-sync ruff check . --config 'lint.isort.known-first-party=["hal","wandb"]'
uv run --no-sync ty check --python-version 3.14 --error-on-warning \
  hal experiments/051_muon_parameterization.py \
  scripts/cache_modal_fixtures.py scripts/launch_gce.py \
  scripts/launch_modal.py scripts/launch_vast.py scripts/replay_policy_fault.py
uv run --no-sync ty check --python-version 3.14 --error-on-warning \
  --output-format concise --exclude research/mechinterp_20260911/artifacts \
  research/mechinterp_20260911
uv run --no-sync pytest -q tests/test_mechinterp_head_positions.py -rs
uv run --no-sync pytest -q --ignore=tests/experiments -m 'not integration' -rs
```

- Formatting passed: 367 files were already formatted.
- The unmodified Ruff command reports 51 existing import-order errors.
  This worktree does not contain the main checkout's ignored `wandb/`
  directory, which changes how Ruff classifies that import. The explicit
  import-classification check passes. No unrelated source was changed.
- Both type-check commands passed. The new derivation and tests also passed
  their focused Ruff and type checks.
- The focused regression tests passed: 7 passed, no skips.
- The full required test command passed: 1,210 passed, 17 integration tests
  deselected, no skips, 30 warnings, 123.97 seconds. The warnings concern
  existing dependency deprecations, NVML initialization, shuffle settings,
  process forking, and uncompiled attention.
- Separate emulator integration tests were not required: no replay,
  representation, wire, controller, or session code changed.
- All 512 plotted positions and peak weights were checked against the raw
  saved attention arrays. Distinct-position and shared-pair counts were
  independently checked for all 32 model/layer combinations.
- The embedded figure data equals the frozen derived JSON. The source
  archive contains exact copies of the derivation, plan, and tests. Frozen
  input and source hashes remain unchanged. A Node child-process archive
  check hit a sandbox permission error; direct read-only archive checks
  then passed.
- `git diff --check` passed. Existing edits in the main checkout were left
  untouched.

The browser check used Playwright 1.63.0 and Chromium 153.0.8010.12:

```sh
node --check research/mechinterp_20260911/attention_head_positions_browser_check.mjs
NODE_PATH=/tmp/policy-anatomy-browser/node_modules \
node research/mechinterp_20260911/attention_head_positions_browser_check.mjs \
  research/mechinterp_20260911/artifacts/attention_head_positions/head-token-positions.html \
  /home/ericgu/.codex/plugins/cache/openai-bundled/visualize/1.0.29/skills/visualize/scripts/render.py \
  research/mechinterp_20260911/artifacts/attention_head_positions/browser_qa
```

Both checks passed. All 16 layers were checked against the embedded data
at widths 736 and 360 pixels in light and dark themes. Both panels show
all 16 heads. The layer selector, hover details, and touch selection work.
The shared-position example reports head 2, position 69, weight 36.5149%,
and other heads 9 and 12. Desktop plot rows align. No horizontal overflow,
clipped labels, or browser JavaScript errors were found.

Final figure SHA-256:
`14afc9c44284723571d1514a56f693d6b613ce34866bbf72606ebacc406ecf0b`.
Browser-check source SHA-256:
`b5ad6336a11593eb6003367e29828e8ef1cc19e57c8d78dc8b3221264d3e76bb`.
Browser-check result SHA-256:
`502ba1cd6ab8fac29c89b0da0c0bbd0335e459f25e41d2e004c8a4ce4e777f2f`.
