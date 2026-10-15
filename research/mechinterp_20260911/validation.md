# Validation and execution log

Worktree: `/home/ericgu/src/hal/runs/mechinterp-20260911-worktree`, based on
`52da5b2de1d402b86404fc4d24e382409649e525`.

Measurements ran from `/tmp/hal-mechinterp-20260911`. After all research jobs
stopped, the worktree moved to disk-backed storage at the path above. Recorded
execution paths in raw artifacts remain unchanged. `git worktree move` could
not cross the filesystem boundary; a scoped directory move followed by
`git worktree repair` preserved the worktree. The original worktree is clean.

This log records failures as well as successful reruns. Final validation uses
the durable worktree and includes the last research additions.

## Environment and fixtures

Commands use the existing Python 3.14.4 environment without changing it:

```sh
export UV_PROJECT_ENVIRONMENT=/home/ericgu/src/hal/.venv
export UV_CACHE_DIR=/tmp/hal-mechinterp-uvcache
export PYTHONPATH=/home/ericgu/src/hal/runs/mechinterp-20260911-worktree
export TMPDIR=/home/ericgu/src/hal/runs/mechinterp-20260911-scratch
export HAL_ISO_PATH=/home/ericgu/src/hal/data/emulator/ssbm.ciso
export HAL_EMULATOR_PATH=/home/ericgu/src/hal/data/emulator/exiai/squashfs-root/AppRun
```

`uv run --no-sync` preserves the shared environment. `PYTHONPATH` selects the
research worktree instead of the environment's editable main-worktree install.
GPU and local service tests ran outside the network/GPU sandbox.

The research worktree links the existing dev archive, dev MDS, ranked-anonymized-1
MDS fixtures, production O26 test checkpoint, and installed netplay node modules.
These are fixture links, not new copies or changes to the original data. An
initial duplicate dev `stats.json` was replaced by the complete dev-MDS link.
Large scratch files now use disk-backed storage, not the 16-GB `/tmp` mount.

## Required checks

| Command | Final result |
| --- | --- |
| `uv run --no-sync ruff format --check .` | Final pass: **359 files already formatted**. Earlier failures on `scaling_audit.py` and the in-progress `historical_gameplay_modal.py` were fixed. |
| `uv run --no-sync ruff check .` | 51 I001 errors in unchanged experiment/script import blocks; see the environment-specific explanation below. |
| `uv run --no-sync ty check --python-version 3.14 --error-on-warning hal experiments/051_muon_parameterization.py scripts/cache_modal_fixtures.py scripts/launch_gce.py scripts/launch_modal.py scripts/launch_vast.py scripts/replay_policy_fault.py` | Pass, zero diagnostics. |
| `uv run --no-sync ty check --python-version 3.14 --error-on-warning --output-format concise --exclude research/mechinterp_20260911/artifacts research/mechinterp_20260911` | Final pass, zero diagnostics in maintained research code. Immutable source artifacts are data, not a maintained type target. |
| `uv run --no-sync pytest -q --ignore=tests/experiments -m 'not integration' -rs` | Final pass with all existing fixtures linked: **1,183 passed, zero skipped, 17 deselected**, 29 warnings, 126.27 seconds. |
| `uv run --no-sync pytest -q research/mechinterp_20260911/test_historical_gameplay.py -rs` | Final pass: **26 passed, zero skipped**, 2.90 seconds. This explicit target is outside the repository's normal `testpaths`. |
| `HAL_REQUIRE_INTEGRATION=1 uv run --no-sync pytest -q tests/test_roundtrip.py tests/test_session_cleanup.py -m integration -rs` | Final pass: **7 passed, zero skipped, 2 deselected**, 7 warnings, 52.61 seconds. |

Bare Ruff passes in the unchanged original worktree. That directory contains an
ignored `wandb/` cache, which makes Ruff classify `wandb` as first-party. The
research worktree lacks the cache and classifies the installed dependency as
third-party. This changes 51 import-order decisions without any source changes.
The final diagnostic command below passes over the entire research worktree:

```sh
uv run --no-sync ruff check . --config 'lint.isort.known-first-party=["hal","wandb"]'
```

This reproduces the existing worktree's grouping; it is not a claim that W&B is
HAL-owned code. No frozen experiment was reformatted to conceal the issue.
The bare required command's failure remains reported.

## Earlier failures and skips

- An initial research-test collection failed because `research` is not an
  installed package. Tests now use the repository's explicit module-loading
  pattern, and worktree execution also sets `PYTHONPATH`.
- A sandboxed full-suite run stalled around 39% in a local-service test. It was
  stopped; it has no successful final result.
- The first outside-sandbox full run had **4 failed, 1,087 passed, 14 skipped,
  17 deselected**. One cleanup test lacked netplay node modules; three replay
  reservoir tests lacked the worktree dev statistics fixture.
- After supplying statistics, the relevant subset had **1 failed, 43 passed**.
  Linking the existing installed node modules resolved the remaining cleanup
  failure; that test passed independently.
- A later full run reached the final section but failed with `OSError: Disk
  quota exceeded`, including during pytest cache writing. It has no trustworthy
  final count. No scientific result relies on that run.
- With disk-backed scratch but incomplete fixture links, the suite had
  **1,082 passed, 36 skipped, 17 deselected**. Skips were 22 dev-archive tests,
  6 dev-MDS tests, 6 local-v7-subset tests, 1 ranked-manifest test, and 1 O26
  checkpoint test. These were not accepted as sufficient validation. All 36
  ran and passed in the subsequent 1,118-test run.
- `spectral_findings.py` initially had one type diagnostic for an optional
  numeric list passed to NumPy. It was fixed with explicit numeric validation;
  its focused type check is now clean. Its five tests plus the five spectral
  measurement tests pass.
- An expanded Ty check of the research directory initially reported 82
  diagnostics, including archived source snapshots. Excluding immutable
  artifact data exposed 61 diagnostics in maintained files while agents were
  still adding code. These concerned module-list types, metadata narrowing,
  and Pandas/Matplotlib interfaces. Type-only cleanup preserves the measured
  source versions in `artifacts/source_snapshots/`. The maintained research
  target subsequently passed with zero diagnostics; final validation repeats it.
- The scaling-reader cleanup changed a list column selector to an equivalent
  name predicate. Its new persisted-column/order parity test passes; all seven
  scaling tests pass. The original measured source is archived under SHA-256
  `a8b921ccaa7da9131427fd9295e3d97bca4e3fc08b4db9425a41ed0474c697d1`.
- The new IID-data analytical counterexample has five passing tests, including
  independent ordered-dataset enumeration of its binomial sum. Its focused
  Ruff format, lint, and Ty checks pass. The artifact is an exact finite-sum
  illustration evaluated in floating point, not a model-training result.

Warnings in the successful suite concern Starlette's HTTPX integration,
TorchScript on Python 3.14, streaming shuffle-block size, fork in a multithreaded
process, and uncompiled FlexAttention. They are reported, not suppressed.

## Research execution issues

- Six model-only exports passed remote size/hash validation. Local downloads
  into `/tmp` exceeded quota; invalid partial copies were removed. The original
  checkpoints and the six complete Modal Volume exports remain intact.
- Modal downloads can preallocate their full destination size before transfer
  completion. Size alone is not a completeness check. Downloads must finish
  and pass SHA-256 validation before any model load; later downloads use a
  `.partial` filename until verified.
- The first spectral app failed at import because a host-relative
  `__file__.parents[2]` path was evaluated in `/root` remotely. That app was
  explicitly stopped. Guarding host mount construction with `modal.is_local()`
  fixed it. The corrected app completed all six spectral audits.
- The historical cloud probe encountered missing remote HAL source. A later
  upload of HAL source and a replay-derived cohort was rejected by the safety
  reviewer. No upload workaround was used. Explicit user approval was requested;
  the investigation continued with checkpoint downloads and the local GPU.
- The first attention summaries mistakenly let CUDA autocast lower QK scores
  and softmax to BF16 despite explicit `.float()` inputs. Trunk parity did not
  catch this because the diagnostic weights were measured separately from the
  public block output. Original artifacts and exact source are retained. All
  six attention summaries were re-audited with FP32 scores/softmax, TF32 off,
  and exact public-trunk parity. The main optimizer contrast remains.
  The first 18-state intervention rerun still used the old attention precision;
  it verifies NLL/action parity only despite its `interventions_fp32` directory
  name. The guard was added after that run. An actual corrected rerun was
  completed in `artifacts/interventions_fp32_corrected/`. All four attention
  traces assert FP32 at runtime and have zero public-trunk error. All 18
  states have exactly unchanged NLL, bootstrap intervals, JS, action-change
  rates, and public greedy mismatch counts. The gauge also preserves the
  corrected attention metrics exactly. No corrected value is inferred from
  the parity-only result. Unlike the six-checkpoint attention re-audit, the
  intervention launch did not explicitly set or record TF32 flags. A post-run
  query reports TF32 off and highest matmul precision, but does not prove the
  completed run's flags. Its supported precision claim is the runtime FP32
  dtype assertion.
- The first hard-routing launch failed its exact soft-path control with maximum
  hidden-state error 0.33594 and wrote no result. Explicit dense SDPA did not
  reproduce the historical config's BF16 variable-length FlashAttention path.
  The corrected implementation keeps the public attention module for untreated
  rows and replaces only each layer's final projected row. Both real-model
  soft controls then had exactly zero error. The failed and successful sources
  are archived. A full-cohort projection-shape rerun was terminated by an
  approval-review timeout after about 84 seconds and wrote no result. A bounded
  numerical control then checked the first eight fixed cohort rows. Projecting
  selected values as one row or as the last row of a full-length buffer was
  bit-exact at all 16 layers in both final models, for both top-key and
  current-key selections. The successful v1 routing result is retained; this
  small control is not represented as a completed full-cohort rerun.
- The six-context learner-replay trace used CPU FP32 only, for about 20 seconds
  while the portable gameplay condition was active. Its frozen one-off source
  is stored as `.py.txt` with unchanged bytes. It is artifact data, not maintained
  source to reformat. The raw per-record `mean_lag` field divides by the number
  of keys twice; the separate per-head `trace.metrics.mean_lag` is correct.
  Raw data are retained, and no main finding uses the incorrect field. The
  derived `artifacts/live_attention/mean_lag_correction.json` recomputes each
  record from the stored weights without a new forward pass. Focused tests
  cover the first-key and current-key reductions and invalid probability data.
- A final process-status query,
  `nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader`,
  failed with NVML exit 18, `Driver/library version mismatch`. The loaded
  kernel module reports 595.84; the NVML library reports 595.91. A separate
  `uv run --no-sync python -c 'import torch; print({"torch": torch.__version__, "cuda_available": torch.cuda.is_available()})'`
  check completed successfully with Torch `2.11.0+cu130` and CUDA available.
  The process table contained no matching research runner, and all owned jobs
  had completed, but global GPU idleness could not be checked with NVML.
  No driver or environment setting was changed. This is an environment issue,
  not a failed scientific result or required test.

There has been no new training, resume, or mutation of the original runs.

## Attention-detail parity tolerance

The first detailed-attention launch used an absolute parity limit of `2e-5`.
It stopped before writing JSON or NPZ results because the Muon 24,576-step
weighted-lag matrix differed from the existing corrected artifact by
`2.1454049e-5`. Its source archive is retained under SHA-256
`b54e8e21ae62155091818b1121356f86bcc1182dd6c6c67bf0dbde100101c6c3`.
The failed launch and final v2 launch used the same command and environment:

```sh
TMPDIR=/home/ericgu/src/hal/runs/mechinterp-20260911-scratch/tmp \
UV_CACHE_DIR=/tmp/hal-mechinterp-uvcache \
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/home/ericgu/src/hal/runs/mechinterp-20260911-worktree:/home/ericgu/src/hal/.venv/lib/python3.14/site-packages \
/home/ericgu/src/hal/.venv/bin/python \
  research/mechinterp_20260911/attention_detail.py \
  --cohort /home/ericgu/src/hal/runs/mechinterp-20260911-scratch/o50-cohort-256.pt \
  --scratch /home/ericgu/src/hal/runs/mechinterp-20260911-scratch \
  --old-attention research/mechinterp_20260911/artifacts/o50_attention_fp32 \
  --output research/mechinterp_20260911/artifacts/attention_detail \
  --device cuda \
  --batch-size 8
```

After that failure, a diagnostic copy wrote only to
`/tmp/hal-attention-detail-probe` and temporarily used `atol=1e-3`. It
completed all six checkpoints and measured the maximum absolute difference
over each 16-by-16 layer-head matrix:

| Checkpoint | Entropy (nats) | Mean exp(entropy) | Weighted lag (frames) |
| --- | ---: | ---: | ---: |
| Muon 24,576 | `1.3221e-7` | `5.2199e-7` | `2.1454e-5` |
| Muon 57,344 | `5.1582e-8` | `2.1098e-7` | `2.0750e-5` |
| Muon 98,304 | `2.1215e-8` | `1.6571e-7` | `1.5854e-5` |
| Muon 131,072 | `5.1242e-8` | `1.9666e-7` | `1.9822e-5` |
| AdamW 57,344 | `6.8857e-7` | `1.3525e-5` | `1.9407e-5` |
| AdamW 131,072 | `6.2357e-7` | `1.4983e-5` | `1.9828e-5` |

The final v2 run used `rtol=0` and `atol=3e-5`. This limit was chosen after the
diagnostic run measured all six checkpoints. The result schema
records this as one ceiling for all three reference matrices, but only the
weighted-lag reduction exceeded the original `2e-5` limit. Entropy errors were
at most `6.89e-7`; mean-exp-entropy errors were at most `1.50e-5`. The revised
limit accommodates float64 summary arithmetic versus the prior float32
summary, without changing the attention weights or model outputs.
An independent CPU audit recomputed the
same six errors from the completed final JSON afterward.
The final run also had FP32 attention tensors, disabled matmul and cuDNN TF32,
`highest` float32 matmul precision, and zero public-trunk trace error.

The final JSON is
`artifacts/attention_detail/attention_detail.json` (SHA-256
`818f0871252acf0e96672e006b8d8a6b76781324d005ce8ddbe130615ff7bbd2`).
Its v2 source archive has SHA-256
`f1ba2de27bacaa960203f6043365d6c152df5e5e3c97619f75e0d5c9976d857e`.

The intermediate diagnostic results are retained under
`artifacts/attention_detail/tolerance_diagnostic/`. Their original execution
paths remain unchanged in the JSON. The diagnostic JSON has SHA-256
`5116fa347afbe56b098ef0d93e07abec2f0452fb301b741d87088f291b3a7c6d`.
Its six summary objects exactly match the final v2 summary objects, and its
NPZ is byte-identical to the final NPZ. The diagnostic's `1e-3` override is
recorded in its JSON; its source archive still contains the default `2e-5`
constant. That archive alone does not encode the override. The final v2
source archive contains the actual `3e-5` setting and matches the maintained
source hash `f09db24acbbb921aebe93d1d05b30e02b354e9279443099761b3e9811f4a1ed5`.

## Attention follow-up validation

The commands below use the environment listed at the top of this log.
The full regression suite ran outside the sandbox, as its local-service and
GPU checks require access unavailable inside it.

| Command | Result |
| --- | --- |
| `uv run --no-sync pytest -q tests/test_mechinterp_attention_detail.py -rs` | 11 passed, zero skipped, 2.82 seconds. |
| `uv run --no-sync pytest -q --ignore=tests/experiments -m 'not integration' -rs` | 1,194 passed, zero skipped, 17 deselected, 30 warnings, 121.39 seconds. |
| `uv run --no-sync ruff format --check .` | Pass, 361 files already formatted. |
| `uv run --no-sync ruff check .` | The same 51 I001 errors in unchanged experiment/script files, as explained above. |
| `uv run --no-sync ruff check research/mechinterp_20260911/attention_detail.py tests/test_mechinterp_attention_detail.py` | Pass. |
| `uv run --no-sync ruff check . --config 'lint.isort.known-first-party=["hal","wandb"]'` | Pass. The standard command also passed in the original main worktree. |
| `uv run --no-sync ty check --python-version 3.14 --error-on-warning hal experiments/051_muon_parameterization.py scripts/cache_modal_fixtures.py scripts/launch_gce.py scripts/launch_modal.py scripts/launch_vast.py scripts/replay_policy_fault.py` | Pass, zero diagnostics. |
| `uv run --no-sync ty check --python-version 3.14 --error-on-warning --output-format concise --exclude research/mechinterp_20260911/artifacts research/mechinterp_20260911` | Pass, zero diagnostics. |
| `node --check research/mechinterp_20260911/attention_detail_browser_check.mjs` | Pass. |

The 17 deselected tests are excluded by the requested non-integration marker;
they are not missing-fixture skips. The extra gameplay-protocol and emulator
integration suites were not rerun: this addition changes only an attention
measurement program and its presentation, not the replay or simulation path.

A preliminary sandboxed Chromium launch failed with
`sandbox_host_linux.cc:41 ... Operation not permitted`. The installed browser
started successfully outside the sandbox. No browser or system dependency
was installed or changed. The final browser check was:

```sh
NODE_PATH=/tmp/policy-anatomy-browser/node_modules \
TMPDIR=/home/ericgu/src/hal/runs/mechinterp-20260911-scratch \
node research/mechinterp_20260911/attention_detail_browser_check.mjs \
/home/ericgu/src/hal/runs/mechinterp-20260911-worktree/research/mechinterp_20260911/artifacts/attention_detail/muon-attention.html \
/home/ericgu/.codex/plugins/cache/openai-bundled/visualize/1.0.29/skills/visualize/scripts/render.py \
/home/ericgu/src/hal/runs/mechinterp-20260911-worktree/research/mechinterp_20260911/artifacts/attention_detail/browser_qa
```

All four combinations of 736/360-pixel width and light/dark theme passed.
The check covers plot and label presence, 512 heatmap cells, 512 probability
stems, no horizontal overflow or clipped SVG labels, 12-pixel labels, native
selection, heatmap click/tap, and both hover displays. There were no browser
JavaScript errors. Desktop and phone screenshots were also inspected.
Results are recorded in `artifacts/attention_detail/browser_qa/browser-check.json`.

The inline figure is 771,744 bytes, SHA-256
`c6a0f4eb3964a06cbebb819ea3858f193d41ed3615fea62b2ff6b3d4853d5f97`.
An independent array comparison confirmed that all embedded float32
probabilities exactly equal the NPZ arrays after one key-to-age reversal.
The mean metrics also match the measurement JSON within `1e-7`.
The final NPZ has SHA-256
`64c549e488f2537793d092b6a708ff4743111e23938546eff3f7ed0bd63e82c3`.

All attention measurement and browser processes launched for this follow-up
have finished. All edits made for this follow-up are in the research
worktree. A concurrent modification
to `notebooks/awr_explorer.py` in the original main worktree was left untouched.

## Repeat on the requested AdamW checkpoints

The supplied W&B URL and timestamped display name both identify `tgpuo1be`.
The new inventory verifies the existing 57,344- and 131,072-step exports and
their training-history evidence. No new run or checkpoint was downloaded.
The plan is `attention_adamw_repeat_plan.md`, SHA-256
`9c5b7d727a7de1d7894816d17bca920a4184d6d4bedcc58bb96f959a1c164a05`.

The frozen measurement program was run without a code change or runtime
override. It measures all six original checkpoints; the new figure displays
only the two AdamW checkpoints. The repeat command was:

```sh
TMPDIR=/home/ericgu/src/hal/runs/mechinterp-20260911-scratch/tmp \
UV_CACHE_DIR=/tmp/hal-mechinterp-uvcache \
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/home/ericgu/src/hal/runs/mechinterp-20260911-worktree:/home/ericgu/src/hal/.venv/lib/python3.14/site-packages \
/home/ericgu/src/hal/.venv/bin/python \
research/mechinterp_20260911/attention_detail.py \
--cohort /home/ericgu/src/hal/runs/mechinterp-20260911-scratch/o50-cohort-256.pt \
--scratch /home/ericgu/src/hal/runs/mechinterp-20260911-scratch \
--old-attention research/mechinterp_20260911/artifacts/o50_attention_fp32 \
--output research/mechinterp_20260911/artifacts/attention_adamw_repeat \
--device cuda --batch-size 8
```

The local RTX 3060 had 12,288,851,968 free bytes before launch. The known NVML
warning remained, but CUDA worked. The run completed successfully, with exact
public-forward parity. All six summary and control objects exactly match the
previous results. The saved example-zero NPZ is byte-identical. These are
repeat measurements, not additional independent training observations.

- Repeat JSON SHA-256:
  `dda10f2a7e62b0a483cb30dc8468bb001d43f4cbe018e56a86135d11c4cd46d5`.
- Unchanged measurement source SHA-256:
  `f09db24acbbb921aebe93d1d05b30e02b354e9279443099761b3e9811f4a1ed5`.
- The agreement is recorded in
  `artifacts/attention_adamw_repeat/repeat-parity.json`.

The new browser check used:

```sh
NODE_PATH=/tmp/policy-anatomy-browser/node_modules \
TMPDIR=/home/ericgu/src/hal/runs/mechinterp-20260911-scratch \
node research/mechinterp_20260911/attention_adamw_browser_check.mjs \
/home/ericgu/src/hal/runs/mechinterp-20260911-worktree/research/mechinterp_20260911/artifacts/attention_adamw_repeat/stable-adamw-attention.html \
/home/ericgu/.codex/plugins/cache/openai-bundled/visualize/1.0.29/skills/visualize/scripts/render.py \
/home/ericgu/src/hal/runs/mechinterp-20260911-worktree/research/mechinterp_20260911/artifacts/attention_adamw_repeat/browser_qa
```

All four 736/360-pixel and light/dark combinations passed, including native
selection, click/tap, hover values checked against the measurement JSON,
visible labels, zero overflow, and no JavaScript errors. Desktop and phone
screenshots were inspected. An independent array comparison found exact
agreement between the embedded float32 weights and the fresh NPZ after one
key-to-age reversal, with summary values agreeing within `1e-7`.
The new fragment is 774,316 bytes, SHA-256
`7bd7166ed3e29cb27fa38de670c89ceab06eb6f8d438006ebe557c881505d47c`.
The original Muon/AdamW figure and all previous numerical artifacts remain
unchanged.

| Command | Repeat validation result |
| --- | --- |
| `uv run --no-sync pytest -q --ignore=tests/experiments -m 'not integration' -rs` | 1,194 passed, zero skipped, 17 deselected, 30 warnings, 121.65 seconds. |
| `uv run --no-sync ruff format --check .` | Pass, 361 files already formatted. |
| `uv run --no-sync ruff check .` | The same 51 baseline I001 errors in unchanged experiment/script import blocks. |
| `uv run --no-sync ruff check . --config 'lint.isort.known-first-party=["hal","wandb"]'` | Pass. |
| `uv run --no-sync ty check --python-version 3.14 --error-on-warning hal experiments/051_muon_parameterization.py scripts/cache_modal_fixtures.py scripts/launch_gce.py scripts/launch_modal.py scripts/launch_vast.py scripts/replay_policy_fault.py` | Pass, zero diagnostics. |
| `uv run --no-sync ty check --python-version 3.14 --error-on-warning --output-format concise --exclude research/mechinterp_20260911/artifacts research/mechinterp_20260911` | Pass, zero diagnostics. |
| `node --check research/mechinterp_20260911/attention_adamw_browser_check.mjs` | Pass. |

No Python source, replay extraction, controller, simulation, or representation
code changed in this repeat. The additional gameplay-protocol and emulator
integration suites were not rerun. All processes launched for this repeat
finished; no training or external run state was changed.

## Matched Muon continuation

The requested Muon control is `7dgv85x8` at update 57,344, compared with
`p1fyyp1z` at the same update. The control was selected from its matching
configuration, parent checkpoint, training health, and gameplay history,
before examining its attention. It is a half-rate continuation from update
24,576, not an independent training seed. Its recorded history ends at
71,225. The full inventory is
`artifacts/attention_muon_control/inventory.json`, SHA-256
`8d8e30403621f4034a687aca107b05067d66f12b0fa817ad73a9b8d61abb4a1a`.

### Retrieval and preflight

The existing read-only `r2:` remote supplied the selected raw checkpoint.
The exact transfer command, R2 size and ETag, and successful transfer status
are recorded in the inventory. No HAL source or replay cohort was uploaded,
no Modal GPU was used, and no training state was changed.

The raw source has SHA-256
`89c941d64f6407eed34bc6b9ad3ccc1466aace67d48f0ef45978bcb7575a4b08`.
The safe exporter uses `weights_only=True` and only the two audited locator
dataclasses `PhysicalRow` and `RingSlotDescriptor` as extra allowed globals.
Their source hash matches the recorded training code. The exporter source
is `export_muon_control.py`, SHA-256
`d6af16129fe7e2a097df6b7e30e842be849cc31c399c1b007d42ac92959790e5`.

An intermediate model export and sidecar were overwritten while preserving
the exporter program. The originally declared `6cbd8755...` export and
`bdb81b10...` sidecar cannot be recovered byte for byte. The raw training
checkpoint was never changed. The surviving intermediate archive and final
export are preserved, with their identities and demonstrated comparisons
in the inventory. No attention measurement had started at the time.

The initial plan is preserved as `attention_muon_control_plan_initial.md`.
The corrected pre-measurement plan is `attention_muon_control_plan.md`,
SHA-256 `ad42f7e30b87590e49d8b510db9e60787d570ea8426e86e892e635dad560208a`.
It pins the final model export to
`2ad0c0f8015c75235def1e8d591a7350145e64a389fca5af63b39b53e6516be2`.
The adjacent metadata SHA-256 is
`9d27c8d84778696423fb6617350e5136886d1e6b01418596117dd5871922b653`.

The main agent independently verified both file hashes, restricted-loaded
the final export with the public O50 loader, and checked that every tensor
is finite. A separate restricted comparison with the raw checkpoint found
zero unequal tensors among all 168 entries. Configuration, stored step,
run ID, data protocol, all 44 source manifests, and source selection also
match exactly. The parent and control inference configurations are equal
apart from their experiment IDs.

One initial diagnostic assertion incorrectly required equality with the
cohort's reference-bundle architecture. That bundle is d256/h4; the historical
parent and control are both d1024/h16. The bundle supplies statistics and
vocabulary, while the public loader uses each checkpoint's own architecture.
The corrected check compares the two historical model configurations and
passed. No model, input, or representation was changed. A mistyped cohort
hash in the draft entry point was also caught before measurement; the final
program uses the frozen module's public cohort-hash constant.

### Attention measurement

The new program is `attention_muon_control.py`, SHA-256
`ce6e688cb30546abf47e7d67206121b36d85be27dad65c0778b6371d912b1471`.
The fixed inputs, batch size eight, retained player-rank inputs, bfloat16
projections, float32 scores and softmax, disabled TF32, and float64 summary
calculation are unchanged. The source archive was captured before the run.
The command was:

```sh
TMPDIR=/home/ericgu/src/hal/runs/mechinterp-20260911-scratch/tmp \
UV_CACHE_DIR=/tmp/hal-mechinterp-uvcache \
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/home/ericgu/src/hal/runs/mechinterp-20260911-worktree:/home/ericgu/src/hal/.venv/lib/python3.14/site-packages \
/home/ericgu/src/hal/.venv/bin/python \
research/mechinterp_20260911/attention_muon_control.py \
--cohort /home/ericgu/src/hal/runs/mechinterp-20260911-scratch/o50-cohort-256.pt \
--scratch /home/ericgu/src/hal/runs/mechinterp-20260911-scratch \
--old-detail research/mechinterp_20260911/artifacts/attention_detail/attention_detail.json \
--old-example research/mechinterp_20260911/artifacts/attention_detail/attention_example0.npz \
--output research/mechinterp_20260911/artifacts/attention_muon_control \
--device cuda --batch-size 8
```

The local RTX 3060 had 12,288,851,968 free bytes before launch. The known
NVML warning remained, but CUDA worked. Both checkpoints completed with
zero traced-versus-public-forward error. The largest probability-sum error
was `2.384185791015625e-7` for both.

The program requires exact parent summary and example-array equality with
the earlier detailed result. This passed, and the main agent independently
confirmed the complete parent summary and all seven example arrays. No
comparison tolerance or runtime override was used.

- Measurement JSON SHA-256:
  `ce30744586fcc6bd5c828dc13f9c1fa930b014465ec157ee5ab506646359f6e2`.
- Example NPZ SHA-256:
  `1518b5581f7beda9a431d2f4c9c7e22c6712c62a350a8fdf135f37c7d8ec38be`.
- Source archive SHA-256:
  `cedc1a778afec1f5920b199a166f8865bafc3e33aec2077f80595bb7bb0b52d3`.

### Interactive figure and regression checks

The figure uses new JSON checkpoints 0 and 1, and new NPZ entries
`checkpoint_00` and `checkpoint_01`. The main agent independently verified
exact equality for every embedded float32 probability after reversing
key order to frame age, and exact equality for the five per-head summary
arrays. The fragment has 773,545 bytes, SHA-256
`99d7c2e8a377dbb200dd53b1407dcbd68bfa3ab8f3e43f8e3c027095de3565e7`.

The browser check command was:

```sh
NODE_PATH=/tmp/policy-anatomy-browser/node_modules \
TMPDIR=/home/ericgu/src/hal/runs/mechinterp-20260911-scratch \
node research/mechinterp_20260911/attention_muon_control_browser_check.mjs \
/home/ericgu/src/hal/runs/mechinterp-20260911-worktree/research/mechinterp_20260911/artifacts/attention_muon_control/muon-baseline-attention.html \
/home/ericgu/.codex/plugins/cache/openai-bundled/visualize/1.0.29/skills/visualize/scripts/render.py \
/home/ericgu/src/hal/runs/mechinterp-20260911-worktree/research/mechinterp_20260911/artifacts/attention_muon_control/browser_qa
```

All four 736/360-pixel and light/dark combinations passed: four SVGs,
512 heatmap cells, 512 probability stems, native selection, click/tap,
hover values, visible labels, and zero overflow or JavaScript errors.
The main agent also inspected the desktop-light and phone-dark screenshots.
The browser-check JSON SHA-256 is
`85e1e782040a502696de884a43c7e60e5c321158bc42a7634250dd527c3d50a7`.

| Command | Muon-control validation result |
| --- | --- |
| `uv run --no-sync pytest -q tests/test_mechinterp_attention_muon_control.py research/mechinterp_20260911/test_export_muon_control.py -rs` | 10 passed, zero skipped, 2.91 seconds. |
| `uv run --no-sync pytest -q --ignore=tests/experiments -m 'not integration' -rs` | 1,203 passed, zero skipped, 17 deselected, 30 warnings, 122.58 seconds. |
| `uv run --no-sync ruff format --check .` | Pass, 365 files already formatted. |
| `uv run --no-sync ruff check . --output-format concise` | The same 51 baseline I001 errors in unchanged experiment/script import blocks. |
| `uv run --no-sync ruff check . --config 'lint.isort.known-first-party=["hal","wandb"]'` | Pass. |
| `uv run --no-sync ty check --python-version 3.14 --error-on-warning hal experiments/051_muon_parameterization.py scripts/cache_modal_fixtures.py scripts/launch_gce.py scripts/launch_modal.py scripts/launch_vast.py scripts/replay_policy_fault.py` | Pass, zero diagnostics. |
| `uv run --no-sync ty check --python-version 3.14 --error-on-warning --output-format concise --exclude research/mechinterp_20260911/artifacts research/mechinterp_20260911` | Pass, zero diagnostics. |

The exporter test initially failed collection when run without `PYTHONPATH=.`;
the research namespace was not importable in that invocation. It passed with
the documented worktree import path. No test was skipped to hide that failure.
The 30 full-suite warnings remain the same dependency, NVML, multiprocessing,
shuffle, and uncompiled-attention warning families described earlier.

Emulator integration tests were not rerun: this change does not alter replay
extraction, wire format, controller input, session stepping, or offline/live
representation. The older attention programs, numerical artifacts, and
figures remain unchanged. No analysis, browser, or test process from this
follow-up is still running, and no training or external run state was changed.

`node --check research/mechinterp_20260911/attention_muon_control_browser_check.mjs`
and `git diff --check` also passed. The final artifact check verified all
31 local report links, the archived source bytes and their recorded hashes,
and the unchanged earlier numerical artifacts and inline figures. The
unrelated main-worktree change to `notebooks/awr_explorer.py` remains untouched.
