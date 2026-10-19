# O65 Modal validation

## Scope and environment

Own code is in `scripts/launch_o65_modal.py`, the profiling/chunk controls in
`experiments/065_selfplay_ppo.py`, `tests/test_o65_modal.py`, and
`tests/test_o65_gpu_profile.py`. No model, decoder or simulator source was
edited. The supplied `data/processed` link was read only. No GCE VM, W&B sink,
R2 upload, or other worktree was changed. Only `o65-modal` was pushed, for
commits needed by Modal jobs.

Local GPU checks used this resource scope, at 16 matches for the new numerical
diagnostic and at most 64 matches for inherited integration tests:

```bash
flock /home/ericgu/data/scratch/o65-tmp/gpu.lock \
  systemd-run --user --scope -p MemoryMax=10G \
  env CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  TORCHINDUCTOR_COMPILE_THREADS=1 \
  HAL_O65_PROXY_CHECKPOINT=/home/ericgu/src/hal-o65-modal/runs/proxy/final.pt \
  PYTHONPATH=.:/home/ericgu/src/melee-sim-light \
  HAL_MSL_SOURCE=/home/ericgu/src/melee-sim-light \
  uv run pytest ... --basetemp "$TMPDIR/<check>"
```

`TMPDIR=/home/ericgu/data/scratch/o65-tmp/modal`. Logs are below
`runs/modal/checks/`. The existing locked Modal SDK is 1.5.3; no dependency
file changed. Local `uv pip install cairosvg` installed 2.9.1 for an existing
type-check dependency. Replay integration used `uv run --with zstandard`
(0.25.0). Neither operation changed the lockfile.

## Handoff commands

| Command | Result |
| --- | --- |
| `uv run ruff format --check .` | Pass, 382 files. |
| `uv run ruff check .` | Pass. |
| `uv run ty check --python-version 3.14 --error-on-warning hal experiments/059_muon_action_sequence.py scripts experiments/065_selfplay_ppo.py tests/test_o65_modal.py tests/test_o65_gpu_profile.py` | Pass, zero diagnostics. |
| `uv run pytest -q tests/test_o65_gpu_profile.py tests/test_o65_modal.py -m 'not integration' --basetemp "$TMPDIR/guards"` | 46 passed. |
| `uv run pytest -q -ra -m 'not integration' --basetemp "$TMPDIR/handoff-final"` with the scope above | **2,581 passed, 8 skipped, 89 deselected, 26 warnings** in 329.88 s; pytest exit 0, but the missing-fixture handoff gate fails. |
| Required `HAL_REQUIRE_INTEGRATION=1 uv run pytest -q tests/test_roundtrip.py tests/test_session_cleanup.py -m integration`, included in the integration batch | All 8 required tests passed, no skips. |
| Integration batch, including native columnar/vectorize tests | Initially 29 passed, 3 failed, 135 deselected; failures were a missing local native adapter. After building it, the 3 failed checks passed, 27 deselected. No integration skips. |

The required fixture rule remains unsatisfied: the full suite skipped the six
real-v7 schema tests. The other two skips require production netplay GPU
qualification (`test_netplay_hardware.py:52` and `:68`). A pytest zero exit status does not make that handoff gate
pass. The fixed-input-only R2 permission does not authorize fetching additional
training shards, and the shared fixture path is read only.

The final evidence audit validates `hal.o65.modal-profile-evidence.v1`, all
four profile identities, 100 raw-artifact SHA-256 values, and both spending
bounds. It passes. Its first invocation expected an incorrect format label
(`hal.o65.modal-analysis.v1`); the corrected audit checks the producer's actual
versioned label. No evidence bytes were changed to satisfy that check.

## Earlier failures, skips and corrections

- The first full suite had **1 failed, 2,558 passed, 11 skipped, 89 deselected**
  in 568.42 s (`pytest-1.log`).
  `test_tick_queues_and_resume_through_odd_resets_and_unsorted_streams` required
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`, absent from that invocation. The final
  invocation sets it. Three skips requested `HAL_O65_PROXY_CHECKPOINT`; the
  final invocation sets the existing verified proxy path. Two skips require
  production netplay hardware qualification. Six skips require
  `data/processed/ranked-anonymized-1/mds-v7-sub4/train`, which is absent.
- An integration invocation was stopped while still queued to add the cuBLAS
  environment setting; it executed no tests (`integration-1.log`). The next
  batch had **3 failed, 29 passed, 135 deselected** in 498.55 s
  (`integration-2.log`). The failures were
  `test_native_columnar_exact_parity[16]`, `[64]`, and
  `test_proxy_tick_numerics_and_update_gradients`, because this worktree lacked
  `build/native_adapter/libhal_native.so`.
- `uv run python -m hal.sim.native_build --native-source
  /home/ericgu/src/melee-sim-light` built the adapter in this worktree. The
  repeat command `uv run pytest -q tests/test_065_columnar.py
  tests/test_065_vectorize.py -m integration` passed **3 tests**, with 27
  deselected, in 498.65 s (`integration-built.log`). Source MSL was read only.
- The local common-input chunk diagnostic failed the unchanged gradient gate
  in three attempts: 13.91 s, 13.61 s, and 14.69 s. The last attempt collected
  both arms before asserting. 32/256 failed actor and critic norm limits;
  64/512 failed the critic norm limit. The measured errors are committed in
  `modal-profile-measurements.json`. The diagnostic snapshot and all failures
  remain in `runs/modal/`. This rejected treatment is not retained as an
  always-failing maintained test. The implemented correction rejects
  nondefault chunks in training and has a focused regression test. No
  threshold was relaxed.
- Initial focused collection imported `scripts` as an installed package and
  failed. It now uses the repository's existing test-module loading pattern.
  Ruff rejected a constructed default and type checking rejected the Modal
  enum import; a module constant and public `modal.types.FileEntryType` fixed
  these. The first commit hook found the existing missing `cairosvg` import;
  the local package installation above fixed the environment. A temporary
  diagnostic mapping had type errors; explicit fixed keys corrected them.
- After removing that temporary diagnostic, one format check requested a
  trailing-blank-line fix in `tests/test_o65_modal.py`. Formatting and the
  final full static checks passed. A shell edit used unavailable `python`;
  it made no changes and was repeated with `uv run python`.
- A read-only GCP secret comparison failed because `gcloud` needed renewed
  authentication. The existing Modal secret was reused, and all fixed input
  bytes passed the committed SHA-256 checks.

## Modal launch attempts

App/call identities, resource requests, commands, source identities, raw-file
hashes and resolved training configurations are retained under `runs/modal/`.
The committed measurement file carries the identities and hashes of each
completed profile. All apps were checked stopped with zero tasks.

| Attempt | Result and correction |
| --- | --- |
| `smoke-16-v1` | Stopped before GPU work to fix the MSL bindings' `PYTHONPATH`. No GPU reservation retained. |
| `smoke-16-v2` | Shared image build inherited the earlier external shutdown. No GPU work; no reservation retained. |
| `smoke-16-v3` | CPU build rejected a positional delivery action. Changed to `--action install`; the focused test parses the real delivery CLI. No GPU work. |
| `smoke-16-v4` | Modal rejected explicit 128 GiB ephemeral disk, requiring 512 GiB. Use the platform default disk. No GPU work. |
| `smoke-16-v5` | GPU setup failed because Modal hides sysfs CPU topology. Use distinct exposed CPU IDs and record that physical topology is unavailable. Full 960 s reservation retained. |
| `smoke-16-v6` | A real PPO update completed, but checkpoint hard links failed on the Modal Volume. Run on local POSIX storage, then copy bytes to the Volume. 27.42 s worker time, exit 1; partial evidence downloaded. |
| `smoke-16-v7` | Passed the real update, checkpoint save, Volume staging and download; exit 0, 33.27 s worker time. |
| `fp32-256-w16-d128` | Passed. |
| `fp32-1024-w16-d128` | Passed. |
| `fp32-1024-w32-d256` | Passed as a disposable timing arm. Training qualification failed locally. |
| `fp32-1024-w64-d512` | Passed as a disposable timing arm. Training qualification failed locally. |

A live `modal container exec` diagnostic had a PTY/stdin-executor error; it did
not stop or fail the training call. Non-PTY live stdout was unreliable, so
completed Volume artifacts are the evidence. The CPU-only interface probe
passed `/proc`, thread CPU accounting and virtual affinity checks and stopped.

## Self-review

Reviewed `git diff o65-selfplay-ppo-plan...HEAD`, the stream's own delta from
`8234dfd2`, the plan's objective/GAE/PPO definitions, execution offsets,
window/episode boundaries, artifact contracts, and the Risks section. The
inherited simulation and model boundaries remain intact. This stream does not
change reward, masks, factor order, sampling, stop-gradient placement,
optimizers, or gameplay selection.

Findings fixed during review:

1. Isolated no-grad chunk controls from actor batch shapes; CPU parity tests
   check row order, outputs, features and actor gradients.
2. Bound chunk sizes into exact-resume configuration identity and preserved
   them in frozen references.
3. Preserved setup-failure output and used local POSIX checkpoint storage
   before durable byte copies. A focused test checks copy contents and cache
   exclusion; the real smoke checks checkpoint publication.
4. Versioned and validated budget/result records; validated run names and
   download paths. Existing owned budget records were explicitly migrated
   with an unchanged backup after all old clients finished.
5. Rejected untested Modal SDK versions and the known-oversized four-group
   profile before allocation.
6. Required every group to be warmed and the resolved update count to match
   warm updates plus two measurements.
7. The numerical diagnostic found batch-shape-dependent gradient drift. Kept
   the training default and rejected unqualified chunks there.
8. Corrected report assumptions: six decoder layers run once per offset;
   actor CPU is split across main and autograd; preparation includes student
   and reference and must not be summed twice; sampled CUDA API time is not
   clean host dispatch latency.

A second review after these fixes found no further defect in the stream
changes. Rechecked the measured-data arithmetic, phase nesting, cache memory
bound, typed boundaries, version guards, conservative budget lifecycle, and
actor batch invariance. The inherited opponent-queue critic pools action
embeddings before its nonlinear encoder, so queue order is lost. This known
model limitation is already recorded in `../validation.md`; fixing it is a
scientific/model change outside this stream and remains open.

## Commits

The Modal runs use published job commits through `8b187a9d`. Final guards and
measurement documentation are local; no later push was needed.

| Commit | Change |
| --- | --- |
| `af017ab7` | Add bounded O65 Modal profiling. |
| `54194324` | Expose MSL bindings in Modal jobs. |
| `52c21ca2` | Isolate no-grad chunk sweep from actor batches. |
| `acc4876c` | Use the delivery action flag in Modal builds. |
| `50913798` | Use the Modal default ephemeral disk. |
| `cd1a90b9` | Use exposed Modal CPU IDs and preserve setup failures. |
| `afaeb7ca` | Stage checkpoints on POSIX storage before Volume copy. |
| `8b187a9d` | Validate Modal budget records and artifact paths. |
| `03375fcb` | Confine O65 chunk sweeps to warmed profiles. |
| `80a4ab00` | Reject unqualified Modal SDK and oversized profiles. |

## Cross-stream changes

The trainer and profiling test changes were necessary shared-file edits:
`experiments/065_selfplay_ppo.py` and `tests/test_o65_gpu_profile.py`.
They add `ReplayChunks`, bind it into resume identity and frozen copies,
isolate it from actor batches, extend trace and thread accounting, and enforce
warmed/profile-only configurations. The decoder stream must preserve these
adjacent fields on integration. No `hal/inference/decision_decoder.py` edit
or `o65-perf` merge was made because its final report was still absent.

## Impasses and remaining work

H3 awaits the decoder stream's final committed report and qualified setting.
After that report is available, merge `o65-perf`, pass decoder precision
explicitly from the Modal launcher, and repeat FP32/BF16 at 1,024 matches
from the same merged commit. The six missing v7 fixtures prevent
a complete handoff gate. Production hardware qualification is outside this
workstream. The 4,096-match four-group arm cannot fit its irreducible cache
storage and was not allocated. Larger no-grad chunks need numerical
qualification before training use. No learned policy was selected.
