# Proxy validation

The refactor comparison starts from `o65-integrate` commit
`55c1aef157dfc7190e0f90e6c040523f3e1c32b5`. The checkpoint, sampler, and
training data hashes are recorded in [the measurement report](performance/report.md).
Tests use the local RTX 3060 12 GB, Python 3.14.4, Torch 2.11.0+cu130, and
NumPy 2.1.3. Temporary files use the scratch directory supplied through
`TMPDIR`.

## Deterministic comparison

Run this command separately before and after the refactor, changing only output:

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=1 \
PYTHONPATH="$HOME/src/melee-sim-light" \
uv run python experiments/065_selfplay_ppo.py train \
  --checkpoint runs/proxy/final.pt --sampler experiments/o65/pilot-sampler.json \
  --data-dir "$HOME/src/melee-sim-light/data" --output runs/o65-cleanup/after \
  --cpus 0 1 --groups 2 --matches-per-shard 1 --updates 2 --compiled \
  --training.windows-per-step 4 --training.critic-warmup-updates 0 \
  --training.checkpoint-interval 2 --training.deterministic \
  --evaluation-every 0 --verify-resume
```

Both runs pass. All 62 training metrics at update 1 and all 60 at update 2
match exactly. The exported actor tensors also match. A third run with the
rebuilt MSL fork repeats the same metrics, tensors, batch hash, and optimizer
hashes. Timing and memory measurements vary between processes.

| Check | Before and after |
| --- | --- |
| Update 1 metric hash | `4d8f1abc06145dde686d2d4ab1f6e27c85349999d5f8a0b3001df52b813f5086` |
| Update 2 metric hash | `7064eddfda37f99b97bcc9a53634ba69b86996eced185c6b0128669bdd3413e5` |
| Next batch hash | `5b90549326e41b4b3cae9c80db83b6777499b5c831ef7d06bdcb43be70b4d48a` |
| Next optimizer/scheduler hash | `282b6cac28591f51c67e97f778d3c501e0190526d8cabfebb40a148da69a3610` |
| Boundary optimizer/scheduler hash | `f5b8e35e6736d787b9b418ea5795d038718b61f6806765b383514aa316d16b79` |
| Maximum model parameter error | 0 |

Metric hashes serialize the sorted non-system metrics. Resume hashes cover
the entire next batch, both optimizers, and scheduler. Each resume check
restores its original boundary after comparison.

| Update | Before match-frames/s | After match-frames/s | Rebuilt MSL match-frames/s |
| ---: | ---: | ---: | ---: |
| 1 | 62.107 | 90.709 | 90.731 |
| 2 | 86.578 | 119.215 | 119.718 |

These short runs include cold graph preparation and run at different times on
a shared host. They do not establish a performance improvement.

## Parity coverage

The native suite replays 12 complete admitted recordings. It compares all 71
stored columns and both reward streams bitwise with replay extraction and
`hal.training.returns`. The recordings include every supported stage, followers,
items, and several character-specific paths. Other native checks compare
controller bytes with a real Dolphin recording and the next frames after
cross-batch restore.

Model tests compare widened BC conditioning, cached and full-window action
probabilities, parallel and sequential gradients, and two-cache handoffs.
Episode tests overwrite earlier-episode bytes and require identical later
outputs. Packed features, controller readback, GPU action counters, and stream
GAE retain bitwise comparisons with their scalar paths.

The refactor retains all 202 tolerance arguments across the 18 affected test
files. After helper renaming, the AST bodies of 13 mathematical functions
match the backup. These include execution masks, rewards, GAE, PPO, and KL.
The removed assertions read an unused factor-count property and a Python signal
constant; parity and resume assertions remain in place.

## Commands and results

Focused validation uses:

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
HAL_O65_PROXY_CHECKPOINT=runs/proxy/final.pt \
uv run pytest -q tests/test_065_math.py tests/test_065_loop.py \
  tests/test_o65_evaluation.py -m 'not integration' --basetemp="$TMPDIR/f1"
```

Result: 157 passed, 4 integration cases deselected, no skips.

Formatting and lint preparation used `uv run ruff format .` and
`uv run ruff check .`. The first lint run reported six unsorted import blocks.
`uv run ruff check --fix .` corrected them. A commit hook also rejected an
incomplete staged rename set with 11 unresolved-import diagnostics. Staging
the module renames with their callers passed all hooks.

The first native integration run returned 1 failed, 18 passed, 88 deselected.
A test still called the deleted private hash helper. Updating that caller to
the shared helper gives 19 passed, 88 deselected, with no skips. The runtime
hash calculation and all tolerance values stay unchanged.

Final results:

| Command | Result |
| --- | --- |
| uv run ruff format --check . | Passed, 370 files |
| uv run ruff check . | Passed |
| Ty command below | Passed, zero diagnostics |
| Non-integration command below | 2,442 passed, 8 skipped, 84 deselected |
| Native integration command below | 19 passed, 88 deselected, no skips after the caller fix |
| Dolphin integration command below | 8 passed, 6 deselected, no skips |
| Deterministic proxy command above, all three runs | Exact metrics, actor tensors, next batch, optimizers, and scheduler |
| Paired evaluation command below | 8 matches published, no failed or dropped matches |

The non-integration suite takes 500.85 s and reports 27 warnings. Native
integration takes 81.13 s; Dolphin integration takes 57.54 s. Warnings concern
TorchScript, diagnostic attention, TF32 configuration, websocket management,
and Python's multi-threaded fork warning.

The type check includes the required targets and every new Python test:

    uv run ty check --python-version 3.14 --error-on-warning \
      hal experiments/059_muon_action_sequence.py scripts \
      experiments/065_selfplay_ppo.py experiments/o65/pilot.py \
      tests/test_065_loop.py tests/test_065_math.py \
      tests/test_block_restart.py tests/test_episode_segments.py \
      tests/test_native_sim_integration.py tests/test_o65_delivery.py \
      tests/test_o65_evaluation.py tests/test_o65_pilot.py \
      tests/test_o65_proxy_checkpoint.py tests/test_packed_context.py \
      tests/test_self_play_model.py

The complete non-integration command is:

    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    HAL_O65_PROXY_CHECKPOINT=runs/proxy/final.pt \
    uv run pytest -q -rs -m "not integration" --basetemp="$TMPDIR/p"

The eight skips are:

- test_netplay_hardware.py::test_declared_profiles_meet_complete_path_budget
- test_netplay_hardware.py::test_prepared_profiles_do_not_compile_or_capture_during_play
- test_policy_world_schema.py::test_real_v7_world_round_trip_preserves_ranks_and_items
- test_policy_world_schema.py::test_world_slice_matches_full_decode
- test_policy_world_schema.py::test_world_to_policy_is_payload_exact
- test_policy_world_schema.py::test_world_columns_round_trip_through_mds
- test_policy_world_schema.py::test_decoder_rejects_reserved_presence_and_metadata_bits
- test_policy_world_schema.py::test_world_slice_decoder_exposes_rank_and_items

The first two require HAL_REQUIRE_NETPLAY_HARDWARE_QUALIFICATION=1 on the
production GPU. The other six require the absent v7 subset and fail the handoff
gate. No substitute data is used.

Native and Dolphin commands run separately:

    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HAL_REQUIRE_INTEGRATION=1 \
    HAL_O65_PROXY_CHECKPOINT=runs/proxy/final.pt \
    PYTHONPATH="$HOME/src/melee-sim-light" xvfb-run -a \
    uv run pytest -q -rs tests/test_native_sim_integration.py \
      tests/test_065_loop.py -m integration --basetemp="$TMPDIR/n2"

    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HAL_REQUIRE_INTEGRATION=1 \
    xvfb-run -a uv run pytest -q -rs tests/test_roundtrip.py \
      tests/test_session_cleanup.py -m integration --basetemp="$TMPDIR/d"

The paired check uses the final two-update manifest against the reference,
both development cadences, both port orientations, selection seed 17, and a
132-frame budget:

    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    PYTHONPATH="$HOME/src/melee-sim-light" xvfb-run -a \
    uv run python -m scripts.eval_o65 \
      --checkpoint runs/o65-cleanup/final/checkpoints/update-00000002.json \
      --anchors runs/proxy/final.pt --output runs/o65-cleanup/paired-evaluation \
      --data-dir "$HOME/src/melee-sim-light/data" \
      --protocol runs/o65-cleanup/evaluation-protocol.json \
      --lambda-grid 1 --cadence-grid 8/4 10/2 --compare-events --device cuda

The private protocol copies pilot-evaluation.json, sets the selection seeds to
[17], and sets max_frames to 132. All eight matches stop at the frame budget.
Win rates remain null. The run tests execution and result publication.
The final proxy and paired runs use clean HAL source
4f9e190bdd6d5ae94d0ec30743e1515db84810aa.

## Simulator fork

MSL branch o65-hal-parity contains cleanup commit
2dbca5e3214d88859ac80333cc19caa328d4be4a. It changes private helper names and
comments while preserving the public replay ABI. Linux x86-64 and GCC 15.2
produce library SHA-256
9f16aefd23377abf4a94a185b702bc34033cf890d63b88e79f410ef5efb7c78a.
The current pilot delivery manifest pins this commit and its Git archive hash.
Earlier performance measurements retain their original simulator identity.

Run from the MSL checkout, using the HAL environment as PY:

    make native python-library validator source-check -j2 PY="$HAL_PYTHON"
    PYTHONPATH="$PWD" "$HAL_PYTHON" -m pytest -q \
      tests/test_public_recorded_replay.py --basetemp="$TMPDIR/m1"
    make native-smoke -j2 PY="$HAL_PYTHON"
    make native-smoke -j2 TIMEOUT="$PWD/reports/triage/o65-cleanup/timeout-60" \
      PY="$HAL_PYTHON"
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 make validation-suite \
      VALIDATION_WORKERS=2 VALIDATION_BUILD_JOBS=2 VALIDATION_ARGS=--no-build \
      PY="$HAL_PYTHON"

Build and source checks pass. The public replay tests pass 2/2. The default
native-smoke command fails at batch-api-smoke with timeout status 124 under
its five-second bound. The separate timeout-60 wrapper replaces each command's
bound with 60 seconds; all native checks then pass. The committed Makefile
retains its original limit. The full replay gate passes 512/512 recordings,
4,721,700 frames, with zero diagnostics, failures, or errors. It takes 87.812 s
with two workers and matches every saved output fingerprint.


## Remaining qualifications

The shared fixture at
`data/processed/ranked-anonymized-1/mds-v7-sub4/train` is absent. Its six
schema tests require that exact input. A skipped required fixture fails the
handoff gate, even when pytest returns zero.

The residual critic averages opponent action embeddings and offset embeddings
before its MLP. It cannot distinguish permutations of a queue with the same
length and actions. Correcting that limitation changes the critic architecture
and needs a separate experiment.

Production O64 gameplay, the RTX PRO 6000 learning pilot, two-GPU NCCL training,
reset-heavy throughput, and deployment deadlines still need qualification.

## RTX PRO 6000 pilot preparation, 2026-10-18

The [protocol](pilot-protocol.md) now defines the treatment, fixed control,
input identities, qualification, warmup, evaluation schedule, resume check,
learning criterion, launch, monitoring and teardown. The delivery manifest
pins MSL 2dbca5e and verifies its commit and Git archive. The launch uses a
container digest, one g4-standard-48 and a 24-hour cap. No VM was created,
no input was uploaded, and no Git commit was pushed during preparation.

The final code changes end at b03e36d9de0c18b1fffe077d11eb359952d9079f.
The handoff includes a later documentation commit; the final report records
its exact launch SHA. The launcher uses the clean HEAD, checks publication
before allocation, and the VM verifies that SHA after checkout.

Local validation uses the RTX 3060, Python 3.14.4 and Torch 2.11.0+cu130.
The clean tiny run uses source 839f9c63bb4d910032dba8e8cc4dbb778b506549.
Later code changes concern VM cleanup; the later diagnostic budget change
does not affect tiny mode, which fixes its own budget at 132 frames.

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 bash docker/o65-pilot.sh \
  --output runs/o65-prep-tiny-final --tiny --no-wandb \
  --local-delivery runs/o65-prep-final-installed
```

Result: passed. Qualification steady rates at 2, 4 and 8 active matches are
124.69, 188.59 and 255.01 match-frames/s. Training reaches update 4 after a
forced process-group kill at update 2 and restore. All 16 scheduled MSL matches
publish; source_dirty is false in all four reports. These 132-frame matches
exercise publication, not learning. Their win rates and interval bounds are
null. Tiny mode bypasses critic warmup and online W&B.

At update 3, exact resume compares the next batch and optimizer update:

| Check | Result |
| --- | --- |
| Next batch, both paths | `41613e2a1f672872d5ea096953389fd895e7fe79cda9310574d8b263815b2716` |
| Optimizers and scheduler, both paths | `fd1e5e28bfe6de593cfb43715796cf27b4e9d1e11f43a65a782e444aff3e4153` |
| Maximum parameter error | 0 |

The pinned container is
`us-west1-docker.pkg.dev/centering-star-502613-k3/hal-netplay/hal-netplay-runner@sha256:93b281d12c0614516eca8b5a508dae2723da80e53933e9616bb08d6e926f3016`.
Its locked environment sync passes. Its Python is 3.14.3, with Torch
2.11.0+cu130 and NumPy 2.1.3. A network-disabled MSL and adapter build passes
inside that image with GCC 13.3. The host build passes with GCC 15.2.
The image's baked entrypoint and source are replaced at boot. Online W&B,
VM timers, cloud IAM, quota and real image pulls still need VM qualification.

The O63 checkpoint was also downloaded directly from R2: 156,426,408 bytes,
SHA-256 `65e7e43e5cbb18a365a005922ec1cfd026ad807c045eae13b331f487b4ec82e6`.
The input archive is absent from R2 (HEAD returns 404). Its verified local
copy and explicit publication command are recorded in the protocol. The
launch wrapper checks remote availability before requesting a VM.

Final validation commands use TMPDIR under the supplied scratch directory:

```sh
uv run ruff format --check .
uv run ruff check .
uv run ty check --python-version 3.14 --error-on-warning \
  hal experiments/059_muon_action_sequence.py scripts \
  experiments/065_selfplay_ppo.py experiments/o65/pilot.py \
  tests/test_o65_delivery.py tests/test_o65_pilot.py \
  tests/test_o65_results.py tests/test_launch_gce.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=1 \
HAL_O65_PROXY_CHECKPOINT=runs/proxy/final.pt \
uv run pytest -q -rs -m 'not integration' --basetemp="$TMPDIR/handoff-final"
```

| Command | Final result |
| --- | --- |
| Ruff format check | Passed, 372 files |
| Ruff lint | Passed |
| Ty, required targets and new/changed Python files | Passed, zero diagnostics |
| Non-integration suite | 2,463 passed, 8 skipped, 84 deselected, 26 warnings; 254.05 s |
| Four preparation test files | 41 passed, no skips |
| Native integration | 19 passed, 88 deselected, no skips; 80.92 s |
| Dolphin integration | 8 passed, 6 deselected, no skips; 57.00 s |
| Launch dry run and outer/inner Bash syntax checks | Passed |

The eight skips are the same eight named above: two production GPU profile
checks and six missing-v7 schema checks. The six required-fixture skips remain
a handoff failure. No substitute fixture or schema was used.

Native and Dolphin commands are:

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=1 \
HAL_REQUIRE_INTEGRATION=1 HAL_O65_PROXY_CHECKPOINT=runs/proxy/final.pt \
HAL_MSL_SOURCE="$PWD/runs/o65-prep-final-installed/msl" \
PYTHONPATH="$PWD/runs/o65-prep-final-installed/msl:$PWD" xvfb-run -a \
uv run pytest -q -rs tests/test_native_sim_integration.py tests/test_065_loop.py \
  -m integration --basetemp="$TMPDIR/native-complete"
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=1 \
HAL_REQUIRE_INTEGRATION=1 xvfb-run -a uv run pytest -q -rs \
  tests/test_roundtrip.py tests/test_session_cleanup.py \
  -m integration --basetemp="$TMPDIR/dolphin-final"
```

Preparation failures and corrections:

- The first local install lacked final.pt in its local-payload directory.
  Linking the verified checkpoint there and using a fresh output succeeds.
- The default GHCR image rejects an anonymous manifest request. The launch
  now selects the existing Artifact Registry image by digest. The remote
  describe command fails because local gcloud authentication needs renewal.
- Offline container sync cannot obtain uncached setuptools/wheel build
  requirements. Normal `uv sync --locked` succeeds with network access.
- The first result-helper test collection could not import the scripts
  namespace. It now uses the repository's existing file-module test loader.
- The first tiny run rejects resume after HEAD changes during the run. The
  complete repeat holds HEAD fixed and passes, including exact resume.
- A new late-TERM regression first fails with signal 15. Cleanup now ignores
  late TERM, and the deadline service no longer signals the startup shell.
  All 41 focused cases and the final full suite pass after this correction.
- Two native attempts each return 12 failed, 7 passed, 88 deselected because
  the fresh clone has LFS pointers for the required recordings. The first
  hydration script also selects a module name and fails before copying.
  The corrected script verifies all 12 existing recordings against each
  committed LFS size and SHA-256 before copying them into this worktree.
  Strict verification initially rejects the stale LFS index after manual
  copying. Refreshing that index leaves identical Git blobs and a clean tree;
  strict delivery verification and the complete native suite then pass.
- The first full suite passes 2,462 tests with the same eight skips. The
  final repeat includes the new late-TERM case and passes 2,463 tests.

Self-review examines the plan-to-HEAD diff and its risks. Fixes reject
non-finite qualification memory, separate held-out W&B keys from selection,
anchor timers to VM boot, protect upload from a late signal, and reduce the
frequent diagnostic budget after measuring eager evaluation cost. Full-match
held-out evaluation remains the learning decision. A second review and final
validation follow the cleanup fix. PPO math, model architecture and simulation
behavior are not changed by this preparation. The existing critic queue-order
limitation and production qualifications above remain open.
