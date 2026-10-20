# O65 dense implementation report

## Summary

The trainer uses dense tick/player tensors, GPU decision selection and window
placement, two simulator buffers, one reference graph and fixed actor graph
steps. PPO uses stored behavior logs. The old decision and grouped paths are
removed. Checkpoint v5 stores pool v3 and validates the resumable counter key.

Final eight-shard updates take 3.157 s at 16 matches and 4.875 s at 64 matches.
At 64 matches, buffer acceptance cuts rollout CPU from 1.788 to 1.001 s and
rollout wall time from 3.992 to 3.211 s. Launch counts stay flat from 16 to 64.
The sampler fix passes the complete FP32-reference precision gate for all seven quantities,
including three actor graph replays. The one Modal profile passes: 11.399 s
per update at 1,024 matches, versus the supplied 67 s baseline. Rollout is
6.669 s and misses its target; actor is 3.260 s and other preparation is
0.346 s. The function used 0.371 GPU-hours; conservative allowance is USD 5.05.
See [phase measurements](performance/dense.md)
and [numerical evidence](performance/dense-precision.json).

The compiled sampler's parallel CDF scan could select a masked class near an
upper draw. It now selects the first positive-probability crossing. Regression
tests cover compiled FP32 and BF16 sampling, GAE overflow, invalid KL values,
actor losses and actor/critic gradients. Invalid training values still fail.
Resume restores complete state exactly and compares the next update with the
measured spread of two control repetitions; it uses training's execution mode.

## Commits

- Final evidence commit: Record 1024-match dense Modal profile
- `1a920ac7` Record completed dense validation
- `d2079d65` Record dense qualification and buffer measurements
- `490af376` Use vector RMS for numerical resume checks
- `c4e26c27` Accept simulator frames once per buffer
- `73c9ba1e` Reject critic loss overflow before stepping
- `537139e1` Measure resume noise across state vectors
- `3fea389f` Check resume against measured update noise
- `694d75a1` Keep compiled samples on valid action support

- `2cb0cc25` Use stored behavior and device selection
- `e1b7456d` Merge branch 'o65-replay' into o65-dense
- `f7bf145b` Store rollout ticks in fixed player slots
- `c456d16e` Record dense rollout measurements
- `382c9300` Overlap simulator buffers with inference
- `7f896aea` Record simulator overlap measurements
- `5228c591` Merge branch 'o65-replay' into o65-dense
- `ba2c2779` Preserve actor loss settings and support
- `2e76e800` Add critic-only window queue replay
- `98ed0d20` Replay dense windows in fixed actor steps
- `056abba5` Declare actor test compilation shapes
- `f5cf13b0` Validate dense window profile coverage
- `d69c5947` Record window replay measurements
- `0c6cc7e6` Merge branch 'o65-replay' into o65-dense
- `7012de66` Evaluate only occupied bootstrap roles
- `2cf4de62` Record smaller bootstrap measurements
- `624868be` Remove the old decision paths
- `2f99e37e` Bound the dense Modal profile
- `0f9b1a43` Clear rewards before reusing rollout slots
- `ed70e87f` Match reference and actor arithmetic
- `3640aae2` Qualify the dense model path
- `4b176fba` Protect qualification evidence
- `13a3cdb3` Match reference window order and padding
- `72a44d36` Validate counter key field types
- `c30c54ba` Use the recorded rollout batch layout
- `03a96478` Update callers for dense training
- `dba11ab0` Match the full reference forward arithmetic
- `2912b618` Check repeated reference and actor replays
- `0572476f` Compare complete match trajectories
- `f7249d78` Use common masked KL for trajectory reports
- `01cd2313` Record qualified dense measurements

## Tests run

All local GPU commands use the shared lock, MemoryMax=10G, OMP_NUM_THREADS=2,
TORCHINDUCTOR_COMPILE_THREADS=1 and CUBLAS_WORKSPACE_CONFIG=:4096:8. Native
commands set PYTHONPATH to the MSL source. Pytest basetemps use TMPDIR.

- `uv run ruff format --check .`: pass, 393 files.
- `uv run ruff check .`: pass.
- `uv run ty check --python-version 3.14 --error-on-warning hal experiments/059_muon_action_sequence.py scripts` plus every new Python file: pass, zero diagnostics. The additional dense/replay changed-test target also passes. An optional broader check of every changed Python file reports 13 existing diagnostics in unchanged lines of tests/test_action_sequence_attention.py; that pre-existing test is outside the maintained/new-file target. No runtime code is omitted from the required check.
- `uv run pytest -q -ra -m "not integration" --basetemp "$TMPDIR/dense-handoff-qualified"`: 2608 passed, 4 failed, 8 skipped, 85 deselected, 30 warnings (783.77 s). One exact Muon test used nondeterministic mode; its deterministic-mode CPU retry passes. Three unrelated netplay failures were Unix socket paths longer than AF_UNIX permits. All 21 netplay checks pass under the shorter `$TMPDIR/n` base. The final short-path GPU run passes below.
- Final full suite: `uv run pytest -q -ra -m "not integration" --basetemp "$TMPDIR/n"`: **2676 passed, 8 skipped, 85 deselected, 27 warnings**, 646.91 s. No test failures. Six missing-fixture skips remain validation failures under AGENTS.md. Log: runs/dense/dense-handoff-final-v5.log.
- Required Dolphin integration: `HAL_REQUIRE_INTEGRATION=1 uv run pytest -q -ra tests/test_roundtrip.py tests/test_session_cleanup.py -m integration`: 8 passed, 6 deselected, 7 dependency warnings, no skips (58.47 s).
- Final expanded integration command also includes tests/test_native_sim_integration.py and tests/test_batched_self_play_integration.py: 13 passed, 12 failed, 6 deselected, 8 warnings (434.37 s). All 12 failures require the missing zstandard package. Both shared-weight and distinct-weight batched/serial action and outcome comparisons pass. Installed local test prerequisite zstandard 0.25.0; native integration retry: **15 passed, no skips**, 58.93 s. No repository dependency or source changed. Logs: runs/dense/dense-integration-final-v5.log and dense-native-integration-retry.log.
- Initial precision qualification: five complete attempts. Final sampler and buffer follow-ups also pass all seven gates. Large-reference and shape-only paths failed the KL gate. The hidden-leaf path passed once, then failed the stricter recorded-student-layout repeat. The final read-only scoring copy passes all seven quantities on three actor replays. Floors were not relaxed.
- Single Modal profile: exit 0, 1,024 matches, 16 shards, one warm update; all three updates report finite metrics. All 21 artifact hashes and trainer source hash verify. The container is stopped; no smoke or retry ran. Timings, counts, cost allowance and unresolved rollout limit are in dense-modal.json and dense.md. One monitoring CLI call failed due to a missing argument separator; the corrected command passed. A final monitoring read reached the already stopped container and reported successful task completion.
- Final `profile` commands at 16 and 64 native matches on eight shards: pass, one warm update each. All five steps and the before/after buffer follow-up have local measurements in dense.md.
- Final buffer-focused checks: 144 passed, one CUDA-only diagnostic skip and 12 deselected on CPU; five passed and 64 deselected with CUDA, including live-lane continuation and native stream/reset/resume cases.
- Final resume comparison unit checks: 17 passed, 71 deselected. The scalar critic-overflow regression failed before the check and passes after it. Sampling regressions: two compiled failures before the fix, 15 passed after; the endpoint-only attempted fix still failed both compiled cases.
- An interim handoff was deliberately interrupted for the RMS contract change: 269 passed, 85 deselected, 16 warnings, 204.24 s. The source-verified final handoff replaces that incomplete run.
- Initial individual-tensor native resume gate failed 70/426 tensors despite exact restoration. The complete-vector RMS/max rule passed once, then failed only the Adam second-moment maximum in a later repetition. Both failed reports remain in evidence. The final RMS-only rule passes; maxima stay diagnostic as permitted by the user.
- Final native resume: pass, every state hash exact on three restores. Next-update model RMS error 1.57611e-7 versus control spread 1.43917e-7. All optimizer-state families pass the calibrated RMS rule; maximum errors remain diagnostic.
- Statistical comparison: three 64-match updates in each arm complete. KL and median advantage/value target differ; reward, entropy, episode-length and reset intervals include zero. Statistical equivalence is not established. See dense-statistical.json for all intervals, including earlier attempts.
- Skips in the full suite: two production netplay hardware qualification tests are disabled; six legacy world-schema tests lack `data/processed/ranked-anonymized-1/mds-v7-sub4/train`. The six missing-fixture skips count as validation failures under AGENTS.md.
- Earlier trajectory-analysis checks: 9 passed, 1 integration test deselected. A later harness-call retry passes 3 tests, 1 deselected.
- Focused CPU migration checks ultimately pass 147 tests, with two CUDA-only diagnostic skips and 19 deselections. The broad reference-path CPU checks pass 89 tests with 11 deselections. The read-only copy checks pass 7 tests with 28 deselections. Reward-padding regression: 1 pass. Counter-field checks: 5 passes. Pilot CLI checks: 10 passes. Modal launcher checks: 22 passes.
- Focused CUDA reset, queue ownership and exact-resume checks: 5 passed, 37 deselected, 15 dependency warnings (536.94 s), before the final reference correction. The final full run rechecks this path.

Earlier failures remain recorded. Step 1 had six stale KL callers (152 passed,
6 failed, 1 unset-proxy-path diagnostic skip); all six pass on retry. Two early
profiles lacked the MSL PYTHONPATH and passed after the environment fix. Step 2
found queue validation, final-bootstrap ordering, capacity-2 padding, borrowed
storage and compiler-inspection defects; focused tests cover each fix. Step 3
found cold-capture stream interference and a BF16 feature-storage cast. Step 4
found an obsolete fixture width, Torch's eight-variant compiler limit and an
old chunk-based profile guard; all retries pass. Step-5 migration runs failed
on stale test APIs, CPU/CUDA fixture devices, clock expectations and a warmup
helper name; the fixtures were corrected. One full unit run was deliberately
interrupted after 119 passes to fix the precision failure. The first statistics
harness run called the frozen control's two-argument KL helper with three
arguments; the common masked report calculation fixes it. The new analysis
helper initially failed import ordering and type annotations; both checks pass
after correction. One scratch edit used unavailable `python`; rerunning with
`uv run python` completed it. Pre-commit checks for documentation-only commits
report no applicable files; these are not test skips.

## Self-review findings and fixes

Reviewed the requested `git diff o65-selfplay-ppo-plan...HEAD`, with detailed
checks of the stream's changes against aacdfee0 and the merged replay boundary.
Checked offsets, execution masks, mixed cadence, queue snapshots before either
policy decides, reset and bootstrap ordering, GAE holes, stop-gradients,
parameter sets, padding and placement, captured-storage ownership and resume.
All 26 changed JSON evidence files parse. An AST scan of all 71 changed Python
files confirms the model/simulator import boundary.

- Clear stale reward padding before reusing a rollout table. GAE already masked it; summary metrics did not. A two-rollout regression covers the fix.
- Preserve configured PPO clipping and infinite KL outside reference support. Tests cover both.
- Serialize cold tick captures before simulator workers use shared queue state. Keep subsequent inference and transfers on each buffer's stream.
- Compute timeout and final bootstrap states before reset. Validate following-state stream, episode, clock and write mask.
- Clear execution caches on restore, evict obsolete tick layouts and preserve controller lookup storage addresses.
- Reject float-valued counter fields before mutating restored state.
- Replace strict Muon update equality with exact restored-state hashes and measured vector RMS in normal training mode. Integer state and scheduler remain exact; maxima are diagnostic.
- Remove the stale pilot `--compiled` argument and parse the real generated CLI in its regression test.
- Match actor reference window order, padding, slot shape and full forward gradient metadata. The stored reference stays frozen. Its independent scoring copy has no optimizer, no backward and no gradient buffers. Three complete precision replays pass.

The next review covered the sampler, non-finite checks, revised resume contract,
coalesced acceptance and their callers and tests. It found and fixed invalid KL
support conversion, dense GAE overflow, actor loss/gradient faults, critic loss
reduction overflow, lossy integer comparison and entropy over padded R=2 frames.
Serial-versus-batched CPU/CUDA parity covers staggered boundaries, unsorted
matches, live-lane continuation and bootstrap ordering. The last review found
no further implementation defect. Statistical differences remain a stated limit.

## Cross-stream changes

- `hal/models/sampling.py` and `tests/test_model_sampling.py`: correct compiled CDF sampling on masked support.
- Plan, pilot protocol and README: state-exact restore and measured next-update noise.

- `hal/models/self_play.py`: a validation switch for captured critic calls; validation remains the public default.
- `hal/models/action_sequence.py`: public critic-only window queue replay.
- `hal/models/replay_actor.py` and `hal/inference/graphed_actor.py`: configured clipping and KL support handling, with tests.
- `hal/eval/self_play_checkpoint.py`: training format v5.
- Delete the unused `hal/inference/action_rng.py`; frozen-control tests recover its recorded version from Git.
- `scripts/launch_o65_modal.py` and its tests: bounded dense profile, one group and one warm update, no obsolete options.
- `experiments/o65/pilot.py`, README and validation notes: update callers for the single dense path.
- Replay interface, implementation and qualification commits merged from local o65-replay. No other worktree was edited.

## Impasses

This host has only 12 logical CPUs. Its distinct-CPU contract prevents a
16-shard local run. Local measurements use eight shards for the requested
16/64 match sizes; the single Modal profile used 16 distinct shard CPUs.

The plan supplies no numerical equivalence margin for statistical parity.
The final 64-match comparison shifts KL and the median advantage and value
target. The preceding run also shifted return spread and tails. Both are kept.
An agreed margin and a longer prespecified comparison are needed to establish
equivalence; nonsignificant differences do not establish it.

The six legacy v7 schema tests lack their recorded dataset fixture. Its expected
path is under the read-only processed-data symlink. No main-worktree fixture
was created or changed. This is an unresolved validation failure.

A timeout between decisions leaves Q=D+R-Delta ego actions committed, while
the critic definition specifies z_(D+1). The existing D-prefix bootstrap is
preserved. A scientific choice between that definition and consuming all Q
remaining ego actions is still needed; this rewrite does not silently change it.

## Remaining work

Implementation, local validation and the single Modal profile are complete.
Rollout and total-update time still miss their targets (6.669 and 11.399 s).
The trace points to outer-worker CPU and transfer synchronization as the next
investigation, without establishing a pure simulator limit. The six missing
fixtures, statistical equivalence and timeout bootstrap choice remain open.
No further cloud job is running or planned. Branch o65-dense is published.
