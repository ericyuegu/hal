# O65 rollout tensors: parity and local performance

The rollout stores observations, processed actions, queues, and decisions in
reserved tensors. The final native tests match the recorded control exactly
for two updates at 16 and 64 matches, with timeouts and resets inside each
rollout. This includes both parameter hashes after each update.

The local performance result is mixed. At update 3, complete update time rises
from 8.200 s to 8.919 s at 16 matches and falls from 15.122 s to 14.723 s at
64 matches. Rollout time rises at both sizes. No 1,024-match or RTX PRO 6000
performance claim follows from these measurements.

## Treatment, control, and inputs

- Control: `4e610a23`, whose experiment source is identical to `e6d768d5`.
- Treatment: `b367dea5`.
- Control experiment file SHA-256: `385071d29e9cfdc1cfcb5c432b62a61a747b5449d24b2e4d4ff17debbec45337`.
- Treatment experiment file SHA-256: `c20eae1a5e413e25a01965b3f72260bf80cfe862f822dfc6145d0ca4a603b243`.
- Both use the O63 bootstrap30 proxy, seed 65, one group, four shards on CPUs
  0–3, 256 rollout frames, delay 8/replan 4 and delay 10/replan 2.
- Both use the default training algorithms, Muon, four critic epochs,
  32 windows per actor step, 16 selected decisions per window, and no critic
  warmup. Sampling graphs are enabled. The trunk replay batch stays at
  16 windows; decoder replay stays at 128 decisions.
- Each arm starts fresh and runs three updates. Update 1 includes graph
  preparation. A separate rollout after update 3 supplies cProfile data.
- These timing runs use `max_frame=28800`. They contain no resets, timeouts,
  or idle frames. Exact parity uses `max_frame=0` and deterministic algorithms.
  Native resume tests also cover `max_frame=140`, odd endings, and retained
  contexts across updates.
- Every GPU process holds the shared lock and runs in a systemd scope with
  `MemoryMax=10G`. `OMP_NUM_THREADS=2`; no local run exceeds 64 matches.

| Input | SHA-256 |
| --- | --- |
| O63 checkpoint | `65e7e43e5cbb18a365a005922ec1cfd026ad807c045eae13b331f487b4ec82e6` |
| Sampler | `c5ffb1b5f75aa0d9245192304871c2979de9a3e59ddf0c5d55e6877994ab661e` |
| MSL data manifest | `4c2b074e872a7ad55176b63a3971fd6a3bbf09dd1988240d1dbdaf61a63616cb` |
| MSL library | `9f16aefd23377abf4a94a185b702bc34033cf890d63b88e79f410ef5efb7c78a` |
| HAL adapter | `e9e2bf2ee60a3cf31034f65a12505f29e06af010971d20d569c44b2849c5feb2` |

The training data manifest is
`b97eab90e761bcf2bf03b48981f0ab6acc1ac3057157c58ae0c5a72c76c43bd8`.
Feature statistics are the proxy's recorded mixture statistics
(`6870bda6c0970826f6467407647cf9bb5901ef246b6947d87518db67252cdde3`).
The environment is Python 3.14.4, Torch 2.11.0+cu130, NumPy 2.1.3, CUDA 13.0,
and an RTX 3060 12 GB. TF32 matmul is disabled; the cuDNN TF32 flag is enabled.
World size is one.

## Complete updates

All durations are seconds. Throughput counts complete match-frames per update.
Checkpoint writing, evaluation, and the profile rollout fall outside these
measurements. The complete update includes diagnostics and phase cleanup.

| Matches | Arm | Update | Rollout | No-grad | Actor | Critic | Complete update | Match-frames/s |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | Control | 1 | 9.293 | 0.796 | 0.592 | 0.130 | 10.865 | 377.0 |
| 16 | Control | 2 | 6.836 | 0.766 | 0.543 | 0.048 | 8.240 | 497.1 |
| 16 | Control | 3 | 6.799 | 0.767 | 0.539 | 0.048 | 8.200 | 499.5 |
| 16 | Tensors | 1 | 9.687 | 0.665 | 0.569 | 0.129 | 11.094 | 369.2 |
| 16 | Tensors | 2 | 7.740 | 0.625 | 0.504 | 0.050 | 8.961 | 457.1 |
| 16 | Tensors | 3 | 7.700 | 0.624 | 0.504 | 0.048 | 8.919 | 459.2 |
| 64 | Control | 1 | 12.098 | 2.961 | 2.016 | 0.244 | 17.479 | 937.3 |
| 64 | Control | 2 | 10.022 | 2.880 | 2.026 | 0.164 | 15.232 | 1075.6 |
| 64 | Control | 3 | 9.848 | 2.945 | 2.025 | 0.164 | 15.122 | 1083.5 |
| 64 | Tensors | 1 | 12.384 | 2.427 | 1.882 | 0.248 | 17.078 | 959.4 |
| 64 | Tensors | 2 | 10.213 | 2.390 | 1.799 | 0.170 | 14.692 | 1115.2 |
| 64 | Tensors | 3 | 10.238 | 2.384 | 1.809 | 0.173 | 14.723 | 1112.8 |

At update 3, no-grad time falls by 18.7% at 16 matches and 19.1% at 64 matches.
Actor time falls by 6.6% and 10.7%. Complete update time rises by 8.8% at
16 matches and falls by 2.6% at 64 matches. Three updates do not establish
long-run variance or a general speed improvement.

| Matches | Arm | Update 3 allocated GiB | Update 3 reserved GiB |
| ---: | --- | ---: | ---: |
| 16 | Control | 2.047 | 2.230 |
| 16 | Tensors | 2.055 | 2.230 |
| 64 | Control | 3.680 | 3.906 |
| 64 | Tensors | 3.716 | 3.926 |

## Rollout profile

cProfile records the next rollout after the three timed updates. Cumulative
function times include CUDA waits and overlap across shard threads; they must
not be added to estimate wall time.

| Matches | Arm | Python/C function calls | `_state` calls | `_packed_inputs` calls | Context gather calls | `_accept` cumulative s |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 16 | Control | 1,114,411 | 3,293 | 698 | 0 | 0.019 |
| 16 | Tensors | 921,045 | 0 | 0 | 698 | 0.597 |
| 64 | Control | 2,644,922 | 12,789 | 760 | 0 | 0.052 |
| 64 | Tensors | 1,276,230 | 0 | 0 | 760 | 0.807 |

At 64 matches, function calls fall by 51.7%. The normal rollout does not create
`DecisionState` objects or call the frame-walking `_packed_inputs` path.
`SimFrame` views remain for reset observations, explicit proxy tests, and
hash diagnostics. They are created only when requested. CPU and GPU policy
paths use the same frame tensors.

The profile also exposes the cost added by device scatter and index operations
in `_accept`. Batching frame and reward coordinates into one transfer reduces
that cost, but does not remove the local rollout regression. Direct transfers
to final frame storage and fewer small queue kernels remain performance work.
The decoder still dominates these small local batches.

## Storage and format choices

Each collection reserves its complete frame store and decision table before
the frame loop. A finished rollout owns those tensors so that a later
collection cannot alter its training data or diagnostic hashes. Reset rows
have separate space; a reset cannot overwrite the preceding timeout bootstrap.
Native staging blocks remain separate from final group storage, which adds a
device copy. This is a deliberate deviation from reusing one observation ring.

Queues, sampled factors, behavior log-probabilities, rewards, context addresses,
and decision status columns reside on the policy device. Host metadata mirrors
drive clocks, resets, RNG keys, and window placement. Scalar status columns are
published once at the end of collection. Episode ends set one episode entry;
execution masks use indexed episode entries instead of scanning old records.

Decoder hidden-state and memory stacks retain their original order to preserve
backward reduction order. The 16-window and 128-decision replay sizes are
unchanged. Reference evaluation during critic warmup and the critic's shuffle
order are unchanged for this stream.

Training format is now `hal.o65.training.v3`. The tensor pool uses
`hal.o65.rollout_pool.v2`; native pool envelopes use `hal.o65.native_pool.v2`.
Old identities are rejected. A checkpoint stores the last 128 previous frames
plus the current observation, queues, heads, lengths, episode metadata, and
clocks. KV caches and decision tables stay out of checkpoints.

The quickfix stream changes critic ordering. After that merge, the integrator
must qualify this golden fixture against a control with the same ordering;
changing the fixture to accept a different actor or critic hash is not evidence
of columnar parity.

## Evidence and reproduction

The committed driver is `tests/test_065_columnar.py`. Its 14 KiB fixture is
`tests/fixtures/o65_columnar.json`. It checks frame bytes and processed actions
for every native output, sampled factors, behavior log-probabilities, interval
rewards, `_batch_hash`, advantages, values, old/reference log-probabilities, and
both parameter hashes after each update. The fixture records resolved sampler,
training, schedule, environment, and artifact identities.

Raw local evidence is under `runs/columnar/`:

- `before-parity-{16,64}/capture.json`: control fingerprints.
- `before-performance-{16,64}/{capture.json,rollout.prof}`: control timing/profile.
- `final-performance-{16,64}/{capture.json,rollout.prof}`: treatment timing/profile.
- `native-final.log`: six passing native integration tests, including both
  exact golden captures and CPU/CUDA native resume checks.
- `handoff-final.log`: final full unit run.
- `required-integration.log`: eight passing Dolphin integration tests.
- `resume-final/`: the maintained native CLI and exact-resume verifier at
  16 matches with `max_frame=140`.

Use the shared lock for each run. Set `TMPDIR` to a directory under
`~/data/scratch/o65-tmp/columnar`, `OMP_NUM_THREADS=2`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTHONPATH` to the MSL checkout, and
`MSL_CORE_LIBRARY` to its `build/melee_core/python/libmelee_core.so`.
Build the HAL adapter in this worktree before running the tests.

```sh
flock /home/ericgu/data/scratch/o65-tmp/gpu.lock \
  systemd-run --user --scope -p MemoryMax=10G \
  uv run python tests/test_065_columnar.py \
  --matches 16 --output runs/columnar/repeat-16 --performance
```

Repeat with 64 matches and a distinct output directory. Omit `--performance`
for the two-update parity capture. Add `--verify-resume` to run the maintained
resume verifier. For the control, export the experiment source from
`4e610a23` into `runs/columnar/control.py` and set
`HAL_O65_CAPTURE_SOURCE` to its absolute path. The held control copy was checked
byte for byte against that Git object before reporting these results.

## Validation and review

Final commands use this worktree's environment. GPU commands use the shared
lock, the 10 GiB scope, and two OMP threads. Pytest basetemps are under `TMPDIR`.

| Command | Result |
| --- | --- |
| `uv run ruff format --check .` | 375 files pass |
| `uv run ruff check .` | Pass |
| `uv run ty check --python-version 3.14 --error-on-warning hal experiments/059_muon_action_sequence.py scripts experiments/065_selfplay_ppo.py tests/test_065_columnar.py` | Zero diagnostics |
| `uv run pytest -q -ra -m "not integration"` | 2,499 pass, 8 skip, 86 deselected; 26 warnings |
| `uv run pytest -q -ra tests/test_065_loop.py tests/test_065_columnar.py -m integration` | 6 pass, 102 deselected; no skips |
| `HAL_REQUIRE_INTEGRATION=1 uv run pytest -q -ra tests/test_roundtrip.py tests/test_session_cleanup.py -m integration` | 8 pass, 6 deselected; no skips; 7 warnings |
| `uv run pytest -q -ra tests/test_065_columnar.py -m "not integration"` | 14 pass, 2 deselected on CPU/CUDA |
| `uv run pytest -q -ra tests/test_065_loop.py -k wire_lookup` | 2 pass, 90 deselected; all codec values match on CPU/CUDA |
| Native CLI with 16 matches, deterministic algorithms, `max_frame=140`, and `verify_resume=True` | Exact next batch, actor/critic optimizer state, scheduler, and model update; maximum parameter error 0 |

The eight unit skips are two production-GPU netplay qualification tests and
six tests whose optional local v7 dataset subset is absent. The proxy fixture,
MSL data, ISO, Dolphin, and required replay fixtures were present. The integration
suites have no missing-fixture skips. Warnings concern Torch JIT and flex
attention, WebSocket connection APIs, and `fork` in a process with multiple
threads.

The final resume verifier records these matching hashes:

- Next batch:
  `d00de96ba6ff40b016a878bf0d9b93d49149990e907781a43c998a14f6d8eaad`.
- Optimizers and scheduler:
  `e608dfc8043dd69a89adb32ea10122c1cb22d6d3f6975e85097422cc1f61be51`.

Earlier diagnostic runs and corrections remain in `runs/columnar/`:

- Initial type checking failed because the new local environment lacked
  CairoSVG. Installing CairoSVG 2.9.1 in this worktree's environment resolved
  that diagnostic. No dependency file changed.
- A CPU-only run was interrupted after 28 passes so that GPU tests would not
  run without the lock. A subsequent run with CUDA hidden had 3 failures,
  65 passes, and 20 skips: one CUDA case lacked an availability guard, and
  two evaluation tests used a shard fixture whose reset and step schemas
  differed. The stock fixture now returns stock columns on reset. Final GPU
  runs pass. CPU diagnostic runs exclude CUDA cases explicitly.
- A focused snapshot test failed once because CPU snapshots shared live queue
  storage (1 failed, 5 passed). Explicit copies fixed it. Later focused CPU
  checks pass, including 77 loop/storage tests with 13 CUDA skips and
  16 deselected cases, and 11 storage tests with 5 deselected cases.
- The first full unit run passed 2,497 tests with the same eight optional
  skips. The final run adds the two native scatter cases and passes 2,499.

The first review found CPU snapshot aliasing, native finite/end-flag checks
lost when frame objects were removed, an explicit packed-input adapter that
read the CPU action copy, and scalar decision columns that had not yet moved
to the device. Those findings are fixed and have focused coverage. Profiling
then identified excess coordinate transfers; one batch now carries frame and
reward coordinates. The second review checked offsets, odd terminal frames,
execution masks, bootstrap ordering, queue snapshots, detach placement, storage
ownership, format rejection, and the complete diff against the O65 plan. It
found no further defects in this stream.
