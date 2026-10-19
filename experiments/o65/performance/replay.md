# Window replay and actor capture

## Summary

`hal.models.window_replay` defines the fixed tensor interface and replays every
decision in a window together. Temporal attention remains separate per decision;
history attention uses one projected K/V per window. Critic features are detached
at D+1, before the first sampled factor. See the [interface](replay-interface.md).

`GraphedActorStep` captures the trunk, decoder, joint PPO against stored behavior
log-probs, lambda/sigma direct KL, and backward. PPO clips at 0.2. There is no
truncated importance weight. Clipping gradients and Muon remain outside capture.
The small actor objective is a tested reimplementation; experiment losses were
not moved into `hal`.

Compile **only `replay_windows`**, leaving the trunk eager inside the same
CUDA graph. Use default mode, `fullgraph=True`, `dynamic=False`, and
`options={"triton.cudagraphs": False, "emulate_precision_casts": True}` on
Torch 2.11.0. Whole-trunk compilation failed the real-target precision gate.
Without cast preservation, gradient and no-gradient fusion also produce
different BF16 distributions and a false KL cost. Remove the option only after
the FP32-reference gate passes without it.

The approved BF16 commits through `8decbe99` were already ancestors of base
`aacdfee0`. No cherry-pick was needed.

### RTX 3060 measurements

The precision gate passed before timing. Raw evidence is in
[replay-measurements.json](replay-measurements.json), with source, input,
weight, simulator, adapter and data hashes. The pinned proxy checkpoint is
`65e7e43e5cbb18a365a005922ec1cfd026ad807c045eae13b331f487b4ec82e6`.

Control is the current `_evaluate`; treatment shares history across fixed slots.
Both use the same weights, decisions, sampled paths and masks from a real
16-match native rollout. L=256, max_delay=10. Both actor paths use common real
rollout GAE advantages, value targets, episode lambda values and the control's
16 selected decisions per window. The student shift is applied before rollout,
so stored behavior probabilities describe that student.

Times include the trunk, decoder, actor objective/backward where applicable,
graph input copies, and returned metric copies. Window assembly precedes the
treatment; control retains `_evaluate`'s assembly. Optimizers and simulation are
outside this measurement. Medians use five eager/compiled samples or ten graph
samples after warmup. Launch counts come from one bounded profiler sample per
operation. CUDA-event latency includes host launch gaps; wall medians are
within 0.05 ms of those values. Kernel counts exclude memcpy and memset
activities.

At W=4:

| K | Phase | Control ms/batch | Captured ms/batch | Control kernels | Captured kernels | Control host launches | Captured host launches |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | No gradient | 31.25 | 7.11 | 1,629 | 1,198 | 1,629 | 118 |
| 32 | Actor backward | 79.90 | 22.62 | 3,984 | 3,038 | 3,984 | 118 |
| 64 | No gradient | 45.68 | 8.94 | 2,287 | 1,198 | 2,287 | 118 |
| 64 | Actor backward | 113.14 | 29.07 | 5,841 | 3,063 | 5,841 | 118 |

Actor graph latency per window:

| K | W=1 | W=2 | W=4 |
| ---: | ---: | ---: | ---: |
| 32 | 11.13 ms | 7.63 ms | 5.65 ms |
| 64 | 13.13 ms | 8.92 ms | 7.27 ms |

Choose W=4 among these measured sizes: K=32 for R=4, K=64 for R=2.
Eager window replay alone takes 99.61/93.71 ms for actor backward at W=4.
Decoder compilation without graph capture takes 19.59/19.45 ms for no-gradient
scoring. Capture provides most of the launch reduction.

Peak allocated GPU memory in the W=4 probe process is 1396/1834 MiB
for K=32/64, including resident rollout/reference state. This is not the complete
trainer's memory use.

The graph's forward/loss/backward body uses one graph launch. Reported host
launches also include staging strided feature inputs. A dense trainer can
instead supply a few packed tensors and build features inside its callback.
Larger W and the complete dense trainer have not been timed by this stream.

### Precision

The reference uses the same inputs and weights, FP32 for every model operation,
and TF32 off. FP32 trunk attention explicitly uses dense SDPA because the
checkpoint's FlashAttention backend requires reduced precision. Both current
and new paths are compared to that reference. The gate checks RMSE, maximum
error and relative L2 error against 1.5 times control error plus the recorded
absolute floor. The floor is 1e-6 for gradient elements and 1e-5 for other
quantities; relative L2 uses 1e-5. It does not use the retired cosine/norm gates.
The final candidate uses outputs and gradients from the actual actor and
reference CUDA graphs, including the eager trunk inside each capture.

The first whole-trunk compiler configuration failed KL: control RMSE was
3.86e-6 and new RMSE was 4.27e-4. That failure is retained in
[replay-initial-precision.json](replay-initial-precision.json).

Cast preservation passed the initial fixed-loss probes and six timing cases;
that historical evidence is retained in
[replay-whole-trunk-measurements.json](replay-whole-trunk-measurements.json).
Those timings do not qualify the final path. Self-review strengthened the
probe to use the rollout's real GAE, selected decisions and episode lambda.
Whole-trunk compilation then failed KL (control RMSE 4.37e-6, candidate 8.94e-6)
and critic gradients (3.41e-6 versus 5.21e-6). See
[replay-compiled-trunk-precision.json](replay-compiled-trunk-precision.json).
The implementation therefore compiles only the decoder.

Every captured quantity passes. Errors against the FP32 reference:

| Quantity | Control RMSE | Captured RMSE | Control max | Captured max |
| --- | ---: | ---: | ---: | ---: |
| Factor log-probs | 0.00532 | 0.00474 | 0.0746 | 0.0634 |
| Factor KL | 4.37e-06 | 4.26e-06 | 6.51e-05 | 5.11e-05 |
| Values | 0.436 | 0.436 | 1.66 | 1.66 |
| Trunk features | 0.00591 | 0.00591 | 0.0457 | 0.0457 |
| Decoder features | 0.00547 | 0.00544 | 0.0252 | 0.0294 |
| Actor gradients | 3.13e-05 | 3.06e-05 | 0.00735 | 0.00633 |
| Critic gradients | 3.41e-06 | 3.42e-06 | 0.000445 | 0.000445 |

Conditions, opponent queues and queue lengths match exactly. Captured versus
functional actor gradients differ by relative L2 0.00376; the captured gradients
themselves pass the FP32-reference gate. This is statistical precision parity,
not bit-exact equality. The evidence records source commit `e15499f3`.

The probe adds a fixed linspace(-0.1, 0.1) to temporal readout biases before
rollout and initializes the critic residual output with std=0.005, so KL and
every critic branch are exercised. Input tensors, model states, hashes,
configuration and environment are retained under `runs/window-replay/scoped-graphs/`.

## Commits

The implementation history starts at `aacdfee0`. The interface was committed
first in `972742da`; shared-history replay is `e3778bd3`; actor capture is
`be0c553d`. Later commits add validation and regression coverage, qualify BF16
compilation, and record the measurement harness and evidence. Use
`git log --oneline aacdfee0..HEAD` for the complete small-commit history.

## Tests run

Final captured-path suite and performance results are pending. All GPU jobs
use the shared `flock` and `systemd-run --user --scope -p MemoryMax=10G`.
`OMP_NUM_THREADS=2`; compiler runs also use
`TORCHINDUCTOR_COMPILE_THREADS=1`. Pytest basetemps are under `$TMPDIR`.

| Command | Result |
| --- | --- |
| `uv run ruff format --check .` | Pass; 389 files already formatted. |
| `uv run ruff check .` | Pass. |
| `uv run ty check --python-version 3.14 --error-on-warning hal experiments/059_muon_action_sequence.py scripts tests/test_window_replay_model.py tests/test_window_replay_actor.py tests/test_window_replay_proxy.py` | Pass; zero diagnostics. |
| `uv run pytest -q -ra -m "not integration" --basetemp "$TMPDIR/handoff-final"` | Before the last captured-output accessor: 2,624 passed, 8 skipped, 91 deselected. |
| `HAL_REQUIRE_INTEGRATION=1 uv run pytest -q -ra tests/test_roundtrip.py tests/test_session_cleanup.py -m integration --basetemp "$TMPDIR/roundtrip"` | 8 passed, 6 deselected; no skips. |
| Focused model/actor/proxy tests with required native integration | Before the last captured-output accessor: 27 passed. |
| `HAL_REQUIRE_INTEGRATION=1 uv run python tests/test_window_replay_proxy.py --output runs/window-replay/scoped-graphs --performance` | Pass on final implementation: all 10 precision quantities and all 6 timing cases. |

The full suite uses `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
`HAL_O65_PROXY_CHECKPOINT=runs/proxy/final.pt`, `HAL_O65_COMPILE_CUDA=1`,
`PYTHONPATH=/home/ericgu/src/melee-sim-light:tests:.`, and
`MSL_CORE_LIBRARY=/home/ericgu/src/melee-sim-light/build/melee_core/python/libmelee_core.so`.
The native probe uses the same environment and `HAL_REQUIRE_INTEGRATION=1`.

The eight full-suite skips are two production-GPU qualifications and six real
v7 schema checks whose `mds-v7-sub4/train` fixture is absent. Per AGENTS.md,
the missing required fixture is a handoff validation failure, even though
pytest reports skips. No substitute fixture was used.

Failed and interrupted attempts, all retained under `runs/window-replay/`:

- The final captured-output suite initially had 3 unrelated stream-test
  failures, 2,621 passes, 8 skips and 91 deselections. Its longer basetemp
  exceeded the AF_UNIX socket path limit. Rerunning under `$TMPDIR/hf` keeps
  socket paths below that limit; no stream code changed.
- Initial full suite: 1 failed, 2,612 passed, 8 skipped, 91 deselected.
  Exact-resume verification required the omitted `CUBLAS_WORKSPACE_CONFIG`.
  The corrected full suite passed all runnable tests. An intermediate corrected
  run was deliberately interrupted after 50 passes to resolve the precision
  failure before repeating the suite.
- Initial GPU actor tests failed raw eager flex-attention capture, then the
  tiny fixture's unsupported head width of 8. Compiling replay and using head
  width 16 fixed both. The proxy's head width is 64.
- Capture then failed with retained default-stream eager autograd nodes. An
  `autograd.grad` attempt had the same failure; releasing the earlier graph
  before standard captured backward fixed it. Regression tests cover repeated
  inputs, parameter changes and stable returned copies in FP32 and BF16.
- A Dynamo fullgraph test rejected an inference-mode query during tracing.
  The eager guard and capture constructor now reject inference mode outside
  tracing; focused tests passed after the fix.
- The first native probe lacked the local adapter. Built it with
  `uv run python -m hal.sim.native_build --native-source /home/ericgu/src/melee-sim-light`.
  Initial harness attempts also exposed a block index (1 instead of 128), a
  checkpoint identity-buffer construction mismatch, and frozen FP32-reference
  parameters. The harness now uses the actual block frame, checkpoint loader,
  and explicit actor-gradient enablement.
- Initial whole-trunk precision failed KL without cast preservation. The
  stronger real-GAE gate later failed KL and critic gradients even with cast
  preservation. Both failures are retained in the linked JSON records. Narrowing
  compilation to the decoder passed the functional gate; final captured results
  are recorded separately. The only source edit after measurement corrected a
  timing comment: the control also includes its slot-layout conversion.
- The first commit-hook type check lacked existing script dependency
  `cairosvg`. Installed it in the local environment with `uv pip install cairosvg`;
  no tracked dependency files changed.

## Self-review findings and fixes

Review used `git diff o65-selfplay-ppo-plan...HEAD`, the plan's objective and
offset definitions, and its failure modes. Substantial findings received a
second review and focused reruns.

- Preserve BF16 casts and keep the trunk eager inside capture. The stronger
  real-target FP32 gate caught whole-trunk compilation errors missed by fixed
  loss probes. Final qualification also reads actual captured outputs.
- Release earlier eager autograd graphs before capture. Torch retains their
  original leaf-gradient stream. Failed preparation restores caller gradients;
  later replay rejects replaced gradient storage.
- Keep critic validity separate from execution validity. Valid decisions whose
  actions never execute still retain critic features and targets.
- Detach behavior, reference, lambda, advantages and all normalization scalars.
  Mask padding before KL arithmetic, including NaN target padding.
- Allow empty local actor ranks when the global denominator is positive.
- Validate all window actions, supported executed offsets, frame-prefix masks,
  conditions and queue lengths before capture. Keep unsupported inference-mode
  checks outside Dynamo tracing.

Focused coverage also checks per-decision output/gradient parity, resets,
poisoned slots, both PPO clipping directions, action-independent critic states,
and updated weights/inputs across FP32 and BF16 graph replays.

## Cross-stream changes

None outside the requested model/inference code, tests and performance notes.
`experiments/065_selfplay_ppo.py` was not edited. The dense stream must adapt to
the committed interface and record the required compiler options.

## Impasses

No scientific choice remains open in this stream. The legacy v7 schema fixture
is absent from the shared read-only fixtures; its six checks skip. This remains
a validation coverage gap. Two production GPU qualification checks also skip
on the RTX 3060.

## Remaining work

Integrate the interface in the dense trainer, then run its joined statistical
parity and exact-resume checks. Complete-update and production-GPU performance
belong to that integration. Larger W remains unmeasured here. The native
precision harness uses the control trainer API from this branch (unchanged
from `aacdfee0`); preserve that control when adapting the dense trainer tests.
