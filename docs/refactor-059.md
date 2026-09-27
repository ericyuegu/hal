# HAL 059 refactor and qualification

Status: implementation in progress. The registered goal remains active. A CPU
suite, a synthetic parity check, or a direct inference benchmark does not satisfy
the required production checkpoint, emulator, service, or hardware gates.

## Scope

Maintain experiment 059 and the complete replay-to-policy path. Preserve its
model, optimization, objectives, controller vocabulary, replay sampling, published
corpus, exact resume, and official dense gameplay evaluation. Retire historical
runtime support. Archive research source and evidence without creating an archive
copy of the library or tests.

Control source: `d7454f9d1a136f7c745d2d4478af22e19cd65faf`.
Implementation branch: `codex/o59-refactor`.
The immutable control checkout is `/tmp/hal-059-control-d7454f9`.
The existing `.gitignore`, edited `notebooks/039_scaling_viz.py`, untracked
`notebooks/main_stick_usage.ipynb`, and `outputs/` are preserved.

Production deployment is a separate action. Qualification does not authorize
replacing the live deployment.

## Commit review order

The feature branch is `codex/o59-refactor`, based on the control source above.
The implementation is recorded as the following ordered migration series.
Validation covers the combined source tree. Intermediate commits have not been
qualified as standalone repositories. Per-commit hooks were disabled only for
recording this series; the required checks were run explicitly on the complete
source tree. No repository or global hook configuration was changed.

| Commit | Concern |
|---|---|
| `233b7e13` | Archive completed research and remove its tests |
| `a167b585` | Controller and wire conversions |
| `8905ec60` | Shared observations, batches, identity, and statistics |
| `c6810ee8` | Canonical action-sequence model |
| `6c7ebdb1` | Replay processing and thin data commands |
| `79134359` | Buffered MDS and validation loading |
| `3b399b4e` | Artifact and capability contracts |
| `079b94ca` | Independent cached streams and prepared batch execution |
| `6585191a` | Shared engine, persistent client, and delivery measurements |
| `486fcb66` | One Dolphin process driver and resource cleanup |
| `6a61978f` | One scheduler, local evaluation, and artifact-based H2H |
| `9700afee` | Maintained 059 training and explicit resume lineage |
| `a581c7a7` | Supervised netplay and bounded recovery |
| `5a61254c` | Maintained tools and launchers |
| `f8c647b4` | Python dependencies and maintained check targets |
| `6e22bf91` | Frontend dependency cleanup |
| `da58c965` | Reproducible qualification capture and comparison tools |
| `537a0fa2` | Generated controls and measured parity evidence |
| `590aa57c` | Accounting, controls, and acceptance evidence |
| `384a732d` | Validate netplay soak measurements and report unmeasured gates |
| `e67a51ea` | Match loader file-cache preparation and remove the benchmark's module-global path override |
| `90046957` | Record the cache-controlled loader comparison and verified evidence archive |
| `99b6244d` | Retire unused dense GPU observation buffers after measuring the maintained staging path |
| `22554be2` | Record the failed dense evaluation attempt and service recovery status |
| `9a68b8ac` | Limit real resume qualification to one production checkpoint |
| `587bbac0` | Stop corpus scanning after representative coverage at the user's request |
| `0234ce36` | Prepare and test the single production next-update comparison |

The protected user edits are outside this series. The commits do not indicate
that the remaining hardware, artifact, gameplay, or soak gates have passed.

## Ownership after the refactor

| Owner | Responsibility |
|---|---|
| `hal/models` | Canonical neural computation, controller codec, categorical sampling |
| `hal/representation` | Shared offline/live observation conversion and feature definitions |
| `hal/data` | Extraction, indexing, selection, MDS construction, publication, statistics and identity sidecars |
| `hal/training` | Buffered MDS loading, replay windows, validation loading, optimization and checkpoint utilities |
| `experiments/059_muon_action_sequence.py` | Scientific treatment, training loop, AWR/BC/value objectives |
| `hal/inference` | Artifact validation, execution, observation history, prepared KV storage, stream sampling, engine and clients |
| `hal/eval` | Timing, action scheduling, process-harness adapters, match protocols and results |
| `hal/sim` | Dolphin lifecycle and frame/controller transport; no Torch imports |
| `hal/netplay_service` | Reservation queue, process supervision, readiness, health and replay publication |
| CLI modules | Parse arguments and dispatch to public library functions |

Model imports must not load Melee. Simulation imports must not load Torch.
Package initializers must preserve these import boundaries.

## File and symbol accounting

This table specifies the final destinations. An entry is not evidence that its
migration and retirement have passed every gate. The final file inventory records
individual archive moves and deletions.

| Before | After |
|---|---|
| 059 model plus serving model copy | `models/action_sequence.py`: `ActionSequenceConfig`, `ActionSequenceTransformer` |
| `training/trunk.py` | `models/attention.py` |
| `training/controller_codec.py` and retained scoring primitives | `models/controller_codec.py` |
| `eval/policy_sampling.py` | `models/sampling.py` and `inference/sampling.py` |
| `training/features.py` | `representation/features.py`; `TrainBatch` in `training/batches.py`; remove AWRBatch and spatial/V6 branches |
| `training/canonical.py` | `representation/observations.py` |
| `training/player_identity.py` | `representation/player_identity.py` and `data/player_identity.py` |
| `training/ego_stats.py` | `data/feature_stats.py` |
| Duplicate controller-vector conversions | `hal/controller.py` |
| Melee identifiers in `wire.py` | `data/slippi.py`; button dispatch in `sim/inputs.py` |
| `PhysicalShardReplayLoader` | `BufferedMDSReplayLoader` in `training/buffered_mds_replay_loader.py` |
| Old `physical_shard_loader.py` | Only `PhysicalRow` and `RingSlotDescriptor`, preserving current checkpoint pickle identities |
| Current window operations in `dataloader.py` | `training/replay_windows.py` |
| Current validation path in `dataloader.py` | `training/validation_replay_loader.py` |
| Generic training loader and reservoir families | Delete after current helpers and parity coverage move |
| Data implementation inside CLI files | `data/index_builder.py`, `replay_selection.py`, `mds_materialization.py`, `mds_publication.py`, `professional_replays.py`, `policy_world_v8.py` |
| `backends/history_decoder/policy.py` | `action_sequence_artifact.py`, `action_sequence_policy.py` |
| `HistoryDecoderPolicy` | `ActionSequencePolicy` |
| `backends/history_decoder/kv_cache.py` | `inference/kv_cache.py`, independent rows and prepared scratch buckets |
| `GpuTokenHistory`, `GpuContextHistory` | `GpuObservationUpdates` retains cached update storage in `gpu_observations.py`. Delete unused dense GPU window helpers; the canonical dense executor retains bulk CPU window collation and transfer. |
| `training/context_history.py` | `inference/observation_history.py`: `ObservationHistory` |
| 059 `BF16Inference` | `inference/window_policy.py` |
| `inference/worker.py` | `inference/engine.py`, `inference/client.py` |
| Backend dispatcher and old temporal-AWR backend | Delete |
| `RecedingHorizon` | Inference owns observations; `ActionScheduler` owns timing; delete old class/module |
| Direct local match loop and thread vector driver | One process vector harness and shared rollout records |
| `eval/self_play.py` | Warmup to inference, telemetry to benchmark, matchups to `eval/matchups.py` |
| Historical H2H imports | Explicit policy artifacts and shared engine routing |
| Numbered programs before 059 | Research archive, including 050–058 |
| `059_muon_history_decoder.py` | `059_muon_action_sequence.py`; historical saved experiment identifier remains unchanged |
| Four maintained experiment benchmark/eval tools | `scripts/` |
| `docker/probe_sm120.py` and its startup/skip hooks | Archive the 022-era diagnostic under `archive/scripts/`; remove the hooks and stale launcher claims. Keep current CUDA, compiler-cache, disk-space, and stall checks. |

`data/slippi.py` is a deliberate adjustment to the proposed tree. The existing
`data/conversions.py` imports replay behavior, which itself needs the character
and stage bridges. A separate small bridge module avoids that import cycle.

The v8 command remains `scripts/rematerialize_policy_world_v8_modal.py`.
The old `hal/scripts/rematerialize_policy_world_v8.py` contained library
functions and no command entrypoint. Its functions now live under `hal/data`;
an empty forwarding module would add no supported operation.

No maintained aliases or re-export bridges will remain after caller migration,
except the two persisted loader records. Temporary bridges during implementation
are not the final architecture.

### Data, loader, and optimizer symbols

| Control owner and symbol | Maintained owner and disposition |
|---|---|
| `training.physical_shard_loader.PhysicalShardReplayLoader` | `training.buffered_mds_replay_loader.BufferedMDSReplayLoader`; the schema-3 cursor, replay ring, shard order, and batch schedule remain. |
| `training.physical_shard_loader.PhysicalRow`, `RingSlotDescriptor` | Stay at the old module path as the only two definitions there; current checkpoints pickle those exact paths. `GenerationDescriptor` and `MultiGenerationRingSlotDescriptor` are deleted. |
| `training.dataloader.make_window`, `relabel_ego`, `collate_windows`, `train_batch_from_columns` | `training.replay_windows` retains the current 059 window and batch operations. |
| `training.dataloader._stable_window_rng` | `training.replay_windows.stable_window_rng` is public for the validation loader. Its hash and seed behavior is unchanged. |
| `training.dataloader.make_loader` | `training.validation_replay_loader.make_validation_replay_loader` owns only the pinned 059 validation path. `ResumableStreamingDataLoader` and generic loader modes are deleted. |
| `training.replay_reservoir.ReservoirBatch`, `ReplayReservoir`, `ReservoirLoader` | Deleted; 059 uses the buffered MDS replay loader. |
| `training.ego_stats.consolidate_key`, `load_consolidated_mixture_stats` | Move to `data.feature_stats`; unused `load_consolidated_stats` is deleted. Its sufficient-stat reader now rejects invalid records while preserving the zero-count sentinel. |
| `scripts.build_index.build_index` and `scripts.materialize.process_replays` | `data.index_builder.build_index` and `data.mds_materialization.process_replays`; commands only parse and dispatch. |
| `scripts.filter.FilterConfig`, `filter_index`, `run` | `data.replay_selection.FilterConfig`, `filter_index`, `select_replay_paths`; the option record is frozen/slotted, and the CLI and professional campaign call the named operation. |
| `scripts.publish_mds.audit`, `publish_mds` | `data.mds_publication.audit`, `publish_mds`; audit identifies full versus packed stored columns and requires `projection.json` only for policy-world rows. This corrects the full-MDS publication audit without changing either stored format. |
| `scripts.prepare_professional.PrepareProfessionalConfig`, `prepare_professional` | `data.professional_replays` owns both; the option record is frozen/slotted. |
| `scripts.rematerialize_policy_world_v8.select_manifest`, `inspect_source`, `audit_dataset`, `publish_dataset`, `rematerialize_corpus` | `data.policy_world_v8` owns these operations. The Modal wrapper calls them directly; no empty local CLI stub remains. |
| `training.muon.Muon`, `SingleDeviceMuon`, `MuonWithAuxAdam` | Deleted unused optimizer families and their distributed dependency. `SingleDeviceMuonWithAuxAdam`, Newton–Schulz, logical splitting/scaling, foreach Adam, diagnostic and state behavior remain for 059. Historical `muon_scale_mode` state now fails explicitly. |

### Model, representation, and artifact symbols

| Control owner and symbol | Maintained owner and disposition |
|---|---|
| 059 and serving `GPT`, `SwiGLU`, `NonlinearActionHead`, `ReturnConditioner`, `CausalTemporalDecoder` | One definition in `models.action_sequence`; `GPT` becomes `ActionSequenceTransformer`. Parameter attribute names remain unchanged because they are checkpoint keys. |
| Serving `Architecture`, `AWRCalibration`, `TrainConfig`, `amp_context` | The serving copies are deleted. `ActionSequenceConfig` contains model construction values; the experiment retains its scientific configuration. |
| `initialize_o51_parameters` | `initialize_action_sequence_parameters`; initialization order and random-number use are unchanged. |
| `training.trunk` attention and rotary operations | `models.attention`; retain the 059 training backend and dense reference, remove other backend selection. |
| `eval.policy_sampling.sample_categorical` and serving `sample_with_temperature` | `models.sampling`; the model no longer imports evaluation. |
| `eval.policy_sampling.SlotGroupRng` | `inference.sampling.StreamGroupRng`; stream/generation identity is passed explicitly, and release removes its counters. |
| `training.features.Context` sampling fields | Remove `slot_ids` and `reset`; neural inputs contain features and padding, not inference lifecycle identity. |
| `ContextHistory`, `ContextWindows`, `push_context_rows`, `stack_context_windows` | `ObservationHistory`, `ObservationWindows`, `push_observation_rows`, `stack_observation_windows` in `inference.observation_history`. Preserve the mirrored ring and prepared layout; remove spatial/V6 support. |
| `history_decoder.policy._config_from_state`, `_checkpoint`, `export_o59_policy`, `load_o59_policy` | `action_sequence_artifact.checkpoint_contract`, `read_checkpoint`, `export_action_sequence_policy`, `load_action_sequence_policy`. Validation and model construction are separate from execution. |
| Backend-specific controller conversions and simulation re-exports | `controller_to_action_vec`, `action_vec_to_controller`, `validate_controller_action`, and wire equality in `hal.controller`. Delete the duplicate `ControllerInputsValue` name. |
| Historical resume unpickler and compatibility switches | Delete. Add `ResumeLineage`, `checkpoint_resume_lineage`, and `validate_resume_provenance` for the declared source transition only. |
| 059 nested learning-rate callback | A module-scope function and a partial. `LearningRateScheduler` preserves format-4's `lr_lambdas=[None,...]` checkpoint record; this avoids a serialization change caused by replacing a function with a callable object. |

### Execution, scheduling, and service symbols

| Control owner and symbol | Maintained owner and disposition |
|---|---|
| `HistoryDecoderPolicy` | `ActionSequencePolicy`; one stream record owns observation history, cache row, generation, consumed frame, and sampling identity. |
| `GpuTokenHistory`, `GpuContextHistory` | `GpuObservationUpdates` retains cached update storage. Delete `GpuObservationWindow` and `GpuObservationWindowBatch` after the staging comparison below; the canonical dense executor keeps its existing bulk transfer path. |
| Batch-one `KVCache` | Retain bounded rings and attention math; add `KVCachePool` for prepared rows and batch scratch. Only real rows are scattered back. |
| 059 `BF16Inference` | `WindowPolicy`; `DenseWindowPredictionPolicy` supplies request/history ownership around the same dense executor. |
| `RecedingHorizon` and `_SlotState` | Delete after moving history and fault capture into inference and timing into `ActionScheduler`. `PolicyBatchAdapter` is the sole bridge into the process harness. |
| Direct `run_local_match`, thread driver in `sim.vec`, single-match harness selection | Delete. Keep `run_matches_vec`, the process driver, and shared `Slot`, `VecMatch`, `PolicyRuntimeSpec`, and `ObservationRow` in `sim.rollout`. |
| `PolicyRuntimeSpec.action_token_groups`, token fields/views in `ArenaSpec` and `RolloutArena` | Delete the unsupported token transport. The only maintained process driver always rejected it. Retain the float controller-action layout, including its byte sizes and shared-memory round trips. Failed arena construction now closes partial views/handles and unlinks only allocations it created. |
| `InferenceWorker` and request-created delivery threads | `InferenceEngine` plus a persistent `InferenceClient` delivery thread. Add prepared-profile admission/release records and bounded response failure handling. |
| Separate checkpoint loads for wrappers | `ModelRegistry` keys by checkpoint hash, resolved device, and inference dtype. Profile wrappers can share the same model object. |
| In-process service inference | A supervised GPU process, explicit ready/pulse/failure messages, and independent Dolphin slot processes. Readiness follows preparation; a failed generation is terminated before recovery. |
| Inference failure used for every plan outage | `NoUsableActionPlan` ends a match after two seconds of unusable plans, including before its first accepted plan. A successful stream release keeps the healthy engine available; a failed release still enters connection-failure handling. |
| `eval.self_play.synthetic_context`, `canonical_context` | `inference.warmup`; `DecodeTelemetry` moves to `inference.benchmark` with bounded samples, and reusable matchup construction moves to `eval.matchups`. |
| Historical H2H experiment loader | Delete. `H2HModel` names an artifact and per-player settings; identical checkpoints share model weights. |
| Old experiment-based fault replay | The current artifact reader plus version-2 fault records containing observation, prefix, and stream RNG inputs. |

## Invariants

The default model has 246,862,205 parameters with unchanged names, ordering,
initialization, RNG consumption and optimizer groups. Trunk: 12 layers, width
1024, 16 heads, context 256. Decoder: six layers, width 1024, feed-forward width
4096. Heads: `1..12,16,20,24,28`. Group decoding remains
`c_stick → main_stick → triggers → buttons`.

AWR retains beta 150, cap 10, gamma 0.99855, activation at update 4097,
next-frame return alignment, 32 policy prefixes and 128 critic/normalizer
positions. Return conditioning retains its population, scale, positive p90,
dropout and RNG. Calibration belongs to training/artifacts, not the neural model.

The buffered loader retains shard traversal, replay/window ordering, ordered
worker delivery, materialization and prefetch, bounded queues, cursor semantics,
`hal-o51-shards`, `o59-replay-ring-v2`, schema 3, and one generation per replay.
Defaults remain batch 512, 131,072 slots, eight windows/generation, phase 25,
minimum gap 200, seed 0 and 24 workers.

Validation retains the ordered 44 sources, Mosaic 0.13.0 compatibility setup,
`py1e`, block 8192, seed 0, zero workers, one stable window/replay, batch 128,
context 256 and target length 28. Preserve current corpus exceptions, selection,
manifest, schema, vocabulary and statistics identities. Do not rematerialize
published corpora to claim the same ordering.

## Inference and scheduling contract

One model allocation per checkpoint SHA-256, device and inference dtype. Bundle
metadata must not create another weight allocation. A stream has one authoritative
record; the cache pool owns storage and no second stream registry.

Allocate and prepare declared capacity before admission. Map arbitrary external
IDs to prepared rows. Prepare powers-of-two ready-batch buckets through capacity,
Q=1/2/4 updates, prefill, resets and empty fixed prefixes. Retain the direct
batch-one path. Batched requests gather ready cache rows, execute the trunk and
decoder along the batch dimension, and scatter real rows only. Never pad time
with fictitious observations. Ring capacity is context + maximum Q - 1 (259).

Sampling draws and counters remain independent of row order, sparse readiness,
other stream lifecycle, and dummy rows. BF16 sampled categories can differ near
a boundary; qualify conditional distributions and exact RNG separately.

Coalesce for at most 0.5 ms, limited by the nearest deadline. Use deadline order
and arrival order for ties. Keep one outstanding request per stream and bounded
queues. Use one Torch CPU thread and the same GC setup in production and tests.

| Profile | Mode | Physical delay | Fixed prefix | Inference allowance | Stride | Horizon | Reserve |
|---|---|---:|---:|---:|---:|---:|---:|
| Official 059 | Dense blocking | 0 | 2 | No game-frame wait | 2 | 4 | 0 |
| Local default | Cached blocking | 0 | 0 | No game-frame wait | 2 | 4 | 2 |
| Local stride-one | Cached blocking | 0 | 0 | No game-frame wait | 1 | 4 | 3 |
| Netplay delay 2 | Cached async | 2 | 3 | 1 frame | 4 | 8 | 1 |
| Netplay delay 3 | Cached async | 3 | 4 | 1 frame | 4 | 8 | 0 |

At choice frame f, a target t is reachable exactly when
`t >= f + physical_delay + 1`. Validate request identity, shape and target frames;
acknowledge consumed observations; compare the pinned/submitted prefix at exact
controller wire precision; accept the whole generated plan only if all targets
remain reachable. Do not salvage a suffix conditioned on missed actions.
A valid rejected response advances the consumed cursor once, independently of
the last accepted plan. A malformed active response is a protocol failure.

Preserve first-seen speculative netplay observations and record the mode.
Completed replay reconciliation is not live rollback correction.

## Recovery and provenance

A request has a one-second monotonic timeout. After useful actions are exhausted,
two seconds without a usable plan ends the match by controlled abort/forfeit.
Dolphin retains its ten-second stall detector. Termination completes within two
further seconds. A recoverable engine restart must prepare within 120 seconds
with cached artifacts; failed recovery leaves admission unavailable.

Current v5/format-4 checkpoints remain valid. Exact resume restores model,
optimizer, scheduler, committed loader position, Python/NumPy/Torch/CUDA RNG,
prefix/identity/return-mask RNG, calibration and scientific counters. A source
transition names parent checkpoint hash, old/new source SHA and parity report
identity; all other provenance comparisons remain strict. Preserve lineage in
subsequent checkpoints. No general provenance bypass is allowed.

On 2026-09-27 the user reduced checkpoint qualification to one representative
production checkpoint. Format/loading compatibility already passes with update
131072. Use the retained update-8192 checkpoint for the real next-batch and
next-update resume comparison, where AWR is active and the learning rate is
nonzero. Keep the existing synthetic coverage before and after AWR activation.
Do not continue searching for the original update-2048 and update-4096 fixtures.

The update-8192 fixture is downloaded and its configuration and conditioning
records validate. Its SHA-256 is
`eb968c0598d314d836c22bf16c976084e7d5bbc00ab28c19fd0e8b340ca98c1f`.
The fixture records source `f05d41502a429b399fe9b64d534c442d1a9a6379`, a B200,
Python 3.14.3, Torch 2.11.0+cu130, and the production batch-512/131,072-slot
loader. [The fixture manifest](../tests/fixtures/o59/artifacts.json) records
its R2 location and environment. The next-update comparison has not run.

`tests/fixtures/o59/capture_resume_update.py` invokes each checkout's existing
loader, prefetcher, training loss, optimizer, and scheduler for update 8193.
It records raw window identity, transformed and masked batches, prefixes,
clipped gradients, updated parameters, and saved optimizer/loader/RNG/calibration
state. `compare_resume_updates.py` verifies capture hashes and compares every
saved field exactly. Capture requires the checkpoint's environment and the
candidate's explicit source transition. It rejects a host memory limit below
100 GiB before constructing the production loader.

Existing version-1 bundles retain delay 2. A new capability version declares the
new local and netplay profiles. Export accepts supported 059 descendants based on
contracts and lineage, without a fixed W&B run identifier.

## Acceptance ledger

Every unchecked row remains required. Missing inputs are failures, not skips that
satisfy a gate. Raw large results belong in the existing run/artifact store.

| Gate | Required evidence | Current status |
|---|---|---|
| A Ownership | Final tree, one model/loader/scheduler/process driver, import boundaries, typed code, protected edits | Runtime retirement and import-boundary tests pass; final inventory/checks in progress. |
| B Data | `.slp` full path/parity, 44-source metadata and representative row audits, hashes, 2048 validation identities/tensors, sampler/resume geometry | Pass for the revised scope: cohort and statistics match exactly; synthetic loader resume and `.slp`→MDS→R2 publication pass; 33 v8 row audits reproduce the publication records. The user waived the remaining 11 row scans on 2026-09-27. |
| C Model/resume | Count/order/init/groups; one representative production next-batch/update comparison including non-unit AWR and all state | Proxy exact, synthetic boundary resume, and default count pass. Update 8192 is downloaded and its saved configuration validates. Capture/comparison tools are prepared; the real update-8193 comparison remains open. |
| D Artifacts | Old artifacts, descendants, new profiles, identity rejection | Actual update-131072 checkpoint validates without changing caller RNG, and re-exports as capability v2 with its checkpoint hash preserved. Artifact contract suite passes on CPU; qualified new profiles remain open. |
| E Cache | B1/2/4/8/16/32, Q1/2/4/decomposition, wraps, sparse/permutation/reset/identity/temp/prefix0/2/3/4, dummy rows, one weights copy | Focused CPU/CUDA independence tests and real-checkpoint B2 BF16 conditional KL pass. Complete capacity/profile matrix remains open. |
| F Scheduling | Exact deadlines, wire prefix, whole-plan rejection, consumed cursor, malformed replies, fallback counters | Focused scheduler/client/engine tests pass. Official intended-action chunk trace matches control. Live qualification remains open. |
| G Recovery | Exit/hang/brokenIPC/malformed/late/stalls/disconnect/rematch/partial-init; bounded detection, cleanup, restart | Focused lifecycle tests and real idle-process hang/exit recovery pass. The stopped GPU child terminates in 1.957 s, replacement readiness takes 31.081 s, and a second failure leaves the service unavailable. Live-match failure injection remains open. |
| H Mechanical speed | Three alternating trials; loader warm200/measure500, training warm100/measure200, matched local concurrency/stride; throughput≥95%, memory≤105% | Final direct cached 3060 pairs and the reduced loader-core comparison pass throughput/memory limits. Production loader, training, and dense-local measurements remain open. |
| I Batching | Ada B2 delivery≥5% faster than two serial B1 calls; 32-admitted/2-ready p95≤105% of2/2; capacity sweep | Real spawned-process 3060 B2 and sparse-load measurements pass these numerical limits. Required Ada measurement remains open. |
| J Real time | 3060 p95≤12ms,p99<16.67ms, matched p95≤105%; Ada≥2 sessions; three2400-frame trials minus300; both30min and10matches/rematches | Direct calls and short process benchmarks are evidence only. Complete-path trials, hardware capacity, and both soaks remain open. |
| K Gameplay | 96×7200-frame CPU protocol,p90,allboots complete,NSM regression≤.2,paired uncertainty; shared/separate weights H2H; separate new-profile results | Maintained commands/profiles are in place. Full matched gameplay runs and H2H qualification remain open. |
| L Repository | Ruff, ty, all maintained CPU tests, required emulator tests, GPU/service tests; frontend lock install/lint/type/build/queueAPI | Ruff, ty, all 1,219 current CPU tests, all seven required emulator cases, and frontend checks pass. Opt-in GPU checks pass at their recorded source revisions; complete service/hardware qualification remains open. |

Cache FP32 tolerances remain trunk/history `atol=2e-6,rtol=2e-5` and decoder
`atol=2e-5,rtol=2e-4`. CUDA BF16 conditional KL limits are mean ≤5e-4 nats and
p99 ≤5e-3, measured before/after eviction against streaming control, with finite
outputs and exact RNG sequences.

The non-fault soak requires at least 59.5 steady FPS, zero accepted late/mismatched
plans, zero skipped controller frames, zero compilation/capture after preparation,
zero inference-caused exhaustion after startup, and stable memory/process/thread
counts. Report startup, countdown, match-end and injected faults separately.

Required commands after integration:

```sh
uv run ruff format --check .
uv run ruff check .
uv run ty check --python-version 3.14 --error-on-warning hal experiments/059_muon_action_sequence.py scripts
uv run pytest -q -m "not integration"
HAL_REQUIRE_INTEGRATION=1 uv run pytest -q tests/test_roundtrip.py tests/test_session_cleanup.py -m integration
```

## Evidence gathered during implementation

- Model proxy report: `tests/fixtures/o59/model_proxy_parity.md`. Separate
  control/candidate processes matched parameter bytes, names/order, post-init RNG,
  dense hidden states, sampled actions and conditional logits. The proxy has
  17,059,325 parameters; it does not substitute for production resume.
- Real validation: `tests/fixtures/o59/validation_cohort.json`, all2048 identities,
  raw and transformed digests match; cohort SHA
  `ba0dc02f3db51685940a790baac795e48d3bf9c69173cf55c2d9928c92244f7d`.
- Existing final checkpoint SHA:
  `52b5233ed506f59f514f7e90a6a6111206152db7413f451dc35be5c30d1e671b`;
  update131072, return p90 `19.9760597229004`.
- First control3060 direct-call workload H8/prefix3/stride4/Q4,2400frames,
  first300excluded: p95 `8.449739497154951ms`, p99 `8.951088953763247ms`.
  Raw files: `runs/refactor-059/control-3060-trial-1/`. This excludes IPC delivery
  and is not the real-time service gate or the required alternating-trial report.
- Initial benchmark attempt failed with temporary-directory quota exhaustion.
  The rerun used `runs/refactor-059/tmp` and completed. No user files were deleted.
- R2 inventory found retained 059 milestones starting at 8192. The user reduced
  the production resume gate to one representative checkpoint; update 8192 is
  available. Loading more checkpoints is not required to establish the format.

### Numerical and process measurements

The [production BF16 report](../tests/fixtures/o59/production_bf16_kl.md)
compares the real checkpoint's batched cached executor with the immutable
control's independent batch-one streaming caches. It includes 960 forced
conditional distributions across two streams and positions before and after
context eviction. Mean KL is `0.00007230` nats and p99 is `0.00114066` nats;
both pass the stated limits. All outputs are finite. Capture programs, raw
artifact hashes, and tested source hashes are included in the report.

The [cache coverage table](../tests/fixtures/o59/acceptance_e_coverage.md)
separates CPU small-model tests, CUDA capture tests, production checkpoint
numerics, and the remaining hardware matrix. In particular, a CPU B32 test
does not qualify production B32 latency.

The [spawned-process batching report](../tests/fixtures/o59/process_batching_3060.md)
measures persistent clients, queueing, cache copies, GPU execution, validation,
and delivery with one shared checkpoint on the RTX 3060. Each condition warms
20 pairs and measures 200 pairs:

| Condition | Pair p50 | Pair p95 | Pair p99 | Peak allocated GPU memory |
|---|---:|---:|---:|---:|
| Two serial batch-one requests | 21.876 ms | 23.369 ms | 24.654 ms | 946.99 MiB |
| Two ready requests in one batch | 11.405 ms | 11.969 ms | 12.275 ms | 946.99 MiB |
| 32 admitted streams, two ready | 11.527 ms | 11.804 ms | 12.822 ms | 2393.72 MiB |

Two-request median delivery improved by 47.9%. Sparse p95 was 98.6% of the
two-admitted control. These runs exercise actual B2 computation, not two
checkpoint copies or a loop of GPU calls. They exclude Dolphin and the service
supervisor. They also precede later lifecycle/device-capture edits, identified
by source hashes in the report, so final-source timing and Ada qualification
remain required. Cold capacity-32 preparation took 253 seconds; that is not
evidence for the 120-second recovery bound or for 32 active sessions.

Three provisional direct-call control/candidate pairs had throughput ratios
`0.9959`, `0.9916`, and `0.9800`, with a median of `0.9916`. Peak GPU allocation
was about 536.34 MiB in both implementations. These exclude transport and have
source changes between trials. They do not close gate H or J. The initial
candidate's repeated BF16 weight conversion caused a measured latency regression;
moving the established one-time linear-weight conversion into artifact model
construction restored the direct-call result. The failed trial is retained.

The [final direct cached comparison](../tests/fixtures/o59/final_direct_cached_3060.md)
repeats three alternating pairs after controller decoding was included in
preparation. Paired throughput is 98.78%, 98.70%, and 99.31% of control; the
median is 98.78%. Peak GPU allocation is 536.323 MiB versus 536.335 MiB for
control. The first candidate calls take 9.43–9.50 ms. Profiling identified a
33 ms first-use cost in controller dequantization before that operation was
warmed. Relevant inference source hashes are identical across these trials.
This closes the measured direct cached throughput comparison only. It does not
replace complete service latency, local emulator throughput, or Ada tests.

### Validation commands and limitations

The following are completed checks at their recorded source revisions. Final
global results are recorded separately when the integration tree is frozen.
Do not merge different revisions into a single claim of full qualification.

| Check | Result and evidence |
|---|---|
| Locked environment install | `UV_CACHE_DIR=/tmp/hal-uv-cache uv sync --locked --extra netplay-server --group analysis` passed. Only HAL was rebuilt; 22 unused installed packages were removed. No retained version changed. `runs/refactor-059/environment-sync.log`. |
| Model extraction | Parameter names/order/bytes, initialization RNG, dense states, actions, and logits match on the 17M proxy; default meta parameter count is 246,862,205. [Report](../tests/fixtures/o59/model_proxy_parity.md). |
| Loader committed cursor | Control/candidate synthetic schema-3 fingerprint matches after 37 batches and 20 restored batches: `f0f80015a1e83a5abdfe5a2a8f67dfc116e198ec4b727802054efd457e30b236`. Durable helper: `tests/fixtures/o59/loader_resume_fingerprint.py`. |
| Validation cohort | All 2,048 replay/start/ego identities and tensor digests match. Durable capture/compare helper: `tests/fixtures/o59/validation_cohort_parity.py`. |
| Published statistics | All 44 sources and 1,584 sufficient-stat records validate. The 16-feature mixture digest remains `49b8299b93f6afdae28e876490065299959148e4389d12ac9190893ada39f775`. Forty zero-count sentinel records are preserved. |
| Local preprocessing | Representative `.slp` indexing, selection, full-MDS materialization, stored-schema validation, and local publication audit pass. [Detailed data checks](../tests/fixtures/o59/data_validation.json). |
| Artifact contracts | Latest focused run: 28 passed, one CUDA test skipped in the CPU invocation. Includes malformed tensors/buffers, incomplete statistics, old/new capability declarations, and resumed descendants. Actual final checkpoint validation also passes and preserves caller RNG. `runs/refactor-059/artifact-contract-tests-final.log` and `artifact-real-validation.log`. |
| Root training/tool regressions | 192 passed, four CUDA tests skipped in the CPU invocation; includes the format-4 learning-rate scheduler state and next-update test. `runs/refactor-059/root-callback-parity.log`. |
| Import boundaries | Five passed; separate interpreters confirm model imports do not load Melee and simulation imports do not load Torch. `runs/refactor-059/package-boundaries.log`. |
| Controller/protocol | 26 passed for malformed action handling and scheduling; 39 passed and five integration cases deselected for controller conversion/session callers. `controller-protocol-tests.log` and `controller-consolidation-tests-rerun.log` under `runs/refactor-059/`. |
| Required emulator tests | After simulation lifecycle edits, `HAL_REQUIRE_INTEGRATION=1 .venv/bin/pytest -q tests/test_roundtrip.py tests/test_session_cleanup.py -m integration` passed: seven passed, two non-integration cases deselected, six fork warnings. Run outside the socket/filesystem sandbox. |
| Frontend | Lockfile installation, lint, TypeScript, production build, and queue/API smoke passed. The smoke suite had 26 passed and one Starlette warning. Exact commands and the earlier sandbox bind failure are in `data_validation.json`. |
| Obsolete startup probe removal | 45 Modal/Vast launcher tests passed after removing the 022-era FlexAttention probe hook and its skip flag. Compiler-cache, free-space, CUDA, and process-stall checks remain. `runs/refactor-059/launcher-retirement-tests.log`. |
| First complete candidate CPU run | 1,132 passed, 22 skipped, 17 integration cases deselected, seven failed. One failure exposed a private cross-module loader import; six exposed test doubles not yet updated for explicit stream admission/profile validation. These were corrected and require a full rerun. `runs/refactor-059/repository-cpu-pass-1.log`. |
| Second complete candidate CPU run | 1,120 passed, 22 skipped, 17 deselected, 14 failed and ten setup errors. Direct Mosaic readers bypassed the required compatibility setup and leaked shared-memory descriptors until the process reached its file limit. This is a defect, not a passing gate; the boundary fix and full rerun are required. `runs/refactor-059/repository-cpu-pass-2.log`. |
| Interrupted CPU reruns | The sandboxed API rerun stalled; the escalated third run unintentionally exposed CUDA and was stopped to avoid competing with GPU qualification. The fourth run exhausted the temporary filesystem's per-user quota and aborted, with only 61 descriptors open. These are incomplete runs. Their logs are retained. Three inactive pytest directories were moved intact to `runs/refactor-059/retained-pytest-temp/`; subsequent runs use the NVMe-backed `TMPDIR`. |
| Mosaic descriptor regression | Explicit compatibility setup now runs at the remaining direct MDS reader boundaries. The focused 42-test sequence ended at 160 open descriptors, versus 808 before the fix; it did not raise the descriptor limit or force collection. A new subprocess regression covers repeated dataset creation. The complete CPU rerun remains required. |
| Standalone evaluation profiles | 15 passed after making the official-059 factory explicitly use intended action history and the checkpoint p90. Cached profiles retain observed action history. The first test attempt used an unavailable script-package import; the corrected test loads the CLI by file, like the other tool tests. `evaluation-profile-tests{,-rerun}.log` under `runs/refactor-059/`. |
| CUDA functional suite and device fix | 211 passed, one failed, one integration case deselected. The failure exposed a redundant global-device argument in the experiment evaluation factory. The executor now derives the device from its model; all eight affected factory/evaluation cases passed with CUDA visible. [Report](../tests/fixtures/o59/cuda_functional_3060.md). |
| RTX 3060 prepared-profile checks | All three opt-in hardware tests passed, including both delay profiles and graph execution across wrap/reset/settings changes. A separate raw capture retained 200 request/delivery samples per profile: delay 2 p95/p99 8.979/9.648 ms; delay 3 8.016/8.918 ms. Preparation took 85.338 seconds. The existing service remained resident and idle at sampled instants. This is not the live Dolphin or soak gate. [Report](../tests/fixtures/o59/netplay_ready_3060.md). |
| First live service smoke | Inference prepared both profiles, but both Dolphins failed GTK initialization without a display. No gameplay occurred. Cleanup took 1.226 seconds and left no descendants. Raw result: `runs/refactor-059/3060-functional-smoke-20260927-1/run-result.json`. The harness now validates DISPLAY and uses separate test ports. A live retry also requires resolving the existing service's ownership of test account A. |
| Isolated GPU-process recovery | `HAL_REQUIRE_IDLE_NETPLAY_FAULTS=1` with the capability-v2 bundle, account B, and a new isolated output directory: `pytest -q tests/test_idle_runner_faults_hardware.py -m integration` passed one hardware test, with two non-integration cases deselected, in 67.33 seconds. Initial readiness took 32.148 s. A stopped child terminated in 1.957 s and its replacement was ready in 31.081 s; killing that replacement left the runner unavailable in 1.157 s. No reservations or Dolphin sessions were created, no descendants survived, and source hashes matched. The existing service retained its PID and readiness. [Evidence](../tests/fixtures/o59/runtime_evidence.md). |
| Recovery fixture startup race | The first isolated attempt failed before preparation or signal injection because an absent startup health file is wrapped in `ValueError`. The fixture now waits only for that exact missing-file cause and rejects malformed content. Both the failed and successful attempt records remain under `runs/refactor-059/3060-idle-faults-20260927-{1,2}`. |
| Static checks during integration | Ruff lint and the complete maintained type target passed. Formatting still reported three agent-owned files being edited. Earlier attempts could not acquire a lock in the read-only default UV cache; reruns used `UV_CACHE_DIR=/tmp/hal-uv-cache`. Final frozen-tree checks remain required. `runs/refactor-059/{ruff,ty,format}-final-pass-{2,3}.log`. |
| Final integration checks | Global Ruff format/lint and the complete type target passed (`{format,ruff,ty}-final-pass-5.log`). The three formatted files retained identical syntax trees (`format-ast-parity.json`). Subsequent changed process/test files also passed focused static checks. |
| CPU suite after retirement | Pass 5 had 1,173 passed and one failure: a test double omitted the required stream-release method. Its correction also verifies that failed release preserves the original forfeit error. Pass 6 had all 1,174 passed. After the final process-constructor cleanup, pass 7 had **1,178 passed, 22 skipped, 18 deselected**, with nine warnings, in 93.63 seconds. Skips are 20 CUDA cases and two opt-in hardware cases; integration cases are selected separately. `runs/refactor-059/repository-cpu-pass-{5,6,7}.log`. |
| Final required emulator integration | `HAL_REQUIRE_INTEGRATION=1 uv run pytest -q -ra tests/test_roundtrip.py tests/test_session_cleanup.py -m integration` passed: **seven passed, two non-integration cases deselected**, six fork warnings, in 56.49 seconds. Fixtures were present. `runs/refactor-059/required-integration-final.log`. |
| Process and arena cleanup | The 42-test IPC/process sequence passed. Additional process construction/start failure cases, including interruption, passed; they are included in the final complete CPU suite. Current float arena sizes remain 156,352, 9,408, and 472,448 bytes for the three recorded control geometries. `transport-retirement-tests.log` and `worker-initialization-cleanup.log` under `runs/refactor-059/`. |
| Commit-series validation | Explicit Ruff format/lint and the full maintained Ty target pass. `uv run pytest -q -ra -m "not integration"` passed **1,180 tests**, with 22 CUDA/opt-in skips, 18 integration deselections, and nine warnings, in 93.75 s. The required replay/cleanup integration command passed **seven tests**, with two non-integration deselections and six fork warnings, in 55.95 s. Logs are under `runs/refactor-059/commit-series/`. The initial format check found one extra blank line in a test; the corrected check passed. |
| Cached local action timeline | The new scheduler matches the frozen delay-2 local transport at exact controller wire precision over startup, eight replans, and the final scheduled actions. The control record and capture-source digests are checked by `tests/test_cached_timeline_parity.py`; the capture helper is in the research archive. This establishes action timing, not matched emulator throughput. |
| Netplay soak assessment | The harness now rejects malformed or mismatched match evidence, slow steady delivery/gameplay, skipped submissions, and plan exhaustion. Its version-2 result distinguishes measured checks from unmeasured qualification gates. Resource sampling finishes before cleanup/reporting. Focused tests: **21 passed**. Full CPU suite: **1,198 passed, 22 skipped, 18 deselected**, nine warnings, 93.96 s. Global Ruff format/lint and the maintained Ty target pass. Initial lint found an omitted `zip(strict=...)`; Ty found two return annotations that needed explicit narrowing. Both were fixed before the passing runs. Logs: `runs/refactor-059/soak-assessment-{focused,cpu,format,lint,types,types-fixed}.log`. This tool change does not substitute for a live soak. |
| Loader measurement follow-up | Five cache-preparation tests pass. Global Ruff format/lint and the maintained Ty target pass. The complete CPU suite has **1,203 passed, 22 skipped, 18 deselected**, nine warnings, in 95.26 s; skip categories are unchanged. Logs: `runs/refactor-059/loader-cache-preparation-tests.log` and `runs/refactor-059/loader-followup-{cpu,format,lint,types}.log`. The selected R2 loader evidence archive passed download verification: 52 matching files, zero differences. Protected user-file hashes remain unchanged, and the accounting now contains 105 new maintained paths. |
| Unused dense GPU storage retirement | The two retired helpers had no maintained caller. Their two implementation-only tests are removed; cached update tests now compare directly with the canonical CPU window representation. The retained cached buffer definitions have identical syntax trees. Focused GPU tests: **22 passed** in 9.70 s. Full CPU suite: **1,201 passed, 22 skipped, 18 deselected**, nine warnings, in 95.17 s. The maintained Ty target passes. Logs: `runs/refactor-059/window-retirement-{gpu,cpu,types}.log`. |
| Production resume capture preparation | Ten new CPU cases verify nested state comparison, tensor shape/dtype rejection, capture hash checks, selected replay identity, and pre-update clipped gradients. Global Ruff format/lint and Ty pass. The full CPU suite has **1,211 passed, 22 skipped, 18 deselected**, nine warnings, in 93.95 s. Skips remain 20 CUDA and two opt-in hardware cases. Logs: `runs/refactor-059/resume-capture-{format,lint,types,cpu}.log`. Initial helper lint required an explicit `zip(strict=False)`; this was corrected before the passing checks. The downloaded checkpoint's first metadata inspection incorrectly treated derived `replay_slots` as a saved config field; parsing `TrainConfig` corrected it. Neither these tool tests nor configuration validation substitute for the real next-update run. |
| Modal source-archive review | H2H and inference capture tools called Git directly even though cloud source archives omit `.git`. They now use the launcher's validated `HAL_GIT_SHA`, retaining Git lookup for local checkouts. Eight new regression cases cover valid, invalid, absent, and local identities. Ruff, Ty, and the full CPU suite pass: **1,219 passed, 22 skipped, 18 deselected**, nine warnings, 93.97 s. Logs: `runs/refactor-059/modal-review-{focused,format,lint,types,cpu}.log`. |
| Dense B32 GPU preflight | The production checkpoint prepared the official dense B32/context-256/prefix-2/horizon-4 decoder in both checkouts. Each completed 40 finite-output calls, with the last 32 measured under `fail_on_recompile`. Peak allocated GPU memory was 3,184,207,872 bytes for control and 3,184,206,848 for candidate. Synthetic-call medians were 135.817/134.083 ms. This is one preparation diagnostic, excluding history, IPC, and Dolphin; it does not qualify throughput. Two earlier capture attempts failed on a removed neutral-action import and a mismatched `inference_mode` context; the successful helper uses the official `no_grad` context. All attempts remain in `runs/refactor-059/dense-32-preflight-*`. |
| Dense B32 emulator attempt and host memory failure | The 32-Dolphin diagnostic on local host `eric-gu-sff` exhausted host memory during startup and exited 137. Kernel logs confirm global OOM kills, including the pre-existing netplay runner PID 2982847 and desktop processes. The capture's resource sampler also failed on a child's inaccessible I/O counters, so no complete memory/throughput report exists. This attempt failed and does not qualify local evaluation. All diagnostic Dolphins exited and the reported shared memory/semaphore objects were removed. The local API on port 8080 stopped; the frontend on port 3000 remained. There were no active reservations. Evidence: `runs/refactor-059/dense-evaluation-diagnostics.json`, `dense-local-32-candidate-smoke-1.log`, and `dense-local-32-memory-failure.log`. |

The remote preprocessing fixture contains six replays from `dev.7z`, two per
split, with no extraction failures. Its nine staged data objects total 842,904
bytes; publication adds `_SUCCESS`. Local, staging, and final object checksums
and lengths match. Both new prefixes are under
`r2:hal/runs/refactor-059/publication-fixture-ad9b1822beb2`; existing published
corpora were not changed. The [fixture identity record](../tests/fixtures/o59/publication_fixture.json)
and its capture helper make this check reproducible. Packed-v8 publication is
covered by the representative row audits below.

The separate read-only v8 object/marker inventory validates all 44 published
prefixes and 1,295,370 train rows. The compressed shards total 172,490,094,626
bytes (160.65 GiB). `runs/refactor-059/v8-published-inventory.json` records the
objects and marker identities, including the original marker form retained by
ranked-1 and Druggedfox. Row audits are recorded separately below.

The first full-row pilot, RapM, passed in 17.47 seconds. It read 15 objects
and 79,506,116 bytes and reproduced the publication marker's row counts,
retained/rejected counts, object lengths, and train frames. Its marker SHA-256
is `36030e304dae352ca1a0636364d8dacead455c749dabdf9d33ebc56c6402d57a`.
The complete report is `runs/refactor-059/v8-full-rapm.json`. This verifies one
of the retained draft exceptions.

The larger Aklo pilot also passed, in 285.88 seconds. It checked 103 objects,
2,168,847,919 compressed bytes, and all 19,177 retained rows, reproducing the
published 169,867,381 train frames. Its report SHA-256 is
`1d22bab6a01b5c4e35ac195fbea4bdb40a3e606479e0057977917b940691b92b`.
`runs/refactor-059/v8-full-audit/summary-attempt-1.json` records both pilots.
On 2026-09-27 the user accepted representative coverage and stopped the full
scan after 33 sources passed. The remaining 11 row scans are no longer required.
The driver stopped with SIGINT (exit 130); no audit processes remained, and its
scratch directory was empty. The interrupted source is not counted as passed.
Individual reports and attempt logs remain under
`runs/refactor-059/v8-full-audit/`. The summary is `summary-user-stop.json`,
SHA-256 `9ba71666335148b12480b47fea18b877de96dcd410ea91816eb06b3bf541fa2e`.
The reports and interrupted-attempt log are preserved at
`r2://hal/runs/refactor-059/evidence/v8-audit-71e8e2339ae24f3aedc5ce5d9c0e22ff9272c2fa7d3d9bb2c416a9e13527d552/`.
Download verification found 72 matching files and zero differences; its log is
`runs/refactor-059/v8-audit-evidence-verification.log`.

The capability-v2 qualification bundle is
`runs/refactor-059/o59-capability-v2.hal`, SHA-256
`0ff1daf80caa36a94a713c4ccba9223db8d7ba7c1379b5865bbc40b8a8c2f3ec`.
It preserves checkpoint SHA-256
`52b5233ed506f59f514f7e90a6a6111206152db7413f451dc35be5c30d1e671b`,
vocabulary SHA-256
`c67c97c995ad033ea7f5b2223efce5b061394566439f091ff6e7aaa6a9d1cfd6`,
and p90 `19.9760597229004`. It declares delays 0, 2, and 3 without rewriting
the existing capability-v1 bundle. The export has not been deployed.

The original full CPU baseline needed an explicit control `PYTHONPATH` and the
shared cached data path. An earlier run imported candidate subprocess modules
and is invalid as a baseline. The corrected run had 1,300 passed, four failures,
ten skips, and 23 deselections. Three failures came from an attempted separate
environment install exhausting temporary space; their seven-test archive subset
passed after using the pinned existing environment. The remaining unchanged
launcher test hit its seven-second timeout; the candidate deployment suite
subsequently passed. The logs retain these failures instead of replacing them.

Other recorded failures include sandbox denial of ENet host creation, restricted
editable-build downloads, frontend localhost binding, one incorrect test path,
and temporary-directory quota exhaustion. Each was rerun with the required
environment or corrected command. The new dense CUDA preparation test initially
used a synthetic head width unsupported by Inductor; production 059 uses a
supported width. Its corrected test and the final checks must be recorded before
closing the corresponding gates.

The default loader-core buffer needs 67.68 GiB for replay slots alone. Its
conservative host estimate is 71.68 GiB, versus about 22 GiB free on this host.
The production 131,072-slot performance gate therefore needs a larger host;
a smaller local geometry must be labeled separately. The selected production
resume comparison still needs the matching training GPU/environment. The RTX
6000 Ada hardware gates are also open. No provenance bypass, lower-concurrency
substitute, or passing CPU suite closes those gates. The memory preflight is
recorded in `runs/refactor-059/loader-preflight-default-v3.json`.

An initial reduced loader comparison completed three alternating pairs with
batch 64, 1,600 replay slots, and two workers. Its median paired throughput
ratio was 97.40%, and its maximum paired peak-RSS ratio was 103.53%.
The first control still materialized raw shards during measurement, so this
series does not close the matched-cache gate. The last two pairs were 96.43%
and 97.40%. The raw
record is `runs/refactor-059/loader-small-pairs-1/summary.json`; it is not a
production-geometry or transformed-training measurement.

The completed-cache second series had a 109.87% median throughput ratio, but
physical disk reads were unequal; that apparent speedup is not attributed to
the refactor. The third series applies the same file-scoped page-cache advice
before every control and candidate warmup. All six trials then read exactly
5,243,879,424 physical bytes and wrote none. Median throughput is **100.66%**
of control, and the highest paired peak-RSS ratio is **97.12%**. This meets the
limits for the reduced loader-core workload only. The
[full report](../tests/fixtures/o59/loader_performance.md) includes every trial,
startup, CPU, I/O, memory, source identity, and the selected R2 evidence archive.

The canonical dense executor retains `ObservationHistory`, bulk CPU window
collation, and transfer. `GpuObservationWindow` and `GpuObservationWindowBatch`
had no maintained runtime caller and are removed. Cached inference still uses
its prepared GPU update buffers; their retained definitions are unchanged.

This differs from the proposed dense GPU-window integration. A staging-only
comparison at source `90046957145317f4659c863dfb4da984669f5f06` used the full
126-feature schema, context 256, two-frame updates, current 059 statistics and
codebooks, and an RTX 3060. Three alternating pairs each warmed 144 updates and
measured 128. Median staging times were:

| Ready streams | Existing bulk CPU path | Mirrored GPU windows |
|---|---:|---:|
| 1 | 0.628 ms | 1.426 ms |
| 32 | 2.939 ms | 42.912 ms |

Feature and controller-token digests matched exactly across all trials. The
mirrored path issued many small per-stream copies and quantization operations;
it did not include the CPU fault-snapshot work an integration would also need.
Retaining the working bulk path avoids that added cost and ownership. This
diagnostic uses synthetic observations, excludes neural computation and IPC,
and ran while a read-only corpus audit and an idle resident service existed.
It does not close the full dense-local throughput gate H.

The six-file evidence archive contains the capture script, all raw timings,
statistics, syntax-tree comparison, and a hash manifest:
`r2://hal/runs/refactor-059/evidence/window-storage-1afaf83f849a4870d5e030b334dc15d39ba3ce7108943b4ac6afeaf5baabc7c7/`.
Manifest SHA-256 is
`1afaf83f849a4870d5e030b334dc15d39ba3ce7108943b4ac6afeaf5baabc7c7`;
download verification found six matching files and zero differences. The raw
measurement SHA-256 is
`ec6b34b5700c1c035eb5a56e99a72329a334efaadcc3a2e94c207ee2fa7f4c9b`.

GPU capacity does not establish host capacity for 32 Dolphin workers. The later
emulator diagnostic caused global host memory exhaustion, including loss of the
existing local netplay backend. A subsequent interrupt found the diagnostic
already killed; this was not a controlled shutdown. No 32-worker retry is
permitted on this host. The user limits local Dolphin concurrency to six
physical CPU cores; subsequent local diagnostics use four workers and remain
separate from the required matched 32-worker qualification on a larger host.
The diagnostic now rejects more than six workers and refuses to launch without
a hard memory limit. Both rejection paths were checked before model loading or
process creation. Further emulator measurements also need a measured
worker-memory budget.
The local systemd user bus currently refuses connections, so the attempted
memory-limited test scope did not start. There is no local Docker test image.

The dense preparation diagnostics, failed emulator capture, original capture
source, and kernel failure log are preserved at
`r2://hal/runs/refactor-059/evidence/dense-diagnostic-c421fd2c7f15295e177a0c22351aac88658ddc3b96c29c78dbbc55687547633a/`.
The manifest SHA-256 is
`c421fd2c7f15295e177a0c22351aac88658ddc3b96c29c78dbbc55687547633a`.
Download verification found 12 matching files and zero differences. The original
capture source is retained separately from the later worker-limit guards.

Recovery of the existing local backend is prepared separately from the refactor
and awaits user approval. The archived source under
`runs/netplay/inference-audit/source` imports with the current environment and
has tree identity
`96c802323243b2245c4bfe8165693286778b791c4d50da17e6527d1b4a1d5653`.
The proposed launcher retains the existing 059 bundle, queue, account A, port
51441, compiled window mode, and recorded sampling seed. The previous health
record's Git SHA was stale, so this snapshot is not asserted to reconstruct the
previous in-memory code exactly. Its configuration and launcher syntax were
validated without starting the backend. The local preparation record is
`runs/refactor-059/service-recovery/proposal.json`.

## Repository trees

These trees compare the pinned control with the implemented source layout.
All maintained runtime modules are expanded below. Unchanged assets and test
families are collapsed. This is a source inventory, not a qualification claim.
The exact per-file archive/deletion/addition ledger is
[`refactor-059-files.json`](refactor-059-files.json).

### Before

```text
/home/ericgu/src/hal/
├── hal/
│   ├── controller.py, wire.py, policy.py, streams.py
│   ├── data/                 # extraction, stored formats, mixed publication ownership
│   ├── training/
│   │   ├── canonical.py, features.py, context_history.py
│   │   ├── controller_codec.py, trunk.py, scoring.py
│   │   ├── player_identity.py, ego_stats.py, rank_metadata.py
│   │   ├── dataloader.py, replay_reservoir.py, physical_shard_loader.py
│   │   ├── closed_loop.py
│   │   └── current optimization, checkpoint, return, and metrics utilities
│   ├── inference/
│   │   ├── api.py, loader.py, transport.py, worker.py
│   │   ├── bundle.py, checkpoints.py, cuda_graph.py, warmup.py, benchmark.py
│   │   └── backends/
│   │       ├── history_decoder/     # model.py, policy.py, kv_cache.py, gpu_history.py
│   │       └── temporal_awr/        # model.py, policy.py
│   ├── eval/
│   │   ├── local.py, policy.py, harness.py, scheduling.py, netplay.py
│   │   ├── self_play.py, policy_sampling.py, slippilab.py
│   │   └── current qualification, matchup, result, and replay utilities
│   ├── sim/
│   │   ├── vec.py, process_vec.py
│   │   └── current Dolphin, transport, replay, and roundtrip utilities
│   ├── netplay_service/       # API, queue, runner, health, and replay publication
│   └── scripts/              # commands mixed with implementation and migrations
├── experiments/
│   ├── 55 numbered programs before 059
│   ├── 059_muon_history_decoder.py
│   ├── benchmark_closed_loop.py, benchmark_kv_cache.py
│   ├── benchmark_kv_netplay.py, eval_kv_cache.py
│   └── o51/, o52/, o53/, o55/, o56/, o58/, o59/
├── scripts/                  # launchers, current tools, old sweeps, setup.sh
├── tests/
│   ├── experiments/          # 48 files, including 059
│   ├── fixtures/             # modal helper and o51 characterization
│   └── current and historical subsystem tests
├── notebooks/                # 22 tracked programs and protected untracked notebook
├── docs/                     # current runtime, historical research/blog/anatomy
├── docker/, deploy/netplay/, vendor/, web/netplay/
└── outputs/                  # protected user files
```

### After

```text
/home/ericgu/src/hal/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── uv.lock
├── .github/
│   └── workflows/
│       └── ci.yml
├── .pre-commit-config.yaml
├── .dockerignore
├── hal/
│   ├── __init__.py
│   ├── controller.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── archive.py
│   │   ├── behavior.py
│   │   ├── bounded_writer.py
│   │   ├── conversions.py
│   │   ├── extract.py
│   │   ├── feature_stats.py
│   │   ├── index.py
│   │   ├── index_builder.py
│   │   ├── mds.py
│   │   ├── mds_materialization.py
│   │   ├── mds_publication.py
│   │   ├── player_identity.py
│   │   ├── policy_schema.py
│   │   ├── policy_world_schema.py
│   │   ├── policy_world_v8.py
│   │   ├── professional_replays.py
│   │   ├── replay_selection.py
│   │   ├── replay_stats.py
│   │   ├── schema.py
│   │   ├── slippi.py
│   │   ├── slp_finalize.py
│   │   └── streaming_compat.py
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── action_trace.py
│   │   ├── behavior.py
│   │   ├── cross_stage.py
│   │   ├── h2h.py
│   │   ├── harness.py
│   │   ├── match_summary.py
│   │   ├── matchups.py
│   │   ├── netplay.py
│   │   ├── observations.py
│   │   ├── paired.py
│   │   ├── policy.py
│   │   ├── qualification.py
│   │   ├── replays.py
│   │   ├── results.py
│   │   └── scheduling.py
│   ├── fixtures.py
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── action_sequence_artifact.py
│   │   ├── action_sequence_policy.py
│   │   ├── api.py
│   │   ├── benchmark.py
│   │   ├── bundle.py
│   │   ├── checkpoints.py
│   │   ├── client.py
│   │   ├── cuda_graph.py
│   │   ├── engine.py
│   │   ├── gpu_observations.py
│   │   ├── kv_cache.py
│   │   ├── observation_history.py
│   │   ├── sampling.py
│   │   ├── warmup.py
│   │   └── window_policy.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── action_sequence.py
│   │   ├── attention.py
│   │   ├── controller_codec.py
│   │   └── sampling.py
│   ├── netplay_service/
│   │   ├── __init__.py
│   │   ├── admin.py
│   │   ├── api.py
│   │   ├── domain.py
│   │   ├── health.py
│   │   ├── queue.py
│   │   ├── replays.py
│   │   └── runner.py
│   ├── paths.py
│   ├── policy.py
│   ├── r2.py
│   ├── representation/
│   │   ├── __init__.py
│   │   ├── features.py
│   │   ├── observations.py
│   │   └── player_identity.py
│   ├── scripts/
│   │   ├── __init__.py
│   │   ├── analyze_replays.py
│   │   ├── build_index.py
│   │   ├── build_player_identity_sidecar.py
│   │   ├── fetch.py
│   │   ├── filter.py
│   │   ├── h2h.py
│   │   ├── materialize.py
│   │   ├── policy.py
│   │   ├── prepare_professional.py
│   │   ├── publish_mds.py
│   │   ├── roundtrip.py
│   │   ├── slippi_jwt.py
│   │   └── slp_link.py
│   ├── sim/
│   │   ├── __init__.py
│   │   ├── diff.py
│   │   ├── inputs.py
│   │   ├── ipc.py
│   │   ├── loop.py
│   │   ├── netplay.py
│   │   ├── pdeathsig_exec.py
│   │   ├── process_vec.py
│   │   ├── rollout.py
│   │   ├── session.py
│   │   ├── sources.py
│   │   ├── trajectory.py
│   │   └── worker.py
│   ├── streams.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── batches.py
│   │   ├── buffered_mds_replay_loader.py
│   │   ├── checkpoints.py
│   │   ├── mfu.py
│   │   ├── muon.py
│   │   ├── physical_shard_loader.py
│   │   ├── replay_windows.py
│   │   ├── returns.py
│   │   ├── runs.py
│   │   ├── system_metrics.py
│   │   └── validation_replay_loader.py
│   └── wire.py
├── scripts/
│   ├── benchmark_closed_loop.py
│   ├── benchmark_kv_cache.py
│   ├── benchmark_kv_netplay.py
│   ├── benchmark_replay_loader.py
│   ├── cache_modal_fixtures.py
│   ├── eval_kv_cache.py
│   ├── launch_gce.py
│   ├── launch_modal.py
│   ├── launch_vast.py
│   ├── qualify_netplay_059.py
│   ├── rematerialize_policy_world_v8_modal.py
│   └── replay_policy_fault.py
├── experiments/
│   ├── 059_muon_action_sequence.py
│   └── o59/
│       └── [preserved launch and measurement records]
├── notebooks/
│   ├── 039_scaling_viz.py
│   ├── alignment_probe.py
│   ├── discretization_roundtrip.py
│   ├── eval_forensics.py
│   └── main_stick_usage.ipynb
├── docs/
│   ├── inference.md
│   ├── kv-cache.md
│   ├── netplay-realtime.md
│   ├── refactor-059-files.json
│   ├── refactor-059.md
│   └── runtime-validation.md
├── tests/
│   ├── experiments/
│   │   └── test_059_muon_action_sequence.py
│   ├── fixtures/
│   │   ├── modal_command.py
│   │   └── o59/
│   │       ├── acceptance_e_coverage.md
│   │       ├── artifacts.json
│   │       ├── audit_published_v8.py
│   │       ├── cached_batch_proxy_parity.md
│   │       ├── cached_timeline_control.json
│   │       ├── capture_model_proxy.py
│   │       ├── capture_netplay_ready.py
│   │       ├── capture_resume_update.py
│   │       ├── checkpoint_config.json
│   │       ├── compare_resume_updates.py
│   │       ├── conditioning_protocol.json
│   │       ├── cuda_functional_3060.md
│   │       ├── data_validation.json
│   │       ├── fd_trace.py
│   │       ├── final_direct_cached_3060.md
│   │       ├── idle_runner_faults.py
│   │       ├── loader_resume_fingerprint.py
│   │       ├── make_loader_control_benchmark.py
│   │       ├── model_proxy_parity.json
│   │       ├── model_proxy_parity.md
│   │       ├── netplay_ready_3060.md
│   │       ├── prepare_publication_fixture.py
│   │       ├── process_batching_3060.md
│   │       ├── production_bf16_kl.md
│   │       ├── production_kl_capture.py
│   │       ├── production_kl_compare.py
│   │       ├── publication_fixture.json
│   │       ├── record_publication_fixture.py
│   │       ├── run_loader_pairs.py
│   │       ├── run_published_v8_audits.py
│   │       ├── runtime_evidence.md
│   │       ├── stats_digest.py
│   │       ├── validation_cohort.json
│   │       └── validation_cohort_parity.py
│   └── [retained and new subsystem tests]
├── archive/
│   ├── README.md
│   ├── experiments/
│   │   └── [55 programs and six metadata directories]
│   ├── notebooks/
│   │   └── [18 programs]
│   ├── scripts/
│   │   └── [historical sweeps, audits, and scale-up campaigns]
│   └── docs/
│       └── [historical research and assets]
├── docker/
│   └── [maintained images]
├── deploy/
│   └── netplay/
│       └── [service configuration, including compose-ada.yaml]
├── vendor/
│   └── [pinned packages]
├── web/
│   └── netplay/
│       └── [application and lockfile]
└── outputs/
    └── [protected user files]
```

There is no archived copy of `hal/` or `tests/`. The two pickle records at
`hal/training/physical_shard_loader.py` are the only retained historical module
boundary. Research programs are preserved as evidence, not supported runtimes.
