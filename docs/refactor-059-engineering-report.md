# HAL 059 refactor: engineering report and merge review

Date: 2026-09-27.

This report covers the committed implementation from d7454f9d to 4895c9d7 on
codex/o59-refactor. That range contains 50 commits. Documentation added during
this review is separate from that runtime snapshot. The working tree also
contains newer frontend and reservation changes; those are identified below
and are not counted as committed refactor work.

## 1. Executive assessment

HAL now has one maintained experiment, one model implementation shared by
training and inference, one buffered training loader, one action scheduler,
and one process-based local evaluation path. Multiple game sessions can use
one loaded checkpoint and execute compatible inference requests in a real
GPU batch.

The work preserved the existing 059 design rather than replacing its core
algorithms. The model, controller vocabulary, AWR objective, replay-buffer
sampling, bounded KV rings, and attention computation remain the basis of the
system. The substantial new runtime behavior is batching across independent
streams, explicit preparation and admission, stricter action-plan validation,
and supervised inference failure handling.

The evidence supports functional training, data loading, artifact loading,
cached inference, and local evaluation. Three matched production training
trials show essentially unchanged throughput. Actual multi-stream inference
is materially faster than serial requests through one shared model. Data
identity and validation-cohort checks provide stronger evidence than a
successful import or a small synthetic training example.

The evidence does not establish every original acceptance target. Exact
bitwise equality of the next production optimizer update was not obtained.
The full clean paired gameplay campaign and the long human-netplay runs did
not finish. The latest local human connection attempts exposed a firewall
block and a reservation-cancellation problem, so this report does not certify
a finished public netplay deployment.

These are different conclusions:

- The refactor is implemented and has substantial correctness coverage.
- The current working tree is not yet a single clean merge candidate.
- The original qualification goal is not complete.
- Merging reviewed code is a separate decision from deploying the service.

The merge recommendation and concrete review findings are recorded at the end.
The detailed historical acceptance ledger remains in
[refactor-059.md](refactor-059.md). This report is the current managerial summary;
the ledger retains failed attempts and measurement details.

## 2. Why the refactor was needed

The previous repository mixed the current training program with infrastructure
for many earlier experiments. Several components performed similar work under
different owners.

Training and serving each contained a model implementation. Several loader
families owned overlapping batching, sampling, and resume logic. Training
evaluation used RecedingHorizon, while deployed policies used ActionScheduler.
Local evaluation had both thread and process drivers plus a separate direct
match loop. The inference worker collected requests, but the policy still ran
them one at a time. The KV implementation required batch size one.

Those overlaps created practical risks. A model change could reach training
without reaching serving. Evaluation and deployment could use different action
timing. A serving benchmark could measure several competing model copies instead
of the capacity of a shared model. A historical compatibility path could keep
obsolete modules inside the maintained package.

The new structure gives each of these decisions one owner. It also separates
scientific choices from runtime mechanics. Experiment 059 still chooses the
objective, architecture values, conditioning, and training schedule. The library
owns reusable computation, data handling, inference execution, and simulation.

## 3. Scope and repository size

The counts below come from Git objects at the two named revisions. Line counts
include comments and blank lines. They describe code ownership and maintenance
surface, not a direct measure of complexity.

| Area | Before | After |
|---|---:|---:|
| Python files under hal/ | 128 | 119 |
| Lines under hal/ | 40,156 | 35,026 |
| Active numbered experiment programs | 56 | 1 |
| Lines in active numbered programs | 135,777 | 4,290 |
| Python files under tests/, including fixture helpers | 145 | 119 |
| Lines in those test and fixture files | 50,687 | 26,914 |
| Tracked Python notebook programs | 22 | 4 |
| Python programs under scripts/ | 11 | 12 |
| Files under archive/ | 0 | 147 |

The maintained library is 5,130 lines smaller, a reduction of about 12.8%.
The full branch changes 501 paths, with 46,684 inserted lines and 54,590 deleted
lines. Those totals include renames, archive moves, tests, documentation, and
new qualification tools. They should not be read as 46,684 lines of new runtime.

Fifty-five earlier numbered experiment programs were archived, including
050–058. Eighteen historical notebook programs were archived. Obsolete runtime
implementations and tests were deleted rather than copied into archive/hal or
archive/tests.

The archive preserves research source and evidence. It is not a supported
runtime. Historical reproduction still requires the corresponding Git checkout,
environment, data, and checkpoint artifacts.

The existing edits to .gitignore and notebooks/039_scaling_viz.py were preserved.
The untracked main_stick_usage.ipynb and outputs/ were preserved. The later
frontend and reservation edits were also preserved.

## 4. The new ownership model

| Package | Responsibility |
|---|---|
| hal/models | Neural computation, controller codec, and primitive categorical sampling |
| hal/representation | Observation conversion and feature definitions shared by training and inference |
| hal/data | Replay extraction, selection, MDS construction, publication, statistics, and identity sidecars |
| hal/training | Buffered loading, replay windows, validation loading, optimizers, and checkpoint utilities |
| experiments/059_muon_action_sequence.py | The scientific treatment and training loop |
| hal/inference | Artifact loading, observation history, policy execution, caches, sampling identity, engine, and clients |
| hal/eval | Timing, action scheduling, evaluation protocols, adapters, and results |
| hal/sim | Dolphin processes and frame/controller transport |
| hal/netplay_service | Reservations, process supervision, readiness, health, and replay publication |
| CLI modules | Argument parsing and calls to public library operations |

The main path is now:

    Slippi replay
      -> extraction, indexing, selection, and MDS materialization
      -> versioned corpus, statistics, and player vocabulary
      -> BufferedMDSReplayLoader
      -> experiment 059 batch treatment and training
      -> ActionSequenceTransformer checkpoint
      -> validated policy artifact
      -> InferenceEngine
      -> ActionScheduler
      -> Dolphin controller input

Dolphin observations travel back through representation conversion to inference.
Independent Dolphin workers share the engine. Each controlled player has its own
stream history and scheduler.

Import-boundary tests verify that loading the model does not load Melee and that
loading simulation does not load Torch. This matters because a Dolphin worker
should not acquire model-runtime dependencies or accidentally allocate GPU state.

## 5. Canonical model and scientific contract

HistoryDecoder was renamed to ActionSequenceTransformer. The runtime policy is
ActionSequencePolicy. There is one maintained neural implementation in
hal/models/action_sequence.py.

The serving copy of the model and the duplicate experiment definitions were
removed. Training and inference construct the same model through a narrow
ActionSequenceConfig. Serving no longer imports an experiment or creates a large
training configuration with optimizer, upload, and corpus settings.

The following 059 choices were preserved:

- The 12-layer, width-1024 trunk and six-layer action decoder.
- Context length 256, 16 attention heads, and the existing decoder width.
- The default total of 246,862,205 parameters.
- Parameter names and order, optimizer grouping, and initialization behavior.
- Prediction offsets 1–12, 16, 20, 24, and 28.
- Controller decoding in C-stick, main-stick, trigger, then button order.
- Codebooks, masks, and trigger/button constraints.
- Muon and AdamW behavior, learning-rate schedule, clipping, and reductions.
- AWR beta 150, cap 10, gamma 0.99855, and the existing activation boundary.
- Next-frame return alignment, prefix sampling, identity conditioning, and
  return calibration.

The dense and cached execution strategies remain separate where they need to
be. They call the same model layers and controller primitives. Cached execution
is not claimed to be identical to recomputing a cropped dense window after
history eviction; those operations have different hidden-state histories.

Model extraction was checked on a smaller 17M-parameter model with exact
parameter, initialization, RNG, state, action, and logit comparisons. The
production parameter count was checked separately. Real-checkpoint cached
numerical comparisons and actual training runs add production-scale evidence.
The smaller model test is useful, but is not presented as the only model check.

## 6. Replay processing and loading

PhysicalShardReplayLoader became BufferedMDSReplayLoader. This is a clearer name
for the mechanism that was retained: workers decode MDS replay data into a large
memory buffer, and training reuses windows from that buffer to avoid reading
from disk for every batch.

The refactor kept shard traversal, replay/window selection, worker result order,
prefetch behavior, buffer geometry, and the committed resume cursor. It retained
the current hash salt, the o59-replay-ring-v2 protocol, and schema-3 loader state.
Historical multi-generation modes were removed.

There is one training loader implementation. The old
hal/training/physical_shard_loader.py contains only PhysicalRow and
RingSlotDescriptor. Current checkpoints pickle those two record types by module
path. Keeping those definitions avoids a checkpoint migration; it does not retain
a second loader.

Reusable replay-window construction and collation moved to replay_windows.py.
The supported validation path moved to validation_replay_loader.py. Generic
loader branches and the reservoir family were retired.

Data construction also gained a proper library owner. Indexing, selection,
materialization, publication, professional replay preparation, and v8 corpus
operations moved out of CLI implementations into hal/data. Commands now dispatch
to those functions. The current v8 publication exceptions and manifest recipes
remain because they identify artifacts still used by 059.

The published corpus was not rewritten. A rematerialized corpus was not treated
as equivalent to the original row order.

Validation results include:

- Exact replay/start/ego identity and tensor-digest agreement for all 2,048
  validation windows.
- Matching consolidated statistics across all 44 sources and 1,584 records.
- Synthetic committed-cursor and restored-batch parity.
- A representative .slp path through extraction, selection, materialization,
  schema validation, and publication.
- Completed row audits for 33 published corpora.

The user explicitly waived the remaining 11 row scans. They are outside the
revised scope, rather than failed checks waiting for another large download.

## 7. Checkpoints, artifact loading, and resume

Current 059 checkpoints and bundle payloads remain supported. Historical
experiment compatibility was removed. Artifact validation now has a separate
owner from policy execution.

The artifact reader validates the architecture contract, tensor payload,
statistics, vocabulary, calibration, and checkpoint identity. It accepts valid
resumed 059 descendants without requiring one hardcoded W&B run identifier.

Existing bundle capability declarations retain their meaning. The new capability
version declares the additional local and netplay timing profiles. A historical
backend identifier remains in the persisted format; changing the Python class
name did not force an unnecessary artifact-format migration.

The source transition for resume is explicit. ResumeLineage records the parent
checkpoint hash, old source, new source, and parity-report identity. It does not
provide a general provenance bypass. Other resume checks remain strict.

The latest checkpoint used for service work was update 131072:

- Checkpoint SHA-256:
  52b5233ed506f59f514f7e90a6a6111206152db7413f451dc35be5c30d1e671b.
- Capability-v2 policy SHA-256:
  0ff1daf80caa36a94a713c4ccba9223db8d7ba7c1379b5865bbc40b8a8c2f3ec.
- Recorded p90 return target: 19.9760597229004.

### What the production resume tests established

At the user's direction, testing used one representative production checkpoint,
update 8192, and its next update 8193. We did not continue loading every historical
checkpoint or require the originally proposed two activation-boundary checkpoints.

The production tests restored the loader and training state, produced the next
batch, applied the AWR objective with non-unit weights, and completed the next
optimizer update. Data identity, masks, prefixes, RNG, calibration, and scheduling
records matched in the recorded comparisons.

They did not pass a strict bitwise comparison of every gradient, parameter, and
optimizer value. The first comparison had very small embedding-related
differences. Repeating the unchanged control also produced embedding differences.
A later comparison had a one-FP32-step loss difference and broader optimizer
differences; the largest updated-parameter difference was 0.00120654.

A separate saved-batch diagnostic then completed eight stages: three compiled
control/candidate pairs and one eager pair, with repeated calls. Forward outputs,
loss, recorded button outputs, and RNG matched between sources. Gradient variation
also occurred when the same source was called again. That diagnostic found no
source-specific forward discrepancy. It did not redo the full optimizer update
or prove strict next-update equality.

The practical conclusion is that production resume and continued training work,
and the input/state restoration checks are strong. The remaining bitwise result
is a documented limit. It is not evidence that resumed training fails to run,
and it is not a completed proof of numerical identity. The user directed us to
stop this numerical investigation and prioritize functional use.

## 8. Shared inference engine and actual batching

The previous worker could collect multiple requests but still executed them
sequentially. The refactor adds genuine batch-dimension execution.

ModelRegistry uses checkpoint hash, resolved device, and inference dtype as its
identity. Two players using the same checkpoint share one model allocation.
Different checkpoints each have their own allocation. Different bundle metadata
does not, by itself, justify another copy of the weights.

A stream owns its history, generation, request sequence, consumed observation
cursor, sampling identity, and cache-row assignment. The cache pool owns the
tensor storage. It does not maintain a competing copy of stream lifecycle state.

Before admission, the engine allocates capacity and prepares the supported
profiles and execution shapes. External stream IDs map to prepared rows; they
do not need to be small consecutive integers. Reset and release reuse prepared
storage. This closes the old warmup mismatch between placeholder stream IDs and
actual match/port IDs.

The retained KV design uses bounded rings. With context 256 and maximum update
length four, capacity is 259. Rows have independent valid lengths and absolute
positions. Observation counts are decomposed into prepared 1-, 2-, and 4-frame
updates without inserting fictitious observations.

For compatible ready requests, the engine gathers the relevant rows, runs the
trunk and decoder as a batch, and scatters real rows back. Prepared batch buckets
cover the ready batch. Dummy rows do not advance real streams, consume their
sampling draws, or produce delivered actions. Batch one retains a direct path.

Requests are grouped by compatible checkpoint, mode, horizon, prefix, and update
shape. Coalescing is bounded at 0.5 ms and limited by the nearest deadline.
Idle admitted streams do not force the engine to execute a larger ready batch.

This work did not introduce vLLM, PagedAttention, FlashInfer, or a new attention
kernel family. It extended the existing working cache and computation.

### Numerical and independence checks

Tests cover independent stream positions, resets, row reuse, sparse readiness,
row permutations, different conditioning settings, empty and nonempty prefixes,
and repeated ring wraps. RNG draws and counters are checked independently of
batch composition.

A production-checkpoint BF16 comparison on the RTX 3060 measured 960 conditional
categorical distributions before and after eviction. Mean KL was 0.00007230 nats;
p99 KL was 0.00114066. Both were below the recorded qualification limits. A later
Blackwell comparison also passed.

These results support equivalent conditional behavior within the declared
numerical tolerance. They do not promise identical stochastic actions at every
category boundary.

## 9. Evaluation and action timing

ActionScheduler is now the one owner of fixed actions, replanning, pending
prediction identity, deadlines, reserve actions, and fallback counters.

RecedingHorizon was removed. Observation history belongs to inference. Training
evaluation, local play, and H2H use PolicyBatchAdapter with the process vector
harness. The direct local loop, thread vector driver, and attribute-based choice
between drivers were removed.

This preserves the existing bulk local execution path. Workers still receive
complete action chunks and retain the supported execution stride. The refactor
did not replace it with per-frame IPC.

Physical controller delay and a fixed action prefix are now separate values.
This distinction prevents a policy's committed future actions from being confused
with transport delay.

| Profile | Execution | Physical delay | Fixed prefix | Replan interval | Horizon |
|---|---|---:|---:|---:|---:|
| Official 059 evaluation | Dense, blocking | 0 | 2 | 2 | 4 |
| Local default | Cached, blocking | 0 | 0 | 2 | 4 |
| Local stride-one measurement | Cached, blocking | 0 | 0 | 1 | 4 |
| Netplay delay 2 | Cached, asynchronous | 2 | 3 | 4 | 8 |
| Netplay delay 3 | Cached, asynchronous | 3 | 4 | 4 | 8 |

The official profile remains the scientific control. The zero-prefix local and
eight-head netplay profiles have separate names and evidence. Their results must
not be substituted for the official profile's historical measurements.

For local zero-delay play, an observation at frame t can produce input for t+1;
Dolphin waits for inference. For delay-2 netplay, t+1 through t+3 are fixed, and
the generated tail targets t+4 through t+8.

A response is accepted only when its identity, shape, frames, and fixed prefix
are valid and every generated action is still reachable. Prefix comparison uses
canonical controller-wire precision. A late plan is rejected as a whole rather
than salvaging a suffix conditioned on missed generated actions.

The consumed observation cursor is separate from the last accepted plan.
Rejecting a valid late response still acknowledges its consumed observations,
which prevents the next request from ingesting them twice. Malformed active
responses terminate through a defined failure path.

Netplay still uses the first observation seen for a speculative frame. The
refactor does not implement rollback-aware cache replay or confirmed-frame-only
inference. That limitation is recorded in match provenance.

## 10. Service lifecycle and future distributed RL

The service keeps its existing HTTP API, SQLite reservation store, Dolphin
workers, Slippi controller transport, and replay publication.

A supervised inference process owns the GPU model and cache storage. Dolphin
workers use persistent clients. The client no longer creates a delivery thread
for each request.

Requests carry stream, generation, sequence, policy identity, source frame, and
timing information. Responses echo request identity and explicit target frames.
These records do not require a Dolphin object or a GPU tensor handle. A future
remote transport can implement the client boundary without moving model ownership
into every actor.

The current work does not add a distributed learner, remote RPC system, replay
service, online weight publication, or rollout database. It establishes interfaces
that those systems can use later.

Failure handling now distinguishes several conditions:

- A request with no valid response has a monotonic timeout.
- Repeated unusable plans cannot keep the match in neutral forever.
- A stalled Dolphin frame stream has a separate timeout.
- A failed inference generation is invalidated and terminated before recovery.
- Readiness follows model/profile preparation.
- Failed recovery leaves the service unavailable rather than admitting more work.

Cold startup and replacement-engine recovery have different budgets. Large cold
compilation can take longer than the recovery target; increasing cold preparation
time did not remove the recovery bound.

An actual idle GPU-process fault test terminated a stopped child in 1.957 seconds,
prepared its replacement in 31.081 seconds, and left the service unavailable after
a second failure. This is real process evidence, but it is not a failure-injection
test during a human match.

## 11. Measured performance

All numbers below retain their workload and scope. A direct model call, a complete
request/response pair, and a running Dolphin match are different measurements.

| Measurement | Result | Interpretation |
|---|---|---|
| Production training, three matched pairs | Median candidate/control throughput 99.963%; largest paired sampled-memory ratio 100.062% | Training speed was preserved in the measured workload |
| Production buffered loader, three pairs | Median throughput ratio 112.32%; third pair was 92.30% | Median improved, but individual trials vary |
| RTX 3060 direct cached path, three pairs | Median throughput ratio 98.78%; GPU allocation effectively unchanged | Within the mechanical performance target |
| RTX 3060 two ready streams through spawned engine/client | Median pair time 21.876 ms serial versus 11.405 ms batched | 47.9% less time to deliver both responses |
| RTX 3060 32 admitted, two ready | p95 11.804 ms versus 11.969 ms for two admitted | Idle streams did not impose a larger active batch |
| RTX PRO 6000 Blackwell batching, three pairs | Pair-time improvements 49.2%, 43.1%, and 42.7% | Actual batched execution benefits repeated on the cloud host |
| Blackwell sparse ready batches | p95 ratios 1.022, 1.043, and 1.038 | All remained within the 1.05 sparse-load target |

Training used batch 512 and the production replay-buffer geometry. Each pair
included 100 warmup updates and 200 measured updates. Loader pairs used 200 warmup
batches and 500 measured batches.

Some cloud peak-memory counters were unavailable. The reports now distinguish a
sampled high-water value from a kernel-reported peak instead of substituting a
misleading zero. The sampled-memory comparisons passed; they are not a claim that
every transient allocation was observed.

The batch results include request transport and cache gather/scatter. They are
stronger evidence than timing a raw forward call. They still exclude the complete
human network connection and long-running Dolphin behavior.

## 12. What the gameplay and netplay tests showed

Model-versus-model evaluation completed eight matches with shared checkpoint
weights and eight with distinct checkpoint weights, without failures in those runs.

The first full dense control and candidate each produced 96 accepted boots at
7,200 frames per boot. Each also had two emulator crashes followed by retries.
The candidate's recorded net stocks per minute were 1.394 versus 1.192 for the
control, and aggregate emulator FPS was about 329 versus 190.

Those figures are retained, but they are not a clean paired gameplay-strength or
throughput result. Retries changed later execution groups and their sampling.
The refactor now preserves failed-attempt replays and reports those attempts
instead of allowing the accepted results to conceal them.

A second control completed all 96 boots without a crash, timeout, or retry.
Its matching candidate was not completed before the user stopped the cloud jobs.
The full clean paired campaign therefore remains unfinished.

Local transport controls found a separate graphics problem on the RTX 3060 host.
A short neutral-input Vulkan run achieved about 28.23 FPS and had 39 gaps longer
than 100 ms. The corresponding OpenGL run achieved about 59.94 FPS with no such
gaps. Enabling Vulkan VSync did not remove the pauses.

The service now accepts an explicit OpenGL backend. The default remains explicit
rather than silently changing based on host behavior. The short OpenGL result
shows a useful local configuration, not a completed model-driven service soak.

During human connection attempts, HAL selected Fox, entered the correct code,
and locked in. The host firewall then blocked incoming UDP traffic to the exact
peer port held by Dolphin. That is a connection-environment failure before policy
execution. The user added a narrowly scoped firewall rule.

Subsequent inspection found that a canceled reservation could leave the worker
waiting for the connection while another job remained queued. A newer, uncommitted
change adds cancellation propagation into the menu wait. That change is separate
from the committed refactor reviewed here.

No successful full human match on the final service configuration was established
in this session. The later logging request was interrupted when the user redirected
the work to merge review. Its partial source edit was removed; a patch copy was
retained with the local review records.

## 13. Other defects found and fixed during the work

The work included several corrections that were necessary to make the retained
paths work under their new owners.

| Defect or weak boundary | Correction |
|---|---|
| Direct netplay observations used different categorical types from stored rows | Project live observations through the stored-column representation |
| Direct Mosaic readers bypassed compatibility setup and leaked descriptors | Call the pinned compatibility setup at each retained reader boundary and add a regression |
| Empty action prefixes failed during stacking | Use an explicit empty action tensor |
| Warmup used placeholder stream IDs | Admit real external IDs into prepared storage |
| Controller decoding first ran after preparation | Warm the decode path before readiness |
| Per-request delivery threads increased lifecycle overhead | Use persistent client delivery machinery |
| Failed local evaluation attempts lost replay evidence | Preserve failed replays separately from accepted attempts |
| A request timestamp overwrote cumulative gameplay timing | Separate request and gameplay clock variables |
| GPU PID namespaces invalidated memory/process assumptions | Identify owned processes correctly and report unavailable counters explicitly |
| Cold preparation was confused with recovery | Give initial preparation its own budget |
| Netplay container lacked graphics support | Add the required graphics libraries and validate the display setup |
| Old development MDS fixture bypassed current schema identity | Replace the fixture and reject stale schemas before replay |
| Qualification could report success without checking the complete timing record | Validate action plans, compilation/capture counts, and complete run records |

These fixes were accompanied by focused tests and recorded in separate commits.
The branch also adds reusable qualification tools rather than keeping all test
logic in one-off terminal commands.

## 14. Retirement, dependencies, and operational documentation

Retired implementations include the reservoir loader, generic historical loader
branches, old closed-loop scheduler, temporal-AWR inference backend, duplicate
serving model, direct local loop, thread vector driver, historical experiment
loader in H2H, and obsolete model-specific commands.

Current data operations remain supported. Old migration and campaign source is
archived where it still has research value. Historical runtime tests are removed;
current contracts were moved to the packages that now own them.

Unused direct Python dependencies were removed. Analysis dependencies were
separated from the training/runtime set, and fsspec was declared directly.
The lockfile was regenerated without an unrelated core dependency upgrade.
Mosaic, Peppi, Slippi/libmelee, and the relevant replay tooling retained their
pinned versions during parity work.

Unused frontend dependencies were removed, and the shadcn CLI moved to development
dependencies. The committed frontend change is dependency cleanup. The much larger
page and styling redesign now visible in the working tree is separate work.

CI, pre-commit configuration, launchers, deployment inputs, and project instructions
were retargeted to 059 and maintained tools. Research archives and generated output
are excluded from maintained-code checks and code-upload inputs.

Runtime documentation was consolidated in inference.md. Operational netplay
instructions, the refactor ledger, file/symbol accounting, source identities,
and test fixtures now point at the maintained layout.

## 15. Repository layout for a new maintainer

The tree below groups tests and unchanged support modules. The complete expanded
tree and symbol accounting are in refactor-059.md.

    hal/
      models/
        action_sequence.py       # canonical model and narrow construction config
        attention.py
        controller_codec.py
        sampling.py
      representation/
        observations.py
        features.py
        player_identity.py
      data/
        extract.py, index_builder.py, replay_selection.py
        mds_materialization.py, mds_publication.py, policy_world_v8.py
        feature_stats.py, player_identity.py, schemas and format support
      training/
        buffered_mds_replay_loader.py
        replay_windows.py, validation_replay_loader.py, batches.py
        checkpoints.py, muon.py, returns.py, metrics
        physical_shard_loader.py  # two persisted record definitions only
      inference/
        action_sequence_artifact.py, action_sequence_policy.py
        engine.py, client.py, api.py
        kv_cache.py, gpu_observations.py, observation_history.py
        window_policy.py, sampling.py, warmup.py, benchmark.py
      eval/
        scheduling.py, policy.py, harness.py, netplay.py, h2h.py
        match protocols, qualification, results, replay utilities
      sim/
        process_vec.py, rollout.py, worker.py, ipc.py
        session.py, netplay.py, inputs.py, replay diagnostics
      netplay_service/
        api.py, queue.py, runner.py, health.py, replays.py
      scripts/
        current data, policy, H2H, authentication, and replay commands
    experiments/
      059_muon_action_sequence.py
      o59/
    scripts/
      benchmarks, cloud launchers, fault replay, qualify_netplay_059.py
    tests/
      current contract tests
      experiments/test_059_muon_action_sequence.py
      fixtures/o59/
    notebooks/
      four maintained programs plus preserved user notebook
    docs/
      inference.md, refactor-059.md, operational and measurement documents
    archive/
      earlier experiments, notebooks, campaign scripts, and research documents
    deploy/netplay/
    web/netplay/
    outputs/                     # preserved user output

Two deliberate changes from the proposed tree are worth knowing. data/slippi.py
holds small identifier bridges without creating a representation import cycle.
The proposed dense GPU observation-window copy was removed after a staging
comparison showed that the existing bulk dense path was preferable.

## 16. Validation record and its limits

At runtime commit 4895c9d7, the recorded CPU suite passed 1,335 tests, with
24 CUDA/opt-in skips and 18 integration deselections. Ruff formatting, Ruff lint,
and the maintained Ty target passed.

The seven required replay/session integration cases passed at the preceding
recorded runtime revision. GPU tests, production-checkpoint numerical checks,
prepared-profile checks, H2H, and process-recovery checks passed at their recorded
revisions. These results are not collapsed into a claim that every check ran on
one identical final tree.

Frontend lockfile installation, lint, type checking, production build, and queue/API
smoke tests passed for the committed dependency cleanup. That does not qualify
the separate uncommitted frontend redesign.

Fresh merge-review checks and code-review findings are appended below. Their
working-tree scope is stated explicitly.

Evidence is organized as follows:

- docs/refactor-059.md: historical accounting, acceptance ledger, and expanded trees.
- tests/fixtures/o59/artifacts.json: control and artifact identities.
- tests/fixtures/o59/validation_cohort.json: ordered validation cohort.
- tests/fixtures/o59/modal_qualification.json: cloud stages, outcomes, hashes,
  cancellations, and scope changes.
- Other reports under tests/fixtures/o59/: model, cache, batching, loader, and
  runtime checks.
- Content-addressed R2 archives: large captures, source snapshots, raw timing,
  logs, and replay evidence.
- runs/refactor-059/merge-review/: this review's local check logs and inventory.

Missing work remains visible. We did not relabel interrupted jobs, failed
comparisons, or skipped hardware tests as passes.

## 17. Scope changes and execution lessons

The user explicitly narrowed several parts of the original qualification plan:

- One representative production resume checkpoint was sufficient.
- The remaining corpus scans were waived after representative coverage.
- The available Blackwell GPU was accepted instead of insisting on Ada.
- Functional training/evaluation and human play took priority over exact
  historical numerical reproduction.
- All Modal jobs were stopped, and no more were authorized.

The earlier 32-Dolphin test on the six-CPU local host was a mistake. It exhausted
host memory and the kernel killed the existing backend. No performance result
from that attempt was accepted. Later local tests used explicit six-CPU,
12-GiB limits and at most two Dolphins.

Several qualification jobs also needed avoidable corrections: cloud disk minimums,
stale fixtures, display setup, preparation budgets, and process-namespace handling.
The resulting tooling is more explicit, but those retries consumed time.

The work also spent too long pursuing exact numerical differences after the
practical training and evaluation questions had strong evidence. The final
saved-batch diagnostic did not find a source-specific forward difference.
The remaining numerical limit should be retained in the record without turning
it into an open-ended prerequisite for every deployment decision.

The useful process change is to separate three questions early: whether the code
implements the intended contract, whether measured performance is preserved, and
whether the complete service works on the target host. Each needs a focused
test. Passing one does not answer the others.

## 18. Merge review and remaining actions

### Fresh checks

The preserved working tree passed the following checks during this review:

| Check | Result |
|---|---|
| Ruff format over the repository | Passed; 256 files already formatted |
| Ruff lint over the repository | Passed |
| Ty over hal/, maintained 059, and scripts/ | Passed with no diagnostics |
| CPU suite with CUDA disabled | 1,338 passed, 24 skipped, 18 deselected, 10 warnings; 102.58 seconds |
| Patch whitespace check | Passed |

The skipped cases require CUDA or an opt-in hardware setting. The deselected
cases are integration tests. They are not counted as passes. No new GPU,
Dolphin, or Modal run was launched for this review. The running local service
was not stopped or restarted by this cleanup.

The first inventory whitespace check flagged empty trailing TSV cells. They
were replaced with explicit absent-path markers, and the final check passed.
The previously modified source files did not change during these checks. The
check logs are under runs/refactor-059/merge-review/. These checks include the
pending cancellation code, but they do not turn that code into a committed or
fully reviewed frontend release.

### Findings in committed runtime

**High priority: a single client failure can stop all sessions.**

In runner.py, _handle_reservation rethrows InferenceUnavailable after forfeiting
the affected reservation. A slot-process exit then becomes a generic RuntimeError
in the supervisor. The outer recovery loop catches _EngineLost, not that generic
slot failure. The generation cleanup stops all workers.

A broken connection for one slot can therefore end healthy sessions and leave
the service unavailable even when the GPU process is still healthy. Failing
closed prevents silent bad play, but the fault scope is wider than the failed
client. This is a service-availability defect, not a model-computation defect.

Relevant current working-tree locations are runner.py lines 1215–1220,
1492–1494, and 1560–1564. The behavior is also present in committed 4895c9d7;
the pending cancellation edits shift the line numbers.

Before merge, define and implement recovery for a failed slot/client connection.
A focused regression should use two slots, keep the engine healthy, break one
client connection, and verify the promised behavior for the surviving match and
replacement admission. Existing tests assert that the affected slot raises;
they do not exercise this multi-slot outcome.

**Medium priority: cached observation types are checked only at stream start.**

ActionSequencePolicy.predict validates canonical scalar types only for the first
observation of a new generation. Other observations pass the general numeric
validator, but not the categorical integer check. ObservationHistory stores
categorical data in integer arrays.

A categorical field such as stage=0.5 can therefore be accepted and converted
instead of rejected when it occurs later in a request or stream. Current live
observation conversion supplies canonical values, which limits normal-path
exposure. The public inference boundary still fails its stated invalid-input
contract.

The relevant location is action_sequence_policy.py lines 577–580. Validate each
new observation before mutating history/cache state. Regression coverage should
place an invalid categorical value in the second observation of a request and
in a later request, and verify that rejection does not advance the stream.

A small CPU reproduction using the existing test model confirmed this finding:
the first malformed observation is rejected, but the same value in the second
observation or a later request is accepted. The result is recorded in
runs/refactor-059/merge-review/observation-validation-repro.json.

### Finding in the uncommitted cancellation change

The pending ConnectAbandoned callback is checked at the top of the menu loop.
It is not checked while _read_state waits for a frame or while the countdown
advances toward frame zero. Cancellation can therefore remain delayed until a
frame arrives, countdown ends, or the frame-stall timeout expires.

The new tests cover an event already set before the first read, plus reservation
handling. They do not cover canceling during a stalled read or countdown.
Those cases should be added before treating immediate requeue as fixed.

This finding belongs to the pending reservation change. It is not part of the
50-commit runtime snapshot.

### Provenance interpretation

The resume-lineage validator checks the parent checkpoint identity, the old and
new source identities, and the format of the report digest. It does not open
the referenced report or verify that the report says the transition passed.

The lineage is therefore an explicit operator declaration, not an automated
proof of parity approval. That distinction must remain clear in training
operations and documentation. This is not a demonstrated tensor/data regression,
and the user's decision to stop pursuing bitwise identity should not be replaced
with another hidden numerical gate. If automated release approval is desired
later, it needs its own explicit contract.

### Cleanup completed

- Removed only the interrupted agent-owned Logger.ini edit. The patch is retained
  in local review records; it is not presented as completed connection logging.
- Removed the generated TypeScript build-cache file; an active frontend task
  regenerated it later. It remains outside the documentation commit.
- Reconciled stale documentation that still described completed or stopped Modal
  jobs as running.
- Recorded the completed saved-batch result, the stopped dense comparison, the
  accepted Blackwell scope, and the fresh repository checks.
- Preserved unrelated source changes, notebooks, and outputs.
- Added this engineering report and an exact committed-path inventory in
  [refactor-059-file-accounting.tsv](refactor-059-file-accounting.tsv). A dash
  means the path does not exist on that side of the comparison.

### What remains outside the committed refactor

The current worktree contains a frontend redesign in page.tsx, globals.css,
layout.tsx, and netplay-api.ts. It also contains API/domain/queue changes that
extend connection/rematch timing, plus cancellation propagation and tests.
Those changes were preserved and must be committed or separated deliberately.

The existing .gitignore and notebook edits remain unrelated user work. They
should not enter a refactor commit by accident.

The first migration commits were recorded as a review series after testing the
combined implementation. Each intermediate commit was not independently
qualified as a standalone checkout. Review the series in dependency order; do
not assume every intermediate point is a green build.

### Recommendation

Do not merge the entire current working tree as one change.

Keep the core refactor, evidence, and documentation together. Address the two
committed runtime findings with small fixes and focused tests. Review and commit
the frontend/reservation changes separately, including the cancellation edge
cases. Then run the repository checks on that selected merge tree.

I would not require another exhaustive corpus scan, every historical checkpoint,
a specific Ada card, or another open-ended numerical investigation to make that
merge decision. The user has already narrowed those requirements.

I would keep production rollout approval separate. A real human match, disconnect,
requeue, rematch, and failure recovery on the intended host remain the most direct
tests of the service that people will use. The original long-run and clean paired
gameplay targets remain unfinished evidence, not completed passes.

The code review was targeted at the highest-risk training, loader, artifact,
inference, scheduling, and service boundaries. It was not an exhaustive proof of
all changed lines. No new model-training or scheduler correctness defect was
identified in that review; the concrete findings above remain open.
