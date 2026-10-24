# Four-group GPU memory: 2026-10-24

## Summary

Four groups at 1024 active matches completed six updates at 26.980 s per timed update. The
retained treatment shares serial graph workspaces, bounds preparation KL temporaries, and
releases unused reference backward activations. It removes the preparation OOM, but reserved
memory reaches 163.801 GiB: the requested headroom is **not qualified**. The lower 110–120 GiB
target is also unmet. Production capacity fields remain unset.

A boundary-cache-release follow-up completed three updates, then failed reference-support
validation on group 3. It was reverted. Its cause is unresolved. The single B200 sandbox was
flushed and shut down; no further cloud machine was started.

Control: measured commit `21c8e8a2` (code retained through `778a4f3e`), with separate
tick/restart/value graph pools and whole-rollout KL.
Invariant: O64 step-0098304, four groups, 1024 active matches, 512 frames, all decisions,
512 windows per optimizer step, eight microbatches, 64-window graph chunks, seed 65,
production sampler, rewards, representation and optimizer. Profiling uses critic warmup zero.

## Memory accounting before edits

The failed control is `/home/ericgu/data/scratch/o65-devbox/runs/bench-20261024-012332-m64-full`.
It captured 48 graphs. Its first preparation failed at 172.91 GiB allocated, with 91.50 GiB
allocated in graph private pools, while requesting another 1.41 GiB. Startup actor peaked at
90.86 GiB allocated and 176.50 GiB reserved. These are different phases.

The control log contains no allocator snapshot or individual pool sizes. Those sizes cannot be
recovered from aggregate peaks. The new inventory records every graph, its input shape key,
private pool identity, pool allocated/reserved bytes, and unique resident tensor storage.
A shared pool is counted once, regardless of how many graph rows name it.

| Captured graph family in control | Pool ownership | Individual pool bytes |
| --- | --- | --- |
| Student tick: both buffers, append 1/2, R2/R4, decoder row buckets | One pool per captured shape | Not recorded |
| Student block restart: both buffers | One pool per buffer | Not recorded |
| Stacked anchor tick and restart | One pool per captured shape | Not recorded |
| Bootstrap value: 2048 and 32 rows, one decision | One pool per shape | Not recorded |
| Reference: 64 windows, 16-decision decoder chunks; (delay,replan) = (16,2/4), (17,2/4), (18,2/4), (19,2/4), (20,2/4), (21,2) | Already shares actor/reference pool | Not recorded |
| Actor: 64 windows, 64 or 32 decisions, queue padded to 21 | Already shares actor/reference pool | Not recorded |

Local sizes below use the checkpoint architecture and production tensor shapes on the meta
device. They are exact tensor sizes, not GPU allocator measurements. Evaluation rows include
one spare backing row, which adds less than 20 KiB to the rounded values.

| Large resident tensor | Shape / storage | GiB |
| --- | --- | ---: |
| Full frame store, records | 646 x 1024 x 440 bytes | 0.271 |
| Full frame store, actions/rewards/flags/lengths | Shared active workspace | 0.075 |
| Rollout columns, including actions/logs/values | 536576 rows; includes two 24-frame queues | 0.978 |
| Behavior distributions | 536577 x 4 x (9+65+25+256), FP32 | 2.838 |
| Behavior critic trunk and decoder features | Two 536577 x 1024 FP32 tensors | 4.094 |
| Other critic features | Conditions and opponent queues/lengths | 0.404 |
| Inactive frame stores | Three 129-frame histories | 0.207 |
| Trunk and history KV | Layer and history K/V, per policy/layout | See measured table below |
| Model master weights | 252958559 parameters, FP32, per model | 0.942 |
| Tick BF16 weights, stacked anchor weights, reference metadata copy | Separate persistent model copies | See measured table below |
| Frozen group queues and episode state | Four group rings and metadata | See measured table below |

Preparation holds the full behavior table; gathered behavior distributions and critic features;
full replay windows (including both queues); full reference window outputs and their gathered
copy; decision metadata; bootstrap values; KL and advantage outputs. Previously, KL additionally
held active probability copies, neutral distributions, masked p/q, logarithms, support masks,
and categorical-KL validation/reduction temporaries for one entire factor. The 256-way button
factor is the largest. Actor fitting later adds dense reference distributions and critic clones.

The grad-enabled reference workaround originally kept the trunk autograd graph through all
decoder chunks. Its `output` local also kept the previous decoder graph alive while the next
decoder forward ran. Returned probabilities and features are already detached: the excess
is temporary activation retention, including during capture, rather than an autograd graph
stored in `PreparedUpdate`. The treatment retains grad-enabled forwards and a grad-enabled
leaf history input. It detaches the history only between the existing compiled boundaries,
deletes the observation graph, and deletes each decoder result after retaining detached outputs.
The regression compares against the original non-leaf history input and retained decoder
graphs, and checks that those objects die before the next decoder forward.

The measured runtime table reconciles the independent budget supplied by the orchestrator.

| Item | Budget GiB | Measured GiB | Explanation |
| --- | ---: | ---: | --- |
| Student weights, gradients, optimizers, BF16 weights | 4 | 3.33 | Includes critic Adam state and gradients; subtotal before these is 3.264 |
| Reference, forward copy, three snapshots, stacked anchors | 10 | 14.123 | Five stacked rows retain FP32 and BF16 copies; see reconciliation below |
| Student trunk and history KV | 26 | 26.313 | Trunk 24.281 plus decoder history 2.031 |
| Anchor caches | 3 | 4.111 | Five-model stack; includes decoder history K/V |
| Active rollout tables and frames | 10 | 8.660 | Table 8.314 plus active frame store 0.346 |
| Three frozen groups and rings/state | <1 | 0.229 | Frozen frames 0.207; ring/state backing about 0.022 |
| Resident total | 53 | 58.375 | First-fit allocation minus the prepared batch; includes graph static tensors and residual |
| Largest actor graph workspace envelope | 42 | <=34.033 | Shared student/training pool; includes about 0.496 GiB graph outputs |
| Peak allocated | 95 | 96.013 (first profile) | Graph replay scratch is reserved and is not counted as live allocation |
| Peak reserved | 110–120 | 163.801 | Above the requested ceiling of about 160 GiB; not qualified |

The reference/anchor weight budget differs by 4.12 GiB. The five-row stack holds two
reference cohorts (the reference lane count is twice each snapshot count) and three snapshots,
with 4.712 GiB FP32 weights and 2.349 GiB BF16 buffers. The separate canonical reference, grad-enabled forward
copy and three snapshots contribute another 7.062 GiB, including the 0.470 GiB forward-copy
buffers omitted by the raw inventory. This accounts for the measured 14.123 GiB.
Student trunk KV is 24.281 GiB; decoder history raises the combined count to 26.313 GiB.
The 42 GiB actor workspace was a napkin estimate, not a measured control. The shared pool's
34.033 GiB reservation bounds the retained graph workspace after activation release and reuse;
those contributions were not isolated, so the difference is not an attributed 8 GiB saving.
At first fit, unique counted storage is 69.268 GiB, with 11.865 GiB in the prepared batch;
allocation is 70.240 GiB. After adding the missing 0.470 GiB forward-copy buffers, the residual
is 0.502 GiB (allocator rounding, small staging/index/rotary tensors and critic gradients).
The 58.375 GiB resident allocation excludes the prepared batch but includes optimizers and
1.128 GiB of unique graph input/output storage. The main differences from the 53 GiB budget
are the additional reference/stack weights and anchor caches.

Immediately before KL, windows occupy 1.963 GiB, gathered old evaluation 5.129 GiB,
gathered reference evaluation 4.379 GiB and decision metadata 0.361 GiB. These coexist with
the resident tables. The reference evaluator's full window output temporarily coexists with
its gathered copy. The reference/preparation allocated peak is 76.567 GiB. At fitting, the
prepared batch retains 11.865 GiB; dense reference distributions and diagnostic/feature clones
add further storage. The eager critic is now the allocated-memory peak. For the first update,
a dense button-factor KL temporary shrinks from 1.413 GiB (370304 x 4 x 256 FP32) to 16 MiB
(4096 x 4 x 256), with the same row/category arithmetic.

Allocator semantics matter: graph working storage remains reserved while many capture-time
allocations appear freed in `memory_allocated()`. In the critic phase, the 35.766 GiB graph
pool reservation is mostly idle and cannot serve ordinary eager allocations. Thus a roughly
96 GiB eager allocated peak can need about 131 GiB before general allocator slack. This
explains why the 110–120 GiB reserved target is harder than the 95 GiB allocated estimate;
the before-boundary-release run then accumulated enough cached eager blocks to reach 163.801 GiB.

## Captured graph inventory after edits

The startup snapshot contains all 48 graphs below. Pool S is `(2, 0)` and pool A is `(1, 0)`.
Every S row names the same 34.033 GiB pool, and every A row the same 1.732 GiB pool; **do not
sum repeated rows**. Together the pools reserve 35.766 GiB. Live allocation at the snapshot is
0.496 GiB in S and zero in A. Pool reservations include working space and graph outputs.
An individual graph has no additive exclusive pool size after sharing. All groups use these
same graph objects and state backing. Anchors can overlap student ticks, so A remains separate.

| # | Graph | Captured shape / cadence | Shared pool reserved GiB |
| ---: | --- | --- | ---: |
| 1 | Student tick | buffer 0; 1024 players; append 1; R4; decoder 0 | S: 34.033 |
| 2 | Student tick | buffer 0; 1024 players; append 1; R4; decoder 1024 | S: 34.033 |
| 3 | Student tick | buffer 1; 1024 players; append 1; R4; decoder 0 | S: 34.033 |
| 4 | Student tick | buffer 1; 1024 players; append 1; R4; decoder 1024 | S: 34.033 |
| 5 | Student tick | buffer 0; 1024 players; append 2; R2; decoder 0 | S: 34.033 |
| 6 | Student tick | buffer 0; 1024 players; append 2; R2; decoder 256 | S: 34.033 |
| 7 | Student tick | buffer 0; 1024 players; append 2; R2; decoder 512 | S: 34.033 |
| 8 | Student tick | buffer 0; 1024 players; append 2; R2; decoder 640 | S: 34.033 |
| 9 | Student tick | buffer 0; 1024 players; append 2; R2; decoder 768 | S: 34.033 |
| 10 | Student tick | buffer 0; 1024 players; append 2; R2; decoder 1024 | S: 34.033 |
| 11 | Student tick | buffer 1; 1024 players; append 2; R2; decoder 0 | S: 34.033 |
| 12 | Student tick | buffer 1; 1024 players; append 2; R2; decoder 256 | S: 34.033 |
| 13 | Student tick | buffer 1; 1024 players; append 2; R2; decoder 512 | S: 34.033 |
| 14 | Student tick | buffer 1; 1024 players; append 2; R2; decoder 640 | S: 34.033 |
| 15 | Student tick | buffer 1; 1024 players; append 2; R2; decoder 768 | S: 34.033 |
| 16 | Student tick | buffer 1; 1024 players; append 2; R2; decoder 1024 | S: 34.033 |
| 17 | Student tick | buffer 0; 1024 players; append 2; R4; decoder 0 | S: 34.033 |
| 18 | Student tick | buffer 0; 1024 players; append 2; R4; decoder 1024 | S: 34.033 |
| 19 | Student tick | buffer 1; 1024 players; append 2; R4; decoder 0 | S: 34.033 |
| 20 | Student tick | buffer 1; 1024 players; append 2; R4; decoder 1024 | S: 34.033 |
| 21 | Student restart | buffer 0; 1024 players; 128-frame restart | S: 34.033 |
| 22 | Student restart | buffer 1; 1024 players; 128-frame restart | S: 34.033 |
| 23 | Student value | 2048 windows; 256 context frames; one decision | S: 34.033 |
| 24 | Student value | 32 windows; 256 context frames; one decision | S: 34.033 |
| 25 | Student actor | 64 windows; 64 decisions; queue 21; R2, delays 16–21 | S: 34.033 |
| 26 | Student actor | 64 windows; 32 decisions; queue 21; R4, delays 16–20 | S: 34.033 |
| 27 | Reference forward | 64 windows; 64 decisions in chunks of 16; delay 16; R2 | S: 34.033 |
| 28 | Reference forward | 64 windows; 32 decisions in chunks of 16; delay 16; R4 | S: 34.033 |
| 29 | Reference forward | 64 windows; 64 decisions in chunks of 16; delay 17; R2 | S: 34.033 |
| 30 | Reference forward | 64 windows; 32 decisions in chunks of 16; delay 17; R4 | S: 34.033 |
| 31 | Reference forward | 64 windows; 64 decisions in chunks of 16; delay 18; R2 | S: 34.033 |
| 32 | Reference forward | 64 windows; 32 decisions in chunks of 16; delay 18; R4 | S: 34.033 |
| 33 | Reference forward | 64 windows; 64 decisions in chunks of 16; delay 19; R2 | S: 34.033 |
| 34 | Reference forward | 64 windows; 32 decisions in chunks of 16; delay 19; R4 | S: 34.033 |
| 35 | Reference forward | 64 windows; 64 decisions in chunks of 16; delay 20; R2 | S: 34.033 |
| 36 | Reference forward | 64 windows; 32 decisions in chunks of 16; delay 20; R4 | S: 34.033 |
| 37 | Reference forward | 64 windows; 64 decisions in chunks of 16; delay 21; R2 | S: 34.033 |
| 38 | Stacked anchors tick | buffer -1; 320 players; append 1; R4; decoder 0 | A: 1.732 |
| 39 | Stacked anchors tick | buffer -1; 320 players; append 1; R4; decoder 320 | A: 1.732 |
| 40 | Stacked anchors tick | buffer -1; 320 players; append 2; R2; decoder 0 | A: 1.732 |
| 41 | Stacked anchors tick | buffer -1; 320 players; append 2; R2; decoder 80 | A: 1.732 |
| 42 | Stacked anchors tick | buffer -1; 320 players; append 2; R2; decoder 160 | A: 1.732 |
| 43 | Stacked anchors tick | buffer -1; 320 players; append 2; R2; decoder 200 | A: 1.732 |
| 44 | Stacked anchors tick | buffer -1; 320 players; append 2; R2; decoder 240 | A: 1.732 |
| 45 | Stacked anchors tick | buffer -1; 320 players; append 2; R2; decoder 320 | A: 1.732 |
| 46 | Stacked anchors tick | buffer -1; 320 players; append 2; R4; decoder 0 | A: 1.732 |
| 47 | Stacked anchors tick | buffer -1; 320 players; append 2; R4; decoder 320 | A: 1.732 |
| 48 | Stacked anchors restart | buffer -1; 320 players; 128-frame restart | A: 1.732 |

## Commits

- `ab95f13e`: share serial graph pools, chunk KL, release reference activations, inventory and parity tests.
- `bf632e41`: allow a scratch-only graph pool to retain reserved storage with no live allocations.
- `ff2eae4d`: validate complete reference and mask shapes before chunking KL.
- `8376587a`: boundary cache release, later rejected.
- `3662ead9`: revert the unqualified boundary release.
- `cb5ac6a9`: add KL support failure diagnostics; preserve validation and valid arithmetic.
- `e7099609`: avoid Torch 2.11 snapshot reference corruption; count forward buffers and parameter gradients.

## Measurements

The approved single B200 session uses sandbox `sb-h7M4OS44kSfSfF9Hu5eaYY`, devbox `main`,
Torch 2.11.0+cu130. The retained six-update profile and gradient probe use source `ff2eae4d`.
The final source adds failure diagnostics and fixes inventory accounting and the Torch snapshot
boundary; policy arithmetic is unchanged. The rejected cache release is absent.

An allocator compatibility probe captured three independent serial graphs into one pool,
replayed them in a different order and copied outputs before reuse. Both native allocation
and `expandable_segments:True` passed exact arithmetic parity. Both reserved exactly 84 MiB,
allocated 28.23 MiB after capture and peaked at 39.30 MiB allocated. The effective expandable
setting and expandable segment flag were checked. There was no reduction in the measured
reserved-minus-allocated gap, so expandable segments were not adopted in either devbox or
production launcher. This small probe is not a full-run fragmentation comparison.
Evidence: `/home/ericgu/data/scratch/o65-main-headroom-allocator-20261024-085604.json`.

The v2 cache volume contained Inductor and Triton files from before this sandbox started.
A slow recursive cache size scan was stopped and the same sandbox continued; no second cloud
machine was started. A shallow cache check confirmed persistence. Final shutdown ran
`sync /cache` successfully before termination; the devbox state file is absent.

The first complete profile is
`/home/ericgu/data/scratch/o65-devbox/runs/bench-20261024-085854-m64-full`, source `ff2eae4d`.
All six updates completed, groups 0, 1, 2, 3, 0, 1. Timed update: 26.980 s; ordinary tick:
28.730 ms (252 host-scoped ticks); boundary tick: 219.945 ms (four); group switch enqueue:
0.801 ms. Its phases were rollout 8.155 s, preparation 4.438 s (reference 3.663 s), critic
0.949 s and actor 13.388 s. Architecture-based MFU: 13.88%, 22.29%, 2.52% and 23.92%
respectively; whole update 19.80%, using 2250 dense BF16 TFLOP/s. This counts modeled work,
including padded slots, rather than hardware counters. Host cgroup peak was 28.980 GiB,
sampled every 0.5 s. There were zero completed games, so mean finished-game length is unavailable.

The reported MFU comes from the runtime `flops/*` and `mfu/*` metrics, computed by
`hal/training/self_play_flops.py` from the actual architecture. `resolved.json.flop_model`
and the bounded-sample `estimates.json` still describe a legacy 9.4M/6.7M proxy. Those proxy
fields were not used for these results and must be corrected before reuse for the 247M policy.

The fourth group raised reserved memory to 163.801 GiB, even though maximum allocated memory
was 96.013 GiB. Eager buffers of different sizes accumulated in the allocator cache. This did
not meet the reserved-memory goal. The rejected follow-up released unused allocator blocks after the
existing device synchronization at each update boundary. Live tensors and graph pools stay
allocated. The CUDA regression reclaims a discarded 128 MiB block and then checks exact replay
of a live graph. The update timer includes this release. An independent review found no new
lifetime issue. The second profile ran in the **same** sandbox after the gradient probe.
Its result is below.

The gradient comparison used checkpoint weights and captured first-update inputs, with no
optimizer step or clipping. All 15 batches were disjoint; real windows were 504 for the first
seven batches and 503 for the remaining eight, padded to the unchanged 512-window step size.
The whole-update FP32 gradient used global advantage moments; batch gradients used their
production per-step moments.

| Gradient comparison | Cosine | Relative L2 (second is denominator) |
| --- | ---: | ---: |
| FP32 batch 0 vs batch 1 | -0.010984 | 1.431905 |
| FP32 batch 0 vs whole update | 0.255069 | 3.096778 |
| Eager BF16 vs FP32, batch 0 | 0.726689 | 0.730865 |
| Eager BF16 vs FP32, batch 1 | 0.656512 | 0.806134 |
| Compiled BF16 vs FP32, batch 0 | 0.698022 | 0.788494 |
| Compiled BF16 vs FP32, batch 1 | 0.675673 | 0.770496 |

These two minibatches differ more than either precision comparison. This is fixed-weight,
one-update evidence, not a convergence result or proof of unbiased precision error.
Probe host peak was 23.138 GiB. Full layer comparisons, hashes and environment are retained in
`/home/ericgu/data/scratch/o65-main-headroom-gradient-20261024-085852.json`; the input manifest
is the adjacent `o65-main-headroom-gradient-manifest-20261024-085852.json`.

The follow-up is
`/home/ericgu/data/scratch/o65-devbox/runs/bench-20261024-092243-m64-full`, measured `8376587a`.
It completed groups 0, 1 and 2 in 26.828, 26.953 and 27.495 s. Its maximum reserved memory
through those updates was 140.371 GiB (startup still 146.643 GiB). Group 3 preparation failed
with `policy and reference must have identical support`. No completed timed/traced pair exists.
Do not select this treatment on memory alone: it was rejected and reverted. The earlier small
CUDA regression established simple live-output graph replay only, and did not qualify the
production path. Independent ownership review found no dangling captured tensor. Underflow
of a finite log-softmax probability to zero is one hypothesis; the artifacts contain no
probability tensors that establish it. The failure could also affect the retained source; the
runs were not bitwise controls, so the cache-release call is not established as the cause.
No support tolerance or probability floor was added.
The final failure diagnostic reports the factor, decision chunk, flat category, shape and
both log probabilities so the next failure is useful.

Warm startup after the sandbox restart (persisted volume cache) was 938.818 s total, including
817.218 s capture/preparation. Repeating in the same sandbox took 300.800 s total, including
195.177 s capture/preparation. These include unchanged head-start work. The previous control
was 957.970 s total and 833.266 s capture. Fresh cold total startup was not measured: the old
cold log records only first-group phases (984.061 + 84.102 + 502.907 + 494.989 = 2066.059 s),
with different older defaults and no complete startup total. This is not a controlled startup
speedup comparison. Cache persistence is established by pre-existing files, the warm repeat,
and successful final `sync /cache` on Volume v2.

The group switch enqueue was 0.801 ms in the timed profile. Full group setup was 21.898 ms;
its separate traced sample spanned 17.59 ms on GPU with 3.05 ms of copy activity. Those scopes
include other setup and do not isolate the GPU switch alone. Tick trace summaries are retained
in `/home/ericgu/data/scratch/o65-main-headroom-tick-trace.txt` and
`/home/ericgu/data/scratch/o65-main-headroom-group-trace.txt`.

| Phase | Control startup allocated / reserved GiB | Retained startup allocated / reserved GiB |
| --- | ---: | ---: |
| Rollout | 70.13 / 150.59 | 69.98 / 93.24 |
| Bootstrap value | 67.66 / 150.59 | 66.64 / 95.63 |
| Reference | 78.72 / 170.58 | 64.30 / 102.80 |
| Actor | 90.86 / 176.50 | 89.81 / 146.64 |

| Update phase | Timed seconds | MFU | Maximum allocated / reserved GiB across all six updates |
| --- | ---: | ---: | ---: |
| Rollout | 8.155 | 13.88% | 58.548 / 163.801 |
| Preparation | 4.438 | 22.29% | 76.567 / 163.801 |
| Reference (within preparation) | 3.663 | Included above | 76.567 / 163.801 |
| Critic | 0.949 | 2.52% | 96.013 / 163.801 |
| Actor | 13.388 | 23.92% | 81.223 / 163.801 |

The retained run's ordinary tick is 28.730 ms and timed update throughput is 19432 match-frames/s.
The control failed before one full update, so matched before tick time, update seconds and MFU
are unavailable; no throughput speedup is claimed for this memory treatment. All changes were
measured together; their separate contributions were not isolated. The traced update took
32.929 s and is not the throughput result. Full machine-readable evidence, exact shapes, bytes,
configuration and hashes are in [memory-headroom-measurement.json](memory-headroom-measurement.json).

## Tests run

The first focused run was interrupted by the orchestrator and was rerun. The first completed
rerun had 49 passes and two test failures: an incorrect nonzero-live-allocation assertion for
scratch-only anchor pools and an anchor test fixture with too few matches. Both were fixed.
The final focused run passed all 13 tests (606.40 s), including CPU/CUDA reference arithmetic
and lifetime parity, five four-group rotations through both buffers, and mixed reference/actor
replays through the shared pool. The KL shape validation run passed nine tests (two deselected,
8.20 s). XML and logs are under `/home/ericgu/data/scratch/o65-main-memory-*` and
`/home/ericgu/data/scratch/o65-main-kl-validation.*`.

The required full command ran under `nice` with the stream-specific TMPDIR and two compiler
threads. It segfaulted near 75%, exit 139, in the inventory's formatting after a filtered Torch
snapshot; no final XML was written. Progress output and crash stack are retained in
`/home/ericgu/data/scratch/o65-main-headroom-full.log`. The isolated startup inventory test
then passed (64.66 s). The dependency defect and fix are described below. Before the crash, progress recorded 2190 passes, 27 skips and four failures; skip names and
final counts were not flushed. After the fix, the affected module and all remaining test files,
plus the four known failing cases, ran in a fresh process: 740 passed, four failed, 12 skipped,
12 deselected (333.22 s). All affected O65 tests passed; the four failures reproduce the known
R2-credential failure and three netplay timeout assertions. The 12 skips were five missing dev
MDS cases, six missing local v7-subset cases, and one missing dev archive case. The full command
was not repeated from the beginning after the fix; it must not be described as a green full suite.

The new large-pool regression passed in 2.60 s. The failure-diagnostic/CPU KL selection passed
10 tests (two deselected) in 9.78 s. Session-cleanup integration passed both tests in 9.32 s,
with `.env` sourced and `HAL_REQUIRE_INTEGRATION=1`; six roundtrip cases remain blocked by
the missing dev MDS fixture. The final commands passed:

- `uv run ruff format --check .` (428 files).
- `uv run ruff check .`.
- `uv run ty check --python-version 3.14 --error-on-warning hal experiments/059_muon_action_sequence.py scripts`.
- `git diff --check`.

Every command, XML hash, failure and available skip reason is recorded in
[memory-headroom-validation.json](memory-headroom-validation.json).

## Self-review findings and fixes

Reviewed the entire branch diff against `a0ae12df`, including the merged library, trainer,
scripts and tests. Chunking initially allowed a trailing reference or mask row to be ignored;
validate the complete shapes first, with two regression cases. The scratch-only pool assertion
and undersized anchor fixture described above were test defects and were fixed.

An independent read-only Codex review found no confirmed stream race, graph-output lifetime,
base/view overwrite or gradient defect. Opus was unavailable; this is a stated substitution.
The reviewer checked prefetch waits, separate anchor pools, copies/scatters before pool reuse,
reference detach boundaries, bootstrap consumption and actor gradient buffers.

Inventory limitations: the measured writer omitted the reference forward copy's BF16 buffers
(about 0.47 GiB); its size can be added from the identically sized canonical reference buffers.
Small staging buffers, helper indices, critic gradients and unregistered rotary caches are
part of the measured allocator residual. The final writer now counts forward-copy buffers
and parameter gradients. Inventories are first-update snapshots: with nonzero critic
warmup they describe that warmup, whereas this zero-warmup run includes PPO/Muon state.
Phase memory peaks still cover the first PPO update with nonzero warmup.
The boundary release passed a small CUDA regression (1 passed in 2.13 s), but failed the
production follow-up. It was reverted despite a second ownership review finding no dangling
captured tensor. Cause remains unresolved. Added and tested error diagnostics without relaxing
the support check; the original valid arithmetic and memory treatment remain.

The full local suite exposed a Torch 2.11.0 reference-count defect in its pool-filtered snapshot
binding: it wraps borrowed tuple entries in owning `THPObjectPtr` values. Large pool IDs can
then corrupt Python memory. The final writer takes one unfiltered public snapshot and selects
`segment_pool_id` in Python. The regression advances capture IDs beyond 256, rejects unsafe
filtered calls, checks exact pool byte totals and replays the graph. Torch remains pinned to
2.11.0 in `pyproject.toml`; remove the workaround after an upstream fix passes this regression.
Sources: [snapshot binding](https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/torch/csrc/cuda/Module.cpp#L683),
[owning destructor](https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/torch/csrc/utils/object_ptr.cpp#L4).
Both B200 runs had only pool IDs 1 and 2, within Python's immortal integer cache; this local
crash is separate from the unresolved group-3 support mismatch.

## Cross-stream changes

The integration change touches student tick/restart/value pool selection and startup/profile
instrumentation in `experiments/065_selfplay_ppo.py`, in addition to reference preparation and
actor pool ownership. Focused tests cover each path. No new library or production-launcher
change was needed.

## Impasses

The control has no per-graph allocator snapshot or complete cold startup total. The current
dev MDS integration fixture is missing locally; six roundtrip integration cases are blocked,
as directed. No completed game was observed, so mean finished-game length is unavailable.
The follow-up's support mismatch cannot be diagnosed from allocator/metric snapshots alone.
The approved one-sandbox session has ended; a further production measurement needs another
approved session. Opus was unavailable; a read-only Codex review was used instead.

## Remaining work

Diagnose the support mismatch from actual offending probabilities and metadata before another
capacity treatment. Reduce or safely reuse eager critic/preparation workspace, then repeat all
six updates below about 160 GiB reserved. The eager critic cannot use idle graph-pool workspace;
reaching 110–120 GiB needs more work. Correct the legacy proxy FLOP metadata before reusing
those profiler fields for O64. Production capacity remains unset, and no launch is qualified
by this report. No training dimensions or policy arithmetic were reduced.
