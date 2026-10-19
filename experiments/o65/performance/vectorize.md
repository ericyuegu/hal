# O65 batched compute

## Merge qualification

The merged experiment at `499b05eb` matches fresh captures from the quickfix
experiment at `a5fb7706` exactly at 16 and 64 native matches. Each capture covers
two deterministic updates, with resets and timeouts (`max_frame=0`), seed 65,
the recorded O63 checkpoint, four shards, and both development cadences.
The capture files identify the complete configuration, environment and inputs.
No merge defect was found.

The experiment source copies were checked against their Git objects before the
fixture was replaced. The new fixture is the **quickfix control**, not a capture
from the merged implementation. Every compared field matches: native frame and
controller bytes, decision tensors, behavior/old/reference log-probabilities,
values, advantages, batch hashes and both post-update parameter hashes.

Raw evidence is in `runs/vectorize/{control,merge}-parity-{16,64}/capture.json`.
The maintained resume verifier also passes on the merge at 16 matches:

- Next batch: `a6a2b43b7e1cf0bd3735a20102f8d1cba561af9babb8faf821c9d3cd6d048b23`.
- Optimizers and scheduler: `f869d946b2013b3a6a4bccfdfa0a40c8c70acad17930da96e70501fcd134dc38`.
- Maximum parameter error: zero.

Every GPU command holds `/home/ericgu/data/scratch/o65-tmp/gpu.lock`, uses a
systemd scope with `MemoryMax=10G`, and sets `OMP_NUM_THREADS=2`. Local jobs
use only 16 or 64 matches. Logs and pytest basetemps stay under `runs/vectorize`
and `$TMPDIR`; jobs run serially.

## Execution mode and resume

Training and the pilot's connected segments use default algorithms. Format
`hal.o65.training.v4` records deterministic mode and the cuBLAS workspace in
the training shard, separately from configuration identity. Old formats are
rejected. All other configuration, environment and artifact checks remain.
The verifier first restores a default-mode checkpoint, runs two deterministic
next updates from that boundary, then restores the boundary and training mode.
Graphs from the other execution mode are discarded.

Focused CPU checks pass: 14 tests for mode changes, pilot commands, retention
and old-format rejection; another 8 cover both resume directions, malformed
execution metadata and the verifier in warmup, at warmup exit and during PPO.
Both directions retain strict rejection of a changed actor learning rate.
The native default-mode checkpoint verifier also passes at 16 matches, with
`max_frame=140`: next-batch hash
`d51d1368cdf13ba5ace0b596adbe106483247a735aa3e14b6640e7820ba21919`,
optimizer/scheduler hash
`84a5fca57a6206cba2410ad03c6240a66113d2c639e19f0703b989b6cbb4b32e`,
and zero parameter error. It records deterministic comparison mode, then returns
to the default training mode and the original boundary.


## Task 2: one graph per policy and tick

Default-algorithm CUDA rollouts use one fixed set of rows for each policy and
active group. Both cadences append two observations each tick. The R=4 rows
still append history on their off ticks; masks prevent a decision or queue
write there. Separate graph variants handle the block's first observation,
all-player ticks, and the intervening R=2 ticks. Each graph includes packed
context construction, the trunk KV append, sampling, controller conversion,
and action/controller ring writes. The simulator receives one pinned controller
copy per shard. There is no per-tick device-to-host scalar validation.

The two shard halves now decide together. This trades their former overlap
for fewer graph calls. At these sizes there are two policy calls per two-frame
tick: one student and one reference anchor. The old path averaged 5.45 calls
at 16 matches and 5.94 at 64. Deterministic verification retains the grouped
path and the exact golden fixture.

Treatment and control use the same resolved settings and input hashes.
[Raw records](vectorize-measurements.json) retain all three updates, source-file
hashes, configuration, environment, input hashes and parity results. The table
averages updates 2 and 3; update 1 includes graph preparation. Warmup has no
actor step. Memory is peak reserved GiB.

| Matches | Phase | Arm | Rollout s | No-grad s | Actor s | Critic s | Update s | Match-frames/s | GiB |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | PPO | Control | 7.729 | 0.623 | 0.504 | 0.017 | 8.915 | 459.451 | 2.230 |
| 16 | PPO | Tick graph | 4.337 | 0.619 | 0.505 | 0.017 | 5.520 | 742.071 | 2.170 |
| 64 | PPO | Control | 10.210 | 2.400 | 1.801 | 0.044 | 14.574 | 1124.160 | 3.928 |
| 64 | PPO | Tick graph | 8.091 | 2.344 | 1.797 | 0.044 | 12.396 | 1321.744 | 4.572 |
| 16 | WARMUP | Control | 7.760 | 0.340 | 0.000 | 0.017 | 8.160 | 501.951 | 1.201 |
| 16 | WARMUP | Tick graph | 4.350 | 0.338 | 0.000 | 0.017 | 4.748 | 862.742 | 1.180 |
| 64 | WARMUP | Control | 10.219 | 1.278 | 0.000 | 0.044 | 11.659 | 1405.209 | 2.605 |
| 64 | WARMUP | Tick graph | 8.117 | 1.261 | 0.000 | 0.044 | 9.542 | 1717.116 | 2.936 |

At 64 matches, complete updates fall 14.9% during PPO and 18.2% during warmup,
so this task passes the 5% stop rule. At 16 matches the reductions are 38.1%
and 41.8%. From 16 to 64, the tick rollout grows 1.87x, no-grad work about
3.8x, actor work 3.56x and critic work 2.69x. Padding the sparse anchor to the
whole group wastes compute and accounts for part of the rollout slope.

The proxy parity check compares the new tick and old grouped cache on common
observations, prefixes and sampled paths. It sums actor gradients over the
update at common weights, preserving each minibatch's placement and advantage
normalization. Results are:

| Quantity | Maximum absolute error | p99 absolute error |
| --- | ---: | ---: |
| Factor log-probability, nats | 0.060425 | 0.013217 |
| Cached value, reward units | 3.148190 | 1.470331 |
| Executed decision KL, nats | 0.002404 | 0.000327 |

Actor gradient cosine is 0.999804 and relative difference norm is 0.019804.
Critic gradients match exactly: the unchanged full-window student pass still
supplies its values, features, rewards and GAE targets. Cached values are
only diagnostics in this task. Their gap to training is max 3.096, p99 1.533;
the old grouped cache already has max 2.710, p99 1.673. Training values have
standard deviation 17.591 and maximum magnitude 105.778.

The test limits are max/p99 0.2/0.06 nats for log-probability, 5/2 reward units
for cached value, and 0.02/0.005 nats for KL. The value limits are 3.3%/1.3%
of the critic's 150-unit loss scale and apply only to unused cached values.
Both gradient cosines must exceed 0.999 and their relative difference norms
must be below 0.03. These bounds admit BF16 cache-kernel differences while
keeping the importance-weight and gradient changes small. They do not justify
replacing the training pass with cached values or probabilities.

Focused qualification passes all five CPU/CUDA tests, including controller
ring parity after every comparison tick, odd timeouts, resets, unsorted logical
stream ids, checkpoint restore across execution modes, and reuse after that
verification. Wire-table addresses stay stable across cache clears.

Two graph trials initially failed because tuple-based factor indexing copied
a CPU index during capture; the decoder now retains a device index. The first
parity test incorrectly checked support on unexecuted padded factors; it now
checks only executed factors. A later run exposed the overly tight cached
value limit. The comparison above records both old and new cache-to-training
gaps. Final qualification has five passes and no skips.


## Task 3: keep the student scoring pass

The precision trial uses the same inputs, sampler, update schedule and tick
graphs. Treatment runs the context encoder, trunk and stored KV history in
FP32. Control uses BF16 there. Both retain the existing FP32 decoder and the
unchanged BF16 student no-grad pass, which matches actor precision. Each arm
runs three complete PPO updates; these steady numbers average updates 2 and 3.
The gap is the absolute **joint decision** log-probability difference over
executed factors, as logged by `numerics/mu_old_gap_*`.

| Matches | Rollout precision | Rollout s | Update s | Mean p99 gap, nats | Maximum gap, nats |
| ---: | --- | ---: | ---: | ---: | ---: |
| 16 | BF16 | 4.337 | 5.520 | 0.049790 | 0.119801 |
| 16 | FP32 | 4.899 | 6.073 | 0.050752 | 0.133454 |
| 64 | BF16 | 8.091 | 12.396 | 0.039873 | 0.138781 |
| 64 | FP32 | 10.515 | 14.836 | 0.051842 | 0.176960 |

The near-FP32 gate was p99 <= 1e-5 nats, allowing some margin over the earlier
4e-6 observation. This implementation does not reproduce that observation
against the current BF16 training pass. FP32 costs another 0.563 s per rollout
at 16 matches and 2.424 s at 64, while its gap stays around 0.05 nats. That is
large enough to change importance weights materially. Task 3 stops at its
specified gate: no stored-value approximation replaces pi_old, and no factor
or critic-feature cache is added. The FP32 trial remains only as local evidence;
production code retains BF16 rollouts and the student scoring pass.


## Task 4: reference batching reverted

The reference prototype uses vectorized window/decision gathers and CUDA graphs
for full-window encoding and teacher-forced replay. The student keeps its
existing pass. On a fixed 64-match rollout, after preparation, three scoring
repeats give:

| Reference path | Mean seconds | Peak reserved GiB |
| --- | ---: | ---: |
| Existing 16 windows / 128 decisions, eager | 1.077 | 2.723 |
| 32 windows / 256 decisions, graphed | 0.825 | 3.010 |
| 64 windows / 512 decisions, graphed | 0.809 | 3.703 |
| 128 windows / 1,024 decisions, graphed | 0.817 | 4.873 |
| Teacher-forced scoring per tick | 2.621 | 5.004 |

Thus prepare-time scoring at 64/512 wins this local comparison. At 16 matches,
its 0.209 s beats eager's 0.274 s and per-tick scoring's 2.357 s. These are
fixed-input scoring measurements, not complete updates; peak reserved memory
includes resident rollout storage and graphs for that arm.

The complete 64-match comparison runs three PPO updates per arm. At updates
2 and 3, mean no-grad time falls from 2.344 to 2.099 s, but complete time falls
only from 12.396 to 12.152 s (1.97%). Peak reserved memory rises from 4.572 to
5.428 GiB. The candidate adds graph ownership and padding code, so it fails the
5% stop rule and is reverted. The shipped reference keeps its 16/128 eager path.

The 16-match common-rollout check passes: max/p99 factor log-probability error
0.059958/0.009307 nats, unused reference value error 2.494663/1.157320 reward
units, and decision KL error 0.000681/0.000249 nats. Actor gradient cosine is
0.999987 with relative difference norm 0.005088; critic cosine is 0.999996 with
relative difference norm 0.003734. That comparison propagates the changed KL
reward through GAE before checking both gradients. Limits were max/p99
0.1/0.03 nats for log-probability, 5/2 reward units for unused reference values,
0.02/0.005 nats for KL, cosine >0.999 and relative norm <0.03. The initial
64-match benchmark failed only its provisional 0.5-unit unused-value bound;
the retained 16-match diagnostic uses the same 150-unit scale rationale as
task 2. No numerical failure caused the revert.

The rejected source and driver remain in local `runs/vectorize/score-treatment.py`
and `score-qualification.py`; their tests are not kept as maintained tests for a
path that is no longer present. Committed raw records retain the source checksum,
environment, timings and parity results.


## Task 5: static actor graph reverted

The actor prototype pads to 32 windows and 512 selected decisions, masks padding
from normalization and both losses, and captures forward, joint PPO, direct KL
and backward together. Optimizer steps stay outside the graph. Its tensor loss
matches the checked loss and stop-gradient behavior in focused FP32/FP64 tests
at 1e-6/1e-12 tolerance. Every update still uses the recorded window placement.

The three-update complete-path comparison gives these means at updates 2 and 3:

| Matches | Arm | Actor s | Update s | Peak reserved GiB |
| ---: | --- | ---: | ---: | ---: |
| 16 | Eager | 0.505 | 5.520 | 2.170 |
| 16 | Static graph | 0.734 | 5.724 | 4.484 |
| 64 | Eager | 1.797 | 12.396 | 4.572 |
| 64 | Static graph | 1.461 | 12.074 | 6.303 |

The 64-match actor phase improves 18.7%, but the complete update improves only
2.6%. At 16 matches, padding about 16 real windows to 32 makes the actor slower.
This candidate adds a second loss path and graph lifecycle code, so it fails
the 5% stop rule and is reverted. No static actor or duplicate loss is shipped.

The 16-match diagnostic checks all four minibatches at common weights. Across
those steps, maximum/p99 log-probability errors reach 0.059480/0.015757 nats;
unused cached value errors reach 1.945038/1.537816 reward units; decision KL
errors reach 0.000838/0.000443 nats. The largest policy-loss difference is
0.0010715 and direct-KL-loss difference is 0.00003257. The worst actor-gradient
cosine is 0.999578 and relative difference norm is 0.030145. Critic gradients
match exactly. One step exceeds the predeclared 0.03 norm bound; that bound was
not relaxed. The qualification command reports 1 failure, 4 passes and
5 deselections. Two of its passes test the tensor loss; two test RoPE ownership.
The rejected source and diagnostic remain in local `runs/vectorize/`.

Review found an independent graph lifetime defect in `Rotary.at`: a capture
could read an eager cache whose storage another sequence length later replaces.
RoPE factors now compute inside explicit CUDA capture, as they already do under
Torch compilation. The focused FP32/BF16 tests overwrite the old eager factors
and require unchanged graph outputs. This small correctness fix is retained;
its graph replay adds no training alternative or loss implementation.


## Task 6: fixed anchor matches and policy buffers

The native training configuration now enables `fixed_anchor_lanes`. A group
reserves `floor(matches * anchor_rate + 0.5)` leading simulator matches for
anchors, with alternating anchor ports. This is an explicit treatment change
requested by task 6, not a numerically exact refactor. The 16/64-match trials
use 2/10 anchor matches (12.5%/15.625%); the sampled control had 3/11 under seed
65. At 1,024 matches the quota is 154 (15.04%). Other episode draws and logical
RNG keys keep their definitions. The resolved native configuration and resume
digest bind this choice. Golden qualification explicitly uses the prior sampled
layout, while real resume keeps its fixed-layout setting in both execution modes.

Each model owns a contiguous static compute buffer: `2M-C` student rows and up
to `C` rows for each anchor generation. Block-boundary gathers fill those buffers
without changing logical stream ids. A generation can occupy several simulator
lanes; its **compute** range and graph shape stay fixed. No active episode changes
opponent. A retired model is reused by loading new weights in place only when
no episode owns that generation. Frozen reference zero is never reused.

Graph ownership now includes the action-ring object, so a policy reused by
another engine cannot replay writes into the previous engine's storage. Keeping
the object in the key also keeps its referenced buffers alive. Focused tests
cover quota/port invariants through resets, release before snapshot reuse,
parameter addresses across the swap, strict resume-layout identity, odd resets,
unsorted stream ids and separate engines. Final focused qualification has
11 passes and no skips.

The final 16-match common-input comparison has max/p99 log-probability error
0.069510/0.012662 nats, unused cached value error 2.603963/1.440185 reward units,
and decision KL error 0.001665/0.000332 nats. Actor-gradient cosine is 0.999769
and relative difference norm is 0.021527. Critic gradients match exactly. It
uses task 2's unchanged bounds. These tests qualify numerical changes on common
inputs; they do not claim that changing anchor match allocation preserves a
whole training trajectory.

## Final complete-update measurements

All runs use the RTX 3060, the hash-pinned O63 proxy, seed 65, default algorithms,
four native shards on CPUs 0–3, one active group and 256 frames per rollout.
Both development cadences remain mixed. Three updates run per arm; the table
averages updates 2 and 3. Startup, evaluation and checkpoints are outside timing.
No timed run has an episode end or idle frame. Raw records retain allocated
and reserved peaks for every update; the table reports the reserved maximum.

| Matches | Phase | Rollout s | No-grad s | Actor s | Critic s | Update s | Match-frames/s | Peak reserved GiB | Decode calls/tick |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | PPO | 3.725 | 0.637 | 0.508 | 0.017 | 4.930 | 830.902 | 2.098 | 2.00 |
| 16 | WARMUP | 3.718 | 0.344 | 0.000 | 0.017 | 4.122 | 993.653 | 1.006 | 2.00 |
| 64 | PPO | 5.739 | 2.380 | 1.825 | 0.045 | 10.109 | 1620.820 | 3.932 | 2.00 |
| 64 | WARMUP | 5.718 | 1.268 | 0.000 | 0.045 | 7.150 | 2291.383 | 2.201 | 2.00 |

Compared with the merged control, complete PPO updates improve 44.7% at 16
matches and 30.6% at 64. Warmup improves 49.5% and 38.7%. The final task alone
improves 64-match PPO by 18.5% and warmup by 25.1% relative to task 2, passing
the stop rule. The shipped changes keep the student scoring pass, eager
reference scoring and eager actor steps.

From 16 to 64 matches, final PPO rollout time grows 1.54x, no-grad time 3.74x,
actor time 3.59x, critic time 2.71x and total time 2.05x. Warmup rollout grows
1.54x, no-grad time 3.69x, critic time 2.63x and total time 1.73x. Thus rollout
still has a large fixed cost; scoring and training scale close to the amount of
data. Phase times are the trainer's existing wall-clock metrics, and diagnostics
outside their timers remain part of the complete update.

## Projection to 1,024 matches on RTX PRO 6000

**Projection only; no 1,024-match or remote job ran.** Fit a straight line to
complete local update times at 16 and 64 matches, separately for control and
final code. Their 1,024-match RTX 3060 extrapolations are 127.8/113.7 s for PPO
and 81.6/67.7 s for warmup. Apply each final/control ratio to the supplied RTX
PRO 6000 observations of 97/73 s. This gives approximately **86 s per PPO
update (3,037 match-frames/s)** and **61 s per warmup update (4,330 match-frames/s)**.

This assumes the relative local gain transfers and per-match work stays linear.
The provided PRO 6000 observations predate this merge, so this calibration also
mixes source revisions. GPU kernels, 16 server shard threads, four resident
groups, graph memory and snapshot generations are not measured by these local
runs. The numbers are planning estimates, not a qualified production rate or
memory fit. The smaller projected gain is consistent with amortizing away much
of the fixed launch cost at larger batches.

## Review fixes and validation

The review covers `git diff o65-selfplay-ppo-plan...HEAD`, the O65 plan's math
and risks, and the performance-stream delta from `499b05eb`. It checks factor
and frame offsets, terminal masks, context ownership, critic detach points,
optimizer separation, counter domains, checkpoint identity, evaluation callers
and persisted formats. Prototype losses and graphs rejected by the stop rule
are absent from maintained code.

The first pass found and fixed three graph ownership issues: wire-table tensor
replacement, eager RoPE cache replacement, and reuse of a graph with another
engine's action ring. It also moved the verifier's temporary mode change inside
its cleanup scope; a failure during graph cleanup now restores the original
mode and update boundary. Focused tests cover each case.

A final check found a scalar device read in block prefill length validation.
Lengths now validate from CPU metadata before transfer; the cache still rejects
invalid lengths before mutation. Empty histories use a fixed-shape masked fill.
CPU/device-length cache parity is checked exactly. This changes validation
placement, not the trunk calculation or representation. It adds no unchecked
fast path. The CPU checks pass (7 tests); their two CUDA cases were explicitly
hidden in that diagnostic run and remain required in GPU qualification.

The first complete unit run passes 2,533 tests, with 8 skips, 89 integration
deselections and 27 warnings. Six skips lack
`data/processed/ranked-anonymized-1/mds-v7-sub4/train` in the supplied read-only
data tree. **Those missing fixtures remain a validation failure under AGENTS.md,
despite pytest's zero exit status.** Two other skips require the production GPU
and `HAL_REQUIRE_NETPLAY_HARDWARE_QUALIFICATION=1`. The reviewed code receives
another full unit run after the prefill change.
