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
