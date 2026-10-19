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
