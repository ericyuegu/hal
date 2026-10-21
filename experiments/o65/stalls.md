# O65 checkpoint and snapshot stalls

Control: the trainer at `3d4ad115`. Treatment: `df79d181`. The treatment changes
checkpoint execution and anchor storage. The sampler, PPO loss, critic fit,
optimizer settings and counter draws are unchanged. AST checks compare the
scientific functions with `3d4ad115` in `runs/stalls/review-inventory.json`.

## RTX 3060 measurements

Host: Ryzen 5 5600X, six physical cores and 12 logical CPUs; RTX 3060 12 GB;
Torch 2.11.0+cu130; Python 3.14.4. Both arms use seed 65, the pinned 17M O63
bootstrap30 checkpoint, (D=8,R=4) and (D=10,R=2), one group of 16 matches on
two native shards, 256 frames per rollout, 32 windows per actor step, one actor
microbatch, four critic epochs, checkpoint interval 1 and snapshot interval 2.
Critic warmup and head starts are disabled for this execution test.

The snapshot probe sets native max_frame=0 and assigns one anchor lane to
pi_ref and the other to snapshot 1 at the first reset after promotion. This
forces a new batch size in the control within three updates. Both arms use
the same reset rule. These are timing probes; their truncated games provide
no learning or gameplay evidence. No other CUDA job ran during either arm.

| Measurement | Control | Treatment |
| --- | ---: | ---: |
| First update, with cached compilation artifacts | 58.976 s | 60.544 s |
| Update 2, before snapshot use | 3.054 s | 3.207 s |
| Update 3, after snapshot use | 206.071 s | 3.500 s |
| Match-frames/s during update 2 | 1,288.9 | 1,227.5 |
| Match-frames/s during update 3 | 19.1 | 1,124.6 |
| Compiler starts during update 3 | 3 | 0 |
| Shard 0 / shard 1 time during update 3 | 1.493 / 1.393 s | 1.587 / 1.408 s |
| Snapshot promotion at update 2 | not instrumented separately | 0.003 s |
| Checkpoint training thread, updates 1 / 2 / 3 | 0.823 / 0.955 / 0.934 s | 0.178 / 0.085 / 0.052 s |
| Checkpoint worker, updates 1 / 2 / 3 | synchronous | 0.678 / 0.838 / 0.761 s |
| Wait at checkpoints 1 / 2 / 3 | synchronous | 0 / 0 / 0.761 s |
| Peak allocated / reserved during update 3 | 1.261 / 6.512 GiB | 1.539 / 6.953 GiB |

The final wait drains the last checkpoint after training ends. There was no
wait at a subsequent checkpoint boundary. A separate control that moved both
anchor lanes to the new policy retained the existing batch size and started
no compilation. New batch sizes cause the cold compilation. New policy
objects also discard their CUDA graphs. Preallocation alone would leave the
batch-size stall; the treatment also keeps fixed anchor batch shapes.
Update 2 took 5% longer in this short probe. Its time plus the next checkpoint
capture fell from 4.009 to 3.292 s. Full-pilot throughput remains unmeasured.

A normal control with max_frame=28800 took 2.997 and 3.042 s for updates 2 and
3. The final normal-length treatment took 3.196 and 3.211 s, with zero compiler
starts in those updates. No match reached the new snapshot in those three
updates in either arm. The snapshot was already prepared in the treatment.
Logs and profiles are under `runs/stalls/`; the split comparison uses
`before-split-forced-clean` and `after-split-forced-clean`.

## Checkpoint breakdown

A profiled synchronous checkpoint at 16 matches took 1.064 s. Training-state
Torch serialization took 0.283 s; simulator save took 0.199 s; outer shard
compression took 0.027 s; evaluation export took 0.085 s; durable file writes
and hashing took 0.397 s. The last figure includes 0.131 s of file fsync and
part of the hashing cost. These categories leave about 0.07 s for other work.

Simulator save already compresses each match. The outer shard compression
therefore compresses an envelope of compressed matches. A separate 64-match
probe measured raw capture at 0.185–0.206 s, inner hashing and compression at
0.494–0.495 s, and outer compression at 0.073–0.074 s. Serial scaling to the
pilot's four groups of 16 shards × 64 matches gives about 48–51 s, before
model serialization and durable writes. This explains the order of the
reported 44 s stall; it is a local projection, not a server measurement.

The treatment submits raw captures to all existing native shard workers before
waiting. With 16 shards sharing this host's 12 logical CPUs, 1,024 matches
capture in 0.568 s after warmup, or 0.784 s on the first trial. The four-group
projection is 2.27 s warm or 3.14 s cold, plus tensor and metadata capture.
The full pilot's capture time on 16 server cores remains unmeasured. The
local 16-match checkpoint meets the 3 s training-thread target.

The first scaling probe mislabeled 12 actual shards as 16. The corrected
probe creates all 16 shards and records the logical CPU count. Its results
are in `runs/stalls/pool-profile.jsonl`; the original file is retained as
`pool-profile-mislabeled.jsonl`.

## Memory and resume

The pilot needs five snapshot slots: three candidates and two retired slots.
Each model has 71,141,026 bytes of weights and buffers. The measured KV state
uses 3,450,408 bytes per anchor lane. With 154 anchor lanes per group and four
groups, five slots require 10.229 GiB: 0.331 GiB of model state and 9.897 GiB
of KV state. This excludes CUDA graph workspace and allocator reservation,
which have not been measured at pilot scale. The two retirement slots account
for 4.092 GiB of that total.

The raw simulator snapshot is about 1.11 GiB for the measured 1,024 matches,
or 4.44 GiB when scaled to four groups. Only one checkpoint can own that
memory at a time. HAL compression, Torch serialization, SHA-256 hashing, disk
writes, compaction, hard links and remote publication run on the checkpoint
worker. The native API still computes its internal savestate checksum during
raw capture; that cost is included in the measurements above. Boundary
capture also freezes publication metadata so settings and logs cannot race
later training writes.

The formats remain training v8, native match v3, native pool v2 and rollout
pool v4. Tests
compare the next batch and optimizer update after synchronous and background
writes, including real native state. Other tests check worker failure, failed
writes before manifest publication, parameter address stability, CUDA graph
outputs after weight copies, and graph reuse after retirement.
