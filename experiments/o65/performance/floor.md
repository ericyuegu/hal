# O65 update cost and measured floor

Baseline source is `57ed1c57` on `o65-dense`. The floor stream starts from that
commit. No training optimization is admitted until hardware rates, bare-model
times, the complete update trace, and the phase floor table are recorded.

Treatment is measurement instrumentation, then active match count in the batch
sweep. Control is the unchanged dense trainer at 1,024 matches. Invariants are
the O63 bootstrap30 checkpoint, seed 65, one group, 16 simulator shards, 256
rollout frames, cadences `(8,4)` and `(10,2)`, BF16 decoding, 32 windows and 16
decisions per actor step, four critic epochs, reward, objective, GAE, simulator
and adapter. These are disposable systems measurements, not policy selection.

The preceding dense measurement is 11.399 s/update, 22,997 match-frames/s and
15,912 decisions/s (181,376 student decisions). The source and complete evidence are in [dense.md](dense.md).
The former launch-count projection is not a hardware floor.

The measurement job records the trainer's actual policy batch sizes, padded
reference window count and actor step count. The bare-model probe uses saved
end-of-update weights and recorded actor inputs. Tick tokens, queues and draws are synthetic
at the observed batch sizes. It measures trunk KV append and sampling without
queue management, critic, RNG, simulation or PPO. Actor backward uses fixed
cotangents on factor log-probabilities and full distributions. The reference
retains the actor's autograd metadata and scores four 16-slot slices per trunk
forward. Its per-chunk timing excludes trainer staging and concatenation.

Hardware GEMM shapes come from PyTorch profiler records of these model calls.
The profile records eager mathematical operations separately from compiled
timing, because the compiled decoder hides ATen matrix dimensions. Hardware
GEMM probes preserve the recorded BF16/FP32 types and operand strides.
The streaming copy uses two 256 MiB buffers and counts reads plus writes.
Empty and one-element kernels are measured at 1, 32, 256 and 1,024 nodes to
separate graph launch cost from the cost per node. Eager wall time includes
Python and Triton dispatch.

Nsight owns CUPTI for one complete warmed update. A separate clean update
provides times and rates. NVTX marks rollout, reference, preparation, critic,
actor, each tick and Muon. CUDA graph nodes are traced individually. Traced
times are never used as clean throughput. Raw reports, allocator snapshots,
model inputs, configuration and hashes stay under `runs/floor/` locally and
`runs/modal-floor/` for downloaded cloud results.

The floor stream has a separate fixed budget ledger at
`runs/modal-floor/budget.jsonl`: at most 7,200 reserved GPU-seconds and USD 20,
including the launcher's time margin and build allowance. Each call stops its
container. Only `o65-floor` may be pushed or launched by this stream.

## Results

Measured on RTX PRO 6000 Blackwell, Torch 2.11.0+cu130, on 2026-10-20 UTC.
The baseline is `5d122bbb`; the supplementary model probe is `ae4eed4f`.
Both use checkpoint SHA-256
`65e7e43e5cbb18a365a005922ec1cfd026ad807c045eae13b331f487b4ec82e6`.
The source, configuration, phase times, profiler shapes, per-shape rates and
call identities are in [floor-measurements.json](floor-measurements.json).
[floor-artifacts.json](floor-artifacts.json) hashes the downloaded evidence.
Raw reports remain in Modal Volume `hal-o65-profile` under
`floor-baseline-5d122bbb-20261020` and `floor-extra-ae4eed4f-20261020`.

### Hardware rates

The long sequences remove most fixed measurement cost. Eager times include
Python/Triton dispatch. Graph times include device scheduling. These are measured
rates, not the old constant multiplied by a launch count.

| Operation, 1,024 repetitions | Eager wall per kernel | Replayed graph per node |
| --- | ---: | ---: |
| Empty | 4.647 us | 0.465 us |
| One FP32 element copied | 5.730 us | 0.737 us |

One-node graph device time is 6.98 us empty and 6.75 us tiny. Small graph
measurements therefore need a fixed launch term. The memory copy reaches
1.466 TB/s with 512 MiB of reads plus writes. Repeating 32 copies gives
2.589, 5.570, 4.469 and 1.459 TB/s for 4, 32, 128 and 512 MiB working sets.
Cache traffic must not be charged at the DRAM rate.

The 207 ordinary-probe GEMM shapes retain their profiler dtype and strides.
BF16 throughput spans 0.008 to 227.055 TFLOP/s; its unweighted median is
24.620 TFLOP/s. FP32 spans 1.330 to 64.043 TFLOP/s. The range is large because
the decoder includes tiny matrix products. The best measured BF16 case is
`[8192,1024] @ [1024,256]`. The JSON contains every case, its latency and count;
the floor calculation below uses each matching shape, rather than the median.

### Bare model and batch sweep

The trained model config has width 256, 12 trunk layers, four heads, six decoder
layers, decoder width 256 and feed-forward width 1,024. Histories have 256
frames. There are three policy batches: 870 student and 154 anchor rows in the
first buffer, and 1,024 student rows in the second. There are 119 actor steps
and 4,128 padded reference windows, or 129 chunks of 32 windows.

Each tick appends two trunk tokens and samples one policy. Times below are
wall milliseconds for eager/compiled and CUDA milliseconds for graph replay.
They omit context encoding, RNG, critic, queue writes and simulation. Whole-tick
Inductor compilation is already present in the dense trainer.

| Policy rows | R | Eager ms | Compiled ms | Graph ms |
| ---: | ---: | ---: | ---: | ---: |
| 154 | 2 | 36.842 | 13.602 | 3.713 |
| 154 | 4 | 49.134 | 18.567 | 4.751 |
| 870 | 2 | 37.510 | 13.683 | 8.505 |
| 870 | 4 | 49.027 | 17.602 | 10.744 |
| 1,024 | 2 | 37.616 | 13.686 | 9.463 |
| 1,024 | 4 | 49.680 | 18.299 | 11.878 |

There are 64 calls of each R per policy batch per update. Thus ordinary bare
ticks total `64 * sum(six graph times) = 3.139410 s`. The two block boundaries
append one token instead of two; this sum slightly overstates that work.
Two prefills per policy batch cost
`2 * (0.008178 + 0.050272 + 0.061915) = 0.240732 s`.
These sums serialize both buffers; the trainer overlaps them, so they are not a
lower bound on its wall time.

Rates in phase tables are **phase-only capacity**: 262,144 match-frames or
181,376 student decisions divided by that phase's seconds. They are not full
training throughput. Bare totals are projections from measured component times.

| Bare work over one update | Seconds | Frames/s | Decisions/s |
| --- | ---: | ---: | ---: |
| Rollout ticks and prefills, serialized | 3.380142 | 77,554 | 53,659 |
| Reference, 129 chunks | 1.143505 | 229,246 | 158,614 |
| Preparation bootstrap, 32 + 2,048 rows | 0.233756 | 1,121,441 | 775,919 |
| Critic, four epochs | 0.119196 | 2,199,270 | 1,521,663 |
| Actor forward/backward, 119 steps | 2.041313 | 128,419 | 88,853 |
| Muon plus auxiliary Adam, eager, 119 steps | 0.630601 | 415,705 | 287,624 |
| Same optimizer, fixed-step graph probe | 0.256381 | 1,022,477 | 707,447 |

The reference repeats the 16 recorded actor slots four times. It preserves the
32/16/64 shapes, but does not reproduce the full reference batch's padding or
actions. Its 1.144 s exceeds the trainer's 1.039 s; subtraction would give a
false negative overhead. Actor backward uses synthetic cotangents, and critic
backward omits Adam. The optimizer graph holds Adam bias correction fixed and
must never be used as training code. These limits prevent an exact subtraction
of model time from trainer time for those phases.

| Matches | Measured seconds/update | Frames/s | Decisions/s | Peak allocated / reserved GiB |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 4.620547 | 14,184 | 10,070 | 6.599 / 16.840 |
| 1,024 | 11.426755 | 22,941 | 15,873 | 23.264 / 46.307 |
| 2,048 | Warm-up timed out | — | — | Not measured |
| 4,096 | Not attempted after deadline | — | — | Not measured |

The first job reached its absolute time limit during 2,048-match warm-up. It
did not record an OOM. No memory ceiling is claimed. The 1,024-match cold warm-up
took 1,468 s, including 1,318 s in rollout compilation/capture. Cold costs are
excluded from throughput, but included in the GPU budget.

### Floor arithmetic and its limits

For an observed GEMM `[batch,m,k] @ [batch,k,n]`, count
`F = 2*batch*m*n*k` FLOPs and
`Q = element_bytes*batch*(m*k + k*n + m*n)` operand bytes.
Multiply by the profiler call count and by 64 ticks, 129 reference chunks, 119
actor/optimizer steps, or the already-recorded four critic epochs. For each
GEMM, the measured-shape estimate is `count * seconds_per_gemm`; sum these.
For example, trunk QKV at two frames and B=1,024 has dimensions
`[2048,256] @ [256,768]`, giving 805,306,368 FLOPs. A trunk layer's four
projections have `3*d*d + d*d + 2*d*(4*d) = 12*d*d` weights. A decoder
layer has `4*d*d + 2*d*1024` weights. The JSON's shapes include backward GEMMs
and the Muon Newton–Schulz products; do not multiply backward counts by three
or critic counts by four again.

The next table records a **partial floor**, not a claim that every operation is
accounted for. FLOPs and bytes are the observed GEMM work only. Q is logical
operand traffic, not measured DRAM traffic. Fused attention, norms, activation,
sampling, loss reductions and simulator work remain outside those totals.
The measured-shape sum includes each GEMM's device scheduling cost; do not add
another per-kernel latency charge to it. Kernel counts are minima for this
unfused GEMM decomposition, not a universal minimum: a persistent kernel could
fuse several. The absolute minimum is one kernel per dependent call, which is
too weak to explain the observed time. No phase has an established complete
floor or a justified claim of being within 2x of it.

| Phase, whole update | GEMM TFLOPs | GEMM operand GB | GEMM kernels in decomposition | Partial floor s | Bare s | Trainer s | Limit / missing work |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Rollout tick, plus prefills | 55.191 | 336.899 | 127,782 | 0.686270 | 3.380142 | 6.703817 | Attention bandwidth, sampling, host waits; buffers overlap |
| Reference | 60.045 | 369.435 | 41,280 | 0.406875 | 1.143505 | 1.038549 | Attention/elementwise latency; bare padding differs |
| Preparation excluding reference | 10.580 | 57.766 | 218 | 0.044988 | 0.233756 | 0.344829 | Bootstrap compute, GAE/data traffic |
| Critic | 0.867 | 22.332 | 84 | 0.022943 | 0.119196 | 0.123855 | FP32 compute and bandwidth |
| Actor, excluding optimizer in bare/floor | 83.694 | 487.814 | 42,007 | 0.515157 | 2.041313 | 3.211275 incl. optimizer | Backward, attention and host dispatch |
| Muon plus auxiliary Adam | 11.582 | 118.795 | 10,710 | 0.082408 | 0.630601 | 0.753907 traced host only | Small matrix/elementwise kernels and eager dispatch |

For rechecking rates beside these seconds, the following gives phase capacity
at the partial floor, bare time and clean trainer time. Muon has no separately
saved clean host time; its last capacity uses traced host time and is marked T.

| Phase | Partial-floor frames/s; decisions/s | Bare frames/s; decisions/s | Trainer frames/s; decisions/s |
| --- | ---: | ---: | ---: |
| Rollout | 381,984; 264,292 | 77,554; 53,659 | 39,104; 27,056 |
| Reference | 644,287; 445,779 | 229,246; 158,614 | 252,414; 174,644 |
| Preparation | 5,826,970; 4,031,648 | 1,121,441; 775,919 | 760,214; 525,988 |
| Critic | 11,425,760; 7,905,421 | 2,199,270; 1,521,663 | 2,116,542; 1,464,424 |
| Actor | 508,862; 352,079 | 128,419; 88,853 | 81,632; 56,481 |
| Muon | 3,181,070; 2,200,965 | 415,705; 287,624 | 347,714; 240,581 T |

For KV size, the trainer already stores BF16. Three policy batches total 2,048
rows. The 12 trunk layers plus history attention require
`13 * 2 * 2048 * 259 * 256 * 2 = 7,061,110,784 bytes` of base KV storage.
This excludes captures, intermediate activations, reference storage and models.
Shortening context or changing groups would change the treatment; neither was
done. The profiler's memory-efficient rollout attention takes 1.089 GPU-kernel
seconds, and flash attention another 0.586. Those are implementation costs, not
hardware bandwidth floors.

### Time reconciliation

Clean phase time is
`6.703817 + 1.038549 + 0.344829 + 0.123855 + 3.211275 + 0.004430`
`= 11.426755 s` (22,941 frames/s; 15,873 decisions/s). Actor includes Muon.
The residual 4.430 ms is between phase ranges. This accounts for the clean wall
total without scaling launch counts.

Nsight increases total time to 13.374384 s (19,600 frames/s; 13,561 decisions/s).
The table partitions the traced ranges. GPU busy is an interval union, not the
sum of concurrent kernels. Muon work can cross its short host range, so its
host-minus-busy difference is an attribution residual, not a critical-path wait.

| Traced phase | Host s | Frames/s | Decisions/s | GPU busy s | Host minus GPU busy s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rollout | 8.191068 | 32,004 | 22,143 | 3.006885 | 5.184183 |
| Reference | 1.035886 | 253,063 | 175,093 | 0.935921 | 0.099965 |
| Preparation excluding reference | 0.353732 | 741,080 | 512,749 | 0.331981 | 0.021751 |
| Critic | 0.124681 | 2,102,524 | 1,454,725 | 0.118853 | 0.005827 |
| Actor excluding Muon | 2.910026 | 90,083 | 62,328 | 1.789846 | 1.120180 |
| Muon | 0.753907 | 347,714 | 240,581 | 0.223186 | 0.530721 |

Every kernel, memcpy and memset has a correlated launching API. There are no
unattributed GPU operations. The selected B=1,024, R=4, append-two graph has
1,532 kernels. [floor-tick-kernels.csv](floor-tick-kernels.csv) records each
name, duration, grid, block, graph node and issuing Python tick module. It does
not resolve the internal compiled graph back to individual submodules.

The trace records 31.123 aggregate thread-seconds in rollout
`cudaStreamSynchronize`; waits overlap across workers and must not be added to
wall time. The native shard's blocking GPU-to-host input copy is one repeated
source. Other rollout host work includes simulation, staging and thread
coordination. The trace proves a large GPU-idle gap, but does not isolate every
cause of that gap. Likewise, the partial floor leaves substantial non-GEMM
GPU work. A complete roofline and exact model/trainer subtraction remain open;
these residuals must not be presented as explained compute or memory costs.

### Next change

If the floor prerequisites are completed, the first candidate is one asynchronous controller-input copy per buffer, with
one readiness event, instead of one blocking copy per simulator shard. It keeps
the two buffers, FP32 controller values and all training math. The trace makes
this a more direct target than a decoder rewrite. Decoder fusion still needs
the fixed FP32 precision gate and a complete numerical review. Moving reference
scoring into rollout changes memory and scheduling substantially. Dropping
overlap is not justified: rollout kernel sums exceed their interval union.
