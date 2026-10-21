# O65 performance

We measured the 17M O63 bootstrap30 policy on an RTX PRO 6000 Blackwell,
using Torch 2.11.0+cu130, seed 65, cadences (8,4) and (10,2), 256 frames per
active match, four critic epochs. The measurements below use 32 windows per
actor step. The pilot uses 512 windows per step and the sampling inputs in
[pilot-protocol.md](pilot-protocol.md). Its complete update time is unmeasured.
The measured episodes use Fox/Falco on Final Destination, u=0.95 and starting
damage drawn from 0, 30, 60 or 90 percent.

## Complete updates

The latest server measurements use `5d122bbb`, with the 2,048-match arm at
`aa807cce`. Each arm times a warmed update with one active group and 16 native
shards. Actor time includes Muon. Other preparation excludes reference replay.

| Active matches | Rollout s | Reference s | Other preparation s | Critic s | Actor s | Update s | Match-frames/s | Student decisions/s | Peak allocated / reserved GiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | | | | | | 4.620547 | 14,184 | 10,070 | 6.599 / 16.840 |
| 1,024 | 6.703817 | 1.038549 | 0.344829 | 0.123855 | 3.211275 | 11.426755 | 22,941 | 15,873 | 23.264 / 46.307 |
| 2,048 | 8.846824 | 1.994630 | 0.694722 | 0.245387 | 6.352961 | 18.142752 | 28,898 | 20,026 | 45.574 / 84.197 |

The 1,024-match phase sum leaves 4.430 ms outside the timed ranges. Different
jobs received different cards and hosts, so the batch comparison also includes
hardware variation. We did not measure 4,096 active matches. Cold preparation
at 1,024 matches took 1,468 s, including 1,318 s for rollout compilation and
capture. These costs are excluded from warmed throughput.

## Partial floor

We replayed recorded GEMM dimensions, dtypes and strides on the same GPU model
at `5d122bbb`; supplementary prefill/bootstrap probes use `ae4eed4f`. We sum
measured time per GEMM over the observed calls. For [b,m,k] times [b,k,n], we
count 2bmnk FLOPs and element_bytes*b*(mk+kn+mn) logical operand bytes.

| Phase over a 1,024-match update | GEMM TFLOPs | Logical operand GB | GEMM calls | Partial floor s | Bare model s | Trainer s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rollout ticks and prefills | 55.191 | 336.899 | 127,782 | 0.685741 | 3.380142 | 6.703817 |
| Reference | 60.045 | 369.435 | 41,280 | 0.405774 | 1.143505 | 1.038549 |
| Other preparation | 10.580 | 57.766 | 218 | 0.044988 | 0.233756 | 0.344829 |
| Critic | 0.867 | 22.332 | 84 | 0.022943 | 0.119196 | 0.123855 |
| Actor, excluding optimizer | 83.694 | 487.814 | 42,007 | 0.513909 | 2.041313 | 3.211275 including optimizer |
| Muon and auxiliary Adam | 11.582 | 118.795 | 10,710 | 0.082408 | 0.630601 | 0.753907 traced |

The calculation excludes attention, norms, activations, sampling, loss
reductions and simulation. Logical operand bytes do not measure DRAM traffic.
GEMM times already include device scheduling. The bare rollout serializes both
buffers; the trainer overlaps them. Bare reference repeats recorded actor
slots with different padding, while bare actor backward uses synthetic
cotangents. These differences prevent an exact subtraction of model time.
We have not established a complete hardware floor.

For context, the measured streaming copy rate is 1.466 TB/s. A 1,024-node graph
averages 0.465 us per empty node and 0.737 us per single-element copy. Measured
BF16 GEMMs span 0.008 to 227.055 TFLOP/s. We use each measured shape in the
table, rather than the median rate or a constant multiplied by launch count.

## Measured changes and projections

| Source | Measurement or change | Complete update |
| --- | --- | ---: |
| `afaeb7ca` | RTX PRO 6000, 1,024 matches, FP32 decoder, 16-window/128-decision scoring chunks | 67.258 s measured |
| `8b187a9d` | Same trainer, scoring chunks increased to 64/512 for profiling | 42.413 s measured |
| `1a920ac7` | Dense rollout storage, stored behavior probabilities, BF16 decoder, window replay and graphs | 11.399 s measured |
| `5d122bbb` | Floor instrumentation and repeat server measurement | 11.426755 s measured |
| `974a9ca0` | Asynchronous controller transfers on the main thread | 4.246 to 4.500 s projected rollout |
| `a1f4a468` | Pilot uses 512 windows per actor step | 7.74 to 8.60 s projected update after async and larger steps |

At `974a9ca0`, local RTX 3060 complete updates decrease from 3.356 to 3.145 s
at 16 matches and from 5.159 to 4.983 s at 64 matches. Persistent compilation
reduces first-update preparation from 1,264.9 to 132.8 s at 16 matches.

For the server projection, we combine 3.006885 s of measured rollout GPU busy
time with 1.239 to 1.493 s of local host residual time. A warmed desktop
simulator probe projects 0.534 s at 1,024 matches on 16 cores, below that GPU
time. This assumes server CPU scaling and GPU work remain comparable. With
another estimated 0.6 to 1.2 s saved by larger actor steps, we project the full
update as 11.4 - (6.703817 - projected_rollout) - step_savings. The pilot must
measure the resulting runtime and memory use.

The original reports and measurement JSON remain available in Git at the
listed commits. Downloaded server results remain under `runs/modal-dense/`
and `runs/modal-floor/`, with remote copies in Modal Volume `hal-o65-profile`.

The earlier dense comparison against `aacdfee0`, recorded at `2ff3e77a`, found
shifts in KL and trajectory medians. It did not establish statistical equivalence.

## Checkpoint and snapshot stalls

The RTX 3060 probes in [stalls.md](stalls.md) reduce checkpoint capture to
0.052–0.178 s at 16 matches. A forced anchor split takes 206.071 s before the
change and 3.500 s after it, with compiler starts reduced from three to zero.
The full-pilot capture and CUDA graph memory figures remain projections.
