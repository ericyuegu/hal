# Proxy throughput and decoder precision

These measurements use the 17M-parameter O63 bootstrap30 checkpoint on an
RTX 3060 12 GB and Ryzen 5 5600X. They measure local execution cost. Learning,
production throughput, and gameplay improvement remain unmeasured.

## Inputs

Each comparison fixes the checkpoint, sampler, seed 65, initial weights,
optimizer settings, and starting states. The proxy executes delay 8/replan 4
and delay 10/replan 2. Updates collect 256 frames per active match. Heavy
processes run separately, with two Torch CPU threads.

| Input | SHA-256 |
| --- | --- |
| O63 `final.pt` | `65e7e43e5cbb18a365a005922ec1cfd026ad807c045eae13b331f487b4ec82e6` |
| Feature statistics | `6870bda6c0970826f6467407647cf9bb5901ef246b6947d87518db67252cdde3` |
| Training data manifest | `b97eab90e761bcf2bf03b48981f0ab6acc1ac3057157c58ae0c5a72c76c43bd8` |
| Sampler | `c5ffb1b5f75aa0d9245192304871c2979de9a3e59ddf0c5d55e6877994ab661e` |
| MSL game-data manifest | `4c2b074e872a7ad55176b63a3971fd6a3bbf09dd1988240d1dbdaf61a63616cb` |

The recorded environment uses Python 3.14.4, Torch 2.11.0+cu130, NumPy 2.1.3,
CUDA 13.0, and MSL source `8fc56d6f2523e978bf6037d8147c57778bd206fa`.
The measured MSL library hash is
`6588de095c024856176e9b7455a2cf4356a4622c817c6915476fc8a7f1ad084c`;
the adapter hash is
`e9e2bf2ee60a3cf31034f65a12505f29e06af010971d20d569c44b2849c5feb2`.

## Decoder execution

This comparison uses a match, a native shard, deterministic algorithms, and
cProfile. Each row includes cold graph preparation. The trunk and stored
history use BF16 on CUDA. The final decoder computes in FP32.

| Execution | Rollout s | No-grad s | Actor s | Critic s | Total s | Match-frames/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sequential eager | 36.830 | 58.107 | 16.970 | 0.107 | 112.356 | 2.28 |
| Batched replay and prefix prefill | 13.438 | 0.339 | 0.484 | 0.107 | 14.426 | 17.75 |
| CUDA sampling graphs, BF16 decoder | 7.805 | 0.338 | 0.488 | 0.107 | 8.795 | 29.11 |
| CUDA sampling graphs, FP32 decoder | 7.562 | 0.272 | 0.461 | 0.109 | 8.459 | 30.26 |

The FP32 decoder adds no material measured cost in this comparison. The 3.8%
difference between the last rows is too small to establish a speed increase.
The final clean-source profile uses implementation commit `fed3bcbf`.
Earlier profiles include uncommitted source changes with incomplete records.

## Sampling graph scaling

Before the packed rollout work, the FP32 decoder runs with a group and 12
native shards. These rows use default algorithms and disabled critic warmup,
without a profiler. The table reports update 3 at 16 matches and update 2 at
64 and 256 matches. Checkpoint writing and evaluation fall outside the timing.

| Matches | Rollout s | No-grad s | Actor s | Critic s | Total s | Match-frames/s | GPU use % | Peak allocated/reserved GiB | Host RSS GiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 20.968 | 1.254 | 0.612 | 0.047 | 22.956 | 178.43 | 37.8 | 1.610 / 1.893 | 2.593 |
| 64 | 31.367 | 4.824 | 2.264 | 0.165 | 38.878 | 421.42 | 41.5 | 3.213 / 3.588 | 2.781 |
| 256 | 53.107 | 19.321 | 8.860 | 0.658 | 82.896 | 790.58 | 47.0 | 5.533 / 7.773 | 3.535 |

## BF16 ablation

Eight native matches supply 16 streams and 512 fixed decisions from block 2.
Anchor assignment is disabled. Each variant scores the same observations,
queues, actions, masks, and conditions. Cached inference uses incremental
trunk updates and sequential action scoring. Full-window inference uses
parallel teacher forcing. The table reports absolute joint log-probability
differences over executed factors, in nats.

| FP32 component | p50 | p99 | Maximum | Diagnostic s |
| --- | ---: | ---: | ---: | ---: |
| BF16 control | 0.005164 | 0.107287 | 0.137884 | 1.929 |
| Last head linear | 0.003601 | 0.073364 | 0.108218 | 1.913 |
| Full output heads | 0.003525 | 0.070371 | 0.104355 | 1.979 |
| Decoder residual stream | 0.003030 | 0.092175 | 0.129845 | 1.960 |
| Heads and decoder residual stream | 0.002485 | 0.038571 | 0.061515 | 1.886 |
| Heads and decoder input projections | 0.002242 | 0.039199 | 0.048572 | 1.857 |
| Decoder including history projection | 0.001108 | 0.018186 | 0.030500 | 1.778 |
| Trunk | 0.004815 | 0.108846 | 0.184667 | 1.919 |
| KV storage | 0.005164 | 0.107287 | 0.137884 | 1.955 |
| All compute | 0.000000283 | 0.000003839 | 0.000006728 | 1.745 |
| Selected decoder, BF16 history projection and storage | 0.001433 | 0.020324 | 0.035253 | 1.722 |

The selected variant has a separate matched BF16 control with the same gap
quantiles and a 1.927 s diagnostic time. Timings exclude simulator collection.
The ablation changes projections before normalization; `log_softmax` already
computes in FP32 in the control.

The remaining trunk error matters for importance correction. At 256 matches,
the measured p99 is 0.03655 nats for delay 10/replan 2 and 0.05121 for delay
8/replan 4, with a maximum of 0.16769. No importance weight hits the cap.
The real-checkpoint regression retains its 0.04-nat joint-gap bound for its
fixed case. That bound does not qualify every rollout state.

## Packed rollout path

The control uses source `f09e124e`. The treatment uses `053c1c44`. The control
export was checked file by file against Git. Each arm uses a group, 12 native
shards, sampling graphs, default Torch algorithms, and disabled critic warmup.
The table reports update 2. Profilers run separately or after update timing.
Checkpoint writing and periodic evaluation fall outside the timed update.

| Matches | Arm | Rollout s | No-grad s | Actor s | Critic s | Total s | Match-frames/s | GPU use % |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | Control | 31.707 | 4.824 | 2.256 | 0.164 | 39.210 | 418 | 40.6 |
| 64 | Packed | 10.702 | 2.633 | 1.853 | 0.163 | 15.476 | 1059 | 76.8 |
| 256 | Control | 53.331 | 19.481 | 8.886 | 0.658 | 83.308 | 787 | 43.7 |
| 256 | Packed | 22.130 | 10.628 | 7.296 | 0.650 | 41.161 | 1592 | 80.0 |
| 512 | Control | 74.876 | 38.615 | 17.695 | 1.306 | 134.363 | 976 | 45.0 |
| 512 | Packed | 33.123 | 21.486 | 14.484 | 1.302 | 71.285 | 1839 | 71.6 |

| Matches | Arm | Peak allocated/reserved GiB | Peak host RSS GiB |
| ---: | --- | ---: | ---: |
| 64 | Control | 3.213 / 3.590 | 2.800 |
| 64 | Packed | 3.277 / 3.750 | 2.769 |
| 256 | Control | 5.533 / 7.773 | 3.527 |
| 256 | Packed | 5.639 / 8.035 | 3.340 |
| 512 | Control | 8.387 / 11.041 | 4.504 |
| 512 | Packed | 8.501 / 11.119 | 4.072 |

These short runs contain no resets or idle frames. GPU use is the mean of
one-second `nvidia-smi` samples. It does not measure SM occupancy. Diagnostics
and boundary cleanup explain the difference between the phase sum and total.
At 16 matches, deterministic execution takes 32.275 s per steady update versus
22.884 s under default algorithms. Resume qualification uses deterministic
execution.

The matched 256-match trace covers 16 student calls and 7,552 stream-frames.
GPU work occupies 885.7 ms of a 2,480.8 ms span in the control and 799.0 ms of
an 885.1 ms span in the treatment. The 512-match trace records 1.148 s of GPU
work in a 1.258 s span. These durations count kernels, copies, and memsets by
the union of their intervals. They exclude user annotations.

Mixed cadence requires separate cached-frame counts for two- and four-frame
appends. At 64 and 256 matches, 264 student calls use a context/cache group and
248 use two groups over two updates. No call uses more than two groups.
The 256-match run prepares four sampling graphs, with buckets 128 and 256.

## Simulator stepping

The native benchmark excludes Torch, features, and policy work. Each shard
steps 16 lanes for 256 frames. Serial and parallel schedules finish with
identical saved-state hashes.

| Shard threads | Match-frames/s |
| ---: | ---: |
| 1 | 27,296 |
| 6 | 131,713 |
| 12 | 166,045 |
| 12, repeated after packed rollout changes | 162,380 |

Reproduce this workload with `python -m hal.sim.native_benchmark`, using
`--lanes 16 --frames 256` and explicit CPUs. The bounded
[local pilot driver](../pilot.py) also measures complete updates during pool
qualification. Raw logs and diagnostic driver copies remain under ignored
`runs/`; they are excluded from the published source history.
