# O65 GPU profile: remote run blocked

The RTX PRO 6000 result is **not measured**. The reserved VM was running
netplay when checked on 2026-10-19 at 07:34, 07:50, and 08:05 UTC. Both
`hal-netplay-runner.service` and `hal-netplay-health.service` were active.
Python PID 1054690 used 2,332 MiB; OBS PID 1054740 used 823 MiB. Xorg and an `AppRun.wrapped` graphics process also held GPU memory. The first check reported 15% GPU utilization.
No training container was started, and no service or VM setting was changed.
The stopped pilot's directory was not accessed. Nothing was uploaded or pushed.

These measurements therefore neither confirm nor refute the proposed bound
for the pilot's 73 s warmup and 97 s PPO updates. One source fact matters:
O65 uses **FP32 decoder computation**, including its attention, with a BF16
trunk. Dividing the entire FLOP count by BF16 peak does not measure utilization
of the arithmetic that this program executes.

## Profiling mode

Implementation commit: `137f2afe` (`Add bounded O65 GPU profiling`). Run:

```bash
uv run python experiments/065_selfplay_ppo.py profile \
  --checkpoint /path/to/delivery/payloads/final.pt \
  --data-dir /path/to/delivery/inputs/msl/data \
  --output runs/gpu-profile-1024
```

Use the delivery, build, and native-library environment from
`docker/o65-existing.sh`, `docker/o65-build-msl.sh`, and
`scripts/o65_delivery.py`. Do not run the pilot launcher: it also starts
learning, evaluation, and uploads. This mode does none of those external
operations. It does execute real PPO updates locally in the process.

Defaults match the pilot layout: CPUs 0–15, 16 shards of 64 matches, four
groups, 1,024 active matches, 256 rollout frames, sampling graphs,
32 windows per actor step, four critic epochs, seed 65, and critic warmup off.
The checkpoint still validates executed offsets for `(8, 4)` and `(10, 2)`.
Select physical cores on the GPU socket explicitly if these CPU numbers do
not match the pilot's topology.

Four initial updates fill the four groups and prepare their graphs. One
subsequent update supplies clean phase wall times and CUDA event spans.
The next update supplies bounded traces:

- Both block-boundary rollout ticks and four ordinary ticks, covering both
  decoder lengths. Each tick includes simulator stepping and device copies.
- One trunk chunk and one decoder chunk for each no-grad policy, plus the
  corresponding bootstrap chunks.
- The critic fit and the first two complete actor steps.

`timed.json`, `traced.json`, `estimates.json`, and `host.jsonl` record the
results. `resolved.json` records inputs, configuration, environment, source,
and artifact identities. Each phase includes Unix timestamps, so GPU samples
can be aligned with the phases. `/proc` supplies CPU ticks for every core and
process thread; `nvidia-smi` supplies GPU utilization, memory utilization,
power, and memory use at roughly half-second intervals.

CUDA event spans contain host gaps. They are **not GPU busy time**. Busy time
comes from the union of device kernel, copy, and memset intervals in each
trace. Kernel execution counts are separate from host CUDA launch counts:
a graph launch can execute thousands of kernels. Host launch counts here
include CUDA runtime events; nested CUDA driver events are not added.
Top-15 shares use summed
kernel durations. Measurement synchronizations, including profiler cleanup,
are excluded by timestamped annotations. Missing CUDA activities or required
trace sites cause an error.

Estimates scale counts by timed call counts, and device durations and bytes
by timed work. Partial batches therefore affect these two scales differently.
They cover sampled sites; validation, setup, concatenation, and diagnostics
between sites remain untraced. The two measured updates have consecutive
policy versions and can have different decision counts. They are not a
paired comparison of identical data. Traced wall times include export and
profiler overhead and must not be used as clean phase timings.

## Local qualification at 64 active matches

The [recorded evidence](gpu-profile-local.json) includes summaries, kernel
names, per-core CPU use, and hashes of the raw files. The raw 148 MiB of traces
remain under `runs/gpu-profile-64-final/`. These are bounded qualification
traces, not evidence about the reserved VM.

This run used the RTX 3060 12 GB, four native shards of 16 matches on CPUs
0–3, **one group and 64 total matches**, four initial updates, then the timed
and traced updates. It held the shared GPU lock and ran under `MemoryMax=10G`.
The treatment is profiling instrumentation; the control is the same
configuration without tracing. Checkpoint, sampler, seed, representation,
reward, optimizer, simulator, and adapter are invariant. No learning or
gameplay comparison is made.

The source is commit `137f2afee32d49ad3f4a4ee11899f31294350917`, with trainer hash
`890ed82b42601cfa2e14d2443e58159393531cf3be2184aa541e6900de61afcb`.
That hash was verified against the committed file. The simulator, adapter,
checkpoint, configuration, and software identities are in the evidence.
`CUBLAS_WORKSPACE_CONFIG=:4096:8`; OMP and MKL each use two threads, and
Inductor compilation uses one worker. Deterministic algorithms remain off.
An earlier local check used 64 active matches in a four-group, 256-match pool.
That exceeded a 64-total-match interpretation of the local limit; the
one-group run replaces it. Its numbers are not used below.

This branch already contains changes after pilot commit `e6d768d5`, including
columnar rollout storage, fixed rollout graphs, and vectorized preparation.
Its timings must not be attributed to the original pilot implementation.

The complete timed update took **10.090 s**, or **1,624 match-frames/s**.
It contains no resets, timeouts, or idle frames.
Busy times and counts below are sample estimates, not whole-phase captures.

| Phase | Wall s | Busy s, estimated | Kernel executions | CUDA runtime launches | Top 15 / kernel time | Model TFLOP/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rollout | 5.729 | 4.536 | 1,075,008 | 54,218 | 71.7% | 0.712 |
| No-grad student | 1.087 | 0.891 | 59,622 | 59,517 | 79.0% | 3.044 |
| No-grad reference | 1.089 | 0.893 | 59,622 | 59,517 | 78.9% | 3.039 |
| Critic | 0.045 | 0.042 | 613 | 597 | 87.6% | 1.358 |
| Actor | 1.813 | 1.606 | 104,496 | 102,912 | 48.2% | 3.051 |

Preparation takes 2.499 s **including** both no-grad passes. Its remaining
0.324 s includes bootstrap inference, GAE, selection, and diagnostics.
Bootstrap samples account for about 0.207 s of device work. Do not add the
inclusive preparation time to the no-grad rows.

| Phase | H2D copies / bytes, estimated | D2H copies / bytes, estimated | Synchronizations, estimated |
| --- | ---: | ---: | ---: |
| Rollout | 7,688 / 16,358,000 | 512 / 1,835,008 | 1,024 event + 5,124 stream |
| No-grad student | 91 / 92,672 | 0 in sampled chunks | 91 stream |
| No-grad reference | 91 / 92,672 | 0 in sampled chunks | 91 stream |
| Critic | 0 in sampled fit | 33 / 48 | 33 stream |
| Actor | 104 / about 133,351 | 864 / about 1,339 | 904 stream |

Zeros refer to sampled sites, not all operations in a phase. Estimated scalar
copy bytes can be fractional before rounding because the updates have
different row counts. Device-to-device copies are also recorded: about
50 MB in rollout, 59 MB in each no-grad pass, and 1.86 GB in actor steps.

The main thread used 87.5% of one CPU over the sampled interval. Mean GPU
utilization was 82.35%, memory utilization 40.9%, and power 97.0 W across
20 samples spanning 10.02 s. These coarse utilization readings do not measure
SM occupancy or achieved memory bandwidth. Native shard step times were
0.337, 0.335, 0.353, and 0.352 s. They include transfer waits and scheduling;
they are not pure simulator compute times and must not be added as wall time.
The GPU decision section occupied 4.832 s of the 5.729 s rollout.

The local evidence supports these limited conclusions:

| Phase | Observed constraint |
| --- | --- |
| Rollout | Many small device kernels and host work; simulator stepping is a much smaller component. FP32 memory-efficient attention alone accounts for about 1.33 s of estimated rollout kernel time. Graphs already reduce host launch count far below kernel count. |
| No-grad student | Device work dominates the sampled path. FP32 attention and SGEMM lead the decoder kernels; the estimated device gap is about 0.20 s. This is not evidence of a mostly idle GPU. |
| No-grad reference | Same pattern as the student. The estimated device gap is about 0.20 s. |
| Critic | Device work dominates this short fit. Layer normalization is the largest kernel. Its 0.045 s total makes it a low-priority target locally. |
| Actor | Small kernels, backward attention, and host launches all matter. Estimated device work occupies 89% of the phase, with about 0.21 s left. |

The no-grad pass keeps trunk chunks at 16 windows and decoder chunks at
128 decisions, regardless of the active match count. More matches therefore
add calls instead of increasing these kernel batch sizes. That is consistent
with the reported linear scaling; it does not establish which resource
bounds the RTX PRO 6000.

These measurements do not distinguish a compute limit from a memory-bandwidth
limit within the active kernels. Hardware counters or controlled changes are
needed for that distinction. An active GPU with low arithmetic throughput can
also be occupied by many small, poorly utilized kernels.

## FLOP model

Count a multiply-add as two FLOPs. With trunk parameter equivalent
`Pt = 9.4e6`, decoder parameter equivalent `Pd = 6.7e6`, executed trunk tokens
`Nt`, and executed decoder frame tokens `Nd`, use:

```text
forward FLOPs = 2 Pt Nt + 2 Pd Nd
actor FLOPs   = 3 × forward FLOPs
critic FLOPs  = 6 × critic_parameters × decision_rows × critic_epochs
```

Decoder tokens are frames, not categorical factors. Replay pads to
`max_delay + 4 = 14` frame tokens per decision. Rollout counts both decoder
lengths, fixed graph rows, and block prefills, including anchor work. The
critic uses its recorded parameter count. This is an approximate parameter
model; not every decoder parameter is used at every forced offset. It omits
attention score/value products, nonlinearities, feature building, optimizer
work, and copies. It is not a hardware-counter FLOP measurement.

The local counts give 4.081 TFLOP for rollout, 3.309 for each no-grad policy,
5.533 for the actor, 0.061 for the critic, and 1.180 for bootstrap inference.
The sum is about 17.47 TFLOP at 64 matches. A count-only multiplication by
16 gives about 280 TFLOP at 1,024 matches, close to the brief's 270 TFLOP.
That arithmetic does **not** predict time on another GPU. In particular,
FP32 decoder work cannot be charged against BF16 peak as if it used BF16
matrix instructions.

## Ranked follow-up measurements

No proposed optimization was implemented. These priorities are based on the
local trace, not a measured 1,024-match benefit:

1. Reduce decoder kernel count and FP32 attention cost. Rollout contains over
   a million kernel executions; attention and small SGEMMs lead its device
   time. Fusion or graph-internal kernel changes can reduce device work as
   well as launch work. Preserve the existing log-probability parity gate.
2. Reduce eager actor launch and pointwise work. It issues about 103,000 CUDA runtime
   launches in 1.813 s. Its top 15 kernels account for only 48.2% of device
   time. A launch-only change cannot be credited with removing device work.
3. Increase useful work per no-grad decoder chunk, then measure complete
   updates and memory. Each policy makes 91 decoder calls locally; sampled
   decoder device work is about 0.70 s per policy. Earlier batching changes
   on this branch include rejected trials, so their parity results must be
   checked before proposing a configuration.
4. Investigate remaining preparation and rollout host work. The local main
   thread is busy, but that alone does not show which work delays the GPU.
5. Defer simulator and critic optimization until the remote profile shows
   otherwise. Their local measured costs are much smaller than rollout decode.

**Estimated time saved at 1,024 matches: unavailable for every item.** No
remote baseline or valid scaling measurements exist in this task. Assigning
seconds would be speculation. If device kernel counts and durations stayed
fixed, the estimated local gaps across the five main phases total about
1.80 s at 64 matches; this is a conditional idle-gap ceiling, not a prediction
for fusion or for the RTX PRO 6000.

The remaining work is to obtain an idle VM, run 1,024 active matches with the
pilot CPU layout, reconcile source differences with `e6d768d5`, and replace
these limits with measured per-phase bounds and savings estimates. The
optional 256- and 4,096-match remote runs were not attempted.
