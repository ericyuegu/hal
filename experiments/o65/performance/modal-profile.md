# O65 Modal RTX PRO 6000 profile

The no-grad passes are limited by host dispatch at the default chunk sizes.
Increasing chunks from 16 windows / 128 decisions to 64 / 512 reduced estimated
CUDA runtime launches by 4.09× and clean scoring time by 3.90×. The simulator
was a small part of rollout. Actor training used about one CPU core in total,
split between the Python main thread and the autograd worker.

H1 and H4 are supported. H2 is inconclusive as a claim about the dominant
rollout limit. H5 is refuted as stated. H3 remains pending the decoder stream.
Larger scoring chunks failed a local gradient qualification check and remain
available only in profiling.

## Inputs and method

These are disposable systems runs from the O63 bootstrap30 proxy. They do not
select a learned policy or establish gameplay improvement. Treatment is active
match count, no-grad chunk size, or decoder precision. Control is 1,024 active
matches, four groups, FP32 decoding, and 16/128 chunks. Seed 65, checkpoint,
statistics, sampler, simulator source/build, reward, cadences `(8,4)` and
`(10,2)`, optimizer settings, 32 actor windows per step, four critic epochs,
and 256 rollout frames stay fixed within each comparison. Checkpoint supervised
offset checks remain enabled.

HAL uses published commits on `o65-modal`; full identities and raw-file hashes
are in `modal-profile-measurements.json`. The first FP32 baselines use
`afaeb7ca`; the chunk sweep uses `8b187a9d`. The trainer file is identical in
these commits. The latter changes launcher bookkeeping. All four runs have
the same simulator SHA `8fde568fb65ff4d0919bd71048808421c6b1a6c174d0d87cb83975696e3f75a8`
and adapter SHA `29f92fca9dfc77ef89ac86de51ccfaab20bbeffb87874b9348d503793f564145`.
The delivery validates the exact R2 keys, archive size and SHA-256, game data,
proxy, statistics, sampler, and MSL source against `pilot-inputs.json` before
GPU allocation. MSL is fork commit `2dbca5e3214d88859ac80333cc19caa328d4be4a`,
built with `docker/o65-build-msl.sh` and GCC 13.3.0.

Modal's GPU name is `RTX-PRO-6000`; each job received one NVIDIA RTX PRO 6000
Blackwell Server Edition with 97,887 MiB exposed memory. The existing locked
Modal SDK 1.5.3 dependency was used. Each job requests 17 CPU cores and 64 GiB
RAM. Modal exposes CPU IDs 0–16 but hides physical-core and NUMA topology;
16 shard threads bind to IDs 0–15. CPU attribution below uses the trainer's
per-thread `/proc` counters and exposed IDs. It does not measure other tenants
or physical host-core occupancy. [Modal GPU names](https://modal.com/docs/guide/gpu).

The existing Modal secret `hal` supplies the four AWS variables to the CPU
image build. No secret value was printed or committed. The GCP credential
comparison could not run because local authentication required renewal; the
fixed input bytes still passed all committed hash checks. R2 was read only.
W&B and evaluation were disabled. Local POSIX storage handles checkpoint hard
links, then completed outputs are copied to Volume `hal-o65-profile` and
downloaded under `runs/modal/`. Every job container stops after its call.

Each profile warms four groups, times one clean update with CUDA events and
host samples, then traces a separate update at bounded sites. Traced wall
times include profiling/export overhead and are excluded. Device busy time
comes from unions of GPU intervals in sampled traces, scaled to clean work;
it is an estimate with incomplete coverage between sites. NVML samples are
coarse and can span phase boundaries. Kernel executions and host runtime
launches are separate counts: CUDA graph launches execute many kernels.
Traced CUDA API duration includes profiler cost and is not a clean launch
latency. Model TFLOP/s uses the existing approximate parameter model, which
omits attention products, pointwise work, optimizer work, and copies. It is
not a hardware FLOP counter or MFU.

Arms share seeds and distributions, but each runs its own PPO warm updates.
Floating-point changes can change trajectories. The timed and traced updates
also have consecutive policy versions. No confidence interval is inferred
from these single measurements. These runs use post-columnar code and Modal's
virtual runtime; the earlier GCE pilot used another source/host environment.

## Clean phase times

All times are seconds. Other preparation is inclusive preparation minus the
student and reference rows; it includes bootstraps, GAE, selection and
diagnostics. This avoids double counting the no-grad passes. The full update
also includes small work outside phase timers. Peak reserved memory is the
clean update's CUDA allocator peak.

| Active matches | Windows/decisions | Rollout | Student | Reference | Other prep | Critic | Actor | Update | Peak GiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 16/128 | 7.433 | 2.918 | 2.945 | 1.010 | 0.031 | 4.162 | 18.505 | 20.15 |
| 1024 | 16/128 | 14.884 | 12.756 | 12.401 | 8.845 | 0.123 | 18.240 | 67.258 | 73.70 |
| 1024 | 32/256 | 13.672 | 5.716 | 5.816 | 6.650 | 0.125 | 15.499 | 47.485 | 73.78 |
| 1024 | 64/512 | 12.974 | 3.264 | 3.187 | 7.662 | 0.124 | 15.189 | 42.413 | 73.08 |

Observed complete-update throughput is 3,898 match-frames/s at 16/128 and
6,181 at 64/512. These are disposable profile arms; the faster configuration
is not qualified for training.

## Hypothesis verdicts

### H1: supported for host dispatch

At 1,024 matches, estimated GPU busy fractions are 25.8% for student scoring,
26.6% for reference scoring, and 29.5% for actor training. Coarse NVML readings
are 32.2%, 31.7%, and 36.0%. The scoring main thread uses 98.2% and 97.2% of a
core; actor main plus autograd uses about one core. Increasing no-grad chunks
holds actor batch shapes fixed.

| Chunks | Combined scoring s | Estimated runtime launches | Wall / launch, µs | Estimated scoring GPU busy |
| --- | ---: | ---: | ---: | ---: |
| 16/128 | 25.157 | 1,790,602 | 14.05 | 26.2% |
| 32/256 | 11.532 | 873,946 | 13.20 | 40.1% |
| 64/512 | 6.451 | 438,290 | 14.72 | 57.7% |

The launch reduction and time reduction nearly match. The effective 13–15 µs
per launch includes Python/Torch dispatch and other host work. CUDA runtime
API duration alone does not explain the phase time. Actor launches scale
from about 373,320 at 256 matches to 1,486,786 at 1,024; its wall time scales
from 4.162 to 18.240 s while device busy time remains a small fraction.

### H2: inconclusive as the dominant rollout limit

The implemented decoder runs six Transformer layers **once per offset**, then
four categorical heads in sequence. It does not rerun six layers per factor.
Forced-prefix prefill is batched. This corrects the hypothesis's description.

| Ordinary tick measurement | 256 matches | 1,024 matches |
| --- | ---: | ---: |
| Clean mean tick wall, ms | 56.362 | 110.234 |
| Kernel executions, R=2 tick | 8,134 | 8,058 |
| Kernel executions, R=4 tick | 10,014 | 9,914 |
| Mean kernel duration, R=2, µs | 2.59 | 5.00 |
| Mean kernel duration, R=4, µs | 2.67 | 5.19 |
| Rollout busy fraction, estimated | 42.3% | 41.8% |
| Rollout model TFLOP/s | 2.20 | 4.39 |

Four times as many matches costs 1.96× tick time; kernel counts stay within
about 1%. Kernels are short. These results support a serial device cost and
useful batching. GPU busy time is about 42%, which does not support the
predicted high utilization or establish that the device chain dominates the
complete rollout. Host work also matters. Boundary ticks are separate:
95.943 and 252.813 ms; they include prefill. A single kernel count is not used
for both cadence variants.

The 4,096-match **four-group** configuration was not run. Its persistent BF16
KV and decoder-history storage alone needs 105.219 GiB:
`4 groups × 2 players × 4096 × 2 K/V × 259 slots × 2 bytes × (12×256 + 256)`.
The GPU exposes 95.593 GiB. The 1,024-match floor is 26.305 GiB, and its
measured complete-update peak is 73.70 GiB. Tick caches retain separate group
storage. Reducing groups would change the comparison; no group count was
silently substituted. Cache reuse is a separate scaling change.

### H3: pending

At the final dependency check on 2026-10-19 at 18:46 UTC, the decoder stream
had not published its final report (latest commit `c74a3f23`). Its implementation
and numerical qualification are still changing. No PRO 6000 BF16 arm has
been run. There is no measured H3 verdict yet. The FP32 scoring decoder
accounts for about 5.45 s of estimated kernel time across student and
reference at 1,024 matches, but this does not measure BF16 savings.

### H4: supported

At 1,024 matches, each shard's total step time is 1.451–2.127 s over 128 ticks,
or 11.34–16.62 ms per tick. The largest mean is 15.1% of the 110.23 ms ordinary
tick, below the 20% threshold. Those step timers include copies, waits and
scheduling, so they bound pure simulator compute from above. Pinned shard
threads use 5.9–7.9% of an exposed CPU over the clean rollout sample. Physical
host topology is unavailable, but the process-level evidence does not show a
simulator throughput limit.

### H5: refuted as stated

The scoring main thread uses about one core. During actor training, the main
thread uses 50.4% and `pt_autograd_0` uses 47.3%; total trainer CPU is 98.7%.
At 256 matches the corresponding split is 49.2% / 48.2%. The actor's host work
moves between these threads. A host-dispatch limit does not imply a main
thread running at 100% throughout backward.

## Chunk qualification limit

A separate local RTX 3060 check used one 16-match rollout, four shards, seed
65, common weights and recorded actions. The reference was frozen before a
0.02 change to one student C-stick output bias, so KL and its effect on GAE
were nonzero. Each arm recomputed no-grad scoring, KL, advantages and critic
targets. Actor gradient sums used the same minibatches and weights.

| Chunks | Actor cosine | Actor relative difference norm | Critic cosine | Critic relative difference norm |
| --- | ---: | ---: | ---: | ---: |
| 32/256 | 0.999293 | 0.037595 | 0.999551 | 0.030112 |
| 64/512 | 0.999572 | 0.029297 | 0.999353 | 0.035969 |

The existing gate requires cosine >0.999 and relative difference norm <0.03
for both gradients. Both settings fail. Factor log-probabilities, values and
executed KL pass their declared limits; full error numbers and the diagnostic
snapshot hash are in the measurement file. No tolerance was relaxed. The CPU
row-order/feature/gradient parity test passes, but it did not predict these
BF16 batch-shape effects. The CLI therefore permits nondefault chunks only
for disposable profiling. A training change needs a new qualified
implementation and gameplay validation.

## Ranked changes at 1,024 matches

1. **Qualify larger no-grad chunks before using them for training.** Scoring alone saves 18.71 s in the
   recorded arms. Normalizing the baseline by the unchanged actor phase gives
   a 14.50 s sensitivity estimate. This is not a confidence interval. The raw
   full-update difference is 24.85 s (36.9%), but actor and rollout also became
   faster, so the full difference is not assigned to scoring. The 32/256 arm
   saves 13.62 s of scoring (about 9.84 s with the same normalization). Peak
   memory stays near 73 GiB. Defaults remain 16/128. Both larger settings failed the local
   gradient gate described above; training rejects them. A path that removes
   dispatch while keeping the qualified batch shapes is a useful alternative
   to test; no speedup for that implementation has been measured.
2. **Actor dispatch/fusion.** About 1.49 million host launches per update and
   roughly one CPU core of dispatch make this the next large target. The
   sampled idle budget is about 9.7–12.9 s across measured hosts; this is a
   conditional ceiling, not a measured speedup. Earlier actor graph work
   failed a gradient-parity gate, so a new implementation needs qualification.
3. **Preparation outside scoring and bootstrap chunks.** About 5.7–7.2 s is
   outside the sampled chunk bodies at 1,024 matches, compared with 0.48 s at
   256. Profile window placement, per-row materialization and diagnostics
   before changing them. That time is an upper bound on potential savings.
4. **Rollout device fusion after host work is measured.** The chain has about
   9,000 kernel executions per ordinary tick, but the main thread already
   uses about 76% of a core. Holding that CPU work fixed limits a purely
   device-side change to roughly 3.5 s of the 14.9 s rollout. No fusion speedup
   was measured here.
5. **Defer simulator and critic work.** The critic takes 0.123 s. Shard CPU use
   and step-time bounds leave little reason to optimize physics first.

BF16 remains unranked until H3 is measured. Savings from different proposals
are not additive. These estimates prioritize follow-up work, not learned
policies.

## Launch path and budget

Use `uv run python -m scripts.launch_o65_modal --mode smoke --matches 16`
first, then `--mode profile --matches 1024 --windows 64 --decisions 512`.
The launcher rejects Modal SDK versions other than tested 1.5.3 and the
known-oversized 4,096-match four-group request before allocation. Smoke uses
default chunk sizes. The launcher requires clean, published `o65-modal` HEAD, checks the named
secret's required keys, reserves time and cost before allocation, and rejects
unknown ledger/result formats. It never pushes automatically. The Modal image
clones that exact HAL commit and verifies fixed inputs on CPU before GPU work.
Outputs and app/call/container identities are retained in `runs/modal/`.

The final ledger reserves 4,443.94 GPU-seconds (1.234 GPU-hours). With the
CPU-build allowance, estimated reserved cost is $8.36; this is not an invoice.
The actual recorded worker time of completed GPU calls is 1,323.94 s. The
failed topology setup retains a full 960 s reservation. All listed apps are
stopped and have zero active tasks. Published rates checked 2026-10-19 are
$0.000842/GPU-second, $0.0000131/CPU-core-second, and
$0.00000222/GiB-second. For 17 cores and 64 GiB, the reserved compute rate is
$0.00120678/s, with an additional $3 CPU-build allowance. The launcher caps
reserved GPU time at three hours and estimated total cost at $30. Failed or
unreconciled calls retain their full reservation. [Modal pricing](https://modal.com/pricing).

The Volume hard-link failure was observed with SDK 1.5.3. The staging
regression test checks copied checkpoint bytes and excluded caches; the final
smoke passes a real checkpoint save and download. Remove staging only after
Modal supports atomic hard-link checkpoint creation and that direct-write
smoke passes. Physical CPU selection can be qualified if Modal exposes stable
host topology. No model, simulator, or checkpoint format was changed to work
around the Volume.

See `modal-profile-validation.md` for every validation result, launch failure,
skip, self-review fix and remaining fixture limit.
