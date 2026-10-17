# O64 throughput investigation

The production recipe adds decoder work. Its estimated training FLOPs per
update are 340.76 TF, versus 255.93 TF for O59: an increase of 33.14%.
The percentile head accounts for only 0.17% of O59's estimated FLOPs.

## Matched B200 measurements

A single Modal B200 ran each case in a separate process. All used global batch
512, BF16, `reduce-overhead`, eager Muon, the same seed, pinned synthetic CPU
batches, the real device prefetcher and conditioning dropout. Each case warmed
for ten updates, then measured three windows of 50 updates. Synchronization
occurred at window boundaries. No dataset was downloaded. Compilation,
profiling and validation are outside the timing windows.

| Model | Windows/s | Seconds/update | Estimated BF16 MFU | Peak reserved GiB |
| --- | ---: | ---: | ---: | ---: |
| O59, 16 offsets | 1,325.0 | 0.3864 | 29.44% | 127.68 |
| O63, 16 offsets + percentile head | 1,305.8 | 0.3921 | 29.06% | 126.67 |
| O64, 26 offsets + percentile head | 958.1 | 0.5344 | 28.34% | 175.85 |

MFU is the experiment's approximate model FLOPs divided by elapsed time and
2.25 PFLOPS, the B200 dense BF16 peak. This is an estimate, not an instruction
counter measurement. NVIDIA's [HGX specifications](https://www.nvidia.com/en-us/data-center/hgx/)
list 36 PFLOPS BF16 for eight B200 GPUs with sparsity; dense performance is
half of that. The estimator does not count all attention and elementwise work.

The linked historical O59 run, `vywk3cih`, has median update time 0.39459 s over
1,312 logged timing windows, or 1,297.5 windows/s. The matched O59 reproduction
is consistent with that reference. See [059_reference_run.json](059_reference_run.json).

The benchmark's overall status is **failed**: the later O64 `max-autotune`
case ran out of memory during CUDA graph setup. The three earlier cases above
completed. The planned final O59 repeat did not run, so this initial comparison
lacks its intended end-of-job timing drift check. Max-autotune is not promoted.
The complete failure is retained in [matched_profile_results.json](matched_profile_results.json).

## Profile and memory

One O64 update was profiled after timing, without shape recording. Among the
retained raw GPU kernels and copies, approximately 49% of time is in matrix
multiplication, 25% in other fused pointwise/reduction kernels, 10% in kernels
that include normalization, and 13% in attention. This is a name-based grouping
of the stored top 100 profiler events. Individual kernels combine operations;
this is not a hardware roofline measurement or a complete phase breakdown.
Parent compiled-graph, optimizer and CPU-operation events are excluded to avoid
double counting. The classification and timings are in [analysis.json](analysis.json).

Rotary/layout kernel names account for about 62 ms in that same subset. The
largest repeated rotary/layout backward kernel alone takes 21.4 ms. These are
candidates for reducing activation traffic. The profile does not establish that
short temporal attention is the dominant bottleneck. The trunk and temporal
functions already use CUDA graphs, and Inductor already fuses many pointwise
operations. A full-model megakernel has not been implemented or benchmarked.

The first prefetch qualification passed eight training updates but failed on
validation batch 128 with resident training graphs. A single 32-row synthetic
validation batch passed in the matched benchmark, with peak reserved memory
176.76 GiB on a 178.34 GiB GPU. A later full-cohort attempt at batch 32 failed
in eager history attention: it requested 1.62 GiB with 956.62 MiB free. That
failed full-cohort attempt is retained in
[../validation32_cohort_preflight.json](../validation32_cohort_preflight.json).
Training batch remains 512.

## Execution candidates

A second bounded benchmark compares the pinned O64 baseline with static device
caches, an explicit rotary input-gradient formula, and Flash SDPA in the
short temporal block. It ends with a repeated baseline. These are execution
candidates; the offsets, coefficients, effective batch, data and objective stay
fixed. Actor prefill and step methods remain inherited and unchanged.

The preliminary harness compares its first two optimizer updates at step
counters 4,096 and 4,097. The scheduler is fresh, so these use the first two
warmup learning rates. It captures post-clipping gradients and absolute model
weights, but does not compare preclip gradients or weight-update deltas.
Conditioner projections start at zero. Its RNG snapshots cover Torch, prefix
and labeler generators, but omit the return masker and Python/NumPy state.
These checks are preliminary and cannot qualify a numerical implementation
change. The timing windows remain useful. Rope and Flash variants also include
the cache candidate, so their incremental speed must be compared with cache.

Any promising numerical candidate needs a separate qualification from a shared
warmed state, with production learning rates, nonzero modulation, preclip
gradients, explicit update deltas, and baseline/repeat numerical noise. A
candidate can advance only after that qualification and when all three timing
windows beat all windows from the baseline and repeated baseline. Numerical or
memory failure rejects that candidate. Sources and launch identities are
retained alongside the results.

The timing benchmark finished. No candidate passed the speed rule:

| Path | Window 1 | Window 2 | Window 3 | Decision |
| --- | ---: | ---: | ---: | --- |
| Baseline | 965.7 | 963.3 | 962.5 | Retained |
| Static caches | 970.6 | 967.0 | 967.6 | Overlaps repeated baseline |
| Caches + explicit rotary backward | 963.9 | 963.5 | 962.0 | Slower than caches |
| Caches + Flash SDPA | — | — | — | CUDA OOM in backward |
| Repeated baseline | 967.8 | 964.5 | 961.9 | Confirms control range |

Flash requested an 8 GiB backward workspace with only 6.55 GiB available.
The separate stronger numerical qualification was not run because no candidate
passed timing. All cache and custom-kernel changes were removed from the
maintained training program. The retained execution change is the final-prefix
validation optimization described below; validation batch is restored to 128.
Rejected sources, source hashes, failures and results remain in this directory. See [candidate_decision.json](candidate_decision.json) and
[candidate_results.json](candidate_results.json).

These measurements support a mixed bottleneck: matrix work plus activation
movement and reductions. They do not show a launch-bound step or establish a
hardware bandwidth roofline. A future custom normalization/modulation backward
kernel can be tested separately. FP8 or reducing supervised work would change
numerical behavior or the training procedure and has not been applied.

## Final-prefix validation

The original validation decoded all 128 suffix prefixes but scored only the last.
The final implementation decodes that one prefix, with its original absolute
position, observed action, target actions, condition and availability. It keeps
the full trunk memory for history attention. Full-suffix value and percentile
computation, labeler draw shape and order, the cohort, and the original batch 128
are preserved. Training and actor paths are unchanged.

Seven focused tests passed. They compare the new path with the original
full-prefix NLL calculation across all three arms on CPU and CUDA, with nonzero
conditioning, padding and unavailable labels; they also check batch aggregation.
The complete B200 qualification passed: ten training updates, all 2,048
synthetic validation samples at batch 128 in 7.623 seconds, and eight further training
updates. It used production learning rates and real prefetch/dropout. All
losses and gradient norms remained finite. Peak allocated/reserved memory was
166.60/175.86 GiB, within the training peak. No dataset was downloaded. The
qualified experiment source SHA-256 matches the committed training source.
See [../full_validation_preflight.json](../full_validation_preflight.json).
