# Dense O65 hot path

Development measurements use the O63 bootstrap30 proxy, BF16 decoder, seed 65,
D=8/R=4 and D=10/R=2, one group, four shards on CPUs 0–3, and 256 rollout
frames. Each profile warms one update, times one update, then traces bounded
sites in a separate update. Local runs hold the shared GPU lock and use
MemoryMax=10G, OMP_NUM_THREADS=2, and at most 64 total matches. They measure
systems behavior, not policy quality. Raw traces remain under
`runs/dense/step1-profile{16,64}`; input hashes, source hashes and summaries
are in [dense-measurements.json](dense-measurements.json).

## Step 1: stored behavior and device selection

Treatment: PPO uses the rollout behavior logs as pi_old; rollout also stores
student distributions, critic features and values. Decision subsampling and
placement use tensor counter draws and stable device sorts. Control: aacdfee0.
Checkpoint, simulator, sampler, reward, KL objective and optimizer families
are invariant. The removed truncated importance correction is an explicit
scientific change approved in the implementation brief.

| Matches | Phase | Wall s | Main-thread CPU s | Launches, estimated | Kernel executions, estimated |
| ---: | --- | ---: | ---: | ---: | ---: |
| 16 | rollout | 4.093 | 3.881 | 72,517 | 1,253,395 |
| 16 | reference | 0.309 | 0.309 | 18,520 | 19,100 |
| 16 | preparation | 0.102 | 0.102 | 4,790 | 4,906 |
| 16 | critic | 0.017 | 0.013 | 605 | 617 |
| 16 | actor | 0.499 | 0.274 | 30,256 | 31,196 |
| 64 | rollout | 5.372 | 5.083 | 72,660 | 1,251,558 |
| 64 | reference | 1.200 | 1.199 | 70,073 | 72,271 |
| 64 | preparation | 0.292 | 0.292 | 15,514 | 15,674 |
| 64 | critic | 0.045 | 0.041 | 597 | 613 |
| 64 | actor | 1.804 | 0.880 | 108,960 | 112,584 |

Preparation excludes reference time. Launch and execution counts cover sampled
sites only; setup and diagnostic gaps remain untraced. They are scaled by
call count, not whole-phase counts. GPU tick main-thread CPU time is 25.712 ms
at 16 matches and 34.692 ms at 64. No student scoring pass runs.

Projection to 1,024 matches: scaling the 64-match reference and actor calls by
16 gives about 1.12M and 1.74M eager launches. At the brief's measured PRO 6000
host costs of 14 us per eager launch, launch cost alone is about 15.7 s and
24.4 s. This misses the targets and motivates window replay. Rollout counts
are already almost flat: about 72.7k launches and 1.25M kernels. Charging
14 us per launch and 5 us for the remaining graph kernels gives about 6.9 s
before other work, also above target. These are count-based projections,
not measured PRO 6000 times or bounds on overlapping execution. Local bounded
traces identify eager metadata/storage writes and many graph-internal kernels;
the next step moves those writes and scheduling into the tick graph.

Validation: initial focused run: 152 passed, 6 failed, 1 skipped (the six
failures were stale KL callers). Focused retry: all six passed. The skip is
the test requiring HAL_O65_PROXY_CHECKPOINT to identify the local artifact;
full handoff and final qualification are recorded after the complete rewrite.
Two profile launches failed before rollout because melee_sim was absent from
PYTHONPATH; setting it to the MSL source root fixed the environment. Both
16- and 64-match profile runs then completed.


## Step 2: dense ticks and storage

The rollout uses fixed tick/player row IDs and device episode settings. Its
compiled tick graph appends the trunk, samples, updates queues, computes values,
and writes behavior distributions and critic features. Dense GAE skips empty
ticks without changing the decision discount. Both CPU tests and CUDA use the
same tick operation. The tensor RNG now uses full 63-bit counters and a
resumable seed-derived key; the interrupted 32-bit counter aliases are removed.

Preliminary local profiles give the following results. These runs precede the
final layout cache fix; source-verified reruns follow before the overlap step.

| Matches | Rollout wall s | Rollout main CPU s | GPU-tick main CPU ms | Launches, estimated | Kernel executions, estimated |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 2.400 | 2.196 | 2.281 | 81,154 | 436,464 |
| 64 | 3.351 | 3.050 | 2.956 | 81,106 | 434,564 |

Counts are almost flat with match count. Using 14 us per eager launch and
5 us per graph kernel gives a **projection** of about 2.9 s for rollout at
1,024 matches, before untraced work and overlap. Reference and actor remain
eager and scale with data; they await the replay integration. Other preparation
is 0.101 s at 16 matches and 0.270 s at 64. Its bootstrap replay still uses the
old evaluator and misses the projected 0.5 s target at 1,024; the local bounded
traces identify those chunks before further work.

Focused checks found and fixed missing queue validation, a bootstrap after a
final reset, padded writes into capacity-2 queues, stale borrowed evaluation
records, and a Torch compiler rejection of dataclass-class inspection. The
GPU graph test passes on CPU and CUDA. The current CPU-focused run passes
127 tests; CUDA-only cases and the unset proxy-path environment were skipped
in that diagnostic run. Full validation remains required.
