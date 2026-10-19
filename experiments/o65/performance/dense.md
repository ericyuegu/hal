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


Source-verified profiles use commit `f7bf145b`, trainer SHA-256
`e05aabb8bd7421ba873186639b98729dbad9a38750fefc2366fc5e701b4ae08d`. The runner checked the source against Git before and after
each job. These replace the preliminary profiles above. Raw outputs are in
`runs/dense/step2-final16` and `runs/dense/step2-final64`.

| Matches | Phase | Wall s | Main-thread CPU s | Eager launches, estimated | Graph launches, estimated | Kernel executions, estimated |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 16 | rollout | 2.368 | 2.164 | 80,898 | 256 | 436,464 |
| 16 | reference | 0.314 | 0.314 | 18,520 | 0 | 19,100 |
| 16 | preparation | 0.101 | 0.100 | 4,790 | 0 | 4,906 |
| 16 | critic | 0.019 | 0.014 | 601 | 0 | 613 |
| 16 | actor | 0.505 | 0.276 | 30,256 | 0 | 31,196 |
| 64 | rollout | 3.284 | 2.998 | 80,850 | 256 | 434,564 |
| 64 | reference | 1.200 | 1.199 | 70,073 | 0 | 72,271 |
| 64 | preparation | 0.264 | 0.264 | 15,514 | 0 | 15,674 |
| 64 | critic | 0.045 | 0.040 | 593 | 0 | 609 |
| 64 | actor | 1.790 | 0.872 | 108,960 | 0 | 112,584 |

16 matches: GPU-tick CPU 2.220 ms; rollout projection 2.910 s; update 3.309 s.

64 matches: GPU-tick CPU 2.859 ms; rollout projection 2.900 s; update 6.587 s.

The projection uses 14 us per eager launch and 5 us per remaining graph
kernel. It is a host-cost projection with sampled coverage, not a measured
PRO 6000 time. The final layout cache avoids per-tick validation loops and
reuses cache rows after anchor assignments change at a block boundary.


## Step 3: two simulator buffers

Each buffer owns a CUDA stream. The hot tick samples A, starts A's shard
workers, then samples B and starts B's workers. Device transfers and accepted
observations stay on their buffer's stream. The default stream joins both
before the next tick. Queue snapshots precede both policies, so critic inputs
exclude either player's new action. The two stepped frames precede the first
new action at D+1 >= 9. Cold captures run before worker CUDA calls; they use
the same tick operation and serialize shared-ring snapshot restoration.

Source-verified profiles use commit `382c9300`, trainer SHA-256
`893dbe6956cd4af33408eaba2e054e7888ace7ae3d46e3cc25d37daa8466caf0`.
Raw outputs are `runs/dense/step3-final16` and `runs/dense/step3-final64`.

| Matches | Phase | Wall s | Main-thread CPU s | Eager launches, estimated | Graph launches, estimated | Kernel executions, estimated |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 16 | rollout | 2.875 | 2.660 | 86,534 | 384 | 611,048 |
| 16 | reference | 0.269 | 0.269 | 18,520 | 0 | 19,100 |
| 16 | preparation | 0.097 | 0.096 | 4,790 | 0 | 4,906 |
| 16 | critic | 0.017 | 0.013 | 601 | 0 | 613 |
| 16 | actor | 0.473 | 0.265 | 30,256 | 0 | 31,196 |
| 64 | rollout | 3.704 | 3.556 | 86,534 | 384 | 618,822 |
| 64 | reference | 1.025 | 1.024 | 70,073 | 0 | 72,271 |
| 64 | preparation | 0.257 | 0.257 | 15,514 | 0 | 15,674 |
| 64 | critic | 0.044 | 0.041 | 593 | 0 | 609 |
| 64 | actor | 1.696 | 0.851 | 108,960 | 0 | 112,584 |

Policy and tick-state CPU time is 3.084 ms per tick at 16 matches and 3.345 ms
at 64. Complete rollout main-thread CPU time divided by 128 ticks is 20.777
and 27.784 ms; this includes shard acceptance, resets and setup. Update wall
time is 3.733 and 6.731 s. Splitting the GPU batch costs time at these local
sizes: rollout was 2.368 and 3.284 s before overlap. The scheduling test proves
that A can step while B samples; these measurements do not show a local speedup.

The sampled rollout launch counts are flat. Applying the supplied PRO 6000
host costs gives a **projection** of 3.834 to 3.873 s at 1,024 matches, close
to the 4 s target limit. Coverage and overlap caveats from step 2 apply.
Reference and actor still miss their projected targets; bounded local traces
identify their eager replay calls. Window replay replaces them next.

Validation: CPU-focused checks passed 94 tests, with 13 diagnostic skips and
19 deselections. The first two-stream CUDA run passed scheduling/reset checks
but failed exact resume on an implicit BF16-to-FP32 indexed write. Making the
conversion explicit fixed it. The retry passed both tests (19 deselections,
15 dependency warnings), including odd-frame timeouts and deterministic exact
resume. Both source-verified native profiles completed.


## Step 4: window reference and actor graphs

Reference replay packs each block/player window with 64 fixed slots. Actor
steps gather 16 selected slots per window into 32 padded windows. The trunk
stays eager inside capture; only the shared-history decoder is compiled, with
BF16 cast preservation. The replay stream's FP32-reference qualification
rejected whole-trunk compilation. Critic fitting uses rollout features;
bootstrap graphs compute queue states and values without action probabilities.
Actor diagnostics reuse its captured forward output. No student scoring pass
runs. The graph objective now accepts the configured PPO clip and preserves an
infinite KL when current mass appears outside reference support.

Source-verified profiles use `f5cf13b0`, trainer SHA-256
`f97ce833dae4d6df6fe4d46eb8e4cfe5e6e03e6af7d022a9eef91a03aecc7a3f`.
Raw outputs are `runs/dense/step4-final16` and `runs/dense/step4-final64`.

| Matches | Phase | Wall s | Main-thread CPU s | Eager launches, estimated | Graph launches, estimated | Kernel executions, estimated |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 16 | rollout | 2.879 | 2.662 | 86,534 | 384 | 611,048 |
| 16 | reference | 0.102 | 0.102 | 16 | 1 | 1,173 |
| 16 | preparation | 0.124 | 0.124 | 74 | 1 | 1,230 |
| 16 | critic | 0.017 | 0.013 | 601 | 0 | 613 |
| 16 | actor | 0.472 | 0.472 | 3,636 | 4 | 18,920 |
| 64 | rollout | 3.685 | 3.521 | 86,534 | 384 | 618,822 |
| 64 | reference | 0.390 | 0.390 | 16 | 1 | 1,174 |
| 64 | preparation | 0.380 | 0.379 | 74 | 1 | 1,228 |
| 64 | critic | 0.044 | 0.041 | 593 | 0 | 609 |
| 64 | actor | 0.931 | 0.931 | 7,272 | 8 | 37,840 |

At 64 matches, reference time falls from 1.025 to 0.390 s and actor time from
1.696 to 0.931 s. Update time falls from 6.731 to 5.433 s. Policy/tick-state
CPU time is 3.146 and 3.292 ms per tick at 16 and 64 matches. Complete rollout
CPU time divided by 128 ticks is 20.796 and 27.509 ms. Reference launch counts
stay flat. Actor counts grow with its four versus eight fixed-shape steps.

The actor count **projection** at 1,024 matches and 119 steps is about 3.79 s,
using 14 us per eager launch and 5 us per remaining graph kernel. Rollout's
projection remains 3.87 s. These omit GPU work growth and untraced setup; they
are not measured PRO 6000 times. Preparation rises from 0.257 to 0.380 s at
64 matches, which needs diagnosis before the next step.

A separate warmed whole-preparation trace at 64 matches captures 5,818 kernel
executions, 2,350 eager launches and three graph launches. Its host-cost
projection is 0.050 s, but GPU work dominates the local measurement. Graph
kernel time is 0.387 s for reference, 0.322 s for bootstrap values, and
0.022 s for the context-gap subset. The bootstrap graph processes all four
reserved rows per player although only the final row is usually used. The
next local change evaluates only the required boundary roles at fixed shapes.
The raw trace and CPU profile are in `runs/dense/step4-preparation`; the JSON
evidence includes their summaries. The CPU profile's waits appear under
`torch.equal` and GAE's `torch.where`, so their apparent time includes preceding
GPU graph work.

Validation: focused CPU checks pass 34 tests, then the queue/value checks pass
28. The broad CPU loop run passes 78 with one stale fixture-width failure and
one unset-proxy-path skip; the width retry passes. Five real-model/Muon resume
checks pass. CUDA's two-stream update and resume pass; a subsequent helper test
hits Torch's shared eight-variant compilation limit. Giving the compiler tests
an explicit preparation scope fixes it: all three pass on retry. The first
native profile finishes its updates but its coverage guard still requires old
chunk sites; updating that guard passes all 15 profiler tests. Both canonical
profiles then finish. A separate diagnostic harness first has a Python syntax
error before execution; the corrected whole-preparation trace completes.


The bootstrap follow-up uses `7012de66`, trainer SHA-256
`debc1bbec7512bc252dd12667e1b45ffabb1d79745128f1c41216f21f1a05d4c`.
It evaluates one fixed player-sized batch for each occupied timeout block or
final boundary and skips empty roles. No model or objective changes. Raw
source-verified outputs are `runs/dense/step4b-final16` and `step4b-final64`.

| Matches | Phase | Wall s | Main-thread CPU s | Eager launches, estimated | Graph launches, estimated | Kernel executions, estimated |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 16 | rollout | 2.853 | 2.642 | 86,534 | 384 | 611,048 |
| 16 | reference | 0.102 | 0.102 | 16 | 1 | 1,173 |
| 16 | preparation | 0.063 | 0.063 | 74 | 1 | 1,230 |
| 16 | critic | 0.017 | 0.013 | 601 | 0 | 613 |
| 16 | actor | 0.473 | 0.473 | 3,636 | 4 | 18,920 |
| 64 | rollout | 3.777 | 3.623 | 86,534 | 384 | 618,822 |
| 64 | reference | 0.394 | 0.394 | 16 | 1 | 1,174 |
| 64 | preparation | 0.139 | 0.139 | 74 | 1 | 1,230 |
| 64 | critic | 0.044 | 0.041 | 593 | 0 | 609 |
| 64 | actor | 0.942 | 0.941 | 7,272 | 8 | 37,840 |

Other preparation falls to 0.063 and 0.139 s, from 0.124 and 0.380 s. Counts
stay flat; the smaller bootstrap batch removes GPU work. Updates take 3.509
and 5.298 s. Policy/tick-state CPU time is 3.077 and 3.443 ms per tick;
complete rollout CPU divided by 128 ticks is 20.643 and 28.304 ms. The count
projections remain about 3.87 s for rollout and 3.79 s for actor at 1,024.
The full-preparation trace's host projection remains 0.050 s; GPU work still
needs the final PRO 6000 measurement. Focused checks pass four tests with one
CUDA-only diagnostic skip. Both canonical follow-up profiles complete.
