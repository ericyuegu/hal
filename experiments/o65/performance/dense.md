# Dense O65 hot path

Initial development measurements use the O63 bootstrap30 proxy, BF16 decoder, seed 65,
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


## Step 5: one dense path and numerical qualification

The trainer now uses only dense tick/player tensors, packed shard frames and
fixed actor steps. The old decision objects, sequence adapters, grouped sampler,
per-decision evaluator, CPU wire conversion, chunk options and BLAKE2b action
sampler are removed. Training checkpoint v5 owns pool snapshot v3 and validates
the device counter key. The verifier uses the training graphs and execution mode. It checks restored
state exactly and the next update against measured control noise.

Review found stale reward padding in reused rollout storage. GAE masked those
cells, but summary metrics could read the prior rollout's values. Clearing the
reward buffer fixes the metrics; a focused two-rollout test covers it.

The complete precision gate initially rejected actor KL: control RMSE was
3.92e-6 and dense RMSE was 1.17e-4. Shape probes found both batch-shape effects
and a Torch 2.11.0 BF16 difference between gradient and no-gradient decoder
execution. Matching only the shapes reduced the error but still failed the
maximum-error gate: 1.00e-4 against an 8.39e-5 limit. Neither failure was
accepted, and the error floors were kept fixed.

The reference follows the actor's window order, padding and decision slots.
It scores the remaining slots in the same fixed shapes and encodes each window
once. One CUDA graph owns the complete reference pass. An initial attempt to
match decoder gradient metadata through a detached hidden tensor passed once,
but failed on a repeated complete run. That result was not accepted.

A separate read-only scoring copy now uses the actor's full autograd metadata.
The stored reference remains frozen. The scoring copy has no optimizer and no
backward runs. All outputs are detached. This Torch 2.11.0 boundary can be
removed when ordinary frozen forwards pass the complete precision gate.

The final gate uses commit `2912b618` and trainer SHA-256
`50269b65282245702246105f51b7baf8ecd45e4f6d6fd7abec6eeb6a4e975483`.
All seven quantities pass RMSE, maximum-error and relative-L2 checks against
1.5 times the control error plus fixed floors: 1e-6 for gradient elements,
1e-5 otherwise, and 1e-5 for relative L2. Three actor graph replays pass. The
FP32 reference uses FP32 for every model operation, TF32 off, and explicit
dense attention because FlashAttention requires reduced precision. Actor and
critic targets stay fixed across comparisons.

The probe uses 16 native matches and actual `[32,16]` actor steps on the RTX
3060. It adds linspace(-0.1,0.1) to action readout biases and initializes the
critic residual output with std 0.005, so KL and the residual critic are
nonzero. The control loads `aacdfee0` directly from Git. Rollout control scores
common actions through its recorded cached sampler, using its actual student
layout (30 slots) and active decision rows. Actor control uses the recorded
per-decision evaluator. Neither control implementation remains in the trainer.

| Quantity | Control RMSE | Dense RMSE | Control max | Dense max |
| --- | ---: | ---: | ---: | ---: |
| log_probs | 0.0056946 | 0.00462679 | 0.0963125 | 0.0879264 |
| kl | 3.92345e-06 | 3.20668e-06 | 4.9293e-05 | 4.11458e-05 |
| actor_gradients | 2.25278e-05 | 1.7069e-05 | 0.00561593 | 0.00351067 |
| rollout_log_probs | 0.00487061 | 0.00420264 | 0.110955 | 0.0759485 |
| rollout_kl | 0.000111086 | 0.000109499 | 0.00179926 | 0.00180453 |
| rollout_values | 0.494738 | 0.296849 | 2.72261 | 1.11101 |
| rollout_critic_gradients | 1.33887e-06 | 9.05466e-07 | 0.000104262 | 6.31101e-05 |

The table shows the first actor replay. All three replay errors and the five
complete attempts are in [dense-precision.json](dense-precision.json), together
with shape probes, source identities and raw input hashes. Raw inputs and logs
are under `runs/dense/precision-{first,canonical,ordered,student-layout,readonly-copy}`.
The earlier ordered-layout pass used all 32 control slots and was superseded
by the stricter recorded-layout gate.

Focused CPU checks pass 89 tests. The CUDA reset, queue ownership and resume
checks pass five tests before this reference correction. The required Dolphin
round-trip and session-cleanup integration run passes 8 tests, with 6 deselected
and no skips. The full unit handoff was interrupted after 119 passes to resolve
the precision failure; its final run is recorded below.

### Final local profiles

Source-verified profiles use `2912b618`, trainer SHA-256
`50269b65282245702246105f51b7baf8ecd45e4f6d6fd7abec6eeb6a4e975483`.
They include the qualified read-only reference copy. Raw outputs are
`runs/dense/step5q-final16` and `step5q-final64`. Earlier step-5 profiles remain
in the JSON as `step5`; they precede the final reference fix.

| Matches | Phase | Wall s | Main-thread CPU s | Eager launches, estimated | Graph launches, estimated | Kernel executions, estimated |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 16 | rollout | 2.329 | 1.060 | 85,638 | 384 | 608,634 |
| 16 | reference | 0.283 | 0.282 | 148 | 1 | 9,995 |
| 16 | preparation | 0.064 | 0.064 | 74 | 1 | 1,230 |
| 16 | critic | 0.018 | 0.013 | 601 | 0 | 613 |
| 16 | actor | 0.470 | 0.470 | 3,672 | 4 | 18,956 |
| 64 | rollout | 3.342 | 1.085 | 85,638 | 384 | 616,408 |
| 64 | reference | 0.506 | 0.506 | 148 | 1 | 17,863 |
| 64 | preparation | 0.138 | 0.138 | 74 | 1 | 1,230 |
| 64 | critic | 0.045 | 0.040 | 593 | 0 | 609 |
| 64 | actor | 0.940 | 0.940 | 7,344 | 8 | 37,912 |

The update takes 3.166 s at 16 matches and 4.971 s at 64 matches. Policy and
tick-state CPU time is 2.831 and 3.046 ms per tick; complete rollout main-thread
CPU time is 8.284 and 8.474 ms per tick. Counts stay flat except for work that
grows with windows and actor steps. The earlier columnar path took 5.433 s at
64 matches before the bootstrap follow-up, with 28 ms of rollout CPU per tick.

Using the measured PRO 6000 host costs, the **1,024-match projection** is
3.85 s rollout, about 1.28 s reference, about 3.80 s actor and below 0.5 s other
preparation. The projected complete update is about 9 s. Rollout uses the
64-match count; actor uses 119 fixed steps; reference scales nine observed
fixed batches to 129. Eager launches cost 14 us and remaining graph kernels
cost 5 us. The whole-preparation diagnostic projects about 0.05 s of host
cost. These are host-count projections, not measurements of GPU work or
measured 1,024-match times. The final Modal profile must test them.

### Resume contract and non-finite sampling fix

The resume verifier keeps training's execution mode. Report v5 checks the
restored model, both optimizers, scheduler, simulator and pool, sampler cursor,
counter key, progress and all RNG states exactly. It repeats the next update
twice as controls from the same boundary and compares a third resumed update.
For the model and each optimizer state family, RMS error may reach twice the
control spread plus eight dtype rounding units (minimum scale 1e-3). Maximum
and per-tensor errors are diagnostic. Integer state and scheduler progress
stay exact. Batch hashes are diagnostic because roundoff can change samples.
This replaces the failed bit-exact claim. Compiled decoder backward variation
is not a failure by itself.

The first native noise check restored every state hash, batch and scheduler
exactly, but failed the initial 1.5x per-tensor maximum-error rule in 70 of 426
tensors. Maximum parameter error was 2.69860e-5 against a control maximum of
1.77175e-5. That initial rule was an engineering choice, not the user's
precision criterion; it is retained in [dense-resume-initial.json](dense-resume-initial.json).
A second check used complete vectors with separate optimizer moment scales.
It passed before buffer coalescing, but a later repetition failed only the
maximum-error rule for Adam second moments; every RMS comparison passed.
Those results remain in dense-resume-prebatch.json and dense-resume-max.json.
The final rule uses RMS acceptance; isolated maxima remain diagnostic. Its
2x-spread rule was fixed before the final native check. It also
compares integer tensors without converting them to float, so low bits above
2^53 remain exact. Tests reject wrong restores, large update errors, mixed
optimizer scales, integer changes and overflow in error statistics.

The earlier 8-shard profile rejected a non-finite behavior log-probability on
its first rollout. A retry and a 12-update diagnostic did not reproduce it.
A forced upper-draw probe exposed the sampler defect: Torch 2.11.0's compiled
parallel prefix sum gives unequal cumulative values across zero-probability
classes. The probe found terminal-plateau drift up to 2.38419e-7. Counting CDF
entries then selected masked classes, whose recorded log-probability is -inf.
Of 8,192 BF16 rows at draw 1-2^-24, 2,434 selected masked classes; eager sampling
selected none. Enabling cast preservation still left 1,653 failures on those
rows. Setting only the normalized endpoint to one also failed because other
entries in the plateau differed.

The sampler now takes the first positive-probability crossing and assigns the
last positive class the exact endpoint. No non-finite score is replaced. The
compiled FP32 and BF16 regression failed before the fix and passes after it;
all 15 sampling tests pass. Dense GAE now rejects non-finite inputs and output
overflow. KL rejects invalid probabilities before support conversion. Actor
losses, KL and joint ratios are checked before the optimizer; gradient norm
checks reject invalid actor and critic gradients. The focused fault tests pass
44 cases, including all three non-finite values and GAE overflow. Additional
verifier tests admit measured noise, reject larger errors, and detect inexact
restoration.

The complete FP32-reference gate was rerun after the sampler fix at `694d75a1`
(trainer SHA-256 `467d65fb27bb9cc76c1bd64b0aa4424175b4d05e695362ddeadde083d1beaee6`).
All seven quantities pass, including all three actor replays, with the same
floors. Actor gradient RMSE is 1.66533e-5 versus control 2.24041e-5; the largest
of the three candidate RMSEs is 1.69370e-5. Raw evidence is in
`runs/dense/precision-valid-support` and the `valid_support_sampler` entry in
[dense-precision.json](dense-precision.json).

This local host exposes 12 logical CPUs (6 physical cores). The trainer rejects
repeated or unavailable shard CPUs. Sixteen local shards therefore cannot run;
the final local profiles use eight shards at 16 and 64 total matches. The Modal
profile retains 16 distinct shard CPUs. This hardware limit does not relax the
configuration check.


### Final eight-shard measurement and acceptance follow-up

The first complete eight-shard profile after the sampler fix projected rollout
above the 4 s target. The local trace showed repeated GPU acceptance work for
each simulator shard. Acceptance now concatenates each buffer's shard frames
and updates dense state once per buffer per frame. It preserves mixed cadence,
unsorted match IDs, first-frame terminations and remaining live lanes. CPU and
CUDA serial-versus-batched parity tests pass, including bootstrap and GAE.

| Matches | Rollout version | Wall s | Main-thread CPU s | Eager launches | Graph launches | Kernels |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 16 | per-shard acceptance | 3.080 | 1.743 | 145,030 | 384 | 668,026 |
| 64 | per-shard acceptance | 3.992 | 1.788 | 145,030 | 384 | 675,800 |
| 16 | per-buffer acceptance | 2.301 | 0.951 | 66,182 | 384 | 589,178 |
| 64 | per-buffer acceptance | 3.211 | 1.001 | 66,182 | 384 | 596,952 |

The source-verified final profiles use `c4e26c27`, trainer SHA-256
`178625eb8d5db0f04cbeb1d6664f30022b9760bcbb53e0d34b68f828e462af72`.
The subsequent resume diagnostic change does not change the training hot path.
Raw profiles are `runs/dense/step5-batched8s-final{16,64}`; the preceding
profiles are `step5-qualified8s-final{16,64}`. Both are retained in the JSON.
Counts below are estimates extrapolated from the profiler's sampled phases.

| Matches | Phase | Wall s | Main-thread CPU s | Eager launches | Graph launches | Kernels |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 16 | rollout | 2.301 | 0.951 | 66,182 | 384 | 589,178 |
| 16 | reference | 0.283 | 0.283 | 148 | 1 | 9,995 |
| 16 | preparation | 0.074 | 0.074 | 74 | 1 | 1,230 |
| 16 | critic | 0.019 | 0.014 | 617 | 0 | 629 |
| 16 | actor | 0.480 | 0.480 | 4,168 | 4 | 19,452 |
| 64 | rollout | 3.211 | 1.001 | 66,182 | 384 | 596,952 |
| 64 | reference | 0.506 | 0.506 | 148 | 1 | 17,863 |
| 64 | preparation | 0.153 | 0.153 | 74 | 1 | 1,230 |
| 64 | critic | 0.045 | 0.040 | 609 | 0 | 625 |
| 64 | actor | 0.958 | 0.958 | 8,336 | 8 | 38,904 |

Complete updates take 3.157 and 4.875 s. Policy and tick-state CPU time is
2.912 and 3.199 ms/tick. Complete rollout main-thread CPU time is 7.426 and
7.817 ms/tick. Rollout launches stay flat as matches grow from 16 to 64.
The critic also rejects a non-finite scalar loss before backward: a focused
regression demonstrates that finite squared elements can overflow on reduction
while their gradients remain finite. This check adds 16 launches per update.

Using 14 us per eager launch and 5 us per remaining graphed kernel, the final
**1,024-match host-count projection** is 3.580 s rollout and 4.009 s actor
(119 steps). Reference projection is about 1.27 s. The whole-preparation trace
projects 0.05 s of host work; measured local other preparation is 0.153 s.
The projected update is about 9 s. Sixteen shards add simulator-input host
work relative to the eight-shard local trace. These are projections, not PRO
6000 measurements.

The final precision rerun (`precision-batched`, source `c4e26c27`) passes all
seven quantities and all three actor graph repetitions. Actor gradient RMSE is
1.67232e-5 versus control 2.23592e-5; the largest candidate repetition RMSE is
1.74655e-5. Full per-quantity errors are retained as `batched_frames` in
[dense-precision.json](dense-precision.json). Neither thresholds nor floors
changed.

The native 64-match resume check at `490af376` passes. Every restored state
hash matches the boundary on all three repetitions. Next scheduler/progress
hashes match exactly. Model update RMS error is 1.57611e-7 versus control
spread 1.43917e-7; maximum error is 2.78503e-5 (diagnostic). Actor first-moment
RMS is 3.90981e-6 versus 3.37266e-6; second-moment RMS is 5.16790e-8 versus
5.95293e-8; Muon momentum RMS is 7.83312e-7 versus 7.20621e-7. Critic optimizer
errors are zero. See [dense-resume.json](dense-resume.json).

### Trajectory comparison

Three 64-match updates compare the recorded `aacdfee0` trainer with the final
dense trainer, seed 65, the same proxy, sampler and native simulator. Both arms
use the current shared model libraries and corrected categorical sampler.
The paired bootstrap resamples 64 logical matches, keeping all three updates
in each match, with 10,000 repetitions. These are descriptive 95% intervals;
no equivalence margin was specified. The 512-frame limit includes countdown,
so reported completed episode length is 635 frames.

| Quantity | Control mean | Dense mean | 95% CI for dense minus control |
| --- | ---: | ---: | --- |
| KL per planned frame | 4.01438e-5 | 5.60193e-5 | [1.31667e-5, 1.87136e-5] |
| Entropy | 0.678647 | 0.677336 | [-0.034458, 0.033325] |
| Student return | -0.696875 | -1.513292 | [-10.768838, 8.292590] |
| Return spread | 55.201119 | 44.414950 | [-25.737427, 3.977830] |
| Advantage mean | -0.387575 | 0.295786 | [-0.411113, 1.833391] |
| Advantage median | -1.057242 | -0.020970 | [0.016542, 2.105633] |
| Value-target mean | 0.453358 | 2.363468 | [-0.516575, 4.522853] |
| Value-target median | 0.011478 | 2.450914 | [0.019312, 4.964797] |
| Episode length | 635 | 635 | [0, 0] |
| Resets per match | 1 | 1 | [0, 0] |

KL and the two medians differ. This short, single-seed comparison does not
establish statistical equivalence. The precision gate independently passes.
The earlier pre-buffer run also shifted return spread and tails; it is retained
in dense-statistical-prebatch.json, rather than discarded. An initial entropy
report included R=2 decoder padding; the corrected report masks unexecuted
frames and has a regression test. All means, spreads, quantiles, source hashes
and invariants are in [dense-statistical.json](dense-statistical.json).


### Final local handoff

`uv run ruff format --check .` passes (393 files); `uv run ruff check .`
passes. The required `ty` command plus every new Python file passes with zero
diagnostics. An optional check of every changed test file reports 13 existing
diagnostics on unchanged lines of test_action_sequence_attention.py.

`uv run pytest -q -ra -m "not integration" --basetemp "$TMPDIR/n"` completes
with **2676 passed, 8 skipped, 85 deselected, 27 warnings**, in 646.91 s.
Two production netplay hardware checks are disabled. Six legacy v7 dataset
tests lack their recorded fixture; those skips count as validation failures
under AGENTS.md. No test fails. The preceding long-path run had three AF_UNIX
path failures and a stale exact-Muon expectation, all resolved in the final
run. Logs and historical failures are listed in the dense implementation report.


Final expanded integration command:

```
HAL_REQUIRE_INTEGRATION=1 uv run pytest -q -ra \
  tests/test_roundtrip.py tests/test_session_cleanup.py \
  tests/test_native_sim_integration.py tests/test_batched_self_play_integration.py \
  -m integration --basetemp "$TMPDIR/i"
```

It reports 13 passed, 12 failed, 6 deselected and 8 warnings in 434.37 s.
Every failure is the missing `zstandard` package required by MSL's .slpz reader.
After installing the local test prerequisite `zstandard==0.25.0`, rerunning
`tests/test_native_sim_integration.py -m integration` passes all 15 tests in
58.93 s, with no skips. Both batched/serial native policy comparisons and all
required Dolphin round-trip and cleanup cases passed in the expanded run.
No source or repository dependency changed for this environment correction.


## Single authorized Modal profile

The profile of published `1a920ac7` completed with exit code 0 on one RTX PRO
6000 Blackwell Server Edition at 1,024 matches, 16 shards and one group. It used
the existing pinned image and Volume, one warm update, one timing update and one
bounded trace update. There was no smoke run or retry. App
`ap-SLJhVzIi3mRTDf5w8YjlSM`, call `fc-01M5BRH5K2TW7MG5X9S5BX6QSD`.
The container has stopped. Raw results are under
`runs/modal-dense/dense-1024-1a920ac7-20261019` and the same run name on
Volume `hal-o65-profile`.

```
uv run python -m scripts.launch_o65_modal --mode profile --matches 1024 \
  --timeout-seconds 2400 --name dense-1024-1a920ac7-20261019 \
  --output runs/modal-dense
```

Trainer SHA-256 is
`302bc1159515840598bb57693694e7a8032f064844c4586b39368902f67946f5`.
The checkpoint is `65e7e43e5cbb18a365a005922ec1cfd026ad807c045eae13b331f487b4ec82e6`;
simulator SHA is `8fde568fb65ff4d0919bd71048808421c6b1a6c174d0d87cb83975696e3f75a8`.
The source hash and all 21 downloaded artifact hashes verify. The resolved
configuration, environment, artifact hashes, warm metrics, timing and launch
estimates are in [dense-modal.json](dense-modal.json). All reported metrics in
all three updates are finite; the training finite-value checks also run.

| Phase | Wall s | Main-thread CPU s | Eager launches, estimated | Graph launches, estimated | Kernels, estimated |
| --- | ---: | ---: | ---: | ---: | ---: |
| rollout | 6.669 | 1.300 | 77,146 | 384 | 600,034 |
| reference | 0.993 | 0.970 | 159 | 1 | 254,442 |
| other preparation | 0.346 | 0.350 | 74 | 1 | 1,234 |
| critic | 0.126 | 0.120 | 613 | 0 | 665 |
| actor | 3.260 | 3.210 | 117,453 | 119 | 585,718 |

The complete update takes **11.399 s**, versus the supplied 67 s baseline
(5.88x faster). This is one measured update, not a confidence interval or an
isolated attribution of every earlier change. Other preparation and actor work
meet their targets. Rollout misses 2–4 s, and the complete update misses
5–10 s. The 3.580 s rollout projection underestimated the measured complete
path. The 4.009 s actor projection is conservative against the measured 3.260 s.

Sampled launch estimates total 195,445 eager launches, 505 graph launches and
1,442,093 kernel executions. They exclude untraced gaps. Policy/tick-state main
CPU time is 3.906 ms/tick; full rollout main CPU time is 10.156 ms/tick. The
extra shard input work raises eager rollout counts from the local eight-shard
66,182 to 77,146 on 16 shards. Kernel counts remain close to the local 64-match
596,952 despite the larger batch.

The rollout trace estimates 3.22 s of device busy time against 6.67 s wall time;
main-thread CPU is 1.30 s. Non-boundary ticks include an estimated 4,032 event
synchronizations and 2,142 stream synchronizations. Native shard timers range
from 4.49 to 5.34 s, but include device-to-host inputs, futures and GPU-copy
fences, so they do not measure pure simulator compute. Over a 6.169 s host
sample, pinned simulator threads use 13.83 CPU seconds in aggregate (14.0% of
each of 16 cores on average); the 16 unpinned outer workers dominate the
34.83 other-thread CPU seconds. This suggests worker/transfer synchronization
and its CPU work remain material; it does not prove a simulator throughput
limit. This is the next rollout performance investigation. The local per-buffer
acceptance profile and this bounded Modal trace are retained; no extra cloud
run was launched after the miss.

Peak PyTorch allocation is 24.98 GB and reservation is 49.73 GB (decimal).
The cold warm update takes 1,283.886 s; the whole function takes 1,335.510 s
(0.371 GPU-hours). Image build steps take 168.75 s before GPU startup. The
2,400 s timeout reserved 2,760 s including startup/teardown. Completion reduces
the ledger to 1,695.510 s. At the launcher's recorded rates, with its retained
360 s margin and USD 3 build reserve, the conservative total allowance is
**USD 5.05**, below USD 10. This estimate is not a provider bill. The stopped
container list is empty.

One monitoring command initially lacked Click's `--` argument separator and
failed to parse `-n`; its corrected form succeeded. A final monitoring read
arrived after the container stopped and reported that the task had already
finished successfully. Neither command launched another GPU job.
