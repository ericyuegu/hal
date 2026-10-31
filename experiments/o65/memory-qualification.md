# O65 memory qualification

The predicted reserved-memory budget is **84.5 GiB for `matchup`**, leaving
**10.5 GiB free** on a 95.0 GiB card. Local update times change by less than
1%. The arm settings and inference batch shapes are unchanged.

Local study on an RTX 3060 (12 GiB), starting from
`89924ff5b8c6df2066749243490d32c1d497ecd8`. The 95 GiB qualification must
run on the target card. No Modal or GCE job was launched during this study.

## Treatment and control

The treatment shares CUDA graph scratch across calls that execute in sequence.
The two rollout streams each have one pool. Bootstrap values, diagnostics,
reference scoring, and actor backward share a third pool on the training
stream. All operations compile and warm up before any shared capture. This
keeps persistent compiler inputs outside the shared scratch pool. Capture
preserves the preparation grad mode and autocast settings.

Groups collect in sequence, so they use the same two rollout streams. This
limits the number of stream-specific allocator caches. The two buffers of an
active group can still overlap. There is no cache flush between updates.

Reference scoring releases each decoder autograd graph before it starts the
next slot chunk. It still computes every padded chunk, with the same shapes.
It copies only real windows into the retained scores. Invalid real windows
are retained; only synthetic padding is omitted.

The optimizer, model, loss, returns, random sampling, lane layout, KV rows,
actor input shapes, and placement code are unchanged. Every native run uses
`pilot.training_command(..., arm=...)`, with only the local scale, output,
CPU list, and run length changed. The production configuration remains four
groups of 1,024 matches, 16 shards of 64, 512 windows per actor step, and four
actor microbatches. The `matchup` arm retains its PFSP flags and scalar
residual critic (width 128, depth zero).

The unmodified training graphs cannot coexist on the 12 GiB card, even at
the minimum useful simulator scale. Therefore the native control shares
**only the training graph pool**, with the decoder graph lifetime correction.
It retains the original rollout graphs, streams, inference batches, and
reference padding. Original graph components are measured in separate
processes. Local before/after trainer numbers below refer to this control,
not to an executable full trainer at `89924ff5`.

## Where the memory goes

Each padded inference row uses 3,450,408 bytes of KV and block history storage
(3.2906 MiB). Both layouts instantiate the same student and four anchors.
The extra memory comes from rows, not additional policy weights.

| Full-scale storage | `old` | `matchup` |
|---|---:|---:|
| Student KV rows | 8,192 | 8,192 |
| Anchor KV rows | 1,280 | 6,144 |
| Total KV rows | 9,472 | 14,336 |
| KV storage, GiB | 30.438 | 46.068 |

The 4,864 extra rows account for **15.630 GiB**. The supplied rollout
allocation difference is 15.7 GiB. The small remaining difference includes
static tick inputs and allocator rounding. Dense rollout tables, stores, and
policy weights do not explain the layout difference.

Original private graph pools reserve about 6.05 GiB (`old`) or 9.11 GiB
(`matchup`) for rollout. These estimates use measured graph sizes against
batch rows for all three tick variants. Reference scoring reserves 8.148 GiB
at its unchanged 128-window microbatch; actor backward reserves 6.418 GiB.
The bootstrap value graph scales to about 6.382 GiB at 2,048 player rows.
Together with the diagnostic value graph, the original private-pool estimate
is about 27.1 / 30.1 GiB. The reported OOM identified 28.72 GiB in private
pools, within about 1.4 GiB of the `matchup` component estimate.

These inactive graph workspaces explain most of the reserved/allocated gap.
The remainder includes general allocator caching and split blocks. For an
eager, production-sized buffer (512 matches), two repeated prefill passes
leave 2.26 / 2.24 GiB cached (`old` / `matchup`), with only 0.12 / 0.11 GiB
in split blocks. This measurement points to reusable cached blocks as the
main remaining cost. It does not support attributing the whole gap to
fragmentation.

Allocated and reserved phase peaks need not occur at the same instant.
Their difference is not an exact allocator partition. Also, allocator
pressure releases cached segments: it affects the supplied 95 GiB runs and
some 12 GiB controls. A direct linear regression of reserved peaks across
those conditions gives invalid extrapolations. The prediction uses separate
storage, graph-pool, and general-allocator terms.

The production training shapes have also been captured together locally:
2,048 bootstrap rows, 32 diagnostic rows, 128 actor windows with 16 selected
slots, and 128 reference windows with 64 scored slots. Their shared pool is
**6.496 GiB**. A separate full-size value capture is 6.404 GiB, only 0.023 GiB
above the value fit. Reference scratch falls from 8.148 to 6.084 GiB.

The isolated CUDA graph timings are 349.152 to 349.169 ms for actor backward
(+0.005%), and 207.231 to 206.739 ms for reference scoring (-0.238%). Each
number is the mean of five groups of ten replays after warmup.

## Measurement and model

All output, temporary files, pytest directories, and compiler caches are
under `/home/ericgu/data/scratch/o65-tmp`. The common Inductor and Triton
caches are reused. The trainer environment is Python 3.14.4,
Torch 2.11.0+cu130, CUDA 13.0, and NumPy 2.1.3. OMP and MKL use two threads;
Inductor uses one compile thread. Native jobs run one at a time.

The native design uses both layouts, groups 1 and 2, and 8, 12, and 16
matches per shard. CPU count equals shard count. The one-group, 8-match
cases have four shards; the other cases have two. Four matches per shard
cannot supply the arm's required self-play trace lanes. The requested arm
flags are preserved.

Persistent storage is fitted against total matches, total anchor lanes, and
groups. Value graph input storage uses its exact row count: `2*M` plus
`min(32, 2*M)`, with identical shapes counted once. Transient phase work is
fitted against one active group's matches and anchor lanes. The reference
fit separates padded input and output storage from the fixed local actor
step floor. One captured input window uses 205,912 bytes; its row and
metadata buffers add 9,728 bytes. One output window uses 498,176 bytes.

The fit is checked against the supplied full-scale allocations before it is
used for the treatment. The numerical results and fit coefficients are in
the companion measurement record. Reserved predictions must include the
pool and cache terms, rather than just extend a small-card reserved slope.

The allocated back-test is below. Each cell is observed / fitted / error,
in GiB. The largest absolute error is 0.85 GiB.

| Phase | `old` | `matchup` |
|---|---:|---:|
| Warmup rollout | 45.10 / 44.35 / -0.75 | 60.80 / 60.74 / -0.06 |
| Warmup preparation | 48.00 / 47.25 / -0.75 | 62.70 / 62.59 / -0.11 |
| Warmup reference | 46.30 / 45.45 / -0.85 | 61.40 / 61.32 / -0.08 |
| Warmup critic | 51.10 / 50.44 / -0.66 | 64.00 / 64.59 / +0.59 |
| PPO preparation | 53.80 / 53.95 / +0.15 | — |
| PPO actor | 50.40 / 50.65 / +0.25 | — |



<!-- RESULTS START -->
## Local results

All values are GiB. A/R denotes peak allocated / reserved over the timed
updates; startup is recorded separately in the JSON. Times are mean seconds
per update. Warmup uses updates 3–32; steady PPO excludes its first update.
The control is the training-pool-only control described above. OOM peaks
include the incomplete failed attempt and are excluded from the fit.

| Arm | Groups / shards / lanes | A/R before → after | Warmup s before → after | PPO s before → after |
|---|---|---|---|---|
| old | 1 / 4 / 8 | 4.02/10.99 → 2.14/8.99 | 5.03 → 5.03 | 14.46 → 14.39 |
| old | 2 / 2 / 8 | 3.77/10.68 OOM → 2.04/9.13 | 4.54 → 4.56 | OOM → 13.90 |
| old | 1 / 2 / 12 | 3.85/10.99 → 1.96/8.80 | 4.72 → 4.68 | 14.08 → 14.08 |
| old | 2 / 2 / 12 | 3.20/10.52 OOM → 2.24/9.13 | 4.69 → 4.65 | OOM → 14.11 |
| old | 1 / 2 / 16 | 3.97/10.99 → 2.09/8.91 | 4.95 → 4.92 | 14.31 → 14.32 |
| old | 2 / 2 / 16 | 3.38/10.63 OOM → 2.44/9.35 | 4.96 → 4.95 | OOM → 14.30 |
| matchup | 1 / 4 / 8 | 4.02/10.99 → 2.14/8.99 | 5.05 → 5.03 | 14.47 → 14.38 |
| matchup | 2 / 2 / 8 | 3.77/10.68 OOM → 2.04/9.13 | 4.54 → 4.55 | OOM → 13.83 |
| matchup | 1 / 2 / 12 | excluded → 2.00/8.82 | excluded → 4.73 | excluded → 14.08 |
| matchup | 2 / 2 / 12 | 3.32/10.61 OOM → 2.35/9.14 | 4.81 → 4.80 | OOM → 14.15 |
| matchup | 1 / 2 / 16 | 4.07/10.99 → 2.17/9.03 | 5.13 → 5.14 | 14.59 → 14.51 |
| matchup | 2 / 2 / 16 | 3.59/10.65 OOM → 2.64/9.49 | 5.13 → 5.09 | OOM → 14.43 |

Paired warmup changes range from -0.8% to +0.4%.
Paired steady-PPO changes range from -0.6% to +0.1%.
The two-group controls completed warmup and then ran out of memory during
reference scoring. Their incomplete PPO phase is excluded from the fit.
Every treatment run completes PPO, with zero post-startup captures and compiles.

## Full-scale prediction

These predictions use four groups, 16 shards, and 64 lanes per shard.
Each cell is allocated / reserved / reserved with allowance, in GiB.
The last value includes the largest permitted reference-padding input.

| Phase | `old` | `matchup` |
|---|---:|---:|
| Warmup rollout | 44.0 / 55.4 / 61.0 | 60.4 / 71.8 / 77.5 |
| Warmup preparation | 46.9 / 58.4 / 63.5 | 62.3 / 73.7 / 78.9 |
| Warmup reference | 45.2 / 56.6 / 61.7 | 61.0 / 72.5 / 77.6 |
| Warmup critic | 50.1 / 62.2 / 67.0 | 64.3 / 76.4 / 81.2 |
| PPO rollout | 44.0 / 60.8 / 66.5 | 60.4 / 77.2 / 83.0 |
| PPO preparation | 52.4 / 63.8 / 68.7 | 68.1 / 79.5 / 84.5 |
| PPO reference | 52.4 / 63.8 / 68.7 | 68.1 / 79.5 / 84.5 |
| PPO critic | 51.3 / 63.5 / 67.9 | 64.9 / 77.1 / 81.5 |
| PPO actor | 49.4 / 60.8 / 65.6 | 64.5 / 76.0 / 80.9 |

Optimization is bounded by its critic and actor child phases. Preparation
includes reference scoring. Startup and head-start use separate component
budgets:

| Additional phase | `old` reserved budget | `matchup` reserved budget |
|---|---:|---:|
| Startup | 66.9 | 82.6 |
| Head-start | 59.1 | 74.8 |

The largest budgets are **68.7 GiB for `old` and 84.5 GiB for `matchup`**.
They leave **26.3 / 10.5 GiB** free on a 95.0 GiB card.

The reserved model adds measured private-pool capacity to fitted allocation
and general-allocator overhead. Rollout pools scale with one buffer’s largest
student batch, independently of group count. General overhead is fitted
against matches per active group, with a floor of two full-buffer prefill
caches. The allowance adds the maximum allocated back-test error, local
cache-fit residual amplified by the row scale, and 1.5 GiB for the observed
original private-pool extrapolation error. This is an engineering allowance,
not a statistical confidence interval. An additional input allowance covers
up to 6,144 padded reference windows while retaining 4,096 real windows.

The original reserved peaks fall inside the component ranges in the JSON.
Those ranges are wide: for example, `old` warmup rollout permits about
70–91 GiB. They do not validate a precise original reserved-memory point
forecast. Cache pressure and allocator history are not identifiable from
small-card phase peaks alone. The allocated back-test and full-size graph
and prefill components provide the useful cross-checks.

The startup budget includes the measured combined-training preparation peak,
two maximum student KV snapshots, both rollout caches, rollout graph scratch,
and 1.5 GiB. It deliberately counts some storage twice. The head-start budget
includes both prefill working sets and caches alongside persistent state.

The 95 GiB card still needs qualification. Target-device graph workspaces,
allocator history over a long run, and full-scale update time remain unmeasured.
No full-scale performance claim is inferred from local seconds alone.

## Evidence and exclusions

[Memory record](memory-qualification.json) contains resolved commands, seeds,
environment, provenance and artifact hashes, buffer sizes, phase peaks,
fits, parity comparisons, and test results. [Phase CSV](memory-phases.csv)
provides the per-phase measurements. [Control patch](memory-control.patch)
reproduces the control trainer from `89924ff5`, using the new HAL graph helper
and `git apply --unidiff-zero`.
Raw logs, probe scripts, checkpoints, and compile caches remain under the
study directory stated above.

`control-matchup-g1-m12` is excluded because a diagnostic overlapped its
startup. An early paused control later failed with illegal CUDA access; its
cause is unresolved and all its data is excluded. Clean repeats of that
configuration completed warmup and reached the ordinary reference OOM.
Lane-compaction prototypes are excluded; they changed BF16 inference shapes.
The final implementation retains the original rows and batch shapes.

Initial deferred-capture tests exposed grad-mode and autocast mismatches;
the helper now preserves both and focused regressions pass. Initial standalone
value probes that captured under the wrong grad mode are excluded. The first
fixed-target probe omitted the inverse conditioning-range transform; both
sides were rerun with that correction. These diagnostic runs do not enter
the memory or time fits.
<!-- RESULTS END -->

## Validation

The shared-pool GPU test runs four updates through resets and anchor
promotions. It compares private and shared rollout pools. Warmup batch
hashes are exact. Selected data and sampled actions remain exact through
PPO. Metrics and parameters use explicit float-order tolerances; a strict
parameter comparison first detected a 3.7e-9 difference in backward
reduction. Separate actor tests compare three eager and graphed optimizer
updates in FP32 and BF16. Reference tests check decoder graph lifetime,
real-window retention, padding, slot order, and independent output storage.
Pool tests cover preparation order, stream binding, scratch reuse, grad mode,
autocast, and cleanup after partial capture failure.

A native four-update comparison also has identical batch, action, and model
state hashes at every update. Longer native comparisons include the fixed
32-update critic warmup and PPO, with all arm flags intact.

The full suite and integration suite are intentionally omitted, as requested.
The opt-in `short_gpu_run` test uses a different actor profile and is
excluded from the focused command; the native pilot runs exercise the
required profile instead.

All 11 valid native control/treatment pairs match through the 32 warmup
updates. At eight lanes and one group, `old` has identical batch and action
hashes through all 36 updates, including four PPO updates; final weights
differ by at most 7.45e-9. At 16 lanes and one group, `matchup` matches all
34 batch and action hashes; final weights differ by at most 1.46e-11.

The eight-lane `matchup` and 16-lane `old` trajectories diverge after PPO
starts. The first pre-update KL difference is about 1e-7. This is consistent
with BF16 float-order noise, but later trajectory metrics are not treated
as a parity check. The JSON retains every pair’s results.

Resuming both paths from the same update-32 checkpoint gives identical
reference inputs, scores, distributions, values, advantages, targets, and
weights before optimization. Separately, the original `89924ff5` reference
path and the new path score those fixed inputs with identical active masks
and mean KL 2.0e-7 on active factors. Scores remain unchanged across three
replays after free allocator memory is filled with NaNs and released.

Three fixed-data Muon steps use the production 17M model, 512 windows, and
four microbatches. They use captured native window data and targets;
the common KL coefficient is reconstructed by inverting the stored normalized
log-lambda condition. Losses and gradient norms match exactly. The maximum
first-step gradient difference is 1.34e-7; the final weight difference is 1.86e-9.
The actor check also reuses free allocator memory with NaN-filled buffers.

Final focused command (all paths relative to the repository):

```sh
PYTHONPATH=/tmp/claude-1000/o65/delivery/msl:$PWD \
MSL_CORE_LIBRARY=/tmp/claude-1000/o65/delivery/msl/build/melee_core/python/libmelee_core.so \
HAL_NATIVE_LIBRARY=/tmp/claude-1000/o65/delivery/libhal_native.so \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=1 \
TMPDIR=$HOME/data/scratch/o65-tmp \
TORCHINDUCTOR_CACHE_DIR=$HOME/data/scratch/o65-tmp/memory-study/inductor \
TRITON_CACHE_DIR=$HOME/data/scratch/o65-tmp/memory-study/triton \
.venv/bin/python -m pytest -q \
  tests/test_o65_tick_memory.py tests/test_cuda_graph_preparation.py \
  tests/test_window_replay_actor.py tests/test_065_dense.py \
  tests/test_o65_fixed_layout.py -k 'not short_gpu_run' \
  --basetemp=$HOME/data/scratch/o65-tmp/pytest-memory-final
```

Result: **127 passed, 1 deselected, 17 warnings**, 380.31 seconds. The warnings
are 14 TorchScript deprecations, the unchanged disabled-TF32 setting, the
deliberately failed empty capture, and an existing scalar conversion in an
actor test. There are no test failures or fixture skips.
The full suite and integration suite were not run.

These checks pass on all six changed Python files:

```sh
uv run --no-sync ruff format --check <changed Python files>
uv run --no-sync ruff check <changed Python files>
uv run --no-sync ty check --python-version 3.14 --error-on-warning <changed Python files>
```

The exact file lists, commands, exit codes, and output are in the memory record.
