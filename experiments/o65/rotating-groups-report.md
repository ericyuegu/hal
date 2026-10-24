# Rotating-group storage: 2026-10-24

## Summary

Four identical groups now share one full frame store, rollout/evaluation table, action ring,
trunk KV cache, and tick/restart graph set. Inactive groups retain their last 129 frames, queues,
and episode state. The four-group B200 run passed startup capture but failed during the first
PPO preparation. No complete update or gradient comparison was measured. The sandbox is down;
1024-match production capacity remains unqualified.

Every group clock must be divisible by 128 before collection. Head start rounds 1200 frames to
1536 with 512-frame rollouts. Checkpoint validation derives every group clock from that origin
and completed group rotations. A misaligned clock is rejected before collection changes the pool.
Training graphs already belong to policies and use shape keys. Preparation owns only the active
group's batch; no further large per-group training allocation was found.

## Commits

- `5072258e` Share rollout storage across rotating groups
- `cf3ccb4d` Preserve production settings and devbox evidence
- `21c8e8a2` Record sandbox memory and flush the mounted cache
- `03cc048a` Capture benchmark provenance before execution
- `72eab0a6` Fix preparation compilation and failure reports

## Measurements

Control: `/home/ericgu/data/scratch/o65-devbox/runs/bench-20261024-003100-m64-full`;
log `/home/ericgu/data/scratch/o65-main-1024-bench.log`.
Treatment: `/home/ericgu/data/scratch/o65-devbox/runs/bench-20261024-012332-m64-full`;
log `/home/ericgu/data/scratch/o65-main-shared-measure.log`.
The treatment trainer SHA256 matches commit `21c8e8a2`; the run's `worktree.json` and
[measurement record](rotating-groups-measurement.json) preserve its identity and artifact hashes.

The treatment requested four groups, 1024 active matches, 512 rollout frames, every decision,
512-window optimizer batches, eight microbatches, 64-window graph chunks, and six updates.
It used the O64 step-0098304 checkpoint, production sampler, seed 65, reward weights
0.1/120/120, head start 1200, snapshot interval 64, and checkpoint interval 32.

| Observation | Before | After |
| --- | --- | --- |
| Startup capture | OOM | 833.266 s, 48 graphs |
| Complete startup | Failed | 957.970 s |
| Failure phase | Startup capture | First preparation, `factor_kl` / `categorical_kl` |
| Allocated memory at OOM | 175.97 GiB | 172.91 GiB |
| Graph private pools at OOM | 93.74 GiB | 91.50 GiB |
| Failed allocation | 1.01 GiB | 1.41 GiB |
| Sampled host RAM peak | Unavailable | 26.015 GiB |
| Completed updates | 0 | 0 |

OOM values are rounded CUDA diagnostics from different phases, not matched peak comparisons.
The treatment process used 177.80 GiB of the 178.34 GiB device at failure. Host telemetry samples
the whole cgroup every 0.5 s; no kernel historical peak counter was available.

Recorded startup phase allocated/reserved peaks, in GiB:

| Phase | Before allocated | After allocated | After reserved |
| --- | ---: | ---: | ---: |
| Rollout | 82.23 | 70.13 | 150.59 |
| Value | 67.65 | 67.66 | 150.59 |
| Reference | 78.72 | 78.72 | 170.58 |
| Actor | 90.86 | 90.86 | 176.50 |

Cold/warm evidence from the first group's reported startup phases:

| Capture phase | Earlier cold run, seconds | Warm treatment, seconds |
| --- | ---: | ---: |
| Rollout | 984.061 | 672.696 |
| Value | 84.102 | 11.675 |
| Reference, all cadences | 502.907 | 79.312 |
| Actor, all cadences | 494.989 | 52.641 |

The cold run never completed whole startup. Configuration defaults and storage also changed, so
these are observations, not a controlled throughput speedup. The other three treatment groups'
rollout capture phases took 3.622, 4.083, and 4.039 s with the existing graphs.
Pre-existing Inductor/Triton files survived restart, and a read-only py-spy sample observed
`cache_hit_post_compile` and `TritonBundler.read_and_emit`. Cache loading still spent substantial
time in volume file operations. `sync /cache` flushed the mounted v2 volume before shutdown,
as specified by the [Modal sandbox filesystem documentation](https://modal.com/docs/guide/sandbox-files).

Training phase seconds, update seconds, MFU, tick time, group switch cost, mean finished-game
length, and gradient cosines are unavailable. Preparation failed before metrics or gradient
inputs were saved. Its phase-memory report was also missing because full profiling did not set
`PhaseMemory.output`; that reporting defect is now fixed and tested locally.

## Tests run

The final checks used `PYTHONPATH=.` and
`TMPDIR=/home/ericgu/data/scratch/o65-tmp-main`. Heavy local tests ran serially under `nice`,
with two OpenMP/MKL threads and two Inductor compile workers. Full-suite execution also used
`MALLOC_ARENA_MAX=2`. Integration tests sourced `.env` without printing its values.
The [validation record](rotating-groups-validation.json) gives commands, XML paths and hashes,
each failed test, and every skip reason.

| Command | Result |
| --- | --- |
| `uv run ruff format --check .` | Passed, 427 files |
| `uv run ruff check .` | Passed |
| `uv run ty check --python-version 3.14 --error-on-warning hal experiments/059_muon_action_sequence.py scripts` | Passed |
| `nice uv run pytest -q -m "not integration"` | 2894 passed, 9 failed, 39 skipped, 90 deselected |
| Focused storage, training batching, and devbox files after fixes | 65 passed, no skips |
| Devbox files after provenance fix | 28 passed, no skips |
| `tests/test_o65_training_probe.py` after reporting fix | 7 passed, no skips |
| Three netplay failures rerun alone | 3 failed again, no skips |
| `HAL_REQUIRE_INTEGRATION=1 uv run pytest -q tests/test_roundtrip.py tests/test_session_cleanup.py -m integration` | 2 passed, 6 failed, no skips, 6 deselected |

Five full-suite O65 failures exhausted Torch's pre-startup frame-acceptance compile budget.
The scoped preparation guard fixed these; the focused rerun includes all five. The four other
full-suite failures are the known missing R2 credentials in control timing recovery and three
netplay timeouts. The full suite ran once, before the fixes; it has not passed as a whole.
Its skips comprise 22 missing dev archive, 11 missing local subsets, and six optional hardware
or checkpoint qualifications.

All six integration failures require a missing v7 dev MDS fixture. The two session cleanup tests
passed. The shared dev manifest has no top-level schema and annotation schema 2, so it was not
substituted. Initial implementation tests also exposed an invalid-row parity assertion, a missing
`PYTHONPATH`, and a missing test constructor argument; each was corrected before its final rerun.

## Self-review findings and fixes

Reviewed `git diff a0ae12df...HEAD`, including capture pools, streams, base/view writes, frame
boundaries, masks, storage lifetime, and resume. An independent Codex reviewer found no further
correctness defect. Fixed the workspace key to include shard topology; expanded parity to all
anchor slots, inactive-slot promotion, five rotations, head start, next-batch resume, and optimizer
and scheduler state. CUDA tests prove exact action/log-prob/value parity and no tick/restart
recapture. No model arithmetic was changed by this storage refactor.

Also fixed unsealed frame-acceptance compile limits, failure-report output wiring, launch-time
provenance, cgroup v1 telemetry, and mounted-volume flushing. Cgroup v2 paths and sandbox SDK
volume commits failed in preflight; the same sandbox was retained while those assumptions were
corrected. No further B200 session was started.

## Cross-stream changes

`scripts/o65_devbox.py` and its tests now preserve production settings, record source identity and
host memory, download failure evidence, and flush the cache. One assignment in full-update
profiling makes phase-memory reports durable. No library, launcher, or production capacity change
was made in this workstream.

## Impasses

The only authorized B200 session ended with preparation OOM. Its missing gradient-input artifact
prevents the requested cosine comparison. The six-update profile set critic warmup to zero to
exercise the actor; production specifies 32. This is an explicit deviation from exact production
settings; production itself is unchanged. The required Opus reviewer was unavailable; an
independent Codex reviewer was used. Current dev MDS and R2 credentials are missing locally.

## Remaining work

Reduce active-update preparation memory and check actor-phase headroom, then obtain a complete
four-group B200 update profile and gradient comparison. A later GPU session needs new user
direction because the single-session allowance is spent. Supply the current integration fixture
and resolve the independent netplay failures before a clean handoff. Keep production capacity
unset until the full update fits with headroom. No 2048-match bench or optimizer-step overhead
study ran in this session.
