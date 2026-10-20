## Summary

Implemented two-buffer host staging with no CUDA calls on native workers, strict
persistent compilation caches, shard/tick timing and a Torch-free simulator probe.

At 16 matches, complete updates fall from **3.356 to 3.145 s**; at 64, from
**5.159 to 4.983 s**. Cache reuse cuts first-update warmup from **1,264.9 to
132.8 s** at 16 matches. Nine tick shape variants remain; no padding was added.
The PRO 6000 rollout projection is **4.25–4.50 s**, dominated by GPU work under
the stated assumptions. No cloud job was launched.

Evidence and full phase/tick tables: [async.md](performance/async.md).

## Commits

Branch `o65-async`, from `o65-floor` at `376eb42b`:

- `2e129429` — Measure native shard work and simulator floor.
- `293be5d6` — Stage rollout transfers on the main thread.
- `8af6acd7` — Persist O65 compilation by source and device identity.
- `bc86f63c` — Keep cache source selection at the experiment boundary.
- `f2901c17` — Preserve CPU shard overlap and bound timing records.
- `13d01af8` — Report tick GPU work and a warmed simulator floor.
- `0577890f` — Cover warmup events in CUDA capture tests.
- `974a9ca0` — Cancel pending simulator work when input copies fail.
- `278af7e3` — Cover interruption of pending controller transfers.

Runtime measurements use `974a9ca0`; subsequent changes add tests and evidence.

## Tests run

Formatting, lint and required/extended type checks pass. The final broad run
passed **2,730 tests**, with **eight skips**: six missing v7 fixture cases count
as validation failures; two production hardware tests are disabled.

Final required Dolphin integration: **8 passed**. Final native/async/cleanup
integration: **21 passed**, with no integration skips.
CPU/CUDA transfer parity, FP32 bit preservation and failed-copy cleanup pass.
The real proxy/native resume check passes all state families and exact restoration
on three rewinds, at the existing noise tolerances.

Every command, prior failure and skip is recorded in
[async-validation.md](performance/async-validation.md).

## Self-review findings and fixes

Fixed a CPU overlap regression and a cleanup hang when a copy event fails.
Exceptional host futures now stop affected workers before they read incomplete
inputs. Corrected a test that compared relocated native pointers across batches;
semantic tensors and same-batch exact restore remain strict. Corrected the
benchmark’s warmup interval, stage comment and provenance records. Updated the
mocked capture test for its new event. The second review found no further issue
in the changed transfer/cache path. Model math, precision gates and resume
tolerances are unchanged.

## Cross-stream changes

Shared changes are `hal/sim/native_shard.py`, `hal/inference/cuda_graph.py`, the
new compile-cache boundary, the Modal launcher and associated tests. No MSL source,
model, objective or stored training format changed. Other worktrees were not edited.

## Impasses

The required local v7 fixture is absent; its six skips are failures under
AGENTS.md. Modal cache reuse cannot be qualified here because cloud launches are
prohibited for this stream.

## Remaining work

Supply the v7 fixture and repeat its six cases. Verify cache persistence and the
rollout projection on the next PRO 6000 run. The conditional optimizer CUDA graph
is deferred while the step 1–3 handoff gate remains incomplete; the eager optimizer
is unchanged.
