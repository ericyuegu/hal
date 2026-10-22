## Summary

Merged O64 and DDP on `o65-247m-ddp`: checkpoint v12, global behavior/MFU,
O64 B200 training and fresh-process measurement sweeps. Final evaluation is
D18/R2, u=0.95, lambda=1, 384 MSL games. Commands: [ddp.md](ddp.md).

## Commits

`e35f6b3b` merges DDP; `99d5eef7` adds O64 launches and measurement.
The protocol and report are in this commit. Branch pushed to origin.

## Tests run (with results)

Ruff format/check and ty pass on all 17 changed Python files. Focused run:
190 passed, four stale-fixture failures fixed; all four then passed. Final DDP
set: 17 passed. Twelve GPU/integration cases deselected; no skips. Three local
launch dry runs passed. No GPU or cloud job ran. Commands and corrected failures:
[247m-ddp-checks.json](247m-ddp-checks.json). Full suites were omitted as instructed.

## Self-review findings and fixes

Fixed rank-local behavior restoration, global counts/FLOPs and MFU normalization,
per-rank match validation/id offsets, the output-directory race, and stale test
fixtures. One review; no repeated parity audit.

## Cross-stream changes

Checkpoint v12 applies to single-GPU and DDP resume. Shared metric helpers,
trainer measurement dispatch, launch scripts and test fixtures changed.

## Impasses

None in implementation. B200 capacity remains unmeasured.

## Remaining work

Run hardware NCCL/Muon/CUDA-graph and exact-resume checks, measure capacity, then
launch the treatment. The 256-update endpoint includes 32 critic warmup updates.
