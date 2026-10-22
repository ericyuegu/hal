## Summary

Added O64 loading, 512-frame configuration, production cadences, behavior counts,
architecture FLOPs/MFU, and the 256-update outcome run configuration. Built the
44-source sampler. Checkpoint SHA-256 is recorded in `production-256.json`.

## Commits

`80921770`, `c2d69a42`, `dd899ce3`, `2aa312bd`, `6c640e9a` plus the evidence/report
commit; pushed to `origin/o65-247m`.

## Tests run

Ruff format, Ruff check and ty pass on all 11 changed Python files. Focused
checks: 19 production/storage, 24 pilot, 16 layout/reward and six evaluation
tests passed. Earlier type, validation-order and fixture failures were fixed.
Full/integration suites were omitted under the speed rule. Exact commands,
deselections and corrected failures are in [production-checks.json](production-checks.json).

Both 3060 attempts ran out of memory before the actor update at four matches
and one window. Zero PPO updates completed; finite PPO losses and measured MFU
remain unverified. Startup: 18 graphs, 516.8 seconds cold and 58.2 on retry.

## Self-review findings and fixes

Fixed countdown exposure, forced-prefix head FLOP counts, nested evaluation
metric keys, metadata validation and rejection order. Behavior state survives
512-frame checkpoint/resume. No PPO objective change.

## Cross-stream changes

Shared loader/evaluation changes; training v11 adds `behavior_state`;
`ShardFrames` adds host `frame_ids`. Updated direct test callers and the legacy
manifest fixture. DDP code is unchanged.

## Impasses

O64 exceeds 3060 memory. B200 match/window capacity and the held-out production
cadence remain open; configuration placeholders are explicit.

## Remaining work

Merge DDP, qualify two PPO updates and capacity on B200, freeze the evaluation
cadence, then run the 256-update treatment.
