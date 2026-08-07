# P1 full-recompute evaluation

Updated: 2026-08-07

## Question

How does the trained P1 policy behave when closed-loop inference rebuilds all 1,024 raw context
frames instead of using its temporal KV cache?

This is an evaluation of the existing `19sowpt8` checkpoint. It is not a new training run.

## Change

Add `--eval-recompute` to the manual evaluation command. The flag changes only
`eval_incremental_kv` after loading the checkpoint. Model dimensions, attention window, weights,
decode settings, and matchup seeds still come from the checkpoint.

Files:

- `experiments/023_mtp_heads.py`: add the manual evaluation override.
- `tests/experiments/test_023_mtp_heads.py`: prove that the policy factory receives
  `eval_incremental_kv=False`.
- This file: record the command, timing, W&B run, and results.

## Evaluation

- Use the final `19sowpt8` checkpoint.
- Run 96 fixed CPU matchups.
- Keep the saved temperature, action-group temperatures, support mask, minimum probability,
  execution horizon, seed, and 7,200-frame limit.
- Save match rows and replay files.
- Log results back to `19sowpt8` with the label `p1-full-recompute`.

Report stocks and damage per active minute, dead-frame rate, terminal-game results, wall time, and
failures. Do not compare P1 against P0 until P0 finishes.

## Limits

The old P1 run used the old replay sampler and a 512-entry action-state table. It is exploratory
evidence. A clean attention comparison requires a new P1 training run with the same compact data,
sampler, action vocabulary, training seed, and evaluation protocol as P0.
