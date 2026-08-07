# P1-old full-recompute rescore

Updated: 2026-08-07

## Question

How does the completed exploratory long-context policy behave when closed-loop inference rebuilds
the full raw rolling window instead of using its temporal KV cache?

This is an evaluation-only systems check. It is not a matched comparison with P0. The checkpoint
used the old full replay sampler, `mds-v7`, and a 512-row action table. Matched P1 uses the compact
policy dataset and a 1,024-row table.

## Frozen input

- W&B run: `19sowpt8`.
- Run name:
  `260807-013656_023_mtp_heads_gpt-d256-L8-h4-Lc1024-o1.5.9.13-linear_ranked-anon-1_e0-normalized-aux-bc`.
- Checkpoint: `final.pt`, 56,305,399 bytes in R2.
- Step: 16,384.
- Model: `d_model=256`, 8 layers, 4 heads, `L_ctx=1024`, SWA window 128.
- Decode: FP16, offset 1, full rolling-window recomputation.

Do not change weights, temperatures, character schedule, frame budget, or execution horizon.

## Files

- Use `experiments/023_mtp_heads.py` at the current experiment branch commit.
- Write evidence under `manual_evals/p1-old-final-recompute-fp16` in the original R2 run.
- Log labeled metrics to W&B run `19sowpt8` without renaming it.
- Do not add model or training code for this rescore.

## Command

```text
uv run scripts/launch_vast.py \
  --max-price 1.50 --disk 100 --min-vram 24 --min-ram 200 \
  --min-dlperf 120 --min-compute-cap 890 --max-compute-cap 890 \
  --data-gb 1 --upload-gb 2 --run-hours 0.75 -- \
  uv run experiments/023_mtp_heads.py \
  --eval-run \
  260807-013656_023_mtp_heads_gpt-d256-L8-h4-Lc1024-o1.5.9.13-linear_ranked-anon-1_e0-normalized-aux-bc \
  --eval-checkpoint-name final.pt --eval-decode recompute \
  --eval-n-matchups 96 --eval-seed 0 \
  --wandb-run-id 19sowpt8 --wandb-label p1-old-final-recompute-fp16
```

## Evidence

Record the offer, instance, commit, model dtype, decode mode, protocol, wall time, 96 scheduled
boots, active games, countdown-only tails, failed boots, replay count, and uploaded rows.

Report stocks and damage per active minute with boot-clustered intervals, dead frames, mean stock
difference, and the terminal subset. Compare point estimates with the checkpoint's old KV result,
but use separate uncertainty intervals. Random restart stages prevent a paired causal estimate.

## Interpretation

- A material change is evidence that temporal KV decoding changed policy semantics.
- Similar results support, but do not prove, practical parity. Offline long-roll and sampled-action
  parity remain the stronger correctness tests.
- Neither outcome selects the P1 architecture because the training data and action table differ
  from P0. The matched P1 gate and training run remain required.

## Results

Pending the official P0 evaluation. Run one evaluation job at a time.

The launch command passed a search-only preflight at commit `eb5eca7`. Two RTX 4090 offers met
the fixed bounds. The best offer had 252 GB RAM, DLPerf 125.6, and an effective price of
$0.829/hour with the 100 GB disk. The recompute, rolling-context, and experiment tests passed:
69 passed and 6 hardware-only tests skipped.

The official rescore launched on Vast instance `47110149` at commit `6e97103`. The selected RTX
4090 host has 252 GB RAM, DLPerf 125.5, and a 100 GB disk at $0.725/hour effective. Provisioning
took about 5 minutes 17 seconds. The evaluator uses the exact command above and is the only active
experiment job.

The rescore completed and the instance destroyed itself after upload. The protocol records FP16,
full recomputation, 96 scheduled boots, 96 concurrent slots, no boot crashes, and 120 active games.
The result is:

- Stocks taken per active minute: 0.798, 95% CI `[0.713, 0.883]`.
- Stocks lost per active minute: 1.405, 95% CI `[1.281, 1.534]`.
- Damage dealt per active minute: 132.5, 95% CI `[125.4, 139.3]`.
- Damage taken per active minute: 116.1, 95% CI `[112.0, 120.0]`.
- Dead frames: 2.14%.
- Mean stock difference: -0.950.
- Twenty-four games reached a terminal stock state; the policy won none.

R2 contains the row manifest and 125 replay files, 231.01 MiB in total. The manifest has 120 active
rows and no countdown-only row. Five 61,396-byte replay files have no match row. Keep them as
unmatched diagnostic artifacts and do not count them as games.

The first replay began about 52 seconds after host readiness. The last replay ended about 22
minutes later, and the manifest uploaded about 25 minutes after readiness. This reaches the
25-minute evaluation warning threshold.

The checkpoint's old KV final point estimates were 0.804 stocks taken/min, 1.480 stocks lost/min,
129.2 damage dealt/min, and 117.7 damage taken/min. The recompute estimates are close. This does not
prove decode parity: the runs used independent game randomness, different concurrency, and
different evaluator versions. The matched P1 checkpoint remains the required P2 comparison.
