# P0 and E0: normalized auxiliary MTP baseline

Updated: 2026-08-07

## Question

What does a plain action policy gain from sparse future-action prediction when the auxiliary loss
has one fixed total weight?

P0 is the systems package used to train E0. E0 is the scientific baseline for later head and AWR
experiments.

## Model

- Standard causal transformer.
- Raw rolling context of 256 frames.
- Full attention over that context.
- Full transformer recomputation during training and closed-loop inference.
- Independent linear action heads at offsets 1, 5, 9, and 13.
- Only offset 1 is executed.
- No AWR, critic, rank weight, temporal conditioning, or within-frame conditioning.

The loss is:

\[
L=L_1+\frac{L_5+L_9+L_{13}}{3}.
\]

The auxiliary term keeps the same total scale if the number of auxiliary heads changes.

## Fixed configuration

- `d_model=256`
- `n_layers=8`
- `n_heads=4`
- `attn_window=0`
- `L_ctx=256`
- `batch_size=512`
- `grad_accum_steps=1`
- `head_offsets=(1,5,9,13)`
- `aux_loss_weight=1.0`
- `max_steps=16384`
- `warmup_steps=500`
- `muon_lr=0.02`
- `adam_lr=8.5e-4`
- `weight_decay=0.01`
- `amp_dtype=bfloat16`
- `compile_trunk=True`
- `action_vocab=1024`
- `data_root=data/processed/ranked-anonymized-1/mds-policy-v7`
- `windows_per_replay=4`
- `reservoir_capacity=4096`
- `predownload=512`, unless the clean-cache gate selects another tested value
- `num_workers=16`
- `prefetch_factor=2`
- `cache_limit_gb=128`
- `seed=0`

The effective token count is 131,072 frames per optimizer step. Do not reduce the batch size to
solve a data problem.

## Data invariants

- Every training batch must contain 512 distinct replay IDs.
- A replay cannot appear in adjacent batches.
- Window sampling depends only on seed, epoch, and stable replay ID.
- The model does not read `misc_as`.
- Compact decoding must reproduce every consumed source value exactly.
- Validation examples stay fixed.

The compact artifact is a policy projection of canonical MDS v7. Each row records source schema 7
and compact policy layout 2.

## Files

- `experiments/023_mtp_heads.py`: model, loss, training, and evaluation.
- `hal/data/policy_schema.py`: exact compact replay format.
- `hal/training/dataloader.py`: replay packs and central reservoir.
- `hal/training/features.py`: early feature projection.
- `scripts/bench_dataloader.py`: repeatable loader measurements.
- `tests/experiments/test_023_mtp_heads.py`: experiment invariants.
- `tests/test_policy_schema.py`: storage exactness.
- `tests/test_dataloader.py`: sampling and reservoir behavior.

## Clean-cache GPU gate

Run 256 steps on a fresh RTX 4090 instance before the full run.

Use `warmup_steps=32`, disable periodic and final closed-loop evaluation, and keep one final
validation batch. The gate is a systems run, not a model result.

Require:

- No nonfinite values.
- 512 distinct replay IDs per batch.
- Mean warm loader wait below 0.5 seconds.
- Mean warm optimizer step below 0.5 seconds.
- GPU use at least 80% after compilation.
- Startup below 30 minutes.

Test `predownload` values 256, 512, 1,024, and 4,096 only if the first setting fails or leaves clear
download stalls. Keep the reservoir at 4,096.

Use a 250 GB disk, at least 200 GB RAM, 24 GB VRAM, and an RTX 4090. The compact data path no longer
needs the old 1 TB disk.

## Full launch

Start from step 0. Do not resume the stopped P0 run.

```text
uv run scripts/launch_vast.py \
  --max-price 1.10 --disk 250 --min-vram 24 --min-ram 200 \
  --min-dlperf 120 --min-compute-cap 890 --max-compute-cap 890 \
  --data-gb 15 --upload-gb 1 --run-hours 3.5 -- \
  uv run experiments/023_mtp_heads.py \
  --cfg.require-flex --cfg.attn-window 0 --cfg.no-eval-incremental-kv \
  --cfg.cache-limit-gb 128 --cfg.predownload 512 --cfg.reservoir-capacity 4096 \
  --comment e0-normalized-aux-bc
```

Record the exact offer, instance, commit, W&B run, command, startup phases, step timing, loader wait,
GPU use, RAM, disk use, and upload time.

## Evaluation

- Validate every 1,024 steps.
- Run 32 fixed CPU matchups every 4,096 steps.
- Run 96 fixed CPU matchups at the final checkpoint.
- Use `eval_max_frames=7200` and `eval_seed=0`.
- Save checkpoints every 2,048 steps.
- Save match rows and replay files.

Report primary and per-offset NLL, group accuracy, transition metrics, gradient interaction, stocks
and damage per active minute, dead-frame rate, terminal-game results, crashes, and wall time.

Closed-loop results decide promotion. Lower auxiliary NLL alone is not a policy improvement.

## Runtime rules

Target 3.0 to 3.5 hours through final evaluation and upload.

Flag:

- Startup over 30 minutes.
- Warm steps over 0.5 seconds.
- GPU use below 80%.
- A periodic CPU evaluation over 25 minutes.
- A step-4,096 projection above 3.5 hours.

Do not let the instance destroy itself until `final.pt`, evaluation rows, replays, and logs are in
the project store.

## Current state

The first P0 attempt stopped at step 1,024 because the old data path projected to 7.2 hours. Its
validation values were finite. It is not an E0 result.

The compact pipeline passes the full 114,768-replay audit. It reduces decoded training arrays from
802.47 GB to 76.19 GB and compressed storage from 29.82 GB to 13.34 GB. The local full-artifact
reservoir benchmark produced 512 distinct replays per batch and a 0.072-second mean loader wait
after startup.

The full test suite passed 861 tests before the artifact rename. The renamed P0 configuration then
passed its 17 focused tests. The artifact is published and verified. The remote clean-cache GPU
gate is active.

Gate attempt 1 used Vast instance `47094969` at commit `e8433d0`. It stopped before model or data
loading because the command set `max_steps=256` but left `warmup_steps=500`. The instance was
destroyed. Attempt 2 sets `warmup_steps=32`.

Gate attempt 2 used instance `47095279`, commit `be31fcc`, and W&B run `xhyznlzb`. It completed all
256 steps and uploaded `final.pt`. Across steps 50 through 255, mean step time was 0.343 seconds and
mean loader wait was 0.209 seconds. Median values were 0.320 and 0.187 seconds. P95 values were
0.487 and 0.354 seconds. Every batch had 512 distinct replays, and all recorded losses and gradient
norms were finite. This meets the wall-time gate but leaves the GPU waiting for data.

Gate attempt 3 changes only `prefetch_factor` from 2 to 32. Sixteen workers can then queue up to 512
replay packs, which matches one training batch. Keep `predownload=512` and all other gate settings
fixed.

## Promotion

E0 is valid only if it reaches step 16,384 and retains the complete final evidence. After E0, train
the matched P1 attention arm on the same compact data path and action vocabulary. The old P1 run is
exploratory because it used the old sampler and a 512-entry action-state table.
