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
- `predownload=512`
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
- GPU use at least 80% after compilation when host telemetry is valid. Otherwise record loader and
  compute time separately and flag the missing measurement.
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
gate passed.

Gate attempt 1 used Vast instance `47094969` at commit `e8433d0`. It stopped before model or data
loading because the command set `max_steps=256` but left `warmup_steps=500`. The instance was
destroyed. Attempt 2 sets `warmup_steps=32`.

Gate attempt 2 used instance `47095279`, commit `be31fcc`, and W&B run `xhyznlzb`. It completed all
256 steps and uploaded `final.pt`. Across steps 50 through 255, mean step time was 0.343 seconds and
mean loader wait was 0.209 seconds. Median values were 0.320 and 0.187 seconds. P95 values were
0.487 and 0.354 seconds. Every batch had 512 distinct replays, and all recorded losses and gradient
norms were finite. This meets the wall-time gate but leaves the GPU waiting for data.

Gate attempt 3 used instance `47095681`, commit `937f973`, and W&B run `cmd0d0ft`. It changed only
`prefetch_factor` from 2 to 32. It completed all 256 steps, uploaded `final.pt`, and destroyed the
instance. Across steps 50 through 255, mean step time was 0.344 seconds and mean loader wait was
0.208 seconds. Median values were 0.321 and 0.185 seconds. P95 values were 0.497 and 0.362 seconds.
Every batch had 512 distinct replays, and all recorded losses and gradient norms were finite.

Prefetch 32 did not improve throughput over prefetch 2. The mean step-time difference was below
0.001 seconds, and the mean loader-wait difference was 0.002 seconds. Keep `prefetch_factor=2` to
avoid 16 times more queued replay packs. W&B and Vast reported zero GPU use during active CUDA
training, so that telemetry is invalid on this host. Step time minus loader wait was 0.134 seconds
with prefetch 2 and 0.136 seconds with prefetch 32. The accepted gate projects about 1.6 hours of
training before validation and closed-loop evaluation.

The full P0 run started from step 0 on Vast instance `47096112`, commit `f3eeaf3`, and W&B run
`obx3o3az`. The selected offer is an RTX 4090 with 409 GB RAM, 250 GB disk, DLPerf 125.6, and an
effective price of $1.428 per hour. The run uses prefetch 2 and the complete default validation,
checkpoint, periodic evaluation, and final evaluation schedule. Startup passed CUDA, disk, fixed
validation, compilation, full-attention, and first-batch checks.

At step 1,024, primary action NLL was 1.219 bits per frame and button log loss was 0.0353. NLL at
offsets 1, 5, 9, and 13 was 1.219, 3.073, 3.898, and 4.397. Every batch through this boundary had
512 distinct replays, and all recorded losses and gradients were finite. The boundary completed
430 seconds after training startup, including compilation and validation.

At step 2,048, NLL at offsets 1, 5, 9, and 13 was 1.172, 2.948, 3.743, and 4.235. Button log loss
was 0.0339. The 56,699,580-byte `latest.pt` checkpoint was saved, uploaded, and verified directly
in R2. The decoded compact cache reached 72 GB while the full instance disk was 31% used.

At step 4,096, primary action NLL was 1.115 and button log loss was 0.033. The 56,705,941-byte
`step_004096.pt` checkpoint was uploaded and verified directly in R2. The first periodic evaluator
used full recomputation. Two model-controlled Sheik boots first appeared as Zelda and were rejected.
Both succeeded on the existing fresh-session retry. All 32 scheduled boots produced trajectories,
giving 47 games and zero crashes. The evaluation took about 8.1 minutes including retries, scoring,
and uploads. No schedule change is needed.

At this checkpoint, stocks taken per active minute were 0.673, 95% CI `[0.556, 0.784]`. Stocks
lost per active minute were 1.795, 95% CI `[1.565, 2.043]`. Damage dealt per active minute was
116.8, 95% CI `[108.2, 125.0]`. Damage taken per active minute was 125.0, 95% CI
`[118.1, 132.7]`. Dead frames were 2.51%. Mean stock difference across the 47 rows was -1.489.
Fifteen games reached a terminal stock state; the policy won none. This is a terminal subset, not a
full-schedule win rate.

At step 5,120, NLL at offsets 1, 5, 9, and 13 was 1.103, 2.822, 3.594, and 4.078. Button log loss
was 0.0325. The recorded gradient norm was finite, and the batch contained 512 distinct replays.

The launch commit applies FP16 when an evaluation worker loads a checkpoint, but its synchronous
final evaluator uses the live FP32 training model. This precision mismatch was found after launch.
Keep the in-run final result as a diagnostic. After `final.pt` uploads, run the official 96-boot
final evaluation by loading that checkpoint through the FP16 decode path. Record the model dtype in
the new evaluation artifact. Do not resume or alter the active training process.

After `final.pt` is verified, launch the corrected evaluation from the current experiment commit:

```text
uv run scripts/launch_vast.py \
  --max-price 1.50 --disk 100 --min-vram 24 --min-ram 200 \
  --min-dlperf 120 --min-compute-cap 890 --max-compute-cap 890 \
  --data-gb 1 --upload-gb 2 --run-hours 0.75 -- \
  uv run experiments/023_mtp_heads.py \
  --eval-run \
  260807-164825_023_mtp_heads_gpt-d256-L8-h4-Lc256-a1024-full-recompute-o1.5.9.13-linear_ranked-anon-1_e0-normalized-aux-bc \
  --eval-checkpoint-name final.pt --eval-n-matchups 96 --eval-seed 0 \
  --wandb-run-id obx3o3az --wandb-label p0-final-fp16
```

The run-based evaluation path downloads the checkpoint, writes to
`manual_evals/p0-final-fp16`, logs labeled metrics without renaming the W&B run, uploads all rows
and replays to the source run, and drains those uploads before the instance exits.

The command passed a no-rent launcher dry run at commit `b422724`. The current qualifying offer was
an RTX 4090 with 1,008 GB RAM, 100 GB disk, DLPerf 125.5, and effective price $1.382 per hour. Select
the offer again at launch; market state may change.

The command passed another no-rent audit at commit `31e151d`, after the run-based evaluator and
decode-mode changes. It still requests `final.pt`, 96 boots, seed 0, W&B run `obx3o3az`, and label
`p0-final-fp16`. The selected offer was an RTX 4090 with 1,008 GB RAM, 100 GB disk, DLPerf 125.3,
and effective price $1.382 per hour.

At step 8,192, NLL at offsets 1, 5, 9, and 13 was 1.074, 2.764, 3.522, and 4.002. Button log loss
was 0.0318. The 56,705,941-byte `step_008192.pt` object was verified directly in R2.

The second periodic evaluation took about 7.8 minutes. All 32 boots succeeded. Instant restart
produced 36 games with active play and two countdown-only tail fragments after a prior game ended
near the frame budget. The launched metric code counted those fragments as matches. A corrected
offline reduction keeps their raw rows but excludes them from rates and confidence intervals.

The corrected step-8,192 result is:

- Stocks taken per active minute: 0.893, 95% CI `[0.764, 1.028]`.
- Stocks lost per active minute: 1.323, 95% CI `[1.138, 1.530]`.
- Damage dealt per active minute: 131.1, 95% CI `[120.5, 141.8]`.
- Damage taken per active minute: 122.4, 95% CI `[114.8, 129.8]`.
- Dead frames: 1.92% across games with active play.
- Mean stock difference: -0.750 across 36 active games.
- Six games reached a terminal stock state; the policy won none.
- Crashed boots: zero.

Compared with step 4,096, both stock rates and damage rates improved. This is one periodic sample,
not a checkpoint-selection rule. Continue to the declared final evaluation.

At step 11,264, NLL at offsets 1, 5, 9, and 13 was 1.050, 2.703, 3.446, and 3.923. Button log
loss was 0.0311. W&B remained live through step 11,530. The latest history row had 512 distinct
replays and a finite gradient norm. Most logged steps remained near 0.33 to 0.38 seconds. One step
took 1.24 seconds, followed by normal timing, so there is no sustained loader regression. The
56,699,580-byte `latest.pt` saved at step 10,240 uploaded successfully.

At step 12,288, NLL at offsets 1, 5, 9, and 13 was 1.042, 2.684, 3.422, and 3.896. Button log
loss was 0.0309. W&B was live through history step 12,306 with 512 distinct replays and a finite
gradient norm. Both `latest.pt` and the 12,288 evaluation checkpoint uploaded successfully. The
third 32-boot periodic evaluation then started; its result is pending.

## Promotion

E0 is valid only if it reaches step 16,384 and retains the complete final evidence. After E0, train
the matched P1 attention arm on the same compact data path and action vocabulary. The old P1 run is
exploratory because it used the old sampler and a 512-entry action-state table.
