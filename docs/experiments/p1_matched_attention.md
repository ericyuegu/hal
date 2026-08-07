# P1: matched long-context SWA package

Updated: 2026-08-07

## Question

Does a longer raw context with local attention improve policy quality when token count, approximate
attention work, data path, model, loss, optimizer, and evaluation stay matched to P0?

This is a package comparison. It does not isolate one variable. P1 changes context length, batch
size, and attention mask together.

## Reference

Use the completed P0 final checkpoint and evaluation rows as the reference. Do not use the old P1
run `19sowpt8` as matched evidence. That run used the old sampler and a 512-entry action-state
table.

## Only training changes from P0

- `L_ctx`: 256 to 1,024.
- `batch_size`: 512 to 128.
- `attn_window`: full causal to 128 frames per layer.

Keep all other P0 fields fixed, including:

- Compact `mds-policy-v7` data and action vocabulary 1,024.
- Four sampled windows per replay and reservoir capacity 4,096.
- Prefetch 2, predownload 512, and 16 workers.
- Linear independent heads at offsets 1, 5, 9, and 13.
- `L_1 + (L_5 + L_9 + L_13) / 3` with no AWR.
- Seed 0, 16,384 steps, optimizer, schedule, checkpoints, and evaluation seeds.
- Full rolling-window recomputation during closed-loop evaluation.

Both arms process 131,072 frame tokens per optimizer step. P0 has 16,842,752 causal attention
edges per layer and step. P1 has about 15,736,832 local attention edges per layer and step.

`windows_per_replay=4` does not put four windows from one replay in the same batch. The loader packs
four non-overlapping windows for later use. The reservoir emits at most one of them per batch and
keeps that replay out of the next batch. P0 must therefore report 512 distinct replays per step and
P1 must report 128 when gradient accumulation is one.

## Files

- Add this plan before implementation.
- Use `experiments/023_mtp_heads.py` for training and evaluation.
- Reuse the compact replay code without schema or sampling changes.
- Add or change tests only if the configuration exposes a missing invariant.
- If a code change is needed, list the exact file and reason here before launch.

The intended core arm is configuration-only. Do not add a new model file merely to encode three
flags.

One systems-only trainer change is planned before the P1 gate. In `experiments/023_mtp_heads.py`,
start the training iterator before building the fixed validation cache, build that cache on one
background thread, and begin the real step-0 forward while the cache builds. This overlaps training
data prefetch, validation reads, and lazy `torch.compile` work. It must use the normal step-0 batch;
do not add a warmup forward or advance any random generator. Poll the cache task each step so an
early cache failure stops training promptly. Wait for it before the first validation use.

Update `tests/experiments/test_023_mtp_heads.py` to prove the cache starts before step 0, completes
before final validation, and does not add a training batch. Record startup phase timing in the P1
gate. If concurrent reads make warm loader wait or total startup worse, revert this systems change
before the full P1 run.

The concurrency audit found that PyTorch `DataLoader` iterator creation uses the process-wide Torch
random generator by default. A validation iterator created on another thread could then race with
model randomness. Update `hal/training/dataloader.py` so every loader owns a private generator
seeded from its existing loader seed. Add a test proving loader setup does not change the process
Torch RNG. Do not launch the concurrent path without this isolation.

Before the P1 gate, update `experiments/023_mtp_heads.py` to log how many replay IDs overlap the
previous optimizer step. This is measurement only and must not change sampling. The reservoir
enforces zero overlap between adjacent batches inside one iterator epoch, but its cooldown resets
when the trainer starts the next epoch. The live metric will expose any rare boundary overlap instead
of claiming a stronger invariant than the code provides.

Also add validation NLL and accuracy for every action group at every future offset. P0 logged only
total future-head NLL and primary-head group accuracy. This is evaluation-only. Rescore the frozen
P0 checkpoint on the same validation cache so the P1 comparison has symmetric diagnostics.

This startup change is now implemented. Creating the training iterator starts worker prefetch, then
one background thread builds the fixed validation cache while step 0 performs the real compiled
forward and update. There is no warmup batch or extra forward. Training logs the exact number of
consumed batches. The cache task is checked each step, and validation blocks on it only if needed.
Train and validation loaders use separate seeded Torch generators, so iterator creation cannot race
with model randomness. The focused trainer, loader, and upload suite passes 47 tests; static checking
adds no new errors. The Vast P1 gate must still measure whether the overlap improves wall time under
real R2 and GPU load.

Startup timing now records the worker-prefetch iterator setup, the background cache task's actual
start-to-finish duration, compiled step-0 wall time, and step-0 loader wait in W&B summary fields.
The cache task records its own finish time, so a long compile cannot inflate the reported cache
duration. The expanded focused suite passes 48 tests.

The audit from P0 launch commit `f3eeaf3` to the planned P1 code found no change to model
initialization, replay selection, window selection, targets, loss, or finite optimizer updates.
Later code adds:

- A private seeded generator for each PyTorch loader. This removes process-global seed draws but
  does not change the replay or window RNGs.
- Concurrent train prefetch, validation caching, and the real step-0 compile.
- A fail-fast check for nonfinite loss or gradients.
- Replay-overlap, startup, dtype, upload, and evaluation-protocol evidence.

P0 has `history_dropout_p=0`, and its train forward has no other random operation. Removing loader
draws from the global Torch RNG therefore cannot change finite training updates. The consumed-batch
counter proves that startup overlap does not skip or duplicate a batch. Treat any P1 gate mismatch
in the first batch, initial logits, or consumed count as a failed matched-control audit.

## Correctness gate

Before the full run:

1. Load the final P0 checkpoint and retain its exact CPU and H2H schedules.
2. Run 256 P1 steps on a clean RTX 4090 host.
3. Confirm FlexAttention reports `window=128`.
4. Confirm each batch has 128 distinct replay IDs and zero adjacent replay reuse inside an iterator
   epoch. Record any reuse on the first batch after an epoch boundary; do not claim the cooldown
   spans that boundary.
5. Confirm all losses, gradients, and parameters stay finite.
6. Confirm training consumes 131,072 frame tokens per optimizer step.
7. Confirm closed-loop decode rebuilds the newest 1,024 raw frames through every layer.
8. Reject any run with temporal KV enabled.
9. Record compile, validation-cache, loader, compute, RAM, VRAM, disk, and startup timing.

Use a 250 GB disk, at least 200 GB RAM, and compute capability exactly 8.9. Flag a projected total
above 3.5 hours.

```text
uv run scripts/launch_vast.py \
  --max-price 1.50 --disk 250 --min-vram 24 --min-ram 200 \
  --min-dlperf 120 --min-compute-cap 890 --max-compute-cap 890 \
  --data-gb 15 --upload-gb 1 --run-hours 0.5 -- \
  uv run experiments/023_mtp_heads.py \
  --cfg.require-flex --cfg.L-ctx 1024 --cfg.batch-size 128 \
  --cfg.attn-window 128 --cfg.no-eval-incremental-kv \
  --cfg.max-steps 256 --cfg.warmup-steps 32 \
  --cfg.val-every 0 --cfg.eval-every 0 --cfg.ckpt-every 0 \
  --cfg.final-eval-n-matchups 0 \
  --cfg.cache-limit-gb 128 --cfg.predownload 512 \
  --cfg.reservoir-capacity 4096 --cfg.prefetch-factor 2 \
  --comment p1-matched-swa-recompute-gate
```

## Full launch shape

```text
uv run scripts/launch_vast.py \
  --max-price 1.50 --disk 250 --min-vram 24 --min-ram 200 \
  --min-dlperf 120 --min-compute-cap 890 --max-compute-cap 890 \
  --data-gb 15 --upload-gb 1 --run-hours 3.5 -- \
  uv run experiments/023_mtp_heads.py \
  --cfg.require-flex --cfg.L-ctx 1024 --cfg.batch-size 128 \
  --cfg.attn-window 128 --cfg.no-eval-incremental-kv \
  --cfg.cache-limit-gb 128 --cfg.predownload 512 \
  --cfg.reservoir-capacity 4096 --cfg.prefetch-factor 2 \
  --cfg.final-h2h-reference-run \
  260807-164825_023_mtp_heads_gpt-d256-L8-h4-Lc256-a1024-full-recompute-o1.5.9.13-linear_ranked-anon-1_e0-normalized-aux-bc \
  --cfg.final-h2h-reference-label p0 --cfg.final-h2h-self-label p1 \
  --cfg.final-h2h-n-configs 64 \
  --comment p1-matched-swa-recompute
```

Audit the final command against the selected P0 configuration before launch. Verify the P0
`final.pt` object before using this command.

The current configuration parser and validator accept this P1 geometry. It resolves to
`gpt-d256-L8-h4-Lc1024-a1024-swa128-recompute-o1.5.9.13-linear` and processes 131,072 tokens per
step. The focused 023 suite passed 20 tests. The shared attention and closed-loop rolling-context
suites passed 38 tests and skipped six GPU-only cases on the local CPU host. This check does not
replace the clean-host runtime gate.

The final evaluator must apply the same configured decode dtype as periodic checkpoint workers.
The challenger and reference sides of H2H must also use the same actual dtype. Match-row and H2H
protocol data must record that dtype. This gate was added after finding that the old in-process final
path kept the live FP32 training weights while checkpoint-loaded policies used FP16.

Both the 256-step gate and full launch command passed no-rent launcher dry runs at commit `b422724`.
All flags, including the P0 H2H reference, were encoded correctly. The current qualifying offer was
an RTX 4090 with 1,008 GB RAM, 250 GB disk, DLPerf 125.5, and effective price $1.451 per hour. Select
the offer again when each run launches.

At commit `6aaa99b`, a direct configuration comparison passed validation and found exactly three
changes from P0: `L_ctx` 256 to 1,024, batch size 512 to 128, and attention window 0 to 128. Both
arms process 131,072 frames per step. Their per-layer attention-edge counts are 16,842,752 and
15,736,832. P1 temporal KV decoding is false. Repeat the launcher dry run at the final launch commit.

The 256-step gate command passed another no-rent launcher audit at commit `dadb0df`. The selected
offer was an RTX 4090 with 1,008 GB RAM, 250 GB disk, DLPerf 125.3, and effective price $1.451 per
hour. The encoded command preserved all planned gate fields.

After the concurrent-startup implementation, the same gate command passed a no-rent audit at commit
`1647cc2`. The encoded SHA and every experiment flag were correct. The selected offer remained the
RTX 4090 with 1,008 GB RAM, 250 GB disk, DLPerf 125.3, and effective price $1.451 per hour.

At commit `c4ae635`, the gate command still encoded every planned flag. No RTX 4090 offer met the
$1.50 effective hourly cap with 250 GB disk at 11:53 PDT. No instance was rented.

At commit `47540cc`, the gate command passed another no-rent audit. Three RTX 4090 offers met the
bounds. The best had 252 GB RAM, DLPerf 125.6, a 250 GB disk, and an effective price of
$0.781/hour. P1-old remained the only rented experiment job.

The 256-step gate launched from branch `exp/p1-matched-attention` on Vast instance `47112838` at
commit `3e9bae8`. The RTX 4090 host has 252 GB RAM, DLPerf 125.5, and a 250 GB disk at
$0.808/hour effective. The container became ready in 33 seconds. No other experiment job was
active.

That gate failed before the first batch. W&B run `b92jh5zw` contains only the parameter count. The
failure was in Mosaic Streaming 0.13.0, not the model: its shared-memory wrapper forwarded an extra
`self` argument to Python 3.14's bound resource-tracker method while the loader created worker
semaphores. The fix adds `hal/data/streaming_compat.py`, loads it from
`hal/training/dataloader.py`, and tests the exact forwarding signature in
`tests/test_dataloader.py`. The focused loader suite passes 20 tests. Relaunch the same gate after
committing this compatibility fix; do not treat instance `47112838` as speed or stability evidence.

## Evaluation

- Use the same deterministic 32 periodic and 96 final CPU character-pair boots as P0.
- Run 64 mirrored H2H configurations, or 128 games, against P0.
- Keep frame limits, matchup seeds, decode seeds, temperature, and concurrency fixed.
- Save checkpoints, match rows, replay files, worker logs, and decode protocol.
- Report per-offset and per-group NLL and accuracy, transition metrics, gradient interaction,
  stocks, damage, dead frames, terminal results, crashes, and wall time.
- Report paired H2H stock difference, non-tied win rate, confidence intervals, and ties.

Report the two CPU sweeps with separate uncertainty intervals. Do not call boot-and-ordinal row
alignment paired evidence because later instant-restart stages are random.

Closed-loop CPU and H2H results decide promotion. NLL or throughput alone cannot select P1.

## Interpretation

If P1 wins, the result supports the long-context local-attention package. It does not prove that
SWA alone caused the gain. If the package decision remains important and unclear, run the separate
I1 mask isolation at context 256 and batch 512.

Temporal KV decoding is a later systems ablation on the same P1 checkpoint. It must first pass
long-roll, mixed-reset, logit, probability, and fixed-seed sampled-action parity tests.
