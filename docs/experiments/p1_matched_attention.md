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

The replacement gate launched on Vast instance `47115519` from commit `0d60273be3`. The RTX 4090
host has 252 GB RAM, DLPerf 125.5, and a 250 GB disk at $0.808/hour effective. The container became
ready in 22 seconds. The command matches the first gate, except the run comment is
`p1-matched-swa-recompute-gate-r2` so its evidence cannot be confused with the failed run.

The full-run command passed a no-rent audit at commit `d899415`. The payload uses full recomputation,
the P0 final checkpoint for H2H, 64 mirrored H2H configurations, and every planned data setting.
Three RTX 4090 offers met the bounds. The best had 252 GB RAM, DLPerf 125.6, a 250 GB disk, and an
effective price of $0.781/hour. Select the offer again only after the replacement gate passes.

The replacement gate passed. W&B run `6ydiy4kq` finished normally in 178 seconds, and Vast removed
instance `47115519` after success. The run used FlexAttention with window 128 and full-recompute
evaluation mode. It consumed exactly 256 batches of 128 replays, or 131,072 frames per step. Every
logged batch had 128 distinct replay IDs. Adjacent replay reuse was zero; the gate did not reach an
epoch boundary, so it provides no boundary-reuse sample.

All logged losses, objectives, gradient norms, step times, and loader waits were finite. All 60
floating checkpoint tensors were finite. Over steps 100 through 255, median step time was 0.296
seconds, mean was 0.312 seconds, p95 was 0.386 seconds, and the maximum was 0.979 seconds. Median
loader wait was 0.138 seconds and mean loader wait was 0.150 seconds. Median throughput was 432.9
samples/s, or about 443,000 frame tokens/s. Step 0 took 54.7 seconds, including compilation, and
waited 19.2 seconds for its batch. The validation cache finished in 9.0 seconds while that work ran.

The 11 W&B system samples are too sparse for a utilization claim on this short run. Sampled GPU use
averaged 7.9% and peaked at 53%, while allocated VRAM reached 12.0 GB and power reached 214 W. The
trainer process reached 8.4 GB RAM, and disk use reached 16.6 GB. The low GPU sample must still be
reported, but the measured step time projects 16,384 training steps to about 81 minutes. The prior
25-minute P1 recompute sweep and the planned periodic and H2H work leave the full run below the
3.5-hour gate.

The gate's final validation NLL at offsets 1, 5, 9, and 13 was 1.543, 3.677, 4.553, and 5.053 bits
per frame. This is a 256-step systems result, not a policy comparison. R2 contains the 56,698,679-byte
`final.pt`; its SHA-256 is `028658c9aec8ac082392af16fc0c47d83e089ded8826e2589197df4f3e19d115`.
The checkpoint records step 256, W&B ID `6ydiy4kq`, action vocabulary 1,024, and temporal KV off.

The full 16,384-step run launched on Vast instance `47117879` from commit `a7a1f6b9a2`. The RTX
4090 host has 252 GB RAM, DLPerf 125.6, and a 250 GB disk at $0.781/hour effective. The container
became ready in 7 minutes 24 seconds. No other Vast experiment job was active. The launch command
uses full recomputation, the verified P0 final checkpoint for H2H, and 64 mirrored H2H
configurations.

At the first 30-minute audit, W&B run `46zi7fgo` had reached step 4,764. All logged loss,
objective, gradient, and timing values were finite. Every batch had 128 distinct replays. One replay
was reused at step 3,506, exactly when the loader ended an epoch after 448,768 emitted windows; no
other adjacent reuse occurred. Over the latest 500 steps, median step time was 0.256 seconds and
median loader wait was 0.103 seconds. The loader still consumed about 40% of step time, but the
total step was faster than the gate.

The step-4,096 evaluation finished in 500 seconds. All 32 boots succeeded, producing 47 active
matches and one zero-active tail. Stocks taken and lost per active minute were 0.609 and 1.635.
Damage dealt and taken per active minute were 106.8 and 137.8. This is weak early policy evidence,
not a stopping rule. R2 contains both 56.7 MB step checkpoints, the match rows, and replay files.

At the same training step, P1 had slightly lower validation NLL than P0 at every offset. P1 versus
P0 was 1.105 versus 1.115 bits at offset 1, 2.817 versus 2.851 at offset 5, 3.584 versus 3.626 at
offset 9, and 4.069 versus 4.115 at offset 13. Most of the primary difference came from main stick:
0.691 versus 0.700 bits. P1's training NLL was higher, 1.130 versus 1.086 bits. These small offline
differences do not override the weak early closed-loop result.

The comparison uses the same complete validation split. P1's startup log records 10 batches and
1,192 samples. P0's 512-sample batch geometry also exhausts that 1,192-sample split before its
32-batch cap. The current branch replaces the batch-count cap with an exact `val_n_samples=1192`
contract. This keeps validation coverage fixed if the training batch changes and fails if the
pinned split is unexpectedly short.

This audit found an evidence-retention bug in the launch commit: periodic evaluation uploaded
replays and `match_rows.json`, but left `metrics.json`, the worker result, and the worker log on the
temporary instance. The current branch now queues all four evidence types, including partial files
after a worker failure. The focused 023 suite passes 37 tests. The active run still uses its launch
commit. Its step-4,096 `metrics.json`, worker result, and worker log were recovered through Vast's
container-copy path and uploaded to their intended R2 keys. Their verified MD5 values are
`3d1586f01faf4d69dbec76317aec3cbf`, `d3196b1466b4c08bffa6252fc16b1086`, and
`30194d3c89fc514bea1ba8d2e72bf1b5`. Repeat this recovery for later periodic evaluations. W&B also
retains the complete numeric metrics.

At the second audit, the run had reached step 9,743. All tracked numerical values remained finite,
and every batch still contained 128 distinct replays. The step-3,506 epoch boundary remained the
only adjacent replay reuse. Over the latest 500 steps, median step time was 0.257 seconds, median
loader wait was 0.105 seconds, and median throughput was 498.2 samples per second. The run remained
within the 3.5-hour budget.

The step-8,192 validation NLL at offsets 1, 5, 9, and 13 was 1.062, 2.715, 3.457, and 3.935 bits per
frame. Its CPU evaluation finished in 570 seconds. It completed all 32 requested boots, produced 41
active matches and one zero-active tail, and reported no crash. Stocks taken and lost per active
minute were 0.687 and 1.518. Damage dealt and taken per active minute were 126.6 and 122.8. These
rates improved from step 4,096, but they remained weak and did not change the predeclared final-step
rule.

At the same step, P0 reported 0.893 stocks taken and 1.323 lost per active minute. P1 therefore
trailed P0 on both stock rates at this checkpoint. The separate 95% intervals overlap: P0 versus P1
was `[0.755, 1.029]` versus `[0.560, 0.800]` for stocks taken and `[1.130, 1.533]` versus
`[1.355, 1.697]` for stocks lost. These are independent small sweeps, not a paired test. Damage was
closer: P0 dealt 131.1 and took 122.4 per active minute, while P1 dealt 126.6 and took 122.8.

One boot first loaded Zelda when the schedule requested Sheik. The evaluator recorded that failed
start and retried it. The final metrics still contain 32 completed boots. The recovered log contains
the mismatch and the successful retry, so this behavior is auditable.

The step-8,192 checkpoint is present in R2 and is 56,706,005 bytes. R2 also contains 43 replay files,
the match rows, and the metrics. The three files affected by the launch commit's retention bug were
recovered from the live instance and uploaded under the canonical `runs/` prefix. The verified MD5
values for `metrics.json`, the worker result, and the worker log are
`57088ebd3bf2ceb5f894ea286a3e3f1d`, `7b1390403e2252afb0923937188e305c`, and
`66a3f817cdac725b5008b133d3fbe918`.

At the third audit, the run had reached step 14,615. All tracked values remained finite. Every batch
still contained 128 distinct replays, and the step-3,506 boundary remained the only adjacent replay
reuse. Over the latest 500 steps, median step time was 0.253 seconds, median loader wait was 0.102
seconds, and median throughput was 506.9 samples per second.

The loader pauses for about 9 seconds at each epoch boundary. The stalls occurred at steps 3,506,
7,012, 10,518, and 14,024, each after 448,768 emitted windows. This is deterministic epoch startup,
not a random I/O failure. It costs about 36 seconds over the full run. A future continuous-epoch
iterator could hide it, but it is not worth changing the active experiment.

At step 12,288, validation NLL at offsets 1, 5, 9, and 13 was 1.027, 2.630, 3.350, and 3.816 bits
per frame. By step 14,336 it reached 1.015, 2.601, 3.312, and 3.776. The step-12,288 CPU evaluation
finished in 553 seconds. It completed all 32 boots, produced 42 active matches and no zero-active
tail, and reported no crash. Stocks taken and lost per active minute were 0.751 and 1.470. Damage
dealt and taken were 124.3 and 124.7.

P0 at step 12,288 reported 0.813 stocks taken and 1.372 lost per active minute. Its intervals overlap
P1's intervals on both rates, so the difference is not resolved by these small independent sweeps.
P1 remains slightly worse on the point estimates. The final CPU sweep and paired H2H remain the
decision evidence.

The 56,706,005-byte step-12,288 checkpoint is present in R2. R2 also contains 43 replay files, the
match rows, and the metrics. The recovered metrics, worker result, and worker log have verified MD5
values `cc6a1bc628725bc630b6d91f8e9ecdad`, `7c6ab44f763b2b9c4ea6958f47fdb070`, and
`f0f467c8623afba02e08bdc46fa164bc`.

Training completed all 16,384 batches. Every logged batch contained 128 distinct replays, and the
only adjacent replay reuse remained the one replay at the first epoch boundary. All logged losses,
objectives, gradients, and timing values were finite. Final validation NLL at offsets 1, 5, 9, and
13 was 1.011, 2.590, 3.297, and 3.761 bits per frame.

The final checkpoint is present in R2. It is 56,698,807 bytes and has SHA-256
`8e9b04c91aa76d1ba49a910c82f1328bc1b0dc3ce7dabf3e9018cb556d964148`. It records step 16,384,
W&B ID `46zi7fgo`, context 1,024, SWA window 128, action vocabulary 1,024, execution horizon 1,
FP16 evaluation, and temporal KV off. All 60 floating model tensors and all 66 floating optimizer
tensors are finite. At this audit, the final CPU and H2H directories had not uploaded yet, and the
Vast instance and W&B run were still active. Do not launch P2 until both packages are complete and
verified.

## Evaluation

- Use the same deterministic 32 periodic and 96 final CPU character-pair boots as P0.
- Run 64 mirrored H2H configurations, or 128 games, against P0.
- Keep frame limits, matchup seeds, decode seeds, temperature, and concurrency fixed.
- Save checkpoints, match rows, replay files, periodic worker logs, and decode protocol.
- Report per-offset and per-group NLL and accuracy, transition metrics, gradient interaction,
  stocks, damage, dead frames, terminal results, crashes, and wall time.
- Report paired H2H stock difference, non-tied stock-lead rate, confidence intervals, and ties.

Report the two CPU sweeps with separate uncertainty intervals. Do not call boot-and-ordinal row
alignment paired evidence because later instant-restart stages are random.

Closed-loop CPU and H2H results decide promotion. NLL or throughput alone cannot select P1.

Periodic CPU sweeps run in worker processes and retain `eval_results/step_*.json` and
`eval_logs/step_*.log`. The final CPU sweep runs in the training process. Its complete artifact is
`replays/final/metrics.json`, `replays/final/match_rows.json`, and the replay files, with the same
metrics also logged to W&B and stdout. It does not create a separate final worker result or log.

The active launch SHA predates the H2H field rename. In its `matches.jsonl`, `winner_port` and
`winner_model` mean the model ahead on stocks at the frame budget; they do not mean a completed-game
winner. The current `MatchRecord.from_dict` maps those fields to `stock_leader_port` and
`stock_leader_model`. Recompute the final table with the current loader and report actual knockouts
through `decided_by_knockout`. Treat any old W&B `wins` field as stock leads.

## Reference selection rule

Keep P0 as the downstream E0 reference unless P1 passes all of these screening checks:

- No new crash or artifact failure.
- Final CPU stocks taken minus stocks lost per active minute is better than P0's point estimate.
- Paired H2H mean stock difference per configuration is positive.
- The H2H config sign test has more P1-ahead configs than P1-behind configs.

Report all intervals and ties even when these point-estimate gates pass. One seed is not enough for a
general architecture claim. If CPU and H2H disagree, keep P0 and call P1 inconclusive. P2 decode
speed cannot override a policy-quality loss.

## Interpretation

If P1 wins, the result supports the long-context local-attention package. It does not prove that
SWA alone caused the gain. If the package decision remains important and unclear, run the separate
I1 mask isolation at context 256 and batch 512.

Temporal KV decoding is a later systems ablation on the same P1 checkpoint. It must first pass
long-roll, mixed-reset, logit, probability, and fixed-seed sampled-action parity tests.
