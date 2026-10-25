# Production continuation

The user authorized preparation and launch on 2026-10-25. Resume checkpoint 32
and continue to update 512 without stopping at 64. Stable memory near 164 GiB
is accepted. Win-rate dips or plateaus are observations, not stop conditions.

Keep BF16, the production sampler, rewards, optimizer, batch sizes and schedule.
The launcher accepts an absolute endpoint and permits later extensions. Reaching
an endpoint saves state without consuming the final evaluation set.

## Fixed inputs

- Use the training settings in [production-256.json](production-256.json): eight
  B200s, 1,024 active matches per GPU, four groups, 512 rollout frames, 512 windows
  per optimizer step, eight microbatches, seed 65. Preserve the historical file;
  record the new endpoint separately.
- Checkpoint 32 producer: `fde1887c6f7170cd817f2497fa1582b071f62c71`.
  Manifest SHA-256:
  `206883a06d62321c672ed1a8141233de8d429e110947e4c83a0dfd2299508968`.
  Verify its manifest and 265 shards before the eight-GPU continuation.
- Restore model, critic, optimizers, scheduler, RNG, simulators, group state,
  sampler cursor, calibrated lambda and anchors. Do not repeat critic warmup
  or the initial head start. Preserve prior attempt records.
- Freeze one tested source commit. Pin Torch 2.11.0 and Triton 3.6.0. Record image,
  full package versions, driver, PDL/precision flags and input hashes. Use a new
  W&B attempt ID linked to checkpoint 32.

## 1. Prepare and validate the launch path

Remove the fixed 256-update restriction. The actor learning rate warms up over
32 optimizer steps, then stays constant; it does not depend on the endpoint.
Version the production identity to separate execution choices from immutable
training settings. Migrate the actual v1 identity without relaxing data or model checks.

Prepare two next-update controls on all ranks. Start fresh rank processes from
that same checkpoint, compare restored state and the next batch/update, then
rewind before production logging. Run this before the continuation so update 64
needs no stop or restart. Preserve rank-local hashes and numerical comparisons.

After a consistent CPU snapshot, write and compress each rank's checkpoint shards
in the background. Keep collectives on the training thread. Rank zero commits
the complete manifest and publishes it in the background. Bound retained snapshots
and propagate failures. Report snapshot, file-write and upload times separately.

Test progress while writes are blocked, snapshot independence, fresh-process
next-batch/update parity on every rank, extension without schedule reset,
off-cadence final checkpoints, and failure propagation. Run required repository
checks and affected CUDA tests before freezing source.

## 2. Prepare the cache on one B200

Run the integrated barrier regression with its independent FP64 oracle and cold
and warm compiler-cache paths, without skips. Build a new v4 production bundle
with tag `hal-inductor-reduction-barrier-v1` from the frozen source.

Use actual per-rank shapes, all four groups, both actor lengths and complete
forward/backward paths. Discard preparation updates. Require the saved cache in
a fresh process and repeat full-path measurement. Record cache identity/misses,
startup time, update time and memory. CUDA graphs need recapture after restart.

## 3. Continue on eight B200s

Resume checkpoint 32, complete the fresh-process resume check, then train toward
512. Observe the first actor update, former failure boundary at 44, and snapshot
rotation at 64 while the job continues. Verify checkpoint 64 becomes durable.

Keep normalization, selected-action gather and mask checks before actor mutation.
Track finite losses/gradients, rank progress, full-path memory, compilation/capture
counts and successful publication. No general eager/compiled equality requirement.

Expected performance is under 30 seconds per update: 4,194,304 match-frames per
update across eight ranks, about 140,000 frames/second at 30 seconds. Interpret
the user's slowdown limit as warmed update times above 45 seconds, sustained
over five ordinary updates. This is 50% longer than 30 seconds. Exclude startup,
resume verification, checkpoint snapshot and profiler windows. Also report total
throughput including checkpoint costs. Investigate before treating a transient
slowdown as terminal.

Continue through win-rate noise and plateaus. Inspect
`gameplay/pi_ref_win_rate_32_updates` with its completed-game count. Stop only for
an unrecoverable failure, invalid training, memory growth that threatens completion,
or a sustained slowdown beyond the limit. Preserve diagnostics and durable state.
Do not silently change precision or batch size, or retry a failing run.

## 4. Finish and evaluate

Keep checkpoints every 32 updates and snapshots every 64. At 512, drain background
writes and publication before releasing the training GPUs. Verify the final
manifest, referenced files and FunctionCall result.

Evaluate explicitly on one B200 against the pinned O64 reference. This amends the
old update-256-only policy: the final checkpoint is now 512. Keep the existing
[match protocol](production-protocol.md), grid, 384 games, seeds, starts, stages,
ports and conditions. Bind the held-out claim to final checkpoint and grid hashes
before play; preserve partial results on failure. Do not use these games to decide
whether to extend training.

## Monitoring and allocation limits

Record App, FunctionCall, container, source, checkpoint and W&B IDs. Follow startup
closely, then inspect progress every 5–10 minutes and at each checkpoint. Use
`global_step` and durable artifacts; missing metrics are not zero. Report the four
stage milestones. Keep retries disabled.

Bound cache preparation to one B200 for 60 minutes, training to eight B200s for
seven hours including startup/verification, and evaluation to one B200 for 60
minutes. These are ceilings, not predicted runtimes. At the listed
[Modal Function rates](https://modal.com/pricing), eight B200s with 64 CPU cores
and 512 GiB host memory cost about $57.11/hour. One B200 with 16 cores and 128 GiB
costs about $8.03/hour. Storage and egress are separate.

Including the launcher's six-minute margin and $3 build reserve per job, these
allocations reserve about $432.12. The prior ledger reserves $1,004.76. Use a
cumulative reservation ceiling of $1,450. Reservations are not confirmed billed
spend. Record each allocation and enforce its deadline; do not silently extend
limits after a failure.
