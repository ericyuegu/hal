# P2: temporal KV decode ablation

Updated: 2026-08-07

Status: strict parity failed; diagnostic rerun pending

## Question

How much closed-loop inference time does temporal KV caching save for the matched P1 model, and is
its behavior equivalent to rebuilding the complete raw rolling window?

P2 is a systems experiment. It does not train another model. Both arms load the same final P1
checkpoint.

## Arms

- P2-R: full recomputation of the newest 1,024 raw frames through all eight transformer layers.
- P2-K: one-frame temporal KV updates with an SWA window of 128 entries per layer.

Keep the checkpoint, FP16 decode setting, action temperature, matchups, boot seeds, CPU level, frame
budget, and concurrency fixed. Change only the decode path.

The comparison is valid only for the P1 geometry. With eight layers and an attention window of 128,
the newest output has a receptive field of 1,017 frames. This is smaller than the 1,024-frame raw
context. Reject the cache when the receptive field reaches or exceeds the raw context. Equality is
not safe: the full rolling-window path masks the oldest row's finite differences because their
predecessor has left the window, while the cached row originally saw that predecessor.

## Files to change

- `experiments/023_mtp_heads.py`: add a manual-evaluation override for `eval_incremental_kv`, a
  checkpoint parity mode, separate output directories, and decode timing and protocol records.
- `tests/experiments/test_023_mtp_heads.py`: test the override, invalid geometry rejection, output
  separation, and protocol logging.
- `tests/test_trunk.py`: keep exact full-versus-incremental trunk checks across left-edge eviction.
- `tests/test_closed_loop_rings.py`: keep full policy checks across warmup, rolling eviction, mixed
  slot resets, and match restart.

Do not change the model weights, attention mask, training loss, loader, or checkpoint format.

## Correctness gate

Run this gate before any closed-loop sweep:

1. Load the same P1 checkpoint twice, once per decode path.
2. Check FP32 logits before and after the raw buffer first rolls.
3. Check FP16 logits under the same sequences.
4. Include different left padding and reset times within one batch.
5. Check all four group probability vectors.
6. Check fixed-seed sampled actions for every frame.
7. Run longer than 1,024 frames so the test covers repeated eviction.
8. Require finite outputs and no stale state after a slot reset.

Use exact equality for sampled action bytes. Set explicit numerical tolerances for logits and report
the largest observed error. If sampled actions differ, reject KV equivalence and promotion. A
report-only closed-loop sweep may still measure speed and behavioral drift after the failed record
is saved.

The checkpoint gate uses three deterministic model-valid feature streams. Slot 0 has no mid-run
reset, slot 1 resets once, and slot 2 resets twice. It runs for `2 * L_ctx + 17` frames. Slot 0
therefore crosses more than `L_ctx` consecutive raw-window evictions, while the other slots test
asynchronous cold starts and mixed cache lengths. It compares every frame in FP32 with
`atol=rtol=1e-4` and in FP16 with `atol=rtol=5e-3`. These values were fixed before the final P1
checkpoint existed. The existing ring tests separately prove that raw online observations build
the same model inputs as training windows.

## Evaluation

Run P2-R first and P2-K second with 96 requested matchups and at most 32 concurrent boots in each
arm. Instant restart may produce more than 96 completed games. The retry code reruns failed boots
only.

Write each arm to its own directory. Save:

- Match rows and replay files.
- Complete protocol data, including `eval_incremental_kv`.
- Total wall time and emulator boot time.
- Policy calls, frames decoded, model forwards, and model-forward wall time.
- Median, p95, and mean time per backbone-forward call, plus total time per forward batch row.
- Peak CPU RAM, VRAM, and GPU use when the host reports them correctly.

The policy metrics should match within independent sampling noise. Report stocks, damage, dead
frames, terminal results, and crashes for each arm. Do not treat boot-and-ordinal row alignment as a
paired estimate because later restart stages are random. A large policy difference is a correctness
warning, not evidence that the systems optimization improved the policy.

## Commands

The source run is
`260807-220243_023_mtp_heads_gpt-d256-L8-h4-Lc1024-a1024-swa128-recompute-o1.5.9.13-linear_ranked-anon-1_p1-matched-swa-recompute`.
Its W&B ID is `46zi7fgo`. Do not launch P2 until its final checkpoint and final P1 evidence are
verified in R2.

Run parity and both evaluation arms in one job so the GPU, CPU, emulator setup, and local checkpoint
stay fixed:

```text
uv run scripts/launch_vast.py \
  --max-price 1.50 --disk 80 --min-vram 24 --min-ram 200 \
  --min-dlperf 120 --min-compute-cap 890 --max-compute-cap 890 \
  --data-gb 0 --upload-gb 1 --run-hours 1.5 -- \
  bash -c '
    set -euo pipefail
    p1_run=260807-220243_023_mtp_heads_gpt-d256-L8-h4-Lc1024-a1024-swa128-recompute-o1.5.9.13-linear_ranked-anon-1_p1-matched-swa-recompute
    uv run experiments/023_mtp_heads.py \
      --parity-run "$p1_run" --parity-checkpoint-name final.pt \
      --parity-frames 2065 --parity-slots 3 --parity-seed 0
    uv run experiments/023_mtp_heads.py \
      --eval-run "$p1_run" --eval-checkpoint-name final.pt \
      --eval-decode recompute --eval-n-matchups 96 --eval-max-parallel 32 --eval-seed 0 \
      --wandb-run-id 46zi7fgo --wandb-label p2-recompute
    uv run experiments/023_mtp_heads.py \
      --eval-run "$p1_run" --eval-checkpoint-name final.pt \
      --eval-decode kv --eval-n-matchups 96 --eval-max-parallel 32 --eval-seed 0 \
      --wandb-run-id 46zi7fgo --wandb-label p2-kv
  '
```

Both arms use 32 concurrent boots. This matches the official P0 evaluation and removes host CPU
count as a policy comparison variable. The 80 GB disk holds the image, ISO, checkpoint, compile
cache, and both evaluation outputs. P2 does not download the training dataset. Do not include model
download or one-time process startup in model-forward latency.

The complete command passed a no-rent launcher audit at commit `977ecf4`. It uses a non-login shell
so the startup environment stays intact. The encoded payload kept the parity, recompute, and KV
order, exact run name, W&B ID, seeds, and 96-boot counts. Two RTX 4090 offers met the hardware
limits. The best had 252 GB RAM, DLPerf 125.7, an 80 GB disk, and an effective price of $0.824 per
hour. Search again after P1 finishes; this audit did not rent a box.

The command passed again at commit `6d247bc`. The encoded payload kept the same order and fields.
One RTX 4090 offer met the limits. It had 252 GB RAM, DLPerf 125.6, an 80 GB disk, and an effective
price of $0.824 per hour. No instance was rented.

## Implementation status

The manual evaluator now accepts `--eval-decode checkpoint`, `recompute`, or `kv`. The override does
not modify the saved checkpoint configuration. It validates the requested decode mode against the
checkpoint attention geometry and records the selected mode in the evaluation protocol and H2H
metadata.

The evaluator now records policy calls, active slot-frames, backbone-forward calls, forward batch
rows, total backbone time, mean/median/p95 forward latency, time per forward row, and total sweep
wall time. CUDA timing uses events, so it does not add a synchronization after every forward. CPU
timing uses the monotonic performance clock. A parity test proves that telemetry does not change
full-context sampled actions. Each evaluation writes these values to `metrics.json` next to
`match_rows.json`, and also returns them for W&B logging. Do not infer decode speed from total
evaluator wall time because Dolphin boot and emulation time dominate that number.

The parity mode downloads a named run checkpoint, checks FP32 and FP16 full-versus-KV behavior,
writes `manual_evals/p2-parity/decode_parity.json`, and records the checkpoint size and SHA-256. It
uploads the record even when a numerical gate fails and returns a failing process status on any
nonfinite value, tolerance failure, or sampled-action mismatch. A CPU test covers rolling eviction
and asynchronous reset schedules with a small model. Each precision result also records synchronized
wall time and comparisons per second, so the gate's own cost is visible.

The current focused CPU suite passes 83 tests. It covers the experiment entry point, transformer
trunk, and closed-loop rolling rings. The FP32 and FP16 GPU parity gate remains blocked on the final
P1 checkpoint.

After the H2H artifact fields were clarified, the complete repository suite passed 892 tests at
commit `d062eb4`. The change does not affect P2 inference, but the full pass checks checkpoint,
evaluation, and artifact compatibility around it.

At commit `16c3e46`, the focused P2 suite passed 77 tests and skipped six GPU-only tests. The
complete repository suite passed 892 tests in 135 seconds. Ruff also passed on the P2 experiment,
rolling-context code, H2H analysis, and their focused tests. The remaining GPU-only check is the
planned parity gate on the final P1 checkpoint.

The parity audit found that every synthetic slot reset near the raw-window boundary. That covered
only a short run of post-eviction frames despite the intended repeated-eviction claim. Commit
`1d95e4f` keeps slot 0 continuous for all `2 * L_ctx + 17` frames, resets slot 1 once, and resets
slot 2 twice. This gives slot 0 more than `L_ctx` consecutive evictions while retaining mixed reset
and cache-length coverage. The corrected focused suite passed 78 tests and skipped the six GPU-only
cases. The complete repository suite passed 893 tests in 136 seconds.

The P1 artifact audit found that its in-process final sweep used 96 concurrent boots while the P0
reference used 32. The manual evaluator now accepts `--eval-max-parallel`, validates it against the
requested boot count, and records it in the protocol. Both P2 arms are pinned to 32. This makes the
recompute arm the clean P1 CPU rerun needed for reference selection. The
`test_023_mtp_heads.py` suite passed 43 tests. The complete suite passed 896 tests in 135 seconds
outside the restricted sandbox, including W&B offline and Dolphin integration tests. Ruff passed.

The P1 H2H audit exposed a separate replay-evidence bug. Budget-cut replays can end with one torn
frame and make peppi reject the port arrays. Future H2H collection now trims that incomplete frame,
retries the parser, and fails if a completed game still lacks input statistics. The repair preserves
all complete frames and the stamped model identity. The focused H2H, finalization, and paired suites
passed 63 tests. The complete repository suite passed 898 tests in 135 seconds.

H2H record schema 3 stores `replay_trimmed` per game. Older records load this field as false.
After the schema change, 64 focused tests and all 899 repository tests passed.

## Decision

Keep full recomputation as the default. Promote temporal KV only if the correctness gate passes,
the closed-loop results show no material regression, and the saved time matters enough to justify
the extra stateful decode path.

If P1 loses to P0, retain P2 as a systems result. Do not let decode speed select the scientific E0
reference.

P1 did not pass its policy promotion gate. Its 128-game mirrored H2H mean was -0.031 stocks per
game, and the mean summed difference over 64 paired configurations was -0.062. P0 remains E0. P2
still runs because it measures the cost and correctness of temporal KV decoding on the P1 SWA
geometry. Neither P2 arm can change the E0 selection.

The P1 source run and its repaired H2H package are complete. W&B is finished, the source Vast
instance is gone, and Rclone verified all 134 audited files. P2 may proceed after one fresh no-rent
launcher check at the exact pushed commit.

The complete command passed a fresh no-rent check at commit `cface89`. The encoded payload kept
parity first, recompute second, KV third, the correct P1 run and W&B IDs, 2,065 parity frames, three
slots, 96 boots per evaluation, and 32 concurrent boots. Three RTX 4090 offers qualified. The best
had 252 GB RAM, DLPerf 125.6, 80 GB disk, and an effective price of $0.824 per hour. No instance was
rented. Repeat this check after committing this record, then launch from that unchanged SHA.

The final no-rent check passed at pushed commit `335b7e7`. It preserved the same payload and found
three qualifying RTX 4090 offers. P2 launched from that exact commit on Vast instance `47129969`.
The selected host has 252 GB RAM, DLPerf 125.6, an 80 GB disk, and an effective price of $0.824 per
hour. Vast reported the instance ready at 18:01 PDT. This is the only active experiment job.

## First parity result

The strict parity gate failed before either closed-loop arm ran. This was a model result, not an
infrastructure or finite-value failure. The artifact is at
`manual_evals/p2-parity/decode_parity.json` under the P1 run. Its SHA-256 is
`4bc514a79dde89c011c387ad19065bb05778c6613c5501cf734bf0a46278d424`.

The test used the final P1 checkpoint, 2,065 frames, three slots, and 6,195 comparisons per dtype.
All values were finite.

- FP32: maximum hidden error `0.02022`, maximum group-logit error `0.08052`, 14 sampled-action
  mismatches, and failure at the `1e-4` tolerance.
- FP16: maximum hidden error `0.03174`, maximum group-logit error `0.09961`, 32 sampled-action
  mismatches, and failure at the `5e-3` tolerance.

The original command used `set -e`, so it stopped after uploading the failed gate. The recompute
and KV closed-loop sweeps did not start. The instance stopped at 18:03 PDT. We verified the parity
artifact in R2 and then destroyed instance `47129969`; its disk is not billing.

The first result compares two kernels as well as two decode methods: full recomputation uses
FlexAttention, while incremental decoding uses dense scaled-dot-product attention. A CPU check with
the trained P1 trunk forced both paths through dense attention. Across a 256-frame sequence, the
maximum hidden difference was `3.48e-5`, and the last-frame difference was `1.87e-5`. This is much
smaller than the GPU result, but it still exceeds `1e-5`. It does not prove that the GPU error comes
only from the kernel change.

The next parity record will compare three paths at fixed frames:

1. Full FlexAttention recomputation.
2. Full dense-attention recomputation.
3. Incremental dense-attention KV decoding.

This separates Flex-versus-dense numerical drift from cache-specific drift before and after the raw
window rolls. The record schema is version 2. Strict failure remains the default. A report-only flag
allows the already-disqualified KV arm to finish its closed-loop speed and behavior measurements;
it does not turn a failed parity result into a pass or permit KV promotion.
