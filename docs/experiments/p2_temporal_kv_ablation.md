# P2: temporal KV decode ablation

Updated: 2026-08-07

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
the largest observed error. Stop P2 if sampled actions differ.

The checkpoint gate uses three deterministic model-valid feature streams with different reset
times. It runs for `2 * L_ctx + 17` frames, so every slot crosses repeated raw-window eviction. It
compares every frame in FP32 with `atol=rtol=1e-4` and in FP16 with `atol=rtol=5e-3`. These values
are fixed before the final P1 checkpoint exists. The existing ring tests separately prove that raw
online observations build the same model inputs as training windows.

## Evaluation

Run P2-R first and P2-K second with 96 scheduled boots each. Instant restart may produce more than
96 completed games. The retry code reruns failed boots only.

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
  bash -lc '
    set -euo pipefail
    p1_run=260807-220243_023_mtp_heads_gpt-d256-L8-h4-Lc1024-a1024-swa128-recompute-o1.5.9.13-linear_ranked-anon-1_p1-matched-swa-recompute
    uv run experiments/023_mtp_heads.py \
      --parity-run "$p1_run" --parity-checkpoint-name final.pt \
      --parity-frames 2065 --parity-slots 3 --parity-seed 0
    uv run experiments/023_mtp_heads.py \
      --eval-run "$p1_run" --eval-checkpoint-name final.pt \
      --eval-decode recompute --eval-n-matchups 96 --eval-seed 0 \
      --wandb-run-id 46zi7fgo --wandb-label p2-recompute
    uv run experiments/023_mtp_heads.py \
      --eval-run "$p1_run" --eval-checkpoint-name final.pt \
      --eval-decode kv --eval-n-matchups 96 --eval-seed 0 \
      --wandb-run-id 46zi7fgo --wandb-label p2-kv
  '
```

The 80 GB disk holds the image, ISO, checkpoint, compile cache, and both evaluation outputs. P2 does
not download the training dataset. Do not include model download or one-time process startup in
model-forward latency.

The complete command passed a no-rent launcher audit at commit `c0606a7`. The encoded payload kept
the parity, recompute, and KV order, exact run name, W&B ID, seeds, and 96-boot counts. Two RTX 4090
offers met the hardware limits. The best had 252 GB RAM, DLPerf 125.7, an 80 GB disk, and an
effective price of $0.824 per hour. Search again after P1 finishes; this audit did not rent a box.

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

The focused CPU suite passes 134 tests and skips six GPU-only tests. The FP32 and FP16 GPU parity
gate remains blocked on the final P1 checkpoint.

## Decision

Keep full recomputation as the default. Promote temporal KV only if the correctness gate passes,
the closed-loop results show no material regression, and the saved time matters enough to justify
the extra stateful decode path.

If P1 loses to P0, retain P2 as a systems result. Do not let decode speed select the scientific E0
reference.
