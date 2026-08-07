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
context. The cache must be rejected when the receptive field reaches or exceeds the raw context.

## Files to change

- `experiments/023_mtp_heads.py`: add a manual-evaluation override for `eval_incremental_kv`, add a
  separate output directory for each manual evaluation, and record decode timing and protocol.
- `tests/experiments/test_023_mtp_heads.py`: test the override, invalid geometry rejection, output
  separation, and protocol logging.
- `tests/test_trunk.py`: keep exact full-versus-incremental trunk checks across left-edge eviction.
- `tests/test_closed_loop_rings.py`: keep full policy checks across warmup, rolling eviction, mixed
  slot resets, and match restart.

Do not change the model weights, attention mask, training loss, loader, or checkpoint format.

## Correctness gate

Run this gate before any Dolphin sweep:

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

## Evaluation

Run P2-R first and P2-K second with 96 scheduled boots each. Instant restart may produce more than
96 completed games. The retry code reruns failed boots only.

Write each arm to its own directory. Save:

- Match rows and replay files.
- Complete protocol data, including `eval_incremental_kv`.
- Total wall time and emulator boot time.
- Policy calls, frames decoded, model forwards, and model-forward wall time.
- Median, p95, and mean model time per decoded frame.
- Peak CPU RAM, VRAM, and GPU use when the host reports them correctly.

The policy metrics should match within paired sampling noise. Report stocks, damage, dead frames,
terminal results, crashes, and paired row deltas. A large policy difference is a correctness warning,
not evidence that the systems optimization improved the policy.

## Commands

The checkpoint path and output paths are placeholders until P1 finishes. The planned interface is:

```text
uv run experiments/023_mtp_heads.py \
  --eval-run P1_RUN --eval-checkpoint-name final.pt \
  --no-eval-incremental-kv \
  --eval-n-matchups 96 --eval-seed 0 \
  --wandb-run-id P1_WANDB_ID --wandb-label p2-recompute

uv run experiments/023_mtp_heads.py \
  --eval-run P1_RUN --eval-checkpoint-name final.pt \
  --eval-incremental-kv \
  --eval-n-matchups 96 --eval-seed 0 \
  --wandb-run-id P1_WANDB_ID --wandb-label p2-kv
```

Run both arms on the same retained Vast instance when possible. Do not include model download or
one-time process startup in model-forward latency.

## Decision

Keep full recomputation as the default. Promote temporal KV only if the correctness gate passes,
the closed-loop results show no material regression, and the saved time matters enough to justify
the extra stateful decode path.

If P1 loses to P0, retain P2 as a systems result. Do not let decode speed select the scientific E0
reference.
