# O59 v4 production run

Train the 246,862,205-parameter return-conditioned policy from seed zero through
update 98,304. This is a fresh stable run. A later decay continuation requires
a separate launch.

## Recipe

The trunk has 12 layers; the temporal decoder has six. Both have width 1024 and
16 attention heads. The decoder MLP has width 4096. The action heads and
trunk-skip heads retain their nonlinear projections and normalization.

A return conditioner embeds the discounted reward over future frames 1 through
60 into 128 channels. Each decoder block receives attention and MLP scale/shift
vectors before its normalized projections. The projections start at zero. Their
biases let a valid zero return differ from absent conditioning. Absent values
are replaced with zero before arithmetic, then the projected vectors are masked.
Return conditioning is enabled with independent 20% dropout per context position.

The trunk, critic, history key/value projection, and trunk-skip heads receive no
return input. History cross-attention remains after the first decoder block.
Action-group FiLM retains its `tanh`; return FiLM has no `tanh`. Residual depth
scaling is unchanged.

AWR uses beta 150, maximum weight 10, and gamma 0.99855. It retains next-frame
alignment, 32 sampled policy prefixes per window, and all 128 suffix positions
for the critic and AWR normalizer. Return conditioning uses the sampled context
position itself. Batch 512, BF16, Muon/AdamW, and WSD schedule rules are unchanged.
The complete 44-source policy-world-v8 corpus, manifests, schemas, identity
artifacts, and natural replay-count mixture remain fixed.

The positive-return p90 freezes after the first 65,536 consumed training windows.
It uses available labels at the final context position before dropout, with
linear quantile interpolation. Checkpoints store the ordered values, validity,
replay IDs, target, and calibration hash. Prefetch does not advance calibration
until training consumes a batch.

## Launch

Use the execution choices in [the throughput report](throughput.md). The launch
uses action-embedding reuse, `reduce-overhead`, production diagnostics, and
eager Muon. History reuse and groupwise loss remain disabled.

```sh
uv run scripts/launch_modal.py \
  --gpu B200 --closed-loop-gpu L40S \
  --cpu 32 --cpu-limit 48 \
  --memory-gib 128 --memory-limit-gib 384 --disk-gib 2048 \
  --timeout-hours 24 \
  -- uv run experiments/059_muon_history_decoder.py train \
  --cfg.train-compile-mode reduce-overhead \
  --stop-after-update 98304 \
  --comment v4-b512
```

Pass the stop explicitly so retries keep the same boundary. The 24-hour timeout
applies to each attempt. The launcher retains infrastructure recovery and refuses
to restart a terminal training failure. It requires a clean, pushed commit and
records Git, Modal App, FunctionCall, launch ID, and state Volume identities.

The experiment identity is `059_muon_history_decoder_v4`, checkpoint format 3,
and data protocol `o59-replay-ring-v2`. Older checkpoints are rejected. Resume
restores the model, optimizer, scheduler, loader cursor, global RNGs, prefix RNG,
identity and return maskers, and calibration. Checkpoints occur only when
prefetched batches have been consumed. Disabling conditioning keeps the same
parameter structure.

## Evaluation and monitoring

- Warmup ends at update 4,096. AWR starts at update 4,097.
- Durable checkpoints occur every 2,048 updates; offline validation every 4,096.
- L40S gameplay evaluations use p90 for 96 matchups every 8,192 updates and at
  completion. Evaluation rejects incomplete p90 calibration.
- The final checkpoint also gets a separate, matched unconditioned evaluation.
  Its directory ends in `-unconditioned`; its W&B namespace is
  `eval_unconditioned`. P90 uses `eval`. Both artifacts record the mode, numeric
  target, calibration hash, checkpoint hash, and matchup schedule.
- Offline validation and its rollout diagnostics use actual available labels.
  Gameplay evaluation determines policy quality; offline loss is diagnostic.
- Completion requires update 98,304, `final.pt`, the numbered checkpoint,
  successful launcher state, and both final evaluation artifacts.

A new Sol babysitter receives the exact Modal and W&B identities after launch.
It should establish this run's own warmed throughput, MFU, loader-wait, and memory
baselines. The fixed-bank benchmark does not measure sustained decoding from all
44 sources. Exclude compilation, validation, checkpoints, and profiling from
throughput comparisons. Track the current container separately from the logical
FunctionCall, since recovery changes the attempt identity.

[Validation results](validation.md) include the required repository checks,
Dolphin integration, CUDA parity, and exact loader-to-update resume coverage.
