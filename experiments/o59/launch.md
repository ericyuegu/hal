# O59 v5 production run

Train the 246,862,205-parameter return-conditioned policy from seed zero through
update 98,304. This is a fresh stable run. A decay continuation requires a
separate launch.

## Recipe

The trunk has 12 layers. The temporal decoder has six layers. Both use width
1024 and 16 attention heads. The decoder MLP width is 4096. The four independent
action heads and the trunk-skip heads keep their 1024-wide hidden layers.

Within each frame, the decoder uses this order:

```text
c_stick -> main_stick -> triggers -> buttons
```

Each action head first applies its RMSNorm to the shared decoder state. For all
groups after `c_stick`, one learned projection reads the cumulative embeddings
of the earlier groups and produces scale and shift vectors. The head applies
`normalized * (1 + tanh(scale)) + shift` and sends that result directly through
its MLP. It does not normalize a second time. Teacher forcing uses the true
earlier groups. Sampling and rollout paths use sampled or forced earlier groups.
The trigger-to-button legality mask is unchanged.

The return conditioner embeds discounted reward over future frames 1 through 60
into 128 channels. Each decoder block receives attention and MLP scale and shift
vectors before its normalized projections. The final projections start at zero.
Biased projections let a valid zero return differ from absent conditioning.
Return conditioning is enabled with independent 20% dropout per context
position. The trunk, critic, history key/value projection, and trunk-skip heads
receive no return input.

AWR uses beta 150, maximum weight 10, and gamma 0.99855. It retains next-frame
alignment, 32 sampled policy prefixes per window, and all 128 suffix positions
for the critic and AWR normalizer. Return conditioning uses the sampled context
position itself. Batch 512, BF16, Muon/AdamW, and the WSD schedule are unchanged.
The run uses all 44 policy-world-v8 sources with seed 0.

The positive-return p90 freezes after the first 65,536 consumed training
windows. Checkpoints store its ordered values, validity, replay IDs, target, and
hash. Calibration observes available final-position labels before return
dropout and only after training consumes the batch.

## Launch

The B200 comparison found no repeatable execution winner, so production uses the
single plain training path. The rejected benchmark implementations and profiler
hooks are not in the production source.

```sh
uv run scripts/launch_modal.py \
  --gpu B200 --closed-loop-gpu L40S \
  --cpu 32 --cpu-limit 48 \
  --memory-gib 128 --memory-limit-gib 384 --disk-gib 2048 \
  --timeout-hours 24 \
  -- uv run experiments/059_muon_history_decoder.py train \
  --cfg.train-compile-mode reduce-overhead \
  --stop-after-update 98304 \
  --comment v5-b512
```

The explicit stop keeps retries on the same boundary. The launcher requires a
clean, pushed commit and records the Git SHA, Modal App, FunctionCall, launch ID,
and state Volume identities.

The experiment identity is `059_muon_history_decoder_v5`, checkpoint format 4,
and data protocol `o59-replay-ring-v2`. Older checkpoints are rejected. Resume
restores the model, optimizer, scheduler, loader cursor, global RNGs, prefix RNG,
identity and return maskers, and calibration. Checkpoints occur only with a
drained prefetch queue.

## Evaluation and monitoring

- Warmup ends at update 4,096. AWR starts at update 4,097.
- Durable checkpoints occur every 2,048 updates. Offline validation runs every
  4,096 updates.
- L40S gameplay evaluation uses p90 for 96 matchups every 8,192 updates and at
  completion. An incomplete calibration is an error.
- The final checkpoint also gets a matched unconditioned evaluation in its own
  output directory and W&B namespace.
- Gameplay evaluation determines policy quality. Offline validation and rollout
  diagnostics use actual available return labels.
- Completion requires update 98,304, `final.pt`, the numbered checkpoint,
  successful launcher state, and both final evaluation artifacts.

The Sol babysitter receives the exact Modal and W&B identities after launch.
It establishes warmed throughput, MFU, loader wait, and memory baselines for the
production data path.

The measured execution evidence is in [throughput.md](throughput.md). Required
checks and regression coverage are in [validation.md](validation.md).
