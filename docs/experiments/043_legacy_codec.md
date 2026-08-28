# O43: exact legacy codec on the O26 baseline

This is one experiment, not an ablation matrix. It copies current O26 and changes only the controller codec.

The run uses ranked-anonymous-1 with the named O26 configuration: d384, 8 layers, 6 heads, context 128, the same temporal decoder and offsets, batch 512, 16,384 updates, and the same H4/H6 evaluation. The trainable parameter count is 15,053,039. O43 keeps every O26 parameter and masks unused output rows.

## What changes

The legacy codec has 37 main-stick classes, 9 C-stick classes, 5 button classes, and 3 shoulder classes. The button classes are A, B, Jump, Z, and None. Jump decodes as X. Shoulder decodes as analog L.

When a replay frame holds several face buttons, the historical stateful reducer selects one. For example:

```text
replay: B, B, B+A, B+A, B
model:  B, B, A,   A,   B
```

That creates a B release followed by a B press. This is intentional: the experiment measures the exact legacy codec, including its lossiness.

Inference emits a complete controller state each frame. The normal HAL/libmelee path sends every button's current pressed or released state. There is no separate learned edge protocol and no hidden button latch.

V7 does not store the old fused logical shoulder. O43 reconstructs it as the larger analog shoulder value, with either digital L/R click forcing full pressure. This is the only historical source field that cannot be recovered exactly.

## Audit result

A read-only scan covered 87,860 of 112,409 locally available training replays, or 78.2%, and 1.881 billion player-frames.

- 1.4432% of frames changed face-button state under the legacy reducer.
- 0.0964% gained a synthetic press edge.
- 0.0464% gained a synthetic release edge.
- A+Jump was the largest lossy chord at 0.9611% of all frames.
- A suppressed hold lasted 6 frames at the median and 20 frames at p90.
- B suppressed hold lasted 2 frames at the median and 6 frames at p90.

These numbers are labeled partial because 63 shards were not cached locally.

## Throughput check

On the local RTX 3060, quantizing a production-shaped batch took 2.47 ms versus 1.32 ms for O26. A complete uncompiled forward/backward step took 110.3 ms versus 109.7 ms, a 0.5% difference. The reducer is a tensor scan rather than a Python loop, so it does not launch one GPU operation per frame.

## Evaluation and launch

Evaluation uses 32-wide waves, but the same spawned-worker path as O41 admits only eight cold Dolphin startups at once. This avoids the CPU thundering herd while preserving the reference matchup count.

```bash
uv run scripts/launch_modal.py --dry-run --gpu L40S --cpu 16 --cpu-limit 16 --memory-gib 64 --disk-gib 512 --timeout-hours 8 --app-name hal-043-legacy-codec -- uv run experiments/043_legacy_codec.py
uv run scripts/launch_modal.py --gpu L40S --cpu 16 --cpu-limit 16 --memory-gib 64 --disk-gib 512 --timeout-hours 8 --app-name hal-043-legacy-codec -- uv run experiments/043_legacy_codec.py
```
