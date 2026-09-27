# O44: fixed prediction delay

O44 asks whether a policy can learn controller actions for a fixed future
release time instead of learning only the next frame. It preserves O43's model,
legacy controller codec, optimizer, loss allocation, and relative auxiliary
targets.

## Timing terms

The model observes game state through frame `t`.

- `prediction_delay_frames` (`d`) is the trained offset to the first predicted
  action.
- `prediction_horizon_frames` (`H`) is the number of consecutive actions in one
  live plan.
- `replan_interval_frames` (`R`) is the number of game frames between inference
  requests.

Thus, one plan covers `[t+d, t+d+H)`. O44 fixes `H=6` and uses `R=1` for the
controlled runs. It does not use “execution horizon,” which previously combined
the last two quantities.

The main relative prediction steps are `0..5`. O43's sparse auxiliary steps are
retained at `8, 11, 15, 19`, so the complete target set is:

```text
d + [0, 1, 2, 3, 4, 5, 8, 11, 15, 19]
```

At `d=1`, this is exactly O43's `[1, 2, 3, 4, 5, 6, 9, 12, 16, 20]` schedule.
The model's prediction-step embeddings remain relative, so changing `d` does
not change its initialization or parameter count.

## Closed-loop scheduling

Each prediction is stored with the absolute frame that anchored its observation.
The scheduler releases it only when its target frame arrives. If plans overlap,
the newest eligible plan wins. At startup and after a reset, the controller is
neutral until the first plan matures.

With `R=1`, the first prediction step normally supplies the executed action.
The other five live steps measure the prediction horizon and permit later
replanning ablations without retraining.

## Controlled runs

Use one initialization seed and the same ranked-anonymous data, 37-frame future
buffer, update count, validation windows, and matchup schedule:

```bash
uv run experiments/044_fixed_prediction_delay.py \
  --cfg.prediction-delay-frames 3 \
  --comment fixed-delay-d3-r1-h6

uv run experiments/044_fixed_prediction_delay.py \
  --cfg.prediction-delay-frames 12 \
  --comment fixed-delay-d12-r1-h6

uv run experiments/044_fixed_prediction_delay.py \
  --cfg.prediction-delay-frames 18 \
  --comment fixed-delay-d18-r1-h6
```

Use W&B run `1imfy8v3` as the `d=1` curve anchor. It is not a fully
sampler-matched control: O43 requested 20 future frames, while every O44 run
requests 37 to keep the three O44 data streams identical.

Compare first-prediction NLL, adjacent-frame transition accuracy, and the two
closed-loop evaluation suites across `d=1,3,12,18`. Evaluation evidence records
all three timing quantities and the absolute training target offsets.
