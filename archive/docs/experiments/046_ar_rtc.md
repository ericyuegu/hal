# O46: autoregressive training-time RTC

O46 tests training-time real-time chunking with O43's discrete autoregressive
controller decoder. It keeps O43's model parameters, historical controller
codec, data, optimizer, and closed-loop suites. It changes the temporal target
schedule from ten sparse offsets to every offset from 1 through 20.

## Training objective

For each valid context prefix, training samples one delay `d` from
`training_delay_frames`. The first `d` target actions are the clean committed
prefix. They remain teacher-forced inputs to the causal decoder, but receive no
loss. The loss is the mean joint controller NLL over the remaining postfix.

The default delay set is `(0, 1, 2, 3, 4)`. Zero must be present. It trains the
full bootstrap chunk used at match start and after a reset.

Autoregression needs no flow timestep or separate delay embedding. At postfix
position `d`, the causal cache already contains the observation, the dense
offset positions, and the `d` committed actions.

## Closed-loop timing

The default policy uses:

- prediction horizon `H = 20`;
- inference delay `d = 4`;
- execution stride `s = 4`.

At each continuing replan, the shared scheduler supplies actions `[s:s+d]`
from the previous plan. The decoder forces those actions through its first `d`
cached steps and samples steps `[d:H]`. The returned prefix is copied exactly
from the previous plan.

O46 pins `s = max(d, 1)`. The RTC overlap constraint is `d <= H - s`, so the
largest supported paper-style delay is 10 frames.

## Controlled runs

Train the dense autoregressive control and RTC treatment with the same seed,
data, updates, and evaluation schedule:

```bash
uv run experiments/046_ar_rtc.py \
  --cfg.training-delay-frames 0 \
  --comment dense-ar-control

uv run experiments/046_ar_rtc.py \
  --comment ar-rtc-d0-4
```

Evaluate both checkpoints at every supported treatment delay. The command sets
the matching execution stride automatically:

```bash
for d in 0 1 2 3 4; do
  uv run experiments/046_ar_rtc.py \
    --eval runs/<run>/final.pt \
    --eval-delay-frames "$d" \
    --eval-output-name "rtc-d${d}"
done
```

The primary comparison is the dense control versus the RTC treatment at the
same evaluation delay. O43 remains an external sparse-offset reference.

## Evidence

Validation reports dense NLL plus conditional postfix and first-postfix NLL for
each training delay. At the configured evaluation delay, it also reports greedy
postfix NLL, exposure gap, first-postfix exact accuracy, and boundary transition
and hold accuracy. Closed-loop evidence records the training delay set,
inference delay, execution stride, dense action offsets, checkpoint hash,
decode latency, and the standard character and Fox suites.
