# 036 residual-readout recursive GRU

Status: preregistered, implemented, and rejected at the RTX 3060 smoke gate on 2026-08-19. Production was not
launched because both validation NLL and compiled latency failed their precommitted limits.

## Hypothesis and sole treatment

Experiment 035 replaced 026's two temporal Transformer blocks with one zero-state vanilla
`torch.nn.GRUCell(128,128)`. It was much faster, but its matched 100-step validation NLL was 7.618324 bits versus
7.209772 for 026, triggering its precommitted stop. Read-only diagnosis found a deficit at every offset including
offset 1, healthy unsaturated gates, a materially used prior-action channel, and a low-rank recurrent state. This
points to a readout bottleneck more directly than to missing recurrent depth.

Experiment 036 changes only the value decoded by the unchanged heads. For the unchanged projected token

\[
x_k=P[\operatorname{RMSNorm}(scene), e(a_{k-1}), e(o_k)]
\]

and unchanged raw recurrent state

\[
h_k=\operatorname{GRUCell}(x_k,h_{k-1}), \qquad h_0=0,
\]

the head input becomes

\[
y_k=\operatorname{RMSNorm}(x_k+h_k).
\]

The raw `h_k`, never `y_k`, recurs. The sum is unscaled. There is no learned or fixed gate, residual state update,
extra projection, scene initialization, custom GRU bias/init, action ablation, second GRU, attention, rank weighting,
or loss/data/optimizer/evaluation change. Teacher forcing, forced stepwise decoding, greedy diagnostics, and sampled
H4/H6 all use this exact readout.

## Frozen controls and identity

- Fresh seed 0, effective batch 512, 16,384 optimizer steps.
- Original ranked-anonymized-1 `mds-policy-v7` storage view and replay/window order.
- Exact offsets `(1,2,3,4,5,6,9,12,16,20)`, codec, absolute-offset and prior-full-action token, scene injection,
  heads, group order, legality, objective, optimizer/schedule, validation, checkpoints, and RNG/reset behavior.
- H4 remains primary; H6 remains diagnostic; final evaluation remains ordinary 96-bootstrap H4 plus 32-bootstrap H6.
- Identity is `036_residual_gru_v1`, decoder architecture version 5. Checkpoints from 026, 034, and 035 are rejected.
- Parameter counts remain exactly 99,072 for the cell and 14,889,967 total.

## Required evidence

Tests must cover the inherited 026/035 contracts plus full teacher/stepwise parity, temporal and group causality,
fresh-state isolation, the exact zero-cell identity `y=RMSNorm(x)`, proof that only raw `h` recurs, H4/H6 eager and
compiled parity, legality/RNG/reset, strict configuration and checkpoint identity, and complete optimizer ownership.

The fresh 100-step RTX 3060 smoke uses the same effective batch 512 geometry as the archived matched 026 control
`260819-111359_026_temporal_mtp_..._026-matched-smoke-rtx3060`. It includes two 1,800-frame boots at H4 and H6,
isolated decoder graph-break checks, compiled latency, and validation NLL by offset/group.

Three read-only branch ablations are reported on the trained checkpoint without changing training:

- full: `RMSNorm(x+h)`;
- x-only: `RMSNorm(x)` by zeroing `h` only at readout;
- h-only: `RMSNorm(h)` by zeroing `x` only at readout.

Report branch norms, NLL, and gradients. Do not call a gain recursive evidence unless full beats x-only. Full worse
than x-only is itself a kill condition.

## Precommitted cloud gates

Launch one fresh Modal L40S production run only if all of the following hold:

1. Loss and gradients are finite and stable; the new decoder introduces zero graph breaks.
2. H4 and H6 each finish with zero crashes and deterministic eager/compiled parity.
3. Aggregate validation NLL is at most 7.359772 bits: no more than 0.15 above matched 026 and at least 0.258552 bits
   recovered from 035.
4. Sampled compiled p95 latency is at most 1.25 times the matched 026/034 same-host control.
5. Full readout is no worse than x-only, and projected L40S runtime is at most eight hours.

If any gate fails, stop before cloud, preserve the negative evidence, commit, and push. If every gate passes, commit
and push before launching exactly one production run; record source SHA, Modal app/function, W&B identity, checkpoint
hash, throughput, validation, H4/H6 evaluation, and final ordinary `net_stock_lcb` against target `1.031887949`.

## Result

The fresh smoke used offline W&B ID `7483dx09` and run directory
`260819-112820_036_residual_recursive_gru_mtp026-d384-L8-h6-Lc128-t128x2-o1-2-3-4-5-6-9-12-16-20-s4-base-gru128-resread_ranked-anon-1_036-smoke-rtx3060`.
Its final checkpoint SHA-256 is `83b0a7876baaa2d0126645b748e2265050ff2a431d4f423eed585c231bfe3cec`.

Training was finite and stable. Mean steady-state throughput over steps 10--99 was 766.16 samples/s. Step-99 train
NLL was 7.895928 bits, peak allocated memory was 2.087 GiB, and peak reserved memory was 2.193 GiB. The final
validation loss was **7.600986 bits**, above the preregistered 7.359772 ceiling. It recovered only 0.017338 bits from
035's 7.618324 and remained 0.391214 bits worse than matched 026. The primary/auxiliary components were
3.143940/4.457045 bits versus 2.959564/4.250208 for 026, regressions of 0.184376/0.206838.

The joint 036-minus-026 NLL deltas at offsets 1, 2, 3, 4, 5, 6, 9, 12, 16, and 20 were respectively
`+0.135, +0.167, +0.212, +0.223, +0.259, +0.262, +0.143, +0.192, +0.187, +0.199` bits. Every group was worse at
every offset. Main-stick contributed the largest delta (`+0.063` to `+0.147` bits), followed by buttons (`+0.027`
to `+0.077`), triggers (`+0.019` to `+0.049`), and c-stick (`+0.003` to `+0.013`). The residual therefore did not
repair 035's broad underfit.

| offset | joint | buttons | main stick | c-stick | triggers |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | +0.134983 | +0.045903 | +0.063170 | +0.002621 | +0.023290 |
| 2 | +0.167393 | +0.054763 | +0.088002 | +0.005365 | +0.019262 |
| 3 | +0.211832 | +0.063995 | +0.105323 | +0.012730 | +0.029783 |
| 4 | +0.223297 | +0.074318 | +0.107953 | +0.007533 | +0.033492 |
| 5 | +0.258557 | +0.070334 | +0.145959 | +0.012210 | +0.030054 |
| 6 | +0.261511 | +0.077338 | +0.146765 | +0.006915 | +0.030494 |
| 9 | +0.143292 | +0.027346 | +0.080067 | +0.005437 | +0.030442 |
| 12 | +0.191500 | +0.047060 | +0.102769 | +0.008510 | +0.033161 |
| 16 | +0.186782 | +0.034806 | +0.100501 | +0.009646 | +0.041829 |
| 20 | +0.199383 | +0.038082 | +0.102923 | +0.009186 | +0.049192 |

H4 and H6 each completed two boots with zero crashes. Actual Inductor compilation of the isolated H4/H6 decoders
had zero graph breaks and deterministic argmax outputs exactly matched eager execution. Sampled compiled p95 decode
latency was 18.07 ms at H4 and 19.74 ms at H6. Both exceed 1.25 times the matched same-host controls: 11.49 ms and
14.68 ms respectively. The residual addition itself is cheap, so this result is likely a changed fusion/codegen path,
but the operational gate is intentionally outcome-based.

Latency was measured by the unchanged closed-loop `DecodeTelemetry` around the complete compiled inference call on
the same NVIDIA RTX 3060 (12 GiB, driver 595.84), with two live model rows, production BF16/default Inductor mode,
and 450 H4 / 302 H6 replans. The telemetry includes the first compile-shaped outlier; its p95 remains dominated by
steady calls at these sample counts. H4 p50/p95/p99 was 12.36/18.07/28.19 ms and H6 was 15.33/19.74/35.23 ms. The
quality gate independently fails by 0.241214 bits, so the latency failure is not causal to the no-launch decision.

The reproducible read-only `scripts/diagnose_residual_gru.py` analysis on the same 128 validation rows found:

| readout | validation NLL, bits | diagnostic objective, bits | token RMS | state RMS | token grad norm | state grad norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full `RMSNorm(x+h)` | 7.600986 | 5.510827 | 0.2783 | 0.1428 | 0.00628 | 0.00801 |
| x-only `RMSNorm(x)` | 7.873916 | 5.701544 | 0.2783 | 0.1428 | 0.00583 | 0 |
| h-only `RMSNorm(h)` | 8.685449 | 6.260115 | 0.2783 | 0.1428 | 0.00567 | 0.01863 |

Full beats x-only by 0.272930 bits, so the recurrent branch contributes useful predictive information and the result
is legitimately recursive evidence. It still fails the absolute matched-control quality and latency gates, so that
contribution is insufficient for promotion.

Formatting, Ruff, Python compilation, 49 focused/inherited tests, same-seed 035/036 state-dict identity, and the
read-only diagnostic CLI passed. `ty` reported 33 dynamic-wrapper unresolved-reference/attribute diagnostics versus
59 for the 035 wrapper control, with no distinct semantic type failure.

Experiment 036 is rejected before Modal. No production app/function, online W&B run, final 96-bootstrap score, or
checkpoint exists. The negative result argues against further unscaled readout stacking under the same fixed-step
recipe.
