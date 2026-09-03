# O51 16-layer sweep preregistration

Registered on 2026-09-03 before the first O51 v3 training launch. The 8-layer
runs are preliminary evidence only. All authoritative tuning endpoints below
use the 14.48M proxy with 16 trunk layers and 4 temporal layers. The 55M and
216M models keep those depths and increase width.

## Prediction

The predicted winning configuration is:

- hidden initialization coefficient `0.5`;
- μP-normal final readouts;
- depth exponent `alpha=0.5`;
- Muon master learning rate `0.014`;
- AdamW master learning rate `2.125e-4`;
- Muon and AdamW weight decay `0.001`;
- batch size `512`;
- fixed Muon batch scaling;
- fixed Muon duration scaling.

The prediction transfers unchanged from the 14.48M proxy to 55M because depth
stays fixed. O51 applies its existing role-specific width rules: Muon keeps its
master learning rate, AdamW input and vector parameters keep theirs, and final
readouts scale by their own fan-in.

## Basis

The completed 8-layer initialization extensions strongly favor `h=0.5` with
μP-normal readouts: validation NLL was `1.585409` for W&B run `q2wqt16c`, versus
`2.431370` for the zero-readout run `cng1219s`. This outweighs the small,
reversed difference at the short initialization screen.

The best completed preliminary 8-layer LR endpoint was Muon `0.014` and AdamW
`2.125e-4`, with validation NLL `1.569904` in W&B run `cl0nqptn`. That pair was
the low-low boundary of the old grid, so the new grid brackets it with Muon
`{0.007, 0.014, 0.028}` and AdamW
`{1.0625e-4, 2.125e-4, 4.25e-4}`.

The reviewed parameterization literature supports an `alpha=0.5` residual
branch multiplier of `1/sqrt(2)` when depth doubles. It does not justify
applying Adam's hidden-weight depth LR rule to Muon. O51 routes all hidden
stack matrices to Muon, while AdamW owns inputs, outputs, biases, gains, and
embeddings. Therefore the prediction keeps both optimizer master rates fixed
and scales only the residual branch contribution with depth. The hidden
initialization coefficient also stays fixed because applying another depth
factor there would scale the branch twice.

The relevant implementation is in
[`051_correct_parameterization.py`](../051_correct_parameterization.py):
`MODEL_FAMILY`, `depth_rule`, `initialize_o51_parameters`, `optimizer_roles`,
and `_role_lr`. The fixed grids and 16-layer stage routing are in
[`o51_sweep.py`](../../hal/training/o51_sweep.py).

This interpretation follows the previously reviewed Depth-μP, CompleteP,
Complete(d)P, μTransfer, and Muon parameterization results. No new literature
search was used for this preregistration.

## Sequence and decision rules

1. Do not rerun the six initialization screens. Run one fresh 16-layer D0/U0
   extension with `h=0.5` and μP-normal readouts.
2. Run the fresh 16-layer 3x3 learning-rate grid.
3. Rank stable exact endpoints by validation NLL, then far NLL, rollout NLL,
   and arm ID. Evaluate the top two final checkpoints over the same fixed 96
   closed-loop matchups.
4. Select the LR winner by greatest net-stock cluster-bootstrap lower bound,
   then mean net stock per minute, net damage per minute, validation NLL, and
   arm ID.
5. Run the independent 16-layer 3x3 Muon/AdamW decay grid and apply the same
   top-two closed-loop rule.
6. Continue with batch, depth-rule, 55M, seed, duration, promotion, and soak
   stages in the registered O51 order.

An infrastructure failure can be retried with the identical specification. A
scientific divergence is excluded. A failed closed-loop worker is retried on
the same checkpoint; completion timing cannot replace either finalist.
