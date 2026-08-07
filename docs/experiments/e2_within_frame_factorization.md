# E2: within-frame action factorization

Status: blocked on E1 evidence

Updated: 2026-08-07

## Question

Does modeling one controller frame as a joint conditional distribution improve closed-loop play
over an equally deep state-only output head?

E2 compares against E1, not only E0. E1 controls for the extra MLP capacity. E2 changes the
probability factorization by conditioning each action group on earlier groups from the same frame.

## Arms

Run two chain orders, one at a time:

- E2-S, stable prefix: `c_stick, triggers, main_stick, buttons`.
- E2-I, intent first: `main_stick, buttons, triggers, c_stick`.

E2-S is the main factorization arm. E2-I tests group order. Do not select the order from
teacher-forced NLL alone.

The first run of each arm uses seed 0. Treat it as screening evidence. Use at least three matched
training seeds before making an architecture claim.

## Files to change

- `experiments/023_mtp_heads.py`: add `state_mlp` and `factored_mlp` output modes, group-order
  validation, action embeddings, teacher-forced training, ancestral decode, and diagnostics.
- `tests/experiments/test_023_mtp_heads.py`: test initialization, conditioning, order, gradients,
  objective alignment, decode, checkpoint loading, and optimizer ownership.
- `docs/experiments/e1_output_head_capacity.md`: record the shared E1 implementation and exact
  parameter counts.
- This file: record the final code, commands, run IDs, timing, metrics, artifacts, and decision.

Do not edit historical experiment files. Do not change the transformer, replay sampler, target
offsets, loss scale, or evaluation protocol.

## Probability model

For chain position `g` at offset `o`, model:

\[
p(a_{t+o,g}\mid s_t,a_{t+o,<g}).
\]

The product over the four groups is one joint distribution for that future controller frame. There
is no temporal conditioning between offsets in E2. The offset-5 prediction does not condition on
the offset-1 action.

During training, use the true earlier group classes from the same target frame. During validation,
report the joint teacher-forced likelihood and a separate rollout-conditioned diagnostic. During
closed-loop decode, sample each group and feed that sample to the later groups.

## Head design

Use these planned configuration fields:

```python
head_mode: Literal["linear", "state_mlp", "factored_mlp"] = "linear"
action_mlp_ratio: int = 2
action_condition_dim: int = 32
action_group_order: tuple[str, ...] = ("c_stick", "triggers", "main_stick", "buttons")
```

Share one embedding table for each action group across offsets. Each table maps its discrete class
to a 32-dimensional vector. For each predicted group, concatenate the embeddings of all earlier
groups in the selected chain order. Concatenation is along the feature dimension.

For offset `o` and group `g`:

\[
c_{o,g}=\operatorname{concat}_{j<g}E_j(a_{t+o,j}),
\]

\[
r_{o,g}=W_{2,o,g}\operatorname{SiLU}
\left(W_{h,o,g}\operatorname{RMSNorm}(h)+W_{c,o,g}c_{o,g}\right),
\]

\[
z_{o,g}=h+r_{o,g},
\qquad
\ell_{o,g}=W_{o,g}z_{o,g}+b_{o,g}.
\]

The first group has no condition and uses the same state-only branch as E1. Initialize `W_2`, its
bias, and every `W_c` to zero. Create the trunk, base classifiers, normalization, `W_h`, and `W_2`
in the same order as E1 so their same-seed values match. Create conditioning tensors afterward.

Implement `W_h` and `W_c` as column blocks of one affine layer over
`concat(RMSNorm(h), c)`. Do not run separate projections and add their outputs. The split notation
only states how to copy the E1 state columns and zero the new condition columns. This gives the
same function with less code and one matrix multiplication.

This gives exact initial E1 logits. The first update can train `W_2`. Later updates can train `W_c`
and the embeddings. Do not add a separate linear condition bypass. `W_c` is already the condition
block of the MLP affine map.

All MLP, embedding, and classifier parameters use AdamW. They must not enter Muon.

## Objective

Keep the normalized E0 objective:

\[
L=L_1+\frac{L_5+L_9+L_{13}}{3}.
\]

Each `L_o` is the sum of the four conditional group NLL values at offset `o`. Keep AWR, rank
weighting, value loss, and critic training off. Execute only offset 1.

The valid-position mask must be identical for all four conditional terms at a given frame. Padding
or a missing target must not supply a condition or a loss.

## Required tests

1. Reject a group order that is not a permutation of all four canonical groups.
2. Check all action class indices against the correct group vocabulary before embedding lookup.
3. Construct same-seed E1 and E2 models. Assert exact shared parameters and exact logits at
   initialization for both chain orders.
4. Assert every `W_2`, `W_2` bias, and `W_c` is zero. Assert `W_h` is not zero.
5. Confirm the first backward pass gives finite nonzero `W_2` gradients.
6. Confirm later backward passes give finite nonzero `W_c` and embedding gradients.
7. Change only an earlier teacher-forced group. The first-group logits must stay fixed, and at least
   one valid later-group logit must change after conditioning has learned.
8. Assert teacher forcing uses target classes, never predicted argmax classes.
9. Assert ancestral decode uses sampled classes in the declared order.
10. Use the same seed at initialization. E1 and E2 must sample identical complete action bytes.
11. Assert the four conditional losses sum to the logged joint frame NLL.
12. Assert auxiliary normalization remains one fixed mean across offsets 5, 9, and 13.
13. Save and reload each order. Check mode, order, dimensions, logits, and sampled decode.
14. Assert every new parameter belongs to AdamW exactly once and never to Muon.
15. Run the small end-to-end training test in both E2 orders.

Run focused tests, Ruff, type checking, Python compilation, and `git diff --check` before the GPU
gate.

## Fixed configuration

Copy the selected E0 package and E1 configuration exactly. E2 may change only:

- `head_mode=factored_mlp`
- `action_condition_dim=32`
- `action_group_order`
- Run labels and H2H reference fields.

Keep `action_mlp_ratio=2`, offsets `(1,5,9,13)`, the normalized auxiliary loss, 16,384 steps, seed
0, 131,072 frames per step, compact policy v7, the replay reservoir, optimizer, schedule,
checkpoint cadence, decode temperature, and CPU protocol fixed.

Report exact total, trunk, classifier, state-MLP, condition, and embedding parameter counts. If E2
has more than 5% more total parameters or is more than 10% slower per training step than E1, plan a
separate capacity control before making a factorization claim.

## Offline records

For every offset and group, log:

- Teacher-forced NLL and argmax accuracy.
- Cross-entropy and sampled accuracy when earlier groups are sampled from the model.
- The gap between teacher-forced and rollout-conditioned cross-entropy.
- Hold and transition NLL and accuracy.
- Predicted transition rate, persistence, and change-event F1.

Also log exact-frame accuracy, each head's trunk-gradient norm, primary-to-auxiliary gradient cosine,
condition-branch norms, embedding norms, parameter counts, step time, loader wait, peak memory, and
closed-loop decode time.

The rollout-conditioned metric is stochastic. It is an exposure-bias diagnostic, not the joint NLL
of the observed action. Freeze its seed and sample count. Do not compare runs that use different
rollout draws.

## Closed-loop evaluation

Run the standard 32-boot periodic and 96-boot final CPU protocol. Instant restart may produce more
completed games than boots. Save all rows and replay files.

For E2-S, run 64 mirrored H2H configurations against E1. For E2-I, run the same H2H schedule against
E2-S. Report non-tied win rate, stock difference, confidence intervals, ties, crashes, stage slices,
and character slices. CPU and H2H results both matter.

Target 3.0 to 3.5 hours per training arm through evaluation and upload. Flag startup over 30
minutes, warm steps over 0.5 seconds, a slowdown over 25% from E1, or a projected total over 3.5
hours. Use one RTX 4090 experiment at a time.

## Decision

- E2-S supports factorization only if it beats E1 in closed loop or gives a clear joint-modeling
  gain without a material control regression.
- E2-I supports an order claim only through a direct E2-I versus E2-S comparison.
- Better teacher-forced NLL with worse ancestral metrics indicates exposure to ancestor errors.
- Better offline metrics without a closed-loop gain do not promote E2.
- A clear regression in both orders is evidence against this factorization at the tested capacity.

Do not start temporal factorization in E3 until E2 evidence is complete and the selected within-frame
head is fixed.

## Results

Pending E1 completion.
