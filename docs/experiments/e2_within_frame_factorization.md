# E2: within-frame action factorization

Status: local implementation complete; GPU gate blocked on E1 evidence

Updated: 2026-08-07

## Question

Does modeling one controller frame as a joint conditional distribution improve closed-loop play
over an equally deep state-only output head?

E2 compares against E1, not only E0. E1 controls for the extra MLP capacity. E2 changes the
probability factorization by conditioning each action group on earlier groups from the same frame.

## Arms

Run two chain orders, one at a time:

- E2-S, stable prefix: `c_stick, triggers, buttons, main_stick`.
- E2-I, intent first: `main_stick, buttons, triggers, c_stick`.

E2-S is the main factorization arm. E2-I tests group order. Do not select the order from
teacher-forced NLL alone.

The E2-S order is fixed from the completed P0 offset-1 validation metrics, before E2 training. NLL
in bits per frame was 0.050 for c-stick, 0.090 for triggers, 0.242 for buttons, and 0.646 for main
stick. Predicted transition rates had the same order: 0.0062, 0.0174, 0.0271, and 0.1136. E2-S
therefore puts the lower-entropy, more persistent groups first. This also lets the main-stick
distribution condition on button intent. The source is W&B run `obx3o3az`.

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
report the joint teacher-forced likelihood and a separate ancestor-sampled diagnostic. For that
diagnostic, sample the earlier groups, then score the true current-group target under the resulting
conditional distribution. Report its NLL and argmax accuracy. Also sample the complete frame once
and report its sampled exact-frame accuracy. During closed-loop decode, sample each group and feed
that sample to the later groups.

Use one live random-number stream per vector slot and named action group. Derive each stream from
the policy seed, stable slot index, slot generation, and group name with a fixed mapping that does
not depend on chain order. Increment the slot generation on reset; resetting one slot must not
change another slot's streams. E1, E2-S, and E2-I must use the same mapping during their direct
comparisons. A single shared stream is invalid here: chain order or one slot ending would reassign
random draws before the model learned any conditioning.

## Head design

Use these planned configuration fields:

```python
head_mode: Literal["linear", "state_mlp", "factored_mlp"] = "linear"
action_mlp_ratio: int = 2
action_condition_dim: int = 32
action_group_order: tuple[str, ...] = ("c_stick", "triggers", "buttons", "main_stick")
factorization_diag_seed: int = 0
factorization_diag_samples: int = 1
```

Share one embedding table for each group that can be an ancestor. The final group has no consumer,
so it has no dead embedding table. Each table maps its discrete class to a 32-dimensional vector.
For each predicted group, concatenate the embeddings of all earlier groups in the selected chain
order. Concatenation is along the feature dimension.

Compute one shared state preactivation:

\[
v(h)=W_hh.
\]

`h` is already the trunk's final RMS-normalized output. Do not normalize it again inside the head.

For offset `o` and group `g`:

\[
c_{o,g}=\operatorname{concat}_{j<g}E_j(a_{t+o,j}),
\]

\[
u_{o,g}=\operatorname{SiLU}\left(v(h)+W_{c,g}c_{o,g}\right),
\]

\[
\ell_{o,g}=W_{o,g}h+b_{o,g}+W_{2,o,g}u_{o,g}.
\]

The first group has no condition and uses the E1 state path. Share `W_h` across all groups and
offsets. Share each group's `W_c` across offsets; its input values still differ by target offset.
Initialize `W_2`, its bias, and every `W_c` to zero. Create the trunk, base classifiers, `W_h`, and
`W_2` in the same order as E1 so their same-seed values match. Create conditioning tensors
afterward.

`W_h h + W_c c` is algebraically one affine map over `concat(h, c)`. Keep the two projections
separate here so the expensive `W_h h` result is computed once and reused across every group and
offset. This is parameter sharing, not an additive logit bypass.

This gives exact initial E1 logits. The first update can train `W_2`. The next update can train
`W_c`; embeddings receive gradients after `W_c` leaves zero. Do not add a separate linear condition
bypass.

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
4. Assert every `W_2`, `W_2` bias, and `W_c` is zero. Assert the shared `W_h` is not zero.
5. Confirm the first backward pass gives finite nonzero `W_2` gradients.
6. Confirm the second backward pass gives finite nonzero `W_c` gradients. Confirm a later backward
   pass gives finite nonzero embedding gradients.
7. Change only an earlier teacher-forced group. The first-group logits must stay fixed, and at least
   one valid later-group logit must change after conditioning has learned.
8. Assert teacher forcing uses target classes, never predicted argmax classes.
9. Assert ancestral decode uses sampled classes in the declared order.
10. Use the same seed and slot-and-group-keyed random streams at initialization. E1 and both E2
    orders must sample identical complete action bytes for many consecutive calls. Check each
    stream state after every call.
11. Assert the four conditional losses sum to the logged joint frame NLL.
12. Assert auxiliary normalization remains one fixed mean across offsets 5, 9, and 13.
13. Save and reload each order. Check mode, order, dimensions, logits, and sampled decode.
14. Assert every new parameter belongs to AdamW exactly once and never to Muon.
15. Run the small end-to-end training test in both E2 orders.
16. Assert ancestor-sampled validation leaves process-wide CPU and CUDA RNG states unchanged.
17. Assert it samples only earlier groups before scoring the current target, and assert its NLL is
    the cross-entropy of that conditional distribution rather than the loss on a sampled class.
18. Assert that reordering the chain does not reassign a random draw from one named group to
    another. Reject missing, extra, or shared group streams. Reset one vector slot and prove that
    every other slot produces the same later samples as an unreset control.

Run focused tests, Ruff, type checking, Python compilation, and `git diff --check` before the GPU
gate.

## Fixed configuration

Copy the selected E0 package and E1 configuration exactly. E2 may change only:

- `head_mode=factored_mlp`
- `action_condition_dim=32`
- `action_group_order`
- `factorization_diag_seed=0`
- `factorization_diag_samples=1`
- Run labels and H2H reference fields.

Keep `action_mlp_ratio=2`, offsets `(1,5,9,13)`, the normalized auxiliary loss, 16,384 steps, seed
0, 131,072 frames per step, compact policy v7, the replay reservoir, optimizer, schedule,
checkpoint cadence, decode temperature, and CPU protocol fixed.

## Planned launches

Fill the reference run and hash only from the verified final checkpoint. Then run a no-rent audit
from the pushed, clean branch. E2-S uses:

```bash
e1_run=<verified E1 run name>
e1_sha=<verified E1 final.pt SHA-256>

uv run scripts/launch_vast.py \
  --max-price 1.1 --disk 250 --min-vram 24 --min-ram 200 \
  --min-dlperf 120 --min-compute-cap 890 --max-compute-cap 890 \
  --data-gb 13.34 --upload-gb 2 --run-hours 3.5 -- \
  uv run experiments/023_mtp_heads.py \
    --cfg.head-mode factored_mlp \
    --cfg.action-mlp-ratio 2 \
    --cfg.action-condition-dim 32 \
    --cfg.action-group-order c_stick triggers buttons main_stick \
    --cfg.factorization-diag-seed 0 \
    --cfg.factorization-diag-samples 1 \
    --cfg.require-flex \
    --cfg.eval-max-parallel 32 \
    --cfg.final-h2h-reference-run "$e1_run" \
    --cfg.final-h2h-reference-sha256 "$e1_sha" \
    --cfg.final-h2h-reference-experiment experiments/023_mtp_heads.py \
    --cfg.final-h2h-reference-label 023-e1-state-mlp \
    --cfg.final-h2h-self-label 023-e2-stable \
    --cfg.final-h2h-n-configs 64 \
    --comment e2-stable
```

E2-I uses the same launcher fields. It changes the order to `main_stick buttons triggers c_stick`,
uses the verified E2-S run and hash as its reference, and uses labels `023-e2-stable` and
`023-e2-intent`. Do not launch both arms together.

The slot-and-group-keyed decode streams are shared evaluation infrastructure for this comparison,
not an E2 treatment. Load the E1 reference through the same sampler during H2H. Report that its
random stream mapping differs from its historical single-stream evaluation.

Report exact total, trunk, classifier, state-MLP, condition, and embedding parameter counts. If E2
has more than 5% more total parameters or is more than 10% slower per training step than E1, plan a
separate capacity control before making a factorization claim.

At the planned dimensions, the shared conditioning projections add 98,304 parameters. E2-S adds
9,280 ancestor-embedding parameters, for 7,786,110 total. E2-I adds 11,072, for 7,787,902 total.
The arms differ by 1,792 parameters, or 0.023% of the model. Keeping an unused final-group table
would make the nominal counts equal but would not equalize effective capacity. Both arms remain
less than 1.43% larger than E1's 7,678,526 parameters. The conditioning projections add about 51.5
billion forward MACs per 131,072-frame step because the conditions differ by offset.

## Offline records

For every offset and group, log:

- Teacher-forced NLL and argmax accuracy.
- NLL and argmax accuracy when earlier groups are sampled from the model.
- The gap between teacher-forced and ancestor-sampled NLL.
- Hold and transition NLL and accuracy.
- Predicted transition rate, persistence, and change-event F1.

Also log exact-frame accuracy, each head's trunk-gradient norm, primary-to-auxiliary gradient cosine,
condition-branch norms, embedding norms, parameter counts, step time, loader wait, peak memory, and
closed-loop decode time.

The ancestor-sampled metric is stochastic. It is an exposure-bias diagnostic, not the joint NLL of
the observed action. Use one ancestor rollout per validation frame with seed 0. Do not compare runs
that use different rollout draws. One rollout is enough because the fixed validation set contains
many valid frames; record its size with every metric.

Use one dedicated diagnostic generator per named action group. Recreate all four from
`factorization_diag_seed` for each validation pass and draw `factorization_diag_samples` ancestor
rollouts per frame. Do not consume the process-wide Torch RNG or any live closed-loop decode
stream. Validation must leave training RNG state byte-identical.

## Closed-loop evaluation

Request 32 periodic matchups and 96 final matchups. Limit both sweeps to 32 concurrent Dolphin
boots. Instant restart may produce more completed games than requested matchups. Save all rows and
replay files.

For E2-S, run 64 mirrored H2H configurations against E1. For E2-I, run the same H2H schedule against
E2-S. Report non-tied stock-lead rate, stock difference, confidence intervals, ties, crashes, stage
slices, and character slices. Expand each policy's existing decode seed into its slot-and-group
streams. The two players keep their existing distinct seeds. CPU and H2H results both matter.

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

The implementation is complete on `exp/e2-within-frame-factorization`. It changes only
`experiments/023_mtp_heads.py`, its focused test file, and experiment records.

Implemented behavior:

- `factored_mlp` keeps E1's base classifiers, shared state projection, and zero-initialized
  residual outputs.
- Each later group receives the feature-axis concatenation of earlier target-group embeddings in
  training and earlier sampled-group embeddings in decode.
- The shared state projection is computed once. Each group's zero-initialized condition projection
  is shared across offsets.
- Teacher-forced validation reports the existing NLL and accuracy. Dedicated group generators
  report ancestor-sampled NLL, accuracy, exposure gap, and exact-frame accuracy without changing
  process RNG state.
- Closed-loop sampling uses deterministic streams keyed by seed, stable slot, reset generation,
  and named group. Reordering groups or resetting one slot does not reassign another stream's draw.
- E2-S has 7,786,110 parameters. E2-I has 7,787,902. The 1,792-parameter difference comes from
  removing each order's unused final-group embedding table.

Local evidence:

- Same-seed E1 and both E2 orders have exact shared parameters and exact initial logits.
- Keyed sampled actions and every stream counter also match E1 exactly at initialization.
- Residual outputs receive gradients on update one, condition projections on update two, and
  action embeddings on update three.
- Tests cover target conditioning, ancestral order, class bounds, reset isolation, checkpoint
  round trips, optimizer ownership, private validation RNG, both training orders, and exact counts.
- Both E2 orders complete the real dev-MDS end-to-end training path and save `final.pt`.
- Focused experiment suite: 79 passed in 10.21 seconds.
- Full repository suite: 935 passed in 136.05 seconds.
- Ruff, the type error gate, Python compilation, and `git diff --check` pass. The type checker
  reports existing warnings and no errors.

The GPU compile, memory, throughput, E1-reference hash, launch command, and no-rent audit remain
pending. Do not launch E2-S until E1's final checkpoint, CPU sweep, H2H record, and decision are
verified.
