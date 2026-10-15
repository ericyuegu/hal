# Earlier small Muon attention controls

Declared before any attention measurement on these checkpoints.

## Question and selection

Do smaller, completed Muon policies also place almost all attention on one
key token per head? Do different heads place their largest weights at
different key positions within the same input and layer?

Use these fixed final checkpoints, not checkpoints selected by validation
loss or by the attention measurements:

| Run | Role | Parameters | Trunk | Final update |
| --- | --- | ---: | --- | ---: |
| `a117chkw` | O37 D1; strongest H1/H2 cell in the prior factorization study | 7,147,504 | d256, 8 layers, 4 heads | 16,384 |
| `5wfk2esf` | O37 D3; strongest H4/H6 cell in that study | 7,147,504 | d256, 8 layers, 4 heads | 16,384 |
| `cqbbbg77` | O26 production policy with full gameplay evaluation | 15,053,039 | d384, 8 layers, 6 heads | 16,384 |
| `1imfy8v3` | O43 Arm A, legacy-codec gameplay anchor | 15,053,039 | d384, 8 layers, 6 heads | 16,384 |

All four are independent of the later large O50 Muon run. All have recorded
training through update 16,384, finite loss and gradient histories, and
full gameplay evaluations. Checkpoint state, configuration, hashes, and
optimizer identity must agree before use. A finished process label alone
does not establish convergence. Record the exact health histories and
gameplay protocol differences.

O39's approximately 16k-update size-sweep endpoints were considered but
not selected as the primary controls: their gameplay results were much
weaker. O23 E0 has a 256-frame context but poor gameplay and is also not
selected. No attention values were used in these choices.

## Fixed inputs and measurement

- Restore the exact 256 replay windows recorded in the earlier frozen O50
  attention metadata, SHA-256
  `ce30744586fcc6bd5c828dc13f9c1fa930b014465ec157ee5ab506646359f6e2`.
- Verify the recorded raw 260-frame hash for every window against the
  original canonical schema-7 MDS data and verify its index and manifest.
- The old models require 128 context frames. Use the last 128 frames of
  each original 256-frame input. Keep the same replay, ego port, and final
  query frame. Do not select new windows or average positions from
  different inputs into the example figure.
- Rebuild inputs with the old base projection and the ranked-policy-v7
  statistics. Do not reuse O50-normalized input tensors. The statistics
  source is the exact key declared by HAL's dataset registry:
  `r2://hal/processed/ranked-anonymized-1/mds-policy-v7/stats.json`.
  Retrieved SHA-256:
  `031dd0e35a7201b299933ec999615f8657dd3e616e0efd3d78b156b1920e0bef`.
- Use each checkpoint's own observation weights, controller codec, and
  trunk. Do not import or alter a historical experiment program. Reject
  unknown checkpoint structure and unsupported observation configurations.
- Trace only the most recent query token, model position 127. Key
  positions run from 0 (oldest) to 127 (most recent), and age is 127 minus
  key position. Each context token represents one frame.
- Use evaluation mode, no gradients, BF16 projections, FP32 QK scores and
  softmax, and disabled TF32, matching the earlier numerical protocol.
  Validate the traced hidden state against the shared dense-SDPA forward
  path, with maximum absolute error at most 1e-5.
- Use the local GPU, batch size 16, and seed 20260911. There is no training
  or new gameplay evaluation in this experiment.

## Outputs and interpretation

Save all last-query probability rows, not only the first example. Report
the largest attention weight, entropy in nats, and `exp(entropy)` for each
row before averaging. Also report distinct maximum-weight key positions
across heads within each replay/layer. Reject exact tied maxima rather
than silently assigning a position. A maximum position is not a complete
description of a head with distributed attention.

Use replay example zero and layer zero as the initial interactive view.
Show all heads together, keep a shared key-position scale, and make the
actual maximum weight and heads sharing the same position visible on hover.
Add the full within-head distribution if broad attention makes a dot-only
view misleading.

These controls differ from O50 in width, depth, number of heads, context
length, data mixture, objective, decoder, training duration, and optimizer
parameter groups. The comparison tests whether the previously observed
attention pattern also occurs here. It cannot identify which difference
caused a change or establish an effect of Muon versus AdamW by itself.
Shared maxima or distinct maxima do not prove shared or distinct head
functions. Preserve all previous study artifacts.
