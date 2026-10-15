# HAL behavior cloning and learning dynamics

Started: 2026-09-11 UTC. Base Git SHA:
`52da5b2de1d402b86404fc4d24e382409649e525`.
Branch: `research/mechinterp-20260911`.

This is an exploratory study. Historical observations generate hypotheses;
they do not establish that an optimizer, architecture, or metric causes a result.
The original experiments and checkpoints remain unchanged.

## Questions

1. Which prediction errors matter for control, and which dominate mean NLL?
2. Does the reported Muon failure change action calibration, dependence on
   history, attention concentration, or the relative scales of model components?
3. Can a small causal intervention distinguish a loss-basin explanation from
   a change in a specific component or in its scale?

## Evidence and controls

- Resolve the exact W&B IDs `tgpuo1be` and `p1fyyp1z`. Record source commits,
  resolved configuration, dataset identity, checkpoint hashes, and evaluation
  settings. Scan exact histories by `global_step`; sparse metrics need separate
  scans. Do not infer a matched comparison from a display name.
- Select checkpoints around the reported transition by training chronology.
  Keep validation and gameplay results visible even when they disagree.
- Use the same replay windows, feature transformations, precision, and random
  seeds for each checkpoint probe. Distinguish human replay states from states
  visited by the evaluated policy.
- Use paired evaluation seeds for interventions where the harness supports it.
  Report the number of games and uncertainty at the game or replay level;
  individual frames are not independent samples.
- Before a causal probe, record its treatment, control, invariant inputs, and
  predicted outcome. Include an identity intervention to test the probe itself.
- A parameter swap changes a component and its compatibility with the rest of
  the model. A successful swap is evidence of localization; an unsuccessful swap
  alone does not prove that the component is unimportant.
- Attention weights and singular values are descriptive. A circuit claim needs
  an intervention and a behavior-specific effect.
- Treat Hessian curvature as dependent on parameterization. Prefer functional
  changes under specified perturbations over an unqualified claim of sharpness.

## Initial tests

1. Reconcile the loss, gradient, and gameplay timelines of the two named runs.
2. Measure matrix norms, spectra, and angular changes across available
   checkpoints, separating optimizer groups and attention Q, K, and V.
3. On fixed replay windows, split NLL by controller group, action changes, and
   simple game situations. Probe history dependence and attention concentration.
4. If a component changes with a gameplay failure, test a bounded restoration
   or rescaling of that component, with the other components held fixed.

These tests may change when artifact inspection rules out a comparison. Changes
and negative results belong in the research log. A fitted offline metric must be
tested on held-out runs before it can support a claim about model selection.

## Follow-up protocols and decision log

These choices follow the initial audits. They are not independent confirmatory
tests of hypotheses chosen before seeing any historical result.

- **Weight interventions:** use the 24,576 and 57,344 Muon exports and the
  same 256-replay cohort. Restore optimizer partitions, readouts, logical
  matrix radii, or Q/K radii; retain whole-model interpolation, identity,
  reversal, and Q×2/K÷2 function-preserving controls. The completed offline
  results do not select a gameplay checkpoint.
- **Historical action likelihood:** frozen source inspection showed two
  discarded sampled ancestor frames. Score the executed +3/+4 tail by
  marginalizing those ancestors, with seed 20260913 and the same identity-
  masked replay contexts. Average likelihoods before log. The 256-particle
  pilot was followed by a common-random-number 64/256/1024 convergence check
  on all six checkpoints. Retain particle drift, likelihood-weight ESS,
  independent-half checks, and the improper expected conditional NLL as a
  contrast. No gameplay-ranking claim follows from the calculation alone.
- **Gameplay protocol treatment:** before seeing new pilot outcomes, choose
  early historical, slump historical, and the same slump weights under the
  portable committed-prefix controller. Use `short32`: seed 0, 32 boots,
  1,800 frames each, delay/replan 2, CPU level 9, the same character schedule,
  eager local CUDA, and no retries. Save every replay. Dolphin game RNG is
  not verified common between conditions; use the public character-blocked,
  boot-clustered comparison, not a frame-paired claim.
- **Attention precision correction:** a CPU autocast test confirmed that
  explicit FP32 operands did not prevent `einsum` from returning BF16 scores
  under the original outer autocast. Preserve the original artifacts and
  source snapshot. Recompute only attention measurements with autocast
  disabled for score/softmax and FP32 matmul precision `highest`. Join the
  corrected attention with unchanged functional metrics by exact hashes.
- **Hard-routing causal probe:** specified before GPU execution, on final
  Muon and final AdamW only. Hold weights, the 256 replay contexts, identity
  masking, and sequential teacher scorer fixed. Compare ordinary soft
  attention with final-query argmax-key V and current-position V in every
  trunk layer/head; keep earlier queries unchanged. Predict that argmax
  routing changes Muon's output less than AdamW's, and that forcing current
  keys is more harmful than selecting learned argmax keys. Test the soft
  reconstruction for exact public-trunk parity. A failed prediction is a
  result; do not tune the treatment to rescue it. This is not a training-
  stability intervention or a gameplay selection rule.
- **Post-hoc controller control:** the slump checkpoint's committed-prefix
  stock benefit in `short32` prompted one additional condition: the early
  checkpoint with that controller. Hold the original pilot protocol fixed
  and save a separate result/replay subtree. The four cells permit a
  checkpoint-by-controller interaction estimate. This is a post-hoc control,
  not an independent confirmation of the first pilot's result.
- **IID-data analytical example:** after the empirical probes, calculate a
  bounded binomial example with one decision state, a fallible expert,
  a fixed pseudocount estimator, and a 20-correct-action success condition.
  Average over all training counts, not Monte Carlo simulations. Independently
  enumerate short ordered label datasets as a check. This addresses a logical
  possibility, not a new explanation fitted to HAL's outcomes.

An independent-plan-per-executed-offset control was considered as a direct
test of temporal correlation. Its correct implementation needs an independent
seeded sampler stream and explicit replan/reset handling. It is documented as
an unrun next experiment, not folded into today's gameplay pilot.

## Deliverables

A report with measured findings, source links, reproducible analysis code,
machine-readable results, standalone figures, and the next experiments justified
by the evidence. New observations will be separated from prior work and from
unconfirmed explanations. No claim of novelty will rest on a literature search
alone.
