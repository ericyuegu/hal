# Matched Muon attention comparison

This plan was recorded before measuring attention in `7dgv85x8`.

## Question and selected runs

Does the low-entropy attention observed in the original Muon run also occur
in its continuation with stable recorded gradient norms?

- Reference: `p1fyyp1z`, original Muon learning rate, update 57,344.
- Control: `7dgv85x8`, half the Muon learning rate after a fork from the
  original run at update 24,576, measured at update 57,344.
- Context: retain the existing attention result for the common starting
  checkpoint at update 24,576. Do not reselect that checkpoint after looking
  at the new attention result.

The control was selected for its matching architecture, full 44-source v8
training mixture, common starting checkpoint, recorded training health, and
gameplay history. It was not selected by validation loss. The local inventory
and exact-config W&B search found no independent Muon run with this same
training identity. The O51 candidates use v7 data and are not equivalent
controls.

The control's recorded gradient norms remain finite and below the clipping
threshold through update 71,225. This is the observed time range, not a claim
that a complete 131,072-update run would stay stable. The history contains
25-update averages, not per-update gradient norms.

## Invariant measurement inputs

- Use the existing 256 distinct validation replays, one 256-frame context
  per replay, and the recorded player-rank inputs. This evaluation cohort
  comes from the ranked-anonymized-1 source; the models were trained on the
  full 44-source mixture.
- Measure only the most recent query position in all 16 layers and 16 heads.
  Retain the same masks and model representation.
- Use batch size eight, bfloat16 model projections, float32 attention scores
  and softmax, disabled TF32, and float64 summary arithmetic.
- Keep cohort example zero as the preselected individual input in the
  interactive figure. Do not select an example for its visual appearance.
- Cohort SHA-256:
  `ec61e97e332672a4e4e2fdc438e208a281a4609e10f7eded72368eccac323437`.
- Model export for the control:
  `7dgv85x8-step-0057344.model.pt`, SHA-256
  `6cbd8755d3b59690bc61a67c87651e859f729a8dd62ee358dcf4b101feb860a7`.
- Source checkpoint SHA-256:
  `89c941d64f6407eed34bc6b9ad3ccc1466aace67d48f0ef45978bcb7575a4b08`.
- Training source-selection SHA-256:
  `2593361352b92e705be3fbeae1b4e9bb1a3c9f1787cd713014a7a95b7df62477`.

## Checks and outputs

Use a new measurement entry point. Preserve the completed attention programs,
their source archives, and all previous numerical artifacts. Use the public
checkpoint loader, attention tracer, and summary calculation already tested
in this study; do not change their module state at runtime.

Before the new run, verify the model export, adjacent metadata, stored step,
run ID, architecture, vocabulary, data protocol, source manifests, and source
selection. Save the exact source snapshot, Git SHA, configuration, environment,
and input hashes. Reject an identity mismatch instead of substituting data.

The remeasured reference must have exact public-forward parity, and its
float64 summary objects and example-zero probabilities must equal the
previous detailed attention artifact exactly. Report a failed check before
interpreting the new control. Save new summaries and example probabilities
under `artifacts/attention_muon_control/`; do not overwrite existing results.

Report the directly measured largest attention weight, attention entropy,
`mean(exp(entropy))`, mean frame age, and current-frame weight. Calculate
entropy and its exponential for each input, layer, and head before averaging.
Do not infer the largest attention weight from entropy or interpret mean age
as one fixed attended position.

## Limits

This is a paired continuation, not an independent training seed. Its starting
checkpoint already has concentrated attention. Thus, finding concentrated
attention in the control would show that concentration can persist during
the observed stable continuation; it would not show how a fresh stable Muon
run develops its attention pattern.

The continuation includes recorded code and diagnostic changes. Check and
report configuration differences. The shared checkpoint and recorded resume
fields do not by themselves prove next-batch or next-update resume parity.

The branch's repeated starting gameplay evaluation differs from the parent's
starting evaluation. Do not treat a raw difference in stock rate between
their later evaluations as a clean causal effect of the learning-rate change.
No new training, resumption, gameplay evaluation, or intervention is included.

Attention weights describe one route through the model. They do not measure
all information used by a frame, and a descriptive comparison cannot show
that attention concentration caused the original gameplay regression.
