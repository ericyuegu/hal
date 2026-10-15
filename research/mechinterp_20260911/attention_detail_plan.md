# Follow-up: attention weights and effective frame count

This follow-up answers which positions receive attention and how much weight
the strongest position receives. It adds measurements and an interactive
figure. It does not train, select, or change a model.

## Comparison and fixed inputs

- Compare the same four Muon and two AdamW exports used in the corrected
  attention audit. These are checkpoints from two runs, not six independent
  training repetitions.
- Use the existing cohort of 256 distinct validation replays, with one
  256-frame input sequence per replay. The cohort SHA-256 is
  `ec61e97e332672a4e4e2fdc438e208a281a4609e10f7eded72368eccac323437`.
- Retain the recorded player-rank inputs, as in the corrected entropy audit.
  Do not substitute the identity-zero inputs used in the gameplay tests.
- Measure the final query position at all 16 layers and 16 heads. Earlier
  queries and the model's output remain unchanged.
- Use the existing public tracing path. Attention scores and softmax must be
  float32, with TF32 disabled. Keep the model's normal projection precision.
- Check that the traced trunk output exactly matches the public forward
  path. Check finite, nonnegative, normalized attention weights.
- Reproduce the previous per-head entropy, effective frame count, and mean
  age before accepting the new measurements. Record tolerances and errors.
- Record checkpoint, cohort, reference-result, and source hashes; archive
  the measurement source at launch. Record the resolved configuration,
  software versions, random seed, and floating-point settings.

## Measurements

For each replay, layer, and head, measure the largest attention weight,
entropy using natural logarithms, its exponential, attention-weighted mean
age, and the age of the position with the largest weight. Age zero is the
current input; age 255 is the oldest.

Average each quantity only after calculating it for each individual
attention distribution. In particular, the mean effective frame count is
the mean of the exponentials, not the exponential of the mean entropy.
Do not infer the largest weight from either entropy measure.

Also measure the mean attention distribution by age, the frequency with
which each age has the largest weight, and the mean weight at the integer
age nearest the overall mean age. These are different measurements: the
weighted mean age need not be a position with a large attention weight.

Keep the full float32 attention weights for cohort example index zero at
every checkpoint. This example is fixed before looking at the new results;
it is an illustration, not a sample selected for a striking pattern.
The interactive figure compares the two final checkpoints and allows
selection of a layer and head. Matching head indices across models does
not establish that the heads perform the same function.

## Checks and limits

Focused tests cover one-position attention, uniform attention, unequal
weights, averaging before versus after exponentiation, changing peak
positions, age direction, and invalid probabilities or tensor shapes.

These are descriptive measurements of a fixed replay cohort. Heads and
layers within one replay are not independent samples. No gameplay effect,
causal optimizer effect, or statistical population claim follows from
these averages. An attention weight is not a measure of the total effect
of an input frame: other heads, earlier layers, and the residual path can
also carry information from that frame.
