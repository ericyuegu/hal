# Validation loss, gameplay, and optimizer behavior in HAL

Research date: 2026-09-11. Base commit:
`52da5b2de1d402b86404fc4d24e382409649e525`.

I examined existing checkpoints; I did not train new models. The study covers
the two named Experiment 050 runs and a half-rate Muon continuation,
Experiment 037's action-prediction comparisons, and Experiment 039's scaling
results.

I tested seven checkpoints on the same 256 validation replay sequences from
`ranked-anonymized-1`. This small diagnostic dataset was not sampled to
represent the full 44-source training dataset. I also ran 128 short gameplay
test sessions across four configurations.

Throughout this report, negative log-likelihood (NLL) measures how well the
model predicts recorded human actions. Lower NLL means a better match to the
demonstrations. It does not necessarily mean better gameplay. Net stocks/min
means stocks taken minus stocks lost per minute; net damage/min means damage
dealt minus damage received per minute.

## Main conclusions

1. The final Muon model concentrates attention on roughly one input position
   per head. AdamW spreads attention across many positions. Keeping only the
   position with the highest attention weight barely changes Muon's outputs,
   but worsens AdamW's NLL. The half-rate Muon continuation also retains
   concentrated attention while its recorded gradient norms stay stable.
2. The old and current inference code use different earlier actions when
   predicting an action sequence. Using the actions actually waiting in the
   controller queue helped both the early and later Muon checkpoints in the
   short gameplay test. This does not explain the training regression.
3. The scaling fit does not establish that a billion-parameter model would
   play best. Its size measure is based on compute, not raw parameter count,
   and its proposed compute budget is about 20 times the largest measured run.
4. Muon's stock rate eventually recovered above its early peak. Its damage
   rate did not. The evidence does not establish that it became permanently
   trapped in a poor local minimum. Restoring earlier weight norms did not
   improve its offline prediction score.

## 1. The scaling estimate is uncertain

I reproduced the Experiment 039 fit from the raw results: 62 runs, 315
evaluations, and 46,655 match records. The fit uses 35 completed results from
the model-size sweep after the experiment's exclusion rules.

The size variable in the fit is not the number of model parameters:

    N_eff = training_FLOPs / (6 × processed_positions)
          = N_trunk + 36 × N_decoder.

Here, `N_trunk` and `N_decoder` are the parameter counts in the main
Transformer and action decoder. The weighting accounts for their different
compute costs. Processed positions include repeated use of the same data.

The largest measured model has about 293 million parameters, but its
`N_eff` is about 504 million. At `4.2e19` FLOPs, the fit predicts an optimum
of `N_eff = 1.335 billion` and 5.24 billion processed positions, or about 2.18
passes through the dataset. That compute budget is 19.9 times the largest
completed measured budget. The result is an extrapolation, not an observed
optimum or a recommendation for a model with one billion weights.

Resampling whole model-size curves and refitting gives a 90% bootstrap
confidence interval of approximately 0.44–6.08 billion for optimal `N_eff`.
This interval assumes the chosen scaling-law model is appropriate. It does
not cover all possible extrapolation errors.

Large changes in the proposed model size and amount of training produce
little change in predicted NLL. Changing how the fit accounts for repeated
examples also changes the recommended model size substantially.

At a four-frame control delay and four-frame replanning interval, NLL and
net stocks/min have a Spearman correlation of −0.72 across the measured
models and training budgets. However, among models trained on the same
number of positions, the ranking sometimes reverses. Those comparisons
contain only four to seven models each. The overall correlation does not
show that NLL reliably selects the best model at a fixed amount of training.

Experiment 039 changes control delay and replanning interval together.
It cannot separate their effects. Future comparisons should vary these
independently and separately check whether inference meets its time limit.

Details: [scaling analysis](scaling_findings.md),
[uncertainty in the fitted optimum](artifacts/scaling/optimum_sensitivity.png).

## 2. The Muon run recovered on one gameplay measure, but not another

The Muon run `p1fyyp1z` and AdamW run `tgpuo1be` use the same 16-layer,
1,024-wide Transformer, action-decoder architecture, ordered list of 44 data
sources, random seed, and evaluation schedule.

Their learning rates differ. Both also train a value function and use
advantage-weighted regression, which gives different weights to demonstration
actions. This is neither a comparison that changes only the optimizer nor
plain, unweighted behavior cloning.

The original Muon run recorded:

| Training updates | Validation NLL | Near-term NLL | Net stocks/min | Net damage/min |
| --- | ---: | ---: | ---: | ---: |
| 24,576 | 1.4455 | 1.1099 | +0.1644 | +50.31 |
| 57,344 | 1.4410 | 1.1123 | −0.1751 | +27.19 |
| 131,072 | 1.3793 | 1.0663 | +0.3450 | +37.59 |

NLL is measured in bits per complete controller action, with the original
dashboard's weighting across prediction horizons. The near-term column
refers to the dashboard's `near` metric. These gameplay results use the
original Experiment 050 evaluation settings; they should not be combined
with results from differently configured experiments.

During the regression, combined NLL improves slightly because predictions
farther into the future improve while near-term predictions get worse.
The F1 score for controller-input changes also gets worse, even though
exact-frame accuracy improves. These distinctions help diagnose errors.
They do not establish F1 as a reliable way to select models.

By the final checkpoint, Muon's net stock rate exceeds its early peak, but
its net damage rate remains lower. The final AdamW checkpoint is better on
both measures: +0.4937 stocks/min and +52.72 damage/min.

After warmup, Muon's logs first show gradient clipping at update 28,950.
AdamW's logs show no clipping after warmup. The logged gradient norms are
averages over 25 updates, so they do not show the size of individual spikes.
The available logs show no collapse in activation magnitudes or concentration
of the training loss weights on a small number of examples.

Both W&B runs end with a `crashed` status, but both reached their final
evaluation. That status alone does not establish that training failed.

![Training loss and gameplay over time](artifacts/dynamics/named_run_dynamics.png)

Details: [run analysis](run_forensics.md),
[metric plots](artifacts/dynamics/metric_trajectories.png).

## 3. Weight norms do not identify a poor local minimum

The attention query and key matrices, Q and K, help determine which input
positions receive attention. Their Frobenius norm is the square root of the
sum of squared matrix entries: a measure of overall weight magnitude.

From Muon update 24,576 to 57,344, the median Q and K norms grow by about
37–38%. Several other matrix groups grow by 41–50%. The singular value
analysis does not show widespread rank collapse. From 98,304 to 131,072,
Q/K norms barely change while the stock rate improves.

I tested several changes to the later checkpoint on the same 256 replay
sequences. These tests use teacher forcing: when predicting a later action,
the model receives the recorded earlier actions.

- Replacing the output layers with their earlier weights worsened four-frame
  NLL. Replacing either the AdamW-trained or Muon-trained parameter group
  with its earlier weights also worsened NLL.
- Rescaling all Muon-trained matrices to their earlier norms worsened NLL.
- Restoring only the earlier Q/K norms made attention less concentrated.
  The NLL change was small and uncertain: +0.00097 nats per controller group
  per frame, with a paired 95% confidence interval of
  [−0.00012, +0.00207]. Positive values mean worse predictions.
- I tested five points along a linear interpolation between the early and
  later weights. None of the tested intermediate points had higher NLL than
  the worse endpoint. This does not rule out a loss increase between the
  sampled points, and it does not measure the full training objective.
- Multiplying Q by 2 and K by 1/2 preserved all measured logits and actions
  from greedy decoding exactly. The scaling cancels in the attention dot
  product, including with rotary position embeddings.

The last test shows why individual matrix norms cannot, on their own,
describe how the model behaves. The same model function can have different
weight norms.

Loss curvature has a related limitation: changing how a function is
parameterized can change its apparent sharpness. A Hessian measurement
therefore needs care when comparing models. I did not calculate Hessians in
this study. [Dinh et al., 2017](https://proceedings.mlr.press/v70/dinh17b.html).

There is no general requirement to keep every matrix's Frobenius norm
constant. Hyperball analyzes weight magnitude and changes in weight direction
under specific assumptions. Its assumptions about scale-invariant networks
and idealized optimizer updates do not fully match HAL's residual network,
approximate Muon updates, and changing learning rate. The paper suggests
useful measurements, but does not establish the cause of this regression.
[Hyperball](https://arxiv.org/abs/2606.16899).

Using AdamW for embeddings and output layers alongside Muon for matrix
weights is a standard arrangement. It is not, by itself, evidence of an
incompatibility. The optimizers transform gradients differently, so matching
their numerical learning-rate or weight-decay settings does not match the
change they cause in model predictions. Clipping the original gradient also
does not directly set the magnitude of the final orthogonalized Muon update.
[PyTorch Muon documentation](https://docs.pytorch.org/docs/main/generated/torch.optim.Muon.html).

Restoring old output weights after a regression is not the same experiment
as freezing those weights before the regression and continuing training.
A proper freeze comparison would start both runs from the same saved training
state, restoring the optimizer, scheduler, data-loader position, and random
number generators.

A useful next test would take one optimizer step on the same training batch
and measure both the change in weight direction and the change in predicted
action probabilities. This study did not run that test: the downloaded
files contain model weights, not optimizer state.

![Weight norms and singular value measurements](artifacts/spectra/matrix_geometry.png)

Details: [singular value analysis](spectral_findings.md),
[weight-change experiments](intervention_findings.md).

## 4. Muon attention is concentrated on almost one frame per head

An attention head combines information from input positions using a set of
weights that sum to one. I measured these weights when each head processes
the most recent input frame. Each head can attend to that frame and the
255 earlier frames in the input.

The measurements use 256 distinct validation replays, with one input
sequence per replay, at the input Transformer's 16 layers and 16 heads per
layer. That gives 65,536 attention distributions per checkpoint. These are
not 65,536 independent replays. All checkpoints use the same input sequences.

| Checkpoint | Mean largest attention weight | Attention entropy, nats | Effective number of attended frames | Average age of attended positions, frames |
| --- | ---: | ---: | ---: | ---: |
| Muon 24,576 | 94.03% | 0.150 | 1.25 | 70.4 |
| Muon 57,344 | 98.21% | 0.043 | 1.06 | 65.6 |
| Muon 98,304 | 98.67% | 0.032 | 1.04 | 62.6 |
| Muon 131,072 | 98.66% | 0.032 | 1.04 | 62.1 |
| AdamW 57,344 | 25.02% | 2.942 | 37.43 | 63.5 |
| AdamW 131,072 | 21.83% | 3.128 | 43.18 | 63.6 |

### How much weight goes to the strongest position?

For final Muon, the strongest position gets **98.66% on average**. The other
255 positions share the remaining 1.34%. For final AdamW, the strongest
position gets 21.83%. I find the largest weight separately for each input,
layer, and head, then average. This is a direct measurement, not a value
inferred from entropy.

The pattern is not identical in every head. Final Muon's per-head mean
largest weights range from 80.01% to 100%. Some individual inputs have more
spread-out attention. The median largest weight across all input/head
distributions is 100% at the recorded float32 precision.

The average age of 62.1 frames is a different quantity: first calculate
`sum(attention weight × frame age)`, then average over inputs, layers, and
heads. It is not the age of one fixed target. In final Muon:

- The exact position 62 frames back gets only **0.35%** of attention on average.
- The current position gets **1.83%** on average.
- The most common strongest position is four frames back, but it is the
  strongest in only **3.88%** of the input/head distributions.

Which position receives most attention changes across inputs, heads, and
layers. Do not read the mean age as the position receiving 98.66%.

### What attention entropy in nats means

Entropy measures how spread out the attention weights are. For weights
`p`, I calculate `H = −sum(p × ln(p))`. A zero weight contributes zero.
The natural logarithm, `ln`, makes the unit **nats**. Using a base-two
logarithm would express the same information in bits: divide nats by
`ln(2)` to convert. Entropy here describes attention weights, not action
prediction error.

If one frame gets all the weight, entropy is zero. Equal attention to two
frames gives `ln(2) = 0.693` nats; equal attention to ten gives
`ln(10) = 2.303` nats. Equal attention to all 256 would give 5.545 nats.
Lower entropy means more concentrated attention.

The **effective number of attended frames** is `exp(H)`. It is the number
of equally weighted frames that would have the same entropy. It is not a
count of nonzero weights. For example, weights of 99% and 1% give 0.056
nats and 1.058 effective frames, although two frames have nonzero weight.

I calculate entropy and its exponential separately for each replay, layer,
and head, then average each measurement. The table therefore reports
`mean(exp(H))`, not `exp(mean(H))`. For final AdamW, these are 43.18 and
22.83, respectively. Unequal concentration across replays and heads makes
them differ. I also do not average the attention weights before calculating
entropy: several heads can each select one frame while their combined
average spreads across many frames.

Both final models attend to positions of a similar average age. Muon's
attention is much more concentrated. This pattern remains after its stock
rate recovers, so concentrated attention is not enough to identify the
gameplay regression.

These measurements concern only the most recent query position, not every
position in the sequence. The table retains the replay's player-rank inputs.
Attention scores and softmax use 32-bit floating point with TF32 disabled;
the preceding projections use the model's normal bfloat16 precision.

The follow-up reproduces the previous table values at their reported
precision. It uses float64 to summarize the stored float32 probabilities;
the previous summary used float32. The largest per-head difference is
0.0000215 frames in mean age. The traced model output still matches the
normal forward path exactly. The validation log records the initial
comparison-tolerance failure and its correction.

### Interactive figure

The [interactive attention comparison](artifacts/attention_detail/attention-comparison.html)
shows the two final checkpoints. Each heatmap cell gives one head's mean
largest weight across the 256 replays. Hover over a cell for entropy,
effective frames, mean age, and current-frame weight.

Select a layer and head to see its individual attention weights for one
fixed replay sequence. This is cohort example zero, chosen before inspecting
the new measurements. These lower plots show one input, not a replay average.
Age zero is the most recent frame. Layer and head numbers start at zero.
The same head number in the two models does not imply the same function.

Concentrated attention does not mean that the whole model uses only one
frame. Different heads and layers can select different positions. A value
vector at a selected position can already contain information from other
frames through earlier layers. The residual path also carries information
without going through that head's attention weights.

### Repeat check on the requested AdamW run

The W&B link for `tgpuo1be` and the long run name beginning
`260909-094223_050_scaled_temporal_awr_` identify the same run. It was already
the AdamW control in the previous figure. I repeated the measurement with
the unchanged program, checkpoints, and replay inputs. All recorded summary
values and the saved example-zero probability arrays reproduced exactly.

The [AdamW checkpoint comparison](artifacts/attention_adamw_repeat/adamw-attention-comparison.html)
shows this run at 57,344 and 131,072 steps, with its run ID in both labels.
The mean largest attention weight falls from **25.02% to 21.83%**, while the
mean effective frame count rises from **37.43 to 43.18**. On average, neither
checkpoint has the near-one-frame concentration seen in Muon.

The recorded training history supports the description of this run as
non-diverged: losses and gradient norms are finite, and post-warm-up gradient
norms stay below 0.76 with no recorded gradient clipping. W&B's process status
is `crashed`; that status alone does not show an optimizer failure.

This is a repeat check of one AdamW training run, not an additional independent
control. It does not establish that concentrated attention causes Muon's
gameplay regression. The [run inventory](attention_controls_inventory.md)
records checkpoint identities and the training-history evidence; the
[repeat check](artifacts/attention_adamw_repeat/repeat-parity.json) records
the exact agreement with the previous measurement.

### Matched Muon control: a stable half-rate continuation

The Muon continuation `7dgv85x8` also has highly concentrated attention.
At update 57,344, its strongest position receives **94.97% on average**,
compared with 98.21% in the original Muon run at the same update. Low
attention entropy is therefore not limited to the period with large
recorded gradient norms.

This is the closest matching Muon control found in the run inventory.
It starts from `p1fyyp1z` at update 24,576 and halves the Muon learning rate.
The architecture, full 44-source v8 training mixture, representation,
vocabulary, objective, and AdamW settings match. The continuation adds
diagnostic logging and uses a later experiment-code version. Both attention
measurements use the same 256 replay inputs and the same calculation as above.

| Muon checkpoint at update 57,344 | Mean largest weight | Entropy, nats | Effective frames | Mean age, frames |
| --- | ---: | ---: | ---: | ---: |
| Original learning rate, `p1fyyp1z` | 98.21% | 0.0429 | 1.061 | 65.6 |
| Half-rate continuation, `7dgv85x8` | 94.97% | 0.1249 | 1.197 | 69.6 |

The current frame receives only **1.17%** of the continuation's attention
on average. Its most common strongest position is four frames back, but
that position is strongest in only 3.14% of the input/head distributions.
Thus, this control also does not attend mainly to the most recent frame.
The selected position changes across inputs, heads, and layers.

The difference is largest in the first layer: its mean largest weight is
81.1% in the continuation, versus 92.2% in the original run. Later layers
remain strongly concentrated. The [matched Muon comparison](artifacts/attention_muon_control/muon-baseline-comparison.html)
shows these differences by layer and head. Hover over a cell for replay
averages, or select a layer and head to inspect the weights for fixed replay
example zero. These are the same controls and example selection as before.

Here, “stable” refers to the recorded numerical training behavior. The
continuation's 25-update mean gradient norms range from 0.248 to 0.338
through update 71,225. It has occasional clipping, but no observed numerical
divergence in that range. This is not a completed 131,072-update run, and
stable gradients do not mean that gameplay performance was constant.

The main limit is that this is a continuation, not an independent run from
initialization. Its starting checkpoint already puts 94.03% of attention
on the strongest position on average. The new result shows that this
concentration persists during the observed stable continuation. It does
not establish how a fresh stable Muon run would develop its attention,
or whether attention concentration caused the original gameplay regression.

Both new measurements have exact agreement between the traced and normal
model outputs. The remeasured original-run summary and saved example
probabilities also match the previous artifacts exactly. The
[control inventory](attention_muon_control_inventory.md),
[measurement plan](attention_muon_control_plan.md), and
[validation log](validation.md) record the selection, checks, and the
model-export file correction made before attention measurement.

### What happens if each head uses only its strongest attention position?

For each head, I replaced its weighted average with the value vector at the
position receiving the highest attention weight. I made this change only for
the most recent query position at every Transformer layer. Earlier query
positions were unchanged, and later layers were computed normally.

This test used the final checkpoints and masked player identity to zero,
as in gameplay.

| Final checkpoint | NLL change, bits/frame | Paired replay 95% confidence interval | Greedy action groups changed |
| --- | ---: | ---: | ---: |
| Muon | −0.0012 | [−0.0044, +0.0020] | 0.37% |
| AdamW | +0.0577 | [+0.0409, +0.0753] | 1.71% |

NLL is averaged over four predicted frames. The action groups are buttons,
main stick, C-stick, and triggers. The mean Jensen–Shannon divergences,
which measure changes in the predicted probability distributions, are
0.000054 nats for Muon and 0.001289 for AdamW under teacher forcing.

On these inputs, Muon behaves almost the same when each head uses only its
strongest attention position. AdamW makes more use of the remaining positions.

### Both models use past input positions

Forcing each head to use only the current position increases NLL by
0.803 bits/frame for Muon and 0.711 for AdamW.
Both models therefore use past positions.

In a separate test, masking all but the last 32 input frames hurts AdamW at
least as much as Muon. Concentrated attention does not imply greater
dependence on distant history.

The unmodified test code reproduced the normal model output exactly.
A separate check on eight replay sequences found no numerical difference
between applying the output projection to one row and applying it to a full
sequence. This held at all 16 layers for both models and both attention
changes. It checks a possible implementation error, but is not a second
full-dataset experiment.

The concentrated Muon pattern also appears in six recorded contexts from
model-controlled games. Those measurements used CPU float32 throughout,
unlike the GPU measurements above. They do not reconstruct the original live
action queue or random sampling state.

These results show a difference in how the models use attention. They do not
show that this difference caused the gameplay regression. The models have
different learning-rate settings, and the attention changes were tested
offline, not in gameplay.

The next question is whether a small state change can switch which position
receives most of a head's attention and cause a large action change. A useful
test would measure how close the two largest attention scores are, change
which value is used, and measure the resulting prediction difference. Then
test relevant state changes in the simulator from reproducible starting
conditions. I have not run this experiment.

Previous work connects low attention entropy with training instability in
other settings. That connection does not establish a failure mechanism here:
HAL's Muon model remains highly concentrated during its stock-rate recovery.
[Zhai et al., 2023](https://proceedings.mlr.press/v202/zhai23a.html).

![Attention concentration and the effect of removing history](artifacts/attention_fp32/attention_contrast.png)

Details: [attention experiments](o50_findings.md),
[corrected attention measurements](artifacts/o50_attention_fp32/),
[detailed attention measurements](artifacts/attention_detail/attention_detail.json),
[follow-up measurement plan](attention_detail_plan.md),
[checks on model-controlled replay inputs](experiment_map.md#learner-visited-replay-feasibility).

## 5. One validation test can penalize consistent action sequences

An autoregressive action model predicts each part of an action sequence
using the earlier parts. One validation test supplies model-generated earlier
actions, then measures how well the model predicts the remaining human
demonstration.

This can pair the beginning of one valid plan with the end of a different
plan. The model may be penalized for following its own plan consistently.
The problem with this training or scoring construction is established
prior work. [Huszár, 2015](https://arxiv.org/abs/1511.05101).

Experiment 037 illustrates why this distinction matters. It compares models
with the same size and training data. The model that predicts future frames
and controller groups independently has almost no increase in the reported
loss when generated actions replace recorded earlier actions. It gets
−0.905 net stocks/min when executing six-frame action sequences.

The fully autoregressive model has the largest increase in that loss,
about 2.469 bits/frame, but the best six-frame result: −0.021 stocks/min.
This does not show that prediction errors help. It shows that the loss
increase is not, by itself, evidence of worse control.

The reported loss averages ten prediction offsets and uses the experiment's
finite-loss calculation. With exact controller legality masks, substituting
generated earlier inputs can make a recorded target impossible, giving
infinite NLL. These finite scores are therefore not the exact likelihood
of the deployed controller's actions.

A simple example makes the issue explicit. Demonstrations are equally likely
to be `000000` or `111111`, and either complete sequence succeeds. As a model
learns to repeat its first choice, both demonstration likelihood and success
improve. A test that combines its generated beginning with the other
demonstration's ending can instead produce a loss that tends to infinity.
This is a mathematical example, not a finding about a specific HAL component.

![An inconsistent target can give a misleading validation score](artifacts/metrics/mixed_prefix_counterexample.png)

To test whether consistency across actions helps gameplay, generate
independent complete plans from the same context and take a different
executed action from each plan. This preserves the probability distribution
at each individual action position but removes dependence between those
positions. Compare the resulting gameplay with executing actions from one
jointly generated plan. Later game states can differ; that is what the test
is intended to measure. This experiment has not been run.

## 6. The old and current inference code use different earlier actions

In both original named runs, the model generates four future actions,
discards the first two, and later executes actions three and four. Because
generation is autoregressive, the discarded actions still affect the actions
that are executed.

The current O50 inference code instead supplies the two actions already
waiting in the controller queue, then predicts actions three and four using
those known actions. The weights can be identical while this change alters
the policy that actually plays.

### Gameplay comparison

I compared these approaches with 32 test sessions per configuration. Each
session used a fixed character matchup, 1,800 frames, a level-9 CPU opponent,
two-frame control delay, replanning every two frames, and 256 frames of
context. Inference ran without compilation on the local RTX 3060. A session
is one emulator launch with a frame budget, not necessarily a complete match.

The initial three configurations and one later control each completed
32/32 sessions, with no crashes, retries, or exclusions.

| Checkpoint and inference method | Net stocks/min | Net damage/min |
| --- | ---: | ---: |
| Early 24,576, original method | −0.134 | +49.71 |
| Same early weights, actual queued actions | +0.738 | +83.20 |
| Later 57,344, original method | −0.738 | +12.86 |
| Same later weights, actual queued actions | 0.000 | +32.75 |

For the later checkpoint, using actual queued actions improves stock rate
by +0.738 stocks/min. The 95% bootstrap confidence interval is
[+0.067, +1.476]. Damage improves by +19.89/min, but its interval,
[−22.57, +62.77], includes zero.

The early-to-later change has the same direction as the original training
regression. However, the short test's confidence intervals do not establish
that checkpoint difference on their own.

The comparisons use matching character schedules and policy random seeds.
I did not verify identical random events inside Dolphin. Confidence
intervals resample whole sessions and keep stocks taken and lost together
when calculating their difference. All sessions have equal active time.
These are exploratory estimates, not confidence intervals adjusted for
every comparison examined.

The original controller comparison was chosen from the source code before
seeing its gameplay results. After observing the later checkpoint's
improvement, I added the early checkpoint under the current inference method.

That early checkpoint improves by +0.872 stocks/min. The difference between
the later and early improvements is −0.134 stocks/min, with a 95% interval
of [−1.208, +0.872]. There is no evidence that the benefit is specific to the
regressed checkpoint. The early checkpoint remains better with either
inference method. The wide interval also does not prove that the effects
are equal.

These short tests are not equivalent to the original evaluations with
96 sessions and 7,200 frames per session.

### Measuring the probability of the actions actually executed

For the original inference method, the probability of actions three and four
must average over all possible discarded first and second actions:

    p(a3, a4 | h) = sum_z p(z | h) p(a3, a4 | z, h),  z = (a1, a2).

Here, `h` is the model's input history. This is a marginal likelihood.
To estimate its NLL by sampling, average the probabilities first and then
take the negative logarithm. Averaging the negative log probabilities
calculates a different quantity.

I implemented this calculation using the existing decoder, its action
legality rules, matched random samples across checkpoints, and the same
player-identity masking as gameplay. With 1,024 sampled first-two-action
sequences per replay, the results are:

| Checkpoint | NLL given recorded earlier actions | Marginal NLL of executed actions | Mean NLL given sampled earlier actions |
| --- | ---: | ---: | ---: |
| Muon 24,576 | 1.053 | 1.543 | 2.708 |
| Muon 57,344 | 1.083 | 1.554 | 2.731 |
| Muon 98,304 | 1.048 | 1.512 | 2.679 |
| Muon 131,072 | 1.043 | 1.491 | 2.650 |
| AdamW 57,344 | 1.033 | 1.464 | 2.532 |
| AdamW 131,072 | 0.982 | 1.403 | 2.532 |

All values are bits per executed frame. The prediction targets remain the
human actions at positions three and four; only the handling of the first
two actions changes. The columns therefore measure different quantities,
and their absolute values are not interchangeable.

This does not establish a new metric for ranking gameplay performance.
The early-to-later difference in marginal NLL is +0.0109 bits/frame, with a
replay-bootstrap interval of [−0.0412, +0.0608]. That interval does not
include uncertainty from sampling the discarded actions.

Splitting the samples into two independent halves gives checkpoint
differences that disagree by 0.0208 bits/frame, more than the estimated
effect. For some replay sequences, almost the entire likelihood estimate
comes from one sample: the effective sample size is close to one despite
drawing 1,024 samples. High effective sample size on other replays does not
rule out important action sequences that were never sampled.

Finally, these inputs still come from human replays, not from the model
playing against a CPU. Matching the inference calculation does not remove
that difference in the states being evaluated. Implementation tests support
the calculation, but do not establish bit-for-bit agreement with the
original compiled GPU code.

Details: [likelihood results](artifacts/tail_likelihood_summary.json),
[gameplay test design](historical_gameplay_plan.md),
[four-configuration comparison](artifacts/forensics/gameplay_short32_factorial_analysis.json).

## 7. When can validation loss predict gameplay?

Poor coverage and a large state space can make behavior cloning difficult.
Neither is necessary for validation NLL and task success to disagree.

For example, suppose a demonstrator makes the correct choice 99% of the
time. Matching it minimizes population NLL at 0.0560 nats. A policy that
makes the correct choice 99.99% of the time has higher NLL, 0.0922 nats,
because it assigns very low probability to the demonstrator's rare mistakes.

A second problem is that average NLL treats errors according to their
frequency and probability, not their effect on the task. Matching common,
harmless choices can outweigh getting a rare, important decision right.
This can happen even when the demonstration distribution includes every
relevant state and action.

### What log loss can guarantee

Under appropriate assumptions, close agreement with the expert can bound
the difference in expected task reward. The mathematical statement uses
KL divergence, a measure of the difference between probability distributions.

Let `D_KL` be the KL divergence from the expert's distribution of complete
trajectories to the model's. If total reward has a range of width `R`, then

    absolute difference in expected reward ≤ R × sqrt(D_KL / 2).

When the starting-state distribution, environment dynamics, and opponent
are the same, `D_KL` is the sum of the action-level KL divergences averaged
over expert trajectories. This requires action probabilities that include
the necessary history and match the policy actually executed.

For a stochastic expert, the action KL corresponds to excess log loss:
NLL above the loss that remains even when the model exactly matches the
expert's action probabilities.

This is an upper bound, not a ranking rule. Reducing an upper bound does not
establish which of two models plays better. The bound can also be loose over
thousands of decisions. Stronger results exist when the model class can
represent the expert and other assumptions on model complexity and total
reward hold. Quadratic error growth with task length is not inevitable in
all behavior-cloning problems.
[Foster, Block, and Misra, 2024](https://proceedings.nips.cc/paper_files/paper/2024/file/da84e39ae51fd26bb5110d9659c06e13-Paper-Conference.pdf).

For HAL, I would evaluate a proposed validation metric as follows:

1. Make its action probabilities and timing match the deployed controller.
2. Report errors separately for held inputs and input changes, near-term
   and later actions, and important and routine game situations.
3. Test the same error measures on states reached by the model, where valid
   target actions or well-defined alternative-action tests are available.
4. Develop the metric on one set of runs. Test whether it ranks new random
   seeds, model sizes, and timing settings correctly using separate gameplay
   results that were not used to design the metric.

The six checkpoints examined here are not enough to establish such a metric.

### Why more data can sometimes reduce success

There is also a simple mathematical example with independent samples from
the same demonstration distribution.

At one decision state, an expert chooses the correct action with probability
0.9. Success requires 20 consecutive correct actions. Fit a policy from `n`
expert decisions, adding a fixed smoothing count of 0.01 to each action.
If `K` of the recorded decisions are correct, the model chooses the correct
action with probability

    q = (K + 0.01) / (n + 0.02),  success probability = q^20.

Averaging over all possible training datasets, increasing `n` from 1 to 512
reduces population validation NLL from 0.841 to 0.326 nats/action. At the
same time, expected success falls from 0.739 to 0.127. Both decrease at
every plotted sample count.

With very little data, the fitted policy usually chooses the correct action
almost deterministically, and occasionally chooses the wrong action almost
deterministically. More data reduces this variation between fitted policies
and makes them resemble the fallible expert. The expert's own success rate
is `0.9^20 ≈ 0.122`.

This result averages over independent training datasets. It does not say
that every individual dataset will show the same trend as examples are
added. Training decisions are sampled independently from the same decision
state, rather than collected along one uninterrupted episode. The fitted
policy samples actions from its learned probabilities during evaluation.

There are no unseen decision states. The model approaches the expert's
performance while absolute task success falls. This illustrates a possible
failure of NLL as a success measure; it is not evidence that HAL should use
less data.

![More data can improve imitation while reducing task success](artifacts/metrics_iid_data/iid_data_counterexample.png)

Adding independent data from the same distribution leaves the population
training target unchanged, but changes the variability of the fitted model.
Adding different players, skill levels, or situations can change the target
itself. Experiments should separate those effects from a fixed compute
budget, fewer training passes per example, and a loss that gives too little
weight to important decisions.

Observation delay and infrequent replanning introduce another issue:
the controller receives less timely feedback. Autoregressive action
sequences can remain internally consistent, but do not replace missing
feedback. A larger model's prediction advantage therefore need not produce
the same gameplay improvement at every delay. Raw NLL values at different
prediction horizons can also reflect different amounts of unavoidable
uncertainty.

## 8. Existing research directly connects training noise and unstable control

The most relevant connection I found is *Butterfly Effects of SGD Noise*,
which studies behavior cloning itself. Small parameter changes during
training can lead to large changes in task reward as their effects
accumulate during control, even while supervised loss changes smoothly.
The paper calls this gradient variance amplification and studies remedies
including averaging model weights across training updates.
[Block et al., ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f8b06ac8601473c3232aec9facbe79ad-Abstract-Conference.html).

This is a possible explanation to test, not an established explanation
of HAL. A suitable experiment would compare nearby checkpoints from the
same simulator starting states and with matched randomness. Measure how
quickly their states and actions diverge. Also compare two identical copies
of the same checkpoint to measure variation from the test setup itself.

A plot that looks fractal is not enough to establish chaos or fractal
structure in the game dynamics.

I found no evidence of grokking or a model remaining stuck in a poor local
minimum. I also did not identify a set of network components responsible for
a specific gameplay behavior. The clearest result is the difference in
attention concentration. The next step is to test whether particular
attention changes cause particular gameplay changes.

Further sources and proposed experiments: [literature review](literature.md).

## Files, limitations, and validation

The report, analysis code, tests, and results are in
`/home/ericgu/src/hal/runs/mechinterp-20260911-worktree`, on branch
`research/mechinterp-20260911`. The research additions are uncommitted.
Model exports and gameplay replays are stored separately in
`/home/ericgu/src/hal/runs/mechinterp-20260911-scratch`.

- Model and data identities were checked against their recorded hashes.
- Offline uncertainty estimates use whole replays; gameplay estimates use
  whole emulator sessions. Individual frames are not counted as independent
  observations.
- These checkpoints come from existing runs. They are not six independent
  training repetitions.
- No original training run, W&B record, or stable HAL source was changed.
- A proposed upload of HAL source and replay-derived data to Modal was
  blocked by the safety reviewer. The remaining model tests ran locally.
- Failed tests and numerical corrections are retained, including the
  attention-precision correction and uncertainty in the sampled likelihood
  estimates.

The [validation log](validation.md) records exact commands, failures, missing
fixtures, and successful reruns. The [research plan](plan.md) records the
initial experiments and controls. The [experiment overview](experiment_map.md)
covers the other runs examined.

The original study passed 1,183 non-integration tests, 26 additional
gameplay-protocol tests, and all seven required emulator integration tests,
with no skips. The attention follow-up passed 11 focused tests and the
expanded suite of 1,194 non-integration tests, also with no skips. Formatting
and both maintained-code type checks passed.

The standard lint command reported 51 import-order errors in unchanged
files because Ruff classifies W&B imports differently without the original
worktree's ignored cache directory. Explicitly matching the original import
grouping passed; the original main worktree also passed the standard lint
command. No frozen experiment was edited to change this result.

The final GPU-status check reported an NVIDIA driver/NVML library version
mismatch. PyTorch still reported CUDA available. Both checks are recorded
in the validation log.

This revision adds direct measurements of the largest attention weight,
position-specific summaries, an interactive figure, and the entropy
explanation in section 4. Previous numerical artifacts are unchanged.
The figure passed browser checks at widths of 736 and 360 pixels in light
and dark themes, including hover, touch, and layer/head selection. Its
embedded individual attention weights exactly match the measured float32
arrays after reversing key order to frame age.

The follow-up changes no replay extraction, controller input, session code,
or offline/live representation. The emulator integration and additional
gameplay-protocol tests were not rerun for this attention-only addition;
their results above belong to the original study.
