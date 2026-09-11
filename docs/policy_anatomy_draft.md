# Anatomy of a *Melee* policy

<!--
WORKING DRAFT

This file is the editable narrative companion to policy_anatomy.html.
Figure blocks are stand-ins for the finished figures in that page.
Internal experiment numbers, run IDs, and implementation notes stay in HTML
comments so they do not appear in a rendered draft.
-->

A year ago, I [trained](https://ericyuegu.com/melee-pt1/) a *Melee*
behavior cloning policy on expert human demonstrations.

My intention was to train a good-enough checkpoint to warm start self-play RL,
but a number of lingering questions compelled me to revisit whether it was
possible to train an even more capable model.

> **FIGURE 1 — Flagship architecture candidates**
>
> Stand-in for three simple views of the same model: a stacked pipeline, a
> dependency graph, and two causal attention patterns.

*Fig 1: A causal observation Transformer compresses the recent game history into
one state. A smaller causal Transformer uses that state to predict a controller
plan.*

<!--
FIGURE 1 NOTES
- Internal model: O50 production.
- Source: experiments/050_scaled_temporal_awr.py
- W&B: https://wandb.ai/ericyuegu/hal/runs/tgpuo1be
- 216,496,794 parameters.
- 8 × 2^30 = 8.59B supervised positions.
- Standard deployment: prediction 4, delay 2, replan 2.
- Result: +0.494 [+0.398] stocks/min; +52.72 [+45.14] damage/min.
- Keep parameters, optimization, scores, and deployment timing out of the
  diagram itself.
-->

- **Better encodings**
  - I made simplifying assumptions in the previous baseline that reduced the
    action space (i.e. discretizing analog controls, reducing concurrent button
    presses to the latest button), which could cause error accumulation in
    closed-loop. Can we do better? Should we directly regress on continuous
    values? How should we represent past actions to the model?
- **Prediction horizon**
  - Previous baseline made single-frame predictions—what if we predicted
    multiple frames with receding horizon control, i.e.
    [action chunking](https://arxiv.org/abs/2304.13705)?
  - The SOTA in robotics is to use flow matching policies to do so for
    continuous trajectories. I was curious whether that’d work for *Melee*.
  - There’s [evidence](https://arxiv.org/abs/2608.02547) that action chunking
    helps not just because of temporal consistency or horizon reduction, but
    due to greater non-Markovian expressivity from effectively ensembling
    policies with random delays.
- **Compute-latency tradeoff**
  - Relatedly, could longer prediction horizons allow for larger (aka smarter)
    models to run in real-time?
  - There’s a frontier of performance defined by the maximum model sizes that
    can run at a given latency on fixed hardware. Larger models might be
    smarter, but they’d have to contend with more control + observation delay.
    I reasoned the curve would have to be an inverted-U, but it wasn’t obvious
    to me a priori where the optimum would be.
  - Because of fixed sequential overhead from emulator steps and tensor
    preprocessing, I’d expect moving from 1 frame (16ms at 60fps) → 2 frames
    (32ms) would yield a disproportionate return on model performance, with
    diminishing returns after that.
- **Offline RL**
  - Can we make better use of the signals we have from the offline dataset?
    Plain behavior cloning objective is misaligned with the actual task that we
    care about: winning matches. Trivial hindsight relabeling should allow us
    to better assign responsibility for damage and stocks to certain controller
    actions.

## What I’m releasing

- Model weights + datasets — public R2 + HF links. **TODO: add links.**
- Online play server. **TODO: add link.**
- Advantage-estimate tool as a human training aid. **TODO: add link.**

## How to read the results

- `mean [LCB]` gives the mean and the saved one-sided 95% lower bound.
- Net stocks per active minute is the primary closed-loop score.
- Net damage per active minute is a denser secondary score.
- I use NLL and transition metrics to diagnose models, not select them.
- Unless stated otherwise, the modern evaluations use 96 boots against the CPU.

<!--
EDITORIAL NOTE
- Add a short glossary for prediction delay, replan interval, action horizon,
  and inference latency.
- Do not imply that validation NLL is a reliable closed-loop selector.
-->

## Experiments

### Encoding and representation learning

Next-frame prediction → independent multi-token prediction → autoregressive
multi-token prediction.

> **FIGURE 2 — From one action to a conditioned plan**
>
> Stand-in for the aligned next-frame, independent MTP, and autoregressive MTP
> diagrams.

*Fig 2: Every predicted action receives the same trunk state. In autoregressive
MTP, action i can also attend to actions 1 through i−1. Four consecutive actions
are shown for clarity.*

<!--
FIGURE 2 NOTES
- Next-frame baseline: O11, https://wandb.ai/ericyuegu/hal/runs/vlim96s9
- Independent MTP: O12, https://wandb.ai/ericyuegu/hal/runs/shjnxxsu
- Factorization ablation: O37.
- Production: O50, https://wandb.ai/ericyuegu/hal/runs/tgpuo1be
- Across-frame decoder input: trunk state + offset embedding + previous selected
  controller frame. It does not use FiLM.
- Training feeds the previous target. Play feeds the realized previous action
  and advances the temporal KV cache.
- Within each frame, decode C-stick → main stick → triggers → buttons.
- Show three candidates back to back until one is selected: dependency wires,
  attention masks, and receptive sets.
-->

The clean comparison separates two questions: should later predicted frames
depend on earlier predicted frames, and should later controller parts depend on
earlier parts from the same frame?

> **FIGURE 3 — Factorization ablation**
>
> Stand-in for the four-arm stock and damage bar chart.

*Fig 3: Across-frame feedback adds +0.361 stocks/min and +13.49 damage/min on
average. Within-frame feedback adds +0.245 stocks/min and +8.48 damage/min.*

<!--
FIGURE 3 NOTES
- Internal experiment: O37.
- Neither: -0.702 stocks/min; +2.77 damage/min.
  https://wandb.ai/ericyuegu/hal/runs/98r9smrj
- Within-frame only: -0.313; +13.05.
  https://wandb.ai/ericyuegu/hal/runs/a117chkw
- Across-frame only: -0.196; +18.06.
  https://wandb.ai/ericyuegu/hal/runs/50q39o9j
- Both / autoregressive MTP: -0.096; +24.75.
  https://wandb.ai/ericyuegu/hal/runs/5wfk2esf
- Same parameters, data, updates, initialization procedure, and deployment.
-->

#### Controller codec

The old codec collapsed simultaneous buttons and used a sparse set of analog
targets. I replaced it with an 8-bit button mask and fixed Cartesian stick
centers, then tested a polar stick layout without changing the rest of the
matched system.

> **FIGURE 4 — Three controller representations**
>
> Stand-in for the Legacy, Cartesian, and Polar codec diagrams and the two
> matched score comparisons.

*Fig 4: In matched comparisons, Cartesian beat legacy by +0.292 stocks/min and
+16.16 damage/min. Polar beat Cartesian by +0.095 and +0.88; stock LCB crossed
zero (−0.074 → +0.016).*

<!--
FIGURE 4 NOTES
- Legacy package: O43 bridge against O26.
  - Legacy: -0.132712 stocks/min; +21.371409 damage/min.
    https://wandb.ai/ericyuegu/hal/runs/acbbeb76
  - Cartesian: +0.159110 stocks/min; +37.529324 damage/min.
    https://wandb.ai/ericyuegu/hal/runs/cqbbbg77
- Polar treatment: O53 against the matched O52 Cartesian baseline.
  - Cartesian: +0.031839 stocks/min; LCB -0.074332; +33.963848
    damage/min; decode p95 15.397853 ms.
    https://wandb.ai/ericyuegu/hal/runs/yxy8brhx
  - Polar: +0.127288 stocks/min; LCB +0.015911; +34.839119
    damage/min; decode p95 15.941710 ms.
    https://wandb.ai/ericyuegu/hal/runs/yi7yyyu0
- Legacy vocabulary: buttons 6, main stick 37, C-stick 9, fused shoulders 5.
- Cartesian vocabulary: buttons 256, main stick 65, C-stick 9, triggers 25.
- Polar vocabulary: buttons 256, main stick 85, C-stick 13, triggers 25.
- Legacy uses the lowest-index newly pressed button and carries that reduced
  output until the raw chord changes.
- Cartesian stick semantics: [x, y].
- Polar stick semantics: [r, sin(theta), cos(theta)].
- Repeat Cartesian in the score chart. These are two different matched
  checkpoints, so a single three-bar ranking would be invalid.
-->

#### Ideas that didn’t work as well

- Geometric feature engineering.
- Temporal output-head cross-attention to earlier trunk context tokens.
- One-hot previous-controller conditioning.
- Flattening controller groups across frames into the main token sequence.
- BPE / RLE action spans.
- Joint action chording.
- A training-only future-state expert.
- Fixed long delay.

> **FIGURE 5 — Representation branches**
>
> Stand-in for a miniature flagship map, an aligned control/treatment diagram,
> and intervention-delta bars for each branch.

*Fig 5: The bars use one scale across the section: ±3 stocks/min and ±180
damage/min. Blank results mean that the retained run does not support an
intervention delta.*

<!--
FIGURE 5 NOTES
- Each row reuses the Figure 1 model shape. Grey the full stack and highlight
  the changed locus before showing the local mechanism.
- Loci: past-action input, observation input, trunk → plan, trunk sequence,
  plan + runtime, controller frame, training heads, and plan + runtime.
- Physical past-action embedding vs one-hot: +0.260 stock; +23.68 damage.
  Runs 1zzstd2d and cqbbbg77.
- Geometric features: no completed closed-loop treatment.
  Runs fxhtoxu8, 6mmt6o4c, 60esc6s8.
- Head cross-attention: retained run 4hoe86s7; no clean delta against cqbbbg77.
- Flattened tokens: 7tlk11io and lxo3id9l; no clean delta.
- BPE/RLE: -2.635 stock; -172.16 damage. Run xgj1dwot.
- Joint chords: only a three-step diagnostic. Run vgwbv53u.
- Future-state expert: -0.048 stock; -12.60 damage. Run qxcjjwl8.
- Fixed long delay: -1.105 stock; -85.71 damage. Runs 1imfy8v3 and
  2qs05au4.
-->

### Optimization

Muon, but not on long runs. AWR.

The small optimizer comparison changed the optimizer and learning rates
together. Muon + AdamW changed stock score by +0.030 and damage by −30.11
relative to AdamW. I used all-AdamW for the final long run.

> **FIGURE 6 — Advantage-weighted behavior cloning**
>
> Stand-in for the AWR mechanism and BC/AWR score bars.

*Fig 6: In the modern comparison, AWR changed stock score by +0.076 and damage
by +10.80. The BC evaluation used 96 boots and the AWR evaluation used 128;
they were not paired.*

<!--
FIGURE 6 NOTES
- Internal experiment: O36 against O37 D3.
- BC: https://wandb.ai/ericyuegu/hal/runs/5wfk2esf
- AWR: https://wandb.ai/ericyuegu/hal/runs/hwzv0k9a
- w = min(3.5, exp(beta A)).
- A = G(t+1) - V(s_t).
- The actor loss does not backpropagate through the value estimate.
-->

#### Ideas that didn’t work as well

- Endpoint flow matching.
- Attached IQL.
- Exploration modeling with one or eight latent trajectories.

> **FIGURE 7 — Objective and decoder branches**
>
> Stand-in for the three control/treatment mechanisms and shared-scale delta
> bars.

*Fig 7: Flow matching, attached IQL, and exploration modeling. K=1 and K=8 are
each compared with the same behavior-cloning baseline; neither beats it.*

<!--
FIGURE 7 NOTES
- IQL: -2.009 stock; -95.85 damage. Runs cqbbbg77 and 06fi5jms.
- Endpoint flow: -0.186 stock; -32.93 damage. Runs 5wfk2esf and
  t4s3gohe.
- XM K=1 against BC: -1.667 stock; -89.37 damage. Run mz8dk2gh.
- XM K=8 against BC: -0.807 stock; -44.33 damage. Run feic18gt.
- K=8 improves XM relative to K=1, but neither XM system beats BC.
- XM comparisons also change the decoder, objective, samples, and work.
-->

### Data & infra

One window per replay, sufficient shuffle block size.

Low-dim data with training sample trajectories being considerably shorter than
full episodes caused a disk-read bottleneck rather than a CPU or memory
bottleneck when training on a large GPU.

The workaround was to read more windows per episode into an in-memory buffer,
then shuffle + sample uniformly from waves of slots to ensure a minimum number
of batches before re-sampling the same window.

> **FIGURE 8 — Replay diversity inside a batch**
>
> Stand-in for the four-windows and one-window sampler diagrams and score bars.

*Fig 8: Four windows per replay → one changed stock score from −0.598 to +0.130,
a +0.728 gain. Damage changed from +40.50 to +34.27.*

<!--
FIGURE 8 NOTES
- Internal experiment: O12.
- Four windows: https://wandb.ai/ericyuegu/hal/runs/uw05bvm2
- One window: https://wandb.ai/ericyuegu/hal/runs/shjnxxsu
- Same parameters, batch, updates, examples, data, and evaluation protocol.
- Interpret this as replay diversity inside a batch, not a data-volume result.
-->

> **FIGURE 9 — Loader lineage**
>
> Stand-in for physical shards → replay ring → window waves → training batch.

*Fig 9: Loader throughput increased from 259 samples/s in the generic-loader
shakedown to 1,377 samples/s in production. These are infrastructure
measurements, not model-quality comparisons.*

<!--
FIGURE 9 NOTES
- 7,191 physical shards.
- 131,072-slot deterministic replay ring.
- Generic loader: 259 samples/s.
- First physical ring: 550 samples/s.
- Production: 1,377 samples/s and about 38 ms uncovered loader wait/update.
- Loader W&B runs: f0nd4xze, bri3007x, vrsbc7jp, a9ud18bc.
-->

## Optimal model size vs. latency

The data here is noisy.

> **FIGURE 10 — Execution horizon**
>
> Stand-in for the four factorization checkpoints across horizons 1, 2, 4,
> and 6.

*Fig 10: Across-frame autoregression is less sensitive to longer execution
horizons. This is not a replan-only result because execution length and replan
interval change together.*

<!--
FIGURE 10 NOTES
- Internal experiment: O37.
- Independent model: +0.096 stocks/min at H1 and -0.905 at H6.
- Full autoregressive model: +0.053 at H1 and -0.021 at H6.
- Same four checkpoints as Fig 3.
-->

> **FIGURE 11 — Capacity against imposed delay**
>
> Stand-in for model selection plus the delay-performance chart.

*Fig 11: Delay 1 is not automatically best, and model size does not give a
stable ordering.*

<!--
FIGURE 11 NOTES
- Internal experiment: O39.
- Source: experiments/039_capacity_scaling.py
- Use checkpoints at D = 2^30 supervised positions.
- The 7.08M baseline is d256 / L8. Display it as L8, not by experiment number.
- Preserve hollow points for models that missed the recorded real-time fit
  check.
-->

> **FIGURE 12 — Measured deployment frontier**
>
> Stand-in for the damage-selected, RTX 3060-feasible point at each delay.

*Fig 12: Best measured deployable damage result at each delay for the fixed-data
checkpoints. The stock view shows those same damage-selected models.*

<!--
FIGURE 12 NOTES
- This is not the true fixed-compute Pareto curve.
- The planned fixed-compute study, O48, was not run.
- Selection uses mean damage at each delay, then the stock view retains those
  same selected models.
-->

## Scaling experiments

Validation NLL has a broad relationship with closed-loop play in the capacity
sweep. It is not reliable enough to select a checkpoint or capacity.

> **FIGURE 13 — Validation NLL vs. closed-loop play**
>
> Stand-in for the model-budget scatterplot and rank correlation by delay.

*Fig 13: At delay 4, n = 33 and Spearman ρ = −0.67. Lower NLL broadly goes with
better play, but within-model paths often reverse.*

<!--
FIGURE 13 NOTES
- Internal experiment: O39.
- Use all healthy model-budget points at delay 4.
- Point size encodes model size; hover gives model, budget, stock, and damage.
- Rank correlations by delay range from -0.353 to -0.817.
- Keep the warning close: this is an association, not a selection rule.
-->

> **FIGURE 14 — Iso-FLOP fit**
>
> Stand-in for the fitted loss curve and training-FLOP control.

*Fig 14: Robust fit over 35 non-diverged cooldown checkpoints; RMS residual
0.0046 NLL, maximum 0.0112. The fit predicts offline NLL only.*

<!--
FIGURE 14 NOTES
- L(N,D) = 1.6350 + 89.11/N^.368 + 2009/D^.498
- Values beyond the measured model range or past one corpus epoch are
  extrapolations.
- Do not convert the NLL optimum into a gameplay claim.
-->

## Current recipe

The final model combines the pieces that survived the small-scale tests.

> **FIGURE 15 — Experiment ledger**
>
> Stand-in for the sortable intervention table linked to the architecture map.

*Fig 15: Mean changes, not lower bounds. Each row needs its comparison caveat.
“No stable order” is retained where a scalar delta would be misleading.*

<!--
FIGURE 15 NOTES
- Keep internal experiment codes and W&B links in source mode only.
- Default view should show category, intervention, and delta, not run IDs.
- Add the polar codec result: +0.095 stock and +0.88 damage.
- The full audited sources are in blog_experiment_evidence.md and
  blog_experiment_runs.csv.
-->

> **FIGURE 16 — Closed-loop performance during training**
>
> Stand-in for the training-checkpoint chart and final training statistics.

*Fig 16: Closed-loop play regressed during the middle of the earlier Muon
scale-up and recovered at the final checkpoint while validation NLL continued
to improve.*

<!--
FIGURE 16 NOTES
- Internal experiment: O50 Muon production run.
- W&B: https://wandb.ai/ericyuegu/hal/runs/p1fyyp1z
- 216,496,794 parameters; 131,072 updates; 8 × 2^30 positions.
- Final training loss 1.358 bits; validation NLL 1.579 → 1.379.
- Throughput 1,377 samples/s; 25.2% MFU; 96 boots; zero aggregate crashes.
- The later all-AdamW production checkpoint is tgpuo1be and reached
  +0.494 [+0.398] stocks/min and +52.72 [+45.14] damage/min.
-->

The old model reached +1.227 net stocks/min after 16.78M supervised examples.
The old and new evaluation systems are not identical, so the gap to the current
model’s +0.494 is descriptive, not a controlled comparison.

<!--
ENDING NOTES
- Historical run: https://wandb.ai/ericyuegu/hal/runs/wa7x0psv
- The honest headline is not “scale solved it.” Sequence factorization, replay
  diversity, optimizer-update count, and deployment timing had clearer effects.
- The Cody/Fox fast-deployment package reached +0.874 stocks/min, but identity,
  character, delay, and replan changed together.
- Possible ending: release links, video, then the remaining clean experiments.
-->
