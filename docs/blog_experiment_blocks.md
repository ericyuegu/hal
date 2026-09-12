# Outline

This is a set of text blocks, not a finished article. Move or remove a block without changing its meaning.

The machine-readable source is [blog_experiment_data.json](blog_experiment_data.json). The older run-level audit is [blog_experiment_runs.csv](blog_experiment_runs.csv). The main existing interactive is [policy_anatomy.html](policy_anatomy.html). Claude's complementary diagrams are in [policy_anatomy_artifact_export.html](policy_anatomy_artifact_export.html).

All scores are net stocks per active minute and net damage per active minute. Higher is better. A delta is treatment minus control. `±` is used only for a direct 95% interval for that delta. Most old runs saved an arm mean and a one-sided lower bound, but not the paired match rows needed for a correct interval on the difference. Those rows show the point delta only. Do not subtract arm lower bounds.

## Experiments

### Action space

- Controller codec

  **Block name: Two controller-codec bridges**

  There is no valid three-way race between the legacy, Cartesian, and polar codecs. There are two matched tests from two model families.

  | Matched test | Control | Treatment | Stock control → treatment | Δ stock | Damage control → treatment | Δ damage | What it shows |
  |---|---|---|---:|---:|---:|---:|---|
  | O26/O43, legacy classes | Legacy | Cartesian | −0.133 → +0.159 | **+0.292** | +21.4 → +37.5 | **+16.2** | The legacy codec was worse. It does not explain the old model's strength. |
  | O52/O53, stick grid | Cartesian | Polar | +0.032 → +0.127 | **+0.095** | +34.0 → +34.8 | **+0.88** | The fixed polar grid gave a small stock gain and almost no damage gain. |

  The first test holds the O26 loss and the 15.05M architecture fixed. It uses batch 512, 16,384 updates, 8.39M examples, ranked-anonymized-1 v7, and matched H4 evaluation. The second test holds the 14.48M O52 proxy, AdamW recipe, batch 512, 16,384 updates, (2^{30}) supervised positions, ranked-anonymized-1 v8, D2/R2 timing, and 96 matchups fixed. O53 has 10,072 extra parameters, or 0.07%, because it has 24 extra stick classes.

  **Sources:** O26 [`cqbbbg77`](https://wandb.ai/ericyuegu/hal/runs/cqbbbg77) · legacy bridge [`acbbeb76`](https://wandb.ai/ericyuegu/hal/runs/acbbeb76) · O52 Cartesian [`yxy8brhx`](https://wandb.ai/ericyuegu/hal/runs/yxy8brhx) · O53 polar [`yi7yyyu0`](https://wandb.ai/ericyuegu/hal/runs/yi7yyyu0) · [O43 code](../experiments/043_legacy_codec.py) · [O53 preregistration](../experiments/o53/preregistration.md)

  - Fig: legacy vs. Cartesian vs. polar; perf bar chart (refine)

    **Figure name: Codec Bridges.** Use two small matched panels. Do not put all three codecs on one shared baseline. Draft: [controller-codec-bridges.svg](blog_figures/controller-codec-bridges.svg).

    **Reusable detail:** the existing `io-paths` figure separates the past-action input path from the predicted-action output path: [artifact](policy_anatomy_artifact_export.html#fig-codec).

- Adding ICs, projectiles

  **Block name: One observation token**

  “ICs” means the Ice Climbers follower. Each frame has four player blocks in a fixed order: ego, ego follower, opponent follower, opponent. Each block contains six normalized continuous values, missing-value masks, and learned categorical embeddings. Missing followers are represented by masks.

  Four projectile slots are processed separately. One shared encoder reads projectile type, state, position, velocity, masks, and presence. Presence gates each encoded slot. The four vectors are summed, so empty slots contribute exactly zero and slot order does not affect the pooled result. The player blocks, past action, characters, stage, and pooled projectile vector are concatenated and projected once. The ego-player identity projection is then added to the token.

  | Test | Stock control → treatment | Δ stock | Damage control → treatment | Δ damage | Evidence |
  |---|---:|---:|---:|---:|---|
  | No projectiles → four pooled slots | −1.428 → −1.336 | **+0.092** | −44.4 → −33.9 | **+10.5** | Strong one-factor test |
  | No follower fields → follower fields | — | — | — | — | No clean gameplay test found |

  The projectile test used a 7.1M policy, batch 128, 16,384 updates, 2.10M examples, and ranked-anonymized-1 v7.

  **Sources:** projectiles off [`v9xku05q`](https://wandb.ai/ericyuegu/hal/runs/v9xku05q) · projectiles on [`9wh7qhr1`](https://wandb.ai/ericyuegu/hal/runs/9wh7qhr1) · [O41 code](../experiments/041_architectural_stability.py) · [shared feature contract](../hal/training/features.py) · [O50 token construction](../experiments/050_scaled_temporal_awr.py)

  - Fig: How these were concatenated/projected (investigate)

    **Figure name: Observation Inputs.** Draft: [observation-input-path.svg](blog_figures/observation-input-path.svg). It shows exact data flow, not tensor dimensions for one model size.

- Past action encoding

  **Block name: Remember the physical control, not only its class**

  The current input adds a learned class embedding to a learned projection of the physical stick or trigger value. The older alternative represented the complete past controller state as a 256-way one-hot class.

  | Control | Treatment | Stock control → treatment | Δ stock | Damage control → treatment | Δ damage |
  |---|---|---:|---:|---:|---:|
  | 256-way one-hot history | Learned class + physical value | −0.101 → +0.159 | **+0.260** | +13.9 → +37.5 | **+23.7** |

  This is not exact resource parity. The one-hot arm had 1.1% more parameters and saw twice as many examples. It still lost. That mismatch strengthens the directional rejection of one-hot history, although it prevents a precise effect-size claim. The matching earlier checkpoint should be used when match rows are recovered.

  A clean O28 subtest pruned the one-hot button table from 256 classes to 52. Stock changed by +0.037, but damage changed by −7.63. Pruning did not give a clear overall gain.

  **Sources:** one-hot [`1zzstd2d`](https://wandb.ai/ericyuegu/hal/runs/1zzstd2d) · 52-class one-hot [`4wyxy1g2`](https://wandb.ai/ericyuegu/hal/runs/4wyxy1g2) · learned physical encoding [`cqbbbg77`](https://wandb.ai/ericyuegu/hal/runs/cqbbbg77) · [O28 code](../experiments/028_onehot_controller.py)

- Ideas that didn't work

  - One-hot prev controller conditioning

    Use the past-action result above. The important fact is that the losing arm received more capacity and data. Do not describe this as a weak result merely because resources were unequal.

  - Hand feature engineering

    **Block name: Derived geometry never reached a valid test**

    O16 added relative positions, relative velocities, range, and stage geometry. Runs [`6mmt6o4c`](https://wandb.ai/ericyuegu/hal/runs/6mmt6o4c) and [`60esc6s8`](https://wandb.ai/ericyuegu/hal/runs/60esc6s8) stopped without a closed-loop result. This is unfinished, not a negative finding.

    A different near-control did finish: the full observation bundle beat the lean O21 bundle by +0.268 stocks/min and +13.37 damage/min. That test changes the complete bundle, so it does not say which field mattered.

    **Sources:** [O16 code](../experiments/016_spatial_features.py) · full bundle [`fxhtoxu8`](https://wandb.ai/ericyuegu/hal/runs/fxhtoxu8) · lean bundle [`pu8jpmbf`](https://wandb.ai/ericyuegu/hal/runs/pu8jpmbf) · [O21 code](../experiments/021_v6_features.py)

  - RLE BPE

    **Block name: Compress holds into action spans**

    O18 packed the four controller groups into one frame symbol. BPE then found repeated holds and made variable-length tokens, capped at 16 frames. A sampled token was executed to completion before the next plan.

    The tested package lost 2.635 stocks/min and 172.16 damage/min against the frame-aligned reference. It also had 14.2% more parameters and estimated training work, changed execution cadence, and crashed in 57.7% of its old evaluation matches. The complete system failed, but this is not a clean rejection of sequence compression alone.

    **Sources:** BPE/RLE [`xgj1dwot`](https://wandb.ai/ericyuegu/hal/runs/xgj1dwot) · frame-aligned reference [`fxhtoxu8`](https://wandb.ai/ericyuegu/hal/runs/fxhtoxu8) · [O18 code](../experiments/018_bpe_rle.py)

    - Small fig: how RLE chunked actions (investigate)

      **Figure name: Run-Length Actions.** Draft: [rle-action-spans.svg](blog_figures/rle-action-spans.svg). It shows frame actions followed by the variable-length span tokens learned by BPE. O18 did not emit a separate duration token.

  - Chording

    **Block name: One token for the full controller**

    The main stick, C-stick, left trigger, right trigger, and buttons were packed into one mixed-radix controller state. A data-derived vocabulary kept observed states and projected unseen states to a nearby entry. The retained run [`vgwbv53u`](https://wandb.ai/ericyuegu/hal/runs/vgwbv53u) stopped after three diagnostic updates. There is no gameplay result.

    **Sources:** [joint-chord decoder](../experiments/006_chord_ar.py) · [RLE event decoder](../experiments/008_rle_events.py)

### Architecture

- Next-frame prediction → independent multi-token prediction → DSv3-style multi-token prediction

  **Block name: From one action to a causal plan**

  This is a design lineage, not a score ladder.

  1. The next-frame model predicted one controller frame from each trunk position.
  2. Independent multi-token prediction reused one trunk state to predict several future offsets. Future predictions could not read one another.
  3. The current temporal decoder predicts the selected offsets in causal order. Each new offset reads the preceding predicted controller frame and the earlier temporal states.

  The clean measurement is O37, not the historical handoff between model families.

  | H4 cell | Across-frame feedback | Within-frame feedback | Stock/min | Damage/min |
  |---|---|---|---:|---:|
  | D0 | No | No learned feedback | −0.702 | +2.8 |
  | D1 | No | Yes | −0.313 | +13.0 |
  | D2 | Yes | No learned feedback | −0.196 | +18.1 |
  | D3 | Yes | Yes | **−0.096** | **+24.7** |

  Averaged across both within-frame settings, across-frame feedback added **+0.361 stocks/min** and **+13.49 damage/min**. Averaged across both across-frame settings, within-frame feedback added **+0.245 stocks/min** and **+8.48 damage/min**. The four arms have identical 7.148M parameter counts, batch 512, 16,384 updates, 8.39M examples, ranked-anonymized-1 v7 data, and H4 evaluation.

  **Sources:** next frame [`vlim96s9`](https://wandb.ai/ericyuegu/hal/runs/vlim96s9) · independent MTP [`shjnxxsu`](https://wandb.ai/ericyuegu/hal/runs/shjnxxsu) · D0 [`98r9smrj`](https://wandb.ai/ericyuegu/hal/runs/98r9smrj) · D1 [`a117chkw`](https://wandb.ai/ericyuegu/hal/runs/a117chkw) · D2 [`50q39o9j`](https://wandb.ai/ericyuegu/hal/runs/50q39o9j) · D3 [`5wfk2esf`](https://wandb.ai/ericyuegu/hal/runs/5wfk2esf) · [O37 report](experiments/037_factorization_matrix.md)

  - Fig: Within-frame AR vs. across-frame AR; bar chart comparing perf (refine)

    **Figure name: Two Axes of Action Feedback.** Draft: [action-factorization.svg](blog_figures/action-factorization.svg). The artifact's `ar-axes` diagram is a useful conceptual companion: [artifact](policy_anatomy_artifact_export.html#fig-ar).

  - within-frame controller ordering

    The current order is C-stick → main stick → triggers → buttons. O37 tests learned feedback versus fixed-zero feedback in that order. It does not compare alternate orders. No clean ordering experiment was found.

- SwiGLU vs. linear head

  **Block name: Nonlinear action readouts matter**

  The exact O43 treatment is not SwiGLU. The control head is RMSNorm → linear → SiLU → linear. The treatment is RMSNorm → bias-free linear. At 8,192 updates, the normalized linear heads changed the character suite by **−2.03 ± 0.28 stocks/min** and the Fox suite by **−2.41 ± 0.32 stocks/min**. These are direct 95% intervals for treatment minus control.

  An older O23 near-control found only +0.11 stocks/min for a larger state MLP over a linear head, with 12.6% more parameters and work. The later O43 test is cleaner and much more decisive.

  **Sources:** O43 control [`1imfy8v3`](https://wandb.ai/ericyuegu/hal/runs/1imfy8v3) · linear-head treatment [`apj3rw8b`](https://wandb.ai/ericyuegu/hal/runs/apj3rw8b) · historical linear [`obx3o3az`](https://wandb.ai/ericyuegu/hal/runs/obx3o3az) · historical MLP [`q3aojgfm`](https://wandb.ai/ericyuegu/hal/runs/q3aojgfm) · [O43 head code](../experiments/043_legacy_codec.py)

- FiLM conditioning, nonlinear heads, trunk skip

  **Block name: Three paths into the action logits**

  O43 tested each component separately against the same control. Arm A uses ordinary random FiLM initialization, nonlinear SiLU action heads, and a direct trunk-to-logit skip.

  | Only change from A | Update | Character suite Δ stock/min | Fox suite Δ stock/min | Result |
  |---|---:|---:|---:|---|
  | Zero-initialize FiLM projections | 4,096 | **−1.09 ± 0.34** | **−1.20 ± 0.37** | Worse at this early checkpoint |
  | Remove trunk-to-logit skip | 4,096 | **−0.89 ± 0.40** | **−1.13 ± 0.46** | Worse |
  | Replace nonlinear heads with normalized linear heads | 8,192 | **−2.03 ± 0.28** | **−2.41 ± 0.32** | Much worse |

  The zero-initialization result is time-specific. It can mean slower learning rather than a worse converged solution. A 16,384-update evaluation would separate those claims.

  O37 gives evidence for within-frame FiLM conditioning versus no learned controller-part feedback. It does **not** compare FiLM with plain concatenation. O43 tests FiLM initialization, not FiLM versus concatenation. That ablation remains missing.

  This also resolves the apparent O26/O42 conflict. O42 did not add these three O43 components to O26. It used normalized linear heads, removed the trunk-logit skip, and zero-initialized bounded FiLM. It also changed trunk and temporal SwiGLU blocks, normalization, residual scaling, and the sampler. The equal-step O26 → O42 package changed score by −0.217 stocks/min and −10.81 damage/min. That package result is consistent with the later one-change tests, but it cannot assign the loss to one component.

  **Sources:** control [`1imfy8v3`](https://wandb.ai/ericyuegu/hal/runs/1imfy8v3) · zero-init FiLM [`vzd242t6`](https://wandb.ai/ericyuegu/hal/runs/vzd242t6) · no skip [`j4skyy83`](https://wandb.ai/ericyuegu/hal/runs/j4skyy83) · linear heads [`apj3rw8b`](https://wandb.ai/ericyuegu/hal/runs/apj3rw8b) · O26 [`cqbbbg77`](https://wandb.ai/ericyuegu/hal/runs/cqbbbg77) · O42 [`k8nb00xv`](https://wandb.ai/ericyuegu/hal/runs/k8nb00xv) · [O42 code](../experiments/042_architecture_data_ablation.py) · [O43 code](../experiments/043_legacy_codec.py)

  - Fig: Show all 3 (investigate)

    **Figure name: Action Logit Paths.** Draft: [action-logit-paths.svg](blog_figures/action-logit-paths.svg). It combines a small path diagram with the direct-delta forest plot.

- Ideas that didn't work

  - Game state expert (flow matching)

    **Block name: Predicting future state did not improve action choice**

    O29 added a training-only future-state expert to the O26 actor. It changed score by −0.048 stocks/min and −12.60 damage/min. The treatment had 8.3% more parameters, twice the examples, and more than twice the estimated work. This unfavorable outcome under favorable resources is strong evidence against the tested package, although it is not an exact effect-size estimate.

    **Sources:** control [`cqbbbg77`](https://wandb.ai/ericyuegu/hal/runs/cqbbbg77) · future-state expert [`qxcjjwl8`](https://wandb.ai/ericyuegu/hal/runs/qxcjjwl8) · [O29 code](../experiments/029_game_state_flow.py)

  - Temporal output head cross-attending to earlier trunk context tokens

    **Block name: Let the output head reread history**

    O24 used 20 action queries that cross-attended to prior trunk tokens. It scored −0.398 stocks/min and +21.55 damage/min. Its nearest reference also changes the trunk, codec, offsets, and decoder size. Keep the absolute result as history, but do not publish a treatment delta.

    **Sources:** O24 [`4hoe86s7`](https://wandb.ai/ericyuegu/hal/runs/4hoe86s7) · [O24 code](../experiments/024_temporal_mtp.py)

  - Flattening all controller groups across frames into long sequence—much more compute intensive to re-use whole trunk

    **Block name: Put actions inside the main sequence**

    A state frame became five main-trunk tokens: state, C-stick, main stick, triggers, buttons. A 256-frame context therefore became 1,280 tokens and used about five times the trunk work.

    The 51-frame arm later scored −0.101 stocks/min and +28.82 damage/min in a 96-boot evaluation. The 256-frame arm scored +0.262 and +41.34 under an older protocol. The arms use different context and evaluation protocols, and neither has a clean equal-work control. Treat them as historical systems, not evidence that flattening helped.

    **Sources:** 51-frame [`7tlk11io`](https://wandb.ai/ericyuegu/hal/runs/7tlk11io) · 256-frame [`lxo3id9l`](https://wandb.ai/ericyuegu/hal/runs/lxo3id9l) · [O13 code](../experiments/013_multi_token.py)

  - Fixed delay

    **Block name: Predicting twelve frames ahead**

    O43 at delay 1 scored about +0.24 stocks/min. O44 at delay 12 scored about −0.86, a change of **−1.105 stocks/min** and **−85.71 damage/min**. O44 also needed a longer future-data buffer, so this is a near-control rather than a strict delay-only test. The tested long-delay setup clearly failed.

    **Sources:** delay 1 [`1imfy8v3`](https://wandb.ai/ericyuegu/hal/runs/1imfy8v3) · delay 12 [`2qs05au4`](https://wandb.ai/ericyuegu/hal/runs/2qs05au4) · [O44 report](experiments/044_fixed_prediction_delay.md)

  - Rank-weighted AWR

    **Terminology correction:** O34 is rank-weighted behavior cloning, not AWR. It gives every demonstrated prefix a replay-rank weight of 1, 2, or 4. Against exact O26 uniform behavior cloning, it changed stock score by **−0.122 stocks/min**. The damage mean was not preserved well enough for a direct delta. This strong one-factor test did not support replay-rank weighting.

    **Sources:** uniform [`cqbbbg77`](https://wandb.ai/ericyuegu/hal/runs/cqbbbg77) · rank-weighted [`c5cxoyk8`](https://wandb.ai/ericyuegu/hal/runs/c5cxoyk8) · [O34 report](experiments/034_rank_weighted_bc.md)

### Learning objective

- AWR

  **Block name: Weight actions by estimated advantage**

  A detached value head estimates the return after each state. The policy weight is a capped exponential function of the demonstrated action's return minus the value estimate. The value input is detached, so the value loss cannot reshape the policy trunk directly.

  In the modern near-control, behavior cloning scored −0.096 stocks/min and +24.7 damage/min. AWR scored −0.020 and +35.5. The point change was **+0.076 stocks/min** and **+10.80 damage/min**. The two evaluations used 96 and 128 boots and were not paired, so a direct interval is not available.

  **Sources:** BC [`5wfk2esf`](https://wandb.ai/ericyuegu/hal/runs/5wfk2esf) · AWR [`hwzv0k9a`](https://wandb.ai/ericyuegu/hal/runs/hwzv0k9a) · [O36 report](experiments/036_advantage_weighted_bc.md)

  - Sensitive to gamma—need to make sure half-life decay of returns reaches their causal source

    **Block name: Reward must travel far enough backward**

    | Gamma | Return half-life | Stock/min | Δ stock vs 0.99618 | Damage/min | Δ damage vs 0.99618 |
    |---:|---:|---:|---:|---:|---:|
    | 0.99618 | 181 frames / 3.0 s | +0.032 | — | +34.0 | — |
    | 0.99855 | 478 frames / 8.0 s | **+0.398** | **+0.366** | **+68.0** | **+34.0** |
    | 0.99920 | 866 frames / 14.4 s | +0.350 | +0.318 | +64.8 | +30.9 |

    This is a strong O52-family sweep: 14.48M parameters, batch 512, 16,384 updates, (2^{30}) positions, ranked-anonymized-1 v8, D2/R2, and 96 boots. The middle gamma was best. A plausible mechanism is that the 3-second return half-life discounts reward before it reaches the action that caused it.

    **Sources:** gamma 0.99618 [`yxy8brhx`](https://wandb.ai/ericyuegu/hal/runs/yxy8brhx) · gamma 0.99855 [`0ksityvv`](https://wandb.ai/ericyuegu/hal/runs/0ksityvv) · gamma 0.99920 [`8t9koblm`](https://wandb.ai/ericyuegu/hal/runs/8t9koblm) · [O52 code](../experiments/052_adamw_temporal_awr.py)

  - Fig: interactive AWR explorer

    **Figure name: AWR Weight Explorer.** Continue from the existing [notebook](../notebooks/awr_explorer.py). The controls should expose return, value, beta, cap, gamma, reward delay, and the resulting normalized weight. Keep the return half-life visible in both frames and seconds.

- Ideas that didn't work as well

  - Flow matching

    **Block name: Refine a complete sparse plan from noise**

    O38 predicts all ten categorical endpoints in parallel and refines them with four flow steps. Against O37 D3, score changed by **−0.186 stocks/min** and **−32.93 damage/min**. The trunk, data, updates, examples, H4 timing, and 96-boot protocol match. Flow has 25.2% more parameters and estimated training work and 7.3% more estimated inference work. The resource asymmetry strengthens the rejection of this tested flow package.

    This is different from O29. O29 predicts future game state as a training-only auxiliary task. O38 uses flow as the deployed action decoder.

    An earlier prototype, O1, did generate continuous main-stick, C-stick, and trigger trajectories. It used a 16-frame continuous action chunk and eight flow steps. It scored −2.262 stocks/min and −112.13 damage/min, but it has no matched control: architecture, objective, action format, update count, and execution all differ. Keep it as historical evidence that continuous action flow was tried, not as an effect estimate.

    **Sources:** categorical [`5wfk2esf`](https://wandb.ai/ericyuegu/hal/runs/5wfk2esf) · endpoint flow [`t4s3gohe`](https://wandb.ai/ericyuegu/hal/runs/t4s3gohe) · continuous flow [`xummemli`](https://wandb.ai/ericyuegu/hal/runs/xummemli) · [O38 report](experiments/038_sparse_endpoint_flow.md) · [O1 code](../experiments/001_flow_matching_baseline.py)

  - IQL

    **Block name: Attach critics to the behavior-cloning policy**

    The tested IQL system changed score by **−2.009 stocks/min** and **−95.85 damage/min** against O26 behavior cloning. It had 1.43% more total parameters, twice the examples, and about twice the estimated training work. That is strong evidence that this implementation failed, but it does not reject every offline Q-learning method.

    **Sources:** BC [`cqbbbg77`](https://wandb.ai/ericyuegu/hal/runs/cqbbbg77) · IQL [`06fi5jms`](https://wandb.ai/ericyuegu/hal/runs/06fi5jms) · [O27 code](../experiments/027_temporal_mtp.py)

  - Explorative modeling (XM) k=1, k=8

    **Terminology correction:** use **Exploration Modeling (XM)**.

    | Arm | Stock/min | Δ stock vs O26 BC | Damage/min | Δ damage vs O26 BC |
    |---|---:|---:|---:|---:|
    | O26 BC | +0.159 | — | +37.5 | — |
    | XM K=1 | −1.508 | **−1.667** | −51.8 | **−89.4** |
    | XM K=8 | −0.648 | **−0.807** | −6.8 | **−44.3** |

    K=8 improved over K=1 by +0.860 stocks/min and +45.04 damage/min, but K=8 used much more trajectory work per example, stopped at 10,240 rather than 16,384 updates, and changed microbatching. The K effect is not isolated. Neither tested XM arm beat O26 BC.

    **Sources:** BC [`cqbbbg77`](https://wandb.ai/ericyuegu/hal/runs/cqbbbg77) · K=1 [`mz8dk2gh`](https://wandb.ai/ericyuegu/hal/runs/mz8dk2gh) · K=8 [`feic18gt`](https://wandb.ai/ericyuegu/hal/runs/feic18gt) · [O33 code](../experiments/033_parallel_latent_trajectory.py)

### Data

- 1 window per replay vs. 4 windows per replay, sufficient shuffle block size

  **Block name: Spend each batch on different games**

  | Control | Treatment | Stock control → treatment | Δ stock | Damage control → treatment | Δ damage |
  |---|---|---:|---:|---:|---:|
  | Four windows per replay | One window per replay | −0.598 → +0.130 | **+0.728** | +40.5 → +34.3 | **−6.23** |

  This historical O12 control holds the model, parameter count, batch 1,024, 16,384 updates, and 16.78M examples fixed. Only the recorded windows-per-replay setting changes. The stock gain is large; damage moves in the opposite direction. Repeat it in the current physical-shard ring before treating it as a universal result.

  A sufficient shuffle block matters for the same reason. Nearby windows from one replay share stage, characters, positions, and local action patterns. They reduce effective replay diversity even when nominal sample count is unchanged.

  **Sources:** four windows [`uw05bvm2`](https://wandb.ai/ericyuegu/hal/runs/uw05bvm2) · one window [`shjnxxsu`](https://wandb.ai/ericyuegu/hal/runs/shjnxxsu) · [O12 code](../experiments/012_multi_token.py)

  - Fig: replay ring buffer diagram

    **Figure name: Replay Diversity Ring.** Reuse and refine Fig. 8 and Fig. 9 in [policy_anatomy.html](policy_anatomy.html). Show physical shards outside the ring, replay slots inside it, several windows decoded per resident replay, and a shuffle queue that enforces a minimum batch gap before the same replay can return.

- Data cleaning, effect of removing 0.5% bad data b/w dataset version 7 vs. 8

  **Block name: A small data-cleaning cut**

  Correct the headline count: policy-world v8 removed **0.405% of training replays** and **0.198% of training frames**, not 0.5% of frames.

  | Corpus total | v7 | v8 | Removed | Removed share |
  |---|---:|---:|---:|---:|
  | Training replays | 1,300,640 | 1,295,370 | 5,270 | 0.405% |
  | Training frames | 13,292,693,140 | 13,266,364,175 | 26,328,965 | 0.198% |

  V8 requires both players to have dealt more than 50 damage and both to have started with four stocks. Ranked-anonymized-1 lost 221 training replays, from 112,409 to 112,188. Across all splits, 214 failed the damage rule and 13 failed the starting-stock rule; some failed both.

  There is no clean v7-versus-v8 gameplay ablation. This is a corpus audit, not evidence that cleaning improved play.

  **Sources:** [v8 publication record](../data/builds/policy-world-v8/publication.json) · [v8 builder](../hal/scripts/rematerialize_policy_world_v8.py) · [frozen stream counts](../hal/streams.py)

- Fig: Effect of more data, scaling laws

  **Figure name: Data Scaling Curves.** Wait for the current O54 rerun. Use equal model size and equal deployment timing within each curve. Show validation NLL and both gameplay measures. Do not splice old protocols into one fitted law.

  A small current result can sit beside, not inside, the scaling curve: in O42, the all-44-source mix changed equal-step score by −0.058 stocks/min and +6.31 damage/min versus ranked-anonymized-1. The full mix improved damage, not stock. **Sources:** ranked only [`k8nb00xv`](https://wandb.ai/ericyuegu/hal/runs/k8nb00xv) · full mix [`bevhq159`](https://wandb.ai/ericyuegu/hal/runs/bevhq159) · [O42 code](../experiments/042_architecture_data_ablation.py)

### Inference

- Inference-time RTC (force-feeding queued actions to decoder)

  **Block name: Replan while actions are already in flight**

  RTC feeds the actions that are already committed in the transport queue through the decoder, then samples only the uncommitted suffix. This should make a new plan consistent with actions the game will receive before the new plan can take effect.

  The retained treatment run [`c6yd7zxx`](https://wandb.ai/ericyuegu/hal/runs/c6yd7zxx) stopped at 4,096 updates. It scored −1.167 stocks/min and −22.3 damage/min in the character suite, and −0.622 and −23.2 in the Fox suite. The planned dense autoregressive control was not run. Therefore RTC is **unresolved**, not a failed idea.

  The minimum decisive test is two O46 runs with the same initialization, dense offsets 1–20, data order, updates, and evaluation match rows. The control trains with committed-prefix length 0 only. The treatment samples committed-prefix lengths 0–4. Evaluate both at each deployment delay with identical queued actions.

  **Sources:** [O46 code](../experiments/046_ar_rtc.py) · [O46 design](experiments/046_ar_rtc.md)

  - Introduce terminology here to distinguish replan interval vs. transport delay vs. action chunk length/horizon vs. inference latency

    Use these terms consistently:

    | Term | Symbol | Meaning |
    |---|---|---|
    | Action chunk length | H | Number of future action frames returned by one plan |
    | Transport delay | d | Number of game frames between the latest observation and the first newly usable action |
    | Replan interval | R | Number of game frames between planning calls |
    | Execution length | E | Number of frames from one plan that are actually used before replacement |
    | Inference latency | — | Wall-clock time to produce a plan; report p50 and p95 in milliseconds |

    In older O37 runs, one “execution horizon” setting tied E and R. In RTC, H, d, and R are separate. Never call all of them “horizon.”

  - Fig: plan-timing

    **Figure name: Plan Timing.** Reuse Claude's `plan-timing` interactive: [artifact](policy_anatomy_artifact_export.html#fig-timing). Keep `stage-map` as the headline overview: [artifact](policy_anatomy_artifact_export.html#diagram).

- Execution horizon

  **Block name: Action feedback protects longer plans**

  O37 evaluated the same four checkpoints while executing and replanning every 1, 2, 4, or 6 frames.

  | Model | H1 stock | H2 | H4 | H6 | H1 → H6 | H1 damage | H6 damage |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | D0: no action feedback | +0.096 | −0.159 | −0.702 | −0.905 | **−1.001** | +35.5 | −19.0 |
  | D1: within-frame only | +0.133 | +0.048 | −0.313 | −0.467 | **−0.600** | +40.8 | −1.4 |
  | D2: across-frame only | −0.186 | −0.064 | −0.196 | −0.234 | −0.048 | +29.5 | +15.7 |
  | D3: both | +0.053 | +0.000 | −0.096 | −0.021 | −0.074 | +36.3 | +25.9 |

  Averaged across models, H1 → H6 changed score by −0.431 stocks/min and −30.20 damage/min. That average hides the main result: the models with across-frame action feedback were much less sensitive to longer execution.

  This is strong same-checkpoint evidence, but it is not a pure action-chunk or replan test. Execution length and replan interval changed together.

  **Sources:** D0 [`98r9smrj`](https://wandb.ai/ericyuegu/hal/runs/98r9smrj) · D1 [`a117chkw`](https://wandb.ai/ericyuegu/hal/runs/a117chkw) · D2 [`50q39o9j`](https://wandb.ai/ericyuegu/hal/runs/50q39o9j) · D3 [`5wfk2esf`](https://wandb.ai/ericyuegu/hal/runs/5wfk2esf) · [full O37 table](experiments/037_factorization_matrix.md)

- Talk about low-dim data with training sample trajectories being considerably shorter than full episodes, causing disk read bottleneck rather than CPU or memory when training on large GPU

  **Block name: Training is a storage-locality problem**

  A training sample is a short sequence cut from a much longer replay. Loading one short sample per replay wastes disk reads and decompression setup. Loading many nearby windows from one replay is efficient, but putting them in the same batch reduces replay diversity. The physical-shard ring separates those concerns: materialize a deterministic set of replay shards, decode several windows while each replay is resident, put those windows in a large in-memory shuffle pool, and delay reuse of the same replay for a minimum number of batches.

  This is why “windows decoded per resident replay” and “windows from one replay admitted to one batch” need different names. The first is a systems setting. The second changes the training distribution.

  - Workaround: read more windows per episode into in-memory buffer, then shuffle + sample uniformly from waves of slots to ensure minimum N batches between re-sampling same window

    O52's fixed ring has 112,128 replay slots, decodes eight windows per generation, uses 25-batch replay phase blocks, and enforces a 195-batch minimum replay gap. O50's full-data ring has 131,072 replay slots. Keep these values in the systems figure, not in the one-window causal claim.

    **Sources:** [physical shard loader](../hal/training/physical_shard_loader.py) · [O52 config](../experiments/052_adamw_temporal_awr.py) · [O50 config](../experiments/050_scaled_temporal_awr.py)

  - Show 4-replay window vs. 1-replay window result

    Use the O12 result above: four windows per replay → one window per replay gave +0.728 stocks/min and −6.23 damage/min. Call it **four windows per replay**, not “four-replay window.”

## Optimal model size vs. latency (TBD, don't worry for now, re-running)

- Fig: horizon-panels

  **Figure name: Capacity by Control Timing.** Reuse the `capacity-explorer` and existing horizon panels only as a layout prototype. Replace their points with O54 when that experiment is frozen: [artifact](policy_anatomy_artifact_export.html#cap).

- Fig: iso-flop compute vs. NLL curves

  **Figure name: Iso-Compute Learning Curves.** Source the final data from [O48](experiments/048_iso_flop_scaling.md) and the new O54 run. Keep one optimizer and data protocol per fitted family.

- Fig: scatterplot b/w val NLL and closed-loop performance

  **Figure name: Likelihood Is Not Gameplay.** Plot checkpoint-level validation NLL against stock and damage for the same run and protocol. Connect checkpoints in time. The O50 Muon trajectory is the strongest example: validation improved steadily while play fell through the middle of training and later recovered.

  **Sources:** Muon production [`p1fyyp1z`](https://wandb.ai/ericyuegu/hal/runs/p1fyyp1z) · all-AdamW production [`tgpuo1be`](https://wandb.ai/ericyuegu/hal/runs/tgpuo1be) · [O50 code](../experiments/050_scaled_temporal_awr.py)

### Any other important experiments I'm forgetting

- **Next-frame loss share.** With the legacy codec and O43 architecture fixed, changing from the O26 loss to 50% of action loss on the next frame changed score from −0.133 to +0.005 stocks/min and +21.4 to +24.9 damage/min: **+0.138 stocks/min, +3.5 damage/min**. Sources: [`acbbeb76`](https://wandb.ai/ericyuegu/hal/runs/acbbeb76) → [`1imfy8v3`](https://wandb.ai/ericyuegu/hal/runs/1imfy8v3).

- **Ego-player identity.** In a 16M, 16,384-update, Fox-only, 96-boot test, the masked baseline scored +0.547 stocks/min and +55.85 damage/min. Cody conditioning scored +0.629 and +87.01: **+0.081 stocks/min, +31.16 damage/min**. Master, Mang0, and Platinum changed stock by −0.021, −0.068, and −0.319, so identity is a behavior control, not a general quality switch. Sources: masked [`u1gvby0r`](https://wandb.ai/ericyuegu/hal/runs/u1gvby0r) · conditioned [`dg0u00w5`](https://wandb.ai/ericyuegu/hal/runs/dg0u00w5) · [O49 code](../experiments/049_player_identity_conditioning.py).

- **All AdamW versus Muon plus AdamW.** At 216.5M parameters and 131,072 updates on the full v8 corpus, Muon plus AdamW scored +0.345 stocks/min and +37.59 damage/min. All AdamW scored +0.494 and +52.72: **+0.149 stocks/min, +15.13 damage/min**. This is a strong optimizer-recipe comparison. It does not isolate optimizer mathematics because rates and parameter routing also change. Sources: [`p1fyyp1z`](https://wandb.ai/ericyuegu/hal/runs/p1fyyp1z) · [`tgpuo1be`](https://wandb.ai/ericyuegu/hal/runs/tgpuo1be).

- **Batch size at fixed examples.** Batches 128, 256, 512, 1,024, and 2,048 scored −0.32, −0.49, −0.72, −0.56, and −0.79 stocks/min after each saw 2.097M examples. Batch 128 received 16 times as many optimizer updates as batch 2,048 and improved by **+0.47 stocks/min and +27.92 damage/min**. Call this an update-frequency result, not batch size in isolation. Endpoint sources: [`nt4gh4mk`](https://wandb.ai/ericyuegu/hal/runs/nt4gh4mk) · [`rri7scjc`](https://wandb.ai/ericyuegu/hal/runs/rri7scjc).

- **Learning-rate pair.** In the six-point O26 sweep, the best Muon/AdamW rate pair was 0.040/0.00170 at −0.37 stocks/min. The worst was 0.014/0.000595 at −0.59. The spread was **+0.22 stocks/min**. Both rates moved together. Sources: [`7hy8xjjh`](https://wandb.ai/ericyuegu/hal/runs/7hy8xjjh) · [`9luwu85n`](https://wandb.ai/ericyuegu/hal/runs/9luwu85n).

- **Checkpoint choice.** The 216.5M O50 Muon run moved from −0.218 stocks/min at 8,192 updates to +0.164 at 24,576, fell to −0.175 at 57,344, and ended at +0.345 at 131,072. Offline likelihood improved throughout. Closed-loop evaluation must select checkpoints; validation likelihood cannot do it alone.

- **Still missing and worth running.** FiLM versus plain concatenation; alternate within-frame controller orders; follower-field ablation; a matched RTC control; separate action-chunk length from replan interval; repeat one versus four windows per replay in the current ring; and a 16,384-update zero-init FiLM evaluation. These are the remaining clean holes, not hidden negative findings.

## Figure inventory

| Figure name | Status | Main evidence | Asset |
|---|---|---|---|
| Policy Stages | Reuse | Headline model and runtime stages | [`stage-map`](policy_anatomy_artifact_export.html#diagram) |
| Codec Bridges | Draft made | Two matched codec tests | [SVG](blog_figures/controller-codec-bridges.svg) |
| Observation Inputs | Draft made | Exact feature projection; projectile ablation | [SVG](blog_figures/observation-input-path.svg) |
| Action History Path | Reuse/refine | One-hot versus learned physical history | [`io-paths`](policy_anatomy_artifact_export.html#fig-codec) |
| Run-Length Actions | Draft made | O18 representation | [SVG](blog_figures/rle-action-spans.svg) |
| Two Axes of Action Feedback | Draft made | O37 2×2 | [SVG](blog_figures/action-factorization.svg) |
| Action Logit Paths | Draft made | O43 FiLM/head/skip tests | [SVG](blog_figures/action-logit-paths.svg) |
| AWR Weight Explorer | Existing work | AWR weights and gamma | [notebook](../notebooks/awr_explorer.py) |
| Replay Diversity Ring | Reuse/refine | O12 window test and current loader | [policy anatomy](policy_anatomy.html) |
| Plan Timing | Reuse | H, d, R, E, and latency | [`plan-timing`](policy_anatomy_artifact_export.html#fig-timing) |
| Capacity by Control Timing | Wait | O54 | [`capacity-explorer` prototype](policy_anatomy_artifact_export.html#cap) |
| Iso-Compute Learning Curves | Wait | O48/O54 | [O48 plan](experiments/048_iso_flop_scaling.md) |
| Likelihood Is Not Gameplay | Ready for data binding | O50 checkpoints | [raw data](blog_experiment_data.json) |
