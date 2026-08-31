# Experiment 045: BYOL player-style representation

Status: protocol frozen; implementation verified on synthetic data. No training result is recorded yet.

## Question

Can a replay encoder identify a player's style across characters, stages, opponents, and game states, including for players that were absent from representation training?

The public artifact is the encoder's 128-dimensional pre-projector representation. The projector and predictor are not downstream features.

## Data protocol

Anonymous data comes from `ranked-anonymized-1/mds-policy-v7`. One replay and one ego side produce both 256-frame views. The anchor start is uniform over the complete replay. The other start is at least

`min(1024, floor((T - 256) / 3))`

frames away and is sampled with probability proportional to squared start distance. Online and target roles are exchanged with probability 0.5. Native MDS `py1e` shuffling supplies replay order. Anonymous examples only appear in their own BYOL pair.

Professional data comes from the 38 player-slug policy-world streams. A row is eligible only when exactly one side has `Rank.PRO`; that side is ego and the stream slug is its identity.

These ten identities are sealed until final evaluation:

`cookbook, solobattle, rapm, siddward, gosu, iliketurtles, friend, nicki, trif, mof`

For each of the other 28 identities, BLAKE2b with personalization `hal-o45-split` hashes `(identity, replay_id)` into 100 buckets:

- 0–79: representation training
- 80–89: held-out gallery and linear-probe training
- 90–99: held-out query

One stratified MDS input batch contains one replay candidate from each development identity. Bounded queues retain at most 32 eligible, distinct replay windows per identity. One update samples 16 identities uniformly and takes 16 windows from each. A maximum-weight assignment creates a cross-replay derangement. Its weights are 8 for a different ego character, 4 for a different stage, 2 for a different opponent character, and 1 for game-state descriptor distance.

The game-state descriptor contains eight 32-frame means of ego/opponent stage-normalized x, y divided by 100, percent divided by 100, stock count, relative stage-normalized x, and relative y divided by 100. It has 80 values. It contains no input, action ID, learned feature, or identity.

Each logical update joins twelve 64-replay anonymous loader batches with 256 professional pairs. It therefore runs 1,024 online examples and 1,024 EMA-target examples. Both MDS pipelines use eight persistent ordered workers, pinned memory, and a prefetch factor of two. MDS owns shuffling, source mixing, shard retrieval, decompression, worker partitioning, and prefetch depth. The local cache has no eviction limit and removes compressed shards after decompression. Device microbatches are 64 online and 128 target.

## Model and objective

The encoder uses the shared causal Transformer implementation with a 256-frame context, width 256, four layers, four heads, and full causal attention. CUDA uses native variable-length FlashAttention; CPU contract tests use dense SDPA. Each frame contains ego, opponent, both Nana state blocks, and raw ego controller inputs. Broadcast stage and character fields do not enter the encoder.

All trunk states are mean-pooled. A learned 256-to-128 layer followed by RMS normalization produces `z`. Retrieval and distance metrics L2-normalize `z`.

The projector and predictor are both 128-to-512-to-128 MLPs with GELU and LayerNorm. There is no BatchNorm.

The online branch is `q(g(f(xA)))`. The target branch is the stopped-gradient `g_ema(f_ema(xB))`. The loss is only the one-way term:

`2 - 2 cosine(online_prediction, target_projection)`

The EMA encoder and projector start as exact copies of their online modules. Their cosine momentum schedule includes 0.996 at update 0 and 0.9995 at update 8,191.

Professional anchors also use a cosine-margin triplet loss on normalized pre-projector values. A negative must be a different known identity with the same ego character. Candidate preference is:

1. Same stage and opponent character.
2. Same stage.
3. Any same-character candidate.

Within the first non-empty group, the sampler keeps the 16 closest game-state descriptors and chooses the most embedding-similar target. An anchor without a safe candidate has no triplet term. Anonymous examples are never candidates.

The total objective is mean BYOL loss plus 0.25 times mean valid triplet loss, with margin 0.2.

AdamW uses learning rate `3e-4`, betas `(0.9, 0.95)`, weight decay `0.05`, gradient clipping at `1.0`, 512 warmup updates, and cosine decay to `3e-5`. The first complete run is 8,192 updates with BF16 activations and TF32 matrix multiplication.

## Checkpoints

Each schema-2 checkpoint contains the online encoder, projector, predictor, EMA encoder and projector, optimizer, explicit learning-rate and EMA schedule positions, all Python/NumPy/Torch/CUDA RNG states, both MDS cursors, professional queues, experiment RNG state, split definition, configuration, feature statistics, completed step, and W&B ID. The failed schema-1 step-0 checkpoint is intentionally incompatible.

Only `BYOL.export_encoder()` is a downstream model. It returns a separate encoder with no projector or predictor.

## Evaluation

Checkpoint selection uses only the 28 development identities. A fixed cache contains 16 gallery and 16 query replays per identity. Gallery replays train the fixed `C=1` multinomial probe and supply retrieval neighbors. Query replays remain separate. Report cross-replay Recall@1, Recall@5, MRR, mAP, cosine kNN at 1/5/15, the linear probe, one/four/sixteen-shot prototypes, and same/different-player distance distributions. Run these metrics before training and after updates 512, 1,024, and 8,192.

For nuisance control, each query is paired with a same-player, different-replay positive, preferentially crossing ego character, opponent character, stage, and then game state. Its different-player comparison has the same ego character and nearest available game-state descriptor. Report the mean distance gap, pairwise ROC AUC, triplet accuracy, and coverage and separate results for character crossing, stage crossing, opponent crossing, and high state distance. The primary selection metric is the distance gap.

The ten sealed identities are evaluated only once after configuration selection. Their replays receive a deterministic support/query split. Report one/four/sixteen-shot prototypes and retrieval among the ten identities.

The same evaluation runs on a random encoder, the metadata/game-state descriptor, and anonymous-only BYOL with the same model and number of anonymous pairs.

Success requires all of the following:

- Bootstrap 95% lower bound of the nuisance-controlled distance gap is above zero.
- Bootstrap 95% lower bound of nuisance triplet accuracy is above 0.5.
- Seen- and sealed-player identification beat random and metadata/state baselines.
- Effective covariance rank of normalized `z` is at least 32.
- Mean coordinate standard deviation is above 0.02.

BYOL loss is not a success metric.

## Run sequence

Run seed 0 first. Inspect held-out nuisance metrics and collapse diagnostics at updates 0, 512, and 1,024. Stop if `z` collapses or held-out distances fail to improve. Otherwise complete 8,192 updates. If the frozen configuration succeeds, repeat seeds 1 and 2.

Symmetric BYOL, 512-frame windows, and other professional-loss weights are later ablations. They are not part of O45.

## Results

| Run | Seed | Updates | Distance gap | Triplet accuracy | Seen ID | Sealed ID | Effective rank | Mean std | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Not run | 0 | 0 | — | — | — | — | — | — | Pending |

Implementation checks cover toy-MDS source tagging and stratification, loader geometry, stable replay-local randomness, exact pair assembly, uniform identity selection, bounded queues, state-exact continuation, stable splits, sealed exclusion, professional derangements, safe negative selection, one-way stop-gradient, EMA endpoints and exact updates, encoder export, evaluation metrics, collapse diagnostics, checkpoint contents, a CPU update, and a CUDA smoke update.
