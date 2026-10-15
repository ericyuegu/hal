# O50 attention/history pilot

## Result

This checkpoint uses history, but its endpoint teacher-forced predictions were insensitive to history older than about 64 frames on this cohort. That is not what its attention span alone suggests.

With the current frame fixed, I masked older context keys by changing `ctx_pad`. The treatment did not insert fake states. The table reports the paired change from the full 256-frame context in NLL, averaged over four future frames and the four controller groups. Intervals are a fixed-seed replay bootstrap over 32 replays.

| Frames retained | Mean NLL change (nats) | 95% bootstrap interval | Replays with increase |
|---:|---:|---:|---:|
| 1 | +0.1781 | [0.1189, 0.2412] | 28/32 |
| 2 | +0.1408 | [0.0895, 0.1974] | 28/32 |
| 4 | +0.1163 | [0.0659, 0.1732] | 28/32 |
| 8 | +0.0725 | [0.0353, 0.1148] | 25/32 |
| 16 | +0.0450 | [0.0140, 0.0828] | 23/32 |
| 32 | +0.0247 | [0.0053, 0.0480] | 21/32 |
| 64 | +0.0005 | [-0.0114, 0.0136] | 17/32 |
| 128 | -0.0001 | [-0.0068, 0.0061] | 18/32 |

At 32 retained frames, the group NLL changes were +0.0316 buttons, +0.0533 main stick, -0.0070 C-stick, and +0.0211 triggers. Teacher-forced top-1 predictions changed on 2.3% of button and main-stick targets and none of the C-stick or trigger targets. At 64 frames, only 0.8% of button top-1 targets changed.

The full-context baseline NLLs were 0.168 buttons, 0.734 main stick, 0.076 C-stick, and 0.183 triggers. Each is natural-log nats per controller group per predicted frame, averaged over four frames. These are not the logged joint-bits metric and not a closed-loop score.

## Attention does not localize this dependence

Across the 64 trunk heads, the last query assigned 2.2% mean mass to the current frame, 14.2% to the last four frames, 46.1% to the last 16, and 28.9% to lags of at least 64 frames. Mean attended lag was 52.4 frames; mean entropy corresponded to 44.8 equally weighted positions. Individual heads ranged from almost one-frame-local to a mean lag of 139 frames.

This separate d256 pilot used the original autocast trace, where CUDA could evaluate the QK score and softmax in BF16 despite explicit FP32 inputs. Its history and head-ablation outputs are unaffected, but its attention summaries have legacy approximate precision. The corrected historical d1024 results below use FP32 scores and softmax.

I then zeroed each head's slice immediately before its layer's attention output projection. This is a causal activation ablation, although it is still off distribution. No single head explained the history effect. The largest mean NLL increase was +0.0113 nats for zero-based layer 1, head 3, with an interval of [-0.0083, 0.0324]. Only zero-based layer 4, head 2 had a nominally positive interval: +0.0031 [0.0005, 0.0060]. This does not survive a claim across 64 uncorrected comparisons.

More distant-attending heads were not more important under this intervention. Distant mass had Spearman correlation -0.18 with absolute NLL effect and -0.57 with Jensen--Shannon output change. This can mean redundancy, small OV/output-projection magnitude, or compensation by later layers. It is direct evidence against using the attention heat map as an importance map; it is not evidence that distant heads have no function.

## Controls and provenance

The cohort contains 32 distinct validation replays selected without replacement with seed 20260911. One full 256+4-frame window and one deterministic ego side were selected per replay. The source is canonical schema 7 `ranked-anonymized-1`; all identities correctly use the rank fallback because these replays are anonymous. The source rows contain schema-7 velocity and flag columns, but the frozen O50 `ITEM_PLAYER_PROJECTION` intentionally excludes them. The exact raw-window, model-context, target, replay, source-index, manifest, and sidecar hashes are in `artifacts/o50/results.json`.

The portable bundle SHA-256 is `47e81cf4400e67eff02fe7f6e800cd189e7af005d83cde4150e7a253cbb0e629`; its manifest records source checkpoint SHA-256 `bd1301d33df43f2fec23e6316004f18ca7e872a6ae182dd8c732d5bf5bab10d0`. The analysis worktree base is `52da5b2de1d402b86404fc4d24e382409649e525`. The final RTX 3060 run took 12.7 seconds with PyTorch 2.11.0+cu130.

The manual teacher-forced decoder reproduced every action from the public `sample_conditioned(..., argmax=True)` path. The attention trace reproduced dense trunk hidden states exactly. Removing all head hooks restored the baseline exactly.

Masking rank identity changed mean NLL by +0.0005 nats, with interval [-0.0021, 0.0034]. Replacing every old controller-action history with a different replay's history, while preserving the current action and all world states, changed it by +0.1685 [0.1024, 0.2404]. The latter is only a sensitivity diagnosis: it is an off-manifold action/world mismatch, and scoring the original expert suffix after a corrupted prefix is not a proper likelihood or performance metric.

## Scope and next test

This is one O50 checkpoint and 32 expert-distribution windows, not either historical success/failure pair. History removal also removes the checkpoint's only way to infer motion that is absent from its projected current frame. Thus, the 32--64-frame threshold does not isolate latent intent from ordinary state estimation. NLL changes do not establish a return effect.

The next small causal test should use real matched transitions. Match windows on current projected state while varying whether the expert is about to continue or change an action. Patch old action embeddings and world-state token components separately, then patch the few local and high-lag heads in both directions. Repeat this exact fixed cohort on the requested historical checkpoints before any circuit claim.

## Historical checkpoints

I repeated the non-head parts of the probe on 256 fixed, distinct replays for four Muon and two AdamW model exports. The cohort, statistics, player vocabulary, source names, 44 source manifests, schema, and decoder protocol were invariant. Each export's byte count and SHA-256 matched its sidecar before restricted loading. The cohort retains anonymous replay rank IDs 1--3; frozen gameplay instead masks identity to 0. The results are under `artifacts/o50_historical/`.

| Run and step | Joint teacher NLL (bits/frame) | Attention entropy (nats/head) | Effective positions/head | Mean lag | Mass at lag >=64 |
|---|---:|---:|---:|---:|---:|
| Muon 24,576 | 1.061 | 0.150 | 1.25 | 70.4 | 0.377 |
| Muon 57,344 | 1.069 | 0.043 | 1.06 | 65.6 | 0.347 |
| Muon 98,304 | 1.035 | 0.032 | 1.04 | 62.6 | 0.328 |
| Muon 131,072 | 1.028 | 0.032 | 1.04 | 62.1 | 0.324 |
| AdamW 57,344 | 1.018 | 2.942 | 37.42 | 63.5 | 0.315 |
| AdamW 131,072 | 0.986 | 3.128 | 43.18 | 63.6 | 0.308 |

The attention columns come from a standalone re-audit with FP32 QK accumulation and softmax, `highest` FP32 matmul precision, and TF32 disabled. Every traced hidden state matched the public trunk exactly. “Effective positions” is the replay mean of `exp(row entropy)`, then averaged over heads and layers; it is not `exp(mean entropy)`. This cohort retains the recorded anonymous replay identity. The corrected artifacts are under `artifacts/o50_attention_fp32/` and join the untouched functional artifacts by exact cohort, checkpoint, and artifact hashes.

Muon's small 24k-to-57k NLL regression has the same direction as its gameplay slump, but the paired difference was only +0.008 bits/frame with a 95% replay-bootstrap interval of [-0.019, +0.035]. This cohort does not resolve it. The 98k and 131k differences from 24k were -0.026 [-0.049, -0.003] and -0.033 [-0.058, -0.009] bits/frame. AdamW was lower than Muon on this cohort by 0.051 [0.022, 0.081] bits/frame at 57k and 0.042 [0.017, 0.067] at 131k. These are paired offline diagnoses, not policy-selection evidence.

The clearest representation difference is attention concentration. After 57k, the average Muon head is almost one-hot, yet its chosen positions remain about as old as AdamW's distributed attention. The pattern persists from the Muon slump through its later recovery, so attention concentration alone is not a failure marker. Removing context older than 32 frames cost 0.019 and 0.023 nats/group/frame for Muon at 57k and 131k, versus 0.026 and 0.029 for AdamW. Sharper attention therefore did not imply greater aggregate dependence on distant history under this intervention.

Muon's centered, pre-mask button-logit absolute p99.9 grew from 46 at 24k to 78, 98, and 103. AdamW's values were 48 and 65. These logits are before trigger-conditioned invalid-button masking. Their magnitude does not measure deployed sampling confidence, and a radial or temperature intervention must preserve the mask and full autoregressive factorization.

The historical teacher scorer evaluates the teacher-forced joint density of all four expert actions. It does not marginalize the first two sampled actions that the frozen delayed-truncation policy discards before executing actions three and four. Its parallel BF16 arithmetic matched 99.83--99.98% of public cached-decoder action elements, but it is not exact logit parity. The JSON records this numerical limit and exact zero error for the separately traced trunk. The separate sequential cached marginal scorer integrates the discarded latent ancestors.

## Final-query routing intervention

I compared the final Muon and AdamW checkpoints on the same 256 replay contexts, with ego identity masked to zero as in frozen gameplay. At every trunk layer, the control kept public soft attention. The argmax treatment replaced only each head's final-query output with the value at its highest-score key. The current-key control used the value at the current position. Earlier queries were unchanged, and every downstream layer recomputed its keys and values. The functional soft path matched the public trunk exactly.

| Model and treatment | NLL change (bits/predicted frame) | 95% paired interval | Output JS (nats) | Greedy group/action change |
|---|---:|---:|---:|---:|
| Muon, argmax key | -0.0012 | [-0.0044, 0.0020] | 0.000054 | 0.37% |
| AdamW, argmax key | +0.0577 | [0.0410, 0.0753] | 0.001289 | 1.71% |
| Muon, current key | +0.8026 | [0.6532, 0.9603] | 0.020625 | 4.93% |
| AdamW, current key | +0.7105 | [0.5794, 0.8481] | 0.019676 | 5.54% |

Muon's nearly one-hot attention is functionally close to hard argmax routing on these replay contexts. AdamW's distributed attention mass is functionally material under the same intervention. Both models reject the trivial current-position route. These output and teacher-NLL effects do not estimate gameplay return or show that hard routing caused Muon's training slump.

The first launch used dense SDPA for its soft reconstruction. It correctly aborted before treatment output because the historical CUDA model's public varlen-flash path differed by 0.33594 in hidden-state maximum absolute error. The successful probe uses the public attention path for every untreated row and only replaces the final projected row. A numerical control on the first eight fixed contexts compared `c_proj([B,D])` with the final row of `c_proj([B,L,D])`; all 16 layers, both models, and both selected-value treatments were bit-exact. The successful result is `artifacts/attention_routing/results.json`; the projection control is `artifacts/attention_routing/projection_shape_control.json`.
