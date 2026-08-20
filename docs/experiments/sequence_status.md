# Action-model experiment status

Updated: 2026-08-19

This file is the result index. Each linked file contains the full plan, command, evidence, and
decision. Git keeps the source that made each historical result.

## Final systems decisions

| Study | Result | Decision |
| --- | --- | --- |
| Data pipeline | The compact policy data passed its full value audit. It has 114,768 replays and about 1.23 billion frames. | Use compact policy schema 2 for current training. |
| P0 | The short, full-causal model completed training and its final evaluation. | Keep P0 as the deployed baseline. |
| P1 | The matched long-context SWA model improved offline NLL but failed the paired H2H gate. | Do not replace P0. |
| P2 | Temporal KV decode was faster, but exact parity failed. | Use full rolling-window recomputation. |

The compact data uses 13.34 GB of compressed storage. The full v7 data uses 29.82 GB. Decoded
training arrays fell from 802.47 GB to 76.19 GB. The replay reservoir keeps one replay per batch
row and a one-batch cooldown. See [data_pipeline.md](data_pipeline.md).

P0 is W&B run `obx3o3az`. Its final checkpoint SHA-256 is
`5d12d010fa3acd1ec07bd86a8e85d2cbb84c584a77b9b79e90dc6fcf03c32e4b`. Its official 96-boot
evaluation reported 0.777 stocks taken and 1.468 stocks lost per active minute. See
[e0_normalized_aux_bc.md](e0_normalized_aux_bc.md).

Matched P1 is W&B run `46zi7fgo`. Its final checkpoint SHA-256 is
`8e9b04c91aa76d1ba49a910c82f1328bc1b0dc3ce7dabf3e9018cb556d964148`. In 64 mirrored H2H pairs,
its mean paired stock difference was -0.062. See [p1_matched_attention.md](p1_matched_attention.md).

P2 had 14 FP32 and 32 FP16 sampled-action mismatches in 6,195 slot-frames. KV reduced model time
per row from 0.390 ms to 0.146 ms. Evaluation wall time fell from 672 to 507 seconds. The mismatch
invalidates the policy path. See [p2_temporal_kv_ablation.md](p2_temporal_kv_ablation.md).

## Scientific sequence

| Experiment | Status | Decision or next gate |
| --- | --- | --- |
| E0 | Complete | Keep P0 fixed as the reference. |
| E1 | Complete | The state-only MLP did not replace P0. Keep it as the E2 capacity control. |
| E2 | Partial result | E2-S stopped after step 2,300. It has no final policy result. |
| 024 temporal MTP | Implemented | No final training result is recorded. |
| 025 flow head | Implemented as a direct follow-on to 024 | No final training result is recorded. |
| E3 to E7 plans | Not selected as final results | Use the linked plans only when a new run starts. |
| 034 rank-weighted BC | Implemented; production run pending | Gate: `net_stock_lcb` vs 026 run `cqbbbg77`. |
| 036 advantage-weighted BC | Implemented; production run pending | Run `--audit-returns` first; gate: `net_stock_lcb` vs 026 run `cqbbbg77`. |

E1 is W&B run `q3aojgfm`. Its final checkpoint SHA-256 is
`c175aa53f1d0f4ff80157b26a51c67cf55c4577ded656b60b97d12e10d8a560f`. Its paired mean stock
difference was -0.391. See [e1_output_head_capacity.md](e1_output_head_capacity.md).

E2-S is W&B run `43ppjdxc`. The run stopped after the last observed step 2,300. The step-2,048
checkpoint reached R2. It has no final checkpoint, CPU result, H2H result, or promotion decision.
See [e2_within_frame_factorization.md](e2_within_frame_factorization.md).

## Plans and result records

- [Action chunk roadmap](action_chunk_roadmap.md)
- [E0 attention ablation](e0_attention_ablation.md)
- [E0 normalized auxiliary BC](e0_normalized_aux_bc.md)
- [P1 matched attention](p1_matched_attention.md)
- [P1 old recompute rescore](p1_old_recompute_rescore.md)
- [P2 temporal KV ablation](p2_temporal_kv_ablation.md)
- [E1 output-head capacity](e1_output_head_capacity.md)
- [E2 within-frame factorization](e2_within_frame_factorization.md)
- [E3 temporal factorization](e3_temporal_factorization.md)
- [E4 dense chunk readiness](e4_dense_chunk_readiness.md)
- [E5 primary AWR](e5_primary_awr.md)
- [E6 chunk critic](e6_chunk_critic.md)
- [E7 chunk AWR execution](e7_chunk_awr_execution.md)
- [034 rank-weighted BC](034_rank_weighted_bc.md)
- [035 recursive chunk decoder](035_recursive_chunk_decoder.md)
- [036 advantage-weighted BC](036_advantage_weighted_bc.md)

## Fixed evaluation rules

- Closed-loop results decide promotion. Offline metrics explain the result.
- Use 32 deterministic CPU boots for periodic checks and 96 boots for final checks.
- Use 64 mirrored H2H configurations for a challenger and its reference.
- Keep the character schedule, decode seed, temperature, frame budget, and concurrency fixed.
- Bootstrap CPU uncertainty by boot. Do not treat games from one boot as independent samples.
- Save match rows, replay files, worker logs, and the resolved decode protocol.
- Use at least three paired training seeds for an architecture claim. Use five for a final result.

## Fixed infrastructure rules

- Run one training experiment at a time.
- Use the compact policy data, a 250 GB disk, and a 128 GB data cache.
- Require FlexAttention for a timing comparison. Do not accept a quiet dense fallback.
- Upload checkpoints before expensive evaluation.
- Treat an upload failure as a run failure.
- Launch only from a clean, pushed commit.
- Record the source commit, run ID, checkpoint hash, host facts, timing, and final decision.
