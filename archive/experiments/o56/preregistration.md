# O56 decoder-capacity reallocation

Registered before the first O56 launch.

## Question

Does moving capacity from the temporal trunk into the within-frame action decoder reduce jump-initiated airdodges that fail to become wavedashes?

The historical control is W&B run `ericyuegu/hal/0ksityvv`, final update 16,384. Its display name is
`260911-182038_052_adamw_temporal_awr_adamw052-d256-L16-h4-Lc256-t128x4-o1-2-3-4-5-6-9-12-16-20-d2r2-nonlinear-head-trunk-skip-projectiles-v8-all-adamw-alr0.0017-awd0.0001-awr-v-near-b199.5-g0.99855-wu512__o52-g99855`.
The run logged no Git SHA and W&B marks it crashed, so it is eligible only if the final checkpoint and complete 96-boot evaluation evidence validate. Its immutable W&B source artifact is `source-hal-experiments_052_adamw_temporal_awr.py:v4`, digest `0251bdb8130e808df574746d1a0972b9`; the experiment source SHA-256 is `d0e75291dc3e0c322aea0cdd8b57573c5fec20b2c843b71943b8e3777f247b69`.

## Treatment and control

| | O52 control | O56 treatment |
| --- | ---: | ---: |
| Trunk | d256, 16 layers, 4 heads | d256, 11 layers, 4 heads |
| Temporal decoder | d128, 4 layers, 2 heads, FF384 | d192, 4 layers, 3 heads, FF576 |
| Action-head width | 128 | 192 |
| Total parameters | 14,480,922 | 11,523,226 |
| Trunk + inputs + 4 × decoder and heads | 17,328,146 | 17,293,842 |
| Effective training FLOPs/update | 14,416,825,417,728 | 15,156,197,326,848 |

The capacity control differs by -0.20%. The training estimate increases by 5.13% because training evaluates the temporal decoder over 128 supervised prefixes and ten offsets. Literal inference FLOPs and measured decode latency are diagnostics, not controls.

## Invariants

The treatment retains the control's seed 0; ranked-anonymized-1 policy-world-v8 corpus, selection and manifest; replay-ring order; identity vocabulary and dropout; feature representation; batch size 512; 16,384 updates; 512 warm-up updates; AdamW learning rate `1.7e-3`, betas `(0.9, 0.95)`, epsilon `1e-12`, weight decay `1e-4`; global gradient clip 1.0; AWR beta 199.5, cap 3.5, gamma 0.99855, value weight 1.0 and far-offset weight 0.5; 256-frame context; and offsets `1,2,3,4,5,6,9,12,16,20`.

The launcher's Git SHA, resolved configuration, W&B identity, checkpoint hashes, data hashes, and evaluation evidence hashes are part of the treatment identity.

## Behavioral measure

A wavedash attempt is an `AIRDODGE` onset with `KNEE_BEND` in the preceding 12 frames. It succeeds when `LANDING_SPECIAL` begins in the next 10 frames and fails otherwise. An attempt whose complete outcome window is not recorded at replay end is censored. The primary measure is failed attempts per active minute. Also report raw failures, successful wavedashes per active minute, and pooled success rate `successful / (successful + failed)`.

Analyze the model port in every completed match from the same fixed 96 boots. The analysis must reject missing or unreadable replays, incomplete boots, changed matchup schedules, changed evaluation protocols, and checkpoint mismatches. Pair boots and use a seeded 2,000-resample two-sided 95% percentile interval for treatment-minus-control deltas. Report net stocks/min, net damage/min, and decode latency beside the behavior measures.

## Eligibility and interpretation

The treatment is eligible only if it reaches update 16,384 with finite training and validation metrics, produces the expected final checkpoint, and completes all 96 final evaluation boots under compiled BF16 inference with prediction 4, delay 2, replan 2, and at most 7,200 frames per boot.

This is a one-seed measurement experiment. No metric automatically selects or adopts the treatment. Offline metrics are diagnostic only.

## Evaluation correction

After training launched, inspection found that the inherited O52 head-to-head adapter discarded the first `delay_frames` predictions instead of conditioning each new plan on the actions already committed to transport. This does not affect training. All control and treatment gameplay evidence used for the comparison must be regenerated with `conditioned_pending_actions_v1`: the committed prefix is forced through the temporal decoder, only its uncommitted tail is sampled, and bootstrap transport contains neutral actions. Match-row schema 7 records this semantic contract. Earlier schema-6 truncation evaluations are ineligible.

## Command

```text
uv run scripts/launch_modal.py --gpu RTX-PRO-6000 -- \
  uv run experiments/056_decoder_capacity_reallocation.py train --proxy \
  --cfg.automatic-evaluation --comment o56-decoder-capacity
```
