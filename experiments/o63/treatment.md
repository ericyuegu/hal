# O63: return percentile conditioning

Status: implementation and validation in progress. No gameplay results yet.

## Fixed comparison

At update **16,384**, compare each O63 arm at **u75** with O59 at **p90**.
Use `net_stock_lcb` over the same 96 standard matchups. This is the primary
comparison. The u sweep is secondary. Validation NLL is diagnostic only.

| Arm | Program | Decoder condition | Primary target |
| --- | --- | --- | --- |
| Control | 059 | Raw 60-frame discounted return | Positive-return p90 |
| `return60` | 063 | Randomized CDF of the 60-frame discounted return given the state | u75 |
| `bootstrap30` | 063 | Randomized CDF of the 30-frame bootstrapped return given the state | u75 |
| `match` | 063 | Randomized CDF of the full-match discounted return given the state | u75 |

## Invariant inputs

- Data: `ranked-anonymized-1-policy-world-v8` only, as requested for these proxies.
  The original full-corpus recipe is not the data source for this comparison.
- Selection SHA-256: `ad28edec1ad37565707d1bb0fb2262d94f05617fc957d4d8c8b234979cf3a381`.
- Training manifest SHA-256: `b97eab90e761bcf2bf03b48981f0ab6acc1ac3057157c58ae0c5a72c76c43bd8`.
- Reward: opponent damage minus ego damage, plus 120 times net opponent stock
  losses, plus 50 times net opponent final-stock losses.
- Discount: 0.99855. AWR: beta 150, weight cap 10, starts at update 4,097.
- Seed: 0. Conditioning dropout: 20%, independent per context position.
- `proxy_config()`: width 256, 12 trunk layers, six decoder layers, batch 512,
  16,384 updates. The current 059 proxy has **17,059,325** parameters. Each
  treatment has **17,133,246** parameters, including the new 73,921-parameter head.
  These are the existing fixed proxy shapes; no change to reach exactly 15M.
- Same Muon/AdamW settings, schedule, loader, feature statistics, identity
  artifacts, validation cohort, and 32 sampled policy prefixes per window.
- The subset contains 112,188 training replays. All four runs use 112,128 replay
  slots (219 batches) and phase blocks of 20 batches. The original 131,072 slots
  exceed this source. The smaller phase block preserves the 200-batch minimum
  replay reuse gap. Full-corpus configurations retain their original geometry.
- Deployment: prediction 4, delay 2, replan 2 frames.
- One RTX PRO 6000 Blackwell GPU per VM. Prefer GCE. If capacity cannot be
  obtained after 10–20 minutes, use Modal for all four runs. Use the same
  launcher and hardware class for all four arms.

## Treatment

The head reads the same detached, normalized suffix states as the value head.
It predicts 65 bins with 66 evenly spaced symlog edges over [-1000, 1000].
Zero is at the center of the middle bin. Labels outside the range are clamped.
The head uses HL-Gauss cross-entropy, sigma 0.75 bin widths, weight 1.0.
It has the same optimizer roles and initialization rules as the value head.
Its construction preserves the control's parameter draws and process RNG.

The label is `u = F_lo + U * p_bin`, detached. A separate device RNG uses
`seed ^ 0x063AB`. Boundary checkpoints save and restore it. The decoder receives
`(2*u - 1)*120` through the existing return input. No change is made to `hal/`.
The head runs in training and offline diagnosis; actors do not call it.
Conditioning stays absent through update 4,096 in all O63 arms.
O59 retains its existing raw-return conditioning behavior.

`returns[t]` is G_(t+1). The value head predicts that same return. For bootstrap:

`y[t] = returns[t] + gamma**30 * (V(s[t+30]) - returns[t+30])`.

Both return positions must be eligible, and both states must be in the window.
Suffix positions 226–255 have no bootstrap label. This removes 30/128 = 23.4375%
of suffix labels before the 20% dropout. With otherwise complete labels, the
expected conditioned fraction is 61.25%, versus 80% in the other treatments.
The fraction and availability are logged. Full-match eligibility does not
inherit the 60-frame availability mask.

## Evaluation and review

- Automatic gameplay evaluation at updates 8,192 and 16,384: u75 for O63,
  p90 for O59. Use the same 96-matchup schedule and seed.
- At the final checkpoint, also evaluate O63 at u50, u90, and unconditioned.
  Evaluate O59 unconditioned too. Each target has separate evidence and a
  separate W&B namespace. Checkpoint and protocol identities are validated.
- Report the primary comparison first. Report all secondary targets after it.
- Compare `bootstrap30`'s conditioned fraction before interpreting a loss.
- Inspect label deciles and calibration after update 4,097. If drift exceeds
  about 0.05, preserve the evidence and freeze the head at that boundary before
  interpreting further results. Record any such change as a new treatment.
- Measure `match`'s dial effect as net stocks at u90 minus net stocks at u50.
- Compare each treatment's unconditioned play with O59's unconditioned play.
- Record launch IDs, Git SHA, resolved configurations, environment, data
  identities, checkpoint hashes, and gameplay evidence before reporting results.

## Primary results

Pending. No arm has been selected.

| Arm at update 16,384 | Target | net_stock_lcb | Difference from control |
| --- | --- | --- | --- |
| 059 control | p90 | Pending | — |
| 063 return60 | u75 | Pending | Pending |
| 063 bootstrap30 | u75 | Pending | Pending |
| 063 match | u75 | Pending | Pending |

## Secondary results

Pending: u50, u90, and unconditioned gameplay for each O63 arm; unconditioned
059; conditioned fractions; percentile calibration; validation diagnostics.
