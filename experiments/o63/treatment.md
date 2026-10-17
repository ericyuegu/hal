# O63: return percentile conditioning

Status: all four Modal runs have started training after GCE capacity failures. The actual
063 closed-loop smoke passed. No comparative gameplay results yet.

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
- Validation uses 1,024 samples in every arm. This source has only 1,189
  validation replays, fewer than the default 2,048 samples. The first Modal
  attempts rejected that invalid request before training; all four were
  relaunched with the same explicit override.
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
- Inspect label deciles and calibration after update 4,097. Keep the heads
  trainable and report drift. The user selected this rule after review of the
  HL-Gauss smoothing bias below; drift alone does not trigger a freeze.
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

## Launch record

- Training source commit: `292b09dc4b77680704e272de63976db46ab00190`.
- Branch: `codex/o63-percentile-proxies-20261017`.
- Launcher: `scripts/launch_modal.py`; one `RTX-PRO-6000` per run.
- Every arm requests 32 CPU cores with a 48-core limit, 128 GiB RAM with a
  384 GiB limit, 2048 GiB scratch SSD, and a 24-hour attempt limit.
- Image: `ghcr.io/ericyuegu/hal:cuda13`, with the same locked dependencies,
  fixture layers, and committed source. Retry state: `hal-modal-state`.
- Every command sets `--proxy`,
  `--cfg.source-names ranked-anonymized-1-policy-world-v8`,
  `--cfg.val-n-samples 1024`, `--stop-after-update 16384`, and
  `--closed-loop-eval inline`.
  Each treatment also states `--cfg.arm` explicitly.
- Inline gameplay evaluation runs on the training GPU. Thus all arms use the
  same launcher and GPU class for both training and evaluation.

| Arm | Modal App | Launcher ID |
| --- | --- | --- |
| control | `ap-U1nRBL1qw3JuznZjWQyYLK` | `8aa3c5d7695b4bd89db86b16eacd0b0a` |
| return60 | `ap-UgHGiyYh5I47RF6qDnH4Q1` | `01444dd6736f4ea08a8992dc7a131aa5` |
| bootstrap30 | `ap-d23X70N3YPljHuPNb7LUik` | `72be731b4dc84294a5da18d2bd805541` |
| match | `ap-RfQekINTahPBkfASCORMFJ` | `561cd553e3fd4a79a16d99a375a63aab` |

### GCE preflight attempts

GCE requests began at 07:50 UTC on 2026-10-17. Only two VMs were allocated.
The remaining arms received capacity errors in `us-west4-a`, `us-central1-b`,
`us-east1-b`, `us-west1-a`, `us-west1-b`, `us-west4-c`, and `us-central1-c`.
The Modal fallback began at 08:02 UTC, after more than ten minutes without
capacity for all four runs.

The two GCE VMs used `g4-standard-48`, one GPU, and a 3000 GB Hyperdisk.
The GPU was confirmed as RTX PRO 6000 Blackwell Server Edition, driver
580.178.04, 97,887 MiB. Both attempts stopped before training because the GCE
launcher omitted the last credential record, the W&B key. The treatment log
reported `UsageError: No API key configured`. No optimizer update or training
checkpoint was produced. The user explicitly approved deletion of both failed
VMs and their disks.
Both were deleted after preserving the failure evidence.

Commit `14bb2957` contains those failed attempts' experiment code. Commit
`292b09dc` fixes the launcher record terminator and adds a Bash regression test.
Every Modal arm uses the latter commit.

### Calibration limitation found before training

HL-Gauss smoothing prevents randomized bin labels from being exactly uniform
in every calibrated state. If the true return is always zero, a perfectly fitted
head puts approximately 0.495 of its mass in the central bin. Its randomized
labels then cover approximately [0.2525, 0.7475]. The first and ninth deciles
are approximately 0.3020 and 0.6980, even though the fitted distribution matches
the requested HL-Gauss objective. Freezing that head cannot remove this bias.
This is a limitation of the specified loss/label combination, not evidence by
itself of training drift. The user chose to retain trainable heads and report
this limitation. The loss,
label formula, and three arms remain as specified. A u75 evaluation is a learned
CDF setting; exact percentile calibration is not assumed.

The FunctionCall and launcher identities are recorded in [launches.json](launches.json).

## Review limits

All heads share the existing global gradient clip. Detaching head features
prevents direct trunk gradients, but the new head can change the shared clip
factor. In the focused fixture, the pre-clip norm changed from 90.85994 to
90.87599, a shared-gradient scale change of about 0.018%. This is part of the
specified added-loss treatment; the optimizer rule is unchanged.

Offline validation uses the last context prefix. That position cannot have a
30-frame bootstrap target, so `bootstrap30` validation policy NLL is
unconditioned. It remains diagnostic only. Gameplay uses the explicit target
in the comparison table.

The actual checkpoint evaluation path passed a compiled CUDA/Dolphin smoke on
the local RTX 3060: one 180-frame match, 90 decoder calls, and zero crashes,
failed workers, or timed-out workers, using deployment timing (4, 2, 2).
This checks execution, not gameplay quality or the complete 96-matchup cohort.
See [closed_loop_smoke.json](closed_loop_smoke.json).

Resume tests compare the next batch, model, optimizer, scheduler, prefix draws,
label draws, metrics, and gradients. Compiled CUDA cases pass with deterministic
kernels. Default Inductor embedding reductions use atomic operations and showed
small numerical differences in optimizer state. Production settings remain
unchanged; bitwise equality with those default compiled kernels is not claimed.
