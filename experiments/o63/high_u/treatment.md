# O63: follow-up high-u evaluation

Status: all six evaluation-only jobs completed. No model was retrained.

This is an exploratory extension of the completed O63 sweep. The original
u75 primary comparison remains fixed.

Evaluate the final return60 and bootstrap30 checkpoints at u95, u98, and u99.
Each point uses the same 96 matchups, seed 0, compiled inference with FP32 weights
and BF16 autocast, maximum parallelism 32, and deployment timing (4, 2, 2) as the original sweep. Use the
existing dashboard cluster-bootstrap `net_stock_lcb` metric.

Run on Modal, one RTX PRO 6000 per evaluation. Each job requests 32 CPU cores
(limit 48), 128 GiB RAM (limit 384), and 512 GiB scratch. No replay datasets or
training data are downloaded. Evaluation reads only its verified final
checkpoint, feature-normalization statistics, and emulator fixtures.

Each target has a separate output directory and W&B namespace (`eval_u95`,
`eval_u98`, `eval_u99`) on the original training run. Checkpoint hashes and full
commands are recorded in [planned_evaluations.json](planned_evaluations.json).
Evaluation source and Modal launch identities are recorded in
[launches.json](launches.json). Training source remains
`292b09dc4b77680704e272de63976db46ab00190`.

Pre-launch rationale: return60 and bootstrap30 improved from u75 to u90. The control p90
is taken over strictly positive returns; its threshold is the empirical 97.67th
percentile of all valid returns in the stored calibration sample. That global
raw-return percentile is not equivalent to a per-state learned percentile.
This motivated testing higher u with the existing checkpoints.

The original warmup difference remains: 059 conditioned from update 1, and 063
from 4,097. These additional evaluations do not resolve that training difference.
HL-Gauss smoothing also means that requested u is not an exact calibrated return
percentile. High settings may have limited training support.

The first submission requested 128 GiB scratch. Modal rejected Function
creation because its minimum is 512 GiB; no evaluation workers ran. The
corrected submissions request 512 GiB. See [failed_launches.json](failed_launches.json).

## Results

The original primary comparison remains fixed at u75 versus 059 p90:

| Arm | Target | Dashboard net_stock_lcb |
| --- | --- | ---: |
| control | p90 | 0.25510 |
| return60 | u75 | -0.04773 |
| bootstrap30 | u75 | -0.02124 |
| match | u75 | -0.01062 |

The following sweep is exploratory. All values use the same final checkpoint
within each arm, seed 0, and the same 96-matchup protocol. Values are net stocks
per active gameplay minute. Selecting a setting from this sweep does not revise
the fixed primary result or provide an independent confirmation.

| Arm | Target | Mean net stocks | Dashboard LCB |
| --- | --- | ---: | ---: |
| return60 | u50 | -0.12734 | -0.24947 |
| return60 | u75 | 0.07433 | -0.04773 |
| return60 | u90 | 0.17590 | 0.03196 |
| return60 | u95 | 0.29793 | 0.17539 |
| return60 | u98 | 0.44755 | 0.33539 |
| return60 | u99 | 0.37244 | 0.24454 |
| bootstrap30 | u50 | -0.04775 | -0.14858 |
| bootstrap30 | u75 | 0.10095 | -0.02124 |
| bootstrap30 | u90 | 0.24460 | 0.13295 |
| bootstrap30 | u95 | 0.56362 | 0.45154 |
| bootstrap30 | u98 | 0.37290 | 0.23430 |
| bootstrap30 | u99 | 0.44216 | 0.30864 |

Gameplay figure: [gameplay_sweep.svg](gameplay_sweep.svg).

| Follow-up cohort | Completed boots | Failed worker attempts | Timed-out worker attempts |
| --- | ---: | ---: | ---: |
| return60 u95 | 96 | 0 | 0 |
| return60 u98 | 96 | 0 | 0 |
| return60 u99 | 96 | 2 | 0 |
| bootstrap30 u95 | 96 | 0 | 0 |
| bootstrap30 u98 | 96 | 1 | 0 |
| bootstrap30 u99 | 96 | 2 | 0 |

The unchanged evaluation protocol allows two retries for boots without a complete
trajectory. Failed worker-attempt counters include recovered attempts; final
scores require all 96 accepted boots and 7,200 recorded frames per boot. Recovery
can change the policy-factory RNG call sequence even though the configured seed
remains zero. This is a source of evaluation variability, and retries are retained
in the evidence rather than described as zero failures.

Reference: the original 059 p90 control has mean net stocks **0.37209** and
cluster-bootstrap LCB **0.25510**. Its raw target is a global positive-return
quantile; treatment u is a learned per-state CDF setting.

See [results.json](results.json) for metrics and raw artifact hashes,
[launches.json](launches.json) for Modal identities and exact commands,
[runtime_asset_audit.json](runtime_asset_audit.json) for live GPU and stats hashes,
[download_scope_audit.json](download_scope_audit.json) for live fixture-fetch
commands, and [validation.md](validation.md) for tests and skips.


The strongest follow-up point is **bootstrap30 at u95**: mean net stocks
**0.56362/min**, cluster-bootstrap LCB **0.45154**, versus the original control's
**0.37209/min** and **0.25510**. Return60 peaks at u98, with LCB **0.33539**.
Increasing u beyond either arm's observed peak did not improve its score.
These are results from an adaptive sweep on seed-0 checkpoints, not an independent
confirmation or a replacement for the original fixed-u75 comparison.

All six new cohorts were independently reconstructed from raw match rows. Each
has 96 complete active boots, for **576 boots and 4,147,200 captured frames**
in total. Five worker-attempt failures were recovered by the existing retry
protocol; no accepted boot is missing or crashed. See
[gameplay_audit.json](gameplay_audit.json) and [recovery_audit.json](recovery_audit.json).

Parallel W&B shared writers initially left points out of chart history. The
verified metrics were republished through one primary writer per original run,
without rerunning gameplay. Both original runs are finished, and all six target
values now match W&B chart history at update 16,384. See
[wandb_evaluation_history.json](wandb_evaluation_history.json) and
[wandb_reconciliation.json](wandb_reconciliation.json).

Completion and source identities are in
[completion_evidence.json](completion_evidence.json); all local evidence hashes
are in [evidence_manifest.json](evidence_manifest.json).
