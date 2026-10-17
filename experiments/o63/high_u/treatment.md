# O63: follow-up high-u evaluation

Status: preparing six evaluation-only jobs, authorized by the user.

This is an exploratory extension of the completed O63 sweep. The original
u75 primary comparison remains fixed. No model is retrained.

Evaluate the final return60 and bootstrap30 checkpoints at u95, u98, and u99.
Each point uses the same 96 matchups, seed 0, compiled FP32 inference, maximum
parallelism 32, and deployment timing (4, 2, 2) as the original sweep. Use the
existing dashboard cluster-bootstrap `net_stock_lcb` metric.

Run on Modal, one RTX PRO 6000 per evaluation. Each job requests 32 CPU cores
(limit 48), 128 GiB RAM (limit 384), and 128 GiB scratch. No replay datasets or
training data are downloaded. Evaluation reads only its verified final
checkpoint, feature-normalization statistics, and emulator fixtures.

Each target has a separate output directory and W&B namespace (`eval_u95`,
`eval_u98`, `eval_u99`) on the original training run. Checkpoint hashes and full
commands are recorded in [planned_evaluations.json](planned_evaluations.json).
Evaluation source and Modal launch identities will be recorded in
`launches.json`. Training source remains
`292b09dc4b77680704e272de63976db46ab00190`.

Rationale: return60 and bootstrap30 improved from u75 to u90. The control p90
is taken over strictly positive returns; its threshold is the empirical 97.67th
percentile of all valid returns in the stored calibration sample. That global
raw-return percentile is not equivalent to a per-state learned percentile.
Higher u is a testable target-choice hypothesis, not an established improvement.

The original warmup difference remains: 059 conditioned from update 1, and 063
from 4,097. These additional evaluations do not resolve that training difference.
HL-Gauss smoothing also means that requested u is not an exact calibrated return
percentile. High settings may have limited training support.
