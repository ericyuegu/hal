# O60: plain BC against O52 AWR

One training seed, one final 96-boot evaluation. No sweep, retuning, or checkpoint
selection by offline metrics. The final update is 16,384.

Control: `ericyuegu/hal/0ksityvv`, trained September 11. Its final checkpoint
SHA-256 is `16c702fe3964a59c2f26d88207ef90137d93c07bf67a5d4213f4fde5d25b8631`.
The checkpoint records step 16,383 (16,384 completed updates). W&B marks the run
crashed and records no Git SHA. The Git SHA remains unknown.
The immutable source artifact is
`source-hal-experiments_052_adamw_temporal_awr.py:v4`, digest
`0251bdb8130e808df574746d1a0972b9`; experiment source SHA-256 is
`d0e75291dc3e0c322aea0cdd8b57573c5fec20b2c843b71943b8e3777f247b69`.
Both the downloaded source and the frozen repository source match this hash.

Use the existing `eval_conditioned_rings_v1_step_0016384_s4` evaluation only:
96/96 completed boots, zero crashes, +0.2919776917246444 net stocks/min and
+64.80461722188957 net damage/min. The older +0.398 score is ineligible.
The recovered config, match rows, and metrics accompany this registration.

The treatment starts at seed 0 and update zero. Every policy weight is one,
and the value-regression objective is removed. The value head stays allocated
and is initialized in the same order. It receives no gradient and therefore
makes no contribution to global gradient clipping. This compares the full AWR
training package against plain BC, not advantage weighting alone.

Preserve all settings in `control-config.json`, except the experiment identity
and value loss weight (zero). Preserve 14,480,922 allocated parameters; d256,
16-layer, 4-head trunk; d128, 4-layer, 2-head, FF384 decoder; context 256;
offsets 1,2,3,4,5,6,9,12,16,20; batch 512; 512 warmup updates; AdamW master
LR 0.0017, betas (0.9, 0.95), epsilon 1e-12, weight decay 1e-4, cosine floor
1/170, global clip 1.0. Preserve near/far weighting, normalization, masks,
128 supervised positions, vocabulary, 10% identity masking, replay order, and
validation cohort. Keep return labels in the loader to preserve its data path.

Data: ranked-anonymized-1 policy-world-v8. Training index SHA-256
`b97eab90e761bcf2bf03b48981f0ab6acc1ac3057157c58ae0c5a72c76c43bd8` matches the
control W&B config and checkpoint loader state. Selection SHA-256 is
`ad28edec1ad37565707d1bb0fb2262d94f05617fc957d4d8c8b234979cf3a381`.
Recovered validation index SHA-256 is
`95738bf46604a7e7b67320095efbcc2dbe1a8ae748aa6a269e19f0cee52da389` and statistics
SHA-256 is `6870bda6c0970826f6467407647cf9bb5901ef246b6947d87518db67252cdde3`.
O52 did not persist a separate statistics or validation-cohort content hash;
these are the recovered objects at its configured locations, not invented
historical hashes. The treatment validates these bytes before any update and
records a validation-tensor hash, resolved statistics, environment, seed, and
Git SHA in `provenance.json`.

Launch once on RTX PRO 6000 with the established Modal proxy recipe (32 CPU
request, 48 CPU limit, 128 GiB RAM request, 384 GiB limit, 2048 GiB disk).
Disable automatic gameplay evaluation. Keep validation and checkpoint cadence.
Disable automatic retries to prevent the launcher's pre-checkpoint fresh-start
fallback. An infrastructure failure may be resumed manually from a complete
checkpoint; a scientific failure must not be restarted. Save Python, NumPy,
Torch CPU/CUDA, loader, and identity-mask RNG state along with optimizer and
scheduler state. Do not change the seed or select an earlier checkpoint.

Evaluate only the final checkpoint, once, with 96 complete level-9 CPU boots.
Use conditioned_pending_actions_v1, H4/d2/r2, masked identity, seed 0, 7,200
frames per boot, 32 process workers, compiled BF16 autocast, reduce-overhead,
dense SDPA, and instant match restart. The persisted `dtype=torch.float32`
field denotes parameter storage; the recovered model config specifies BF16
autocast. Match protocol version 2, match-row schema 7, start retries 2, stage
24, and schedule SHA-256
`a2202b353e3e769f2ab25e673226ef29fb6f949f4391c2b9f3003afdc7ce3c15`.
An incomplete evaluation is a failure, not a score. Never rerun the control.

Report pooled net stocks/min and net damage/min over active frames, plus
AWR-minus-BC differences with paired boot-bootstrap percentile 95% intervals:
2,000 resamples, NumPy default_rng seed 60. Pair by boot index and matchup;
pool all match fragments within a boot, excluding zero-active fragments from
rates. Report completion, crashes, cost, artifacts, and deviations. Interpret
all results as one training seed.
