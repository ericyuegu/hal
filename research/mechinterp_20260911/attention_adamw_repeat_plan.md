# Repeat check: the requested AdamW run

The W&B URL for `tgpuo1be` and the supplied run name starting
`260909-094223_050_scaled_temporal_awr_` identify the same training run.
Its 57,344- and 131,072-step checkpoints were already included in the
previous attention measurement. This follow-up repeats those measurements
and presents the two AdamW checkpoints together. It does not add an
independent training run.

## Fixed comparison

- Keep the existing 256-replay cohort, recorded player-rank inputs, all
  16 layers and 16 heads, and the most recent query position.
- Keep the existing float32 score and softmax calculation, disabled TF32,
  float64 summary arithmetic, batch size eight, and fixed example zero.
- Use the unchanged `attention_detail.py` program. Its SHA-256 is
  `f09db24acbbb921aebe93d1d05b30e02b354e9279443099761b3e9811f4a1ed5`.
  The cohort SHA-256 is
  `ec61e97e332672a4e4e2fdc438e208a281a4609e10f7eded72368eccac323437`.
- The frozen program measures all six original checkpoints. Run it as-is
  into the new `artifacts/attention_adamw_repeat/` directory. Use its two
  AdamW records for the new comparison. Do not modify its checkpoint list
  or comparison tolerances at runtime.
- Select the same two stored AdamW checkpoints, not checkpoints chosen
  by validation loss. Verify their export hashes and recorded run identity.
- Check exact public-forward parity and the recorded reference comparisons.
  Compare all new summary objects and example-zero arrays with the previous
  measurement. Record any difference rather than overwriting either result.
- The previous numerical results and interactive figure remain unchanged.

The comparison is descriptive. A W&B process marked `crashed` does not by
itself show optimizer divergence. Use the recorded loss and gradient history
to describe training behavior, and do not infer that attention concentration
caused a gameplay failure from a comparison between different optimizers.

No training, resumption, gameplay evaluation, or external write is required.
