# E0: normalized auxiliary BC baseline

Status: design audit

Owner: Codex with a focused implementation agent

Starting commit: `41692c3`

## Question

What is the closed-loop performance of a plain `(1, 5, 9, 13)` MTP policy when the three discarded
heads have one fixed total loss weight?

E0 is the reference for E1 and E2. It does not use AWR, rank weights, action-group conditioning, or
temporal conditioning.

## Intended files

- Add `experiments/023_mtp_heads.py`.
- Add `tests/experiments/test_023_mtp_heads.py`.
- Update this file with implementation, launch, W&B, and evaluation findings.
- Do not change historical experiment files 020, 021, or 022 for E0.
- Do not change shared training or evaluation code unless a failing parity test proves that E0 needs
  a shared fix. Record any such change here before making it.

## Lineage

Use one 022-derived experiment file for E0, E1, and E2. A model-mode field will identify the linear,
independent-MLP, and factored-MLP arms. This avoids copying the training and evaluation harness for
each head ablation.

Fork the current 022 training and evaluation path after its concurrent SWA and closed-loop fixes are
committed. Keep:

- The shared sliding-window trunk.
- V7 observation features and data.
- Compiled training and eager evaluation.
- Incremental fp16 closed-loop decode.
- Checkpoint upload and resume support.
- Fixed CPU-opponent evaluation and replay artifacts.
- Optional final mirrored head-to-head evaluation.

Remove:

- Return and reward labeling used only by AWR.
- The value head and critic losses.
- AWR and rank sample weights.
- Within-frame autoregressive conditioning.
- Factored-head MLPs.

E0 uses one linear projection at each offset. Its four slices are independent categorical groups.
This matches the naive MTP probability model. E1 and E2 will add their head modes in later audited
commits.

## Core objective

Use offsets `(1, 5, 9, 13)` and:

\[
L=L_1+\lambda_{aux}\frac{L_5+L_9+L_{13}}{3}.
\]

Set `aux_loss_weight = 1.0`. The old per-head value of `1.0` gave total auxiliary weight `3.0` and is
not the same experiment.

The objective must have tests for:

- No auxiliary heads.
- One auxiliary head.
- Three auxiliary heads.
- Invariance of total auxiliary scale to the number of heads.
- Transition weighting without advantage weighting.
- Exact equality between the configured loss and a hand calculation.

## Model and training configuration

Planned defaults:

- `d_model = 256`
- `n_layers = 8`
- `n_heads = 4`
- `attn_window = 128`
- `L_ctx = 1024`
- `batch_size = 128`
- `grad_accum_steps = 1`
- `head_offsets = (1, 5, 9, 13)`
- `max_steps = 16384`
- `warmup_steps = 500`
- `muon_lr = 0.02`
- `adam_lr = 8.5e-4`
- `weight_decay = 0.01`
- `amp_dtype = bfloat16`
- `compile_trunk = True`
- `data_root = data/processed/ranked-anonymized-1/mds-v7`
- `mds_schema_version = 7`
- `windows_per_replay = 2`
- `seed = 0`

`L_ctx = 1024` is required by the current eight-layer, 128-frame SWA incremental decoder. The
effective receptive field is 1017 frames, and the raw rolling context must extend beyond its left
edge.

The step token budget is 131,072. Do not change it to fit a specific GPU without recording the
change here.

## Validation metrics

Required offline metrics:

- Total and per-group offset-1 NLL in bits per frame.
- Per-group argmax accuracy.
- Hold and transition NLL and accuracy.
- Predicted transition rate and persistence.
- Per-offset auxiliary NLL.
- Primary and auxiliary trunk-gradient norms.
- Primary-to-auxiliary gradient cosine and sign conflict.
- Training step time and samples per second.

The default `val_n_batches = 32` with batch 128 gives 4,096 validation replays. This is half the
replay count used by the old batch-256 setup, but each replay contributes twice the context length.
Keep 32 for E0 to control wall time. Revisit only if the observed NLL confidence interval is too wide.

## Closed-loop evaluation

Use the existing fixed protocol:

- Periodic CPU-opponent evaluation every 4,096 steps.
- 32 fixed matchups for periodic evaluation.
- 96 fixed matchups at the final checkpoint.
- `eval_max_frames = 7200`.
- `eval_seed = 0`.
- Save match rows and replays.

Primary results:

- Win rate against the level-9 CPU.
- Mean stock difference.
- Paired uncertainty intervals.

Also record action entropy, no-op rate, transition rate, and any repeated-action failure pattern that
is visible in replays.

## Head-to-head policy

E0 becomes the head-to-head reference for E1 and E2. E0 itself does not need an in-process
head-to-head run unless a same-geometry unnormalized checkpoint is available.

For each challenger:

- Run 64 mirrored configurations against E0.
- Use the same matchup and policy seeds in both orientations.
- Report paired stock difference and win rate.
- Keep the 96-matchup CPU evaluation as the common external reference.
- Treat head-to-head as sensitive but potentially non-transitive. Do not hide a CPU regression behind
  one favorable direct matchup.

If a suitable old unnormalized checkpoint is found, E0 may run a secondary head-to-head comparison
against it. The comparison must be labeled historical if its data, geometry, or training length
differs.

## Vast launch and runtime budget

Run one experiment at a time.

Require at least 64 GB of system RAM and prefer 128 GB or more. Prefer a validated RTX 4090 until the
concurrent RTX 5090 compile-cache and smoke-probe changes are
committed and pass a production start. A 5090 is allowed after that gate.

Target total wall time: 3.0 to 3.5 hours, including final evaluation and uploads.

Runtime checks:

- Flag startup longer than 30 minutes.
- Flag sustained training step time above 0.5 seconds.
- Flag GPU utilization below 80% after caches are warm.
- Flag any periodic CPU evaluation longer than 25 minutes.
- After the first periodic evaluation, project the final wall time. Notify the user if it exceeds
  3.5 hours.
- Record dataset download time, compile time, median step time, validation time, closed-loop time,
  upload time, GPU model, CPU count, RAM, and disk throughput when visible.

Use enough system RAM for the streaming dataset. Prior runs showed that low RAM can make the same GPU
about 3.5 times slower. The offer audit must include RAM and download speed, not only GPU type.

## Promotion rule

E0 is a baseline, so completion does not require beating a historical model. Completion requires:

- All tests pass.
- The objective is confirmed as fixed-total auxiliary BC.
- Training reaches step 16,384.
- The final checkpoint and replay evidence upload successfully.
- The final CPU evaluation completes.
- Runtime and throughput are recorded here.

E1 starts only after this document contains the final run name and E0 checkpoint reference.

## Implementation findings

- 020 is the clean independent-head ancestor, but it uses the older full-attention trunk, v5 data,
  and context geometry. A raw 020 rerun would not be a durable base for E1-E7.
- 021 adds v6 features but still contains the old summed auxiliary objective.
- 022 contains the current v7, shared-trunk, compiled training, incremental evaluation, checkpoint,
  and head-to-head paths. It is the correct infrastructure source, but its factored head is not E0.
- Use `L_ctx = 1024`, not the previously run 512-frame geometry. With eight 128-frame SWA layers,
  the cached receptive field reaches 1,017 frames. A 512-frame rolling input cannot be equivalent to
  that incremental cache.
- The local 020 and 022 timing records suggest that 16,384 steps should fit the requested budget on
  a healthy 4090 or 5090, but RAM and page-cache performance can dominate GPU speed.

## Training findings

Pending.

## Evaluation findings

Pending.

## Throughput and infrastructure findings

Pending.
