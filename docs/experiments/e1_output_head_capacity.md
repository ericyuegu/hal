# E1: output-head capacity control

Status: blocked on E0 completion

The final E0 checkpoint ID and measured E0 wall time are pending. Do not implement or launch E1
until E0 reaches step 16,384, uploads `final.pt`, and completes its final CPU evaluation.

## Question

Does a state-only nonlinear output head improve the deployed policy, without action-group
conditioning?

E1 is the capacity control for E2. It changes output-head capacity only. It does not change the
probability factorization.

Hypothesis: a residual MLP will reduce training NLL and may reduce validation NLL, but will not give
a clear closed-loop gain over E0. A positive paired head-to-head result without a CPU regression
promotes E1 to a confirmation run. A mixed result is inconclusive.

The first run uses seed 0 as a screening run because E0 also uses seed 0. Do not treat one training
seed as a final architecture claim. Confirm a promoted result with at least three paired training
seeds before publication.

## Intended files

- Edit `experiments/023_mtp_heads.py` to add the state-only MLP head mode.
- Edit `tests/experiments/test_023_mtp_heads.py` to test the mode, initialization, gradients,
  objective parity, checkpoint loading, and decode parity.
- Update this file with the final implementation, launch command, run IDs, timing, W&B findings,
  CPU results, head-to-head results, and decision.

Do not edit historical experiment files. Do not edit shared training or evaluation code unless a
failing test proves that a shared fix is required. Record that need here before making the change.

## Model change

Add a mode field with stable values such as:

```python
head_mode: Literal["linear", "state_mlp", "factored_mlp"] = "linear"
action_mlp_ratio: int = 2
```

E0 remains `linear`. E1 uses `state_mlp`. The run name and checkpoint configuration must include the
mode and MLP ratio.

Keep E0's `Linear(d_model, 355)` projection at every offset. It supplies one classifier slice for
each action group. For offset `o` and group `g`, add:

\[
r_{o,g}(h)=W_{2,o,g}\,\operatorname{SiLU}
\left(W_{1,o,g}\,\operatorname{RMSNorm}(h)\right),
\]

\[
z_{o,g}=h+r_{o,g}(h),
\]

\[
\ell_{o,g}=W_{o,g}z_{o,g}+b_{o,g}.
\]

`W_{o,g}` and `b_{o,g}` are slices of the unchanged E0 classifier. Use a hidden width of
`action_mlp_ratio * d_model`.

Initialize the residual output projection `W_2` and its bias to zero. Do not zero the action
classifier. At initialization, `r(h) = 0`, so E1 must produce exactly the same logits as a same-seed
E0 model for every input.

Create the trunk and E0 classifier projections before the optional MLP modules. This preserves the
same-seed E0 parameters. A test must compare the shared state dictionaries, not only the final
logits.

There is no `W_c` in E1. E1 has no action condition `c`, so a conditioning projection would either
be unused or add another treatment. E2 will let the first MLP affine layer read the concatenated
state and earlier-action embeddings. Its state block is the E1 `W_1`; its condition block is the
learned conditioning path. A separate additive linear bypass is not part of this control.

All new head parameters use the existing output-head AdamW path. Do not route them into the trunk's
Muon group.

## Parameter matching to E2

Use the same number of residual branches and the same hidden width in E1 and E2: one branch for each
of four groups at each of four offsets, with ratio 2.

At `d_model = 256`, E1 adds 4,206,592 parameters:

- One branch has `Linear(256, 512)` and `Linear(512, 256)`, including biases: 262,912 parameters.
- Four groups at four offsets give 16 branches.

E2 will be slightly larger because it also needs action embeddings and condition columns in its
first affine layers. Do not add unused tensors to E1 to claim an exact parameter match. Report the
exact total, trunk, classifier, residual-MLP, embedding, and conditioning parameter counts for both
arms.

If E2 changes total parameters by more than 5% or training step time by more than 10% relative to
E1, add a separate capacity-matched control before making a factorization claim. Do not change E1
after seeing E2 results.

## Objective

Use the exact E0 objective:

\[
L=L_1+\frac{L_5+L_9+L_{13}}{3}.
\]

Fixed settings:

- `head_offsets = (1, 5, 9, 13)`
- `aux_loss_weight = 1.0`
- `transition_loss_weight = 1.0`
- No AWR, critic, value loss, rank weight, or sample weight.
- Execute only offset 1.

The loss, target alignment, reduction, and decode distribution must not depend on `head_mode`.

## Fixed run configuration

Copy the final E0 checkpoint configuration. Only these fields may differ:

- Head mode and MLP ratio.
- Run comment and model tag.
- Head-to-head self label and E0 reference run.

Expected fixed values are:

- Model: `d_model=256`, `n_layers=8`, `n_heads=4`, `attn_window=0`.
- Context: `L_ctx=1024`, `batch_size=128`, `grad_accum_steps=1`.
- Training: `max_steps=16384`, `warmup_steps=500`, `seed=0`.
- Optimizer: `muon_lr=0.02`, `adam_lr=8.5e-4`, `weight_decay=0.01`,
  `head_weight_decay=True`.
- Numeric path: `amp_dtype=bfloat16`, `allow_tf32=True`, `compile_trunk=True`.
- Input regularization: `history_dropout_p=0.0`.
- Data: `data/processed/ranked-anonymized-1/mds-v7`, schema 7,
  `windows_per_replay=2`.
- Loader: `shuffle_block_size=2000`, `num_workers=16`, `prefetch_factor=4`.
- Validation: `val_every=1024`, `val_n_batches=32`,
  `gradient_diagnostic_batch_size=16`.
- Checkpoints: `ckpt_every=2048`.
- Decode: temperature 1, no per-group override, no support mask, no min-p filter, no click repair,
  `exec_horizon=1`, full-window recomputation, fp16 evaluation enabled.
- CPU evaluation: every 4,096 steps with 32 matchups, then 96 final matchups,
  `eval_max_frames=7200`, `eval_seed=0`.
- Evaluation and training do not overlap on one GPU.

The token budget is 131,072 frames per optimizer step. Use `require_flex=True` on Vast. If final E0
used a different value, record and resolve the difference before launch.

## Required tests

1. Construct same-seed E0 and E1 models. Assert exact equality for the trunk, input projection,
   embeddings, and all four action classifiers.
2. Run both models on the same batch in evaluation mode. Assert exact equality for every offset and
   group logit.
3. Use the same decode seed and context. Assert that E0 and initial E1 sample the same action.
4. Assert that every residual output weight and bias is zero. Assert that the first-layer weights
   are not all zero.
5. Backpropagate the real BC objective. At initialization, assert a finite, nonzero gradient on each
   residual output projection. A zero first-layer gradient at this point is expected because the
   zero output projection blocks it.
6. Take one optimizer step, then backpropagate a new batch. Assert a finite, nonzero first-layer
   gradient. This proves the branch is not permanently dead.
7. Assert that E1 logits change after its residual branch learns.
8. Assert exact E0/E1 objective equality when given equal logits and targets.
9. Assert all new parameters appear once in the AdamW partition and never in the Muon partition.
10. Load an E0 checkpoint after the new mode field exists. It must resolve to `linear` and reproduce
    the E0 policy.
11. Save and reload E1. Assert mode, MLP ratio, logits, and decode settings round-trip.
12. Run the existing small end-to-end train test in `state_mlp` mode. It must save `final.pt` and
    must not log AWR, rank, value, or critic fields.

Run focused tests, Ruff, Python compilation, and `git diff --check` before launch.

## Offline records

Use the same frozen validation split as E0. Log:

- Total and per-group offset-1 NLL.
- Per-group argmax accuracy and exact four-group frame accuracy.
- Hold and transition NLL and accuracy for every group.
- Predicted transition rate, persistence, and change-event F1.
- NLL at offsets 1, 5, 9, and 13.
- Per-offset shared-trunk gradient norms.
- Primary-to-auxiliary gradient cosine, norm ratio, and sign conflict.
- Residual output norm relative to trunk-state norm, by group and offset.
- First-layer and residual-output gradient norms.
- Total, trunk, classifier, and MLP parameter counts.
- Median warm step time, samples per second, tokens per second, peak GPU memory, and attention path.
- Validation and closed-loop evaluation duration.

Compare E1 with E0 at the same steps. Lower training NLL alone is not evidence of a better policy.

## Closed-loop evaluation

Run the final 96-matchup level-9 CPU protocol with the exact E0 matchup seed, decode seed, frame
budget, and concurrency rule. Save `match_rows.json` and all replay files.

Report:

- Stocks taken and lost per active minute.
- Damage dealt and taken per active minute.
- Seeded bootstrap intervals.
- Paired E1-minus-E0 deltas from aligned CPU match rows.
- Crash rate and completed-match count.

The CPU protocol does not report game win rate. Do not infer a win from a stock lead at the frame
limit.

After the CPU evaluation, run head-to-head against the final E0 checkpoint:

- Set `final_h2h_reference_run` to the final E0 run name.
- Use `experiments/023_mtp_heads.py` for both policies.
- Use self label `023-e1-state-mlp` and reference label `023-e0`.
- Run 64 mirrored configurations, two games per configuration, for 128 games total.
- Keep `eval_max_frames=7200` and `eval_seed=0`.
- Report non-tied win rate, per-game stock difference, paired per-configuration stock difference,
  confidence intervals, stage results, and character results.

Head-to-head is sensitive and can be non-transitive. It does not replace the CPU result.

## Timing gates

Target total wall time is 3.0 to 3.5 hours, including CPU evaluation, head-to-head evaluation,
uploads, and shutdown. Use one Vast experiment at a time.

- Record provisioning, data staging, validation caching, compilation, training, each evaluation,
  head-to-head, and upload time separately.
- Flag startup longer than 30 minutes.
- After compilation, compare the median of at least 100 warm E1 steps with E0. Flag a slowdown
  above 25%.
- Flag sustained step time above 0.5 seconds or GPU utilization below 80% after warmup.
- Flag any 32-match periodic CPU evaluation longer than 25 minutes.
- At step 4,096, use measured training and evaluation time to project the complete run. Notify the
  user before continuing silently if the projection exceeds 3.5 hours.
- Record CPU count, RAM, GPU model, peak GPU memory, disk throughput, and download speed when
  available.

Use at least 64 GB of system RAM and prefer 128 GB. Do not accept a faster GPU on a low-RAM host if
it makes the streaming loader slower. E0 showed eviction and download stalls with a 440 GB cache on
a 500 GB disk. Audit a 1 TB disk for E1 so the materialized training shards can remain cached. Record
the storage cost and compare the stall time with E0.

## Evidence to retain

W&B must retain:

- Git commit, complete configuration, run ID, Vast instance ID, GPU and host facts.
- Training, validation, gradient, parameter-count, memory, throughput, and timing metrics.
- Periodic and final CPU metrics at their source steps.
- Final head-to-head summary.

R2 and the run directory must retain:

- `latest.pt`, step checkpoints, and `final.pt`.
- Final CPU `match_rows.json` and replay files.
- Head-to-head `meta.json`, `matches.jsonl`, and replay files for both orientations.
- Evaluation worker logs and result JSON files.

Record the final E0 checkpoint reference before launch. Verify that E1 fetched it at startup. Do not
allow the Vast instance to destroy itself until the final checkpoint and evaluation evidence have
uploaded.

## Decision rules

The E1 run is invalid if:

- Initial E0/E1 logits differ.
- The residual branch remains gradient-dead after one update.
- Any fixed data, optimizer, target, objective, seed, decode, or evaluation setting differs from E0.
- Training does not reach step 16,384.
- The final checkpoint, CPU rows, head-to-head records, or replays are missing.
- Evaluation crashes make the planned comparison incomplete.

Interpret a valid run as follows:

- Clear H2H and paired CPU improvement: nonlinear head capacity matters. E2 must beat E1, not only
  E0, to support a factorization claim.
- Better offline metrics but no closed-loop gain: capacity improves fitting but not control. Keep
  E1 as the required E2 control.
- Closed-loop regression: the MLP is harmful at this scale or optimization setting. Do not tune E1
  after seeing E2. E2 must beat both E0 and E1 without an unacceptable latency cost.
- Conflicting CPU and H2H results: call the result inconclusive. Inspect paired rows and replays, and
  run more predeclared seeds before promotion.

Do not promote E1 as the new policy baseline from NLL alone. E2 planning may begin while E1 trains,
but E2 may not launch until E1 evidence is complete.

## Results

Pending E0 completion.
