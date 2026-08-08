# E1: output-head capacity control

Status: active on Vast instance `47136334`, W&B `q3aojgfm`

P0 is the E0 reference. P1 did not pass its paired head-to-head gate, so P2 cannot change this
scientific choice. P2 only tests decode systems on the P1 checkpoint. E1 code preparation may
overlap P2, but the E1 GPU gate and training run must wait until P2 exits.

The fixed E0 checkpoint is:

- Run: `260807-164825_023_mtp_heads_gpt-d256-L8-h4-Lc256-a1024-full-recompute-o1.5.9.13-linear_ranked-anon-1_e0-normalized-aux-bc`
- W&B ID: `obx3o3az`
- Step: 16,384
- `final.pt` SHA-256: `5d12d010fa3acd1ec07bd86a8e85d2cbb84c584a77b9b79e90dc6fcf03c32e4b`

The checkpoint, final CPU rows, and replays are verified. E1 must record this identity before launch
and must not replace it after seeing E1 results.

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

E1 adds these fields:

```python
head_mode: Literal["linear", "state_mlp"] = "linear"
action_mlp_ratio: int = 2
```

E2 will add `factored_mlp` after E1 is frozen.

E0 remains `linear`. E1 uses `state_mlp`. The run name and checkpoint configuration must include the
mode and MLP ratio.

Keep `model.heads` as the existing `IndependentHead` list. It owns the unchanged base projections
and keeps old 023 checkpoint keys stable. Add one optional adapter module after that list. In linear
mode, do not create adapter parameters. In state-MLP mode, the adapter owns the shared state
projection and every group-and-offset residual projection.

Add one model-level group-logit path and make training, validation, parity, and decode call it.
Linear mode must return the existing head logits without another operation. State-MLP mode adds its
residual. E2 will extend the same path with teacher-forced or sampled earlier groups. Do not keep a
second sampling implementation inside the adapter; one ancestral sampler must apply support masks,
temperature, min-p, trigger repair, and RNG handling for every head mode.

Keep E0's `Linear(d_model, 355)` projection at every offset. It supplies one classifier slice for
each action group. Add one shared state projection:

\[
u(h)=\operatorname{SiLU}\left(W_hh\right).
\]

`h` is already the trunk's final RMS-normalized output. Do not normalize it again inside the head.

For offset `o` and group `g`, add a logit residual:

\[
\ell_{o,g}=W_{o,g}h+b_{o,g}+W_{2,o,g}u(h).
\]

`W_{o,g}` and `b_{o,g}` are slices of the unchanged E0 classifier. `W_h` has hidden width
`action_mlp_ratio * d_model`. Share `W_h` across groups and offsets. Each `W_2` maps that shared
hidden state directly to one group's logits.

Initialize every residual output projection `W_2` and its bias to zero. Do not zero the action
classifier. At initialization, the logit residual is zero, so E1 must produce exactly the same
logits as a same-seed E0 model for every input.

Create the trunk and E0 classifier projections before the optional MLP modules. This preserves the
same-seed E0 parameters. A test must compare the shared state dictionaries, not only the final
logits.

There is no `W_c` in E1. E1 has no action condition `c`, so a conditioning projection would either
be unused or add another treatment. E2 adds `W_c c` to the shared state preactivation before SiLU.
It is not a separate logit bypass.

All new head parameters use the existing output-head AdamW path. Do not route them into the trunk's
Muon group.

The current optimizer identifies no-decay head parameters only through `model.heads`. If the shared
state projection or residual modules live beside that container, extend the explicit head-parameter
set to include them. Do not rely on a name prefix. Test both values of `head_weight_decay` so an
adapter cannot silently receive the wrong decay rule.

Gradient diagnostics must select the shared input encoder and transformer explicitly. They must not
define the representation as “every parameter outside `model.heads`,” because that would silently
include the new adapter and later value or critic modules. The shared selector now names
`cat_embeds`, `char_emb`, `stage_emb`, `ctx_proj`, and `trunk` directly.

## Parameter matching to E2

Use the same `W_h`, hidden width, and group-and-offset `W_2` projections in E1 and E2. The ratio is
2.

At `d_model = 256`, E1 adds 860,044 parameters:

- Shared `Linear(256, 512)` with bias: 131,584 parameters.
- Group logit residuals at one offset: `512 * 355 + 355 = 182,115` parameters.
- Four offsets: 728,460 group logit residual parameters.

The old 16-MLP plan added 4,206,592 parameters and about 550 billion forward MACs per training step.
It repeated the same state projection for every group and offset, so it was rejected before
implementation. E2 will be slightly larger because it also needs action embeddings and shared
group-conditioning projections. Do not add unused tensors to E1 to claim an exact parameter match.
Report the exact total, trunk, classifier, state projection, logit-residual, embedding, and
conditioning parameter counts for both arms. The revised E1 adapter adds about 112.5 billion
forward MACs per 131,072-frame step.

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

Copy P0 exactly. Do not copy P1 geometry. Only these fields may differ:

- `head_mode=state_mlp`
- `action_mlp_ratio=2`
- Run comment and model tag.
- Head-to-head labels and E0 reference run.

In particular, keep the selected E0 transformer, context, batch, attention, optimizer, step count,
seed, compact data, replay sampler, action vocabulary, validation, decode, and evaluation settings.
The expected data path is `data/processed/ranked-anonymized-1/mds-policy-v7`, with four windows per
replay and a 4,096-replay reservoir. Keep 131,072 frames per optimizer step and use
`require_flex=True` on Vast.

The inherited settings are:

- Model: `d_model=256`, `n_layers=8`, `n_heads=4`, `L_ctx=256`, `action_vocab=1024`, full causal
  attention, and full-window recomputation.
- Objective: offsets `(1, 5, 9, 13)`, normalized auxiliary weight `1.0`, and transition weight
  `1.0`.
- Optimization: batch size `512`, one accumulation step, Muon learning rate `0.02`, AdamW learning
  rate `8.5e-4`, weight decay `0.01`, head weight decay enabled, 500 warmup steps, and 16,384 total
  steps.
- Data: v7 compact policy data, `windows_per_replay=4`, `reservoir_capacity=4096`,
  `shuffle_block_size=2000`, `predownload=512`, `cache_limit_gb=128`, `num_workers=16`, and
  `prefetch_factor=2`.
- Validation: every 1,024 steps on the fixed 1,192-example split.
- Closed loop: periodic 32-match sweeps, final 96-match sweep, seed `0`, 7,200 frames per boot,
  FP16 decode, execution horizon `1`, and full recomputation.
- Training seed: `0`.

P0's saved configuration predates `val_n_samples` and stores `val_n_batches=32`. The current loader
correctly drops that stale host-era field and uses `val_n_samples=1192`. This preserves the frozen
v7 split. It is not an allowed treatment change.

Use exactly 32 concurrent Dolphin boots for periodic and final CPU sweeps. The saved
`eval_max_parallel=32` field controls the background, final, and H2H evaluators. P0 used 32 boots.
Host CPU count cannot change this protocol.

## Planned launch

The launch command is:

```bash
p0_run=260807-164825_023_mtp_heads_gpt-d256-L8-h4-Lc256-a1024-full-recompute-o1.5.9.13-linear_ranked-anon-1_e0-normalized-aux-bc
p0_sha=5d12d010fa3acd1ec07bd86a8e85d2cbb84c584a77b9b79e90dc6fcf03c32e4b

uv run scripts/launch_vast.py \
  --max-price 1.1 --disk 250 --min-vram 24 --min-ram 200 \
  --min-dlperf 120 --min-compute-cap 890 --max-compute-cap 890 \
  --data-gb 13.34 --upload-gb 2 --run-hours 3.5 -- \
  uv run experiments/023_mtp_heads.py \
    --cfg.head-mode state_mlp \
    --cfg.action-mlp-ratio 2 \
    --cfg.require-flex \
    --cfg.eval-max-parallel 32 \
    --cfg.final-h2h-reference-run "$p0_run" \
    --cfg.final-h2h-reference-sha256 "$p0_sha" \
    --cfg.final-h2h-reference-experiment experiments/023_mtp_heads.py \
    --cfg.final-h2h-reference-label 023-e0 \
    --cfg.final-h2h-self-label 023-e1-state-mlp \
    --cfg.final-h2h-n-configs 64 \
    --comment e1-state-mlp
```

The 250 GB disk leaves room for the 128 GB cache limit, the 13.34 GB compact dataset, compile
files, the image, the ISO, checkpoints, and replays. Do not reduce it from the audited command.

The current code default has `require_flex=False`, while P0 used `True`. The launch therefore passes
`--cfg.require-flex` explicitly. After old configuration fields are normalized, the launch differs
from P0 only in `head_mode`, the H2H reference run, and the H2H self label. The MLP ratio, fixed
evaluation concurrency, H2H schedule, reference experiment, and reference label already equal their
planned values, but the command keeps them explicit for auditability.

E1 downloads the P0 checkpoint before training and verifies its SHA-256. A missing or different
checkpoint stops the run before model compilation or data loading.

## Required tests

1. Construct same-seed E0 and E1 models. Assert exact equality for the trunk, input projection,
   embeddings, and all four action classifiers.
2. Run both models on the same batch in evaluation mode. Assert exact equality for every offset and
   group logit.
3. Use the same decode seed and context. Assert that E0 and initial E1 sample the same action.
4. Assert that every residual output weight and bias is zero. Assert that the shared state
   projection weights are not all zero.
5. Backpropagate the real BC objective. At initialization, assert a finite, nonzero gradient on each
   residual output projection. A zero shared-state gradient at this point is expected because the
   zero output projection blocks it.
6. Take one optimizer step, then backpropagate a new batch. Assert a finite, nonzero shared-state
   gradient. This proves the branch is not permanently dead.
7. Assert that E1 logits change after its residual branch learns.
8. Assert exact E0/E1 objective equality when given equal logits and targets.
9. Assert all new parameters appear once in the AdamW partition and never in the Muon partition.
   Check that `head_weight_decay=False` also places every new head weight in the no-decay group.
10. Load an E0 checkpoint after the new mode field exists. It must resolve to `linear` and reproduce
    the E0 policy.
11. Save and reload E1. Assert mode, MLP ratio, logits, and decode settings round-trip.
12. Run the existing small end-to-end train test in `state_mlp` mode. It must save `final.pt` and
    must not log AWR, rank, value, or critic fields.

Run focused tests, Ruff, Python compilation, and `git diff --check` before launch.

## Offline records

Use the same frozen validation split as E0. Log:

The pinned v7 split has 1,192 samples. Cache exactly those 1,192 samples, independent of training
batch size.

- Total and per-group offset-1 NLL.
- Per-group argmax accuracy and exact four-group frame accuracy.
- Hold and transition NLL and accuracy for every group.
- Predicted transition rate, persistence, and change-event F1.
- NLL at offsets 1, 5, 9, and 13.
- Per-offset shared-trunk gradient norms.
- Primary-to-auxiliary gradient cosine, norm ratio, and sign conflict.
- Residual-logit norm relative to base-logit norm, by group and offset.
- Shared-state and residual-output gradient norms.
- Total, trunk, classifier, state-projection, and residual-output parameter counts.
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
- E1-minus-E0 differences between the separately estimated CPU rates.
- Crash rate and completed-match count.

The CPU protocol does not report game win rate. Do not infer a win from a stock lead at the frame
limit.

After the CPU evaluation, run head-to-head against the final E0 checkpoint:

- Set `final_h2h_reference_run` to the final E0 run name.
- Use `experiments/023_mtp_heads.py` for both policies.
- Use self label `023-e1-state-mlp` and reference label `023-e0`.
- Run 64 mirrored configurations, two games per configuration, for 128 games total.
- Keep `eval_max_frames=7200` and `eval_seed=0`.
- Report non-tied stock-lead rate, per-game stock difference, paired per-configuration stock difference,
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

Use the same RTX 4090 host class as E0, at least 200 GB of system RAM, and a 250 GB disk. Record the
storage cost and compare loader wait with E0.

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
- Periodic evaluation worker logs and result JSON files. The in-process final sweep instead retains
  `replays/final/metrics.json` with its rows and replays.

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

- Clear H2H and CPU improvement: nonlinear head capacity matters. E2 must beat E1, not only
  E0, to support a factorization claim.
- Better offline metrics but no closed-loop gain: capacity improves fitting but not control. Keep
  E1 as the required E2 control.
- Closed-loop regression: the MLP is harmful at this scale or optimization setting. Do not tune E1
  after seeing E2. E2 must beat both E0 and E1 without an unacceptable latency cost.
- Conflicting CPU and H2H results: call the result inconclusive. Inspect rows and replays, and
  run more predeclared seeds before promotion.

Do not promote E1 as the new policy baseline from NLL alone. E2 planning may begin while E1 trains,
but E2 may not launch until E1 evidence is complete.

## Results

Local implementation is complete on `exp/e1-output-head-capacity`.

- `StateMLPAdapter.state_proj` is the shared `256 -> 512` projection.
- `StateMLPAdapter.residual_projs` contains one zero-initialized projection for each offset and
  action group.
- Training computes the shared state feature once for all offsets.
- Multi-offset decode also computes the shared state feature once, then applies only the requested
  output heads.
- Training, validation, full-window decode, chunk decode, and KV parity use the same model-level
  logit path.
- One sampler handles support masks, temperature, min-p, trigger repair, and random generators for
  both head modes.
- The adapter adds 131,584 state-projection parameters and 728,460 residual-output parameters. The
  total increase is 860,044 parameters, from 6,818,482 to 7,678,526.
- Validation records the residual-to-base logit RMS ratio for every group and offset. Gradient
  diagnostics record state-projection and residual-output norms.
- W&B records exact parameter groups, token throughput, and peak allocated and reserved GPU memory.
- Old host-scaled evaluation settings map to the fixed 32-boot limit.

Correctness evidence:

- Same-seed E0 and E1 trunks, input modules, and classifiers are exactly equal.
- Initial logits, objectives, and sampled actions are exactly equal.
- Every residual output starts at zero and receives a finite nonzero gradient on the first update.
- The shared state projection has zero gradient on the first update, as expected, and a finite
  nonzero gradient on the second update.
- Both head-weight-decay modes place all adapter parameters in AdamW exactly once and never in
  Muon.
- A saved E1 state round-trips exactly.
- The real P0 `final.pt` loads as linear with 6,818,482 parameters, no adapter,
  `eval_max_parallel=32`, and step 16,384.
- E1 checks the downloaded P0 checkpoint against the audited SHA-256 before compilation or data
  loading. Tests cover the matching, mismatching, malformed, and missing-run cases.
- Focused experiment tests: 60 passed in 8.45 seconds.
- Full repository suite: 916 passed in 134.73 seconds.
- Ruff, the type error gate, Python compilation, and `git diff --check` passed. The type checker
  still reports existing warnings but no errors.
- The exact launch command passed a no-rent audit at pushed commit `a3c6b4b`. It kept the 250 GB
  disk request, encoded the complete training command, and found three qualifying RTX 4090 offers.
  Their effective prices were $0.781, $0.808, and $0.871 per hour.
- After P2 closed, the complete command passed another no-rent audit at pushed commit `9987ed5`.
  It kept the 250 GB disk and every declared E1 and P0-reference field. Four RTX 4090 offers
  qualified; the best effective price was $0.755 per hour. Commit this record, then repeat the audit
  and launch from the unchanged final SHA.
- The final no-rent audit passed at pushed commit `bba9b87`. E1 launched from that unchanged SHA on
  Vast instance `47136334`. The selected RTX 4090 host has 251 GB RAM, 250 GB disk, DLPerf 125.6,
  and an effective price of $0.755 per hour. It became ready in 43 seconds. W&B run `q3aojgfm` is
  `260808-023741_023_mtp_heads_gpt-d256-L8-h4-Lc256-a1024-full-recompute-o1.5.9.13-state-mlp-r2_ranked-anon-1_e1-state-mlp`.
  Startup verified `sm_89`, the exact source SHA, the 250 GB compile cache, the complete command,
  the P0 reference SHA-256, 7,678,526 model parameters, and concurrent validation-cache and training
  prefetch startup. No other Vast experiment is active.

The first routine check found a healthy run at step 4,350. Warm training steps are usually 0.27 to
0.30 seconds. The largest observed step was 0.72 seconds; it was isolated rather than sustained.
The GPU used 12.0 of 49.1 GiB and was 74% active during the check.

At step 4,096, validation action NLL was 1.109 bits per frame and button log loss was 0.033. The
periodic CPU sweep finished in 166 seconds. All 32 boots completed without a crash and produced 41
active matches plus one zero-active tail. Stocks taken and lost per active minute were 0.639 and
1.534. This early closed-loop point is weak and is not a final decision. R2 received the checkpoint,
44 replays, rows, metrics, result, and worker log.

Training continues. Final validation, CPU evaluation, H2H, artifact verification, cost, and the
decision are pending.
