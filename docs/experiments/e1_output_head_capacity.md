# E1: output-head capacity control

Status: blocked on the P0/P1/P2 package decision

The final E0 reference and measured wall time are pending. Finish P0, matched P1, and the planned P2
decode comparison before choosing that reference. Do not implement or launch E1 until the selected
reference reaches step 16,384, uploads `final.pt`, and completes its final CPU and H2H evaluation.

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

Copy the selected E0 configuration exactly. Do not copy the old P1 geometry from an exploratory
checkpoint. Only these fields may differ:

- `head_mode=state_mlp`
- `action_mlp_ratio=2`
- Run comment and model tag.
- Head-to-head labels and E0 reference run.

In particular, keep the selected E0 transformer, context, batch, attention, optimizer, step count,
seed, compact data, replay sampler, action vocabulary, validation, decode, and evaluation settings.
The expected data path is `data/processed/ranked-anonymized-1/mds-policy-v7`, with four windows per
replay and a 4,096-replay reservoir. Keep 131,072 frames per optimizer step and use
`require_flex=True` on Vast.

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

Pending E0 completion.
