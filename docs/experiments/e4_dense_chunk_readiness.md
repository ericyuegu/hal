# E4: dense chunk readiness

Status: blocked on E3 evidence

Updated: 2026-08-07

## Question

Can the selected temporal model produce a coherent dense four-frame action sequence without harming
the one-frame closed-loop policy?

E4 changes sparse offsets `(1,5,9,13)` to dense offsets `(1,2,3,4)`. It still executes only offset
1. This is a bridge to chunk critics and chunk execution. It is not expected to improve policy
quality.

Earlier dense MTP work was slightly worse in closed loop. Keep that result as prior evidence, but do
not treat it as a valid matched control unless its target alignment and loss scale pass the current
tests.

## Files to change

- Prefer a configuration-only change in `experiments/023_mtp_heads.py`.
- Add tests to `tests/experiments/test_023_mtp_heads.py` only if dense offsets expose a missing
  invariant or metric.
- Update this file with the exact command, reference, run IDs, timing, metrics, artifacts, and
  decision.

Do not create a new experiment file only to change offsets. Do not change the transformer, temporal
module, within-frame factorization, loader, optimizer, or evaluator.

## Probability model

E4 models:

\[
p(a_{t+1:t+4}\mid s_t)
=\prod_{k=1}^{4}p(a_{t+k}\mid s_t,a_{t+1:t+k-1}).
\]

Each frame action uses the selected E2 within-frame group factorization. Training uses the true
previous frame action. Free-running validation uses the model's previous sampled action.

The offset-1 head remains the deployed policy and must not depend on future targets or temporal
modules. The temporal chain begins only after offset 1.

## Objective

Keep one fixed auxiliary weight:

\[
L=L_1+\frac{L_2+L_3+L_4}{3}.
\]

Do not copy the old objective that gave every auxiliary head weight 1.0. Keep AWR, rank weighting,
value loss, and critic training off.

The primary loss is directly comparable with E3. Auxiliary NLL is not directly comparable because
E3 predicts different time offsets.

## Required tests

1. Use a synthetic action sequence with a unique class at every frame. Assert exact targets at
   offsets 1, 2, 3, and 4.
2. Assert a context ending at frame `t` never reads action `t+1` as an input feature.
3. Assert offset 2 receives the true offset-1 action during teacher forcing, offset 3 receives the
   true offset-2 action, and offset 4 receives the true offset-3 action.
4. Assert changing a later target cannot change an earlier logit.
5. Assert padding and episode boundaries invalidate every target and condition that crosses them.
6. Assert the loss is `L1 + mean(L2,L3,L4)` over valid entries.
7. Assert closed-loop execution requests offset 1 only.
8. Assert requesting a four-frame execution horizon remains a separate explicit setting.
9. Save and reload dense offsets, temporal mode, group order, and decode settings.
10. Run the small end-to-end train test with dense offsets and finite losses.

Run focused tests, Ruff, type checking, Python compilation, and `git diff --check` before launch.

## Fixed configuration

Copy the selected E3 configuration exactly. Change only:

- `head_offsets=(1,2,3,4)`
- Run and H2H labels.

Keep execution horizon 1, 16,384 steps, seed 0, 131,072 frames per step, attention package,
within-frame order, temporal MLP, normalized auxiliary weight, compact data, replay sampler,
optimizer, schedule, decode temperature, and evaluation protocol fixed.

Use the same RTX 4090 class, at least 200 GB system RAM, and 250 GB disk. Target 3.0 to 3.5 hours
through evaluation and upload.

## Diagnostics

Dense controller streams contain many holds. Report metrics that cannot be won by copying the last
action:

- Teacher-forced and free-running NLL by frame offset and group.
- Exact full-action accuracy by offset.
- Hold and transition accuracy by offset and group.
- Change-event precision, recall, and F1.
- Free-running four-frame sequence exact match.
- Action-run length and transition rate in targets and samples.
- Performance of a copy-the-previous-action baseline on the same validation rows.
- Exposure gap from teacher forcing to free running.

Also report primary NLL, trunk-gradient interaction, parameter counts, training time, loader wait,
memory, and inference time for sampling all four planned frames. Sampling all four is a diagnostic;
the live CPU evaluator still executes one.

## Closed-loop evaluation

Run the standard 32-boot periodic and 96-boot final CPU protocol with execution horizon 1. Run 64
mirrored H2H configurations against E3. Save all match rows and replays.

Report stocks, damage, dead frames, terminal results, crashes, paired CPU deltas, H2H stock
difference, non-tied win rate, confidence intervals, and ties. A closed-loop regression is allowed
as a scientific result, but it must be explicit.

Flag startup over 30 minutes, warm steps over 0.5 seconds, a slowdown over 25% from E3, a periodic
evaluation over 25 minutes, or a projected total over 3.5 hours.

## Decision

E4 passes the chunk-readiness gate if:

- Target alignment and causality tests pass.
- Free-running predictions beat the copy baseline on transition events.
- Four-frame samples are finite and structurally valid.
- The final checkpoint and complete evaluation evidence exist.

E4 does not need to beat E3 in closed loop to proceed to critic validation. If it regresses badly or
cannot predict transitions beyond copying holds, do not execute its chunks. Diagnose temporal
exposure and representation first.

Do not apply a chunk advantage in E4. E6 must validate a chunk-conditioned critic before E7 changes
the actor or execution horizon.

## Results

Pending E3 completion.
