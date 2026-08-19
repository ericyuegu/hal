# 035 zero-state recursive GRU chunk decoder

Status: preregistered, implemented, and rejected at the matched RTX 3060 smoke gate on 2026-08-19. Production was
not launched because primary validation NLL regressed clearly.

## Question and single intervention

Can a small recurrent plan state improve coherent closed-loop action chunks relative to 026 without changing its
observation model, action representation, autoregressive conditioning, objective, data, or evaluation?

Experiment 026 already autoregresses across selected future offsets. Each token contains the normalized 384-wide
scene state, the previous full controller-frame embedding (four groups times 16 dimensions), and a learned 16-wide
**absolute offset** embedding, projected by the existing randomly initialized linear layer to width 128. Experiment
035 changes only the mixer after that projection: the two width-128 causal temporal Transformer blocks become one
`torch.nn.GRUCell(128, 128)`. This is a mixer swap, not a new autoregressive scheme.

For every independent context prefix or live decode, initialize `h_0 = 0`. For selected offset `k`, let `x_k` be the
unchanged projected scene/action/absolute-offset token. The exact PyTorch convention is

\[
r_k=\sigma(W_{ir}x_k+b_{ir}+W_{hr}h_{k-1}+b_{hr}),
\]
\[
z_k=\sigma(W_{iz}x_k+b_{iz}+W_{hz}h_{k-1}+b_{hz}),
\]
\[
n_k=\tanh(W_{in}x_k+b_{in}+r_k\odot(W_{hn}h_{k-1}+b_{hn})),
\qquad h_k=(1-z_k)\odot n_k+z_k\odot h_{k-1}.
\]

This deliberately uses PyTorch's candidate equation, including reset after the recurrent affine transform, rather
than silently substituting the original Cho formulation. See the
[PyTorch GRU documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html) and
[Cho et al. (2014)](https://arxiv.org/abs/1406.1078).

The raw recurrent state is carried between offsets. `RMSNorm(h_k)` is used only by the unchanged output heads. The
same statically unrolled cell is used for full teacher forcing, forced-stepwise diagnostics, greedy validation
rollout, sampled H4 deployment, and sampled H6 diagnostics. Every decode starts from fresh zeros; no recurrent state
crosses contexts, slots, or calls.

There is no scene initialization, residual wrapper, custom gate bias or initialization, zero action slice, learned
null control, scheduled sampling, SSM, rank weighting, or other change. In particular, the action projection remains
ordinary random initialization and the normalized scene plus absolute offset remain injected at every step.

## Frozen eligible recipe

- Fresh seed 0; effective batch 512; 16,384 optimizer steps.
- Exact original `ranked-anonymized-1/mds-policy-v7` stream, storage view, replay/window order, and sparse alignment.
- Offsets `(1, 2, 3, 4, 5, 6, 9, 12, 16, 20)` in the existing controller-group order.
- Exact 026 observation trunk, codec, token projection, group conditioning, direct trunk residual heads, nonlinear
  heads, legality mask, optimizer, learning-rate schedule, loss reduction, auxiliary weight, validation cache,
  checkpoints, compilation settings, and RNG/reset semantics.
- H4 is the primary deployment contract. H6 is diagnostic only.
- Final evaluation is the ordinary fixed 96-bootstrap H4 protocol plus the fixed 32-bootstrap H6 diagnostic.
- Experiment identity is `035_recursive_gru_v1`. Checkpoints from 026, 034, or any other identity must be rejected.

`GRUCell(128,128)` has exactly 99,072 parameters. The preregistered total is approximately 14,889,967 parameters,
a net reduction of 163,072 from 026; tests must assert the exact counts produced by repository code.

## Lightweight diagnostics

The existing validation set reports teacher-forced and greedy-rollout NLL/accuracy at every offset, a scalar named
`exposure_gap`, dense-frame/chunk exactness, and hold/change transition metrics. These are coarse health diagnostics,
not causal exposure-bias isolation. In particular, the inherited `exposure_gap` subtracts teacher-forced NLL pooled
over every valid prefix from rollout NLL measured only on each example's last prefix; rollout also mixes temporal
feedback with within-frame greedy ancestors. It must not support a mechanistic claim.

Checkpoint analysis may add two lightweight, validation-only views without changing training: compare teacher and
rollout NLL on the same last-prefix rows, and separately hold within-frame ancestors to target values while changing
only temporal feedback. Cross-offset error/disagreement from those matched rows is descriptive. No new training
signal, sampled dataset, or heavyweight logging path is part of 035.

## Mechanistic expectation and limits of the evidence

A GRU can preserve or revise a compact latent plan while consuming the same scene/action/offset evidence at each
step. It may improve error structure or cross-offset coordination, but this is an inference to test, not an
established property of GRUs in Melee.

The August 2026 study
[Why Does Action Chunking Improve Behavioral Cloning Performance in Robotic Control?](https://arxiv.org/abs/2608.02547)
directly challenges the usual smoothness story: its experiments do not support temporal consistency, horizon
reduction, or representation learning as the general explanation. The paper instead attributes benefits to greater
non-Markov conditioning, reduced compounding error that delayed policies can sometimes capture, and implicit
ensembling over multiple observation-to-action delays. Therefore 035 is not justified as a smoothness mechanism.
Its recurrent mixer may help exploit the already-present delayed-offset relationships, but it does not explicitly
implement the paper's randomized-delay ensemble and should not be described as doing so.

Related action-chunk work is only directional context: [ACT](https://arxiv.org/abs/2304.13705),
[Diffusion Policy](https://arxiv.org/abs/2303.04137), and
[Chunking Causal Transformer](https://arxiv.org/abs/2410.03132) support predicting action sequences in other control
domains, but do not establish that a GRU is better here.

## Deferred failure-contingent experiments

Modern minGRU and minLSTM ([Feng et al., 2024](https://arxiv.org/abs/2410.01201)), RG-LRU
([De et al., 2024](https://arxiv.org/abs/2402.19427)), LSTM/xLSTM
([Beck et al., 2024](https://arxiv.org/abs/2405.04517)), Mamba
([Gu and Dao, 2023](https://arxiv.org/abs/2312.00752)), Gated DeltaNet
([Yang et al., 2024](https://arxiv.org/abs/2412.06464)), and Tiny Recursive Models
([Jolicoeur-Martineau, 2025](https://arxiv.org/abs/2510.04871)) are separately named, failure-contingent experiments.
Their parallel scans, expanded memories, fast-weight states, or iterative refinement would confound this first clean
test and are not part of 035.

Exposure-bias treatments are also deferred. Standard scheduled sampling changes the objective and RNG path and is
not a proper likelihood ([Huszar, 2015](https://arxiv.org/abs/1511.05101)). If 035 improves teacher-forced NLL while
the own-sample exposure gap worsens materially with depth, a later preregistration may compare a principled
on-policy or dataset-aggregation treatment rather than folding it into this architecture ablation.

## Gates and stop conditions

Before cloud launch:

1. Assert cell, temporal, and total parameter counts; exact teacher-forced versus forced-stepwise parity; target
   temporal causality; fresh-state isolation; legality, RNG, and reset behavior; H4/H6 eager and compiled parity;
   checkpoint rejection/roundtrip; optimizer ownership; and strict production/smoke config isolation.
2. Run formatting, Ruff, ty, Python compilation, focused experiment tests, and proportionate full tests.
3. Run a 100-step effective-batch-512 RTX 3060 smoke and H4/H6 closed-loop evaluation on its checkpoint.
4. Kill before production for graph breaks, non-finite loss/gradients, compiled p95 latency above 1.25 times matched
   same-host 026, a clear primary validation-NLL regression, or projected L40S runtime above eight hours.

If all gates pass, launch exactly one production run on Modal L40S and babysit it through terminal validation and
ordinary final evaluation. Record the source SHA, Modal app/function, W&B run, checkpoint SHA-256, throughput,
validation metrics, H4/H6 evaluation, and final ordinary `net_stock_lcb`. The promotion target is strictly above the
eligible 026 value `1.031887949`; negative evidence remains useful only if the intervention and controls stay clean.

## Smoke evidence and decision

Both treatments were trained fresh on the same RTX 3060 with seed 0, the same replay stream/order, effective batch
512 split into eight 64-example microbatches, 100 optimizer steps, production trunk/temporal compilation, and the
same 128-example validation cache geometry. The 035 offline W&B ID is `fn6eu9qs`; its run directory is
`260819-110729_035_recursive_gru_mtp026-d384-L8-h6-Lc128-t128x2-o1-2-3-4-5-6-9-12-16-20-s4-base-gru128_ranked-anon-1_035-smoke-rtx3060`.
Its final checkpoint SHA-256 is `e4d057ded511c3b1f4db9903ab21e8c09002b0f0e2ac8b51bab443796e8c2058`.
The fresh matched 026 control has offline W&B ID `67pj9axs` and run directory
`260819-111359_026_temporal_mtp_mtp026-d384-L8-h6-Lc128-t128x2-o1-2-3-4-5-6-9-12-16-20-s4-base_ranked-anon-1_026-matched-smoke-rtx3060`.

| Metric | 035 GRU | matched 026 | Difference/ratio |
| --- | ---: | ---: | ---: |
| step-99 train NLL, bits | 7.927566 | 7.503688 | +0.423878 |
| final validation NLL, bits | 7.618324 | 7.209772 | **+0.408552** |
| steady-state samples/s, steps 10--99 | 822.76 | 536.69 | 1.533x |
| peak allocated GiB | 2.069 | 2.486 | -0.417 |
| peak reserved GiB | 2.174 | 2.908 | -0.734 |

The 035 H4 and H6 smoke evaluations each completed two 1,800-frame boots with zero crashes. Sampled compiled decode
p95 was 7.70 ms at H4 and 11.22 ms at H6, versus 9.19 ms and 11.75 ms for the same 026 architecture in the preceding
matched-host 034 smoke. Deterministic eager/compiled argmax outputs matched at H4 and H6. Isolated compilation of the
new sampled GRU decoders produced zero graph breaks. The full inference counter also reports four breaks from the
inherited trunk's intentional `torch.compiler.disable`-wrapped attention-path resolver; this is baseline behavior,
not a new decoder break.

Formatting, Ruff, Python compilation, the 10 focused 035 tests, and all 25 experiment-026 coverage tests passed. `ty`
reported 55 unresolved-reference/attribute diagnostics caused by the dynamic 026-module wrapper; the structurally
identical 034 wrapper reports 78 of the same class. There were no semantic/type diagnostics beyond dynamically
re-exported 026 names.

The final validation regression is large and agrees with the worse step-99 training NLL. It triggers the explicitly
preregistered "clear primary NLL regression" stop condition. The faster, smaller recurrence is useful systems
evidence, but does not justify an eight-hour production run. Experiment 035 is therefore rejected before Modal; no
production app/function, online W&B run, or final 96-bootstrap score exists.

### Read-only checkpoint diagnosis

`scripts/diagnose_recursive_gru.py` compares the two smoke checkpoints on the same cached validation rows and
inspects the trained GRU without modifying either checkpoint. On 128 rows, the 035-minus-026 joint-NLL deltas were
positive at every offset: `+0.103, +0.154, +0.192, +0.251, +0.291, +0.254, +0.151, +0.252, +0.215, +0.238` bits for
offsets 1, 2, 3, 4, 5, 6, 9, 12, 16, and 20. Most of the deficit came from main-stick and button prediction, rather
than appearing only in the sparse tail.

The cell was not gate-saturated: reset/update means were `0.5043/0.4883`, with zero measured elements below `0.01`
or above `0.99`. Raw-state RMS rose from `0.066` at offset 1 to `0.126` at offset 6 and ended at `0.133`; mean relative
update norm after the undefined zero-state first step ranged from `0.37` to `0.67`. Effective rank stayed about
24--27 of 128 dimensions and covariance participation ratio about 13--15. The prior-action channel was not ignored:
its token-projection gradient norm was `1.240`, or `33.18%` of the full projection-gradient norm, and the action class
embeddings had gradient norm `0.0648` on the diagnostic batch.

These measurements argue against dead gates or a disconnected autoregressive action path. The broad NLL deficit and
low-dimensional state instead suggest an underfit recurrent representation or fixed-step optimization disadvantage.
They do not justify stacking a second GRU as the immediate follow-up. Any residual/initialization or optimization
rescue must be a separately preregistered single-variable experiment; it cannot retroactively rescue 035.
