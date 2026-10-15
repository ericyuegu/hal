# O50 checkpoint interventions

This probe compares the Muon run `p1fyyp1z` at boundaries 24,576 and 57,344.
It uses the fixed 256-replay normalized cohort prepared by the O50 probe. The
cohort, model-export hashes, sidecar hashes, and role commit are recorded in
the completed `artifacts/interventions/results.json`. A byte-for-byte
protocol rerun is preserved separately in
`artifacts/interventions_fp32/results.json`.

The transform layer is tested before any model evaluation. It preserves all
non-parameter buffers, rejects schema or dtype mismatches, and uses the exact
optimizer roles from commit
`cdf84f904d322d2a8c342fe3ea75d926e10346ba`. Fused QKV matrices are split into
logical Q/K/V blocks; `value_head.up.weight` is split into its two logical
blocks.

The O50 module registers its codec below both `codec` and `temporal.codec`.
The loader treats these eight state-dict aliases as parameter slots, maps
their roles to the canonical `codec.*` names, and checks their values as part
of each intervention.

The fixed cohort's reference bundle is a smaller d256 O50 model. The loader
uses it only for representation identity: context length, prediction heads,
codec fields, statistics, schema, and player vocabulary. It then builds the
historical d1024 model from the checkpoint config. A width mismatch is not a
cohort mismatch.

The planned evaluation reports NLL in natural-log nats by hold/change status,
paired replay-bootstrap intervals, output Jensen–Shannon divergence, argmax
change rate, and the largest teacher-forced logit change by controller group.
Each unique transformed state is evaluated once. The reversal and identity
controls are retained, so duplicate states are labeled rather than rerun.
These are four-frame joint teacher-forced diagnostics. They are not a
deployed-tail gameplay score: the frozen named runs sampled four actions and
committed the executed suffix after discarding the first two, while the
portable policy path uses a different pending-prefix convention.

For the early, slump, Q/K norm-restored, and Q×2/K÷2 states, it also traces
the final trunk query. It records per-layer/head entropy, effective context,
lag mass, and the maximum absolute difference between the traced trunk output
and the public dense trunk output. A trace error above `2e-2` aborts that
state. The sequential public greedy decoder is also compared with the full
teacher-forced logits; rare CUDA bfloat16 argmax differences are recorded with
their action count and do not change the four-frame NLL score.

Two small attention controls were added after the spectral audit, not as
post-hoc findings:

1. `restore_attention_qk_norms` restores only logical Q and K Frobenius norms
   from the early checkpoint. V, O/V, MLP, embeddings, and heads remain from
   the slump checkpoint.
2. `qk_gauge_q2_khalf` multiplies fused Q by 2 and K by 1/2 in the slump
   checkpoint. The QK scores should be unchanged because the factors cancel
   before attention and commute with RoPE. The run records output JS, argmax
   changes, and maximum logit deltas as the public-output parity control.

Offline NLL, output changes, and attention are diagnostic only. They are not a
checkpoint or treatment-selection rule and do not support a gameplay or basin
claim. Any gameplay follow-up must be a separate, replay-matched evaluation
with its treatment choice defined by that protocol.

## Fixed-cohort result

The run used the verified `p1fyyp1z` exports at checkpoint steps 24,575 and
57,343, 256 distinct replays, batch size 8, and an RTX 3060. The slump mean
score was 0.185240 nats per controller group and predicted frame. Intervals
below are paired bootstrap 95% intervals over replay means; positive ΔNLL is
worse than the slump baseline.

Provenance: cohort SHA-256
`ec61e97e332672a4e4e2fdc438e208a281a4609e10f7eded72368eccac323437`; early
export SHA-256
`289e80ea88fb0b23e48fbff2e142a92a3abbaff78a80ac3f3a97843189ae3a4c`; slump
export SHA-256
`fda4dd85a1f60c06c3c19e507834071e54cb000b483bdac40315482b69215466`.

| State | ΔNLL (nats) | 95% interval | output JS (nats) | argmax change |
| --- | ---: | ---: | ---: | ---: |
| Early checkpoint | −0.00141 | [−0.00612, 0.00325] | 0.00322 | 1.15% |
| Interpolation α=.25 | −0.00078 | [−0.00548, 0.00387] | 0.00294 | 1.17% |
| Interpolation α=.50 | −0.00109 | [−0.00499, 0.00281] | 0.00214 | 0.81% |
| Interpolation α=.75 | −0.00223 | [−0.00502, 0.00042] | 0.00097 | 0.51% |
| Early readouts restored | +0.00636 | [+0.00292, 0.01020] | 0.00145 | 1.10% |
| Early AdamW partition restored | +0.15950 | [+0.12865, 0.19095] | 0.02233 | 3.00% |
| Early Muon partition restored | +0.06825 | [+0.05529, 0.08191] | 0.01790 | 2.15% |
| Early Muon norms restored | +0.02805 | [+0.02221, 0.03383] | 0.00805 | 1.34% |
| Early Q/K norms only | +0.00097 | [−0.00012, 0.00207] | 0.00017 | 0.39% |
| Q×2, K×½ gauge | +0.00000 | [0.00000, 0.00000] | 6.9×10⁻¹⁰ | 0.00% |

The Q×2/K×½ control preserved every recorded public greedy action and had
zero maximum finite logit difference; the tiny JS value is floating-point
roundoff. Its trunk trace error was zero. This validates the fused Q/K split,
RoPE placement, and output comparison. The early, slump, Q/K-only, and gauge
traces also had zero maximum dense-trunk parity error.

The original intervention trace was evaluated under the model's CUDA autocast
context. Its `.float()` operands did not prevent the `einsum` from being
autocast back to bfloat16. The first byte-for-byte rerun therefore confirms
the NLL and action metrics below, but its attention fields remain legacy
routing diagnostics, not corrected FP32 measurements.

The final corrected FP32 rerun on the same fixed cohort gives the following
values. Effective context is
`mean(exp(per-example/head entropy))`, in context positions; entropy is in
natural-log nats. Mean lag is in context frames, where zero is the current
frame. The six-checkpoint join under `artifacts/attention_fp32/` has summary
SHA-256
`e9ce0f0d6d4f32c88579f4f79a954f63e9c2caaec2a59079201a2f57fbecf2b1`.

| State | Entropy (nats) | Effective context | Mean lag (frames) |
| --- | ---: | ---: | ---: |
| Early checkpoint, corrected FP32 | 0.14994 | 1.24792 | 70.4303 |
| Slump checkpoint, corrected FP32 | 0.04289 | 1.06091 | 65.5998 |
| Early Q/K norms only, corrected FP32 | 0.07281 | 1.10685 | 65.5943 |
| Q×2/K×½ gauge, corrected FP32 | 0.04289 | 1.06091 | 65.5998 |

All four traced states recorded runtime attention dtype `torch.float32` and
zero dense-trunk parity error. The gauge's entropy, effective context, and
lag matrices are exactly equal to the slump identity trace, as expected from
the Q/K score-preserving control.

## Rerun parity and provenance

The unchanged 18-state reruns used the same checkpoints, fixed cohort, seed,
batch size 8, and 256 windows. The first parity-only result is
`artifacts/interventions_fp32/results.json` (SHA-256
`b0869ed07b36e92c9b2c480a5254d83a24e5c90b3ef25a6535c21f7988f28ff2`). The
original result is `artifacts/interventions/results.json` (SHA-256
`82df4fdbb121439c93d8eaa373dc9d7669cd881c5c4880fb25086fce7a69f5b2`). The
parity rerun's source is archived at
`artifacts/interventions_fp32/checkpoint_interventions_rerun_source.tar`
(SHA-256
`a46a8207410bdd87b24e7eabea2b0f779d48958cfc8bcd26ae76cc28cde8ae81`), and
the source file used for that parity rerun has SHA-256
`e524985c423795637999dd1dee90e0136d4c200ab72f896610c6f27d609273a8`.
Its NLL and action fields are retained as an exact parity record; its
attention fields are legacy precision.

The final corrected result is
`artifacts/interventions_fp32_corrected/results.json` (SHA-256
`997949b848aa46d5f9beb571b1c8eebda4b88c8a6b8a07824427fd991a0abdb5`). Its
source archive is
`artifacts/interventions_fp32_corrected/checkpoint_interventions_corrected_source.tar`
(SHA-256
`324fd6b14ee034a7756538d1f9370f1571d71c8535b79fc5e4b536fbf1fd1857`), and
the source file used for this corrected run has SHA-256
`d3112c07e147f31a76eece7df158af2e53032f40e2ee9821988894ea5a9e3ca6`.
All 18 intervention records have exactly equal paired NLL means and bootstrap
intervals, output JS, argmax-change rates, public greedy mismatch counts, and
the gauge parity control between the original result and this corrected run.
The baseline slump NLL is exactly
`0.18523982167243958` nats per controller group and predicted frame in both
files. Each of the four traced states in the corrected result records
`attention_dtype=["torch.float32"]`; this is a runtime assertion, not an
inference from source comments.

The exact corrected launch was:

```text
env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/tmp/hal-mechinterp-20260911:/home/ericgu/src/hal/.venv/lib/python3.14/site-packages /home/ericgu/src/hal/.venv/bin/python research/mechinterp_20260911/checkpoint_interventions.py --early-checkpoint /home/ericgu/src/hal/runs/mechinterp-20260911-scratch/p1fyyp1z-step-0024576.model.pt --slump-checkpoint /home/ericgu/src/hal/runs/mechinterp-20260911-scratch/p1fyyp1z-step-0057344.model.pt --early-metadata /home/ericgu/src/hal/runs/mechinterp-20260911-scratch/p1fyyp1z-step-0024576.model.metadata.json --slump-metadata /home/ericgu/src/hal/runs/mechinterp-20260911-scratch/p1fyyp1z-step-0057344.model.metadata.json --cohort /home/ericgu/src/hal/runs/mechinterp-20260911-scratch/o50-cohort-256.pt --output research/mechinterp_20260911/artifacts/interventions_fp32_corrected/results.json --device cuda --batch-size 8 --windows 256
```

The wrapper did not set TF32 or float32 matmul-precision flags, and the
result does not contain an in-run flag snapshot. A post-run, no-forward query
of the same environment reported PyTorch `2.11.0+cu130`,
`torch.backends.cuda.matmul.allow_tf32=False`, and
`torch.get_float32_matmul_precision()="highest"`; these observations are not
used to infer mantissa precision for the completed run. The supported numeric
claim is the runtime FP32 attention-dtype assertion above.

The pre-Ty source is preserved at
`artifacts/interventions/checkpoint_interventions_pre_tyfix.tar` (SHA-256
`3d13f32ffa34900a5375f9d430f71e693f762877cc34621a6dbfdc524f607f0e`) with
source SHA-256
`6715342166c8454ea7ae30642f1be63ea950e50920f02e71c95008cc313a9276`.

Restoring all AdamW or all Muon parameters moved the score strongly away from
the slump checkpoint. Restoring only Q/K norms produced a small, mostly
uncertain NLL shift and partly moved attention sharpness toward early. This is
evidence about this fixed offline cohort, not a causal decomposition of the
training regression or a gameplay prediction.
