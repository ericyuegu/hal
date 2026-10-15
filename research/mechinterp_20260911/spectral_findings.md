# Weight geometry across the six historical checkpoints

The large Muon weight norms are real. A collapse of matrix rank, or a norm
threshold that identifies the gameplay slump, is not supported by these saved
checkpoints.

The audit computed full singular-value spectra for 189 matrix views per
checkpoint on an H100. Fused QKV weights and their logical Q, K, and V views are
recorded separately; summaries never add them together. Every model export
passed its recorded size and SHA-256 check. The source checkpoints were not
changed. See [measurements](artifacts/spectra/historical_spectra.json),
[group summaries](artifacts/spectra/summary.json), and the
[figure](artifacts/spectra/matrix_geometry.png).

## Main observations

Values below are medians over the 16 trunk layers, not confidence intervals.

| Checkpoint | Q Frobenius norm | Q stable rank | K Frobenius norm | Net stocks/min in the historical evaluation |
| --- | ---: | ---: | ---: | ---: |
| Muon 24,576 | 261.2 | 152.1 | 262.7 | +0.164 |
| Muon 57,344 | 357.9 | 165.6 | 360.6 | -0.175 |
| Muon 98,304 | 376.8 | 161.6 | 380.3 | +0.138 |
| Muon 131,072 | 376.4 | 160.4 | 379.9 | +0.345 |
| AdamW 57,344 | 55.6 | 44.8 | 53.8 | +0.387 |
| AdamW 131,072 | 59.6 | 45.4 | 58.3 | +0.494 |

Stable rank is `||W||_F² / ||W||_op²`. It measures spectral concentration, not
the number of learned features. The shared geometry permits this descriptive
comparison, but the two runs have different optimizers and learning rates.
Larger stable rank is not evidence of better features or control.

From Muon 24,576 to 57,344, the median *paired* norm ratios are 1.37 for Q,
1.38 for K, 1.50 for V, 1.46 for the attention output, 1.45 for the MLP input,
and 1.41 for the MLP output. Q and K change direction by about 48 degrees.
Final action readouts change much less: their median norm ratio is 1.08 and
angle is about 17 degrees. A swap of early readouts therefore tests a specific
compatibility hypothesis; it does not simply undo the largest parameter change.

From 98,304 to 131,072, median Q and K norms change by approximately -0.1%.
Their directions still move by about four degrees. Gameplay improves over this
interval without reversing the earlier scale growth. This rules out a simple
story in which recovery requires those norms to return to their early values.
It does not rule out a role for scale in the onset of instability.

## Two controls prompted by these measurements

These were added before the intervention results were available:

1. Restore only each Q and K matrix's early Frobenius norm, preserving its
   current direction and leaving V, output projections, MLPs, and readouts
   unchanged. Neither the trunk nor temporal decoder uses QK normalization.
   This tests the effect of changing attention-score scale more directly than
   restoring every Muon matrix at once.
2. Multiply each Q matrix by two and each K matrix by one half. This should
   leave every attention score, and thus the policy, unchanged. The scalar
   factors commute with RoPE and cancel in `Q Kᵀ`. Numerical output parity is
   required; the algebra alone is not the implementation check.

The second control is also a concrete reason not to diagnose a bad basin from
one matrix norm or an unqualified Hessian condition number. Equivalent
parameterizations can have different radii and Euclidean curvature. This is
established theory, not a new claim about optimization
([Dinh et al., 2017](https://proceedings.mlr.press/v70/dinh17b.html)).

## Limits

- Saved-checkpoint angles are net displacements, not per-update angular
  learning rates or measurements of momentum autocorrelation.
- The audit is on weights, not data-conditioned activations. Attention entropy
  and causal output changes are separate tests.
- No rank collapse at these four Muon snapshots does not exclude a transient
  between snapshots or a collapse restricted to a behavioral feature.
- The model uses residual paths, RMS normalization, nonlinear heads, mixed
  optimizers, and a changing learning rate. Fixed-radius, scale-invariant
  idealizations do not establish that all its Frobenius norms must be fixed.

Reproduce the local summaries with:

```sh
python research/mechinterp_20260911/spectral_findings.py
```

`remote_spectra.py` records the exact mounted HAL source hashes and numerical
environment used for the already-completed remote weight audit.
