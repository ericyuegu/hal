# 035 recursive chunk decoder research design

Status: researched candidate, not preregistered and not implemented. Do not combine with 034.

## Aim

Test whether a tiny recurrent plan state produces more coherent controller chunks than 026's temporal-attention
decoder while preserving the expensive observation trunk and the real-time H4 execution contract.

Important premise correction: 026 does not use independent temporal-offset heads. It uses a shared two-layer causal
temporal Transformer (`d=128`, four heads, FFN 256) over the ten future-frame tokens. Each token reads the trunk scene
embedding, offset embedding, and previous full controller frame; controller groups are then decoded in the frozen
within-frame order. Candidate 035 replaces only those two temporal attention blocks with a recurrent state update.

## First hypothesis: one standard GRU cell

A state-conditioned gated recurrence is the smallest direct test of the proposed register. The scene embedding stays
available at every step, so the 128-wide register need only track the evolving plan rather than compress the observed
game history.

Let `c = RMSNorm(h_trunk)`, initialize `s_0 = P_0 c`, and at future offset `o_k` form

\[
x_k = P_c c + P_o e(o_k) + P_a e(a_{k-1}) + b.
\]

Use a standard GRU update ([Cho et al., 2014](https://arxiv.org/abs/1406.1078)):

\[
r_k=\sigma(W_r x_k + U_r s_{k-1}+b_r),\qquad
z_k=\sigma(W_z x_k + U_z s_{k-1}+b_z),
\]

\[
n_k=\tanh(W_n x_k + U_n(r_k\odot s_{k-1})+b_n),\qquad
s_k=(1-z_k)\odot s_{k-1}+z_k\odot n_k.
\]

Decode all four groups from `RMSNorm(s_k)` through 026's unchanged direct trunk residual heads, nonlinear shared
heads, FiLM-like within-frame group conditioning, and trigger/button legality mask. During training, `a_{k-1}` is the
ground-truth previous frame. During rollout it is the sampled previous frame. Do not recur once per controller group;
that would simultaneously change temporal recurrence and within-frame factorization.

Suggested initialization: Xavier input matrices, orthogonal recurrent matrices per gate, zero candidate/reset bias,
and update bias `-1`. Initialize `P_a` to zero. A parameter-matched null-action control permanently feeds one learned
null vector; it is function-identical to the action-feedback treatment at initialization. This isolates information
from the generated previous action from recurrence and capacity alone.

## Frozen experiment contract

- Exact 026 observation trunk, data/order, batch 512, seed 0, optimizer/schedule, 16,384 steps, validation cache, and
  H4 deployment; H6 remains a diagnostic.
- Exact offsets `(1, 2, 3, 4, 5, 6, 9, 12, 16, 20)` and within-frame group order.
- Full teacher-forced conditional maximum likelihood in the first experiment.
- Treatment and null-action control start from paired initialization and see paired batches.
- Report parameter count and same-host eager/compiled latency against 026. The GRU is estimated at about 99k cell
  parameters and under 1 MFLOP for H4, negligible beside the observation trunk, but kernel-launch cost must be measured.

Do not add scheduled sampling to the first run. It would change the objective and RNG stream; moreover, the standard
scheduled-sampling objective is not a proper likelihood. If teacher-forced metrics improve while the own-sample gap
grows sharply with depth, a separately named follow-up may test stop-gradient sampled histories.

## Required evidence

Before training:

1. Batched teacher-forced logits equal forced stepwise logits.
2. Changing target frame `j` cannot change logits before `j`.
3. Changing the prior frame changes later logits but not current-frame logits.
4. Treatment and null-action logits are identical at initialization; only treatment receives a useful `P_a` gradient.
5. Within-frame causality, trigger/button legality, sampled log-probability rescoring, slot-keyed RNG, H4/H6 eager and
   compiled parity, and reset isolation remain exact.

During training, log teacher-forced and own-sample NLL/accuracy by offset and group, their depth-wise exposure gap,
full-frame and chunk exact match, transition/hold metrics, entropy/calibration, gate means and saturation, state RMS,
throughput, VRAM, and inference p50/p95/p99. Closed-loop stocks and damage remain the promotion evidence.

## Why the modern alternatives come later

- `minGRU` and `minLSTM` remove state dependence from their gates so teacher-forced sequences can use a parallel scan
  ([Feng et al., 2024](https://arxiv.org/abs/2410.01201)). Here the sequence has only ten tokens, and free-running
  inputs depend on sampled prior actions, so the scan advantage is small while the lost state-conditioned transition
  is exactly what this hypothesis wants to test.
- Mamba's selective SSM combines input-dependent recurrence with a hardware-aware scan and is compelling for long
  sequences ([Gu and Dao, 2023](https://arxiv.org/abs/2312.00752)). A custom kernel/state expansion is not justified
  for H4-H10 until a simple cell exposes a throughput or capacity limit.
- xLSTM adds exponential gates and, in mLSTM, a matrix memory
  ([Beck et al., 2024](https://arxiv.org/abs/2405.04517)). Its memory capacity targets long-context storage rather than
  this tiny plan register.
- Gated DeltaNet's matrix state combines adaptive erasure with targeted delta-rule writes
  ([Yang et al., 2025](https://arxiv.org/abs/2412.06464)). Its evidence is strongest on retrieval and long-context
  modeling; an `O(d^2)` fast-weight state and specialized kernels would make a first failure hard to interpret.

Action-chunk evidence is directionally supportive, not Melee-specific. ACT predicts coherent action chunks
([Zhao et al., 2023](https://arxiv.org/abs/2304.13705)); Diffusion Policy uses receding-horizon action sequences
([Chi et al., 2023](https://arxiv.org/abs/2303.04137)); and Chunking Causal Transformer reports gains from
autoregressive chunk generation over one-shot chunks in robot control
([Zhou et al., 2024](https://arxiv.org/abs/2410.03132)). None establishes which recurrent cell is best here.

## Decision gate

The first mechanistic comparison is standard-GRU action feedback versus its parameter-matched null-action control,
with 026 as the fixed external reference. Promote to a full run only if own-sample depth curves improve without a
primary NLL regression and same-host compiled p95 latency stays within 1.25 times 026. A failure under those controls
means either a 128-wide recurrent bottleneck or exposure bias; only then consider a wider cell, minGRU throughput
ablation, or a separately preregistered sampled-history objective.
