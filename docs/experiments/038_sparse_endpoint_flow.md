# 038 sparse categorical-endpoint flow

Status: implemented and locally validated; no production training or Dolphin
quality result has been run yet.

## Question

Experiment 038 replaces 037's selected-offset autoregressive action chain with a
parallel categorical-endpoint flow. It asks whether four iterative refinements
of one complete sparse plan improve closed-loop temporal coherence while keeping
the d256 observation policy, four-frame execution cadence, and measured latency
practical.

## Fixed comparison

The production configuration keeps 037's base observation bundle, structured
controller codec, causal d256/L8/H4 trunk, 128-frame context, batch 512, 16,384
updates, Muon/AdamW partition and learning rates, 500-step warmup, BF16 training,
ranked-anonymized v7 stream, replay reservoir, seed, checkpoint format, and final
96-boot level-9 CPU protocol. It trains from step zero; there is no actor or trunk
warm start.

The future offsets are exactly `(1,2,3,4,5,6,9,12,16,20)`. The policy always
predicts all ten offsets and executes only offsets 1 through 4. Unlike 037, the
only policy objective is the equal mean of the four endpoint categorical CEs;
the detached AWR value-head auxiliary is intentionally absent.

## Training geometry

The trunk runs once on each `[B, 128]` sequence. Each row contributes 32 causal
prefix problems: the final prefix plus one random sample from each of 31 strata.
The gathered context states flatten to `[B * 32, 256]`, and the complete flow
expert runs in one vectorized call. Opening/cold-start windows with fewer than 32
distinct real prefixes retain 32 valid problems by reusing the nearest valid
prefix for empty strata; this preserves the parent data distribution and keeps
the final deployment-like prefix present. No prefix microbatching is implemented.

Each prefix draws one scalar `tau ~ Uniform(0, 1)` and named iid Gaussian states
for buttons (256), main stick (65), C-stick (9), and triggers (25). The four named
one-hot endpoints are interpolated as `X_tau = (1 - tau) Z + tau X_clean`. The
same scalar time applies to every offset and group within that prefix.

## Action expert

The group projections are `256->64`, `65->48`, `9->16`, and `25->24`; their
concatenation passes through `152->256->192` with SiLU. Learned embeddings use
the actual offset IDs, not dense token indices. The bidirectional expert has
three d192 blocks, three 64-dimensional heads, no attention mask, RMSNorm, and a
complete d512 SwiGLU FFN.

History and sinusoidal flow time are separately projected and fused into one
global condition. Every block owns a zero-initialized six-way AdaLN-Zero
modulator for attention and FFN shift, scale, and residual gates. Context and
time are never added to the noisy action-token content.

Four direct linear heads emit raw endpoint logits. The training loss averages
each group over batch, 32 prefixes, and ten offsets, then averages the four group
means. Gradients reach the expert and causal trunk; context states are not
detached.

The default model has 8,946,063 trainable parameters: 6,291,456 in the causal
trunk and 2,508,499 in the flow expert including its heads. The static estimator
reports 16,146,288 MACs (32.293 MFLOPs) per expert NFE and 1.830 GFLOPs for one
batch-one H4 replan including the trunk and four NFEs. These are audit estimates;
the primary systems result remains measured wall-clock latency.

## Inference and evaluation

Inference starts from FP32 iid Gaussian noise and uses exactly four endpoint
network calls on the left-endpoint Euler grid `[0, 0.333, 0.666, 0.999]`, with
`epsilon=1e-3`. The Transformer runs under BF16 autocast. Softmax, velocity
`(p_clean - X_tau) / (1 - tau)`, solver state, and Euler updates remain FP32.
The final NFE supplies the logits for groupwise argmax; there is no extra call or
intermediate discretization.

Independent group argmax can emit trigger/button conflicts. Validation and
closed-loop telemetry record their pre-repair rate, then the existing codec's
click-to-full-trigger canonicalization runs before actions reach Dolphin.

Offline validation includes group/offset/time-bucket CE and accuracy, shuffled-
context CE deltas, 4/8/16-NFE reconstruction with shared noise, eight-seed plan
diversity, AdaLN gates and subsystem gradients, action transitions/holds, and
invalid combinations. Parent-compatible joint/primary/auxiliary NLL, group
accuracy, exact-frame/dense-prefix accuracy, transition, quantization, and invalid
count names are retained where they have a direct flow analogue. Dolphin
evaluation is disabled during training and runs
once, after the final checkpoint, at H4/NFE4/argmax. The evaluator caps active
Dolphins at a power-of-two no larger than the allocated/declared CPU count and
runs the 96 boots in waves.

Latency uses the 037 `BF16Inference`, synthetic context, and decode telemetry
paths. It reports replan/model-decode p50, p95, and p99; amortized p50 per executed
frame; exact NFE; action-expert and total inference FLOPs; closed-loop decode
timing; broker environment-step timing; and end-to-end control-loop timing.

## Commands

Focused validation:

```bash
uv run pytest -q tests/experiments/test_038_sparse_endpoint_flow.py
uv run ruff check experiments/038_sparse_endpoint_flow.py tests/experiments/test_038_sparse_endpoint_flow.py
uv run experiments/038_sparse_endpoint_flow.py --benchmark
```

No-rent Modal audit:

```bash
uv run scripts/launch_modal.py --dry-run --gpu L40S --cpu 16 --cpu-limit 16 --memory-gib 64 --disk-gib 512 --timeout-hours 8 --app-name hal-038-flow -- uv run experiments/038_sparse_endpoint_flow.py
```

Production launch after the no-rent audit and a clean pushed commit:

```bash
uv run scripts/launch_modal.py --gpu L40S --cpu 16 --cpu-limit 16 --memory-gib 64 --disk-gib 512 --timeout-hours 8 --app-name hal-038-flow -- uv run experiments/038_sparse_endpoint_flow.py
```

Manual final-checkpoint evaluation, if evidence repair is ever required, remains:

```bash
uv run experiments/038_sparse_endpoint_flow.py --eval runs/<run>/final.pt
```
