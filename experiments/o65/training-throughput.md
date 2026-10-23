# O65 training throughput

The control is `a0ae12df`, with the O64 checkpoint
`e6d6303fbf25993c768b1e86fecbb350433a2fb229235b556334e9eb6ae868cf`,
seed 65, 1,024 matches on one B200 and 512 rollout frames. Changes preserve
the policy, rewards, window placement, optimizer steps and precision contract.

## Existing profile

Source: `~/data/scratch/o65-o64-profile4/profile/`. This run uses 64 windows
per actor step, one microbatch and 16 selected decisions per window. Its
training code matches the control; its rollout code predates the current
overlap between anchor graphs and the student graph.

| Phase | Wall s | Model TFLOP | BF16 compute floor s | Achieved TFLOP/s | MFU |
|---|---:|---:|---:|---:|---:|
| Preparation | 6.258 | 2917.5 | 1.297 | 466.2 | 20.7% |
| Actor | 10.204 | 3260.6 | 1.449 | 319.5 | 14.2% |
| Critic | 0.946 | 53.9 | 0.024 | 56.9 | 2.5% |

The floor divides model FLOPs by 2,250 TFLOP/s. It excludes memory traffic,
optimizer work, nonlinearities and host work. The critic uses FP32, so its
BF16 floor is only a common normalization, not an attainable FP32 target.
FLOPs come from `timed.json` metrics. `estimates.json` retains a proxy-model
formula that understates O64 work and must not be used for O64 MFU.

The bounded traces show where to investigate. They are instrumented samples,
not replacements for the untraced timings above.

| Sample | Kernels | GEMM ms | Attention ms | Triton pointwise ms | Triton reductions ms | Other/index ms |
|---|---:|---:|---:|---:|---:|---:|
| Reference windows | 280108 | 2162.4 | 695.4 | 1209.8 | 662.6 | 1252.7 |
| Actor step 0 | 5767 | 26.8 | 4.3 | 4.8 | 4.7 | 42.4 |
| Bootstrap | 1386 | 102.0 | 35.5 | 4.0 | 2.0 | 275.1 |
| Critic fit | 727 | 717.8 | 0 | 0 | 0 | 233.0 |

The actor graph itself takes about 65.2 ms of summed kernel time. Eager trunk
pointwise operations dominate its non-GEMM work. The reference trace includes
128 graph replays; each encodes a trunk and four groups of decoder slots.
The critic's largest kernels are FP32 SIMT GEMMs.

Reproduce both tables with:

```sh
uv run python -m scripts.o65_training_trace ~/data/scratch/o65-o64-profile4/profile
```
