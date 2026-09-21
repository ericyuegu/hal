# O59 v5 BF16 throughput results

O59 v5 uses batch 512, BF16, `reduce-overhead`, and eager Muon. The treatment is
adaptive RMSNorm for within-frame action groups. The invariant inputs were the
saved initial model and optimizer state, fixed replay bank, masks, prefix draws,
conditioning state, and example order.

No execution candidate passed the promotion rule. Production therefore uses the
plain implementation. The rejected implementations and benchmark-only profiler
code were removed after the evidence was saved.

## Results

A candidate could advance only when all three active-AWR prefetch windows beat
all three control windows. Each window contains 51,200 examples. Compilation and
warmup are excluded.

| Path | Window 1 | Window 2 | Window 3 | Decision |
|---|---:|---:|---:|---|
| Control | 1,286.6 | 1,287.8 | 1,283.4 | Control |
| Reuse quantized history | 1,286.4 | 1,288.6 | 1,283.7 | Rejected |
| Reuse action embeddings | 1,289.4 | 1,289.9 | 1,285.3 | Rejected |
| Groupwise training loss | 1,288.2 | 1,285.4 | 1,285.4 | Rejected |
| Repeated control | 1,285.3 | 1,287.3 | 1,285.5 | Confirmed control range |

The fastest candidate window did not establish a repeatable gain. The repeated
control median was 1,285.5 samples/s, close to the original control median of
1,286.6 samples/s. The control's active-AWR device-resident windows were 1,288.5,
1,289.5, and 1,286.3 samples/s. Peak reserved memory was 128.45 GiB.

The numerical gate ran before timing with inactive and active AWR. It required
identical batches, masks, prefix draws, global and dedicated RNG states, and
conditioning state. It compared the objective and every parameter, gradient,
and optimizer update. All candidates passed. Across candidates, the largest
absolute parameter or update difference was 5.49e-7, the largest gradient
difference was 1.24e-4, and the largest objective difference was 2.48e-5. No
measured window recompiled.

## Profile decision

The five-update control profile reported these mean phase costs:

| Phase | Time |
|---|---:|
| Host to device | 0.010 ms |
| Target preparation | 0.962 ms |
| Trunk forward | 53.454 ms |
| Temporal forward | 74.757 ms |
| Objective | 2.393 ms |
| Backward | 255.454 ms |
| Gradient guard and clipping | 0.859 ms |
| Optimizer and scheduler | 15.906 ms |

RMSNorm kernels accounted for 7.68% of GPU kernel time. Fixed-length attention,
including its GEMMs and all softmax kernels as a conservative upper bound, was
below 2%. The optimizer was about 3.9% of update time. None reached the planned
10% threshold, so this pass did not implement custom RMSNorm/FiLM or short
attention kernels and did not test compiled Muon.

## Artifacts

One B200 worker executed immutable jobs serially with no automatic retries. It
used a 24-hour timeout and an 86,100-second working deadline. It completed in
2,206.9 seconds, saved the compiler cache and reports, and stopped cleanly.

- Modal App: `ap-vxworW0PW7QS6LVbKwkIwm`
- FunctionCall: `fc-01M30P63N6TQ6ZHG26R39Y4Z2A`
- [Modal dashboard](https://modal.com/apps/ap-vxworW0PW7QS6LVbKwkIwm)
- Volume directory: `0f5290d426da49f88192ddbf983b5aa9`
- Bank SHA-256: `625ded22595305fdb15df5836f29eb10602e2776d639c1cb553a4c3d0ad4979c`
- Reference SHA-256: `d13edf0295b7563482dca07fd9f68cfad0653674eb2a6de34afa750098218505`
- Compiler-cache SHA-256: `ad92c8bb1229c3eb78e6287d033df9dfe6c5dad0383f54b932787747b04dee60`
- Environment: PyTorch 2.11.0+cu130, CUDA 13.0, Python 3.14.3, NVIDIA B200

[Machine-readable results](throughput-v5-results.json) contain the exact active
windows, immutable job and source identities, selection rule, profile threshold,
and final decision. The raw source archives, reports, reference state, and trace
remain in the `hal-o59-benchmarks` Modal Volume.

The production command is in [launch.md](launch.md). Repository and model checks
are in [validation.md](validation.md).
