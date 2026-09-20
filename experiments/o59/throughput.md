# O59 v4 BF16 throughput results

O59 v4 uses batch 512, `reduce-overhead`, production diagnostics, eager Muon,
and action-embedding reuse. On one B200, the selected path reached a median
**1,295.4 samples/s** at active AWR. The matched production control reached
**1,286.6 samples/s**, so the measured gain was **0.68%**. Each of the three
treatment windows exceeded every control window.

The selected path does not change the batch, precision, loss, optimizer,
schedule, masks, example order, or prefix draws. It embeds observed and target
controller groups once, shifts the observed embeddings for previous actions,
and reuses target embeddings in the action-group features.

## Production comparison

Each row is one unprofiled window of 51,200 examples through pinned CPU
prefetching. Compilation and warmup are outside the windows.

| Path | Window 1 | Window 2 | Window 3 | Median | Update | MFU |
|---|---:|---:|---:|---:|---:|---:|
| Control | 1,290.2 | 1,286.6 | 1,286.5 | 1,286.6 samples/s | 397.94 ms | 28.58% |
| Reuse action embeddings | 1,297.6 | 1,295.4 | 1,295.2 | **1,295.4 samples/s** | **395.26 ms** | **28.78%** |

Peak reserved memory was 129.63 GiB for the control and 129.33 GiB for the
treatment. The selected path's three device-resident windows were 1,307.4,
1,302.3, and 1,299.6 samples/s. Its first update took 12.13 seconds and its
estimated compilation portion was 11.65 seconds with the shared cache.

The first comparison used the same bank and reference state before the benchmark
path was aligned with the production CPU-validation setting. Action reuse also
passed there: 1,294.8, 1,292.4, and 1,289.8 samples/s versus control windows of
1,286.8, 1,286.4, and 1,283.9. A later repeated control measured 1,262.6,
1,260.0, and 1,257.5. The production comparison above is the promotion result.

## Candidate selection

All candidates used the saved initial state, fixed bank, and recorded example
order. A candidate could advance only if all three active-AWR prefetch windows
beat all three control windows.

| Candidate | Active-AWR prefetch windows, samples/s | Decision |
|---|---|---|
| Reuse quantized history | 1,266.2; 1,261.5; 1,261.2 | Rejected |
| Reuse action embeddings | 1,294.8; 1,292.4; 1,289.8 | Advanced |
| Groupwise training loss | 1,291.3; 1,288.4; 1,283.6 | Rejected |
| Production action reuse | 1,297.6; 1,295.4; 1,295.2 | Promoted |

The numerical gate ran before timing with inactive and active AWR. It required
identical batches, masks, prefix draws, global and dedicated RNG states, and
conditioning state. It compared the objective and every parameter, gradient,
and optimizer update. Tolerance pairs were `2e-6/2e-5` for parameters,
`2e-4/1e-2` for gradients, and `2e-6/2e-2` for updates. The final treatment's
largest absolute differences were `2.36e-7` for parameters and updates and
`1.32e-4` for gradients. Its objective difference was `3.81e-6`. The gate
passed and no measured window recompiled.

## Profile decision

The five-update control profile reported these mean phase costs:

| Phase | Time |
|---|---:|
| Host to device | 0.011 ms |
| Target preparation | 1.038 ms |
| Trunk forward | 53.360 ms |
| Temporal forward | 72.505 ms |
| Objective | 2.307 ms |
| Backward | 256.960 ms |
| Gradient guard and clipping | 0.854 ms |
| Optimizer and scheduler | 16.021 ms |

The raw trace put all RMSNorm kernels at 7.52% of GPU kernel time. Short
attention, including its GEMMs and all softmax kernels as a conservative upper
bound, was 7.13%. The optimizer was about 3.9% of profiled update time. None met
the planned 10% threshold, so this pass did not add custom RMSNorm/FiLM or short
attention kernels and did not test the compiled-Muon hook.

## Protocol and artifacts

The fixed bank contains 2,048 real windows from the selected replay source and
stores return labels, availability, presence masks, and their identities. The
44-source production statistics, data manifests, schemas, and player identity
artifacts were validated. This isolates execution cost; it does not measure
sustained decoding from all 44 sources.

One B200 worker executed immutable jobs serially in separate child processes.
The jobs shared dataset and compiler caches. The worker had no automatic retries,
a 24-hour Modal timeout, and an 86,100-second work deadline. It completed in
3,217.3 seconds (0.894 B200-hours), preserved all reports, saved the compiler
cache, and then closed cleanly.

- Modal App: `ap-6uQmiSy24usVefPXgl56On`
- FunctionCall: `fc-01M30GC4MHVSJXGB3N7B8PZPYG`
- [Modal dashboard](https://modal.com/apps/ap-6uQmiSy24usVefPXgl56On)
- Volume directory: `5aa49af7ca9942989a12ce3f5ed7a31c`
- Base Git SHA: `6c0ec8690e5a82cd50dc0102b3f4bf1b6d8e26ba`
- Bank SHA-256: `88dc62db8e3bfca85c3d70bed103716357c0c72ce57e4afe3d0e63a6294d926d`
- Reference SHA-256: `43ac38b2c2bebb4d2e3c6dc1a5d4a88ca2215d21d311d8457e7780694ee73998`
- Compiler-cache SHA-256: `cab8700f5fe73ea8da3d69a2209b76110d127ab84c96e6a87d477483a2e737d8`
- Environment: PyTorch 2.11.0+cu130, CUDA 13.0, Python 3.14.3, NVIDIA B200

[Machine-readable results](throughput-v4-results.json) contain the exact windows,
job and source identities, selection rule, profile thresholds, and final choice.
Raw reports, source archives, logs, the bank, reference state, and traces remain
in the `hal-o59-benchmarks` Modal Volume.

The production launch and evaluation protocol are in [launch.md](launch.md).
Repository and parity checks are in [validation.md](validation.md).
