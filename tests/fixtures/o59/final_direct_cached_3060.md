# Final-source direct cached inference, RTX 3060

Three alternating control/candidate trials used the immutable control checkout
`d7454f9d1a136f7c745d2d4478af22e19cd65faf` and the current refactor
working tree. Each trial replayed 2,400 frames of the same Slippi game with
the same 059 bundle, seed 1001, CUDA graphs, BF16, H8/prefix3/delay2/stride4,
and Q4 observations. The steady sample excludes the first 300 frames and has
525 prediction calls. Both processes used PyTorch 2.11.0, one Torch CPU thread,
frozen long-lived GC objects, and the same TorchInductor cache location.

Bundle SHA-256:
`13a2a2922ea3a015d583f3efa345368ca41ee81adc614d5fd182700337a926fa`.
Replay SHA-256:
`81af02a949c3c436d5e527b5dd83600d42338887d4445de4dc515e489e6cbe94`.

| Pair | Control FPS | Candidate FPS | Paired ratio | Control/candidate p95, ms | Control/candidate p99, ms | Control/candidate preparation, s |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 489.413 | 483.464 | 98.78% | 8.256 / 8.314 | 9.242 / 9.212 | 14.098 / 12.728 |
| 2 | 487.771 | 481.416 | 98.70% | 8.382 / 8.593 | 8.997 / 9.278 | 14.057 / 12.773 |
| 3 | 486.386 | 483.020 | 99.31% | 8.421 / 8.356 | 9.086 / 8.984 | 14.507 / 12.697 |

FPS is `4,000 / mean(prediction_ms)` over the 525 steady calls. The median
paired throughput ratio is **98.78%**, above the 95% mechanical gate. GPU peak
allocated memory was 536.335 MiB for every control and 536.323 MiB for every
candidate, within the 105% gate. Candidate peak process RSS was 1,974,352,
1,974,988, and 1,974,076 KiB versus 1,933,852, 1,932,144, and 1,933,072
KiB for control. Candidate user/system CPU times were 22.81/3.08,
22.89/3.08, and 22.80/3.06 s; control was 23.68/3.18, 23.68/3.13, and
24.10/3.24 s. OS file-input blocks were 118,000 for the first control and
zero in the other five runs; profiler trace output dominated file writes.

The candidate's first prediction took 9.43, 9.50, and 9.50 ms, versus 9.90,
9.88, and 9.83 ms for control. Before controller decoding was warmed during
preparation, a profiling run took 38.91 ms on the first candidate request;
`DiscreteControllerCodec.dequantize` accounted for 33 ms. A matched profiling
run after the fix took 10.54 ms first and 9.17 ms second. The benchmark uses
direct policy calls; it excludes IPC, Dolphin, and controller submission, so
it does not qualify the 3060 netplay service deadline.

The six raw result directories and adjacent `.log` files are
`runs/refactor-059/final-warm-{control,candidate}-3060-{1,2,3}`. Each `results.json`
contains all per-call times, configuration, GPU peak, and relevant source
hashes. `runs/refactor-059/final-warm-source-manifest-{start,end}.json`
records hashes for 123 maintained source/configuration files and the input
artifacts. The measured inference source hashes match across all three runs
of each side. Three unrelated candidate files, `hal/eval/netplay.py`,
`hal/eval/results.py`, and `hal/netplay_service/runner.py`, changed during the
trial sequence; none is imported by this direct cached path. The exact command
and process resource record for each trial are in its `.log` file. The six
pre-warmup-change runs remain under `runs/refactor-059/final-{control,candidate}-3060-{1,2,3}`
as diagnosis, not the final-source comparison.
