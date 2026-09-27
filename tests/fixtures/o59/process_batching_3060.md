# Spawned-process cached batching on RTX 3060

The control sent two requests serially through persistent `InferenceClient`
connections to one spawned `InferenceEngine`. The treatment submitted both
ready requests before waiting for either response. One 059 checkpoint was
loaded once per worker process. All three runs used PyTorch `2.11.0+cu130`,
one Torch CPU thread, the same H8/prefix3/delay2/Q4 CUDA-graph profile, the
same 059 checkpoint bundle (SHA-256
`13a2a2922ea3a015d583f3efa345368ca41ee81adc614d5fd182700337a926fa`),
and the same replay (SHA-256
`81af02a949c3c436d5e527b5dd83600d42338887d4445de4dc515e489e6cbe94`).
Each run warmed 20 request pairs and measured 200. Pair time starts before
the first client submission and ends after both plans are delivered and polled;
it includes connection delivery, queueing, cache gathering/scattering, model
execution, plan validation, and response delivery.

| Run | Admitted / ready | Engine calls / items | Pair p50 / p95 / p99, ms | Prepare, s | Peak allocated, MiB |
|---|---:|---:|---:|---:|---:|
| Serial control | 2 / 1 at a time | 440 / 440 | 21.876 / 23.369 / 24.654 | 20.277 | 946.99 |
| Concurrent treatment | 2 / 2 | 220 / 440 | 11.405 / 11.969 / 12.275 | 19.990 | 946.99 |
| Sparse load | 32 / 2 | 220 / 440 | 11.527 / 11.804 / 12.822 | 253.000 | 2393.72 |

The concurrent pair's median completion time was 47.9% below serial, beyond
the proposed 5% batching benefit. Every concurrent pair used one actual
two-row model call. The sparse-load p95 was 98.6% of the two-admitted p95,
within the proposed 105% limit. Thirty admitted streams were idle in the
sparse run; this does not qualify 32 simultaneously active streams. The first
cold capacity-32 preparation took 253 seconds, so it does not establish the
120-second cached-artifact recovery gate. This benchmark does not include
Dolphin, the service runner, or Ada hardware.

The raw timing samples, full source hashes, and environment records are:

- `runs/refactor-059/process-serial-2-1/results.json` and `.log`
- `runs/refactor-059/process-batch-2-1/results.json` and `.log`
- `runs/refactor-059/process-batch-32-1/results.json` and `.log`

The benchmark CLI SHA-256 was
`787aed8e388cd83a1931568f028bdffe9fa6b77bb9f3e5af2743681621f0ed64`.
The benchmark library, engine, client, cached policy, and cache SHA-256 values
were respectively `0a911b7c68b724bd361db7de7a444f3a542e85aa386b0db1c6820c95383272ae`,
`e69eea64a2c66fb27c67acf4336284781b5b02ad232405364ef18e01412e0d1b`,
`92a68f752b86e14fbfb9fb144db81ae0af3c3f2509f49d3d75462110ee4df93d`,
`2031757d0abcc2931b7924e3558246b0d9710933340d04c36dd95a2cdaac8868`,
and `6d75628c4688d610c1bed93c94873d960ba25572d3104af61bc5a128b2d8dadc`.
Those five hashes match across the three runs. `window_policy.py`, unused by
this cached profile, changed while the sparse run prepared. An explicit-device
CUDA graph correction landed after these runs; final-source timing still needs
measurement.

To repeat a run, use the maintained checkout and substitute `OUTPUT` with a
new directory. Add `--serial-requests` for the serial control, or set
`--capacity 32` for the sparse run:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  TMPDIR=/home/ericgu/src/hal/runs/refactor-059/tmp \
  TORCHINDUCTOR_CACHE_DIR=/home/ericgu/src/hal/runs/refactor-059/tmp/torchinductor \
  /home/ericgu/src/hal/.venv/bin/python \
  /home/ericgu/src/hal/scripts/benchmark_kv_cache.py \
  /home/ericgu/src/hal/runs/netplay/o59-vywk3cih.hal \
  /home/ericgu/src/hal/runs/netplay/selfplay-o59-20260924-retry1/replays-a/Game_20260924T152607.slp \
  OUTPUT --process-batching --update-frames 4 --prediction-horizon 8 \
  --fixed-prefix 3 --replan-interval 4 --physical-delay 2 \
  --capacity 2 --warmup-calls 20 --measured-calls 200
```
