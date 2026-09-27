# Reduced buffered-loader comparison

The three pairs with controlled file-cache preparation meet the throughput and
memory limits **for this reduced loader-core workload**. Median candidate/control
throughput is 100.66%; the largest paired peak-RSS ratio is 97.12%.

This does not qualify the production 131,072-slot loader or the transformed 059
training path. It measures raw window batches, using the same no-label and
no-transform callbacks in both checkouts.

## Controls

- Control: `d7454f9d1a136f7c745d2d4478af22e19cd65faf`.
- Candidate: `e67a51ea7489874cdca60a82e24df8e196f398fa`.
- Source: `professional-aklo-policy-world-v8`, train split.
- Manifest SHA-256: `1ae04b2ffd57fe0bb1bac86933f61b4fbad151bfe7956f607f6ff19521895f64`.
- Selection SHA-256: `5b7c0e8e34d0502f3284c57060db9a1dac10f7129344715a7b5acea209ac029e`.
- Batch 64; 1,600 replay slots; two workers; eight windows per replay generation;
  25-batch phase blocks; seed zero; four materialization threads.
- Warm 200 batches, then measure 500 batches (32,000 samples) per trial.
- AMD Ryzen 5 5600X, affinity 12 logical CPUs; one OMP/MKL thread; Python 3.14.4,
  Torch 2.11.0, Mosaic 0.13.0. CUDA is hidden from these loader processes.

Both checkouts resolve to the same local corpus files through the control data
symlink. All 92 raw shards, totaling 24,437,937,463 bytes, must already exist at
their declared lengths. Before each trial the driver issues
`POSIX_FADV_DONTNEED` for those raw files, then runs the identical warmup. This is
a file-scoped kernel hint, not a global cache flush. The measured physical reads
confirm equal disk traffic in these six trials.

The common capture code differs only in imports needed to call the immutable
control. No loader source or saved sampling protocol was modified for measurement.

## Results

| Pair | Control samples/s | Candidate samples/s | Throughput ratio | Peak-RSS ratio |
|---|---:|---:|---:|---:|
| 1 | 3,915.07 | 3,955.18 | 101.02% | 96.82% |
| 2 | 3,907.76 | 3,933.50 | 100.66% | 97.01% |
| 3 | 3,959.19 | 3,933.22 | 99.34% | 97.12% |

Every trial read **5,243,879,424 physical bytes** during measurement and wrote
zero bytes. Startup below means adapter setup plus delivery of the first batch;
Python import time is outside that interval. Peak RSS sums process high-water
marks, so it is a conservative process-tree upper bound.

| Trial | Startup seconds | Measurement CPU seconds | Peak RSS bytes |
|---|---:|---:|---:|
| Control 1 | 3.345 | 20.25 | 3,168,485,376 |
| Candidate 1 | 3.247 | 19.96 | 3,067,813,888 |
| Control 2 | 3.301 | 20.29 | 3,164,246,016 |
| Candidate 2 | 3.303 | 19.88 | 3,069,607,936 |
| Control 3 | 3.276 | 19.92 | 3,163,152,384 |
| Candidate 3 | 3.345 | 20.09 | 3,072,147,456 |

Raw record: [series 3 summary](../../../runs/refactor-059/loader-small-pairs-3/summary.json),
SHA-256 `441aa38e1d0d2591913c3a5a7b34acde8a2ce4465c644fba5243c834a12e3c4d`.
It references all six trial records and their individual cache-preparation records.

The preserved diagnostic attempts explain why cache preparation matters:

- Series 1 had median throughput ratio 97.40%, but the first control still
  materialized missing shards. It is not a matched-cache comparison.
- Series 2 had complete local files and median ratio 109.87%, but control
  trials read 4.04–4.63 GB physically while candidate trials read 0.39–0.84 GB.
  The apparent speedup is not attributed to the refactor.

All three series, including these diagnostic records, are retained. The selected
52-file archive contains 465,371 bytes, with manifest SHA-256
`b2b1973f5c03a2e3ee4e67caf3bba7fab3154bced192a210e957b5cef1a5fdd0`.
Its artifact prefix is
`r2://hal/runs/refactor-059/evidence/loader-b2b1973f5c03a2e3ee4e67caf3bba7fab3154bced192a210e957b5cef1a5fdd0/`.
The upload used `--immutable --checksum`; `rclone check --download` then
verified all 52 files with zero differences. The local verification log is
`runs/refactor-059/loader-evidence-verify.log`.

## Reproduce

Use the pinned control checkout, the shared data symlink, and the existing
environment and R2 configuration. The command requires a new output directory:

```bash
TMPDIR=/home/ericgu/src/hal/runs/refactor-059/tmp \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python tests/fixtures/o59/run_loader_pairs.py \
  --repo /home/ericgu/src/hal \
  --control /tmp/hal-059-control-d7454f9 \
  --output-root runs/refactor-059/loader-small-pairs-3
```
