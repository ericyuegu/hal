# Data-pipeline de-risk plan

Updated: 2026-08-07

## Reason

P0 used the intended model configuration, but the GPU waited for data. After the cache filled, steps
600 through 800 averaged 1.53 seconds. Training projected to about 7.2 hours before evaluation. The
target is 3.0 to 3.5 hours including evaluation.

The v7 training split contains 112,409 replay rows. Its 375 Zstd shards occupy 29.82 GB, but Mosaic
Streaming expands them to 802.60 GB before reading. One row contains a complete replay. P0 loads a
complete row, then keeps two 269-frame windows. The full run needs 8,388,608 windows and scans the
replay corpus at least 37 times.

This work is infrastructure. It must not change the model, loss, effective batch size, or evaluation
protocol.

## Invariants

- Keep the effective batch at 512.
- Measure the number of distinct replay IDs in every batch.
- Keep action and observation values exact through the first optimization stage.
- Do not use `misc_as` as a model feature.
- Keep validation examples fixed.
- Compare batch tensors and losses before comparing speed.
- Treat a new sample order or new window sampler as a new training baseline.

## Stage D0: reduce speculative work

Keep the current MDS and sampler. Benchmark:

- `predownload`: 256, 512, 1,024, and the current 4,096.
- `prefetch_factor`: 1, 2, and the current 4.
- `shuffle_block_size`: keep 2,000 at first. Test a smaller value only after the other settings.

Record time to first batch, loader wait time, process read bytes, raw shards created, compressed bytes
downloaded, CPU use, RAM, and batch hashes. `predownload` must remain greater than one.

Files:

- `hal/training/dataloader.py`: expose explicit settings and counters.
- `scripts/bench_dataloader.py`: repeatable loader benchmark.
- `tests/test_dataloader.py`: order and tensor-equality tests.

Gate: lower startup pressure without changing the first fixed batches.

## Stage D1: project before preprocessing

Declare the raw columns needed by the consumer. After choosing the ego port, retain only those
columns before window copies and preprocessing. Do not derive spatial features or build opponent
controller tensors when the model does not read them.

Files:

- `hal/training/features.py`: consumer input declaration.
- `hal/training/dataloader.py`: early projection.
- `experiments/023_mtp_heads.py`: pass the declaration.
- `tests/test_dataloader.py` and `tests/experiments/test_023_mtp_heads.py`: exact output checks.

Gate: every tensor consumed by experiment 023 is equal to the current path. No required tensor may be
missing. Measure worker CPU time and bytes passed to the main process.

## Stage D2: exact compact replay storage

Build a model-facing dataset from v7. Store only required raw fields. First preserve current dtypes.
Then test exact reversible packing for categorical values, buttons, sticks, and triggers. Keep the
canonical v7 dataset as the source of truth.

The projected current-dtype dataset should be about 360 GB. Exact packed storage should be smaller,
but its size is a measurement, not an assumption.

Files:

- New storage schema under `hal/data/`.
- New conversion command under `hal/scripts/`.
- New round-trip and size tests under `tests/`.
- A new stream registry entry only after the artifact passes validation.

Gate: decode every projected value exactly on the development corpus and a representative v7 shard.
Compare frame counts, masks, actions, model input tensors, target tensors, and losses.

## Stage D3: replay-aware window reuse

Reading more windows from one replay saves I/O but must not collapse replay diversity. Prototype a
bounded reservoir:

1. Read one replay and sample several non-overlapping windows.
2. Tag each window with a stable replay ID.
3. Mix windows across a reservoir containing many replay IDs.
4. Emit batches with no repeated replay ID.
5. Apply a replay cooldown across nearby batches and measure that correlation.

Window seeds must depend on the experiment seed, epoch, and stable replay ID. They must not depend on
DataLoader worker ID. The current worker-dependent seed prevents worker-count changes from being pure
systems tests and must be removed.

Gate: 512 distinct replay IDs per batch when enough replays exist, deterministic batches across worker
counts, correct marginal window-start distribution, and lower replay bytes per training window.

## Representative benchmark

Use `mds-v7-sub4` for host benchmarks before another full run. It contains real v7 shards and avoids
an 803 GB expansion. Compare the current loader, D0, D1, D2, and D3 independently.

Do not relaunch P0 until one configuration projects below 0.5 seconds per optimizer step on a suitable
host, sustains at least 80% GPU use after warm-up, and preserves the correctness gates above.

## Current implementation

The `exp/data-pipeline` branch now has:

- A compact replay schema that keeps only P0 inputs and targets.
- Exact packing for controller values, buttons, state values, and missing data.
- A path-based 128-bit replay ID. The old 32-bit manifest ID has three collisions.
- A central replay reservoir with one replay per batch and a one-batch cooldown.
- Stable window sampling based on the seed, epoch, and replay ID.
- Slice decoding. Workers decode only the selected windows, not the full replay.
- Early feature projection in training and closed-loop evaluation.
- W&B metrics for loader wait, replay diversity, and dropped epoch-tail windows.

The artifact is named `mds-policy-v7`. It is a policy projection of canonical MDS v7, not a new
canonical v8 schema. Each row records source schema 7 and compact policy layout 2.

The full audit passed all 114,768 ranked replays and about 1.23 billion frames. The compact train
arrays use 76.19 GB instead of 802.47 GB, a 10.53x reduction. The complete compact artifact has 291
compressed shards and uses 13.34 GB. The v7 artifact uses 29.82 GB compressed.

The audit also found 6,933 training frames in 15 replays with action states above the old 511 limit.
The maximum was 525. New P0 runs use a 1,024-row action embedding. Old checkpoints keep 512 rows when
loaded, so their parameter shapes do not change.

On the complete local artifact with batch 512 and 16 workers, the compact reservoir produced 512
distinct replays per batch. Capacity 4,096 took 7.16 seconds to start, then loader waits averaged
0.072 seconds. It used 2.77 GB peak RAM. Across 8,192 samples, it used 3,840 distinct replays. The
median replay reuse gap was four batches; 21.9% of reuse gaps were two batches. Capacity 1,024 is the
minimum that enforces the cooldown, but it nearly alternates two fixed replay groups. P0 therefore
uses capacity 4,096. This does not measure remote downloads or GPU use. A Vast probe must still test
`predownload` 256, 512, 1,024, and 4,096.

## Startup overlap

Train prefetch, validation caching, and Torch compilation are still sequential in experiment 023.
Overlap them only after the loader is correct. Add separate timers and keep one serial control run.
Concurrent reads can increase disk pressure, so total startup time is the deciding measure.

## Stopped P0 evidence

- Vast instance: `47051823`.
- W&B run: `ws7a72tw`.
- The run was stopped after step 1,024 because of throughput, not numerical failure.
- Validation at step 1,024 was finite. Primary NLL was 1.221 bits per frame.
- The instance was destroyed on 2026-08-07.
