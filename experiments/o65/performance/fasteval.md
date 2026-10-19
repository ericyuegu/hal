# Batched O65 evaluation on the RTX 3060

The evaluator now batches each MSL cell's matches and both players by actor
weights. Dolphin uses the same policy with concurrent `process_vec` workers.
Inference keeps FP32, the two block caches, processed controller readback,
committed factors, and `evaluation_uniforms`. Different actor weights use
separate batches. The serial path remains available with `--execution serial`.

## Measurements

These local measurements use the O63 bootstrap30 proxy, an RTX 3060 12 GB,
Ryzen 5 5600X, Python 3.14.4, Torch 2.11.0+cu130, and two Torch CPU threads.
The checkpoint SHA-256 is
`65e7e43e5cbb18a365a005922ec1cfd026ad807c045eae13b331f487b4ec82e6`.
The control uses `SelfPlayPlayer` from `e6d768d5`. Its actor and both serial
match runners remain unchanged in the implementation commits `56ee1722` and
`0a6cad0f`. The 256-match measurement's inference source hash matches `56ee1722`.

The treatment changes batching and graph execution. Both arms use delay 8,
replan 4, lambda 1, masked identity, the same weights and statistics, FD
Fox/Falco starts, and a 512-frame budget per match. The serial case uses seed 17
on focal port 1. Batched cases use seeds `17 + match_index // 2`, with paired
orientations. These runs measure execution cost and do not select a policy.

| Backend | Execution | Concurrent matches | Captured match-frames/s | Match wall s |
| --- | --- | ---: | ---: | ---: |
| MSL | Serial eager | 1 | 15.61 | 32.799 |
| MSL | Batched graphs | 4 | 482.31 | 4.246 |
| MSL | Batched graphs | 64 | 1,441.49 | 22.732 |
| MSL | Batched graphs | 256 | 1,602.03 | 81.816 |
| Dolphin | Serial eager | 1 | 12.69 | 40.341 |
| Dolphin | Batched graphs | 2 | 83.51 | 12.262 |
| Dolphin | Batched graphs | 4 | 138.86 | 14.749 |
| Dolphin | Batched graphs | 8 | 191.87 | 21.348 |

Rates count captured observations, including reset and countdown, once per
match. Match wall time includes reset, policy work, stepping, trace collection,
and cold graph capture. Dolphin also includes worker startup, shutdown, and
replay processing. Native library initialization, checkpoint loading, and policy
allocation occur outside match timing. The JSON also records the outer timing.
Every measurement uses cProfile. No match ended before its frame budget.

The four-match MSL cell improves by 30.9 times. The eight-worker Dolphin wave
improves by 15.1 times. Eight is the best worker count tested, so it is the
default limit; a four-match cell still uses four workers. At 256 matches,
evaluation uses 7.98 GiB peak reserved GPU memory. The recorded training
rollout at 256 matches takes 22.130 s for 65,536 match-frames, about 2,961/s.
That training path uses a BF16 trunk and a single block cache. Evaluation
keeps FP32 and dual caches for parity. Dolphin remains slower; its eight-worker
profile spends 13.42 s polling worker pipes, including about 9 s of startup.
A separate eight-worker run with the pilot's 1,924-frame budget reaches
360.15 match-frames/s in 42.737 s, including startup and replay processing.

The serial MSL profile attributes 32.65 of 32.80 match seconds to policy calls:
17.95 s in cache appends, 7.24 s in decoding, and 3.56 s in feature construction.
The batched path uses `PackedColumnContextBuilder`,
`forward_tokens_with_block_handoff`, and `DecisionDecoder` CUDA graphs.

Raw measurements and profile hashes are in
[`fasteval-measurements.json`](fasteval-measurements.json). Commands, results,
and skip reasons are in [`fasteval-validation.md`](fasteval-validation.md). Local profiles,
logs, and replay files are under `runs/fasteval/`. Reproduce a measurement with:

```sh
OMP_NUM_THREADS=2 \
PYTHONPATH=.:/home/ericgu/src/melee-sim-light \
MSL_CORE_LIBRARY=/home/ericgu/src/melee-sim-light/build/melee_core/python/libmelee_core.so \
uv run python -m scripts.profile_o65_evaluation \
  --checkpoint runs/proxy/final.pt \
  --data-dir /home/ericgu/src/melee-sim-light/data \
  --output runs/fasteval/repeat-msl --execution batched --matches 4
```

Use `--simulator dolphin --matches 8` for Dolphin. Use `--execution serial
--matches 1` for the control. The profiler records inputs and source hashes
before play, then saves cProfile data, timings, and peak GPU memory.

## Parity and review

The proxy parity test compares sampled factors, submitted actions, and every
native frame byte against independent per-frame `SelfPlayPlayer` instances.
It uses seeds 17 and 29, both orientations, mixed 8/4 and 10/2 cadence, cache
handoffs, and asynchronous resets. One-stock, 100-percent starts with damage
ratio 100 produce seven stockouts over 560 ticks. Sampled actions, simulator
states, and outcomes match exactly. The largest log-probability gap is
0.0000038147 nats. The test also replays each trace through a one-match native
batch and checks timeout and budget stops at different frames.

CPU tests cover distinct model weights, missing-stream returns, partial resets,
invalid cadences, and unsupported sliding history. Batched results use
`hal.o65.evaluation-results.v2`: each match receives an equal share of batch
wall time, so the sum measures throughput. The training and pilot readers accept
both versions. Gameplay metric definitions retain the serial arithmetic,
including float64 trajectory columns.

Review found an unnecessary cache copy during first graph capture. It caused
the initial 256-match measurement to fail with CUDA OOM. Capture now discards
its empty warmup state; the repeat completed at 1,602/s. Review also strengthened
the simulator reference with one-match replay and updated both result readers.
A two-action Dolphin request trial measured 138.46/s at four workers and
192.69/s at eight. The implementation keeps one-frame requests because that
trial showed no useful gain. A second review found no further implementation
defects.
