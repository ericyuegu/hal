# Experiment 026 closed-loop profile and implementation

Date: 2026-08-10

## Outcome

Closed-loop evaluation had three independent costs:

1. Torch compiled new inference shapes inside the live loop.
2. All Session CPU work ran as threads in the coordinator process and shared one GIL.
3. Match conversion and a ten-second Dolphin shutdown wait stopped progress at wave boundaries.

The implementation removes the first two causes and bounds the third:

- Evaluation compiles its inference programs synchronously on first use; training never compiles CUDA programs from another thread.
- Each Dolphin Session now runs in a spawned Python process. The main process only owns the policy and GPU.
- Observations, action plans, and final trajectories use typed shared memory. Hot control messages are fixed 64-byte binary records. The hot path does not use pickle.
- One Session worker can publish more than one model slot. Self-play uses two slots per Dolphin, one for each controller port, and the broker decodes all live ports in one batch.
- The loaded policy supplies its context length, prediction length, execution stride, committed prefix, and action width. The worker does not contain a four-frame constant.
- Worker and inference sizes come from the process CPU affinity. The automatic wave size is the nearest power of two, with ties rounded up and no maximum cap.
- Trajectory transposition runs in each Session worker. Dolphin gets 0.25 seconds to accept SIGTERM, then Session uses its existing hard-kill and replay-repair path.

## Before and after

Host: 12 allowed CPUs, RTX 3060, Python 3.14.4. Dolphin used EXI input, fast-forward, blocking input, and uncapped emulation speed.

| Probe | Old thread driver | Spawned process driver |
|---|---:|---:|
| 12 boots, neutral policy | 249.5 lockstep fps | 397 lockstep fps steady |
| 16 boots, neutral policy | not measured | 307 lockstep fps steady |
| 30 boots, neutral policy | 96.1 lockstep fps | not used as one wave on this 12-CPU host |
| Dolphin shutdown grace | 10.0 s | 0.25 s |

The 12-worker figure comes from the interval between the 600-frame and 1,200-frame progress records: 600 frames in 1.511 seconds, or 397.1 lockstep fps. The 16-worker figure is 600 frames in 1.954 seconds, or 307.1 lockstep fps. The automatic policy selects 16 on this host because 12 is equally distant from 8 and 16, and ties go up.

These are steady rollout rates after all Dolphin processes reach the game. They include Session stepping, libmelee parsing, canonical flattening, controller conversion, shared observation writes, fixed-message IPC, broker dispatch, and a configurable neutral action-chunk policy. They exclude CUDA work because that benchmark is intended to isolate the non-GPU ceiling.

End-to-end short-run rates are lower because Dolphin startup dominates. The 12-worker, 1,200-frame run completed at 79.7 lockstep fps when process spawn, menu navigation, rollout, trajectory transfer, and teardown were all included. The last Dolphin reached the game about ten seconds after the command began. Long instant-restart evaluations amortize that one-time boot cost over many matches.

Reproduce the probe with:

```text
uv run experiments/benchmark_closed_loop.py --workers 12 --frames 1200
uv run experiments/benchmark_closed_loop.py --workers 16 --frames 1200
```

## Why the old architecture slowed down linearly

The old driver had one coordinator process:

```text
                         one Python process
  +----------------------------------------------------------------+
  | main thread: build observations -> policy -> collect futures    |
  |                                                                |
  | thread 0: Session.step -> socket receive -> parse -> dict       |
  | thread 1: Session.step -> socket receive -> parse -> dict       |
  | ...                                                            |
  | thread N: Session.step -> socket receive -> parse -> dict       |
  +----------------------------------------------------------------+
        |             |                                      |
     Dolphin 0     Dolphin 1                              Dolphin N
```

The socket receive releases the GIL. The event parser, canonical-frame builder, observation builder, ring update, and controller conversion do not. More Session threads therefore added more serialized Python work to the same interpreter. A fixed-work parser test had flat throughput from one to twelve threads.

That GIL effect was real, but the first estimate of 1.4 ms per boot was wrong. Direct measurements were:

| Old-path component | CPU time per boot and frame |
|---|---:|
| libmelee event parse | 0.125 ms |
| canonical dictionary conversion | 0.065 ms |
| all Session.step CPU work | 0.318 ms |

The corrected model for the old 30-boot loop was:

- Emulator and Session coordinator: 10.8 ms per lockstep frame at 92.6 fps.
- Policy-side CPU and GPU: about 7.0 ms per frame, amortized over a four-frame stride.
- Predicted total: about 17.8 ms per frame, or 56 fps.
- Measured full random-weight policy: 53.7 fps after compilation.

The costs were added because the old coordinator performed them in sequence. Dolphin workers returned to the coordinator, then the coordinator built policy input and ran inference, then it submitted the next Session steps.

## Current process architecture

```text
                           MAIN PROCESS
  +---------------------------------------------------------------------+
  | inference broker                                                    |
  |                                                                     |
  |  receive 64-byte PLAN_REQUEST records                               |
  |       |                                                             |
  |       v                                                             |
  |  zero-copy views of numeric observation rows                        |
  |       |                                                             |
  |       v                                                             |
  |  policy ring update -> packed H2D -> one batched decode              |
  |       |                                                             |
  |       v                                                             |
  |  write double-buffered action plans -> PLAN_READY records           |
  +-------------------------------+-------------------------------------+
                                  |
             one parent-owned POSIX shared-memory arena
                                  |
        +-------------------------+--------------------------+
        |                         |                          |
  +-----v---------+         +-----v---------+          +-----v---------+
  | worker 0      |         | worker 1      |   ...    | worker N      |
  |               |         |               |          |               |
  | Session       |         | Session       |          | Session       |
  | libmelee      |         | libmelee      |          | libmelee      |
  | flatten row   |         | flatten row   |          | flatten row   |
  | controller    |         | controller    |          | controller    |
  | trajectory    |         | trajectory    |          | trajectory    |
  +-----+---------+         +-----+---------+          +-----+---------+
        |                         |                          |
     Dolphin 0                 Dolphin 1                  Dolphin N
```

Each worker has its own interpreter and GIL. Session CPU work therefore runs on separate CPUs. The broker waits for one request from each live worker at a plan boundary and sends all live slots through one GPU batch.

The policy runtime contract is:

```text
L = context_frames
P = prediction_frames
S = execution_stride
D = committed_frames
A = action_dim
```

For the present experiment the values are `L=128`, `P=4`, `S=4`, `D=0`, and `A=14`. These values come from `RecedingHorizon.runtime_spec`. A policy with a different valid `P`, `S`, or `D` changes worker scheduling without a driver edit.

At bootstrap, a worker publishes one real observation paired with a neutral action. It waits for a plan, executes the first `S` actions, publishes the `S` resulting observations, and requests the next plan. On an instant-restart frame-id reset, it discards the old plan, publishes the reset row with a neutral producing action, and requests a fresh plan immediately.

For self-play, one Session worker owns one Dolphin and both controllers:

```text
                    one Session worker
       +------------------------------------------+
       | parse one shared game state              |
       | flatten it once                          |
       | publish slot (match, port 1) in row 2i   |
       | publish slot (match, port 2) in row 2i+1 |
       | wait for both plans                      |
       | step both controllers together           |
       +------------------------------------------+
                           |
                        Dolphin i
```

Twelve self-play Dolphins therefore produce 24 real model rows. Experiment 026 pads those rows to its batch-32 compiled inference program. The observation payload is shared between the two local slots; only the producing action and slot identity differ.

## IPC format

Cap'n Proto is not useful for the hot path here. It would still need an ownership protocol for large arrays, and its schema traversal would add work to a fixed record that is only 64 bytes.

The control wire is a versioned little-endian struct:

```text
64 bytes total

magic | version | message type | worker | flags
task generation | task id | row sequence | auxiliary sequence
count | plan slot | model port | status | reserved
```

The receiver checks message length, magic, version, type, and the reserved field. It uses `send_bytes` and `recv_bytes_into`, not object send/receive.

Bulk live data stays in one parent-owned shared-memory allocation:

```text
observation generation     uint64 [worker, ring]
frame id                   int32  [worker, ring]
reset flag                 uint8  [worker, ring]
observation floats         float32[worker, ring, float column]
observation integers       int32  [worker, ring, integer column]
producing action            float32[worker, ring, A]
optional action tokens      int32  [worker, ring, token group]
action plans, buffer 0/1    float32[worker, 2, P, A]
```

The observation schema is generated from the MDS schema. It has one ordered column list and one SHA-256 layout identity. A worker writes payload fields first and publishes the absolute generation last. The broker rejects a ring row if its stored generation does not match the requested generation. Plans use two buffers so the broker does not overwrite the plan a worker is executing.

Final trajectories do not use pickle. Each worker transposes its local frame segments, writes one exact-size shared-memory result slab, sends `RESULT_READY`, and waits. The parent copies the columnar arrays into the returned `Trajectory` objects, unlinks the result slab, and sends `RESULT_RELEASED`. This is a cold boundary operation and it runs independently for each worker.

## Compilation stalls

The old adaptive compiled path selected buckets 1, 2, 4, 8, 16, and 32 as matches ended at different times. Torch compiled a trunk and decoder on the first use of each bucket. The observed first calls were:

| Real rows | Bucket | First call | Second call |
|---:|---:|---:|---:|
| 30 | 32 | 14.24 s | 0.027 s |
| 1 | 1 | 13.59 s | 0.017 s |
| 3 | 4 | 19.52 s | 0.019 s |
| 5 | 8 | 17.92 s | 0.021 s |
| 9 | 16 | 17.91 s | 0.022 s |

Those first-use stalls totaled 83.2 seconds. They caused the 45-to-55-second match-boundary cliffs.

The new compiled path uses the smallest precompiled hardware bucket that covers the active wave and pads finished rows. The bucket is not fixed at 32:

```text
allowed CPUs 12 -> automatic wave 16 -> compiled bucket 16
allowed CPUs 32 -> automatic wave 32 -> compiled bucket 32
allowed CPUs 96 -> automatic wave 128 -> compiled bucket 128
```

There is no hard maximum. Explicit power-of-two overrides remain available.

Safety correction (2026-08-12): background inference compilation was removed repository-wide after it deadlocked training on both H100 and L40S hosts. CUDA compilation and training must not overlap across Python threads. Evaluation now keeps one inference engine over the live policy and compiles each required bucket/horizon synchronously on first use; later evaluations reuse those programs with the policy's updated parameter storage.

On the RTX 3060, the steady decode measurements were:

| Batch and horizon | Decode time | Executed action frames/s |
|---|---:|---:|
| batch 12, horizon 4 | 7.525 ms | 6,378 |
| batch 16, horizon 4 | 7.542 ms | 8,485 |
| batch 32, horizon 4 | 9.88 ms | 12,955 |

Inductor `default` and `reduce-overhead` were within measurement noise for this model. Evaluation evidence records the selected wave, compiled bucket, compile mode, and maximum decode duration.
The added hardware and bucket fields raise the match-row evidence schema to version 6.

## Checkpoint self-play benchmark

The requested production-checkpoint benchmark used:

```text
checkpoint: runs/260810-071709_026_temporal_mtp_.../final.pt
checkpoint SHA-256: 22333d1d61d6b648c757f0f1f3e887925fbb12a08fdffd1cb4ae72d6d6f2ef88
workers: 12 Dolphin processes on 12 allowed CPUs
model slots: 24, two per Dolphin
compiled inference bucket: 32
execution horizon: 4
frame cap: 14,400 per match
match mode: one normal head-to-head match per boot
```

All 12 boots completed with no worker failure. Normal games can end before the cap, so the run captured 106,814 frames rather than `12 * 14,400`. It took 132.65 seconds after compilation. This is 805.2 aggregate emulator-frames per second, or 67.1 frames per second averaged across the 12 original boots, including startup, finished games, result transfer, and teardown.

While all 12 games were active, each 600-frame interval took approximately 4.47 seconds. The steady 12-way lockstep rate was therefore approximately 134 fps. As games ended, the live row count fell from 24 to 2 and some intervals reached approximately 210 fps. The fixed batch-32 program continued to run; the savings came from fewer active Session workers and less policy ingestion work.

The wall-clock phase counters were:

| Broker phase | Seconds | Share of 131.91 s broker run |
|---|---:|---:|
| Wait for worker control messages | 66.57 | 50.5% |
| Policy ingestion, context build, transfer, and decode | 60.95 | 46.2% |
| Shared observation reads | 0.67 | 0.5% |
| Shared plan writes and replies | 1.51 | 1.1% |
| Shared result reads | 0.02 | <0.1% |
| Other broker and process lifecycle work | 2.19 | 1.7% |

The 66.57 seconds of control wait includes 20.18 seconds before the first complete batch, when the 12 Dolphins spawned and navigated menus. It also includes Session stepping, match-end trajectory conversion, and final worker exit. Policy work includes 49.85 seconds inside the measured decode boundary. The difference, 11.10 seconds, is ring ingestion, context stacking, host-to-device setup, and policy bookkeeping.

Decode timing across 3,600 replans was 13.51 ms median, 14.28 ms at p95, and 14.71 ms at p99. One first-live-shape call took 1.37 seconds. The separate prewarm took 6.91 seconds in this later process, compared with 22.80 seconds in the first cold benchmark process. The reduction shows that Inductor reused its disk cache, but loading and initializing cached kernels still has a cost.

The result proves why full policy speed is near 134 fps instead of 240 fps. After startup, the strict lockstep broker does two dependent phases:

```text
Session workers execute four frames  ->  broker builds and decodes next plans
          about 13 ms                +            about 17 ms
```

This is approximately 30 ms per four-frame stride, or 133 four-frame strides per second. The observed rate matches that arithmetic. A 240 fps target permits only 16.7 ms for the complete four-frame stride. The median decode alone consumes 13.5 ms, so small IPC changes cannot reach 240 fps.

Reproduce the benchmark with:

```text
uv run experiments/026_temporal_mtp.py \
  --self-play-eval runs/260810-071709_026_temporal_mtp_mtp026-d384-L8-h6-Lc128-t128x2-o1-2-3-4-5-6-9-12-16-20-s4-base_ranked-anon-1_production-seed0-d384-b512/final.pt \
  --self-play-matches 12 \
  --self-play-frames 14400
```

The machine-readable metrics and replays are in `self_play_benchmark_12x14400_single_match_run02` under the checkpoint run directory.

## What remains

The non-GPU Session path has exceeded the 240 fps target when measured alone. It is no longer the only limit on full-policy evaluation.

The next full-policy ceiling is the serial broker work between Session strides:

- policy ring ingestion and feature transforms;
- window stacking;
- host-to-device transfer;
- quantization and sampling setup;
- output synchronization and CPU conversion.

The production-checkpoint benchmark now measures this interval. Non-decode policy work was 11.10 seconds across 3,600 replans, or 3.08 ms per replan on average. The full policy phase averaged 16.93 ms per replan. Shared-memory request reads and plan writes added another 0.61 ms per replan.

The largest remaining architectural gain is to pipeline worker groups. The present broker waits for every live Session slot, decodes one batch, sends every plan, and only then lets all Sessions advance. Two or more groups could execute Session work while another group uses the GPU. This can overlap the approximately 13 ms Session phase with the approximately 17 ms policy phase. It trades some inference batch size for overlap, so it needs a measured group-size search on each host.

After pipelining, the next targets are the 13.5 ms median batch-32 decode and the 3.1 ms policy preparation interval. Reaching 240 fps needs the complete four-frame cycle below 16.7 ms. It will probably require overlap plus faster inference, not an IPC serializer change.

Other remaining work is narrower:

- `flatten_canonical_frame` still constructs a local dictionary inside each worker. It does not cross IPC, and it runs in parallel, but a direct canonical-frame-to-shared-row encoder can remove that allocation.
- The optional Instant Match Gecko restart was unstable in one 12-way model-versus-itself stress run: one Dolphin emitted invalid memory reads and then timed out after approximately 9,000 frames. Standard one-match head-to-head mode is clean. Instant restart needs a separate Dolphin/Gecko investigation before it is the default for self-play.
- The process worker now skips policy publication for the terminal menu frame. It still retains that frame in the trajectory. A regression test covers terminal frames that omit live player position fields.
- Short evaluations still pay Dolphin process spawn and menu navigation. Persistent worker pools across evaluation calls would remove repeated boots, but they need a clear checkpoint and replay-directory lifecycle.
- Result slabs are zero-copy between processes, but the parent copies them into durable NumPy arrays before unlinking shared memory. This is intentional cold-path ownership. A longer-lived result arena could remove that copy if scoring can consume borrowed buffers safely.
