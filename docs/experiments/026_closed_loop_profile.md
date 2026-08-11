# Experiment 026 closed-loop profile and implementation

Date: 2026-08-10

## Outcome

Closed-loop evaluation had three independent costs:

1. Torch compiled new inference shapes inside the live loop.
2. All Session CPU work ran as threads in the coordinator process and shared one GIL.
3. Match conversion and a ten-second Dolphin shutdown wait stopped progress at wave boundaries.

The implementation removes the first two causes and bounds the third:

- Training starts a separate no-grad inference replica and compiles the scheduled inference programs in the background.
- Each Dolphin Session now runs in a spawned Python process. The main process only owns the policy and GPU.
- Observations, action plans, and final trajectories use typed shared memory. Hot control messages are fixed 64-byte binary records. The hot path does not use pickle.
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

Training constructs a separate inference replica after the first training step. A background thread runs its static inference programs under `torch.no_grad()` on a dedicated CUDA stream while later training steps continue. It uses Inductor `default` mode because CUDA graph capture conflicted with concurrent trainer synchronization. Before evaluation, the latest training weights are copied into the replica's existing parameter storage. The compiled program is code specialized to tensor shapes and operations; it is not a second set of frozen weights.

On the RTX 3060, the steady decode measurements were:

| Batch and horizon | Decode time | Executed action frames/s |
|---|---:|---:|
| batch 12, horizon 4 | 7.525 ms | 6,378 |
| batch 16, horizon 4 | 7.542 ms | 8,485 |
| batch 32, horizon 4 | 9.88 ms | 12,955 |

Inductor `default` and `reduce-overhead` were within measurement noise for this model. Evaluation evidence records the selected wave, compiled bucket, compile mode, maximum decode duration, total prewarm time, and the time the first evaluation still had to wait.
The added hardware and bucket fields raise the match-row evidence schema to version 6.

## What remains

The non-GPU Session path has exceeded the 240 fps target. It is no longer the only limit on full-policy evaluation.

The next full-policy ceiling is the serial broker work between Session strides:

- policy ring ingestion and feature transforms;
- window stacking;
- host-to-device transfer;
- quantization and sampling setup;
- output synchronization and CPU conversion.

The old measurement assigned about 4.5 ms per frame to these non-decode policy operations at 30 slots. The process implementation moves canonical flattening and controller conversion to workers, but the model-specific ring transforms, stacking, transfer, and output synchronization remain in the broker. A complete GPU-enabled benchmark is still required to split this new broker interval and report the final full-policy speedup. The GPU driver was not available during the final CPU benchmark.

Other remaining work is narrower:

- The spawned driver currently supports one model-controlled port per match, which is the experiment-026 versus-CPU protocol. The legacy driver still covers two-model-port self-play. A future shared arena can assign multiple model slots to one Session worker.
- `flatten_canonical_frame` still constructs a local dictionary inside each worker. It does not cross IPC, and it runs in parallel, but a direct canonical-frame-to-shared-row encoder can remove that allocation.
- Short evaluations still pay Dolphin process spawn and menu navigation. Persistent worker pools across evaluation calls would remove repeated boots, but they need a clear checkpoint and replay-directory lifecycle.
- Result slabs are zero-copy between processes, but the parent copies them into durable NumPy arrays before unlinking shared memory. This is intentional cold-path ownership. A longer-lived result arena could remove that copy if scoring can consume borrowed buffers safely.
