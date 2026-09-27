# Inference and evaluation

HAL serves the maintained 059 `ActionSequenceTransformer` through an
`ActionSequencePolicy`. A policy artifact carries the checkpoint and its
representation contract. Evaluation joins that policy to Dolphin; simulation
itself does not load Torch, and model imports do not load Melee.

## Artifacts and commands

Export a supported 059 checkpoint, then pass the artifact to local play,
evaluation, or H2H:

```bash
uv run hal-policy export /path/to/step.pt policy.hal
uv run hal-policy eval policy.hal --profile official-059 --n-matches 1
uv run hal-policy eval policy.hal --profile local --n-matches 1
uv run hal-play policy.hal OPPONENT#123 --online-delay 2
```

`official-059` runs the dense window control. `local` runs cached inference
with no fixed future action. `local-stride-one` is a separate local timing
measurement. `hal-play` reads the bot login from `HAL_SLIPPI_USER_JSON`, unless
`--user-json` is supplied. `--imitate` selects a connect code in the checkpoint
vocabulary, or `PLATINUM`, `DIAMOND`, or `MASTER`; it does not select the live
opponent. Local evaluation uses the checkpoint's exact p90 return target.

For explicit model-versus-model play, supply both artifacts:

```bash
uv run -m hal.scripts.h2h \
  --model-a.name A --model-a.artifact /path/to/a.hal \
  --model-b.name B --model-b.artifact /path/to/b.hal \
  --out-dir runs/h2h-example --profile local
```

The H2H command mirrors ports and records matchup results. Two players with
the same checkpoint can share weights while keeping independent stream state.
Different checkpoints are separate model allocations.

The pinned 96-boot level-9 CPU protocol has its own command:

```bash
HAL_GIT_SHA=<committed-source-sha> uv run scripts/eval_kv_cache.py --profile official-059
```

It uses 7,200 frames per boot, instant restart, the recorded matchup and
seed-stage order, and the checkpoint's p90 target. `cached-prefix-two` keeps
the official timing but changes the executor; `local` measures zero-prefix
cached play under a separate protocol name. The command records hardware and
raw outcomes. Its W&B baseline is historical evidence; a matched same-hardware
control and the full 96-boot run are still required for qualification.

Existing format-4/v5 checkpoints and current bundles retain their identities.
The artifact reader validates checkpoint tensors, configuration, corpus and
vocabulary identity, return calibration, and declared capability version.
The older bundle capability declares delay 2. New exports can declare the
supported local and netplay delays; loading an older bundle does not silently
change its delay contract.

## Request and frame contract

A `PolicyInput` is one observed frame plus the controller action that produced
it. A `PredictionRequest` names its stream, generation, sequence, source frame,
contiguous new observations, and exact fixed future actions. An `ActionPlan`
echoes those identifiers and returns only newly generated actions, each with
an absolute target frame. Requests contain no Dolphin object or GPU handle.

`FrameTiming` separates physical controller delay, inference allowance, fixed
prefix, replan interval, and prediction horizon. The profile values are:

| Profile | Execution | Physical delay | Allowance | Fixed prefix | Replan | Horizon | Reserve |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Official 059 | Dense window, blocking | 0 | 0 | 2 | 2 | 4 | 0 |
| Local default | Cached, blocking | 0 | 0 | 0 | 2 | 4 | 2 |
| Local stride one | Cached, blocking | 0 | 0 | 0 | 1 | 4 | 3 |
| Netplay delay 2 | Cached, asynchronous | 2 | 1 | 3 | 4 | 8 | 1 |
| Netplay delay 3 | Cached, asynchronous | 3 | 1 | 4 | 4 | 8 | 0 |

For local zero-delay play, an observation at frame `t` can produce the action
for `t+1`; Dolphin waits for inference. For delay-2 netplay, `t+1..t+3` are
already fixed and the model generates `t+4..t+8`. The action at `t+8` is one
frame of reserve. Delay-3 netplay has its own prefix-4 prepared shape.

One `ActionScheduler` owns each controlled port's committed actions, pending
request, plan validity, reserve use, and fallback counts. At controller choice
frame `f`, an action for target `t` can still be submitted exactly when
`t >= f + physical_delay + 1`. Equality is valid. A response must match the
active stream, generation, sequence, source frame, shape, and target frames.
The scheduler compares fixed actions at canonical controller wire precision.
If a generated action has missed its deadline, or the fixed prefix differs,
it rejects the whole generated plan and keeps the prior schedule. It never
uses a suffix conditioned on a missed generated action.

A valid response acknowledges its observed frames even when its plan is too
late. The next request starts after that consumed source frame, so the policy
does not ingest a duplicate observation. A malformed active response is a
protocol failure. Exhaustion and neutral fallback are counted separately from
controller-submission gaps.

Netplay currently ingests the first-seen speculative frame and ignores a
duplicate rollback frame. The completed replay supplies final reconciliation;
this is not confirmed-frame inference or live rollback correction.

## Runtime ownership

| Owner | Responsibility |
| --- | --- |
| `hal/models/action_sequence.py` | One neural implementation and the 059 controller vocabulary |
| `hal/representation/` | Offline/live features, observations, and player identity encoding |
| `hal/inference/action_sequence_artifact.py` | Checkpoint/bundle validation, export, and loading |
| `hal/inference/action_sequence_policy.py` | Dense or cached execution, per-stream history, cache, and sampling |
| `hal/inference/engine.py` and `client.py` | Ready-request batching and persistent local-process delivery |
| `hal/eval/scheduling.py` and `policy.py` | Per-port action scheduling and process-vector adapter |
| `hal/sim/` | Dolphin, controller transport, and replay capture |
| `hal/netplay_service/` | Reservations, workers, health, and replay publication |

The process vector harness collects bulk requests and sends complete action
chunks to independent Dolphin workers. A cached policy keeps a bounded KV ring
per stream. Compatible ready requests can use one batched trunk/decoder call;
other admitted streams do not need to be ready. Sampling counters and RNG are
stream-keyed, including when rows are reordered or padded. Preparation fixes
the checkpoint hash, execution mode, horizon, exact prefix, update shapes,
and cache capacity before admission. The engine groups ready requests by that
profile and waits at most 0.5 ms, shortened by the nearest deadline. A client
admits one stream generation, permits one outstanding request, releases its
cache row at match close, and has a one-second monotonic response timeout
independent of game-frame deadlines.

The model registry keys weight allocations by checkpoint hash, device, and
inference dtype. Real-service OpenGL gameplay, two-session admission on the
target GPU, recovery, and sparse-load capacity still need measured evidence.
The present deployment configuration does not prove that capacity. New local
and netplay profiles likewise need their own gameplay and timing results
before they can replace a validated profile.

The original KV-cache and runtime measurements remain in
[KV-cache evidence](kv-cache.md) and [runtime validation](runtime-validation.md)
with their recorded hardware and protocol labels. The current acceptance
record is [refactor-059.md](refactor-059.md).

`scripts/qualify_netplay_059.py` runs a separate service and a neutral-input
peer with two explicit Slippi accounts and separate ports. Run it under an X
display (for example, `xvfb-run -a`); accounts must not belong to another live
runner. `--graphics-backend OGL` selects OpenGL for both Dolphins; Vulkan is the
default. The default workload requires ten completed matches and 1,800 gameplay
seconds. It retains local replays and does not publish them.

The harness validates each match's bundle, source revision, graphics backend,
timing profile, frame sequence, latency samples, and scheduling counters. It
excludes the first 300 frames and the terminal menu interval from steady
measurements. Missing or
malformed evidence fails the run. Delivery p99 must remain below one frame;
delay-2 p95 must be at most 12 ms. Steady gameplay must reach 59.5 FPS, with no
skipped controller submissions or action-plan exhaustion after startup.

The schema-4 `run-result.json` reports measured failures and gates still
unmeasured. The harness checks the prepared engine's ready and final compile
and CUDA-graph counts; missing audit evidence fails the run. Resource stability
and matched control trials still need separate evidence. Deadline and prefix
rejection counts are reported separately; they do not mean invalid plans were
accepted. A short `--smoke-frames` run cannot pass the soak checks.
