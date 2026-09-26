# Inference runtime

HAL has two gameplay execution modes: synchronous local evaluation and
nonblocking netplay. Both use `PredictionPolicy.predict()` and `ActionScheduler`.
The model microbenchmark calls the same prediction operation directly.

## Ownership and imports

| Module | Owner |
| --- | --- |
| `hal/sim/session.py` | Local Dolphin startup, controller submission, and blocking advancement |
| `hal/sim/netplay.py` | Netplay setup, controller submission, and independent observation reads |
| `hal/eval/observations.py` | Canonical game observations and required-field projection |
| `hal/eval/scheduling.py` | `FrameTiming` and one `ActionScheduler` per controlled port |
| `hal/eval/local.py` | Synchronous match loop, including two local model ports |
| `hal/eval/policy.py` | The same scheduling contract at the vector evaluator boundary |
| `hal/eval/netplay.py` | Nonblocking match loop |
| `hal/inference/api.py` | `PredictionRequest`, `ActionPlan`, and `PredictionPolicy` |
| `hal/inference/worker.py` | `InferenceClient`, `InferenceWorker`, and their lifecycle |
| `hal/inference/backends/history_decoder/policy.py` | `HistoryDecoderPolicy`, model history, sampling, and KV state |
| `hal/inference/benchmark.py` | Timed request delivery at a declared model shape |
| `hal/eval/qualification.py` | Startup inference budget checks and schedule selection |
| `hal/netplay_service/runner.py` | Reservations, process supervision, health, and replay publication |

Deployment and CLI code depend on evaluation. Evaluation joins simulation and
inference. Simulation does not import Torch; model code does not import Melee.
Neither evaluation nor inference imports deployment service code.

Local sessions use `step(inputs)`. Netplay uses `submit(inputs)` and
`read_frames()`. These mechanics stay explicit; the loops share observation
construction, action scheduling, model prediction, and result types.

## Frame convention

An observation at frame `f` includes the controller action that produced that
frame. An input submitted after observing `f`, with input delay `d`, applies at
`f + d + 1`.

`FrameTiming` has four independent values:

- `input_delay_frames`: additional controller submission-to-application delay.
- `thinking_allowance_frames`: game frames allowed for inference before using its new plan.
- `replan_interval_frames`: spacing between prediction requests.
- `prediction_horizon_frames`: last predicted offset from the source observation.

For thinking allowance `a`, the request fixes actions at `f + 1` through `f + a + d`.
Its first new prediction applies at `f + a + d + 1`. The horizon must cover this
fixed prefix and the replan interval. Any remaining predictions form the reserve.
For example, input delay 2, thinking allowance 1, replan interval 4, and horizon 8
fix actions through `f + 3`. The model predicts `f + 4` through `f + 8`; the last
action is one frame of reserve. A result ready by the `f + 1` submission can
control `f + 4`, so its measured latency must be at most one frame (16.7 ms).

In synchronous local execution, waiting for inference consumes wall time but not
game frames. The thinking allowance can therefore be zero. Local input delay is explicitly
emulated by the controller transport queue; netplay input delay is applied by
Slippi. These mechanisms must not both delay a netplay submission.

## Prediction and state

A request identifies a stream, generation, sequence, and source frame. Observed
history and fixed future actions are separate fields. An `ActionPlan` contains
only newly predicted actions, each with an absolute target frame. The response
does not echo the fixed prefix.

The backend owns observation history, KV state, and sampling state. It accepts
an initial observation batch, then only contiguous new observations. It rejects
missing frames and stale request identities. A new generation resets
that stream. The scheduler sends observations after the previous accepted request's
source frame. It keeps a bounded recent history for applied-action checks. If
unsent observations exceed that capacity, it fails instead of sending a truncated
history as a new start.

The scheduler owns committed inputs, plan handoff, deadlines, and fallback.
Early plans wait for handoff. Late plans retain only reachable target frames.
Actual applied actions are checked against both the fixed prefix and expired
predictions. Exhaustion submits neutral. Confirmed inference failure drains the
remaining plan before the netplay driver exits.

The client is an IPC endpoint, not a model. It permits one outstanding request.
The worker groups available requests and invokes `predict`; the backend controls
whether execution is actually batched on the GPU.

## Measurement and compatibility

`check_realtime_budget()` measures the declared request shape through the same
IPC worker implementation used by serving. Synthetic observations use distinct
dictionaries for each frame. The first request sends a full context; later requests
send only the observations since the previous request at the configured replan
interval. Warmup and measured calls are separate, and measured
calls forbid compilation. This startup check runs clients and the inference
thread in one process without Dolphin. It is not evidence of live netplay FPS;
full gameplay qualification is a separate integration run.

The runner writes `.budget.json` schema 3 with resolved timings, measurements,
checkpoint hash, seed, environment, and Git SHA. Slot health is schema 3 and
runner health is schema 4; older status files are rejected. The deployment shape
override is `--prediction-shape HORIZON PREFIX`, exposed by
`HAL_NETPLAY_PREDICTION_HORIZON` and `HAL_NETPLAY_PREDICTION_PREFIX`.
`--replan-interval` selects the independent request interval; the deployment
variable is `HAL_NETPLAY_REPLAN_INTERVAL`. For the 2/1/4/8 timing example, use
shape `8 3` and replan interval `4`.
Checkpoint bundle identifiers and formats are unchanged.

`hal-policy eval` and `hal-play` expose `--history-mode`, `--kv-update-frames`,
`--prediction-horizon`, `--thinking-allowance`, and `--replan-interval`. Local
evaluation defaults to zero thinking frames because Dolphin waits for inference.
Both commands can run the explicit 2/1/4/8 schedule without changing their
default prediction horizon or replan interval.

The model-only benchmark records its horizon, fixed prefix, replan interval, and
cache update size. Its schema 2 `prediction_ms` measures complete predictions;
it must not be compared directly with the earlier per-frame mixture of queued
and replanning calls.

`prepare_prediction()`, `reset_prediction()`, and `predict()` are the sole model
lifecycle and execution interface. There is no model `step()` compatibility path
or model-owned action queue. The scheduler owns actions waiting to be played.
Streaming KV inference intentionally retains information beyond the explicit
attention window. See [the accepted numerical contract](kv-cache.md).

Measured latency, gameplay throughput, numerical checks, and validation results
are recorded in [Runtime validation](runtime-validation.md).
