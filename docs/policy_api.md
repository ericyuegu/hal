# Portable policy API

HAL policies consume flat numeric observations and produce one semantic GameCube
controller action. A backend owns its model, preprocessing, context, and sampling.
It does not expose model-specific tensors to the match loop.

```python
policy = load_policy("policy.halpolicy", device="cuda")
runtime = RuntimeConfig(max_batch_size=1, transport_delay_frames=2)
policy.prepare(runtime)  # Load and compile before Dolphin starts.
outputs = policy.step(inputs)
```

Each `PolicyInput` has three action-time fields:

```text
observed state                         future states
     S_t              S_t+1              S_t+2              S_t+3
      │                  │                  │                  │
applied_action      pending[0]         pending[1]         new output
```

For Slippi delay 2, the policy observes `S_t`, conditions on the real actions
already committed for `S_t+1` and `S_t+2`, and predicts the action for `S_t+3`.
A Dolphin step submits that prediction and returns `S_t+1`; the prediction first
appears two returned frames later. Delay 3 is the same rule with three pending
actions and a new output for `S_t+4`.

Local CPU evaluation implements the same delay with a software queue. The EXI
Dolphin applies one submitted input to the next returned frame. HAL waits for
that frame before it flushes another input, so one policy call maps to one game
frame.

Export a supported training checkpoint once, then use only the portable bundle:

```text
hal-policy export r2://hal/runs/<run>/checkpoints/<step>.pt policy.halpolicy
hal-play policy.halpolicy <opponent-code>
hal-policy eval policy.halpolicy --transport-delay 2
```

`hal-play` reads the bot login from `HAL_SLIPPI_USER_JSON`. `--user-json`
overrides it. `--imitate` accepts an exact connect code or `PLATINUM`,
`DIAMOND`, or `MASTER`. It conditions policy behavior and does not affect
matchmaking. Its defaults are Fox, `IBDW#0`, and Slippi delay 2.

The pinned Slippi 3.6.4 session uses Vulkan. Its default OpenGL renderer can
stall CUDA inference when Dolphin and the policy share an NVIDIA GPU.
