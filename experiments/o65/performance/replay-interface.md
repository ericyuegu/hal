# Window replay interface

Qualification is in progress. The first compiled BF16 proxy trial failed the
KL precision gate while other measured quantities passed. Keep the trainer on
its existing path until the replay report records a passing configuration.
The follow-up trial uses `options={"triton.cudagraphs": False,
"emulate_precision_casts": True}`. The latter preserves BF16 rounding between
fused operators; without it Torch 2.11 produces different gradient/no-gradient
distributions and a false KL cost. The common-input diagnostic now agrees
across gradient modes, and the full FP32-reference gate remains pending.

The replay stream owns `hal.models.window_replay`. The interface commit defines
`ReplaySlots`, `ReplayCriticFeatures`, and `ReplayOutput`. All are frozen, slotted
dataclasses. Tensor suffixes state the dimensions; their docstrings state order
and dtypes. `W` and `K` stay fixed within a compiled or captured call.

The implementation provides:

```python
validate_replay_inputs(model, hidden_WLd, observed_WL4, slots)
replay_windows(model, hidden_WLd, observed_WL4, slots, *, grad: bool)
```

Validate at the input assembly boundary, before compilation or graph capture.
`hidden_WLd` is the trunk output, computed once per window. Observed actions and
all queues use codec order: buttons, main stick, C-stick, triggers. Conditions
are already normalized by `model.ranges.normalize`. `history_starts_WL` gives
the first visible frame at each history position, including episode boundaries.
Use zero for windows with one segment and no left padding.

`positions_WK` locates each decision's observation in its window. `valid_WK`
identifies occupied slots. `executed_WKF` is the terminal execution mask in
frame-major decode order: C-stick, main stick, triggers, buttons. It has 16
factors even for R=2. `sampled_actions_WK44` contains the sampled path in codec
order. `forced_queue_WKD4.shape[2]` fixes max_delay; it must leave four padded
offsets within the checkpoint's decoder capacity. Executed offsets D+1..D+R
must all belong to `model.head_offsets`.

`ReplayOutput.log_probs_WKF` and `.distributions` use decode order, as the
experiment's `PolicyEvaluation` does. The four distributions have shapes
`[W,K,4,V_group]`. `factor_mask_WKF` intersects valid slots, cadence and the
execution mask. Every policy or KL reduction must use that mask. Critic
features are detached, including z at D+1 before the first sampled factor.
Flatten W and K to call `model.self_play_value`. Use `slots.valid_WK` for critic
fitting, including decisions with zero executed factors. Do not derive the
critic mask from `factor_mask_WKF`.

## Actor capture

`hal.models.replay_actor` provides `ActorTargets`, `ActorLoss`,
`validate_actor_targets`, and `actor_loss`. Targets contain stored behavior
factor log-probs, raw advantages, selection, lambda, reference probabilities,
the common advantage mean/sigma, and a scalar denominator. Use the local
selected decision count on one device. With DDP, use global count / world
size, because DDP averages gradients. This boundary has no importance weight.
An empty local rank contributes zero loss when the global count is positive.
The PPO clip is 0.2 and direct KL is weighted by lambda / sigma.

`hal.inference.graphed_actor.GraphedActorStep` takes a forward callback, a tuple
of tensor inputs, targets, and `model.actor_parameters()`. The callback must
rebuild the trunk and call `replay_windows(..., grad=True)` from the supplied
static tensor inputs. Include all changing slot tensors in that tuple; do not
close over decision data that changes between calls. Compile the callback
with `fullgraph=True`, `dynamic=False`, and `options={"triton.cudagraphs": False}`
(default mode), or use `mode="max-autotune-no-cudagraphs"`. Compiled flex
attention requires head width at least 16; the proxy uses 64.

Release earlier eager autograd graphs before constructing capture buckets.
Torch 2.11 retains the original stream on their leaf gradient nodes. Capture
owns fixed gradient storage and rejects replacement of that storage. Run
`step.backward(inputs, targets)`, then clipping and Muon outside the graph.
Do not call `zero_grad(set_to_none=True)` after preparation. Backward clears
the captured gradients itself. Returned loss tensors are copies, so later
replays cannot overwrite an earlier result. Validate each new input payload
at assembly time before entering the captured hot path.

The graph helper contains no experiment import. Its small objective is a
tested reimplementation at this model boundary; experiment losses were not
moved or imported. The dense stream still owns trainer integration, selection,
stored behavior probabilities, critic fitting, and persisted format changes.

The approved BF16 decoder commits through `8decbe99` were already ancestors of
this branch's base `aacdfee0`; no cherry-pick was needed.
