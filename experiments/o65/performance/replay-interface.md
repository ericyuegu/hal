# Window replay interface

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
Flatten W and K to call `model.self_play_value`.

The approved BF16 decoder commits through `8decbe99` were already ancestors of
this branch's base `aacdfee0`; no cherry-pick was needed.
