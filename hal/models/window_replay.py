"""Replay fixed decision slots with one history memory per window."""

from dataclasses import dataclass
from dataclasses import fields

import torch
from torch import Tensor

from hal.models.controller_codec import BUTTONS_GROUP
from hal.models.controller_codec import CONTROLLER_DECODE_ORDER
from hal.models.controller_codec import CONTROLLER_GROUP_INDEX
from hal.models.controller_codec import TRIGGERS_GROUP
from hal.models.self_play import SelfPlayTransformer
from hal.models.self_play import validate_executed_offsets


@dataclass(frozen=True, slots=True)
class ReplaySlots:
    """W windows, K slots, L history frames, D forced frames, Q opponent frames.

    Controller tensors use codec order (buttons, main stick, C-stick, triggers).
    F=16 uses frame-major decode order (C-stick, main stick, triggers, buttons).
    Conditions are normalized [log lambda, delay, replan]. Invalid slots and
    unexecuted factors never enter a loss reduction. All indices are int64,
    masks bool, and conditions float32. Validate once before compiled replay.
    """

    positions_WK: Tensor
    valid_WK: Tensor
    history_starts_WL: Tensor
    forced_queue_WKD4: Tensor
    delay_WK: Tensor
    replan_WK: Tensor
    condition_WK3: Tensor
    opponent_condition_WK3: Tensor
    opponent_queue_WKQ4: Tensor
    opponent_lengths_WK: Tensor
    sampled_actions_WK44: Tensor
    executed_WKF: Tensor


@dataclass(frozen=True, slots=True)
class ReplayCriticFeatures:
    """Detached critic inputs per [W, K] slot; d/t are trunk/decoder widths."""

    trunk_WKd: Tensor
    decoder_WKt: Tensor
    condition_WK3: Tensor
    opponent_condition_WK3: Tensor
    opponent_queue_WKQ4: Tensor
    opponent_lengths_WK: Tensor


@dataclass(frozen=True, slots=True)
class ReplayOutput:
    """FP32 factor logs [W,K,16], group probabilities [W,K,4,V_group].

    Both outputs use decode order. Log-probs are zero outside executed_WKF and
    valid_WK. Distributions retain the normalized conditional support, including
    unexecuted sampled frames; reductions must use factor_mask_WKF.
    """

    log_probs_WKF: Tensor
    distributions: tuple[Tensor, ...]
    factor_mask_WKF: Tensor
    features: ReplayCriticFeatures


def _check_shapes(model: SelfPlayTransformer, hidden: Tensor, observed: Tensor, slots: ReplaySlots) -> None:
    if hidden.ndim != 3 or hidden.shape[-1] != model.cfg.d_model or min(hidden.shape[:2]) < 1:
        raise ValueError("window hidden states must be nonempty [W,L,d_model]")
    windows, length, _ = hidden.shape
    if length > model.cfg.L_ctx or observed.shape != (windows, length, 4):
        raise ValueError("observed window actions must be [W,L,4] within context capacity")
    if slots.positions_WK.ndim != 2 or slots.positions_WK.shape[0] != windows or slots.positions_WK.shape[1] < 1:
        raise ValueError("slot positions must be nonempty [W,K]")
    shape = tuple(slots.positions_WK.shape)
    if slots.forced_queue_WKD4.ndim != 4 or slots.forced_queue_WKD4.shape[:2] != shape:
        raise ValueError("forced queue must be [W,K,max_delay,4]")
    max_delay = slots.forced_queue_WKD4.shape[2]
    expected = {
        "positions_WK": shape,
        "valid_WK": shape,
        "history_starts_WL": (windows, length),
        "forced_queue_WKD4": (*shape, max_delay, 4),
        "delay_WK": shape,
        "replan_WK": shape,
        "condition_WK3": (*shape, 3),
        "opponent_condition_WK3": (*shape, 3),
        "opponent_queue_WKQ4": (*shape, model.critic_residual.max_queue, 4),
        "opponent_lengths_WK": shape,
        "sampled_actions_WK44": (*shape, 4, 4),
        "executed_WKF": (*shape, 16),
    }
    if max_delay + 4 > model.cfg.sample_chunk_length:
        raise ValueError("padded replay exceeds decoder offset capacity")
    for field in fields(slots):
        tensor: Tensor = getattr(slots, field.name)
        dtype = torch.bool if field.name in ("valid_WK", "executed_WKF") else torch.int64
        if field.name in ("condition_WK3", "opponent_condition_WK3"):
            dtype = torch.float32
        if tuple(tensor.shape) != expected[field.name] or tensor.dtype != dtype or tensor.device != hidden.device:
            raise ValueError(f"{field.name} has the wrong shape, dtype or device")
    if observed.dtype != torch.int64 or observed.device != hidden.device or not hidden.is_floating_point():
        raise ValueError("observed actions must be int64 on the hidden states' device")


def validate_replay_inputs(
    model: SelfPlayTransformer, hidden_WLd: Tensor, observed_WL4: Tensor, slots: ReplaySlots
) -> None:
    """Validate data at assembly time, outside compiled/captured replay.

    Invalid slots and queue padding can contain arbitrary values. Live slots
    must use supported offsets, legal actions and a prefix execution mask.
    """
    _check_shapes(model, hidden_WLd, observed_WL4, slots)
    if any(parameter.dtype != torch.float32 for parameter in model.temporal.parameters()):
        raise ValueError("window replay requires FP32 decoder parameters")
    valid = slots.valid_WK
    length = hidden_WLd.shape[1]
    positions = slots.positions_WK[valid]
    if bool(((positions < 0) | (positions >= length)).any()):
        raise ValueError("slot position is outside its window")
    starts = slots.history_starts_WL
    frame = torch.arange(length, device=hidden_WLd.device)
    if bool(((starts < 0) | (starts > frame)).any()) or bool((starts[:, 1:] < starts[:, :-1]).any()):
        raise ValueError("history starts must be causal and nondecreasing")
    delay, replan = slots.delay_WK[valid], slots.replan_WK[valid]
    if bool(((delay < 0) | (delay > slots.forced_queue_WKD4.shape[2]) | ((replan != 2) & (replan != 4))).any()):
        raise ValueError("slot delay or replan exceeds replay capacity")
    configurations = tuple(sorted(set(zip(delay.tolist(), replan.tolist(), strict=True))))
    if configurations:
        validate_executed_offsets(model.head_offsets, configurations)
    condition = slots.condition_WK3[valid]
    opponent = slots.opponent_condition_WK3[valid]
    for value in (condition, opponent):
        if bool((~torch.isfinite(value) | (value < -1) | (value > 1)).any()):
            raise ValueError("slot conditions must be normalized to [-1,1]")
    expected = model.ranges.normalize(
        torch.full_like(delay, model.ranges.log_lambda[0], dtype=torch.float32), delay, replan
    )
    if not torch.equal(condition[:, 1:], expected[:, 1:]):
        raise ValueError("slot delay and replan disagree with policy conditioning")
    opponent_lengths = slots.opponent_lengths_WK[valid]
    if bool(((opponent_lengths < 0) | (opponent_lengths > model.critic_residual.max_queue)).any()):
        raise ValueError("opponent queue length exceeds its capacity")
    executed = slots.executed_WKF[valid].reshape(-1, 4, 4)
    frame_executed = executed[..., 0]
    if (
        bool((executed != frame_executed[..., None]).any())
        or bool((~frame_executed[:, :-1] & frame_executed[:, 1:]).any())
        or bool((frame_executed & (torch.arange(4, device=delay.device) >= replan[:, None])).any())
    ):
        raise ValueError("executed factors must be a prefix of complete frames within R")
    rows = torch.arange(hidden_WLd.shape[0], device=delay.device)[:, None].expand_as(valid)[valid]
    group_sizes = torch.tensor(model.codec.group_vocabs, device=delay.device)
    for actions in (
        observed_WL4[rows, positions],
        slots.forced_queue_WKD4[valid][
            torch.arange(slots.forced_queue_WKD4.shape[2], device=delay.device) < delay[:, None]
        ],
        slots.sampled_actions_WK44[valid][torch.arange(4, device=delay.device) < replan[:, None]],
        slots.opponent_queue_WKQ4[valid][
            torch.arange(model.critic_residual.max_queue, device=delay.device) < opponent_lengths[:, None]
        ],
    ):
        if bool(((actions < 0) | (actions >= group_sizes)).any()):
            raise ValueError("replay contains an invalid controller factor")
        if bool(
            model.codec.button_mask(actions[..., TRIGGERS_GROUP]).gather(-1, actions[..., BUTTONS_GROUP, None]).any()
        ):
            raise ValueError("replay buttons are incompatible with triggers")
    if not bool(torch.isfinite(hidden_WLd).all()):
        raise ValueError("window hidden states must be finite")


def replay_windows(
    model: SelfPlayTransformer,
    hidden_WLd: Tensor,
    observed_WL4: Tensor,
    slots: ReplaySlots,
    *,
    grad: bool,
) -> ReplayOutput:
    """Replay previously validated slots with static shapes and shared history K/V.

    The caller runs the trunk once per window. With grad=True its autograd
    history is preserved. This function owns decoder precision, as the rollout
    decoder does. On Torch 2.11, compile in default mode with
    triton.cudagraphs=False and emulate_precision_casts=True before explicit
    CUDA capture. Cast preservation prevents gradient/no-gradient BF16 drift;
    remove it only after the proxy FP32-reference gate passes without it.
    Critic features never carry actor gradients.
    """
    if grad and not torch.compiler.is_compiling() and torch.is_inference_mode_enabled():
        raise ValueError("gradient replay requires autograd outside inference_mode")
    _check_shapes(model, hidden_WLd, observed_WL4, slots)
    valid = slots.valid_WK
    position = torch.where(valid, slots.positions_WK, 0)
    delay = torch.where(valid, slots.delay_WK, 0)
    replan = torch.where(valid, slots.replan_WK, 2)
    condition = torch.where(valid[..., None], slots.condition_WK3, 0)
    opponent_condition = torch.where(valid[..., None], slots.opponent_condition_WK3, 0)
    depth = torch.arange(slots.forced_queue_WKD4.shape[2], device=hidden_WLd.device)
    queue = torch.where((valid[..., None] & (depth < delay[..., None]))[..., None], slots.forced_queue_WKD4, 0)
    actions = torch.where(
        (valid[..., None] & (torch.arange(4, device=hidden_WLd.device) < replan[..., None]))[..., None],
        slots.sampled_actions_WK44,
        0,
    )
    with (
        torch.set_grad_enabled(grad),
        torch.autocast(
            hidden_WLd.device.type,
            dtype=torch.bfloat16,
            enabled=model.decoder_precision == "bf16",
            cache_enabled=False,
        ),
    ):
        output = model.temporal.replay_window_decisions(
            hidden_WLd.float(),
            observed_WL4,
            position,
            slots.history_starts_WL,
            queue,
            delay,
            replan,
            condition,
            actions,
        )
        group_order = tuple(CONTROLLER_GROUP_INDEX[name] for name in CONTROLLER_DECODE_ORDER)
        mask = slots.executed_WKF & valid[..., None] & output.valid[..., None].expand(-1, -1, -1, 4).flatten(-2)
        logs = output.log_probs[..., list(group_order)].flatten(-2)
        rows = torch.arange(hidden_WLd.shape[0], device=hidden_WLd.device)[:, None]
        lengths = torch.where(valid, slots.opponent_lengths_WK, 0)
        opponent_queue = torch.where(
            (torch.arange(model.critic_residual.max_queue, device=hidden_WLd.device) < lengths[..., None])[..., None],
            slots.opponent_queue_WKQ4,
            0,
        )
        return ReplayOutput(
            torch.where(mask, logs, 0),
            tuple(output.distributions[group] for group in group_order),
            mask,
            ReplayCriticFeatures(
                torch.where(valid[..., None], hidden_WLd[rows, position], 0).detach(),
                torch.where(valid[..., None], output.decoder_state, 0).detach(),
                condition.detach(),
                opponent_condition.detach(),
                opponent_queue.detach(),
                lengths.detach(),
            ),
        )
