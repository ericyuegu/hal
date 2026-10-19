"""Train a self-play actor with decision-grid rewards and KL-regularized PPO.

Flatten each frame's factors in decode order: C-stick, main stick, triggers,
buttons. Pad mixed cadences to four frames. Mask execution at match ends,
retain live rollout tails, and compute GAE from every decision's value.
"""

import contextlib
import copy
import dataclasses
import hashlib
import json
import math
import os
import random
import struct
import subprocess
import time
import zlib
from collections import defaultdict
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal
from typing import Protocol
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from hal.data.feature_stats import FeatureStats
from hal.data.schema import MDS_PER_FRAME_DTYPES
from hal.data.schema import Rank
from hal.eval.periodic_self_play import PeriodicEvaluator
from hal.eval.self_play_checkpoint import PROXY_SHA256
from hal.eval.self_play_checkpoint import TRAINING_FORMAT
from hal.eval.self_play_checkpoint import TRAINING_VERSION
from hal.eval.self_play_checkpoint import evaluation_actor_path
from hal.eval.self_play_checkpoint import load_evaluation_checkpoint
from hal.eval.self_play_checkpoint import read_training_manifest
from hal.eval.self_play_checkpoint import save_evaluation_checkpoint
from hal.eval.self_play_checkpoint import sha256
from hal.inference.column_context import ColumnContextBuilder
from hal.inference.column_context import PackedColumnContextBuilder
from hal.inference.cuda_graph import CapturedCall
from hal.inference.decision_decoder import DecisionDecoder
from hal.inference.kv_cache import BlockRestartCache
from hal.inference.kv_cache import forward_tokens_with_kv_cache
from hal.models.action_sequence import ActionSequenceConfig
from hal.models.attention import KVMemory
from hal.models.conditioning import ConditioningRanges
from hal.models.controller_codec import CONTROLLER_GROUP_NAMES
from hal.models.self_play import SelfPlayTransformer
from hal.representation.player_identity import vocabulary_from_checkpoint_buffer
from hal.sim import native
from hal.sim import native_bridge
from hal.sim.inputs import native_controller_readback
from hal.sim.native_shard import NativeShard
from hal.training.muon import SingleDeviceMuonWithAuxAdam
from hal.wire import ACTION_DIM

FACTOR_ORDER = ("c_stick", "main_stick", "triggers", "buttons")
MAX_REPLAN = 4
GAMMA = 0.99855
GAE_LAMBDA = 0.99


@dataclasses.dataclass(frozen=True, slots=True)
class DelayReplan:
    delay: int
    replan: int

    def __post_init__(self) -> None:
        if type(self.delay) is not int or self.delay < 0:
            raise ValueError("delay must be a non-negative integer")
        if type(self.replan) is not int or self.replan not in (2, 4):
            raise ValueError("replan must be 2 or 4")
        if self.delay + self.replan > 24:
            raise ValueError("delay + replan must not exceed 24")


@dataclasses.dataclass(frozen=True, slots=True)
class ScheduleConfig:
    configurations: tuple[DelayReplan, ...]
    supervised_offsets: tuple[int, ...]
    rollout_frames: int = 256

    def __post_init__(self) -> None:
        if not self.configurations or len(set(self.configurations)) != len(self.configurations):
            raise ValueError("configurations must be nonempty and unique")
        if (
            not self.supervised_offsets
            or any(type(offset) is not int or offset < 1 for offset in self.supervised_offsets)
            or tuple(sorted(set(self.supervised_offsets))) != self.supervised_offsets
        ):
            raise ValueError("supervised offsets must be positive, ordered, unique integers")
        if type(self.rollout_frames) is not int or self.rollout_frames < 1:
            raise ValueError("rollout_frames must be a positive integer")
        for config in self.configurations:
            if self.rollout_frames % config.replan:
                raise ValueError("rollout_frames must be divisible by every replan")
            missing = set(range(config.delay + 1, config.delay + config.replan + 1)) - set(self.supervised_offsets)
            if missing:
                raise ValueError(f"checkpoint never supervised executed offsets {sorted(missing)}")


def development_config(supervised_offsets: tuple[int, ...], *, rollout_frames: int = 256) -> ScheduleConfig:
    return ScheduleConfig((DelayReplan(8, 4), DelayReplan(10, 2)), supervised_offsets, rollout_frames)


def production_config(supervised_offsets: tuple[int, ...], *, rollout_frames: int = 256) -> ScheduleConfig:
    configurations = tuple(
        DelayReplan(delay, replan) for delay in range(16, 22) for replan in (2, 4) if delay + replan <= 24
    )
    return ScheduleConfig(configurations, supervised_offsets, rollout_frames)


def _check_tensor(
    name: str, value: Tensor, shape: tuple[int, ...], kind: Literal["bool", "int", "float"], device: torch.device
) -> None:
    if value.shape != shape or value.device != device:
        raise ValueError(f"{name} must have shape {shape} on {device}")
    valid_dtype = value.dtype == torch.bool if kind == "bool" else value.dtype == torch.int64
    if kind == "float":
        valid_dtype = value.is_floating_point()
    if not valid_dtype:
        raise ValueError(f"{name} must have {kind} dtype")


def _check_finite(name: str, value: Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


def _loss_precision(value: Tensor) -> Tensor:
    return value if value.dtype == torch.float64 else value.float()


def _check_discount(name: str, value: float) -> None:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")


def _check_positive_count(count: Tensor) -> None:
    if count.ndim != 0 or (not count.is_floating_point() and count.dtype not in (torch.int32, torch.int64)):
        raise ValueError("count must be a real numeric scalar")
    _check_finite("count", count)
    if bool(count <= 0) or bool(count != count.round()):
        raise ValueError("count must be a positive integer")


def executed_factor_mask(delay_J: Tensor, replan_J: Tensor, frames_remaining_J: Tensor) -> Tensor:
    """Include offset D+1 through min(D+R, frames remaining in the match).

    The terminal frame counts as an executed frame. For an ongoing match, the
    caller supplies at least D+R; the rollout's remaining length is irrelevant.
    A time-out also discards queued actions and uses its last pre-reset frame.
    """
    shape = tuple(delay_J.shape)
    if len(shape) != 1:
        raise ValueError("delay must have shape [J]")
    for name, value in (("delay", delay_J), ("replan", replan_J), ("frames_remaining", frames_remaining_J)):
        _check_tensor(name, value, shape, "int", delay_J.device)
    if bool(((delay_J < 0) | ((replan_J != 2) & (replan_J != 4)) | (delay_J > 24 - replan_J)).any()):
        raise ValueError("invalid delay or replan")
    if bool((frames_remaining_J < 0).any()):
        raise ValueError("frames_remaining must be non-negative")
    frame = torch.arange(1, MAX_REPLAN + 1, device=delay_J.device)
    executed = (frame <= replan_J[:, None]) & (delay_J[:, None] + frame <= frames_remaining_J[:, None])
    return executed.repeat_interleave(4, dim=-1)


@dataclasses.dataclass(frozen=True, slots=True)
class DecisionBatch:
    """Decision rows, with all action indices stored in FACTOR_ORDER.

    The prefix is padded to [J, 24, 4]; only its first D frames are read.
    Sampled indices and behavior log-probs are padded to [J, 16]. The execution
    mask consists of complete frames and can be shorter than the sampled chunk.
    Observation/decision frames use the same clock. Lambda belongs to ego.
    """

    observation_frame_J: Tensor
    decision_frame_J: Tensor
    sampled_indices_JF: Tensor
    committed_prefix_JD4: Tensor
    behavior_log_probs_JF: Tensor
    executed_factors_JF: Tensor
    duration_J: Tensor
    bootstrap_mask_J: Tensor
    trace_mask_J: Tensor
    kl_coefficient_J: Tensor
    delay_J: Tensor
    replan_J: Tensor
    policy_version_J: Tensor
    episode_id_J: Tensor
    segment_id_J: Tensor

    def __post_init__(self) -> None:
        shape = tuple(self.delay_J.shape)
        if len(shape) != 1:
            raise ValueError("decision columns must have shape [J]")
        device = self.delay_J.device
        for name, value in (
            ("observation frame", self.observation_frame_J),
            ("decision frame", self.decision_frame_J),
            ("duration", self.duration_J),
            ("delay", self.delay_J),
            ("replan", self.replan_J),
            ("policy version", self.policy_version_J),
            ("episode id", self.episode_id_J),
            ("segment id", self.segment_id_J),
        ):
            _check_tensor(name, value, shape, "int", device)
        for name, value in (("bootstrap mask", self.bootstrap_mask_J), ("trace mask", self.trace_mask_J)):
            _check_tensor(name, value, shape, "bool", device)
        _check_tensor("lambda", self.kl_coefficient_J, shape, "float", device)
        _check_finite("lambda", self.kl_coefficient_J)
        if bool((self.kl_coefficient_J <= 0).any()):
            raise ValueError("lambda must be positive")
        j = shape[0]
        _check_tensor("indices", self.sampled_indices_JF, (j, 16), "int", device)
        _check_tensor("prefix", self.committed_prefix_JD4, (j, 24, 4), "int", device)
        _check_tensor("behavior log-probs", self.behavior_log_probs_JF, (j, 16), "float", device)
        _check_tensor("execution mask", self.executed_factors_JF, (j, 16), "bool", device)
        full_mask = executed_factor_mask(self.delay_J, self.replan_J, self.delay_J + self.replan_J)
        if bool((self.sampled_indices_JF[full_mask] < 0).any()):
            raise ValueError("sampled indices must be non-negative")
        prefix_mask = torch.arange(24, device=device) < self.delay_J[:, None]
        if bool((self.committed_prefix_JD4[prefix_mask] < 0).any()):
            raise ValueError("committed indices must be non-negative")
        executed = self.executed_factors_JF.reshape(j, 4, 4)
        if (
            bool((self.executed_factors_JF & ~full_mask).any())
            or not torch.equal(executed, executed[:, :, :1].expand_as(executed))
            or bool((~executed[:, :-1, 0] & executed[:, 1:, 0]).any())
        ):
            raise ValueError("execution mask must be a prefix of complete sampled frames")
        _check_finite("executed behavior log-probs", self.behavior_log_probs_JF[self.executed_factors_JF])
        if bool(((self.duration_J < 1) | (self.duration_J > self.replan_J)).any()):
            raise ValueError("duration must be in [1, R]")
        if bool((self.trace_mask_J & ~self.bootstrap_mask_J).any()):
            raise ValueError("a terminal decision cannot continue a trace")
        if bool(((self.duration_J < self.replan_J) & self.trace_mask_J).any()):
            raise ValueError("a shortened interval must end the trace")
        if bool((self.observation_frame_J > self.decision_frame_J).any()):
            raise ValueError("observation frame cannot follow the decision frame")

    @property
    def loss_mask_J(self) -> Tensor:
        return self.executed_factors_JF.any(-1)


@dataclasses.dataclass(frozen=True, slots=True)
class ControllerLogits:
    """Categorical logits [J, 4, K_group], in decoder order."""

    c_stick: Tensor
    main_stick: Tensor
    triggers: Tensor
    buttons: Tensor


@dataclasses.dataclass(frozen=True, slots=True)
class FactorEvaluation:
    log_probs_JF: Tensor
    reference_log_probs_JF: Tensor
    exact_kl_JF: Tensor


def categorical_log_probs(logits: Tensor, invalid_mask: Tensor | None = None) -> Tensor:
    """Apply the same support mask before normalizing either policy."""
    if not logits.is_floating_point() or logits.ndim < 1 or logits.shape[-1] < 1:
        raise ValueError("logits must be floating point with a nonempty category dimension")
    if invalid_mask is None:
        invalid_mask = torch.zeros_like(logits, dtype=torch.bool)
    _check_tensor("category mask", invalid_mask, tuple(logits.shape), "bool", logits.device)
    if bool(invalid_mask.all(-1).any()):
        raise ValueError("a categorical distribution needs a legal category")
    _check_finite("legal logits", logits[~invalid_mask])
    return _loss_precision(logits).masked_fill(invalid_mask, -torch.inf).log_softmax(-1)


def categorical_kl(log_probs: Tensor, reference_log_probs: Tensor) -> Tensor:
    """Exact forward KL; the reference is always a stop-gradient target.

    Inputs must be normalized log-probabilities with identical support. Illegal
    categories have log-probability -inf; they contribute zero, including in
    the backward pass.
    """
    _check_tensor("reference", reference_log_probs, tuple(log_probs.shape), "float", log_probs.device)
    if not log_probs.is_floating_point() or log_probs.ndim < 1 or log_probs.shape[-1] < 1:
        raise ValueError("log-probs must have a category dimension")
    log_probs = _loss_precision(log_probs)
    reference_log_probs = _loss_precision(reference_log_probs.detach())
    legal = torch.isfinite(log_probs)
    if not torch.equal(legal, torch.isfinite(reference_log_probs)):
        raise ValueError("policy and reference must have identical support")
    for value in (log_probs, reference_log_probs):
        if bool((torch.isnan(value) | torch.isposinf(value)).any()) or not bool(
            torch.isclose(value.logsumexp(-1), torch.zeros_like(value[..., 0]), atol=1e-5).all()
        ):
            raise ValueError("log-probs must be normalized")
    difference = torch.where(legal, log_probs, 0) - torch.where(legal, reference_log_probs, 0)
    return (log_probs.exp() * difference).sum(-1)


def evaluate_factors(
    policy: ControllerLogits,
    reference: ControllerLogits,
    sampled_indices_JF: Tensor,
    button_mask_JRK: Tensor,
) -> FactorEvaluation:
    """Evaluate sampled contexts, applying one trigger-derived button mask.

    The caller supplies logits conditioned on the identical forced prefix and
    earlier sampled factors. The mask uses HAL's convention: True is illegal.
    """
    if policy.c_stick.ndim != 3:
        raise ValueError("controller logits must have shape [J, 4, K_group]")
    j = policy.c_stick.shape[0]
    _check_tensor("sampled indices", sampled_indices_JF, (j, 16), "int", policy.c_stick.device)
    indices = sampled_indices_JF.reshape(j, 4, 4)
    log_probs: list[Tensor] = []
    reference_log_probs: list[Tensor] = []
    kls: list[Tensor] = []
    groups = (
        (policy.c_stick, reference.c_stick),
        (policy.main_stick, reference.main_stick),
        (policy.triggers, reference.triggers),
        (policy.buttons, reference.buttons),
    )
    for group, (logits, ref_logits) in enumerate(groups):
        if logits.ndim != 3 or logits.shape[:2] != (j, 4):
            raise ValueError("controller logits must have shape [J, 4, K_group]")
        _check_tensor("reference logits", ref_logits, tuple(logits.shape), "float", logits.device)
        mask = button_mask_JRK if group == 3 else None
        log_p = categorical_log_probs(logits, mask)
        log_ref = categorical_log_probs(ref_logits.detach(), mask)
        picks = indices[:, :, group, None]
        if bool(((picks < 0) | (picks >= logits.shape[-1])).any()):
            raise ValueError("sampled index outside categorical support")
        sampled = log_p.gather(-1, picks).squeeze(-1)
        _check_finite("sampled log-probs", sampled)
        log_probs.append(sampled)
        reference_log_probs.append(log_ref.gather(-1, picks).squeeze(-1))
        kls.append(categorical_kl(log_p, log_ref))
    return FactorEvaluation(
        torch.stack(log_probs, -1).flatten(1),
        torch.stack(reference_log_probs, -1).flatten(1),
        torch.stack(kls, -1).flatten(1),
    )


def _masked_factors(values_JF: Tensor, executed_JF: Tensor) -> Tensor:
    _check_tensor("factor values", values_JF, tuple(executed_JF.shape), "float", executed_JF.device)
    _check_finite("executed factor values", values_JF[executed_JF])
    return torch.where(executed_JF, _loss_precision(values_JF), 0)


def _factor_sum(values_JF: Tensor, executed_JF: Tensor) -> Tensor:
    return _masked_factors(values_JF, executed_JF).sum(-1)


@torch.no_grad()
def regularized_reward(
    decisions: DecisionBatch, frame_rewards_JR: Tensor, factor_kl_JF: Tensor, *, gamma: float = GAMMA
) -> Tensor:
    """Discount unshifted interval rewards and subtract lambda times old-policy KL."""
    _check_discount("gamma", gamma)
    _check_tensor("frame rewards", frame_rewards_JR, (len(decisions.delay_J), 4), "float", decisions.delay_J.device)
    frame = torch.arange(4, device=frame_rewards_JR.device)
    valid = frame < decisions.duration_J[:, None]
    _check_finite("interval rewards", frame_rewards_JR[valid])
    rewards = torch.where(valid, _loss_precision(frame_rewards_JR), 0)
    rewards = rewards * gamma ** frame.to(rewards.dtype)
    kl = _factor_sum(factor_kl_JF, decisions.executed_factors_JF)
    return rewards.sum(-1) - decisions.kl_coefficient_J * kl


@torch.no_grad()
def sampled_kl_penalty(decisions: DecisionBatch, reference_log_probs_JF: Tensor) -> Tensor:
    """The k1 cross-check uses log mu, with no differentiable KL loss."""
    behavior = _masked_factors(decisions.behavior_log_probs_JF, decisions.executed_factors_JF)
    reference = _masked_factors(reference_log_probs_JF, decisions.executed_factors_JF)
    return decisions.kl_coefficient_J * (behavior - reference).sum(-1)


@dataclasses.dataclass(frozen=True, slots=True)
class AdvantageTargets:
    advantages: Tensor
    value_targets: Tensor


@torch.no_grad()
def generalized_advantage(
    rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    durations: Tensor,
    bootstrap_mask: Tensor,
    trace_mask: Tensor,
    *,
    gamma: float = GAMMA,
    gae_lambda: float = GAE_LAMBDA,
) -> AdvantageTargets:
    """Time-major GAE [T, ...], with explicit pre-reset and rollout-end values.

    next_values is V' for every interval, not a shifted tensor spanning resets.
    The final trace mask must be false. Terminal next-values may be NaN because
    the bootstrap mask removes them before arithmetic.
    """
    _check_discount("gamma", gamma)
    _check_discount("gae_lambda", gae_lambda)
    if rewards.ndim < 1 or rewards.shape[0] == 0:
        raise ValueError("GAE needs a nonempty time dimension")
    shape = tuple(rewards.shape)
    for name, value in (("rewards", rewards), ("values", values), ("next_values", next_values)):
        _check_tensor(name, value, shape, "float", rewards.device)
    _check_tensor("durations", durations, shape, "int", rewards.device)
    for name, value in (("bootstrap mask", bootstrap_mask), ("trace mask", trace_mask)):
        _check_tensor(name, value, shape, "bool", rewards.device)
    if bool(((durations < 1) | (durations > MAX_REPLAN)).any()):
        raise ValueError("durations must be in [1, 4]")
    if bool(trace_mask[-1].any()) or bool((trace_mask & ~bootstrap_mask).any()):
        raise ValueError("traces must stop at terminals and rollout end")
    _check_finite("rewards", rewards)
    _check_finite("values", values)
    _check_finite("bootstrap values", next_values[bootstrap_mask])
    values = _loss_precision(values)
    next_values = torch.where(bootstrap_mask, _loss_precision(next_values), 0)
    delta = _loss_precision(rewards) + gamma ** durations.to(values.dtype) * next_values - values
    trace_discount = (gamma * gae_lambda) ** durations.to(values.dtype)
    advantages = torch.empty_like(delta)
    tail = torch.zeros_like(delta[-1])
    for t in range(len(delta) - 1, -1, -1):
        tail = delta[t] + trace_discount[t] * torch.where(trace_mask[t], tail, 0)
        advantages[t] = tail
    return AdvantageTargets(advantages, advantages + values)


@torch.no_grad()
def stream_advantages(
    rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    durations: Tensor,
    bootstrap_mask: Tensor,
    trace_mask: Tensor,
    streams: Sequence[Sequence[int]],
) -> AdvantageTargets:
    """Run the same reverse recurrence over padded independent stream columns."""
    if rewards.ndim != 1:
        raise ValueError("stream GAE needs flat decision rows")
    count = len(rewards)
    if (
        not streams
        or any(not stream for stream in streams)
        or sorted(index for stream in streams for index in stream) != list(range(count))
    ):
        raise ValueError("GAE streams must partition every decision exactly once")
    if bool(trace_mask[[stream[-1] for stream in streams]].any()):
        raise ValueError("traces must stop at every stream's rollout end")
    positions = np.full((max(map(len, streams)), len(streams)), count, dtype=np.int64)
    inverse = np.empty(count, dtype=np.int64)
    for column, stream in enumerate(streams):
        positions[: len(stream), column] = stream
        inverse[list(stream)] = np.arange(len(stream)) * len(streams) + column
    at = torch.as_tensor(positions, device=rewards.device)
    result = generalized_advantage(
        *(
            torch.cat((value, value.new_full((1,), padding)))[at]
            for value, padding in (
                (rewards, 0),
                (values, 0),
                (next_values, 0),
                (durations, 1),
                (bootstrap_mask, False),
                (trace_mask, False),
            )
        )
    )
    order = torch.as_tensor(inverse, device=rewards.device)
    return AdvantageTargets(result.advantages.flatten()[order], result.value_targets.flatten()[order])


@torch.no_grad()
def advantage_moments(advantages_J: Tensor, selected_J: Tensor) -> Tensor:
    """Return float64 [count, sum, sum of squares] for an external SUM all-reduce."""
    if not advantages_J.is_floating_point():
        raise ValueError("advantages must have floating dtype")
    _check_tensor("selection mask", selected_J, tuple(advantages_J.shape), "bool", advantages_J.device)
    _check_finite("selected advantages", advantages_J[selected_J])
    values = torch.where(selected_J, advantages_J.double(), 0)
    return torch.stack((selected_J.sum().double(), values.sum(), values.square().sum()))


@dataclasses.dataclass(frozen=True, slots=True)
class AdvantageStatistics:
    mean: Tensor
    sigma: Tensor
    count: Tensor

    def __post_init__(self) -> None:
        _check_positive_count(self.count)
        for name, value in (("mean", self.mean), ("sigma", self.sigma)):
            _check_tensor(name, value, (), "float", self.count.device)
            _check_finite(name, value)
        if bool(self.sigma <= 0):
            raise ValueError("sigma must be positive")


@torch.no_grad()
def advantage_statistics(global_moments: Tensor, *, minimum_sigma: float = 1e-8) -> AdvantageStatistics:
    """Population standard deviation with an explicit numerical floor.

    The actor and differentiable KL must use this same sigma. Moments cover
    exactly the selected training decisions across all ranks.
    """
    if global_moments.shape != (3,) or not global_moments.is_floating_point():
        raise ValueError("global moments must have shape [3] and floating dtype")
    _check_finite("global moments", global_moments)
    if not math.isfinite(minimum_sigma) or minimum_sigma <= 0:
        raise ValueError("minimum_sigma must be finite and positive")
    count, total, squares = global_moments.unbind()
    _check_positive_count(count)
    if bool(squares < 0):
        raise ValueError("sum of squares must be non-negative")
    mean = total / count
    variance = squares / count - mean.square()
    roundoff = 8 * torch.finfo(global_moments.dtype).eps * (squares / count + mean.square())
    if bool(variance < -roundoff):
        raise ValueError("inconsistent advantage moments give negative variance")
    return AdvantageStatistics(mean, variance.clamp_min(0).sqrt().clamp_min(minimum_sigma), count)


@dataclasses.dataclass(frozen=True, slots=True)
class DecisionReduction:
    """External global count and the number of ranks averaging gradients.

    DDP averages rank gradients, so each local sum is scaled by world_size / J.
    A critic whose gradients are SUM-reduced instead uses gradient_average_ranks=1.
    """

    global_count: Tensor
    gradient_average_ranks: int = 1

    def __post_init__(self) -> None:
        _check_positive_count(self.global_count)
        if type(self.gradient_average_ranks) is not int or self.gradient_average_ranks < 1:
            raise ValueError("gradient_average_ranks must be a positive integer")


def _decision_mean(values_J: Tensor, selected_J: Tensor, reduction: DecisionReduction | None) -> Tensor:
    _check_tensor("selected decisions", selected_J, tuple(values_J.shape), "bool", values_J.device)
    _check_finite("selected loss values", values_J[selected_J])
    local_count = selected_J.sum()
    if reduction is None:
        reduction = DecisionReduction(local_count)
    if bool(reduction.global_count < local_count):
        raise ValueError("global count cannot be smaller than the local count")
    return torch.where(selected_J, values_J, 0).sum() * (
        reduction.gradient_average_ranks
        / reduction.global_count.detach().to(device=values_J.device, dtype=values_J.dtype)
    )


def _training_mask(
    decisions: DecisionBatch,
    selected_J: Tensor,
    statistics: AdvantageStatistics,
    reduction: DecisionReduction | None,
) -> None:
    _check_tensor("selected decisions", selected_J, tuple(decisions.delay_J.shape), "bool", decisions.delay_J.device)
    if bool((selected_J & ~decisions.loss_mask_J).any()):
        raise ValueError("cannot train a decision with no executed factors")
    count = selected_J.sum() if reduction is None else reduction.global_count
    if bool(statistics.count.to(count.device) != count):
        raise ValueError("advantage statistics and loss must use the same global decision count")


@dataclasses.dataclass(frozen=True, slots=True)
class PolicyLoss:
    loss: Tensor
    joint_ratio_J: Tensor
    importance_weight_J: Tensor


def ppo_loss(
    decisions: DecisionBatch,
    new_log_probs_JF: Tensor,
    old_log_probs_JF: Tensor,
    advantages_J: Tensor,
    selected_J: Tensor,
    statistics: AdvantageStatistics,
    *,
    clip_epsilon: float,
    importance_cap: float,
    reduction: DecisionReduction | None = None,
) -> PolicyLoss:
    """Joint PPO on selected decisions using detached global advantage statistics."""
    _training_mask(decisions, selected_J, statistics, reduction)
    if not math.isfinite(clip_epsilon) or not 0 < clip_epsilon < 1:
        raise ValueError("clip_epsilon must be in (0, 1)")
    if not math.isfinite(importance_cap) or importance_cap <= 0:
        raise ValueError("importance_cap must be finite and positive")
    _check_tensor("advantages", advantages_J, tuple(decisions.delay_J.shape), "float", decisions.delay_J.device)
    executed = decisions.executed_factors_JF & selected_J[:, None]
    new = _masked_factors(new_log_probs_JF, executed)
    old = _masked_factors(old_log_probs_JF.detach(), executed)
    behavior = _masked_factors(decisions.behavior_log_probs_JF.detach(), executed)
    # Subtract per factor before summing to preserve small kernel differences.
    ratio = (new - old).sum(-1).exp()
    _check_finite("joint ratio", ratio)
    # Cap in log space so an extreme parity failure cannot overflow before truncation.
    weight = (old - behavior).sum(-1).clamp(max=math.log(importance_cap)).exp().detach()
    advantages = torch.where(selected_J, advantages_J.detach(), 0)
    normalized = (advantages - statistics.mean.detach()) / statistics.sigma.detach()
    surrogate = torch.minimum(ratio * normalized, ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * normalized)
    return PolicyLoss(_decision_mean(-weight * surrogate, selected_J, reduction), ratio, weight)


def differentiable_kl_loss(
    decisions: DecisionBatch,
    factor_kl_JF: Tensor,
    selected_J: Tensor,
    statistics: AdvantageStatistics,
    *,
    reduction: DecisionReduction | None = None,
) -> Tensor:
    """Direct KL derivative on the same decisions and sigma used by PPO."""
    _training_mask(decisions, selected_J, statistics, reduction)
    kl = _factor_sum(factor_kl_JF, decisions.executed_factors_JF & selected_J[:, None])
    weighted = decisions.kl_coefficient_J.detach() / statistics.sigma.detach() * kl
    return _decision_mean(weighted, selected_J, reduction)


def critic_loss(
    values_J: Tensor,
    targets_J: Tensor,
    selected_J: Tensor,
    *,
    value_scale: float = 150.0,
    reduction: DecisionReduction | None = None,
) -> Tensor:
    """O64's mean squared error in beta=150 reward units; targets are detached.

    Selection includes all valid critic decisions, even ones with no executed
    factors. Cached critic features must be detached by the model boundary.
    """
    if not math.isfinite(value_scale) or value_scale <= 0:
        raise ValueError("value_scale must be finite and positive")
    if not values_J.is_floating_point():
        raise ValueError("critic values must have floating dtype")
    _check_tensor("targets", targets_J, tuple(values_J.shape), "float", values_J.device)
    _check_tensor("selection mask", selected_J, tuple(values_J.shape), "bool", values_J.device)
    values = torch.where(selected_J, _loss_precision(values_J), 0)
    targets = torch.where(selected_J, _loss_precision(targets_J.detach()), 0)
    return _decision_mean(((values - targets) / value_scale).square(), selected_J, reduction)


# PPO rows use FACTOR_ORDER. Index codec tensors with DECODE_GROUP_INDICES
# to read that order; CODEC_FACTOR_INDICES orders RNG factors for the codec.
DECODE_GROUP_INDICES = (2, 1, 3, 0)
CODEC_FACTOR_INDICES = (3, 1, 0, 2)
CHECKPOINT_VERSION = TRAINING_VERSION


@dataclasses.dataclass(frozen=True, slots=True)
class CounterRng:
    seed: int

    def __post_init__(self) -> None:
        if type(self.seed) is not int:
            raise ValueError("counter RNG seed must be an integer")

    def uniform(self, domain: str, stream: int, episode: int, frame: int, offset: int = 0, factor: int = 0) -> float:
        key = json.dumps((self.seed, domain, stream, episode, frame, offset, factor), separators=(",", ":"))
        digest = hashlib.blake2b(key.encode(), digest_size=8, person=b"hal-o65-rng-v1").digest()
        return (int.from_bytes(digest, "little") >> 11) * 2.0**-53

    def choice(self, weights: Sequence[float], domain: str, stream: int, episode: int, frame: int) -> int:
        if not weights or any(not math.isfinite(x) or x < 0 for x in weights) or sum(weights) <= 0:
            raise ValueError("sampling weights must be finite, nonnegative, and have positive mass")
        draw = self.uniform(domain, stream, episode, frame) * sum(weights)
        cumulative = 0.0
        for index, weight in enumerate(weights):
            cumulative += weight
            if draw < cumulative:
                return index
        raise ArithmeticError("categorical CDF did not cover the draw")


@dataclasses.dataclass(frozen=True, slots=True)
class IdentityCount:
    identity: int
    character: int
    count: int

    def __post_init__(self) -> None:
        if (
            any(type(value) is not int for value in (self.identity, self.character, self.count))
            or min(self.identity, self.character) < 0
            or self.count < 1
        ):
            raise ValueError("identity counts require nonnegative ids and positive counts")


@dataclasses.dataclass(frozen=True, slots=True)
class Matchup:
    characters: tuple[int, int]
    weight: float

    def __post_init__(self) -> None:
        if (
            len(self.characters) != 2
            or any(type(value) is not int or value < 0 for value in self.characters)
            or not math.isfinite(self.weight)
            or self.weight <= 0
        ):
            raise ValueError("invalid matchup prior")


@dataclasses.dataclass(frozen=True, slots=True)
class SamplerConfig:
    identities: tuple[IdentityCount, ...]
    matchups: tuple[Matchup, ...]
    stages: tuple[int, ...]
    data_floor: int = 100
    tempering: float = 0.5
    masked_rate: float = 0.1
    deployment_identity: int = int(Rank.MASTER)
    deployment_rate: float = 0.225
    lambda_min: float = 0.01
    lambda_max: float = 100.0
    stocks: tuple[int, ...] = (1, 2, 3, 4)
    percents: tuple[float, ...] = (0.0, 30.0, 60.0, 90.0)
    anchor_rate: float = 0.15

    def __post_init__(self) -> None:
        if (
            not self.identities
            or not self.matchups
            or not self.stages
            or any(type(stage) is not int or stage < 0 for stage in self.stages)
        ):
            raise ValueError("sampler tables and stages must be nonempty")
        if len({(row.identity, row.character) for row in self.identities}) != len(self.identities):
            raise ValueError("duplicate identity and character pair")
        if (
            type(self.data_floor) is not int
            or self.data_floor < 1
            or not math.isfinite(self.tempering)
            or not 0 <= self.tempering <= 1
        ):
            raise ValueError("invalid identity floor or tempering")
        for rate in (self.masked_rate, self.deployment_rate, self.anchor_rate):
            if not math.isfinite(rate) or not 0 <= rate <= 1:
                raise ValueError("sampler rates must be in [0, 1]")
        ordinary_rate = 1 - (
            self.deployment_rate if self.deployment_identity == 0 else self.masked_rate + self.deployment_rate
        )
        if type(self.deployment_identity) is not int or self.deployment_identity < 0 or ordinary_rate < 0:
            raise ValueError("invalid deployment identity or identity rates")
        if not 0 < self.lambda_min <= self.lambda_max or not math.isfinite(self.lambda_max):
            raise ValueError("invalid lambda range")
        if not self.stocks or any(type(stock) is not int or not 1 <= stock <= 4 for stock in self.stocks):
            raise ValueError("stocks must be in [1, 4]")
        if not self.percents or any(not math.isfinite(x) or x < 0 for x in self.percents):
            raise ValueError("starting percents must be finite and nonnegative")
        for matchup in self.matchups:
            for character in matchup.characters:
                rows = [row for row in self.identities if row.character == character and row.count >= self.data_floor]
                ordinary = [row for row in rows if row.identity not in (0, self.deployment_identity)]
                if not rows or (ordinary_rate > 0 and not ordinary):
                    raise ValueError(f"character {character} has no admitted ordinary identities")
                if (
                    self.deployment_identity
                    and self.deployment_rate > 0
                    and not any(row.identity == self.deployment_identity for row in rows)
                ):
                    raise ValueError(f"character {character} has no admitted deployment identity")


@dataclasses.dataclass(frozen=True, slots=True)
class PlayerEpisode:
    identity: int
    character: int
    log_lambda: float
    timing: DelayReplan
    stocks: int
    percent: float
    anchor: int | None


@dataclasses.dataclass(frozen=True, slots=True)
class Episode:
    number: int
    stage: int
    players: tuple[PlayerEpisode, PlayerEpisode]
    seed: int


class EpisodeSampler:
    def __init__(self, config: SamplerConfig, schedule: ScheduleConfig, rng: CounterRng) -> None:
        self.config = config
        self.schedule = schedule
        self.rng = rng
        self.lambda_max = config.lambda_max
        self.cursor = 0

    def sample(self, match_id: int, episode: int, anchors: tuple[int, ...]) -> Episode:
        cfg = self.config
        rng = self.rng
        matchup = cfg.matchups[rng.choice([m.weight for m in cfg.matchups], "matchup", match_id, episode, 0)]
        stage = cfg.stages[rng.choice([1.0] * len(cfg.stages), "stage", match_id, episode, 0)]
        stocks = cfg.stocks[rng.choice([1.0] * len(cfg.stocks), "stocks", match_id, episode, 0)]
        anchor_side = None
        if rng.uniform("anchor_share", match_id, episode, 0) < cfg.anchor_rate:
            anchor_side = rng.choice((1.0, 1.0), "anchor_side", match_id, episode, 0)
        players: list[PlayerEpisode] = []
        for side, character in enumerate(matchup.characters):
            stream = 2 * match_id + side
            identity_draw = rng.uniform("identity_kind", stream, episode, 0)
            masked_mass = cfg.deployment_rate if cfg.deployment_identity == 0 else cfg.masked_rate
            if identity_draw < masked_mass:
                identity = 0
            elif cfg.deployment_identity != 0 and identity_draw < masked_mass + cfg.deployment_rate:
                identity = cfg.deployment_identity
            else:
                rows = [
                    row
                    for row in cfg.identities
                    if row.character == character
                    and row.count >= cfg.data_floor
                    and row.identity not in (0, cfg.deployment_identity)
                ]
                identity = rows[
                    rng.choice([row.count**cfg.tempering for row in rows], "identity", stream, episode, 0)
                ].identity
            log_lambda = math.log(cfg.lambda_min) + rng.uniform("lambda", stream, episode, 0) * (
                math.log(self.lambda_max) - math.log(cfg.lambda_min)
            )
            timing = self.schedule.configurations[
                rng.choice([1.0] * len(self.schedule.configurations), "timing", stream, episode, 0)
            ]
            percent = cfg.percents[rng.choice([1.0] * len(cfg.percents), "percent", stream, episode, 0)]
            anchor = None
            if side == anchor_side:
                if not anchors:
                    raise ValueError("anchor sampling requires a reference policy")
                anchor = anchors[rng.choice([1.0] * len(anchors), "anchor", stream, episode, 0)]
            players.append(PlayerEpisode(identity, character, log_lambda, timing, stocks, percent, anchor))
        self.cursor += 1
        seed = int(rng.uniform("simulator_seed", match_id, episode, 0) * 2**32)
        return Episode(episode, stage, (players[0], players[1]), seed)


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class NativeColumns(Mapping[str, int | float]):
    """Decode an immutable adapter row when diagnostics or CPU features need columns."""

    words: NDArray[np.uint32]

    def __post_init__(self) -> None:
        if (
            self.words.shape != (len(native.OBSERVATION_FIELDS),)
            or self.words.dtype != np.uint32
            or self.words.flags.writeable
        ):
            raise ValueError("native columns require an immutable uint32 adapter row")

    def __getitem__(self, name: str) -> int | float:
        index, integer = NATIVE_COLUMN_LAYOUT[name]
        word = self.words[index]
        return int(word.view(np.int32)) if integer else float(word.view(np.float32))

    def __iter__(self) -> Iterator[str]:
        return iter(native.OBSERVATION_FIELDS)

    def __len__(self) -> int:
        return len(native.OBSERVATION_FIELDS)


NATIVE_COLUMN_LAYOUT = {
    name: (index, np.dtype(MDS_PER_FRAME_DTYPES[name]).kind == "i")
    for index, name in enumerate(native.OBSERVATION_FIELDS)
}


@dataclasses.dataclass(frozen=True, slots=True)
class DeviceFrameBlock:
    """Write-once packed frames; recorded decisions keep their storage alive."""

    records: Tensor  # [128, lanes, bytes]
    actions: Tensor  # [128, lanes, 2, ACTION_DIM]


@dataclasses.dataclass(frozen=True, slots=True)
class SimFrame:
    """One post-frame observation. No timer column is admitted by the builder."""

    columns: Mapping[str, int | float]
    applied_actions: NDArray[np.float32]  # [2, ACTION_DIM], HAL wire units
    rewards: tuple[float, float] = (0.0, 0.0)
    terminal: bool = False
    timeout: bool = False
    device_record: Tensor | None = None  # packed HalFrame bytes, copied from pinned output
    device_actions: Tensor | None = None  # processed readback [2, ACTION_DIM]
    device_block: DeviceFrameBlock | None = None
    device_slot: tuple[int, int] | None = None  # frame, lane in device_block

    def __post_init__(self) -> None:
        if self.applied_actions.shape != (2, ACTION_DIM) or self.applied_actions.dtype != np.float32:
            raise ValueError("simulator actions must be float32 [2, ACTION_DIM]")
        if not np.isfinite(self.applied_actions).all() or not all(math.isfinite(x) for x in self.rewards):
            raise ValueError("simulator actions and rewards must be finite")
        if self.terminal and self.timeout:
            raise ValueError("a match end cannot also be a time-out")


class SimulatorShard(Protocol):
    """Step queued actions until the first terminal or timeout in the shard.

    Inputs contain [frames, lanes, 2, ACTION_DIM]; step returns 1..frames rows.
    Reset returns the initial observation while the group clock stays fixed.
    Callers copy borrowed observations before the next write and serialize calls
    on each shard. Save and restore include simulator RNG and build identity.
    """

    @property
    def match_ids(self) -> tuple[int, ...]: ...
    def step(self, inputs: NDArray[np.float32], active: NDArray[np.bool_]) -> tuple[tuple[SimFrame, ...], ...]: ...
    def reset(self, lane: int, episode: Episode) -> SimFrame: ...
    def save(self) -> bytes: ...
    def restore(self, state: bytes) -> None: ...


class NativeSimulatorShard:
    """Step native matches with pinned buffers and retain copied rollout frames.

    Wait for the previous GPU copy before writing output again. Allocate device
    blocks once, then retain them through decision replay.
    """

    def __init__(
        self, data_dir: Path, match_ids: tuple[int, ...], *, cpu: int, device: torch.device, max_frame: int = 28800
    ) -> None:
        if not match_ids or len(set(match_ids)) != len(match_ids) or min(match_ids) < 0:
            raise ValueError("native shard needs distinct nonnegative match ids")
        self.match_ids = match_ids
        self.device = device
        self.max_frame = max_frame
        count = len(match_ids)
        pinned = device.type == "cuda"
        self._output = torch.zeros(count * native.FRAME_DTYPE.itemsize, dtype=torch.uint8, pin_memory=pinned)
        self._records = self._output.numpy().view(native.FRAME_DTYPE)
        self._input = torch.empty(count, 2, ACTION_DIM, dtype=torch.float32, pin_memory=pinned)
        self._observed = torch.empty_like(self._input, pin_memory=pinned)
        self._event: torch.cuda.Event | None = None
        self._device_block: DeviceFrameBlock | None = None
        self._device_frame = 128
        self.step_seconds = 0.0
        self.reset_seconds = 0.0
        self.stepped_frames = 0
        self.copied_bytes = 0
        self._mask = np.ones(count, dtype=np.bool_)
        self._configs = [native.NativeMatch(25, (native.NativePlayer(1, 1), native.NativePlayer(2, 22)), 0)] * count
        self.shard = NativeShard(data_dir, count, cpu=cpu, records=self._records)

    def _before_write(self) -> None:
        if self._event is not None:
            self._event.synchronize()

    def _frames(self, frame: native.NativeFrameBatch) -> tuple[SimFrame, ...]:
        if frame.records is None:
            raise ValueError("native rollout requires packed adapter records")
        columns = frame.records["columns"].copy()
        columns.flags.writeable = False
        observed = native_controller_readback(frame.applied_action)
        packed_gpu = actions_gpu = None
        block = None
        if self.device.type == "cuda":
            self._observed.copy_(torch.from_numpy(observed))
            if self._device_frame == 128:
                self._device_block = DeviceFrameBlock(
                    torch.zeros(
                        128, len(self.match_ids), native.FRAME_DTYPE.itemsize, device=self.device, dtype=torch.uint8
                    ),
                    torch.zeros(128, len(self.match_ids), 2, ACTION_DIM, device=self.device),
                )
                self._device_frame = 0
            block = self._device_block
            if block is None:
                raise RuntimeError("native device frame storage was not allocated")
            packed_gpu = block.records[self._device_frame]
            actions_gpu = block.actions[self._device_frame]
            packed_gpu.copy_(self._output.reshape(len(self.match_ids), -1), non_blocking=True)
            actions_gpu.copy_(self._observed, non_blocking=True)
            self._event = torch.cuda.Event()
            self._event.record(torch.cuda.current_stream(self.device))
            self.copied_bytes += self._output.numel() + self._observed.numel() * self._observed.element_size()
        result = tuple(
            SimFrame(
                NativeColumns(columns[lane]),
                observed[lane].copy(),
                (float(frame.reward[lane, 0]), float(frame.reward[lane, 1])),
                bool(frame.terminated[lane]),
                bool(frame.truncated[lane]),
                None if packed_gpu is None else packed_gpu[lane],
                None if actions_gpu is None else actions_gpu[lane],
                block,
                None if block is None else (self._device_frame, lane),
            )
            for lane in range(len(self.match_ids))
        )
        if block is not None:
            self._device_frame += 1
        return result

    def step(
        self, inputs: NDArray[np.float32], active: NDArray[np.bool_] | None = None
    ) -> tuple[tuple[SimFrame, ...], ...]:
        started = time.perf_counter()
        if (
            inputs.dtype != np.float32
            or inputs.ndim != 4
            or inputs.shape[1:] != (len(self.match_ids), 2, ACTION_DIM)
            or not len(inputs)
        ):
            raise ValueError("native step needs float32 [frames, lanes, 2, ACTION_DIM]")
        if active is None:
            active = self._mask
        if active.shape != (len(self.match_ids),) or active.dtype != np.bool_:
            raise ValueError("native active mask must be bool [lanes]")
        rows = []
        for actions in inputs:
            self._input.copy_(torch.from_numpy(actions))
            frame = self.shard.step(self._input.numpy(), active, before_write=self._before_write).result()
            rows.append(self._frames(frame))
            self.stepped_frames += int(active.sum())
            if ((frame.terminated | frame.truncated) & active).any():
                break
        self.step_seconds += time.perf_counter() - started
        return tuple(rows)

    def reset(self, lane: int, episode: Episode) -> SimFrame:
        started = time.perf_counter()
        if type(lane) is not int or not 0 <= lane < len(self.match_ids):
            raise ValueError("native reset lane is out of range")
        left, right = episode.players
        if left.stocks != right.stocks:
            raise ValueError("native reset does not support unequal starting stocks")
        if any(p.percent != int(p.percent) for p in episode.players):
            raise ValueError("native reset requires integer starting percents")
        config = native.NativeMatch(
            episode.stage,
            (
                native.NativePlayer(1, left.character, start_percent=int(left.percent)),
                native.NativePlayer(2, right.character, start_percent=int(right.percent)),
            ),
            episode.seed,
            max_frame=self.max_frame,
            stocks=left.stocks,
        )
        self._configs[lane] = config
        mask = np.arange(len(self.match_ids)) == lane
        frame = self.shard.reset(self._configs, mask, before_write=self._before_write).result()
        result = self._frames(frame)[lane]
        self.reset_seconds += time.perf_counter() - started
        return result

    def save(self) -> bytes:
        states = self.shard.save_many(tuple(range(len(self.match_ids)))).result()
        header = json.dumps(
            {"format": "hal.o65.native_pool.v1", "matches": self.match_ids, "max_frame": self.max_frame}
        ).encode()
        return (
            struct.pack("<I", len(header))
            + header
            + b"".join(struct.pack("<I", len(state)) + state for state in states)
        )

    def restore(self, state: bytes) -> None:
        if len(state) < 4:
            raise ValueError("truncated native pool snapshot")
        length = struct.unpack_from("<I", state)[0]
        at = 4 + length
        expected = {"format": "hal.o65.native_pool.v1", "matches": list(self.match_ids), "max_frame": self.max_frame}
        if json.loads(state[4:at]) != expected:
            raise ValueError("native pool snapshot identity mismatch")
        states = []
        for _ in self.match_ids:
            if at + 4 > len(state):
                raise ValueError("truncated native pool snapshot")
            size = struct.unpack_from("<I", state, at)[0]
            at += 4
            if at + size > len(state):
                raise ValueError("truncated native pool snapshot")
            states.append(state[at : at + size])
            at += size
        if at != len(state):
            raise ValueError("unexpected native pool snapshot bytes")
        self.shard.restore_many(tuple(range(len(self.match_ids))), states, before_write=self._before_write).result()

    def close(self) -> None:
        try:
            self._before_write()
        finally:
            self.shard.close()


@dataclasses.dataclass(frozen=True, slots=True)
class DecisionState:
    stream: int
    match: int
    side: int
    episode: Episode
    clock: int
    window: tuple[int, int, int, int]  # update, stream, block, episode
    frames: tuple[SimFrame, ...]
    prefix: Tensor  # [D, 4], codec order
    opponent_queue: Tensor  # [up to D_opp+R_opp, 4], codec order

    @property
    def player(self) -> PlayerEpisode:
        return self.episode.players[self.side]


@dataclasses.dataclass(frozen=True, slots=True)
class CriticFeatures:
    trunk: Tensor
    decoder: Tensor
    condition: Tensor
    opponent_condition: Tensor
    opponent_queue: Tensor
    opponent_lengths: Tensor

    def detached(self) -> CriticFeatures:
        return CriticFeatures(*(getattr(self, field.name).detach().clone() for field in dataclasses.fields(self)))


@dataclasses.dataclass(frozen=True, slots=True)
class PolicyEvaluation:
    actions: Tensor  # [J, 4, 4], codec order
    log_probs: Tensor  # [J, 16], decode order
    distributions: tuple[Tensor, ...]  # [J, 4, K], decode order; button mask already applied
    features: CriticFeatures


@dataclasses.dataclass(frozen=True, slots=True)
class CachedStream:
    window: tuple[int, int, int, int]
    history_length: int
    cache: BlockRestartCache
    row: int


class LoopPolicy(Protocol):
    """Run rollout sampling and differentiable replay through separate kernels."""

    @property
    def model(self) -> torch.nn.Module: ...
    @property
    def device(self) -> torch.device: ...
    def actor_parameters(self) -> tuple[torch.nn.Parameter, ...]: ...
    def critic_parameters(self) -> tuple[torch.nn.Parameter, ...]: ...
    def sample(self, states: Sequence[DecisionState], rng: CounterRng) -> PolicyEvaluation: ...
    def evaluate(self, states: Sequence[DecisionState], actions: Tensor) -> PolicyEvaluation: ...
    def values(self, features: CriticFeatures) -> Tensor: ...
    def wire_actions(self, indices: Tensor) -> NDArray[np.float32]: ...
    def clear_caches(self) -> None: ...
    def frozen_copy(self) -> LoopPolicy: ...


def _concatenate_evaluations(rows: Sequence[PolicyEvaluation]) -> PolicyEvaluation:
    return PolicyEvaluation(
        torch.cat([row.actions for row in rows]),
        torch.cat([row.log_probs for row in rows]),
        tuple(torch.cat([row.distributions[group] for row in rows]) for group in range(4)),
        CriticFeatures(
            *(
                torch.cat([getattr(row.features, field.name) for row in rows])
                for field in dataclasses.fields(CriticFeatures)
            )
        ),
    )


def _take_evaluation(evaluation: PolicyEvaluation, indices: Tensor) -> PolicyEvaluation:
    return PolicyEvaluation(
        evaluation.actions[indices],
        evaluation.log_probs[indices],
        tuple(value[indices] for value in evaluation.distributions),
        _take_features(evaluation.features, indices),
    )


class TransformerPolicy:
    """Single-device correctness path; each training window encodes once.

    Rollout uses the block cache. Training and the no-grad pass use the same
    full-window trunk and replay the recorded conditional action path.
    """

    def __init__(
        self,
        model: SelfPlayTransformer,
        schedule: ScheduleConfig,
        statistics: dict[str, FeatureStats],
        *,
        compiled: bool = False,
    ) -> None:
        self.model = model.eval()
        self.compiled = compiled
        self.schedule = schedule
        self.statistics = statistics
        for stat in statistics.values():
            if (
                not all(math.isfinite(value) for value in (stat.mean, stat.std, stat.min, stat.max))
                or stat.std < 0
                or stat.min > stat.max
            ):
                raise ValueError("invalid observation statistics")
        self.device = next(model.parameters()).device
        if schedule.supervised_offsets != model.head_offsets:
            raise ValueError("schedule supervised offsets must match the model checkpoint")
        configurations = tuple((c.delay, c.replan) for c in schedule.configurations)
        self.decoder = DecisionDecoder(model, configurations, compiled=compiled)
        self.group_sizes_cpu = self.decoder.group_sizes.cpu()
        self.button_valid_cpu = self.model.codec.button_valid_for_trigger.cpu()
        self.builders = (ColumnContextBuilder(statistics, 1), ColumnContextBuilder(statistics, 2))
        self.packed_builder = PackedColumnContextBuilder(statistics, native.OBSERVATION_FIELDS, self.device)
        self.caches: dict[int, CachedStream] = {}
        self.rollout_caches: dict[tuple[int, ...], BlockRestartCache] = {}
        self.used_cache_keys: set[tuple[int, ...]] = set()
        self.sample_group_counts: dict[int, int] = defaultdict(int)
        self.sample_batch_counts: dict[int, int] = defaultdict(int)
        self.wire_tables: tuple[NDArray[np.float32], ...] = ()
        self._prepare_wire_tables()

    @torch.no_grad()
    def _prepare_wire_tables(self) -> None:
        self.wire_tables = tuple(
            self.model.codec.semantic_values(name, torch.arange(size, device=self.device)).float().cpu().numpy()
            for name, size in zip(CONTROLLER_GROUP_NAMES, self.model.codec.group_vocabs, strict=True)
        )

    def actor_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        return self.model.actor_parameters()

    def critic_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        return self.model.critic_parameters()

    def clear_caches(self) -> None:
        self.caches.clear()
        self._prepare_wire_tables()
        if self.used_cache_keys:
            self.rollout_caches = {
                key: cache
                for key, cache in self.rollout_caches.items()
                if key in self.used_cache_keys
                and (not self.compiled or self.device.type != "cuda" or cache.cache.updates)
            }
            self.used_cache_keys.clear()

    def frozen_copy(self) -> LoopPolicy:
        model = copy.deepcopy(self.model).requires_grad_(False)
        model.zero_grad(set_to_none=True)
        return TransformerPolicy(model, self.schedule, self.statistics, compiled=self.compiled)

    def _validate_states(self, states: Sequence[DecisionState], actions: Tensor | None = None) -> None:
        if not states:
            raise ValueError("policy batch must be nonempty")
        for state in states:
            if type(state.side) is not int or state.side not in (0, 1):
                raise ValueError("decision side must be HAL port 1 or 2's zero-based index")
            if not 1 <= len(state.frames) <= 256:
                raise ValueError("invalid rollout history span")
            if not 0 <= state.player.identity < self.model.cfg.player_vocab_size:
                raise ValueError("episode identity is outside the checkpoint vocabulary")
            for player in state.episode.players:
                if player.timing not in self.schedule.configurations:
                    raise ValueError("decision batch contains an unsupported configuration")
                if (
                    not math.isfinite(player.log_lambda)
                    or not self.model.ranges.log_lambda[0] <= player.log_lambda <= self.model.ranges.log_lambda[1]
                ):
                    raise ValueError("episode conditioning is outside its declared range")
            if state.prefix.shape != (state.player.timing.delay, 4):
                raise ValueError("decision prefix must contain D committed frames")
            if (
                state.opponent_queue.ndim != 2
                or state.opponent_queue.shape[1] != 4
                or len(state.opponent_queue) > self.model.critic_residual.max_queue
            ):
                raise ValueError("opponent queue exceeds the critic capacity or has the wrong shape")
        queues = torch.cat([queue for state in states for queue in (state.prefix, state.opponent_queue)])
        self._validate_actions(queues)
        if actions is not None:
            self._validate_actions(actions)

    def _validate_actions(self, actions: Tensor) -> None:
        if actions.dtype not in (torch.int32, torch.int64) or actions.shape[-1] != 4:
            raise ValueError("actions must contain integer controller factors")
        sizes = self.group_sizes_cpu if actions.device.type == "cpu" else self.decoder.group_sizes
        if not bool(((actions >= 0) & (actions < sizes)).all()):
            raise ValueError("actions contain an invalid controller factor")
        valid_buttons = (
            self.button_valid_cpu if actions.device.type == "cpu" else self.model.codec.button_valid_for_trigger
        )
        if bool((~valid_buttons[actions[..., 3], actions[..., 0]]).any()):
            raise ValueError("actions contain buttons incompatible with their triggers")

    def _packed_inputs(
        self, states: Sequence[DecisionState], frames: Sequence[tuple[SimFrame, ...]], length: int
    ) -> tuple[Tensor, ...]:
        if all(frame.device_block is not None and frame.device_slot is not None for row in frames for frame in row):
            blocks: dict[int, tuple[DeviceFrameBlock, list[tuple[int, int, int]]]] = {}
            for row, history in enumerate(frames):
                for position, frame in enumerate(history):
                    block = cast(DeviceFrameBlock, frame.device_block)
                    slot = cast(tuple[int, int], frame.device_slot)
                    if id(block) not in blocks:
                        blocks[id(block)] = (block, [])
                    blocks[id(block)][1].append((row * length + position, *slot))
            coordinates = torch.tensor([item for _, items in blocks.values() for item in items])
            device_coordinates = coordinates.to(self.device, non_blocking=True)
            records = torch.zeros(
                len(states) * length, native.FRAME_DTYPE.itemsize, dtype=torch.uint8, device=self.device
            )
            actions = torch.zeros(len(states) * length, 2, ACTION_DIM, device=self.device)
            start = 0
            for block, items in blocks.values():
                end = start + len(items)
                at = device_coordinates[start:end]
                source_at = at if block.records.device == self.device else coordinates[start:end]
                records.index_copy_(0, at[:, 0], block.records[source_at[:, 1], source_at[:, 2]].to(self.device))
                actions.index_copy_(0, at[:, 0], block.actions[source_at[:, 1], source_at[:, 2]].to(self.device))
                start = end
            records = records.reshape(len(states), length, -1)
            actions = actions.reshape(len(states), length, 2, ACTION_DIM)
        else:
            packed = [self._packed_history(row) for row in frames]
            records = torch.stack([torch.nn.functional.pad(row[0], (0, 0, 0, length - len(row[0]))) for row in packed])
            actions = torch.stack(
                [torch.nn.functional.pad(row[1], (0, 0, 0, 0, 0, length - len(row[1]))) for row in packed]
            )
        sides = torch.tensor([state.side for state in states]).to(self.device, non_blocking=True)
        identities = torch.tensor([state.player.identity for state in states]).to(self.device, non_blocking=True)
        lengths = torch.tensor([len(row) for row in frames]).to(self.device, non_blocking=True)
        return records, actions, sides, identities, lengths

    def _packed_history(self, frames: tuple[SimFrame, ...]) -> tuple[Tensor, Tensor]:
        return (
            torch.stack([cast(Tensor, frame.device_record).to(self.device, non_blocking=True) for frame in frames]),
            torch.stack([cast(Tensor, frame.device_actions).to(self.device, non_blocking=True) for frame in frames]),
        )

    def _packed_context(self, inputs: tuple[Tensor, ...]) -> tuple[dict[str, Tensor], Tensor]:
        records, actions, sides, identities, lengths = inputs
        features = self.packed_builder(records, actions, sides, identities, lengths)
        length = records.shape[1]
        ego = actions.gather(2, sides[:, None, None, None].expand(-1, length, 1, ACTION_DIM)).squeeze(2)
        valid = torch.arange(length, device=self.device)[None] < lengths[:, None]
        return features, torch.where(valid[..., None], self.model.codec.quantize(ego), 0)

    def _append_packed(
        self, states: Sequence[DecisionState], count: int, cache: BlockRestartCache
    ) -> tuple[Tensor, Tensor]:
        if not 1 <= count <= 4:
            raise ValueError("cached context append must contain one to four frames")
        if any(not 0 <= state.clock - state.window[2] < 128 for state in states):
            raise ValueError("block is exhausted; restart before appending")
        frames = [state.frames[-count:] for state in states]
        inputs = self._packed_inputs(states, frames, count)
        key = 2 * count + int(torch.is_inference_mode_enabled())
        captured = cache.cache.updates.get(key)
        if captured is None:
            static = tuple(value.clone() for value in inputs)

            def append() -> tuple[Tensor, Tensor]:
                # Autocast's eager weight cache dies when sample() returns.
                # Capture the casts so replay also reads updated actor weights.
                with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                    features, observed = self._packed_context(static)
                    tokens = self.model.context_tokens(features, observed)
                    hidden = forward_tokens_with_kv_cache(self.model, tokens, cache.cache)
                    cache.frames_in_block.add_(count)
                    return hidden, observed[:, -1]

            captured = CapturedCall(
                append, static, (*cache.cache.buffers(), cache.frames_in_block), device=self.device
            )
            cache.cache.updates[key] = captured
        hidden, observed = cast(CapturedCall[tuple[Tensor, Tensor]], captured)(inputs)
        return hidden, observed

    def _contexts(
        self, states: Sequence[DecisionState], *, length: int | None = None, tail: int | None = None
    ) -> tuple[dict[str, Tensor], Tensor]:
        frames = [state.frames if tail is None else state.frames[-tail:] for state in states]
        length = max(len(row) for row in frames) if length is None else length
        if any(len(row) > length for row in frames):
            raise ValueError("context padding is shorter than its history")
        columns: dict[str, Tensor] = {}
        packed = all(
            frame.device_record is not None and frame.device_actions is not None for row in frames for frame in row
        )
        if packed:
            return self._packed_context(self._packed_inputs(states, frames, length))
        for name, value in frames[0][0].columns.items():
            dtype = torch.int64 if isinstance(value, int) else torch.float32
            columns[name] = torch.tensor(
                [[frame.columns[name] for frame in row] + [0] * (length - len(row)) for row in frames],
                dtype=dtype,
                device=self.device,
            )
        host_actions = np.zeros((len(states), length, 2, ACTION_DIM), dtype=np.float32)
        for index, row in enumerate(frames):
            host_actions[index, : len(row)] = np.stack([frame.applied_actions for frame in row])
        actions = torch.as_tensor(host_actions, device=self.device)
        parts: list[dict[str, Tensor]] = []
        observed_parts: list[Tensor] = []
        order: list[int] = []
        for side in range(2):
            indices = [index for index, state in enumerate(states) if state.side == side]
            if not indices:
                continue
            at = torch.tensor(indices, device=self.device)
            segments = torch.zeros(len(indices), length, device=self.device, dtype=torch.long)
            context = self.builders[side](
                {name: value[at] for name, value in columns.items()},
                actions[at],
                segments,
                torch.zeros_like(segments, dtype=torch.bool),
                torch.tensor([states[index].player.identity for index in indices], device=self.device),
            )
            valid = (
                torch.arange(length, device=self.device)[None]
                < torch.tensor([len(frames[index]) for index in indices], device=self.device)[:, None]
            )
            parts.append({name: torch.where(valid, value, 0) for name, value in context.features.items()})
            observed_parts.append(torch.where(valid[..., None], self.model.codec.quantize(actions[at, :, side]), 0))
            order.extend(indices)
        inverse = torch.tensor(np.argsort(order)).to(self.device, non_blocking=True)
        return (
            {name: torch.cat([part[name] for part in parts])[inverse] for name in parts[0]},
            torch.cat(observed_parts)[inverse],
        )

    def _decode(
        self,
        states: Sequence[DecisionState],
        hidden: Tensor,
        observed: Tensor,
        uniforms: Tensor,
        actions: Tensor | None = None,
        *,
        history: KVMemory,
        count: int,
    ) -> PolicyEvaluation:
        players = [state.player for state in states]
        opponents = [state.episode.players[1 - state.side] for state in states]
        delay = torch.tensor([player.timing.delay for player in players]).to(self.device, non_blocking=True)
        replan = torch.tensor([player.timing.replan for player in players]).to(self.device, non_blocking=True)
        log_lambda = torch.tensor([player.log_lambda for player in players]).to(self.device, non_blocking=True)
        queue = torch.stack(
            [
                torch.nn.functional.pad(state.prefix, (0, 0, 0, self.decoder.max_delay - len(state.prefix)))
                for state in states
            ]
        ).to(self.device, non_blocking=True)
        output = self.decoder(
            hidden,
            observed,
            queue,
            log_lambda,
            delay,
            replan,
            uniforms[:count],
            sampled_offsets=count,
            history=history,
            sampled_actions=None if actions is None else actions[:, :count],
            validate=False,
        )
        condition = self.model.ranges.normalize(log_lambda, delay, replan, validate=False)
        opponent_condition = self.model.ranges.normalize(
            torch.tensor([player.log_lambda for player in opponents]).to(self.device, non_blocking=True),
            torch.tensor([player.timing.delay for player in opponents]).to(self.device, non_blocking=True),
            torch.tensor([player.timing.replan for player in opponents]).to(self.device, non_blocking=True),
            validate=False,
        )
        capacity = self.model.critic_residual.max_queue
        opponent_queue = torch.stack(
            [
                torch.nn.functional.pad(state.opponent_queue, (0, 0, 0, capacity - len(state.opponent_queue)))
                for state in states
            ]
        ).to(self.device, non_blocking=True)
        features = CriticFeatures(
            hidden[:, -1].detach(),
            output.decoder_state.detach(),
            condition.detach(),
            opponent_condition.detach(),
            opponent_queue,
            torch.tensor([len(state.opponent_queue) for state in states]).to(self.device, non_blocking=True),
        )
        distributions = []
        for group in DECODE_GROUP_INDICES:
            probabilities = output.distributions[group]
            padding = probabilities.new_zeros(len(states), 4 - count, probabilities.shape[-1])
            padding[..., 0] = 1
            distributions.append(torch.cat((probabilities, padding), dim=1))
        return PolicyEvaluation(
            torch.nn.functional.pad(output.actions, (0, 0, 0, 4 - count)),
            torch.nn.functional.pad(output.log_probs[..., DECODE_GROUP_INDICES], (0, 0, 0, 4 - count)).flatten(1),
            tuple(distributions),
            features,
        )

    @torch.no_grad()
    def sample(self, states: Sequence[DecisionState], rng: CounterRng) -> PolicyEvaluation:
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            return self._sample(states, rng)

    def _sample(self, states: Sequence[DecisionState], rng: CounterRng) -> PolicyEvaluation:
        self._validate_states(states)
        if any(state.clock % state.player.timing.replan or state.clock % 4 != states[0].clock % 4 for state in states):
            raise ValueError("sampling batch must share a decision tick and respect each cadence")
        groups: dict[tuple[int, int], list[tuple[int, DecisionState]]] = defaultdict(list)
        for index, state in enumerate(states):
            entry = self.caches.get(state.stream)
            seen = entry.history_length if entry is not None and entry.window == state.window else 0
            groups[len(state.frames), seen].append((index, state))
        self.sample_group_counts[len(groups)] += 1
        rows: list[PolicyEvaluation] = []
        order: list[int] = []
        for (_, seen), entries in groups.items():
            batch = [state for _, state in entries]
            self.sample_batch_counts[len(batch)] += 1
            hidden, last_observed, cache = self._advance_rollout_cache(batch, seen)
            uniforms = self._action_uniforms(batch, rng)
            count = 4 if batch[0].clock % 4 == 0 else 2
            rows.append(
                self._decode(batch, hidden[:, -1:], last_observed, uniforms, history=cache.memory(), count=count)
            )
            order.extend(index for index, _ in entries)
        return _take_evaluation(
            _concatenate_evaluations(rows), torch.tensor(np.argsort(order)).to(self.device, non_blocking=True)
        )

    def _advance_rollout_cache(
        self, batch: Sequence[DecisionState], seen: int
    ) -> tuple[Tensor, Tensor, BlockRestartCache]:
        length = len(batch[0].frames)
        if not seen:
            cache = self._rollout_cache_for(tuple(state.stream for state in batch))
            features, observed = self._contexts(batch)
            tokens = self.model.context_tokens(features, observed)
            previous = tokens[:, :-1]
            if previous.shape[1] > 128:
                raise ValueError("first decision in a block must prefill at most 128 previous frames")
            cache.restart(previous, torch.full((len(batch),), previous.shape[1], device=self.device))
            tokens = tokens[:, -1:]
        else:
            sources = [self.caches[state.stream] for state in batch]
            cache = sources[0].cache
            if cache.cache.batch_size != len(batch) or any(
                source.cache is not cache or source.row != row for row, source in enumerate(sources)
            ):
                cache = self._rollout_cache_for(tuple(state.stream for state in batch))
                for target_index, target in enumerate(cache.cache.buffers()):
                    axis = 1 if target.ndim == 5 else 0
                    pieces = [
                        source.cache.cache.buffers()[target_index].narrow(axis, source.row, 1) for source in sources
                    ]
                    target.copy_(torch.cat(pieces, dim=axis))
                cache.frames_in_block.copy_(
                    torch.cat([source.cache.frames_in_block[source.row : source.row + 1] for source in sources])
                )
        packed_append = (
            seen
            and self.compiled
            and self.device.type == "cuda"
            and all(
                frame.device_record is not None and frame.device_actions is not None
                for state in batch
                for frame in state.frames[-(length - seen) :]
            )
        )
        if packed_append:
            hidden, last_observed = self._append_packed(batch, length - seen, cache)
        else:
            if seen:
                features, observed = self._contexts(batch, tail=length - seen)
                tokens = self.model.context_tokens(features, observed)
            hidden = cache.append(tokens)
            last_observed = observed[:, -1]
        for row, state in enumerate(batch):
            self.caches[state.stream] = CachedStream(state.window, length, cache, row)
        return hidden, last_observed, cache

    def _rollout_cache_for(self, streams: tuple[int, ...]) -> BlockRestartCache:
        self.used_cache_keys.add(streams)
        cache = self.rollout_caches.get(streams)
        if cache is None:
            dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
            cache = BlockRestartCache(self.model, len(streams), self.device, dtype=dtype)
            self.rollout_caches[streams] = cache
        return cache

    def _action_uniforms(self, batch: Sequence[DecisionState], rng: CounterRng) -> Tensor:
        if self.device.type == "cuda":
            from hal.inference.action_rng import action_uniforms

            return action_uniforms(
                rng.seed,
                [(state.stream, state.episode.number, state.clock, state.player.timing.delay) for state in batch],
                self.device,
            )
        draws = [
            [
                [
                    rng.uniform(
                        "action",
                        state.stream,
                        state.episode.number,
                        state.clock,
                        state.player.timing.delay + offset + 1,
                        factor,
                    )
                    for state in batch
                ]
                for factor in CODEC_FACTOR_INDICES
            ]
            for offset in range(4)
        ]
        return torch.tensor(draws, dtype=torch.float64, device=self.device)

    def evaluate(self, states: Sequence[DecisionState], actions: Tensor) -> PolicyEvaluation:
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            return self._evaluate(states, actions)

    def _evaluate(self, states: Sequence[DecisionState], actions: Tensor) -> PolicyEvaluation:
        self._validate_states(states, actions)
        windows: dict[tuple[int, int, int, int], DecisionState] = {}
        for state in states:
            if state.window not in windows or len(state.frames) > len(windows[state.window].frames):
                windows[state.window] = state
        encoded: dict[tuple[int, int, int, int], tuple[Tensor, Tensor, Tensor]] = {}
        window_states = list(windows.values())
        for start in range(0, len(window_states), 16):
            batch = window_states[start : start + 16]
            # The old-policy and actor passes use the same 256-frame trunk shape.
            features, observed = self._contexts(batch, length=256)
            hidden = self.model(features, torch.zeros(len(batch), dtype=torch.long, device=self.device), observed)
            keys, values = self.model.temporal.history_attention.project_memory(hidden)
            memory = torch.stack((keys, values))
            for row, state in enumerate(batch):
                encoded[state.window] = (hidden[row], observed[row], memory[:, row])
        outputs: list[PolicyEvaluation] = []
        for start in range(0, len(states), 128):
            batch = states[start : start + 128]
            hidden = torch.stack([encoded[state.window][0][len(state.frames) - 1] for state in batch])[:, None]
            observed = torch.stack([encoded[state.window][1][len(state.frames) - 1] for state in batch])
            memory = KVMemory(
                torch.stack([encoded[state.window][2] for state in batch], dim=1),
                torch.arange(256, device=self.device)[None].expand(len(batch), -1),
                torch.tensor([len(state.frames) - 1 for state in batch]).to(self.device, non_blocking=True),
                256,
            )
            outputs.append(
                self._decode(
                    batch,
                    hidden,
                    observed,
                    torch.full((4, 4, len(batch)), 0.5, device=self.device),
                    actions[start : start + len(batch)],
                    history=memory,
                    count=4,
                )
            )
        return _concatenate_evaluations(outputs)

    def values(self, features: CriticFeatures) -> Tensor:
        return self.model.self_play_value(
            features.trunk,
            features.decoder,
            features.condition,
            features.opponent_condition,
            features.opponent_queue,
            features.opponent_lengths,
        )

    def wire_actions(self, indices: Tensor) -> NDArray[np.float32]:
        host = indices.detach().cpu().numpy()
        if host.dtype not in (np.int32, np.int64) or host.shape[-1:] != (4,):
            raise ValueError("wire actions require integer controller factors")
        if any(
            ((host[..., group] < 0) | (host[..., group] >= len(table))).any()
            for group, table in enumerate(self.wire_tables)
        ):
            raise ValueError("wire actions contain an invalid controller factor")
        result = np.empty((*host.shape[:-1], ACTION_DIM), dtype=np.float32)
        result[..., :2] = self.wire_tables[1][host[..., 1]]
        result[..., 2:4] = self.wire_tables[2][host[..., 2]]
        result[..., 4:6] = self.wire_tables[3][host[..., 3]]
        result[..., 6:] = self.wire_tables[0][host[..., 0]]
        return result


@dataclasses.dataclass(slots=True)
class MatchState:
    episode: Episode
    episode_start: int
    history_start: int
    history: list[SimFrame]
    queues: tuple[list[Tensor], list[Tensor]]
    returns: tuple[float, float] = (0.0, 0.0)
    end_clock: int | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class EpisodeResult:
    match: int
    episode: Episode
    duration: int
    returns: tuple[float, float]
    timeout: bool
    stocks: tuple[int, int] | None


@dataclasses.dataclass(slots=True)
class DecisionRecord:
    state: DecisionState
    actions: Tensor  # [4, 4], codec order
    behavior: Tensor  # [16], decode order
    rewards: list[float]
    terminal: bool = False
    boundary: bool = False
    frames_remaining: int | None = None
    next_state: DecisionState | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class Rollout:
    records: tuple[DecisionRecord, ...]
    group: int
    update: int
    frames: int
    resets: int
    timeouts: int
    elapsed: float
    results: tuple[EpisodeResult, ...]
    idle_frames: int = 0


def _decision_batch(records: Sequence[DecisionRecord], device: torch.device) -> DecisionBatch:
    if not records:
        raise ValueError("rollout has no student decisions")

    def integers(values: Sequence[int]) -> Tensor:
        return torch.tensor(values, dtype=torch.long, device=device)

    delay = integers([r.state.player.timing.delay for r in records])
    replan = integers([r.state.player.timing.replan for r in records])
    prefixes = torch.zeros(len(records), 24, 4, dtype=torch.long)
    for index, record in enumerate(records):
        prefixes[index, : len(record.state.prefix)] = record.state.prefix[..., DECODE_GROUP_INDICES].cpu()
    # Only an end observed in this rollout masks execution. A live match's
    # queued tail trains in full, even if the next rollout ends before it runs.
    remaining = integers(
        [
            r.frames_remaining
            if r.frames_remaining is not None
            else r.state.player.timing.delay + r.state.player.timing.replan
            for r in records
        ]
    )
    return DecisionBatch(
        observation_frame_J=integers([r.state.clock for r in records]),
        decision_frame_J=integers([r.state.clock for r in records]),
        sampled_indices_JF=torch.stack([r.actions[..., DECODE_GROUP_INDICES].flatten() for r in records]).to(device),
        committed_prefix_JD4=prefixes.to(device, non_blocking=True),
        behavior_log_probs_JF=torch.stack([r.behavior for r in records]).to(device),
        executed_factors_JF=executed_factor_mask(delay, replan, remaining),
        duration_J=integers([len(r.rewards) for r in records]),
        bootstrap_mask_J=torch.tensor([not r.terminal for r in records], device=device),
        trace_mask_J=torch.tensor([not r.boundary for r in records], device=device),
        kl_coefficient_J=torch.tensor([math.exp(r.state.player.log_lambda) for r in records], device=device),
        delay_J=delay,
        replan_J=replan,
        policy_version_J=integers([r.state.window[0] for r in records]),
        episode_id_J=integers([r.state.episode.number for r in records]),
        segment_id_J=integers([r.state.episode.number for r in records]),
    )


class RolloutEngine:
    def __init__(
        self,
        groups: tuple[tuple[SimulatorShard, ...], ...],
        policy: LoopPolicy,
        sampler: EpisodeSampler,
        anchors: dict[int, LoopPolicy],
    ) -> None:
        if not groups or any(not group for group in groups):
            raise ValueError("rollout groups and shards must be nonempty")
        ids = [match for group in groups for shard in group for match in shard.match_ids]
        if not ids or len(set(ids)) != len(ids) or min(ids) < 0:
            raise ValueError("logical match ids must be nonempty, nonnegative, and globally unique")
        if 0 not in anchors:
            raise ValueError("anchor zero must be the frozen reference")
        student_parameters = {id(parameter) for parameter in policy.model.parameters()}
        for anchor in anchors.values():
            if any(parameter.requires_grad for parameter in anchor.model.parameters()) or any(
                id(parameter) in student_parameters for parameter in anchor.model.parameters()
            ):
                raise ValueError("anchors must be frozen and have independent parameters")
        self.groups = groups
        self.policy = policy
        self.sampler = sampler
        self.anchors = anchors
        self.anchor_candidates = tuple(sorted(anchors))
        self.clocks = [0] * len(groups)
        self.cursor = 0
        self.matches: dict[int, MatchState] = {}
        self._initialized = False
        self.completed_update = -1
        self.results: list[EpisodeResult] = []

    def initialize(self) -> None:
        if self._initialized:
            raise ValueError("pool is already initialized")
        for group in self.groups:
            for shard in group:
                for lane, match in enumerate(shard.match_ids):
                    self._reset(shard, lane, match, 0, 0)
        self._initialized = True

    def _reset(self, shard: SimulatorShard, lane: int, match: int, clock: int, number: int) -> None:
        if clock % 128:
            raise ValueError("episodes must start on a block boundary")
        episode = self.sampler.sample(match, number, self.anchor_candidates)
        frame = shard.reset(lane, episode)
        if frame.terminal or frame.timeout or bool(frame.applied_actions.any()):
            raise ValueError("reset must return a live initial state with neutral applied inputs")
        frame = dataclasses.replace(frame, columns=dict(frame.columns), applied_actions=frame.applied_actions.copy())
        queues: tuple[list[Tensor], list[Tensor]] = ([], [])
        for side, player in enumerate(episode.players):
            queues[side].extend(torch.zeros(4, dtype=torch.long) for _ in range(player.timing.delay))
        self.matches[match] = MatchState(episode, clock, clock, [frame], queues)

    def _state(self, match: int, side: int, clock: int, update: int, *, bootstrap: bool = False) -> DecisionState:
        state = self.matches[match]
        block = (clock // 128) * 128
        if bootstrap and clock % 128 == 0:
            block -= 128
        start = max(state.episode_start, block - 128, clock - 255)
        frames = tuple(state.history[start - state.history_start : clock - state.history_start + 1])
        if not frames or len(frames) > 256:
            raise ValueError("invalid rollout history span")
        delay = state.episode.players[side].timing.delay
        if len(state.queues[side]) < delay:
            raise ValueError("committed queue is shorter than D")
        prefix = torch.stack(state.queues[side][:delay]) if delay else torch.empty(0, 4, dtype=torch.long)
        other = state.queues[1 - side]
        opponent = torch.stack(other) if other else torch.empty(0, 4, dtype=torch.long)
        return DecisionState(
            2 * match + side,
            match,
            side,
            state.episode,
            clock,
            (update, 2 * match + side, block, state.episode.number),
            frames,
            prefix,
            opponent,
        )

    def _decide(
        self,
        shards: Sequence[SimulatorShard],
        clock: int,
        update: int,
        records: list[DecisionRecord],
        pending: dict[int, DecisionRecord],
    ) -> None:
        states: dict[int | None, list[DecisionState]] = defaultdict(list)
        for shard in shards:
            for match in shard.match_ids:
                state = self.matches[match]
                if state.end_clock is not None:
                    continue
                for side, player in enumerate(state.episode.players):
                    if clock % player.timing.replan:
                        continue
                    if len(state.queues[side]) != player.timing.delay:
                        raise ValueError("decision must see exactly D committed frames")
                    decision = self._state(match, side, clock, update)
                    states[player.anchor].append(decision)
        # Capture both queues before either side samples; the baseline cannot read
        # an opponent action sampled at the same clock tick.
        for anchor, decisions in states.items():
            policy = self.policy if anchor is None else self.anchors[anchor]
            with torch.no_grad():
                output = policy.sample(decisions, self.sampler.rng)
            sampled_actions = output.actions.detach().cpu()
            sampled_log_probs = output.log_probs.detach().cpu()
            for index, decision in enumerate(decisions):
                actions = sampled_actions[index].clone()
                self.matches[decision.match].queues[decision.side].extend(
                    action.clone() for action in actions[: decision.player.timing.replan]
                )
                if anchor is not None:
                    continue
                previous = pending.pop(decision.stream, None)
                if previous is not None:
                    previous.next_state = decision
                record = DecisionRecord(decision, actions, sampled_log_probs[index].clone(), [])
                records.append(record)
                pending[decision.stream] = record

    def _inputs(self, shard: SimulatorShard, count: int) -> NDArray[np.float32]:
        indices = torch.zeros(count, len(shard.match_ids), 2, 4, dtype=torch.long)
        for lane, match in enumerate(shard.match_ids):
            if self.matches[match].end_clock is not None:
                continue
            for side in range(2):
                queue = self.matches[match].queues[side]
                if len(queue) < count:
                    raise ValueError("simulator step would consume an uncommitted action")
                indices[:, lane, side] = torch.stack(queue[:count])
        return self.policy.wire_actions(indices)

    def _accept(
        self,
        shard: SimulatorShard,
        rows: tuple[tuple[SimFrame, ...], ...],
        clock: int,
        update: int,
        records: list[DecisionRecord],
        pending: dict[int, DecisionRecord],
    ) -> tuple[int, int]:
        frames_stepped = timeouts = 0
        for at, frames in enumerate(rows, start=1):
            if len(frames) != len(shard.match_ids):
                raise ValueError("simulator returned the wrong lane count")
            for match, frame in zip(shard.match_ids, frames, strict=True):
                if not isinstance(shard, NativeSimulatorShard):
                    frame = dataclasses.replace(
                        frame, columns=dict(frame.columns), applied_actions=frame.applied_actions.copy()
                    )
                state = self.matches[match]
                if state.end_clock is not None:
                    continue
                frames_stepped += 1
                state.history.append(frame)
                state.returns = (state.returns[0] + frame.rewards[0], state.returns[1] + frame.rewards[1])
                for side in range(2):
                    state.queues[side].pop(0)
                    record = pending.get(2 * match + side)
                    if record is not None:
                        record.rewards.append(frame.rewards[side])
                if not (frame.terminal or frame.timeout):
                    continue
                if at != len(rows):
                    raise ValueError("simulator stepped beyond an episode boundary")
                end_clock = clock + at
                stocks = None
                if "p1_stock" in frame.columns and "p2_stock" in frame.columns:
                    stocks = (int(frame.columns["p1_stock"]), int(frame.columns["p2_stock"]))
                self.results.append(
                    EpisodeResult(
                        match, state.episode, end_clock - state.episode_start, state.returns, frame.timeout, stocks
                    )
                )
                for record in records:
                    if record.state.match == match and record.state.episode.number == state.episode.number:
                        record.frames_remaining = end_clock - record.state.clock
                for side in range(2):
                    record = pending.pop(2 * match + side, None)
                    if record is not None:
                        record.boundary = True
                        record.terminal = frame.terminal
                        if frame.timeout:
                            record.next_state = self._state(match, side, end_clock, update, bootstrap=True)
                state.end_clock = end_clock
                timeouts += int(frame.timeout)
        return frames_stepped, timeouts

    def _reset_waiting(self, shards: Sequence[SimulatorShard], clock: int) -> int:
        if clock % 128:
            return 0
        resets = 0
        for shard in shards:
            for lane, match in enumerate(shard.match_ids):
                state = self.matches[match]
                if state.end_clock is not None:
                    self._reset(shard, lane, match, clock, state.episode.number + 1)
                    resets += 1
        return resets

    def _active(self, shard: SimulatorShard) -> NDArray[np.bool_]:
        return np.array([self.matches[match].end_clock is None for match in shard.match_ids], dtype=np.bool_)

    def collect(self, update: int) -> Rollout:
        if not self._initialized:
            raise ValueError("initialize or restore the pool before rollout")
        started = time.perf_counter()
        self.results = []
        group_id = self.cursor
        shards = self.groups[group_id]
        start = self.clocks[group_id]
        count = self.sampler.schedule.rollout_frames
        if count % 128:
            raise ValueError("rollouts must end on a 128-frame block boundary")
        self.policy.clear_caches()
        for anchor in self.anchors.values():
            anchor.clear_caches()
        records: list[DecisionRecord] = []
        pending: dict[int, DecisionRecord] = {}
        resets = timeouts = frames_stepped = 0
        split = (len(shards) + 1) // 2
        with ThreadPoolExecutor(max_workers=len(shards)) as workers:
            for clock in range(start, start + count, 2):
                resets += self._reset_waiting(shards, clock)
                futures = []
                for half in (shards[:split], shards[split:]):
                    self._decide(half, clock, update, records, pending)
                    for shard in half:
                        active = self._active(shard)
                        if active.any():
                            futures.append((shard, workers.submit(shard.step, self._inputs(shard, 2), active)))
                for shard, future in futures:
                    rows = future.result()
                    if not 1 <= len(rows) <= 2:
                        raise ValueError("simulator must return between one and two frames")
                    stepped, ended_timeouts = self._accept(shard, rows, clock, update, records, pending)
                    frames_stepped += stepped
                    timeouts += ended_timeouts
                    if len(rows) == 1:
                        active = self._active(shard)
                        if not active.any():
                            continue
                        rest = shard.step(self._inputs(shard, 1), active)
                        if len(rest) != 1:
                            raise ValueError("one-frame step returned the wrong duration")
                        stepped, ended_timeouts = self._accept(shard, rest, clock + 1, update, records, pending)
                        frames_stepped += stepped
                        timeouts += ended_timeouts
        end = start + count
        resets += self._reset_waiting(shards, end)
        for record in pending.values():
            record.boundary = True
            record.next_state = self._state(record.state.match, record.state.side, end, update, bootstrap=True)
        for shard in shards:
            for match in shard.match_ids:
                state = self.matches[match]
                # The current observation is separate from the previous 128 frames.
                state.history = state.history[-129:]
                state.history_start = end - len(state.history) + 1
        self.clocks[group_id] = end
        self.cursor = (group_id + 1) % len(self.groups)
        self.completed_update = update
        return Rollout(
            tuple(records),
            group_id,
            update,
            frames_stepped,
            resets,
            timeouts,
            time.perf_counter() - started,
            tuple(self.results),
            count * sum(len(s.match_ids) for s in shards) - frames_stepped,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class TrainConfig:
    seed: int = 65
    actor_optimizer: Literal["muon", "adam"] = "muon"
    actor_lr: float = 1e-5
    critic_lr: float = 3e-4
    critic_epochs: int = 4
    windows_per_step: int = 32
    decisions_per_window: int = 16
    clip_epsilon: float = 0.2
    importance_cap: float = 2.0
    gradient_clip: float = 1.0
    value_scale: float = 150.0
    warmup_steps: int = 32
    critic_warmup_updates: int = 172
    warmup_match_frames: int = 11008
    plateau_patience: int = 8
    plateau_tolerance: float = 0.01
    lambda_spread_multiplier: float = 3.0
    snapshot_interval: int = 256
    snapshot_capacity: int = 3
    checkpoint_interval: int = 32
    deterministic: bool = False

    def __post_init__(self) -> None:
        if type(self.deterministic) is not bool:
            raise ValueError("deterministic must be a boolean")
        for name in (
            "critic_epochs",
            "windows_per_step",
            "decisions_per_window",
            "plateau_patience",
            "snapshot_interval",
            "snapshot_capacity",
            "checkpoint_interval",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 1 <= self.decisions_per_window <= 16:
            raise ValueError("at most 16 decisions can train per window")
        for value in (self.warmup_steps, self.critic_warmup_updates, self.warmup_match_frames):
            if type(value) is not int or value < 0:
                raise ValueError("warmup lengths must be nonnegative integers")
        for name in (
            "actor_lr",
            "critic_lr",
            "importance_cap",
            "gradient_clip",
            "value_scale",
            "lambda_spread_multiplier",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.clip_epsilon < 1 or not math.isfinite(self.plateau_tolerance) or self.plateau_tolerance < 0:
            raise ValueError("invalid clip or plateau tolerance")
        if self.actor_optimizer not in ("muon", "adam"):
            raise ValueError("unsupported actor optimizer")


def make_actor_optimizer(policy: LoopPolicy, config: TrainConfig) -> torch.optim.Optimizer:
    parameters = policy.actor_parameters()
    if config.actor_optimizer == "adam":
        return torch.optim.AdamW(parameters, lr=config.actor_lr, weight_decay=0)
    allowed = {id(p) for p in parameters}
    buckets: dict[tuple[bool, int], list[torch.nn.Parameter]] = defaultdict(list)
    for name, parameter in policy.model.named_parameters():
        if id(parameter) not in allowed:
            continue
        muon = parameter.ndim == 2 and name.startswith(
            ("trunk.blocks.", "temporal.blocks.", "temporal.history_attention.", "temporal.token_projection.weight")
        )
        if name.startswith(("temporal.outputs.", "temporal.trunk_outputs.")) and name.endswith("up.weight"):
            muon = True
        splits = 3 if name.endswith(("attn.c_attn.weight", "qkv.weight")) else 1
        if name.endswith("key_value.weight"):
            splits = 2
        buckets[muon, splits if muon else 1].append(parameter)
    groups: list[dict[str, object]] = []
    for (muon, splits), values in buckets.items():
        group: dict[str, object] = {"params": values, "lr": config.actor_lr, "weight_decay": 0.0, "use_muon": muon}
        if muon:
            group.update(momentum=0.95, logical_splits=splits, muon_scale_clamp_min_one=False)
        else:
            group.update(betas=(0.9, 0.95), eps=1e-8)
        groups.append(group)
    return SingleDeviceMuonWithAuxAdam(groups)


def _take_features(features: CriticFeatures, indices: Tensor) -> CriticFeatures:
    return CriticFeatures(
        *(getattr(features, field.name).index_select(0, indices) for field in dataclasses.fields(features))
    )


def _take_decisions(decisions: DecisionBatch, indices: Tensor) -> DecisionBatch:
    return DecisionBatch(
        *(getattr(decisions, field.name).index_select(0, indices) for field in dataclasses.fields(decisions))
    )


def _log_distribution(probabilities: Tensor) -> Tensor:
    positive = probabilities > 0
    return torch.where(positive, probabilities, 1).log().masked_fill(~positive, -torch.inf)


def factor_kl(policy: PolicyEvaluation, reference: PolicyEvaluation) -> Tensor:
    return torch.stack(
        [
            categorical_kl(_log_distribution(p), _log_distribution(q))
            for p, q in zip(policy.distributions, reference.distributions, strict=True)
        ],
        -1,
    ).flatten(1)


def explained_variance(values: Tensor, targets: Tensor) -> float:
    variance = targets.double().var(unbiased=False)
    if float(variance) == 0:
        return math.nan
    return float(1 - (targets.double() - values.double()).var(unbiased=False) / variance)


@dataclasses.dataclass(frozen=True, slots=True)
class PreparedUpdate:
    rollout: Rollout
    decisions: DecisionBatch
    old: PolicyEvaluation
    reference: PolicyEvaluation | None
    values: Tensor
    advantages: Tensor
    targets: Tensor
    selected: Tensor
    steps: tuple[tuple[int, ...], ...]
    metrics: dict[str, float]


def select_decisions(
    records: Sequence[DecisionRecord],
    eligible: Tensor,
    rng: CounterRng,
    update: int,
    count: int,
    windows_per_step: int,
) -> tuple[Tensor, tuple[tuple[int, ...], ...]]:
    windows: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        windows[record.state.window[:3]].append(index)
    selected = torch.zeros_like(eligible)
    eligible_host = eligible.cpu().tolist()
    chosen: list[int] = []
    by_match: dict[int, list[list[int]]] = defaultdict(list)
    for indices in windows.values():
        candidates = [index for index in indices if eligible_host[index]]
        candidates.sort(
            key=lambda j: rng.uniform(
                "decision_subsample", records[j].state.stream, records[j].state.episode.number, records[j].state.clock
            )
        )
        chosen.extend(candidates[:count])
        if candidates:
            by_match[records[indices[0]].state.match].append(indices)
    selected[chosen] = True
    if not by_match:
        return selected, ()
    total = sum(len(windows) for windows in by_match.values())
    slots = max(max(len(windows) for windows in by_match.values()), math.ceil(total / windows_per_step))
    placement: list[list[list[int]]] = [[] for _ in range(slots)]
    match_ids = sorted(by_match, key=lambda match: rng.uniform("minibatch_placement", match, 0, update))
    for match in match_ids:
        used: set[int] = set()
        for window in by_match[match]:
            choices = [slot for slot in range(slots) if slot not in used and len(placement[slot]) < windows_per_step]
            if not choices:
                raise ArithmeticError("cannot place match windows in distinct actor steps")
            slot = min(
                choices, key=lambda slot: (len(placement[slot]), rng.uniform("placement_tie", match, update, slot))
            )
            placement[slot].append(window)
            used.add(slot)
    steps = tuple(tuple(index for window in step for index in window) for step in placement if step)
    return selected, steps


class Trainer:
    def __init__(self, engine: RolloutEngine, config: TrainConfig) -> None:
        if torch.are_deterministic_algorithms_enabled() != config.deterministic:
            raise ValueError("trainer determinism differs from the configured execution mode")
        if engine.sampler.rng.seed != config.seed:
            raise ValueError("trainer and counter RNG seeds must agree")
        self.engine = engine
        self.policy = engine.policy
        self.reference = engine.anchors[0]
        self.config = config
        actor_ids = {id(p) for p in self.policy.actor_parameters()}
        critic_ids = {id(p) for p in self.policy.critic_parameters()}
        if not actor_ids or not critic_ids or actor_ids & critic_ids:
            raise ValueError("actor and critic parameter sets must be nonempty and disjoint")
        self.actor_optimizer = make_actor_optimizer(self.policy, config)
        self.critic_optimizer = torch.optim.AdamW(self.policy.critic_parameters(), lr=config.critic_lr, weight_decay=0)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.actor_optimizer, lambda step: min(1.0, (step + 1) / max(1, config.warmup_steps))
        )
        self.update = 0
        self.actor_steps = 0
        self.warming = config.critic_warmup_updates > 0
        self.warmup_moments: dict[DelayReplan, tuple[int, float, float]] = {}
        self.fresh_variances: list[float] = []
        self.next_anchor = max(engine.anchors) + 1
        self.evaluation_cursor = 0
        self.updating = False

    @torch.no_grad()
    def prepare(self, rollout: Rollout) -> PreparedUpdate:
        started = time.perf_counter()
        device = self.policy.device
        records = rollout.records
        decisions = _decision_batch(records, device)
        states = [r.state for r in records]
        actions = torch.stack([r.actions for r in records]).to(device)
        old = self.policy.evaluate(states, actions)
        reference = None if self.warming else self.reference.evaluate(states, actions)
        kl = torch.zeros_like(old.log_probs) if reference is None else factor_kl(old, reference)
        frame_rewards = torch.tensor(
            [record.rewards + [0.0] * (4 - len(record.rewards)) for record in records], device=device
        )
        rewards = regularized_reward(decisions, frame_rewards, kl)
        values = self.policy.values(old.features)
        next_values = torch.zeros_like(values)
        # Bootstrap from the next decision inside an episode, or from the final
        # pre-reset observation at a timeout.
        streams: dict[int, list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            streams[record.state.stream].append(index)
        boundary_indices: list[int] = []
        boundary_states: list[DecisionState] = []
        interior_indices: list[int] = []
        following_indices: list[int] = []
        for indices in streams.values():
            for position, index in enumerate(indices):
                record = records[index]
                if record.terminal:
                    continue
                if record.boundary:
                    if record.next_state is None:
                        raise ValueError("nonterminal boundary is missing its bootstrap state")
                    boundary_indices.append(index)
                    boundary_states.append(record.next_state)
                else:
                    following = indices[position + 1]
                    if records[following].state.episode.number != record.state.episode.number:
                        raise ValueError("GAE trace crossed an episode reset")
                    interior_indices.append(index)
                    following_indices.append(following)
        next_values[interior_indices] = values[following_indices]
        if boundary_states:
            neutral = torch.zeros(len(boundary_states), 4, 4, dtype=torch.long, device=device)
            bootstrap = self.policy.evaluate(boundary_states, neutral)
            next_values[boundary_indices] = self.policy.values(bootstrap.features)
        result = stream_advantages(
            rewards,
            values,
            next_values,
            decisions.duration_J,
            decisions.bootstrap_mask_J,
            decisions.trace_mask_J,
            list(streams.values()),
        )
        advantages, targets = result.advantages, result.value_targets
        selected, steps = select_decisions(
            records,
            decisions.loss_mask_J,
            self.engine.sampler.rng,
            rollout.update,
            self.config.decisions_per_window,
            self.config.windows_per_step,
        )
        executed = decisions.executed_factors_JF
        gap = torch.where(executed, old.log_probs - decisions.behavior_log_probs_JF, 0).sum(-1)
        metrics = {
            "critic/explained_variance": explained_variance(values, targets),
            "numerics/mu_old_gap_mean": float(-gap.mean()),
            "numerics/mu_old_gap_p50": float(gap.abs().quantile(0.5)),
            "numerics/mu_old_gap_p99": float(gap.abs().quantile(0.99)),
            "numerics/mu_old_gap_max": float(gap.abs().max()),
            "numerics/truncated_weight_fraction": float((gap > math.log(self.config.importance_cap)).float().mean()),
            "policy/kl_per_frame": float(torch.where(executed, kl, 0).sum() / decisions.replan_J.sum()),
            "return/interval_reward": float(frame_rewards.sum(-1).mean()),
            "data/anchor_share": sum(
                player.anchor is not None for state in self.engine.matches.values() for player in state.episode.players
            )
            / len(self.engine.matches)
            / 2,
            "data/resets": float(rollout.resets),
            "data/timeouts": float(rollout.timeouts),
            "systems/rollout_seconds": rollout.elapsed,
            "systems/no_grad_seconds": time.perf_counter() - started,
            "systems/simulator_fps": rollout.frames / rollout.elapsed,
            "systems/rollout_frames": float(rollout.frames),
            "systems/reset_idle_frames": float(rollout.idle_frames),
            "systems/reset_idle_fraction": rollout.idle_frames / (rollout.frames + rollout.idle_frames),
            "data/decisions": float(len(records)),
            "data/trained_decisions": float(selected.sum()),
        }
        log_min = math.log(self.engine.sampler.config.lambda_min)
        log_max = math.log(self.engine.sampler.config.lambda_max)
        log_values = decisions.kl_coefficient_J.log()
        bins = ((log_values - log_min) / max(log_max - log_min, 1e-12) * 4).long().clamp(0, 3)
        for index in range(4):
            mask = bins == index
            if bool(mask.any()):
                metrics[f"return/interval_reward_lambda{index}"] = float(frame_rewards[mask].sum(-1).mean())
                metrics[f"policy/kl_per_frame_lambda{index}"] = float(
                    torch.where(executed[mask], kl[mask], 0).sum() / decisions.replan_J[mask].sum()
                )
        for position in range(4):
            mask = (decisions.decision_frame_J % 128) // 32 == position
            if bool(mask.any()):
                metrics[f"critic/explained_variance_position{position}"] = explained_variance(
                    values[mask], targets[mask]
                )
        for name, getter in (
            ("identity", lambda r: r.state.player.identity),
            ("character", lambda r: r.state.player.character),
            ("stage", lambda r: r.state.episode.stage),
        ):
            counts: dict[int, int] = defaultdict(int)
            for record in records:
                counts[getter(record)] += 1
            for key, value in counts.items():
                metrics[f"data/{name}_{key}"] = float(value)
        final_indices = []
        for j in boundary_indices:
            state = records[j].next_state
            if state is not None and state.clock % 256 == 0:
                final_indices.append(j)
        if final_indices:
            shorter = []
            for j in final_indices:
                state = records[j].next_state
                if state is None:
                    raise ValueError("missing rollout bootstrap state")
                shorter.append(
                    dataclasses.replace(
                        state,
                        frames=state.frames[-129:],
                        window=(state.window[0], state.stream, state.clock, state.episode.number),
                    )
                )
            replay = self.policy.evaluate(shorter, torch.zeros(len(shorter), 4, 4, dtype=torch.long, device=device))
            metrics["critic/bootstrap_context_gap"] = float(
                (self.policy.values(replay.features) - next_values[final_indices]).abs().mean()
            )
        for timing in self.engine.sampler.schedule.configurations:
            mask = (decisions.delay_J == timing.delay) & (decisions.replan_J == timing.replan)
            if not bool(mask.any()):
                continue
            suffix = f"d{timing.delay}_r{timing.replan}"
            metrics[f"critic/advantage_mean_{suffix}"] = float(advantages[mask].mean())
            metrics[f"critic/advantage_std_{suffix}"] = float(advantages[mask].std(unbiased=False))
            metrics[f"critic/explained_variance_{suffix}"] = explained_variance(values[mask], targets[mask])
            metrics[f"numerics/mu_old_gap_p99_{suffix}"] = float(gap[mask].abs().quantile(0.99))
        for factor, distribution in zip(FACTOR_ORDER, old.distributions, strict=True):
            entropy = -(distribution * torch.where(distribution > 0, distribution, 1).log()).sum(-1)
            valid = torch.arange(4, device=device)[None] < decisions.replan_J[:, None]
            metrics[f"policy/entropy_{factor}"] = float(entropy[valid].mean())
        return PreparedUpdate(
            rollout, decisions, old, reference, values, advantages, targets, selected, steps, metrics
        )

    def fit(self, batch: PreparedUpdate) -> dict[str, float]:
        if not self.warming and batch.reference is None:
            raise ValueError("actor update requires a reference evaluation")
        cfg = self.config
        metrics = dict(batch.metrics)
        started = time.perf_counter()
        features = batch.old.features.detached()
        count = len(batch.values)
        selected = torch.ones(count, dtype=torch.bool, device=self.policy.device)
        for _ in range(cfg.critic_epochs):
            self.critic_optimizer.zero_grad(set_to_none=True)
            loss = critic_loss(
                self.policy.values(features),
                batch.targets,
                selected,
                value_scale=cfg.value_scale,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.critic_parameters(), cfg.gradient_clip, error_if_nonfinite=True)
            self.critic_optimizer.step()
        metrics["critic/loss"] = float(loss.detach())
        metrics["systems/critic_seconds"] = time.perf_counter() - started
        if self.warming:
            self._warmup(batch)
            return metrics
        metrics["policy/empty_update"] = float(not batch.steps)
        if not batch.steps:
            return metrics
        assert batch.reference is not None
        started = time.perf_counter()
        policy_losses: list[float] = []
        kl_losses: list[float] = []
        ratios: list[Tensor] = []
        old_log_ratios: list[Tensor] = []
        selected_host = batch.selected.cpu().tolist()
        for step, indices in enumerate(batch.steps):
            indices = tuple(index for index in indices if selected_host[index])
            at = torch.tensor(indices, device=self.policy.device)
            decisions = _take_decisions(batch.decisions, at)
            selected = batch.selected[at]
            statistics = advantage_statistics(advantage_moments(batch.advantages[at], selected))
            records = [batch.rollout.records[index] for index in indices]
            self.actor_optimizer.zero_grad(set_to_none=True)
            new = self.policy.evaluate([record.state for record in records], batch.old.actions[at])
            reference = _take_evaluation(batch.reference, at)
            policy_loss = ppo_loss(
                decisions,
                new.log_probs,
                batch.old.log_probs[at],
                batch.advantages[at],
                selected,
                statistics,
                clip_epsilon=cfg.clip_epsilon,
                importance_cap=cfg.importance_cap,
            )
            kl_loss = differentiable_kl_loss(decisions, factor_kl(new, reference), selected, statistics)
            loss = policy_loss.loss + kl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.actor_parameters(), cfg.gradient_clip, error_if_nonfinite=True)
            self.actor_optimizer.step()
            self.scheduler.step()
            self.actor_steps += 1
            policy_losses.append(float(policy_loss.loss.detach()))
            kl_losses.append(float(kl_loss.detach()))
            ratios.append(policy_loss.joint_ratio_J[selected].detach())
            old_gap = torch.where(
                decisions.executed_factors_JF, new.log_probs.detach() - batch.old.log_probs[at], 0
            ).sum(-1)
            old_log_ratios.append(old_gap[selected])
            metrics[f"policy/old_log_ratio_step{step}"] = float(old_gap[selected].mean())
            old_policy = _take_evaluation(batch.old, at)
            old_kl = factor_kl(new, old_policy).detach()
            metrics[f"policy/old_kl_step{step}"] = float(
                torch.where(decisions.executed_factors_JF & selected[:, None], old_kl, 0).sum() / selected.sum()
            )
        ratio = torch.cat(ratios)
        metrics.update(
            {
                "policy/loss": sum(policy_losses) / len(policy_losses),
                "policy/kl_loss": sum(kl_losses) / len(kl_losses),
                "policy/clip_fraction": float(((ratio - 1).abs() > cfg.clip_epsilon).float().mean()),
                "policy/ratio_p01": float(ratio.quantile(0.01)),
                "policy/ratio_p99": float(ratio.quantile(0.99)),
                "policy/old_log_ratio_p99": float(torch.cat(old_log_ratios).abs().quantile(0.99)),
                "systems/actor_seconds": time.perf_counter() - started,
            }
        )
        return metrics

    def _warmup(self, batch: PreparedUpdate) -> None:
        for timing in self.engine.sampler.schedule.configurations:
            mask = (batch.decisions.delay_J == timing.delay) & (batch.decisions.replan_J == timing.replan)
            values = batch.advantages[mask].double()
            previous = self.warmup_moments.get(timing, (0, 0.0, 0.0))
            self.warmup_moments[timing] = (
                previous[0] + len(values),
                previous[1] + float(values.sum()),
                previous[2] + float(values.square().sum()),
            )
        self.fresh_variances.append(batch.metrics["critic/explained_variance"])
        patience = self.config.plateau_patience
        if self.update + 1 < self.config.critic_warmup_updates or len(self.fresh_variances) < patience:
            return
        if min(self.engine.clocks) < self.config.warmup_match_frames:
            return
        recent = self.fresh_variances[-patience:]
        if not all(math.isfinite(x) for x in recent) or max(recent) - min(recent) > self.config.plateau_tolerance:
            return
        spreads = []
        for count, total, squares in self.warmup_moments.values():
            if count < 2:
                raise ValueError("lambda calibration needs advantages for every timing configuration")
            spreads.append(math.sqrt(max(0.0, squares / count - (total / count) ** 2)))
        maximum = self.config.lambda_spread_multiplier * max(spreads)
        sampler = self.engine.sampler
        if not sampler.config.lambda_min < maximum <= sampler.config.lambda_max:
            raise ValueError("calibrated lambda_max is outside the declared conditioner range")
        sampler.lambda_max = maximum
        self.warming = False

    def run_update(self) -> tuple[PreparedUpdate, dict[str, float]]:
        if self.updating:
            raise ValueError("previous update did not finish; restore an update-boundary checkpoint")
        self.updating = True
        started = time.perf_counter()
        native_shards = [
            shard for shard in self.engine.groups[self.engine.cursor] if isinstance(shard, NativeSimulatorShard)
        ]
        before = [
            (shard.step_seconds, shard.reset_seconds, shard.stepped_frames, shard.copied_bytes)
            for shard in native_shards
        ]
        if self.policy.device.type == "cuda":
            torch.cuda.synchronize(self.policy.device)
            torch.cuda.reset_peak_memory_stats(self.policy.device)
        rollout = self.engine.collect(self.update)
        self.policy.clear_caches()
        for anchor in self.engine.anchors.values():
            anchor.clear_caches()
        batch = self.prepare(rollout)
        metrics = self.fit(batch)
        for index, (shard, previous) in enumerate(zip(native_shards, before, strict=True)):
            elapsed = shard.step_seconds - previous[0]
            metrics[f"systems/native_shard{index}_seconds"] = elapsed
            metrics[f"systems/native_shard{index}_fps"] = (shard.stepped_frames - previous[2]) / elapsed
            metrics[f"systems/native_shard{index}_reset_seconds"] = shard.reset_seconds - previous[1]
            metrics[f"systems/native_shard{index}_h2d_bytes"] = float(shard.copied_bytes - previous[3])
        self.update += 1
        metrics.update(
            global_step=float(self.update),
            **{
                "train/actor_steps": float(self.actor_steps),
                "train/warming": float(self.warming),
                "train/lambda_max": self.engine.sampler.lambda_max,
            },
        )
        if not self.warming and self.update % self.config.snapshot_interval == 0:
            self.engine.anchors[self.next_anchor] = self.policy.frozen_copy()
            self.next_anchor += 1
            self.engine.anchor_candidates = (
                0,
                *sorted(key for key in self.engine.anchors if key)[-self.config.snapshot_capacity :],
            )
            # Keep an assigned snapshot until its last episode finishes.
            assigned = {p.anchor for m in self.engine.matches.values() for p in m.episode.players}
            snapshots = sorted(key for key in self.engine.anchors if key)
            for key in snapshots[: -self.config.snapshot_capacity]:
                if key not in assigned:
                    del self.engine.anchors[key]
        self.policy.clear_caches()
        if self.policy.device.type == "cuda":
            torch.cuda.synchronize(self.policy.device)
            metrics["systems/peak_allocated_bytes"] = float(torch.cuda.max_memory_allocated(self.policy.device))
            metrics["systems/peak_reserved_bytes"] = float(torch.cuda.max_memory_reserved(self.policy.device))
        metrics["systems/update_seconds"] = time.perf_counter() - started
        metrics["systems/match_frames_per_second"] = rollout.frames / metrics["systems/update_seconds"]
        self.updating = False
        return batch, metrics


def _durable_file(directory: Path, payload: bytes, suffix: str) -> tuple[str, str]:
    digest = hashlib.sha256(payload).hexdigest()
    name = f"{digest}.{suffix}"
    target = directory / name
    temporary = directory / f".{name}.{os.getpid()}.partial"
    with temporary.open("wb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, target)
    return name, digest


def _environment() -> dict[str, str]:
    import platform

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": str(torch.version.cuda),
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "world_size": "1",
        "tf32_matmul": str(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": str(torch.backends.cudnn.allow_tf32),
    }


def _configuration_digest(trainer: Trainer) -> str:
    engine = trainer.engine
    model = trainer.policy.model
    schema = [(name, tuple(value.shape), str(value.dtype)) for name, value in model.state_dict().items()]
    identity = {
        "train": dataclasses.asdict(trainer.config),
        "sampler": dataclasses.asdict(engine.sampler.config),
        "schedule": dataclasses.asdict(engine.sampler.schedule),
        "seed": engine.sampler.rng.seed,
        "schema": schema,
        "execution_device": str(trainer.policy.device),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "groups": [[s.match_ids for s in group] for group in engine.groups],
        "native_limits": [
            [s.max_frame if isinstance(s, NativeSimulatorShard) else None for s in group] for group in engine.groups
        ],
        "episode_reset": "shared_stocks_block_wait_v1",
        "rollout_storage": "owned_columns_write_once_device_blocks_v1",
    }
    if isinstance(trainer.policy, TransformerPolicy):
        identity["compiled"] = trainer.policy.compiled
        identity["decision_kernels"] = "fp32_packed_context_graph_append_blake2b_v2"
        identity["model"] = dataclasses.asdict(trainer.policy.model.cfg)
        identity["ranges"] = dataclasses.asdict(trainer.policy.model.ranges)
        identity["statistics"] = {key: dataclasses.asdict(value) for key, value in trainer.policy.statistics.items()}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def save_checkpoint(trainer: Trainer, directory: Path, provenance: dict[str, str]) -> Path:
    """Publish one manifest after the model and every simulator shard are durable.

    No cache is saved: every resumed rollout prefills under restored weights.
    Filenames are content hashes, so an interrupted replacement cannot corrupt
    a checkpoint which already has a manifest.
    """
    import io

    if not trainer.engine._initialized or trainer.updating or trainer.engine.completed_update != trainer.update - 1:
        raise ValueError("checkpoint requires a complete update boundary")
    required = {"git_sha", "checkpoint_sha256", "data_manifest_sha256", "simulator_sha256", "adapter_sha256"}
    if set(provenance) != required or any(not value for value in provenance.values()):
        raise ValueError("checkpoint provenance must identify source, BC checkpoint, data, simulator, and adapter")
    directory.mkdir(parents=True, exist_ok=True)
    engine = trainer.engine
    state = {
        "version": CHECKPOINT_VERSION,
        "model": trainer.policy.model.state_dict(),
        "actor_optimizer": trainer.actor_optimizer.state_dict(),
        "critic_optimizer": trainer.critic_optimizer.state_dict(),
        "scheduler": trainer.scheduler.state_dict(),
        "reference": trainer.reference.model.state_dict(),
        "anchors": {key: value.model.state_dict() for key, value in engine.anchors.items()},
        "anchor_candidates": engine.anchor_candidates,
        "matches": engine.matches,
        "clocks": engine.clocks,
        "cursor": engine.cursor,
        "sampler_cursor": engine.sampler.cursor,
        "lambda_max": engine.sampler.lambda_max,
        "update": trainer.update,
        "actor_steps": trainer.actor_steps,
        "warming": trainer.warming,
        "warmup_moments": trainer.warmup_moments,
        "fresh_variances": trainer.fresh_variances,
        "next_anchor": trainer.next_anchor,
        "evaluation_cursor": trainer.evaluation_cursor,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    buffer = io.BytesIO()
    torch.save(state, buffer)
    name, digest = _durable_file(directory, buffer.getvalue(), "rank0.pt")
    shards = [{"name": name, "sha256": digest, "role": "training"}]
    for group_id, group in enumerate(engine.groups):
        for shard_id, shard in enumerate(group):
            name, digest = _durable_file(directory, zlib.compress(shard.save()), "sim.zlib")
            shards.append({"name": name, "sha256": digest, "role": f"simulator_{group_id}_{shard_id}"})
    if isinstance(trainer.policy, TransformerPolicy):
        actor = directory / f".actor-{os.getpid()}-{time.time_ns()}.pt"
        try:
            save_evaluation_checkpoint(
                actor,
                trainer.policy.model,
                trainer.policy.statistics,
                provenance={
                    "reference_sha256": provenance["checkpoint_sha256"],
                    "git_sha": provenance["git_sha"],
                    "update": trainer.update,
                    "data_manifests": {"training": provenance["data_manifest_sha256"]},
                },
            )
            name, digest = _durable_file(directory, actor.read_bytes(), "actor.pt")
            shards.append({"name": name, "sha256": digest, "role": "actor"})
        finally:
            actor.unlink(missing_ok=True)
    manifest = {
        "format": TRAINING_FORMAT,
        "version": CHECKPOINT_VERSION,
        "experiment": "065_selfplay_ppo",
        "update": trainer.update,
        "configuration_sha256": _configuration_digest(trainer),
        "environment": _environment(),
        "provenance": provenance,
        "shards": shards,
    }
    temporary = directory / f".manifest-{trainer.update}-{os.getpid()}.partial"
    target = directory / f"update-{trainer.update:08d}.json"
    with temporary.open("w") as output:
        json.dump(manifest, output, sort_keys=True)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, target)
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return target


def restore_checkpoint(trainer: Trainer, manifest_path: Path, provenance: dict[str, str]) -> None:
    manifest = read_training_manifest(manifest_path)
    if manifest["configuration_sha256"] != _configuration_digest(trainer):
        raise ValueError("resume configuration mismatch")
    if manifest["environment"] != _environment() or manifest["provenance"] != provenance:
        raise ValueError("resume environment or checkpoint mismatch")
    paths: dict[str, Path] = {}
    for shard in manifest["shards"]:
        if set(shard) != {"name", "sha256", "role"} or Path(shard["name"]).name != shard["name"]:
            raise ValueError("invalid checkpoint shard entry")
        path = manifest_path.parent / shard["name"]
        if shard["role"] in paths or sha256(path) != shard["sha256"]:
            raise ValueError("checkpoint shard hash mismatch or duplicate role")
        paths[shard["role"]] = path
    expected_roles = {"training"} | {
        f"simulator_{g}_{s}" for g, group in enumerate(trainer.engine.groups) for s in range(len(group))
    }
    if isinstance(trainer.policy, TransformerPolicy):
        expected_roles.add("actor")
        evaluation_actor_path(manifest_path)
    if set(paths) != expected_roles:
        raise ValueError("checkpoint simulator topology mismatch")
    saved = torch.load(paths["training"], map_location="cpu", weights_only=False)
    if saved["version"] != CHECKPOINT_VERSION or saved["update"] != manifest["update"]:
        raise ValueError("checkpoint shard identity mismatch")
    if "actor" in paths:
        actor = torch.load(paths["actor"], map_location="cpu", weights_only=False)["model"]
        if actor.keys() != saved["model"].keys() or any(
            not torch.equal(value, saved["model"][name]) for name, value in actor.items()
        ):
            raise ValueError("checkpoint training state and evaluation actor disagree")
    trainer.policy.model.load_state_dict(saved["model"], strict=True)
    trainer.actor_optimizer.load_state_dict(saved["actor_optimizer"])
    trainer.critic_optimizer.load_state_dict(saved["critic_optimizer"])
    trainer.scheduler.load_state_dict(saved["scheduler"])
    engine = trainer.engine
    engine.anchors.clear()
    for key, tensors in saved["anchors"].items():
        policy = trainer.policy.frozen_copy()
        policy.model.load_state_dict(tensors, strict=True)
        engine.anchors[key] = policy
    trainer.reference = engine.anchors[0]
    for name, value in saved["reference"].items():
        if not torch.equal(trainer.reference.model.state_dict()[name].cpu(), value):
            raise ValueError("checkpoint reference and anchor zero disagree")
    for group_id, group in enumerate(engine.groups):
        for shard_id, shard in enumerate(group):
            shard.restore(zlib.decompress(paths[f"simulator_{group_id}_{shard_id}"].read_bytes()))
    engine.matches = saved["matches"]
    engine.clocks = saved["clocks"]
    engine.cursor = saved["cursor"]
    engine.anchor_candidates = saved["anchor_candidates"]
    engine.sampler.cursor = saved["sampler_cursor"]
    engine.sampler.lambda_max = saved["lambda_max"]
    engine._initialized = True
    trainer.update = saved["update"]
    trainer.updating = False
    engine.completed_update = trainer.update - 1
    trainer.actor_steps = saved["actor_steps"]
    trainer.warming = saved["warming"]
    trainer.warmup_moments = saved["warmup_moments"]
    trainer.fresh_variances = saved["fresh_variances"]
    trainer.next_anchor = saved["next_anchor"]
    trainer.evaluation_cursor = saved["evaluation_cursor"]
    trainer.policy.clear_caches()
    random.setstate(saved["python_rng"])
    np.random.set_state(saved["numpy_rng"])
    torch.set_rng_state(saved["torch_rng"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(saved["cuda_rng"])


def load_development_policy(
    checkpoint: Path,
    statistics: dict[str, FeatureStats],
    *,
    lambda_min: float = 0.01,
    lambda_max: float = 100.0,
    device: str = "cuda",
    compiled: bool = False,
) -> TransformerPolicy:
    """Load the recorded O63 bootstrap30 checkpoint with fresh RL optimizer state."""
    if sha256(checkpoint) != PROXY_SHA256:
        raise ValueError("development checkpoint is not the recorded O63 bootstrap30 final.pt")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = saved["cfg"]
    if (config["experiment_id"], config["checkpoint_format_version"], config["arm"], saved["step"]) != (
        "063_percentile_conditioning_v1",
        5,
        "bootstrap30",
        16383,
    ):
        raise ValueError("development checkpoint identity mismatch")
    cfg = ActionSequenceConfig(
        d_model=256,
        n_layers=12,
        n_heads=4,
        temporal_d_model=256,
        temporal_layers=6,
        temporal_heads=4,
        temporal_ff_dim=1024,
        group_head_dim=256,
        value_hidden_dim=128,
    )
    if config["architecture"] != {name: getattr(cfg, name) for name in config["architecture"]}:
        raise ValueError("development checkpoint architecture mismatch")
    offsets = tuple(config["architecture"]["head_offsets"])
    schedule = development_config(offsets)
    if not 0 < lambda_min < lambda_max or not math.isfinite(lambda_max):
        raise ValueError("invalid development lambda range")
    state = saved["model"]
    vocabulary = vocabulary_from_checkpoint_buffer(state["player_code_bytes"].numpy())
    ranges = ConditioningRanges((math.log(lambda_min), math.log(lambda_max)), (8, 10), (2, 4))
    model = SelfPlayTransformer(cfg, ranges, vocabulary, max_opponent_queue=14, critic_hidden_dim=128)
    model.load_bc_state_dict(state)
    return TransformerPolicy(model.to(device).eval(), schedule, statistics, compiled=compiled)


def run_training(
    trainer: Trainer,
    updates: int,
    output: Path,
    provenance: dict[str, str],
    *,
    wandb_project: str | None = None,
    evaluation_shards: tuple[SimulatorShard, ...] | None = None,
    evaluation_config: EvaluationConfig | None = None,
    evaluation_interval: int = 128,
    periodic_evaluator: PeriodicEvaluator | None = None,
    run_configuration: dict[str, object] | None = None,
) -> None:
    """Train from initialized shards, publish checkpoints, and collect periodic results."""
    import wandb

    if updates <= trainer.update:
        raise ValueError("updates must exceed the restored update cursor")
    if (evaluation_shards is None) != (evaluation_config is None) or evaluation_interval < 1:
        raise ValueError("evaluation needs dedicated shards, a configuration, and a positive interval")
    output.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, object] = {
        "configuration_sha256": _configuration_digest(trainer),
        "environment": _environment(),
        "provenance": provenance,
        "train": dataclasses.asdict(trainer.config),
        "sampler": dataclasses.asdict(trainer.engine.sampler.config),
        "schedule": dataclasses.asdict(trainer.engine.sampler.schedule),
        "treatment": "KL-regularized joint PPO from pi_ref at u95",
        "control": "frozen pi_ref with matched sampling, starts, u95, and schedules",
        "invariants": "BC checkpoint, representation, data, simulator, adapter, reward, evaluation seeds",
        "run": run_configuration,
    }
    (output / "resolved.json").write_text(json.dumps(resolved, indent=2, sort_keys=True))
    with wandb.init(project=wandb_project, mode="disabled" if wandb_project is None else "online", config=resolved):
        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")
        with (output / "metrics.jsonl").open("a") as log:
            while trainer.update < updates:
                batch, metrics = trainer.run_update()
                del batch
                if periodic_evaluator is not None:
                    completed = periodic_evaluator.poll()
                    if completed is not None:
                        metrics.update(periodic_evaluation_metrics(completed))
                    if trainer.update % periodic_evaluator.every == 0:
                        manifest = save_checkpoint(trainer, output / "checkpoints", provenance)
                        if periodic_evaluator.submit(manifest, update=trainer.update):
                            trainer.evaluation_cursor += 1
                if (
                    evaluation_shards is not None
                    and evaluation_config is not None
                    and trainer.update % evaluation_interval == 0
                ):
                    metrics.update(evaluate_anchors(trainer, evaluation_shards, evaluation_config))
                resolved["calibrated_lambda_max"] = trainer.engine.sampler.lambda_max
                (output / "resolved.json").write_text(json.dumps(resolved, indent=2, sort_keys=True))
                wandb.log(metrics)
                log.write(json.dumps(metrics, sort_keys=True) + "\n")
                log.flush()
                if trainer.update % trainer.config.checkpoint_interval == 0:
                    save_checkpoint(trainer, output / "checkpoints", provenance)
            save_checkpoint(trainer, output / "checkpoints", provenance)
            if periodic_evaluator is not None:
                completed = periodic_evaluator.finish()
                if completed is not None:
                    metrics = {"global_step": float(trainer.update), **periodic_evaluation_metrics(completed)}
                    wandb.log(metrics)
                    log.write(json.dumps(metrics, sort_keys=True) + "\n")


def periodic_evaluation_metrics(path: Path) -> dict[str, float]:
    result = json.loads(path.read_text())
    if (
        result.get("format") not in ("hal.o65.evaluation-results.v1", "hal.o65.evaluation-results.v2")
        or result.get("split") != "selection"
    ):
        raise ValueError("unsupported periodic evaluation result identity")
    metrics = {"evaluation/update": float(result["checkpoint_provenance"]["update"])}
    for index, group in enumerate(result["groups"]):
        for simulator, values in group["metrics"].items():
            prefix = f"evaluation/{simulator}_cell{index}"
            metrics[f"{prefix}_matches"] = float(values["matches"])
            metrics[f"{prefix}_completed_pairs"] = float(values["completed_pairs"])
            for key in ("win_rate", "net_stocks_per_min"):
                value = values[key]["mean"]
                if value is not None:
                    metrics[f"{prefix}_{key}"] = float(value)
            if values["net_stock_lcb"] is not None:
                metrics[f"{prefix}_net_stock_lcb"] = float(values["net_stock_lcb"])
    return metrics


@dataclasses.dataclass(frozen=True, slots=True)
class ProxyCli:
    checkpoint: Path
    statistics: Path
    output: Path
    device: str = "cuda"
    compiled: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class NativeCli:
    checkpoint: Path
    sampler: Path
    output: Path
    data_dir: Path
    cpus: tuple[int, ...]
    updates: int
    groups: int = 4
    matches_per_shard: int = 2
    device: str = "cuda"
    compiled: bool = False
    max_frame: int = 28800
    resume: Path | None = None
    training: TrainConfig = TrainConfig()
    evaluation_every: int = 128
    evaluation_protocol: Path | None = None
    evaluation_simulators: tuple[Literal["msl", "dolphin"], ...] = ("msl", "dolphin")
    evaluation_device: str = "cpu"
    verify_resume: bool = False
    wandb_project: str | None = None

    def __post_init__(self) -> None:
        if (
            not self.cpus
            or len(set(self.cpus)) != len(self.cpus)
            or any(cpu not in os.sched_getaffinity(0) for cpu in self.cpus)
        ):
            raise ValueError("native runs need distinct available CPUs")
        if any(type(value) is not int or value < 1 for value in (self.groups, self.matches_per_shard, self.updates)):
            raise ValueError("groups, matches per shard, and updates must be positive integers")
        if type(self.evaluation_every) is not int or self.evaluation_every < 0:
            raise ValueError("evaluation interval must be nonnegative")
        if self.max_frame != -1 and (type(self.max_frame) is not int or self.max_frame < 0):
            raise ValueError("max frame must be -1 or nonnegative")
        if self.verify_resume and not self.training.deterministic:
            raise ValueError("exact resume verification requires training.deterministic=True")


def read_native_sampler(path: Path) -> SamplerConfig:
    raw = json.loads(path.read_text())
    if (
        not isinstance(raw, dict)
        or set(raw) != {"format", "identities", "matchups", "stages", "stocks", "percents"}
        or raw["format"] != "hal.o65.sampler.v1"
    ):
        raise ValueError("unsupported native sampler schema")
    identities = []
    for row in raw["identities"]:
        if set(row) != {"identity", "character", "count"}:
            raise ValueError("invalid sampler identity fields")
        identities.append(IdentityCount(row["identity"], row["character"], row["count"]))
    matchups = []
    for row in raw["matchups"]:
        if set(row) != {"characters", "weight"} or len(row["characters"]) != 2:
            raise ValueError("invalid sampler matchup fields")
        matchups.append(Matchup((row["characters"][0], row["characters"][1]), row["weight"]))
    stocks = tuple(raw["stocks"])
    return SamplerConfig(
        tuple(identities), tuple(matchups), tuple(raw["stages"]), stocks=stocks, percents=tuple(raw["percents"])
    )


def train_native(config: NativeCli) -> None:
    torch.set_num_threads(2)
    if config.training.deterministic:
        workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if workspace not in (None, ":4096:8"):
            raise ValueError("deterministic native runs require CUBLAS_WORKSPACE_CONFIG=:4096:8")
        if torch.cuda.is_initialized() and workspace is None:
            raise ValueError("configure deterministic execution before CUDA initialization")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(config.training.deterministic)
    if sha256(config.checkpoint) != PROXY_SHA256:
        raise ValueError("native development run requires the recorded O63 proxy")
    torch.manual_seed(config.training.seed)
    checkpoint = load_evaluation_checkpoint(config.checkpoint, device=torch.device(config.device))
    checkpoint.model.requires_grad_(True)
    checkpoint.model.percentile_head.requires_grad_(False)
    schedule = development_config(checkpoint.model.head_offsets)
    policy = TransformerPolicy(checkpoint.model, schedule, checkpoint.statistics, compiled=config.compiled)
    sampling = read_native_sampler(config.sampler)
    for row in sampling.identities:
        if row.identity >= policy.model.cfg.player_vocab_size:
            raise ValueError("sampler identity is outside the checkpoint vocabulary")
    manifests = checkpoint.provenance["data_manifests"]
    if not isinstance(manifests, dict) or set(manifests) != {"ranked-anonymized-1-policy-world-v8"}:
        raise ValueError("development checkpoint data manifests changed")
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if config.output.exists() and config.resume is None:
        raise FileExistsError(config.output)
    with contextlib.ExitStack() as resources:
        groups = []
        next_match = 0
        for _ in range(config.groups):
            shards: list[NativeSimulatorShard] = []
            for cpu in config.cpus:
                ids = tuple(range(next_match, next_match + config.matches_per_shard))
                shard = NativeSimulatorShard(
                    config.data_dir, ids, cpu=cpu, device=policy.device, max_frame=config.max_frame
                )
                resources.callback(shard.close)
                shards.append(shard)
                next_match += config.matches_per_shard
            groups.append(tuple(shards))
        provenance = {
            "git_sha": git_sha,
            "checkpoint_sha256": PROXY_SHA256,
            "data_manifest_sha256": str(cast(dict[str, object], manifests)["ranked-anonymized-1-policy-world-v8"]),
            "simulator_sha256": groups[0][0].shard.library_sha256,
            "adapter_sha256": native_bridge.library_sha256(),
        }
        engine = RolloutEngine(
            tuple(groups),
            policy,
            EpisodeSampler(sampling, schedule, CounterRng(config.training.seed)),
            {0: policy.frozen_copy()},
        )
        trainer = Trainer(engine, config.training)
        if config.resume is None:
            engine.initialize()
        else:
            restore_checkpoint(trainer, config.resume, provenance)
        evaluator = None
        if config.evaluation_every:
            arguments = [
                "--anchors",
                str(config.checkpoint.resolve()),
                "--data-dir",
                str(config.data_dir.resolve()),
                "--simulators",
                *config.evaluation_simulators,
                "--device",
                config.evaluation_device,
                "--cadence-grid",
                "8/4",
                "10/2",
            ]
            if config.evaluation_protocol is not None:
                arguments.extend(("--protocol", str(config.evaluation_protocol.resolve())))
            evaluator = PeriodicEvaluator(
                config.output / "evaluation", every=config.evaluation_every, arguments=tuple(arguments)
            )
            evaluator.last_update = trainer.update
            resources.callback(evaluator.close)
        native_config = json.loads(json.dumps(dataclasses.asdict(config), default=str))
        native_config["simulator_manifest_sha256"] = sha256(config.data_dir / "raw" / "manifest.json")
        native_config["sampler_sha256"] = sha256(config.sampler)
        run_training(
            trainer,
            config.updates,
            config.output,
            provenance,
            wandb_project=config.wandb_project,
            periodic_evaluator=evaluator,
            run_configuration=native_config,
        )
        if config.verify_resume:
            verify_checkpoint_resume(trainer, config.output, provenance)


def _state_hash(value: object) -> str:
    digest = hashlib.sha256()

    def visit(item: object) -> None:
        if isinstance(item, Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(str((item.dtype, item.shape)).encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            for key, child in item.items():
                visit(key)
                visit(child)
        elif isinstance(item, tuple | list):
            digest.update(str(len(item)).encode())
            for child in item:
                visit(child)
        elif item is None or isinstance(item, str | int | float | bool):
            digest.update(json.dumps(item).encode() + b"\x00")
        else:
            raise TypeError(f"unsupported resume diagnostic value: {type(item)}")

    visit(value)
    return digest.hexdigest()


def _batch_hash(batch: PreparedUpdate) -> str:
    return _state_hash(
        {
            "decisions": {
                field.name: getattr(batch.decisions, field.name) for field in dataclasses.fields(batch.decisions)
            },
            "selected": batch.selected,
            "steps": batch.steps,
            "values": batch.values,
            "advantages": batch.advantages,
            "targets": batch.targets,
            "old": batch.old.log_probs,
            "reference": None if batch.reference is None else batch.reference.log_probs,
            "observations": tuple(
                (
                    record.state.stream,
                    record.state.clock,
                    record.state.window,
                    dataclasses.asdict(record.state.episode),
                    record.state.opponent_queue,
                    tuple(
                        (dict(frame.columns), frame.applied_actions, frame.rewards, frame.terminal, frame.timeout)
                        for frame in record.state.frames
                    ),
                )
                for record in batch.rollout.records
            ),
        }
    )


def verify_checkpoint_resume(trainer: Trainer, output: Path, provenance: dict[str, str]) -> None:
    """Run the next native batch and update twice from one published boundary."""
    manifest = save_checkpoint(trainer, output / "checkpoints", provenance)
    boundary_optimizer = _state_hash(
        (trainer.actor_optimizer.state_dict(), trainer.critic_optimizer.state_dict(), trainer.scheduler.state_dict())
    )
    started = time.perf_counter()
    expected, expected_metrics = trainer.run_update()
    expected_batch = _batch_hash(expected)
    expected_model = {name: value.detach().cpu().clone() for name, value in trainer.policy.model.state_dict().items()}
    expected_optimizer = _state_hash(
        (trainer.actor_optimizer.state_dict(), trainer.critic_optimizer.state_dict(), trainer.scheduler.state_dict())
    )
    del expected
    restore_checkpoint(trainer, manifest, provenance)
    restored_optimizer = _state_hash(
        (trainer.actor_optimizer.state_dict(), trainer.critic_optimizer.state_dict(), trainer.scheduler.state_dict())
    )
    if restored_optimizer != boundary_optimizer:
        raise ValueError("checkpoint did not restore the exact optimizer and scheduler state")
    actual, actual_metrics = trainer.run_update()
    actual_batch = _batch_hash(actual)
    actual_optimizer = _state_hash(
        (trainer.actor_optimizer.state_dict(), trainer.critic_optimizer.state_dict(), trainer.scheduler.state_dict())
    )
    maximum = max(
        float((expected_model[name].float() - value.detach().cpu().float()).abs().max())
        for name, value in trainer.policy.model.state_dict().items()
        if value.numel()
    )
    report = {
        "format": "hal.o65.resume-check.v1",
        "manifest": str(manifest),
        "next_batch_expected_sha256": expected_batch,
        "next_batch_actual_sha256": actual_batch,
        "optimizer_expected_sha256": expected_optimizer,
        "optimizer_actual_sha256": actual_optimizer,
        "boundary_optimizer_sha256": boundary_optimizer,
        "restored_optimizer_sha256": restored_optimizer,
        "maximum_parameter_error": maximum,
        "uninterrupted_metrics": expected_metrics,
        "resumed_metrics": actual_metrics,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "resume-check.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    if expected_batch != actual_batch or expected_optimizer != actual_optimizer or maximum != 0:
        raise ValueError(f"exact native resume failed; see {output / 'resume-check.json'}")
    restore_checkpoint(trainer, manifest, provenance)


def qualify_development(config: ProxyCli) -> None:
    raw = json.loads(config.statistics.read_text())
    if not isinstance(raw, dict) or any(set(value) != {"mean", "std", "min", "max"} for value in raw.values()):
        raise ValueError("statistics must contain finalized FeatureStats objects")
    stats = {
        name: FeatureStats(value["mean"], value["std"], value["min"], value["max"]) for name, value in raw.items()
    }
    policy = load_development_policy(config.checkpoint, stats, device=config.device, compiled=config.compiled)
    report = {
        "checkpoint_sha256": PROXY_SHA256,
        "schedule": dataclasses.asdict(policy.schedule),
        "parameters": sum(p.numel() for p in policy.model.parameters()),
        "actor_parameters": sum(p.numel() for p in policy.actor_parameters()),
        "critic_parameters": sum(p.numel() for p in policy.critic_parameters()),
        "statistics_sha256": sha256(config.statistics),
        "environment": _environment(),
    }
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(json.dumps(report, sort_keys=True, indent=2))


@dataclasses.dataclass(frozen=True, slots=True)
class EvaluationConfig:
    lambdas: tuple[float, ...]
    seed: int
    matches_per_cell: int
    max_rollouts: int = 64

    def __post_init__(self) -> None:
        if not self.lambdas or any(not math.isfinite(x) or x <= 0 for x in self.lambdas):
            raise ValueError("evaluation lambdas must be finite and positive")
        if min(self.matches_per_cell, self.max_rollouts) < 1:
            raise ValueError("evaluation needs positive match and rollout counts")


@torch.no_grad()
def _episode_diagnostics(
    policy: LoopPolicy, reference: LoopPolicy, records: Sequence[DecisionRecord], results: dict[int, EpisodeResult]
) -> dict[str, float]:
    for record in records:
        record.frames_remaining = results[record.state.match].duration - record.state.clock
    batch = _decision_batch(records, policy.device)
    actions = torch.stack([record.actions for record in records]).to(policy.device)
    states = [record.state for record in records]
    current = policy.evaluate(states, actions)
    control = reference.evaluate(states, actions)
    kl = _factor_sum(factor_kl(current, control), batch.executed_factors_JF)
    metrics = {
        "kl_cost": float((kl * batch.kl_coefficient_J).sum()) / len(results),
        "kl_per_frame": float(kl.sum()) / sum(result.duration for result in results.values()),
    }
    rewards = torch.zeros(len(records), 4, device=policy.device)
    by_match: dict[int, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        rewards[index, : len(record.rewards)] = torch.tensor(record.rewards, device=policy.device)
        if not results[record.state.match].timeout:
            by_match[record.state.match].append(index)
    regularized = regularized_reward(batch, rewards, factor_kl(current, control))
    targets = torch.zeros(len(records), device=policy.device)
    selected = torch.zeros(len(records), device=policy.device, dtype=torch.bool)
    for indices in by_match.values():
        tail = torch.zeros((), device=policy.device)
        for index in reversed(indices):
            tail = regularized[index] + GAMMA ** len(records[index].rewards) * tail
            targets[index] = tail
            selected[index] = True
    metrics["monte_carlo_count"] = float(selected.sum())
    if bool(selected.any()):
        values = policy.values(current.features)[selected]
        metrics["monte_carlo_rmse"] = float((values - targets[selected]).square().mean().sqrt())
        metrics["monte_carlo_explained_variance"] = explained_variance(values, targets[selected])
    return metrics


@torch.no_grad()
def evaluate_anchors(
    trainer: Trainer, shards: tuple[SimulatorShard, ...], config: EvaluationConfig
) -> dict[str, float]:
    """Fixed first-episode seeds and starts, separate from every training lane.

    Every cell uses the same logical matches at matched D and R. An incomplete
    cell is an error, rather than a selection biased toward short matches.
    The caller supplies dedicated shards and controls the evaluation interval.
    """
    if any(shard is training for shard in shards for group in trainer.engine.groups for training in group):
        raise ValueError("evaluation cannot step a training shard")
    ids = sorted(match for shard in shards for match in shard.match_ids)
    if len(ids) < config.matches_per_cell:
        raise ValueError("evaluation requires one lane per fixed initial match")
    wanted = set(ids[: config.matches_per_cell])
    metrics: dict[str, float] = {}
    for anchor_id in trainer.engine.anchor_candidates:
        for timing in trainer.engine.sampler.schedule.configurations:
            for bin_id, coefficient in enumerate(config.lambdas):
                sampler_config = dataclasses.replace(
                    trainer.engine.sampler.config, lambda_min=coefficient, lambda_max=coefficient, anchor_rate=1.0
                )
                schedule = ScheduleConfig((timing,), trainer.engine.sampler.schedule.supervised_offsets)
                evaluation_seed = int(CounterRng(config.seed).uniform("evaluation", 0, 0, 0) * 2**63)
                sampling = EpisodeSampler(sampler_config, schedule, CounterRng(evaluation_seed))
                policy = trainer.policy.frozen_copy()
                anchors = {0: trainer.engine.anchors[anchor_id].frozen_copy()}
                engine = RolloutEngine((shards,), policy, sampling, anchors)
                engine.initialize()
                results: dict[int, EpisodeResult] = {}
                records: list[DecisionRecord] = []
                for update in range(config.max_rollouts):
                    rollout = engine.collect(update)
                    records.extend(
                        record
                        for record in rollout.records
                        if record.state.match in wanted and record.state.episode.number == 0
                    )
                    for result in rollout.results:
                        if result.match in wanted and result.episode.number == 0:
                            results[result.match] = result
                    if set(results) == wanted:
                        break
                if set(results) != wanted:
                    raise ValueError("evaluation cell did not finish all fixed matches")
                scores: list[float] = []
                lengths: list[float] = []
                stocks: list[float] = []
                wins: list[float] = []
                for result in results.values():
                    student = 0 if result.episode.players[0].anchor is None else 1
                    scores.append(result.returns[student])
                    lengths.append(float(result.duration))
                    if result.stocks is None:
                        raise ValueError("gameplay evaluation requires stock columns")
                    stock = result.stocks[student] - result.stocks[1 - student]
                    initial = result.episode.players[student].stocks - result.episode.players[1 - student].stocks
                    stocks.append(float(stock - initial))
                    if not result.timeout:
                        wins.append(float(stock > 0) + 0.5 * float(stock == 0))
                suffix = f"a{anchor_id}_d{timing.delay}_r{timing.replan}_lambda{bin_id}"
                metrics[f"eval/return_{suffix}"] = float(np.mean(scores))
                metrics[f"eval/match_frames_{suffix}"] = float(np.mean(lengths))
                metrics[f"eval/net_stocks_{suffix}"] = float(np.mean(stocks))
                metrics[f"eval/timeout_fraction_{suffix}"] = 1 - len(wins) / len(results)
                if wins:
                    metrics[f"eval/win_rate_{suffix}"] = float(np.mean(wins))
                diagnostics = _episode_diagnostics(policy, trainer.reference, records, results)
                for name, value in diagnostics.items():
                    metrics[f"eval/{name}_{suffix}"] = value
    trainer.evaluation_cursor += 1
    return metrics


def main() -> None:
    import argparse

    import tyro

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("train", add_help=False)
    commands.add_parser("qualify", add_help=False)
    args, remaining = parser.parse_known_args()
    if args.command == "train":
        train_native(tyro.cli(NativeCli, args=remaining))
    else:
        qualify_development(tyro.cli(ProxyCli, args=remaining))


if __name__ == "__main__":
    main()
