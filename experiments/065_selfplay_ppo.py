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
import threading
import time
import zlib
from collections import defaultdict
from collections.abc import Iterator
from collections.abc import Sequence
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal
from typing import Protocol
from typing import Self
from typing import TypedDict
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.special import betaincinv
from torch import Tensor

from hal.data.feature_stats import FeatureStats
from hal.data.schema import MDS_PER_FRAME_DTYPES
from hal.data.schema import Rank
from hal.eval.periodic_self_play import PeriodicEvaluator
from hal.eval.self_play_checkpoint import EVALUATION_FORMAT
from hal.eval.self_play_checkpoint import PROXY_SHA256
from hal.eval.self_play_checkpoint import TRAINING_FORMAT
from hal.eval.self_play_checkpoint import TRAINING_VERSION
from hal.eval.self_play_checkpoint import evaluation_actor_path
from hal.eval.self_play_checkpoint import load_evaluation_checkpoint
from hal.eval.self_play_checkpoint import read_training_manifest
from hal.eval.self_play_checkpoint import save_evaluation_checkpoint
from hal.eval.self_play_checkpoint import sha256
from hal.inference import compile_cache
from hal.inference.column_context import PackedColumnContextBuilder
from hal.inference.cuda_graph import CapturedCall
from hal.inference.cuda_graph import prepared_shape_compilation
from hal.inference.decision_decoder import DecisionDecoder
from hal.inference.graphed_actor import GraphedActorStep
from hal.inference.kv_cache import BlockRestartCache
from hal.inference.kv_cache import forward_tokens_with_kv_cache
from hal.models.action_sequence import ActionSequenceConfig
from hal.models.attention import KVMemory
from hal.models.conditioning import ConditioningRanges
from hal.models.controller_codec import CONTROLLER_GROUP_NAMES
from hal.models.replay_actor import ActorLoss
from hal.models.replay_actor import ActorTargets
from hal.models.replay_actor import actor_loss
from hal.models.self_play import DecoderPrecision
from hal.models.self_play import SelfPlayTransformer
from hal.models.self_play import validate_decoder_precision
from hal.models.window_replay import ReplayCriticFeatures
from hal.models.window_replay import ReplayOutput
from hal.models.window_replay import ReplaySlots
from hal.models.window_replay import replay_windows
from hal.representation.player_identity import vocabulary_from_checkpoint_buffer
from hal.sim import native
from hal.sim import native_bridge
from hal.sim import native_shard
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
def dense_advantages(
    rewards_TP: Tensor,
    values_TP: Tensor,
    next_values_TP: Tensor,
    duration_TP: Tensor,
    bootstrap_TP: Tensor,
    trace_TP: Tensor,
    decided_TP: Tensor,
) -> AdvantageTargets:
    """Empty ticks preserve the next decision's trace without adding a discount."""
    _check_finite("decision rewards", rewards_TP[decided_TP])
    _check_finite("decision values", values_TP[decided_TP])
    _check_finite("decision bootstrap values", next_values_TP[decided_TP & bootstrap_TP])
    values = torch.where(decided_TP, values_TP, 0)
    next_values = torch.where(decided_TP & bootstrap_TP, next_values_TP, 0)
    rewards = torch.where(decided_TP, rewards_TP, 0)
    duration = duration_TP.to(values.dtype)
    delta = rewards + GAMMA**duration * next_values - values
    discount = (GAMMA * GAE_LAMBDA) ** duration
    advantages = torch.zeros_like(delta)
    tail = torch.zeros_like(delta[-1])
    for tick in range(len(delta) - 1, -1, -1):
        candidate = delta[tick] + discount[tick] * torch.where(trace_TP[tick], tail, 0)
        tail = torch.where(decided_TP[tick], candidate, tail)
        advantages[tick] = torch.where(decided_TP[tick], tail, 0)
    targets = advantages + values
    _check_finite("decision advantages and targets", torch.stack((advantages, targets)))
    return AdvantageTargets(advantages, targets)


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


def ppo_loss(
    decisions: DecisionBatch,
    new_log_probs_JF: Tensor,
    advantages_J: Tensor,
    selected_J: Tensor,
    statistics: AdvantageStatistics,
    *,
    clip_epsilon: float,
    reduction: DecisionReduction | None = None,
) -> PolicyLoss:
    """Joint PPO on selected decisions using detached global advantage statistics."""
    _training_mask(decisions, selected_J, statistics, reduction)
    if not math.isfinite(clip_epsilon) or not 0 < clip_epsilon < 1:
        raise ValueError("clip_epsilon must be in (0, 1)")
    _check_tensor("advantages", advantages_J, tuple(decisions.delay_J.shape), "float", decisions.delay_J.device)
    executed = decisions.executed_factors_JF & selected_J[:, None]
    new = _masked_factors(new_log_probs_JF, executed)
    old = _masked_factors(decisions.behavior_log_probs_JF.detach(), executed)
    # Subtract per factor before summing to preserve small kernel differences.
    ratio = (new - old).sum(-1).exp()
    _check_finite("joint ratio", ratio)
    advantages = torch.where(selected_J, advantages_J.detach(), 0)
    normalized = (advantages - statistics.mean.detach()) / statistics.sigma.detach()
    surrogate = torch.minimum(ratio * normalized, ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * normalized)
    return PolicyLoss(_decision_mean(-surrogate, selected_J, reduction), ratio)


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
    key: int = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.seed) is not int:
            raise ValueError("counter RNG seed must be an integer")
        digest = hashlib.blake2b(str(self.seed).encode(), digest_size=8, person=b"hal-o65-key-v3").digest()
        object.__setattr__(self, "key", int.from_bytes(digest, "little") & 0x7FFFFFFFFFFFFFFF)

    def uniform(self, domain: str, stream: int, episode: int, frame: int, offset: int = 0, factor: int = 0) -> float:
        key = json.dumps((self.seed, domain, stream, episode, frame, offset, factor), separators=(",", ":"))
        digest = hashlib.blake2b(key.encode(), digest_size=8, person=b"hal-o65-rng-v1").digest()
        return (int.from_bytes(digest, "little") >> 11) * 2.0**-53

    def tensor_uniform(self, domain: int, stream: Tensor, episode: Tensor, frame: Tensor) -> Tensor:
        """Stateless 63-bit integer counters with identical CPU and CUDA arithmetic."""
        mask = 0x7FFFFFFFFFFFFFFF
        value = torch.full_like(stream, self.key)
        for counter in (domain, stream, episode, frame):
            value = (value ^ counter) & mask
            value = ((value ^ (value >> 30)) * 6364136223846793005) & mask
            value = ((value ^ (value >> 27)) * 1442695040888963407) & mask
            value = value ^ (value >> 31)
        return (value >> 10).double() * 2.0**-53

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
    data_floor: int = 1
    tempering: float = 1.0
    masked_rate: float = 0.1
    deployment_identity: int = int(Rank.MASTER)
    deployment_rate: float = 0.0
    lambda_min: float = 0.01
    lambda_max: float = 100.0
    stocks: tuple[int, ...] = (1, 2, 3, 4)
    percents: tuple[float, ...] = (0.0,)
    anchor_rate: float = 0.15
    percentile_alpha: float = 5.0
    percentile_beta: float = 1.3
    fixed_percentile: float | None = None

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
            self.deployment_rate
            if self.deployment_identity == 0 and self.deployment_rate > 0
            else self.masked_rate + self.deployment_rate
        )
        if type(self.deployment_identity) is not int or self.deployment_identity < 0 or ordinary_rate < 0:
            raise ValueError("invalid deployment identity or identity rates")
        if not 0 < self.lambda_min <= self.lambda_max or not math.isfinite(self.lambda_max):
            raise ValueError("invalid lambda range")
        if not self.stocks or any(type(stock) is not int or not 1 <= stock <= 4 for stock in self.stocks):
            raise ValueError("stocks must be in [1, 4]")
        if not self.percents or any(not math.isfinite(x) or x < 0 for x in self.percents):
            raise ValueError("starting percents must be finite and nonnegative")
        if any(not math.isfinite(x) or x <= 0 for x in (self.percentile_alpha, self.percentile_beta)):
            raise ValueError("percentile Beta parameters must be finite and positive")
        if self.fixed_percentile is not None and (
            not math.isfinite(self.fixed_percentile) or not 0.1 <= self.fixed_percentile <= 0.999
        ):
            raise ValueError("fixed percentile must be in [0.1, 0.999]")
        for matchup in self.matchups:
            for character in matchup.characters:
                rows = [row for row in self.identities if row.character == character and row.count >= self.data_floor]
                ordinary = [
                    row for row in rows if (self.deployment_rate == 0 or row.identity != self.deployment_identity)
                ]
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
    percentile: float = 0.95


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
        characters = matchup.characters
        if rng.uniform("orientation", match_id, episode, 0) < 0.5:
            characters = characters[::-1]
        for side, character in enumerate(characters):
            stream = 2 * match_id + side
            identity_draw = rng.uniform("identity_kind", stream, episode, 0)
            masked_mass = (
                cfg.deployment_rate if cfg.deployment_identity == 0 and cfg.deployment_rate > 0 else cfg.masked_rate
            )
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
                    and (cfg.deployment_rate == 0 or row.identity != cfg.deployment_identity)
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
            percentile = cfg.fixed_percentile
            if percentile is None:
                percentile = float(
                    np.clip(
                        betaincinv(
                            cfg.percentile_alpha, cfg.percentile_beta, rng.uniform("percentile", stream, episode, 0)
                        ),
                        0.1,
                        0.999,
                    )
                )
            players.append(PlayerEpisode(identity, character, log_lambda, timing, stocks, percent, anchor, percentile))
        self.cursor += 1
        seed = int(rng.uniform("simulator_seed", match_id, episode, 0) * 2**32)
        return Episode(episode, stage, (players[0], players[1]), seed)


NATIVE_COLUMN_LAYOUT = {
    name: (index, np.dtype(MDS_PER_FRAME_DTYPES[name]).kind == "i")
    for index, name in enumerate(native.OBSERVATION_FIELDS)
}


@dataclasses.dataclass(frozen=True, slots=True)
class DeviceFrameBlock:
    """Stage write-once native transfers before copying them into group storage."""

    records: Tensor  # [128, lanes, bytes]
    actions: Tensor  # [128, lanes, 2, ACTION_DIM]


class SimulatorShard(Protocol):
    """Return borrowed whole-shard frames, stopping at the first active episode end."""

    @property
    def match_ids(self) -> tuple[int, ...]: ...
    def step(self, inputs: Tensor, active: NDArray[np.bool_]) -> tuple[ShardFrames, ...]: ...
    def reset(self, lane: int, episode: Episode) -> ShardFrames: ...
    def save(self) -> bytes: ...
    def restore(self, state: bytes) -> None: ...


@dataclasses.dataclass(frozen=True, slots=True)
class ShardTiming:
    started: float
    wall: float
    simulator: float
    packing: float
    input_wait: float
    other: float


class NativeSimulatorShard:
    """Step native matches with pinned buffers and batch their device transfers.

    Wait for the previous GPU copy before writing output again. The rollout owns
    each batch's observations after it copies them into its group storage.
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
        self._step_input = torch.empty(2, count, 2, ACTION_DIM, dtype=torch.float32, pin_memory=pinned)
        self._observed = torch.empty(count, 2, ACTION_DIM, dtype=torch.float32, pin_memory=pinned)
        self._event: torch.cuda.Event | None = None
        self._device_block: DeviceFrameBlock | None = None
        self._device_frame = 128
        self.step_seconds = 0.0
        self.reset_seconds = 0.0
        self.stepped_frames = 0
        self.copied_bytes = 0
        self.timings: list[ShardTiming] = []
        self._configs = [native.NativeMatch(25, (native.NativePlayer(1, 1), native.NativePlayer(2, 22)), 0)] * count
        self.shard = NativeShard(data_dir, count, cpu=cpu, records=self._records)

    def _before_write(self) -> None:
        if self._event is not None:
            self._event.synchronize()

    def _frames(self, frame: native.NativeFrameBatch) -> ShardFrames:
        if frame.records is None:
            raise ValueError("native rollout requires packed adapter records")
        columns = frame.records["columns"].copy()
        columns.flags.writeable = False
        observed = native_controller_readback(frame.applied_action)
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
        packed = block.records[self._device_frame]
        actions = block.actions[self._device_frame]
        packed.copy_(self._output.reshape(len(self.match_ids), -1), non_blocking=True)
        actions.copy_(self._observed, non_blocking=True)
        if self.device.type == "cuda":
            self._event = torch.cuda.Event()
            self._event.record(torch.cuda.current_stream(self.device))
            self.copied_bytes += self._output.numel() + self._observed.numel() * self._observed.element_size()
        self._device_frame += 1
        return ShardFrames(
            packed, actions, columns, frame.reward.copy(), frame.terminated.copy(), frame.truncated.copy(), observed
        )

    def step(self, inputs: Tensor, active: NDArray[np.bool_]) -> tuple[ShardFrames, ...]:
        started = time.perf_counter()
        simulator = packing = 0.0
        source = inputs
        if (
            source.dtype != torch.float32
            or source.ndim != 4
            or source.shape[1:] != (len(self.match_ids), 2, ACTION_DIM)
            or not len(source)
        ):
            raise ValueError("native step needs float32 [frames, lanes, 2, ACTION_DIM]")
        if active.shape != (len(self.match_ids),) or active.dtype != np.bool_:
            raise ValueError("native active mask must be bool [lanes]")
        if self._step_input.shape[0] < len(source):
            self._step_input = torch.empty(source.shape, dtype=torch.float32, pin_memory=self.device.type == "cuda")
        # A single device-to-pinned-host transfer supplies the whole tick.
        waiting = time.perf_counter()
        self._step_input[: len(source)].copy_(source)
        input_wait = time.perf_counter() - waiting
        rows = []
        for actions in self._step_input[: len(source)]:
            self._before_write()
            frame = self.shard.step(actions.numpy(), active).result()
            simulator += self.shard.last_step_seconds
            packing_started = time.perf_counter()
            rows.append(self._frames(frame))
            packing += time.perf_counter() - packing_started
            self.stepped_frames += int(active.sum())
            if ((frame.terminated | frame.truncated) & active).any():
                break
        elapsed = time.perf_counter() - started
        self.step_seconds += elapsed
        self.timings.append(
            ShardTiming(started, elapsed, simulator, packing, input_wait, elapsed - simulator - packing - input_wait)
        )
        return tuple(rows)

    def reset(self, lane: int, episode: Episode) -> ShardFrames:
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
        self._before_write()
        frame = self.shard.reset(self._configs, mask).result()
        result = self._frames(frame)
        self.reset_seconds += time.perf_counter() - started
        return result

    def save(self) -> bytes:
        states = self.shard.save_many(tuple(range(len(self.match_ids)))).result()
        header = json.dumps(
            {"format": "hal.o65.native_pool.v2", "matches": self.match_ids, "max_frame": self.max_frame}
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
        expected = {"format": "hal.o65.native_pool.v2", "matches": list(self.match_ids), "max_frame": self.max_frame}
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
        self._before_write()
        self.shard.restore_many(tuple(range(len(self.match_ids))), states).result()

    def close(self) -> None:
        try:
            self._before_write()
        finally:
            self.shard.close()


def _step_shard(shard: SimulatorShard, inputs: Tensor, active: NDArray[np.bool_]) -> tuple[ShardFrames, ...]:
    return shard.step(inputs, active)


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


CRITIC_FEATURE_NAMES = ("trunk", "decoder", "condition", "opponent_condition", "opponent_queue", "opponent_lengths")


@dataclasses.dataclass(frozen=True, slots=True)
class PolicyEvaluation:
    actions: Tensor  # [J, 4, 4], codec order
    log_probs: Tensor  # [J, 16], decode order
    distributions: tuple[Tensor, ...]  # [J, 4, K], decode order; button mask already applied
    features: CriticFeatures


class LoopPolicy(Protocol):
    @property
    def model(self) -> torch.nn.Module: ...
    @property
    def device(self) -> torch.device: ...
    @property
    def max_delay(self) -> int: ...
    @property
    def max_opponent_queue(self) -> int: ...
    @property
    def player_vocab_size(self) -> int: ...
    @property
    def condition_ranges(self) -> ConditioningRanges: ...
    def retain_tick_layouts(self, ring: ActionRing, layouts: tuple[TickLayout, ...]) -> None: ...
    def tick_ready(self, layout: TickLayout, ring: ActionRing, *, clock: int) -> bool: ...
    def sample_tick(
        self, states: TickInputs, rng: CounterRng, layout: TickLayout, ring: ActionRing, *, clock: int
    ) -> None: ...
    def evaluate_reference(self, batch: ReplayBatch, positions: Tensor, window_order: Tensor) -> ReplayOutput: ...
    def evaluate_values(self, batch: ReplayBatch) -> Tensor: ...
    def actor_step(
        self, batch: ReplayBatch, targets: ActorTargets, *, clip_epsilon: float
    ) -> tuple[ActorLoss, ReplayOutput]: ...
    def values(self, features: CriticFeatures) -> Tensor: ...
    def device_wire_actions(self, indices: Tensor, *, validate: bool = True) -> Tensor: ...
    def actor_parameters(self) -> tuple[torch.nn.Parameter, ...]: ...
    def critic_parameters(self) -> tuple[torch.nn.Parameter, ...]: ...
    def clear_caches(self) -> None: ...
    def frozen_copy(self) -> LoopPolicy: ...


def _take_evaluation(evaluation: PolicyEvaluation, indices: Tensor) -> PolicyEvaluation:
    return PolicyEvaluation(
        evaluation.actions[indices],
        evaluation.log_probs[indices],
        tuple(value[indices] for value in evaluation.distributions),
        _take_features(evaluation.features, indices),
    )


@dataclasses.dataclass(frozen=True, slots=True)
class TickLayout:
    streams: tuple[int, ...]
    lanes: tuple[int, ...]
    buffer: int = 0

    def __post_init__(self) -> None:
        if type(self.buffer) is not int or self.buffer not in (0, 1):
            raise ValueError("tick buffer must be zero or one")
        if not self.streams or len(self.streams) != len(self.lanes):
            raise ValueError("tick layout lanes must match nonempty streams")
        if len(set(self.streams)) != len(self.streams):
            raise ValueError("tick layout streams must be distinct")
        if any(type(value) is not int or value < 0 for value in (*self.streams, *self.lanes)):
            raise ValueError("tick layout coordinates must be nonnegative integers")
        if len({(lane, stream % 2) for lane, stream in zip(self.lanes, self.streams, strict=True)}) != len(
            self.streams
        ):
            raise ValueError("tick layout streams must own distinct queues")


@dataclasses.dataclass(frozen=True, slots=True)
class RolloutBuffer:
    index: int
    shards: tuple[SimulatorShard, ...]
    lanes: NDArray[np.int64]
    stream: torch.cuda.Stream | None


@dataclasses.dataclass(frozen=True, slots=True)
class BufferTiming:
    buffer: int
    started: float
    input_copy_submit: float
    input_event_wait: float
    output_reuse_wait: float
    worker_wait: float
    output_copy_submit: float


class BufferStaging:
    def __init__(self, lanes: int, device: torch.device) -> None:
        pinned = device.type == "cuda"
        self.inputs = torch.empty(2, lanes, 2, ACTION_DIM, dtype=torch.float32, pin_memory=pinned)
        self.output = torch.zeros(2, lanes, native_shard.STAGING_DTYPE.itemsize, dtype=torch.uint8, pin_memory=pinned)
        self.host_inputs = self.inputs.numpy()
        self.host_output = self.output.numpy().view(native_shard.STAGING_DTYPE).reshape(2, lanes)
        self.device_output = torch.empty(self.output.shape, dtype=torch.uint8, device=device)
        self.ready: Future[None] = Future()
        self.ready.set_result(None)
        self.input_copied = torch.cuda.Event() if pinned else None
        self.output_copied = torch.cuda.Event() if pinned else None
        self.output_pending = False
        self.started = 0.0
        self.input_copy_submit = 0.0
        self.input_event_wait = 0.0
        self.output_reuse_wait = 0.0

    def release_inputs(self) -> None:
        if self.ready.done():
            return
        started = time.perf_counter()
        if self.input_copied is not None:
            self.input_copied.synchronize()
        self.ready.set_result(None)
        self.input_event_wait = time.perf_counter() - started


type PendingStep = tuple[
    RolloutBuffer, SimulatorShard, Future[native_shard.PackedStep] | Future[tuple[ShardFrames, ...]]
]


@dataclasses.dataclass(frozen=True, slots=True)
class ShardFrames:
    """Borrowed native columns for every match in a shard, on host and device."""

    records: Tensor
    actions: Tensor
    columns: NDArray[np.uint32]
    rewards: NDArray[np.float32]
    terminal: NDArray[np.bool_]
    timeout: NDArray[np.bool_]
    host_actions: NDArray[np.float32]

    def __post_init__(self) -> None:
        count = len(self.rewards)
        if (
            self.records.shape != (count, native.FRAME_DTYPE.itemsize)
            or self.records.dtype != torch.uint8
            or self.actions.shape != (count, 2, ACTION_DIM)
            or self.actions.dtype != torch.float32
            or self.host_actions.shape != (count, 2, ACTION_DIM)
            or self.host_actions.dtype != np.float32
            or self.columns.shape != (count, len(native.OBSERVATION_FIELDS))
            or self.columns.dtype != np.uint32
            or self.columns.flags.writeable
            or self.rewards.shape != (count, 2)
            or self.rewards.dtype != np.float32
            or self.terminal.shape != (count,)
            or self.timeout.shape != (count,)
            or self.terminal.dtype != np.bool_
            or self.timeout.dtype != np.bool_
        ):
            raise ValueError("native shard frame schema mismatch")
        if not np.isfinite(self.host_actions).all() or not np.isfinite(self.rewards).all():
            raise ValueError("simulator actions and rewards must be finite")
        if np.any(self.terminal & self.timeout):
            raise ValueError("a match end cannot also be a time-out")

    def __len__(self) -> int:
        return len(self.rewards)


class FrameStore:
    def __init__(self, capacity: int, matches: int, device: torch.device) -> None:
        self.records = torch.zeros(capacity, matches, native.FRAME_DTYPE.itemsize, dtype=torch.uint8, device=device)
        self.actions = torch.zeros(capacity, matches, 2, ACTION_DIM, device=device)
        self.rewards = torch.zeros(capacity, matches, 2, device=device)
        self.flags = torch.zeros(capacity, matches, 2, device=device, dtype=torch.bool)
        self.lengths = torch.zeros(matches, device=device, dtype=torch.long)

    def write(self, positions: Tensor, lanes: Tensor, batch: ShardFrames, source: Tensor) -> Tensor:
        selected = batch.records.index_select(0, source)
        self.records[positions, lanes] = selected
        self.actions[positions, lanes] = batch.actions.index_select(0, source)
        fields = native.FRAME_DTYPE.fields
        if fields is None:
            raise RuntimeError("native records must have a structured dtype")
        reward_offset = fields["reward"][1]
        flag_offset = fields["terminated"][1]
        rewards = selected[:, reward_offset : reward_offset + 8].contiguous().view(torch.float32)
        self.rewards[positions, lanes] = rewards
        self.flags[positions, lanes] = selected[:, flag_offset : flag_offset + 2].bool()
        return rewards


@dataclasses.dataclass(frozen=True, slots=True)
class TickInputs:
    table: DenseRollout
    ids: Tensor

    def __len__(self) -> int:
        return self.ids.numel()

    def tensor(self, value: Tensor) -> Tensor:
        return value.index_select(0, self.ids)

    def packed(self, length: int, tail: int | None = None) -> tuple[Tensor, ...]:
        meta = self.tensor(self.table.integers)
        lengths = meta[:, LENGTH] if tail is None else meta[:, LENGTH].clamp_max(tail)
        starts = meta[:, START] + meta[:, LENGTH] - lengths
        valid = torch.arange(length, device=meta.device)[None] < lengths[:, None]
        positions = starts[:, None] + torch.arange(length, device=meta.device)[None]
        positions = torch.where(valid, positions, 0)
        flat = (positions * self.table.store.records.shape[1] + meta[:, LANE, None]).flatten()
        records = self.table.store.records.flatten(0, 1).index_select(0, flat).reshape(len(self), length, -1)
        actions = (
            self.table.store.actions.flatten(0, 1).index_select(0, flat).reshape(len(self), length, 2, ACTION_DIM)
        )
        return (
            torch.where(valid[..., None], records, 0),
            torch.where(valid[..., None, None], actions, 0),
            meta[:, SIDE],
            meta[:, IDENTITY],
            lengths,
        )


@dataclasses.dataclass(slots=True)
class MatchState:
    episode: Episode
    episode_start: int
    end_clock: int | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class Rollout:
    """Dense group storage stays valid until that group is collected again."""

    table: DenseRollout
    ids: Tensor
    group: int
    update: int
    frames: int
    resets: int
    timeouts: int
    elapsed: float
    results: tuple[EpisodeResult, ...]
    idle_frames: int = 0


class DenseRollout:
    """Flatten [tick,player] only for indexed writes; reserve final rows for bootstraps."""

    def __init__(self, ticks: int, store: FrameStore) -> None:
        if type(ticks) is not int or ticks < 64 or ticks % 64:
            raise ValueError("dense rollout ticks must contain whole 128-frame blocks")
        self.ticks = ticks
        self.players = store.records.shape[1] * 2
        capacity = (ticks + ticks // 64 + 2) * self.players
        self.store = store
        self.device = store.records.device
        self.integers = torch.zeros(capacity, 18, dtype=torch.long, device=self.device)
        self.conditions = torch.zeros(capacity, 4, device=self.device)
        self.coefficients = torch.zeros(capacity, device=self.device)
        self.age = torch.zeros(capacity, dtype=torch.long, device=self.device)
        self.prefix = torch.zeros(capacity, 24, 4, dtype=torch.long, device=self.device)
        self.opponent = torch.zeros_like(self.prefix)
        self.actions = torch.zeros(capacity, 4, 4, dtype=torch.long, device=self.device)
        self.behavior = torch.zeros(capacity, 16, device=self.device)
        self.rewards = torch.zeros(capacity, 4, device=self.device)
        self.duration = torch.zeros(capacity, dtype=torch.long, device=self.device)
        self.remaining = torch.full((capacity,), -1, dtype=torch.long, device=self.device)
        self.terminal = torch.zeros(capacity, dtype=torch.bool, device=self.device)
        self.boundary = torch.zeros(capacity, dtype=torch.bool, device=self.device)
        self.following = torch.full((capacity,), -1, dtype=torch.long, device=self.device)
        self.values = torch.zeros(capacity, device=self.device)
        self.active = torch.zeros(capacity, device=self.device, dtype=torch.bool)
        self.valid = torch.zeros(capacity, device=self.device, dtype=torch.bool)
        self.bootstrap = torch.zeros(capacity, device=self.device, dtype=torch.bool)
        self.queue_error = torch.zeros((), device=self.device, dtype=torch.bool)
        self.evaluation: PolicyEvaluation | None = None

    def allocate_evaluation(self, model: SelfPlayTransformer) -> None:
        capacity = len(self.values)
        self.evaluation = PolicyEvaluation(
            self.actions,
            self.behavior,
            tuple(
                torch.zeros(capacity, 4, model.codec.group_vocabs[group], device=self.device)
                for group in DECODE_GROUP_INDICES
            ),
            CriticFeatures(
                torch.zeros(capacity, model.cfg.d_model, device=self.device),
                torch.zeros(capacity, model.cfg.temporal_d_model, device=self.device),
                torch.zeros(capacity, 4, device=self.device),
                torch.zeros(capacity, 4, device=self.device),
                torch.zeros(capacity, model.critic_residual.max_queue, 4, dtype=torch.long, device=self.device),
                torch.zeros(capacity, dtype=torch.long, device=self.device),
            ),
        )

    def reset(self) -> None:
        self.valid.zero_()
        self.bootstrap.zero_()
        self.active.zero_()
        self.duration.zero_()
        self.rewards.zero_()
        self.remaining.fill_(-1)
        self.terminal.zero_()
        self.boundary.zero_()
        self.following.fill_(-1)
        self.queue_error.zero_()

    def record_evaluation(self, ids: Tensor, output: PolicyEvaluation, values: Tensor) -> None:
        if self.evaluation is None:

            def storage(value: Tensor) -> Tensor:
                return value.new_zeros((len(self.values), *value.shape[1:]))

            self.evaluation = PolicyEvaluation(
                self.actions,
                self.behavior,
                tuple(storage(p) for p in output.distributions),
                CriticFeatures(*(storage(getattr(output.features, name)) for name in CRITIC_FEATURE_NAMES)),
            )
        self.actions[ids] = output.actions
        self.behavior[ids] = output.log_probs
        self.values[ids] = values
        for target, source in zip(self.evaluation.distributions, output.distributions, strict=True):
            target[ids] = source
        for name in CRITIC_FEATURE_NAMES:
            target = getattr(self.evaluation.features, name)
            target[ids] = getattr(output.features, name).to(target.dtype)


class TransformerPolicy:
    """Append fixed rollout slots and replay one history per training window."""

    def __init__(
        self,
        model: SelfPlayTransformer,
        schedule: ScheduleConfig,
        statistics: dict[str, FeatureStats],
    ) -> None:
        self.profile: UpdateProfile | None = None
        self.model = model.eval()
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
        self.packed_builder = PackedColumnContextBuilder(statistics, native.OBSERVATION_FIELDS, self.device)
        self.decode_order = torch.tensor(DECODE_GROUP_INDICES, device=self.device)
        self.tick_decoder = DecisionDecoder(model, configurations, compiled=False)
        self.tick_caches: dict[tuple[ActionRing, tuple[int, ...]], BlockRestartCache] = {}
        self.tick_graphs: dict[tuple[tuple[ActionRing, tuple[int, ...]], int, int], CapturedCall[None]] = {}
        self.current_layouts: dict[tuple[ActionRing, int], TickLayout] = {}
        self.reference_graphs: dict[tuple[tuple[int, ...], ...], CapturedCall[ReplayOutput]] = {}
        self.reference_forward_model: SelfPlayTransformer | None = None
        self.value_graphs: dict[tuple[tuple[int, ...], ...], CapturedCall[Tensor]] = {}
        self.queue_replay = (
            torch.compile(
                self.model.temporal.replay_window_queue_state,
                fullgraph=True,
                dynamic=False,
                options={"triton.cudagraphs": False, "emulate_precision_casts": True},
            )
            if self.device.type == "cuda"
            else self.model.temporal.replay_window_queue_state
        )
        self.actor_graph: GraphedActorStep | None = None
        self.actor_clip_epsilon: float | None = None
        self.window_replay = (
            torch.compile(
                replay_windows,
                fullgraph=True,
                dynamic=False,
                options={"triton.cudagraphs": False, "emulate_precision_casts": True},
            )
            if self.device.type == "cuda"
            else replay_windows
        )
        self.group_sizes = torch.tensor(model.codec.group_vocabs, device=self.device)
        self.device_wire_tables: tuple[Tensor, ...] = ()
        self._prepare_wire_tables()

    @torch.no_grad()
    def _prepare_wire_tables(self) -> None:
        tables = tuple(
            self.model.codec.semantic_values(name, torch.arange(size, device=self.device)).float()
            for name, size in zip(CONTROLLER_GROUP_NAMES, self.model.codec.group_vocabs, strict=True)
        )
        if self.device_wire_tables:
            for target, source in zip(self.device_wire_tables, tables, strict=True):
                target.copy_(source)
        else:
            self.device_wire_tables = tables

    @property
    def player_vocab_size(self) -> int:
        return self.model.cfg.player_vocab_size

    @property
    def max_delay(self) -> int:
        return self.tick_decoder.max_delay

    @property
    def max_opponent_queue(self) -> int:
        return self.model.critic_residual.max_queue

    @property
    def condition_ranges(self) -> ConditioningRanges:
        return self.model.ranges

    def actor_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        return self.model.actor_parameters()

    def critic_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        return self.model.critic_parameters()

    def clear_caches(self) -> None:
        self.tick_caches.clear()
        self.tick_graphs.clear()
        self.current_layouts.clear()
        self.value_graphs.clear()
        self.reference_graphs.clear()
        self.reference_forward_model = None
        self.actor_graph = None

    def frozen_copy(self) -> LoopPolicy:
        model = copy.deepcopy(self.model).requires_grad_(False)
        model.zero_grad(set_to_none=True)
        return TransformerPolicy(model, self.schedule, self.statistics)

    def _validate_actions(self, actions: Tensor) -> None:
        if actions.dtype not in (torch.int32, torch.int64) or actions.shape[-1] != 4:
            raise ValueError("actions must contain integer controller factors")
        sizes = self.group_sizes
        if not bool(((actions >= 0) & (actions < sizes)).all()):
            raise ValueError("actions contain an invalid controller factor")
        valid_buttons = self.model.codec.button_valid_for_trigger
        if bool((~valid_buttons[actions[..., 3], actions[..., 0]]).any()):
            raise ValueError("actions contain buttons incompatible with their triggers")

    def _packed_context(self, inputs: tuple[Tensor, ...]) -> tuple[dict[str, Tensor], Tensor]:
        records, actions, sides, identities, lengths = inputs
        features = self.packed_builder(records, actions, sides, identities, lengths)
        length = records.shape[1]
        ego = actions.gather(2, sides[:, None, None, None].expand(-1, length, 1, ACTION_DIM)).squeeze(2)
        valid = torch.arange(length, device=self.device)[None] < lengths[:, None]
        return features, torch.where(valid[..., None], self.model.codec.quantize(ego), 0)

    def _decode_tensors(
        self,
        integers: Tensor,
        conditions: Tensor,
        prefix: Tensor,
        opponent: Tensor,
        hidden: Tensor,
        observed: Tensor,
        uniforms: Tensor,
        actions: Tensor | None,
        *,
        history: KVMemory,
        count: int,
        decoder: DecisionDecoder,
    ) -> PolicyEvaluation:
        delay = integers[:, DELAY].contiguous()
        replan = integers[:, REPLAN].contiguous()
        log_lambda = conditions[:, 0].contiguous()
        queue = prefix[:, : decoder.max_delay].contiguous()
        output = decoder(
            hidden,
            observed,
            queue,
            log_lambda,
            delay,
            replan,
            uniforms[:count],
            sampled_offsets=count,
            percentile=conditions[:, 2].contiguous(),
            history=history,
            sampled_actions=None if actions is None else actions[:, :count],
            validate=False,
        )
        condition = self.model.ranges.normalize(
            log_lambda, delay, replan, percentile=conditions[:, 2].contiguous(), validate=False
        )
        opponent_condition = self.model.ranges.normalize(
            conditions[:, 1].contiguous(),
            integers[:, OPP_DELAY].contiguous(),
            integers[:, OPP_REPLAN].contiguous(),
            percentile=conditions[:, 3].contiguous(),
            validate=False,
        )
        capacity = self.model.critic_residual.max_queue
        features = CriticFeatures(
            hidden[:, -1].detach(),
            output.decoder_state.detach(),
            condition.detach(),
            opponent_condition.detach(),
            opponent[:, :capacity].contiguous(),
            integers[:, OPP_LENGTH].contiguous(),
        )
        distributions = []
        for group in DECODE_GROUP_INDICES:
            probabilities = output.distributions[group]
            padding = probabilities.new_zeros(hidden.shape[0], 4 - count, probabilities.shape[-1])
            padding[..., 0] = 1
            distributions.append(torch.cat((probabilities, padding), dim=1))
        return PolicyEvaluation(
            torch.nn.functional.pad(output.actions, (0, 0, 0, 4 - count)),
            torch.nn.functional.pad(
                output.log_probs.index_select(-1, self.decode_order), (0, 0, 0, 4 - count)
            ).flatten(1),
            tuple(distributions),
            features,
        )

    def retain_tick_layouts(self, ring: ActionRing, layouts: tuple[TickLayout, ...]) -> None:
        current = {(ring, (layout.buffer, len(layout.streams))) for layout in layouts}
        self.tick_caches = {
            key: cache for key, cache in self.tick_caches.items() if key[0] is not ring or key in current
        }
        self.tick_graphs = {
            key: graph for key, graph in self.tick_graphs.items() if key[0][0] is not ring or key[0] in current
        }
        buffers = {layout.buffer for layout in layouts}
        self.current_layouts = {
            key: layout for key, layout in self.current_layouts.items() if key[0] is not ring or key[1] in buffers
        }

    def tick_ready(self, layout: TickLayout, ring: ActionRing, *, clock: int) -> bool:
        if self.device.type != "cuda":
            return True
        key = ring, (layout.buffer, len(layout.streams))
        graph_key = key, (1 if clock % 128 == 0 else 2), (4 if clock % 4 == 0 else 2)
        return graph_key in self.tick_graphs

    @torch.no_grad()
    def sample_tick(
        self, states: TickInputs, rng: CounterRng, layout: TickLayout, ring: ActionRing, *, clock: int = 0
    ) -> None:
        if len(layout.streams) != len(states):
            raise ValueError("tick layout contains an unknown stream or row count")
        key = ring, (layout.buffer, len(states))
        layout_key = ring, layout.buffer
        previous_layout = self.current_layouts.get(layout_key)
        if previous_layout is not layout:
            if previous_layout is not None and clock % 128 and previous_layout.streams != layout.streams:
                raise ValueError("tick layout may change only at a block boundary")
            expected_streams = torch.tensor(layout.streams, device=self.device)
            expected_lanes = torch.tensor(layout.lanes, device=self.device)
            declared = states.tensor(states.table.integers)
            if bool((declared[:, STREAM] != expected_streams).any()):
                raise ValueError("rollout tick contains an unknown stream")
            if bool((declared[:, LANE] != expected_lanes).any()):
                raise ValueError("rollout tick metadata disagrees with its lane layout")
            self.current_layouts[layout_key] = layout
        cache = self.tick_caches.get(key)
        if cache is None:
            cache = BlockRestartCache(
                self.model,
                len(states),
                self.device,
                dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            )
            self.tick_caches[key] = cache
        meta = states.tensor(states.table.integers)
        restart = clock % 128 == 0
        enabled = self.device.type == "cuda"
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=enabled, cache_enabled=False):
            if restart:
                features, observed = self._packed_context(states.packed(129))
                tokens = self.model.context_tokens(features, observed)
                lengths = meta[:, LENGTH] - 1
                positions = torch.arange(128, device=self.device)[None] - (128 - lengths[:, None])
                previous = tokens.gather(1, positions.clamp_min(0)[..., None].expand(-1, -1, tokens.shape[-1]))
                previous = torch.where((positions >= 0)[..., None], previous, 0)
                cache.restart(previous, lengths)
            appended = 1 if restart else 2
            inputs = (
                *states.packed(appended, tail=appended),
                meta,
                states.tensor(states.table.conditions),
                states.tensor(states.table.prefix),
                states.tensor(states.table.opponent),
                states.tensor(states.table.active),
                states.ids,
            )
            count = 4 if clock % 4 == 0 else 2
            graph_key = key, appended, count
            captured = self.tick_graphs.get(graph_key)
            if captured is None:
                static = tuple(value.clone() for value in inputs)

                def tick() -> None:
                    with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=enabled, cache_enabled=False):
                        context, observed = self._packed_context(static[:5])
                        tokens = self.model.context_tokens(context, observed)
                        hidden = forward_tokens_with_kv_cache(self.model, tokens, cache.cache)
                        cache.frames_in_block.add_(appended)
                        result = self._decode_tensors(
                            static[5],
                            static[6],
                            static[7],
                            static[8],
                            hidden[:, -1:],
                            observed[:, -1],
                            self._tensor_action_uniforms(static[5], rng),
                            None,
                            history=cache.memory(),
                            count=count,
                            decoder=self.tick_decoder,
                        )
                        rows = static[5]
                        due = static[9] & (rows[:, CLOCK] % rows[:, REPLAN] == 0)
                        ring.commit(
                            rows,
                            due,
                            result.actions,
                            self.device_wire_actions(result.actions, validate=False),
                            count=count,
                        )
                        values = self.values(result.features, validate=False)
                        states.table.record_evaluation(static[10], result, values)

                if enabled:
                    operation = torch.compile(
                        tick, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False}
                    )
                    with prepared_shape_compilation(64):
                        captured = CapturedCall(
                            operation,
                            static,
                            (
                                *cache.cache.buffers(),
                                cache.frames_in_block,
                                ring.values,
                                ring.controllers,
                                ring.length_device,
                            ),
                            device=self.device,
                        )
                    self.tick_graphs[graph_key] = captured
                else:
                    # CPU tests execute the same fixed-slot operation without capture.
                    captured = None
            if self.profile is not None:
                self.profile.work["rollout/trunk_tokens"] += len(states) * (appended + (128 if restart else 0))
                self.profile.work["rollout/decoder_tokens"] += len(states) * (self.tick_decoder.max_delay + count)
            if captured is None:
                tick()
            else:
                with (
                    torch.cuda.nvtx.range(
                        f"hal.o65.TransformerPolicy.sample_tick/B{len(states)}/append{appended}/R{count}"
                    )
                    if self.profile is not None
                    else contextlib.nullcontext()
                ):
                    captured(inputs)

    def _tensor_action_uniforms(self, meta: Tensor, rng: CounterRng) -> Tensor:
        offsets = meta[:, DELAY][None, None] + torch.arange(1, 5, device=self.device)[:, None, None]
        factors = torch.tensor(CODEC_FACTOR_INDICES, device=self.device)[None, :, None]
        return rng.tensor_uniform(
            3,
            meta[:, STREAM][None, None].expand(4, 4, -1),
            meta[:, EPISODE][None, None],
            meta[:, CLOCK][None, None] * 128 + offsets * 4 + factors,
        )

    def window_forward(self, inputs: tuple[Tensor, ...], *, grad: bool) -> ReplayOutput:
        with (
            torch.set_grad_enabled(grad),
            torch.autocast(
                self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda", cache_enabled=False
            ),
        ):
            context, observed = self._packed_context(inputs[:5])
            hidden = self.model(context, torch.zeros(len(observed), dtype=torch.long, device=self.device), observed)
        # Torch 2.11 whole-trunk compilation failed the proxy precision gate.
        # Capture the eager trunk together with the compiled shared-history decoder.
        return self.window_replay(self.model, hidden, observed, ReplaySlots(*inputs[5:]), grad=grad)

    def reference_forward(self, inputs: tuple[Tensor, ...], windows_per_step: int, count: int) -> ReplayOutput:
        model = self.reference_forward_model
        if model is None:
            raise ValueError("reference forward model is not prepared")
        batches = []
        for start in range(0, len(inputs[0]), windows_per_step):
            end = start + windows_per_step
            with (
                torch.enable_grad(),
                torch.autocast(
                    self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda", cache_enabled=False
                ),
            ):
                context, observed = self._packed_context(tuple(value[start:end] for value in inputs[:5]))
                hidden = model(context, torch.zeros(len(observed), dtype=torch.long, device=self.device), observed)
            parts = []
            for slot in range(0, inputs[5].shape[1], count):
                fields = tuple(
                    value[start:end].contiguous()
                    if name == "history_starts_WL"
                    else value[start:end, slot : slot + count].contiguous()
                    for name, value in zip(REPLAY_SLOT_NAMES, inputs[5:], strict=True)
                )
                with torch.enable_grad():
                    output = self.window_replay(model, hidden, observed, ReplaySlots(*fields), grad=True)
                parts.append(
                    ReplayOutput(
                        output.log_probs_WKF.detach(),
                        tuple(p.detach() for p in output.distributions),
                        output.factor_mask_WKF,
                        output.features,
                    )
                )
            batches.append(concatenate_replays(parts, 1))
        return concatenate_replays(batches, 0)

    @torch.no_grad()
    def evaluate_reference(self, batch: ReplayBatch, positions: Tensor, window_order: Tensor) -> ReplayOutput:
        if torch.is_inference_mode_enabled():
            raise ValueError("reference replay requires autograd metadata outside inference_mode")
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise ValueError("reference parameters must stay frozen")
        windows, slots = batch.rows.shape
        count = positions.shape[1] if positions.ndim == 2 else 0
        windows_per_step = window_order.shape[1] if window_order.ndim == 2 else 0
        if (
            windows_per_step < 1
            or window_order.dtype != torch.long
            or window_order.device != self.device
            or positions.shape != (windows, count)
            or not 1 <= count <= slots
            or positions.dtype != torch.long
            or positions.device != self.device
        ):
            raise ValueError("invalid reference actor shape")
        ordered_positions = positions.sort(-1).values
        if bool(((positions < 0) | (positions >= slots)).any()) or bool(
            (ordered_positions[:, 1:] == ordered_positions[:, :-1]).any()
        ):
            raise ValueError("reference positions must be distinct slots inside each window")
        order = window_order.flatten()
        live = order >= 0
        used = order[live]
        if bool(((order < -1) | (order >= windows)).any()) or len(used.unique()) != len(used):
            raise ValueError("reference window order must contain distinct windows or -1 padding")
        seen = torch.zeros(windows, dtype=torch.bool, device=self.device)
        seen[used] = True
        rest = torch.nonzero(~seen).flatten()
        tail = torch.nn.functional.pad(rest, (0, (-len(rest)) % windows_per_step), value=-1)
        order = torch.cat((order, tail))
        live = order >= 0
        window_inverse = torch.empty(windows, device=self.device, dtype=torch.long)
        window_inverse[order[live]] = torch.arange(len(order), device=self.device)[live]
        priority = (torch.arange(slots, device=self.device)[None] + count).expand(windows, -1).clone()
        priority.scatter_(1, positions, torch.arange(count, device=self.device)[None].expand_as(positions))
        permutation = priority.argsort(-1)
        inverse = permutation.argsort(-1)
        padded_slots = math.ceil(slots / count) * count
        permutation = torch.nn.functional.pad(permutation, (0, padded_slots - slots))
        padded_windows = len(order)
        packed = batch.take(order, permutation, live)
        packed = dataclasses.replace(
            packed,
            slots=dataclasses.replace(
                packed.slots,
                valid_WK=packed.slots.valid_WK & (torch.arange(padded_slots, device=self.device) < slots),
            ),
        )
        inputs = packed.inputs()
        if self.reference_forward_model is None:
            # Torch 2.11.0 BF16 forwards with frozen parameter metadata fail
            # the proxy KL gate. A read-only copy matches the actor's autograd
            # arithmetic. It has no optimizer or backward; the stored reference
            # stays frozen and every score is detached. Remove this boundary
            # when frozen forwards pass the complete gate.
            self.reference_forward_model = copy.deepcopy(self.model).requires_grad_(True)
            self.reference_forward_model.percentile_head.requires_grad_(False)
        if self.device.type == "cpu":
            output = self.reference_forward(inputs, windows_per_step, count)
        else:
            key = ((windows_per_step, count), *(tuple(value.shape) for value in inputs))
            captured = self.reference_graphs.get(key)
            if captured is None:
                static = tuple(value.clone() for value in inputs)
                with prepared_shape_compilation(64):
                    captured = CapturedCall(
                        lambda: self.reference_forward(static, windows_per_step, count), static, device=self.device
                    )
                self.reference_graphs[key] = captured
            output = captured(inputs)
        if self.profile is not None:
            phase_name = self.profile.phase_name
            self.profile.work[f"{phase_name}/trunk_tokens"] += padded_windows * 256
            self.profile.work[f"{phase_name}/decoder_tokens"] += padded_windows * padded_slots * (self.max_delay + 4)

        def reorder(value: Tensor) -> Tensor:
            index = inverse.reshape(windows, slots, *([1] * (value.ndim - 2)))
            return value[window_inverse].gather(1, index.expand(windows, slots, *value.shape[2:]))

        return ReplayOutput(
            reorder(output.log_probs_WKF),
            tuple(reorder(p) for p in output.distributions),
            reorder(output.factor_mask_WKF),
            ReplayCriticFeatures(
                *(reorder(getattr(output.features, field.name)) for field in dataclasses.fields(output.features))
            ),
        )

    def value_forward(self, inputs: tuple[Tensor, ...]) -> Tensor:
        slots = ReplaySlots(*inputs[5:])
        valid = slots.valid_WK
        positions = torch.where(valid, slots.positions_WK, 0)
        delay = torch.where(valid, slots.delay_WK, 0)
        condition = torch.where(valid[..., None], slots.condition_WK4, 0)
        with torch.autocast(
            self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda", cache_enabled=False
        ):
            context, observed = self._packed_context(inputs[:5])
            hidden = self.model(context, torch.zeros(len(observed), dtype=torch.long, device=self.device), observed)
        with torch.autocast(
            self.device.type, dtype=torch.bfloat16, enabled=self.model.decoder_precision == "bf16", cache_enabled=False
        ):
            state = self.queue_replay(
                hidden.float(), observed, positions, slots.history_starts_WL, slots.forced_queue_WKD4, delay, condition
            )
        rows = torch.arange(len(observed), device=self.device)[:, None]
        lengths = torch.where(valid, slots.opponent_lengths_WK, 0)
        opponent_queue = torch.where(
            (torch.arange(self.max_opponent_queue, device=self.device) < lengths[..., None])[..., None],
            slots.opponent_queue_WKQ4,
            0,
        )
        features = CriticFeatures(
            hidden[rows, positions].flatten(0, 1).detach(),
            state.flatten(0, 1).detach(),
            condition.flatten(0, 1),
            torch.where(valid[..., None], slots.opponent_condition_WK4, 0).flatten(0, 1),
            opponent_queue.flatten(0, 1),
            lengths.flatten(),
        )
        return self.values(features, validate=False).reshape(valid.shape)

    @torch.no_grad()
    def evaluate_values(self, batch: ReplayBatch) -> Tensor:
        if self.profile is not None:
            phase_name = self.profile.phase_name
            self.profile.work[f"{phase_name}/trunk_tokens"] += batch.records.shape[0] * batch.records.shape[1]
            self.profile.work[f"{phase_name}/decoder_tokens"] += batch.rows.numel() * (self.max_delay + 1)
        inputs = batch.inputs()
        if self.device.type == "cpu":
            return self.value_forward(inputs)
        key = tuple(tuple(value.shape) for value in inputs)
        captured = self.value_graphs.get(key)
        if captured is None:
            static = tuple(value.clone() for value in inputs)
            with prepared_shape_compilation(64):
                captured = CapturedCall(lambda: self.value_forward(static), static, device=self.device)
            self.value_graphs[key] = captured
        return captured(inputs)

    def actor_step(
        self, batch: ReplayBatch, targets: ActorTargets, *, clip_epsilon: float
    ) -> tuple[ActorLoss, ReplayOutput]:
        if self.profile is not None:
            phase_name = self.profile.phase_name
            self.profile.work[f"{phase_name}/trunk_tokens"] += batch.records.shape[0] * batch.records.shape[1]
            self.profile.work[f"{phase_name}/decoder_tokens"] += batch.rows.numel() * (self.max_delay + 4)
        inputs = batch.inputs()
        if self.device.type == "cpu":
            self.model.zero_grad(set_to_none=True)
            output = self.window_forward(inputs, grad=True)
            result = actor_loss(output, targets, clip_epsilon=clip_epsilon)
            result.loss.backward()
            detached = ReplayOutput(
                output.log_probs_WKF.detach(),
                tuple(value.detach() for value in output.distributions),
                output.factor_mask_WKF,
                output.features,
            )
            return result, detached
        if self.actor_graph is not None and self.actor_clip_epsilon != clip_epsilon:
            raise ValueError("actor clip epsilon differs from the captured objective")
        if self.actor_graph is None:
            with prepared_shape_compilation(64):
                self.actor_graph = GraphedActorStep(
                    lambda values: self.window_forward(values, grad=True),
                    inputs,
                    targets,
                    self.actor_parameters(),
                    clip_epsilon=clip_epsilon,
                )
            self.actor_clip_epsilon = clip_epsilon
        return self.actor_graph.backward(inputs, targets), self.actor_graph.evaluation()

    def values(self, features: CriticFeatures, *, validate: bool = True) -> Tensor:
        return self.model.self_play_value(
            features.trunk,
            features.decoder,
            features.condition,
            features.opponent_condition,
            features.opponent_queue,
            features.opponent_lengths,
            validate=validate,
        )

    def device_wire_actions(self, indices: Tensor, *, validate: bool = True) -> Tensor:
        if validate:
            if indices.device != self.device:
                raise ValueError("controller factors must use the policy device")
            self._validate_actions(indices)
        tables = self.device_wire_tables
        result = torch.empty(*indices.shape[:-1], ACTION_DIM, device=indices.device, dtype=torch.float32)
        result[..., :2] = tables[1][indices[..., 1]]
        result[..., 2:4] = tables[2][indices[..., 2]]
        result[..., 4:6] = tables[3][indices[..., 3]]
        result[..., 6:] = tables[0][indices[..., 0]]
        return result


@dataclasses.dataclass(frozen=True, slots=True)
class EpisodeResult:
    match: int
    episode: Episode
    duration: int
    returns: tuple[float, float]
    timeout: bool
    stocks: tuple[int, int] | None


class ActionRing:
    def __init__(self, matches: int, capacity: int, device: torch.device) -> None:
        self.values = torch.zeros(matches, 2, capacity, 4, dtype=torch.long, device=device)
        self.controllers = torch.zeros(matches, 2, capacity, ACTION_DIM, device=device)
        self.head = torch.zeros(matches, 2, dtype=torch.long, device=device)
        self.length_device = torch.zeros(matches, 2, dtype=torch.long, device=device)
        self.capacity = capacity
        self.offsets = torch.arange(capacity, device=device)

    def reset(self, lane: int, delays: tuple[int, int]) -> None:
        self.values[lane].zero_()
        self.head[lane].zero_()
        self.length_device[lane] = torch.tensor(delays, device=self.values.device)

    def read(self, lanes: Tensor, sides: Tensor, count: int) -> Tensor:
        positions = (self.head[lanes, sides, None] + self.offsets[:count]) % self.capacity
        return self.values[lanes[:, None], sides[:, None], positions]

    def commit(self, metadata: Tensor, due: Tensor, actions: Tensor, controllers: Tensor, *, count: int) -> None:
        lanes, sides = metadata[:, LANE], metadata[:, SIDE]
        written = min(count, self.capacity)
        offsets = self.offsets[None, :written]
        positions = (self.head[lanes, sides, None] + self.length_device[lanes, sides, None] + offsets) % self.capacity
        valid = due[:, None] & (offsets < metadata[:, REPLAN, None])
        previous = self.values[lanes[:, None], sides[:, None], positions]
        self.values[lanes[:, None], sides[:, None], positions] = torch.where(
            valid[..., None], actions[:, :written], previous
        )
        previous_controls = self.controllers[lanes[:, None], sides[:, None], positions]
        self.controllers[lanes[:, None], sides[:, None], positions] = torch.where(
            valid[..., None], controllers[:, :written], previous_controls
        )
        self.length_device[lanes, sides] += torch.where(due, metadata[:, REPLAN], 0)


# Device metadata columns for fixed player slots.
(
    STREAM,
    MATCH,
    SIDE,
    CLOCK,
    UPDATE,
    BLOCK,
    EPISODE,
    LANE,
    START,
    LENGTH,
    IDENTITY,
    CHARACTER,
    STAGE,
    DELAY,
    REPLAN,
    OPP_DELAY,
    OPP_REPLAN,
    OPP_LENGTH,
) = range(18)


@dataclasses.dataclass(frozen=True, slots=True)
class PoolMatchSnapshot:
    match: int
    episode: Episode
    episode_start: int
    history_start: int
    history_length: int
    returns: tuple[float, float]
    end_clock: int | None


@dataclasses.dataclass(frozen=True, slots=True)
class PoolGroupSnapshot:
    fields: tuple[str, ...]
    records: Tensor
    actions: Tensor
    rewards: Tensor
    flags: Tensor
    queue: Tensor
    head: Tensor
    length: Tensor
    matches: tuple[PoolMatchSnapshot, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class PoolSnapshot:
    format: str
    groups: tuple[PoolGroupSnapshot, ...]


class RolloutEngine:
    def __init__(
        self,
        groups: tuple[tuple[SimulatorShard, ...], ...],
        policy: LoopPolicy,
        sampler: EpisodeSampler,
        anchors: dict[int, LoopPolicy],
        *,
        fixed_anchor_lanes: bool = False,
    ) -> None:
        if not groups or any(not group or any(not shard.match_ids for shard in group) for group in groups):
            raise ValueError("rollout groups and shards must be nonempty")
        if type(fixed_anchor_lanes) is not bool:
            raise ValueError("fixed_anchor_lanes must be a boolean")
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
        self.profile: UpdateProfile | None = None
        self.groups = groups
        self.policy = policy
        self.sampler = sampler
        self.configurations = sampler.schedule.configurations
        self.anchors = anchors
        self.fixed_anchor_lanes = fixed_anchor_lanes
        self.tick_layouts: dict[tuple[int, int | None], TickLayout] = {}
        self.group_layouts: list[dict[tuple[int, int | None], TickLayout]] = [{} for _ in groups]
        self.anchor_candidates = tuple(sorted(anchors))
        self.clocks = [0] * len(groups)
        self.cursor = 0
        self.matches: dict[int, MatchState] = {}
        self._initialized = False
        self.completed_update = -1
        self.results: list[EpisodeResult] = []
        self.group_matches = tuple(tuple(match for shard in group for match in shard.match_ids) for group in groups)
        capacity = max(t.delay + t.replan for t in sampler.schedule.configurations)
        self.rings = tuple(ActionRing(len(matches), capacity, policy.device) for matches in self.group_matches)
        for ring in self.rings:
            ring.controllers.copy_(policy.device_wire_actions(ring.values))
        self.anchor_counts = tuple(
            math.floor(len(matches) * sampler.config.anchor_rate + 0.5) for matches in self.group_matches
        )
        self.locations = {
            match: (group, lane)
            for group, matches in enumerate(self.group_matches)
            for lane, match in enumerate(matches)
        }
        self.shard_lanes = {
            id(shard): np.array([self.locations[match][1] for match in shard.match_ids], dtype=np.int64)
            for group in groups
            for shard in group
        }
        self.device_lanes = {key: torch.from_numpy(lanes).to(policy.device) for key, lanes in self.shard_lanes.items()}
        self.buffer_device_lanes: dict[int, Tensor] = {}
        self.staging: dict[int, BufferStaging] = {}
        self.transfer_timings: list[BufferTiming] = []
        buffers: list[tuple[RolloutBuffer, ...]] = []
        for shards in groups:
            split = (len(shards) + 1) // 2
            prepared: list[RolloutBuffer] = []
            for index, half in enumerate((shards[:split], shards[split:])):
                if not half:
                    continue
                lanes = np.concatenate([self.shard_lanes[id(shard)] for shard in half])
                lanes.flags.writeable = False
                stream = torch.cuda.Stream(device=policy.device) if policy.device.type == "cuda" else None
                buffer = RolloutBuffer(index, half, lanes, stream)
                prepared.append(buffer)
                if all(isinstance(shard, NativeSimulatorShard) for shard in half):
                    self.staging[id(buffer)] = BufferStaging(len(lanes), policy.device)
                elif any(isinstance(shard, NativeSimulatorShard) for shard in half):
                    raise ValueError("a rollout buffer cannot mix native and test shards")
                self.buffer_device_lanes[id(buffer)] = torch.from_numpy(lanes.copy()).to(policy.device)
            buffers.append(tuple(prepared))
        self.buffers = tuple(buffers)
        self.group_slots: list[dict[tuple[int, int | None], Tensor]] = [{} for _ in groups]
        self.stores: list[FrameStore | None] = [None] * len(groups)
        self.dense_meta = [
            torch.zeros(2 * len(ids), 18, dtype=torch.long, device=policy.device) for ids in self.group_matches
        ]
        self.dense_logs = [torch.zeros(2 * len(ids), 4, device=policy.device) for ids in self.group_matches]
        self.stops = [torch.zeros(len(ids), dtype=torch.long, device=policy.device) for ids in self.group_matches]
        self.episode_clocks = [torch.zeros_like(value) for value in self.stops]
        self.observed_clocks = [torch.zeros_like(value) for value in self.stops]
        self.active = [torch.ones(len(ids), dtype=torch.bool, device=policy.device) for ids in self.group_matches]
        self.live_host = [np.ones(len(ids), dtype=np.bool_) for ids in self.group_matches]
        self.assignments = [np.full(2 * len(ids), -1, dtype=np.int64) for ids in self.group_matches]
        self.owners = [
            torch.full((2 * len(ids),), -1, dtype=torch.long, device=policy.device) for ids in self.group_matches
        ]
        self.layout_assignments = [np.full_like(values, -2) for values in self.assignments]
        self.tables: list[DenseRollout | None] = [None] * len(groups)
        self.episode_returns = [torch.zeros(len(ids), 2, device=policy.device) for ids in self.group_matches]

    def snapshot(self) -> PoolSnapshot:
        groups = []
        for group_id, matches in enumerate(self.group_matches):
            store = self.stores[group_id]
            if store is None:
                raise ValueError("pool has no frame storage")
            lengths = (self.observed_clocks[group_id] - self.episode_clocks[group_id] + 1).clamp_max(129)
            time = torch.arange(129, device=self.policy.device)[:, None]
            valid = time < lengths[None]
            positions = torch.where(valid, self.stops[group_id][None] - lengths[None] + time, 0)
            lanes = torch.arange(len(matches), device=self.policy.device)[None]
            packed = []
            for source in (store.records, store.actions, store.rewards, store.flags):
                mask = valid.reshape(*valid.shape, *([1] * (source.ndim - 2)))
                packed.append(torch.where(mask, source[positions, lanes], source.new_zeros(())).cpu())
            retained = lengths.cpu().tolist()
            starts = (self.observed_clocks[group_id] - lengths + 1).cpu().tolist()
            returns = self.episode_returns[group_id].cpu().tolist()
            metadata = tuple(
                PoolMatchSnapshot(
                    match,
                    self.matches[match].episode,
                    self.matches[match].episode_start,
                    starts[lane],
                    retained[lane],
                    (returns[lane][0], returns[lane][1]),
                    self.matches[match].end_clock,
                )
                for lane, match in enumerate(matches)
            )
            ring = self.rings[group_id]
            groups.append(
                PoolGroupSnapshot(
                    fields=native.OBSERVATION_FIELDS,
                    records=packed[0],
                    actions=packed[1],
                    rewards=packed[2],
                    flags=packed[3],
                    queue=ring.values.to(device="cpu", copy=True),
                    head=ring.head.to(device="cpu", copy=True),
                    length=ring.length_device.to(device="cpu", copy=True),
                    matches=metadata,
                )
            )
        return PoolSnapshot("hal.o65.rollout_pool.v4", tuple(groups))

    def restore_pool(self, snapshot: PoolSnapshot) -> None:
        if not isinstance(snapshot, PoolSnapshot) or snapshot.format != "hal.o65.rollout_pool.v4":
            raise ValueError("unsupported rollout pool snapshot identity")
        if len(snapshot.groups) != len(self.groups):
            raise ValueError("rollout pool topology mismatch")
        restored: dict[int, MatchState] = {}
        for group_id, (saved, ids) in enumerate(zip(snapshot.groups, self.group_matches, strict=True)):
            if tuple(match.match for match in saved.matches) != ids or saved.fields != native.OBSERVATION_FIELDS:
                raise ValueError("rollout pool match or schema mismatch")
            store = FrameStore(129, len(ids), self.policy.device)
            ring = self.rings[group_id]
            pairs = (
                (store.records, saved.records),
                (store.actions, saved.actions),
                (store.rewards, saved.rewards),
                (store.flags, saved.flags),
                (ring.values, saved.queue),
                (ring.head, saved.head),
                (ring.length_device, saved.length),
            )
            for target, source in pairs:
                if target.shape != source.shape or target.dtype != source.dtype:
                    raise ValueError("rollout pool tensor schema mismatch")
                target.copy_(source)
            if bool(((saved.head < 0) | (saved.head >= ring.capacity)).any()) or bool(
                ((saved.length < 0) | (saved.length > ring.capacity)).any()
            ):
                raise ValueError("invalid rollout pool queues")
            ring.controllers.copy_(self.policy.device_wire_actions(ring.values))
            for lane, match in enumerate(saved.matches):
                if not 1 <= match.history_length <= 129 or match.history_start < match.episode_start:
                    raise ValueError("invalid rollout pool history span")
                restored[match.match] = MatchState(match.episode, match.episode_start, match.end_clock)
                self._publish_episode(group_id, lane, match.match, 0, match.episode, match.episode_start)
                self.stops[group_id][lane] = match.history_length
                self.observed_clocks[group_id][lane] = match.history_start + match.history_length - 1
                self.active[group_id][lane] = match.end_clock is None
                self.live_host[group_id][lane] = match.end_clock is None
                self.episode_returns[group_id][lane] = torch.tensor(match.returns, device=self.policy.device)
            store.lengths.copy_(self.stops[group_id])
            self.stores[group_id] = store
        self.matches = restored
        self.tables = [None] * len(self.groups)
        for policy in (self.policy, *self.anchors.values()):
            policy.clear_caches()

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
        if self.fixed_anchor_lanes:
            group, lane_index = self.locations[match]
            side = lane_index % 2 if lane_index < self.anchor_counts[group] else None
            players = []
            for player_side, player in enumerate(episode.players):
                anchor = None
                if player_side == side:
                    stream = 2 * match + player_side
                    choice = self.sampler.rng.choice([1.0] * len(self.anchor_candidates), "anchor", stream, number, 0)
                    anchor = self.anchor_candidates[choice]
                players.append(dataclasses.replace(player, anchor=anchor))
            episode = dataclasses.replace(episode, players=(players[0], players[1]))
        for player in episode.players:
            if not 0 <= player.identity < self.policy.player_vocab_size:
                raise ValueError("episode identity is outside the checkpoint vocabulary")
            if player.timing not in self.configurations:
                raise ValueError("episode uses an unsupported decision configuration")
            low, high = self.policy.condition_ranges.log_lambda
            if not low <= player.log_lambda <= high:
                raise ValueError("episode conditioning is outside its declared range")
        group, at = self.locations[match]
        store = self.stores[group]
        if store is None:
            store = FrameStore(1, len(self.group_matches[group]), self.policy.device)
            self.stores[group] = store
        previous = self.matches.get(match)
        position = int(self.stops[group][at]) if previous is not None else 0
        if position >= len(store.records):
            raise ValueError("frame store capacity exhausted during reset")
        frame = shard.reset(lane, episode)
        if frame.terminal[lane] or frame.timeout[lane] or bool(frame.host_actions[lane].any()):
            raise ValueError("reset must return a live initial state with neutral applied inputs")
        store.write(
            torch.tensor([position], device=self.policy.device),
            torch.tensor([at], device=self.policy.device),
            frame,
            torch.tensor([lane], device=self.policy.device),
        )
        ring = self.rings[group]
        ring.reset(at, (episode.players[0].timing.delay, episode.players[1].timing.delay))
        ring.controllers[at].copy_(self.policy.device_wire_actions(ring.values[at]))
        self.matches[match] = MatchState(episode, clock)
        self._publish_episode(group, at, match, position, episode, clock)

    def _publish_episode(
        self, group: int, lane: int, match: int, position: int, episode: Episode, episode_start: int
    ) -> None:
        rows = []
        for side, player in enumerate(episode.players):
            if not math.isfinite(player.percentile) or not 0.1 <= player.percentile <= 0.999:
                raise ValueError("episode percentile must be in [0.1, 0.999]")
            opponent = episode.players[1 - side]
            rows.append(
                [
                    2 * match + side,
                    match,
                    side,
                    episode_start,
                    0,
                    episode_start,
                    episode.number,
                    lane,
                    position,
                    1,
                    player.identity,
                    player.character,
                    episode.stage,
                    player.timing.delay,
                    player.timing.replan,
                    opponent.timing.delay,
                    opponent.timing.replan,
                    opponent.timing.delay,
                ]
            )
            self.assignments[group][2 * lane + side] = -1 if player.anchor is None else player.anchor
        at = slice(2 * lane, 2 * lane + 2)
        self.dense_meta[group][at] = torch.tensor(rows, device=self.policy.device)
        self.owners[group][at] = torch.from_numpy(self.assignments[group][at].copy()).to(self.policy.device)
        logs = tuple(player.log_lambda for player in episode.players)
        u = tuple(player.percentile for player in episode.players)
        self.dense_logs[group][at] = torch.tensor(((*logs, *u), (*logs[::-1], *u[::-1])), device=self.policy.device)
        self.stops[group][lane] = position + 1
        self.episode_clocks[group][lane] = episode_start
        self.observed_clocks[group][lane] = episode_start
        self.active[group][lane] = True
        self.live_host[group][lane] = True
        self.episode_returns[group][lane].zero_()

    def _dense_metadata(self, clock: int, update: int, *, bootstrap: bool = False) -> Tensor:
        group = self.cursor
        meta = self.dense_meta[group].clone()
        observed = self.observed_clocks[group].repeat_interleave(2)
        clocks = observed if bootstrap else torch.full_like(observed, clock)
        block = ((clocks - int(bootstrap)) // 128) * 128
        start = torch.maximum(self.episode_clocks[group].repeat_interleave(2), block - 128)
        start = torch.maximum(start, clocks - 255)
        # Inactive rows still occupy slots. Their observations stop at the end.
        length = (observed - start + 1).clamp(1, 256)
        meta[:, CLOCK] = clocks
        meta[:, UPDATE] = update
        meta[:, BLOCK] = block
        meta[:, START] = self.stops[group].repeat_interleave(2) - length
        meta[:, LENGTH] = length
        meta[:, OPP_LENGTH] = self.rings[group].length_device.flip(1).flatten()
        return meta

    def _stage_tick(
        self,
        clock: int,
        update: int,
        table: DenseRollout,
        pending: Tensor,
    ) -> Tensor:
        group = self.cursor
        players = len(self.dense_meta[group])
        tick = (clock - self.clocks[group]) // 2
        at = torch.arange(players, device=self.policy.device) + tick * players
        meta = self._dense_metadata(clock, update)
        table.integers[at] = meta
        table.age[at] = meta[:, CLOCK] - self.episode_clocks[group].repeat_interleave(2)
        table.remaining[at] = -1
        table.conditions[at] = self.dense_logs[group]
        table.coefficients[at] = self.dense_logs[group][:, 0].exp()
        ring = self.rings[group]
        lanes, sides = meta[:, LANE], meta[:, SIDE]
        offsets = torch.arange(24, device=self.policy.device)[None]
        queue = ring.read(lanes, sides, min(24, ring.capacity))
        queue = torch.nn.functional.pad(queue, (0, 0, 0, 24 - queue.shape[1]))
        table.prefix[at] = torch.where((offsets < meta[:, DELAY, None])[..., None], queue, 0)
        opponent = queue.reshape(-1, 2, 24, 4).flip(1).flatten(0, 1)
        table.opponent[at] = torch.where((offsets < meta[:, OPP_LENGTH, None])[..., None], opponent, 0)
        active = self.active[group].repeat_interleave(2)
        table.active[at] = active
        due = active & (clock % meta[:, REPLAN] == 0)
        table.queue_error.logical_or_((due & (ring.length_device.flatten() != meta[:, DELAY])).any())
        assignments = self.assignments[group]
        student = self.owners[group] < 0
        table.valid[at] = due & student
        previous = torch.where(pending >= 0, pending, torch.arange(players, device=self.policy.device))
        continuing = due & student & (pending >= 0)
        table.following[previous] = torch.where(continuing, at, table.following[previous])
        pending.copy_(torch.where(due & student, at, pending))
        if clock % 128 == 0 and not np.array_equal(assignments, self.layout_assignments[group]):
            self.tick_layouts.clear()
            self.group_slots[group].clear()
            self.layout_assignments[group][:] = assignments
            logical = np.asarray(self.group_matches[group], dtype=np.int64).repeat(2) * 2 + np.tile(
                (0, 1), len(self.group_matches[group])
            )
            for buffer in self.buffers[group]:
                in_buffer = np.isin(np.arange(players) // 2, buffer.lanes)
                for owner in np.unique(assignments[in_buffer]):
                    selected = np.flatnonzero(in_buffer & (assignments == owner))
                    anchor = None if owner < 0 else int(owner)
                    key = buffer.index, anchor
                    self.group_slots[group][key] = torch.from_numpy(selected).to(self.policy.device)
                    self.tick_layouts[key] = TickLayout(
                        tuple(logical[selected].tolist()),
                        tuple((selected // 2).tolist()),
                        buffer.index,
                    )
            if self.policy.device.type == "cuda":
                layout_ready = torch.cuda.Event()
                layout_ready.record()
                layout_ready.synchronize()
            for owner, policy in ((None, self.policy), *self.anchors.items()):
                layouts = tuple(layout for (_, anchor), layout in self.tick_layouts.items() if anchor == owner)
                policy.retain_tick_layouts(ring, layouts)
        return at

    def _sample_buffer(
        self,
        buffer: RolloutBuffer,
        clock: int,
        table: DenseRollout,
        ids: Tensor,
    ) -> None:
        ring = self.rings[self.cursor]
        for key, layout in self.tick_layouts.items():
            buffer_index, anchor = key
            if buffer_index != buffer.index:
                continue
            policy = self.policy if anchor is None else self.anchors[anchor]
            rows = TickInputs(table, ids[self.group_slots[self.cursor][key]])
            policy.sample_tick(rows, self.sampler.rng, layout, ring, clock=clock)

    def _tick_ready(self, buffer: RolloutBuffer, clock: int) -> bool:
        ring = self.rings[self.cursor]
        for (buffer_index, anchor), layout in self.tick_layouts.items():
            if buffer_index != buffer.index:
                continue
            policy = self.policy if anchor is None else self.anchors[anchor]
            if not policy.tick_ready(layout, ring, clock=clock):
                return False
        return True

    @contextlib.contextmanager
    def _host_transfers(self, buffers: tuple[RolloutBuffer, ...]) -> Iterator[None]:
        try:
            yield
            for buffer in buffers:
                staging = self.staging.get(id(buffer))
                if staging is not None:
                    staging.release_inputs()
        except BaseException as error:
            # A failed copy must wake every pending worker without exposing its input bytes.
            for buffer in buffers:
                staging = self.staging.get(id(buffer))
                if staging is not None and not staging.ready.done():
                    staging.ready.set_exception(error)
            raise

    def _submit_buffer(self, workers: ThreadPoolExecutor, buffer: RolloutBuffer) -> list[PendingStep]:
        pending: list[PendingStep] = []
        staging = self.staging.get(id(buffer))
        if staging is None:
            # CUDA test seams run on the caller; CPU seams retain simulator overlap.
            for shard in buffer.shards:
                active = self._active(shard)
                if active.any():
                    inputs = self._inputs(shard, 2)
                    if self.policy.device.type == "cpu":
                        future = workers.submit(_step_shard, shard, inputs, active)
                    else:
                        future: Future[tuple[ShardFrames, ...]] = Future()
                        future.set_result(shard.step(inputs, active))
                    pending.append((buffer, shard, future))
            return pending
        staging.started = time.perf_counter()
        if staging.output_pending and staging.output_copied is not None:
            staging.output_copied.synchronize()
        staging.output_reuse_wait = time.perf_counter() - staging.started
        staging.ready = Future()
        staging.output.zero_()
        copying_started = time.perf_counter()
        staging.inputs.copy_(self._controller_inputs(self.buffer_device_lanes[id(buffer)], 2), non_blocking=True)
        if staging.input_copied is not None:
            staging.input_copied.record()
        staging.input_copy_submit = time.perf_counter() - copying_started
        offset = 0
        for shard in buffer.shards:
            active = self._active(shard)
            stop = offset + len(shard.match_ids)
            if active.any():
                if not isinstance(shard, NativeSimulatorShard):
                    raise TypeError("native staging requires native shards")
                pending.append(
                    (
                        buffer,
                        shard,
                        shard.shard.step_packed(
                            staging.host_inputs[:, offset:stop],
                            active,
                            staging.host_output[:, offset:stop],
                            staging.ready,
                        ),
                    )
                )
            offset = stop
        return pending

    def _complete_buffer(
        self, buffer: RolloutBuffer, pending: list[PendingStep]
    ) -> list[tuple[SimulatorShard, tuple[ShardFrames, ...]]]:
        staging = self.staging.get(id(buffer))
        if staging is None:
            return [
                (shard, cast(Future[tuple[ShardFrames, ...]], future).result())
                for owner, shard, future in pending
                if owner is buffer
            ]
        waiting_started = time.perf_counter()
        results = {
            id(shard): cast(Future[native_shard.PackedStep], future).result()
            for owner, shard, future in pending
            if owner is buffer
        }
        worker_wait = time.perf_counter() - waiting_started
        copying_started = time.perf_counter()
        staging.device_output.copy_(staging.output, non_blocking=True)
        if staging.output_copied is not None:
            staging.output_copied.record()
            staging.output_pending = True
        self.transfer_timings.append(
            BufferTiming(
                buffer.index,
                staging.started,
                staging.input_copy_submit,
                staging.input_event_wait,
                staging.output_reuse_wait,
                worker_wait,
                time.perf_counter() - copying_started,
            )
        )
        completed = []
        offset = 0
        for shard in buffer.shards:
            stop = offset + len(shard.match_ids)
            result = results.get(id(shard))
            if isinstance(shard, NativeSimulatorShard):
                shard.copied_bytes += 2 * len(shard.match_ids) * native_shard.STAGING_DTYPE.itemsize
            if result is not None:
                if not isinstance(shard, NativeSimulatorShard):
                    raise TypeError("native staging requires native shards")
                shard.timings.append(
                    ShardTiming(
                        result.started, result.wall, result.simulator, result.packing, result.input_wait, result.other
                    )
                )
                shard.step_seconds += result.wall
                rows = []
                for frame in range(result.frames):
                    packed = staging.device_output[frame, offset:stop]
                    host = staging.host_output[frame, offset:stop]
                    records = host["records"]
                    columns = records["columns"].copy()
                    columns.flags.writeable = False
                    rows.append(
                        ShardFrames(
                            packed[:, : native.FRAME_DTYPE.itemsize],
                            packed[:, native.FRAME_DTYPE.itemsize :].view(torch.float32).reshape(-1, 2, ACTION_DIM),
                            columns,
                            records["reward"].copy(),
                            records["terminated"].copy(),
                            records["truncated"].copy(),
                            host["observed"].copy(),
                        )
                    )
                completed.append((shard, tuple(rows)))
            offset = stop
        return completed

    def _inputs(self, shard: SimulatorShard, count: int) -> Tensor:
        return self._controller_inputs(self.device_lanes[id(shard)], count)

    def _controller_inputs(self, lanes: Tensor, count: int) -> Tensor:
        ring = self.rings[self.cursor]
        sides = torch.arange(2, device=self.policy.device)[None, :, None]
        positions = (ring.head[lanes, :, None] + ring.offsets[:count]) % ring.capacity
        result = ring.controllers[lanes[:, None, None], sides, positions].permute(2, 0, 1, 3)
        return torch.where(self.active[self.cursor][lanes][None, :, None, None], result, 0)

    def _accept(
        self,
        buffer: RolloutBuffer,
        shards: tuple[SimulatorShard, ...],
        batches: tuple[ShardFrames, ...],
        end_clock: int,
        update: int,
        table: DenseRollout,
        pending: Tensor,
    ) -> tuple[int, int]:
        for shard, frames in zip(shards, batches, strict=True):
            if len(frames) != len(shard.match_ids):
                raise ValueError("simulator returned the wrong lane count")
        group = self.cursor
        players = len(self.dense_meta[group])
        ring = self.rings[group]
        if len(shards) == len(buffer.shards):
            host_lanes, lanes = buffer.lanes, self.buffer_device_lanes[id(buffer)]
        else:
            host_lanes = np.concatenate([self.shard_lanes[id(shard)] for shard in shards])
            lanes = torch.cat([self.device_lanes[id(shard)] for shard in shards])
        frames = concatenate_shard_frames(batches)
        slots = (2 * lanes[:, None] + torch.arange(2, device=self.policy.device)[None]).flatten()
        timeouts = 0
        active_host = self.live_host[group][host_lanes]
        active = self.active[group][lanes]
        positions = self.stops[group][lanes]
        rewards = table.store.write(positions, lanes, frames, torch.arange(len(lanes), device=lanes.device))
        terminals, truncations = frames.terminal, frames.timeout
        ids = torch.where(pending[slots] >= 0, pending[slots], slots)
        valid = (pending[slots] >= 0) & active.repeat_interleave(2)
        duration = table.duration[ids]
        table.rewards[ids, duration.clamp_max(3)] = torch.where(
            valid, rewards.flatten(), table.rewards[ids, duration.clamp_max(3)]
        )
        table.duration[ids] += valid.long()
        self.episode_returns[group][lanes] += torch.where(active[:, None], rewards, 0)
        self.stops[group][lanes] += active.long()
        self.observed_clocks[group][lanes] += active.long()
        ring.head[lanes] = (ring.head[lanes] + active[:, None].long()) % ring.capacity
        ring.length_device[lanes] -= active[:, None].long()
        frames_stepped = int(active_host.sum())
        for shard in shards:
            if isinstance(shard, NativeSimulatorShard):
                shard.stepped_frames += int(self._active(shard).sum())
        ends = active_host & (terminals | truncations)
        for source in np.flatnonzero(ends):
            lane = int(host_lanes[source])
            match = self.group_matches[group][lane]
            state = self.matches[match]
            returns = (
                float(self.episode_returns[group][lane, 0]),
                float(self.episode_returns[group][lane, 1]),
            )
            stocks = tuple(
                int(frames.columns[source, NATIVE_COLUMN_LAYOUT[f"p{side}_stock"][0]].view(np.int32))
                for side in (1, 2)
            )
            self.results.append(
                EpisodeResult(
                    match,
                    state.episode,
                    end_clock - state.episode_start,
                    returns,
                    bool(truncations[source]),
                    cast(tuple[int, int] | None, stocks),
                )
            )
            match_slots = torch.arange(2 * lane, 2 * lane + 2, device=self.policy.device)
            pending_ids = pending[match_slots]
            valid_pending = pending_ids >= 0
            target = pending_ids[valid_pending]
            table.boundary[target] = True
            table.terminal[target] = bool(terminals[source])
            episode_rows = (table.integers[:, MATCH] == match) & (table.integers[:, EPISODE] == state.episode.number)
            episode_rows &= table.valid
            ended_rows = torch.nonzero(episode_rows).flatten()
            table.remaining[ended_rows] = end_clock - table.integers[ended_rows, CLOCK]
            if truncations[source]:
                boot = (
                    self.sampler.schedule.rollout_frames // 2 + (end_clock - self.clocks[group] - 1) // 128
                ) * players + match_slots
                self._write_bootstrap(table, boot, match_slots, end_clock, update)
                table.following[target] = boot[valid_pending]
            pending[match_slots] = -1
            state.end_clock = end_clock
            self.active[group][lane] = False
            self.live_host[group][lane] = False
            timeouts += int(truncations[source])
        return frames_stepped, timeouts

    def _write_bootstrap(self, table: DenseRollout, ids: Tensor, slots: Tensor, clock: int, update: int) -> None:
        meta = self._dense_metadata(clock, update, bootstrap=True)[slots]
        table.bootstrap[ids] = True
        table.integers[ids] = meta
        table.conditions[ids] = self.dense_logs[self.cursor][slots]
        ring = self.rings[self.cursor]
        lanes, sides = meta[:, LANE], meta[:, SIDE]
        count = min(24, ring.capacity)
        queue = torch.nn.functional.pad(ring.read(lanes, sides, count), (0, 0, 0, 24 - count))
        opponent = torch.nn.functional.pad(ring.read(lanes, 1 - sides, count), (0, 0, 0, 24 - count))
        offsets = torch.arange(24, device=self.policy.device)[None]
        table.prefix[ids] = torch.where((offsets < meta[:, DELAY, None])[..., None], queue, 0)
        table.opponent[ids] = torch.where((offsets < meta[:, OPP_LENGTH, None])[..., None], opponent, 0)

    def _reset_waiting(self, shards: Sequence[SimulatorShard], clock: int) -> int:
        if clock % 128:
            return 0
        resets = 0
        for shard in shards:
            waiting = np.flatnonzero(~self._active(shard))
            for lane in waiting:
                match = shard.match_ids[lane]
                state = self.matches[match]
                self._reset(shard, int(lane), match, clock, state.episode.number + 1)
                resets += 1
        return resets

    def _active(self, shard: SimulatorShard) -> NDArray[np.bool_]:
        return self.live_host[self.cursor][self.shard_lanes[id(shard)]]

    def collect(self, update: int) -> Rollout:
        if not self._initialized:
            raise ValueError("initialize or restore the pool before rollout")
        started = time.perf_counter()
        self.results = []
        group_id = self.cursor
        shards = self.groups[group_id]
        self.transfer_timings.clear()
        for shard in shards:
            if isinstance(shard, NativeSimulatorShard):
                shard.timings.clear()
        start = self.clocks[group_id]
        count = self.sampler.schedule.rollout_frames
        if count % 128:
            raise ValueError("rollouts must end on a 128-frame block boundary")
        self.tick_layouts = self.group_layouts[group_id]
        previous_store = self.stores[group_id]
        if previous_store is None:
            raise ValueError("pool has no frame storage")
        matches = self.group_matches[group_id]
        capacity = 129 + count + count // 128 + 1
        store = (
            previous_store
            if len(previous_store.records) == capacity
            else FrameStore(capacity, len(matches), self.policy.device)
        )
        lengths = (self.observed_clocks[group_id] - self.episode_clocks[group_id] + 1).clamp_max(129)
        positions = self.stops[group_id][None] - lengths[None] + torch.arange(129, device=self.policy.device)[:, None]
        valid = torch.arange(129, device=self.policy.device)[:, None] < lengths[None]
        positions = torch.where(valid, positions, 0)
        lanes = torch.arange(len(matches), device=self.policy.device)[None]
        for target, source in (
            (store.records, previous_store.records),
            (store.actions, previous_store.actions),
            (store.rewards, previous_store.rewards),
            (store.flags, previous_store.flags),
        ):
            mask = valid.reshape(*valid.shape, *([1] * (target.ndim - 2)))
            target[:129].copy_(torch.where(mask, source[positions, lanes], 0))
        self.stops[group_id].copy_(lengths)
        self.stores[group_id] = store
        table = self.tables[group_id]
        if table is None:
            table = DenseRollout(count // 2, store)
            self.tables[group_id] = table
            if isinstance(self.policy, TransformerPolicy):
                table.allocate_evaluation(self.policy.model)
        table.reset()
        pending = torch.full((2 * len(matches),), -1, dtype=torch.long, device=self.policy.device)
        resets = timeouts = frames_stepped = 0
        buffers = self.buffers[group_id]
        default_stream = torch.cuda.current_stream(self.policy.device) if self.policy.device.type == "cuda" else None
        with ThreadPoolExecutor(max_workers=len(shards)) as workers:
            for clock in range(start, start + count, 2):
                boundary = clock % 128 == 0
                with (
                    self._host_transfers(buffers),
                    sample(
                        self.profile,
                        "boundary_tick" if boundary else "tick",
                        work=float(len(matches)),
                        limit=2 if boundary else 4,
                        skip=0 if boundary else 16,
                    ),
                ):
                    if default_stream is not None:
                        for buffer in buffers:
                            default_stream.wait_stream(cast(torch.cuda.Stream, buffer.stream))
                    resets += self._reset_waiting(shards, clock)
                    with sample(self.profile, "tick_state", work=float(len(matches)), limit=0):
                        ids = self._stage_tick(clock, update, table, pending)
                    ready = all(self._tick_ready(buffer, clock) for buffer in buffers)
                    futures: list[PendingStep] = []
                    for buffer in buffers:
                        inference_stream = buffer.stream if ready else default_stream
                        if ready and buffer.stream is not None:
                            buffer.stream.wait_stream(cast(torch.cuda.Stream, default_stream))
                        with (
                            torch.cuda.stream(inference_stream)
                            if inference_stream is not None
                            else contextlib.nullcontext()
                        ):
                            with sample(self.profile, "gpu_tick", work=float(len(buffer.lanes)), limit=0):
                                self._sample_buffer(buffer, clock, table, ids)
                            if ready:
                                futures.extend(self._submit_buffer(workers, buffer))
                    if not ready:
                        # Cold captures restore the shared ring; serialize them before staging controller inputs.
                        for buffer in buffers:
                            if buffer.stream is not None:
                                buffer.stream.wait_stream(cast(torch.cuda.Stream, default_stream))
                            with (
                                torch.cuda.stream(buffer.stream)
                                if buffer.stream is not None
                                else contextlib.nullcontext()
                            ):
                                futures.extend(self._submit_buffer(workers, buffer))
                    for buffer in buffers:
                        staging = self.staging.get(id(buffer))
                        if staging is not None:
                            staging.release_inputs()
                    for buffer in buffers:
                        with (
                            torch.cuda.stream(buffer.stream) if buffer.stream is not None else contextlib.nullcontext()
                        ):
                            completed = self._complete_buffer(buffer, futures)
                        for shard, rows in completed:
                            if not 1 <= len(rows) <= 2:
                                raise ValueError("simulator must return between one and two frames")
                            if (
                                not isinstance(shard, NativeSimulatorShard)
                                and len(rows) == 2
                                and bool(((rows[0].terminal | rows[0].timeout) & self._active(shard)).any())
                            ):
                                raise ValueError("simulator stepped beyond an episode boundary")
                        with (
                            torch.cuda.stream(buffer.stream) if buffer.stream is not None else contextlib.nullcontext()
                        ):
                            for offset in range(2):
                                accepted: list[tuple[SimulatorShard, ShardFrames]] = []
                                for shard, rows in completed:
                                    if offset < len(rows):
                                        accepted.append((shard, rows[offset]))
                                    elif self._active(shard).any():
                                        rest = _step_shard(shard, self._inputs(shard, 1), self._active(shard))
                                        if len(rest) != 1:
                                            raise ValueError("one-frame step returned the wrong duration")
                                        accepted.append((shard, rest[0]))
                                if not accepted:
                                    continue
                                stepped, ended_timeouts = self._accept(
                                    buffer,
                                    tuple(shard for shard, _ in accepted),
                                    tuple(frames for _, frames in accepted),
                                    clock + offset + 1,
                                    update,
                                    table,
                                    pending,
                                )
                                frames_stepped += stepped
                                timeouts += ended_timeouts
                    if default_stream is not None:
                        for buffer in buffers:
                            default_stream.wait_stream(cast(torch.cuda.Stream, buffer.stream))
        end = start + count
        live_slots = torch.nonzero(pending >= 0).flatten()
        pending_ids = pending[live_slots]
        table.boundary[pending_ids] = True
        boot_ids = (count // 2 + count // 128) * (2 * len(matches)) + live_slots
        self._write_bootstrap(table, boot_ids, live_slots, end, update)
        table.following[pending_ids] = boot_ids
        # Preserve the pre-reset observation and queue before resetting a final boundary.
        resets += self._reset_waiting(shards, end)
        store.lengths.copy_(self.stops[group_id])
        if bool(table.queue_error):
            raise ValueError("decision must see exactly D committed frames")
        ids = torch.nonzero(table.valid).flatten()
        self.clocks[group_id] = end
        self.cursor = (group_id + 1) % len(self.groups)
        self.completed_update = update
        return Rollout(
            table,
            ids,
            group_id,
            update,
            frames_stepped,
            resets,
            timeouts,
            time.perf_counter() - started,
            tuple(self.results),
            count * len(matches) - frames_stepped,
        )


def concatenate_shard_frames(frames: Sequence[ShardFrames]) -> ShardFrames:
    if not frames:
        raise ValueError("frame concatenation needs a nonempty shard batch")
    if len(frames) == 1:
        return frames[0]
    columns = np.concatenate([frame.columns for frame in frames])
    columns.flags.writeable = False
    return ShardFrames(
        torch.cat([frame.records for frame in frames]),
        torch.cat([frame.actions for frame in frames]),
        columns,
        np.concatenate([frame.rewards for frame in frames]),
        np.concatenate([frame.terminal for frame in frames]),
        np.concatenate([frame.timeout for frame in frames]),
        np.concatenate([frame.host_actions for frame in frames]),
    )


@dataclasses.dataclass(frozen=True, slots=True)
class TrainConfig:
    seed: int = 65
    actor_optimizer: Literal["muon", "adam"] = "muon"
    actor_lr: float = 1e-5
    critic_lr: float = 3e-4
    critic_epochs: int = 4
    windows_per_step: int = 512
    decisions_per_window: int = 16
    clip_epsilon: float = 0.2
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


def _log_distribution(probabilities: Tensor) -> Tensor:
    positive = probabilities > 0
    return torch.where(positive, probabilities, 1).log().masked_fill(~positive, -torch.inf)


def factor_kl(policy: PolicyEvaluation, reference: PolicyEvaluation, executed_JF: Tensor) -> Tensor:
    rows = []
    for group, (p, q) in enumerate(zip(policy.distributions, reference.distributions, strict=True)):
        valid = executed_JF[:, group::4, None]
        for probabilities in (p, q):
            active = probabilities[valid.expand_as(probabilities)]
            _check_finite("executed KL probabilities", active)
            if bool((active < 0).any()):
                raise ValueError("executed KL probabilities must be nonnegative")
        neutral = torch.nn.functional.one_hot(torch.zeros_like(p[..., 0], dtype=torch.long), p.shape[-1]).to(p.dtype)
        p = torch.where(valid, p, neutral)
        q = torch.where(valid, q, neutral)
        rows.append(categorical_kl(_log_distribution(p), _log_distribution(q)))
    return torch.stack(rows, -1).flatten(1)


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
    kl: Tensor
    advantages: Tensor
    targets: Tensor
    selected: Tensor
    steps: tuple[Tensor, ...]
    metrics: dict[str, float]
    windows: ReplayBatch
    placement: WindowPlacement


@dataclasses.dataclass(frozen=True, slots=True)
class ReplayBatch:
    """Packed windows and their fixed decision slots; rows index dense storage."""

    records: Tensor
    applied_actions: Tensor
    sides: Tensor
    identities: Tensor
    lengths: Tensor
    slots: ReplaySlots
    rows: Tensor
    metadata: Tensor

    def inputs(self) -> tuple[Tensor, ...]:
        return (
            self.records,
            self.applied_actions,
            self.sides,
            self.identities,
            self.lengths,
            *(getattr(self.slots, name) for name in REPLAY_SLOT_NAMES),
        )

    def take(self, windows: Tensor, positions: Tensor, live: Tensor) -> ReplayBatch:
        """Gather a padded actor step without materializing per-decision history."""
        at = windows.clamp(0, len(self.records) - 1)
        selected_positions = positions[at]
        fields = []
        for name in REPLAY_SLOT_NAMES:
            value = getattr(self.slots, name)
            selected = value[at]
            if name != "history_starts_WL":
                selected = selected[torch.arange(len(at), device=at.device)[:, None], selected_positions]
            if name == "valid_WK":
                selected = selected & live[:, None]
            fields.append(selected)
        return ReplayBatch(
            self.records[at],
            self.applied_actions[at],
            self.sides[at],
            self.identities[at],
            self.lengths[at],
            ReplaySlots(*fields),
            self.rows[at[:, None], selected_positions],
            self.metadata[at[:, None], selected_positions],
        )


def concatenate_replays(outputs: Sequence[ReplayOutput], dimension: int) -> ReplayOutput:
    return ReplayOutput(
        torch.cat([output.log_probs_WKF for output in outputs], dimension),
        tuple(torch.cat([output.distributions[group] for output in outputs], dimension) for group in range(4)),
        torch.cat([output.factor_mask_WKF for output in outputs], dimension),
        ReplayCriticFeatures(
            *(
                torch.cat([getattr(output.features, field.name) for output in outputs], dimension)
                for field in dataclasses.fields(outputs[0].features)
            )
        ),
    )


REPLAY_SLOT_NAMES = tuple(field.name for field in dataclasses.fields(ReplaySlots))


def replay_batch(policy: LoopPolicy, table: DenseRollout, rows: Tensor, valid: Tensor) -> ReplayBatch:
    """Assemble device slots from one episode per block, guaranteed by reset cadence."""
    meta = table.integers[rows]
    active_lengths = torch.where(valid, meta[..., LENGTH], 0)
    longest = active_lengths.argmax(1)
    index = torch.arange(len(rows), device=rows.device)
    window = meta[index, longest]
    lengths = active_lengths.amax(1).clamp_min(1)
    time = torch.arange(256, device=rows.device)[None]
    in_history = time < lengths[:, None]
    positions = torch.where(in_history, window[:, START, None] + time, 0)
    flat = (positions * table.store.records.shape[1] + window[:, LANE, None]).flatten()
    records = table.store.records.flatten(0, 1)[flat].reshape(len(rows), 256, -1)
    applied = table.store.actions.flatten(0, 1)[flat].reshape(len(rows), 256, 2, ACTION_DIM)
    records = torch.where(in_history[..., None], records, 0)
    applied = torch.where(in_history[..., None, None], applied, 0)
    conditions = table.conditions[rows]
    remaining = table.remaining[rows]
    remaining = torch.where(remaining < 0, meta[..., DELAY] + meta[..., REPLAN], remaining)
    frame = torch.arange(1, 5, device=rows.device)
    mask = (frame <= meta[..., REPLAN, None]) & (meta[..., DELAY, None] + frame <= remaining[..., None])
    mask = mask.repeat_interleave(4, -1)
    slots = ReplaySlots(
        (meta[..., LENGTH] - 1).clamp_min(0),
        valid,
        torch.zeros(len(rows), 256, dtype=torch.long, device=rows.device),
        table.prefix[rows, : policy.max_delay],
        meta[..., DELAY],
        meta[..., REPLAN],
        policy.condition_ranges.normalize(
            conditions[..., 0], meta[..., DELAY], meta[..., REPLAN], percentile=conditions[..., 2], validate=False
        ),
        policy.condition_ranges.normalize(
            conditions[..., 1],
            meta[..., OPP_DELAY],
            meta[..., OPP_REPLAN],
            percentile=conditions[..., 3],
            validate=False,
        ),
        table.opponent[rows, : policy.max_opponent_queue],
        meta[..., OPP_LENGTH],
        table.actions[rows],
        mask.reshape(*rows.shape, 16),
    )
    return ReplayBatch(records, applied, window[:, SIDE], window[:, IDENTITY], lengths, slots, rows, meta)


@dataclasses.dataclass(frozen=True, slots=True)
class WindowPlacement:
    selected: Tensor  # [W,K]
    positions: Tensor  # [W,decisions_per_window]
    windows: Tensor  # [steps,windows_per_step], -1 padding


class PhaseMemory:
    """Measure the first update and first actor update, including nested phases."""

    def __init__(self, device: torch.device, windows_per_step: int) -> None:
        self.device = device
        self.windows_per_step = windows_per_step
        self.label = "first_update"
        self.active = False
        self.first_checked = False
        self.actor_checked = False
        self.peaks: dict[str, tuple[int, int]] = {}
        self.stack: list[str] = []
        self.output: Path | None = None

    @contextlib.contextmanager
    def phase(self, name: str) -> Iterator[None]:
        measure = self.active and self.device.type == "cuda"
        if measure:
            torch.cuda.synchronize(self.device)
            self.record()
            torch.cuda.reset_peak_memory_stats(self.device)
            self.stack.append(f"{self.label}_{name}")
        try:
            yield
        except torch.cuda.OutOfMemoryError as error:
            if measure:
                self.record()
                self.publish()
            raise RuntimeError(
                f"O65 CUDA OOM in {name}; configured actor step is {self.windows_per_step} windows. "
                "Stop this run and inspect phase-memory.json. No automatic batch-size change is allowed."
            ) from error
        finally:
            if measure:
                self.record()
                self.stack.pop()
                torch.cuda.reset_peak_memory_stats(self.device)
                self.publish()
        if measure and name == "actor":
            self.actor_checked = True

    def record(self) -> None:
        allocated = torch.cuda.max_memory_allocated(self.device)
        reserved = torch.cuda.max_memory_reserved(self.device)
        for name in self.stack:
            old = self.peaks.get(name, (0, 0))
            self.peaks[name] = (max(old[0], allocated), max(old[1], reserved))

    def publish(self) -> None:
        if self.output is not None:
            self.output.mkdir(parents=True, exist_ok=True)
            temporary = self.output / ".phase-memory.partial"
            temporary.write_text(json.dumps(self.metrics(), indent=2))
            temporary.replace(self.output / "phase-memory.json")

    def metrics(self) -> dict[str, float]:
        return {
            f"memory/{name}_peak_{kind}_bytes": float(value)
            for name, values in self.peaks.items()
            for kind, value in zip(("allocated", "reserved"), values, strict=True)
        }


def gradient_metrics(name: str, norms: Sequence[Tensor], cap: float) -> dict[str, float]:
    if not norms:
        return {}
    values = torch.stack(tuple(norms)).float()
    return {
        f"{name}/gradient_norm": float(values.mean()),
        f"{name}/gradient_norm_max": float(values.max()),
        f"{name}/gradient_cap_fraction": float((values > cap).float().mean()),
    }


class GradientNoise:
    """Online window-scale estimate from pre-clip gradients across actor steps.

    Consecutive gradients use moving weights. This cheap diagnostic is not a
    fixed-weight measurement; report signal and validity with the estimate.
    """

    def __init__(self, parameters: tuple[torch.nn.Parameter, ...]) -> None:
        self.parameters = parameters
        self.total = tuple(torch.zeros_like(p) for p in parameters)
        self.squares = parameters[0].new_zeros(())
        self.windows = 0
        self.decisions = 0
        self.steps = 0

    @torch.no_grad()
    def add(self, windows: int, decisions: int) -> None:
        for parameter, total in zip(self.parameters, self.total, strict=True):
            if parameter.grad is not None:
                gradient = parameter.grad.detach()
                total.add_(gradient, alpha=decisions)
                self.squares.add_(gradient.float().square().sum(), alpha=decisions)
        self.windows += windows
        self.decisions += decisions
        self.steps += 1

    def metrics(self) -> dict[str, float]:
        result = {"actor/gradient_noise_valid": 0.0, "actor/gradient_noise_steps": float(self.steps)}
        if self.steps < 2:
            return result
        mean_square = torch.stack([t.float().square().sum() for t in self.total]).sum() / self.decisions**2
        noise = ((self.squares - self.decisions * mean_square) / (self.steps - 1)).clamp_min(0)
        signal = mean_square - noise / self.decisions
        result.update({"actor/gradient_noise_variance": float(noise), "actor/gradient_signal_squared": float(signal)})
        if float(signal) > 0:
            estimate = noise / signal * (self.windows / self.decisions)
            _check_finite("gradient noise scale", estimate)
            result.update({"actor/gradient_noise_scale_windows": float(estimate), "actor/gradient_noise_valid": 1.0})
        return result


@torch.no_grad()
def parameter_metrics(
    name: str, optimizer: torch.optim.Optimizer, before: tuple[Tensor, ...], policy: LoopPolicy, reference: LoopPolicy
) -> dict[str, float]:
    parameters = policy.actor_parameters() if name == "actor" else policy.critic_parameters()
    previous = dict(zip(map(id, parameters), before, strict=True))
    keys = {id(p): key for key, p in policy.model.named_parameters()}
    base = dict(reference.model.named_parameters())
    metrics = {}
    for index, group in enumerate(optimizer.param_groups):
        params = cast(list[torch.nn.Parameter], group["params"])
        weight = torch.stack([previous[id(p)].float().square().sum() for p in params]).sum().sqrt()
        delta = torch.stack([(p.float() - previous[id(p)]).square().sum() for p in params]).sum().sqrt()
        distance = torch.stack([(p.float() - base[keys[id(p)]]).square().sum() for p in params]).sum().sqrt()
        metrics[f"{name}/reference_distance_group{index}"] = float(distance)
        metrics[f"{name}/update_norm_group{index}"] = float(delta)
        # A zero group has no defined relative update. Keep its absolute norm.
        if float(weight) > 0:
            metrics[f"{name}/update_to_weight_group{index}"] = float(delta / weight)
    return metrics


@torch.no_grad()
def action_statistics(
    model: SelfPlayTransformer, old: PolicyEvaluation, reference: PolicyEvaluation, executed: Tensor
) -> dict[str, float]:
    metrics = {}
    for factor, name in enumerate(FACTOR_ORDER):
        valid = executed[:, factor::4]
        if not bool(valid.any()):
            continue
        size = old.distributions[factor].shape[-1]
        semantics = model.codec.semantic_values(name, torch.arange(size, device=executed.device)).float()
        for label, evaluation in (("student", old), ("reference", reference)):
            mean = evaluation.distributions[factor][valid].mean(0)
            if name == "buttons":
                for index, value in enumerate(mean @ semantics):
                    metrics[f"action/{label}_button{index}_press_rate"] = float(value)
            elif name == "triggers":
                for index, value in enumerate(mean @ semantics):
                    metrics[f"action/{label}_trigger{index}_mean"] = float(value)
            else:
                magnitude = semantics.square().sum(-1).sqrt()
                bins = torch.bucketize(magnitude, magnitude.new_tensor((0.01, 0.25, 0.5, 0.75, 1.01)))
                histogram = mean.new_zeros(6).scatter_add_(0, bins, mean)
                for index, value in enumerate(histogram):
                    metrics[f"action/{label}_{name}_magnitude_bin{index}"] = float(value)
    return metrics


@torch.no_grad()
def reward_component_metrics(rollout: Rollout) -> dict[str, float]:
    table = rollout.table
    meta = table.integers[rollout.ids]
    positions = meta[:, START] + meta[:, LENGTH] - 1
    lane, side = meta[:, LANE], meta[:, SIDE]
    valid = torch.arange(4, device=meta.device)[None] < table.duration[rollout.ids, None]
    at = (positions[:, None] + torch.arange(5, device=meta.device)).clamp_max(len(table.store.records) - 1)
    records = table.store.records[at, lane[:, None]]
    columns = records[:, :, : 4 * len(native.OBSERVATION_FIELDS)].contiguous().view(torch.int32)
    stock = torch.stack([columns[..., NATIVE_COLUMN_LAYOUT[f"p{s}_stock"][0]] for s in (1, 2)], -1)
    percent = torch.stack([columns[..., NATIVE_COLUMN_LAYOUT[f"p{s}_percent"][0]] for s in (1, 2)], -1).view(
        torch.float32
    )
    losses = (stock[:, 1:] < stock[:, :-1]).float()
    damage = (percent[:, 1:] - percent[:, :-1]).clamp_min(0)
    wins = losses * (stock[:, 1:] == 0)
    result = {}
    rows = torch.arange(len(meta), device=meta.device)[:, None]
    for name, value, scale in (("damage", damage, 1.0), ("stock", losses, 120.0), ("win", wins, 50.0)):
        positive = value[rows, torch.arange(4, device=meta.device), 1 - side[:, None]] * scale
        negative = -value[rows, torch.arange(4, device=meta.device), side[:, None]] * scale
        net = (positive + negative) / scale
        result[f"return/{name}_positive"] = float(torch.where(valid, positive, 0).sum() / len(meta))
        result[f"return/{name}_negative"] = float(torch.where(valid, negative, 0).sum() / len(meta))
        result[f"return/{name}_reward"] = float(torch.where(valid, net * scale, 0).sum() / len(meta))
    return result


def place_windows(
    batch: ReplayBatch, rng: CounterRng, update: int, count: int, windows_per_step: int
) -> WindowPlacement:
    if not 1 <= count <= batch.rows.shape[1] or windows_per_step < 1:
        raise ValueError("invalid decision/window selection shape")
    eligible = batch.slots.valid_WK & batch.slots.executed_WKF.any(-1)
    meta = batch.metadata
    draws = rng.tensor_uniform(1, meta[..., STREAM], meta[..., EPISODE], meta[..., CLOCK])
    positions = torch.argsort(torch.where(eligible, draws, torch.inf), stable=True)[:, :count]
    selected = torch.zeros_like(eligible)
    selected.scatter_(1, positions, eligible.gather(1, positions))
    live = selected.any(-1)
    total = int(live.sum())
    if not total:
        return WindowPlacement(
            selected, positions, torch.empty(0, windows_per_step, dtype=torch.long, device=meta.device)
        )
    matches, inverse = torch.unique(meta[:, 0, MATCH], return_inverse=True)
    sizes = torch.zeros(len(matches), dtype=torch.long, device=meta.device)
    sizes.scatter_add_(0, inverse, live.long())
    steps = max(int(sizes.max()), math.ceil(total / windows_per_step))
    match_order = torch.argsort(
        rng.tensor_uniform(2, matches, torch.zeros_like(matches), torch.full_like(matches, update)), stable=True
    )
    rank = torch.empty_like(match_order)
    rank[match_order] = torch.arange(len(matches), device=meta.device)
    ordered = torch.argsort(rank[inverse], stable=True)
    ordered = ordered[live[ordered]]
    index = torch.arange(total, device=meta.device)
    placed = torch.full((steps, windows_per_step), -1, dtype=torch.long, device=meta.device)
    placed[index % steps, index // steps] = ordered
    return WindowPlacement(selected, positions, placed)


def window_indices(rows: Tensor, players: int) -> Tensor:
    tick = rows // players
    return ((tick // 64) * players + rows % players) * 64 + tick % 64


def window_evaluation(output: ReplayOutput, actions: Tensor, indices: Tensor) -> PolicyEvaluation:
    def flatten(value: Tensor) -> Tensor:
        return value.flatten(0, 1)[indices]

    features = output.features
    return PolicyEvaluation(
        actions,
        flatten(output.log_probs_WKF),
        tuple(flatten(value) for value in output.distributions),
        CriticFeatures(
            *(
                flatten(getattr(features, name))
                for name in (
                    "trunk_WKd",
                    "decoder_WKt",
                    "condition_WK4",
                    "opponent_condition_WK4",
                    "opponent_queue_WKQ4",
                    "opponent_lengths_WK",
                )
            )
        ),
    )


def decision_batch(rollout: Rollout) -> DecisionBatch:
    table, ids = rollout.table, rollout.ids
    if not len(ids):
        raise ValueError("rollout has no student decisions")
    meta = table.integers[ids]
    delay, replan = meta[:, DELAY].contiguous(), meta[:, REPLAN].contiguous()
    remaining = table.remaining[ids]
    remaining = torch.where(remaining < 0, delay + replan, remaining)
    return DecisionBatch(
        observation_frame_J=meta[:, CLOCK].contiguous(),
        decision_frame_J=meta[:, CLOCK].contiguous(),
        sampled_indices_JF=table.actions[ids][..., DECODE_GROUP_INDICES].flatten(1),
        committed_prefix_JD4=table.prefix[ids][..., DECODE_GROUP_INDICES],
        behavior_log_probs_JF=table.behavior[ids],
        executed_factors_JF=executed_factor_mask(delay, replan, remaining),
        duration_J=table.duration[ids],
        bootstrap_mask_J=~table.terminal[ids],
        trace_mask_J=~table.boundary[ids],
        kl_coefficient_J=table.coefficients[ids],
        delay_J=delay,
        replan_J=replan,
        policy_version_J=meta[:, UPDATE].contiguous(),
        episode_id_J=meta[:, EPISODE].contiguous(),
        segment_id_J=meta[:, EPISODE].contiguous(),
    )


def percentile_bin(u: float) -> int:
    return sum(u > edge for edge in (0.5, 0.75, 0.9, 0.97))


class Trainer:
    def __init__(self, engine: RolloutEngine, config: TrainConfig) -> None:
        if torch.are_deterministic_algorithms_enabled() != config.deterministic:
            raise ValueError("trainer determinism differs from the configured execution mode")
        if engine.sampler.rng.seed != config.seed:
            raise ValueError("trainer and counter RNG seeds must agree")
        self.profile: UpdateProfile | None = None
        self.memory = PhaseMemory(engine.policy.device, config.windows_per_step)
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
        self.episode_metrics: dict[tuple[int, int], tuple[float, float, float, float]] = {}

    @torch.no_grad()
    def completed_metrics(self, rollout: Rollout, kl: Tensor) -> dict[str, float]:
        """Compare the initial value with full-episode discounted task return.

        Exclude truncations. KL cost follows the trained rollout-tail convention
        in the objective; also report the task return without regularization.
        """
        table, ids = rollout.table, rollout.ids
        meta, age = table.integers[ids], table.age[ids]
        pairs, inverse = torch.unique(meta[:, [STREAM, EPISODE]], dim=0, return_inverse=True)
        count = len(pairs)
        first = torch.full((count,), len(ids), device=ids.device, dtype=torch.long)
        first.scatter_reduce_(0, inverse, torch.arange(len(ids), device=ids.device), reduce="amin")
        frames = torch.arange(4, device=ids.device)
        rewards = torch.where(frames < table.duration[ids, None], table.rewards[ids], 0)
        discounted = (rewards * GAMMA ** (age[:, None] + frames)).sum(-1)
        costs = torch.where(decision_batch(rollout).executed_factors_JF, kl, 0).sum(-1) * table.coefficients[ids]
        totals = torch.zeros(count, 3, device=ids.device)
        totals.scatter_add_(
            0, inverse[:, None].expand(-1, 3), torch.stack((discounted, costs, costs * GAMMA**age), -1)
        )
        packed = torch.cat((totals, table.values[ids[first], None], age[first, None]), -1).cpu().tolist()
        for key, row in zip(pairs.cpu().tolist(), packed, strict=True):
            identity = (int(key[0]), int(key[1]))
            previous = self.episode_metrics.get(identity)
            if previous is None and row[4] != 0:
                raise ValueError("episode metrics lost the episode's initial state")
            self.episode_metrics[identity] = (
                row[0] + (previous[0] if previous else 0),
                row[1] + (previous[1] if previous else 0),
                previous[2] if previous else row[3],
                row[2] + (previous[3] if previous else 0),
            )
        metrics: dict[str, float] = {}
        costs_done, values, returns, task_returns = [], [], [], []
        bins: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for result in rollout.results:
            match_cost = 0.0
            for side, player in enumerate(result.episode.players):
                if player.anchor is not None:
                    continue
                discounted_return, cost, value, discounted_cost = self.episode_metrics.pop(
                    (2 * result.match + side, result.episode.number)
                )
                match_cost += cost
                if result.timeout:
                    continue
                values.append(value)
                returns.append(discounted_return - discounted_cost)
                task_returns.append(discounted_return)
                if result.stocks is not None:
                    bins[percentile_bin(player.percentile)].append(
                        (result.returns[side], float(result.stocks[side] > result.stocks[1 - side]))
                    )
            costs_done.append(match_cost)
        metrics["critic/monte_carlo_episode_count"] = float(len(values))
        if values:
            v, target = torch.tensor(values, dtype=torch.float64), torch.tensor(returns, dtype=torch.float64)
            metrics["critic/monte_carlo_initial_value"] = float(v.mean())
            metrics["critic/monte_carlo_task_return"] = sum(task_returns) / len(task_returns)
            metrics["critic/monte_carlo_return"] = float(target.mean())
            metrics["critic/monte_carlo_rmse"] = float((v - target).square().mean().sqrt())
        if costs_done:
            metrics["return/completed_match_kl_cost"] = sum(costs_done) / len(costs_done)
        for index, rows in bins.items():
            metrics[f"return/completed_episode_u{index}"] = sum(row[0] for row in rows) / len(rows)
            metrics[f"gameplay/win_rate_u{index}"] = sum(row[1] for row in rows) / len(rows)
        return metrics

    @torch.no_grad()
    def prepare(self, rollout: Rollout) -> PreparedUpdate:
        started = time.perf_counter()
        device = self.policy.device
        table, ids = rollout.table, rollout.ids
        decisions = decision_batch(rollout)
        if table.evaluation is None:
            raise ValueError("rollout is missing its behavior evaluation")
        players = 2 * len(self.engine.group_matches[rollout.group])
        ticks = self.engine.sampler.schedule.rollout_frames // 2
        size = ticks * players
        rows = torch.arange(size, device=device).reshape(-1, 64, players).transpose(1, 2).reshape(-1, 64)
        windows = replay_batch(self.policy, table, rows, table.valid[rows])
        placement = place_windows(
            windows,
            self.engine.sampler.rng,
            rollout.update,
            self.config.decisions_per_window,
            self.config.windows_per_step,
        )
        compact = window_indices(ids, players)
        assert table.evaluation is not None
        old = _take_evaluation(table.evaluation, ids)
        with (
            self.memory.phase("reference"),
            phase(self.profile, "reference"),
            sample(self.profile, "windows", work=float(rows.numel())),
        ):
            reference = (
                None
                if self.warming
                else window_evaluation(
                    self.reference.evaluate_reference(windows, placement.positions, placement.windows),
                    old.actions,
                    compact,
                )
            )
        kl = (
            torch.zeros_like(old.log_probs)
            if reference is None
            else factor_kl(old, reference, decisions.executed_factors_JF)
        )
        frame_rewards = table.rewards[ids]
        rewards = regularized_reward(decisions, frame_rewards, kl)
        values = table.values[ids]
        next_values = torch.zeros_like(values)
        following = table.following[ids]
        terminal = table.terminal[ids]
        boundary = table.boundary[ids]
        linked = following.clamp_min(0)
        meta = table.integers[ids]
        next_meta = table.integers[linked]
        exists = torch.where(boundary, table.bootstrap[linked], table.valid[linked])
        invalid = (
            (following < 0)
            | ~exists
            | (next_meta[:, STREAM] != meta[:, STREAM])
            | (next_meta[:, EPISODE] != meta[:, EPISODE])
            | (next_meta[:, CLOCK] != meta[:, CLOCK] + table.duration[ids])
        )
        if bool((~terminal & invalid).any()):
            raise ValueError("decision transition is missing its next state or crosses a stream, episode or clock")
        boundary_mask = boundary & ~terminal
        boundary_at = torch.nonzero(boundary_mask).flatten()
        all_values = table.values.clone()
        bootstrap_valid = torch.zeros_like(table.valid)
        bootstrap_valid[following[boundary_at]] = True
        for role in range(ticks // 64 + 1):
            rows = torch.arange(size + role * players, size + (role + 1) * players, device=device)[:, None]
            valid = bootstrap_valid[rows]
            if not bool(valid.any()):
                continue
            with sample(self.profile, "bootstrap", work=float(players)):
                bootstrap_batch = replay_batch(self.policy, table, rows, valid)
                all_values[rows[:, 0]] = self.policy.evaluate_values(bootstrap_batch).flatten()
        next_values = torch.where(terminal, 0, all_values[following.clamp_min(0)])
        dense_rewards = torch.zeros_like(table.values)
        dense_next = torch.zeros_like(table.values)
        dense_rewards[ids] = rewards
        dense_next[ids] = next_values
        result = dense_advantages(
            dense_rewards[:size].reshape(ticks, players),
            table.values[:size].reshape(ticks, players),
            dense_next[:size].reshape(ticks, players),
            table.duration[:size].reshape(ticks, players),
            ~table.terminal[:size].reshape(ticks, players),
            ~table.boundary[:size].reshape(ticks, players),
            table.valid[:size].reshape(ticks, players),
        )
        advantages = result.advantages.flatten()[ids]
        targets = result.value_targets.flatten()[ids]
        selected = placement.selected.flatten()[compact]
        steps = tuple(placement.windows.unbind(0))
        executed = decisions.executed_factors_JF
        metrics = {
            "critic/explained_variance": explained_variance(values, targets),
            "policy/kl_per_frame": float(torch.where(executed, kl, 0).sum() / executed[:, ::4].sum().clamp_min(1)),
            "return/interval_reward": float(frame_rewards.sum(-1).mean()),
            "data/anchor_share": sum(
                player.anchor is not None for state in self.engine.matches.values() for player in state.episode.players
            )
            / len(self.engine.matches)
            / 2,
            "data/resets": float(rollout.resets),
            "data/timeouts": float(rollout.timeouts),
            "systems/rollout_seconds": rollout.elapsed,
            "systems/preparation_seconds": time.perf_counter() - started,
            "systems/simulator_fps": rollout.frames / rollout.elapsed,
            "systems/rollout_frames": float(rollout.frames),
            "systems/reset_idle_frames": float(rollout.idle_frames),
            "systems/reset_idle_fraction": rollout.idle_frames / (rollout.frames + rollout.idle_frames),
            "data/decisions": float(len(ids)),
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
                    torch.where(executed[mask], kl[mask], 0).sum() / executed[mask, ::4].sum().clamp_min(1)
                )
        for position in range(4):
            mask = (decisions.decision_frame_J % 128) // 32 == position
            if bool(mask.any()):
                metrics[f"critic/explained_variance_position{position}"] = explained_variance(
                    values[mask], targets[mask]
                )
        for name, column in (("identity", IDENTITY), ("character", CHARACTER), ("stage", STAGE)):
            keys, counts = torch.unique(meta[:, column], return_counts=True)
            for key, count in zip(keys.cpu().tolist(), counts.cpu().tolist(), strict=True):
                metrics[f"data/{name}_{key}"] = float(count)
        final_rows = torch.arange(size + ticks // 64 * players, size + (ticks // 64 + 1) * players, device=device)[
            : min(32, players)
        ]
        final_valid = bootstrap_valid[final_rows]
        metrics["critic/bootstrap_context_count"] = float(final_valid.sum())
        if bool(final_valid.any()):
            shorter = replay_batch(self.policy, table, final_rows[:, None], final_valid[:, None])
            length = shorter.lengths.clamp_max(129)
            shift = shorter.lengths - length
            frame = torch.arange(256, device=device)[None]
            source = (shift[:, None] + frame).clamp_max(255)
            records_short = shorter.records.gather(1, source[..., None].expand(-1, -1, shorter.records.shape[-1]))
            applied_short = shorter.applied_actions.gather(1, source[..., None, None].expand(-1, -1, 2, ACTION_DIM))
            shorter = dataclasses.replace(
                shorter,
                records=records_short,
                applied_actions=applied_short,
                lengths=length,
                slots=dataclasses.replace(shorter.slots, positions_WK=length[:, None] - 1),
            )
            difference = (self.policy.evaluate_values(shorter).flatten() - all_values[final_rows]).abs()
            metrics["critic/bootstrap_context_gap"] = float(
                torch.where(final_valid, difference, 0).sum() / final_valid.sum()
            )
        for timing in self.engine.sampler.schedule.configurations:
            mask = (decisions.delay_J == timing.delay) & (decisions.replan_J == timing.replan)
            if not bool(mask.any()):
                continue
            suffix = f"d{timing.delay}_r{timing.replan}"
            metrics[f"policy/kl_per_frame_{suffix}"] = float(kl[mask].sum() / executed[mask, ::4].sum().clamp_min(1))
            metrics[f"critic/advantage_mean_{suffix}"] = float(advantages[mask].mean())
            metrics[f"critic/advantage_std_{suffix}"] = float(advantages[mask].std(unbiased=False))
            metrics[f"critic/explained_variance_{suffix}"] = explained_variance(values[mask], targets[mask])
        for factor, distribution in zip(FACTOR_ORDER, old.distributions, strict=True):
            entropy = -(distribution * torch.where(distribution > 0, distribution, 1).log()).sum(-1)
            valid = torch.arange(4, device=device)[None] < decisions.replan_J[:, None]
            metrics[f"policy/entropy_{factor}"] = float(entropy[valid].mean())
        for index, factor in enumerate(FACTOR_ORDER):
            frames = executed[:, index::4]
            if bool(frames.any()):
                metrics[f"policy/kl_{factor}_per_frame"] = float(kl[:, index::4][frames].mean())
        if reference is not None and isinstance(self.policy, TransformerPolicy):
            metrics.update(action_statistics(self.policy.model, old, reference, executed))
        metrics.update(reward_component_metrics(rollout))

        ages = table.age[ids]
        age_bins = torch.bucketize(ages, ages.new_tensor((128, 512, 2048, 8192)))
        for index, value in enumerate(torch.bincount(age_bins, minlength=5).float() / len(ids)):
            metrics[f"data/episode_age_bin{index}"] = float(value)
        metrics["data/episode_age_mean_frames"] = float(ages.float().mean())
        u_bins = torch.bucketize(table.conditions[ids, 2], table.conditions.new_tensor((0.5, 0.75, 0.9, 0.97)))
        entropy = torch.stack([-(p * torch.where(p > 0, p, 1).log()).sum(-1) for p in old.distributions]).sum(0)
        for index in range(5):
            mask = u_bins == index
            frames = executed[:, ::4] & mask[:, None]
            if bool(mask.any()):
                metrics[f"return/interval_reward_u{index}"] = float(frame_rewards[mask].sum(-1).mean())
            if bool(frames.any()):
                metrics[f"policy/kl_per_frame_u{index}"] = float(
                    torch.where(executed & mask[:, None], kl, 0).sum() / frames.sum()
                )
                metrics[f"policy/entropy_u{index}"] = float(entropy[frames].mean())
        return PreparedUpdate(
            rollout,
            decisions,
            old,
            reference,
            values,
            kl,
            advantages,
            targets,
            selected,
            steps,
            metrics,
            windows,
            placement,
        )

    def fit(self, batch: PreparedUpdate) -> dict[str, float]:
        if not self.warming and batch.reference is None:
            raise ValueError("actor update requires a reference evaluation")
        cfg = self.config
        metrics = dict(batch.metrics)
        actor_before = () if self.warming else tuple(p.detach().clone() for p in self.policy.actor_parameters())
        critic_before = tuple(p.detach().clone() for p in self.policy.critic_parameters())
        critic_norms: list[Tensor] = []
        started = time.perf_counter()
        with (
            self.memory.phase("critic"),
            phase(self.profile, "critic"),
            sample(self.profile, "fit", work=float(len(batch.values))),
        ):
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
                _check_finite("critic loss", loss)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.critic_parameters(), cfg.gradient_clip, error_if_nonfinite=True
                )
                critic_norms.append(norm.detach())
                self.critic_optimizer.step()
            metrics["critic/loss"] = float(loss.detach())
            metrics["systems/critic_seconds"] = time.perf_counter() - started
        metrics.update(gradient_metrics("critic", critic_norms, cfg.gradient_clip))
        metrics.update(parameter_metrics("critic", self.critic_optimizer, critic_before, self.policy, self.reference))
        if self.warming:
            self._warmup(batch)
            return metrics
        metrics["policy/empty_update"] = float(not batch.steps)
        if not batch.steps:
            return metrics
        assert batch.reference is not None
        started = time.perf_counter()
        policy_losses: list[Tensor] = []
        kl_losses: list[Tensor] = []
        ratios: list[Tensor] = []
        old_log_ratios: list[Tensor] = []
        actor_norms: list[Tensor] = []
        noise = GradientNoise(self.policy.actor_parameters())
        table = batch.rollout.table
        dense_advantage = torch.zeros_like(table.values)
        dense_advantage[batch.rollout.ids] = batch.advantages
        assert batch.reference is not None
        reference_dense = tuple(
            value.new_zeros(len(table.values), *value.shape[1:]) for value in batch.reference.distributions
        )
        for target, value in zip(reference_dense, batch.reference.distributions, strict=True):
            target[batch.rollout.ids] = value
        with self.memory.phase("actor"), phase(self.profile, "actor"):
            for step, window_ids in enumerate(batch.steps):
                with sample(
                    self.profile, "step", work=float(cfg.windows_per_step * cfg.decisions_per_window), limit=2
                ):
                    live = window_ids >= 0
                    windows = batch.windows.take(window_ids, batch.placement.positions, live)
                    selected = (
                        batch.placement.selected[window_ids.clamp_min(0)].gather(
                            1, batch.placement.positions[window_ids.clamp_min(0)]
                        )
                        & live[:, None]
                    )
                    advantages = dense_advantage[windows.rows]
                    statistics = advantage_statistics(advantage_moments(advantages.flatten(), selected.flatten()))
                    targets = ActorTargets(
                        table.behavior[windows.rows],
                        advantages,
                        selected,
                        table.coefficients[windows.rows],
                        tuple(value[windows.rows] for value in reference_dense),
                        statistics.mean.float(),
                        statistics.sigma.float(),
                        selected.sum().float(),
                    )
                    result, current = self.policy.actor_step(windows, targets, clip_epsilon=cfg.clip_epsilon)
                    _check_finite(
                        "actor loss, KL and joint ratio",
                        torch.cat(
                            (
                                torch.stack((result.loss, result.policy_loss, result.kl_loss)),
                                result.factor_kl_WKF.flatten(),
                                result.joint_ratio_WK[selected],
                            )
                        ),
                    )
                    noise.add(int(live.sum()), int(selected.sum()))
                    norm = torch.nn.utils.clip_grad_norm_(
                        self.policy.actor_parameters(), cfg.gradient_clip, error_if_nonfinite=True
                    )
                    actor_norms.append(norm.detach())
                    with sample(self.profile, "muon", limit=0):
                        self.actor_optimizer.step()
                    self.scheduler.step()
                    self.actor_steps += 1
                    policy_losses.append(result.policy_loss.detach())
                    kl_losses.append(result.kl_loss.detach())
                    ratio = result.joint_ratio_WK[selected].detach()
                    ratios.append(ratio)
                    executed = current.factor_mask_WKF & selected[..., None]
                    log_ratio = (
                        torch.where(executed, current.log_probs_WKF - table.behavior[windows.rows], 0)
                        .sum(-1)[selected]
                        .detach()
                    )
                    old_log_ratios.append(log_ratio)
                    metrics[f"policy/old_log_ratio_step{step}"] = float(log_ratio.mean())
                    assert table.evaluation is not None
                    new = window_evaluation(
                        current,
                        table.actions[windows.rows].flatten(0, 1),
                        torch.arange(windows.rows.numel(), device=windows.rows.device),
                    )
                    previous = _take_evaluation(table.evaluation, windows.rows.flatten())
                    mask = (current.factor_mask_WKF & selected[..., None]).flatten(0, 1)
                    old_kl = factor_kl(new, previous, mask)
                    metrics[f"policy/old_kl_step{step}"] = float(torch.where(mask, old_kl, 0).sum() / selected.sum())
        ratio = torch.cat(ratios)
        metrics.update(gradient_metrics("actor", actor_norms, cfg.gradient_clip))
        metrics.update(noise.metrics())
        metrics.update(parameter_metrics("actor", self.actor_optimizer, actor_before, self.policy, self.reference))
        metrics.update(
            {
                "policy/loss": float(torch.stack(policy_losses).mean()),
                "policy/kl_loss": float(torch.stack(kl_losses).mean()),
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
        self.memory.active = self.policy.device.type == "cuda" and (
            not self.memory.first_checked or not self.memory.actor_checked and not self.warming
        )
        self.memory.label = "first_warmup" if self.warming else "first_ppo"
        with self.memory.phase("rollout"), phase(self.profile, "rollout"):
            rollout = self.engine.collect(self.update)
        if self.profile is not None:
            self.profile.transfer_timings = [dataclasses.asdict(t) for t in self.engine.transfer_timings]
            self.profile.shard_timings = {
                str(i): [dataclasses.asdict(t) for t in shard.timings] for i, shard in enumerate(native_shards)
            }
        with self.memory.phase("preparation"), phase(self.profile, "preparation"):
            batch = self.prepare(rollout)
        with self.memory.phase("optimization"):
            metrics = self.fit(batch)
        metrics.update(self.completed_metrics(rollout, batch.kl))
        metrics.update(self.memory.metrics())
        self.memory.first_checked = True
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
            assigned = {p.anchor for m in self.engine.matches.values() for p in m.episode.players}
            snapshots = sorted(key for key in self.engine.anchors if key)
            retired = snapshots[: max(0, len(snapshots) - self.config.snapshot_capacity + 1)]
            available = [key for key in retired if key not in assigned]
            if available:
                anchor = self.engine.anchors.pop(available[0])
                anchor.model.load_state_dict(self.policy.model.state_dict(), strict=True)
                anchor.clear_caches()
            else:
                anchor = self.policy.frozen_copy()
            self.engine.anchors[self.next_anchor] = anchor
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
        if self.policy.device.type == "cuda":
            torch.cuda.synchronize(self.policy.device)
            metrics["systems/peak_allocated_bytes"] = float(
                max(
                    torch.cuda.max_memory_allocated(self.policy.device),
                    max(
                        (
                            value[0]
                            for key, value in self.memory.peaks.items()
                            if self.memory.active and key.startswith(self.memory.label)
                        ),
                        default=0,
                    ),
                )
            )
            metrics["systems/peak_reserved_bytes"] = float(
                max(
                    torch.cuda.max_memory_reserved(self.policy.device),
                    max(
                        (
                            value[1]
                            for key, value in self.memory.peaks.items()
                            if self.memory.active and key.startswith(self.memory.label)
                        ),
                        default=0,
                    ),
                )
            )
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
    training = dataclasses.asdict(trainer.config)
    del training["deterministic"]
    identity = {
        "train": training,
        "sampler": dataclasses.asdict(engine.sampler.config),
        "schedule": dataclasses.asdict(engine.sampler.schedule),
        "seed": engine.sampler.rng.seed,
        "schema": schema,
        "execution_device": str(trainer.policy.device),
        "resume_contract": "execution_mode_independent_v1",
        "groups": [[s.match_ids for s in group] for group in engine.groups],
        "native_limits": [
            [s.max_frame if isinstance(s, NativeSimulatorShard) else None for s in group] for group in engine.groups
        ],
        "episode_reset": "shared_stocks_block_wait_v1",
        "rollout_storage": "dense_windows_v6",
        "selection_rng": "integer_counter_v3",
        "opponent_layout": "fixed_anchor_ranges_v1" if engine.fixed_anchor_lanes else "sampled_anchor_lanes_v1",
    }
    if isinstance(trainer.policy, TransformerPolicy):
        identity["decision_kernels"] = "dense_tick_window_graph_v5"
        identity["reference_kernels"] = "actor_order_readonly_forward_v3"
        identity["decoder_precision"] = trainer.policy.model.decoder_precision
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

    if torch.are_deterministic_algorithms_enabled() != trainer.config.deterministic:
        raise ValueError("checkpoint execution mode differs from the trainer configuration")
    if not trainer.engine._initialized or trainer.updating or trainer.engine.completed_update != trainer.update - 1:
        raise ValueError("checkpoint requires a complete update boundary")
    required = {"git_sha", "checkpoint_sha256", "data_manifest_sha256", "simulator_sha256", "adapter_sha256"}
    if set(provenance) != required or any(not value for value in provenance.values()):
        raise ValueError("checkpoint provenance must identify source, BC checkpoint, data, simulator, and adapter")
    directory.mkdir(parents=True, exist_ok=True)
    engine = trainer.engine
    state = {
        "version": CHECKPOINT_VERSION,
        "execution": {
            "deterministic": trainer.config.deterministic,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "model": trainer.policy.model.state_dict(),
        "actor_optimizer": trainer.actor_optimizer.state_dict(),
        "critic_optimizer": trainer.critic_optimizer.state_dict(),
        "scheduler": trainer.scheduler.state_dict(),
        "reference": trainer.reference.model.state_dict(),
        "anchors": {key: value.model.state_dict() for key, value in engine.anchors.items()},
        "anchor_candidates": engine.anchor_candidates,
        "pool": engine.snapshot(),
        "clocks": engine.clocks,
        "cursor": engine.cursor,
        "sampler_cursor": engine.sampler.cursor,
        "lambda_max": engine.sampler.lambda_max,
        "update": trainer.update,
        "actor_steps": trainer.actor_steps,
        "warming": trainer.warming,
        "warmup_moments": trainer.warmup_moments,
        "episode_metrics": trainer.episode_metrics,
        "memory_guard": (trainer.memory.first_checked, trainer.memory.actor_checked, trainer.memory.peaks),
        "fresh_variances": trainer.fresh_variances,
        "next_anchor": trainer.next_anchor,
        "evaluation_cursor": trainer.evaluation_cursor,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "counter_rng": {"seed": engine.sampler.rng.seed, "key": engine.sampler.rng.key},
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
    if (
        type(saved["version"]) is not int
        or type(saved["update"]) is not int
        or saved["version"] != CHECKPOINT_VERSION
        or saved["update"] != manifest["update"]
    ):
        raise ValueError("checkpoint shard identity mismatch")
    groups = len(trainer.engine.groups)
    frames = trainer.engine.sampler.schedule.rollout_frames
    expected_clocks = [((saved["update"] + groups - 1 - group) // groups) * frames for group in range(groups)]
    if (
        not isinstance(saved["clocks"], list)
        or any(type(clock) is not int for clock in saved["clocks"])
        or saved["clocks"] != expected_clocks
        or type(saved["cursor"]) is not int
        or saved["cursor"] != saved["update"] % groups
    ):
        raise ValueError("checkpoint pool progress mismatch")
    for field in ("actor_steps", "sampler_cursor", "evaluation_cursor", "next_anchor"):
        if type(saved[field]) is not int or saved[field] < 0:
            raise ValueError("checkpoint progress counters must be nonnegative integers")
    counter = saved.get("counter_rng")
    if (
        not isinstance(counter, dict)
        or set(counter) != {"seed", "key"}
        or any(type(counter[name]) is not int for name in ("seed", "key"))
        or counter != {"seed": trainer.engine.sampler.rng.seed, "key": trainer.engine.sampler.rng.key}
    ):
        raise ValueError("checkpoint counter RNG key mismatch")
    execution = saved.get("execution")
    if (
        not isinstance(execution, dict)
        or set(execution) != {"deterministic", "cublas_workspace_config"}
        or type(execution["deterministic"]) is not bool
        or (
            execution["cublas_workspace_config"] is not None
            and not isinstance(execution["cublas_workspace_config"], str)
        )
    ):
        raise ValueError("invalid checkpoint execution mode")
    if "actor" in paths:
        published = torch.load(paths["actor"], map_location="cpu", weights_only=False)
        if (
            published.get("format") != EVALUATION_FORMAT
            or published.get("decoder_precision") != trainer.policy.model.decoder_precision
        ):
            raise ValueError("checkpoint evaluation actor precision or format mismatch")
        actor = published["model"]
        if actor.keys() != saved["model"].keys() or any(
            not torch.equal(value, saved["model"][name]) for name, value in actor.items()
        ):
            raise ValueError("checkpoint training state and evaluation actor disagree")
    if trainer.policy.device.type == "cuda":
        torch.cuda.synchronize(trainer.policy.device)
    trainer.updating = True
    trainer.policy.model.load_state_dict(saved["model"], strict=True)
    if isinstance(trainer.policy, TransformerPolicy):
        trainer.policy._prepare_wire_tables()
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
    engine.restore_pool(saved["pool"])
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
    trainer.episode_metrics = saved["episode_metrics"]
    trainer.memory.first_checked, trainer.memory.actor_checked, trainer.memory.peaks = saved["memory_guard"]
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
    decoder_precision: DecoderPrecision = "bf16",
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
    model = SelfPlayTransformer(
        cfg, ranges, vocabulary, max_opponent_queue=14, critic_hidden_dim=128, decoder_precision=decoder_precision
    )
    model.load_bc_state_dict(state)
    return TransformerPolicy(model.to(device).eval(), schedule, statistics)


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
    trainer.memory.output = output
    resolved: dict[str, object] = {
        "configuration_sha256": _configuration_digest(trainer),
        "environment": _environment(),
        "provenance": provenance,
        "train": dataclasses.asdict(trainer.config),
        "sampler": dataclasses.asdict(trainer.engine.sampler.config),
        "schedule": dataclasses.asdict(trainer.engine.sampler.schedule),
        "treatment": "KL-regularized joint PPO; independent per-player clipped Beta(5,1.3) u",
        "actor_step_change": {
            "old_windows_per_step": 32,
            "windows_per_step": trainer.config.windows_per_step,
            "actor_epochs": 1,
        },
        "control": "frozen pi_ref with matched sampling, starts, episode u, and schedules",
        "invariants": "BC checkpoint, representation, data, simulator, adapter, reward, evaluation seeds",
        "run": run_configuration,
    }
    if isinstance(trainer.policy, TransformerPolicy):
        resolved["decoder_precision"] = trainer.policy.model.decoder_precision
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
    decoder_precision: DecoderPrecision = "bf16"


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
    max_frame: int = 28800
    resume: Path | None = None
    training: TrainConfig = TrainConfig()
    evaluation_every: int = 128
    evaluation_protocol: Path | None = None
    evaluation_simulators: tuple[Literal["msl", "dolphin"], ...] = ("msl", "dolphin")
    evaluation_device: str = "cpu"
    verify_resume: bool = False
    verify_only: bool = False
    fixed_anchor_lanes: bool = True
    wandb_project: str | None = None
    profiling: ProfileConfig | None = None
    decoder_precision: DecoderPrecision = "bf16"
    compile_cache: Path | None = None

    def __post_init__(self) -> None:
        validate_decoder_precision(self.decoder_precision)
        if self.verify_only and (not self.verify_resume or self.resume is None):
            raise ValueError("verify-only needs a resume checkpoint and verify-resume")
        if self.profiling is not None and (
            self.device != "cuda"
            or self.training.critic_warmup_updates
            or self.resume is not None
            or self.evaluation_every
            or self.wandb_project is not None
            or self.verify_resume
        ):
            raise ValueError(
                "profiling requires CUDA, warmup off, fresh state, no evaluation, no uploads, and no resume check"
            )
        if type(self.fixed_anchor_lanes) is not bool:
            raise ValueError("fixed_anchor_lanes must be a boolean")
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
        if self.profiling is not None and (
            self.profiling.warm_updates < self.groups or self.updates != self.profiling.warm_updates + 2
        ):
            raise ValueError("profiling must warm every group and run exactly two measured updates")


def read_native_sampler(path: Path) -> SamplerConfig:
    raw = json.loads(path.read_text())
    if (
        not isinstance(raw, dict)
        or set(raw)
        != {
            "format",
            "identities",
            "matchups",
            "stages",
            "stocks",
            "percents",
            "data_floor",
            "tempering",
            "masked_rate",
            "deployment_rate",
        }
        or raw["format"] != "hal.o65.sampler.v2"
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
        tuple(identities),
        tuple(matchups),
        tuple(raw["stages"]),
        stocks=stocks,
        percents=tuple(raw["percents"]),
        data_floor=raw["data_floor"],
        tempering=raw["tempering"],
        masked_rate=raw["masked_rate"],
        deployment_rate=raw["deployment_rate"],
    )


def train_native(config: NativeCli) -> None:
    startup_started = time.perf_counter()
    torch.set_num_threads(2)
    if config.training.deterministic:
        workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if workspace not in (None, ":4096:8"):
            raise ValueError("deterministic native runs require CUBLAS_WORKSPACE_CONFIG=:4096:8")
        if torch.cuda.is_initialized() and workspace is None:
            raise ValueError("configure deterministic execution before CUDA initialization")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(config.training.deterministic)
    cache_started = time.perf_counter()
    source_root = Path(__file__).resolve().parents[1]
    cache_sources = (
        *source_root.joinpath("hal").rglob("*.py"),
        Path(__file__).resolve(),
        source_root / "pyproject.toml",
        source_root / "uv.lock",
    )
    cache = (
        compile_cache.CompileCache(
            config.compile_cache,
            compile_cache.current_identity(
                compile_cache.source_hash(source_root, cache_sources), torch.device(config.device)
            ),
        )
        if config.compile_cache is not None
        else None
    )
    cache_load_seconds = time.perf_counter() - cache_started
    if sha256(config.checkpoint) != PROXY_SHA256:
        raise ValueError("native development run requires the recorded O63 proxy")
    torch.manual_seed(config.training.seed)
    checkpoint = load_evaluation_checkpoint(
        config.checkpoint, device=torch.device(config.device), decoder_precision=config.decoder_precision
    )
    checkpoint.model.requires_grad_(True)
    checkpoint.model.percentile_head.requires_grad_(False)
    schedule = development_config(checkpoint.model.head_offsets)
    policy = TransformerPolicy(checkpoint.model, schedule, checkpoint.statistics)
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
        if cache is not None:
            resources.callback(cache.save)
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
            fixed_anchor_lanes=config.fixed_anchor_lanes,
        )
        trainer = Trainer(engine, config.training)
        if config.resume is None:
            engine.initialize()
        else:
            restore_checkpoint(trainer, config.resume, provenance)
        if config.verify_only:
            verify_checkpoint_resume(trainer, config.output, provenance)
            return
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
                "--decoder-precision",
                config.decoder_precision,
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
        native_config["startup_seconds"] = time.perf_counter() - startup_started
        native_config["compile_cache_status"] = {
            "directory": str(cache.directory) if cache is not None else None,
            "loaded_artifacts": cache.loaded if cache is not None else False,
            "loaded_counts": cache.loaded_counts if cache is not None else {},
            "load_seconds": cache_load_seconds,
        }
        if config.profiling is not None:
            try:
                profile_native(trainer, config, provenance, native_config)
            except torch.cuda.OutOfMemoryError:
                (config.output / "oom-allocator.json").write_text(json.dumps(torch.cuda.memory_snapshot()))
                (config.output / "oom-memory.txt").write_text(torch.cuda.memory_summary())
                raise
            return
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


class TraceEvent(TypedDict, total=False):
    cat: str
    name: str
    ts: float
    dur: float
    args: dict[str, str | int | float]


@dataclasses.dataclass(frozen=True, slots=True)
class ProfileConfig:
    warm_updates: int = 4
    nsys: bool = False

    def __post_init__(self) -> None:
        if type(self.warm_updates) is not int or self.warm_updates < 1:
            raise ValueError("profiling requires at least one warm update")
        if type(self.nsys) is not bool:
            raise ValueError("nsys must be a boolean")


def interval_union(intervals: list[tuple[float, float]]) -> float:
    end = total = 0.0
    for start, stop in sorted(intervals):
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def summarize_trace(path: Path) -> dict[str, object]:
    """Count device activities, not CPU operators that launch those activities."""
    events = cast(list[TraceEvent], json.loads(path.read_text())["traceEvents"])
    roots = [
        (event["ts"], event["ts"] + event["dur"])
        for event in events
        if event.get("cat") == "user_annotation" and event.get("name", "").startswith("o65_profile/")
    ]
    markers = [
        (event["ts"], event["ts"] + event["dur"])
        for event in events
        if event.get("cat") == "user_annotation" and event.get("name") == "o65_profile_sync"
    ]
    if len(roots) != 1:
        raise ValueError("trace requires exactly one O65 profile annotation")
    root_start, root_stop = roots[0]
    kernels: dict[str, float] = defaultdict(float)
    copies: dict[str, int] = defaultdict(int)
    syncs: dict[str, int] = defaultdict(int)
    measurement_syncs: dict[str, int] = defaultdict(int)
    intervals: list[tuple[float, float]] = []
    kernel_count = launches = graph_launches = 0
    launch_seconds = 0.0
    for event in events:
        category = event.get("cat", "")
        name = event.get("name", "")
        duration = event.get("dur", 0.0)
        timestamp = event.get("ts", -1.0)
        in_root = root_start <= timestamp <= root_stop
        if category == "cuda_runtime" and "Synchronize" in name:
            measurement = not in_root or any(start <= timestamp <= stop for start, stop in markers)
            if measurement:
                measurement_syncs[name] += 1
            else:
                syncs[name] += 1
        if not in_root:
            continue
        if category in ("kernel", "gpu_memcpy", "gpu_memset"):
            start = event["ts"]
            intervals.append((start, start + duration))
        if category == "kernel":
            kernels[name] += duration
            kernel_count += 1
        if category == "cuda_runtime" and ("LaunchKernel" in name or "GraphLaunch" in name):
            launches += 1
            graph_launches += int("GraphLaunch" in name)
            launch_seconds += duration / 1e6
        if category == "gpu_memcpy":
            if "HtoD" in name:
                direction = "h2d"
            elif "DtoH" in name:
                direction = "d2h"
            else:
                direction = "other"
            args = event.get("args", {})
            if "bytes" not in args:
                raise ValueError(f"copy event has no byte count: {name}")
            copies[f"{direction}_count"] += 1
            copies[f"{direction}_bytes"] += int(args["bytes"])
    if not kernel_count:
        raise ValueError("profile captured no CUDA kernels; check CUPTI availability")
    total = sum(kernels.values())
    top = sorted(kernels.items(), key=lambda item: item[1], reverse=True)[:15]
    return {
        "device_busy_seconds": interval_union(intervals) / 1e6,
        "kernel_seconds": total / 1e6,
        "kernel_count": kernel_count,
        "runtime_launch_count": launches,
        "runtime_eager_launch_count": launches - graph_launches,
        "runtime_graph_launch_count": graph_launches,
        "runtime_launch_seconds": launch_seconds,
        "kernel_seconds_by_name": {name: value / 1e6 for name, value in kernels.items()},
        "top15_kernel_share": sum(value for _, value in top) / total if total else 0.0,
        "top15_kernels": [{"name": name, "seconds": value / 1e6} for name, value in top],
        "copies": dict(copies),
        "synchronizations": dict(syncs),
        "measurement_synchronizations_excluded": dict(measurement_syncs),
    }


class UpdateProfile:
    """Measure one update, then trace bounded samples in a separate update.

    CUDA event spans contain idle gaps. Only trace interval unions measure busy
    time. Extrapolated samples remain estimates and are stored separately.
    """

    def __init__(self, output: Path, device: torch.device, *, traces: bool) -> None:
        self.output = output
        self.device = device
        self.traces = traces
        self.phase_name = "outside"
        self.active_trace = False
        self.phases: dict[str, dict[str, float]] = {}
        self.calls: dict[str, int] = defaultdict(int)
        self.work: dict[str, float] = defaultdict(float)
        self.samples: list[dict[str, object]] = []
        self.host_seconds: dict[str, float] = defaultdict(float)
        self.shard_timings: dict[str, list[dict[str, float]]] = {}
        self.transfer_timings: list[dict[str, float]] = []

    @contextlib.contextmanager
    def phase(self, name: str) -> Iterator[None]:
        previous = self.phase_name
        self.phase_name = name
        torch.cuda.synchronize(self.device)
        start_event = torch.cuda.Event(enable_timing=True)
        stop_event = torch.cuda.Event(enable_timing=True)
        started = time.perf_counter()
        cpu_started = time.thread_time()
        started_unix = time.time()
        start_event.record()
        torch.cuda.nvtx.range_push(f"o65/{name}")
        try:
            yield
        finally:
            stop_event.record()
            stop_event.synchronize()
            torch.cuda.nvtx.range_pop()
            self.phases[name] = {
                "started_unix": started_unix,
                "finished_unix": time.time(),
                "wall_seconds": time.perf_counter() - started,
                "main_thread_cpu_seconds": time.thread_time() - cpu_started,
                "cuda_span_seconds": start_event.elapsed_time(stop_event) / 1000,
            }
            self.phase_name = previous

    @contextlib.contextmanager
    def sample(self, name: str, *, work: float = 0.0, limit: int = 1, skip: int = 0) -> Iterator[None]:
        started = time.perf_counter()
        cpu_started = time.thread_time()
        key = f"{self.phase_name}/{name}"
        try:
            with self._sample(name, work=work, limit=limit, skip=skip):
                yield
        finally:
            self.host_seconds[key] += time.perf_counter() - started
            self.host_seconds[f"{key}/main_thread_cpu"] += time.thread_time() - cpu_started

    @contextlib.contextmanager
    def _sample(self, name: str, *, work: float, limit: int, skip: int) -> Iterator[None]:
        key = f"{self.phase_name}/{name}"
        index = self.calls[key]
        self.calls[key] += 1
        self.work[key] += work
        if not self.traces or self.active_trace or not skip <= index < skip + limit:
            yield
            return
        self.active_trace = True
        try:
            torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            with (
                torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                ) as profiler,
                torch.profiler.record_function(f"o65_profile/{key}"),
            ):
                with torch.profiler.record_function("o65_profile_sync"):
                    torch.cuda.synchronize(self.device)
                yield
                with torch.profiler.record_function("o65_profile_sync"):
                    torch.cuda.synchronize(self.device)
            path = self.output / f"{key.replace('/', '-')}-{index}.json"
            profiler.export_chrome_trace(str(path))
            self.samples.append(
                {
                    "site": key,
                    "index": index,
                    "work": work,
                    "instrumented_seconds_including_export": time.perf_counter() - started,
                    "trace": path.name,
                    **summarize_trace(path),
                }
            )
        finally:
            self.active_trace = False

    def write(self, metrics: dict[str, float]) -> None:
        name = "traced" if self.traces else "timed"
        payload = {
            "format": "hal.o65.gpu-profile.v1",
            "mode": name,
            "phases": self.phases,
            "calls": dict(self.calls),
            "work": dict(self.work),
            "site_host_seconds": dict(self.host_seconds),
            "samples": self.samples,
            "metrics": metrics,
            "shard_timings": self.shard_timings,
            "transfer_timings": self.transfer_timings,
        }
        (self.output / f"{name}.json").write_text(json.dumps(payload, indent=2, sort_keys=True))


def estimate_profile(timed: UpdateProfile, traced: UpdateProfile) -> dict[str, object]:
    """Scale bounded samples by timed work; preserve the coverage limitation."""
    estimates: dict[str, dict[str, object]] = {}
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in traced.samples:
        grouped[cast(str, row["site"])].append(row)
    for site, rows in grouped.items():
        sample_work = sum(cast(float, row["work"]) for row in rows)
        if sample_work <= 0 or site not in timed.work:
            raise ValueError(f"missing timed work for trace site {site}")
        scale = timed.work[site] / sample_work
        call_scale = timed.calls[site] / len(rows)
        kernels: dict[str, float] = defaultdict(float)
        copies: dict[str, float] = defaultdict(float)
        syncs: dict[str, float] = defaultdict(float)
        for row in rows:
            for name, duration in cast(dict[str, float], row["kernel_seconds_by_name"]).items():
                kernels[name] += duration * scale
            for name, count in cast(dict[str, int], row["copies"]).items():
                copies[name] += count * (call_scale if name.endswith("_count") else scale)
            for name, count in cast(dict[str, int], row["synchronizations"]).items():
                syncs[name] += count * call_scale
        total = sum(kernels.values())
        top = sorted(kernels.items(), key=lambda item: item[1], reverse=True)[:15]
        estimates[site] = {
            "scale": scale,
            "call_scale": call_scale,
            "timed_calls": timed.calls[site],
            "sampled_calls": len(rows),
            "device_busy_seconds": sum(cast(float, row["device_busy_seconds"]) for row in rows) * scale,
            "kernel_seconds": total,
            "kernel_count": sum(cast(int, row["kernel_count"]) for row in rows) * call_scale,
            "runtime_launch_count": sum(cast(int, row["runtime_launch_count"]) for row in rows) * call_scale,
            "runtime_eager_launch_count": sum(
                cast(int, row.get("runtime_eager_launch_count", row["runtime_launch_count"])) for row in rows
            )
            * call_scale,
            "runtime_graph_launch_count": sum(cast(int, row.get("runtime_graph_launch_count", 0)) for row in rows)
            * call_scale,
            "runtime_launch_seconds": sum(cast(float, row["runtime_launch_seconds"]) for row in rows) * call_scale,
            "copies": dict(copies),
            "synchronizations_excluding_injected": dict(syncs),
            "top15_kernel_share": sum(value for _, value in top) / total if total else 0,
            "top15_kernels": [{"name": name, "seconds": value} for name, value in top],
        }
    flops: dict[str, float] = {}
    for name in timed.phases:
        forward = 2 * (
            9_400_000 * timed.work.get(f"{name}/trunk_tokens", 0)
            + 6_700_000 * timed.work.get(f"{name}/decoder_tokens", 0)
        )
        flops[name] = forward * (3 if name == "actor" else 1)
    return {
        "format": "hal.o65.gpu-profile-estimates.v1",
        "sites": estimates,
        "coverage": "sampled sites only; gaps between chunks and rollout setup are untraced",
        "flops": flops,
        "tflops_per_wall_second": {
            name: value / timed.phases[name]["wall_seconds"] / 1e12 for name, value in flops.items()
        },
    }


@contextlib.contextmanager
def phase(profile: UpdateProfile | None, name: str) -> Iterator[None]:
    with profile.phase(name) if profile is not None else contextlib.nullcontext():
        yield


@contextlib.contextmanager
def sample(
    profile: UpdateProfile | None,
    name: str,
    *,
    work: float = 0.0,
    limit: int = 1,
    skip: int = 0,
) -> Iterator[None]:
    with (
        torch.cuda.nvtx.range(f"o65/{profile.phase_name}/{name}")
        if profile is not None and profile.device.type == "cuda"
        else contextlib.nullcontext(),
        profile.sample(name, work=work, limit=limit, skip=skip) if profile is not None else contextlib.nullcontext(),
    ):
        yield


class HostSampler:
    """Sample per-core and per-thread CPU ticks and GPU use during timing only."""

    def __init__(self, output: Path) -> None:
        self.output = output
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="o65-host-sampler")
        self.failure: OSError | subprocess.SubprocessError | None = None

    def _run(self) -> None:
        try:
            with (self.output / "host.jsonl").open("x") as log:
                while not self.stop.is_set():
                    threads: dict[str, list[int]] = {}
                    thread_affinity: dict[str, list[int]] = {}
                    for path in Path(f"/proc/{os.getpid()}/task").glob("*/stat"):
                        try:
                            fields = path.read_text().rsplit(")", 1)[1].split()
                            affinity = sorted(os.sched_getaffinity(int(path.parent.name)))
                        except FileNotFoundError, ProcessLookupError:
                            continue
                        threads[path.parent.name] = [int(fields[11]), int(fields[12])]
                        thread_affinity[path.parent.name] = affinity
                    cores = [line for line in Path("/proc/stat").read_text().splitlines() if line.startswith("cpu")]
                    gpu = subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=utilization.gpu,utilization.memory,power.draw,memory.used",
                            "--format=csv,noheader,nounits",
                        ],
                        text=True,
                        timeout=10,
                    ).strip()
                    log.write(
                        json.dumps(
                            {
                                "time": time.time(),
                                "pid": os.getpid(),
                                "threads": threads,
                                "thread_affinity": thread_affinity,
                                "cores": cores,
                                "gpu": gpu,
                            }
                        )
                        + "\n"
                    )
                    log.flush()
                    self.stop.wait(0.5)
        except (OSError, subprocess.SubprocessError) as error:
            self.failure = error

    def __enter__(self) -> Self:
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop.set()
        self.thread.join()
        if self.failure is not None:
            raise RuntimeError("host sampling failed") from self.failure


def validate_profile_sites(sites: set[str], *, bootstrap: bool) -> None:
    required = {"rollout/tick", "rollout/boundary_tick", "reference/windows", "critic/fit", "actor/step"}
    if bootstrap:
        required.add("preparation/bootstrap")
    missing = required - sites
    if missing:
        raise ValueError(f"profile is missing required sites: {sorted(missing)}")


def profile_nsys(trainer: Trainer, output: Path) -> None:
    """Keep CUPTI ownership in nsys; trace one complete warmed update only."""
    for traced in (False, True):
        recorder = UpdateProfile(output, trainer.policy.device, traces=False)
        trainer.profile = trainer.engine.profile = recorder
        try:
            for policy in (trainer.policy, *trainer.engine.anchors.values()):
                if not isinstance(policy, TransformerPolicy):
                    raise TypeError("profiling requires TransformerPolicy")
                policy.profile = recorder
            if traced:
                torch.cuda.synchronize()
                torch.cuda.profiler.start()
            try:
                batch, metrics = trainer.run_update()
            finally:
                if traced:
                    torch.cuda.profiler.stop()
            name = "nsys" if traced else "timed"
            (output / f"{name}.json").write_text(
                json.dumps(
                    {
                        "phases": recorder.phases,
                        "metrics": metrics,
                        "work": recorder.work,
                        "shard_timings": recorder.shard_timings,
                        "transfer_timings": recorder.transfer_timings,
                    },
                    indent=2,
                )
            )
            if not traced:
                windows = batch.windows.take(batch.steps[0], batch.placement.positions, batch.steps[0] >= 0)
                policy = trainer.policy
                assert isinstance(policy, TransformerPolicy)
                features, observed = policy._packed_context(windows.inputs()[:5])
                torch.save(
                    {
                        "format": "hal.o65.floor-inputs.v1",
                        "features": {key: value.cpu() for key, value in features.items()},
                        "observed": observed.cpu(),
                        "slots": {
                            field.name: getattr(windows.slots, field.name).cpu()
                            for field in dataclasses.fields(windows.slots)
                        },
                        "model": {key: value.cpu() for key, value in policy.model.state_dict().items()},
                    },
                    output / "floor-inputs.pt",
                )
                layouts = [
                    {"buffer": buffer, "anchor": anchor, "batch": len(layout.streams)}
                    for (buffer, anchor), layout in trainer.engine.tick_layouts.items()
                ]
                (output / "floor-layout.json").write_text(
                    json.dumps(
                        {
                            "format": "hal.o65.floor-layout.v1",
                            "tick_layouts": layouts,
                            "actor_steps": len(batch.steps),
                            "actor_windows": trainer.config.windows_per_step,
                            "actor_slots": trainer.config.decisions_per_window,
                            "reference_windows": int(recorder.work["reference/trunk_tokens"] // 256),
                            "reference_slots": 64,
                            "frames": batch.rollout.frames,
                            "decisions": len(batch.rollout.ids),
                            "matches": len(trainer.engine.group_matches[batch.rollout.group]),
                            "critic_parameters": [
                                list(parameter.shape) for parameter in trainer.policy.critic_parameters()
                            ],
                            "optimizer_groups": [
                                {
                                    "use_muon": group["use_muon"],
                                    "logical_splits": group.get("logical_splits", 1),
                                    "shapes": [list(parameter.shape) for parameter in group["params"]],
                                }
                                for group in trainer.actor_optimizer.param_groups
                            ],
                        },
                        indent=2,
                    )
                )
                (output / "allocator.json").write_text(json.dumps(torch.cuda.memory_snapshot()))
            del batch
        finally:
            trainer.profile = trainer.engine.profile = None
            for policy in (trainer.policy, *trainer.engine.anchors.values()):
                if isinstance(policy, TransformerPolicy):
                    policy.profile = None


def compilation_report(engine: RolloutEngine) -> dict[str, object]:
    ticks: list[dict[str, object]] = []
    captures: list[dict[str, object]] = []
    variants: set[tuple[int, int, int]] = set()
    for name, policy in (
        ("student", engine.policy),
        *((f"anchor{key}", value) for key, value in engine.anchors.items()),
    ):
        if not isinstance(policy, TransformerPolicy):
            raise TypeError("compilation report requires TransformerPolicy")
        for ((_, (buffer, batch)), append, replan), graph in policy.tick_graphs.items():
            variants.add((batch, replan, append))
            ticks.append(
                {
                    "policy": name,
                    "buffer": buffer,
                    "batch": batch,
                    "R": replan,
                    "append": append,
                    "compile_and_warmup_seconds": graph.preparation_seconds,
                    "capture_seconds": graph.capture_seconds,
                }
            )
        graphs: list[
            tuple[
                str, CapturedCall[ReplayOutput] | CapturedCall[Tensor] | CapturedCall[tuple[ActorLoss, ReplayOutput]]
            ]
        ] = [
            *(("reference", graph) for graph in policy.reference_graphs.values()),
            *(("value", graph) for graph in policy.value_graphs.values()),
        ]
        if policy.actor_graph is not None:
            graphs.append(("actor", policy.actor_graph.captured))
        for kind, graph in graphs:
            captures.append(
                {
                    "policy": name,
                    "kind": kind,
                    "compile_and_warmup_seconds": graph.preparation_seconds,
                    "capture_seconds": graph.capture_seconds,
                }
            )
    return {
        "format": "hal.o65.compilation-report.v1",
        "tick_variants": ticks,
        "distinct_tick_variants": len(variants),
        "other_captures": captures,
        "note": "preparation includes compilation, allocation and two warmup calls; graph capture is separate",
    }


def profile_native(
    trainer: Trainer,
    config: NativeCli,
    provenance: dict[str, str],
    native_config: dict[str, object],
) -> None:
    if config.profiling is None:
        raise ValueError("missing profile configuration")
    config.output.mkdir(parents=True, exist_ok=False)
    (config.output / "resolved.json").write_text(
        json.dumps(
            {
                "format": "hal.o65.gpu-profile-inputs.v1",
                "provenance": provenance,
                "trainer_source_sha256": sha256(Path(__file__)),
                "configuration": native_config,
                "environment": _environment(),
                "treatment": "bounded profiler instrumentation",
                "control": "same configuration without tracing",
                "invariants": "checkpoint, sampler, seed, representation, reward, optimizer, simulator and adapter",
                "gpu": torch.cuda.get_device_name(),
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "clock_ticks_per_second": os.sysconf("SC_CLK_TCK"),
                "flop_model": {
                    "multiply_add_flops": 2,
                    "trunk_parameter_equivalent": 9_400_000,
                    "decoder_parameter_equivalent": 6_700_000,
                    "actor_forward_backward_factor": 3,
                    "critic_parameters": sum(p.numel() for p in trainer.policy.critic_parameters()),
                    "formula": "2*9.4e6*trunk_tokens + 2*6.7e6*decoder_tokens; actor times 3; critic 6*parameters*rows*epochs",
                    "excludes": "attention score/value products, nonlinearities, feature building, optimizer, copies",
                    "decoder_precision": config.decoder_precision,
                    "trunk_precision": "BF16",
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    warmup_started = time.perf_counter()
    with (config.output / "warm.jsonl").open("x") as log:
        for _ in range(config.profiling.warm_updates):
            batch, metrics = trainer.run_update()
            del batch
            log.write(json.dumps(metrics) + "\n")
            log.flush()
    compilation = compilation_report(trainer.engine)
    compilation["warmup_seconds"] = time.perf_counter() - warmup_started
    (config.output / "compilation.json").write_text(json.dumps(compilation, indent=2))
    if config.profiling.nsys:
        profile_nsys(trainer, config.output)
        return
    recorders: list[UpdateProfile] = []
    try:
        for traces in (False, True):
            recorder = UpdateProfile(config.output, trainer.policy.device, traces=traces)
            recorders.append(recorder)
            trainer.profile = trainer.engine.profile = recorder
            for policy in (trainer.policy, *trainer.engine.anchors.values()):
                if not isinstance(policy, TransformerPolicy):
                    raise TypeError("profiling requires TransformerPolicy")
                policy.profile = recorder
            sampler = contextlib.nullcontext() if traces else HostSampler(config.output)
            with sampler:
                batch, metrics = trainer.run_update()
            del batch
            recorder.write(metrics)
        estimates = estimate_profile(*recorders)
        validate_profile_sites(
            set(cast(dict[str, object], estimates["sites"])),
            bootstrap=bool(recorders[0].calls.get("preparation/bootstrap", 0)),
        )
        critic_flops = (
            6
            * sum(p.numel() for p in trainer.policy.critic_parameters())
            * recorders[0].work["critic/fit"]
            * trainer.config.critic_epochs
        )
        cast(dict[str, float], estimates["flops"])["critic"] = critic_flops
        cast(dict[str, float], estimates["tflops_per_wall_second"])["critic"] = (
            critic_flops / recorders[0].phases["critic"]["wall_seconds"] / 1e12
        )
        (config.output / "estimates.json").write_text(json.dumps(estimates, indent=2, sort_keys=True))
    finally:
        trainer.profile = trainer.engine.profile = None
        for policy in (trainer.policy, *trainer.engine.anchors.values()):
            if isinstance(policy, TransformerPolicy):
                policy.profile = None


@dataclasses.dataclass(frozen=True, slots=True)
class GpuProfileCli:
    checkpoint: Path
    data_dir: Path
    output: Path
    sampler: Path = Path("experiments/o65/pilot-sampler.json")
    cpus: tuple[int, ...] = tuple(range(16))
    matches_per_shard: int = 64
    groups: int = 4
    warm_updates: int = 4
    decoder_precision: DecoderPrecision = "bf16"
    nsys: bool = False
    compile_cache: Path | None = None

    def native(self) -> NativeCli:
        return NativeCli(
            self.checkpoint,
            self.sampler,
            self.output,
            self.data_dir,
            self.cpus,
            updates=self.warm_updates + 2,
            groups=self.groups,
            matches_per_shard=self.matches_per_shard,
            training=TrainConfig(critic_warmup_updates=0, windows_per_step=32),
            evaluation_every=0,
            profiling=ProfileConfig(self.warm_updates, self.nsys),
            compile_cache=self.compile_cache,
            decoder_precision=self.decoder_precision,
        )


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
        elif dataclasses.is_dataclass(item) and not isinstance(item, type):
            visit(type(item).__name__)
            visit(tuple((field.name, getattr(item, field.name)) for field in dataclasses.fields(item)))
        elif isinstance(item, bytes):
            digest.update(item)
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
    table = batch.rollout.table
    store = table.store
    frames = torch.arange(len(store.records), device=table.device)[:, None] < store.lengths[None]
    states = table.valid | table.bootstrap
    return _state_hash(
        {
            "decisions": {
                field.name: getattr(batch.decisions, field.name) for field in dataclasses.fields(batch.decisions)
            },
            "selected": batch.selected,
            "steps": batch.steps,
            "placement_positions": batch.placement.positions,
            "values": batch.values,
            "advantages": batch.advantages,
            "targets": batch.targets,
            "behavior": batch.old.log_probs,
            "reference": None if batch.reference is None else batch.reference.log_probs,
            "state_mask": states,
            "metadata": table.integers[states],
            "conditions": table.conditions[states],
            "prefix": table.prefix[states],
            "opponent": table.opponent[states],
            "frames": (
                store.lengths,
                store.records[frames],
                store.actions[frames],
                store.rewards[frames],
                store.flags[frames],
            ),
            "episodes": tuple(dataclasses.asdict(result) for result in batch.rollout.results),
        }
    )


def _rng_hash() -> str:
    return _state_hash(
        (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else (),
        )
    )


def _resume_state_hashes(trainer: Trainer) -> dict[str, str]:
    engine = trainer.engine
    return {
        "model": _state_hash(trainer.policy.model.state_dict()),
        "actor_optimizer": _state_hash(trainer.actor_optimizer.state_dict()),
        "critic_optimizer": _state_hash(trainer.critic_optimizer.state_dict()),
        "scheduler": _state_hash(trainer.scheduler.state_dict()),
        "pool": _state_hash(engine.snapshot()),
        "simulators": _state_hash(tuple(shard.save() for group in engine.groups for shard in group)),
        "sampler": _state_hash((engine.sampler.cursor, engine.sampler.lambda_max, engine.sampler.rng)),
        "rng": _rng_hash(),
        "anchors": _state_hash({key: policy.model.state_dict() for key, policy in engine.anchors.items()}),
        "progress": _state_hash(
            (
                engine.clocks,
                engine.cursor,
                engine.anchor_candidates,
                trainer.update,
                trainer.actor_steps,
                trainer.warming,
                trainer.warmup_moments,
                trainer.fresh_variances,
                trainer.next_anchor,
                trainer.evaluation_cursor,
                trainer.episode_metrics,
            )
        ),
    }


def _update_tensors(trainer: Trainer) -> dict[str, Tensor]:
    tensors: dict[str, Tensor] = {}

    def visit(path: str, value: object) -> None:
        if isinstance(value, Tensor):
            _check_finite(path, value)
            tensors[path] = value.detach().to(device="cpu", copy=True)
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(f"{path}/{key}", child)
        elif isinstance(value, tuple | list):
            for index, child in enumerate(value):
                visit(f"{path}/{index}", child)

    visit("model", trainer.policy.model.state_dict())
    visit("actor_optimizer", trainer.actor_optimizer.state_dict())
    visit("critic_optimizer", trainer.critic_optimizer.state_dict())
    return tensors


@dataclasses.dataclass(frozen=True, slots=True)
class ResumeComparison:
    tensors: dict[str, dict[str, float | bool]]
    groups: dict[str, dict[str, float | bool]]

    @property
    def passed(self) -> bool:
        return all(row["passed"] for row in self.groups.values())


def _compare_resume_updates(
    first: dict[str, Tensor], second: dict[str, Tensor], resumed: dict[str, Tensor]
) -> ResumeComparison:
    """Compare model and optimizer-state vectors with two-control noise.

    RMS error may reach twice the control spread plus eight dtype rounding
    units. Maximum and per-tensor errors remain diagnostic: reduction noise
    can move between tensors and produce isolated outliers. Integer state stays exact.
    Optimizer state families have separate scales (e.g. first/second moments).
    """
    if first.keys() != second.keys() or first.keys() != resumed.keys():
        raise ValueError("next update tensor identities differ")
    tensors: dict[str, dict[str, float | bool]] = {}
    accumulated: dict[str, dict[str, float]] = {}
    for name, a in first.items():
        b, c = second[name], resumed[name]
        if a.shape != b.shape or a.shape != c.shape or a.dtype != b.dtype or a.dtype != c.dtype:
            raise ValueError(f"next update tensor schema differs: {name}")
        for value in (a, b, c):
            _check_finite(name, value)
        if not a.numel():
            continue
        root = name.split("/", 1)[0]
        group = f"{root}/{name.rsplit('/', 1)[-1]}" if root.endswith("_optimizer") else root
        totals = accumulated.setdefault(
            group,
            dict.fromkeys(
                (
                    "count",
                    "control_square",
                    "first_square",
                    "second_square",
                    "floor_square",
                    "control_max",
                    "error_max",
                    "floor_max",
                    "integer_mismatch",
                ),
                0.0,
            ),
        )
        if not a.is_floating_point():
            exact = torch.equal(a, b) and torch.equal(a, c)
            totals["integer_mismatch"] += float(not exact)
            tensors[name] = {"exact": exact}
            continue
        a, b, c = a.double(), b.double(), c.double()
        control = a - b
        first_error, second_error = a - c, b - c
        spread = float(control.abs().max())
        error = max(float(first_error.abs().max()), float(second_error.abs().max()))
        floor = 8 * torch.finfo(first[name].dtype).eps * max(float(a.abs().max()), float(b.abs().max()), 1e-3)
        control_square = float(control.square().sum())
        first_square, second_square = float(first_error.square().sum()), float(second_error.square().sum())
        if not all(
            math.isfinite(value) for value in (spread, error, floor, control_square, first_square, second_square)
        ):
            raise ValueError("resume comparison errors must be finite")
        tensors[name] = {
            "control_spread": spread,
            "error": error,
            "control_rms": math.sqrt(control_square / a.numel()),
            "error_rms": math.sqrt(max(first_square, second_square) / a.numel()),
        }
        totals["count"] += a.numel()
        totals["control_square"] += control_square
        totals["first_square"] += first_square
        totals["second_square"] += second_square
        totals["floor_square"] += a.numel() * floor**2
        totals["control_max"] = max(totals["control_max"], spread)
        totals["error_max"] = max(totals["error_max"], error)
        totals["floor_max"] = max(totals["floor_max"], floor)
    groups: dict[str, dict[str, float | bool]] = {}
    for name, totals in accumulated.items():
        if not all(math.isfinite(value) for value in totals.values()):
            raise ValueError("resume comparison totals must be finite")
        count = max(1.0, totals["count"])
        control_rms = math.sqrt(totals["control_square"] / count)
        error_rms = math.sqrt(max(totals["first_square"], totals["second_square"]) / count)
        tolerance_rms = 2 * control_rms + math.sqrt(totals["floor_square"] / count)
        tolerance_max = 2 * totals["control_max"] + totals["floor_max"]
        integer_equal = totals["integer_mismatch"] == 0
        groups[name] = {
            "control_rms": control_rms,
            "error_rms": error_rms,
            "tolerance_rms": tolerance_rms,
            "control_max": totals["control_max"],
            "error_max": totals["error_max"],
            "tolerance_max": tolerance_max,
            "integer_equal": integer_equal,
            "passed": integer_equal and error_rms <= tolerance_rms,
        }
    return ResumeComparison(tensors, groups)


def verify_checkpoint_resume(trainer: Trainer, output: Path, provenance: dict[str, str]) -> None:
    """Check exact restoration and calibrate next-update error on two controls.

    Each control repeats the uninterrupted next update from the same boundary.
    Restoring between controls rewinds their inputs. A third update tests resume
    with the same training graphs and execution mode. Batch hashes are evidence;
    floating-point noise can change categorical choices, so they need not match.
    """
    manifest = save_checkpoint(trainer, output / "checkpoints", provenance)
    boundary = _resume_state_hashes(trainer)
    started = time.perf_counter()
    controls: list[dict[str, Tensor]] = []
    batches: list[str] = []
    metrics: list[dict[str, float]] = []
    restored: list[dict[str, str]] = []
    schedules: list[str] = []
    try:
        for _ in range(3):
            _clear_execution_caches(trainer)
            batch, measured = trainer.run_update()
            batches.append(_batch_hash(batch))
            metrics.append(measured)
            controls.append(_update_tensors(trainer))
            schedules.append(_state_hash((trainer.scheduler.state_dict(), trainer.update, trainer.actor_steps)))
            del batch
            restore_checkpoint(trainer, manifest, provenance)
            restored.append(_resume_state_hashes(trainer))
            if restored[-1] != boundary:
                changed = [name for name in boundary if restored[-1][name] != boundary[name]]
                raise ValueError(f"checkpoint did not restore exact state: {changed}")
        comparisons = _compare_resume_updates(*controls)
        passed = comparisons.passed and len(set(schedules)) == 1
        report = {
            "format": "hal.o65.resume-check.v5",
            "manifest": str(manifest),
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "boundary": boundary,
            "restored": restored,
            "next_batch_sha256": batches,
            "next_scheduler_sha256": schedules,
            "tolerance": "RMS per model/optimizer-state family: 2 * two-control spread + 8 dtype rounding units; max errors diagnostic",
            "comparisons": comparisons.tensors,
            "groups": comparisons.groups,
            "maximum_parameter_error": max(
                float(row.get("error", 0.0)) for name, row in comparisons.tensors.items() if name.startswith("model/")
            ),
            "control_metrics": metrics[:2],
            "resumed_metrics": metrics[2],
            "passed": passed,
            "elapsed_seconds": time.perf_counter() - started,
        }
        (output / "resume-check.json").write_text(json.dumps(report, indent=2, sort_keys=True))
        if not passed:
            raise ValueError(f"native resume exceeded measured update noise; see {output / 'resume-check.json'}")
    finally:
        restore_checkpoint(trainer, manifest, provenance)
        _clear_execution_caches(trainer)


def _clear_execution_caches(trainer: Trainer) -> None:
    for policy in (trainer.policy, *trainer.engine.anchors.values()):
        policy.clear_caches()


def qualify_development(config: ProxyCli) -> None:
    raw = json.loads(config.statistics.read_text())
    if not isinstance(raw, dict) or any(set(value) != {"mean", "std", "min", "max"} for value in raw.values()):
        raise ValueError("statistics must contain finalized FeatureStats objects")
    stats = {
        name: FeatureStats(value["mean"], value["std"], value["min"], value["max"]) for name, value in raw.items()
    }
    policy = load_development_policy(
        config.checkpoint,
        stats,
        device=config.device,
        decoder_precision=config.decoder_precision,
    )
    report = {
        "checkpoint_sha256": PROXY_SHA256,
        "schedule": dataclasses.asdict(policy.schedule),
        "parameters": sum(p.numel() for p in policy.model.parameters()),
        "actor_parameters": sum(p.numel() for p in policy.actor_parameters()),
        "critic_parameters": sum(p.numel() for p in policy.critic_parameters()),
        "statistics_sha256": sha256(config.statistics),
        "decoder_precision": policy.model.decoder_precision,
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


@dataclasses.dataclass(frozen=True, slots=True)
class EpisodeDiagnostics:
    metadata: Tensor
    rewards: Tensor
    duration: Tensor
    coefficient: Tensor
    values: Tensor
    factor_kl: Tensor


@torch.no_grad()
def episode_diagnostics(
    policy: LoopPolicy,
    reference: LoopPolicy,
    rollout: Rollout,
    wanted: Tensor,
    *,
    windows_per_step: int,
    decisions_per_window: int,
) -> EpisodeDiagnostics:
    table, ids = rollout.table, rollout.ids
    if table.evaluation is None:
        raise ValueError("evaluation rollout is missing its behavior distributions")
    players, ticks = table.players, table.ticks
    rows = torch.arange(ticks * players, device=policy.device).reshape(-1, 64, players).transpose(1, 2).reshape(-1, 64)
    windows = replay_batch(policy, table, rows, table.valid[rows])
    full = (torch.arange(4, device=policy.device) < windows.slots.replan_WK[..., None]).repeat_interleave(4, -1)
    windows = dataclasses.replace(windows, slots=dataclasses.replace(windows.slots, executed_WKF=full))
    old = _take_evaluation(table.evaluation, ids)
    positions = torch.arange(decisions_per_window, device=policy.device)[None].expand(len(rows), -1)
    order = torch.arange(math.ceil(len(rows) / windows_per_step) * windows_per_step, device=policy.device)
    order = torch.where(order < len(rows), order, -1).reshape(-1, windows_per_step)
    control = window_evaluation(
        reference.evaluate_reference(windows, positions, order),
        old.actions,
        window_indices(ids, players),
    )
    meta = table.integers[ids]
    mask = (torch.arange(4, device=policy.device) < meta[:, REPLAN, None]).repeat_interleave(4, -1)
    kl = factor_kl(old, control, mask)
    keep = (meta[:, EPISODE] == 0) & torch.isin(meta[:, MATCH], wanted)
    selected = ids[keep]
    return EpisodeDiagnostics(
        meta[keep],
        table.rewards[selected],
        table.duration[selected],
        table.coefficients[selected],
        table.values[selected],
        kl[keep],
    )


def _episode_diagnostics(chunks: Sequence[EpisodeDiagnostics], results: dict[int, EpisodeResult]) -> dict[str, float]:
    if not chunks:
        raise ValueError("evaluation episode diagnostics are empty")
    metadata = torch.cat([chunk.metadata for chunk in chunks])
    device = metadata.device
    matches = sorted(results)
    match_ids = torch.tensor(matches, device=device)
    lengths = torch.tensor([results[match].duration for match in matches], device=device)
    terminal = torch.tensor([not results[match].timeout for match in matches], device=device)
    at = torch.searchsorted(match_ids, metadata[:, MATCH])
    remaining = lengths[at] - metadata[:, CLOCK]
    executed = executed_factor_mask(metadata[:, DELAY].contiguous(), metadata[:, REPLAN].contiguous(), remaining)
    kl = torch.where(executed, torch.cat([chunk.factor_kl for chunk in chunks]), 0).sum(-1)
    coefficients = torch.cat([chunk.coefficient for chunk in chunks])
    metrics = {
        "kl_cost": float((kl * coefficients).sum()) / len(results),
        "kl_per_frame": float(kl.sum()) / sum(result.duration for result in results.values()),
    }
    duration = torch.cat([chunk.duration for chunk in chunks])
    frames = torch.arange(4, device=device)
    rewards = torch.cat([chunk.rewards for chunk in chunks])
    discounted = torch.where(frames < duration[:, None], rewards * GAMMA**frames, 0).sum(-1) - coefficients * kl
    _, streams = torch.unique(metadata[:, STREAM], return_inverse=True)
    ticks = metadata[:, CLOCK] // 2
    shape = (int(ticks.max()) + 1, int(streams.max()) + 1)
    dense_rewards = torch.zeros(shape, device=device)
    discount = torch.ones(shape, device=device)
    decided = torch.zeros(shape, device=device, dtype=torch.bool)
    dense_rewards[ticks, streams] = discounted
    discount[ticks, streams] = GAMMA**duration
    decided[ticks, streams] = True
    targets = torch.zeros(shape, device=device)
    tail = torch.zeros(shape[1], device=device)
    for tick in range(shape[0] - 1, -1, -1):
        tail = torch.where(decided[tick], dense_rewards[tick] + discount[tick] * tail, tail)
        targets[tick] = torch.where(decided[tick], tail, 0)
    selected = terminal[at]
    metrics["monte_carlo_count"] = float(selected.sum())
    if bool(selected.any()):
        values = torch.cat([chunk.values for chunk in chunks])[selected]
        target = targets[ticks, streams][selected]
        metrics["monte_carlo_rmse"] = float((values - target).square().mean().sqrt())
        metrics["monte_carlo_explained_variance"] = explained_variance(values, target)
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
                    trainer.engine.sampler.config,
                    lambda_min=coefficient,
                    lambda_max=coefficient,
                    anchor_rate=1.0,
                    fixed_percentile=0.95,
                )
                schedule = ScheduleConfig((timing,), trainer.engine.sampler.schedule.supervised_offsets)
                evaluation_seed = int(CounterRng(config.seed).uniform("evaluation", 0, 0, 0) * 2**63)
                sampling = EpisodeSampler(sampler_config, schedule, CounterRng(evaluation_seed))
                policy = trainer.policy.frozen_copy()
                anchors = {0: trainer.engine.anchors[anchor_id].frozen_copy()}
                engine = RolloutEngine((shards,), policy, sampling, anchors)
                engine.initialize()
                results: dict[int, EpisodeResult] = {}
                chunks: list[EpisodeDiagnostics] = []
                wanted_ids = torch.tensor(sorted(wanted), device=policy.device)
                for update in range(config.max_rollouts):
                    rollout = engine.collect(update)
                    chunks.append(
                        episode_diagnostics(
                            policy,
                            trainer.reference,
                            rollout,
                            wanted_ids,
                            windows_per_step=trainer.config.windows_per_step,
                            decisions_per_window=trainer.config.decisions_per_window,
                        )
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
                diagnostics = _episode_diagnostics(chunks, results)
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
    commands.add_parser("profile", add_help=False)
    args, remaining = parser.parse_known_args()
    if args.command == "train":
        train_native(tyro.cli(NativeCli, args=remaining))
    elif args.command == "profile":
        train_native(tyro.cli(GpuProfileCli, args=remaining).native())
    else:
        qualify_development(tyro.cli(ProxyCli, args=remaining))


if __name__ == "__main__":
    main()
