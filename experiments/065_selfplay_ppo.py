"""Train a self-play actor with delayed actions and KL-regularized PPO."""

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
from hal.eval.action_behavior import COUNT_NAMES
from hal.eval.action_behavior import BehaviorCounts
from hal.eval.action_behavior import BehaviorTracker
from hal.eval.self_play_checkpoint import DISTRIBUTED_TRAINING_FORMAT
from hal.eval.self_play_checkpoint import DISTRIBUTED_TRAINING_VERSION
from hal.eval.self_play_checkpoint import EVALUATION_FORMAT
from hal.eval.self_play_checkpoint import TRAINING_FORMAT
from hal.eval.self_play_checkpoint import TRAINING_VERSION
from hal.eval.self_play_checkpoint import CheckpointShard
from hal.eval.self_play_checkpoint import evaluation_actor_path
from hal.eval.self_play_checkpoint import evaluation_checkpoint_metadata
from hal.eval.self_play_checkpoint import load_evaluation_checkpoint
from hal.eval.self_play_checkpoint import read_training_manifest
from hal.eval.self_play_checkpoint import sha256
from hal.inference import compile_cache
from hal.inference.column_context import PackedColumnContextBuilder
from hal.inference.cuda_graph import CapturedCall
from hal.inference.cuda_graph import CompilationStartCounter
from hal.inference.cuda_graph import count_compilation_starts
from hal.inference.cuda_graph import prepared_shape_compilation
from hal.inference.decision_decoder import DecisionDecoder
from hal.inference.graphed_actor import GraphedActorStep
from hal.inference.kv_cache import BlockRestartCache
from hal.inference.kv_cache import forward_tokens_with_kv_cache
from hal.models.action_sequence import CausalTemporalDecoder
from hal.models.action_sequence import NonlinearActionHead
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
from hal.sim import native
from hal.sim import native_bridge
from hal.sim import native_shard
from hal.sim.inputs import native_controller_readback
from hal.sim.native_shard import NativeShard
from hal.training import mfu
from hal.training import self_play_distributed as distributed
from hal.training import self_play_publication
from hal.training.muon import SingleDeviceMuonWithAuxAdam
from hal.training.self_play_flops import SelfPlayFlops
from hal.training.self_play_flops import phase_metrics
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
        if self.delay < 0:
            raise ValueError("delay must be a non-negative integer")
        if self.replan not in (2, 4):
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
            or any(offset < 1 for offset in self.supervised_offsets)
            or tuple(sorted(set(self.supervised_offsets))) != self.supervised_offsets
        ):
            raise ValueError("supervised offsets must be positive, ordered, unique integers")
        if self.rollout_frames < 1:
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
    """Store actions in decode order, padding prefixes to [J,24,4] and sampled factors to [J,16]."""

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


def categorical_kl(log_probs: Tensor, reference_log_probs: Tensor) -> Tensor:
    """Sum forward KL on shared support, detaching the reference.

    Illegal categories use -inf log-probabilities and contribute zero gradients.
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


@dataclasses.dataclass(frozen=True, slots=True)
class AdvantageTargets:
    advantages: Tensor
    value_targets: Tensor


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
    """Normalize PPO and direct KL with the same sigma over selected decisions."""
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
    """Scale local sums by gradient_average_ranks/global_count before gradient averaging."""

    global_count: Tensor
    gradient_average_ranks: int = 1

    def __post_init__(self) -> None:
        _check_positive_count(self.global_count)
        if self.gradient_average_ranks < 1:
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


def critic_loss(
    values_J: Tensor,
    targets_J: Tensor,
    selected_J: Tensor,
    *,
    value_scale: float = 150.0,
    reduction: DecisionReduction | None = None,
) -> Tensor:
    """Fit detached targets at O64 reward scale, including decisions with no executed factors."""
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
    percentile_alpha: float = 5.0
    percentile_beta: float = 1.3

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
        for rate in (self.masked_rate, self.deployment_rate):
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

    def sample(self, match_id: int, episode: int, anchor: int | None) -> Episode:
        if anchor is not None and (type(anchor) is not int or anchor not in (0, 1, 2, 3)):
            raise ValueError("opponent slot must be pi_ref or snapshot 1..3")
        cfg = self.config
        rng = self.rng
        matchup = cfg.matchups[rng.choice([m.weight for m in cfg.matchups], "matchup", match_id, episode, 0)]
        stage = cfg.stages[rng.choice([1.0] * len(cfg.stages), "stage", match_id, episode, 0)]
        stocks = cfg.stocks[rng.choice([1.0] * len(cfg.stocks), "stocks", match_id, episode, 0)]
        anchor_side = None
        if anchor is not None:
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
            percentile = float(
                np.clip(
                    betaincinv(
                        cfg.percentile_alpha, cfg.percentile_beta, rng.uniform("percentile", stream, episode, 0)
                    ),
                    0.1,
                    0.999,
                )
            )
            players.append(
                PlayerEpisode(
                    identity,
                    character,
                    log_lambda,
                    timing,
                    stocks,
                    percent,
                    anchor if side == anchor_side else None,
                    percentile,
                )
            )
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
    """Return borrowed shard frames through the first active episode end."""

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


@dataclasses.dataclass(frozen=True, slots=True)
class NativePoolSnapshot:
    match_ids: tuple[int, ...]
    max_frame: int
    states: tuple[native.NativeSnapshot, ...]


def encode_native_pool(snapshot: NativePoolSnapshot) -> bytes:
    header = json.dumps(
        {"format": "hal.o65.native_pool.v2", "matches": snapshot.match_ids, "max_frame": snapshot.max_frame}
    ).encode()
    states = tuple(native.encode_snapshot(state) for state in snapshot.states)
    return (
        struct.pack("<I", len(header)) + header + b"".join(struct.pack("<I", len(state)) + state for state in states)
    )


class NativeSimulatorShard:
    """Step native matches using pinned buffers for device transfers."""

    def __init__(
        self,
        data_dir: Path,
        match_ids: tuple[int, ...],
        *,
        cpu: int,
        device: torch.device,
        max_frame: int = 28800,
        reward_weights: native.RewardWeights = native.DEFAULT_REWARD_WEIGHTS,
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
        self.shard = NativeShard(data_dir, count, cpu=cpu, records=self._records, reward_weights=reward_weights)

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
            packed,
            actions,
            columns,
            frame.reward.copy(),
            frame.terminated.copy(),
            frame.truncated.copy(),
            observed,
            frame.frame_id.copy(),
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

    def capture(self) -> Future[tuple[native.NativeSnapshot, ...]]:
        return self.shard.capture_many(tuple(range(len(self.match_ids))))

    def save(self) -> bytes:
        return encode_native_pool(NativePoolSnapshot(self.match_ids, self.max_frame, self.capture().result()))

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
        if self.buffer not in (0, 1, ALL_BUFFERS):
            raise ValueError("tick buffer must be zero, one or ALL_BUFFERS")
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


def tick_decoder_buckets(rows: int, count: int) -> tuple[int, ...]:
    if rows < 1 or count not in (2, 4):
        raise ValueError("tick decoder buckets require positive rows and two or four offsets")
    if count == 4:
        return (0, rows)
    step = max(1, (rows + 7) // 8)
    # Keep finer padding near the usual mixed R2/R4 cohort sizes. Every extra
    # decoder shape needs a separate compilation, even on synthetic rows.
    buckets = {0, rows}
    buckets.update(min(rows, step * factor) for factor in (2, 4, 5, 6))
    return tuple(sorted(buckets))


# Anchor layouts span every buffer: one graph per anchor and tick, sampled with the first buffer.
ALL_BUFFERS = -1
STACKED_ANCHORS = -2


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
    frame_ids: NDArray[np.int32]

    def __post_init__(self) -> None:
        count = len(self.rewards)
        if (
            self.records.shape != (count, native.FRAME_DTYPE.itemsize)
            or self.records.dtype != torch.uint8
            or self.actions.shape != (count, 2, ACTION_DIM)
            or self.actions.dtype != torch.float32
            or self.host_actions.shape != (count, 2, ACTION_DIM)
            or self.host_actions.dtype != np.float32
            or self.frame_ids.shape != (count,)
            or self.frame_ids.dtype != np.int32
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
        fields = native.FRAME_DTYPE.fields
        if fields is None:
            raise RuntimeError("native records must have a structured dtype")
        self.reward_offset = fields["reward"][1]
        self.flag_offset = fields["terminated"][1]
        self.records = torch.zeros(capacity, matches, native.FRAME_DTYPE.itemsize, dtype=torch.uint8, device=device)
        self.actions = torch.zeros(capacity, matches, 2, ACTION_DIM, device=device)
        self.rewards = torch.zeros(capacity, matches, 2, device=device)
        self.flags = torch.zeros(capacity, matches, 2, device=device, dtype=torch.bool)
        self.lengths = torch.zeros(matches, device=device, dtype=torch.long)

    def write(self, positions: Tensor, lanes: Tensor, batch: ShardFrames, source: Tensor) -> Tensor:
        selected = batch.records.index_select(0, source)
        self.records[positions, lanes] = selected
        self.actions[positions, lanes] = batch.actions.index_select(0, source)
        rewards = selected[:, self.reward_offset : self.reward_offset + 8].contiguous().view(torch.float32)
        self.rewards[positions, lanes] = rewards
        self.flags[positions, lanes] = selected[:, self.flag_offset : self.flag_offset + 2].bool()
        return rewards


@dataclasses.dataclass(frozen=True, slots=True)
class TickInputs:
    table: DenseRollout
    ids: Tensor
    enabled: Tensor | None = None
    decoder_rows: tuple[int, ...] | None = None

    def __len__(self) -> int:
        return self.ids.numel()

    def tensor(self, value: Tensor) -> Tensor:
        selected = value.index_select(0, self.ids)
        if value is self.table.active and self.enabled is not None:
            return selected & self.enabled
        return selected

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
    """Retain group tensors until the next collection from that group."""

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
    behavior: tuple[tuple[str, BehaviorCounts], ...] = ()


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
        # Evaluation tensors view the first rows of storage with one spare row. Tick graphs
        # send rows they do not own to the spare row, so concurrent graphs never rewrite
        # each other's rows.
        self.spare = capacity
        self.evaluation_storage: list[Tensor] = []
        self.integers = torch.zeros(capacity, 18, dtype=torch.long, device=self.device)
        self.conditions = torch.zeros(capacity, 4, device=self.device)
        self.coefficients = torch.zeros(capacity, device=self.device)
        self.age = torch.zeros(capacity, dtype=torch.long, device=self.device)
        self.opponent_anchor = torch.full((capacity,), -1, dtype=torch.long, device=self.device)
        self.prefix = torch.zeros(capacity, 24, 4, dtype=torch.long, device=self.device)
        self.opponent = torch.zeros_like(self.prefix)
        self.actions = self._evaluation_rows(torch.zeros(capacity + 1, 4, 4, dtype=torch.long, device=self.device))
        self.behavior = self._evaluation_rows(torch.zeros(capacity + 1, 16, device=self.device))
        self.rewards = torch.zeros(capacity, 4, device=self.device)
        self.duration = torch.zeros(capacity, dtype=torch.long, device=self.device)
        self.remaining = torch.full((capacity,), -1, dtype=torch.long, device=self.device)
        self.terminal = torch.zeros(capacity, dtype=torch.bool, device=self.device)
        self.boundary = torch.zeros(capacity, dtype=torch.bool, device=self.device)
        self.following = torch.full((capacity,), -1, dtype=torch.long, device=self.device)
        self.values = self._evaluation_rows(torch.zeros(capacity + 1, device=self.device))
        self.active = torch.zeros(capacity, device=self.device, dtype=torch.bool)
        self.valid = torch.zeros(capacity, device=self.device, dtype=torch.bool)
        self.bootstrap = torch.zeros(capacity, device=self.device, dtype=torch.bool)
        self.queue_error = torch.zeros((), device=self.device, dtype=torch.bool)
        self.evaluation: PolicyEvaluation | None = None

    def _evaluation_rows(self, storage: Tensor) -> Tensor:
        self.evaluation_storage.append(storage)
        return storage[: self.spare]

    def allocate_evaluation(self, model: SelfPlayTransformer) -> None:
        rows = self.spare + 1

        def zeros(*shape: int, dtype: torch.dtype = torch.float32) -> Tensor:
            return self._evaluation_rows(torch.zeros(rows, *shape, dtype=dtype, device=self.device))

        self.evaluation = PolicyEvaluation(
            self.actions,
            self.behavior,
            tuple(zeros(4, model.codec.group_vocabs[group]) for group in DECODE_GROUP_INDICES),
            CriticFeatures(
                zeros(model.cfg.d_model),
                zeros(model.cfg.temporal_d_model),
                zeros(4),
                zeros(4),
                zeros(model.critic_residual.max_queue, 4, dtype=torch.long),
                zeros(dtype=torch.long),
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

    def record_evaluation(
        self, ids: Tensor, output: PolicyEvaluation, values: Tensor, enabled: Tensor | None = None
    ) -> None:
        if self.evaluation is None:

            def storage(value: Tensor) -> Tensor:
                return self._evaluation_rows(value.new_zeros((self.spare + 1, *value.shape[1:])))

            self.evaluation = PolicyEvaluation(
                self.actions,
                self.behavior,
                tuple(storage(p) for p in output.distributions),
                CriticFeatures(*(storage(getattr(output.features, name)) for name in CRITIC_FEATURE_NAMES)),
            )
        sources = (
            output.actions,
            output.log_probs,
            values,
            *output.distributions,
            *(getattr(output.features, name) for name in CRITIC_FEATURE_NAMES),
        )
        rows = ids if enabled is None else torch.where(enabled, ids, self.spare)
        for target, source in zip(self.evaluation_storage, sources, strict=True):
            target[rows] = source.to(target.dtype)


class TickLinear(torch.nn.Linear):
    """Retain the exact autocast weight conversion outside captured tick replays."""

    tick_weight: Tensor
    tick_bias: Tensor | None

    def __init__(self, source: torch.nn.Linear) -> None:
        super().__init__(source.in_features, source.out_features, source.bias is not None, device="meta")
        self.weight = source.weight
        self.bias = source.bias
        self.register_buffer("tick_weight", source.weight.detach().to(torch.bfloat16), persistent=False)
        self.register_buffer(
            "tick_bias", None if source.bias is None else source.bias.detach().to(torch.bfloat16), persistent=False
        )
        self.tick = False
        self.train(source.training)

    @torch.no_grad()
    def refresh(self) -> None:
        self.tick_weight.copy_(self.weight)
        if self.bias is not None:
            assert self.tick_bias is not None
            self.tick_bias.copy_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        if self.tick and torch.is_autocast_enabled(input.device.type):
            return torch.nn.functional.linear(input, self.tick_weight, self.tick_bias)
        return torch.nn.functional.linear(input, self.weight, self.bias)


class AnchorLinear(TickLinear):
    """Apply one frozen model per equal-sized, contiguous group of rows."""

    def __init__(self, sources: tuple[torch.nn.Linear, ...]) -> None:
        super().__init__(sources[0])
        self.sources = sources
        self.weight = torch.nn.Parameter(
            torch.stack([source.weight.detach() for source in sources]), requires_grad=False
        )
        self.bias = (
            None
            if sources[0].bias is None
            else torch.nn.Parameter(
                torch.stack([cast(Tensor, source.bias).detach() for source in sources]), requires_grad=False
            )
        )
        self.tick_weight = self.weight.to(torch.bfloat16)
        self.tick_bias = None if self.bias is None else self.bias.to(torch.bfloat16)

    @torch.no_grad()
    def refresh(self) -> None:
        for index, source in enumerate(self.sources):
            self.weight[index].copy_(source.weight)
            if self.bias is not None:
                self.bias[index].copy_(cast(Tensor, source.bias))
        super().refresh()

    def project(self, inputs: Tensor, start: int = 0, stop: int | None = None, *, bias: bool = True) -> Tensor:
        groups = len(self.sources)
        if inputs.shape[0] % groups:
            raise ValueError("stacked linear rows must contain equal anchor groups")
        tick = self.tick and torch.is_autocast_enabled(inputs.device.type)
        weight = (self.tick_weight if tick else self.weight)[:, :, start:stop]
        biases = (self.tick_bias if tick else self.bias) if bias else None
        grouped = inputs.reshape(groups, -1, inputs.shape[-1])
        result = (
            torch.bmm(grouped, weight.transpose(1, 2))
            if biases is None
            else torch.baddbmm(biases[:, None], grouped, weight.transpose(1, 2))
        )
        return result.reshape(*inputs.shape[:-1], self.out_features)

    def add_bias(self, inputs: Tensor) -> Tensor:
        if self.bias is None:
            raise ValueError("stacked readout requires a bias")
        return (inputs.reshape(len(self.sources), -1, self.out_features) + self.bias[:, None]).reshape_as(inputs)

    def forward(self, input: Tensor) -> Tensor:
        return self.project(input)


class AnchorEmbedding(torch.nn.Embedding):
    def __init__(self, sources: tuple[torch.nn.Embedding, ...]) -> None:
        source = sources[0]
        super().__init__(source.num_embeddings, source.embedding_dim, device="meta")
        self.sources = sources
        self.weight = torch.nn.Parameter(
            torch.stack([source.weight.detach() for source in sources]), requires_grad=False
        )

    @torch.no_grad()
    def refresh(self) -> None:
        for index, source in enumerate(self.sources):
            self.weight[index].copy_(source.weight)

    def forward(self, input: Tensor) -> Tensor:
        groups = len(self.sources)
        if input.shape[0] % groups:
            raise ValueError("stacked embedding rows must contain equal anchor groups")
        torch._assert_async(
            ((input >= 0) & (input < self.num_embeddings)).all(), "stacked embedding index is out of range"
        )
        indices = input.reshape(groups, -1) + torch.arange(groups, device=input.device)[:, None] * self.num_embeddings
        result = torch.nn.functional.embedding(indices, self.weight.flatten(0, 1))
        return result.reshape(*input.shape, self.embedding_dim)


class AnchorTemporalDecoder(CausalTemporalDecoder):
    def _state_bias(self, trunk: Tensor) -> Tensor:
        return cast(AnchorLinear, self.token_projection).project(trunk, stop=self.trunk_width)

    def _step_features(self, previous: Tensor, offsets: Tensor, embedded: Tensor | None = None) -> Tensor:
        projection = cast(AnchorLinear, self.token_projection)
        action = projection.project(
            self.codec.embed_frame(previous) if embedded is None else embedded,
            self.trunk_width,
            self.trunk_width + self.controller_width,
            bias=False,
        )
        offset = projection.project(
            self.offset_embedding(offsets.expand(previous.shape[:-1])),
            self.trunk_width + self.controller_width,
            bias=False,
        )
        return action.float() + offset.float() if self.return_conditioner.ranges is not None else action + offset

    def _project_action_head(self, head: NonlinearActionHead, features: Tensor) -> Tensor:
        if self.return_conditioner.ranges is None or not torch.is_autocast_enabled(features.device.type):
            return head.project(features)
        hidden = torch.nn.functional.silu(head.up(features))
        down = cast(AnchorLinear, head.down)
        return down.add_bias(down.project(hidden, bias=False).float())


class TransformerPolicy:
    """Append fixed rollout slots and replay one history per training window."""

    def __init__(
        self,
        model: SelfPlayTransformer,
        schedule: ScheduleConfig,
        statistics: dict[str, FeatureStats],
    ) -> None:
        self.profile: UpdateProfile | None = None
        self.startup_finished = False
        self.post_startup_captures = 0
        self.capture_count = 0
        self.tick_compile_instances = 16
        self.model = model.eval()
        self.flops = SelfPlayFlops.from_model(model)
        self.work_flops: dict[str, int] = defaultdict(int)
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
        linears = []
        if self.device.type == "cuda":
            for name, module in tuple(model.named_modules()):
                if isinstance(module, torch.nn.Linear):
                    linear = module if isinstance(module, TickLinear) else TickLinear(module)
                    model.set_submodule(name, linear)
                    linears.append(linear)
        self.tick_linears = tuple(linears)
        if schedule.supervised_offsets != model.head_offsets:
            raise ValueError("schedule supervised offsets must match the model checkpoint")
        configurations = tuple((c.delay, c.replan) for c in schedule.configurations)
        self.packed_builder = PackedColumnContextBuilder(statistics, native.OBSERVATION_FIELDS, self.device)
        self.decode_order = torch.tensor(DECODE_GROUP_INDICES, device=self.device)
        self.tick_decoder = DecisionDecoder(model, configurations, compiled=False)
        self.tick_caches: dict[tuple[ActionRing, tuple[int, ...]], BlockRestartCache] = {}
        self.tick_sources: dict[tuple[ActionRing, tuple[int, ...]], tuple[DenseRollout, FrameStore]] = {}
        self.tick_graphs: dict[tuple[tuple[ActionRing, tuple[int, ...]], int, int, int], CapturedCall[None]] = {}
        self.restart_graphs: dict[tuple[ActionRing, tuple[int, ...]], CapturedCall[None]] = {}
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

    def admit_capture(self, key: object) -> None:
        if self.startup_finished:
            self.post_startup_captures += 1
            raise RuntimeError(f"CUDA graph capture after startup: {key}")
        self.capture_count += 1

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
        self.startup_finished = False
        self.tick_caches.clear()
        self.tick_sources.clear()
        self.tick_graphs.clear()
        self.restart_graphs.clear()
        self.current_layouts.clear()
        self.value_graphs.clear()
        self.reference_graphs.clear()
        self.reference_forward_model = None
        self.actor_graph = None
        self.prepare_tick_weights()

    @torch.no_grad()
    def prepare_tick_weights(self) -> None:
        """Refresh captured BF16 weights after any master-weight load or update."""
        for module in self.tick_linears:
            module.refresh()

    @contextlib.contextmanager
    def _tick_weight_casts(self) -> Iterator[None]:
        for module in self.tick_linears:
            module.tick = True
        try:
            yield
        finally:
            for module in self.tick_linears:
                module.tick = False

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
        self.tick_sources = {
            key: source for key, source in self.tick_sources.items() if key[0] is not ring or key in current
        }
        self.tick_graphs = {
            key: graph for key, graph in self.tick_graphs.items() if key[0][0] is not ring or key[0] in current
        }
        # A restart graph writes the cache it captured, so it goes with that cache.
        self.restart_graphs = {
            key: graph for key, graph in self.restart_graphs.items() if key[0] is not ring or key in current
        }
        buffers = {layout.buffer for layout in layouts}
        self.current_layouts = {
            key: layout for key, layout in self.current_layouts.items() if key[0] is not ring or key[1] in buffers
        }

    def tick_ready(self, layout: TickLayout, ring: ActionRing, *, clock: int) -> bool:
        if self.device.type != "cuda":
            return True
        key = ring, (layout.buffer, len(layout.streams))
        appended, count = (1 if clock % 128 == 0 else 2), (4 if clock % 4 == 0 else 2)
        return all(
            (key, appended, count, bucket) in self.tick_graphs
            for bucket in self.decoder_buckets(len(layout.streams), count)
        ) and (clock % 128 != 0 or key in self.restart_graphs)

    def decoder_buckets(self, rows: int, count: int) -> tuple[int, ...]:
        return tick_decoder_buckets(rows, count)

    def decoder_layout(
        self, rows: tuple[int, ...], batch: int, bucket: int
    ) -> tuple[tuple[int, ...], tuple[bool, ...]]:
        return rows + rows[:1] * (bucket - len(rows)), (True,) * len(rows) + (False,) * (bucket - len(rows))

    def decoder_bucket(self, rows: tuple[int, ...], batch: int, count: int) -> int:
        if self.device.type != "cuda":
            return len(rows)
        return next(size for size in self.decoder_buckets(batch, count) if size >= len(rows))

    @torch.no_grad()
    def sample_tick(
        self, states: TickInputs, rng: CounterRng, layout: TickLayout, ring: ActionRing, *, clock: int = 0
    ) -> None:
        if len(layout.streams) != len(states):
            raise ValueError("tick layout contains an unknown stream or row count")
        if self.device.type == "cuda" and states.decoder_rows is None:
            raise ValueError("CUDA ticks require host-selected decoder rows")
        if states.decoder_rows is not None and (
            not isinstance(states.decoder_rows, tuple)
            or any(type(row) is not int or not 0 <= row < len(states) for row in states.decoder_rows)
            or tuple(sorted(set(states.decoder_rows))) != states.decoder_rows
        ):
            raise ValueError("tick decoder rows must be sorted, distinct row indices")
        key = ring, (layout.buffer, len(states))
        layout_key = ring, layout.buffer
        if self.device.type == "cuda":
            sources = states.table, states.table.store
            if key in self.tick_sources and self.tick_sources[key] != sources:
                raise ValueError("tick graph storage changed; clear caches before replacing rollout storage")
            self.tick_sources[key] = sources
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
        restart = clock % 128 == 0
        enabled = self.device.type == "cuda"
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=enabled, cache_enabled=False):
            if restart:
                restart_inputs = (*states.packed(129), states.tensor(states.table.integers))
                if not enabled:
                    self._restart_block(cache, restart_inputs, validate=True)
                else:
                    captured_restart = self.restart_graphs.get(key)
                    if captured_restart is None:
                        self.admit_capture(("restart", key))
                        static_restart = tuple(value.clone() for value in restart_inputs)

                        def restart_block(static: tuple[Tensor, ...] = static_restart) -> None:
                            self._restart_block(cache, static, validate=False)

                        operation = torch.compile(
                            restart_block,
                            fullgraph=True,
                            dynamic=False,
                            options={"triton.cudagraphs": False},
                        )
                        with prepared_shape_compilation(64), self._tick_weight_casts():
                            captured_restart = CapturedCall(
                                operation,
                                static_restart,
                                (*cache.cache.buffers(), cache.frames_in_block),
                                device=self.device,
                            )
                        self.restart_graphs[key] = captured_restart
                    captured_restart(restart_inputs)
            appended = 1 if restart else 2
            count = 4 if clock % 4 == 0 else 2
            decoder_rows = states.decoder_rows
            if decoder_rows is None:
                meta = states.tensor(states.table.integers)
                due = states.tensor(states.table.active) & (meta[:, CLOCK] % meta[:, REPLAN] == 0)
                decoder_rows = tuple(torch.nonzero(due).flatten().tolist())
            buckets = self.decoder_buckets(len(states), count)
            bucket = self.decoder_bucket(decoder_rows, len(states), count)
            # Padded slots repeat a due row, so no live row that is not due enters
            # the decoder. Their writes go only to the ring and table spare rows.
            padded_rows, decoder_valid = self.decoder_layout(decoder_rows, len(states), bucket)
            selected = torch.tensor(padded_rows, dtype=torch.long, pin_memory=enabled).to(
                self.device, non_blocking=True
            )
            inputs = (
                states.ids,
                torch.ones(len(states), dtype=torch.bool, device=self.device)
                if states.enabled is None
                else states.enabled,
                selected,
                torch.tensor(
                    decoder_valid,
                    dtype=torch.bool,
                    pin_memory=enabled,
                ).to(self.device, non_blocking=True),
            )
            graph_key = key, appended, count, bucket
            captured = self.tick_graphs.get(graph_key)
            if captured is None:
                if enabled:
                    self.tick_decoder.prepare_rotary()
                for capture_bucket in buckets if enabled else (bucket,):
                    capture_key = key, appended, count, capture_bucket
                    if enabled and capture_key in self.tick_graphs:
                        continue
                    capture_inputs = inputs
                    if capture_bucket != bucket:
                        # Prepare every bucket on synthetic inactive rows before
                        # startup seals graph capture. Resets can change the count.
                        capture_inputs = (
                            inputs[0],
                            torch.zeros_like(inputs[1]),
                            torch.zeros(capture_bucket, dtype=torch.long, device=self.device),
                            torch.zeros(capture_bucket, dtype=torch.bool, device=self.device),
                        )
                    static = tuple(value.clone() for value in capture_inputs)

                    def tick(graph_inputs: tuple[Tensor, ...] = static, decoder_bucket: int = capture_bucket) -> None:
                        with torch.autocast(
                            self.device.type, dtype=torch.bfloat16, enabled=enabled, cache_enabled=False
                        ):
                            staged = TickInputs(states.table, graph_inputs[0], graph_inputs[1])
                            static = (
                                *staged.packed(appended, tail=appended),
                                staged.tensor(states.table.integers),
                                staged.tensor(states.table.conditions),
                                staged.tensor(states.table.prefix),
                                staged.tensor(states.table.opponent),
                                staged.tensor(states.table.active),
                                staged.ids,
                                graph_inputs[2],
                                graph_inputs[3],
                            )
                            packed = static[:5]
                            if not restart:
                                # A staggered first episode can begin between block boundaries.
                                young = static[5][:, LENGTH] <= 2
                                cache.cache.positions.masked_fill_(young[:, None], -1)
                                cache.cache.next_position.copy_(
                                    torch.where(young, static[5][:, LENGTH] - appended, cache.cache.next_position)
                                )
                                records, actions, sides, identities, lengths = packed
                                single = lengths == 1
                                records = torch.where(single[:, None, None], records.flip(1), records)
                                actions = torch.where(single[:, None, None, None], actions.flip(1), actions)
                                packed = (records, actions, sides, identities, torch.full_like(lengths, appended))
                            context, observed = self._packed_context(packed)
                            tokens = self.model.context_tokens(context, observed)
                            hidden = forward_tokens_with_kv_cache(self.model, tokens, cache.cache)
                            cache.frames_in_block.add_(appended)
                            due = static[9] & (static[5][:, CLOCK] % static[5][:, REPLAN] == 0)
                            indices, valid = static[11], static[12]
                            torch._assert_async(
                                (~valid | due.index_select(0, indices)).all(), "tick selected a row that is not due"
                            )
                            torch._assert_async(valid.sum() == due.sum(), "tick decoder rows omit a due row")
                            if decoder_bucket == 0:
                                return
                            rows = static[5].index_select(0, indices)
                            memory = cache.memory()
                            history = KVMemory(
                                memory.kv,
                                memory.positions.index_select(0, indices),
                                memory.query_position.index_select(0, indices),
                                memory.window,
                                memory.prefix,
                                indices,
                            )
                            result = self._decode_tensors(
                                rows,
                                static[6].index_select(0, indices),
                                static[7].index_select(0, indices),
                                static[8].index_select(0, indices),
                                hidden.index_select(0, indices)[:, -1:],
                                observed.index_select(0, indices)[:, -1],
                                self._tensor_action_uniforms(rows, rng),
                                None,
                                history=history,
                                count=count,
                                decoder=self.tick_decoder,
                            )
                            ring.commit(
                                rows,
                                valid,
                                result.actions,
                                self.device_wire_actions(result.actions, validate=False),
                                count=count,
                            )
                            values = self.values(result.features, validate=False)
                            states.table.record_evaluation(static[10].index_select(0, indices), result, values, valid)

                    if enabled:
                        self.admit_capture(("tick", capture_key))
                        operation = torch.compile(
                            tick, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False}
                        )
                        # Cover four groups and the startup plus three resume
                        # controls without exhausting Dynamo's lifetime counter.
                        with (
                            prepared_shape_compilation(64, instance_count=self.tick_compile_instances),
                            self._tick_weight_casts(),
                        ):
                            self.tick_graphs[capture_key] = CapturedCall(
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
                captured = self.tick_graphs.get(graph_key)
            self.work_flops["rollout"] += self.flops.rollout(
                len(states),
                appended,
                self.tick_decoder.max_delay + count,
                count,
                restart=restart,
            ) - (len(states) - bucket) * (
                self.flops.decode(self.tick_decoder.max_delay + count, count) + self.flops.critic
            )
            if self.profile is not None:
                self.profile.work["rollout/trunk_tokens"] += len(states) * (appended + (128 if restart else 0))
                self.profile.work["rollout/decoder_tokens"] += bucket * (self.tick_decoder.max_delay + count)
            if captured is None:
                tick(inputs)
            else:
                with (
                    torch.cuda.nvtx.range(
                        f"hal.o65.TransformerPolicy.sample_tick/B{len(states)}/append{appended}/R{count}"
                    )
                    if self.profile is not None
                    else contextlib.nullcontext()
                ):
                    captured(inputs)

    def _restart_block(self, cache: BlockRestartCache, inputs: tuple[Tensor, ...], *, validate: bool) -> None:
        """Re-encode the previous 128 frames of each row into a fresh block cache."""
        enabled = self.device.type == "cuda"
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=enabled, cache_enabled=False):
            features, observed = self._packed_context(inputs[:5])
            tokens = self.model.context_tokens(features, observed)
            lengths = inputs[5][:, LENGTH] - 1
            positions = torch.arange(128, device=self.device)[None] - (128 - lengths[:, None])
            previous = tokens.gather(1, positions.clamp_min(0)[..., None].expand(-1, -1, tokens.shape[-1]))
            previous = torch.where((positions >= 0)[..., None], previous, 0)
            cache.restart(previous, lengths, validate=validate)

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
        # Keep the trunk eager to meet the Torch 2.11 proxy precision limits.
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
            chunks = []
            for start in range(0, padded_windows, windows_per_step):
                part = tuple(value[start : start + windows_per_step] for value in inputs)
                key = ((windows_per_step, count), *(tuple(value.shape) for value in part))
                captured = self.reference_graphs.get(key)
                if captured is None:
                    self.admit_capture(("reference", key))
                    static = tuple(value.clone() for value in part)
                    with prepared_shape_compilation(64):
                        captured = CapturedCall(
                            lambda static=static: self.reference_forward(static, windows_per_step, count),
                            static,
                            device=self.device,
                        )
                    self.reference_graphs[key] = captured
                # Each replay reuses output storage; retain this chunk before the next.
                chunks.append(concatenate_replays((captured(part),), 0))
            output = concatenate_replays(chunks, 0)
        self.work_flops["reference"] += self.flops.replay(padded_windows, padded_slots, self.max_delay + 4, 4)
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
        self.work_flops["value"] += (
            self.flops.replay(len(batch.records), batch.rows.shape[1], self.max_delay + 1, 0)
            + batch.rows.numel() * self.flops.critic
        )
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
            self.admit_capture(("value", key))
            static = tuple(value.clone() for value in inputs)
            with prepared_shape_compilation(64):
                captured = CapturedCall(lambda: self.value_forward(static), static, device=self.device)
            self.value_graphs[key] = captured
        return captured(inputs)

    def actor_step(
        self, batch: ReplayBatch, targets: ActorTargets, *, clip_epsilon: float
    ) -> tuple[ActorLoss, ReplayOutput]:
        self.work_flops["actor"] += 3 * self.flops.replay(
            len(batch.records), batch.rows.shape[1], self.max_delay + 4, 4
        )
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
            self.admit_capture(("actor", tuple(tuple(value.shape) for value in inputs)))
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


class StackedAnchorPolicy(TransformerPolicy):
    def __init__(self, sources: tuple[TransformerPolicy, ...]) -> None:
        if len(sources) < 2:
            raise ValueError("a stacked anchor policy requires at least two groups")
        first = sources[0]
        shapes = tuple((name, tuple(parameter.shape)) for name, parameter in first.model.named_parameters())
        if any(
            source.device != first.device
            or source.model.cfg != first.model.cfg
            or source.model.ranges != first.model.ranges
            or source.model.decoder_precision != first.model.decoder_precision
            or source.schedule != first.schedule
            or source.statistics != first.statistics
            or tuple((name, tuple(parameter.shape)) for name, parameter in source.model.named_parameters()) != shapes
            for source in sources
        ):
            raise ValueError("stacked anchors require matching model and representation contracts")
        for name, _ in shapes:
            parent, _, local = name.rpartition(".")
            module = first.model.get_submodule(parent)
            linear = isinstance(module, torch.nn.Linear) and local in ("weight", "bias")
            embedding = isinstance(module, torch.nn.Embedding) and local == "weight"
            if not (linear or embedding):
                raise ValueError("stacked anchors require linear or embedding parameters")
        model = copy.deepcopy(first.model).requires_grad_(False)
        temporal = model.temporal
        with torch.device("meta"):
            model.temporal = AnchorTemporalDecoder(
                model.cfg,
                model.codec,
                live_horizons=temporal.live_horizons,
                training_diagnostics=temporal.training_diagnostics,
            )
        model.temporal.return_conditioner = temporal.return_conditioner
        for name, buffer in tuple(model.named_buffers()):
            if buffer.device.type == "meta":
                parent, _, local_name = name.rpartition(".")
                model.get_submodule(parent).register_buffer(local_name, first.model.get_buffer(name).clone())
        embeddings = []
        for name, module in tuple(model.named_modules()):
            if isinstance(module, torch.nn.Linear):
                model.set_submodule(
                    name,
                    AnchorLinear(tuple(cast(torch.nn.Linear, source.model.get_submodule(name)) for source in sources)),
                )
            elif isinstance(module, torch.nn.Embedding):
                embedding = AnchorEmbedding(
                    tuple(cast(torch.nn.Embedding, source.model.get_submodule(name)) for source in sources)
                )
                model.set_submodule(name, embedding)
                embeddings.append(embedding)
        self.anchor_groups = len(sources)
        self.anchor_sources = sources
        self.anchor_embeddings = tuple(embeddings)
        super().__init__(model, first.schedule, first.statistics)
        # Each row uses one model; counting the stacked parameter storage would
        # multiply the per-row GEMM work by the number of anchor groups.
        self.flops = first.flops

    @torch.no_grad()
    def prepare_tick_weights(self) -> None:
        super().prepare_tick_weights()
        # CPU policies do not install the CUDA tick-weight lifecycle.
        if self.device.type != "cuda":
            for module in self.model.modules():
                if isinstance(module, AnchorLinear):
                    module.refresh()
        for embedding in self.anchor_embeddings:
            embedding.refresh()

    def decoder_buckets(self, rows: int, count: int) -> tuple[int, ...]:
        if rows % self.anchor_groups:
            raise ValueError("stacked tick rows must contain equal anchor groups")
        return tuple(size * self.anchor_groups for size in tick_decoder_buckets(rows // self.anchor_groups, count))

    def decoder_bucket(self, rows: tuple[int, ...], batch: int, count: int) -> int:
        width = batch // self.anchor_groups
        maximum = max(sum(start <= row < start + width for row in rows) for start in range(0, batch, width))
        return next(size for size in self.decoder_buckets(batch, count) if size >= maximum * self.anchor_groups)

    def decoder_layout(
        self, rows: tuple[int, ...], batch: int, bucket: int
    ) -> tuple[tuple[int, ...], tuple[bool, ...]]:
        width = batch // self.anchor_groups
        if bucket == 0:
            return (), ()
        per_group = bucket // self.anchor_groups
        padded: list[int] = []
        valid: list[bool] = []
        for start in range(0, batch, width):
            selected = tuple(row for row in rows if start <= row < start + width)
            padded.extend(selected + (selected[0] if selected else rows[0],) * (per_group - len(selected)))
            valid.extend((True,) * len(selected) + (False,) * (per_group - len(selected)))
        return tuple(padded), tuple(valid)

    def validate_layout(self, policies: tuple[LoopPolicy, ...], counts: tuple[int, ...]) -> None:
        total = sum(counts)
        if total % self.anchor_groups:
            raise ValueError("stacked anchor row groups changed")
        width = total // self.anchor_groups
        if width < 1 or any(count % width for count in counts):
            raise ValueError("stacked anchor row groups changed")
        expected = tuple(policy for policy, count in zip(policies, counts, strict=True) for _ in range(count // width))
        if expected != self.anchor_sources:
            raise ValueError("stacked anchor weights disagree with the row layout")


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
        # Queues view the first lanes of storage with one spare lane. Tick graphs send rows
        # they do not own to the spare lane, so concurrent graphs never rewrite each other's lanes.
        self.spare = matches
        self.value_storage = torch.zeros(matches + 1, 2, capacity, 4, dtype=torch.long, device=device)
        self.controller_storage = torch.zeros(matches + 1, 2, capacity, ACTION_DIM, device=device)
        self.length_storage = torch.zeros(matches + 1, 2, dtype=torch.long, device=device)
        self.values = self.value_storage[:matches]
        self.controllers = self.controller_storage[:matches]
        self.length_device = self.length_storage[:matches]
        self.head = torch.zeros(matches, 2, dtype=torch.long, device=device)
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
        # Reading the view beside a write to its base makes Inductor copy the whole base back,
        # which would overwrite lengths a concurrent graph commits.
        positions = (self.head[lanes, sides, None] + self.length_storage[lanes, sides, None] + offsets) % self.capacity
        valid = due[:, None] & (offsets < metadata[:, REPLAN, None])
        targets = torch.where(valid, lanes[:, None], self.spare)
        self.value_storage[targets, sides[:, None], positions] = actions[:, :written]
        self.controller_storage[targets, sides[:, None], positions] = controllers[:, :written]
        owned = torch.where(due, lanes, self.spare)
        self.length_storage[owned, sides] += metadata[:, REPLAN]


def accept_frame_tensors(
    frames: ShardFrames,
    lanes: Tensor,
    pending: Tensor,
    table: DenseRollout,
    ring: ActionRing,
    active_lanes: Tensor,
    stops: Tensor,
    observed_clocks: Tensor,
    episode_returns: Tensor,
) -> None:
    slots = (2 * lanes[:, None] + torch.arange(2, device=lanes.device)[None]).flatten()
    active = active_lanes[lanes]
    rewards = table.store.write(stops[lanes], lanes, frames, torch.arange(len(lanes), device=lanes.device))
    ids = torch.where(pending[slots] >= 0, pending[slots], slots)
    valid = (pending[slots] >= 0) & active.repeat_interleave(2)
    duration = table.duration[ids].clamp_max(3)
    table.rewards[ids, duration] = torch.where(valid, rewards.flatten(), table.rewards[ids, duration])
    table.duration[ids] += valid.long()
    episode_returns[lanes] += torch.where(active[:, None], rewards, 0)
    stops[lanes] += active.long()
    observed_clocks[lanes] += active.long()
    ring.head[lanes] = (ring.head[lanes] + active[:, None].long()) % ring.capacity
    # Read and write the base so Inductor cannot copy another buffer's lengths.
    ring.length_storage[lanes] -= active[:, None].long()


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


def opponent_slots(lanes: int) -> tuple[int | None, ...]:
    """Spread 4:2:2:2 anchors over 64 lanes; sub-four test shards use self-play."""
    if type(lanes) is not int or lanes < 1:
        raise ValueError("shard lanes must be positive")
    counts = (max(1, round(lanes / 16)), *(max(1, round(lanes / 32)) for _ in range(3)))
    if lanes < 4:
        return (None,) * lanes
    roles: list[int | None] = [None] * lanes
    anchors = [slot for slot, count in enumerate(counts) for _ in range(count)]
    for index, slot in enumerate(anchors):
        roles[(2 * index + 1) * lanes // (2 * len(anchors))] = slot
    return tuple(roles)


class RolloutEngine:
    def __init__(
        self,
        groups: tuple[tuple[SimulatorShard, ...], ...],
        policy: LoopPolicy,
        sampler: EpisodeSampler,
        anchors: dict[int, LoopPolicy],
    ) -> None:
        if not groups or any(not group or any(not shard.match_ids for shard in group) for group in groups):
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
        self.profile: UpdateProfile | None = None
        self.groups = groups
        self.policy = policy
        self.sampler = sampler
        self.configurations = sampler.schedule.configurations
        self.anchors = anchors
        self.anchor_stacks: list[StackedAnchorPolicy | None] = [None for _ in groups]
        self.stacked_owners: dict[int, Tensor] = {}
        self.tick_layouts: dict[tuple[int, int | None], TickLayout] = {}
        self.tick_decoder_rows: dict[tuple[int, int | None], tuple[int, ...]] = {}
        self.tick_decoder_candidates: dict[tuple[int, int | None], tuple[NDArray[np.int64], NDArray[np.int64]]] = {}
        self.group_layouts: list[dict[tuple[int, int | None], TickLayout]] = [{} for _ in groups]
        self.layout_anchors: list[tuple[int, ...]] = [() for _ in groups]
        self.clocks = [0] * len(groups)
        self.cursor = 0
        self.matches: dict[int, MatchState] = {}
        self._initialized = False
        self.staggering = False
        self.completed_update = -1
        self.results: list[EpisodeResult] = []
        self.group_matches = tuple(tuple(match for shard in group for match in shard.match_ids) for group in groups)
        capacity = max(t.delay + t.replan for t in sampler.schedule.configurations)
        self.rings = tuple(ActionRing(len(matches), capacity, policy.device) for matches in self.group_matches)
        for ring in self.rings:
            ring.controllers.copy_(policy.device_wire_actions(ring.values))
        self.lane_roles = {
            match: role
            for group in groups
            for shard in group
            for match, role in zip(shard.match_ids, opponent_slots(len(shard.match_ids)), strict=True)
        }
        self.behavior_tracker = BehaviorTracker(2 * (max(ids) + 1))
        self.behavior_totals: dict[str, NDArray[np.int64]] = {}
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
        self.buffer_player_slots: dict[int, Tensor] = {}
        self.player_slots = tuple(torch.arange(2 * len(ids), device=policy.device) for ids in self.group_matches)
        self.staging: dict[int, BufferStaging] = {}
        self.anchor_streams: dict[tuple[int, int | None], torch.cuda.Stream] = {}
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
                slots = (2 * lanes[:, None] + np.arange(2)[None]).flatten()
                self.buffer_player_slots[id(buffer)] = torch.from_numpy(slots).to(policy.device)
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
        self.stage_tick_tensors = (
            torch.compile(self._stage_tick_tensors, fullgraph=True, dynamic=True, options={"triton.cudagraphs": False})
            if policy.device.type == "cuda"
            else self._stage_tick_tensors
        )
        self.accept_frame = (
            torch.compile(accept_frame_tensors, fullgraph=True, dynamic=True, options={"triton.cudagraphs": False})
            if policy.device.type == "cuda"
            else accept_frame_tensors
        )

    def prepare_anchor_stacks(self) -> None:
        """Build execution copies before capture; restore retains their storage."""
        if self.policy.device.type != "cuda" or not all(
            isinstance(policy, TransformerPolicy) for policy in self.anchors.values()
        ):
            return
        # Reference replay also uses these immutable factors. Its canonical
        # model no longer visits the live tick path that used to prepare them.
        for policy in self.anchors.values():
            cast(TransformerPolicy, policy).tick_decoder.prepare_rotary()
        prepared: dict[tuple[int, ...], StackedAnchorPolicy] = {}
        for group, matches in enumerate(self.group_matches):
            owners = tuple(
                owner for owner in self.anchors if any(self.lane_roles[match] == owner for match in matches)
            )
            if len(owners) < 2:
                self.anchor_stacks[group] = None
                continue
            counts = tuple(2 * sum(self.lane_roles[match] == owner for match in matches) for owner in owners)
            width = math.gcd(*counts)
            # Split the larger pi_ref cohort into equal groups. This preserves
            # every real row without padding the other three anchor trunks.
            groups = tuple(owner for owner, count in zip(owners, counts, strict=True) for _ in range(count // width))
            if groups not in prepared:
                prepared[groups] = StackedAnchorPolicy(
                    tuple(cast(TransformerPolicy, self.anchors[owner]) for owner in groups)
                )
            self.anchor_stacks[group] = prepared[groups]

    def execution_policies(self) -> tuple[LoopPolicy, ...]:
        stacks = tuple(dict.fromkeys(stack for stack in self.anchor_stacks if stack is not None))
        return self.policy, *self.anchors.values(), *stacks

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
                if sorted(p.anchor for p in match.episode.players if p.anchor is not None) != (
                    [] if self.lane_roles[match.match] is None else [self.lane_roles[match.match]]
                ):
                    raise ValueError("checkpoint episode disagrees with its fixed opponent slot")
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
        for policy in self.execution_policies():
            policy.clear_caches()

    def initialize(self) -> None:
        if self._initialized:
            raise ValueError("pool is already initialized")
        for group in self.groups:
            for shard in group:
                for lane, match in enumerate(shard.match_ids):
                    self._reset(shard, lane, match, 0, 0)
        self._initialized = True

    def stagger_starts(self, maximum: int) -> tuple[int, ...]:
        """Wait before each initial episode so its age at the common end is uniform.

        Odd starts take their first neutral frame here, then join the two-frame
        tick loop. Their observations and queues are identical to stepping that
        frame at its logical clock. No policy decision occurs on the odd frame.
        """
        if not self._initialized or self.staggering or any(self.clocks) or self.completed_update != -1:
            raise ValueError("head start requires a fresh initialized pool")
        if type(maximum) is not int or maximum < 1:
            raise ValueError("head start maximum must be a positive integer")
        self.staggering = True
        count = self.sampler.schedule.rollout_frames
        end = math.ceil(maximum / count) * count
        offsets = []
        for group, shards in enumerate(self.groups):
            store = self.stores[group]
            if store is None:
                raise ValueError("pool has no initial frame storage")
            # Startup graphs retain the full store's addresses through head start.
            if len(store.records) < 2:
                previous = store
                store = FrameStore(2, len(self.group_matches[group]), self.policy.device)
                for target, source in (
                    (store.records, previous.records),
                    (store.actions, previous.actions),
                    (store.rewards, previous.rewards),
                    (store.flags, previous.flags),
                ):
                    target[0].copy_(source[0])
                self.stores[group] = store
            ring = self.rings[group]
            for shard in shards:
                odd = np.zeros(len(shard.match_ids), dtype=np.bool_)
                for lane, match in enumerate(shard.match_ids):
                    offset = int(self.sampler.rng.uniform("head_start", match, 0, 0) * (maximum + 1))
                    offsets.append(offset)
                    start = end - offset
                    at = self.locations[match][1]
                    state = self.matches[match]
                    state.episode_start = start
                    self._publish_episode(group, at, match, 0, state.episode, start)
                    ring.length_device[at] = torch.tensor(
                        [p.timing.delay + (-start) % p.timing.replan for p in state.episode.players],
                        device=self.policy.device,
                    )
                    odd[lane] = bool(start % 2)
                if odd.any():
                    neutral = ring.controllers[self.device_lanes[id(shard)], :, 0].unsqueeze(0)
                    frames = shard.step(neutral, odd)
                    if len(frames) != 1 or ((frames[0].terminal | frames[0].timeout) & odd).any():
                        raise ValueError("initial neutral head-start frame must remain live")
                    source = torch.from_numpy(np.flatnonzero(odd)).to(self.policy.device)
                    lanes = self.device_lanes[id(shard)][source]
                    store.write(torch.ones_like(lanes), lanes, frames[0], source)
                    self.stops[group][lanes] += 1
                    self.observed_clocks[group][lanes] += 1
                    ring.head[lanes] = 1
                    ring.length_device[lanes] -= 1
            self.active[group].zero_()
            self.live_host[group].fill(False)
        return tuple(offsets)

    def _reset(self, shard: SimulatorShard, lane: int, match: int, clock: int, number: int) -> None:
        if clock % 128:
            raise ValueError("episodes must start on a block boundary")
        episode = self.sampler.sample(match, number, self.lane_roles[match])
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
        self.behavior_tracker.reset(np.array([2 * match, 2 * match + 1], dtype=np.int64))
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

    def _dense_metadata(
        self, clock: int | Tensor, update: int | Tensor, *, bootstrap: bool = False, slots: Tensor | None = None
    ) -> Tensor:
        group = self.cursor
        meta = self.dense_meta[group].clone() if slots is None else self.dense_meta[group].index_select(0, slots)
        lanes = meta[:, LANE]
        observed = self.observed_clocks[group][lanes]
        if bootstrap:
            clocks = observed
        elif isinstance(clock, Tensor):
            clocks = clock.expand_as(observed)
        else:
            clocks = torch.full_like(observed, clock)
        block = ((clocks - int(bootstrap)) // 128) * 128
        start = torch.maximum(self.episode_clocks[group][lanes], block - 128)
        start = torch.maximum(start, clocks - 255)
        # Inactive rows still occupy slots. Their observations stop at the end.
        length = (observed - start + 1).clamp(1, 256)
        meta[:, CLOCK] = clocks
        meta[:, UPDATE] = update
        meta[:, BLOCK] = block
        meta[:, START] = self.stops[group][lanes] - length
        meta[:, LENGTH] = length
        meta[:, OPP_LENGTH] = self.rings[group].length_device[lanes, 1 - meta[:, SIDE]]
        return meta

    def _stage_tick_tensors(self, control: Tensor, table: DenseRollout, pending: Tensor, slots: Tensor) -> Tensor:
        clock, update, start = control.unbind()
        group = self.cursor
        players = len(self.dense_meta[group])
        all_at = self.player_slots[group] + (clock - start).div(2, rounding_mode="floor") * players
        at = all_at[slots]
        meta = self._dense_metadata(clock, update, slots=slots)
        table.integers[at] = meta
        table.age[at] = meta[:, CLOCK] - self.episode_clocks[group].repeat_interleave(2)[slots]
        table.remaining[at] = -1
        table.conditions[at] = self.dense_logs[group][slots]
        table.coefficients[at] = self.dense_logs[group][slots, 0].exp()
        ring = self.rings[group]
        lanes, sides = meta[:, LANE], meta[:, SIDE]
        offsets = torch.arange(24, device=self.policy.device)[None]
        queue = ring.read(lanes, sides, min(24, ring.capacity))
        queue = torch.nn.functional.pad(queue, (0, 0, 0, 24 - queue.shape[1]))
        table.prefix[at] = torch.where((offsets < meta[:, DELAY, None])[..., None], queue, 0)
        opponent = queue.reshape(-1, 2, 24, 4).flip(1).flatten(0, 1)
        table.opponent[at] = torch.where((offsets < meta[:, OPP_LENGTH, None])[..., None], opponent, 0)
        active = self.active[group].repeat_interleave(2)[slots]
        table.active[at] = active
        due = active & (clock % meta[:, REPLAN] == 0)
        table.queue_error.logical_or_((due & (ring.length_device[lanes, sides] != meta[:, DELAY])).any())
        student = self.owners[group][slots] < 0
        table.valid[at] = due & student
        previous = torch.where(pending[slots] >= 0, pending[slots], slots)
        continuing = due & student & (pending[slots] >= 0)
        table.following[previous] = torch.where(continuing, at, table.following[previous])
        pending[slots] = torch.where(due & student, at, pending[slots])
        return all_at

    @torch.no_grad()
    def _stage_tick(
        self,
        clock: int,
        update: int,
        table: DenseRollout,
        pending: Tensor,
        *,
        buffer: RolloutBuffer | None = None,
    ) -> Tensor:
        group = self.cursor
        players = len(self.dense_meta[group])
        tick = (clock - self.clocks[group]) // 2
        slots = self.player_slots[group]
        if buffer is not None:
            if clock % 128 == 0:
                raise ValueError("a block boundary must stage every buffer")
            slots = self.buffer_player_slots[id(buffer)]
        control = torch.tensor(
            (clock, update, self.clocks[group]),
            dtype=torch.long,
            pin_memory=self.policy.device.type == "cuda",
        ).to(self.policy.device, non_blocking=True)
        preparing = self.policy.device.type == "cuda" and (
            not isinstance(self.policy, TransformerPolicy) or not self.policy.startup_finished
        )
        # Resume controls can create more engine variants than Torch's default
        # budget. Keep the normal budget after production startup.
        with (
            prepared_shape_compilation(64, instance_count=4 * len(self.groups))
            if preparing
            else contextlib.nullcontext()
        ):
            all_at = self.stage_tick_tensors(control, table, pending, slots)
        if clock % 128 == 0:
            # Opponents change only at resets on block boundaries.
            table.opponent_anchor[tick * players : (tick + 64) * players].view(64, players).copy_(
                self.owners[group].reshape(-1, 2).flip(1).flatten()
            )
        ring = self.rings[group]
        assignments = self.assignments[group]
        anchor_ids = tuple(self.anchors)
        if clock % 128 == 0 and (
            not np.array_equal(assignments, self.layout_assignments[group]) or self.layout_anchors[group] != anchor_ids
        ):
            self.layout_anchors[group] = anchor_ids
            self.tick_layouts.clear()
            self.group_slots[group].clear()
            self.layout_assignments[group][:] = assignments
            logical = np.asarray(self.group_matches[group], dtype=np.int64).repeat(2) * 2 + np.tile(
                (0, 1), len(self.group_matches[group])
            )
            # The student runs per buffer, so each buffer's simulator step overlaps the next
            # buffer's student graph. Anchor rows are few and latency-bound, so each anchor
            # runs once per tick over its rows in every buffer.
            student_layouts = tuple(
                (buffer.index, np.isin(np.arange(players) // 2, buffer.lanes), -1) for buffer in self.buffers[group]
            )
            anchor_layouts = tuple((ALL_BUFFERS, np.ones(players, dtype=np.bool_), owner) for owner in anchor_ids)
            for buffer_index, in_buffer, owner in (*student_layouts, *anchor_layouts):
                # Both ports occupy slots; only the episode's owner writes queues.
                owned = np.array(
                    [owner == -1 or self.lane_roles[match] == owner for match in self.group_matches[group]]
                ).repeat(2)
                selected = np.flatnonzero(in_buffer & owned)
                if not len(selected):
                    continue
                anchor = None if owner < 0 else int(owner)
                key = buffer_index, anchor
                self.group_slots[group][key] = torch.from_numpy(selected).to(self.policy.device)
                self.tick_layouts[key] = TickLayout(
                    tuple(logical[selected].tolist()),
                    tuple((selected // 2).tolist()),
                    buffer_index,
                )
            stack = self.anchor_stacks[group]
            if stack is not None:
                keys = tuple((ALL_BUFFERS, owner) for owner in anchor_ids if (ALL_BUFFERS, owner) in self.tick_layouts)
                layouts = tuple(self.tick_layouts.pop(key) for key in keys)
                slots = tuple(self.group_slots[group].pop(key) for key in keys)
                stack.validate_layout(
                    tuple(self.anchors[owner] for _, owner in keys),
                    tuple(len(layout.streams) for layout in layouts),
                )
                key = ALL_BUFFERS, STACKED_ANCHORS
                self.group_slots[group][key] = torch.cat(slots)
                self.tick_layouts[key] = TickLayout(
                    tuple(stream for layout in layouts for stream in layout.streams),
                    tuple(lane for layout in layouts for lane in layout.lanes),
                    ALL_BUFFERS,
                )
                self.stacked_owners[group] = torch.tensor(
                    [owner for (_, owner), layout in zip(keys, layouts, strict=True) for _ in layout.streams],
                    device=self.policy.device,
                )
            if self.policy.device.type == "cuda":
                layout_ready = torch.cuda.Event()
                layout_ready.record()
                layout_ready.synchronize()
            for owner, policy in (
                (None, self.policy),
                *self.anchors.items(),
                *(() if stack is None else ((STACKED_ANCHORS, stack),)),
            ):
                layouts = tuple(layout for (_, anchor), layout in self.tick_layouts.items() if anchor == owner)
                if isinstance(policy, TransformerPolicy):
                    policy.tick_compile_instances = max(policy.tick_compile_instances, 4 * len(self.groups))
                policy.retain_tick_layouts(ring, layouts)
        if clock % 128 == 0:
            # Ownership and cadence change only at block-boundary episode resets.
            self.tick_decoder_candidates.clear()
            for key, layout in self.tick_layouts.items():
                coordinates = np.asarray(
                    [
                        (row, lane, self.matches[stream // 2].episode.players[stream % 2].timing.replan)
                        for row, (stream, lane) in enumerate(zip(layout.streams, layout.lanes, strict=True))
                        if self.assignments[group][2 * lane + stream % 2]
                        == (
                            self.lane_roles[stream // 2]
                            if key[1] == STACKED_ANCHORS
                            else (-1 if key[1] is None else key[1])
                        )
                    ],
                    dtype=np.int64,
                ).reshape(-1, 3)
                every_tick = coordinates[coordinates[:, 2] == 2, :2]
                both_cadences = coordinates[:, :2]
                every_tick.flags.writeable = both_cadences.flags.writeable = False
                self.tick_decoder_candidates[key] = both_cadences, every_tick
        phase = int(clock % 4 != 0)
        self.tick_decoder_rows = {}
        for key, candidates in self.tick_decoder_candidates.items():
            coordinates = candidates[phase]
            live = self.live_host[group][coordinates[:, 1]]
            self.tick_decoder_rows[key] = tuple(coordinates[live, 0].tolist())
        return all_at

    def _sample_buffer(
        self,
        buffer: RolloutBuffer,
        clock: int,
        table: DenseRollout,
        ids: Tensor,
        *,
        concurrent: bool = False,
        sample_student: bool = True,
    ) -> None:
        first = buffer.index == self.buffers[self.cursor][0].index
        anchors = [key for key in self.tick_layouts if first and key[0] == ALL_BUFFERS]
        students = [key for key in self.tick_layouts if sample_student and key[0] == buffer.index]
        if not (concurrent and anchors):
            for key in (*anchors, *students):
                self._sample_layout(key, clock, table, ids)
            return
        # Each graph writes only the rows it owns and sends the rest to spare rows, so the
        # anchors run beside the student. They fork before the student's launch so they never
        # wait for it, and they join before this buffer's controller inputs go to the simulator.
        current = torch.cuda.current_stream(self.policy.device)
        side = []
        for key in anchors:
            stream = self.anchor_streams.get(key)
            if stream is None:
                stream = self.anchor_streams[key] = torch.cuda.Stream(device=self.policy.device)
            stream.wait_stream(current)
            side.append(stream)
        for key in students:
            self._sample_layout(key, clock, table, ids)
        for key, stream in zip(anchors, side, strict=True):
            with torch.cuda.stream(stream):
                self._sample_layout(key, clock, table, ids)
        for stream in side:
            current.wait_stream(stream)

    def _sample_layout(self, key: tuple[int, int | None], clock: int, table: DenseRollout, ids: Tensor) -> None:
        buffer_index, anchor = key
        layout = self.tick_layouts[key]
        policy = (
            self.policy
            if anchor is None
            else cast(StackedAnchorPolicy, self.anchor_stacks[self.cursor])
            if anchor == STACKED_ANCHORS
            else self.anchors[anchor]
        )
        selected = ids[self.group_slots[self.cursor][key]]
        owner = -1 if anchor is None else anchor
        enabled = self.owners[self.cursor][self.group_slots[self.cursor][key]] == (
            self.stacked_owners[self.cursor] if anchor == STACKED_ANCHORS else owner
        )
        rows = TickInputs(table, selected, enabled, self.tick_decoder_rows[key])
        with (
            torch.profiler.record_function(f"o65_graph/buffer{buffer_index}/policy{owner}/rows{len(layout.streams)}")
            if self.profile is not None and self.profile.active_trace
            else contextlib.nullcontext()
        ):
            policy.sample_tick(rows, self.sampler.rng, layout, self.rings[self.cursor], clock=clock)

    def _tick_ready(self, buffer: RolloutBuffer, clock: int) -> bool:
        ring = self.rings[self.cursor]
        first = buffer.index == self.buffers[self.cursor][0].index
        for (buffer_index, anchor), layout in self.tick_layouts.items():
            if buffer_index != buffer.index and not (first and buffer_index == ALL_BUFFERS):
                continue
            policy = (
                self.policy
                if anchor is None
                else cast(StackedAnchorPolicy, self.anchor_stacks[self.cursor])
                if anchor == STACKED_ANCHORS
                else self.anchors[anchor]
            )
            if not policy.tick_ready(layout, ring, clock=clock):
                return False
        return True

    def _prefetch_tick(
        self,
        clock: int,
        end_clock: int,
        update: int,
        table: DenseRollout,
        pending: Tensor,
        *,
        ready: bool,
    ) -> torch.cuda.Event | None:
        buffers = self.buffers[self.cursor]
        if (
            not ready
            or self.policy.device.type != "cuda"
            or self.staggering
            or len(buffers) != 2
            or clock >= end_clock
            or clock % 128 == 0
        ):
            return None
        first, last = buffers
        if not self._tick_ready(first, clock):
            return None
        first_stream = cast(torch.cuda.Stream, first.stream)
        completion_stream = cast(torch.cuda.Stream, last.stream)
        # The first buffer is complete. Its next student graph can cover the
        # second simulator's tail, before that buffer's observations are ready.
        with torch.cuda.stream(completion_stream):
            ids = self._stage_tick(clock, update, table, pending, buffer=first)
        ids.record_stream(first_stream)
        first_stream.wait_stream(completion_stream)
        with (
            torch.cuda.stream(first_stream),
            torch.profiler.record_function(f"o65_prefetch/clock{clock}")
            if self.profile is not None and self.profile.active_trace
            else contextlib.nullcontext(),
        ):
            self._sample_layout((first.index, None), clock, table, ids)
            done = torch.cuda.Event()
            done.record(first_stream)
        return done

    @torch.no_grad()
    def prepare_frame_acceptance(self, table: DenseRollout) -> None:
        """Compile both native frame layouts without changing the live rollout."""
        group = self.cursor
        trial = copy.copy(table)
        trial.store = copy.deepcopy(table.store)
        trial.rewards = table.rewards.clone()
        trial.duration = table.duration.clone()
        ring = copy.copy(self.rings[group])
        ring.head = ring.head.clone()
        ring.length_storage = ring.length_storage.clone()
        stops = self.stops[group].clone()
        observed = self.observed_clocks[group].clone()
        returns = self.episode_returns[group].clone()
        pending = torch.full_like(self.player_slots[group], -1)
        maximum = max(len(buffer.lanes) for buffer in self.buffers[group])
        for count in sorted({1, min(2, maximum), maximum}):
            lanes = torch.arange(count, device=self.policy.device)
            packed = torch.zeros(
                count, native_shard.STAGING_DTYPE.itemsize, device=self.policy.device, dtype=torch.uint8
            )
            records = packed[:, : native.FRAME_DTYPE.itemsize]
            actions = packed[:, native.FRAME_DTYPE.itemsize :].view(torch.float32).reshape(count, 2, ACTION_DIM)
            columns = np.zeros((count, len(native.OBSERVATION_FIELDS)), dtype=np.uint32)
            columns.flags.writeable = False
            for contiguous in (False, True):
                frames = ShardFrames(
                    torch.empty_like(records, memory_format=torch.contiguous_format).copy_(records)
                    if contiguous
                    else records,
                    torch.empty_like(actions, memory_format=torch.contiguous_format).copy_(actions)
                    if contiguous
                    else actions,
                    columns,
                    np.zeros((count, 2), dtype=np.float32),
                    np.zeros(count, dtype=np.bool_),
                    np.zeros(count, dtype=np.bool_),
                    np.zeros((count, 2, ACTION_DIM), dtype=np.float32),
                    np.zeros(count, dtype=np.int32),
                )
                self.accept_frame(frames, lanes, pending, trial, ring, self.active[group], stops, observed, returns)

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
        # Convert submitted inputs to observed controller state in place, before the device copy.
        applied = staging.host_output["applied"]
        applied[...] = native_controller_readback(applied)
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
                            host["applied"].copy(),
                            records["frame_id"].copy(),
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

    @torch.no_grad()
    def _accept(
        self,
        buffer: RolloutBuffer,
        shards: tuple[SimulatorShard, ...],
        batches: tuple[ShardFrames, ...],
        end_clock: int,
        update: int,
        table: DenseRollout,
        pending: Tensor,
        *,
        behavior: list[tuple[NDArray[np.uint32], NDArray[np.int64], NDArray[np.int32]]] | None = None,
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
        timeouts = 0
        active_host = self.live_host[group][host_lanes]
        terminals, truncations = frames.terminal, frames.timeout
        self.accept_frame(
            frames,
            lanes,
            pending,
            table,
            ring,
            self.active[group],
            self.stops[group],
            self.observed_clocks[group],
            self.episode_returns[group],
        )
        frames_stepped = int(active_host.sum())
        for shard in shards:
            if isinstance(shard, NativeSimulatorShard):
                shard.stepped_frames += int(self._active(shard).sum())
        observed = frames.columns[active_host], host_lanes[active_host], frames.frame_ids[active_host]
        if behavior is None:
            self.observe_behavior(*observed)
        else:
            behavior.append(observed)
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
        meta = self._dense_metadata(clock, update, bootstrap=True, slots=slots)
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

    def truncate_slot(self, slot: int) -> tuple[tuple[int, int], ...]:
        """End old-weight episodes at their saved boundary values before promotion."""
        truncated = []
        for match, role in self.lane_roles.items():
            if role != slot:
                continue
            state = self.matches[match]
            if state.end_clock is not None:
                continue
            group, lane = self.locations[match]
            # The preceding rollout ended with a nonterminal bootstrap and a trace
            # boundary. Reset on the next visit uses that same truncation contract.
            state.end_clock = self.clocks[group]
            self.active[group][lane] = False
            self.live_host[group][lane] = False
            truncated.append((match, state.episode.number))
        return tuple(truncated)

    def _reset_waiting(self, shards: Sequence[SimulatorShard], clock: int) -> int:
        if clock % 128 and not self.staggering:
            return 0
        resets = 0
        for shard in shards:
            waiting = np.flatnonzero(~self._active(shard))
            for lane in waiting:
                match = shard.match_ids[lane]
                state = self.matches[match]
                at = self.locations[match][1]
                if state.end_clock is None:
                    if clock >= state.episode_start + state.episode_start % 2:
                        self.active[self.cursor][at] = True
                        self.live_host[self.cursor][at] = True
                elif clock % 128 == 0:
                    self._reset(shard, int(lane), match, clock, state.episode.number + 1)
                    resets += 1
        return resets

    def observe_behavior(
        self, columns: NDArray[np.uint32], lanes: NDArray[np.int64], frame_ids: NDArray[np.int32]
    ) -> None:
        if not len(lanes):
            return
        group = self.cursor
        matches = np.asarray(self.group_matches[group], dtype=np.int64)[lanes]
        streams = (2 * matches[:, None] + np.arange(2)[None]).ravel()

        def player_bits(suffix: str) -> NDArray[np.uint32]:
            return np.stack([columns[:, NATIVE_COLUMN_LAYOUT[f"p{side}_{suffix}"][0]] for side in (1, 2)], -1)

        stage = columns[:, NATIVE_COLUMN_LAYOUT["stage"][0]].view(np.int32).astype(np.int64)
        tracker = self.behavior_tracker
        tracker.observe(
            streams,
            player_bits("action").view(np.int32).astype(np.int64),
            player_bits("stock").view(np.int32).astype(np.int64),
            player_bits("percent").view(np.float32).astype(np.float64),
            player_bits("position_x").view(np.float32).astype(np.float64),
            stage,
            playing=frame_ids >= 0,
        )
        counts = tracker.counts[streams]
        roles = self.assignments[group].reshape(-1, 2)[lanes]
        for role, mask in (
            ("student", roles < 0),
            ("student_pi_ref", (roles < 0) & (roles[:, ::-1] == 0)),
            ("pi_ref", roles == 0),
        ):
            self.behavior_totals[role] += counts[mask.ravel()].sum(0)
        tracker.counts[streams] = 0

    def _active(self, shard: SimulatorShard) -> NDArray[np.bool_]:
        return self.live_host[self.cursor][self.shard_lanes[id(shard)]]

    def collect(self, update: int) -> Rollout:
        if not self._initialized:
            raise ValueError("initialize or restore the pool before rollout")
        started = time.perf_counter()
        for policy in self.execution_policies():
            if isinstance(policy, TransformerPolicy):
                policy.prepare_tick_weights()
        self.results = []
        self.behavior_totals = {
            role: np.zeros(len(COUNT_NAMES), dtype=np.int64) for role in ("student", "student_pi_ref", "pi_ref")
        }
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
        table.store = store
        table.reset()
        pending = torch.full((2 * len(matches),), -1, dtype=torch.long, device=self.policy.device)
        resets = timeouts = frames_stepped = 0
        buffers = self.buffers[group_id]
        default_stream = torch.cuda.current_stream(self.policy.device) if self.policy.device.type == "cuda" else None
        prefetched: torch.cuda.Event | None = None
        with ThreadPoolExecutor(max_workers=len(shards)) as workers:
            for clock in range(start, start + count, 2):
                had_prefetch = prefetched
                prefetched = None
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
                            if not (had_prefetch and buffer is buffers[0]):
                                default_stream.wait_stream(cast(torch.cuda.Stream, buffer.stream))
                    resets += self._reset_waiting(shards, clock)
                    with sample(self.profile, "tick_state", work=float(len(matches)), limit=0):
                        ids = self._stage_tick(
                            clock, update, table, pending, buffer=buffers[1] if had_prefetch else None
                        )
                    ready = all(self._tick_ready(buffer, clock) for buffer in buffers)
                    futures: list[PendingStep] = []
                    previous_stream: torch.cuda.Stream | None = None
                    for buffer in buffers:
                        inference_stream = buffer.stream if ready else default_stream
                        if ready and buffer.stream is not None:
                            buffer.stream.wait_stream(cast(torch.cuda.Stream, default_stream))
                            # Each buffer's graphs follow the previous buffer's, so its inputs
                            # reach the simulator while the GPU runs the next buffer.
                            if previous_stream is not None:
                                if had_prefetch is not None:
                                    # The first student already ran. Its anchors now run
                                    # beside this student, before either simulator starts.
                                    buffer.stream.wait_event(had_prefetch)
                                else:
                                    buffer.stream.wait_stream(previous_stream)
                            previous_stream = buffer.stream
                        with (
                            torch.cuda.stream(inference_stream)
                            if inference_stream is not None
                            else contextlib.nullcontext()
                        ):
                            with sample(self.profile, "gpu_tick", work=float(len(buffer.lanes)), limit=0):
                                self._sample_buffer(
                                    buffer,
                                    clock,
                                    table,
                                    ids,
                                    concurrent=ready and buffer.stream is not None,
                                    sample_student=not (had_prefetch and buffer is buffers[0]),
                                )
                            if ready:
                                if had_prefetch is not None and buffer is buffers[1]:
                                    # Sampling can overlap the anchors, but both ports'
                                    # committed actions must precede the controller copy.
                                    cast(torch.cuda.Stream, buffer.stream).wait_stream(
                                        cast(torch.cuda.Stream, buffers[0].stream)
                                    )
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
                    # Completions follow every current-tick graph on the last buffer's stream.
                    # Only the next first-buffer graph can overlap them; it reads and writes
                    # disjoint buffer rows.
                    completion_stream = buffers[-1].stream
                    for buffer in buffers:
                        behavior: list[tuple[NDArray[np.uint32], NDArray[np.int64], NDArray[np.int32]]] = []
                        staging = self.staging.get(id(buffer))
                        if staging is not None:
                            staging.release_inputs()
                        with (
                            torch.cuda.stream(completion_stream)
                            if completion_stream is not None
                            else contextlib.nullcontext(),
                            torch.profiler.record_function(f"o65_host/complete/buffer{buffer.index}")
                            if self.profile is not None and self.profile.active_trace
                            else contextlib.nullcontext(),
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
                            torch.cuda.stream(completion_stream)
                            if completion_stream is not None
                            else contextlib.nullcontext(),
                            torch.profiler.record_function(f"o65_host/accept/buffer{buffer.index}")
                            if self.profile is not None and self.profile.active_trace
                            else contextlib.nullcontext(),
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
                                    behavior=behavior,
                                )
                                frames_stepped += stepped
                                timeouts += ended_timeouts
                        if buffer is buffers[0]:
                            prefetched = self._prefetch_tick(
                                clock + 2, start + count, update, table, pending, ready=ready
                            )
                        for observed in behavior:
                            self.observe_behavior(*observed)
                    if default_stream is not None:
                        for buffer in buffers:
                            if not (prefetched and buffer is buffers[0]):
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
            tuple((role, BehaviorCounts(counts)) for role, counts in self.behavior_totals.items()),
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
        np.concatenate([frame.frame_ids for frame in frames]),
    )


@dataclasses.dataclass(frozen=True, slots=True)
class TrainConfig:
    seed: int = 65
    reward_weights: native.RewardWeights = native.RewardWeights()
    actor_optimizer: Literal["muon", "adam"] = "muon"
    actor_lr: float = 1e-5
    critic_lr: float = 3e-4
    critic_epochs: int = 4
    windows_per_step: int = 512
    # Each actor step and the reference pass run in this many window chunks,
    # because every captured CUDA graph keeps a private pool at its peak size.
    actor_microbatches: int = 1
    decisions_per_window: int = 16
    clip_epsilon: float = 0.2
    gradient_clip: float = 1.0
    value_scale: float = 150.0
    warmup_steps: int = 32
    critic_warmup_updates: int = 32
    head_start_frames: int = 0
    lambda_spread_multiplier: float = 3.0
    snapshot_interval: int = 256
    snapshot_capacity: int = 3
    checkpoint_interval: int = 32
    deterministic: bool = False

    def __post_init__(self) -> None:
        if type(self.snapshot_capacity) is not int or self.snapshot_capacity != 3:
            raise ValueError("fixed opponent layout requires exactly three snapshot slots")
        for name in (
            "critic_epochs",
            "windows_per_step",
            "actor_microbatches",
            "decisions_per_window",
            "snapshot_interval",
            "snapshot_capacity",
            "checkpoint_interval",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 1 <= self.decisions_per_window <= 16:
            raise ValueError("at most 16 decisions can train per window")
        if self.windows_per_step % self.actor_microbatches:
            raise ValueError("actor microbatches must divide windows per step")
        for value in (self.warmup_steps, self.critic_warmup_updates, self.head_start_frames):
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
        if not 0 < self.clip_epsilon < 1:
            raise ValueError("invalid clip epsilon")
        if self.actor_optimizer not in ("muon", "adam"):
            raise ValueError("unsupported actor optimizer")


@dataclasses.dataclass(frozen=True, slots=True)
class HeadStart:
    offsets: tuple[int, ...]
    episode_ages: tuple[int, ...]
    match_frames: int
    seconds: float

    def metrics(self) -> dict[str, float]:
        return {
            "systems/head_start_seconds": self.seconds,
            "systems/head_start_match_fps": self.match_frames / self.seconds,
            **{
                f"coverage/head_start_age_{name}": float(value)
                for name, value in zip(
                    ("min", "p25", "median", "p75", "max"),
                    np.quantile(self.episode_ages, [0, 0.25, 0.5, 0.75, 1]),
                    strict=True,
                )
            },
        }


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
        self.measure_all = False
        self.peaks: dict[str, tuple[int, int]] = {}
        self.seconds: dict[str, float] = {}
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
        started = time.perf_counter()
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
                torch.cuda.synchronize(self.device)
                self.seconds[f"{self.label}_{name}"] = time.perf_counter() - started
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
            **{f"systems/{name}_seconds": seconds for name, seconds in self.seconds.items()},
            **{
                f"memory/{name}_peak_{kind}_bytes": float(value)
                for name, values in self.peaks.items()
                for kind, value in zip(("allocated", "reserved"), values, strict=True)
            },
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
    """Estimate gradient noise from pre-clip gradients and trained counts.

    Weights change between samples, so this is a training diagnostic.
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
    model: SelfPlayTransformer,
    old: PolicyEvaluation,
    reference: PolicyEvaluation,
    executed: Tensor,
    *,
    totals: distributed.MetricTotals | None = None,
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
    if totals is not None:
        count = float(executed[:, ::4].sum())
        for key, value in metrics.items():
            totals.ratio(key, value * count, count)
    return metrics


@torch.no_grad()
def frame_components(table: DenseRollout, ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    meta = table.integers[ids]
    positions = meta[:, START] + meta[:, LENGTH] - 1
    lane = meta[:, LANE]
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
    return losses, damage, wins


@torch.no_grad()
def reward_component_metrics(
    rollout: Rollout,
    weights: native.RewardWeights = native.DEFAULT_REWARD_WEIGHTS,
    *,
    totals: distributed.MetricTotals | None = None,
) -> dict[str, float]:
    table = rollout.table
    meta = table.integers[rollout.ids]
    side = meta[:, SIDE]
    valid = torch.arange(4, device=meta.device)[None] < table.duration[rollout.ids, None]
    losses, damage, wins = frame_components(table, rollout.ids)
    result = {}
    rows = torch.arange(len(meta), device=meta.device)[:, None]
    for name, value, scale in (
        ("damage", damage, weights.damage),
        ("stock", losses, weights.stock),
        ("win", wins, weights.game_win),
    ):
        positive = value[rows, torch.arange(4, device=meta.device), 1 - side[:, None]] * scale
        negative = -value[rows, torch.arange(4, device=meta.device), side[:, None]] * scale
        result[f"return/{name}_positive"] = float(torch.where(valid, positive, 0).sum() / len(meta))
        result[f"return/{name}_negative"] = float(torch.where(valid, negative, 0).sum() / len(meta))
        result[f"return/{name}_reward"] = float(torch.where(valid, positive + negative, 0).sum() / len(meta))
    if totals is not None:
        for key, value in result.items():
            totals.ratio(key, value * len(meta), len(meta))
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
        self.metric_totals = distributed.MetricTotals(engine.policy.device)
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
        self.pool_start_update = 0
        self.warmup_start_update = 0
        self.warming = config.critic_warmup_updates > 0
        self.warmup_moments: dict[DelayReplan, tuple[int, float, float]] = {}
        if set(engine.anchors) != {0}:
            raise ValueError("fresh trainer requires only pi_ref")
        self.snapshot_slots = tuple(self.reference.frozen_copy() for _ in range(3))
        engine.anchors.update(enumerate(self.snapshot_slots, start=1))
        engine.prepare_anchor_stacks()
        self.slot_updates = {1: 0, 2: 0, 3: 0}
        self.compilation_counter: CompilationStartCounter | None = None
        self.startup_seconds = 0.0
        self.evaluation_cursor = 0
        self.updating = False
        self.episode_metrics: dict[tuple[int, int], tuple[float, float, float, float]] = {}
        self.partial_episodes: set[tuple[int, int]] = set()
        self.anchor_win_history: list[dict[int, tuple[int, int]]] = []

    @torch.no_grad()
    def capture_startup(self, *, rollout_only: bool = False) -> None:
        if not isinstance(self.policy, TransformerPolicy) or self.policy.device.type != "cuda":
            return
        policies = self.engine.execution_policies()
        if any(not isinstance(policy, TransformerPolicy) or policy.startup_finished for policy in policies):
            raise ValueError("startup capture requires unsealed transformer policies")
        started = time.perf_counter()
        engine = self.engine
        for policy in policies:
            assert isinstance(policy, TransformerPolicy)
            policy.prepare_tick_weights()
        cursor = engine.cursor
        frames = engine.sampler.schedule.rollout_frames
        if distributed.rank() == 0:
            print(
                f"startup_capture_started groups={len(engine.groups)} matches_per_group={[len(ids) for ids in engine.group_matches]}",
                flush=True,
            )
        for group, matches in enumerate(engine.group_matches):
            engine.cursor = group
            previous = engine.stores[group]
            if previous is None:
                raise ValueError("startup capture requires an initialized pool")
            store = FrameStore(129 + frames + frames // 128 + 1, len(matches), self.policy.device)
            for target, source in (
                (store.records, previous.records),
                (store.actions, previous.actions),
                (store.rewards, previous.rewards),
                (store.flags, previous.flags),
            ):
                target[: len(source)].copy_(source)
            engine.stores[group] = store
            table = DenseRollout(frames // 2, store)
            table.allocate_evaluation(self.policy.model)
            engine.tables[group] = table
            engine.tick_layouts = engine.group_layouts[group]
            pending = torch.full((2 * len(matches),), -1, dtype=torch.long, device=self.policy.device)
            for offset in (0, 2, 4):
                clock = engine.clocks[group] + offset
                with prepared_shape_compilation(64, instance_count=len(engine.groups)):
                    ids = engine._stage_tick(clock, self.update, table, pending)
                    if offset == 2:
                        for buffer in engine.buffers[group]:
                            engine._stage_tick(clock, self.update, table, pending, buffer=buffer)
                table.active[ids] = False
                engine.tick_decoder_rows = dict.fromkeys(engine.tick_layouts, ())
                for buffer in engine.buffers[group]:
                    engine._sample_buffer(buffer, clock, table, ids)
            with prepared_shape_compilation(64, instance_count=len(engine.groups)):
                engine.prepare_frame_acceptance(table)
            if rollout_only:
                table.reset()
                continue
            # Bootstrap and context-gap graphs have fixed player/diagnostic row counts.
            for count in {2 * len(matches), min(32, 2 * len(matches))}:
                rows = torch.zeros(count, 1, dtype=torch.long, device=self.policy.device)
                batch = replay_batch(self.policy, table, rows, torch.zeros_like(rows, dtype=torch.bool))
                self.policy.evaluate_values(batch)
            chunk = self.config.windows_per_step // self.config.actor_microbatches
            rows = torch.zeros(chunk, 64, dtype=torch.long, device=self.policy.device)
            batch = replay_batch(self.policy, table, rows, torch.zeros_like(rows, dtype=torch.bool))
            positions = torch.arange(self.config.decisions_per_window, device=self.policy.device)[None].expand(
                chunk, -1
            )
            order = torch.arange(chunk, device=self.policy.device)[None]
            self.reference.evaluate_reference(batch, positions, order)
            part = batch.take(
                order.flatten(), positions, torch.ones(chunk, dtype=torch.bool, device=self.policy.device)
            )
            assert isinstance(self.reference, TransformerPolicy)
            output = self.reference.window_forward(part.inputs(), grad=False)
            shape = part.slots.valid_WK.shape
            zero = torch.zeros(shape, device=self.policy.device)
            targets = ActorTargets(
                output.log_probs_WKF.float(),
                zero,
                torch.zeros_like(zero, dtype=torch.bool),
                zero,
                output.distributions,
                zero.new_zeros(()),
                zero.new_ones(()),
                zero.new_ones(()),
            )
            self.policy.actor_step(part, targets, clip_epsilon=self.config.clip_epsilon)
            table.reset()
        engine.cursor = cursor
        engine.tick_layouts = engine.group_layouts[cursor]
        torch.cuda.synchronize(self.policy.device)
        for policy in policies:
            assert isinstance(policy, TransformerPolicy)
            policy.startup_finished = True
        self.startup_seconds = time.perf_counter() - started
        if self.compilation_counter is not None:
            self.compilation_counter.starts = 0
        if distributed.rank() == 0:
            print(
                f"startup_capture seconds={self.startup_seconds:.3f} graphs={sum(p.capture_count for p in policies if isinstance(p, TransformerPolicy))}",
                flush=True,
            )

    @torch.no_grad()
    def head_start(self) -> HeadStart:
        if self.update or self.actor_steps or self.critic_optimizer.state or self.actor_optimizer.state:
            raise ValueError("head start must precede training")
        if _state_hash(self.policy.model.state_dict()) != _state_hash(self.reference.model.state_dict()):
            raise ValueError("head start requires pi_ref weights")
        started = time.perf_counter()
        offsets = self.engine.stagger_starts(self.config.head_start_frames)
        frames = self.engine.sampler.schedule.rollout_frames
        end = math.ceil(self.config.head_start_frames / frames) * frames
        stepped = sum(offset % 2 for offset in offsets)
        while min(self.engine.clocks) < end:
            rollout = self.engine.collect(-1)
            stepped += rollout.frames
            del rollout
        ages = []
        for group, matches in enumerate(self.engine.group_matches):
            for lane, match in enumerate(matches):
                state = self.engine.matches[match]
                age = int(self.engine.observed_clocks[group][lane]) - state.episode_start
                ages.append(age)
                if age:
                    self.partial_episodes.update(
                        (2 * match + side, state.episode.number)
                        for side, player in enumerate(state.episode.players)
                        if player.anchor is None
                    )
        self.engine.results.clear()
        self.engine.staggering = False
        if self.policy.device.type == "cuda":
            torch.cuda.synchronize(self.policy.device)
        return HeadStart(offsets, tuple(ages), stepped, time.perf_counter() - started)

    @torch.no_grad()
    def anchor_metrics(self, rollout: Rollout) -> dict[str, float]:
        """Measure training exposure and stockout wins from the student side."""
        table, ids = rollout.table, rollout.ids
        ids = ids[table.opponent_anchor[ids] >= 0]
        opponents = table.opponent_anchor[ids]
        side = table.integers[ids, SIDE]
        frames = torch.arange(4, device=ids.device)[None]
        valid = frames < table.duration[ids, None]
        losses, damage, _ = frame_components(table, ids)
        rows = torch.arange(len(ids), device=ids.device)[:, None]
        net = torch.stack(
            [
                value[rows, frames, 1 - side[:, None]] - value[rows, frames, side[:, None]]
                for value in (losses, damage)
            ],
            -1,
        )
        totals = torch.where(valid[..., None], net, 0).sum(1)
        completed: dict[int, tuple[int, int]] = {}
        for result in rollout.results:
            if result.timeout:
                continue
            for side_index, player in enumerate(result.episode.players):
                if player.anchor is None:
                    continue
                if result.stocks is None:
                    raise ValueError("completed anchor match is missing stocks")
                wins, count = completed.get(player.anchor, (0, 0))
                completed[player.anchor] = (
                    wins + int(result.stocks[1 - side_index] > result.stocks[side_index]),
                    count + 1,
                )
        self.anchor_win_history.append(completed)
        self.anchor_win_history = self.anchor_win_history[-32:]
        anchors = sorted(
            set(self.engine.anchors) | set(completed) | {key for row in self.anchor_win_history for key in row}
        )
        exposure = (
            torch.stack(
                [
                    torch.cat(
                        (totals[opponents == anchor].sum(0), table.duration[ids][opponents == anchor].sum()[None])
                    )
                    for anchor in anchors
                ]
            )
            .cpu()
            .tolist()
        )
        metrics: dict[str, float] = {
            f"gameplay/anchor{slot}_weights_update": float(update) for slot, update in self.slot_updates.items()
        }
        for anchor, (stocks, damage_total, duration) in zip(anchors, exposure, strict=True):
            prefix = "gameplay/pi_ref" if anchor == 0 else f"gameplay/anchor{anchor}"
            if duration:
                metrics[f"{prefix}_net_stocks_per_min"] = self.metric_totals.ratio(
                    f"{prefix}_net_stocks_per_min", stocks * 3600, duration
                )
                metrics[f"{prefix}_net_damage_per_min"] = self.metric_totals.ratio(
                    f"{prefix}_net_damage_per_min", damage_total * 3600, duration
                )
            wins, count = completed.get(anchor, (0, 0))
            metrics[f"{prefix}_count"] = self.metric_totals.total(f"{prefix}_count", count)
            if count:
                metrics[f"{prefix}_win_rate"] = self.metric_totals.ratio(f"{prefix}_win_rate", wins, count)
            recent_wins = sum(row.get(anchor, (0, 0))[0] for row in self.anchor_win_history)
            recent_count = sum(row.get(anchor, (0, 0))[1] for row in self.anchor_win_history)
            metrics[f"{prefix}_count_32_updates"] = self.metric_totals.total(
                f"{prefix}_count_32_updates", recent_count
            )
            if recent_count:
                metrics[f"{prefix}_win_rate_32_updates"] = self.metric_totals.ratio(
                    f"{prefix}_win_rate_32_updates", recent_wins, recent_count
                )
        return metrics

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
            if identity in self.partial_episodes:
                continue
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
            complete = True
            for side, player in enumerate(result.episode.players):
                if player.anchor is not None:
                    continue
                identity = (2 * result.match + side, result.episode.number)
                if identity in self.partial_episodes:
                    self.partial_episodes.remove(identity)
                    complete = False
                    continue
                discounted_return, cost, value, discounted_cost = self.episode_metrics.pop(identity)
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
            if complete:
                costs_done.append(match_cost)
        metrics["critic/monte_carlo_episode_count"] = float(len(values))
        if values:
            v, target = torch.tensor(values, dtype=torch.float64), torch.tensor(returns, dtype=torch.float64)
            metrics["critic/monte_carlo_initial_value"] = self.metric_totals.mean(
                "critic/monte_carlo_initial_value", v
            )
            metrics["critic/monte_carlo_task_return"] = self.metric_totals.ratio(
                "critic/monte_carlo_task_return", sum(task_returns), len(task_returns)
            )
            metrics["critic/monte_carlo_return"] = self.metric_totals.mean("critic/monte_carlo_return", target)
            metrics["critic/monte_carlo_rmse"] = math.sqrt(
                self.metric_totals.ratio(
                    "critic/monte_carlo_rmse", float((v - target).square().sum()), v.numel(), kind="rmse"
                )
            )
        if costs_done:
            metrics["return/completed_match_kl_cost"] = self.metric_totals.ratio(
                "return/completed_match_kl_cost", sum(costs_done), len(costs_done)
            )
        for index, rows in bins.items():
            metrics[f"return/completed_episode_u{index}"] = self.metric_totals.ratio(
                f"return/completed_episode_u{index}", sum(row[0] for row in rows), len(rows)
            )
            metrics[f"gameplay/win_rate_u{index}"] = self.metric_totals.ratio(
                f"gameplay/win_rate_u{index}", sum(row[1] for row in rows), len(rows)
            )
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
        old = _take_evaluation(table.evaluation, ids)
        with (
            self.memory.phase("reference"),
            phase(self.profile, "reference"),
            sample(self.profile, "windows", work=float(rows.numel())),
        ):
            reference = (
                None
                if self.warming and self.warmup_start_update == 0
                else window_evaluation(
                    self.reference.evaluate_reference(
                        windows,
                        placement.positions,
                        placement.windows.reshape(-1, self.config.windows_per_step // self.config.actor_microbatches),
                    ),
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
        step_count = distributed.maximum(len(placement.windows), device)
        step_windows = torch.full((step_count, self.config.windows_per_step), -1, dtype=torch.long, device=device)
        step_windows[: len(placement.windows)] = placement.windows
        steps = tuple(step_windows.unbind(0))
        executed = decisions.executed_factors_JF
        metrics = {
            "critic/explained_variance": self.metric_totals.explained_variance(
                "critic/explained_variance", values, targets
            ),
            "policy/kl_per_frame": self.metric_totals.ratio(
                "policy/kl_per_frame", float(torch.where(executed, kl, 0).sum()), float(executed[:, ::4].sum())
            ),
            "return/interval_reward": self.metric_totals.mean("return/interval_reward", frame_rewards.sum(-1)),
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
                metrics[f"return/interval_reward_lambda{index}"] = self.metric_totals.mean(
                    f"return/interval_reward_lambda{index}", frame_rewards[mask].sum(-1)
                )
                metrics[f"policy/kl_per_frame_lambda{index}"] = self.metric_totals.ratio(
                    f"policy/kl_per_frame_lambda{index}",
                    float(torch.where(executed[mask], kl[mask], 0).sum()),
                    float(executed[mask, ::4].sum()),
                )
        for position in range(4):
            mask = (decisions.decision_frame_J % 128) // 32 == position
            if bool(mask.any()):
                metrics[f"critic/explained_variance_position{position}"] = self.metric_totals.explained_variance(
                    f"critic/explained_variance_position{position}", values[mask], targets[mask]
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
            metrics["critic/bootstrap_context_gap"] = self.metric_totals.mean(
                "critic/bootstrap_context_gap", difference[final_valid]
            )
        for timing in self.engine.sampler.schedule.configurations:
            mask = (decisions.delay_J == timing.delay) & (decisions.replan_J == timing.replan)
            if not bool(mask.any()):
                continue
            suffix = f"d{timing.delay}_r{timing.replan}"
            metrics[f"policy/kl_per_frame_{suffix}"] = self.metric_totals.ratio(
                f"policy/kl_per_frame_{suffix}",
                float(torch.where(executed[mask], kl[mask], 0).sum()),
                float(executed[mask, ::4].sum()),
            )
            metrics[f"critic/advantage_mean_{suffix}"] = self.metric_totals.mean(
                f"critic/advantage_mean_{suffix}", advantages[mask]
            )
            metrics[f"critic/advantage_std_{suffix}"] = self.metric_totals.std(
                f"critic/advantage_std_{suffix}", advantages[mask]
            )
            metrics[f"critic/explained_variance_{suffix}"] = self.metric_totals.explained_variance(
                f"critic/explained_variance_{suffix}", values[mask], targets[mask]
            )
        for factor, distribution in zip(FACTOR_ORDER, old.distributions, strict=True):
            entropy = -(distribution * torch.where(distribution > 0, distribution, 1).log()).sum(-1)
            valid = torch.arange(4, device=device)[None] < decisions.replan_J[:, None]
            metrics[f"policy/entropy_{factor}"] = self.metric_totals.mean(f"policy/entropy_{factor}", entropy[valid])
        for index, factor in enumerate(FACTOR_ORDER):
            frames = executed[:, index::4]
            if bool(frames.any()):
                metrics[f"policy/kl_{factor}_per_frame"] = self.metric_totals.mean(
                    f"policy/kl_{factor}_per_frame", kl[:, index::4][frames]
                )
        if reference is not None and isinstance(self.policy, TransformerPolicy):
            metrics.update(action_statistics(self.policy.model, old, reference, executed, totals=self.metric_totals))
        metrics.update(reward_component_metrics(rollout, self.config.reward_weights, totals=self.metric_totals))

        ages = table.age[ids]
        age_bins = torch.bucketize(ages, ages.new_tensor((128, 512, 2048, 8192)))
        for index, value in enumerate(torch.bincount(age_bins, minlength=5).float() / len(ids)):
            metrics[f"data/episode_age_bin{index}"] = self.metric_totals.ratio(
                f"data/episode_age_bin{index}", float(value) * len(ids), len(ids)
            )
        metrics["data/episode_age_mean_frames"] = self.metric_totals.mean("data/episode_age_mean_frames", ages)
        u_bins = torch.bucketize(table.conditions[ids, 2], table.conditions.new_tensor((0.5, 0.75, 0.9, 0.97)))
        entropy = torch.stack([-(p * torch.where(p > 0, p, 1).log()).sum(-1) for p in old.distributions]).sum(0)
        for index in range(5):
            mask = u_bins == index
            frames = executed[:, ::4] & mask[:, None]
            if bool(mask.any()):
                metrics[f"return/interval_reward_u{index}"] = self.metric_totals.mean(
                    f"return/interval_reward_u{index}", frame_rewards[mask].sum(-1)
                )
            if bool(frames.any()):
                metrics[f"policy/kl_per_frame_u{index}"] = self.metric_totals.ratio(
                    f"policy/kl_per_frame_u{index}",
                    float(torch.where(executed & mask[:, None], kl, 0).sum()),
                    float(frames.sum()),
                )
                metrics[f"policy/entropy_u{index}"] = self.metric_totals.mean(
                    f"policy/entropy_u{index}", entropy[frames]
                )
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
            global_count = distributed.sum_tensor(selected.sum())
            for _ in range(cfg.critic_epochs):
                self.critic_optimizer.zero_grad(set_to_none=True)
                loss = critic_loss(
                    self.policy.values(features),
                    batch.targets,
                    selected,
                    value_scale=cfg.value_scale,
                    reduction=DecisionReduction(global_count, distributed.world_size()),
                )
                _check_finite("critic loss", loss)
                loss.backward()
                distributed.average_gradients(self.policy.critic_parameters())
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
            metrics.update(self._warmup(batch))
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
                    statistics = advantage_statistics(
                        distributed.sum_tensor(advantage_moments(advantages.flatten(), selected.flatten()))
                    )
                    # Chunks share the step's statistics and denominator, so their
                    # gradients sum to the gradient of the whole step.
                    chunk = cfg.windows_per_step // cfg.actor_microbatches
                    parts: list[tuple[ActorLoss, ReplayOutput]] = []
                    total = tuple(torch.zeros_like(p) for p in self.policy.actor_parameters())
                    for start in range(0, cfg.windows_per_step, chunk):
                        ids = window_ids[start : start + chunk]
                        part = batch.windows.take(ids, batch.placement.positions, ids >= 0)
                        targets = ActorTargets(
                            table.behavior[part.rows],
                            advantages[start : start + chunk],
                            selected[start : start + chunk],
                            table.coefficients[part.rows],
                            tuple(value[part.rows] for value in reference_dense),
                            statistics.mean.float(),
                            statistics.sigma.float(),
                            statistics.count.float() / distributed.world_size(),
                        )
                        parts.append(self.policy.actor_step(part, targets, clip_epsilon=cfg.clip_epsilon))
                        if cfg.actor_microbatches > 1:
                            for gradient, parameter in zip(total, self.policy.actor_parameters(), strict=True):
                                if parameter.grad is not None:
                                    gradient.add_(parameter.grad)
                    if cfg.actor_microbatches > 1:
                        for gradient, parameter in zip(total, self.policy.actor_parameters(), strict=True):
                            if parameter.grad is None:
                                parameter.grad = gradient
                            else:
                                parameter.grad.copy_(gradient)
                    result = ActorLoss(
                        *(
                            torch.stack([getattr(loss, f.name) for loss, _ in parts]).sum(0)
                            if getattr(parts[0][0], f.name).ndim == 0
                            else torch.cat([getattr(loss, f.name) for loss, _ in parts])
                            for f in dataclasses.fields(ActorLoss)
                        )
                    )
                    current = concatenate_replays([output for _, output in parts], 0)
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
                    distributed.average_gradients(self.policy.actor_parameters())
                    global_windows = distributed.sum_tensor(live.sum())
                    noise.add(int(global_windows), int(statistics.count))
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
                    metrics[f"policy/old_log_ratio_step{step}"] = self.metric_totals.mean(
                        f"policy/old_log_ratio_step{step}", log_ratio
                    )
                    assert table.evaluation is not None
                    new = window_evaluation(
                        current,
                        table.actions[windows.rows].flatten(0, 1),
                        torch.arange(windows.rows.numel(), device=windows.rows.device),
                    )
                    previous = _take_evaluation(table.evaluation, windows.rows.flatten())
                    mask = (current.factor_mask_WKF & selected[..., None]).flatten(0, 1)
                    old_kl = factor_kl(new, previous, mask)
                    metrics[f"policy/old_kl_step{step}"] = self.metric_totals.ratio(
                        f"policy/old_kl_step{step}", float(torch.where(mask, old_kl, 0).sum()), float(selected.sum())
                    )
        ratio = torch.cat(ratios)
        metrics.update(gradient_metrics("actor", actor_norms, cfg.gradient_clip))
        metrics.update(noise.metrics())
        metrics.update(parameter_metrics("actor", self.actor_optimizer, actor_before, self.policy, self.reference))
        metrics.update(
            {
                "policy/loss": float(torch.stack(policy_losses).mean()),
                "policy/kl_loss": float(torch.stack(kl_losses).mean()),
                "policy/clip_fraction": self.metric_totals.mean(
                    "policy/clip_fraction", ((ratio - 1).abs() > cfg.clip_epsilon).float()
                ),
                "policy/ratio_p01": float(ratio.quantile(0.01)) if ratio.numel() else math.nan,
                "policy/ratio_p99": float(ratio.quantile(0.99)) if ratio.numel() else math.nan,
                "policy/old_log_ratio_p99": float(torch.cat(old_log_ratios).abs().quantile(0.99))
                if ratio.numel()
                else math.nan,
                "systems/actor_seconds": time.perf_counter() - started,
            }
        )
        return metrics

    def _warmup(self, batch: PreparedUpdate) -> dict[str, float]:
        for timing in self.engine.sampler.schedule.configurations:
            mask = (batch.decisions.delay_J == timing.delay) & (batch.decisions.replan_J == timing.replan)
            values = batch.advantages[mask].double()
            moments = distributed.sum_tensor(advantage_moments(values, torch.ones_like(values, dtype=torch.bool)))
            previous = self.warmup_moments.get(timing, (0, 0.0, 0.0))
            self.warmup_moments[timing] = (
                previous[0] + int(moments[0]),
                previous[1] + float(moments[1]),
                previous[2] + float(moments[2]),
            )
        if self.update + 1 < self.warmup_start_update + self.config.critic_warmup_updates:
            return {}
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
        return {"train/warmup_ended": 1.0}

    def flops_metrics(self, batch: PreparedUpdate, metrics: dict[str, float]) -> dict[str, float]:
        if not isinstance(self.policy, TransformerPolicy):
            return {}
        policies = self.engine.execution_policies()
        rollout = sum(
            policy.work_flops.get("rollout", 0) for policy in policies if isinstance(policy, TransformerPolicy)
        )
        actor = self.policy.work_flops.get("actor", 0)
        preparation = self.policy.work_flops.get("value", 0) + (
            self.reference.work_flops.get("reference", 0) if isinstance(self.reference, TransformerPolicy) else 0
        )
        critic = 3 * self.policy.flops.critic * len(batch.values) * self.config.critic_epochs
        peak = (
            mfu.bf16_dense_peak_flops(torch.cuda.get_device_name(self.policy.device))
            if self.policy.device.type == "cuda"
            else None
        )
        return phase_metrics(
            (rollout, actor, preparation, critic),
            batch.rollout.frames,
            len(batch.values),
            metrics,
            device=self.policy.device,
            peak=peak,
        )

    def run_update(self) -> tuple[PreparedUpdate, dict[str, float]]:
        if self.updating:
            raise ValueError("previous update did not finish; restore an update-boundary checkpoint")
        self.updating = True
        for policy in self.engine.execution_policies():
            if isinstance(policy, TransformerPolicy):
                policy.work_flops.clear()
        self.metric_totals = distributed.MetricTotals(self.policy.device)
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
            self.memory.measure_all
            or not self.memory.first_checked
            or not self.memory.actor_checked
            and not self.warming
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
        metrics.update(self.anchor_metrics(rollout))
        for role, counts in rollout.behavior:
            metrics.update(counts.metrics(role))
            active = float(counts.counts[-2])
            for name, count in zip(COUNT_NAMES, counts.counts, strict=True):
                self.metric_totals.total(f"behavior/{role}_{name}_count", float(count))
            for name, count in zip(COUNT_NAMES[:-2], counts.counts[:-2], strict=True):
                self.metric_totals.ratio(f"behavior/{role}_{name}_per_min", float(count) * 3600, active)
            self.metric_totals.ratio(f"behavior/{role}_offstage_fraction", float(counts.counts[-1]), active)
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
            snapshot_started = time.perf_counter()
            promoted = min(self.slot_updates, key=lambda slot: (self.slot_updates[slot], slot))
            truncated = self.engine.truncate_slot(promoted)
            for match, number in truncated:
                for side in (0, 1):
                    self.episode_metrics.pop((2 * match + side, number), None)
                    self.partial_episodes.discard((2 * match + side, number))
            self.engine.anchors[promoted].model.load_state_dict(self.policy.model.state_dict(), strict=True)
            self.slot_updates[promoted] = self.update
            # Rolling win counts must not combine different weights in one slot.
            for row in self.anchor_win_history:
                row.pop(promoted, None)
            if self.policy.device.type == "cuda":
                torch.cuda.synchronize(self.policy.device)
            metrics["systems/snapshot_seconds"] = time.perf_counter() - snapshot_started
            if distributed.rank() == 0:
                print(
                    f"snapshot update={self.update} anchor={promoted} seconds={metrics['systems/snapshot_seconds']:.3f}",
                    flush=True,
                )
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
        self.metric_totals.ratio(
            "systems/reset_idle_fraction", rollout.idle_frames, rollout.frames + rollout.idle_frames
        )
        metrics["systems/startup_capture_seconds"] = self.startup_seconds
        metrics["systems/post_startup_compiles"] = float(
            self.compilation_counter.snapshot() if self.compilation_counter is not None else 0
        )
        metrics["systems/post_startup_captures"] = float(
            sum(
                policy.post_startup_captures
                for policy in self.engine.execution_policies()
                if isinstance(policy, TransformerPolicy)
            )
        )
        metrics["systems/update_seconds"] = time.perf_counter() - started
        metrics["systems/match_frames_per_second"] = rollout.frames / metrics["systems/update_seconds"]
        if self.update == 1 or self.update % 16 == 0 or self.update % self.config.snapshot_interval == 0:
            distributed.check_parameters(tuple(self.policy.model.parameters()))
        metrics = self.metric_totals.reduce(metrics)
        metrics.update(self.flops_metrics(batch, metrics))
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
        "world_size": str(distributed.world_size()),
        "tf32_matmul": str(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": str(torch.backends.cudnn.allow_tf32),
    }


def _configuration_identity(trainer: Trainer) -> dict[str, object]:
    engine = trainer.engine
    model = trainer.policy.model
    schema = [(name, tuple(value.shape), str(value.dtype)) for name, value in model.state_dict().items()]
    training = dataclasses.asdict(trainer.config)
    del training["deterministic"]
    identity: dict[str, object] = {
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
        "opponent_layout": "fixed_opponent_slots_v2",
        "lane_roles": sorted(engine.lane_roles.items()),
    }
    if isinstance(trainer.policy, TransformerPolicy):
        identity["decision_kernels"] = "dense_tick_window_graph_v6"
        identity["reference_kernels"] = "actor_order_chunk_graph_v4"
        identity["decoder_precision"] = trainer.policy.model.decoder_precision
        identity["model"] = dataclasses.asdict(trainer.policy.model.cfg)
        identity["ranges"] = dataclasses.asdict(trainer.policy.model.ranges)
        identity["statistics"] = {key: dataclasses.asdict(value) for key, value in trainer.policy.statistics.items()}
    if distributed.world_size() > 1:
        base = min(engine.matches)
        identity["world_size"] = distributed.world_size()
        identity["execution_device"] = trainer.policy.device.type
        identity["groups"] = [
            [[match - base for match in shard.match_ids] for shard in group] for group in engine.groups
        ]
        identity["lane_roles"] = sorted((match - base, role) for match, role in engine.lane_roles.items())
    return identity


def _configuration_digest(trainer: Trainer) -> str:
    return hashlib.sha256(json.dumps(_configuration_identity(trainer), sort_keys=True).encode()).hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class CheckpointSnapshot:
    update: int
    state: dict[str, object]
    simulators: tuple[tuple[str, bytes | NativePoolSnapshot], ...]
    actor: dict[str, object] | None
    manifest: dict[str, object]
    rank_shards: tuple[CheckpointShard, ...] = ()


def _checkpoint_tensors(value: object) -> Iterator[Tensor]:
    if isinstance(value, Tensor):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _checkpoint_tensors(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _checkpoint_tensors(child)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            yield from _checkpoint_tensors(getattr(value, field.name))


def capture_checkpoint(trainer: Trainer, provenance: dict[str, str]) -> CheckpointSnapshot:
    """Own every mutable byte before the next update can change it."""
    if torch.are_deterministic_algorithms_enabled() != trainer.config.deterministic:
        raise ValueError("checkpoint execution mode differs from the trainer configuration")
    if (
        not trainer.engine._initialized
        or trainer.engine.staggering
        or trainer.updating
        or trainer.engine.completed_update != trainer.update - 1
    ):
        raise ValueError("checkpoint requires a complete update boundary")
    required = {"git_sha", "checkpoint_sha256", "data_manifest_sha256", "simulator_sha256", "adapter_sha256"}
    if set(provenance) != required or any(not value for value in provenance.values()):
        raise ValueError("checkpoint provenance must identify source, BC checkpoint, data, simulator, and adapter")
    engine = trainer.engine
    anchors = {key: value.model.state_dict() for key, value in engine.anchors.items()}
    state: dict[str, object] = {
        "version": CHECKPOINT_VERSION,
        "execution": {
            "deterministic": trainer.config.deterministic,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "model": trainer.policy.model.state_dict(),
        "actor_optimizer": trainer.actor_optimizer.state_dict(),
        "critic_optimizer": trainer.critic_optimizer.state_dict(),
        "scheduler": trainer.scheduler.state_dict(),
        "reference": anchors[0],
        "anchors": anchors,
        "slot_updates": trainer.slot_updates,
        "pool": engine.snapshot(),
        "clocks": engine.clocks,
        "cursor": engine.cursor,
        "sampler_cursor": engine.sampler.cursor,
        "lambda_max": engine.sampler.lambda_max,
        "update": trainer.update,
        "actor_steps": trainer.actor_steps,
        "pool_start_update": trainer.pool_start_update,
        "warming": trainer.warming,
        "warmup_start_update": trainer.warmup_start_update,
        "warmup_moments": trainer.warmup_moments,
        "episode_metrics": trainer.episode_metrics,
        "behavior_state": engine.behavior_tracker.state.copy(),
        "partial_episodes": sorted(trainer.partial_episodes),
        "anchor_win_history": trainer.anchor_win_history,
        "memory_guard": (trainer.memory.first_checked, trainer.memory.actor_checked, trainer.memory.peaks),
        "evaluation_cursor": trainer.evaluation_cursor,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": (
            [torch.cuda.get_rng_state(trainer.policy.device)]
            if distributed.world_size() > 1
            else torch.cuda.get_rng_state_all()
        )
        if trainer.policy.device.type == "cuda" or (distributed.world_size() == 1 and torch.cuda.is_available())
        else [],
        "counter_rng": {"seed": engine.sampler.rng.seed, "key": engine.sampler.rng.key},
    }
    if distributed.world_size() > 1:
        state["version"] = DISTRIBUTED_TRAINING_VERSION
        state["distributed_rank"] = distributed.rank()
        if distributed.rank() != 0:
            for name in ("model", "actor_optimizer", "critic_optimizer", "scheduler", "reference", "anchors"):
                del state[name]
    # Prepopulate deepcopy's memo so it preserves shared tensor references while
    # batching device-to-pinned-host copies on the current training stream.
    memo: dict[int, object] = {}
    for tensor in _checkpoint_tensors(state):
        if id(tensor) in memo:
            continue
        host = torch.empty_like(tensor, device="cpu", pin_memory=tensor.device.type == "cuda")
        host.copy_(tensor.detach(), non_blocking=tensor.device.type == "cuda")
        memo[id(tensor)] = host
    if trainer.policy.device.type == "cuda":
        torch.cuda.synchronize(trainer.policy.device)
    state = copy.deepcopy(state, memo)
    simulators: list[tuple[str, bytes | NativePoolSnapshot]] = []
    pending: list[tuple[str, NativeSimulatorShard, Future[tuple[native.NativeSnapshot, ...]]]] = []
    for group_id, group in enumerate(engine.groups):
        for shard_id, shard in enumerate(group):
            role = f"simulator_{group_id}_{shard_id}"
            if isinstance(shard, NativeSimulatorShard):
                pending.append((role, shard, shard.capture()))
            else:
                simulators.append((role, shard.save()))
    for role, shard, future in pending:
        simulators.append((role, NativePoolSnapshot(shard.match_ids, shard.max_frame, future.result())))
    actor: dict[str, object] | None = None
    if isinstance(trainer.policy, TransformerPolicy) and distributed.rank() == 0:
        actor = {
            **evaluation_checkpoint_metadata(
                trainer.policy.model,
                trainer.policy.statistics,
                provenance={
                    "reference_sha256": provenance["checkpoint_sha256"],
                    "git_sha": provenance["git_sha"],
                    "update": trainer.update,
                    "data_manifests": {"training": provenance["data_manifest_sha256"]},
                },
            ),
            "model": state["model"],
        }
    manifest: dict[str, object] = {
        "format": DISTRIBUTED_TRAINING_FORMAT if distributed.world_size() > 1 else TRAINING_FORMAT,
        "version": DISTRIBUTED_TRAINING_VERSION if distributed.world_size() > 1 else CHECKPOINT_VERSION,
        "experiment": "065_selfplay_ppo",
        "update": trainer.update,
        "configuration_sha256": _configuration_digest(trainer),
        "reward_weights": dataclasses.asdict(trainer.config.reward_weights),
        "environment": _environment(),
        "provenance": dict(provenance),
    }
    return CheckpointSnapshot(trainer.update, state, tuple(simulators), actor, manifest)


def write_checkpoint(snapshot: CheckpointSnapshot, directory: Path) -> Path:
    """Publish the manifest only after all immutable files are durable."""
    import io

    directory.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    torch.save(snapshot.state, buffer)
    name, digest = _durable_file(directory, buffer.getvalue(), "rank0.pt")
    shards: list[CheckpointShard] = [{"name": name, "sha256": digest, "role": "training"}, *snapshot.rank_shards]
    for role, state in snapshot.simulators:
        payload = encode_native_pool(state) if isinstance(state, NativePoolSnapshot) else state
        name, digest = _durable_file(directory, zlib.compress(payload), "sim.zlib")
        shards.append({"name": name, "sha256": digest, "role": role})
    if snapshot.actor is not None:
        buffer = io.BytesIO()
        torch.save(snapshot.actor, buffer)
        name, digest = _durable_file(directory, buffer.getvalue(), "actor.pt")
        shards.append({"name": name, "sha256": digest, "role": "actor"})
    manifest = {**snapshot.manifest, "shards": shards}
    temporary = directory / f".manifest-{snapshot.update}-{os.getpid()}.partial"
    target = directory / f"update-{snapshot.update:08d}.json"
    with temporary.open("w") as output:
        json.dump(manifest, output, sort_keys=True)
        output.flush()
        os.fsync(output.fileno())
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        # Persist shard names before the manifest can refer to them.
        os.fsync(descriptor)
        os.replace(temporary, target)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return target


def save_checkpoint(trainer: Trainer, directory: Path, provenance: dict[str, str]) -> Path:
    snapshot = capture_checkpoint(trainer, provenance)
    if distributed.world_size() == 1:
        return write_checkpoint(snapshot, directory)
    import io

    directory.mkdir(parents=True, exist_ok=True)
    local: list[CheckpointShard] = []
    if distributed.rank() != 0:
        buffer = io.BytesIO()
        torch.save(snapshot.state, buffer)
        name, digest = _durable_file(directory, buffer.getvalue(), "rank.pt")
        local.append({"name": name, "sha256": digest, "role": f"rank{distributed.rank()}"})
        for role, state in snapshot.simulators:
            payload = encode_native_pool(state) if isinstance(state, NativePoolSnapshot) else state
            name, digest = _durable_file(directory, zlib.compress(payload), "sim.zlib")
            local.append({"name": name, "sha256": digest, "role": f"rank{distributed.rank()}_{role}"})
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    shards = tuple(shard for rows in distributed.gather_objects(local) for shard in rows)
    target = directory / f"update-{trainer.update:08d}.json"
    if distributed.rank() == 0:
        write_checkpoint(dataclasses.replace(snapshot, rank_shards=shards), directory)
    distributed.gather_objects(True)
    return target


def _write_and_publish_checkpoint(
    snapshot: CheckpointSnapshot,
    directory: Path,
    publisher: self_play_publication.CheckpointPublisher | None,
    metadata: dict[Path, bytes] | None,
) -> float:
    started = time.perf_counter()
    checkpoint = write_checkpoint(snapshot, directory)
    if publisher is not None:
        publisher.submit(checkpoint, metadata=metadata)
        publisher.wait()
    return time.perf_counter() - started


class CheckpointWorker:
    """Keep one independent snapshot until its write and publication finish."""

    def __init__(
        self,
        directory: Path,
        provenance: dict[str, str],
        publisher: self_play_publication.CheckpointPublisher | None = None,
    ) -> None:
        self.directory = directory
        self.provenance = dict(provenance)
        self.publisher = publisher
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="checkpoint-write")
        self.pending: Future[float] | None = None
        self.update = 0
        self.capture_seconds = 0.0

    def wait(self) -> None:
        if self.pending is None:
            return
        waiting = not self.pending.done()
        started = time.perf_counter()
        if waiting:
            print(f"checkpoint_wait update={self.update}", flush=True)
        worker_seconds = self.pending.result()
        wait_seconds = time.perf_counter() - started if waiting else 0.0
        print(
            f"checkpoint update={self.update} training_seconds={self.capture_seconds:.3f} "
            f"worker_seconds={worker_seconds:.3f} wait_seconds={wait_seconds:.3f}",
            flush=True,
        )
        self.pending = None

    def check(self) -> None:
        if self.pending is not None and self.pending.done():
            self.wait()

    def submit(self, trainer: Trainer) -> None:
        self.wait()
        started = time.perf_counter()
        snapshot = capture_checkpoint(trainer, self.provenance)
        metadata = self.publisher.capture_metadata() if self.publisher is not None else None
        self.capture_seconds = time.perf_counter() - started
        self.update = snapshot.update
        self.pending = self.executor.submit(
            _write_and_publish_checkpoint, snapshot, self.directory, self.publisher, metadata
        )

    def close(self) -> None:
        try:
            self.wait()
        finally:
            self.executor.shutdown(wait=True)


def restore_checkpoint(trainer: Trainer, manifest_path: Path, provenance: dict[str, str]) -> None:
    manifest = read_training_manifest(manifest_path)
    distributed.validate_world_size(manifest["environment"]["world_size"])
    expected_version = DISTRIBUTED_TRAINING_VERSION if distributed.world_size() > 1 else CHECKPOINT_VERSION
    if manifest["version"] != expected_version:
        raise ValueError("exact resume requires the fixed-slot checkpoint format for this world size")
    if manifest["reward_weights"] != dataclasses.asdict(trainer.config.reward_weights) or manifest[
        "configuration_sha256"
    ] != _configuration_digest(trainer):
        raise ValueError("resume configuration mismatch")
    if manifest["environment"] != _environment() or manifest["provenance"] != provenance:
        raise ValueError("resume environment or checkpoint mismatch")
    paths: dict[str, Path] = {}
    for shard in manifest["shards"]:
        path = manifest_path.parent / shard["name"]
        if sha256(path) != shard["sha256"]:
            raise ValueError("checkpoint shard hash mismatch")
        paths[shard["role"]] = path
    expected_roles = {"training"} | {
        f"simulator_{g}_{s}" for g, group in enumerate(trainer.engine.groups) for s in range(len(group))
    }
    if isinstance(trainer.policy, TransformerPolicy):
        expected_roles.add("actor")
        evaluation_actor_path(manifest_path)
    if distributed.world_size() > 1:
        expected_roles |= {f"rank{rank}" for rank in range(1, distributed.world_size())}
        expected_roles |= {
            f"rank{rank}_simulator_{g}_{s}"
            for rank in range(1, distributed.world_size())
            for g, group in enumerate(trainer.engine.groups)
            for s in range(len(group))
        }
    if set(paths) != expected_roles:
        raise ValueError("checkpoint simulator topology mismatch")
    saved = torch.load(paths["training"], map_location="cpu", weights_only=False)
    if distributed.rank() != 0:
        local = torch.load(paths[f"rank{distributed.rank()}"], map_location="cpu", weights_only=False)
        shared_names = {"model", "actor_optimizer", "critic_optimizer", "scheduler", "reference", "anchors"}
        if set(local) != set(saved) - shared_names:
            raise ValueError("checkpoint rank state schema mismatch")
        saved.update(local)
    if distributed.world_size() > 1 and saved.get("distributed_rank") != distributed.rank():
        raise ValueError("checkpoint rank identity mismatch")
    behavior = BehaviorTracker(len(trainer.engine.behavior_tracker.state))
    behavior.restore(saved["behavior_state"])
    warmup_start = saved["warmup_start_update"]
    if (
        type(warmup_start) is not int
        or not 0 <= warmup_start <= saved["update"]
        or saved["warming"] != (saved["update"] < warmup_start + trainer.config.critic_warmup_updates)
    ):
        raise ValueError("invalid checkpoint critic warmup progress")
    if (
        type(saved["version"]) is not int
        or type(saved["update"]) is not int
        or saved["version"] != expected_version
        or saved["update"] != manifest["update"]
    ):
        raise ValueError("checkpoint shard identity mismatch")
    groups = len(trainer.engine.groups)
    frames = trainer.engine.sampler.schedule.rollout_frames
    initial_clock = math.ceil(trainer.config.head_start_frames / frames) * frames
    pool_start = saved["pool_start_update"]
    if type(pool_start) is not int or not 0 <= pool_start <= saved["update"] or pool_start % groups:
        raise ValueError("invalid fresh-pool update origin")
    pool_updates = saved["update"] - pool_start
    expected_clocks = [
        initial_clock + ((pool_updates + groups - 1 - group) // groups) * frames for group in range(groups)
    ]
    if (
        not isinstance(saved["clocks"], list)
        or any(type(clock) is not int for clock in saved["clocks"])
        or saved["clocks"] != expected_clocks
        or type(saved["cursor"]) is not int
        or saved["cursor"] != saved["update"] % groups
    ):
        raise ValueError("checkpoint pool progress mismatch")
    for field in ("actor_steps", "sampler_cursor", "evaluation_cursor"):
        if type(saved[field]) is not int or saved[field] < 0:
            raise ValueError("checkpoint progress counters must be nonnegative integers")
    partial = saved.get("partial_episodes")
    streams = {2 * match + side for matches in trainer.engine.group_matches for match in matches for side in (0, 1)}
    if (
        not isinstance(partial, list)
        or any(
            not isinstance(pair, tuple)
            or len(pair) != 2
            or any(type(value) is not int or value < 0 for value in pair)
            or pair[0] not in streams
            or pair in saved["episode_metrics"]
            for pair in partial
        )
        or len(set(partial)) != len(partial)
        or (partial and not trainer.config.head_start_frames)
    ):
        raise ValueError("invalid checkpoint partial episodes")
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
    if set(saved["anchors"]) != {0, 1, 2, 3} or set(saved["slot_updates"]) != {1, 2, 3}:
        raise ValueError("checkpoint fixed snapshot slot mismatch")
    if any(type(update) is not int or not 0 <= update <= saved["update"] for update in saved["slot_updates"].values()):
        raise ValueError("invalid checkpoint slot weight update")
    reference = trainer.reference
    engine.anchors.clear()
    slots = iter(trainer.snapshot_slots)
    for key, tensors in saved["anchors"].items():
        policy = reference if key == 0 else next(slots)
        policy.model.load_state_dict(tensors, strict=True)
        engine.anchors[key] = policy
    engine.layout_anchors = [() for _ in engine.groups]
    trainer.reference = engine.anchors[0]
    for name, value in saved["reference"].items():
        if not torch.equal(trainer.reference.model.state_dict()[name].cpu(), value):
            raise ValueError("checkpoint reference and anchor zero disagree")
    for group_id, group in enumerate(engine.groups):
        for shard_id, shard in enumerate(group):
            role = f"simulator_{group_id}_{shard_id}"
            if distributed.rank() != 0:
                role = f"rank{distributed.rank()}_{role}"
            shard.restore(zlib.decompress(paths[role].read_bytes()))
    engine.restore_pool(saved["pool"])
    engine.clocks = saved["clocks"]
    engine.cursor = saved["cursor"]
    trainer.slot_updates = saved["slot_updates"]
    engine.sampler.cursor = saved["sampler_cursor"]
    engine.sampler.lambda_max = saved["lambda_max"]
    engine._initialized = True
    trainer.update = saved["update"]
    trainer.updating = False
    engine.completed_update = trainer.update - 1
    trainer.actor_steps = saved["actor_steps"]
    trainer.pool_start_update = saved["pool_start_update"]
    trainer.warming = saved["warming"]
    trainer.warmup_start_update = warmup_start
    trainer.warmup_moments = saved["warmup_moments"]
    engine.behavior_tracker.restore(saved["behavior_state"])
    trainer.episode_metrics = saved["episode_metrics"]
    trainer.partial_episodes = set(cast(list[tuple[int, int]], partial))
    trainer.anchor_win_history = saved["anchor_win_history"]
    trainer.memory.first_checked, trainer.memory.actor_checked, trainer.memory.peaks = saved["memory_guard"]
    trainer.evaluation_cursor = saved["evaluation_cursor"]
    trainer.policy.clear_caches()
    random.setstate(saved["python_rng"])
    np.random.set_state(saved["numpy_rng"])
    torch.set_rng_state(saved["torch_rng"])
    if distributed.world_size() > 1 and trainer.policy.device.type == "cuda":
        torch.cuda.set_rng_state(saved["cuda_rng"][0], trainer.policy.device)
    elif distributed.world_size() == 1 and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
    distributed.check_parameters(tuple(trainer.policy.model.parameters()))


PILOT_SOURCE = "3d4ad115881857d0c6c1ba039e41225bd7df3fe8"
PILOT_ADAPTER_SHA256 = "29f92fca9dfc77ef89ac86de51ccfaab20bbeffb87874b9348d503793f564145"


def legacy_pilot_digest(trainer: Trainer) -> str:
    identity = _configuration_identity(trainer)
    del cast(dict[str, object], identity["train"])["reward_weights"]
    identity["sampler"] = {**dataclasses.asdict(trainer.engine.sampler.config), "anchor_rate": 0.15}
    identity["opponent_layout"] = "fixed_anchor_ranges_v1"
    del identity["lane_roles"]
    if isinstance(trainer.policy, TransformerPolicy):
        identity["decision_kernels"] = "dense_tick_window_graph_v5"
        identity["reference_kernels"] = "actor_order_readonly_forward_v3"
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def restore_pilot_learner(trainer: Trainer, manifest_path: Path, provenance: dict[str, str]) -> HeadStart | None:
    """Migrate only the stopped update-256 learner; fresh matches are a new treatment."""
    manifest = read_training_manifest(manifest_path)
    if (manifest["format"], manifest["version"], manifest["update"]) != ("hal.o65.training.v8", 8, 256):
        raise ValueError("learner resume requires the stopped pilot's v8 update-256 checkpoint")
    if manifest["provenance"] != {
        **provenance,
        "git_sha": PILOT_SOURCE,
        "adapter_sha256": PILOT_ADAPTER_SHA256,
    }:
        raise ValueError("learner resume source or artifact mismatch")
    if manifest["configuration_sha256"] != legacy_pilot_digest(trainer) or manifest["environment"] != _environment():
        raise ValueError("learner resume configuration or environment mismatch")
    paths = {}
    for shard in manifest["shards"]:
        path = manifest_path.parent / shard["name"]
        if sha256(path) != shard["sha256"]:
            raise ValueError("learner resume shard hash mismatch")
        paths[shard["role"]] = path
    expected = {"training", "actor"} | {
        f"simulator_{g}_{s}" for g, group in enumerate(trainer.engine.groups) for s in range(len(group))
    }
    if set(paths) != expected:
        raise ValueError("learner resume simulator topology mismatch")
    actor_path = evaluation_actor_path(manifest_path)
    saved = torch.load(paths["training"], map_location="cpu", weights_only=False)
    if (
        saved["version"] != 8
        or saved["update"] != 256
        or saved["warming"]
        or set(saved["anchors"]) != {0, 1}
        or saved["anchor_candidates"] != (0, 1)
        or saved["next_anchor"] != 2
        or saved["evaluation_cursor"] != 0
        or saved["execution"]
        != {
            "deterministic": trainer.config.deterministic,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        }
        or saved["counter_rng"] != {"seed": trainer.config.seed, "key": trainer.engine.sampler.rng.key}
        or not trainer.engine.sampler.config.lambda_min
        <= saved["lambda_max"]
        <= trainer.engine.sampler.config.lambda_max
    ):
        raise ValueError("learner resume state differs from the stopped pilot")
    groups = len(trainer.engine.groups)
    frames = trainer.engine.sampler.schedule.rollout_frames
    initial = math.ceil(trainer.config.head_start_frames / frames) * frames
    clocks = [initial + ((256 + groups - 1 - group) // groups) * frames for group in range(groups)]
    if saved["clocks"] != clocks or saved["cursor"] != 256 % groups:
        raise ValueError("learner resume pool progress mismatch")
    actor = torch.load(actor_path, map_location="cpu", weights_only=False)
    if actor["model"].keys() != saved["model"].keys() or any(
        not torch.equal(value, saved["model"][name]) for name, value in actor["model"].items()
    ):
        raise ValueError("learner resume actor disagrees with training state")
    if saved["anchors"][1].keys() != saved["model"].keys() or any(
        not torch.equal(value, saved["model"][name]) for name, value in saved["anchors"][1].items()
    ):
        raise ValueError("learner resume snapshot is not the update-256 student")
    reference = trainer.reference.model.state_dict()
    if saved["reference"].keys() != reference.keys() or any(
        not torch.equal(value, reference[name].cpu()) or not torch.equal(value, saved["anchors"][0][name])
        for name, value in saved["reference"].items()
    ):
        raise ValueError("learner resume pi_ref mismatch")
    if trainer.engine._initialized:
        raise ValueError("learner migration requires a fresh uninitialized pool")
    trainer.engine.sampler.lambda_max = saved["lambda_max"]
    trainer.engine.initialize()
    trainer.capture_startup()
    head_start = trainer.head_start() if trainer.config.head_start_frames else None
    trainer.policy.model.load_state_dict(saved["model"], strict=True)
    trainer.actor_optimizer.load_state_dict(saved["actor_optimizer"])
    trainer.critic_optimizer.load_state_dict(saved["critic_optimizer"])
    trainer.scheduler.load_state_dict(saved["scheduler"])
    trainer.engine.anchors[1].model.load_state_dict(saved["anchors"][1], strict=True)
    # Head-start episodes used pi_ref, so end slot 1 before its first learned-weight tick.
    for match, number in trainer.engine.truncate_slot(1):
        for side in (0, 1):
            trainer.partial_episodes.discard((2 * match + side, number))
    trainer.slot_updates = {1: 256, 2: 0, 3: 0}
    trainer.update = trainer.pool_start_update = 256
    trainer.engine.completed_update = 255
    trainer.actor_steps = saved["actor_steps"]
    reward_changed = manifest["reward_weights"] != dataclasses.asdict(trainer.config.reward_weights)
    trainer.warmup_start_update = trainer.update if reward_changed else 0
    trainer.warming = reward_changed and trainer.config.critic_warmup_updates > 0
    trainer.warmup_moments = {} if reward_changed else saved["warmup_moments"]
    trainer.engine.sampler.lambda_max = saved["lambda_max"]
    trainer.anchor_win_history = saved["anchor_win_history"]
    # No simulator, episode accumulator or queue belongs to the new match pool.
    random.setstate(saved["python_rng"])
    np.random.set_state(saved["numpy_rng"])
    torch.set_rng_state(saved["torch_rng"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(saved["cuda_rng"])

    return head_start


def print_progress(metrics: dict[str, float], *, warming: bool) -> None:
    games = int(metrics.get("gameplay/pi_ref_count_32_updates", 0))
    win_rate = f"{metrics['gameplay/pi_ref_win_rate_32_updates']:.3f}" if games else "n/a"
    gib = 1024**3
    print(
        f"update={int(metrics['global_step'])} warming={int(warming)} "
        f"seconds={metrics['systems/update_seconds']:.3f} "
        f"compiles={int(metrics['systems/post_startup_compiles'])} "
        f"captures={int(metrics['systems/post_startup_captures'])} "
        f"rollout={metrics['systems/rollout_seconds']:.3f} "
        f"preparation={metrics['systems/preparation_seconds']:.3f} "
        f"actor={metrics.get('systems/actor_seconds', 0):.3f} "
        f"critic={metrics['systems/critic_seconds']:.3f} "
        f"match_frames/s={metrics['systems/match_frames_per_second']:.1f} "
        f"kl/frame={metrics['policy/kl_per_frame']:.6g} "
        f"ev={metrics['critic/explained_variance']:.4f} "
        f"pi_ref_win32={win_rate} games={games} "
        f"peak_allocated_gib={metrics.get('systems/peak_allocated_bytes', 0) / gib:.3f} "
        f"peak_reserved_gib={metrics.get('systems/peak_reserved_bytes', 0) / gib:.3f}",
        flush=True,
    )


def run_training(
    trainer: Trainer,
    updates: int,
    output: Path,
    provenance: dict[str, str],
    *,
    wandb_project: str | None = None,
    run_configuration: dict[str, object] | None = None,
    publisher: self_play_publication.CheckpointPublisher | None = None,
    head_start: HeadStart | None = None,
) -> None:
    import wandb

    if updates <= trainer.update:
        raise ValueError("updates must exceed the restored update cursor")
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
        "control": "frozen pi_ref with matched sampling, starts, episode u, and schedules",
        "invariants": "BC checkpoint, representation, data, simulator, adapter, reward, evaluation seeds",
        "run": run_configuration,
    }
    if isinstance(trainer.policy, TransformerPolicy):
        resolved["decoder_precision"] = trainer.policy.model.decoder_precision
    if distributed.world_size() > 1:
        run_distributed_training(
            trainer,
            updates,
            output,
            provenance,
            resolved,
            wandb_project=wandb_project,
            publisher=publisher,
            head_start=head_start,
        )
        return
    (output / "resolved.json").write_text(json.dumps(resolved, indent=2, sort_keys=True))
    with wandb.init(project=wandb_project, mode="disabled" if wandb_project is None else "online", config=resolved):
        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")
        if trainer.pool_start_update and trainer.update == trainer.pool_start_update:
            for line in (output / "metrics.jsonl").read_text().splitlines():
                wandb.log(json.loads(line))
        if head_start is not None:
            metrics = head_start.metrics()
            (output / "head-start.json").write_text(
                json.dumps({**dataclasses.asdict(head_start), "metrics": metrics}, indent=2)
            )
            wandb.log({"global_step": trainer.update, **metrics})
            print(f"head_start_ended seconds={head_start.seconds:.3f}", flush=True)
        with (
            contextlib.closing(CheckpointWorker(output / "checkpoints", provenance, publisher)) as checkpoints,
            (output / "metrics.jsonl").open("a") as log,
        ):
            while trainer.update < updates:
                checkpoints.check()
                warming = trainer.warming
                batch, metrics = trainer.run_update()
                del batch
                print_progress(metrics, warming=warming)
                if metrics.get("train/warmup_ended"):
                    print(
                        f"critic_warmup_ended update={trainer.update} "
                        f"lambda_max={trainer.engine.sampler.lambda_max:.6g}",
                        flush=True,
                    )
                resolved["calibrated_lambda_max"] = trainer.engine.sampler.lambda_max
                (output / "resolved.json").write_text(json.dumps(resolved, indent=2, sort_keys=True))
                wandb.log(metrics)
                log.write(json.dumps(metrics, sort_keys=True) + "\n")
                log.flush()
                if trainer.update % trainer.config.checkpoint_interval == 0 or trainer.update == updates:
                    checkpoints.submit(trainer)


def run_distributed_training(
    trainer: Trainer,
    updates: int,
    output: Path,
    provenance: dict[str, str],
    resolved: dict[str, object],
    *,
    wandb_project: str | None,
    publisher: self_play_publication.CheckpointPublisher | None,
    head_start: HeadStart | None,
) -> None:
    import wandb

    primary = distributed.rank() == 0
    trainer.memory.output = output if primary else output / f"rank{distributed.rank()}"
    if not primary:
        trainer.memory.output.mkdir(exist_ok=True)
    with contextlib.ExitStack() as resources:
        log = None
        if primary:
            resources.enter_context(
                wandb.init(
                    project=wandb_project, mode="disabled" if wandb_project is None else "online", config=resolved
                )
            )
            wandb.define_metric("global_step")
            wandb.define_metric("*", step_metric="global_step")
            log = resources.enter_context((output / "metrics.jsonl").open("a"))
            (output / "resolved.json").write_text(json.dumps(resolved, indent=2, sort_keys=True))
            if head_start is not None:
                (output / "head-start.json").write_text(json.dumps(dataclasses.asdict(head_start), indent=2))
                wandb.log({"global_step": trainer.update, **head_start.metrics()})
        while trainer.update < updates:
            warming = trainer.warming
            batch, metrics = trainer.run_update()
            del batch
            if primary:
                assert log is not None
                print_progress(metrics, warming=warming)
                resolved["calibrated_lambda_max"] = trainer.engine.sampler.lambda_max
                (output / "resolved.json").write_text(json.dumps(resolved, indent=2, sort_keys=True))
                wandb.log(metrics)
                log.write(json.dumps(metrics, sort_keys=True) + "\n")
                log.flush()
            if trainer.update % trainer.config.checkpoint_interval == 0 or trainer.update == updates:
                # All ranks finish durable local state before rank zero publishes.
                checkpoint = save_checkpoint(trainer, output / "checkpoints", provenance)
                if primary and publisher is not None:
                    publisher.submit(checkpoint)
                    publisher.wait()
                distributed.gather_objects(True)


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
    rollout_frames: int = 256
    cadence: Literal["development", "production"] = "development"
    configurations: tuple[DelayReplan, ...] = ()
    matches_per_shard: int = 16
    active_matches_per_gpu: int | None = None
    device: str = "cuda"
    max_frame: int = 28800
    resume: Path | None = None
    resume_learner: Path | None = None
    training: TrainConfig = TrainConfig()
    verify_resume: bool = False
    verify_only: bool = False
    wandb_project: str | None = None
    profiling: ProfileConfig | None = None
    decoder_precision: DecoderPrecision = "bf16"
    compile_cache: Path | None = None
    publication: self_play_publication.Args | None = None
    measurement: bool = False

    def __post_init__(self) -> None:
        validate_decoder_precision(self.decoder_precision)
        if self.measurement and (
            self.training.critic_warmup_updates
            or self.resume is not None
            or self.resume_learner is not None
            or self.publication is not None
            or self.profiling is not None
            or self.wandb_project is not None
            or self.verify_resume
        ):
            raise ValueError("measurement requires fresh PPO, no warmup, publication, profiling, W&B or resume")
        if type(self.rollout_frames) is not int or self.rollout_frames < 128 or self.rollout_frames % 128:
            raise ValueError("rollout frames must contain whole 128-frame blocks")
        if self.cadence not in ("development", "production"):
            raise ValueError("unsupported cadence")
        if self.resume_learner is not None and (
            self.resume is not None or self.verify_resume or self.profiling is not None
        ):
            raise ValueError("learner resume requires a fresh pool, without exact resume or profiling")
        if self.publication is not None:
            self_play_publication.validate_args(self.publication)
            if self.output.resolve() != (self.publication.source / "train").resolve():
                raise ValueError("publication source must contain this run's train directory")
            if self.verify_only:
                raise ValueError("verify-only does not publish checkpoints")
        if self.verify_only and (not self.verify_resume or self.resume is None):
            raise ValueError("verify-only needs a resume checkpoint and verify-resume")
        if self.profiling is not None and (
            self.device != "cuda"
            or self.training.critic_warmup_updates
            or self.resume is not None
            or self.wandb_project is not None
            or self.verify_resume
            or self.publication is not None
        ):
            raise ValueError(
                "profiling requires CUDA, warmup off, fresh state, disabled W&B, no publication or resume check"
            )
        if (
            not self.cpus
            or len(set(self.cpus)) != len(self.cpus)
            or any(cpu not in os.sched_getaffinity(0) for cpu in self.cpus)
        ):
            raise ValueError("native runs need distinct available CPUs")
        if any(value < 1 for value in (self.groups, self.matches_per_shard, self.updates)):
            raise ValueError("groups, matches per shard, and updates must be positive integers")
        processes = int(os.environ.get("WORLD_SIZE", "1"))
        if self.active_matches_per_gpu is not None and (
            type(self.active_matches_per_gpu) is not int
            or processes < 1
            or len(self.cpus) % processes
            or len(self.cpus) < processes
            or self.active_matches_per_gpu < 4 * (len(self.cpus) // processes)
            or self.active_matches_per_gpu % (len(self.cpus) // processes)
            or self.matches_per_shard != 16
        ):
            raise ValueError(
                "active matches must divide across CPUs with at least four per shard; leave matches_per_shard at its default"
            )
        if self.matches_per_shard < 4:
            raise ValueError("native fixed-slot shards require at least four matches")
        if self.max_frame != -1 and self.max_frame < 0:
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
    with distributed.process_group(config.device) as device:
        if distributed.world_size() == 1:
            # Single-GPU runs compile the same graphs as DDP rank 0 on cuda:0.
            train_native_rank(
                dataclasses.replace(
                    config,
                    compile_cache=config.compile_cache / "rank0" if config.compile_cache is not None else None,
                )
            )
            return
        if config.profiling is not None or config.verify_resume or config.resume_learner is not None:
            raise ValueError(
                "DDP supports fresh training and exact resume; profiling and learner migration require one GPU"
            )
        cpus = distributed.split_cpus(config.cpus, distributed.world_size(), distributed.rank())
        controller_cpus = sorted(set(os.sched_getaffinity(0)) - set(config.cpus))
        affinity = set(cpus)
        if len(controller_cpus) >= distributed.world_size():
            affinity.add(controller_cpus[distributed.rank()])
        os.sched_setaffinity(0, affinity)
        train_native_rank(
            dataclasses.replace(
                config,
                cpus=cpus,
                matches_per_shard=config.matches_per_shard
                if config.active_matches_per_gpu is None
                else config.active_matches_per_gpu // len(cpus),
                active_matches_per_gpu=None,
                device=str(device),
                compile_cache=config.compile_cache / f"rank{distributed.rank()}"
                if config.compile_cache is not None
                else None,
            )
        )


def train_native_rank(config: NativeCli) -> None:
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
    torch.manual_seed(config.training.seed)
    checkpoint = load_evaluation_checkpoint(
        config.checkpoint, device=torch.device(config.device), decoder_precision=config.decoder_precision
    )
    checkpoint.model.requires_grad_(True)
    checkpoint.model.percentile_head.requires_grad_(False)
    factory = production_config if config.cadence == "production" else development_config
    schedule = factory(checkpoint.model.head_offsets, rollout_frames=config.rollout_frames)
    if config.configurations:
        if not set(config.configurations) <= set(schedule.configurations):
            raise ValueError("requested configurations are outside the selected cadence")
        schedule = dataclasses.replace(schedule, configurations=config.configurations)
    policy = TransformerPolicy(checkpoint.model, schedule, checkpoint.statistics)
    sampling = read_native_sampler(config.sampler)
    for row in sampling.identities:
        if row.identity >= policy.model.cfg.player_vocab_size:
            raise ValueError("sampler identity is outside the checkpoint vocabulary")
    manifests = checkpoint.provenance["data_manifests"]
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    output_exists = config.output.exists() if distributed.rank() == 0 else False
    if (
        distributed.gather_objects(output_exists)[0]
        and config.resume is None
        and (config.resume_learner is None or set(config.output.iterdir()) != {config.output / "metrics.jsonl"})
    ):
        raise FileExistsError(config.output)
    with contextlib.ExitStack() as resources:
        if cache is not None:
            resources.callback(cache.save)
        groups = []
        matches_per_shard = (
            config.matches_per_shard
            if config.active_matches_per_gpu is None
            else config.active_matches_per_gpu // len(config.cpus)
        )
        next_match = distributed.rank() * config.groups * len(config.cpus) * matches_per_shard
        for _ in range(config.groups):
            shards: list[NativeSimulatorShard] = []
            for cpu in config.cpus:
                ids = tuple(range(next_match, next_match + matches_per_shard))
                shard = NativeSimulatorShard(
                    config.data_dir,
                    ids,
                    cpu=cpu,
                    device=policy.device,
                    max_frame=config.max_frame,
                    reward_weights=config.training.reward_weights,
                )
                resources.callback(shard.close)
                shards.append(shard)
                next_match += matches_per_shard
            groups.append(tuple(shards))
        provenance = {
            "git_sha": git_sha,
            "checkpoint_sha256": checkpoint.checkpoint_sha256,
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
        distributed.check_parameters(tuple(policy.model.parameters()))
        head_start = None
        trainer.compilation_counter = resources.enter_context(count_compilation_starts())
        if config.resume_learner is not None:
            head_start = restore_pilot_learner(trainer, config.resume_learner, provenance)
        elif config.resume is None:
            engine.initialize()
            trainer.capture_startup(rollout_only=config.profiling is not None and config.profiling.rollout_only)
            if config.training.head_start_frames:
                head_start = trainer.head_start()
        else:
            restore_checkpoint(trainer, config.resume, provenance)
            trainer.capture_startup()
        if config.verify_only:
            verify_checkpoint_resume(trainer, config.output, provenance)
            return
        publisher = (
            resources.enter_context(contextlib.closing(self_play_publication.CheckpointPublisher(config.publication)))
            if config.publication is not None and distributed.rank() == 0
            else None
        )
        native_config = json.loads(json.dumps(dataclasses.asdict(config), default=str))
        native_config["simulator_manifest_sha256"] = sha256(config.data_dir / "raw" / "manifest.json")
        native_config["sampler_sha256"] = sha256(config.sampler)
        native_config["startup_capture_seconds"] = trainer.startup_seconds
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
        if config.measurement:
            run_measurement(trainer, config, provenance, native_config)
            return
        run_training(
            trainer,
            config.updates,
            config.output,
            provenance,
            wandb_project=config.wandb_project,
            run_configuration=native_config,
            publisher=publisher,
            head_start=head_start,
        )
        if config.verify_resume:
            verify_checkpoint_resume(trainer, config.output, provenance)


def run_measurement(
    trainer: Trainer, config: NativeCli, provenance: dict[str, str], native_config: dict[str, object]
) -> None:
    primary = distributed.rank() == 0
    config.output.mkdir(parents=True, exist_ok=True)
    trainer.memory.output = config.output / f"rank{distributed.rank()}"
    trainer.memory.measure_all = True
    if primary:
        (config.output / "resolved.json").write_text(
            json.dumps({"run": native_config, "provenance": provenance, "environment": _environment()}, indent=2)
        )
    for _ in range(config.updates):
        batch, metrics = trainer.run_update()
        del batch
        if primary:
            with (config.output / "metrics.jsonl").open("a") as log:
                log.write(json.dumps(metrics, sort_keys=True) + "\n")
            print(json.dumps({"measurement": True, **metrics}, sort_keys=True), flush=True)


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
    # Capture only the rollout graphs and time rollouts without PPO updates.
    rollout_only: bool = False

    def __post_init__(self) -> None:
        if self.warm_updates < 1:
            raise ValueError("profiling requires at least one warm update")
        if self.rollout_only and self.nsys:
            raise ValueError("rollout-only profiling uses torch.profiler traces")


def interval_union(intervals: list[tuple[float, float]]) -> float:
    end = total = 0.0
    for start, stop in sorted(intervals):
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def summarize_trace(path: Path) -> dict[str, object]:
    """Count GPU activities through their correlated runtime launches."""
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


def graph_breakdown(path: Path) -> dict[str, object]:
    """Split a traced tick's device time by policy graph through annotated graph launches."""
    events = cast(list[TraceEvent], json.loads(path.read_text())["traceEvents"])
    root = next(
        event
        for event in events
        if event.get("cat") == "user_annotation" and event.get("name", "").startswith("o65_profile/")
    )
    graphs = [
        event
        for event in events
        if event.get("cat") == "user_annotation" and event.get("name", "").startswith("o65_graph/")
    ]
    owner: dict[object, str] = {}
    for event in events:
        if event.get("cat") == "cuda_runtime" and event.get("name") == "cudaGraphLaunch":
            for graph in graphs:
                if graph["ts"] <= event["ts"] <= graph["ts"] + graph["dur"]:
                    owner[event.get("args", {}).get("correlation")] = graph["name"].removeprefix("o65_graph/")
    work: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for event in events:
        if event.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"):
            name = owner.get(event.get("args", {}).get("correlation"), "outside graphs")
            work[name].append((event["ts"], event["ts"] + event["dur"]))
    busy = interval_union([interval for intervals in work.values() for interval in intervals])
    return {
        "wall_ms": root["dur"] / 1e3,
        "device_busy_ms": busy / 1e3,
        "device_idle_ms": (root["dur"] - busy) / 1e3,
        "graphs": {
            name: {
                "kernels": len(intervals),
                "busy_ms": interval_union(intervals) / 1e3,
                "kernel_ms": sum(stop - start for start, stop in intervals) / 1e3,
            }
            for name, intervals in work.items()
        },
    }


def profile_rollout(trainer: Trainer, config: NativeCli, native_config: dict[str, object]) -> None:
    """Time rollouts without PPO updates, then trace sampled ticks and split them by policy graph."""
    profiling = cast(ProfileConfig, config.profiling)
    engine = trainer.engine
    policies = engine.execution_policies()
    warm_started = time.perf_counter()
    for _ in range(profiling.warm_updates):
        engine.collect(trainer.update)
    report: dict[str, object] = {
        "configuration": native_config,
        "compilation": compilation_report(engine),
        "startup_capture_seconds": trainer.startup_seconds,
        "warm_seconds": time.perf_counter() - warm_started,
        "post_startup_captures": sum(cast(TransformerPolicy, policy).post_startup_captures for policy in policies),
    }
    try:
        for traces in (False, True):
            before_flops = sum(cast(TransformerPolicy, policy).work_flops.get("rollout", 0) for policy in policies)
            recorder = UpdateProfile(config.output, trainer.policy.device, traces=traces)
            engine.profile = recorder
            for policy in policies:
                cast(TransformerPolicy, policy).profile = recorder
            with recorder.phase("rollout"):
                rollout = engine.collect(trainer.update)
            ticks = recorder.host_seconds.get("rollout/tick", 0.0)
            boundaries = recorder.host_seconds.get("rollout/boundary_tick", 0.0)
            flops = (
                sum(cast(TransformerPolicy, policy).work_flops.get("rollout", 0) for policy in policies) - before_flops
            )
            peak = mfu.bf16_dense_peak_flops(torch.cuda.get_device_name(trainer.policy.device))
            report["traced" if traces else "timed"] = {
                "rollout_seconds": recorder.phases["rollout"]["wall_seconds"],
                "flops": flops,
                "mfu": None if peak is None else flops / (recorder.phases["rollout"]["wall_seconds"] * peak),
                "frames": rollout.frames,
                "tick_ms": 1e3 * ticks / max(1, recorder.calls.get("rollout/tick", 0)),
                "boundary_tick_ms": 1e3 * boundaries / max(1, recorder.calls.get("rollout/boundary_tick", 0)),
                "transfer_timings": recorder.transfer_timings,
            }
            if traces:
                report["ticks"] = {
                    path.name: graph_breakdown(path) for path in sorted(config.output.glob("rollout-*tick-*.json"))
                }
    finally:
        engine.profile = None
        for policy in policies:
            cast(TransformerPolicy, policy).profile = None
    device = trainer.policy.device
    report["memory"] = {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "device_total_bytes": torch.cuda.get_device_properties(device).total_memory,
    }
    (config.output / "rollout-profile.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    timed = cast(dict[str, object], report["timed"])
    print(
        json.dumps(
            {
                "rollout_profile": {key: value for key, value in timed.items() if key != "transfer_timings"},
                "post_startup_captures": report["post_startup_captures"],
                "startup_capture_seconds": trainer.startup_seconds,
            }
        ),
        flush=True,
    )
    for name, tick in cast(dict[str, dict[str, object]], report.get("ticks", {})).items():
        print(
            f"{name}: wall {tick['wall_ms']:.1f} ms busy {tick['device_busy_ms']:.1f} idle {tick['device_idle_ms']:.1f}",
            flush=True,
        )
        for graph, stats in sorted(cast(dict[str, dict[str, float]], tick["graphs"]).items()):
            print(
                f"   {graph:40s} {stats['kernels']:6.0f} kernels {stats['busy_ms']:7.2f} ms busy",
                flush=True,
            )


class UpdateProfile:
    """Time an update, then trace bounded samples in a separate update.

    Use unions of trace intervals for device busy time. CUDA event spans include
    idle gaps; scaled samples estimate only their recorded sites.
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
            for policy in trainer.engine.execution_policies():
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
                (output / "allocator.json").write_text(json.dumps(torch.cuda.memory_snapshot()))
            del batch
        finally:
            trainer.profile = trainer.engine.profile = None
            for policy in trainer.engine.execution_policies():
                if isinstance(policy, TransformerPolicy):
                    policy.profile = None


def compilation_report(engine: RolloutEngine) -> dict[str, object]:
    ticks: list[dict[str, object]] = []
    captures: list[dict[str, object]] = []
    variants: set[tuple[int, int, int, int]] = set()
    for name, policy in (
        ("student", engine.policy),
        *((f"anchor{key}", value) for key, value in engine.anchors.items()),
        *(
            (f"anchor_stack{index}", policy)
            for index, policy in enumerate(dict.fromkeys(stack for stack in engine.anchor_stacks if stack is not None))
        ),
    ):
        if not isinstance(policy, TransformerPolicy):
            raise TypeError("compilation report requires TransformerPolicy")
        for ((_, (buffer, batch)), append, replan, decoder_batch), graph in policy.tick_graphs.items():
            variants.add((batch, replan, append, decoder_batch))
            ticks.append(
                {
                    "policy": name,
                    "buffer": buffer,
                    "batch": batch,
                    "R": replan,
                    "decoder_batch": decoder_batch,
                    "append": append,
                    "compile_and_warmup_seconds": graph.preparation_seconds,
                    "capture_seconds": graph.capture_seconds,
                }
            )
        graphs: list[
            tuple[
                str,
                CapturedCall[ReplayOutput]
                | CapturedCall[Tensor]
                | CapturedCall[tuple[ActorLoss, ReplayOutput]]
                | CapturedCall[None],
            ]
        ] = [
            *(("restart", graph) for graph in policy.restart_graphs.values()),
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
        "format": "hal.o65.compilation-report.v2",
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
    if config.profiling.rollout_only:
        profile_rollout(trainer, config, native_config)
        return
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
            for policy in trainer.engine.execution_policies():
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
        for policy in trainer.engine.execution_policies():
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
                trainer.slot_updates,
                trainer.update,
                trainer.actor_steps,
                trainer.pool_start_update,
                trainer.warming,
                trainer.warmup_moments,
                trainer.evaluation_cursor,
                trainer.episode_metrics,
                trainer.engine.behavior_tracker.state,
                sorted(trainer.partial_episodes),
                trainer.anchor_win_history,
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
    for policy in (*trainer.engine.execution_policies(), *trainer.snapshot_slots):
        policy.clear_caches()


def qualify_development(config: ProxyCli) -> None:
    raw = json.loads(config.statistics.read_text())
    if not isinstance(raw, dict) or any(set(value) != {"mean", "std", "min", "max"} for value in raw.values()):
        raise ValueError("statistics must contain finalized FeatureStats objects")
    stats = {
        name: FeatureStats(value["mean"], value["std"], value["min"], value["max"]) for name, value in raw.items()
    }
    loaded = load_evaluation_checkpoint(
        config.checkpoint, device=torch.device(config.device), decoder_precision=config.decoder_precision
    )
    model = loaded.model.requires_grad_(True)
    model.percentile_head.requires_grad_(False)
    policy = TransformerPolicy(model, development_config(model.head_offsets), stats)
    report = {
        "checkpoint_sha256": sha256(config.checkpoint),
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
