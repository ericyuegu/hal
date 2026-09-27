"""Double-Q Implicit Q-Learning for four-frame controller chunks.

This is the paper-faithful IQL successor to experiment 027.  The deployed
policy remains a temporal controller, while Q1, Q2, V, and target critics are
fully independent training-only networks.  ``scalar`` is the canonical IQL
objective; ``hl_gauss`` is a controlled distributional-Q ablation.

Run:
    uv run experiments/030_iql.py
    uv run experiments/030_iql.py --cfg.q-objective hl_gauss
    uv run experiments/030_iql.py --eval runs/<run>/final.pt
"""

from __future__ import annotations

import copy
import functools
import hashlib
import importlib
import importlib.util
import math
import sys
import time
from collections.abc import Iterable
from collections.abc import Iterator
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR

import wandb
from hal import streams
from hal.data.feature_stats import FeatureStats
from hal.eval.h2h import run_h2h
from hal.eval.paired import summarize_paired
from hal.training.checkpoints import BackgroundUploader
from hal.training.checkpoints import download_latest
from hal.training.checkpoints import load_for_resume
from hal.training.checkpoints import save_checkpoint
from hal.training.dataloader import collate_windows
from hal.training.dataloader import make_loader
from hal.training.ego_stats import load_consolidated_stats
from hal.training.features import A_DIM
from hal.training.features import BASE_ACTION_PROJECTION
from hal.training.features import V6_PLAYER_COLUMNS
from hal.training.features import Context
from hal.training.features import FeatureProjection
from hal.training.features import TrainBatch
from hal.training.features import preprocess
from hal.training.features import stack_actions
from hal.training.replay_reservoir import make_reservoir_loader
from hal.training.runs import make_run_name
from hal.training.runs import profile
from hal.training.runs import setup_run_dir
from hal.training.trunk import varlen_flash_is_usable


def _experiment_module(name: str, filename: str):
    """Import a numeric experiment both as a script and under pytest's import mode."""
    qualified = f"experiments.{filename.removesuffix('.py')}"
    try:
        return importlib.import_module(qualified)
    except ModuleNotFoundError as error:
        if error.name != "experiments":
            raise
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load experiment dependency {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_base = _experiment_module("hal_exp027_for_030", "027_temporal_mtp.py")
_exp026 = _experiment_module("hal_exp026_for_030", "026_temporal_mtp.py")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXECUTED_CHUNK = 4
EVAL_MAX_PARALLEL = 4
GROUP_NAMES = _base.GROUP_NAMES
GROUP_VOCABS = _base.GROUP_VOCABS
GROUP_ORDER = _base.GROUP_ORDER
N_GROUPS = _base.N_GROUPS
StructuredControllerCodec = _base.StructuredControllerCodec
BF16Inference = _base.BF16Inference
_eval_inference_bucket = _base._eval_inference_bucket
RecedingHorizon = _base.RecedingHorizon
amp_context = _base.amp_context
canonical_context = _base.canonical_context
eval_vs_cpu = _base.eval_vs_cpu
make_policy = _base.make_policy
synthetic_context = _base.synthetic_context

_REWARD_SUFFIX = "iql_reward"
_RETURN_SUFFIX = "iql_return"
_CONTINUATION_SUFFIX = "iql_continuation"
EGO_REWARD_COLUMN = f"ego_{_REWARD_SUFFIX}"
EGO_RETURN_COLUMN = f"ego_{_RETURN_SUFFIX}"
EGO_CONTINUATION_COLUMN = f"ego_{_CONTINUATION_SUFFIX}"

REFERENCE_026_RUN = (
    "260810-071709_026_temporal_mtp_mtp026-d384-L8-h6-Lc128-t128x2-"
    "o1-2-3-4-5-6-9-12-16-20-s4-base_ranked-anon-1_production-seed0-d384-b512"
)
REFERENCE_026_SHA256 = "22333d1d61d6b648c757f0f1f3e887925fbb12a08fdffd1cb4ae72d6d6f2ef88"


@dataclass
class TrainConfig:
    # Deployed policy.
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 6
    attn_window: int = 0
    require_flex: bool = False
    attention_backend: str = "varlen_flash"
    L_ctx: int = 128

    decoder_arch_version: int = 3
    sample_chunk_length: int = EXECUTED_CHUNK
    head_offsets: tuple[int, ...] = (1, 2, 3, 4)
    temporal_d_model: int = 128
    temporal_layers: int = 2
    temporal_heads: int = 4
    temporal_ff_dim: int = 256
    group_head_dim: int = 256
    action_embed_dim: int = 16
    offset_embed_dim: int = 16
    group_order: tuple[str, ...] = GROUP_ORDER

    action_vocab: int = 1024
    action_state_embed_dim: int = 48
    char_vocab: int = 32
    char_dim: int = 8
    stage_vocab: int = 32
    stage_dim: int = 4
    observation_bundle: str = "base"

    exec_horizon: int = EXECUTED_CHUNK
    decode_temp: float = 1.0
    inference_mode: str = "compiled"
    compiled_inference_bucket: int | None = None

    # Independent training-only estimators.
    critic_d_model: int = 256
    critic_layers: int = 4
    critic_heads: int = 4
    critic_hidden_dim: int = 256

    q_objective: str = "scalar"  # scalar (canonical IQL) or hl_gauss
    iql_expectile: float = 0.7
    iql_temperature: float = 3.0
    iql_weight_max: float = 100.0
    iql_discount: float = 0.99  # one four-frame macro transition
    target_tau: float = 0.005
    iql_q_min: float = -4.0
    iql_q_max: float = 4.0
    iql_q_bins: int = 51
    iql_hl_gauss_sigma_ratio: float = 0.75
    iql_damage_shaping: float = 0.01
    iql_win_reward: float = 0.5

    seed: int = 0
    eval_seed: int = 0
    batch_size: int = 512
    grad_accum_steps: int = 1
    learning_rate: float = 3e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    max_steps: int = 16_384
    amp_dtype: str = "bfloat16"
    allow_tf32: bool = True
    compile_trunk: bool = True
    compile_temporal: bool = True

    wandb_log_code: bool = True
    val_every: int = 1024
    val_n_samples: int = 1192
    val_batch_size: int = 128
    ckpt_every: int = 1024
    eval_every: int = 4096
    eval_max_frames: int = 7200
    eval_n_matchups: int = 32
    final_eval_n_matchups: int = 96
    # One Dolphin plus one slippstream child per boot. Modal training containers
    # have 8 CPUs, so four concurrent boots avoid starving first-control startup.
    eval_max_parallel: int | None = EVAL_MAX_PARALLEL

    final_h2h_reference_run: str = REFERENCE_026_RUN
    final_h2h_reference_sha256: str = REFERENCE_026_SHA256
    final_h2h_n_configs: int = 64
    final_h2h_max_parallel: int = 32

    data_root: str = "data/processed/ranked-anonymized-1/mds-policy-v7"
    compact_data: bool = True
    mds_schema_version: int = 7
    cache_limit_gb: int = 128
    shuffle_block_size: int = 2000
    predownload: int = 512
    windows_per_replay: int = 4
    reservoir_capacity: int = 4096
    val_split: str = "val"
    num_workers: int = 16
    prefetch_factor: int = 2
    prefetch_batches: int = 4
    push_to_r2: bool = True


def validate_config(cfg: TrainConfig) -> None:
    integer_positive = {
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "L_ctx": cfg.L_ctx,
        "critic_d_model": cfg.critic_d_model,
        "critic_layers": cfg.critic_layers,
        "critic_heads": cfg.critic_heads,
        "critic_hidden_dim": cfg.critic_hidden_dim,
        "batch_size": cfg.batch_size,
        "max_steps": cfg.max_steps,
    }
    for name, value in integer_positive.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if cfg.sample_chunk_length != EXECUTED_CHUNK or tuple(cfg.head_offsets) != (1, 2, 3, 4):
        raise ValueError("experiment 030 trains and executes exactly offsets (1, 2, 3, 4)")
    if cfg.exec_horizon != EXECUTED_CHUNK:
        raise ValueError("experiment 030 always replans after four frames")
    if cfg.grad_accum_steps != 1:
        raise ValueError("experiment 030 freezes grad_accum_steps=1")
    if cfg.d_model % cfg.n_heads or cfg.critic_d_model % cfg.critic_heads:
        raise ValueError("model dimensions must be divisible by their head counts")
    if cfg.temporal_d_model % cfg.temporal_heads:
        raise ValueError("temporal_d_model must be divisible by temporal_heads")
    if cfg.decoder_arch_version != 3 or tuple(cfg.group_order) != GROUP_ORDER:
        raise ValueError("experiment 030 requires decoder v3 and the canonical controller group order")
    if cfg.q_objective not in ("scalar", "hl_gauss"):
        raise ValueError("q_objective must be 'scalar' or 'hl_gauss'")
    if not 0.0 < cfg.iql_expectile < 1.0:
        raise ValueError("iql_expectile must be in (0, 1)")
    if not 0.0 <= cfg.iql_discount <= 1.0:
        raise ValueError("iql_discount must be in [0, 1]")
    for name in ("iql_temperature", "iql_weight_max", "target_tau", "learning_rate", "adam_eps"):
        value = getattr(cfg, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if cfg.target_tau > 1.0:
        raise ValueError("target_tau must not exceed one")
    if cfg.iql_q_bins < 2 or cfg.iql_q_min >= cfg.iql_q_max:
        raise ValueError("HL-Gauss support must have at least two increasing centers")
    if cfg.iql_hl_gauss_sigma_ratio <= 0:
        raise ValueError("iql_hl_gauss_sigma_ratio must be positive")
    if cfg.observation_bundle not in ("base", "v6_lean"):
        raise ValueError("observation_bundle must be 'base' or 'v6_lean'")
    if cfg.decode_temp != 1.0 or cfg.inference_mode not in ("compiled", "eager"):
        raise ValueError("decode_temp is fixed at 1 and inference_mode must be compiled or eager")
    if cfg.amp_dtype not in ("bfloat16", "float32"):
        raise ValueError("amp_dtype must be bfloat16 or float32")
    if cfg.attention_backend != "varlen_flash" or cfg.require_flex:
        raise ValueError("experiment 030 freezes attention_backend='varlen_flash' and require_flex=False")
    if cfg.amp_dtype != "bfloat16":
        raise ValueError("varlen_flash requires amp_dtype='bfloat16'")
    if cfg.final_h2h_reference_run != REFERENCE_026_RUN:
        raise ValueError("the final H2H reference run is pinned")
    if cfg.final_h2h_reference_sha256.lower() != REFERENCE_026_SHA256:
        raise ValueError("the final H2H reference SHA-256 is pinned")
    if cfg.final_h2h_n_configs != 64 or cfg.final_h2h_max_parallel != 32:
        raise ValueError("the final H2H protocol is frozen to 64 configs and parallelism 32")
    if cfg.reservoir_capacity < 2 * cfg.batch_size:
        raise ValueError("reservoir_capacity must be at least twice batch_size")


def micro_batch_size(cfg: TrainConfig) -> int:
    return cfg.batch_size


class GPT(_base.GPT):
    """Inference policy without experiment-027's attached critic heads."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__(cfg)
        del self.q_head
        del self.value_head
        del self.q_bin_values


class StateEncoder(GPT):
    """Private causal encoder used by exactly one Q or V network."""

    def __init__(self, cfg: TrainConfig) -> None:
        critic_cfg = replace(
            cfg,
            d_model=cfg.critic_d_model,
            n_layers=cfg.critic_layers,
            n_heads=cfg.critic_heads,
            compile_trunk=False,
            compile_temporal=False,
        )
        super().__init__(critic_cfg)
        del self.temporal

    def final(self, context: Context) -> Tensor:
        actions = self.codec.quantize(stack_actions(context.features))
        return self(context.features, context.ctx_pad, actions)[:, -1]


def _mlp(d_in: int, hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, d_out)
    )


class QNetwork(nn.Module):
    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = StateEncoder(cfg)
        action_width = EXECUTED_CHUNK * N_GROUPS * cfg.action_embed_dim
        output = 1 if cfg.q_objective == "scalar" else cfg.iql_q_bins
        self.head = _mlp(cfg.critic_d_model + action_width, cfg.critic_hidden_dim, output)
        self.register_buffer("bin_values", torch.linspace(cfg.iql_q_min, cfg.iql_q_max, cfg.iql_q_bins))

    def logits(self, context: Context, chunk: Tensor) -> Tensor:
        if chunk.shape[1:] != (EXECUTED_CHUNK, N_GROUPS):
            raise ValueError(f"Q action must be [B, 4, {N_GROUPS}], got {tuple(chunk.shape)}")
        state = self.encoder.final(context)
        action = self.encoder.codec.embed_frame(chunk).flatten(start_dim=1)
        return self.head(torch.cat((state, action), dim=-1)).float()

    def decode(self, output: Tensor) -> Tensor:
        if self.cfg.q_objective == "scalar":
            return output.squeeze(-1).float()
        return (torch.softmax(output.float(), dim=-1) * self.bin_values.float()).sum(dim=-1)

    def forward(self, context: Context, chunk: Tensor) -> Tensor:
        return self.decode(self.logits(context, chunk))


class ValueNetwork(nn.Module):
    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.encoder = StateEncoder(cfg)
        self.head = _mlp(cfg.critic_d_model, cfg.critic_hidden_dim, 1)

    def forward(self, context: Context) -> Tensor:
        return self.head(self.encoder.final(context)).float().squeeze(-1)


class TrainingModel(nn.Module):
    """Full resumable model. Only ``policy`` is loaded for deployment."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.policy = GPT(cfg)
        self.q1 = QNetwork(cfg)
        self.q2 = QNetwork(cfg)
        self.value = ValueNetwork(cfg)
        self.target_q1 = copy.deepcopy(self.q1).requires_grad_(False)
        self.target_q2 = copy.deepcopy(self.q2).requires_grad_(False)

    def train(self, mode: bool = True) -> TrainingModel:
        super().train(mode)
        # Target networks are frozen estimators, never stochastic training modules.
        self.target_q1.eval()
        self.target_q2.eval()
        return self


def parameter_id_sets(model: TrainingModel) -> dict[str, set[int]]:
    return {
        name: {id(parameter) for parameter in module.parameters()}
        for name, module in (("policy", model.policy), ("q1", model.q1), ("q2", model.q2), ("value", model.value))
    }


def polyak_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_parameter, parameter in zip(target.parameters(), online.parameters(), strict=True):
            target_parameter.mul_(1.0 - tau).add_(parameter, alpha=tau)
        for target_buffer, buffer in zip(target.buffers(), online.buffers(), strict=True):
            target_buffer.copy_(buffer)


@dataclass(frozen=True, slots=True)
class IQLBatch:
    batch: TrainBatch
    extended_context: Context  # [B, L_ctx + 4]
    rewards: Tensor  # [B, 4]
    returns: Tensor  # [B], diagnostic Monte-Carlo return from the first action frame
    continuation: Tensor  # [B], zero iff s' is terminal

    def to(self, device: str | torch.device) -> IQLBatch:
        return IQLBatch(
            self.batch.to(device),
            self.extended_context.to(device),
            self.rewards.to(device, non_blocking=True),
            self.returns.to(device, non_blocking=True),
            self.continuation.to(device, non_blocking=True),
        )

    def pin_memory(self) -> IQLBatch:
        return IQLBatch(
            self.batch.pin_memory(),
            self.extended_context.pin_memory(),
            self.rewards.pin_memory(),
            self.returns.pin_memory(),
            self.continuation.pin_memory(),
        )

    def take(self, count: int) -> IQLBatch:
        context = self.batch.context
        return IQLBatch(
            TrainBatch(
                context=Context(
                    features={name: value[:count] for name, value in context.features.items()},
                    ctx_pad=context.ctx_pad[:count],
                    slot_ids=None if context.slot_ids is None else context.slot_ids[:count],
                    reset=None if context.reset is None else context.reset[:count],
                ),
                target=self.batch.target[:count],
                replay_ids=None if self.batch.replay_ids is None else self.batch.replay_ids[:count],
            ),
            Context(
                features={name: value[:count] for name, value in self.extended_context.features.items()},
                ctx_pad=self.extended_context.ctx_pad[:count],
            ),
            self.rewards[:count],
            self.returns[:count],
            self.continuation[:count],
        )


def frame_gamma(cfg: TrainConfig) -> float:
    return cfg.iql_discount ** (1.0 / EXECUTED_CHUNK)


def label_iql_replay(sample: dict, *, discount: float, damage_shaping: float, win_reward: float) -> dict:
    out = dict(sample)
    gamma = discount ** (1.0 / EXECUTED_CHUNK)
    length = len(sample["p1_stock"])
    continuation = np.zeros(length, dtype=np.float32)
    if length > EXECUTED_CHUNK + 1:
        continuation[: -(EXECUTED_CHUNK + 1)] = 1.0
    for port, other in (("p1", "p2"), ("p2", "p1")):
        reward = _base.frame_reward(sample, ego=port, opp=other, damage_shaping=damage_shaping, win_reward=win_reward)
        out[f"{port}_{_REWARD_SUFFIX}"] = reward
        out[f"{port}_{_RETURN_SUFFIX}"] = _base.discounted_returns(reward, gamma)
        out[f"{port}_{_CONTINUATION_SUFFIX}"] = continuation.copy()
    return out


def aligned_iql_labels(windows: list[dict], L_ctx: int) -> tuple[Tensor, Tensor, Tensor]:
    rewards = np.stack([window[EGO_REWARD_COLUMN][L_ctx : L_ctx + EXECUTED_CHUNK] for window in windows])
    returns = np.asarray([window[EGO_RETURN_COLUMN][L_ctx] for window in windows], dtype=np.float32)
    continuation = np.asarray([window[EGO_CONTINUATION_COLUMN][L_ctx - 1] for window in windows], dtype=np.float32)
    return tuple(torch.from_numpy(np.ascontiguousarray(value)) for value in (rewards, returns, continuation))  # type: ignore[return-value]


def collate_iql_batch(
    windows: list[dict],
    batch: TrainBatch,
    *,
    stats: dict[str, FeatureStats],
    L_ctx: int,
    extra,
    projection: FeatureProjection | None,
) -> IQLBatch:
    stacked = collate_windows(windows)
    features = preprocess(stacked, stats, extra=extra, projection=projection)
    extended = Context(
        features={name: value[:, : L_ctx + EXECUTED_CHUNK] for name, value in features.items()},
        ctx_pad=batch.context.ctx_pad,
    )
    rewards, returns, continuation = aligned_iql_labels(windows, L_ctx)
    return IQLBatch(batch, extended, rewards, returns, continuation)


def rolling_next_context(extended: Context, L_ctx: int) -> Context:
    wrong = {name: tuple(value.shape) for name, value in extended.features.items() if value.shape[1] != L_ctx + 4}
    if wrong:
        raise ValueError(f"extended features must have length L_ctx+4: {wrong}")
    return Context(
        features={name: value[:, EXECUTED_CHUNK:] for name, value in extended.features.items()},
        ctx_pad=(extended.ctx_pad - EXECUTED_CHUNK).clamp_min(0),
        slot_ids=extended.slot_ids,
        reset=extended.reset,
    )


def quantized_chunk(policy: GPT, batch: TrainBatch) -> Tensor:
    if batch.target.shape[1:] != (EXECUTED_CHUNK, A_DIM):
        raise ValueError(f"target must be [B, 4, {A_DIM}]")
    return policy.codec.quantize(batch.target)


def actor_nll(policy: GPT, batch: TrainBatch, chunk: Tensor | None = None) -> Tensor:
    history = policy.codec.quantize(stack_actions(batch.context.features))
    target = quantized_chunk(policy, batch) if chunk is None else chunk
    hidden = policy(batch.context.features, batch.context.ctx_pad, history)[:, -1:]
    losses = policy.temporal.teacher_forced_nll(hidden, history[:, -1:], target[:, None])
    # One macro-action likelihood, normalized across its 4 x 4 categorical
    # decisions so its gradient scale is independent of the chosen factorization.
    return losses[:, 0].mean(dim=(-2, -1))


def expectile_loss(q: Tensor, value: Tensor, expectile: float) -> Tensor:
    delta = q.detach() - value
    weight = torch.where(delta > 0, expectile, 1.0 - expectile)
    return (weight * delta.square()).mean()


def actor_weights(advantage: Tensor, *, temperature: float, weight_max: float) -> tuple[Tensor, Tensor]:
    if advantage.requires_grad:
        raise ValueError("actor weights require a detached advantage")
    scaled = advantage.float() * temperature
    log_max = math.log(weight_max)
    clipped = scaled >= log_max
    return torch.exp(scaled.clamp(max=log_max)), clipped


def chunk_td_target(rewards: Tensor, next_value: Tensor, continuation: Tensor, *, macro_discount: float) -> Tensor:
    if rewards.shape != (*next_value.shape, EXECUTED_CHUNK) or continuation.shape != next_value.shape:
        raise ValueError("TD target needs rewards [B,4], next_value [B], and continuation [B]")
    gamma = macro_discount ** (1.0 / EXECUTED_CHUNK)
    discounts = rewards.new_tensor([gamma**index for index in range(EXECUTED_CHUNK)])
    return (rewards.float() * discounts).sum(-1) + macro_discount * continuation.float() * next_value.detach()


def encode_hl_gauss(target: Tensor, bins: Tensor, sigma_ratio: float = 0.75) -> Tensor:
    if bins.ndim != 1 or bins.numel() < 2:
        raise ValueError("HL-Gauss bins must be a one-dimensional support")
    spacing = bins[1] - bins[0]
    sigma = spacing * sigma_ratio
    midpoints = 0.5 * (bins[:-1] + bins[1:])
    edges = torch.cat((bins.new_tensor([float("-inf")]), midpoints, bins.new_tensor([float("inf")])))
    z = (edges - target.detach().float()[..., None]) / sigma
    cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    probabilities = (cdf[..., 1:] - cdf[..., :-1]).clamp_min(0.0)
    return probabilities / probabilities.sum(-1, keepdim=True).clamp_min(torch.finfo(probabilities.dtype).tiny)


def q_loss(network: QNetwork, context: Context, chunk: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    output = network.logits(context, chunk)
    prediction = network.decode(output)
    if network.cfg.q_objective == "scalar":
        return F.mse_loss(prediction, target.detach()), prediction
    encoded = encode_hl_gauss(target, network.bin_values.float(), network.cfg.iql_hl_gauss_sigma_ratio)
    loss = -(encoded * F.log_softmax(output.float(), dim=-1)).sum(-1).mean()
    return loss, prediction


@dataclass(frozen=True, slots=True)
class StepMetrics:
    value_loss: float
    actor_loss: float
    unweighted_actor_loss: float
    q1_loss: float
    q2_loss: float
    grad_value: float
    grad_actor: float
    grad_q: float
    q1: Tensor
    q2: Tensor
    q_min: Tensor
    value: Tensor
    advantage: Tensor
    weight: Tensor
    clipped: Tensor
    target: Tensor
    continuation: Tensor


def _grad_norm(parameters: Iterable[nn.Parameter]) -> float:
    selected = tuple(parameters)
    total = torch.zeros((), device=selected[0].device)
    for parameter in selected:
        if parameter.grad is not None:
            total += parameter.grad.detach().float().square().sum()
    return float(total.sqrt())


def iql_update(
    model: TrainingModel,
    batch: IQLBatch,
    optimizers: OptimizerBundle,
    cfg: TrainConfig,
) -> StepMetrics:
    policy, q1, q2, value = model.policy, model.q1, model.q2, model.value
    device = batch.rewards.device
    chunk = quantized_chunk(policy, batch.batch)
    next_context = rolling_next_context(batch.extended_context, cfg.L_ctx)

    # 1. V learns an upper expectile of the conservative frozen double-Q.
    optimizers.value.zero_grad(set_to_none=True)
    with torch.no_grad(), amp_context(cfg, device):
        target_q1 = model.target_q1(batch.batch.context, chunk)
        target_q2 = model.target_q2(batch.batch.context, chunk)
        q_min = torch.minimum(target_q1, target_q2)
    with amp_context(cfg, device):
        value_prediction = value(batch.batch.context)
        value_objective = expectile_loss(q_min, value_prediction, cfg.iql_expectile)
    value_objective.backward()
    grad_value = _grad_norm(value.parameters())
    optimizers.value.step()

    # 2. Actor uses the freshly updated V and the frozen target critics.
    optimizers.actor.zero_grad(set_to_none=True)
    with torch.no_grad(), amp_context(cfg, device):
        updated_value = value(batch.batch.context)
        advantage = (q_min - updated_value).detach()
        weight, clipped = actor_weights(advantage, temperature=cfg.iql_temperature, weight_max=cfg.iql_weight_max)
    with amp_context(cfg, device):
        nll = actor_nll(policy, batch.batch, chunk)
        actor_objective = (weight * nll).mean()
    actor_objective.backward()
    grad_actor = _grad_norm(policy.parameters())
    optimizers.actor.step()

    # 3. Both online Qs regress to the same one-step target from the new V.
    optimizers.q.zero_grad(set_to_none=True)
    with torch.no_grad(), amp_context(cfg, device):
        next_value = value(next_context)
        target = chunk_td_target(
            batch.rewards, next_value, batch.continuation, macro_discount=cfg.iql_discount
        ).detach()
    with amp_context(cfg, device):
        q1_objective, q1_prediction = q_loss(q1, batch.batch.context, chunk, target)
        q2_objective, q2_prediction = q_loss(q2, batch.batch.context, chunk, target)
        q_objective = q1_objective + q2_objective
    q_objective.backward()
    grad_q = _grad_norm(list(q1.parameters()) + list(q2.parameters()))
    optimizers.q.step()

    # 4. Slow target critics track the just-updated online critics.
    polyak_update(model.target_q1, q1, cfg.target_tau)
    polyak_update(model.target_q2, q2, cfg.target_tau)
    return StepMetrics(
        float(value_objective.detach()),
        float(actor_objective.detach()),
        float(nll.detach().mean()),
        float(q1_objective.detach()),
        float(q2_objective.detach()),
        grad_value,
        grad_actor,
        grad_q,
        q1_prediction.detach(),
        q2_prediction.detach(),
        q_min.detach(),
        updated_value.detach(),
        advantage,
        weight.detach(),
        clipped.detach(),
        target.detach(),
        batch.continuation.detach(),
    )


class OptimizerBundle:
    def __init__(self, actor: torch.optim.Optimizer, q: torch.optim.Optimizer, value: torch.optim.Optimizer) -> None:
        self.actor, self.q, self.value = actor, q, value

    def state_dict(self) -> dict[str, Any]:
        return {"actor": self.actor.state_dict(), "q": self.q.state_dict(), "value": self.value.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.q.load_state_dict(state["q"])
        self.value.load_state_dict(state["value"])


class SchedulerBundle:
    def __init__(self, actor: LambdaLR) -> None:
        self.actor = actor

    def state_dict(self) -> dict[str, Any]:
        return {"actor": self.actor.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.actor.load_state_dict(state["actor"])


def make_optimizers(model: TrainingModel, cfg: TrainConfig) -> tuple[OptimizerBundle, SchedulerBundle]:
    kwargs = dict(
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_eps,
        weight_decay=0.0,
    )
    actor = torch.optim.Adam(model.policy.parameters(), **kwargs)
    q = torch.optim.Adam(list(model.q1.parameters()) + list(model.q2.parameters()), **kwargs)
    value = torch.optim.Adam(model.value.parameters(), **kwargs)
    optimizers = OptimizerBundle(actor, q, value)

    def cosine(step: int) -> float:
        return 0.5 * (1.0 + math.cos(math.pi * min(step / max(cfg.max_steps, 1), 1.0)))

    return optimizers, SchedulerBundle(LambdaLR(actor, cosine))


def effective_sample_size(weight: Tensor) -> float:
    values = weight.float()
    return float(values.sum().square() / values.square().sum().clamp_min(1e-12))


def _correlation(left: Tensor, right: Tensor) -> float:
    x, y = left.double(), right.double()
    x, y = x - x.mean(), y - y.mean()
    denominator = x.square().sum().sqrt() * y.square().sum().sqrt()
    return float((x * y).sum() / denominator) if denominator > 0 else float("nan")


def _distribution_metrics(name: str, values: Tensor) -> dict[str, float]:
    selected = values.detach().float()
    quantiles = torch.quantile(selected, selected.new_tensor([0.05, 0.5, 0.95]))
    return {
        f"{name}_mean": float(selected.mean()),
        f"{name}_std": float(selected.std(unbiased=False)),
        f"{name}_p05": float(quantiles[0]),
        f"{name}_p50": float(quantiles[1]),
        f"{name}_p95": float(quantiles[2]),
    }


def training_metrics(parts: StepMetrics, cfg: TrainConfig) -> dict[str, float]:
    disagreement = (parts.q1 - parts.q2).abs()
    delta = parts.q_min - parts.value
    positive = delta.clamp_min(0).mean()
    negative = (-delta).clamp_min(0).mean()
    balance_denominator = (1.0 - cfg.iql_expectile) * negative
    metrics = {
        "value_loss": parts.value_loss,
        "actor_loss": parts.actor_loss,
        "actor_nll": parts.unweighted_actor_loss,
        "q1_loss": parts.q1_loss,
        "q2_loss": parts.q2_loss,
        "q1_mean": float(parts.q1.mean()),
        "q2_mean": float(parts.q2.mean()),
        "q_min_mean": float(parts.q_min.mean()),
        "value_mean": float(parts.value.mean()),
        "advantage_mean": float(parts.advantage.mean()),
        "advantage_positive_frac": float((parts.advantage > 0).float().mean()),
        "actor_weight_mean": float(parts.weight.mean()),
        "actor_weight_p95": float(torch.quantile(parts.weight.float(), 0.95)),
        "actor_weight_cap_frac": float(parts.clipped.float().mean()),
        "actor_weight_ess": effective_sample_size(parts.weight),
        "actor_weight_ess_frac": effective_sample_size(parts.weight) / max(parts.weight.numel(), 1),
        "q_disagreement_mae": float(disagreement.mean()),
        "q_correlation": _correlation(parts.q1, parts.q2),
        "td_target_mean": float(parts.target.mean()),
        "q1_td_residual_mean": float((parts.target - parts.q1).mean()),
        "q2_td_residual_mean": float((parts.target - parts.q2).mean()),
        "terminal_frac": float((parts.continuation == 0).float().mean()),
        "grad_norm_actor": parts.grad_actor,
        "grad_norm_q": parts.grad_q,
        "grad_norm_value": parts.grad_value,
        "expectile_positive_mass": float(positive),
        "expectile_negative_mass": float(negative),
        "expectile_balance": (
            float(cfg.iql_expectile * positive / balance_denominator) if balance_denominator > 0 else float("nan")
        ),
    }
    for name, values in (
        ("q1", parts.q1),
        ("q2", parts.q2),
        ("q_min", parts.q_min),
        ("value", parts.value),
        ("advantage", parts.advantage),
        ("td_target", parts.target),
    ):
        metrics.update(_distribution_metrics(name, values))
    if cfg.q_objective == "hl_gauss":
        metrics["q_target_oob_frac"] = float(
            ((parts.target < cfg.iql_q_min) | (parts.target > cfg.iql_q_max)).float().mean()
        )
    return metrics


@torch.no_grad()
def val_metrics(model: TrainingModel, batches: list[IQLBatch], cfg: TrainConfig) -> dict[str, float]:
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    rows: dict[str, list[Tensor]] = {name: [] for name in ("q1", "q2", "q", "v", "g", "target")}
    actor_nll_sum = q1_loss_sum = q2_loss_sum = value_loss_sum = samples = 0.0
    q1_edge_sum = q2_edge_sum = 0.0
    try:
        for cpu_batch in batches:
            batch = cpu_batch.to(device)
            with amp_context(cfg, device):
                chunk = quantized_chunk(model.policy, batch.batch)
                next_context = rolling_next_context(batch.extended_context, cfg.L_ctx)
                q1 = model.target_q1(batch.batch.context, chunk)
                q2 = model.target_q2(batch.batch.context, chunk)
                q = torch.minimum(q1, q2)
                value = model.value(batch.batch.context)
                target = chunk_td_target(
                    batch.rewards, model.value(next_context), batch.continuation, macro_discount=cfg.iql_discount
                )
                loss1, _ = q_loss(model.q1, batch.batch.context, chunk, target)
                loss2, _ = q_loss(model.q2, batch.batch.context, chunk, target)
                actor_loss = actor_nll(model.policy, batch.batch, chunk).mean()
            count = chunk.shape[0]
            actor_nll_sum += float(actor_loss) * count
            q1_loss_sum += float(loss1) * count
            q2_loss_sum += float(loss2) * count
            value_loss_sum += float(expectile_loss(q, value, cfg.iql_expectile)) * count
            if cfg.q_objective == "hl_gauss":
                with amp_context(cfg, device):
                    probability1 = torch.softmax(model.q1.logits(batch.batch.context, chunk).float(), dim=-1)
                    probability2 = torch.softmax(model.q2.logits(batch.batch.context, chunk).float(), dim=-1)
                q1_edge_sum += float((probability1[:, 0] + probability1[:, -1]).sum())
                q2_edge_sum += float((probability2[:, 0] + probability2[:, -1]).sum())
            samples += count
            for name, tensor in (
                ("q1", q1),
                ("q2", q2),
                ("q", q),
                ("v", value),
                ("g", batch.returns),
                ("target", target),
            ):
                rows[name].append(tensor.cpu())
    finally:
        model.train(was_training)
    joined = {name: torch.cat(values).float() for name, values in rows.items()}
    out = {
        "actor_nll": actor_nll_sum / samples,
        "q1_loss": q1_loss_sum / samples,
        "q2_loss": q2_loss_sum / samples,
        "value_loss": value_loss_sum / samples,
        "q1_q2_corr": _correlation(joined["q1"], joined["q2"]),
        "q_disagreement_mae": float((joined["q1"] - joined["q2"]).abs().mean()),
        "corr_q_g": _correlation(joined["q"], joined["g"]),
        "corr_value_g": _correlation(joined["v"], joined["g"]),
        "q_g_mae": float((joined["q"] - joined["g"]).abs().mean()),
        "value_g_mae": float((joined["v"] - joined["g"]).abs().mean()),
        "td_target_mean": float(joined["target"].mean()),
    }
    if cfg.q_objective == "hl_gauss":
        out.update(
            {
                "q_target_oob_frac": float(
                    ((joined["target"] < cfg.iql_q_min) | (joined["target"] > cfg.iql_q_max)).float().mean()
                ),
                "q1_edge_mass": q1_edge_sum / samples,
                "q2_edge_mass": q2_edge_sum / samples,
            }
        )
    return out


def validate_batch_geometry(batch: IQLBatch, cfg: TrainConfig, expected_batch_size: int | None = None) -> None:
    size = batch.batch.target.shape[0]
    if expected_batch_size is not None and size != expected_batch_size:
        raise ValueError(f"fixed batch must contain {expected_batch_size} rows, got {size}")
    if batch.batch.target.shape != (size, 4, A_DIM):
        raise ValueError("target geometry is not [B,4,A_DIM]")
    if batch.batch.context.ctx_pad.shape != (size,) or batch.extended_context.ctx_pad.shape != (size,):
        raise ValueError("ctx_pad shape does not match batch")
    if any(value.shape[:2] != (size, cfg.L_ctx) for value in batch.batch.context.features.values()):
        raise ValueError("current context feature geometry is wrong")
    if any(value.shape[:2] != (size, cfg.L_ctx + 4) for value in batch.extended_context.features.values()):
        raise ValueError("extended context feature geometry is wrong")
    if batch.rewards.shape != (size, 4) or batch.returns.shape != (size,) or batch.continuation.shape != (size,):
        raise ValueError("IQL label geometry is wrong")
    if not torch.all((batch.continuation == 0) | (batch.continuation == 1)):
        raise ValueError("continuation must be binary")


def cache_validation(loader: Iterable[IQLBatch], n_samples: int) -> list[IQLBatch]:
    batches, count = [], 0
    for batch in loader:
        if count >= n_samples:
            break
        batch = batch.take(min(batch.batch.target.shape[0], n_samples - count))
        batches.append(batch)
        count += batch.batch.target.shape[0]
    if count != n_samples:
        raise RuntimeError(f"validation yielded {count} samples, expected {n_samples}")
    return batches


def device_batches(
    cpu_batches: list[IQLBatch], device: str | torch.device, copy_stream: torch.cuda.Stream | None
) -> Iterator[IQLBatch]:
    # A batch is one optimizer step in 030; this remains an iterator for testability.
    del copy_stream
    for batch in cpu_batches:
        yield batch.to(device)


def loader_kwargs(cfg: TrainConfig, stats: dict[str, FeatureStats]) -> dict[str, Any]:
    v6 = cfg.observation_bundle == "v6_lean"
    projection = None
    if not v6:
        projection = FeatureProjection(
            columns=BASE_ACTION_PROJECTION.columns | {EGO_REWARD_COLUMN, EGO_RETURN_COLUMN, EGO_CONTINUATION_COLUMN},
            derive_spatial=BASE_ACTION_PROJECTION.derive_spatial,
        )
    return dict(
        data_root=cfg.data_root,
        remote=streams.remote_for_local(cfg.data_root),
        cache_limit=f"{cfg.cache_limit_gb}gb",
        shuffle_block_size=cfg.shuffle_block_size,
        stats=stats,
        L_ctx=cfg.L_ctx,
        L_chunk=EXECUTED_CHUNK,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        schema_version=cfg.mds_schema_version,
        extra=V6_PLAYER_COLUMNS if v6 else None,
        projection=projection,
    )


def _make_loaders(cfg: TrainConfig, stats: dict[str, FeatureStats]):
    kwargs = loader_kwargs(cfg, stats)
    replay_transform = functools.partial(
        label_iql_replay,
        discount=cfg.iql_discount,
        damage_shaping=cfg.iql_damage_shaping,
        win_reward=cfg.iql_win_reward,
    )
    batch_transform = functools.partial(
        collate_iql_batch,
        stats=stats,
        L_ctx=cfg.L_ctx,
        extra=kwargs["extra"],
        projection=kwargs["projection"],
    )
    if cfg.compact_data:
        train_loader = make_reservoir_loader(
            split="train",
            num_workers=cfg.num_workers,
            prefetch_factor=cfg.prefetch_factor,
            predownload=cfg.predownload,
            windows_per_replay=cfg.windows_per_replay,
            reservoir_capacity=cfg.reservoir_capacity,
            prefetch_batches=cfg.prefetch_batches,
            replay_transform=replay_transform,
            batch_transform=batch_transform,
            **kwargs,
        )
    else:
        train_loader = make_loader(
            split="train",
            num_workers=cfg.num_workers,
            prefetch_factor=cfg.prefetch_factor,
            windows_per_replay=cfg.windows_per_replay,
            compact=False,
            replay_transform=replay_transform,
            batch_transform=batch_transform,
            **kwargs,
        )
    val_loader = make_loader(
        split=cfg.val_split,
        num_workers=0,
        compact=cfg.compact_data,
        replay_transform=replay_transform,
        batch_transform=batch_transform,
        **{**kwargs, "batch_size": cfg.val_batch_size},
    )
    return train_loader, cache_validation(val_loader, cfg.val_n_samples)


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_h2h_reference(cfg: TrainConfig, run_dir: Path) -> Path:
    local = Path("runs") / cfg.final_h2h_reference_run / "final.pt"
    checkpoint = (
        local.resolve()
        if local.is_file()
        else download_latest(cfg.final_h2h_reference_run, run_dir / "h2h_reference", name="final.pt")
    )
    if checkpoint is None:
        raise RuntimeError(f"no final.pt for pinned H2H reference {cfg.final_h2h_reference_run!r}")
    actual = _checkpoint_sha256(checkpoint)
    if actual != cfg.final_h2h_reference_sha256:
        raise RuntimeError(f"H2H reference SHA-256 mismatch: expected {cfg.final_h2h_reference_sha256}, got {actual}")
    return checkpoint


def final_h2h(
    policy: GPT,
    stats: dict[str, FeatureStats],
    cfg: TrainConfig,
    *,
    run_dir: Path,
    reference: Path,
    uploader: BackgroundUploader | None,
    inference: BF16Inference | None = None,
) -> dict[str, object]:
    reference_model, reference_cfg, reference_stats, reference_state = _exp026.load_checkpoint(str(reference))
    if cfg.L_ctx != reference_cfg.L_ctx or cfg.exec_horizon != reference_cfg.exec_horizon:
        raise RuntimeError("030 and 026 H2H context/execution protocols differ")
    if tuple(cfg.head_offsets[:4]) != tuple(reference_cfg.head_offsets[:4]):
        raise RuntimeError("both H2H policies must expose offsets 1..4")
    if cfg.observation_bundle != reference_cfg.observation_bundle or cfg.decode_temp != reference_cfg.decode_temp:
        raise RuntimeError("030 and 026 H2H observation/decoding protocols differ")

    def build_self(seed: int):
        return make_policy(policy, stats, cfg, decode_seed=seed, inference=inference)

    def build_reference(seed: int):
        return _exp026.make_policy(reference_model, reference_stats, reference_cfg, decode_seed=seed)

    out_dir = run_dir / "h2h_final"

    def upload_orientation(_orientation: int) -> None:
        if uploader is not None:
            uploader.upload_tree(out_dir, base=run_dir)

    try:
        records = run_h2h(
            build_self,
            build_reference,
            name_a="030-iql",
            name_b="026-reference",
            n_configs=cfg.final_h2h_n_configs,
            out_dir=out_dir,
            max_frames=cfg.eval_max_frames,
            max_parallel=cfg.final_h2h_max_parallel,
            seed=cfg.eval_seed,
            meta={
                "models": {
                    "030-iql": {"experiment": str(Path(__file__)), "step": cfg.max_steps},
                    "026-reference": {
                        "experiment": "experiments/026_temporal_mtp.py",
                        "checkpoint": str(reference),
                        "checkpoint_sha256": cfg.final_h2h_reference_sha256,
                        "step": int(reference_state["step"]),
                    },
                }
            },
            on_orientation_done=upload_orientation,
        )
    finally:
        if uploader is not None:
            uploader.upload_tree(out_dir, base=run_dir)
    summary = summarize_paired(records, focal_model="030-iql")
    print(summary.format_table(), flush=True)
    return summary.as_dict()


def subsystem_parameter_counts(model: TrainingModel) -> dict[str, int]:
    return {
        name: sum(parameter.numel() for parameter in module.parameters())
        for name, module in (
            ("policy", model.policy),
            ("q1", model.q1),
            ("q2", model.q2),
            ("value", model.value),
            ("target_q1", model.target_q1),
            ("target_q2", model.target_q2),
        )
    }


def model_tag(cfg: TrainConfig) -> str:
    return (
        f"iql030-{cfg.q_objective}-p{cfg.d_model}x{cfg.n_layers}-"
        f"c{cfg.critic_d_model}x{cfg.critic_layers}-b{cfg.batch_size}-"
        f"tau{cfg.iql_expectile:g}-g{cfg.iql_discount:g}-t{cfg.iql_temperature:g}-varlenflash"
    )


def train(
    cfg: TrainConfig,
    stats: dict[str, FeatureStats],
    *,
    comment: str = "",
    resume_run: str | None = None,
    resume_state: dict | None = None,
) -> None:
    validate_config(cfg)
    longest_context = cfg.L_ctx + EXECUTED_CHUNK
    head_dims = (cfg.d_model // cfg.n_heads, cfg.critic_d_model // cfg.critic_heads)
    if DEVICE == "cuda" and not all(
        varlen_flash_is_usable("cuda", longest_context, heads, head_dim)
        for heads, head_dim in ((cfg.n_heads, head_dims[0]), (cfg.critic_heads, head_dims[1]))
    ):
        raise RuntimeError("experiment 030 requires native varlen FlashAttention forward/backward on CUDA")
    run_name = resume_run or make_run_name(Path(__file__).stem, model_tag(cfg), cfg.data_root, comment)
    run_dir, replay_dir = setup_run_dir(run_name)
    reference = resolve_h2h_reference(cfg, run_dir)
    uploader = BackgroundUploader(run_name) if cfg.push_to_r2 else None
    wandb.init(
        project="hal",
        name=run_name,
        id=None if resume_state is None else resume_state.get("wandb_id"),
        resume="allow" if resume_state is not None else None,
        tags=["gpt", "iql", "double-q", "030", cfg.q_objective],
        config=asdict(cfg),
    )
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high" if cfg.allow_tf32 else "highest")
    model = TrainingModel(cfg).to(DEVICE)
    optimizers, schedulers = make_optimizers(model, cfg)
    start_step = 0
    if resume_state is not None:
        model.load_state_dict(resume_state["model"], strict=True)
        optimizers.load_state_dict(resume_state["opt"])
        schedulers.load_state_dict(resume_state["sched"])
        start_step = int(resume_state["step"]) + 1
    if wandb.run is not None:
        for name, count in subsystem_parameter_counts(model).items():
            wandb.run.summary[f"parameters/{name}"] = count
        if cfg.wandb_log_code:
            _base.log_wandb_code(wandb.run)

    if DEVICE == "cuda" and cfg.compile_trunk:
        trunks = (
            model.policy.trunk,
            model.q1.encoder.trunk,
            model.q2.encoder.trunk,
            model.value.encoder.trunk,
            model.target_q1.encoder.trunk,
            model.target_q2.encoder.trunk,
        )
        for trunk in trunks:
            trunk.compile(dynamic=False)
    if DEVICE == "cuda" and cfg.compile_temporal:
        model.policy.temporal.compile(dynamic=False)

    train_loader, validation = _make_loaders(cfg, stats)
    iterator = iter(train_loader)
    run_started = time.monotonic()
    policy = model.policy
    model.train()
    inference: BF16Inference | None = None
    try:
        for step in range(start_step, cfg.max_steps):
            loader_started = time.monotonic()
            try:
                cpu_batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                cpu_batch = next(iterator)
            validate_batch_geometry(cpu_batch, cfg, cfg.batch_size)
            loader_wait = time.monotonic() - loader_started
            batch = cpu_batch.to(DEVICE)
            with profile("step") as stopwatch:
                parts = iql_update(model, batch, optimizers, cfg)
                schedulers.actor.step()
                if DEVICE == "cuda":
                    torch.cuda.synchronize()
            metrics = training_metrics(parts, cfg)
            wandb.log(
                {
                    "global_step": step,
                    "samples": (step + 1) * cfg.batch_size,
                    **{f"train/{name}": value for name, value in metrics.items()},
                    "lr/actor": optimizers.actor.param_groups[0]["lr"],
                    "lr/q": optimizers.q.param_groups[0]["lr"],
                    "lr/value": optimizers.value.param_groups[0]["lr"],
                    "throughput/step_s": stopwatch.elapsed,
                    "throughput/loader_wait_s": loader_wait,
                    "throughput/samples_per_s": cfg.batch_size / stopwatch.elapsed,
                }
            )
            if step < 10 or step % 50 == 0:
                print(
                    f"[t+{time.monotonic() - run_started:.0f}s] step {step}: "
                    f"actor={parts.actor_loss:.3f} q={parts.q1_loss + parts.q2_loss:.3f} "
                    f"v={parts.value_loss:.3f}",
                    flush=True,
                )
            val_due = cfg.val_every > 0 and step > 0 and step % cfg.val_every == 0
            eval_due = cfg.eval_every > 0 and step > 0 and step % cfg.eval_every == 0
            ckpt_due = cfg.ckpt_every > 0 and step > 0 and step % cfg.ckpt_every == 0
            checkpoint = run_dir / "latest.pt"
            if val_due or eval_due or ckpt_due:
                save_checkpoint(
                    checkpoint,
                    step=step,
                    model=model,
                    opt=optimizers,
                    sched=schedulers,
                    cfg=asdict(cfg),
                    wandb_id=None if wandb.run is None else wandb.run.id,
                    uploader=uploader,
                )
            if val_due:
                values = val_metrics(model, validation, cfg)
                wandb.log({"global_step": step, **{f"val/{name}": value for name, value in values.items()}})
            if eval_due:
                inference = inference or BF16Inference(
                    policy, cfg, bucket=_eval_inference_bucket(cfg, cfg.eval_n_matchups)
                )
                values = eval_vs_cpu(
                    policy,
                    stats,
                    cfg,
                    n_matchups=cfg.eval_n_matchups,
                    replay_dir=replay_dir / f"step_{step:06d}",
                    checkpoint_sha256=_checkpoint_sha256(checkpoint),
                    inference=inference,
                )
                wandb.log({"global_step": step, **{f"eval/{name}": value for name, value in values.items()}})

        final_path = run_dir / "final.pt"
        save_checkpoint(
            final_path,
            step=cfg.max_steps,
            model=model,
            opt=optimizers,
            sched=schedulers,
            cfg=asdict(cfg),
            wandb_id=None if wandb.run is None else wandb.run.id,
            uploader=uploader,
        )
        final_val = val_metrics(model, validation, cfg)
        wandb.log({"global_step": cfg.max_steps, **{f"val/{name}": value for name, value in final_val.items()}})
        final_bucket = _eval_inference_bucket(cfg, cfg.final_eval_n_matchups)
        if inference is None or (inference.compiled and inference.bucket != final_bucket):
            inference = BF16Inference(policy, cfg, bucket=final_bucket)
        values = eval_vs_cpu(
            policy,
            stats,
            cfg,
            n_matchups=cfg.final_eval_n_matchups,
            replay_dir=replay_dir / "final",
            checkpoint_sha256=_checkpoint_sha256(final_path),
            inference=inference,
        )
        wandb.log({"global_step": cfg.max_steps, **{f"eval/{name}": value for name, value in values.items()}})
        model.q1.to("cpu")
        model.q2.to("cpu")
        model.value.to("cpu")
        model.target_q1.to("cpu")
        model.target_q2.to("cpu")
        del optimizers
        del schedulers
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        h2h = final_h2h(
            policy, stats, cfg, run_dir=run_dir, reference=reference, uploader=uploader, inference=inference
        )
        if wandb.run is not None:
            wandb.run.summary["h2h_final"] = h2h
    finally:
        if uploader is not None:
            uploader.upload_tree(replay_dir, base=run_dir)
            uploader.close()
        wandb.finish()


_CHECKPOINT_ARCH_FIELDS = {
    "decoder_arch_version",
    "head_offsets",
    "sample_chunk_length",
    "temporal_d_model",
    "temporal_layers",
    "temporal_heads",
    "critic_d_model",
    "critic_layers",
    "critic_heads",
    "critic_hidden_dim",
    "q_objective",
}


def config_from_state(values: dict[str, Any]) -> TrainConfig:
    missing = _CHECKPOINT_ARCH_FIELDS - values.keys()
    if missing:
        raise ValueError(f"checkpoint is not experiment 030; missing {sorted(missing)}")
    known = {item.name for item in fields(TrainConfig)}
    cfg = TrainConfig(**{name: value for name, value in values.items() if name in known})
    if cfg.eval_max_parallel is None or cfg.eval_max_parallel > EVAL_MAX_PARALLEL:
        cfg = replace(cfg, eval_max_parallel=EVAL_MAX_PARALLEL)
    return cfg


def policy_state_dict(checkpoint_model: dict[str, Tensor]) -> dict[str, Tensor]:
    prefix = "policy."
    extracted = {
        name.removeprefix(prefix): value for name, value in checkpoint_model.items() if name.startswith(prefix)
    }
    if extracted:
        return extracted
    training_prefixes = ("q1.", "q2.", "value.", "target_q1.", "target_q2.")
    if any(name.startswith(training_prefixes) for name in checkpoint_model):
        raise ValueError("training checkpoint contains critics but no policy.* parameters")
    return checkpoint_model


def load_checkpoint(path: str, *, device: str = DEVICE) -> tuple[GPT, TrainConfig, dict[str, FeatureStats], dict]:
    state = torch.load(path, map_location=device, weights_only=False)
    cfg = config_from_state(state["cfg"])
    validate_config(cfg)
    policy = GPT(cfg).to(device)
    policy_state = policy_state_dict(state["model"])
    if not policy_state:
        raise ValueError("experiment-030 checkpoint has no policy.* parameters")
    policy.load_state_dict(policy_state, strict=True)
    policy.eval()
    stats = load_consolidated_stats(Path(cfg.data_root) / "stats.json")
    return policy, cfg, stats, state


def eval_checkpoint(
    path: str,
    *,
    n_matchups: int | None = None,
    eager: bool = False,
    max_parallel: int | None = None,
    output_name: str | None = None,
) -> dict[str, float]:
    checkpoint = Path(path)
    if not checkpoint.is_file():
        downloaded = download_latest(path, Path("runs") / path)
        if downloaded is None:
            raise FileNotFoundError(f"no local checkpoint or remote latest.pt for {path!r}")
        checkpoint = downloaded
    policy, cfg, stats, state = load_checkpoint(str(checkpoint))
    cfg = replace(
        cfg,
        inference_mode="eager" if eager else cfg.inference_mode,
        eval_max_parallel=cfg.eval_max_parallel if max_parallel is None else max_parallel,
    )
    name = "eval_replays" if output_name is None else output_name
    if Path(name).name != name or name in ("", ".", ".."):
        raise ValueError("evaluation output name must be one directory name")
    values = eval_vs_cpu(
        policy,
        stats,
        cfg,
        n_matchups=cfg.final_eval_n_matchups if n_matchups is None else n_matchups,
        replay_dir=checkpoint.resolve().parent / name,
        checkpoint_sha256=_checkpoint_sha256(checkpoint),
    )
    print(f"[eval] step={state['step']}: {values}", flush=True)
    return values


@dataclass
class Args:
    cfg: TrainConfig = field(default_factory=TrainConfig)
    comment: str = ""
    resume: str | None = None
    eval: str | None = None
    eval_n_matchups: int | None = None
    eval_eager: bool = False
    eval_max_parallel: int | None = None
    eval_output_name: str | None = None


def main(args: Args) -> None:
    if args.eval is not None and args.resume is not None:
        raise SystemExit("pass only one of --eval or --resume")
    if args.eval is not None:
        eval_checkpoint(
            args.eval,
            n_matchups=args.eval_n_matchups,
            eager=args.eval_eager,
            max_parallel=args.eval_max_parallel,
            output_name=args.eval_output_name,
        )
        return
    resume_run = resume_state = None
    cfg = args.cfg
    if args.resume is not None:
        resume_state = load_for_resume(args.resume, Path("runs") / args.resume, device=DEVICE)
        if resume_state is None:
            raise SystemExit(f"no latest.pt for run {args.resume!r}")
        if resume_state["cfg"].get("attention_backend") != "varlen_flash":
            raise SystemExit("cannot resume a pre-varlen experiment 030 checkpoint; restart training from step zero")
        resume_run = args.resume
        cfg = config_from_state(resume_state["cfg"])
    stats = load_consolidated_stats(Path(cfg.data_root) / "stats.json")
    train(cfg, stats, comment=args.comment, resume_run=resume_run, resume_state=resume_state)


if __name__ == "__main__":
    main(tyro.cli(Args))
