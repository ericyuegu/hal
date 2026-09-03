"""Experiment 051: correct parameterization over nested replay data.

O51 keeps O50's task, objective, controller factorization, context, and closed-
loop evaluation protocol.  It changes model/optimizer parameterization, assigns
each duration endpoint a fixed nested replay pool, and makes the B200 throughput
preflight part of the launch contract.

Examples:
    uv run experiments/051_correct_parameterization.py describe
    uv run experiments/051_correct_parameterization.py coordinate-checks
    uv run experiments/051_correct_parameterization.py benchmark
    uv run experiments/051_correct_parameterization.py loader-benchmark
    uv run experiments/051_correct_parameterization.py preflight --report preflight.json
    uv run experiments/051_correct_parameterization.py train --preflight-report preflight.json
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import math
import sys
from collections import defaultdict
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from dataclasses import fields
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Annotated
from typing import Final
from typing import Literal
from typing import cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from torch import Tensor

from hal import streams
from hal.training.dataloader import StreamSamplePrefix
from hal.training.dataloader import make_loader
from hal.training.ego_stats import load_consolidated_mixture_stats
from hal.training.muon import SingleDeviceMuonWithAuxAdam
from hal.training.o51_data import D0
from hal.training.o51_data import DATA_PROTOCOL
from hal.training.o51_data import O51_RETURN_SUFFIX
from hal.training.o51_data import OFFICIAL_TIER_REPLAYS
from hal.training.o51_data import OFFICIAL_TIER_TARGETS
from hal.training.o51_data import TIER_SCALES
from hal.training.o51_data import CorpusSelection
from hal.training.o51_data import DirectO51ReplayLabels
from hal.training.o51_data import corpus_selection
from hal.training.player_identity import ReplayPlayerLookup
from hal.training.replay_reservoir import ReservoirLoader
from hal.training.replay_reservoir import make_reservoir_loader
from hal.training.trunk import Trunk
from hal.training.trunk import TrunkConfig


def _load_o50() -> ModuleType:
    path = Path(__file__).with_name("050_scaled_temporal_awr.py")
    owner = __name__.replace(".", "_")
    name = f"_hal_experiment_050_for_051_{owner}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_o50 = _load_o50()
_O50_DEFAULT_CONFIG = _o50.TrainConfig()
_O50_FINALIZE_TRAINING = _o50._finalize_training
_O50_LAYER_ACTIVATION_RMS_LOG = _o50.layer_activation_rms_log
_O50_TRAINING_DIAGNOSTICS = _o50._training_diagnostics

DEVICE = _o50.DEVICE
GROUP_NAMES = _o50.GROUP_NAMES
GROUP_ORDER = _o50.GROUP_ORDER
GROUP_INDEX = _o50.GROUP_INDEX
GROUP_VOCABS = _o50.GROUP_VOCABS
N_GROUPS = _o50.N_GROUPS
BUTTONS_G = _o50.BUTTONS_G
TRIG_G = _o50.TRIG_G
EVAL_HORIZONS = _o50.EVAL_HORIZONS
DIRECT_LOSS_START = _o50.DIRECT_LOSS_START
AWRCalibration = _o50.AWRCalibration
AWR_CALIBRATION = _o50.AWR_CALIBRATION
AWRBatch = _o50.AWRBatch

_EXPERIMENT_ID: Final[str] = "051_correct_parameterization_v2"
_TRUNK_BASE_LAYERS: Final[int] = 8
_TEMPORAL_BASE_LAYERS: Final[int] = 2
_TRUNK_BASE_ATTENTION_SCALE: Final[float] = 0.25
_TEMPORAL_BASE_ATTENTION_SCALE: Final[float] = 0.5
_BASE_MLP_SCALE: Final[float] = 1.0
_BASE_BATCH: Final[int] = 512
_BASE_ADAM_BETAS: Final[tuple[float, float]] = (0.9, 0.95)
_BASE_ADAM_EPS: Final[float] = 1e-12
_SUPERVISED_POSITIONS_PER_WINDOW: Final[int] = 128
_REPLAY_COOLDOWN_BATCHES: Final[int] = 16
_MIN_FREE_CACHE_GIB: Final[int] = 256
_LONG_RUN_POSITIONS: Final[int] = D0
# A 55M batch-1024 B200 benchmark reached 16.73% compiled MFU. Keep enough
# margin for run-to-run variance, then require the loader to retain 90% of it.
_MIN_SYNTHETIC_MFU: Final[float] = 0.15
_MIN_FULL_TIER_MFU: Final[float] = 0.135

MODEL_LEVELS: Final[tuple[str, ...]] = ("base", "proxy", "mid", "large")
EXPECTED_PARAMETER_COUNTS: Final[dict[str, int]] = {
    "base": 7_861_786,
    "proxy": 14_480_922,
    "mid": 55_015_322,
    "large": 216_496_794,
}


@dataclass(frozen=True)
class Architecture(_o50.Architecture):
    """One member of the O51 width/depth family; finite encoders stay fixed."""

    d_model: int = 512
    n_layers: int = 16
    n_heads: int = 8
    temporal_d_model: int = 256
    temporal_layers: int = 4
    temporal_heads: int = 4
    temporal_ff_dim: int = 768
    group_head_dim: int = 256
    value_hidden_dim: int = 256


ARCHITECTURE = Architecture()
MODEL_FAMILY: Final[dict[str, Architecture]] = {
    "base": replace(
        ARCHITECTURE,
        d_model=256,
        n_layers=8,
        n_heads=4,
        temporal_d_model=128,
        temporal_layers=2,
        temporal_heads=2,
        temporal_ff_dim=384,
        group_head_dim=128,
        value_hidden_dim=128,
    ),
    "proxy": replace(
        ARCHITECTURE,
        d_model=256,
        n_layers=16,
        n_heads=4,
        temporal_d_model=128,
        temporal_layers=4,
        temporal_heads=2,
        temporal_ff_dim=384,
        group_head_dim=128,
        value_hidden_dim=128,
    ),
    "mid": ARCHITECTURE,
    "large": replace(
        ARCHITECTURE,
        d_model=1024,
        n_layers=16,
        n_heads=16,
        temporal_d_model=512,
        temporal_layers=4,
        temporal_heads=8,
        temporal_ff_dim=1536,
        group_head_dim=512,
        value_hidden_dim=512,
    ),
}

_MODEL_GEOMETRY_FIELDS: Final[tuple[str, ...]] = (
    "d_model",
    "n_layers",
    "n_heads",
    "temporal_d_model",
    "temporal_layers",
    "temporal_heads",
    "temporal_ff_dim",
    "group_head_dim",
    "value_hidden_dim",
)
_FROZEN_ARCHITECTURE_FIELDS: Final[tuple[str, ...]] = tuple(
    item.name for item in fields(Architecture) if item.name not in _MODEL_GEOMETRY_FIELDS
)
_FROZEN_RUNTIME_FIELDS: Final[tuple[str, ...]] = (
    "prediction_frames",
    "delay_frames",
    "replan_interval_frames",
    "eval_seed",
    "lr_schedule_kind",
    "val_split",
    "val_n_samples",
    "val_batch_size",
    "eval_max_frames",
    "eval_n_matchups",
    "final_eval_n_matchups",
    "mds_schema_version",
    "policy_world_schema_version",
    "identity_dropout",
    "player_sidecar_sha256",
    "player_vocab_sha256",
    "player_vocab_size",
)


def model_level(architecture: Architecture) -> str:
    geometry = tuple(getattr(architecture, name) for name in _MODEL_GEOMETRY_FIELDS)
    matches = [
        name
        for name, candidate in MODEL_FAMILY.items()
        if geometry == tuple(getattr(candidate, field) for field in _MODEL_GEOMETRY_FIELDS)
    ]
    if len(matches) != 1:
        raise ValueError(f"architecture is not an O51 family member: {geometry}")
    return matches[0]


@dataclass(frozen=True)
class TrainConfig(_o50.TrainConfig):
    arch: Annotated[Architecture, tyro.conf.Suppress] = ARCHITECTURE
    awr: Annotated[AWRCalibration, tyro.conf.Suppress] = AWR_CALIBRATION

    target_positions: int = D0
    tier_scale: int = 1
    depth_alpha: Literal[0.5, 1.0] = 0.5
    hidden_std_multiplier: Literal[0.5, 1.0, 2.0] = 1.0
    readout_init: Literal["zero", "mup-normal"] = "zero"
    muon_duration_scaling: Literal["fixed", "inverse-sqrt"] = "fixed"
    muon_batch_scaling: Literal["fixed", "sqrt"] = "fixed"

    max_steps: int = 16_384
    warmup_steps: int = 512
    muon_lr: float = 0.028
    adam_lr: float = 4.25e-4
    adam_beta1: float = _BASE_ADAM_BETAS[0]
    adam_beta2: float = _BASE_ADAM_BETAS[1]
    adam_eps: float = _BASE_ADAM_EPS
    muon_weight_decay: float = 0.001
    adam_weight_decay: float = 0.001

    compile_mode: Literal["reduce-overhead", "max-autotune"] = "reduce-overhead"
    temporal_attention_chunk: int | None = 16_384
    windows_per_replay: int = 4
    reservoir_capacity: Annotated[int, tyro.conf.Suppress] = _BASE_BATCH * (_REPLAY_COOLDOWN_BATCHES + 1)
    replay_cooldown_batches: Annotated[int, tyro.conf.Suppress] = _REPLAY_COOLDOWN_BATCHES
    replay_pack_batch_size: Literal[16, 32, 64] = 64
    loader_prefetch_factor: Literal[1, 2, 4] = 2
    shuffle_algo: Literal["py1s", "py1e"] = "py1s"
    predownload: int = 1024
    num_workers: int = 16
    cache_limit_gb: int = 1700
    stability_every: int = 25

    def __post_init__(self) -> None:
        supervised = self.arch.L_ctx - self.arch.L_ctx // 2
        positions_per_update = self.batch_size * supervised
        updates, remainder = divmod(self.target_positions, positions_per_update)
        if remainder:
            raise ValueError(
                f"target_positions={self.target_positions} is not divisible by "
                f"batch_size={self.batch_size} x supervised_positions={supervised}"
            )
        warmup_positions, warmup_remainder = divmod(self.target_positions, 32)
        if warmup_remainder:
            raise ValueError("target_positions must be divisible by 32")
        warmup_updates, update_remainder = divmod(warmup_positions, positions_per_update)
        if update_remainder:
            raise ValueError("D/32 warmup does not land on an optimizer boundary")
        object.__setattr__(self, "max_steps", updates)
        object.__setattr__(self, "warmup_steps", warmup_updates)
        object.__setattr__(
            self,
            "reservoir_capacity",
            self.batch_size * (self.replay_cooldown_batches + 1),
        )


def config_for(
    level: str = "mid",
    *,
    target_positions: int = D0,
    tier_scale: int = 1,
    **changes: object,
) -> TrainConfig:
    """Construct one family member while deriving its position schedule."""
    if level not in MODEL_FAMILY:
        raise ValueError(f"level must be one of {MODEL_LEVELS}, got {level!r}")
    return TrainConfig(
        arch=MODEL_FAMILY[level],
        target_positions=target_positions,
        tier_scale=tier_scale,
        **changes,
    )


@dataclass(frozen=True, slots=True)
class DepthRule:
    attention: float
    mlp: float
    learning_rate: float


def depth_rule(stack: Literal["trunk", "temporal"], layers: int, alpha: float) -> DepthRule:
    """Return O51's branch and optimizer multipliers for one stack."""
    if alpha not in (0.5, 1.0):
        raise ValueError(f"depth alpha must be 0.5 or 1.0, got {alpha}")
    if stack == "trunk":
        base_layers = _TRUNK_BASE_LAYERS
        base_attention = _TRUNK_BASE_ATTENTION_SCALE
    elif stack == "temporal":
        base_layers = _TEMPORAL_BASE_LAYERS
        base_attention = _TEMPORAL_BASE_ATTENTION_SCALE
    else:
        raise ValueError(f"unknown stack {stack!r}")
    if layers < 1:
        raise ValueError("stack depth must be positive")
    multiplier = layers / base_layers
    branch = multiplier ** (-alpha)
    return DepthRule(
        attention=base_attention * branch,
        mlp=_BASE_MLP_SCALE * branch,
        learning_rate=multiplier ** (alpha - 1),
    )


def scaling_multipliers(cfg: TrainConfig) -> tuple[float, float]:
    return cfg.batch_size / _BASE_BATCH, cfg.target_positions / D0


def scaled_adam_betas(
    betas: tuple[float, float],
    *,
    batch_multiplier: float,
    duration_multiplier: float,
) -> tuple[float, float]:
    ratio = batch_multiplier / duration_multiplier
    scaled = tuple(1 - (1 - beta) * ratio for beta in betas)
    if any(not 0 <= beta < 1 for beta in scaled):
        raise ValueError(f"scaled Adam betas are invalid: {scaled}")
    return cast(tuple[float, float], scaled)


def scaled_adam_epsilon(
    epsilon: float,
    *,
    batch_multiplier: float,
    duration_multiplier: float,
) -> float:
    return epsilon * math.sqrt(duration_multiplier / batch_multiplier)


def scaled_adam_lr(
    master_lr: float,
    *,
    batch_multiplier: float,
    duration_multiplier: float,
    fan_in_multiplier: float = 1.0,
    output: bool = False,
    depth_multiplier: float = 1.0,
) -> float:
    if fan_in_multiplier <= 0 or depth_multiplier <= 0:
        raise ValueError("fan-in and depth multipliers must be positive")
    value = master_lr * math.sqrt(batch_multiplier / duration_multiplier) * depth_multiplier
    return value / fan_in_multiplier if output else value


def muon_lr_multiplier(cfg: TrainConfig) -> float:
    batch_multiplier, duration_multiplier = scaling_multipliers(cfg)
    duration = duration_multiplier**-0.5 if cfg.muon_duration_scaling == "inverse-sqrt" else 1.0
    batch = math.sqrt(batch_multiplier) if cfg.muon_batch_scaling == "sqrt" else 1.0
    return duration * batch


def scaled_weight_decays(cfg: TrainConfig) -> tuple[float, float]:
    """Return AdamW and Muon decay under O51's integrated-shrink invariant."""
    batch_multiplier, duration_multiplier = scaling_multipliers(cfg)
    muon_scale = muon_lr_multiplier(cfg)
    adam = cfg.adam_weight_decay * math.sqrt(batch_multiplier / duration_multiplier)
    muon = cfg.muon_weight_decay * batch_multiplier / (duration_multiplier * muon_scale)
    return adam, muon


def validate_config(cfg: TrainConfig) -> None:
    """Reject geometry or protocol changes that would no longer be O51."""
    level = model_level(cfg.arch)
    del level
    changed_architecture = {
        name: (getattr(cfg.arch, name), getattr(ARCHITECTURE, name))
        for name in _FROZEN_ARCHITECTURE_FIELDS
        if getattr(cfg.arch, name) != getattr(ARCHITECTURE, name)
    }
    if changed_architecture:
        raise ValueError(f"O51 changed frozen O50 architecture fields: {changed_architecture}")
    if cfg.awr != AWR_CALIBRATION:
        raise ValueError("O51 keeps O50's AWR objective and calibration fixed")
    changed_runtime = {
        name: (getattr(cfg, name), getattr(_O50_DEFAULT_CONFIG, name))
        for name in _FROZEN_RUNTIME_FIELDS
        if getattr(cfg, name) != getattr(_O50_DEFAULT_CONFIG, name)
    }
    if changed_runtime:
        raise ValueError(f"O51 changed frozen O50 task or evaluation fields: {changed_runtime}")
    if cfg.arch.d_model // cfg.arch.n_heads != 64:
        raise ValueError("O51 fixes trunk attention head width at 64")
    if cfg.arch.temporal_d_model // cfg.arch.temporal_heads != 64:
        raise ValueError("O51 fixes temporal attention head width at 64")
    if cfg.depth_alpha not in (0.5, 1.0):
        raise ValueError("depth_alpha must be 0.5 or 1.0")
    if cfg.hidden_std_multiplier not in (0.5, 1.0, 2.0):
        raise ValueError("hidden_std_multiplier is outside the six-arm sweep")
    if cfg.readout_init not in ("zero", "mup-normal"):
        raise ValueError("readout_init must be zero or mup-normal")
    if cfg.tier_scale not in TIER_SCALES:
        raise ValueError(f"tier_scale must be one of {TIER_SCALES}")
    if not 0 < cfg.target_positions <= 8 * D0:
        raise ValueError("target_positions must be in (0, 8D0]")
    official_scale = cfg.target_positions // D0 if cfg.target_positions % D0 == 0 else None
    if official_scale in TIER_SCALES and cfg.tier_scale != official_scale:
        raise ValueError("an official D endpoint must use its matched nested U tier")
    if cfg.max_steps * cfg.batch_size * _SUPERVISED_POSITIONS_PER_WINDOW != cfg.target_positions:
        raise ValueError("max_steps does not stop exactly at D valid supervised positions")
    if cfg.warmup_steps * cfg.batch_size * _SUPERVISED_POSITIONS_PER_WINDOW != cfg.target_positions // 32:
        raise ValueError("warmup must consume exactly D/32 valid positions")
    if cfg.lr_floor_ratio != 1 / 170 or cfg.grad_clip != 1.0:
        raise ValueError("O51 fixes cosine floor=1/170 and global clipping=1.0")
    if cfg.batch_size not in (128, 256, 512, 1024):
        raise ValueError("batch_size is outside the O51 sweep")
    if cfg.replay_cooldown_batches != _REPLAY_COOLDOWN_BATCHES:
        raise ValueError(f"O51 fixes replay cooldown to {_REPLAY_COOLDOWN_BATCHES} batches")
    expected_capacity = cfg.batch_size * (cfg.replay_cooldown_batches + 1)
    if cfg.reservoir_capacity != expected_capacity:
        raise ValueError(f"O51 replay reservoir must contain exactly {expected_capacity} replay IDs")
    if cfg.windows_per_replay != 4:
        raise ValueError("O51 emits four deterministic windows per physical replay read")
    if cfg.replay_pack_batch_size not in (16, 32, 64):
        raise ValueError("replay-pack batch is outside the O51 preflight grid")
    if cfg.loader_prefetch_factor not in (1, 2, 4):
        raise ValueError("loader prefetch factor is outside the O51 preflight grid")
    if cfg.predownload not in (8 * cfg.replay_pack_batch_size, 16 * cfg.replay_pack_batch_size):
        raise ValueError("predownload must be 8x or 16x the replay-pack batch")
    if cfg.temporal_attention_chunk not in (8192, 16_384, 32_768, None):
        raise ValueError("temporal attention chunk is outside the O51 preflight grid")
    if cfg.compile_mode not in ("reduce-overhead", "max-autotune"):
        raise ValueError("unsupported compile mode")
    if cfg.stability_every != 25:
        raise ValueError("O51 fixes stability diagnostics to the 25-update logging cadence")
    if tuple(cfg.source_names) != tuple(source.name for source in streams.POLICY_WORLD_V7_SOURCES):
        raise ValueError("O51 normalization and validation require all 44 policy-world-v7 sources")
    for name, value in (
        ("muon_lr", cfg.muon_lr),
        ("adam_lr", cfg.adam_lr),
        ("adam_eps", cfg.adam_eps),
        ("loader_timeout_s", cfg.loader_timeout_s),
        ("system_metrics_interval_s", cfg.system_metrics_interval_s),
        ("process_metrics_interval_s", cfg.process_metrics_interval_s),
        ("cache_metrics_interval_s", cfg.cache_metrics_interval_s),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name, value in (
        ("adam_weight_decay", cfg.adam_weight_decay),
        ("muon_weight_decay", cfg.muon_weight_decay),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if cfg.muon_duration_scaling not in ("fixed", "inverse-sqrt"):
        raise ValueError("unsupported Muon duration scaling")
    if cfg.muon_batch_scaling not in ("fixed", "sqrt"):
        raise ValueError("unsupported Muon batch scaling")
    if (cfg.adam_beta1, cfg.adam_beta2, cfg.adam_eps) != (*_BASE_ADAM_BETAS, _BASE_ADAM_EPS):
        raise ValueError("O51 scales the fixed base Adam betas and epsilon")
    if not isinstance(cfg.num_workers, int) or isinstance(cfg.num_workers, bool) or not 0 <= cfg.num_workers <= 48:
        raise ValueError("num_workers must be an integer in [0, 48]")
    if not isinstance(cfg.cache_limit_gb, int) or isinstance(cfg.cache_limit_gb, bool) or cfg.cache_limit_gb <= 0:
        raise ValueError("cache_limit_gb must be a positive integer")
    scaled_adam_betas(
        (cfg.adam_beta1, cfg.adam_beta2),
        batch_multiplier=scaling_multipliers(cfg)[0],
        duration_multiplier=scaling_multipliers(cfg)[1],
    )


def center_class_logits(logits: Tensor) -> Tensor:
    """Remove one softmax-invariant common mode from each class group."""
    return logits - logits.mean(dim=-1, keepdim=True)


class TemporalBlock(nn.Module):
    """O50's temporal block with O51's explicit depth rule on both branches."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.n_heads = cfg.arch.temporal_heads
        self.d_model = cfg.arch.temporal_d_model
        self.head_dim = self.d_model // self.n_heads
        rule = depth_rule("temporal", cfg.arch.temporal_layers, cfg.depth_alpha)
        self.scale = rule.attention
        self.mlp_scale = rule.mlp
        self.attention_chunk = cfg.temporal_attention_chunk
        self.qkv = nn.Linear(self.d_model, 3 * self.d_model, bias=False)
        self.proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.rotary = _o50.Rotary(self.head_dim)
        self.up = nn.Linear(self.d_model, cfg.arch.temporal_ff_dim, bias=False)
        self.down = nn.Linear(cfg.arch.temporal_ff_dim, self.d_model, bias=False)

    def _qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, length, _ = x.shape
        q, k, v = self.qkv(_o50.decoder_rmsnorm(x)).split(self.d_model, dim=-1)
        shape = (batch, length, self.n_heads, self.head_dim)
        return q.view(shape), k.view(shape), v.view(shape)

    def forward(self, x: Tensor) -> Tensor:
        q, k, v = self._qkv(x)
        cos, sin = self.rotary(q)
        q = _o50.apply_rotary_emb(q, cos, sin).transpose(1, 2)
        k = _o50.apply_rotary_emb(k, cos, sin).transpose(1, 2)
        v = v.transpose(1, 2)
        if self.attention_chunk is None:
            attended = _o50.short_causal_attention(q, k, v)
        else:
            attended = torch.cat(
                [
                    _o50.short_causal_attention(query, key, values)
                    for query, key, values in zip(
                        q.split(self.attention_chunk),
                        k.split(self.attention_chunk),
                        v.split(self.attention_chunk),
                        strict=True,
                    )
                ],
                dim=0,
            )
        attended = attended.transpose(1, 2).contiguous().view_as(x)
        x = x + self.scale * self.proj(attended)
        return x + self.mlp_scale * self.down(F.silu(self.up(_o50.decoder_rmsnorm(x))))

    def forward_step(self, x: Tensor, past: tuple[Tensor, Tensor] | None) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        q, k, v = self._qkv(x[:, None])
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if past is not None:
            k = torch.cat((past[0], k), dim=2)
            v = torch.cat((past[1], v), dim=2)
        cos, sin = self.rotary.at(k.shape[2], x.device)
        q = _o50.apply_rotary_emb(q, cos[:, -1:], sin[:, -1:]).transpose(1, 2)
        rotated_k = _o50.apply_rotary_emb(k.transpose(1, 2), cos, sin).transpose(1, 2)
        attended = F.scaled_dot_product_attention(q, rotated_k, v)
        attended = attended.transpose(1, 2).contiguous().view_as(x)
        x = x + self.scale * self.proj(attended)
        x = x + self.mlp_scale * self.down(F.silu(self.up(_o50.decoder_rmsnorm(x))))
        return x, (k, v)


class CausalTemporalDecoder(_o50.CausalTemporalDecoder):
    """O50's factorized controller with centered combined group logits."""

    @staticmethod
    def _center(logits: Tensor) -> Tensor:
        return center_class_logits(logits)

    def _raw_teacher_forced_outputs(
        self,
        hidden: Tensor,
        observed: Tensor,
        targets: Tensor,
    ) -> tuple[dict[str, Tensor], tuple[Tensor, Tensor, Tensor, Tensor]]:
        states = self.teacher_forced_states(hidden, observed, targets)
        embedded = self.codec.embed_groups(targets)
        logits: dict[str, Tensor] = {}
        button_values: tuple[Tensor, Tensor, Tensor] | None = None
        for name in GROUP_NAMES:
            features = self.group_features(states, name, embedded)
            if name == "buttons":
                head = cast(_o50.NonlinearActionHead, self.outputs[name])
                combined, head_input = head.forward_with_input(features)
                combined = combined + self.trunk_outputs[name](hidden)[..., None, :]
                button_values = (features, head_input, combined)
            else:
                combined = self.outputs[name](features) + self.trunk_outputs[name](hidden)[..., None, :]
            logits[name] = combined
        if button_values is None:
            raise RuntimeError("button head was not evaluated")
        button_mask = self.codec.button_mask(targets[..., TRIG_G])
        return logits, (*button_values, button_mask)

    @staticmethod
    def _mask_buttons(logits: dict[str, Tensor], button_mask: Tensor) -> dict[str, Tensor]:
        masked = dict(logits)
        masked["buttons"] = masked["buttons"].masked_fill(button_mask, float("-inf"))
        return masked

    def _teacher_forced_outputs(
        self,
        hidden: Tensor,
        observed: Tensor,
        targets: Tensor,
    ) -> tuple[dict[str, Tensor], tuple[Tensor, Tensor, Tensor, Tensor]]:
        raw_logits, button_values = self._raw_teacher_forced_outputs(hidden, observed, targets)
        centered = {name: self._center(logits) for name, logits in raw_logits.items()}
        return self._mask_buttons(centered, button_values[-1]), button_values

    def teacher_forced_nll(self, hidden: Tensor, observed: Tensor, targets: Tensor) -> Tensor:
        """Compute the policy loss without a softmax-invariant common mode."""
        logits, button_values = self._raw_teacher_forced_outputs(hidden, observed, targets)
        return self.nll_from_logits(self._mask_buttons(logits, button_values[-1]), targets)

    def teacher_forced_nll_with_diagnostics(
        self,
        hidden: Tensor,
        observed: Tensor,
        targets: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        raw_logits, button_values = self._raw_teacher_forced_outputs(hidden, observed, targets)
        features, head_input, raw_button_logits, button_mask = button_values
        feature_rms = features.detach().float().square().mean(dim=-1).sqrt()
        button_targets = targets[..., BUTTONS_G, None]
        target_logits = raw_button_logits.detach().gather(-1, button_targets).squeeze(-1).float()
        legal_logits = raw_button_logits.detach().masked_fill(button_mask, float("-inf"))
        competing = legal_logits.scatter(-1, button_targets, float("-inf")).amax(dim=-1).float()
        centered_values = {name: self._center(logits) for name, logits in raw_logits.items()}
        centered_p999 = torch.stack(
            [_o50._sampled_quantile(value.detach(), 99.9, absolute=True) for value in centered_values.values()]
        ).amax()
        metrics = {
            "stability/action_pre_norm_rms": features.detach().float().square().mean().sqrt(),
            "stability/button_pre_norm_rms_min": feature_rms.amin(),
            "stability/button_input_abs_p999": _o50._sampled_quantile(head_input.detach(), 99.9, absolute=True),
            # Keep O50's name as the required uncentered diagnostic.
            "stability/button_logit_abs_p999": _o50._sampled_quantile(raw_button_logits.detach(), 99.9, absolute=True),
            "stability/uncentered_button_logit_abs_p999": _o50._sampled_quantile(
                raw_button_logits.detach(), 99.9, absolute=True
            ),
            "stability/centered_logit_abs_p999": centered_p999,
            "stability/button_margin_mean": (target_logits - competing).mean(),
        }
        logits = self._mask_buttons(raw_logits, button_mask)
        return self.nll_from_logits(logits, targets), metrics

    def forced_stepwise_logits(self, hidden: Tensor, observed: Tensor, targets: Tensor) -> list[dict[str, Tensor]]:
        if targets.shape != (hidden.shape[0], len(self.head_offsets), N_GROUPS):
            raise ValueError("stepwise targets have the wrong shape")
        raw_trunk = hidden[:, -1]
        state_bias = self._state_bias(_o50.decoder_rmsnorm(raw_trunk))
        previous = observed
        caches: list[tuple[Tensor, Tensor] | None] = [None] * len(self.blocks)
        out: list[dict[str, Tensor]] = []
        for depth, offset in enumerate(self.head_offsets):
            state, caches = self._decode_step(previous, offset, state_bias, caches)
            target = targets[:, depth]
            embedded = self.codec.embed_groups(target)
            group_logits = {
                name: self._center(
                    self.outputs[name](self.group_features(state, name, embedded))
                    + self.trunk_outputs[name](raw_trunk)
                )
                for name in GROUP_NAMES
            }
            group_logits["buttons"] = group_logits["buttons"].masked_fill(
                self.codec.button_mask(target[:, TRIG_G]), float("-inf")
            )
            out.append(group_logits)
            previous = target
        return out

    def sample_indices(
        self,
        hidden: Tensor,
        observed: Tensor,
        offsets: tuple[int, ...],
        *,
        argmax: bool,
        uniforms: Tensor | None = None,
        gen: torch.Generator | None = None,
    ) -> Tensor:
        allowed = tuple(self.head_offsets[:horizon] for horizon in EVAL_HORIZONS)
        if offsets not in allowed:
            raise ValueError(f"live decode offsets must select one of the dense prefixes {allowed}")
        if uniforms is not None and uniforms.shape != (len(offsets), N_GROUPS, hidden.shape[0]):
            raise ValueError("uniform table must be [frames, groups, batch]")
        raw_trunk = hidden[:, -1]
        state_bias = self._state_bias(_o50.decoder_rmsnorm(raw_trunk))
        previous = observed
        caches: list[tuple[Tensor, Tensor] | None] = [None] * len(self.blocks)
        frames: list[Tensor] = []
        for depth, offset in enumerate(offsets):
            state, caches = self._decode_step(previous, offset, state_bias, caches)
            embedded: dict[str, Tensor] = {}
            picks: dict[str, Tensor] = {}
            for name in GROUP_ORDER:
                logits = self._center(
                    self.outputs[name](self.group_features(state, name, embedded))
                    + self.trunk_outputs[name](raw_trunk)
                )
                if name == "buttons":
                    logits = logits.masked_fill(self.codec.button_mask(picks["triggers"]), float("-inf"))
                group = GROUP_INDEX[name]
                uniform = None if uniforms is None else uniforms[depth, group]
                pick = _o50.sample_categorical(logits, argmax=argmax, uniform=uniform, gen=gen)
                picks[name] = pick
                embedded[name] = self.codec.group_embedding(name, pick)
            previous = torch.stack([picks[name] for name in GROUP_NAMES], dim=-1)
            frames.append(previous)
        return torch.stack(frames, dim=1)

    def rollout_conditioned_logits(self, hidden: Tensor, observed: Tensor) -> tuple[list[dict[str, Tensor]], Tensor]:
        raw_trunk = hidden[:, -1]
        state_bias = self._state_bias(_o50.decoder_rmsnorm(raw_trunk))
        previous = observed
        caches: list[tuple[Tensor, Tensor] | None] = [None] * len(self.blocks)
        frames: list[Tensor] = []
        all_logits: list[dict[str, Tensor]] = []
        for offset in self.head_offsets:
            state, caches = self._decode_step(previous, offset, state_bias, caches)
            embedded: dict[str, Tensor] = {}
            picks: dict[str, Tensor] = {}
            frame_logits: dict[str, Tensor] = {}
            for name in GROUP_ORDER:
                logits = self._center(
                    self.outputs[name](self.group_features(state, name, embedded))
                    + self.trunk_outputs[name](raw_trunk)
                )
                if name == "buttons":
                    logits = logits.masked_fill(self.codec.button_mask(picks["triggers"]), float("-inf"))
                pick = logits.argmax(dim=-1)
                frame_logits[name] = logits
                picks[name] = pick
                embedded[name] = self.codec.group_embedding(name, pick)
            previous = torch.stack([picks[name] for name in GROUP_NAMES], dim=-1)
            frames.append(previous)
            all_logits.append(frame_logits)
        return all_logits, torch.stack(frames, dim=1)


def mup_readout_std(fan_in: int, base_fan_in: int) -> float:
    """Use sigma(n0)=1/sqrt(n0), then sigma(n)=sigma(n0)*n0/n."""
    if fan_in < 1 or base_fan_in < 1:
        raise ValueError("readout fan-ins must be positive")
    return math.sqrt(base_fan_in) / fan_in


def _final_readouts(model: GPT) -> tuple[tuple[nn.Linear, int], ...]:
    action = tuple((cast(nn.Linear, model.temporal.outputs[name].down), 128) for name in GROUP_NAMES)
    trunk_skip = tuple((cast(nn.Linear, model.temporal.trunk_outputs[name]), 256) for name in GROUP_NAMES)
    return (*action, *trunk_skip, (cast(nn.Linear, model.value_head.down), 128))


def initialize_o51_parameters(model: GPT, cfg: TrainConfig) -> None:
    """Initialize without any depth multiplier, then apply the readout arm."""
    final_modules = {id(module) for module, _ in _final_readouts(model)}
    for module in model.modules():
        if isinstance(module, nn.Linear):
            if id(module) in final_modules:
                continue
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=cfg.hidden_std_multiplier / math.sqrt(module.weight.shape[1]),
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=cfg.hidden_std_multiplier / math.sqrt(module.embedding_dim),
            )
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    for module, base_fan_in in _final_readouts(model):
        if cfg.readout_init == "zero":
            nn.init.zeros_(module.weight)
        else:
            nn.init.normal_(module.weight, mean=0.0, std=mup_readout_std(module.in_features, base_fan_in))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    # Identity absence must remain an exact no-op in every initialization arm.
    nn.init.zeros_(model.player_projection.weight)


class GPT(_o50.GPT):
    """The frozen O50 policy with O51 depth, initialization, and logits."""

    def __init__(self, cfg: TrainConfig, vocabulary: _o50.PlayerVocabulary | None = None) -> None:
        nn.Module.__init__(self)
        self.cfg = cfg
        self.L_chunk = cfg.arch.sample_chunk_length
        self.head_offsets = tuple(cfg.arch.head_offsets)
        self.codec = _o50.StructuredControllerCodec(cfg.arch.action_embed_dim)
        self.cat_specs = {
            **_o50.CAT_FEATURES,
            "action": (cfg.arch.action_vocab, cfg.arch.action_state_embed_dim),
        }
        self.cat_embeds = nn.ModuleDict(
            {name: nn.Embedding(vocab, dim) for name, (vocab, dim) in self.cat_specs.items()}
        )
        self.char_emb = nn.Embedding(cfg.arch.char_vocab, cfg.arch.char_dim)
        self.stage_emb = nn.Embedding(cfg.arch.stage_vocab, cfg.arch.stage_dim)
        per_player = len(_o50.FLOAT_FEATURES) * 2 + sum(dim for _, dim in self.cat_specs.values())
        d_in = (
            len(_o50._PLAYER_PREFIXES) * per_player
            + N_GROUPS * cfg.arch.action_embed_dim
            + 2 * cfg.arch.char_dim
            + cfg.arch.stage_dim
        )
        self.item_type_emb = nn.Embedding(_o50._ITEM_CAT_VOCABS["type"], cfg.arch.item_type_dim)
        self.item_state_emb = nn.Embedding(_o50._ITEM_CAT_VOCABS["state"], cfg.arch.item_state_dim)
        slot_width = cfg.arch.item_type_dim + cfg.arch.item_state_dim + 2 * len(_o50._ITEM_FLOATS) + 1
        self.item_encoder = _o50.SwiGLU(slot_width, cfg.arch.item_hidden_dim, cfg.arch.item_dim)
        d_in += cfg.arch.item_dim
        self.observation_encoder = nn.Linear(d_in, cfg.arch.d_model)
        self.player_embedding = nn.Embedding(
            cfg.player_vocab_size,
            _o50.PLAYER_EMBED_DIM,
            padding_idx=_o50.MASKED_PLAYER_ID,
        )
        self.player_projection = nn.Linear(_o50.PLAYER_EMBED_DIM, cfg.arch.d_model, bias=False)
        code_payload = b"" if vocabulary is None else _o50.vocabulary_buffer(vocabulary)
        if vocabulary is not None and (
            vocabulary.size != cfg.player_vocab_size or vocabulary.sha256 != cfg.player_vocab_sha256
        ):
            raise ValueError("identity vocabulary does not match the frozen O51 contract")
        self.register_buffer(
            "player_code_bytes",
            torch.from_numpy(np.frombuffer(code_payload, dtype=np.uint8).copy()),
        )
        trunk_rule = depth_rule("trunk", cfg.arch.n_layers, cfg.depth_alpha)
        self.trunk = Trunk(
            TrunkConfig(
                d_model=cfg.arch.d_model,
                n_layers=cfg.arch.n_layers,
                n_heads=cfg.arch.n_heads,
                L_ctx=cfg.arch.L_ctx,
                attn_window=cfg.arch.attn_window,
                attention_backend=_o50._TRUNK_ATTENTION_BACKEND,
                attention_scale=trunk_rule.attention,
                mlp_scale=trunk_rule.mlp,
            )
        )
        self.temporal = CausalTemporalDecoder(cfg, self.codec)
        self.value_head = _o50.SwiGLU(cfg.arch.d_model, cfg.arch.value_hidden_dim, 1, output_bias=True)
        initialize_o51_parameters(self, cfg)

    def forward_unpadded(
        self,
        features: dict[str, Tensor],
        _ctx_pad: Tensor,
        action_indices: Tensor | None = None,
    ) -> Tensor:
        """Training forward for the loader's guaranteed full contexts."""
        return self.trunk.forward_unpadded(self.context_tokens(features, action_indices))


def subsystem_parameter_counts(model: GPT) -> dict[str, int]:
    """Count the O50 subsystems and pin each O51 family total."""
    all_parameters = tuple(model.parameters())
    trunk_ids = {id(parameter) for parameter in model.trunk.parameters()}
    head_modules = nn.ModuleList([model.temporal.outputs, model.temporal.trunk_outputs])
    head_ids = {id(parameter) for parameter in head_modules.parameters()}
    temporal_ids = {id(parameter) for parameter in model.temporal.parameters() if id(parameter) not in head_ids}
    value_ids = {id(parameter) for parameter in model.value_head.parameters()}
    other_ids = {id(parameter) for parameter in all_parameters} - trunk_ids - temporal_ids - head_ids - value_ids
    partitions = {
        "trunk": trunk_ids,
        "temporal_decoder": temporal_ids,
        "group_heads": head_ids,
        "value_head": value_ids,
        "other": other_ids,
    }
    counts = {
        name: sum(parameter.numel() for parameter in all_parameters if id(parameter) in parameter_ids)
        for name, parameter_ids in partitions.items()
    }
    counts["total"] = sum(parameter.numel() for parameter in all_parameters)
    if sum(value for name, value in counts.items() if name != "total") != counts["total"]:
        raise RuntimeError("parameter subsystem partition is incomplete")
    expected = EXPECTED_PARAMETER_COUNTS[model_level(model.cfg.arch)]
    if counts["total"] != expected:
        raise RuntimeError(f"O51 parameter contract changed: {counts['total']} != {expected}")
    return counts


@dataclass(frozen=True, slots=True)
class OptimizerRole:
    optimizer: Literal["muon", "adamw"]
    lr_kind: Literal["hidden", "input", "output", "vector"]
    decay: bool
    stack: Literal["trunk", "temporal"] | None = None
    logical_splits: int = 1
    fan_in_multiplier: float = 1.0


def _is_final_readout(name: str) -> bool:
    return (
        (name.startswith("temporal.outputs.") and ".down." in name)
        or name.startswith("temporal.trunk_outputs.")
        or name.startswith("value_head.down.")
    )


def _output_fan_in_multiplier(name: str, cfg: TrainConfig) -> float:
    if name.startswith("temporal.outputs."):
        return cfg.arch.group_head_dim / 128
    if name.startswith("temporal.trunk_outputs."):
        return cfg.arch.d_model / 256
    if name.startswith("value_head.down."):
        return cfg.arch.value_hidden_dim / 128
    raise ValueError(f"{name!r} is not a final readout")


def optimizer_roles(model: GPT, cfg: TrainConfig) -> dict[str, OptimizerRole]:
    """Assign every tensor by semantic role instead of by dimensionality."""
    embedding_prefixes = (
        "codec.class_embeddings.",
        "cat_embeds.",
        "char_emb.",
        "stage_emb.",
        "item_type_emb.",
        "item_state_emb.",
        "player_embedding.",
        "temporal.offset_embedding.",
    )
    finite_prefixes = (
        "codec.semantic_projections.",
        "item_encoder.",
        "observation_encoder.",
        "player_projection.",
        "temporal.group_condition.",
    )
    roles: dict[str, OptimizerRole] = {}
    for name, parameter in model.named_parameters():
        if _is_final_readout(name):
            roles[name] = OptimizerRole(
                "adamw",
                "output",
                parameter.ndim >= 2,
                fan_in_multiplier=_output_fan_in_multiplier(name, cfg),
            )
        elif name.startswith("trunk.blocks."):
            splits = 3 if name.endswith("attn.c_attn.weight") else 1
            roles[name] = OptimizerRole("muon", "hidden", True, stack="trunk", logical_splits=splits)
        elif name.startswith("temporal.blocks."):
            splits = 3 if name.endswith("qkv.weight") else 1
            roles[name] = OptimizerRole("muon", "hidden", True, stack="temporal", logical_splits=splits)
        elif name == "temporal.token_projection.weight" or (
            name.startswith("temporal.outputs.") and name.endswith("up.weight")
        ):
            roles[name] = OptimizerRole("muon", "hidden", True)
        elif name == "value_head.up.weight":
            roles[name] = OptimizerRole("muon", "hidden", True, logical_splits=2)
        elif name.startswith(embedding_prefixes):
            roles[name] = OptimizerRole("adamw", "input", False)
        elif name.startswith(finite_prefixes):
            roles[name] = OptimizerRole("adamw", "input" if parameter.ndim >= 2 else "vector", parameter.ndim >= 2)
        elif name == "temporal.token_projection.bias":
            roles[name] = OptimizerRole("adamw", "vector", False)
        else:
            raise RuntimeError(f"O51 has no optimizer role for {name} {tuple(parameter.shape)}")
    if set(roles) != {name for name, _ in model.named_parameters()}:
        raise RuntimeError("O51 optimizer-role coverage is incomplete")
    return roles


def _role_lr(role: OptimizerRole, cfg: TrainConfig) -> float:
    batch_multiplier, duration_multiplier = scaling_multipliers(cfg)
    if role.optimizer == "muon":
        value = cfg.muon_lr * muon_lr_multiplier(cfg)
        if role.stack is not None:
            layers = cfg.arch.n_layers if role.stack == "trunk" else cfg.arch.temporal_layers
            value *= depth_rule(role.stack, layers, cfg.depth_alpha).learning_rate
        return value
    depth = 1.0
    if role.stack is not None:
        layers = cfg.arch.n_layers if role.stack == "trunk" else cfg.arch.temporal_layers
        depth = depth_rule(role.stack, layers, cfg.depth_alpha).learning_rate
    return scaled_adam_lr(
        cfg.adam_lr,
        batch_multiplier=batch_multiplier,
        duration_multiplier=duration_multiplier,
        fan_in_multiplier=role.fan_in_multiplier,
        output=role.lr_kind == "output",
        depth_multiplier=depth,
    )


def make_optimizer(model: GPT, cfg: TrainConfig) -> SingleDeviceMuonWithAuxAdam:
    """Build exact O51 Muon/AdamW groups, including logical QKV matrices."""
    validate_config(cfg)
    roles = optimizer_roles(model, cfg)
    named = dict(model.named_parameters())
    adam_decay, muon_decay = scaled_weight_decays(cfg)
    batch_multiplier, duration_multiplier = scaling_multipliers(cfg)
    betas = scaled_adam_betas(
        (cfg.adam_beta1, cfg.adam_beta2),
        batch_multiplier=batch_multiplier,
        duration_multiplier=duration_multiplier,
    )
    epsilon = scaled_adam_epsilon(
        cfg.adam_eps,
        batch_multiplier=batch_multiplier,
        duration_multiplier=duration_multiplier,
    )

    buckets: dict[tuple[object, ...], list[nn.Parameter]] = defaultdict(list)
    for name, role in roles.items():
        lr = _role_lr(role, cfg)
        if role.optimizer == "muon":
            key = ("muon", lr, muon_decay if role.decay else 0.0, role.logical_splits)
        else:
            key = ("adamw", lr, adam_decay if role.decay else 0.0)
        buckets[key].append(named[name])

    groups: list[dict[str, object]] = []
    for key, parameters in buckets.items():
        if key[0] == "muon":
            _, lr, decay, logical_splits = key
            groups.append(
                {
                    "params": parameters,
                    "lr": lr,
                    "momentum": 0.95,
                    "weight_decay": decay,
                    "use_muon": True,
                    "muon_scale_mode": "o51",
                    "logical_splits": logical_splits,
                }
            )
        else:
            _, lr, decay = key
            groups.append(
                {
                    "params": parameters,
                    "lr": lr,
                    "betas": betas,
                    "eps": epsilon,
                    "weight_decay": decay,
                    "update_clip_threshold": None,
                    "use_muon": False,
                }
            )
    optimizer = SingleDeviceMuonWithAuxAdam(groups)
    optimizer.o51_roles = roles
    return optimizer


def _button_adam_parameters(model: GPT) -> dict[str, nn.Parameter]:
    """Retain O50's diagnostics only for tensors that remain on AdamW."""
    button_head = cast(_o50.NonlinearActionHead, model.temporal.outputs["buttons"])
    return {
        "buttons_output_weight": cast(nn.Parameter, button_head.down.weight),
        "buttons_condition_weight": cast(nn.Parameter, model.temporal.group_condition["buttons"].weight),
    }


def microbatch_loss(
    model: GPT,
    batch: AWRBatch,
    cfg: TrainConfig,
    *,
    step: int,
    valid_prefixes: int,
    trunk_fn: Callable,
    temporal_fn: Callable,
    phase_timer: _o50.CudaPhaseTimer | None = None,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """O50's AWR loss, activated only after D/32 valid positions."""
    if not isinstance(batch, AWRBatch):
        raise TypeError(f"advantage training needs an AWRBatch, got {type(batch).__name__}")
    history, targets, valid = _o50.prepared_targets(model, batch)
    if phase_timer is not None:
        phase_timer.record("target_prep_end")
    with _o50.amp_context(cfg, DEVICE):
        hidden = trunk_fn(batch.context.features, batch.context.ctx_pad, None)
        if phase_timer is not None:
            phase_timer.record("trunk_end")
        suffix_start = _o50.direct_loss_start(cfg)
        hidden = hidden[:, suffix_start:]
        temporal_output = temporal_fn(hidden, history, targets)
        if isinstance(temporal_output, Tensor):
            dense_nll = temporal_output
            button_diagnostics: dict[str, Tensor] = {}
        else:
            dense_nll, button_diagnostics = temporal_output
        if phase_timer is not None:
            phase_timer.record("temporal_end")
    value = model.value_head(_o50.decoder_rmsnorm(hidden).detach().float()).squeeze(-1)
    value_loss, advantage, value_stats = _o50.value_objective(
        value,
        batch.returns[:, suffix_start:],
        batch.eligible[:, suffix_start:],
        beta=cfg.awr.beta,
        valid=valid,
    )
    active = step + 1 > cfg.warmup_steps
    weights, stats = _o50.advantage_weights(
        advantage,
        batch.eligible[:, suffix_start:],
        beta=cfg.awr.beta,
        weight_max=cfg.awr.weight_max,
        active=active,
        valid=valid,
    )
    button_loss = dense_nll[..., BUTTONS_G].float().mean(dim=-1)
    stats["weight_button_loss_correlation"] = _o50.masked_correlation(
        weights,
        button_loss,
        batch.eligible[:, suffix_start:] & valid,
    )
    near, far, policy_loss = _o50.temporal_objective_parts(
        dense_nll,
        weights,
        valid_prefixes=valid_prefixes,
        aux_loss_weight=cfg.awr.auxiliary_loss_weight,
        valid=valid,
    )
    loss = policy_loss + cfg.awr.value_loss_weight * value_loss
    nll_sum = torch.where(valid[..., None, None], dense_nll.float(), 0).sum(dim=(0, 1))
    extra = {
        "train/loss": policy_loss.detach() / _o50._LN2,
        "train/near_loss": near.detach() / _o50._LN2,
        "train/far_nll": far.detach() / _o50._LN2,
        "train/objective": loss.detach(),
        "value/loss": value_stats["value_loss"],
        "value/rmse": value_stats["value_rmse"],
        "awr/active": torch.ones_like(loss) if active else torch.zeros_like(loss),
        "awr/eligible_fraction": stats["eligible_frac"],
        "awr/weight_mean": stats["weight_mean"],
        "awr/weight_max": stats["weight_max"],
        "awr/ess_fraction": stats["weight_ess"],
        "awr/cap_fraction": stats["weight_clip_frac"],
        "awr/button_loss_correlation": stats["weight_button_loss_correlation"],
        **button_diagnostics,
    }
    if phase_timer is not None:
        phase_timer.record("objective_end")
    return loss, nll_sum.detach(), extra


@dataclass(frozen=True, slots=True)
class ArmDecision:
    status: Literal["pass", "pause", "reject"]
    reasons: tuple[str, ...]


def arm_decision(
    metrics: dict[str, float],
    *,
    post_warmup_clip_fraction: float,
    initial_action_pre_norm_rms: float | None = None,
    initial_centered_logit_p999: float | None = None,
    sustained_growth: bool = False,
) -> ArmDecision:
    """Apply O51's numerical rejection and diagnostic-pause rules."""
    rejected: list[str] = []
    paused: list[str] = []
    nonfinite = sorted(name for name, value in metrics.items() if not math.isfinite(value))
    if nonfinite:
        rejected.append(f"non-finite metrics: {nonfinite}")
    centered = metrics.get("stability/centered_logit_abs_p999", 0.0)
    if centered > 64:
        rejected.append(f"centered-logit p999 {centered:g} exceeds 64")
    if post_warmup_clip_fraction > 0.10:
        rejected.append(f"post-warmup clipping {post_warmup_clip_fraction:.3f} exceeds 0.10")
    if sustained_growth and initial_centered_logit_p999 is not None and centered >= 4 * initial_centered_logit_p999:
        rejected.append("centered logits sustained fourfold growth")
    current_rms = metrics.get("stability/action_pre_norm_rms", metrics.get("stability/button_pre_norm_rms_min"))
    if (
        sustained_growth
        and initial_action_pre_norm_rms is not None
        and current_rms is not None
        and current_rms >= 4 * initial_action_pre_norm_rms
    ):
        rejected.append("action pre-norm RMS sustained fourfold growth")
    raw_button = metrics.get("stability/uncentered_button_logit_abs_p999", 0.0)
    if raw_button > 128:
        paused.append(f"raw button-logit p999 {raw_button:g} exceeds 128")
    if rejected:
        return ArmDecision("reject", tuple(rejected))
    if paused:
        return ArmDecision("pause", tuple(paused))
    return ArmDecision("pass", ())


class _ArmGuard:
    """Accumulate post-warmup clipping and require sustained growth to persist."""

    def __init__(self, warmup_updates: int) -> None:
        self._warmup_updates = warmup_updates
        self._last_training_update = 0
        self._clip_sum = 0.0
        self._clip_updates = 0
        self._initial_rms: float | None = None
        self._initial_centered: float | None = None
        self._centered_growth_windows = 0
        self._rms_growth_windows = 0

    def observe(self, metrics: dict[str, object]) -> ArmDecision | None:
        update = metrics.get("global_step")
        if not isinstance(update, int):
            return None
        centered = metrics.get("stability/centered_logit_abs_p999")
        rms = metrics.get("stability/action_pre_norm_rms", metrics.get("stability/button_pre_norm_rms_min"))
        if not isinstance(centered, int | float) and not isinstance(rms, int | float):
            return None
        numeric = {name: float(value) for name, value in metrics.items() if isinstance(value, int | float)}
        clip_fraction = metrics.get("optimizer/clip_fraction")
        window_updates = update - self._last_training_update
        if (
            isinstance(clip_fraction, int | float)
            and window_updates > 0
            and self._last_training_update >= self._warmup_updates
        ):
            self._clip_sum += float(clip_fraction) * window_updates
            self._clip_updates += window_updates
        self._last_training_update = update
        post_warmup_clip = self._clip_sum / max(self._clip_updates, 1)

        if update >= self._warmup_updates:
            if self._initial_centered is None and isinstance(centered, int | float):
                self._initial_centered = max(float(centered), torch.finfo(torch.float32).tiny)
            if self._initial_rms is None and isinstance(rms, int | float):
                self._initial_rms = max(float(rms), torch.finfo(torch.float32).tiny)
            if isinstance(centered, int | float) and self._initial_centered is not None:
                grew = float(centered) >= 4 * self._initial_centered
                self._centered_growth_windows = self._centered_growth_windows + 1 if grew else 0
            if isinstance(rms, int | float) and self._initial_rms is not None:
                grew = float(rms) >= 4 * self._initial_rms
                self._rms_growth_windows = self._rms_growth_windows + 1 if grew else 0
        return arm_decision(
            numeric,
            post_warmup_clip_fraction=post_warmup_clip,
            initial_action_pre_norm_rms=self._initial_rms,
            initial_centered_logit_p999=self._initial_centered,
            sustained_growth=max(self._centered_growth_windows, self._rms_growth_windows) >= 4,
        )


def data_selection(cfg: TrainConfig) -> CorpusSelection:
    """Return the pinned direct-source tier definitions."""
    del cfg
    return corpus_selection()


def load_stats(cfg: TrainConfig) -> dict[str, _o50.FeatureStats]:
    """Combine existing source statistics with the selected direct-prefix mix."""
    tier = data_selection(cfg).tier(cfg.tier_scale)
    sources = tuple(streams.BY_NAME[name] for name in cfg.source_names)
    selected = tier.source_replay_counts()
    return load_consolidated_mixture_stats(
        [source.local_root / "stats.json" for source in sources],
        [float(selected[source.name]) for source in sources],
        expected_mds_schema_version=cfg.mds_schema_version,
    )


def _make_train_loader(
    cfg: TrainConfig,
    stats: dict[str, _o50.FeatureStats],
    player_lookup: ReplayPlayerLookup,
) -> ReservoirLoader:
    """Build the direct source-prefix training loader."""
    tier = data_selection(cfg).tier(cfg.tier_scale)
    sources = tuple(streams.BY_NAME[name] for name in cfg.source_names)
    views = tier.sources
    if tuple(view.source for view in views) != cfg.source_names:
        raise RuntimeError("direct prefix views do not match configured source order")
    prefixes = tuple(StreamSamplePrefix(view.stop, view.excluded_rows) for view in views)
    projection = replace(
        _o50.MODEL_PROJECTION,
        columns=_o50.MODEL_PROJECTION.columns
        | {
            f"ego_{O51_RETURN_SUFFIX}",
            f"ego_{O51_RETURN_SUFFIX}_valid",
        },
    )
    train_loader = make_reservoir_loader(
        data_root=None,
        sources=sources,
        source_prefixes=prefixes,
        split="train",
        cache_limit=f"{cfg.cache_limit_gb}gb",
        shuffle=True,
        shuffle_seed=cfg.seed,
        shuffle_algo=cfg.shuffle_algo,
        shuffle_block_size=cfg.shuffle_block_size,
        stats=stats,
        L_ctx=cfg.arch.L_ctx,
        L_chunk=cfg.arch.sample_chunk_length,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        schema_version=cfg.mds_schema_version,
        extra=_o50.MODEL_COLUMNS,
        projection=projection,
        num_workers=cfg.num_workers,
        prefetch_factor=cfg.loader_prefetch_factor,
        predownload=cfg.predownload,
        download_retry=cfg.download_retry,
        windows_per_replay=cfg.windows_per_replay,
        reservoir_capacity=cfg.reservoir_capacity,
        prefetch_batches=0,
        replay_format="policy-world",
        replay_labels=DirectO51ReplayLabels(
            player_lookup=player_lookup,
            gamma=cfg.awr.gamma,
            damage_shaping=cfg.awr.damage_shaping,
            win_reward=cfg.awr.win_reward,
            stock_value=cfg.awr.stock_value,
        ),
        batch_transform=functools.partial(_o50.collate_awr_batch, L_ctx=cfg.arch.L_ctx),
        replay_pack_batch_size=cfg.replay_pack_batch_size,
        worker_independent_resume=True,
        deterministic_out_of_order=True,
        cooldown_batches=cfg.replay_cooldown_batches,
        limit_worker_threads=True,
        require_full_context=True,
    )
    expected_train_rows = OFFICIAL_TIER_REPLAYS[cfg.tier_scale]
    actual_train_rows = sum(train_loader.source_sample_counts.values())
    if actual_train_rows != expected_train_rows:
        raise ValueError(
            f"direct U{cfg.tier_scale} view exposes {actual_train_rows} rows, expected {expected_train_rows}"
        )
    _o50.TRAIN_REPLAYS = expected_train_rows
    return train_loader


def _make_loaders(
    cfg: TrainConfig,
    stats: dict[str, _o50.FeatureStats],
) -> tuple[ReservoirLoader, list[_o50.TrainBatch]]:
    """Build direct source-prefix training and the fixed validation cohort."""
    player_lookup = ReplayPlayerLookup(_o50.load_identity_sidecar(cfg).by_replay)
    train_loader = _make_train_loader(cfg, stats, player_lookup)

    common = _o50.loader_kwargs(cfg, stats)
    validation = {
        **common,
        "batch_size": _O50_DEFAULT_CONFIG.val_batch_size,
        "shuffle_block_size": _O50_DEFAULT_CONFIG.shuffle_block_size,
        "seed": 0,
        "shuffle_seed": 0,
    }
    val_loader = make_loader(
        split=cfg.val_split,
        num_workers=0,
        replay_format="policy-world",
        replay_labels=player_lookup,
        require_full_context=True,
        shuffle=True,
        **validation,
    )
    return train_loader, _o50.cache_validation(val_loader, cfg.val_n_samples)


def validate_shuffle_config(
    algorithm: str,
    block_size: int,
) -> None:
    if algorithm not in ("py1s", "py1e"):
        raise ValueError(f"unsupported shuffle algorithm {algorithm!r}")
    if block_size not in (4096, 8192):
        raise ValueError("O51 benchmarks shuffle blocks 4096 and 8192")


LOADER_WORKERS: Final[tuple[int, ...]] = (8, 16, 24, 32)
REPLAY_PACK_BATCHES: Final[tuple[int, ...]] = (16, 32, 64)
LOADER_PREFETCH_FACTORS: Final[tuple[int, ...]] = (1, 2, 4)
PREDOWNLOAD_MULTIPLIERS: Final[tuple[int, ...]] = (8, 16)
PHYSICAL_BATCHES: Final[tuple[int, ...]] = (128, 256, 512, 1024)
COMPILE_MODES: Final[tuple[str, ...]] = ("reduce-overhead", "max-autotune")
TEMPORAL_ATTENTION_CHUNKS: Final[tuple[int | None, ...]] = (8192, 16_384, 32_768, None)
REQUIRED_PREFLIGHT_TELEMETRY: Final[tuple[str, ...]] = (
    "system/network/read_mib_s",
    "system/cache/allocated_gib",
    "system/pinned_memory_gib",
    "system/cgroup/current_gib",
    "profile/target_prep_s",
    "profile/trunk_s",
    "profile/temporal_s",
    "profile/backward_s",
    "profile/grad_norm_s",
    "profile/diagnostics_s",
    "profile/optimizer_s",
    "throughput/mfu_wall_clock",
    "throughput/mfu_steady_state",
)


def preflight_fingerprint(cfg: TrainConfig, selection: CorpusSelection) -> str:
    """Bind throughput evidence to its model and direct-source selection."""
    payload = {
        "protocol": _EXPERIMENT_ID,
        "data_protocol": DATA_PROTOCOL,
        "corpus_hash": selection.corpus_hash,
        "source_manifest_sha256": selection.source_manifest_sha256,
        "tiers": {
            scale: {
                "sha256": tier.sha256,
                "sources": [
                    {
                        "source": source.source,
                        "stop": source.stop,
                        "excluded_rows": source.excluded_rows,
                    }
                    for source in tier.sources
                ],
            }
            for scale, tier in selection.tiers.items()
        },
        "player_sidecar_sha256": cfg.player_sidecar_sha256,
        "player_vocabulary_sha256": cfg.player_vocab_sha256,
        "return_parameters": asdict(cfg.awr),
        "model_level": model_level(cfg.arch),
        "architecture": {name: getattr(cfg.arch, name) for name in _MODEL_GEOMETRY_FIELDS},
        "batch_size": cfg.batch_size,
        "compile_mode": cfg.compile_mode,
        "temporal_attention_chunk": cfg.temporal_attention_chunk,
        "shuffle_algo": cfg.shuffle_algo,
        "shuffle_block_size": cfg.shuffle_block_size,
        "num_workers": cfg.num_workers,
        "cache_limit_gb": cfg.cache_limit_gb,
        "reservoir_capacity": cfg.reservoir_capacity,
        "replay_cooldown_batches": cfg.replay_cooldown_batches,
        "worker_completion_order": "deterministic-sample-id-tasks-v1",
        "windows_per_replay": cfg.windows_per_replay,
        "replay_pack_batch_size": cfg.replay_pack_batch_size,
        "loader_prefetch_factor": cfg.loader_prefetch_factor,
        "predownload": cfg.predownload,
        "download_retry": cfg.download_retry,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class PreflightReport:
    fingerprint: str
    synthetic_mfu: float
    compute_only_updates_per_s: float
    loader_only_windows_per_s: float
    raw_bytes_per_window: float
    o50_raw_bytes_per_window: float
    peak_memory_fraction: float
    graph_gap_fraction: float
    optimizer_time_fraction: float
    disk_capacity_bytes: int
    exact_resume: bool
    memory_passed: bool
    shuffle_passed: bool
    telemetry: dict[str, float]


@dataclass(frozen=True, slots=True)
class SoakReport(PreflightReport):
    """Preflight measurements plus the final U8 endurance evidence."""

    tier_scale: int
    end_to_end_updates_per_s: float
    gpu_windows_per_s: float
    loader_wait_mean_fraction: float
    loader_wait_p95_fraction: float
    end_to_end_mfu: float
    soak_seconds: float
    judged_seconds: float


def _numeric_report_values(
    report: object,
    names: tuple[str, ...],
    unit_fractions: tuple[str, ...],
    *,
    label: str,
) -> tuple[dict[str, float], tuple[str, ...]]:
    numeric: dict[str, float] = {}
    invalid: list[str] = []
    for name in names:
        value = getattr(report, name)
        if not isinstance(value, int | float) or isinstance(value, bool):
            invalid.append(name)
            continue
        numeric[name] = float(value)
        if not math.isfinite(numeric[name]) or numeric[name] < 0:
            invalid.append(name)
    invalid.extend(name for name in unit_fractions if numeric.get(name, 0) > 1)
    if invalid:
        return numeric, (f"{label} measurements are invalid {sorted(set(invalid))}",)
    return numeric, ()


def preflight_failures(cfg: TrainConfig, report: PreflightReport) -> tuple[str, ...]:
    """Check the short compute and loader evidence required before long runs."""
    selection = data_selection(cfg)
    failures: list[str] = []
    if report.fingerprint != preflight_fingerprint(cfg, selection):
        failures.append("preflight fingerprint does not match the selected shape and data")
    numeric_names = (
        "synthetic_mfu",
        "compute_only_updates_per_s",
        "loader_only_windows_per_s",
        "raw_bytes_per_window",
        "o50_raw_bytes_per_window",
        "peak_memory_fraction",
        "graph_gap_fraction",
        "optimizer_time_fraction",
        "disk_capacity_bytes",
    )
    unit_fractions = (
        "synthetic_mfu",
        "peak_memory_fraction",
        "graph_gap_fraction",
        "optimizer_time_fraction",
    )
    numeric, numeric_failures = _numeric_report_values(
        report,
        numeric_names,
        unit_fractions,
        label="preflight",
    )
    failures.extend(numeric_failures)
    if numeric_failures:
        return tuple(failures)
    if model_level(cfg.arch) == "mid" and numeric["synthetic_mfu"] < _MIN_SYNTHETIC_MFU:
        failures.append("55M synthetic compiled MFU is below the measured 15% floor")
    if numeric["compute_only_updates_per_s"] <= 0:
        failures.append("compute-only throughput is zero")
    if numeric["loader_only_windows_per_s"] <= 0:
        failures.append("loader-only throughput is zero")
    if (
        numeric["o50_raw_bytes_per_window"] <= 0
        or numeric["raw_bytes_per_window"] > 0.35 * numeric["o50_raw_bytes_per_window"]
    ):
        failures.append("raw bytes per window exceed 35% of O50 K=1")
    if numeric["peak_memory_fraction"] >= 0.95:
        failures.append("peak device memory is not below 95%")
    if numeric["graph_gap_fraction"] > 0.05:
        failures.append("loss-path graph gaps exceed 5%")
    if numeric["optimizer_time_fraction"] > 0.10:
        failures.append("optimizer time exceeds 10%")
    flags = {
        "exact_resume": report.exact_resume,
        "memory_passed": report.memory_passed,
        "shuffle_passed": report.shuffle_passed,
    }
    invalid_flags = sorted(name for name, value in flags.items() if not isinstance(value, bool))
    if invalid_flags:
        failures.append(f"preflight pass flags are not boolean {invalid_flags}")
    if report.exact_resume is not True:
        failures.append("tensor-exact resume did not pass")
    if report.memory_passed is not True:
        failures.append("host/pinned-memory gate did not pass")
    if report.shuffle_passed is not True:
        failures.append("shuffle audit did not pass")
    if not isinstance(report.telemetry, dict):
        failures.append("preflight telemetry must be an object")
        return tuple(failures)
    missing_telemetry = sorted(set(REQUIRED_PREFLIGHT_TELEMETRY) - report.telemetry.keys())
    invalid_telemetry = sorted(
        name
        for name, value in report.telemetry.items()
        if name in REQUIRED_PREFLIGHT_TELEMETRY
        and (not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value) or value < 0)
    )
    if missing_telemetry:
        failures.append(f"preflight telemetry is missing {missing_telemetry}")
    if invalid_telemetry:
        failures.append(f"preflight telemetry is invalid {invalid_telemetry}")
    required_disk = (cfg.cache_limit_gb + _MIN_FREE_CACHE_GIB) * 2**30
    if numeric["disk_capacity_bytes"] < required_disk:
        failures.append("streaming cache does not leave 256 GiB of free disk")
    try:
        validate_shuffle_config(cfg.shuffle_algo, cfg.shuffle_block_size)
    except ValueError as error:
        failures.append(str(error))
    return tuple(failures)


def soak_failures(cfg: TrainConfig, report: SoakReport) -> tuple[str, ...]:
    """Check the final two-hour 55M run over the full direct U8 view."""
    failures = list(preflight_failures(cfg, report))
    if model_level(cfg.arch) != "mid":
        failures.append("the authoritative soak must use the 55M model")
    if not isinstance(report.tier_scale, int) or isinstance(report.tier_scale, bool):
        failures.append("soak tier must be an integer")
    elif report.tier_scale != 8 or cfg.tier_scale != 8:
        failures.append("the authoritative soak must use the full U8 tier")
    numeric, numeric_failures = _numeric_report_values(
        report,
        (
            "compute_only_updates_per_s",
            "loader_only_windows_per_s",
            "end_to_end_updates_per_s",
            "gpu_windows_per_s",
            "loader_wait_mean_fraction",
            "loader_wait_p95_fraction",
            "end_to_end_mfu",
            "soak_seconds",
            "judged_seconds",
        ),
        (
            "loader_wait_mean_fraction",
            "loader_wait_p95_fraction",
            "end_to_end_mfu",
        ),
        label="soak",
    )
    failures.extend(numeric_failures)
    if numeric_failures:
        return tuple(failures)
    if numeric["soak_seconds"] < 7200 or numeric["judged_seconds"] < 1800:
        failures.append("soak does not cover two hours with a final 30-minute judgment window")
    if numeric["end_to_end_mfu"] < _MIN_FULL_TIER_MFU:
        failures.append("full-tier end-to-end MFU is below the revised 13.5% floor")
    if numeric["loader_only_windows_per_s"] < 1.25 * numeric["gpu_windows_per_s"]:
        failures.append("loader-only throughput is below 125% of GPU consumption")
    if numeric["end_to_end_updates_per_s"] < 0.9 * numeric["compute_only_updates_per_s"]:
        failures.append("end-to-end throughput retains less than 90% of compute-only throughput")
    if numeric["loader_wait_mean_fraction"] >= 0.05:
        failures.append("mean loader wait is not below 5%")
    if numeric["loader_wait_p95_fraction"] >= 0.10:
        failures.append("p95 loader wait is not below 10%")
    return tuple(failures)


def load_preflight(path: Path) -> PreflightReport:
    return PreflightReport(**json.loads(path.read_text()))


def load_soak(path: Path) -> SoakReport:
    return SoakReport(**json.loads(path.read_text()))


def _weighted_percentile(values: list[tuple[float, float]], percentile: float) -> float:
    if not values or not 0 <= percentile <= 1:
        raise ValueError("weighted percentile needs samples and a percentile in [0, 1]")
    threshold = percentile * sum(weight for _, weight in values)
    cumulative = 0.0
    for value, weight in sorted(values):
        cumulative += weight
        if cumulative >= threshold:
            return value
    return max(value for value, _ in values)


def collect_soak_report(
    cfg: TrainConfig,
    preflight: PreflightReport,
    history: Iterable[Mapping[str, object]],
    *,
    judged_seconds: float = 1800.0,
) -> SoakReport:
    """Aggregate the final judgment window from a fresh W&B training history."""
    if not math.isfinite(judged_seconds) or judged_seconds <= 0:
        raise ValueError("judged_seconds must be finite and positive")
    rows: list[dict[str, float]] = []
    for raw in history:
        names = (
            "global_step",
            "progress/elapsed_s",
            "throughput/update_s",
            "loader/wait_s",
        )
        if any(name not in raw for name in names):
            continue
        mfu = raw.get("throughput/mfu_wall_clock", raw.get("throughput/mfu"))
        values = {name: raw[name] for name in names} | {"throughput/mfu_wall_clock": mfu}
        if any(
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0
            for value in values.values()
        ):
            continue
        row = {name: float(value) for name, value in values.items()}
        if row["throughput/update_s"] <= 0:
            continue
        for name in REQUIRED_PREFLIGHT_TELEMETRY:
            value = raw.get(name)
            if (
                isinstance(value, int | float)
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and float(value) >= 0
            ):
                row[name] = float(value)
        rows.append(row)
    rows.sort(key=lambda row: (row["progress/elapsed_s"], row["global_step"]))
    if len(rows) < 2:
        raise ValueError("soak history has fewer than two complete throughput windows")
    last_elapsed = rows[-1]["progress/elapsed_s"]
    cutoff = last_elapsed - judged_seconds
    if cutoff < 0 or rows[0]["progress/elapsed_s"] > cutoff:
        raise ValueError("soak history does not cover the requested judgment window")

    samples: list[tuple[dict[str, float], float]] = []
    for previous, row in zip(rows, rows[1:], strict=False):
        start = previous["progress/elapsed_s"]
        stop = row["progress/elapsed_s"]
        updates = row["global_step"] - previous["global_step"]
        if stop <= cutoff or stop <= start or updates <= 0:
            continue
        overlap = (stop - max(start, cutoff)) / (stop - start)
        samples.append((row, updates * overlap))
    if not samples:
        raise ValueError("soak history has no training windows in the judgment period")

    total_updates = sum(weight for _, weight in samples)

    def update_mean(name: str) -> float:
        return sum(row[name] * weight for row, weight in samples) / total_updates

    update_s = update_mean("throughput/update_s")
    wait_fractions = [
        (row["loader/wait_s"] / max(row["throughput/update_s"], 1e-12), weight) for row, weight in samples
    ]
    elapsed_weights = [
        (row["throughput/mfu_wall_clock"], weight * row["throughput/update_s"]) for row, weight in samples
    ]
    total_elapsed_weight = sum(weight for _, weight in elapsed_weights)
    telemetry = {
        name: float(np.mean([row[name] for row, _ in samples if name in row]))
        for name in REQUIRED_PREFLIGHT_TELEMETRY
        if any(name in row for row, _ in samples)
    }
    telemetry["throughput/mfu_wall_clock"] = (
        sum(value * weight for value, weight in elapsed_weights) / total_elapsed_weight
    )
    report = SoakReport(
        **{**asdict(preflight), "telemetry": telemetry},
        tier_scale=cfg.tier_scale,
        end_to_end_updates_per_s=1.0 / update_s,
        gpu_windows_per_s=cfg.batch_size / update_s,
        loader_wait_mean_fraction=sum(value * weight for value, weight in wait_fractions) / total_updates,
        loader_wait_p95_fraction=_weighted_percentile(wait_fractions, 0.95),
        end_to_end_mfu=telemetry["throughput/mfu_wall_clock"],
        soak_seconds=last_elapsed,
        judged_seconds=judged_seconds,
    )
    return report


@dataclass(frozen=True, slots=True)
class PromotionEvidence:
    seeds: int
    stock_per_min_gain: float
    paired_90_lower_bound: float
    damage_per_min_loss: float
    validation_nll: float
    best_validation_nll: float


def validate_large_promotion(evidence: PromotionEvidence) -> None:
    numeric = tuple(
        getattr(evidence, name)
        for name in (
            "stock_per_min_gain",
            "paired_90_lower_bound",
            "damage_per_min_loss",
            "validation_nll",
            "best_validation_nll",
        )
    )
    if any(not math.isfinite(value) for value in numeric):
        raise ValueError("216M promotion evidence contains non-finite values")
    if evidence.seeds != 3:
        raise ValueError("216M promotion requires three 55M seeds")
    if evidence.stock_per_min_gain < 0.10:
        raise ValueError("55M does not beat proxy by at least +0.10 stock/min")
    if evidence.paired_90_lower_bound <= 0:
        raise ValueError("paired 90% lower bound is not positive")
    if evidence.damage_per_min_loss > 5:
        raise ValueError("55M loses more than 5 damage/min")
    if evidence.validation_nll > 1.01 * evidence.best_validation_nll:
        raise ValueError("55M validation NLL is more than 1% from the best")


def model_tag(cfg: TrainConfig) -> str:
    level = model_level(cfg.arch)
    duration = "d-fixed" if cfg.muon_duration_scaling == "fixed" else "d-invsqrt"
    batch = "b-fixed" if cfg.muon_batch_scaling == "fixed" else "b-sqrt"
    return (
        f"o51-{level}-a{cfg.depth_alpha:g}-D{cfg.target_positions // (D0 // 8):03d}eighth-"
        f"U{cfg.tier_scale}-{cfg.readout_init}-hstd{cfg.hidden_std_multiplier:g}-{duration}-{batch}"
    )


def source_mixture_weights(cfg: TrainConfig) -> tuple[float, ...]:
    """Return the selected tier's actual replay-count mixture."""
    selected = data_selection(cfg).tier(cfg.tier_scale).source_replay_counts()
    return tuple(float(selected[name]) for name in cfg.source_names)


def _install_wandb_log_guard(cfg: TrainConfig) -> None:
    """Install O51 metric semantics and arm gates after W&B initializes."""
    arm_guard = _ArmGuard(cfg.warmup_steps)
    original_log = _o50.wandb.log

    def guarded_log(values: dict[str, object], *log_args: object, **log_kwargs: object) -> object:
        payload = dict(values)
        wall_mfu = payload.get("throughput/mfu")
        update_s = payload.get("throughput/update_s")
        if isinstance(wall_mfu, int | float) and isinstance(update_s, int | float):
            payload["throughput/mfu_wall_clock"] = wall_mfu
            phase_s = sum(
                value
                for name, value in payload.items()
                if name.startswith("profile/") and name.endswith("_s") and isinstance(value, int | float)
            )
            if phase_s > 0:
                payload["throughput/mfu_steady_state"] = wall_mfu * update_s / phase_s
        result = original_log(payload, *log_args, **log_kwargs)
        decision = arm_guard.observe(payload)
        if decision is not None and decision.status != "pass":
            reasons = "; ".join(decision.reasons)
            raise RuntimeError(f"O51 arm {decision.status}: {reasons}")
        return result

    _o50.wandb.log = guarded_log


def _init_wandb(cfg: TrainConfig, run_name: str, resume_state: dict[str, object] | None) -> None:
    """Start an O51 run with its nested-data identity in the immutable config."""
    selection = data_selection(cfg)
    _o50.wandb.init(
        project="hal",
        group="o51-correct-parameterization",
        name=run_name,
        id=None if resume_state is None else resume_state.get("wandb_id"),
        resume="allow" if resume_state is not None else None,
        tags=[
            "051",
            "nested-data",
            "muon-o51-scale",
            "centered-logits",
            model_level(cfg.arch),
            f"U{cfg.tier_scale}",
        ],
        config={
            **asdict(cfg),
            "data_protocol": DATA_PROTOCOL,
            "data_corpus_hash": selection.corpus_hash,
            "data_tier_hash": selection.tier(cfg.tier_scale).sha256,
        },
        settings=_o50.wandb.Settings(
            x_stats_sampling_interval=5.0,
            x_stats_track_process_tree=True,
        ),
    )
    # wandb.init() rebinds wandb.log. Install the guard only after that rebind.
    _install_wandb_log_guard(cfg)
    if _o50.wandb.run is None:
        return
    _o50.wandb.define_metric("global_step")
    _o50.wandb.define_metric("*", step_metric="global_step")
    summary = _o50.wandb.run.summary
    summary["nll_semantics"] = (
        "train/loss is weighted policy loss in bits; train/nll is unweighted; "
        "train/objective is the optimized policy-plus-value objective"
    )
    summary["layer_rms_semantics"] = (
        "activation=block input; residual_branch=block output-input; attention and MLP branch metrics "
        "include O51 depth multipliers"
    )
    summary["architecture/treatment"] = (
        "O50 task and controller with O51 depth parameterization, semantic optimizer roles, "
        "direct source-prefix data, runtime AWR labels, and centered class logits"
    )
    summary["optimizer/muon_scale"] = "sqrt(d_out/d_in) per logical matrix"
    summary["optimizer/lr_schedule"] = cfg.lr_schedule_kind
    summary["data/corpus_hash"] = selection.corpus_hash
    summary["data/tier_hash"] = selection.tier(cfg.tier_scale).sha256
    if cfg.wandb_log_code:
        _o50.log_wandb_code(_o50.wandb.run)


def _log_training_summary(
    cfg: TrainConfig,
    parameter_counts: dict[str, int],
    *,
    flops_per_update: int,
    device_name: str | None,
    peak_flops: float | None,
) -> None:
    """Record the selected nested pool and its actual per-source counters."""
    if _o50.wandb.run is None:
        return
    summary = _o50.wandb.run.summary
    selection = data_selection(cfg)
    tier = selection.tier(cfg.tier_scale)
    selected = tier.source_replay_counts()
    for name, value in parameter_counts.items():
        summary[f"parameters/{name}"] = value
    unique_replays = tier.unique_replays
    potential_targets = tier.potential_targets
    if unique_replays != OFFICIAL_TIER_REPLAYS[cfg.tier_scale]:
        raise RuntimeError("selected-tier replay accounting changed after loader construction")
    if potential_targets != OFFICIAL_TIER_TARGETS[cfg.tier_scale]:
        raise RuntimeError("selected-tier target accounting changed after loader construction")
    summary["data/unique_replays"] = unique_replays
    summary["data/unique_frames"] = tier.frames
    summary["data/potential_port_frame_targets"] = potential_targets
    summary["data/processed_loss_positions"] = cfg.target_positions
    summary["data/effective_target_epochs"] = cfg.target_positions / potential_targets
    summary["data/windows_per_replay"] = cfg.target_positions / (128 * unique_replays)
    summary["data/D_over_N"] = cfg.target_positions / parameter_counts["total"]
    summary["data/valid_positions_per_update"] = cfg.batch_size * _SUPERVISED_POSITIONS_PER_WINDOW
    summary["data/replay_pack_batch_size"] = cfg.replay_pack_batch_size
    summary["data/loader_prefetch_factor"] = cfg.loader_prefetch_factor
    summary["data/source_mixing"] = "direct_source_prefixes"
    summary["data/corpus_hash"] = selection.corpus_hash
    summary["data/tier_hash"] = selection.tier(cfg.tier_scale).sha256
    summary["training/approx_flops_per_update"] = flops_per_update
    summary["training/flops_formula"] = "6*B*L_ctx*(N_trunk+N_other+N_value+n_offsets*(N_temporal+N_group_heads))"
    if device_name is not None:
        summary["hardware/gpu_name"] = device_name
    if peak_flops is not None:
        summary["hardware/bf16_dense_peak_tflops"] = peak_flops / 1e12
        source = _o50.bf16_peak_source(device_name or "")
        if source is not None:
            summary["hardware/bf16_dense_peak_source"] = source
    for source_name in cfg.source_names:
        source_replays = selected[source_name]
        summary[f"data/source_sampling_share/{source_name}"] = source_replays / unique_replays
        summary[f"data/source_replays/{source_name}"] = source_replays


def _training_functions(model: GPT, cfg: TrainConfig) -> tuple[Callable, Callable]:
    """Compile O51's selected trunk and complete temporal loss entrypoints."""
    trunk_fn: Callable = model.forward_unpadded
    temporal_fn: Callable = model.temporal.teacher_forced_nll
    if DEVICE == "cuda" and cfg.compile_trunk:
        trunk_fn = torch.compile(
            trunk_fn,
            dynamic=False,
            fullgraph=True,
            mode=cfg.compile_mode,
        )
    if DEVICE == "cuda" and cfg.compile_temporal:
        temporal_fn = torch.compile(
            temporal_fn,
            dynamic=False,
            fullgraph=True,
            mode=cfg.compile_mode,
        )
    return trunk_fn, temporal_fn


@torch.no_grad()
@torch.compiler.disable
def stability_diagnostics_log(
    model: GPT,
    batch: AWRBatch,
    cfg: TrainConfig,
    *,
    max_rows: int,
) -> dict[str, float]:
    """Measure O51 rejection signals outside the compiled training path."""
    diagnostic = _o50._diagnostic_awr_batch(batch, max_rows)
    history, targets, _valid = _o50.prepared_targets(model, diagnostic)
    device = next(model.parameters()).device
    with _o50.amp_context(cfg, device):
        hidden = model(diagnostic.context.features, diagnostic.context.ctx_pad, None)
        hidden = hidden[:, _o50.direct_loss_start(cfg) :]
        _nll, metrics = model.temporal.teacher_forced_nll_with_diagnostics(hidden, history, targets)
    values = torch.stack(tuple(metrics.values())).double().cpu()
    payload = {name: float(value) for name, value in zip(metrics, values, strict=True)}
    nonfinite = {name: value for name, value in payload.items() if not math.isfinite(value)}
    if nonfinite:
        raise FloatingPointError(f"stability diagnostic produced non-finite metrics: {nonfinite}")
    return payload


def _training_diagnostics(model: GPT, batch: AWRBatch, cfg: TrainConfig, update: int) -> dict[str, object]:
    """Collect O50 diagnostics plus O51's periodic arm-rejection signals."""
    metrics = _O50_TRAINING_DIAGNOSTICS(model, batch, cfg, update)
    if update % cfg.stability_every == 0:
        metrics.update(stability_diagnostics_log(model, batch, cfg, max_rows=cfg.layer_rms_batch_size))
    return metrics


def layer_activation_rms_log(
    model: GPT,
    batch: _o50.TrainBatch | AWRBatch,
    cfg: TrainConfig,
    *,
    max_rows: int,
) -> dict[str, float]:
    """Correct O50's unscaled temporal-MLP hook for O51's branch rule."""
    metrics = _O50_LAYER_ACTIVATION_RMS_LOG(model, batch, cfg, max_rows=max_rows)
    multiplier = depth_rule("temporal", cfg.arch.temporal_layers, cfg.depth_alpha).mlp
    return {
        name: value * multiplier if name.startswith("mlp_branch_rms/temporal_block_") else value
        for name, value in metrics.items()
    }


def _minimal_system_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Retain the full O51 host bottleneck panel collected off the hot path."""
    selected = {
        name: value
        for name, value in metrics.items()
        if name.startswith(
            (
                "system/cpu/",
                "system/disk/",
                "system/network/",
                "system/page_faults/",
                "system/major_page_faults/",
                "system/cache/",
                "system/process_tree/",
                "system/cgroup/",
                "system/pinned_memory_",
            )
        )
    }
    selected["system/telemetry_errors"] = metrics.get("system/telemetry_errors", 0.0)
    return selected


def _finalize_training(**kwargs: object) -> None:
    cfg = cast(TrainConfig, kwargs["cfg"])
    update = int(cast(int, kwargs["update"]))
    actual = int(cast(int, kwargs["actual_loss_positions"]))
    if update == cfg.max_steps and actual != cfg.target_positions:
        raise RuntimeError(f"O51 stopped at {actual} valid positions instead of D={cfg.target_positions}")
    _O50_FINALIZE_TRAINING(**kwargs)
    smoke = bool(kwargs["smoke"])
    if not smoke and model_level(cfg.arch) == "large" and cfg.target_positions == D0 and cfg.tier_scale == 1:
        run_dir = cast(Path, kwargs["run_dir"])
        selection = data_selection(cfg)
        evidence_path = run_dir / "large-d0-evidence.json"
        evidence = {
            "completed": True,
            "experiment_id": _EXPERIMENT_ID,
            "model_level": "large",
            "target_positions": D0,
            "tier_scale": 1,
            "corpus_hash": selection.corpus_hash,
            "checkpoint_sha256": _o50._checkpoint_sha256(run_dir / "final.pt"),
        }
        evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        uploader = kwargs["uploader"]
        if uploader is not None:
            uploader.upload(evidence_path, key=evidence_path.name)


_ARCHITECTURE_FIELDS = frozenset(item.name for item in fields(Architecture))
_AWR_FIELDS = frozenset(item.name for item in fields(AWRCalibration))
_RUNTIME_CONFIG_FIELDS = frozenset(item.name for item in fields(TrainConfig)) - {"arch", "awr"}


def _checkpoint_config(cfg: TrainConfig) -> dict[str, object]:
    values = asdict(cfg)
    architecture = values.pop("arch")
    calibration = values.pop("awr")
    return {
        "experiment_id": _EXPERIMENT_ID,
        "architecture": architecture,
        "awr_calibration": calibration,
        **values,
    }


def config_from_state(values: dict[str, object]) -> TrainConfig:
    expected = {"experiment_id", "architecture", "awr_calibration", *_RUNTIME_CONFIG_FIELDS}
    missing = expected - values.keys()
    unexpected = values.keys() - expected
    if missing or unexpected:
        raise ValueError(f"checkpoint config mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    if values["experiment_id"] != _EXPERIMENT_ID:
        raise ValueError(f"checkpoint experiment_id {values['experiment_id']!r} != {_EXPERIMENT_ID!r}")
    architecture_values = cast(dict[str, object], values["architecture"])
    calibration_values = cast(dict[str, object], values["awr_calibration"])
    if set(architecture_values) != _ARCHITECTURE_FIELDS:
        raise ValueError("checkpoint architecture does not match O51")
    if set(calibration_values) != _AWR_FIELDS:
        raise ValueError("checkpoint AWR calibration does not match O51")
    runtime = {name: values[name] for name in _RUNTIME_CONFIG_FIELDS}
    cfg = TrainConfig(
        arch=Architecture(**architecture_values),
        awr=AWRCalibration(**calibration_values),
        **runtime,
    )
    if cfg.max_steps != values["max_steps"] or cfg.warmup_steps != values["warmup_steps"]:
        raise ValueError("checkpoint position schedule is not derivable from D and batch size")
    validate_config(cfg)
    return cfg


def validate_production_config(cfg: TrainConfig) -> None:
    """O51's declared grids are treatments, not forbidden production overrides."""
    validate_config(cfg)
    if cfg.num_workers not in LOADER_WORKERS:
        raise ValueError(f"production num_workers must be one of {LOADER_WORKERS}")
    if cfg.shuffle_block_size not in (4096, 8192):
        raise ValueError("production shuffle block must be 4096 or 8192")
    if cfg.shuffle_algo not in ("py1s", "py1e"):
        raise ValueError("production shuffle algorithm must be py1s or py1e")
    if not cfg.compile_trunk or not cfg.compile_temporal:
        raise ValueError("production O51 requires compiled trunk and temporal loss paths")
    if cfg.amp_dtype != "bfloat16":
        raise ValueError("production O51 requires BFloat16 AMP")


def describe() -> dict[str, object]:
    return {
        "experiment": _EXPERIMENT_ID,
        "models": {
            name: {
                **{field: getattr(architecture, field) for field in _MODEL_GEOMETRY_FIELDS},
                "parameters": EXPECTED_PARAMETER_COUNTS[name],
            }
            for name, architecture in MODEL_FAMILY.items()
        },
        "depth_alpha": [0.5, 1.0],
        "initialization": {
            "hidden_std_multiplier": [0.5, 1.0, 2.0],
            "readout": ["zero", "mup-normal"],
        },
        "lr_grid": {
            "muon": [0.014, 0.028, 0.056],
            "adam": [2.125e-4, 4.25e-4, 8.5e-4],
            "adam_epsilon": _BASE_ADAM_EPS,
        },
        "decay_grid": [0.0, 0.001, 0.01],
        "batch_grid": list(PHYSICAL_BATCHES),
        "tiers": {
            scale: {
                "D": scale * D0,
                "updates_at_B512": scale * 16_384,
                "unique_replays": OFFICIAL_TIER_REPLAYS[scale],
                "potential_targets": OFFICIAL_TIER_TARGETS[scale],
            }
            for scale in TIER_SCALES
        },
        "loader_grid": {
            "workers": LOADER_WORKERS,
            "pack_batches": REPLAY_PACK_BATCHES,
            "prefetch_factors": LOADER_PREFETCH_FACTORS,
            "predownload_multipliers": PREDOWNLOAD_MULTIPLIERS,
            "shuffle_algorithms": ["py1s", "py1e"],
            "shuffle_blocks": [4096, 8192],
        },
        "compute_grid": {
            "physical_batches": PHYSICAL_BATCHES,
            "compile_modes": COMPILE_MODES,
            "temporal_attention_chunks": TEMPORAL_ATTENTION_CHUNKS,
        },
    }


@dataclass(frozen=True, slots=True)
class CoordinateCheck:
    level: str
    depth_alpha: float
    steps: int = 128


def coordinate_check_grid() -> tuple[CoordinateCheck, ...]:
    return tuple(CoordinateCheck(level, alpha) for level in MODEL_LEVELS for alpha in (0.5, 1.0))


def benchmark_train_step(
    cfg: TrainConfig,
    *,
    warmup_steps: int = 3,
    measured_steps: int = 20,
) -> dict[str, float]:
    """Measure one selected compiled shape without touching replay data."""
    if DEVICE != "cuda":
        raise RuntimeError("the O51 train-step benchmark requires CUDA")
    if warmup_steps < 1 or measured_steps < 1:
        raise ValueError("warmup_steps and measured_steps must be positive")
    validate_config(cfg)
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high" if cfg.allow_tf32 else "highest")
    model = GPT(cfg).to(DEVICE).train()
    counts = subsystem_parameter_counts(model)
    flops_per_update = _o50.approximate_training_flops_per_update(cfg, counts)
    optimizer = make_optimizer(model, cfg)
    scheduler = _o50.LambdaLR(optimizer, _o50.lr_schedule(cfg))
    trunk_fn, temporal_fn = _training_functions(model, cfg)
    batch = _o50.synthetic_awr_batch(cfg, torch.device(DEVICE))
    valid_prefixes = cfg.batch_size * _SUPERVISED_POSITIONS_PER_WINDOW
    for index in range(warmup_steps):
        _o50.train_step(
            model,
            batch,
            cfg,
            step=index,
            update=index + 1,
            valid_prefixes=valid_prefixes,
            trunk_fn=trunk_fn,
            temporal_fn=temporal_fn,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    timers: list[_o50.CudaPhaseTimer] = []
    started = _o50.time.monotonic()
    for index in range(measured_steps):
        timer = _o50.CudaPhaseTimer()
        timer.record("start")
        timer.record("h2d_end")
        _o50.train_step(
            model,
            batch,
            cfg,
            step=warmup_steps + index,
            update=warmup_steps + index + 1,
            valid_prefixes=valid_prefixes,
            trunk_fn=trunk_fn,
            temporal_fn=temporal_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            phase_timer=timer,
        )
        timers.append(timer)
    torch.cuda.synchronize()
    elapsed = _o50.time.monotonic() - started
    update_s = elapsed / measured_steps
    device_name = torch.cuda.get_device_name()
    peak_flops = _o50.bf16_dense_peak_flops(device_name)
    phase_metrics = _o50._mean_phase_metrics(timers)
    metrics = {
        "batch_size": float(cfg.batch_size),
        "measured_steps": float(measured_steps),
        "update_s": update_s,
        "samples_per_s": cfg.batch_size / update_s,
        "synthetic_mfu": 0.0
        if peak_flops is None
        else _o50.model_flops_utilization(flops_per_update, update_s, peak_flops),
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
        "muon_group_count": float(sum(group["use_muon"] for group in optimizer.param_groups)),
        "adam_group_count": float(sum(not group["use_muon"] for group in optimizer.param_groups)),
        **phase_metrics,
    }
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    return metrics


def benchmark_loader(
    cfg: TrainConfig,
    *,
    warmup_batches: int = 8,
    measured_batches: int = 50,
) -> dict[str, object]:
    """Measure the direct-source loader without running the model."""
    if warmup_batches < 1 or measured_batches < 1:
        raise ValueError("loader benchmark batch counts must be positive")
    validate_config(cfg)
    stats = load_stats(cfg)
    player_lookup = ReplayPlayerLookup(_o50.load_identity_sidecar(cfg).by_replay)
    loader = _make_train_loader(cfg, stats, player_lookup)
    iterator = iter(loader)
    for _ in range(warmup_batches):
        next(iterator)

    batch_seconds: list[float] = []
    replay_ids: set[str] = set()
    last_seen_batch: dict[str, int] = {}
    cooldown_passed = True
    started = _o50.time.monotonic()
    for batch_index in range(measured_batches):
        batch_started = _o50.time.monotonic()
        batch = next(iterator)
        batch_seconds.append(_o50.time.monotonic() - batch_started)
        if not isinstance(batch, AWRBatch) or batch.batch.replay_ids is None:
            raise TypeError("O51 loader benchmark requires replay-aware AWR batches")
        if len(batch.batch.replay_ids) != cfg.batch_size or len(set(batch.batch.replay_ids)) != cfg.batch_size:
            raise RuntimeError("O51 loader emitted a batch with repeated or missing replay IDs")
        for replay_id in batch.batch.replay_ids:
            previous = last_seen_batch.get(replay_id)
            if previous is not None and batch_index - previous <= cfg.replay_cooldown_batches:
                cooldown_passed = False
            last_seen_batch[replay_id] = batch_index
        replay_ids.update(batch.batch.replay_ids)
    elapsed = _o50.time.monotonic() - started
    windows = measured_batches * cfg.batch_size
    metrics: dict[str, object] = {
        "fingerprint": preflight_fingerprint(cfg, data_selection(cfg)),
        "tier_scale": cfg.tier_scale,
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "replay_pack_batch_size": cfg.replay_pack_batch_size,
        "loader_prefetch_factor": cfg.loader_prefetch_factor,
        "predownload": cfg.predownload,
        "warmup_batches": warmup_batches,
        "measured_batches": measured_batches,
        "measured_seconds": elapsed,
        "loader_only_windows_per_s": windows / elapsed,
        "batch_seconds_mean": float(np.mean(batch_seconds)),
        "batch_seconds_p95": float(np.percentile(batch_seconds, 95)),
        "distinct_replays": len(replay_ids),
        "within_batch_unique": True,
        "cooldown_batches": cfg.replay_cooldown_batches,
        "cooldown_passed": cooldown_passed,
        "source_sample_counts": loader.source_sample_counts,
    }
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    return metrics


def run_coordinate_checks(*, batch_size: int = 512, steps: int = 128) -> list[dict[str, object]]:
    """Run every model/depth coordinate for exactly 128 synthetic updates."""
    if steps != 128:
        raise ValueError("O51 coordinate checks are fixed to 128 updates")
    reports: list[dict[str, object]] = []
    for spec in coordinate_check_grid():
        cfg = config_for(
            spec.level,
            batch_size=batch_size,
            depth_alpha=spec.depth_alpha,
            push_to_r2=False,
            wandb_log_code=False,
        )
        metrics = benchmark_train_step(cfg, warmup_steps=1, measured_steps=steps)
        reports.append({"level": spec.level, "depth_alpha": spec.depth_alpha, **metrics})
    return reports


def _validate_large_d0_evidence(
    path: Path,
    selection: CorpusSelection,
) -> None:
    payload = json.loads(path.read_text())
    checkpoint_hash = payload.get("checkpoint_sha256")
    if (
        payload.get("completed") is not True
        or payload.get("experiment_id") != _EXPERIMENT_ID
        or payload.get("model_level") != "large"
        or payload.get("target_positions") != D0
        or payload.get("tier_scale") != 1
        or payload.get("corpus_hash") != selection.corpus_hash
        or not isinstance(checkpoint_hash, str)
        or len(checkpoint_hash) != 64
        or any(character not in "0123456789abcdef" for character in checkpoint_hash)
    ):
        raise ValueError("large-model D0 evidence does not match a completed O51 D0/U0 run")


def _install_o50_bindings() -> None:
    """Route O50's frozen training/evaluation machinery through O51 types."""
    _o50._EXPERIMENT_ID = _EXPERIMENT_ID
    _o50.Architecture = Architecture
    _o50.ARCHITECTURE = ARCHITECTURE
    _o50.TrainConfig = TrainConfig
    _o50.TemporalBlock = TemporalBlock
    _o50.CausalTemporalDecoder = CausalTemporalDecoder
    _o50.GPT = GPT
    _o50.validate_config = validate_config
    _o50.validate_production_config = validate_production_config
    _o50.make_optimizer = make_optimizer
    _o50._button_adam_parameters = _button_adam_parameters
    _o50.microbatch_loss = microbatch_loss
    _o50.subsystem_parameter_counts = subsystem_parameter_counts
    _o50.model_tag = model_tag
    _o50.source_mixture_weights = source_mixture_weights
    _o50._init_wandb = _init_wandb
    _o50._log_training_summary = _log_training_summary
    _o50.load_stats = load_stats
    _o50._make_loaders = _make_loaders
    _o50._training_functions = _training_functions
    _o50._training_diagnostics = _training_diagnostics
    _o50.layer_activation_rms_log = layer_activation_rms_log
    _o50._minimal_system_metrics = _minimal_system_metrics
    _o50._finalize_training = _finalize_training
    _o50._checkpoint_config = _checkpoint_config
    _o50.config_from_state = config_from_state
    _o50._ARCHITECTURE_FIELDS = _ARCHITECTURE_FIELDS
    _o50._AWR_FIELDS = _AWR_FIELDS
    _o50._RUNTIME_CONFIG_FIELDS = _RUNTIME_CONFIG_FIELDS


_install_o50_bindings()


@dataclass
class TrainArgs:
    cfg: TrainConfig = dataclass_field(default_factory=TrainConfig)
    level: Literal["base", "proxy", "mid", "large"] | None = None
    comment: str = ""
    resume: str | None = None
    resume_checkpoint: str = "latest.pt"
    resume_as: str | None = None
    resume_num_workers: int | None = None
    resume_predownload: int | None = None
    smoke: bool = False
    stop_after_update: int | None = None
    smoke_eval_matchups: int = 4
    eval_max_parallel: int | None = None
    preflight_report: Path | None = None
    promotion_evidence: Path | None = None
    large_d0_evidence: Path | None = None


@dataclass
class EvalArgs(_o50.EvalArgs):
    pass


@dataclass
class BenchmarkArgs:
    level: Literal["base", "proxy", "mid", "large"] = "mid"
    batch_size: Literal[128, 256, 512, 1024] = 512
    depth_alpha: Literal[0.5, 1.0] = 0.5
    compile_mode: Literal["reduce-overhead", "max-autotune"] = "reduce-overhead"
    temporal_attention_chunk: int | None = 16_384
    warmup_steps: int = 3
    measured_steps: int = 20


@dataclass
class LoaderBenchmarkArgs:
    level: Literal["base", "proxy", "mid"] = "mid"
    tier_scale: Literal[1, 2, 4, 8] = 8
    batch_size: Literal[128, 256, 512, 1024] = 512
    num_workers: Literal[8, 16, 24, 32] = 16
    replay_pack_batch_size: Literal[16, 32, 64] = 64
    loader_prefetch_factor: Literal[1, 2, 4] = 2
    predownload: int = 1024
    shuffle_algo: Literal["py1s", "py1e"] = "py1s"
    shuffle_block_size: Literal[4096, 8192] = 8192
    warmup_batches: int = 8
    measured_batches: int = 50


@dataclass
class CoordinateChecksArgs:
    batch_size: Literal[128, 256, 512, 1024] = 512


@dataclass
class DescribeArgs:
    pass


@dataclass
class AuditDataArgs:
    cfg: TrainConfig = dataclass_field(default_factory=TrainConfig)


@dataclass
class PreflightArgs:
    report: Path
    cfg: TrainConfig = dataclass_field(default_factory=TrainConfig)


@dataclass
class CollectSoakArgs:
    run: str
    """Full W&B run path: entity/project/run_id."""

    preflight_report: Path
    output: Path
    cfg: TrainConfig = dataclass_field(default_factory=TrainConfig)
    judged_seconds: float = 1800.0


type Command = (
    Annotated[TrainArgs, tyro.conf.subcommand(name="train")]
    | Annotated[EvalArgs, tyro.conf.subcommand(name="eval")]
    | Annotated[BenchmarkArgs, tyro.conf.subcommand(name="benchmark")]
    | Annotated[LoaderBenchmarkArgs, tyro.conf.subcommand(name="loader-benchmark")]
    | Annotated[CoordinateChecksArgs, tyro.conf.subcommand(name="coordinate-checks")]
    | Annotated[DescribeArgs, tyro.conf.subcommand(name="describe")]
    | Annotated[AuditDataArgs, tyro.conf.subcommand(name="audit-data")]
    | Annotated[PreflightArgs, tyro.conf.subcommand(name="preflight")]
    | Annotated[CollectSoakArgs, tyro.conf.subcommand(name="collect-soak")]
)


def _require_launch_evidence(cfg: TrainConfig, args: TrainArgs) -> None:
    selection = data_selection(cfg)
    if cfg.target_positions >= _LONG_RUN_POSITIONS:
        if args.preflight_report is None:
            raise ValueError("runs at D0 or longer require a throughput preflight report")
        failures = preflight_failures(cfg, load_preflight(args.preflight_report))
        if failures:
            raise ValueError("throughput preflight failed: " + "; ".join(failures))
    if model_level(cfg.arch) == "large":
        if args.promotion_evidence is None:
            raise ValueError("216M training requires the three-seed 55M promotion evidence")
        validate_large_promotion(PromotionEvidence(**json.loads(args.promotion_evidence.read_text())))
        if (cfg.target_positions, cfg.tier_scale) not in ((D0, 1), (8 * D0, 8)):
            raise ValueError("216M trains first at D0/U0 and may then run only a fresh 8D0/U8 endpoint")
        if cfg.target_positions == 8 * D0:
            if args.large_d0_evidence is None:
                raise ValueError("216M 8D0/U8 requires evidence from its completed D0/U0 run")
            _validate_large_d0_evidence(args.large_d0_evidence, selection)


def _run_train(args: TrainArgs) -> None:
    resume_run = None
    resume_state = None
    cfg = args.cfg
    if args.resume is None:
        cfg = replace(cfg, arch=MODEL_FAMILY[args.level or model_level(cfg.arch)])
    if args.resume is None and (args.resume_checkpoint != "latest.pt" or args.resume_as is not None):
        raise SystemExit("--resume-checkpoint and --resume-as require --resume")
    if args.resume is None and (args.resume_num_workers is not None or args.resume_predownload is not None):
        raise SystemExit("--resume-num-workers and --resume-predownload require --resume")
    if args.resume is not None:
        checkpoint = Path(args.resume_checkpoint)
        if checkpoint.is_absolute() or ".." in checkpoint.parts or checkpoint.suffix != ".pt":
            raise SystemExit("--resume-checkpoint must be a relative .pt object within the run")
        resume_state = _o50.load_for_resume(
            args.resume,
            Path("runs") / args.resume,
            device=DEVICE,
            name=args.resume_checkpoint,
        )
        if resume_state is None:
            raise SystemExit(f"no {args.resume_checkpoint!r} for run {args.resume!r}")
        resume_run = args.resume_as or args.resume
        cfg = config_from_state(resume_state["cfg"])
        if args.level is not None and model_level(cfg.arch) != args.level:
            raise SystemExit(f"--level {args.level} does not match resumed {model_level(cfg.arch)} architecture")
        cfg = replace(
            cfg,
            num_workers=cfg.num_workers if args.resume_num_workers is None else args.resume_num_workers,
            predownload=cfg.predownload if args.resume_predownload is None else args.resume_predownload,
        )
        if args.resume_as is not None:
            if Path(args.resume_as).name != args.resume_as or args.resume_as in ("", ".", "..", args.resume):
                raise SystemExit("--resume-as must be one new run-name component")
            destination_exists = (Path("runs") / args.resume_as).exists()
            if cfg.push_to_r2:
                destination_exists = destination_exists or _o50._remote_run_exists(args.resume_as)
            if destination_exists:
                raise SystemExit(f"resume destination {args.resume_as!r} already exists")
            resume_state = {**resume_state, "wandb_id": None}
    if args.eval_max_parallel is not None:
        cfg = replace(cfg, eval_max_parallel=args.eval_max_parallel)
    validate_config(cfg)
    if not args.smoke:
        _require_launch_evidence(cfg, args)
    stats = load_stats(cfg)
    original_log = _o50.wandb.log
    original_host_metrics_sampler = _o50.HostMetricsSampler
    cache_roots = tuple(streams.BY_NAME[name].local_root for name in cfg.source_names)

    def o51_host_metrics_sampler(
        _ignored_roots: tuple[Path, ...],
        **kwargs: object,
    ) -> object:
        return original_host_metrics_sampler(cache_roots, **kwargs)

    _o50.HostMetricsSampler = o51_host_metrics_sampler
    original_o50_file = _o50.__file__
    _o50.__file__ = __file__
    try:
        _o50.train(
            cfg,
            stats,
            comment=args.comment,
            resume_run=resume_run,
            resume_state=resume_state,
            smoke=args.smoke,
            stop_after_update=args.stop_after_update,
            smoke_eval_matchups=args.smoke_eval_matchups,
        )
    finally:
        _o50.__file__ = original_o50_file
        _o50.wandb.log = original_log
        _o50.HostMetricsSampler = original_host_metrics_sampler


def main(args: Command) -> None:
    if isinstance(args, DescribeArgs):
        print(json.dumps(describe(), indent=2, sort_keys=True))
        return
    if isinstance(args, AuditDataArgs):
        validate_config(args.cfg)
        selection = data_selection(args.cfg)
        print(json.dumps(asdict(selection), indent=2, sort_keys=True))
        return
    if isinstance(args, PreflightArgs):
        validate_config(args.cfg)
        failures = preflight_failures(args.cfg, load_preflight(args.report))
        if failures:
            raise SystemExit("preflight failed: " + "; ".join(failures))
        print("preflight passed")
        return
    if isinstance(args, CollectSoakArgs):
        validate_config(args.cfg)
        if args.run.count("/") != 2:
            raise SystemExit("--run must be entity/project/run_id")
        run = _o50.wandb.Api().run(args.run)
        expected = asdict(args.cfg)
        bound_fields = (
            "arch",
            "awr",
            "target_positions",
            "tier_scale",
            "batch_size",
            "player_sidecar_sha256",
            "player_vocab_sha256",
            "compile_mode",
            "temporal_attention_chunk",
            "shuffle_algo",
            "shuffle_block_size",
            "num_workers",
            "cache_limit_gb",
            "reservoir_capacity",
            "replay_cooldown_batches",
            "windows_per_replay",
            "replay_pack_batch_size",
            "loader_prefetch_factor",
            "predownload",
            "download_retry",
        )
        mismatched = [
            name
            for name in bound_fields
            if json.dumps(run.config.get(name), sort_keys=True) != json.dumps(expected[name], sort_keys=True)
        ]
        if mismatched:
            raise SystemExit(f"W&B run does not match the selected soak config: {mismatched}")
        report = collect_soak_report(
            args.cfg,
            load_preflight(args.preflight_report),
            run.scan_history(page_size=1000),
            judged_seconds=args.judged_seconds,
        )
        temporary = args.output.with_name(f".{args.output.name}.tmp")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n")
        temporary.replace(args.output)
        failures = soak_failures(args.cfg, report)
        if failures:
            raise SystemExit("soak failed: " + "; ".join(failures))
        print(f"soak passed; report written to {args.output}")
        return
    if isinstance(args, BenchmarkArgs):
        cfg = config_for(
            args.level,
            batch_size=args.batch_size,
            depth_alpha=args.depth_alpha,
            compile_mode=args.compile_mode,
            temporal_attention_chunk=args.temporal_attention_chunk,
            push_to_r2=False,
            wandb_log_code=False,
        )
        benchmark_train_step(cfg, warmup_steps=args.warmup_steps, measured_steps=args.measured_steps)
        return
    if isinstance(args, LoaderBenchmarkArgs):
        cfg = config_for(
            args.level,
            target_positions=args.tier_scale * D0,
            tier_scale=args.tier_scale,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            replay_pack_batch_size=args.replay_pack_batch_size,
            loader_prefetch_factor=args.loader_prefetch_factor,
            predownload=args.predownload,
            shuffle_algo=args.shuffle_algo,
            shuffle_block_size=args.shuffle_block_size,
            push_to_r2=False,
            wandb_log_code=False,
        )
        benchmark_loader(
            cfg,
            warmup_batches=args.warmup_batches,
            measured_batches=args.measured_batches,
        )
        return
    if isinstance(args, CoordinateChecksArgs):
        reports = run_coordinate_checks(batch_size=args.batch_size)
        print(json.dumps(reports, indent=2, sort_keys=True))
        return
    if isinstance(args, EvalArgs):
        checkpoint = _o50._resolve_eval_checkpoint(args.checkpoint, args.run)
        _o50.eval_checkpoint(
            str(checkpoint),
            n_matchups=args.n_matchups,
            eager=args.eager,
            max_parallel=args.max_parallel,
            output_name=args.output_name,
            upload_run=args.run,
            backfill_wandb=args.backfill_wandb,
            companion_wandb=args.companion_wandb,
        )
        return
    _run_train(args)


def __getattr__(name: str) -> object:
    """Expose unchanged O50 task/evaluation helpers for focused tests."""
    return getattr(_o50, name)


if __name__ == "__main__":
    main(tyro.cli(cast(type[Command], Command)))
