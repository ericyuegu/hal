"""Experiment 051: Muon parameterization over nested replay data.

This standalone program owns the O51 model, optimizer, data tiers, training,
evaluation, and sweep policy. Its persisted experiment identity remains
``051_correct_parameterization_v5`` so existing checkpoints remain valid.
"""

from __future__ import annotations

import contextlib
import fcntl
import functools
import hashlib
import itertools
import json
import math
import os
import re
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Mapping
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from contextlib import contextmanager
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from dataclasses import fields
from dataclasses import replace
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Annotated
from typing import Any
from typing import ClassVar
from typing import Final
from typing import Literal
from typing import TypedDict
from typing import cast

import melee
import modal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from beartype import beartype
from jaxtyping import Bool
from jaxtyping import Float
from jaxtyping import Int
from jaxtyping import jaxtyped
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR

import wandb
from hal import r2
from hal import streams
from hal.data.feature_stats import FeatureStats
from hal.data.policy_world_schema import POLICY_WORLD_SCHEMA_VERSION
from hal.eval.cross_stage import BOOTSTRAP_RESAMPLES
from hal.eval.cross_stage import PRIOR_SWEEP_SEED_STAGE
from hal.eval.cross_stage import MatchRow
from hal.eval.cross_stage import sweep_vs_cpu_prior_with_rows
from hal.eval.cross_stage import vs_cpu_metrics
from hal.eval.harness import DEFAULT_START_RETRIES
from hal.eval.harness import automatic_parallelism
from hal.eval.harness import default_session_cfg
from hal.eval.harness import resolve_parallelism
from hal.eval.harness import usable_cpus
from hal.eval.matchups import matchups_for_vs_cpu
from hal.eval.policy_sampling import SlotGroupRng
from hal.eval.policy_sampling import sample_categorical
from hal.eval.self_play import DecodeTelemetry
from hal.eval.self_play import canonical_context
from hal.eval.self_play import synthetic_context as build_synthetic_context
from hal.sim.inputs import ControllerInputs
from hal.sim.inputs import action_vec_to_controller
from hal.sim.rollout import PolicyRuntimeSpec
from hal.sim.rollout import covering_power_of_two
from hal.sim.vec import Slot
from hal.training import returns as returns_lib
from hal.training.checkpoints import BackgroundUploader
from hal.training.checkpoints import download_latest
from hal.training.checkpoints import load_for_resume
from hal.training.checkpoints import save_checkpoint
from hal.training.closed_loop import RecedingHorizon
from hal.training.controller_codec import BUTTONS_GROUP
from hal.training.controller_codec import C_STICK_GROUP
from hal.training.controller_codec import CONTROLLER_DECODE_ORDER
from hal.training.controller_codec import CONTROLLER_GROUP_COUNT
from hal.training.controller_codec import CONTROLLER_GROUP_INDEX
from hal.training.controller_codec import CONTROLLER_GROUP_NAMES
from hal.training.controller_codec import CONTROLLER_GROUP_VOCABS
from hal.training.controller_codec import MAIN_STICK_GROUP
from hal.training.controller_codec import TRIGGERS_GROUP
from hal.training.controller_codec import DiscreteControllerCodec
from hal.training.dataloader import collate_train_batch
from hal.training.dataloader import make_loader
from hal.training.ego_stats import load_consolidated_mixture_stats
from hal.training.features import ACTION_CHANNELS
from hal.training.features import BASE_ITEMS_PROJECTION
from hal.training.features import BASE_PLAYER_PREFIXES
from hal.training.features import CAT_FEATURES
from hal.training.features import FLOAT_FEATURES
from hal.training.features import ITEM_COLUMNS
from hal.training.features import NEUTRAL_ACTION
from hal.training.features import AWRBatch
from hal.training.features import Context
from hal.training.features import ExtraColumns
from hal.training.features import FeatureProjection
from hal.training.features import TrainBatch
from hal.training.features import stack_actions
from hal.training.mfu import bf16_dense_peak_flops
from hal.training.mfu import bf16_peak_source
from hal.training.mfu import model_flops_utilization
from hal.training.muon import SingleDeviceMuonWithAuxAdam
from hal.training.physical_shard_loader import PREFETCH_FACTOR
from hal.training.physical_shard_loader import MDSStorageAdapter
from hal.training.physical_shard_loader import PhysicalShardReplayLoader
from hal.training.physical_shard_loader import PhysicalShardSelection
from hal.training.physical_shard_loader import SourceRowSelection
from hal.training.physical_shard_loader import build_shard_plan
from hal.training.physical_shard_loader import estimate_host_memory
from hal.training.player_identity import MASKED_PLAYER_ID
from hal.training.player_identity import PlayerIdentitySidecar
from hal.training.player_identity import PlayerVocabulary
from hal.training.player_identity import ReplayPlayerLookup
from hal.training.player_identity import decode_player_codes
from hal.training.player_identity import load_player_identity_sidecar
from hal.training.player_identity import vocabulary_buffer
from hal.training.runs import make_run_name
from hal.training.runs import setup_run_dir
from hal.training.system_metrics import HostMetricsSampler
from hal.training.trunk import Rotary
from hal.training.trunk import Trunk
from hal.training.trunk import TrunkConfig
from hal.training.trunk import apply_rotary_emb
from hal.wire import ACTION_DIM
from hal.wire import ITEM_SLOTS
from hal.wire import item_column

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_LN2 = math.log(2.0)
_N_CONT = 6
_PLAYER_PREFIXES = BASE_PLAYER_PREFIXES
_EXPERIMENT_ID: Final[str] = "051_correct_parameterization_v5"
_RETURN_SUFFIX = "awr_return"
EGO_RETURN = f"ego_{_RETURN_SUFFIX}"
EGO_RETURN_VALID = f"{EGO_RETURN}_valid"
_INFERENCE_BUCKETS = (1, 2, 4, 8, 16, 32, 64)
_PRODUCTION_UPDATES = 2**17
_PRODUCTION_EVAL_MATCHUPS = 96
_N_NEAR = 6
EVAL_HORIZONS = (4,)
DIRECT_LOSS_START = 128
PREDICTION_FRAMES = 4
DELAY_FRAMES = 2
REPLAN_INTERVAL_FRAMES = 2
AWR_START_UPDATE = 4097
PLAYER_SIDECAR_LOCAL = "data/processed/player-identity-v1/professional-code-v1.jsonl.gz"
PLAYER_SIDECAR_REMOTE = "s3://hal/processed/player-identity-v1/professional-code-v1.jsonl.gz"
PLAYER_SIDECAR_SHA256 = "54ccf8a2497fe240313117297ca2ea31158e08db2cc53c67e7aa46853a8dac1c"
PLAYER_VOCAB_SHA256 = "c67c97c995ad033ea7f5b2223efce5b061394566439f091ff6e7aaa6a9d1cfd6"
PLAYER_VOCAB_SIZE = 21_181
PLAYER_EMBED_DIM = 32
TRAIN_REPLAYS = 1_300_640
DATA_PROTOCOL: Final[str] = "o51-dense-shard-replay-v2"
D0: Final[int] = 2**30
_TRUNK_BASE_LAYERS: Final[int] = 8
_TEMPORAL_BASE_LAYERS: Final[int] = 2
_TRUNK_BASE_ATTENTION_SCALE: Final[float] = 0.25
_TEMPORAL_BASE_ATTENTION_SCALE: Final[float] = 0.5
_BASE_MLP_SCALE: Final[float] = 1.0
_BASE_BATCH: Final[int] = 512
_BASE_ADAM_BETAS: Final[tuple[float, float]] = (0.9, 0.95)
_BASE_ADAM_EPS: Final[float] = 1e-12
_SUPERVISED_POSITIONS_PER_WINDOW: Final[int] = 128
_LONG_RUN_POSITIONS: Final[int] = D0
_MIN_SYNTHETIC_MFU: Final[float] = 0.15
_MIN_FULL_TIER_MFU: Final[float] = 0.135
MODEL_LEVELS: Final[tuple[str, ...]] = ("base", "proxy", "mid", "large")
EXPECTED_PARAMETER_COUNTS: Final[dict[str, int]] = {
    "base": 7_861_786,
    "proxy": 14_480_922,
    "mid": 55_015_322,
    "large": 216_496_794,
}
_TRAIN_METRICS_EVERY = 25
_TRAIN_PREFETCH_FACTOR = 4
_TRAIN_COMPILE_MODE = "reduce-overhead"
_TRUNK_ATTENTION_BACKEND = "varlen_flash"
_ACTIVATION_PERCENTILE_SAMPLE_SIZE = 65_536
_ARCHITECTURE_POWER_ITERATIONS = 8
_PRODUCTION_ABLATION_FIELDS = frozenset({"lr_schedule_kind"})
_PRODUCTION_OVERRIDE_FIELDS = frozenset(
    {
        "cache_limit_gb",
        "cache_metrics_interval_s",
        "architecture_metrics_every",
        "compile_temporal",
        "compile_trunk",
        "compiled_inference_bucket",
        "eval_max_parallel",
        "gradient_hist_every",
        "layer_rms_batch_size",
        "layer_rms_every",
        "muon_lr",
        "num_workers",
        "phase_timing_every",
        "predownload",
        "process_metrics_interval_s",
        "push_to_r2",
        "system_metrics_every",
        "system_metrics_interval_s",
        "wandb_log_code",
        "weight_hist_every",
    }
)

GROUP_NAMES = CONTROLLER_GROUP_NAMES
GROUP_VOCABS = CONTROLLER_GROUP_VOCABS
N_GROUPS = CONTROLLER_GROUP_COUNT
BUTTONS_G = BUTTONS_GROUP
MAIN_G = MAIN_STICK_GROUP
C_G = C_STICK_GROUP
TRIG_G = TRIGGERS_GROUP
GROUP_INDEX = CONTROLLER_GROUP_INDEX
GROUP_ORDER = CONTROLLER_DECODE_ORDER

TRIGGER_L_CH = ACTION_CHANNELS.index("trigger_l")
TRIGGER_R_CH = ACTION_CHANNELS.index("trigger_r")
BUTTON_L_CH = ACTION_CHANNELS.index("button_l")
BUTTON_R_CH = ACTION_CHANNELS.index("button_r")

# The projectile block, per slot: four normalized floats, their validity sidecars, and
# two categorical ids. Table sizes come from the routing declaration; the embedding
# widths live in the frozen Architecture. The two ids get one table each, so a
# third routed categorical must fail here rather than pass unread into the model.
_ITEM_FLOATS = tuple(ITEM_COLUMNS.floats)
_ITEM_CAT_VOCABS = {name: spec[0] for name, spec in ITEM_COLUMNS.cats.items() if spec is not None}
assert set(ITEM_COLUMNS.cats) == {"type", "state"}
# The sidecar that gates a slot: an item with an unusable position tells the model
# nothing about where it is, so the slot pools to zero.
_ITEM_PRESENCE_SUFFIX = "pos_x"
# The column whose absence means the observation cannot carry projectiles at all.
_ITEM_PROBE_COLUMN = item_column(0, _ITEM_PRESENCE_SUFFIX)
# Only the policy-world decoder emits the projectile block. The compact "policy"
# decoder builds its dict from POLICY_MDS_COLUMNS, which carries no item columns.
_POLICY_WORLD_NAMES = frozenset(source.name for source in streams.POLICY_WORLD_V7_SOURCES)
_DEFAULT_SOURCE_NAMES = tuple(source.name for source in streams.POLICY_WORLD_V7_SOURCES)
SOURCE_LIST_SHA256 = hashlib.sha256(json.dumps(_DEFAULT_SOURCE_NAMES, separators=(",", ":")).encode()).hexdigest()
MODEL_COLUMNS = ExtraColumns(
    floats=ITEM_COLUMNS.floats,
    cats={**ITEM_COLUMNS.cats, "player_id": None},
)
MODEL_PROJECTION = replace(
    BASE_ITEMS_PROJECTION,
    columns=BASE_ITEMS_PROJECTION.columns | {"ego_player_id"},
)


def direct_loss_start(cfg: TrainConfig) -> int:
    """Return the midpoint; this is position 128 in the frozen production shape."""
    if cfg.arch.L_ctx % 2:
        raise ValueError("context length must be even for suffix supervision")
    return cfg.arch.L_ctx // 2


@dataclass(frozen=True)
class Architecture:
    d_model: int = 512
    n_layers: int = 16
    n_heads: int = 8
    attn_window: int = 0
    L_ctx: int = 256
    sample_chunk_length: int = 20
    head_offsets: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 9, 12, 16, 20)
    temporal_d_model: int = 256
    temporal_layers: int = 4
    temporal_heads: int = 4
    temporal_ff_dim: int = 768
    group_head_dim: int = 256
    action_embed_dim: int = 32
    offset_embed_dim: int = 16
    action_vocab: int = 1024
    action_state_embed_dim: int = 48
    char_vocab: int = 32
    char_dim: int = 8
    stage_vocab: int = 32
    stage_dim: int = 4
    item_type_dim: int = 16
    item_state_dim: int = 4
    item_hidden_dim: int = 64
    item_dim: int = 32
    value_hidden_dim: int = 256
    "One member of the O51 width/depth family; finite encoders stay fixed."


@dataclass(frozen=True)
class AWRCalibration:
    beta: float = 199.5
    weight_max: float = 3.5
    gamma: float = 0.99618
    stock_value: float = 120.0
    damage_shaping: float = 1.0
    win_reward: float = 50.0
    # Regressing the value error in beta units keeps the critic loss O(1)
    # despite the reward's roughly hundred-point scale.
    value_loss_weight: float = 1.0
    auxiliary_loss_weight: float = 0.5


ARCHITECTURE = Architecture()
AWR_CALIBRATION = AWRCalibration()


@dataclass(frozen=True)
class TrainConfig:
    arch: Annotated[Architecture, tyro.conf.Suppress] = ARCHITECTURE
    awr: Annotated[AWRCalibration, tyro.conf.Suppress] = AWR_CALIBRATION
    prediction_frames: int = PREDICTION_FRAMES
    delay_frames: int = DELAY_FRAMES
    replan_interval_frames: int = REPLAN_INTERVAL_FRAMES
    inference_mode: str = "compiled"
    compiled_inference_bucket: int | None = None
    seed: int = 0
    eval_seed: int = 0
    batch_size: int = 512
    max_steps: int = 16384
    muon_lr: float = 0.028
    muon_weight_decay: float = 0.001
    adam_lr: float = 0.000425
    adam_weight_decay: float = 0.001
    grad_clip: float = 1.0
    warmup_steps: int = 512
    lr_floor_ratio: float = 1 / 170
    lr_schedule_kind: Literal["late-cosine", "cosine"] = "cosine"
    amp_dtype: str = "bfloat16"
    allow_tf32: bool = True
    compile_trunk: bool = True
    compile_temporal: bool = True
    wandb_log_code: bool = True
    gradient_hist_every: int = 0
    weight_hist_every: int = 0
    layer_rms_every: int = 0
    layer_rms_batch_size: int = 8
    architecture_metrics_every: int = 0
    val_every: int = 4096
    val_n_samples: int = 2048
    val_batch_size: int = 128
    ckpt_every: int = 2048
    eval_every: int = 8192
    eval_max_frames: int = 7200
    eval_n_matchups: int = _PRODUCTION_EVAL_MATCHUPS
    final_eval_n_matchups: int = _PRODUCTION_EVAL_MATCHUPS
    eval_max_parallel: int | None = 32
    source_names: tuple[str, ...] = _DEFAULT_SOURCE_NAMES
    mds_schema_version: int = 7
    policy_world_schema_version: int = POLICY_WORLD_SCHEMA_VERSION
    cache_limit_gb: ClassVar[int] = 0
    shuffle_block_size: ClassVar[int] = 0
    predownload: ClassVar[int] = 0
    download_retry: ClassVar[int] = 8
    loader_timeout_s: float = 300.0
    val_split: str = "val"
    num_workers: int = 16
    push_to_r2: bool = True
    system_metrics_every: int = 25
    system_metrics_interval_s: float = 5.0
    process_metrics_interval_s: float = 30.0
    cache_metrics_interval_s: float = 30.0
    phase_timing_every: int = 256
    identity_dropout: float = 0.1
    player_sidecar_local: str = PLAYER_SIDECAR_LOCAL
    player_sidecar_sha256: str = PLAYER_SIDECAR_SHA256
    player_vocab_sha256: str = PLAYER_VOCAB_SHA256
    player_vocab_size: int = PLAYER_VOCAB_SIZE
    target_positions: int = D0
    tier_scale: int = 1
    depth_alpha: float = 0.5
    hidden_std_multiplier: float = 1.0
    readout_init: Literal["zero", "mup-normal"] = "zero"
    muon_duration_scaling: Literal["fixed", "inverse-sqrt"] = "fixed"
    muon_batch_scaling: Literal["fixed", "sqrt"] = "fixed"
    adam_beta1: float = _BASE_ADAM_BETAS[0]
    adam_beta2: float = _BASE_ADAM_BETAS[1]
    adam_eps: float = _BASE_ADAM_EPS
    compile_mode: Literal["reduce-overhead", "max-autotune"] = "reduce-overhead"
    temporal_attention_chunk: int | None = 16384
    stability_every: int = 25

    def __post_init__(self) -> None:
        supervised = self.arch.L_ctx - self.arch.L_ctx // 2
        positions_per_update = self.batch_size * supervised
        updates, remainder = divmod(self.target_positions, positions_per_update)
        if remainder:
            raise ValueError(
                f"target_positions={self.target_positions} is not divisible by batch_size={self.batch_size} x supervised_positions={supervised}"
            )
        warmup_positions, warmup_remainder = divmod(self.target_positions, 32)
        if warmup_remainder:
            raise ValueError("target_positions must be divisible by 32")
        warmup_updates, update_remainder = divmod(warmup_positions, positions_per_update)
        if update_remainder:
            raise ValueError("D/32 warmup does not land on an optimizer boundary")
        object.__setattr__(self, "max_steps", updates)
        object.__setattr__(self, "warmup_steps", warmup_updates)


def synthetic_context(cfg: TrainConfig, batch_size: int, device: torch.device) -> Context:
    """Build the fixed base observation with projectile columns."""
    context = build_synthetic_context(
        cfg,
        batch_size,
        device,
        context_length=cfg.arch.L_ctx,
        observation_bundle="base",
        items=True,
    )
    return Context(
        features={
            **context.features,
            "ego_player_id": torch.zeros(batch_size, cfg.arch.L_ctx, dtype=torch.long, device=device),
        },
        ctx_pad=context.ctx_pad,
        slot_ids=context.slot_ids,
        reset=context.reset,
    )


def synthetic_awr_batch(cfg: TrainConfig, device: torch.device) -> AWRBatch:
    """Build one fully valid production-shaped batch without touching the corpus."""
    context = synthetic_context(cfg, cfg.batch_size, device)
    target = torch.zeros(cfg.batch_size, cfg.arch.sample_chunk_length, ACTION_DIM, device=device)
    returns = torch.zeros(cfg.batch_size, cfg.arch.L_ctx, device=device)
    eligible = torch.ones(cfg.batch_size, cfg.arch.L_ctx, dtype=torch.bool, device=device)
    return AWRBatch(batch=TrainBatch(context=context, target=target), returns=returns, eligible=eligible)


def _eval_parallelism(cfg: TrainConfig, n_matchups: int) -> int:
    return resolve_parallelism(n_matchups, cfg.eval_max_parallel)


def _eval_inference_bucket(cfg: TrainConfig, n_matchups: int) -> int:
    rows = _eval_parallelism(cfg, n_matchups)
    override = cfg.compiled_inference_bucket
    if override is not None:
        if rows > override:
            raise ValueError(
                f"evaluation needs {rows} rows, but compiled_inference_bucket is only {override}; "
                "reduce eval_max_parallel too or increase the inference bucket"
            )
        return override
    return covering_power_of_two(rows)


def _planned_inference_buckets(cfg: TrainConfig) -> tuple[int, ...]:
    matchups = (cfg.eval_n_matchups, cfg.final_eval_n_matchups)
    return tuple(sorted({_eval_inference_bucket(cfg, n) for n in matchups}))


def amp_context(cfg: TrainConfig, device: torch.device | str):
    if cfg.amp_dtype == "bfloat16" and torch.device(device).type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


_CUDA_PHASES: tuple[tuple[str, str, str], ...] = (
    ("h2d", "start", "h2d_end"),
    ("target_prep", "h2d_end", "target_prep_end"),
    ("trunk", "target_prep_end", "trunk_end"),
    ("temporal", "trunk_end", "temporal_end"),
    ("objective", "temporal_end", "objective_end"),
    ("backward", "objective_end", "backward_end"),
    ("grad_norm", "backward_end", "grad_norm_end"),
    ("diagnostics", "grad_norm_end", "diagnostics_end"),
    ("optimizer", "diagnostics_end", "optimizer_end"),
)


class CudaPhaseTimer:
    """Measure named phases on the current CUDA stream with one final sync."""

    def __init__(self) -> None:
        self._events: dict[str, torch.cuda.Event] = {}

    def record(self, name: str) -> None:
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self._events[name] = event

    def metrics(self) -> dict[str, float]:
        """Return seconds for every phase whose boundary events were recorded."""
        metrics: dict[str, float] = {}
        for metric, start, end in _CUDA_PHASES:
            if start in self._events and end in self._events:
                metrics[f"profile/{metric}_s"] = self._events[start].elapsed_time(self._events[end]) / 1000
        return metrics


def decoder_rmsnorm(x: Tensor) -> Tensor:
    return F.rms_norm(x, (x.shape[-1],), eps=1e-6)


def action_rmsnorm(x: Tensor) -> Tensor:
    """Normalize an action-boundary tensor with a safer near-zero Jacobian."""
    return F.rms_norm(x, (x.shape[-1],), eps=1e-5)


class SwiGLU(nn.Module):
    """Gated MLP used by every nonlinear projection in the policy."""

    def __init__(self, d_input: int, d_hidden: int, d_output: int, *, output_bias: bool = False) -> None:
        super().__init__()
        self.up = nn.Linear(d_input, 2 * d_hidden, bias=False)
        self.down = nn.Linear(d_hidden, d_output, bias=output_bias)

    def activations(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return the activated gate, value branch, and their product."""
        gate_projection, value = self.up(x).chunk(2, dim=-1)
        gate = F.silu(gate_projection)
        return gate, value, gate * value

    def forward(self, x: Float[Tensor, "... d_input"]) -> Float[Tensor, "... d_output"]:
        _, _, product = self.activations(x)
        return self.down(product)


class NonlinearActionHead(nn.Module):
    """O26 RMSNorm-SiLU controller readout."""

    def __init__(self, d_model: int, d_hidden: int, vocab: int) -> None:
        super().__init__()
        self.up = nn.Linear(d_model, d_hidden, bias=False)
        self.down = nn.Linear(d_hidden, vocab)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.up(decoder_rmsnorm(x))))

    def forward_with_input(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return logits and the normalized tensor read by the hidden layer."""
        normalized = action_rmsnorm(x)
        return self.down(F.silu(self.up(normalized))), normalized


def _sampled_quantile(tensor: Tensor, percentile: float, *, absolute: bool = False) -> Tensor:
    """Estimate a quantile from a deterministic bounded-size flat sample."""
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"percentile must be in [0, 100], got {percentile}")
    values = tensor.detach().flatten()
    stride = max((values.numel() + _ACTIVATION_PERCENTILE_SAMPLE_SIZE - 1) // _ACTIVATION_PERCENTILE_SAMPLE_SIZE, 1)
    if stride > 1:
        stride += 1
    sample = values[::stride].float()
    if absolute:
        sample = sample.abs()
    rank = min(max(math.ceil(percentile * sample.numel() / 100.0), 1), sample.numel())
    return torch.kthvalue(sample, rank).values


def short_causal_attention(
    query: Float[Tensor, "B H L D"],
    key: Float[Tensor, "B H L D"],
    value: Float[Tensor, "B H L D"],
) -> Float[Tensor, "B H L D"]:
    """Use explicit causal attention for the 11-token training sequence.

    On a B200, cuDNN flash SDPA used 149 ms per step for forward and
    backward. At this length, its launch and layout costs were more than the
    cost to materialize 121 scores per head. This implementation reduced the
    full train step by approximately 100 ms.
    """
    scores = query @ key.transpose(-2, -1)
    scores = scores.float() * (query.shape[-1] ** -0.5)
    causal = torch.ones(scores.shape[-2:], dtype=torch.bool, device=scores.device).tril()
    weights = F.softmax(scores.masked_fill(~causal, -torch.inf), dim=-1).to(query.dtype)
    return weights @ value


TEMPORAL_ATTENTION_BATCH = 16_384


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
        self.rotary = Rotary(self.head_dim)
        self.up = nn.Linear(self.d_model, cfg.arch.temporal_ff_dim, bias=False)
        self.down = nn.Linear(cfg.arch.temporal_ff_dim, self.d_model, bias=False)

    def _qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, length, _ = x.shape
        q, k, v = self.qkv(decoder_rmsnorm(x)).split(self.d_model, dim=-1)
        shape = (batch, length, self.n_heads, self.head_dim)
        return (q.view(shape), k.view(shape), v.view(shape))

    def forward(self, x: Tensor) -> Tensor:
        q, k, v = self._qkv(x)
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin).transpose(1, 2)
        k = apply_rotary_emb(k, cos, sin).transpose(1, 2)
        v = v.transpose(1, 2)
        if self.attention_chunk is None:
            attended = short_causal_attention(q, k, v)
        else:
            attended = torch.cat(
                [
                    short_causal_attention(query, key, values)
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
        return x + self.mlp_scale * self.down(F.silu(self.up(decoder_rmsnorm(x))))

    def forward_step(self, x: Tensor, past: tuple[Tensor, Tensor] | None) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        q, k, v = self._qkv(x[:, None])
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if past is not None:
            k = torch.cat((past[0], k), dim=2)
            v = torch.cat((past[1], v), dim=2)
        cos, sin = self.rotary.at(k.shape[2], x.device)
        q = apply_rotary_emb(q, cos[:, -1:], sin[:, -1:]).transpose(1, 2)
        rotated_k = apply_rotary_emb(k.transpose(1, 2), cos, sin).transpose(1, 2)
        attended = F.scaled_dot_product_attention(q, rotated_k, v)
        attended = attended.transpose(1, 2).contiguous().view_as(x)
        x = x + self.scale * self.proj(attended)
        x = x + self.mlp_scale * self.down(F.silu(self.up(decoder_rmsnorm(x))))
        return (x, (k, v))


class CausalTemporalDecoder(nn.Module):
    """O50's factorized controller with centered combined group logits."""

    def __init__(self, cfg: TrainConfig, codec: DiscreteControllerCodec) -> None:
        super().__init__()
        self.codec = codec
        self.head_offsets = tuple(cfg.arch.head_offsets)
        self.d_model = cfg.arch.temporal_d_model
        controller_width = N_GROUPS * cfg.arch.action_embed_dim
        self.offset_embedding = nn.Embedding(cfg.arch.sample_chunk_length + 1, cfg.arch.offset_embed_dim)
        self.token_projection = nn.Linear(
            cfg.arch.d_model + controller_width + cfg.arch.offset_embed_dim, self.d_model
        )
        self.blocks = nn.ModuleList([TemporalBlock(cfg) for _ in range(cfg.arch.temporal_layers)])
        self.group_condition = nn.ModuleDict(
            {
                name: nn.Linear(position * cfg.arch.action_embed_dim, 2 * self.d_model)
                for position, name in enumerate(GROUP_ORDER)
                if position
            }
        )
        self.outputs = nn.ModuleDict(
            {
                name: NonlinearActionHead(self.d_model, cfg.arch.group_head_dim, GROUP_VOCABS[GROUP_INDEX[name]])
                for name in GROUP_NAMES
            }
        )
        self.trunk_outputs = nn.ModuleDict(
            {name: nn.Linear(cfg.arch.d_model, GROUP_VOCABS[GROUP_INDEX[name]], bias=False) for name in GROUP_NAMES}
        )
        self.trunk_width = cfg.arch.d_model
        self.controller_width = controller_width

    def _state_bias(self, trunk: Tensor) -> Tensor:
        """The trunk share of the token projection, computed once per position.

        A linear layer over a concatenation decomposes into a sum of per-part
        linears: ``W [h | a | o] + b = W_h h + W_a a + W_o o + b``. The trunk
        part is constant across the chain's steps, so it never needs the
        per-step copy the concatenation implied. The single ``token_projection``
        parameter is kept (same shape, same initialization as the concatenating
        form); only the compute schedule changes.
        """
        weight = self.token_projection.weight
        return F.linear(trunk, weight[:, : self.trunk_width], self.token_projection.bias)

    def _step_features(self, previous: Tensor, offsets: Tensor) -> Tensor:
        """The per-step share of the token projection: previous action and offset."""
        weight = self.token_projection.weight
        action_weight = weight[:, self.trunk_width : self.trunk_width + self.controller_width]
        offset_weight = weight[:, self.trunk_width + self.controller_width :]
        action = F.linear(self.codec.embed_frame(previous), action_weight)
        return action + F.linear(self.offset_embedding(offsets), offset_weight)

    def _decode_step(
        self, previous: Tensor, offset: int, state_bias: Tensor, caches: list[tuple[Tensor, Tensor] | None]
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor] | None]]:
        """Advance the temporal chain by one selected frame offset."""
        offsets = torch.full((previous.shape[0],), offset, device=previous.device, dtype=torch.long)
        state = decoder_rmsnorm(state_bias + self._step_features(previous, offsets))
        next_caches: list[tuple[Tensor, Tensor] | None] = []
        for module, past in zip(self.blocks, caches, strict=True):
            block = cast(TemporalBlock, module)
            state, present = block.forward_step(state, past)
            next_caches.append(present)
        return (decoder_rmsnorm(state), next_caches)

    def teacher_forced_states(self, hidden: Tensor, observed: Tensor, targets: Tensor) -> Tensor:
        expected = (*hidden.shape[:2], len(self.head_offsets), N_GROUPS)
        if observed.shape != (*hidden.shape[:2], N_GROUPS) or targets.shape != expected:
            raise ValueError(
                f"expected observed {(*hidden.shape[:2], N_GROUPS)} and targets {expected}, got {tuple(observed.shape)} and {tuple(targets.shape)}"
            )
        previous = torch.cat((observed[:, :, None], targets[..., :-1, :]), dim=2)
        trunk = decoder_rmsnorm(hidden)
        offsets = torch.tensor(self.head_offsets, device=hidden.device)
        x = self._state_bias(trunk)[:, :, None] + self._step_features(previous, offsets)
        x = decoder_rmsnorm(x)
        x = x.reshape(hidden.shape[0] * hidden.shape[1], len(self.head_offsets), self.d_model)
        for block in self.blocks:
            x = block(x)
        return decoder_rmsnorm(x.view(*hidden.shape[:2], len(self.head_offsets), self.d_model))

    def group_features(self, states: Tensor, name: str, embedded: dict[str, Tensor]) -> Tensor:
        position = GROUP_ORDER.index(name)
        if position == 0:
            return states
        prefix = torch.cat([embedded[group] for group in GROUP_ORDER[:position]], dim=-1)
        raw_scale, raw_shift = self.group_condition[name](prefix).chunk(2, dim=-1)
        scale = torch.tanh(raw_scale)
        shift = raw_shift
        return states * (1.0 + scale) + shift

    def _teacher_forced_outputs(
        self, hidden: Tensor, observed: Tensor, targets: Tensor
    ) -> tuple[dict[str, Tensor], tuple[Tensor, Tensor, Tensor, Tensor]]:
        raw_logits, button_values = self._raw_teacher_forced_outputs(hidden, observed, targets)
        centered = {name: self._center(logits) for name, logits in raw_logits.items()}
        return (self._mask_buttons(centered, button_values[-1]), button_values)

    def teacher_forced_logits_by_group(self, hidden: Tensor, observed: Tensor, targets: Tensor) -> dict[str, Tensor]:
        logits, _ = self._teacher_forced_outputs(hidden, observed, targets)
        return logits

    @staticmethod
    def nll_from_logits(logits: dict[str, Tensor], targets: Tensor) -> Tensor:
        losses = [
            F.cross_entropy(
                logits[name].float().reshape(-1, GROUP_VOCABS[group]),
                targets[..., group].reshape(-1),
                reduction="none",
            ).view(*targets.shape[:-1])
            for group, name in enumerate(GROUP_NAMES)
        ]
        return torch.stack(losses, dim=-1)

    def teacher_forced_nll(self, hidden: Tensor, observed: Tensor, targets: Tensor) -> Tensor:
        """Compute the policy loss without a softmax-invariant common mode."""
        logits, button_values = self._raw_teacher_forced_outputs(hidden, observed, targets)
        return self.nll_from_logits(self._mask_buttons(logits, button_values[-1]), targets)

    def teacher_forced_nll_with_diagnostics(
        self, hidden: Tensor, observed: Tensor, targets: Tensor
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
            [_sampled_quantile(value.detach(), 99.9, absolute=True) for value in centered_values.values()]
        ).amax()
        metrics = {
            "stability/action_pre_norm_rms": features.detach().float().square().mean().sqrt(),
            "stability/button_pre_norm_rms_min": feature_rms.amin(),
            "stability/button_input_abs_p999": _sampled_quantile(head_input.detach(), 99.9, absolute=True),
            "stability/button_logit_abs_p999": _sampled_quantile(raw_button_logits.detach(), 99.9, absolute=True),
            "stability/uncentered_button_logit_abs_p999": _sampled_quantile(
                raw_button_logits.detach(), 99.9, absolute=True
            ),
            "stability/centered_logit_abs_p999": centered_p999,
            "stability/button_margin_mean": (target_logits - competing).mean(),
        }
        logits = self._mask_buttons(raw_logits, button_mask)
        return (self.nll_from_logits(logits, targets), metrics)

    def teacher_forced_logits(self, hidden: Tensor, observed: Tensor, targets: Tensor) -> list[dict[str, Tensor]]:
        values = self.teacher_forced_logits_by_group(hidden, observed, targets)
        return [
            {name: logits[..., depth, :] for name, logits in values.items()} for depth in range(len(self.head_offsets))
        ]

    def forced_stepwise_logits(self, hidden: Tensor, observed: Tensor, targets: Tensor) -> list[dict[str, Tensor]]:
        if targets.shape != (hidden.shape[0], len(self.head_offsets), N_GROUPS):
            raise ValueError("stepwise targets have the wrong shape")
        raw_trunk = hidden[:, -1]
        state_bias = self._state_bias(decoder_rmsnorm(raw_trunk))
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
        state_bias = self._state_bias(decoder_rmsnorm(raw_trunk))
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
                pick = sample_categorical(logits, argmax=argmax, uniform=uniform, generator=gen)
                picks[name] = pick
                embedded[name] = self.codec.group_embedding(name, pick)
            previous = torch.stack([picks[name] for name in GROUP_NAMES], dim=-1)
            frames.append(previous)
        return torch.stack(frames, dim=1)

    def rollout_conditioned_logits(self, hidden: Tensor, observed: Tensor) -> tuple[list[dict[str, Tensor]], Tensor]:
        raw_trunk = hidden[:, -1]
        state_bias = self._state_bias(decoder_rmsnorm(raw_trunk))
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
        return (all_logits, torch.stack(frames, dim=1))

    @staticmethod
    def _center(logits: Tensor) -> Tensor:
        return center_class_logits(logits)

    def _raw_teacher_forced_outputs(
        self, hidden: Tensor, observed: Tensor, targets: Tensor
    ) -> tuple[dict[str, Tensor], tuple[Tensor, Tensor, Tensor, Tensor]]:
        states = self.teacher_forced_states(hidden, observed, targets)
        embedded = self.codec.embed_groups(targets)
        logits: dict[str, Tensor] = {}
        button_values: tuple[Tensor, Tensor, Tensor] | None = None
        for name in GROUP_NAMES:
            features = self.group_features(states, name, embedded)
            if name == "buttons":
                head = cast(NonlinearActionHead, self.outputs[name])
                combined, head_input = head.forward_with_input(features)
                combined = combined + self.trunk_outputs[name](hidden)[..., None, :]
                button_values = (features, head_input, combined)
            else:
                combined = self.outputs[name](features) + self.trunk_outputs[name](hidden)[..., None, :]
            logits[name] = combined
        if button_values is None:
            raise RuntimeError("button head was not evaluated")
        button_mask = self.codec.button_mask(targets[..., TRIG_G])
        return (logits, (*button_values, button_mask))

    @staticmethod
    def _mask_buttons(logits: dict[str, Tensor], button_mask: Tensor) -> dict[str, Tensor]:
        masked = dict(logits)
        masked["buttons"] = masked["buttons"].masked_fill(button_mask, float("-inf"))
        return masked


class Policy(nn.Module):
    def __init__(self, cfg: TrainConfig, vocabulary: PlayerVocabulary | None = None) -> None:
        nn.Module.__init__(self)
        self.cfg = cfg
        self.L_chunk = cfg.arch.sample_chunk_length
        self.head_offsets = tuple(cfg.arch.head_offsets)
        self.codec = DiscreteControllerCodec(cfg.arch.action_embed_dim)
        self.cat_specs = {**CAT_FEATURES, "action": (cfg.arch.action_vocab, cfg.arch.action_state_embed_dim)}
        self.cat_embeds = nn.ModuleDict(
            {name: nn.Embedding(vocab, dim) for name, (vocab, dim) in self.cat_specs.items()}
        )
        self.char_emb = nn.Embedding(cfg.arch.char_vocab, cfg.arch.char_dim)
        self.stage_emb = nn.Embedding(cfg.arch.stage_vocab, cfg.arch.stage_dim)
        per_player = len(FLOAT_FEATURES) * 2 + sum((dim for _, dim in self.cat_specs.values()))
        d_in = (
            len(_PLAYER_PREFIXES) * per_player
            + N_GROUPS * cfg.arch.action_embed_dim
            + 2 * cfg.arch.char_dim
            + cfg.arch.stage_dim
        )
        self.item_type_emb = nn.Embedding(_ITEM_CAT_VOCABS["type"], cfg.arch.item_type_dim)
        self.item_state_emb = nn.Embedding(_ITEM_CAT_VOCABS["state"], cfg.arch.item_state_dim)
        slot_width = cfg.arch.item_type_dim + cfg.arch.item_state_dim + 2 * len(_ITEM_FLOATS) + 1
        self.item_encoder = SwiGLU(slot_width, cfg.arch.item_hidden_dim, cfg.arch.item_dim)
        d_in += cfg.arch.item_dim
        self.observation_encoder = nn.Linear(d_in, cfg.arch.d_model)
        self.player_embedding = nn.Embedding(cfg.player_vocab_size, PLAYER_EMBED_DIM, padding_idx=MASKED_PLAYER_ID)
        self.player_projection = nn.Linear(PLAYER_EMBED_DIM, cfg.arch.d_model, bias=False)
        code_payload = b"" if vocabulary is None else vocabulary_buffer(vocabulary)
        if vocabulary is not None and (
            vocabulary.size != cfg.player_vocab_size or vocabulary.sha256 != cfg.player_vocab_sha256
        ):
            raise ValueError("identity vocabulary does not match the frozen O51 contract")
        self.register_buffer("player_code_bytes", torch.from_numpy(np.frombuffer(code_payload, dtype=np.uint8).copy()))
        trunk_rule = depth_rule("trunk", cfg.arch.n_layers, cfg.depth_alpha)
        self.trunk = Trunk(
            TrunkConfig(
                d_model=cfg.arch.d_model,
                n_layers=cfg.arch.n_layers,
                n_heads=cfg.arch.n_heads,
                L_ctx=cfg.arch.L_ctx,
                attn_window=cfg.arch.attn_window,
                attention_backend=_TRUNK_ATTENTION_BACKEND,
                attention_scale=trunk_rule.attention,
                mlp_scale=trunk_rule.mlp,
            )
        )
        self.temporal = CausalTemporalDecoder(cfg, self.codec)
        self.value_head = SwiGLU(cfg.arch.d_model, cfg.arch.value_hidden_dim, 1, output_bias=True)
        initialize_o51_parameters(self, cfg)

    def _per_player_features(self, features: dict[str, Tensor], prefix: str) -> Tensor:
        ref = features[f"{prefix}_position_x"]
        batch, length = ref.shape
        values: list[Tensor] = []
        masks: list[Tensor] = []
        for name in FLOAT_FEATURES:
            value = features[f"{prefix}_{name}"]
            mask = features.get(f"{prefix}_{name}_mask", torch.zeros_like(ref))
            values.append(value[..., None])
            masks.append(mask[..., None])
        parts: list[Tensor] = values + masks
        for name, (vocab, _) in self.cat_specs.items():
            parts.append(self.cat_embeds[name](features[f"{prefix}_{name}"].clamp(0, vocab - 1)))
        return torch.cat(parts, dim=-1)

    def _item_features(self, features: dict[str, Tensor]) -> Tensor:
        """Pool the four projectile slots into one permutation-invariant vector.

        One shared encoder reads each slot, the presence flag gates its output, and the
        gated outputs are summed. An empty slot therefore contributes the exact zero
        vector, the pooled value does not depend on WHICH slots the live items occupy,
        and the item count stays implicit in the sum.
        """
        if _ITEM_PROBE_COLUMN not in features:
            raise ValueError(
                f"the observation carries no {_ITEM_PROBE_COLUMN!r} column; training needs policy-world sources and closed-loop evaluation needs projectile routing"
            )
        zeros = torch.zeros_like(features[_ITEM_PROBE_COLUMN])
        slots: list[Tensor] = []
        presence: list[Tensor] = []
        for slot in range(ITEM_SLOTS):
            type_ids = features[item_column(slot, "type")].clamp(0, self.item_type_emb.num_embeddings - 1)
            state_ids = features[item_column(slot, "state")].clamp(0, self.item_state_emb.num_embeddings - 1)
            masks = {name: features.get(f"{item_column(slot, name)}_mask", zeros) for name in _ITEM_FLOATS}
            live = 1.0 - masks[_ITEM_PRESENCE_SUFFIX]
            parts = [self.item_type_emb(type_ids), self.item_state_emb(state_ids)]
            parts += [features[item_column(slot, name)][..., None] for name in _ITEM_FLOATS]
            parts += [masks[name][..., None] for name in _ITEM_FLOATS]
            parts.append(live[..., None])
            slots.append(torch.cat(parts, dim=-1))
            presence.append(live)
        encoded = self.item_encoder(torch.stack(slots, dim=-2))
        return (encoded * torch.stack(presence, dim=-1)[..., None]).sum(dim=-2)

    def context_tokens(self, features: dict[str, Tensor], action_indices: Tensor | None = None) -> Tensor:
        if "opp_player_id" in features:
            raise ValueError("opponent identity must never enter the model")
        if "ego_player_id" not in features:
            raise KeyError("context is missing ego_player_id")
        if action_indices is None:
            action_indices = self.codec.quantize(stack_actions(features))
        parts = [self._per_player_features(features, prefix) for prefix in _PLAYER_PREFIXES]
        parts.append(self.codec.embed_frame(action_indices))
        parts.append(self.char_emb(features["ego_character"].clamp(0, self.char_emb.num_embeddings - 1)))
        parts.append(self.char_emb(features["opp_character"].clamp(0, self.char_emb.num_embeddings - 1)))
        parts.append(self.stage_emb(features["stage"].clamp(0, self.stage_emb.num_embeddings - 1)))
        parts.append(self._item_features(features))
        observation = self.observation_encoder(torch.cat(parts, dim=-1))
        player_ids = features["ego_player_id"].clamp(0, self.player_embedding.num_embeddings - 1)
        return observation + self.player_projection(self.player_embedding(player_ids))

    def forward(self, features: dict[str, Tensor], ctx_pad: Tensor, action_indices: Tensor | None = None) -> Tensor:
        return self.trunk(self.context_tokens(features, action_indices), ctx_pad)

    def forward_dense(
        self, features: dict[str, Tensor], ctx_pad: Tensor, action_indices: Tensor | None = None
    ) -> Tensor:
        """Run the shared model weights through the dense inference trunk."""
        return self.trunk.forward_dense(self.context_tokens(features, action_indices), ctx_pad)

    "The frozen O50 policy with O51 depth, initialization, and logits."

    def forward_unpadded(
        self, features: dict[str, Tensor], _ctx_pad: Tensor, action_indices: Tensor | None = None
    ) -> Tensor:
        """Training forward for the loader's guaranteed full contexts."""
        return self.trunk.forward_unpadded(self.context_tokens(features, action_indices))


class IdentityMasker:
    """Checkpointable, per-window identity dropout independent of all other RNGs."""

    def __init__(self, seed: int, probability: float) -> None:
        self.probability = probability
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.forced = 0
        self.total = 0
        self.masked = 0

    def __call__(self, batch: AWRBatch) -> AWRBatch:
        features = dict(batch.context.features)
        player_id = features["ego_player_id"]
        drop = torch.rand(player_id.shape[0], generator=self.generator) < self.probability
        naturally_masked = player_id[:, 0] == MASKED_PLAYER_ID
        features["ego_player_id"] = player_id.masked_fill(drop[:, None], MASKED_PLAYER_ID)
        self.forced += int(drop.sum())
        self.masked += int((drop | naturally_masked).sum())
        self.total += len(drop)
        train_batch = TrainBatch(
            context=Context(features=features, ctx_pad=batch.context.ctx_pad),
            target=batch.target,
            replay_ids=batch.batch.replay_ids,
        )
        return AWRBatch(train_batch, batch.returns, batch.eligible)

    def state_dict(self) -> dict[str, object]:
        return {
            "generator": self.generator.get_state(),
            "forced": self.forced,
            "masked": self.masked,
            "total": self.total,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        generator_state = state["generator"]
        if not isinstance(generator_state, Tensor):
            raise TypeError("identity-mask generator state must be a tensor")
        self.generator.set_state(generator_state.detach().cpu())
        self.forced = int(cast(int, state["forced"]))
        self.masked = int(cast(int, state["masked"]))
        self.total = int(cast(int, state["total"]))

    def metrics(self) -> dict[str, float]:
        denominator = max(self.total, 1)
        return {
            "data/id_dropout_fraction": self.forced / denominator,
            "data/id_masked_fraction": self.masked / denominator,
        }


@dataclass(slots=True)
class PreparedTrainingData:
    """An optional loader path started before the training process touches CUDA."""

    loader: PhysicalShardReplayLoader[AWRBatch]
    validation: list[TrainBatch]
    iterator: Iterator[AWRBatch]
    first_batch_future: Future[AWRBatch]
    executor: ThreadPoolExecutor
    resources: ExitStack


@jaxtyped(typechecker=beartype)
def prepared_targets(
    model: Policy, batch: TrainBatch | AWRBatch
) -> tuple[
    Int[Tensor, "B L_ctx n_groups"],
    Int[Tensor, "B L_ctx n_offsets n_groups"],
    Bool[Tensor, "B L_ctx"],
]:
    """Quantize history+future exactly once, then align every selected offset."""
    history = stack_actions(batch.context.features)
    full = model.codec.quantize(torch.cat((history, batch.target[:, : model.L_chunk]), dim=1))
    length = history.shape[1]
    targets = torch.stack([full[:, offset : offset + length] for offset in model.head_offsets], dim=2)
    valid = torch.arange(length, device=full.device)[None, :] >= batch.context.ctx_pad[:, None]
    suffix = slice(length // 2, None)
    return full[:, :length][:, suffix], targets[:, suffix], valid[:, suffix]


class DeviceBatchPrefetcher:
    """Fetch the next CPU batch during GPU compute and stage it for transfer."""

    def __init__(
        self,
        loader: Iterable[AWRBatch],
        cfg: TrainConfig,
        device: str | torch.device,
        identity_masker: IdentityMasker | None = None,
        *,
        iterator: Iterator[AWRBatch] | None = None,
        first_batch_future: Future[AWRBatch] | None = None,
    ) -> None:
        self._loader = loader
        self._iterator = iter(loader) if iterator is None else iterator
        self._cfg = cfg
        self._device = torch.device(device)
        self._identity_masker = identity_masker
        self._copy_stream = torch.cuda.Stream(device=self._device) if self._device.type == "cuda" else None
        self._staged: tuple[AWRBatch, AWRBatch, int] | None = None
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="device-batch-prefetch")
        self._future = first_batch_future
        if first_batch_future is None:
            self.preload()
        else:
            self.finish_preload()

    def _load_cpu_batch(self) -> AWRBatch:
        try:
            cpu_batch = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self._loader)
            cpu_batch = next(self._iterator)
        return self._prepare_cpu_batch(cpu_batch)

    def _prepare_cpu_batch(self, cpu_batch: AWRBatch) -> AWRBatch:
        """Apply the parent-side transforms to an already-fetched batch."""
        if not isinstance(cpu_batch, AWRBatch):
            raise TypeError(f"advantage loader yielded {type(cpu_batch).__name__}, expected AWRBatch")
        if self._identity_masker is not None:
            cpu_batch = self._identity_masker(cpu_batch)
        validate_batch_geometry(cpu_batch, self._cfg, self._cfg.batch_size)
        return cpu_batch

    def _stage(self, cpu_batch: AWRBatch) -> None:
        start = direct_loss_start(self._cfg)
        suffix_pad = (cpu_batch.context.ctx_pad - start).clamp_min(0)
        valid_prefixes = int((self._cfg.arch.L_ctx - start - suffix_pad).sum())
        if valid_prefixes <= 0:
            raise RuntimeError("training batch contains no valid context prefixes")
        if self._copy_stream is None:
            device_batch = cpu_batch.to(self._device)
        else:
            with torch.cuda.stream(self._copy_stream):
                device_batch = cpu_batch.to(self._device)
        self._staged = (device_batch, cpu_batch, valid_prefixes)

    def preload(self) -> None:
        """Synchronously stage a batch at startup and checkpoint boundaries."""
        if self._staged is not None:
            raise RuntimeError("consume the staged batch before preloading another")
        if self._future is not None:
            raise RuntimeError("finish the background preload before preloading synchronously")
        cpu_batch = self._load_cpu_batch()
        self._stage(cpu_batch)

    def start_preload(self) -> None:
        """Start loading the next CPU batch on a background thread."""
        if self._staged is not None:
            raise RuntimeError("consume the staged batch before starting another preload")
        if self._future is not None:
            raise RuntimeError("a background preload is already running")
        self._future = self._pool.submit(self._load_cpu_batch)

    def finish_preload(self) -> float:
        """Stage a background-loaded batch and measure only its uncovered wait."""
        if self._future is None:
            raise RuntimeError("no background preload is running")
        started = time.monotonic()
        future, self._future = self._future, None
        cpu_batch = self._prepare_cpu_batch(future.result())
        loader_wait = time.monotonic() - started
        self._stage(cpu_batch)
        return loader_wait

    def next(self) -> tuple[AWRBatch, int]:
        """Wait only for the uncovered tail of the staged transfer."""
        if self._staged is None:
            raise RuntimeError("preload a batch before consuming it")
        device_batch, cpu_batch, valid_prefixes = self._staged
        if self._copy_stream is not None:
            compute_stream = torch.cuda.current_stream(self._device)
            compute_stream.wait_stream(self._copy_stream)
            device_batch.record_stream(compute_stream)
        self._staged = None
        del cpu_batch
        return device_batch, valid_prefixes

    def close(self) -> None:
        """Release the background loader thread."""
        self._pool.shutdown(wait=True, cancel_futures=True)


def collate_awr_batch(windows: list[dict], batch: TrainBatch, *, L_ctx: int) -> AWRBatch:
    """Attach ``G_{t+1}`` and its validity mask to each context position."""
    next_frames = slice(1, L_ctx + 1)
    returns = np.stack([window[EGO_RETURN] for window in windows])[:, next_frames]
    eligible = np.stack([window[EGO_RETURN_VALID] for window in windows])[:, next_frames]
    return AWRBatch(
        batch=batch,
        returns=torch.from_numpy(np.ascontiguousarray(returns)),
        eligible=torch.from_numpy(np.ascontiguousarray(eligible)).bool(),
    )


@jaxtyped(typechecker=beartype)
def advantage_weights(
    advantage: Float[Tensor, "*batch"],
    eligible: Bool[Tensor, "*batch"],
    *,
    beta: float,
    weight_max: float,
    active: bool = True,
    valid: Bool[Tensor, "*batch"] | None = None,
) -> tuple[Float[Tensor, "*batch"], dict[str, Float[Tensor, ""]]]:
    """Compute capped, batch-normalized weights from detached advantages."""
    if valid is None:
        valid = torch.ones_like(eligible)
    eligible = eligible & valid
    eligible_float = eligible.float()
    eligible_count = eligible_float.sum()
    eligible_denominator = eligible_count.clamp_min(1)
    valid_count = valid.float().sum().clamp_min(1)

    safe_advantage = torch.where(eligible, advantage, 0).float()

    max_log_weight = math.log(weight_max)
    log_weights = (safe_advantage / beta).clamp(max=max_log_weight)
    eligible_log_weights = log_weights.masked_fill(~eligible, -torch.inf)
    log_mean_weight = torch.logsumexp(eligible_log_weights.reshape(-1), dim=0) - eligible_count.clamp_min(1).log()
    log_mean_weight = torch.where(eligible_count > 0, log_mean_weight, torch.zeros_like(log_mean_weight))
    normalized_log_weights = torch.where(eligible, log_weights - log_mean_weight, 0)
    normalized_weights = torch.exp(normalized_log_weights)
    active_weights = normalized_weights if active else torch.ones_like(normalized_weights)
    weights = torch.where(eligible, active_weights, torch.ones_like(active_weights))

    has_eligible = eligible_count > 0
    advantage_scale = (safe_advantage.abs() * eligible_float).max().clamp_min(torch.finfo(torch.float32).tiny)
    scaled_advantage = safe_advantage / advantage_scale
    scaled_mean = (scaled_advantage * eligible_float).sum() / eligible_denominator
    scaled_variance = ((scaled_advantage - scaled_mean).square() * eligible_float).sum() / eligible_denominator
    advantage_mean = scaled_mean * advantage_scale
    advantage_std = scaled_variance.sqrt() * advantage_scale
    weight_sum = (active_weights * eligible_float).sum()
    squared_sum = (active_weights.square() * eligible_float).sum()
    raw_ess = weight_sum.square() / (eligible_count * squared_sum).clamp_min(torch.finfo(torch.float32).tiny)
    zero = torch.zeros((), device=advantage.device)
    stats = {
        "advantage_mean": torch.where(has_eligible, advantage_mean, zero),
        "advantage_std": torch.where(has_eligible, advantage_std, zero),
        "weight_ess": torch.where(has_eligible, raw_ess, torch.ones_like(zero)),
        "weight_clip_frac": torch.where(
            has_eligible,
            ((log_weights >= max_log_weight).float() * eligible_float).sum() / eligible_denominator,
            zero,
        ),
        "weight_mean": torch.where(has_eligible, weight_sum / eligible_denominator, torch.ones_like(zero)),
        "weight_max": torch.where(
            has_eligible,
            active_weights.masked_fill(~eligible, 0).max(),
            torch.ones_like(zero),
        ),
        "weight_min": torch.where(
            has_eligible,
            active_weights.masked_fill(~eligible, float("inf")).min(),
            torch.ones_like(zero),
        ),
        "eligible_frac": eligible_count / valid_count,
    }
    return weights, stats


def masked_correlation(x: Tensor, y: Tensor, selected: Tensor) -> Tensor:
    """Return a finite Pearson correlation over selected entries."""
    selected_float = selected.float()
    count = selected_float.sum()
    denominator = count.clamp_min(1)
    x_values = torch.where(selected, x.float(), 0)
    y_values = torch.where(selected, y.float(), 0)
    x_centered = torch.where(selected, x_values - x_values.sum() / denominator, 0)
    y_centered = torch.where(selected, y_values - y_values.sum() / denominator, 0)
    covariance = (x_centered * y_centered).sum()
    scale = (x_centered.square().sum() * y_centered.square().sum()).sqrt()
    correlation = covariance / scale.clamp_min(torch.finfo(torch.float32).tiny)
    return torch.where((count > 1) & (scale > 0), correlation, torch.zeros_like(correlation))


@jaxtyped(typechecker=beartype)
def value_objective(
    value: Float[Tensor, "B L_ctx"],
    return_target: Float[Tensor, "B L_ctx"],
    eligible: Bool[Tensor, "B L_ctx"],
    *,
    beta: float,
    valid: Bool[Tensor, "B L_ctx"],
) -> tuple[Float[Tensor, ""], Float[Tensor, "B L_ctx"], dict[str, Float[Tensor, ""]]]:
    """Fit ``V(s_t)`` to ``G_{t+1}`` and return a detached advantage."""
    selected = eligible & valid
    selected_float = selected.float()
    count = selected_float.sum().clamp_min(1)
    value_float = value.float()
    return_float = return_target.float()
    error = torch.where(selected, (value_float - return_float) / beta, 0)
    value_loss = error.square().sum() / count
    advantage = torch.where(selected, return_float - value_float.detach(), 0)
    stats = {
        "value_loss": value_loss.detach(),
        "value_rmse": value_loss.detach().sqrt() * beta,
        "value_mean": (value_float.detach() * selected_float).sum() / count,
        "return_mean": torch.where(selected, return_float, 0).sum() / count,
    }
    return value_loss, advantage, stats


@jaxtyped(typechecker=beartype)
def temporal_objective_parts(
    nll: Float[Tensor, "*prefix n_offsets n_groups"],
    weight: Float[Tensor, "*prefix"],
    *,
    valid_prefixes: int,
    aux_loss_weight: float,
    valid: Bool[Tensor, "*prefix"],
) -> tuple[Float[Tensor, ""], Float[Tensor, ""], Float[Tensor, ""]]:
    """Return weighted near loss, unweighted far loss, and normalized total."""
    n_offsets = nll.shape[-2]
    joint_nll = nll.float().sum(dim=-1)
    joint_nll = torch.where(valid[..., None], joint_nll, 0)
    weights = weight.float()[..., None]
    near = (joint_nll[..., :_N_NEAR] * weights).sum() / (valid_prefixes * _N_NEAR)
    far = joint_nll[..., _N_NEAR:].sum() / (valid_prefixes * (n_offsets - _N_NEAR))
    total = (near + aux_loss_weight * far) / (1.0 + aux_loss_weight)
    return near, far, total


def nll_mean_metrics(
    mean_nll: Tensor,
    offsets: tuple[int, ...],
    *,
    aux_loss_weight: float = 0.5,
) -> dict[str, float]:
    if mean_nll.shape != (len(offsets), N_GROUPS):
        raise ValueError(f"mean NLL has shape {tuple(mean_nll.shape)}")
    joint = mean_nll.sum(dim=-1) / _LN2
    if len(offsets) <= _N_NEAR:
        raise ValueError(f"the {_N_NEAR} near offsets must leave at least one far offset, got {len(offsets)}")
    near = joint[:_N_NEAR].mean()
    far = joint[_N_NEAR:].mean()
    total = (near + aux_loss_weight * far) / (1.0 + aux_loss_weight)
    out = {
        "loss_unweighted": float(total),
        "temporal_loss_near_unweighted": float(near),
        "temporal_loss_far_unweighted": float(far),
    }
    for depth, offset in enumerate(offsets):
        out[f"nll_o{offset:02d}"] = float(joint[depth])
        for group, name in enumerate(GROUP_NAMES):
            out[f"nll_o{offset:02d}_{name}"] = float(mean_nll[depth, group] / _LN2)
    return out


def _transition_metrics(target: Tensor, prediction: Tensor, observed: Tensor) -> dict[str, float]:
    previous_target = torch.cat((observed[:, None], target[:, :-1]), dim=1)
    previous_prediction = torch.cat((observed[:, None], prediction[:, :-1]), dim=1)
    target_change = target != previous_target
    sampled_change = prediction != previous_prediction
    true_positive = (target_change & sampled_change).sum().float()
    precision = true_positive / sampled_change.sum().clamp_min(1)
    recall = true_positive / target_change.sum().clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {
        "hold_acc": float(((~target_change) & (prediction == target)).sum() / (~target_change).sum().clamp_min(1)),
        "transition_acc": float((target_change & (prediction == target)).sum() / target_change.sum().clamp_min(1)),
        "change_precision": float(precision),
        "change_recall": float(recall),
        "change_f1": float(f1),
        "target_transition_rate": float(target_change.float().mean()),
        "sampled_transition_rate": float(sampled_change.float().mean()),
        "copy_last_acc": float((target == previous_target).float().mean()),
    }


@torch.no_grad()
def val_metrics(model: Policy, batches: list[TrainBatch], cfg: TrainConfig) -> dict[str, float]:
    """Score teacher-forced and rollout policies on the same final-prefix rows."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    nll_sum = torch.zeros(len(model.head_offsets), N_GROUPS, dtype=torch.float64)
    correct = torch.zeros_like(nll_sum)
    count = 0
    rollout_correct = torch.zeros_like(nll_sum)
    rollout_nll = torch.zeros_like(nll_sum)
    teacher_exposure_nll = torch.zeros_like(nll_sum)
    exposure_count = torch.zeros_like(nll_sum)
    button_incompatible = torch.zeros(len(model.head_offsets), dtype=torch.float64)
    target_rows: list[Tensor] = []
    sampled_rows: list[Tensor] = []
    observed_rows: list[Tensor] = []
    quantization_squared = quantization_count = invalid_triggers = 0.0
    try:
        for cpu_batch in batches:
            batch = cpu_batch.to(device)
            history, targets, valid = prepared_targets(model, batch)
            with amp_context(cfg, device):
                hidden = model.forward_dense(batch.context.features, batch.context.ctx_pad, None)
                hidden = hidden[:, direct_loss_start(cfg) :]
                logits = model.temporal.teacher_forced_logits_by_group(hidden, history, targets)
                dense_nll = model.temporal.nll_from_logits(logits, targets)
            row_valid = batch.context.ctx_pad < cfg.arch.L_ctx
            if not bool(row_valid.any()):
                continue
            selected_nll = dense_nll[:, -1][row_valid]
            nll_sum += selected_nll.double().sum(dim=0).cpu()
            count += selected_nll.shape[0]
            target_last = targets[:, -1][row_valid]
            for group, name in enumerate(GROUP_NAMES):
                correct[:, group] += (
                    (logits[name][:, -1][row_valid].argmax(dim=-1) == target_last[..., group])
                    .double()
                    .sum(dim=0)
                    .cpu()
                )

            # Rollout-conditioned diagnostics use the last real context prefix;
            # temporal and within-frame prefixes are sampled greedily.
            last_observed = history[:, -1][row_valid]
            with amp_context(cfg, device):
                rollout_logits, sampled_all = model.temporal.rollout_conditioned_logits(
                    hidden[row_valid], last_observed
                )
            sampled = sampled_all[:, :6]
            target_rows.append(target_last.cpu())
            sampled_rows.append(sampled.cpu())
            observed_rows.append(last_observed.cpu())
            for depth in range(len(model.head_offsets)):
                compatible = model.codec.button_valid_for_trigger[
                    sampled_all[:, depth, TRIG_G], target_last[:, depth, BUTTONS_G]
                ]
                button_incompatible[depth] += float((~compatible).sum())
                for group, name in enumerate(GROUP_NAMES):
                    expected = target_last[:, depth, group]
                    step_logits = rollout_logits[depth][name]
                    rollout_correct[depth, group] += (step_logits.argmax(-1) == expected).double().sum().cpu()
                    selected = compatible if group == BUTTONS_G else torch.ones_like(compatible)
                    selected_count = int(selected.sum())
                    if selected_count:
                        rollout_nll[depth, group] += (
                            F.cross_entropy(step_logits[selected].float(), expected[selected], reduction="sum")
                            .double()
                            .cpu()
                        )
                        teacher_exposure_nll[depth, group] += selected_nll[selected, depth, group].double().sum().cpu()
                        exposure_count[depth, group] += selected_count

            raw = torch.cat((stack_actions(batch.context.features), batch.target[:, : model.L_chunk]), dim=1)
            canonical = model.codec.canonicalize(raw)
            reconstructed = model.codec.dequantize(model.codec.quantize(raw))
            quantization_squared += float((canonical[..., :6] - reconstructed[..., :6]).square().sum())
            quantization_count += canonical[..., :6].numel()
            invalid_triggers += float(
                (
                    ((raw[..., BUTTON_L_CH] > 0.5) & (raw[..., TRIGGER_L_CH] < 1.0))
                    | ((raw[..., BUTTON_R_CH] > 0.5) & (raw[..., TRIGGER_R_CH] < 1.0))
                ).sum()
            )
    finally:
        model.train(was_training)
    if count == 0:
        raise RuntimeError("validation contained no valid prefixes")
    out = nll_mean_metrics(
        nll_sum / count,
        model.head_offsets,
        aux_loss_weight=cfg.awr.auxiliary_loss_weight,
    )
    for depth, offset in enumerate(model.head_offsets):
        for group, name in enumerate(GROUP_NAMES):
            out[f"acc_o{offset:02d}_{name}"] = float(correct[depth, group] / count)
            denominator = float(exposure_count[depth, group])
            if denominator <= 0:
                raise RuntimeError(f"validation has no compatible rollout rows for offset {offset} group {name}")
            roll_nll = float(rollout_nll[depth, group] / denominator / _LN2)
            teacher_nll = float(teacher_exposure_nll[depth, group] / denominator / _LN2)
            out[f"rollout_nll_o{offset:02d}_{name}"] = roll_nll
            out[f"exposure_gap_o{offset:02d}_{name}"] = roll_nll - teacher_nll
            out[f"rollout_acc_o{offset:02d}_{name}"] = float(rollout_correct[depth, group] / count)
        out[f"rollout_button_target_masked_rate_o{offset:02d}"] = float(button_incompatible[depth] / count)
    target = torch.cat(target_rows)
    sampled = torch.cat(sampled_rows)
    observed = torch.cat(observed_rows)
    dense_target = target[:, :6]
    matches = sampled == dense_target
    out["exact_frame_acc"] = float(matches.all(dim=-1).float().mean())
    out["dense_four_sequence_acc"] = float(matches[:, :4].all(dim=-1).all(dim=-1).float().mean())
    out.update(_transition_metrics(dense_target, sampled, observed))
    out["action_quantization_mse"] = quantization_squared / max(quantization_count, 1)
    out["invalid_trigger_count_raw"] = invalid_triggers
    out["invalid_trigger_count_sampled"] = float(
        (~model.codec.button_valid_for_trigger[sampled[..., TRIG_G], sampled[..., BUTTONS_G]]).sum()
    )
    nonfinite = {name: value for name, value in out.items() if not math.isfinite(value)}
    if nonfinite:
        raise FloatingPointError(f"validation produced non-finite metrics: {nonfinite}")
    return out


def _validation_wandb_metrics(values: dict[str, float], cfg: TrainConfig) -> dict[str, float]:
    """Reduce detailed validation evidence to orthogonal W&B signals."""
    horizon = cfg.prediction_frames
    rollout_nll = sum(values[f"rollout_nll_o{horizon:02d}_{name}"] for name in GROUP_NAMES)
    exposure_gap = sum(values[f"exposure_gap_o{horizon:02d}_{name}"] for name in GROUP_NAMES)
    return {
        "nll": values["loss_unweighted"],
        "near_nll": values["temporal_loss_near_unweighted"],
        "far_nll": values["temporal_loss_far_unweighted"],
        "rollout_nll": rollout_nll,
        "exposure_gap": exposure_gap,
        "exact_frame_acc": values["exact_frame_acc"],
        "sequence_acc": values["dense_four_sequence_acc"],
        "change_f1": values["change_f1"],
        "sampled_transition_rate": values["sampled_transition_rate"],
    }


def _pad_context(ctx: Context, bucket: int) -> Context:
    rows = ctx.ctx_pad.shape[0]
    if rows == bucket:
        return ctx
    if rows > bucket:
        raise ValueError("cannot pad a context to a smaller bucket")
    extra = bucket - rows
    features = {
        name: torch.cat((value, torch.zeros((extra, *value.shape[1:]), dtype=value.dtype, device=value.device)))
        for name, value in ctx.features.items()
    }
    ctx_pad = torch.cat(
        (
            ctx.ctx_pad,
            torch.full(
                (extra,),
                ctx.features[next(iter(ctx.features))].shape[1] - 1,
                dtype=ctx.ctx_pad.dtype,
                device=ctx.ctx_pad.device,
            ),
        )
    )
    slot_ids = None
    reset = None
    if ctx.slot_ids is not None:
        slot_ids = torch.cat(
            (ctx.slot_ids, torch.full((extra,), -1, dtype=ctx.slot_ids.dtype, device=ctx.slot_ids.device))
        )
    if ctx.reset is not None:
        reset = torch.cat((ctx.reset, torch.ones(extra, dtype=ctx.reset.dtype, device=ctx.reset.device)))
    return Context(features=features, ctx_pad=ctx_pad, slot_ids=slot_ids, reset=reset)


class BF16Inference:
    """Hardware-bucketed compiled trunk and unrolled dense-prefix decoders.

    Evaluation compiles each required program synchronously on first use. Runtime
    calls use the smallest compiled bucket that fits. Padding and slot-keyed random
    streams leave real rows unchanged.
    """

    def __init__(
        self,
        model: Policy,
        cfg: TrainConfig,
        *,
        bucket: int | None = None,
        compiled: bool | None = None,
        compile_mode: str = "default",
        compiled_buckets: tuple[int, ...] | None = None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        if bucket is not None and compiled_buckets is not None:
            raise ValueError("pass bucket or compiled_buckets, not both")
        chosen = (bucket,) if bucket is not None else compiled_buckets
        chosen = _planned_inference_buckets(cfg) if chosen is None else chosen
        self.compiled_buckets = tuple(sorted(set(chosen)))
        if not self.compiled_buckets:
            raise ValueError("compiled_buckets must contain at least one bucket")
        if any(bucket < 1 or bucket & (bucket - 1) for bucket in self.compiled_buckets):
            raise ValueError(f"compiled_buckets must be positive powers of two, got {self.compiled_buckets}")
        requested = cfg.inference_mode == "compiled" if compiled is None else compiled
        self.compiled = bool(requested and next(model.parameters()).device.type == "cuda")
        self.compile_mode = compile_mode
        self.attention_backend = "dense_sdpa"
        self.compile_seconds = 0.0
        self._warmed: set[tuple[int, int]] = set()
        self._trunks: dict[int, Callable] = {}
        self._decoders: dict[tuple[int, int], Callable] = {}

    @property
    def uses_cuda_graphs(self) -> bool:
        return self.compiled and self.compile_mode == "reduce-overhead"

    def _bucket(self, rows: int) -> int:
        if self.compiled:
            try:
                return next(bucket for bucket in self.compiled_buckets if bucket >= rows)
            except StopIteration as exc:
                raise ValueError(
                    f"inference batch {rows} exceeds largest compiled bucket {self.compiled_buckets[-1]}"
                ) from exc
        try:
            return next(bucket for bucket in _INFERENCE_BUCKETS if bucket >= rows)
        except StopIteration:
            return covering_power_of_two(rows)

    def _trunk(self, bucket: int) -> Callable:
        if bucket not in self._trunks:
            forward = self.model.forward_dense
            self._trunks[bucket] = (
                torch.compile(forward, dynamic=False, fullgraph=True, mode=self.compile_mode)
                if self.compiled
                else forward
            )
        return self._trunks[bucket]

    def _decoder(self, bucket: int, horizon: int) -> Callable:
        key = (bucket, horizon)
        if key not in self._decoders:
            offsets = self.model.head_offsets[:horizon]

            def fn(hidden, observed, uniforms):
                return self.model.temporal.sample_indices(hidden, observed, offsets, argmax=False, uniforms=uniforms)

            self._decoders[key] = torch.compile(fn, dynamic=False, mode=self.compile_mode) if self.compiled else fn
        return self._decoders[key]

    @torch.no_grad()
    def prewarm(self, rows: int, horizon: int) -> float:
        """Compile and replay the exact evaluation program before Dolphin starts."""
        bucket = self._bucket(rows)
        key = (bucket, horizon)
        if key in self._warmed or not self.compiled:
            self._warmed.add(key)
            return 0.0
        device = next(self.model.parameters()).device
        started = time.perf_counter()
        context = synthetic_context(self.cfg, rows, device)
        self.decode(context, horizon)
        self.decode(context, horizon)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        self.compile_seconds += elapsed
        self._warmed.add(key)
        print(
            f"[inference] synchronously compiled batch {bucket}, horizon {horizon} in {elapsed:.1f}s",
            flush=True,
        )
        return elapsed

    @torch.no_grad()
    def decode(
        self,
        ctx: Context,
        horizon: int,
        *,
        streams: SlotGroupRng | None = None,
        argmax: bool = False,
        gen: torch.Generator | None = None,
    ) -> Tensor:
        if horizon not in EVAL_HORIZONS:
            raise ValueError(f"horizon must be one of {EVAL_HORIZONS}")
        rows = ctx.ctx_pad.shape[0]
        bucket = self._bucket(rows)
        padded = canonical_context(_pad_context(ctx, bucket), "base", items=True)
        padded = Context(
            features={
                **padded.features,
                "ego_player_id": torch.zeros_like(padded.features["stage"]),
            },
            ctx_pad=padded.ctx_pad,
            slot_ids=padded.slot_ids,
            reset=padded.reset,
        )
        observed = self.model.codec.quantize(stack_actions(padded.features))
        uniform_parts: list[Tensor] = []
        if streams is not None:
            streams.begin(ctx)
        for _ in range(horizon):
            groups = []
            for name in GROUP_NAMES:
                if streams is None:
                    real = torch.rand(rows, device=ctx.ctx_pad.device, generator=gen)
                else:
                    real = streams.uniforms(name)
                groups.append(F.pad(real, (0, bucket - rows), value=0.5))
            uniform_parts.append(torch.stack(groups))
        uniforms = torch.stack(uniform_parts)
        if self.uses_cuda_graphs:
            # The trunk and decoder are separate CUDA Graph trees.  Mark one
            # complete decode as a graph step so the next trunk replay may
            # safely reuse its managed output storage after the decoder has
            # consumed it.
            torch.compiler.cudagraph_mark_step_begin()
        with amp_context(self.cfg, ctx.ctx_pad.device):
            hidden = self._trunk(bucket)(padded.features, padded.ctx_pad, observed)
            if argmax:
                indices = self.model.temporal.sample_indices(
                    hidden, observed[:, -1], self.model.head_offsets[:horizon], argmax=True
                )
            else:
                indices = self._decoder(bucket, horizon)(hidden, observed[:, -1], uniforms)
        return self.model.codec.dequantize(indices[:rows])


@dataclass
class DelayedTruncationPolicy(RecedingHorizon):
    """D2/R2 controller: discard +1/+2 and execute +3/+4 two frames later."""

    _queues: dict[Slot, list[np.ndarray]] = dataclass_field(default_factory=dict)
    _phases: dict[Slot, int] = dataclass_field(default_factory=dict)
    neutral_actions: int = 0
    total_actions: int = 0

    @property
    def runtime_spec(self) -> PolicyRuntimeSpec:
        return PolicyRuntimeSpec(
            context_frames=self.L_ctx,
            prediction_frames=PREDICTION_FRAMES,
            execution_stride=REPLAN_INTERVAL_FRAMES,
            committed_frames=DELAY_FRAMES,
            action_dim=len(ACTION_CHANNELS),
        )

    def __call__(
        self,
        frame_index: int,
        obs: Mapping[Slot, dict],
    ) -> Mapping[Slot, ControllerInputs]:
        del frame_index
        live = list(obs)
        self._ingest(live, obs)
        due: list[Slot] = []
        for slot in live:
            state = self._slots[slot]
            if state.reset_pending or slot not in self._queues:
                self._queues[slot] = [NEUTRAL_ACTION.copy() for _ in range(DELAY_FRAMES)]
                self._phases[slot] = 0
            if self._phases[slot] % REPLAN_INTERVAL_FRAMES == 0:
                due.append(slot)
        if due:
            context = self._context(due)
            plans = self.predict_chunk(context, None)
            if plans.shape[:2] != (len(due), PREDICTION_FRAMES):
                raise ValueError("D2/R2 predictor must return four frames per live slot")
            for row, slot in enumerate(due):
                self._queues[slot].extend(plans[row, DELAY_FRAMES:].astype(np.float32))
        actions = {}
        for slot in live:
            action = self._queues[slot].pop(0)
            actions[slot] = action
            self._push_ego(slot, action)
            self._phases[slot] += 1
            self.total_actions += 1
            self.neutral_actions += int(np.array_equal(action, NEUTRAL_ACTION))
        return {slot: action_vec_to_controller(action) for slot, action in actions.items()}

    @property
    def neutral_action_fraction(self) -> float:
        return self.neutral_actions / max(self.total_actions, 1)


def make_policy(
    model: Policy,
    stats: dict[str, FeatureStats],
    cfg: TrainConfig,
    *,
    decode_seed: int | None = None,
    inference: BF16Inference | None = None,
    telemetry: DecodeTelemetry | None = None,
    device: str = DEVICE,
) -> DelayedTruncationPolicy:
    horizon = cfg.prediction_frames
    if horizon not in EVAL_HORIZONS:
        raise ValueError(f"execution horizon must be one of {EVAL_HORIZONS}")
    engine = BF16Inference(model, cfg) if inference is None else inference
    random_streams = None if decode_seed is None else SlotGroupRng(decode_seed, GROUP_NAMES)
    generator = None if decode_seed is None else torch.Generator(device=device).manual_seed(decode_seed)

    @torch.no_grad()
    def predict(ctx: Context, committed: np.ndarray | None) -> np.ndarray:
        if committed is not None:
            raise ValueError("O50 uses truncation only and never conditions on a committed prefix")
        started = time.perf_counter()
        result = engine.decode(ctx, horizon, streams=random_streams, gen=generator).cpu().numpy()
        if telemetry is not None:
            telemetry.record(rows=ctx.ctx_pad.shape[0], horizon=horizon, seconds=time.perf_counter() - started)
        return result

    return DelayedTruncationPolicy(
        predict_chunk=predict,
        stats=stats,
        L_ctx=cfg.arch.L_ctx,
        L_chunk=horizon,
        s=REPLAN_INTERVAL_FRAMES,
        d=0,
        device=device,
        float_dtype=next(model.parameters()).dtype,
        extra=ITEM_COLUMNS,
        projection=BASE_ITEMS_PROJECTION,
    )


@dataclass(frozen=True, slots=True)
class EvalProtocol:
    n_matchups: int
    allowed_cpus: int
    hardware_wave_bucket: int
    max_parallel: int
    max_frames: int
    seed: int
    cpu_level: int
    ego_port: Literal[1, 2]
    seed_stage: int
    matchup_schedule_sha256: str
    oriented_pairs: int
    ego_characters: int
    cpu_characters: int
    prediction_frames: int
    delay_frames: int
    replan_interval_frames: int
    dtype: str
    inference_mode: str
    inference_compile_mode: str
    inference_attention_backend: str
    compiled_inference_bucket: int
    checkpoint_sha256: str
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES
    start_retries: int = DEFAULT_START_RETRIES


def matchup_diversity(n_matchups: int) -> tuple[int, int, int, str]:
    matchups = matchups_for_vs_cpu(n_matchups)
    schedule = [(int(ego.value), int(cpu.value)) for ego, cpu in matchups]
    sha = hashlib.sha256(json.dumps(schedule, separators=(",", ":")).encode()).hexdigest()
    return len(set(schedule)), len({ego for ego, _ in schedule}), len({cpu for _, cpu in schedule}), sha


def assert_protocol_diversity(n_matchups: int) -> tuple[int, int, int, str]:
    diversity = matchup_diversity(n_matchups)
    expected = {32: (26, 8, 8), 96: (58, 13, 14)}.get(n_matchups)
    if expected is not None and diversity[:3] != expected:
        raise AssertionError(
            f"deterministic {n_matchups}-matchup schedule changed: got {diversity[:3]}, expected {expected}"
        )
    return diversity


def _eval_protocol(
    cfg: TrainConfig,
    model: Policy,
    *,
    n_matchups: int,
    checkpoint_sha256: str,
    inference_compile_mode: str = "reduce-overhead",
    inference_attention_backend: str = "dense_sdpa",
) -> EvalProtocol:
    pairs, egos, cpus, schedule_sha = assert_protocol_diversity(n_matchups)
    return EvalProtocol(
        n_matchups=n_matchups,
        allowed_cpus=usable_cpus(),
        hardware_wave_bucket=automatic_parallelism(),
        max_parallel=_eval_parallelism(cfg, n_matchups),
        max_frames=cfg.eval_max_frames,
        seed=cfg.eval_seed,
        cpu_level=9,
        ego_port=1,
        seed_stage=int(PRIOR_SWEEP_SEED_STAGE.value),
        matchup_schedule_sha256=schedule_sha,
        oriented_pairs=pairs,
        ego_characters=egos,
        cpu_characters=cpus,
        prediction_frames=cfg.prediction_frames,
        delay_frames=cfg.delay_frames,
        replan_interval_frames=cfg.replan_interval_frames,
        dtype=str(next(model.parameters()).dtype),
        inference_mode=cfg.inference_mode,
        inference_compile_mode=inference_compile_mode,
        inference_attention_backend=inference_attention_backend,
        compiled_inference_bucket=_eval_inference_bucket(cfg, n_matchups),
        checkpoint_sha256=checkpoint_sha256,
    )


def _write_eval_evidence(
    replay_dir: Path, rows: list[MatchRow], metrics: dict[str, float], protocol: EvalProtocol
) -> None:
    replay_dir.mkdir(parents=True, exist_ok=True)
    rows_payload = {
        "schema_version": 6,
        "protocol": asdict(protocol),
        "rows": [row.as_dict() for row in rows],
    }
    for path, payload in (
        (replay_dir / "match_rows.json", rows_payload),
        (replay_dir / "metrics.json", metrics),
    ):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True))
        temporary.replace(path)


def eval_vs_cpu(
    model: Policy,
    stats: dict[str, FeatureStats],
    cfg: TrainConfig,
    *,
    n_matchups: int,
    replay_dir: Path,
    checkpoint_sha256: str = "unavailable",
    inference: BF16Inference | None = None,
) -> dict[str, float]:
    horizon = cfg.prediction_frames
    inference = BF16Inference(model, cfg) if inference is None else inference
    if inference.model is not model:
        raise ValueError("the supplied inference engine must own the evaluation model")
    protocol = _eval_protocol(
        cfg,
        model,
        n_matchups=n_matchups,
        checkpoint_sha256=checkpoint_sha256,
        inference_compile_mode=inference.compile_mode,
        inference_attention_backend=inference.attention_backend,
    )
    if next(model.parameters()).device.type == "cuda" and (
        protocol.inference_mode != "compiled" or not inference.compiled
    ):
        raise RuntimeError("official CUDA evaluation requires compiled BF16 inference")
    telemetry = DecodeTelemetry()
    policy_index = itertools.count()
    policies: list[DelayedTruncationPolicy] = []

    def factory() -> RecedingHorizon:
        policy = make_policy(
            model,
            stats,
            cfg,
            decode_seed=protocol.seed + next(policy_index),
            inference=inference,
            telemetry=telemetry,
        )
        policies.append(policy)
        return policy

    was_training = model.training
    model.eval()
    total_started = time.perf_counter()
    try:
        compile_seconds = inference.prewarm(protocol.max_parallel, horizon)
        started = time.perf_counter()
        with torch.compiler.set_stance("fail_on_recompile"):
            results, rows = sweep_vs_cpu_prior_with_rows(
                factory,
                session_cfg=default_session_cfg(replay_dir, instant_match_restart=True),
                n_matchups=protocol.n_matchups,
                max_parallel=protocol.max_parallel,
                max_frames=protocol.max_frames,
                cpu_level=protocol.cpu_level,
                ego_port=protocol.ego_port,
                seed_stage=melee.Stage(protocol.seed_stage),
                start_retries=protocol.start_retries,
            )
    finally:
        model.train(was_training)
    metrics = vs_cpu_metrics(results, seed=protocol.seed)
    metrics["eval_wall_seconds"] = time.perf_counter() - started
    metrics["eval_total_wall_seconds"] = time.perf_counter() - total_started
    metrics["inference_compile_seconds"] = compile_seconds
    neutral = sum(policy.neutral_actions for policy in policies)
    actions = sum(policy.total_actions for policy in policies)
    metrics["neutral_action_fraction"] = neutral / max(actions, 1)
    metrics["prediction_frames"] = float(cfg.prediction_frames)
    metrics["delay_frames"] = float(cfg.delay_frames)
    metrics["replan_interval_frames"] = float(cfg.replan_interval_frames)
    metrics.update(telemetry.metrics())
    _write_eval_evidence(replay_dir, rows, metrics, protocol)
    return metrics


def _eval_wandb_metrics(values: dict[str, float]) -> dict[str, float]:
    """Keep closed-loop quality, uncertainty, reliability, and latency."""
    names = {
        "stocks_taken_per_min": "stocks_taken_per_min",
        "stocks_lost_per_min": "stocks_lost_per_min",
        "damage_dealt_per_min": "damage_dealt_per_min",
        "damage_taken_per_min": "damage_taken_per_min",
        "net_stock_per_min": "net_stock_per_min",
        "net_dmg_per_min": "net_dmg_per_min",
        "net_stock_cluster_bootstrap_lcb": "net_stock_lcb",
        "net_dmg_cluster_bootstrap_lcb": "net_dmg_lcb",
        "boots": "boots",
        "crashed": "crashed",
        "dead_frame_frac": "dead_frame_fraction",
        "decode_p95_ms": "decode_p95_ms",
        "eval_total_wall_seconds": "wall_s",
        "neutral_action_fraction": "neutral_action_fraction",
        "prediction_frames": "prediction_frames",
        "delay_frames": "delay_frames",
        "replan_interval_frames": "replan_interval_frames",
    }
    return {output: values[source] for source, output in names.items() if source in values}


def require_complete_eval(metrics: dict[str, float], expected_boots: int) -> None:
    """Fail unless every scheduled boot reached active gameplay."""
    scheduled = int(metrics.get("scheduled_boots", 0.0))
    completed = int(metrics.get("completed_boots", 0.0))
    active = int(metrics.get("boots", 0.0))
    if scheduled != expected_boots or completed != expected_boots or active != expected_boots:
        raise RuntimeError(
            "closed-loop evaluation is incomplete: "
            f"scheduled={scheduled}/{expected_boots}, completed={completed}/{expected_boots}, "
            f"active={active}/{expected_boots}"
        )


def lr_schedule(cfg: TrainConfig):
    def schedule(step: int) -> float:
        update = step + 1
        if update <= cfg.warmup_steps:
            return update / cfg.warmup_steps
        progress = (update - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return cfg.lr_floor_ratio + (1.0 - cfg.lr_floor_ratio) * cosine

    return schedule


def _button_gradient_abs_max(model: Policy) -> Tensor:
    """Return one pre-clipping action-path gradient guardrail."""
    maxima: list[Tensor] = []
    for name, parameter in _button_adam_parameters(model).items():
        if parameter.grad is None:
            raise RuntimeError(f"button parameter {name!r} has no gradient")
        maxima.append(parameter.grad.detach().float().abs().amax())
    return torch.stack(maxima).amax()


def _stable_adam_metrics(diagnostics: dict[str, Tensor]) -> dict[str, Tensor]:
    """O50 intentionally has no per-tensor adaptive-update clipping."""
    del diagnostics
    return {}


def approximate_training_flops_per_update(cfg: TrainConfig, parameter_counts: dict[str, int]) -> int:
    """Estimate forward-backward FLOPs from each subsystem's parameter uses."""
    full = cfg.arch.L_ctx
    suffix = full - DIRECT_LOSS_START
    trunk_and_inputs = parameter_counts["trunk"] + parameter_counts["other"]
    temporal_and_heads = parameter_counts["temporal_decoder"] + parameter_counts["group_heads"]
    parameter_uses = (
        full * trunk_and_inputs
        + suffix * parameter_counts["value_head"]
        + suffix * len(cfg.arch.head_offsets) * temporal_and_heads
    )
    return 6 * cfg.batch_size * parameter_uses


def log_wandb_code(run: wandb.Run) -> None:
    root = Path(__file__).resolve().parents[1]
    allowed_dirs = {"docker", "experiments", "hal", "notebooks", "scripts", "tests"}

    def include(path: str, code_root: str) -> bool:
        try:
            relative = Path(path).resolve().relative_to(Path(code_root).resolve())
        except ValueError:
            return False
        return (
            bool(relative.parts)
            and relative.parts[0] in allowed_dirs
            and relative.suffix in {".py", ".sh", ".toml", ".yaml", ".yml"}
        )

    run.log_code(root=str(root), include_fn=include)


def _wandb_parameter_group(name: str) -> str:
    parts = name.split(".")
    if len(parts) >= 3 and parts[:2] == ["trunk", "blocks"]:
        return f"trunk/block_{parts[2]}"
    if len(parts) >= 3 and parts[:2] == ["temporal", "blocks"]:
        return f"decoder/temporal_block_{parts[2]}"
    if parts[0] == "temporal":
        return "decoder/heads" if parts[1] == "outputs" else "decoder/other"
    if parts[0] == "value_head":
        return "critic/value_head"
    return "trunk/input"


def wandb_weight_log(model: nn.Module, sample_limit: int = 65_536) -> dict[str, object]:
    """Return sampled parameter histograms grouped by subsystem."""
    buckets: dict[str, list[Tensor]] = {}
    for name, parameter in model.named_parameters():
        value = parameter
        if value.is_sparse:
            value = value.coalesce().values()
        buckets.setdefault(_wandb_parameter_group(name), []).append(value.detach())

    payload: dict[str, object] = {}
    for group, values in buckets.items():
        count = sum(value.numel() for value in values)
        stride = max(1, math.ceil(count / sample_limit))
        samples = torch.cat([value.reshape(-1)[::stride] for value in values])[:sample_limit]
        payload[f"weights/{group}"] = wandb.Histogram(samples.float().cpu().numpy())
    return payload


def histogram_due(update: int, every: int) -> bool:
    """Whether global update-count cadence requests parameter histograms."""
    return every > 0 and (update == 1 or update % every == 0)


def _residual_layers(model: Policy) -> tuple[tuple[str, nn.Module], ...]:
    """Return the residual blocks whose scale health is monitored."""
    trunk = tuple((f"trunk_block_{index:02d}", block) for index, block in enumerate(model.trunk.blocks))
    temporal = tuple((f"temporal_block_{index:02d}", block) for index, block in enumerate(model.temporal.blocks))
    return trunk + temporal


def _diagnostic_batch(batch: TrainBatch | AWRBatch, max_rows: int) -> TrainBatch:
    """Take a small device-resident prefix for the occasional eager diagnostic."""
    if max_rows < 1:
        raise ValueError(f"max_rows must be positive, got {max_rows}")
    source = batch.batch if isinstance(batch, AWRBatch) else batch
    rows = min(max_rows, source.target.shape[0])
    context = source.context
    return TrainBatch(
        context=Context(
            features={name: value[:rows] for name, value in context.features.items()},
            ctx_pad=context.ctx_pad[:rows],
            slot_ids=None if context.slot_ids is None else context.slot_ids[:rows],
            reset=None if context.reset is None else context.reset[:rows],
        ),
        target=source.target[:rows],
        replay_ids=None if source.replay_ids is None else source.replay_ids[:rows],
    )


def _diagnostic_awr_batch(batch: AWRBatch, max_rows: int) -> AWRBatch:
    """Take a device-resident prefix while preserving critic targets."""
    diagnostic = _diagnostic_batch(batch, max_rows)
    rows = diagnostic.target.shape[0]
    return AWRBatch(
        batch=diagnostic,
        returns=batch.returns[:rows],
        eligible=batch.eligible[:rows],
    )


def _top_right_singular_vector(weight: Tensor) -> tuple[Tensor, Tensor]:
    """Estimate the leading singular value and right vector by power iteration."""
    matrix = weight.detach().float()
    right = torch.ones(matrix.shape[1], device=matrix.device, dtype=matrix.dtype)
    right = F.normalize(right, dim=0)
    for _ in range(_ARCHITECTURE_POWER_ITERATIONS):
        left = F.normalize(matrix @ right, dim=0)
        right = F.normalize(matrix.T @ left, dim=0)
    return (matrix @ right).norm(), right


def _matrix_geometry_metrics(prefix: str, weight: Tensor) -> tuple[dict[str, Tensor], Tensor]:
    """Measure gain, effective rank, row scale, and class-common mode."""
    matrix = weight.detach().float()
    top_singular, top_right = _top_right_singular_vector(matrix)
    frobenius = matrix.norm()
    row_norms = matrix.norm(dim=1)
    common_row = matrix.mean(dim=0, keepdim=True)
    common_mode = common_row.norm() * math.sqrt(matrix.shape[0])
    centered = (matrix - common_row).norm()
    tiny = torch.finfo(matrix.dtype).tiny
    return (
        {
            f"{prefix}/top_singular_value_estimate": top_singular,
            f"{prefix}/frobenius_norm": frobenius,
            f"{prefix}/stable_rank_estimate": frobenius.square() / top_singular.square().clamp_min(tiny),
            f"{prefix}/row_norm_mean": row_norms.mean(),
            f"{prefix}/row_norm_max": row_norms.amax(),
            f"{prefix}/row_norm_p99": _sampled_quantile(row_norms, 99.0),
            f"{prefix}/centered_frobenius_norm": centered,
            f"{prefix}/common_mode_frobenius_norm": common_mode,
        },
        top_right,
    )


@torch.compiler.disable
def architecture_diagnostics_log(
    model: Policy,
    batch: AWRBatch,
    cfg: TrainConfig,
    *,
    max_rows: int,
) -> dict[str, float]:
    """Measure head geometry and per-objective shared-state gradients."""
    diagnostic = _diagnostic_awr_batch(batch, max_rows)
    history, targets, valid = prepared_targets(model, diagnostic)
    device = next(model.parameters()).device
    with torch.no_grad(), amp_context(cfg, device):
        hidden_base = model.forward_dense(diagnostic.context.features, diagnostic.context.ctx_pad, None)
        hidden_base = hidden_base[:, direct_loss_start(cfg) :]
    hidden = hidden_base.detach().requires_grad_(True)
    with amp_context(cfg, device):
        logits, button_values = model.temporal._teacher_forced_outputs(hidden, history, targets)
        dense_nll = model.temporal.nll_from_logits(logits, targets)

    metrics: dict[str, Tensor] = {}
    valid_float = valid.float()
    denominator = valid_float.sum().clamp_min(1) * len(cfg.arch.head_offsets)
    group_gradient_rms: dict[str, Tensor] = {}
    for group, name in enumerate(GROUP_NAMES):
        group_loss = (dense_nll[..., group] * valid_float[..., None]).sum() / denominator
        gradient = torch.autograd.grad(group_loss, hidden, retain_graph=group + 1 < N_GROUPS)[0].detach().float()
        gradient_rms = gradient.square().mean().sqrt()
        group_gradient_rms[name] = gradient_rms
        metrics[f"multi_objective_grad/{name}_rms"] = gradient_rms
        metrics[f"multi_objective_grad/{name}_abs_max"] = gradient.abs().amax()

    hypothetical_value = model.value_head(decoder_rmsnorm(hidden).float()).squeeze(-1)
    hypothetical_value_loss, _, _ = value_objective(
        hypothetical_value,
        diagnostic.returns[:, direct_loss_start(cfg) :],
        diagnostic.eligible[:, direct_loss_start(cfg) :],
        beta=cfg.awr.beta,
        valid=valid,
    )
    value_gradient = torch.autograd.grad(hypothetical_value_loss, hidden)[0].detach().float()
    metrics["multi_objective_grad/value_actual_rms"] = torch.zeros((), device=device)
    metrics["multi_objective_grad/value_hypothetical_rms"] = value_gradient.square().mean().sqrt()
    metrics["multi_objective_grad/value_hypothetical_abs_max"] = value_gradient.abs().amax()
    shared_gradient_total = (
        torch.stack(tuple(group_gradient_rms.values())).sum().clamp_min(torch.finfo(torch.float32).tiny)
    )
    metrics["multi_objective_grad/buttons_fraction"] = group_gradient_rms["buttons"] / shared_gradient_total

    output_vectors: dict[str, Tensor] = {}
    for name in GROUP_NAMES:
        head = cast(NonlinearActionHead, model.temporal.outputs[name])
        geometry, output_vectors[name] = _matrix_geometry_metrics(
            f"head_geometry/{name}_output",
            head.down.weight,
        )
        metrics.update(geometry)
    token_geometry, _ = _matrix_geometry_metrics(
        "head_geometry/token_projection",
        model.temporal.token_projection.weight,
    )
    metrics.update(token_geometry)
    for name, module in model.temporal.group_condition.items():
        condition = cast(nn.Linear, module)
        condition_geometry, _ = _matrix_geometry_metrics(
            f"head_geometry/{name}_condition",
            condition.weight,
        )
        metrics.update(condition_geometry)

    _, head_input, _, _ = button_values
    flat_input = head_input.detach().float().reshape(-1, head_input.shape[-1])
    alignment = (flat_input @ output_vectors["buttons"]).abs() / flat_input.norm(dim=-1).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    metrics["head_geometry/buttons_input_top_vector_alignment_mean"] = alignment.mean()
    metrics["head_geometry/buttons_input_top_vector_alignment_max"] = alignment.amax()
    metrics["head_geometry/buttons_input_top_vector_alignment_p99"] = _sampled_quantile(alignment, 99.0)

    values = torch.stack(tuple(metrics.values())).double().cpu()
    payload = {name: float(value) for name, value in zip(metrics, values, strict=True)}
    nonfinite = {name: value for name, value in payload.items() if not math.isfinite(value)}
    if nonfinite:
        raise FloatingPointError(f"architecture diagnostic produced non-finite metrics: {nonfinite}")
    return payload


def _baseline_layer_activation_rms_log(
    model: Policy,
    batch: TrainBatch | AWRBatch,
    cfg: TrainConfig,
    *,
    max_rows: int,
) -> dict[str, float]:
    """Measure residual-stream and branch RMS on a small eager forward pass.

    A residual block is treated as one layer: ``x_{l+1} = x_l + F_l(x_l)``.
    The branch is recovered exactly as ``F_l(x_l) = x_{l+1} - x_l``. Hooks are
    present only for this infrequent eager pass, so they do not enter or
    invalidate the compiled training graph.
    """
    measurements: dict[str, tuple[Tensor, Tensor]] = {}
    branch_measurements: dict[tuple[str, str], Tensor] = {}

    def capture(name: str):
        def hook(_module: nn.Module, inputs: tuple[object, ...], output: object) -> None:
            layer_input = inputs[0]
            if not isinstance(layer_input, Tensor) or not isinstance(output, Tensor):
                raise TypeError(f"{name} RMS hook expected tensor input and output")
            activation_rms = layer_input.detach().float().square().mean().sqrt()
            residual_rms = (output.detach().float() - layer_input.detach().float()).square().mean().sqrt()
            measurements[name] = (activation_rms, residual_rms)

        return hook

    def capture_branch(name: str, branch: str, scale: float):
        def hook(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
            if not isinstance(output, Tensor):
                raise TypeError(f"{name} {branch} RMS hook expected a tensor output")
            branch_measurements[(name, branch)] = scale * output.detach().float().square().mean().sqrt()

        return hook

    handles = [layer.register_forward_hook(capture(name)) for name, layer in _residual_layers(model)]
    temporal_names: list[str] = []
    for index, module in enumerate(model.temporal.blocks):
        block = cast(TemporalBlock, module)
        name = f"temporal_block_{index:02d}"
        temporal_names.append(name)
        handles.append(block.proj.register_forward_hook(capture_branch(name, "attention", block.scale)))
        handles.append(block.down.register_forward_hook(capture_branch(name, "mlp", 1.0)))
    try:
        diagnostic = _diagnostic_batch(batch, max_rows)
        history, targets, _ = prepared_targets(model, diagnostic)
        device = next(model.parameters()).device
        with amp_context(cfg, device):
            hidden = model.forward_dense(diagnostic.context.features, diagnostic.context.ctx_pad, None)
            hidden = hidden[:, direct_loss_start(cfg) :]
            model.temporal.teacher_forced_states(hidden, history, targets)
    finally:
        for handle in handles:
            handle.remove()

    layer_names = tuple(name for name, _ in _residual_layers(model))
    missing = set(layer_names) - measurements.keys()
    if missing:
        raise RuntimeError(f"RMS diagnostic did not observe residual layers {sorted(missing)}")
    expected_branches = {(name, branch) for name in temporal_names for branch in ("attention", "mlp")}
    missing_branches = expected_branches - branch_measurements.keys()
    if missing_branches:
        raise RuntimeError(f"RMS diagnostic did not observe temporal branches {sorted(missing_branches)}")
    payload_tensors: dict[str, Tensor] = {}
    for name in layer_names:
        activation_rms, residual_rms = measurements[name]
        payload_tensors[f"activation_rms/{name}"] = activation_rms
        payload_tensors[f"residual_branch_rms/{name}"] = residual_rms
        payload_tensors[f"residual_ratio/{name}"] = residual_rms / activation_rms.clamp_min(
            torch.finfo(torch.float32).tiny
        )
    for name in temporal_names:
        payload_tensors[f"attention_branch_rms/{name}"] = branch_measurements[(name, "attention")]
        payload_tensors[f"mlp_branch_rms/{name}"] = branch_measurements[(name, "mlp")]
    scalars = torch.stack(tuple(payload_tensors.values())).double().cpu()
    payload = {name: float(value) for name, value in zip(payload_tensors, scalars, strict=True)}
    nonfinite = {name: value for name, value in payload.items() if not math.isfinite(value)}
    if nonfinite:
        raise FloatingPointError(f"layer activation diagnostic produced non-finite metrics: {nonfinite}")
    return payload


class _LoaderKwargs(TypedDict):
    data_root: str | None
    sources: tuple[streams.StreamSource, ...]
    cache_limit: str
    shuffle_block_size: int
    shuffle_seed: int
    stats: dict[str, FeatureStats]
    L_ctx: int
    L_chunk: int
    batch_size: int
    seed: int
    schema_version: int
    extra: ExtraColumns
    projection: FeatureProjection


def loader_kwargs(cfg: TrainConfig, stats: dict[str, FeatureStats]) -> _LoaderKwargs:
    sources = tuple(streams.BY_NAME[name] for name in cfg.source_names)
    return dict(
        data_root=None,
        sources=sources,
        cache_limit=f"{cfg.cache_limit_gb}gb",
        shuffle_block_size=cfg.shuffle_block_size,
        shuffle_seed=cfg.seed,
        stats=stats,
        L_ctx=cfg.arch.L_ctx,
        L_chunk=cfg.arch.sample_chunk_length,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        schema_version=cfg.mds_schema_version,
        extra=MODEL_COLUMNS,
        projection=MODEL_PROJECTION,
    )


def validate_batch_geometry(
    batch: TrainBatch | AWRBatch, cfg: TrainConfig, expected_batch_size: int | None = None
) -> None:
    if batch.target.shape[1:] != (cfg.arch.sample_chunk_length, ACTION_DIM):
        raise ValueError(
            f"target must be [B, {cfg.arch.sample_chunk_length}, {ACTION_DIM}], got {tuple(batch.target.shape)}"
        )
    batch_size = batch.target.shape[0]
    if expected_batch_size is not None and batch_size != expected_batch_size:
        raise ValueError(f"fixed training batch must contain {expected_batch_size} rows, got {batch_size}")
    if batch.context.ctx_pad.shape != (batch_size,):
        raise ValueError("ctx_pad shape does not match the batch")
    wrong = {
        name: tuple(value.shape)
        for name, value in batch.context.features.items()
        if value.shape[:2] != (batch_size, cfg.arch.L_ctx)
    }
    if wrong:
        raise ValueError(f"context features have the wrong geometry: {wrong}")


def cache_validation(loader: Iterable[TrainBatch], n_samples: int) -> list[TrainBatch]:
    batches: list[TrainBatch] = []
    count = 0
    for batch in loader:
        remaining = n_samples - count
        if remaining <= 0:
            break
        if batch.target.shape[0] > remaining:
            batch = TrainBatch(
                context=Context(
                    features={name: value[:remaining] for name, value in batch.context.features.items()},
                    ctx_pad=batch.context.ctx_pad[:remaining],
                ),
                target=batch.target[:remaining],
                replay_ids=None if batch.replay_ids is None else batch.replay_ids[:remaining],
            )
        batches.append(batch)
        count += batch.target.shape[0]
    if count != n_samples:
        raise RuntimeError(f"validation yielded {count} samples, expected {n_samples}")
    return batches


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _replace_link(source: Path, destination: Path) -> None:
    """Atomically point ``destination`` at an immutable checkpoint inode."""
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    os.link(source, temporary)
    os.replace(temporary, destination)


def save_boundary_checkpoint(
    run_dir: Path,
    *,
    update: int,
    model: Policy,
    optimizer: SingleDeviceMuonWithAuxAdam,
    scheduler: LambdaLR,
    cfg: TrainConfig,
    uploader: BackgroundUploader | None,
    milestone: bool,
    wandb_id: str | None,
    actual_loss_positions: int,
    loader_state: dict[str, object],
    identity_masker_state: dict[str, object] | None = None,
) -> Path:
    """Save one immutable boundary snapshot, then atomically advance latest."""
    snapshot = run_dir / f"boundary-step-{update:07d}.pt"
    if snapshot.exists():
        raise FileExistsError(f"immutable checkpoint already exists: {snapshot}")
    temporary = snapshot.with_suffix(snapshot.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    save_checkpoint(
        temporary,
        step=update - 1,
        model=model,
        opt=optimizer,
        sched=scheduler,
        cfg=_checkpoint_config(cfg),
        wandb_id=wandb_id,
        uploader=None,
        extra_state={
            "actual_loss_positions": actual_loss_positions,
            "loader": loader_state,
            "identity_masker": (
                loader_state.get("identity_masker") if identity_masker_state is None else identity_masker_state
            ),
        },
    )
    os.replace(temporary, snapshot)
    latest = run_dir / "latest.pt"
    _replace_link(snapshot, latest)
    if uploader is not None:
        uploader.upload(snapshot, key="latest.pt")
        if milestone:
            uploader.upload(snapshot, key=f"checkpoints/step-{update:07d}.pt")
    return snapshot


def load_identity_sidecar(cfg: TrainConfig) -> PlayerIdentitySidecar:
    """Download, hash-check, and load the frozen O49 identity artifact."""
    path = Path(cfg.player_sidecar_local)
    if not path.is_file():
        bucket, _, key = PLAYER_SIDECAR_REMOTE.removeprefix("s3://").partition("/")
        path.parent.mkdir(parents=True, exist_ok=True)
        r2.client().download_file(bucket, key, str(path))
    sidecar = load_player_identity_sidecar(path, expected_sha256=cfg.player_sidecar_sha256)
    if sidecar.vocabulary.size != cfg.player_vocab_size or sidecar.vocabulary.sha256 != cfg.player_vocab_sha256:
        raise ValueError("identity sidecar vocabulary differs from the frozen O50 contract")
    return sidecar


def _watch_gradients(model: nn.Module, cfg: TrainConfig) -> None:
    """Reject the legacy watcher; O50 logs only aggregated scalar telemetry."""
    del model
    if cfg.gradient_hist_every:
        raise ValueError("O50 does not use wandb.watch; gradient_hist_every must be zero")


def _baseline_training_diagnostics(model: Policy, batch: AWRBatch, cfg: TrainConfig, update: int) -> dict[str, object]:
    """Collect the infrequent parameter and layer diagnostics due this update."""
    metrics: dict[str, object] = {}
    if histogram_due(update, cfg.weight_hist_every):
        metrics.update(wandb_weight_log(model))
    if histogram_due(update, cfg.layer_rms_every):
        metrics.update(layer_activation_rms_log(model, batch, cfg, max_rows=cfg.layer_rms_batch_size))
    if histogram_due(update, cfg.architecture_metrics_every):
        metrics.update(architecture_diagnostics_log(model, batch, cfg, max_rows=cfg.layer_rms_batch_size))
    return metrics


@dataclass(frozen=True, slots=True)
class TrainStepResult:
    nll_sum: Tensor
    gradient_norm: Tensor
    metrics: dict[str, Tensor]
    diagnostics: dict[str, object]
    muon_lr: float
    adam_lr: float


@dataclass(slots=True)
class _TrainingMetricAccumulator:
    """Accumulate device metrics and download one payload per log window."""

    _sum: Tensor | None = None
    _metric_names: tuple[str, ...] = ()
    updates: int = 0
    valid_prefixes: int = 0

    def add(self, result: TrainStepResult, valid_prefixes: int) -> None:
        if valid_prefixes <= 0:
            raise ValueError("valid_prefixes must be positive")
        metric_names = tuple(result.metrics)
        if self._metric_names and metric_names != self._metric_names:
            raise RuntimeError(f"training metric names changed from {self._metric_names} to {metric_names}")
        scalar_metrics = torch.stack(
            [result.gradient_norm.detach().float(), *(result.metrics[name].detach().float() for name in metric_names)]
        )
        payload = torch.cat((result.nll_sum.detach().reshape(-1).float(), scalar_metrics))
        if self._sum is None:
            self._sum = payload
            self._metric_names = metric_names
        else:
            self._sum.add_(payload)
        self.updates += 1
        self.valid_prefixes += valid_prefixes

    def flush(self, cfg: TrainConfig, *, update: int) -> tuple[dict[str, float], int, int]:
        """Synchronize once, return window means, and reset the accumulator."""
        if self._sum is None or self.updates == 0 or self.valid_prefixes == 0:
            raise RuntimeError("cannot flush an empty training metric accumulator")
        payload = self._sum.cpu()
        if not torch.isfinite(payload).all():
            raise FloatingPointError(f"update {update}: accumulated training metrics contain a non-finite value")

        nll_values = len(cfg.arch.head_offsets) * N_GROUPS
        mean_nll = payload[:nll_values].reshape(len(cfg.arch.head_offsets), N_GROUPS) / self.valid_prefixes
        scalar_values = payload[nll_values:] / self.updates
        nll_metrics = nll_mean_metrics(
            mean_nll,
            cfg.arch.head_offsets,
            aux_loss_weight=cfg.awr.auxiliary_loss_weight,
        )
        values = {
            "train/nll": nll_metrics["loss_unweighted"],
            "optimizer/grad_norm": float(scalar_values[0]),
        }
        values.update({name: float(value) for name, value in zip(self._metric_names, scalar_values[1:], strict=True)})

        updates = self.updates
        valid_prefixes = self.valid_prefixes
        self._sum = None
        self._metric_names = ()
        self.updates = 0
        self.valid_prefixes = 0
        return values, updates, valid_prefixes


def train_step(
    model: Policy,
    batch: AWRBatch,
    cfg: TrainConfig,
    *,
    step: int,
    update: int,
    valid_prefixes: int,
    trunk_fn: Callable,
    temporal_fn: Callable,
    optimizer: SingleDeviceMuonWithAuxAdam,
    scheduler: LambdaLR,
    phase_timer: CudaPhaseTimer | None = None,
) -> TrainStepResult:
    """Run one complete optimization step on a device-resident batch."""
    if DEVICE == "cuda" and (cfg.compile_trunk or cfg.compile_temporal):
        torch.compiler.cudagraph_mark_step_begin()
    optimizer.zero_grad()
    loss, nll_sum, metrics = microbatch_loss(
        model,
        batch,
        cfg,
        step=step,
        valid_prefixes=valid_prefixes,
        trunk_fn=trunk_fn,
        temporal_fn=temporal_fn,
        phase_timer=phase_timer,
    )
    loss.backward()
    if phase_timer is not None:
        phase_timer.record("backward_end")
    metrics["stability/action_grad_abs_max"] = _button_gradient_abs_max(model)
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    metrics["optimizer/clip_fraction"] = (gradient_norm > cfg.grad_clip).float()
    if phase_timer is not None:
        phase_timer.record("grad_norm_end")
    diagnostics = _training_diagnostics(model, batch, cfg, update)
    if phase_timer is not None:
        phase_timer.record("diagnostics_end")
    muon_lr = float(next(group["lr"] for group in optimizer.param_groups if group["use_muon"]))
    adam_lr = float(next(group["lr"] for group in optimizer.param_groups if not group["use_muon"]))
    optimizer.step()
    metrics.update(_stable_adam_metrics(optimizer.last_adam_diagnostics))
    scheduler.step()
    if phase_timer is not None:
        phase_timer.record("optimizer_end")
    return TrainStepResult(nll_sum, gradient_norm, metrics, diagnostics, muon_lr, adam_lr)


def _cadence_in_window(first_update: int, last_update: int, every: int) -> bool:
    """Return whether update one or a cadence boundary is in the window."""
    if every <= 0:
        return False
    return first_update == 1 or last_update // every > (first_update - 1) // every


def _mean_phase_metrics(timers: list[CudaPhaseTimer]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for timer in timers:
        for name, value in timer.metrics().items():
            totals[name] = totals.get(name, 0.0) + value
    return {name: value / len(timers) for name, value in totals.items()} if timers else {}


def _write_eval_request(
    run_dir: Path,
    update: int,
    checkpoint_sha256: str,
    n_matchups: int,
    *,
    final: bool,
    uploader: BackgroundUploader | None = None,
) -> Path:
    """Atomically queue one SHA-deduplicated RTX Pro 6000 request."""
    directory = run_dir / "eval_requests"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"step-{update:07d}-{checkpoint_sha256[:16]}.json"
    payload = {
        "schema": 1,
        "update": update,
        "checkpoint_sha256": checkpoint_sha256,
        "n_matchups": n_matchups,
        "final": final,
        "gpu": "RTX-PRO-6000",
        "cpus": 32,
        "memory_gib": 64,
        "disk_gib": 128,
        "timeout_hours": 2,
        "retries": 2,
        "prediction_frames": PREDICTION_FRAMES,
        "delay_frames": DELAY_FRAMES,
        "replan_interval_frames": REPLAN_INTERVAL_FRAMES,
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True))
    temporary.replace(path)
    if uploader is not None:
        uploader.upload(path, key=f"eval_requests/{path.name}")
    return path


def _baseline_finalize_training(
    *,
    model: Policy,
    optimizer: SingleDeviceMuonWithAuxAdam,
    scheduler: LambdaLR,
    cfg: TrainConfig,
    stats: dict[str, FeatureStats],
    val_cache: list[TrainBatch],
    run_dir: Path,
    replay_dir: Path,
    uploader: BackgroundUploader | None,
    loader_wait_fractions: list[float],
    loader_state: dict[str, object],
    identity_masker_state: dict[str, object],
    update: int,
    actual_loss_positions: int,
    smoke: bool,
    smoke_eval_matchups: int,
    arm_guard: _ArmGuard,
) -> None:
    """Save the final model and queue evaluation for a separate L40S worker."""
    snapshot = save_boundary_checkpoint(
        run_dir,
        update=update,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        uploader=uploader,
        milestone=cfg.ckpt_every > 0 and update % cfg.ckpt_every == 0,
        wandb_id=None if wandb.run is None else wandb.run.id,
        actual_loss_positions=actual_loss_positions,
        loader_state=loader_state,
        identity_masker_state=identity_masker_state,
    )
    final_path = run_dir / ("smoke-final.pt" if smoke else "final.pt")
    _replace_link(snapshot, final_path)
    if uploader is not None:
        uploader.upload(snapshot, key=final_path.name)

    checkpoint_sha = _checkpoint_sha256(final_path)
    validation = _validation_wandb_metrics(val_metrics(model, val_cache, cfg), cfg)
    final_metrics = {f"val/{name}": value for name, value in validation.items()}
    final_matchups = smoke_eval_matchups if smoke else cfg.final_eval_n_matchups
    _write_eval_request(run_dir, update, checkpoint_sha, final_matchups, final=True, uploader=uploader)
    _log_wandb({"global_step": update, **final_metrics}, arm_guard)

    mean_wait = float(np.mean(loader_wait_fractions)) if loader_wait_fractions else 0.0
    p95_wait = float(np.percentile(loader_wait_fractions, 95)) if loader_wait_fractions else 0.0
    print(f"[loader] mean wait={100 * mean_wait:.2f}%, p95={100 * p95_wait:.2f}%", flush=True)
    if smoke and (mean_wait > 0.05 or p95_wait > 0.10):
        raise RuntimeError("smoke loader gate failed: require mean wait <=5% and p95 <=10%")


def _compile_synthetic_forward_backward(
    model: Policy,
    cfg: TrainConfig,
    *,
    step: int,
    trunk_fn: Callable,
    temporal_fn: Callable,
) -> None:
    """Compile production-shaped loss graphs without changing training state."""
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if DEVICE == "cuda" else None
    try:
        model.train()
        model.zero_grad(set_to_none=True)
        if DEVICE == "cuda" and (cfg.compile_trunk or cfg.compile_temporal):
            torch.compiler.cudagraph_mark_step_begin()
        batch = synthetic_awr_batch(cfg, torch.device(DEVICE))
        valid_prefixes = cfg.batch_size * (cfg.arch.L_ctx - DIRECT_LOSS_START)
        loss, _nll, _metrics = microbatch_loss(
            model,
            batch,
            cfg,
            step=step,
            valid_prefixes=valid_prefixes,
            trunk_fn=trunk_fn,
            temporal_fn=temporal_fn,
        )
        loss.backward()
        if DEVICE == "cuda":
            torch.cuda.synchronize()
    finally:
        model.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)


def _training_loader_state(
    train_loader: PhysicalShardReplayLoader[AWRBatch],
    identity_masker: IdentityMasker,
) -> dict[str, object]:
    state = train_loader.state_dict()
    if not isinstance(state, dict):
        raise TypeError("training loader state must be a dict")
    if getattr(train_loader, "separate_identity_checkpoint", False):
        return state
    return {**state, "identity_masker": identity_masker.state_dict()}


def _checkpoint_mapping(state: Mapping[str, object], key: str) -> dict[str, object]:
    value = state.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"resume checkpoint has no {key!r} mapping")
    return cast(dict[str, object], value)


def train(
    cfg: TrainConfig,
    stats: dict[str, FeatureStats],
    *,
    comment: str = "",
    resume_run: str | None = None,
    resume_state: dict[str, object] | None = None,
    smoke: bool = False,
    stop_after_update: int | None = None,
    smoke_eval_matchups: int = 4,
    prepared_data_factory: Callable[
        [TrainConfig, dict[str, FeatureStats], PlayerIdentitySidecar, dict[str, object] | None],
        PreparedTrainingData,
    ]
    | None = None,
) -> None:
    validate_config(cfg)
    if not smoke:
        validate_production_config(cfg)
        if stop_after_update is not None:
            raise ValueError("stop_after_update is a smoke-only control")
    if stop_after_update is not None and not 1 <= stop_after_update <= cfg.max_steps:
        raise ValueError(f"stop_after_update must be in [1, {cfg.max_steps}], got {stop_after_update}")
    if smoke_eval_matchups < 0:
        raise ValueError("smoke_eval_matchups must be non-negative")
    run_stop = cfg.max_steps if stop_after_update is None else stop_after_update
    run_name = resume_run or make_run_name(Path(__file__).stem, model_tag(cfg), "policy-world-v7-35", comment)
    uploader = BackgroundUploader(run_name) if cfg.push_to_r2 else None
    _init_wandb(cfg, run_name, resume_state)
    arm_guard = _ArmGuard(cfg.warmup_steps, cfg.max_steps)
    run_dir, replay_dir = setup_run_dir(run_name)
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high" if cfg.allow_tf32 else "highest")
    sidecar = load_identity_sidecar(cfg)
    prepared_data = None if prepared_data_factory is None else prepared_data_factory(cfg, stats, sidecar, resume_state)
    model = Policy(cfg, sidecar.vocabulary).to(DEVICE)
    _watch_gradients(model, cfg)
    counts = subsystem_parameter_counts(model)
    flops_per_update = approximate_training_flops_per_update(cfg, counts)
    device_name = torch.cuda.get_device_name() if DEVICE == "cuda" else None
    peak_flops = bf16_dense_peak_flops(device_name or "")
    _log_training_summary(
        cfg,
        counts,
        flops_per_update=flops_per_update,
        device_name=device_name,
        peak_flops=peak_flops,
    )
    optimizer = make_optimizer(model, cfg)
    scheduler = LambdaLR(optimizer, lr_schedule(cfg))
    start_step = 0
    actual_positions = 0
    identity_masker = IdentityMasker(cfg.seed ^ 0x0501D, cfg.identity_dropout)
    if resume_state is not None:
        model.load_state_dict(_checkpoint_mapping(resume_state, "model"))
        optimizer.load_state_dict(_checkpoint_mapping(resume_state, "opt"))
        scheduler.load_state_dict(_checkpoint_mapping(resume_state, "sched"))
        identity_state = resume_state.get("identity_masker")
        if not isinstance(identity_state, dict):
            raise ValueError("resume checkpoint has no identity-mask RNG state")
        identity_masker.load_state_dict(cast(dict[str, object], identity_state))
        start_step = int(cast(int, resume_state["step"])) + 1
        positions_per_update = cfg.batch_size * (cfg.arch.L_ctx - DIRECT_LOSS_START)
        actual_positions = int(cast(int, resume_state.get("actual_loss_positions", start_step * positions_per_update)))
        if not 0 <= actual_positions <= start_step * positions_per_update:
            raise ValueError(
                f"checkpoint actual_loss_positions={actual_positions} is invalid after {start_step} updates"
            )

    trunk_fn, temporal_fn = _training_functions(model, cfg)

    if prepared_data is None:
        train_loader, val_cache = _make_loaders(cfg, stats)
    else:
        train_loader, val_cache = prepared_data.loader, prepared_data.validation
        _compile_synthetic_forward_backward(
            model,
            cfg,
            step=start_step,
            trunk_fn=trunk_fn,
            temporal_fn=temporal_fn,
        )
    if resume_state is not None and prepared_data is None:
        loader_state = resume_state.get("loader")
        if not isinstance(loader_state, dict):
            raise ValueError("resume checkpoint does not contain Mosaic streaming loader state")
        train_loader.load_state_dict(cast(dict[str, object], loader_state))
    source_sample_counts = getattr(train_loader, "source_sample_counts", None)
    train_replays = sum(source_sample_counts.values()) if isinstance(source_sample_counts, dict) else TRAIN_REPLAYS
    run_started = time.monotonic()
    if prepared_data is None:
        batch_prefetcher = DeviceBatchPrefetcher(train_loader, cfg, DEVICE, identity_masker)
    else:
        try:
            batch_prefetcher = DeviceBatchPrefetcher(
                train_loader,
                cfg,
                DEVICE,
                identity_masker,
                iterator=prepared_data.iterator,
                first_batch_future=prepared_data.first_batch_future,
            )
        finally:
            prepared_data.resources.close()
    loader_wait_fractions: list[float] = []
    cache_roots = tuple(streams.BY_NAME[name].local_root for name in cfg.source_names)
    host_metrics = HostMetricsSampler(
        cache_roots,
        interval_s=cfg.system_metrics_interval_s,
        process_interval_s=cfg.process_metrics_interval_s,
        cache_interval_s=cfg.cache_metrics_interval_s,
    )
    host_metrics.start()
    metric_accumulator = _TrainingMetricAccumulator()
    window_loader_wait_seconds: list[float] = []
    window_phase_timers: list[CudaPhaseTimer] = []
    window_diagnostics: dict[str, object] = {}
    window_peak_allocated_gb = 0.0
    metrics_window_started = time.monotonic()
    # CUDA compilation must remain on the training thread. Background compilation
    # deadlocked training on both H100 and B200 hosts.
    model.train()
    try:
        for step in range(start_step, run_stop):
            update = step + 1
            if DEVICE == "cuda":
                torch.cuda.reset_peak_memory_stats()

            val_due = cfg.val_every > 0 and update % cfg.val_every == 0 and update < run_stop
            eval_request_due = update == 9 or (
                cfg.eval_every > 0 and update % cfg.eval_every == 0 and update < run_stop
            )
            ckpt_due = cfg.ckpt_every > 0 and update % cfg.ckpt_every == 0 and update < run_stop
            preflight_boundary = update in (8, 9)
            boundary_due = val_due or eval_request_due or ckpt_due or preflight_boundary
            overlap_preload = update < run_stop and not boundary_due

            phase_due = (
                DEVICE == "cuda"
                and cfg.phase_timing_every > 0
                and (update == 1 or update % cfg.phase_timing_every == 0)
            )
            phase_timer = CudaPhaseTimer() if phase_due else None
            if phase_timer is not None:
                phase_timer.record("start")
            batch, valid_prefixes = batch_prefetcher.next()
            if phase_timer is not None:
                phase_timer.record("h2d_end")
            if overlap_preload:
                batch_prefetcher.start_preload()
            result = train_step(
                model,
                batch,
                cfg,
                step=step,
                update=update,
                valid_prefixes=valid_prefixes,
                trunk_fn=trunk_fn,
                temporal_fn=temporal_fn,
                optimizer=optimizer,
                scheduler=scheduler,
                phase_timer=phase_timer,
            )
            loader_wait = batch_prefetcher.finish_preload() if overlap_preload else 0.0
            actual_positions += valid_prefixes
            metric_accumulator.add(result, valid_prefixes)
            window_loader_wait_seconds.append(loader_wait)
            window_diagnostics.update(result.diagnostics)
            if phase_timer is not None:
                window_phase_timers.append(phase_timer)
            if DEVICE == "cuda":
                window_peak_allocated_gb = max(
                    window_peak_allocated_gb,
                    torch.cuda.max_memory_allocated() / 2**30,
                )

            metrics_due = update % _TRAIN_METRICS_EVERY == 0 or update == run_stop
            if metrics_due:
                window_metric_values, window_updates, window_valid_prefixes = metric_accumulator.flush(
                    cfg,
                    update=update,
                )
                if len(window_loader_wait_seconds) != window_updates:
                    raise RuntimeError("training telemetry window lost an update")
                loader_wait_s = sum(window_loader_wait_seconds) / window_updates
                training_elapsed_wall_s = time.monotonic() - run_started
                completed_updates = update - start_step
                projected_training_remaining_s = training_elapsed_wall_s * (run_stop - update) / completed_updates
                first_window_update = update - window_updates + 1
                log: dict[str, object] = {
                    "data/windows": update * cfg.batch_size,
                    "data/supervised_prefixes": actual_positions,
                    "data/future_targets": actual_positions * len(cfg.arch.head_offsets),
                    "data/epoch": update * cfg.batch_size / train_replays,
                    "data/dropped_windows": (update * cfg.batch_size // train_replays) * 160,
                    "loader/wait_s": loader_wait_s,
                    "progress/elapsed_s": training_elapsed_wall_s,
                    "progress/remaining_s": projected_training_remaining_s,
                    "schedule/muon_lr": result.muon_lr,
                    "schedule/adam_lr": result.adam_lr,
                    **window_metric_values,
                    **window_diagnostics,
                    **identity_masker.metrics(),
                    **_mean_phase_metrics(window_phase_timers),
                }
                loader_metrics = getattr(train_loader, "metrics", None)
                if isinstance(loader_metrics, dict):
                    log.update(loader_metrics)
                if _cadence_in_window(first_window_update, update, cfg.system_metrics_every):
                    log.update(_minimal_system_metrics(host_metrics.snapshot()))
                if DEVICE == "cuda":
                    log["system/gpu_memory_gb"] = window_peak_allocated_gb

                update_s = (time.monotonic() - metrics_window_started) / window_updates
                samples_per_s = cfg.batch_size / update_s
                loader_wait_fraction = loader_wait_s / max(update_s, 1e-12)
                loader_wait_fractions.extend([loader_wait_fraction] * window_updates)
                log["throughput/update_s"] = update_s
                log["throughput/samples_per_s"] = samples_per_s
                if peak_flops is not None:
                    log["throughput/mfu"] = model_flops_utilization(
                        flops_per_update,
                        update_s,
                        peak_flops,
                    )
                wandb_started = time.monotonic()
                _log_wandb({"global_step": update, **log}, arm_guard)
                if update <= _TRAIN_METRICS_EVERY or update % 50 == 0 or update == run_stop:
                    print(
                        f"[t+{time.monotonic() - run_started:.0f}s] update {update}: "
                        f"{window_metric_values['train/loss']:.3f} bits objective, "
                        f"{samples_per_s:.0f} samples/s, "
                        f"projected training remaining {projected_training_remaining_s / 60:.1f}m",
                        flush=True,
                    )
                window_loader_wait_seconds.clear()
                window_phase_timers.clear()
                window_diagnostics.clear()
                window_peak_allocated_gb = 0.0
                metrics_window_started = wandb_started
            checkpoint_path: Path | None = None
            if val_due or eval_request_due or ckpt_due or preflight_boundary:
                checkpoint_path = save_boundary_checkpoint(
                    run_dir,
                    update=update,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    cfg=cfg,
                    uploader=uploader,
                    milestone=preflight_boundary or (update % 8192 == 0),
                    wandb_id=None if wandb.run is None else wandb.run.id,
                    actual_loss_positions=actual_positions,
                    loader_state=_training_loader_state(train_loader, identity_masker),
                    identity_masker_state=identity_masker.state_dict(),
                )
            boundary_metrics: dict[str, float] = {}
            if val_due:
                values = _validation_wandb_metrics(val_metrics(model, val_cache, cfg), cfg)
                boundary_metrics.update({f"val/{name}": value for name, value in values.items()})
            if eval_request_due:
                assert checkpoint_path is not None
                _write_eval_request(
                    run_dir,
                    update,
                    _checkpoint_sha256(checkpoint_path),
                    4 if update == 9 else cfg.eval_n_matchups,
                    final=False,
                    uploader=uploader,
                )
            if boundary_metrics:
                _log_wandb({"global_step": update, **boundary_metrics}, arm_guard)
            if update < run_stop and boundary_due:
                batch_prefetcher.preload()
                metrics_window_started = time.monotonic()
        _finalize_training(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            stats=stats,
            val_cache=val_cache,
            run_dir=run_dir,
            replay_dir=replay_dir,
            uploader=uploader,
            loader_wait_fractions=loader_wait_fractions,
            loader_state=_training_loader_state(train_loader, identity_masker),
            identity_masker_state=identity_masker.state_dict(),
            update=run_stop,
            actual_loss_positions=actual_positions,
            smoke=smoke,
            smoke_eval_matchups=smoke_eval_matchups,
            arm_guard=arm_guard,
        )
    finally:
        batch_prefetcher.close()
        close_loader = getattr(train_loader, "close", None)
        if callable(close_loader):
            close_loader()
        host_metrics.close()
        if uploader is not None:
            uploader.upload_tree(replay_dir, base=run_dir)
            uploader.close()
        wandb.finish()


_ARCHITECTURE_FIELDS = frozenset(item.name for item in fields(Architecture))
_AWR_FIELDS = frozenset(item.name for item in fields(AWRCalibration))
_RUNTIME_CONFIG_FIELDS = frozenset(item.name for item in fields(TrainConfig)) - {"arch", "awr"}


def _upload_eval_evidence(run_name: str, replay_dir: Path) -> None:
    uploader = BackgroundUploader(run_name)
    uploader.upload_tree(replay_dir, base=(Path("runs") / run_name).resolve())
    uploader.close()


def _backfill_eval_metrics(wandb_id: str, update: int, values: dict[str, float]) -> None:
    wandb.init(project="hal", id=wandb_id, resume="must")
    try:
        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")
        wandb.define_metric("eval/*", step_metric="global_step")
        metrics = {f"eval/{name}": value for name, value in _eval_wandb_metrics(values).items()}
        wandb.log({"global_step": update, "eval/backfilled": 1, **metrics})
    finally:
        wandb.finish()


def _log_companion_eval_metrics(wandb_id: str, update: int, values: dict[str, float]) -> None:
    """Log live evals without concurrently resuming the training process."""
    companion_id = f"{wandb_id}-eval96"
    run = wandb.init(
        project="hal",
        id=companion_id,
        resume="allow",
        name=f"{wandb_id} eval96",
        group=wandb_id,
        job_type="evaluation",
        config={"training_wandb_id": wandb_id, "eval_matchups": _PRODUCTION_EVAL_MATCHUPS},
        allow_val_change=True,
    )
    if run is None:
        raise RuntimeError("W&B companion run initialization returned no run")
    try:
        wandb.define_metric("global_step")
        wandb.define_metric("eval/*", step_metric="global_step")
        metrics = {f"eval/{name}": value for name, value in _eval_wandb_metrics(values).items()}
        wandb.log({"global_step": update, **metrics})
        companion_url = run.url
        entity = run.entity
    finally:
        wandb.finish()

    if not entity:
        raise RuntimeError("W&B companion run has no entity")

    training_run = wandb.Api().run(f"{entity}/hal/{wandb_id}")
    training_run.config["eval96_run_url"] = companion_url
    training_run.config["eval96_companion_id"] = companion_id
    training_run.summary["eval96/run_url"] = companion_url
    training_run.summary["eval96/latest_step"] = update
    for name in ("boots", "crashed", "net_stock_per_min", "net_dmg_per_min"):
        if name in values:
            training_run.summary[f"eval96/latest_{name}"] = values[name]
    training_run.update()


def eval_checkpoint(
    path: str,
    *,
    n_matchups: int | None = None,
    eager: bool = False,
    max_parallel: int | None = None,
    output_name: str | None = None,
    upload_run: str | None = None,
    backfill_wandb: bool = False,
    companion_wandb: bool = False,
) -> dict[str, float]:
    if backfill_wandb and companion_wandb:
        raise ValueError("choose either direct W&B backfill or companion W&B logging")
    model, cfg, stats, state = load_checkpoint(path)
    cfg = replace(
        cfg,
        inference_mode="eager" if eager else cfg.inference_mode,
        eval_max_parallel=cfg.eval_max_parallel if max_parallel is None else max_parallel,
    )
    validate_config(cfg)
    horizon = cfg.prediction_frames
    update = int(cast(int, state["step"])) + 1
    if upload_run is not None:
        default_name = f"eval_backfill_step_{update:07d}_s{horizon}"
    else:
        default_name = "eval_replays_s6" if horizon == 6 else "eval_replays"
    if output_name is not None and (Path(output_name).name != output_name or output_name in ("", ".", "..")):
        raise ValueError(f"evaluation output name must be one directory name, got {output_name!r}")
    replay_dir = Path(path).resolve().parent / (default_name if output_name is None else output_name)
    values = eval_vs_cpu(
        model,
        stats,
        cfg,
        n_matchups=cfg.final_eval_n_matchups if n_matchups is None else n_matchups,
        replay_dir=replay_dir,
        checkpoint_sha256=_checkpoint_sha256(Path(path)),
    )
    require_complete_eval(values, cfg.final_eval_n_matchups if n_matchups is None else n_matchups)
    if upload_run is not None:
        _upload_eval_evidence(upload_run, replay_dir)
    if backfill_wandb:
        wandb_id = state.get("wandb_id")
        if not isinstance(wandb_id, str):
            raise RuntimeError("checkpoint has no W&B run id to backfill")
        _backfill_eval_metrics(wandb_id, update, values)
    if companion_wandb:
        wandb_id = state.get("wandb_id")
        if not isinstance(wandb_id, str):
            raise RuntimeError("checkpoint has no W&B run id for companion logging")
        _log_companion_eval_metrics(wandb_id, update, values)
    print(f"[eval] step={update} horizon={horizon}: {values}", flush=True)
    return values


def _resolve_eval_checkpoint(checkpoint: str, run: str | None) -> Path:
    if run is None:
        return Path(checkpoint)
    path = download_latest(run, Path("runs") / run, name=checkpoint)
    if path is None:
        raise SystemExit(f"no {checkpoint!r} for run {run!r}")
    return path


def _remote_run_exists(run_name: str) -> bool:
    """Return whether R2 contains an object for a run name."""
    response = r2.client().list_objects_v2(Bucket=r2.bucket(), Prefix=f"runs/{run_name}/", MaxKeys=1)
    return bool(response.get("KeyCount", len(response.get("Contents", ()))))


@dataclass
class EvalArgs:
    checkpoint: str
    run: str | None = None
    n_matchups: int | None = None
    eager: bool = False
    max_parallel: int | None = None
    output_name: str | None = None
    backfill_wandb: bool = False
    companion_wandb: bool = False


# O51 direct data contract.
TIER_SCALES: Final[tuple[int, ...]] = (1, 2, 4, 8)
OFFICIAL_RAW_REPLAYS: Final[int] = 1_300_640
OFFICIAL_UNIQUE_REPLAYS: Final[int] = 1_300_638
OFFICIAL_CORPUS_TARGETS: Final[int] = 26_582_742_076
OFFICIAL_TIER_REPLAYS: Final[dict[int, int]] = {
    1: 162_598,
    2: 325_176,
    4: 650_331,
    8: 1_300_638,
}
OFFICIAL_TIER_TARGETS: Final[dict[int, int]] = {
    1: 3_353_805_100,
    2: 6_686_081_812,
    4: 13_291_247_716,
    8: 26_582_742_076,
}

CORPUS_SHA256: Final[str] = "e6a83dfb8c98d2cf1ddfacce41c1c2a3e3a8db0a50496f6ecce1c686f5a6ef95"
TIER_SHA256: Final[dict[int, str]] = {
    1: "0df4c7f80d7bcc4ae135a8ec0151a4e50e8cae0a76107dcab4637a9a79b801ce",
    2: "3d57c4836ecd50c9dbaea87434527baef847870ea63286600aa5816770614bb6",
    4: "ce671cc83ca4fe7fd251bf27993c91451ff5812162123528482dbe206629f9d7",
    8: "b8ead0b579b725fa67d6b0a5176251c0057b947886350b61d591cf7b82a0a15c",
}
EXCLUDED_SOURCE_ROWS: Final[dict[str, tuple[int, ...]]] = {
    "professional-monotheon-policy-world-v7": (14_160, 14_163),
}
SOURCE_MANIFEST_SHA256: Final[dict[str, str]] = {
    "professional-aklo-policy-world-v7": "c8635058e791874e76371e3b473fcd0552e6bf69a9f3d30fa35f7967de5a3b79",
    "professional-amsa-policy-world-v7": "c7b213e6ada0bbaec4899b142bfc6deb52716162f07ef3e4dd92fd07165839db",
    "professional-axe-policy-world-v7": "b9568dbb6c4bf931e7c7786ba49bf56614b07c7cb9530857529c0b8fb78df212",
    "professional-billybopeep-policy-world-v7": ("d07fe88191500a77c70e540163a3d645cdd71331a0cbfd4d4b5f9660f4681b52"),
    "professional-bobbybigballz-policy-world-v7": ("670a8920ea533f83d3e8e3bae9fe6a935161657570a598d41752e79f1b802e4c"),
    "professional-cody-policy-world-v7": "d931e14d275451f61a4bcf11af30fb435ff7f4a7b02e8da32778cfc6ab972311",
    "professional-cookbook-policy-world-v7": "6380ebefced8a37154f85780a0eb4ec975c1bd32f70d3bfacb0759037981b1fc",
    "professional-daniel-policy-world-v7": "08496d5eaa8819065303ee9e12e7e29f5192a0c81115727db94e8e9f3c4138f0",
    "professional-desertsnoopy-policy-world-v7": ("0694ee88c6d63993be361442159e9b4a814551bbc1339bfaa85fe15a2d5ca0ef"),
    "professional-druggedfox-policy-world-v7": ("5f382cb09714eb3cb369bc02991ac36db3665efa79483dd92c8c031c8ac8346b"),
    "professional-fknsilver-policy-world-v7": ("92af994560b698ae4c3e5a7dcccdbe6e0897a946f0bb6b21893542875573449b"),
    "professional-franz-policy-world-v7": "0e2972e21fb0dd536022ddd07aaf57a2cd5c8ece412c000532fac606ade02705",
    "professional-frenzy-policy-world-v7": "a78ce331906741bee9841bea20821e1da49dc557ffe4545dcfe67450c8ef435a",
    "professional-friend-policy-world-v7": "012e9f2878f50813984b43dd612cd4fe7a9d8a7e1ad3f9df4ecb0ac457ac6443",
    "professional-ginger-policy-world-v7": "5fce487fb31b7c954f396d7b0bc47f2254d464ada683a92ad7abc88b07d41c29",
    "professional-gosu-policy-world-v7": "410f7456656a72158be5326d247e69a731a6e98a85819ca97a794bc1f9711e4d",
    "professional-grab2win-policy-world-v7": ("3a386a12110b407ec0158a62c941c5a10bfb92a7f0d5b67cc5e95cc9f25f07d3"),
    "professional-iliketurtles-policy-world-v7": ("ba241befd7cafc216d60cdf4b9c357bf5931c51b4ea17213927961cd4f12b3f9"),
    "professional-isdsar-policy-world-v7": "9f4b6f543e9a0a3bcdab5d4304d8bbfc6b31992fc20f2fd6f9478c01920352ca",
    "professional-jahridin-policy-world-v7": "428fa4e7051925fecf8fb0545f5cbb84f10524f965fed92403f1c19ae1c170e9",
    "professional-jchu-policy-world-v7": "9096fcd2424135280c9f0163ae9c04c2fdffa4dbc71f4ca9fdb9472f65a1f4ab",
    "professional-kjh-policy-world-v7": "09d78720633ca3809ccbeb92be2a5f377cc173584435cc7e054dfc5c4b1d6d03",
    "professional-kodorin-policy-world-v7": "8b0d9809da17d1be49f4c8eb5e0a82061b992f84dfbf33636bbeebe94118dae0",
    "professional-krudo-policy-world-v7": "e3b572be20e6c6ee6d249c669a2559b69d22ce24fae653001deb04cc16fc7f54",
    "professional-m2k-policy-world-v7": "b372dd7ba6a46ee01b977f9cf992d18a08e9213354796b45a1449b7f1ff2a5aa",
    "professional-mang0-policy-world-v7": "f78a84bdf70885fb691a202933b936b1921bbf74b8daaea92cffb4d21250ac21",
    "professional-mof-policy-world-v7": "1ee81b47cc8dddf1f7ebae5f9139f00a406188bc7b7c75638a9fcd1e9476e030",
    "professional-monotheon-policy-world-v7": ("9ff5057f8826ef78e1d31f3b687cd6cadc7d66d1656ba2cd98c8e396dd68c38a"),
    "professional-nicki-policy-world-v7": "738157bed96c6843b5b9825b4abdb3000a4faa82c210ead17121b61120250034",
    "professional-rapm-policy-world-v7": "31e0ee64a551c5dffb23ad8dca7a18ff713fe011fb7f55d5b8b6e347479048cc",
    "professional-redx-policy-world-v7": "4bf776186d9470b0280b951ff9a7eab804a4354c277ef855e16518ff3aba0cc2",
    "professional-siddward-policy-world-v7": ("c4c483c0165b379089ee1f20084a2c46107d9faf0a5e7c58c50f53614c94bf8b"),
    "professional-solobattle-policy-world-v7": ("4c29eab3e77c7fdd56f8a816733af2f837c82b560f8f77d2291ae50e20bd993e"),
    "professional-technospider-policy-world-v7": ("624f3e27df17c043b7368ca56a758742dbf827f4777244c6f6c7c972812222e2"),
    "professional-trif-policy-world-v7": "55c0a5154159ebded5eaece1efd3a9b01f1d8cf63c68b008c22f8d84ac58b37c",
    "professional-uhhei-policy-world-v7": "2fcbf81b2afc3bc7f9e21a22fa8ec93f809903fb594b9ddbc8666fcc4be65e64",
    "professional-ycz-policy-world-v7": "7e97cb1c4dd6b75afd0d65c69f0dfb05b582e5349e391b9ff53f75c48700554a",
    "professional-zain-policy-world-v7": "0a3e39757fd68d4aabd37831f63f5b8e0e9aa9dafa580fa7ee5494b445c9ee12",
    "ranked-anonymized-1-policy-world-v7": "a563c62603b8cfdef219cd133324cb090f7e6488fcfc4b6941d03e863a255d16",
    "ranked-anonymized-2-policy-world-v7": "f2bb6f33d53bdcffb0038dca83e926c8102d9f6b82cf0f3a346064e93ec73d29",
    "ranked-anonymized-3-policy-world-v7": "8e68c15dfbe898e44c7104c355ae86cbac61ba3bbf28c711266405fd5a633694",
    "ranked-anonymized-4-policy-world-v7": "946d76f03282e5b23ef4cee4552c6dabd2668423408ae1f7510ee9a0671e31f8",
    "ranked-anonymized-5-policy-world-v7": "ebbb72a7a4e26221ee20a187b0ac60684657b70c7188cab26c81ec4be29ed8e3",
    "ranked-anonymized-6-policy-world-v7": "471a945aab225483735db9dbef05cfd26e215178630115e4753a3312116bb019",
}
O51_RETURN_SUFFIX: Final[str] = "awr_return"


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


@dataclass(frozen=True, slots=True)
class SourceSlice:
    """The selected rows from one existing source MDS."""

    source: str
    stop: int
    excluded_rows: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("source name must not be empty")
        if self.stop < 1:
            raise ValueError(f"source slice stop must be positive for {self.source}")
        if self.excluded_rows != tuple(sorted(set(self.excluded_rows))):
            raise ValueError(f"excluded rows are not sorted and unique for {self.source}")
        if any(not 0 <= row < self.stop for row in self.excluded_rows):
            raise ValueError(f"excluded row is outside {self.source}[0:{self.stop}]")

    @property
    def unique_replays(self) -> int:
        return self.stop - len(self.excluded_rows)


@dataclass(frozen=True, slots=True)
class TierSelection:
    """The direct source slices for one O51 data tier."""

    scale: int
    sources: tuple[SourceSlice, ...]
    potential_targets: int
    sha256: str

    def __post_init__(self) -> None:
        if self.scale not in TIER_SCALES:
            raise ValueError(f"invalid tier scale {self.scale}")
        if not self.sources or len({source.source for source in self.sources}) != len(self.sources):
            raise ValueError(f"U{self.scale} source slices are empty or repeated")
        if self.potential_targets < 1 or self.potential_targets % 2:
            raise ValueError(f"U{self.scale} potential-target count is invalid")
        if not _is_sha256(self.sha256):
            raise ValueError(f"U{self.scale} selection hash is invalid")

    @property
    def unique_replays(self) -> int:
        return sum(source.unique_replays for source in self.sources)

    @property
    def frames(self) -> int:
        return self.potential_targets // 2 + self.unique_replays

    def source_replay_counts(self) -> dict[str, int]:
        return {source.source: source.unique_replays for source in self.sources}


@dataclass(frozen=True, slots=True)
class CorpusSelection:
    """All pinned O51 views of the existing policy-world-v7 sources."""

    corpus_hash: str
    source_manifest_sha256: Mapping[str, str]
    tiers: Mapping[int, TierSelection]

    def tier(self, scale: int) -> TierSelection:
        try:
            return self.tiers[scale]
        except KeyError as error:
            raise ValueError(f"tier scale must be one of {TIER_SCALES}, got {scale}") from error


def _prefix_stop(unique_replays: int, excluded_rows: tuple[int, ...]) -> int:
    """Convert a unique-row count to an exclusive raw-row cutoff."""
    stop = unique_replays
    for row in excluded_rows:
        if row < stop:
            stop += 1
    return stop


def corpus_selection() -> CorpusSelection:
    """Return the four pinned tiers without reading or copying replay data."""
    source_names = tuple(source.name for source in streams.POLICY_WORLD_V7_SOURCES)
    raw_counts = streams.POLICY_WORLD_V7_TRAIN_REPLAYS
    if set(raw_counts) != set(source_names):
        raise RuntimeError("policy-world-v7 replay counts do not cover all 44 sources")
    if sum(raw_counts.values()) != OFFICIAL_RAW_REPLAYS:
        raise RuntimeError("policy-world-v7 replay count changed")
    if set(SOURCE_MANIFEST_SHA256) != set(source_names):
        raise RuntimeError("pinned source manifests do not cover all 44 sources")
    if any(not _is_sha256(digest) for digest in SOURCE_MANIFEST_SHA256.values()):
        raise RuntimeError("a pinned source manifest hash is invalid")
    if sum(map(len, EXCLUDED_SOURCE_ROWS.values())) != OFFICIAL_RAW_REPLAYS - OFFICIAL_UNIQUE_REPLAYS:
        raise RuntimeError("the duplicate-row sidecar must exclude exactly two rows")

    tiers: dict[int, TierSelection] = {}
    for scale in TIER_SCALES:
        slices: list[SourceSlice] = []
        for source in source_names:
            excluded = EXCLUDED_SOURCE_ROWS.get(source, ())
            unique_source_replays = raw_counts[source] - len(excluded)
            selected_replays = (scale * unique_source_replays + 7) // 8
            stop = _prefix_stop(selected_replays, excluded)
            selected_exclusions = tuple(row for row in excluded if row < stop)
            source_slice = SourceSlice(source, stop, selected_exclusions)
            if source_slice.unique_replays != selected_replays:
                raise RuntimeError(f"U{scale} slice accounting failed for {source}")
            slices.append(source_slice)
        tier = TierSelection(
            scale=scale,
            sources=tuple(slices),
            potential_targets=OFFICIAL_TIER_TARGETS[scale],
            sha256=TIER_SHA256[scale],
        )
        if tier.unique_replays != OFFICIAL_TIER_REPLAYS[scale]:
            raise RuntimeError(f"U{scale} replay count changed")
        tiers[scale] = tier

    full_exclusions = {source.source: source.excluded_rows for source in tiers[8].sources if source.excluded_rows}
    if full_exclusions != EXCLUDED_SOURCE_ROWS:
        raise RuntimeError("the full direct view does not contain the duplicate-row sidecar")
    return CorpusSelection(
        corpus_hash=CORPUS_SHA256,
        source_manifest_sha256={name: SOURCE_MANIFEST_SHA256[name] for name in source_names},
        tiers=tiers,
    )


@dataclass(frozen=True, slots=True)
class ParameterizationReplayLabels:
    """Compute O51 returns and player IDs after reading an existing MDS row."""

    player_lookup: ReplayPlayerLookup
    gamma: float
    damage_shaping: float
    win_reward: float
    stock_value: float

    def __call__(self, compact: Mapping[str, object]) -> dict[str, np.ndarray]:
        labels = returns_lib.compact_policy_returns(
            compact,
            gamma=self.gamma,
            damage_shaping=self.damage_shaping,
            win_reward=self.win_reward,
            stock_value=self.stock_value,
            suffix=O51_RETURN_SUFFIX,
        )
        p1_id, p2_id = self.player_lookup.ids(compact)
        labels["p1_player_id"] = np.asarray(p1_id, dtype=np.int32)
        labels["p2_player_id"] = np.asarray(p2_id, dtype=np.int32)
        return labels


# O51 sweep contract.
EXPERIMENT = "experiments/051_muon_parameterization.py"
INIT_STD_GRID = (0.5, 1.0, 2.0)
READOUT_GRID = ("zero", "mup-normal")
MUON_LR_GRID = (0.007, 0.014, 0.028)
ADAM_LR_GRID = (1.0625e-4, 2.125e-4, 4.25e-4)
DECAY_GRID = (0.0, 0.001, 0.01)
BATCH_GRID = (128, 256, 512, 1024)
SUPERVISED_POSITIONS_PER_WINDOW = 128
GRID_EVAL_MATCHUPS = 96
_ARM_ID = re.compile(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?")

Stage = Literal[
    "initialization-screen",
    "initialization-extension",
    "lr",
    "decay",
    "batch",
    "proxy-transfer",
    "mid-search",
    "seed-repeat",
    "duration",
]
GridStage = Literal["lr", "decay"]

TERMINAL_RUN_STATES = frozenset({"finished", "failed", "crashed", "killed"})


@dataclass(frozen=True, slots=True)
class Treatment:
    """One selected center passed between O51 sweep stages."""

    hidden_std_multiplier: float = 1.0
    readout_init: Literal["zero", "mup-normal"] = "zero"
    muon_lr: float = 0.028
    adam_lr: float = 4.25e-4
    muon_weight_decay: float = 0.001
    adam_weight_decay: float = 0.001
    batch_size: int = 512
    depth_alpha: float = 0.5
    muon_batch_scaling: Literal["fixed", "sqrt"] = "fixed"
    muon_duration_scaling: Literal["fixed", "inverse-sqrt"] = "fixed"
    compile_mode: Literal["reduce-overhead", "max-autotune"] = "reduce-overhead"
    temporal_attention_chunk: int | None = 16_384
    num_workers: int = 16

    def __post_init__(self) -> None:
        positive_floats = {
            "hidden_std_multiplier": self.hidden_std_multiplier,
            "muon_lr": self.muon_lr,
            "adam_lr": self.adam_lr,
        }
        invalid_positive = sorted(
            name
            for name, value in positive_floats.items()
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value <= 0
        )
        if invalid_positive:
            raise ValueError(f"treatment values must be positive and finite: {invalid_positive}")
        decays = {
            "muon_weight_decay": self.muon_weight_decay,
            "adam_weight_decay": self.adam_weight_decay,
        }
        invalid_decays = sorted(
            name
            for name, value in decays.items()
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0
        )
        if invalid_decays:
            raise ValueError(f"treatment decays must be non-negative and finite: {invalid_decays}")
        if self.hidden_std_multiplier not in INIT_STD_GRID:
            raise ValueError(f"hidden_std_multiplier must be one of {INIT_STD_GRID}")
        if self.readout_init not in READOUT_GRID:
            raise ValueError(f"readout_init must be one of {READOUT_GRID}")
        if self.batch_size not in BATCH_GRID:
            raise ValueError("batch_size is outside the O51 sweep")
        if self.depth_alpha not in (0.5, 1.0):
            raise ValueError("depth_alpha must be 0.5 or 1.0")
        if self.muon_batch_scaling not in ("fixed", "sqrt"):
            raise ValueError("muon_batch_scaling must be fixed or sqrt")
        if self.muon_duration_scaling not in ("fixed", "inverse-sqrt"):
            raise ValueError("muon_duration_scaling must be fixed or inverse-sqrt")
        if self.compile_mode not in ("reduce-overhead", "max-autotune"):
            raise ValueError("compile_mode is outside the O51 preflight grid")
        if self.temporal_attention_chunk not in (8192, 16_384, 32_768, None):
            raise ValueError("temporal_attention_chunk is outside the O51 preflight grid")
        if self.num_workers not in (8, 16, 24, 32):
            raise ValueError("num_workers is outside the O51 preflight grid")

    @classmethod
    def load(cls, path: Path) -> Treatment:
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f"O51 treatment must be a JSON object: {path}")
        expected = {field.name for field in fields(cls)}
        extra = sorted(payload.keys() - expected)
        if extra:
            raise ValueError(f"O51 treatment has unknown fields: {extra}")
        return cls(**payload)

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")
        temporary.replace(path)


@dataclass(frozen=True, slots=True)
class SweepArm:
    """One independently launchable O51 training arm."""

    arm_id: str
    stage: Stage
    level: Literal["base", "proxy", "mid", "large"]
    treatment: Treatment
    target_positions: int = D0
    tier_scale: int = 1
    seed: int = 0
    stop_after_update: int | None = None

    def __post_init__(self) -> None:
        if not _ARM_ID.fullmatch(self.arm_id):
            raise ValueError(f"invalid sweep arm ID: {self.arm_id!r}")
        if (
            not isinstance(self.tier_scale, int)
            or isinstance(self.tier_scale, bool)
            or self.tier_scale not in TIER_SCALES
            or not isinstance(self.target_positions, int)
            or isinstance(self.target_positions, bool)
            or self.target_positions != self.tier_scale * D0
        ):
            raise ValueError("each sweep arm must use an exact matched D/U endpoint")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        positions_per_update = self.treatment.batch_size * SUPERVISED_POSITIONS_PER_WINDOW
        max_updates, remainder = divmod(self.target_positions, positions_per_update)
        if remainder:
            raise ValueError("target positions must end on an optimizer update")
        if self.stop_after_update is not None and (
            not isinstance(self.stop_after_update, int)
            or isinstance(self.stop_after_update, bool)
            or not 1 <= self.stop_after_update < max_updates
        ):
            raise ValueError("stop_after_update must select an early optimizer update")
        if self.stage == "initialization-screen" and self.stop_after_update is None:
            raise ValueError("initialization-screen arms require an early stop")
        if self.stage != "initialization-screen" and self.stop_after_update is not None:
            raise ValueError("only initialization-screen arms can stop early")

    @property
    def requires_preflight(self) -> bool:
        """Return whether the O51 train command requires launch evidence."""
        return self.stop_after_update is None

    def argv(self, *, preflight_report: Path | None = None) -> tuple[str, ...]:
        if self.stop_after_update is not None and preflight_report is not None:
            raise ValueError("initialization-screen arms do not use a preflight report")
        cfg = {
            "target-positions": self.target_positions,
            "tier-scale": self.tier_scale,
            "seed": self.seed,
            "hidden-std-multiplier": self.treatment.hidden_std_multiplier,
            "readout-init": self.treatment.readout_init,
            "muon-lr": self.treatment.muon_lr,
            "adam-lr": self.treatment.adam_lr,
            "muon-weight-decay": self.treatment.muon_weight_decay,
            "adam-weight-decay": self.treatment.adam_weight_decay,
            "batch-size": self.treatment.batch_size,
            "depth-alpha": self.treatment.depth_alpha,
            "muon-batch-scaling": self.treatment.muon_batch_scaling,
            "muon-duration-scaling": self.treatment.muon_duration_scaling,
            "compile-mode": self.treatment.compile_mode,
            "temporal-attention-chunk": self.treatment.temporal_attention_chunk,
            "num-workers": self.treatment.num_workers,
        }
        command = [
            "uv",
            "run",
            EXPERIMENT,
            "train",
            "--level",
            self.level,
            "--comment",
            self.arm_id,
        ]
        if self.stop_after_update is not None:
            command.extend(
                (
                    "--smoke",
                    "--smoke-eval-matchups",
                    "0",
                    "--stop-after-update",
                    str(self.stop_after_update),
                )
            )
        if preflight_report is not None:
            command.extend(("--preflight-report", str(preflight_report)))
        for name, value in cfg.items():
            command.extend((f"--cfg.{name}", "None" if value is None else str(value)))
        return tuple(command)


@dataclass(frozen=True, slots=True)
class ValidationOutcome:
    """Final validation evidence for one declared sweep arm."""

    arm_id: str
    run_path: str
    state: str
    processed_positions: float | None
    final_update: float | None
    val_nll: float | None
    val_far_nll: float | None
    val_rollout_nll: float | None


@dataclass(frozen=True, slots=True)
class ValidationSelection:
    """Deterministic validation ranking and its selected treatment."""

    winner: SweepArm
    ranking: tuple[ValidationOutcome, ...]
    excluded: tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True, slots=True)
class ClosedLoopOutcome:
    """Final 96-matchup evidence for one validation-screened arm."""

    arm_id: str
    run_path: str
    state: str
    final_update: float | None
    boots: float | None
    crashed: float | None
    net_stock_per_min: float | None
    net_stock_lcb: float | None
    net_dmg_per_min: float | None


@dataclass(frozen=True, slots=True)
class ClosedLoopSelection:
    """Deterministic closed-loop ranking of a grid's two finalists."""

    winner: SweepArm
    ranking: tuple[ClosedLoopOutcome, ...]


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def _outcome_failures(arm: SweepArm, outcome: ValidationOutcome) -> tuple[str, ...]:
    failures: list[str] = []
    if outcome.state != "finished":
        failures.append(f"run state is {outcome.state}")
    if outcome.processed_positions != arm.target_positions:
        failures.append("run did not reach its exact D endpoint")
    expected_update = arm.target_positions // (arm.treatment.batch_size * SUPERVISED_POSITIONS_PER_WINDOW)
    if outcome.final_update != expected_update:
        failures.append("run did not reach its final optimizer update")
    for name in ("val_nll", "val_far_nll", "val_rollout_nll"):
        if not _finite(getattr(outcome, name)):
            failures.append(f"{name} is missing or non-finite")
    return tuple(failures)


def select_validation_winner(
    arms: tuple[SweepArm, ...],
    outcomes: dict[str, ValidationOutcome],
) -> ValidationSelection:
    """Select by final NLL, with fixed far-NLL and rollout-NLL tie breaks."""
    if not arms:
        raise ValueError("validation selection needs at least one sweep arm")
    expected = {arm.arm_id for arm in arms}
    actual = set(outcomes)
    if expected != actual:
        raise ValueError(
            f"validation outcomes do not match the stage: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    nonterminal = sorted(outcome.arm_id for outcome in outcomes.values() if outcome.state not in TERMINAL_RUN_STATES)
    if nonterminal:
        raise ValueError(f"validation runs are not terminal: {nonterminal}")

    by_id = {arm.arm_id: arm for arm in arms}
    eligible: list[ValidationOutcome] = []
    excluded: list[tuple[str, tuple[str, ...]]] = []
    for arm in arms:
        outcome = outcomes[arm.arm_id]
        failures = _outcome_failures(arm, outcome)
        if failures:
            excluded.append((arm.arm_id, failures))
        else:
            eligible.append(outcome)
    if not eligible:
        raise ValueError("no sweep arm completed with valid final validation evidence")
    eligible.sort(
        key=lambda outcome: (
            outcome.val_nll,
            outcome.val_far_nll,
            outcome.val_rollout_nll,
            outcome.arm_id,
        )
    )
    return ValidationSelection(
        winner=by_id[eligible[0].arm_id],
        ranking=tuple(eligible),
        excluded=tuple(excluded),
    )


def select_closed_loop_winner(
    arms: tuple[SweepArm, ...],
    validation: ValidationSelection,
    outcomes: dict[str, ClosedLoopOutcome],
) -> ClosedLoopSelection:
    """Adjudicate the two best validation arms by complete closed-loop evidence."""
    finalists = validation.ranking[:2]
    if len(finalists) != 2:
        raise ValueError("closed-loop adjudication needs two eligible validation finalists")
    expected = {outcome.arm_id for outcome in finalists}
    actual = set(outcomes)
    if actual != expected:
        raise ValueError(
            f"closed-loop outcomes do not match the top two validation arms: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )

    by_id = {arm.arm_id: arm for arm in arms}
    validation_by_id = {outcome.arm_id: outcome for outcome in finalists}
    failures: list[str] = []
    for arm_id in sorted(expected):
        arm = by_id[arm_id]
        outcome = outcomes[arm_id]
        final_update = arm.target_positions // (arm.treatment.batch_size * SUPERVISED_POSITIONS_PER_WINDOW)
        if outcome.state != "finished":
            failures.append(f"{arm_id}: evaluation state is {outcome.state}")
        if outcome.final_update != final_update:
            failures.append(f"{arm_id}: evaluation does not target the final checkpoint")
        if outcome.boots != GRID_EVAL_MATCHUPS:
            failures.append(f"{arm_id}: evaluation did not complete {GRID_EVAL_MATCHUPS} boots")
        if outcome.crashed != 0:
            failures.append(f"{arm_id}: evaluation had crashes")
        for name in ("net_stock_per_min", "net_stock_lcb", "net_dmg_per_min"):
            if not _finite(getattr(outcome, name)):
                failures.append(f"{arm_id}: {name} is missing or non-finite")
    if failures:
        raise ValueError("invalid closed-loop evidence: " + "; ".join(failures))

    ranking = sorted(
        outcomes.values(),
        key=lambda outcome: (
            -float(outcome.net_stock_lcb),
            -float(outcome.net_stock_per_min),
            -float(outcome.net_dmg_per_min),
            validation_by_id[outcome.arm_id].val_nll,
            outcome.arm_id,
        ),
    )
    return ClosedLoopSelection(winner=by_id[ranking[0].arm_id], ranking=tuple(ranking))


def _safe_float(value: float) -> str:
    return f"{value:g}".replace("-", "n").replace(".", "p")


def _arm(
    stage: Stage,
    suffix: str,
    level: Literal["base", "proxy", "mid", "large"],
    treatment: Treatment,
    *,
    target_positions: int = D0,
    tier_scale: int = 1,
    seed: int = 0,
    stop_after_update: int | None = None,
) -> SweepArm:
    return SweepArm(
        f"o51-{stage}-{suffix}",
        stage,
        level,
        treatment,
        target_positions=target_positions,
        tier_scale=tier_scale,
        seed=seed,
        stop_after_update=stop_after_update,
    )


def initialization_screen_arms(center: Treatment | None = None) -> tuple[SweepArm, ...]:
    """Six base-model arms stopped at D0/8 while retaining a D0/U0 contract."""
    center = center or Treatment()
    updates = (D0 // 8) // (center.batch_size * 128)
    return tuple(
        _arm(
            "initialization-screen",
            f"h{_safe_float(hidden)}-{readout}",
            "base",
            replace(center, hidden_std_multiplier=hidden, readout_init=readout),
            stop_after_update=updates,
        )
        for hidden in INIT_STD_GRID
        for readout in READOUT_GRID
    )


def initialization_extension_arms(center: Treatment) -> tuple[SweepArm, ...]:
    """Run one selected initialization treatment on the 16-layer proxy."""
    suffix = f"h{_safe_float(center.hidden_std_multiplier)}-{center.readout_init}"
    return (_arm("initialization-extension", suffix, "proxy", center),)


def lr_arms(center: Treatment) -> tuple[SweepArm, ...]:
    return tuple(
        _arm(
            "lr",
            f"m{_safe_float(muon)}-a{_safe_float(adam)}",
            "proxy",
            replace(center, muon_lr=muon, adam_lr=adam),
        )
        for muon in MUON_LR_GRID
        for adam in ADAM_LR_GRID
    )


def decay_arms(center: Treatment) -> tuple[SweepArm, ...]:
    return tuple(
        _arm(
            "decay",
            f"m{_safe_float(muon)}-a{_safe_float(adam)}",
            "proxy",
            replace(center, muon_weight_decay=muon, adam_weight_decay=adam),
        )
        for muon in DECAY_GRID
        for adam in DECAY_GRID
    )


def batch_arms(center: Treatment) -> tuple[SweepArm, ...]:
    return tuple(
        _arm(
            "batch",
            f"b{batch}-muon-{rule}",
            "proxy",
            replace(center, batch_size=batch, muon_batch_scaling=rule),
        )
        for batch in BATCH_GRID
        for rule in ("fixed", "sqrt")
    )


def proxy_transfer_arms(center: Treatment) -> tuple[SweepArm, ...]:
    return tuple(
        _arm(
            "proxy-transfer",
            f"alpha-{_safe_float(alpha)}",
            "proxy",
            replace(center, depth_alpha=alpha),
        )
        for alpha in (0.5, 1.0)
    )


def mid_search_arms(center: Treatment) -> tuple[SweepArm, ...]:
    if center.muon_weight_decay not in DECAY_GRID or center.adam_weight_decay not in DECAY_GRID:
        raise ValueError("mid-search center decays must come from the O51 decay grid")
    candidates = [
        ("center", center),
        ("muon-half", replace(center, muon_lr=center.muon_lr / 2)),
        ("muon-double", replace(center, muon_lr=center.muon_lr * 2)),
        ("adam-half", replace(center, adam_lr=center.adam_lr / 2)),
        ("adam-double", replace(center, adam_lr=center.adam_lr * 2)),
    ]
    candidates.extend(
        (f"muon-decay-{_safe_float(value)}", replace(center, muon_weight_decay=value))
        for value in DECAY_GRID
        if value != center.muon_weight_decay
    )
    candidates.extend(
        (f"adam-decay-{_safe_float(value)}", replace(center, adam_weight_decay=value))
        for value in DECAY_GRID
        if value != center.adam_weight_decay
    )
    return tuple(_arm("mid-search", name, "mid", treatment) for name, treatment in candidates)


def seed_repeat_arms(center: Treatment) -> tuple[SweepArm, ...]:
    return tuple(_arm("seed-repeat", f"seed-{seed}", "mid", center, seed=seed) for seed in (1, 2))


def duration_arms(center: Treatment) -> tuple[SweepArm, ...]:
    return tuple(
        _arm(
            "duration",
            f"s{scale}-{rule}",
            "proxy",
            replace(center, muon_duration_scaling=rule),
            target_positions=scale * D0,
            tier_scale=scale,
        )
        for scale in (2, 4, 8)
        for rule in ("fixed", "inverse-sqrt")
    )


def stage_arms(stage: Stage, center: Treatment) -> tuple[SweepArm, ...]:
    generators: dict[Stage, Callable[[], tuple[SweepArm, ...]]] = {
        "initialization-screen": lambda: initialization_screen_arms(center),
        "initialization-extension": lambda: initialization_extension_arms(center),
        "lr": lambda: lr_arms(center),
        "decay": lambda: decay_arms(center),
        "batch": lambda: batch_arms(center),
        "proxy-transfer": lambda: proxy_transfer_arms(center),
        "mid-search": lambda: mid_search_arms(center),
        "seed-repeat": lambda: seed_repeat_arms(center),
        "duration": lambda: duration_arms(center),
    }
    return generators[stage]()


# O51 sweep operations.
ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "launch_modal.py"
DEFAULT_STATE = ROOT / "runs" / "o51-sweep-v3-16l" / "launches.jsonl"
APP_PATTERN = re.compile(r"submitted Modal App (ap-[A-Za-z0-9]+), Function call (fc-[A-Za-z0-9]+)")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
STATE_SCHEMA = 2

LaunchEvent = Literal["launching", "failed", "uncertain", "launched"]


@dataclass(frozen=True, slots=True)
class PlanArgs:
    stage: Stage
    treatment: Path | None = None
    preflight_reports: Path | None = None
    """JSON object that maps production arm IDs, or ``*``, to report paths."""


@dataclass(frozen=True, slots=True)
class LaunchArgs:
    stage: Stage
    treatment: Path | None = None
    preflight_reports: Path | None = None
    """JSON object that maps production arm IDs, or ``*``, to report paths."""
    state: Path = DEFAULT_STATE
    max_arms: int | None = None
    gpu: str = "B200"
    disk_gib: int = 2048
    dry_run: bool = False


@dataclass(frozen=True, slots=True)
class SelectArgs:
    """Select a completed stage by its fixed validation rule."""

    stage: Stage
    runs: Path
    """JSON object that maps every arm ID to its full W&B run path."""
    output: Path
    """New treatment JSON for the next stage."""
    evidence: Path
    """New durable JSON record of the complete ranking."""
    treatment: Path | None = None


@dataclass(frozen=True, slots=True)
class RankArgs:
    """Screen a 3x3 grid to its two closed-loop finalists."""

    stage: GridStage
    runs: Path
    """JSON object that maps every arm ID to its full W&B training-run path."""
    evidence: Path
    """New durable JSON record of the validation ranking and finalists."""
    treatment: Path | None = None


@dataclass(frozen=True, slots=True)
class AdjudicateArgs:
    """Select a 3x3 grid winner from two complete closed-loop evaluations."""

    stage: GridStage
    runs: Path
    """JSON object that maps every arm ID to its full W&B training-run path."""
    evaluations: Path
    """JSON object that maps finalists to W&B runs containing closed-loop metrics."""
    output: Path
    """New treatment JSON for the next stage."""
    evidence: Path
    """New durable JSON record of validation screening and closed-loop ranking."""
    treatment: Path | None = None


@dataclass(frozen=True, slots=True)
class PreparedArm:
    arm: SweepArm
    preflight_report: Path | None
    train_argv: tuple[str, ...]
    spec_sha256: str


def _treatment(path: Path | None) -> Treatment:
    return Treatment() if path is None else Treatment.load(path)


def _preflight_reports(path: Path | None) -> dict[str, Path]:
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read the preflight report map {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("the preflight report map must be a JSON object")
    reports: dict[str, Path] = {}
    for arm_id, report in payload.items():
        if not isinstance(arm_id, str) or not arm_id:
            raise ValueError("each preflight report key must be a non-empty arm ID")
        if not isinstance(report, str) or not report:
            raise ValueError(f"the preflight report path for {arm_id!r} must be a non-empty string")
        reports[arm_id] = Path(report)
    return reports


def _run_paths(path: Path, arms: tuple[SweepArm, ...]) -> dict[str, str]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read the W&B run map {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("the W&B run map must be a JSON object")
    expected = {arm.arm_id for arm in arms}
    actual = set(payload)
    if expected != actual:
        raise ValueError(
            f"W&B run map does not match the stage: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    invalid = sorted(
        arm_id for arm_id, run_path in payload.items() if not isinstance(run_path, str) or run_path.count("/") != 2
    )
    if invalid:
        raise ValueError(f"W&B run paths must be entity/project/run_id: {invalid}")
    return cast(dict[str, str], payload)


def _evaluation_paths(path: Path, finalists: tuple[str, str]) -> dict[str, str]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read the W&B evaluation map {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("the W&B evaluation map must be a JSON object")
    expected = set(finalists)
    actual = set(payload)
    if expected != actual:
        raise ValueError(
            f"W&B evaluation map does not match the finalists: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    invalid = sorted(
        arm_id for arm_id, run_path in payload.items() if not isinstance(run_path, str) or run_path.count("/") != 2
    )
    if invalid:
        raise ValueError(f"W&B evaluation paths must be entity/project/run_id: {invalid}")
    return cast(dict[str, str], payload)


def _run_config_mismatches(arm: SweepArm, run: Any) -> tuple[str, ...]:
    expected = {
        **asdict(arm.treatment),
        "target_positions": arm.target_positions,
        "tier_scale": arm.tier_scale,
        "seed": arm.seed,
        "data_protocol": DATA_PROTOCOL,
        "adam_eps": 1e-12,
    }
    config = dict(run.config)
    mismatches = [
        name
        for name, value in expected.items()
        if json.dumps(config.get(name), sort_keys=True) != json.dumps(value, sort_keys=True)
    ]
    name = str(run.name)
    if not name.endswith(f"__{arm.arm_id}"):
        mismatches.append("run_name")
    if f"_o51-{arm.level}-" not in name:
        mismatches.append("model_level")
    return tuple(sorted(mismatches))


def _optional_float(value: object) -> float | None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    return float(value)


def _validation_outcome(arm: SweepArm, run_path: str, run: Any) -> ValidationOutcome:
    mismatches = _run_config_mismatches(arm, run)
    if mismatches:
        raise ValueError(f"W&B run {run_path} does not match {arm.arm_id}: {list(mismatches)}")
    summary = dict(run.summary)
    return ValidationOutcome(
        arm_id=arm.arm_id,
        run_path=run_path,
        state=str(run.state),
        processed_positions=_optional_float(summary.get("data/processed_loss_positions")),
        final_update=_optional_float(summary.get("global_step")),
        val_nll=_optional_float(summary.get("val/nll")),
        val_far_nll=_optional_float(summary.get("val/far_nll")),
        val_rollout_nll=_optional_float(summary.get("val/rollout_nll")),
    )


def _closed_loop_outcome(
    arm: SweepArm,
    training_run_path: str,
    evaluation_run_path: str,
    run: Any,
) -> ClosedLoopOutcome:
    config = dict(run.config)
    summary = dict(run.summary)
    mismatches: list[str] = []
    if evaluation_run_path == training_run_path:
        if summary.get("eval/backfilled") != 1:
            mismatches.append("eval/backfilled")
    else:
        expected_training_id = training_run_path.rsplit("/", maxsplit=1)[1]
        if config.get("training_wandb_id") != expected_training_id:
            mismatches.append("training_wandb_id")
        if config.get("eval_matchups") != GRID_EVAL_MATCHUPS:
            mismatches.append("eval_matchups")
    if mismatches:
        raise ValueError(f"W&B evaluation {evaluation_run_path} does not match {arm.arm_id}: {sorted(mismatches)}")
    return ClosedLoopOutcome(
        arm_id=arm.arm_id,
        run_path=evaluation_run_path,
        state=str(run.state),
        final_update=_optional_float(summary.get("global_step")),
        boots=_optional_float(summary.get("eval/boots")),
        crashed=_optional_float(summary.get("eval/crashed")),
        net_stock_per_min=_optional_float(summary.get("eval/net_stock_per_min")),
        net_stock_lcb=_optional_float(summary.get("eval/net_stock_lcb")),
        net_dmg_per_min=_optional_float(summary.get("eval/net_dmg_per_min")),
    )


def _validated_preflight_reports(reports: dict[str, Path]) -> dict[str, Path]:
    """Return report paths that the Modal source image will contain."""
    if not reports:
        return {}
    ignored = modal.FilePatternMatcher.from_file(ROOT / ".dockerignore")
    tracked = {Path(value) for value in _git("ls-files", "-z").split("\0") if value}
    validated: dict[str, Path] = {}
    root = ROOT.resolve()
    for arm_id, report in reports.items():
        if report.is_absolute():
            raise ValueError(f"preflight report for {arm_id!r} must be relative to the repository")
        try:
            candidate = (ROOT / report).resolve(strict=True)
            relative = candidate.relative_to(root)
        except (FileNotFoundError, ValueError) as error:
            raise ValueError(f"preflight report for {arm_id!r} is not a repository file: {report}") from error
        if not candidate.is_file():
            raise ValueError(f"preflight report for {arm_id!r} is not a regular file: {report}")
        if relative not in tracked:
            raise ValueError(f"preflight report for {arm_id!r} is not tracked by Git: {relative}")
        if ignored(relative):
            raise ValueError(f"preflight report for {arm_id!r} is excluded from the Modal image: {relative}")
        validated[arm_id] = relative
    return validated


def _report_for(arm: SweepArm, reports: dict[str, Path], *, required: bool) -> Path | None:
    if not arm.requires_preflight:
        return None
    report = reports.get(arm.arm_id, reports.get("*"))
    if report is None and required:
        raise ValueError(
            f"production arm {arm.arm_id} needs a preflight report; add its arm ID or '*' to --preflight-reports"
        )
    return report


def _spec_sha256(
    arm: SweepArm,
    train_argv: tuple[str, ...],
    *,
    git_sha: str,
    gpu: str,
    disk_gib: int,
) -> str:
    payload = {
        "arm_id": arm.arm_id,
        "stage": arm.stage,
        "git_sha": git_sha,
        "gpu": gpu,
        "disk_gib": disk_gib,
        "train_argv": train_argv,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _prepare_arms(
    arms: tuple[SweepArm, ...],
    reports: dict[str, Path],
    *,
    git_sha: str,
    gpu: str,
    disk_gib: int,
    require_preflight: bool,
) -> tuple[PreparedArm, ...]:
    prepared: list[PreparedArm] = []
    for arm in arms:
        report = _report_for(arm, reports, required=require_preflight)
        train_argv = arm.argv(preflight_report=report)
        prepared.append(
            PreparedArm(
                arm=arm,
                preflight_report=report,
                train_argv=train_argv,
                spec_sha256=_spec_sha256(
                    arm,
                    train_argv,
                    git_sha=git_sha,
                    gpu=gpu,
                    disk_gib=disk_gib,
                ),
            )
        )
    return tuple(prepared)


def _validate_record(record: object, path: Path, line_number: int) -> dict[str, object]:
    prefix = f"invalid launch state {path}:{line_number}"
    if not isinstance(record, dict):
        raise ValueError(f"{prefix}: each line must contain a JSON object")
    record = cast(dict[str, object], record)
    if record.get("schema_version") != STATE_SCHEMA:
        raise ValueError(f"{prefix}: expected schema version {STATE_SCHEMA}")
    required_strings = ("arm_id", "stage", "event", "attempt_id", "spec_sha256", "git_sha", "recorded_at")
    invalid_strings = [name for name in required_strings if not isinstance(record.get(name), str) or not record[name]]
    if invalid_strings:
        raise ValueError(f"{prefix}: invalid string fields {invalid_strings}")
    if record["event"] not in ("launching", "failed", "uncertain", "launched"):
        raise ValueError(f"{prefix}: unknown event {record['event']!r}")
    if not SHA256_PATTERN.fullmatch(cast(str, record["spec_sha256"])):
        raise ValueError(f"{prefix}: spec_sha256 is not a lowercase SHA-256 digest")
    argv = record.get("argv")
    if not isinstance(argv, list) or not all(isinstance(value, str) for value in argv):
        raise ValueError(f"{prefix}: argv must be a list of strings")
    if not isinstance(record.get("treatment"), dict):
        raise ValueError(f"{prefix}: treatment must be a JSON object")
    event = cast(LaunchEvent, record["event"])
    app_id = record.get("app_id")
    function_call_id = record.get("function_call_id")
    if event == "launched" and not (isinstance(app_id, str) and isinstance(function_call_id, str)):
        raise ValueError(f"{prefix}: only a launched event must contain both Modal IDs")
    if event != "launched" and (app_id is not None or function_call_id is not None):
        raise ValueError(f"{prefix}: only a launched event can contain Modal IDs")
    return record


def _records(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    latest: dict[str, dict[str, object]] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid launch state {path}:{line_number}: {error.msg}") from error
        record = _validate_record(raw, path, line_number)
        arm_id = cast(str, record["arm_id"])
        event = cast(LaunchEvent, record["event"])
        previous = latest.get(arm_id)
        if previous is None:
            if event != "launching":
                raise ValueError(f"invalid launch state {path}:{line_number}: first arm event must be launching")
        else:
            if record["spec_sha256"] != previous["spec_sha256"]:
                raise ValueError(f"invalid launch state {path}:{line_number}: arm ID changed its launch specification")
            previous_event = cast(LaunchEvent, previous["event"])
            same_attempt = record["attempt_id"] == previous["attempt_id"]
            if event == "launching":
                if previous_event != "failed" or same_attempt:
                    raise ValueError(
                        f"invalid launch state {path}:{line_number}: a new attempt can follow only a failed attempt"
                    )
            elif previous_event != "launching" or not same_attempt:
                raise ValueError(
                    f"invalid launch state {path}:{line_number}: terminal event does not match a launching attempt"
                )
        latest[arm_id] = record
    return latest


def _append(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


@contextmanager
def _state_lock(path: Path) -> Iterator[None]:
    lock_path = path.with_suffix(f"{path.suffix}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another O51 sweep launcher holds {lock_path}") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _pending_arms(
    arms: tuple[PreparedArm, ...],
    previous: dict[str, dict[str, object]],
) -> tuple[PreparedArm, ...]:
    pending: list[PreparedArm] = []
    for prepared in arms:
        record = previous.get(prepared.arm.arm_id)
        if record is None:
            pending.append(prepared)
            continue
        if record["spec_sha256"] != prepared.spec_sha256:
            raise RuntimeError(f"arm ID {prepared.arm.arm_id} already has a different launch specification")
        event = cast(LaunchEvent, record["event"])
        if event == "failed":
            pending.append(prepared)
        elif event != "launched":
            raise RuntimeError(
                f"arm {prepared.arm.arm_id} has an unresolved {event} attempt; reconcile Modal before retrying"
            )
    return tuple(pending)


def _event_payload(prepared: PreparedArm, git_sha: str, attempt_id: str, event: LaunchEvent) -> dict[str, object]:
    return {
        "schema_version": STATE_SCHEMA,
        "event": event,
        "attempt_id": attempt_id,
        "arm_id": prepared.arm.arm_id,
        "stage": prepared.arm.stage,
        "spec_sha256": prepared.spec_sha256,
        "git_sha": git_sha,
        "recorded_at": datetime.now(UTC).isoformat(),
        "argv": prepared.train_argv,
        "treatment": asdict(prepared.arm.treatment),
    }


def launch_command(prepared: PreparedArm, args: LaunchArgs, git_sha: str) -> tuple[str, ...]:
    app_name = f"hal-{prepared.arm.arm_id}-{git_sha[:7]}"
    return (
        sys.executable,
        str(LAUNCHER),
        "--gpu",
        args.gpu,
        "--disk-gib",
        str(args.disk_gib),
        "--cpu",
        "32",
        "--cpu-limit",
        "48",
        "--memory-gib",
        "192",
        "--memory-limit-gib",
        "384",
        "--app-name",
        app_name,
        "--",
        *prepared.train_argv,
    )


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _git_sha() -> str:
    if _git("status", "--porcelain"):
        raise RuntimeError("commit all working-tree changes before an O51 launch")
    return _git("rev-parse", "HEAD")


def _validate_launch_args(args: LaunchArgs) -> None:
    if args.max_arms is not None and args.max_arms < 1:
        raise ValueError("max_arms must be positive")
    if args.gpu != "B200":
        raise ValueError("O51 sweep launches require a B200")
    if args.disk_gib < 2048:
        raise ValueError("O51 sweep launches require at least 2048 GiB of ephemeral disk")


def _launch_pending(
    pending: tuple[PreparedArm, ...],
    args: LaunchArgs,
    git_sha: str,
) -> None:
    for prepared in pending:
        attempt_id = str(uuid.uuid4())
        launching = _event_payload(prepared, git_sha, attempt_id, "launching")
        _append(args.state, launching)
        command = launch_command(prepared, args, git_sha)
        try:
            result = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
        except BaseException:
            uncertain = _event_payload(prepared, git_sha, attempt_id, "uncertain")
            _append(args.state, uncertain)
            raise
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
        if result.returncode:
            uncertain = _event_payload(prepared, git_sha, attempt_id, "uncertain")
            uncertain["returncode"] = result.returncode
            _append(args.state, uncertain)
            raise RuntimeError(f"launch failed for {prepared.arm.arm_id} with status {result.returncode}")
        match = APP_PATTERN.search(result.stdout + result.stderr)
        if match is None:
            uncertain = _event_payload(prepared, git_sha, attempt_id, "uncertain")
            _append(args.state, uncertain)
            raise RuntimeError(f"launch output for {prepared.arm.arm_id} did not contain Modal IDs")
        launched = _event_payload(prepared, git_sha, attempt_id, "launched")
        launched["app_id"] = match.group(1)
        launched["function_call_id"] = match.group(2)
        _append(args.state, launched)


def launch(args: LaunchArgs) -> None:
    _validate_launch_args(args)
    git_sha = _git_sha()
    arms = stage_arms(args.stage, _treatment(args.treatment))
    prepared = _prepare_arms(
        arms,
        _validated_preflight_reports(_preflight_reports(args.preflight_reports)),
        git_sha=git_sha,
        gpu=args.gpu,
        disk_gib=args.disk_gib,
        require_preflight=True,
    )
    if args.dry_run:
        pending = _pending_arms(prepared, _records(args.state))
        if args.max_arms is not None:
            pending = pending[: args.max_arms]
        for arm in pending:
            print(json.dumps({"arm_id": arm.arm.arm_id, "command": launch_command(arm, args, git_sha)}))
        return
    with _state_lock(args.state):
        pending = _pending_arms(prepared, _records(args.state))
        if args.max_arms is not None:
            pending = pending[: args.max_arms]
        _launch_pending(pending, args, git_sha)


def _plan(args: PlanArgs) -> None:
    reports = _validated_preflight_reports(_preflight_reports(args.preflight_reports))
    arms = stage_arms(args.stage, _treatment(args.treatment))
    payload = []
    for arm in arms:
        report = _report_for(arm, reports, required=False)
        payload.append(
            {
                "arm_id": arm.arm_id,
                "requires_preflight": arm.requires_preflight,
                "preflight_report": None if report is None else str(report),
                "argv": arm.argv(preflight_report=report),
            }
        )
    print(json.dumps(payload, indent=2))


def _json_text(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _check_output(path: Path, text: str) -> None:
    if path.exists() and path.read_text() != text:
        raise ValueError(f"refusing to replace different selection output: {path}")


def _write_output(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def _collect_validation(
    stage: Stage,
    center: Treatment,
    runs_path: Path,
    api: Any,
) -> tuple[tuple[SweepArm, ...], dict[str, ValidationOutcome], Any]:
    arms = stage_arms(stage, center)
    run_paths = _run_paths(runs_path, arms)
    outcomes = {
        arm.arm_id: _validation_outcome(arm, run_paths[arm.arm_id], api.run(run_paths[arm.arm_id])) for arm in arms
    }
    return arms, outcomes, select_validation_winner(arms, outcomes)


def _validation_rule() -> list[str]:
    return [
        "Every run must be terminal before screening.",
        "A candidate must finish its exact D/U endpoint and final optimizer update.",
        "Final validation NLL, far NLL, and rollout NLL must be finite.",
        "Rank eligible candidates by validation NLL, then far NLL, then rollout NLL, then arm ID.",
    ]


def rank(args: RankArgs) -> None:
    """Record the two validation finalists that require closed-loop evaluation."""
    center = _treatment(args.treatment)
    arms, outcomes, selection = _collect_validation(args.stage, center, args.runs, wandb.Api(timeout=30))
    del arms
    finalists = selection.ranking[:2]
    if len(finalists) != 2:
        raise ValueError("a 3x3 grid needs two eligible finalists")
    evidence = {
        "schema_version": 1,
        "stage": args.stage,
        "rule": _validation_rule(),
        "source_treatment": asdict(center),
        "final_eval_matchups": GRID_EVAL_MATCHUPS,
        "finalists": [{"arm_id": outcome.arm_id, "training_run": outcome.run_path} for outcome in finalists],
        "ranking": [asdict(outcome) for outcome in selection.ranking],
        "excluded": [{"arm_id": arm_id, "reasons": list(reasons)} for arm_id, reasons in selection.excluded],
    }
    text = _json_text(evidence)
    _check_output(args.evidence, text)
    _write_output(args.evidence, text)
    print(f"screened {args.stage}; evaluate {', '.join(outcome.arm_id for outcome in finalists)}")


def adjudicate(args: AdjudicateArgs) -> None:
    """Select a 3x3 grid only after evaluating its two validation finalists."""
    center = _treatment(args.treatment)
    api = wandb.Api(timeout=30)
    arms, validation_outcomes, validation = _collect_validation(args.stage, center, args.runs, api)
    finalists = validation.ranking[:2]
    if len(finalists) != 2:
        raise ValueError("a 3x3 grid needs two eligible finalists")
    finalist_ids = (finalists[0].arm_id, finalists[1].arm_id)
    training_paths = {outcome.arm_id: outcome.run_path for outcome in finalists}
    evaluation_paths = _evaluation_paths(args.evaluations, finalist_ids)
    arms_by_id = {arm.arm_id: arm for arm in arms}
    closed_loop_outcomes = {
        arm_id: _closed_loop_outcome(
            arms_by_id[arm_id],
            training_paths[arm_id],
            evaluation_paths[arm_id],
            api.run(evaluation_paths[arm_id]),
        )
        for arm_id in finalist_ids
    }
    selection = select_closed_loop_winner(arms, validation, closed_loop_outcomes)
    treatment_payload = asdict(selection.winner.treatment)
    evidence_payload = {
        "schema_version": 1,
        "stage": args.stage,
        "validation_rule": _validation_rule(),
        "closed_loop_rule": [
            f"Evaluate exactly the two best validation arms over {GRID_EVAL_MATCHUPS} fixed matchups.",
            "Both evaluations must finish the final checkpoint with all boots and no crashes.",
            "Rank by net-stock cluster-bootstrap lower bound, then mean net stock per minute.",
            "Break any remaining tie by net damage per minute, validation NLL, then arm ID.",
        ],
        "source_treatment": asdict(center),
        "winner_arm_id": selection.winner.arm_id,
        "winner_training_run": training_paths[selection.winner.arm_id],
        "winner_evaluation_run": closed_loop_outcomes[selection.winner.arm_id].run_path,
        "winner_treatment": treatment_payload,
        "validation_ranking": [asdict(outcome) for outcome in validation.ranking],
        "validation_excluded": [
            {"arm_id": arm_id, "reasons": list(reasons)} for arm_id, reasons in validation.excluded
        ],
        "closed_loop_ranking": [asdict(outcome) for outcome in selection.ranking],
    }
    treatment_text = _json_text(treatment_payload)
    evidence_text = _json_text(evidence_payload)
    _check_output(args.output, treatment_text)
    _check_output(args.evidence, evidence_text)
    _write_output(args.output, treatment_text)
    _write_output(args.evidence, evidence_text)
    print(f"selected {selection.winner.arm_id}; wrote {args.output} and {args.evidence}")


def select(args: SelectArgs) -> None:
    """Collect final W&B values and write one reproducible stage decision."""
    if args.stage in ("lr", "decay"):
        raise ValueError(f"{args.stage} is a 3x3 grid; use rank, evaluate both finalists, then adjudicate")
    center = _treatment(args.treatment)
    _arms, outcomes, selection = _collect_validation(args.stage, center, args.runs, wandb.Api(timeout=30))
    treatment_payload = asdict(selection.winner.treatment)
    evidence_payload = {
        "schema_version": 1,
        "stage": args.stage,
        "rule": _validation_rule(),
        "source_treatment": asdict(center),
        "winner_arm_id": selection.winner.arm_id,
        "winner_run": outcomes[selection.winner.arm_id].run_path,
        "winner_treatment": treatment_payload,
        "ranking": [asdict(outcome) for outcome in selection.ranking],
        "excluded": [{"arm_id": arm_id, "reasons": list(reasons)} for arm_id, reasons in selection.excluded],
    }
    treatment_text = _json_text(treatment_payload)
    evidence_text = _json_text(evidence_payload)
    _check_output(args.output, treatment_text)
    _check_output(args.evidence, evidence_text)
    _write_output(args.output, treatment_text)
    _write_output(args.evidence, evidence_text)
    print(f"selected {selection.winner.arm_id}; wrote {args.output} and {args.evidence}")


WINDOWS_PER_GENERATION: Final[int] = 4
DEFAULT_REPLAY_SLOTS: Final[int] = 131_072
RESERVED_DISK_BYTES: Final[int] = 256 * 2**30
O51_EXTRA_COLUMNS: Final[ExtraColumns] = ExtraColumns(
    floats=ITEM_COLUMNS.floats,
    cats={**ITEM_COLUMNS.cats, "player_id": None},
)
_FROZEN_RUNTIME_DEFAULT_CONFIG = TrainConfig()
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
    base = TrainConfig(
        arch=MODEL_FAMILY[level],
        target_positions=target_positions,
        tier_scale=tier_scale,
    )
    return replace(base, **changes)


@dataclass(frozen=True, slots=True)
class DepthRule:
    attention: float
    mlp: float


def depth_rule(stack: Literal["trunk", "temporal"], layers: int, alpha: float) -> DepthRule:
    """Return O51's residual-branch multipliers for one stack."""
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
) -> float:
    if fan_in_multiplier <= 0:
        raise ValueError("fan-in multiplier must be positive")
    value = master_lr * math.sqrt(batch_multiplier / duration_multiplier)
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
        name: (getattr(cfg, name), getattr(_FROZEN_RUNTIME_DEFAULT_CONFIG, name))
        for name in _FROZEN_RUNTIME_FIELDS
        if getattr(cfg, name) != getattr(_FROZEN_RUNTIME_DEFAULT_CONFIG, name)
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
    scaled_adam_betas(
        (cfg.adam_beta1, cfg.adam_beta2),
        batch_multiplier=scaling_multipliers(cfg)[0],
        duration_multiplier=scaling_multipliers(cfg)[1],
    )


def center_class_logits(logits: Tensor) -> Tensor:
    """Remove one softmax-invariant common mode from each class group."""
    return logits - logits.mean(dim=-1, keepdim=True)


def mup_readout_std(fan_in: int, base_fan_in: int) -> float:
    """Use sigma(n0)=1/sqrt(n0), then sigma(n)=sigma(n0)*n0/n."""
    if fan_in < 1 or base_fan_in < 1:
        raise ValueError("readout fan-ins must be positive")
    return math.sqrt(base_fan_in) / fan_in


def _final_readouts(model: Policy) -> tuple[tuple[nn.Linear, int], ...]:
    action = tuple((cast(nn.Linear, model.temporal.outputs[name].down), 128) for name in GROUP_NAMES)
    trunk_skip = tuple((cast(nn.Linear, model.temporal.trunk_outputs[name]), 256) for name in GROUP_NAMES)
    return (*action, *trunk_skip, (model.value_head.down, 128))


def initialize_o51_parameters(model: Policy, cfg: TrainConfig) -> None:
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


def subsystem_parameter_counts(model: Policy) -> dict[str, int]:
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


def optimizer_roles(model: Policy, cfg: TrainConfig) -> dict[str, OptimizerRole]:
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
            roles[name] = OptimizerRole("muon", "hidden", True, logical_splits=splits)
        elif name.startswith("temporal.blocks."):
            splits = 3 if name.endswith("qkv.weight") else 1
            roles[name] = OptimizerRole("muon", "hidden", True, logical_splits=splits)
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
        return cfg.muon_lr * muon_lr_multiplier(cfg)
    return scaled_adam_lr(
        cfg.adam_lr,
        batch_multiplier=batch_multiplier,
        duration_multiplier=duration_multiplier,
        fan_in_multiplier=role.fan_in_multiplier,
        output=role.lr_kind == "output",
    )


def make_optimizer(model: Policy, cfg: TrainConfig) -> SingleDeviceMuonWithAuxAdam:
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
                    "muon_scale_clamp_min_one": False,
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
    return SingleDeviceMuonWithAuxAdam(groups)


def _button_adam_parameters(model: Policy) -> dict[str, nn.Parameter]:
    """Retain O50's diagnostics only for tensors that remain on AdamW."""
    button_head = cast(NonlinearActionHead, model.temporal.outputs["buttons"])
    return {
        "buttons_output_weight": cast(nn.Parameter, button_head.down.weight),
        "buttons_condition_weight": cast(nn.Parameter, model.temporal.group_condition["buttons"].weight),
    }


def microbatch_loss(
    model: Policy,
    batch: AWRBatch,
    cfg: TrainConfig,
    *,
    step: int,
    valid_prefixes: int,
    trunk_fn: Callable,
    temporal_fn: Callable,
    phase_timer: CudaPhaseTimer | None = None,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """O50's AWR loss, activated only after D/32 valid positions."""
    if not isinstance(batch, AWRBatch):
        raise TypeError(f"advantage training needs an AWRBatch, got {type(batch).__name__}")
    history, targets, valid = prepared_targets(model, batch)
    if phase_timer is not None:
        phase_timer.record("target_prep_end")
    with amp_context(cfg, DEVICE):
        hidden = trunk_fn(batch.context.features, batch.context.ctx_pad, None)
        if phase_timer is not None:
            phase_timer.record("trunk_end")
        suffix_start = direct_loss_start(cfg)
        hidden = hidden[:, suffix_start:]
        temporal_output = temporal_fn(hidden, history, targets)
        if isinstance(temporal_output, Tensor):
            dense_nll = temporal_output
            button_diagnostics: dict[str, Tensor] = {}
        else:
            dense_nll, button_diagnostics = temporal_output
        if phase_timer is not None:
            phase_timer.record("temporal_end")
    value = model.value_head(decoder_rmsnorm(hidden).detach().float()).squeeze(-1)
    value_loss, advantage, value_stats = value_objective(
        value,
        batch.returns[:, suffix_start:],
        batch.eligible[:, suffix_start:],
        beta=cfg.awr.beta,
        valid=valid,
    )
    active = step + 1 > cfg.warmup_steps
    weights, stats = advantage_weights(
        advantage,
        batch.eligible[:, suffix_start:],
        beta=cfg.awr.beta,
        weight_max=cfg.awr.weight_max,
        active=active,
        valid=valid,
    )
    button_loss = dense_nll[..., BUTTONS_G].float().mean(dim=-1)
    stats["weight_button_loss_correlation"] = masked_correlation(
        weights,
        button_loss,
        batch.eligible[:, suffix_start:] & valid,
    )
    near, far, policy_loss = temporal_objective_parts(
        dense_nll,
        weights,
        valid_prefixes=valid_prefixes,
        aux_loss_weight=cfg.awr.auxiliary_loss_weight,
        valid=valid,
    )
    loss = policy_loss + cfg.awr.value_loss_weight * value_loss
    nll_sum = torch.where(valid[..., None, None], dense_nll.float(), 0).sum(dim=(0, 1))
    extra = {
        "train/loss": policy_loss.detach() / _LN2,
        "train/near_loss": near.detach() / _LN2,
        "train/far_nll": far.detach() / _LN2,
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
    """Apply live magnitude gates and the endpoint clipping gate."""

    def __init__(self, warmup_updates: int, final_update: int) -> None:
        self._warmup_updates = warmup_updates
        self._final_update = final_update
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
        has_stability = isinstance(centered, int | float) or isinstance(rms, int | float)
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

        if has_stability and update >= self._warmup_updates:
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
        if not has_stability and update < self._final_update:
            return None
        return arm_decision(
            numeric,
            post_warmup_clip_fraction=post_warmup_clip if update >= self._final_update else 0.0,
            initial_action_pre_norm_rms=self._initial_rms,
            initial_centered_logit_p999=self._initial_centered,
            sustained_growth=max(self._centered_growth_windows, self._rms_growth_windows) >= 4,
        )


def data_selection(cfg: TrainConfig) -> CorpusSelection:
    """Return the pinned direct-source tier definitions."""
    del cfg
    return corpus_selection()


def load_stats(cfg: TrainConfig) -> dict[str, FeatureStats]:
    """Combine existing source statistics with the selected direct-prefix mix."""
    tier = data_selection(cfg).tier(cfg.tier_scale)
    sources = tuple(streams.BY_NAME[name] for name in cfg.source_names)
    selected = tier.source_replay_counts()
    return load_consolidated_mixture_stats(
        [source.local_root / "stats.json" for source in sources],
        [float(selected[source.name]) for source in sources],
        expected_mds_schema_version=cfg.mds_schema_version,
    )


def _collate_o51_batch(
    replay_ids: tuple[str, ...],
    windows: list[dict[str, np.ndarray]],
    *,
    stats: dict[str, FeatureStats],
    projection: FeatureProjection,
    context_length: int,
) -> AWRBatch:
    batch = collate_train_batch(
        windows,
        stats=stats,
        L_ctx=context_length,
        extra=O51_EXTRA_COLUMNS,
        projection=projection,
    )
    batch = TrainBatch(batch.context, batch.target, replay_ids)
    next_frames = slice(1, context_length + 1)
    return_name = f"ego_{O51_RETURN_SUFFIX}"
    returns = np.stack([window[return_name] for window in windows])[:, next_frames]
    eligible = np.stack([window[f"{return_name}_valid"] for window in windows])[:, next_frames]
    return AWRBatch(
        batch,
        torch.from_numpy(np.ascontiguousarray(returns)),
        torch.from_numpy(np.ascontiguousarray(eligible)).bool(),
    )


def _make_train_loader(
    cfg: TrainConfig,
    stats: dict[str, FeatureStats],
    player_lookup: ReplayPlayerLookup,
) -> PhysicalShardReplayLoader[AWRBatch]:
    """Build the direct source-prefix training loader."""
    tier = data_selection(cfg).tier(cfg.tier_scale)
    if tuple(view.source for view in tier.sources) != cfg.source_names:
        raise RuntimeError("direct prefix views do not match configured source order")
    projection = replace(
        MODEL_PROJECTION,
        columns=MODEL_PROJECTION.columns
        | {
            f"ego_{O51_RETURN_SUFFIX}",
            f"ego_{O51_RETURN_SUFFIX}_valid",
        },
    )
    selection = PhysicalShardSelection(
        tuple(SourceRowSelection(view.source, view.stop, view.excluded_rows) for view in tier.sources),
        tier.sha256,
    )
    adapter = MDSStorageAdapter(selection, download_retry=8)
    tasks = build_shard_plan(selection, adapter.manifests)
    train_loader = PhysicalShardReplayLoader[AWRBatch](
        selection=selection,
        adapter=adapter,
        tasks=tasks,
        data_protocol=DATA_PROTOCOL,
        source_manifest_sha256={view.source: SOURCE_MANIFEST_SHA256[view.source] for view in tier.sources},
        batch_transform=functools.partial(
            _collate_o51_batch,
            stats=stats,
            projection=projection,
            context_length=cfg.arch.L_ctx,
        ),
        batch_size=cfg.batch_size,
        replay_slots=DEFAULT_REPLAY_SLOTS,
        seed=cfg.seed,
        num_workers=cfg.num_workers,
        labels=ParameterizationReplayLabels(
            player_lookup=player_lookup,
            gamma=cfg.awr.gamma,
            damage_shaping=cfg.awr.damage_shaping,
            win_reward=cfg.awr.win_reward,
            stock_value=cfg.awr.stock_value,
        ),
        projection=projection,
        context_length=cfg.arch.L_ctx,
        chunk_length=cfg.arch.sample_chunk_length,
        windows_per_generation=WINDOWS_PER_GENERATION,
        schema_version=cfg.mds_schema_version,
        reserved_disk_bytes=RESERVED_DISK_BYTES,
        pin_memory=torch.cuda.is_available(),
    )
    expected_train_rows = OFFICIAL_TIER_REPLAYS[cfg.tier_scale]
    actual_train_rows = sum(train_loader.source_sample_counts.values())
    if actual_train_rows != expected_train_rows:
        raise ValueError(
            f"direct U{cfg.tier_scale} view exposes {actual_train_rows} rows, expected {expected_train_rows}"
        )
    return train_loader


def _make_loaders(
    cfg: TrainConfig,
    stats: dict[str, FeatureStats],
    player_lookup: ReplayPlayerLookup | None = None,
) -> tuple[PhysicalShardReplayLoader[AWRBatch], list[TrainBatch]]:
    """Build direct source-prefix training and the fixed validation cohort."""
    if player_lookup is None:
        player_lookup = ReplayPlayerLookup(load_identity_sidecar(cfg).by_replay)
    train_loader = _make_train_loader(cfg, stats, player_lookup)

    val_loader = make_loader(
        data_root=None,
        split=cfg.val_split,
        stats=stats,
        L_ctx=cfg.arch.L_ctx,
        L_chunk=cfg.arch.sample_chunk_length,
        batch_size=128,
        seed=0,
        sources=tuple(streams.BY_NAME[name] for name in cfg.source_names),
        cache_limit="1792gb",
        shuffle_block_size=8192,
        shuffle_seed=0,
        num_workers=0,
        schema_version=cfg.mds_schema_version,
        extra=MODEL_COLUMNS,
        projection=MODEL_PROJECTION,
        replay_format="policy-world",
        replay_labels=player_lookup,
        require_full_context=True,
        shuffle=True,
    )
    return train_loader, cache_validation(val_loader, cfg.val_n_samples)


def _next_awr_batch(iterator: Iterator[AWRBatch]) -> AWRBatch:
    """Fetch one typed batch on the pre-CUDA setup thread."""
    return next(iterator)


def _prepare_training_data(
    cfg: TrainConfig,
    stats: dict[str, FeatureStats],
    sidecar: PlayerIdentitySidecar,
    resume_state: dict[str, object] | None,
) -> PreparedTrainingData:
    """Start O51 shard workers and its first batch before CUDA allocation."""
    train_loader, validation = _make_loaders(cfg, stats, ReplayPlayerLookup(sidecar.by_replay))
    if resume_state is not None:
        loader_state = resume_state.get("loader")
        if not isinstance(loader_state, dict):
            raise ValueError("resume checkpoint does not contain O51 replay-loader state")
        train_loader.load_state_dict(cast(dict[str, object], loader_state))
    train_iterator = iter(train_loader)
    with ExitStack() as setup:
        executor = setup.enter_context(ThreadPoolExecutor(max_workers=1, thread_name_prefix="o51-first-batch"))
        first_batch = executor.submit(_next_awr_batch, train_iterator)
        resources = setup.pop_all()
    return PreparedTrainingData(train_loader, validation, train_iterator, first_batch, executor, resources)


LOADER_WORKERS: Final[tuple[int, ...]] = (8, 16, 24, 32)
PHYSICAL_BATCHES: Final[tuple[int, ...]] = (128, 256, 512, 1024)
COMPILE_MODES: Final[tuple[str, ...]] = ("reduce-overhead", "max-autotune")
TEMPORAL_ATTENTION_CHUNKS: Final[tuple[int | None, ...]] = (8192, 16_384, 32_768, None)
REQUIRED_PREFLIGHT_TELEMETRY: Final[tuple[str, ...]] = (
    "system/network/read_mib_s",
    "system/cache/allocated_gib",
    "system/pinned_memory_gib",
    "system/cgroup/current_gib",
    "system/cgroup/projected_peak_gib",
    "system/disk/required_bytes",
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
        "num_workers": cfg.num_workers,
        "worker_completion_order": "in-order-keyed-shards-v1",
        "replay_slots": DEFAULT_REPLAY_SLOTS,
        "windows_per_generation": WINDOWS_PER_GENERATION,
        "loader_prefetch_factor": PREFETCH_FACTOR,
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
    disk_free_bytes: int
    exact_resume: bool
    memory_passed: bool
    shuffle_passed: bool
    loader_stability_passed: bool
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
        "disk_free_bytes",
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
        "loader_stability_passed": report.loader_stability_passed,
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
    if report.loader_stability_passed is not True:
        failures.append("loader-only stability audit did not pass")
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
    required_disk = report.telemetry.get("system/disk/required_bytes")
    if isinstance(required_disk, int | float) and numeric["disk_free_bytes"] < float(required_disk):
        failures.append("current free disk does not cover the manifest, download workspace, and 256 GiB reserve")
    cgroup_peak = max(
        report.telemetry.get("system/cgroup/current_gib", 0.0),
        report.telemetry.get("system/cgroup/projected_peak_gib", 0.0),
    )
    if cgroup_peak > 300:
        failures.append("projected or observed cgroup memory exceeds 300 GiB")
    compute_demand = 1.25 * numeric["compute_only_updates_per_s"] * cfg.batch_size
    if numeric["loader_only_windows_per_s"] < compute_demand:
        failures.append("loader-only throughput is below 1.25x measured compute demand")
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
        row = {name: float(cast(int | float, value)) for name, value in values.items()}
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
        fingerprint=preflight.fingerprint,
        synthetic_mfu=preflight.synthetic_mfu,
        compute_only_updates_per_s=preflight.compute_only_updates_per_s,
        loader_only_windows_per_s=preflight.loader_only_windows_per_s,
        raw_bytes_per_window=preflight.raw_bytes_per_window,
        o50_raw_bytes_per_window=preflight.o50_raw_bytes_per_window,
        peak_memory_fraction=preflight.peak_memory_fraction,
        graph_gap_fraction=preflight.graph_gap_fraction,
        optimizer_time_fraction=preflight.optimizer_time_fraction,
        disk_free_bytes=preflight.disk_free_bytes,
        exact_resume=preflight.exact_resume,
        memory_passed=preflight.memory_passed,
        shuffle_passed=preflight.shuffle_passed,
        loader_stability_passed=preflight.loader_stability_passed,
        telemetry=telemetry,
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


def _log_wandb(values: dict[str, object], arm_guard: _ArmGuard) -> None:
    """Add O51 throughput semantics, log, then apply the arm gate."""
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
    wandb.log(payload)
    decision = arm_guard.observe(payload)
    if decision is not None and decision.status != "pass":
        raise RuntimeError(f"O51 arm {decision.status}: {'; '.join(decision.reasons)}")


def _init_wandb(cfg: TrainConfig, run_name: str, resume_state: dict[str, object] | None) -> None:
    """Start an O51 run with its nested-data identity in the immutable config."""
    selection = data_selection(cfg)
    wandb_id = None if resume_state is None else resume_state.get("wandb_id")
    if wandb_id is not None and not isinstance(wandb_id, str):
        raise TypeError("resume checkpoint W&B id must be a string or None")
    wandb.init(
        project="hal",
        group="o51-correct-parameterization",
        name=run_name,
        id=wandb_id,
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
        settings=wandb.Settings(
            x_stats_sampling_interval=5.0,
            x_stats_track_process_tree=True,
        ),
    )
    if wandb.run is None:
        return
    wandb.define_metric("global_step")
    wandb.define_metric("*", step_metric="global_step")
    summary = wandb.run.summary
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
        log_wandb_code(wandb.run)


def _log_training_summary(
    cfg: TrainConfig,
    parameter_counts: dict[str, int],
    *,
    flops_per_update: int,
    device_name: str | None,
    peak_flops: float | None,
) -> None:
    """Record the selected nested pool and its actual per-source counters."""
    if wandb.run is None:
        return
    summary = wandb.run.summary
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
    summary["data/replay_slots"] = min(DEFAULT_REPLAY_SLOTS, unique_replays)
    summary["data/generation_windows"] = WINDOWS_PER_GENERATION
    summary["data/loader_prefetch_factor"] = PREFETCH_FACTOR
    summary["data/source_mixing"] = "identity_uniform_dense_shards"
    summary["data/corpus_hash"] = selection.corpus_hash
    summary["data/tier_hash"] = selection.tier(cfg.tier_scale).sha256
    summary["training/approx_flops_per_update"] = flops_per_update
    summary["training/flops_formula"] = "6*B*L_ctx*(N_trunk+N_other+N_value+n_offsets*(N_temporal+N_group_heads))"
    if device_name is not None:
        summary["hardware/gpu_name"] = device_name
    if peak_flops is not None:
        summary["hardware/bf16_dense_peak_tflops"] = peak_flops / 1e12
        source = bf16_peak_source(device_name or "")
        if source is not None:
            summary["hardware/bf16_dense_peak_source"] = source
    for source_name in cfg.source_names:
        source_replays = selected[source_name]
        summary[f"data/source_sampling_share/{source_name}"] = source_replays / unique_replays
        summary[f"data/source_replays/{source_name}"] = source_replays


def _training_functions(model: Policy, cfg: TrainConfig) -> tuple[Callable, Callable]:
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
    model: Policy,
    batch: AWRBatch,
    cfg: TrainConfig,
    *,
    max_rows: int,
) -> dict[str, float]:
    """Measure O51 rejection signals outside the compiled training path."""
    diagnostic = _diagnostic_awr_batch(batch, max_rows)
    history, targets, _valid = prepared_targets(model, diagnostic)
    device = next(model.parameters()).device
    with amp_context(cfg, device):
        hidden = model(diagnostic.context.features, diagnostic.context.ctx_pad, None)
        hidden = hidden[:, direct_loss_start(cfg) :]
        _nll, metrics = model.temporal.teacher_forced_nll_with_diagnostics(hidden, history, targets)
    values = torch.stack(tuple(metrics.values())).double().cpu()
    payload = {name: float(value) for name, value in zip(metrics, values, strict=True)}
    nonfinite = {name: value for name, value in payload.items() if not math.isfinite(value)}
    if nonfinite:
        raise FloatingPointError(f"stability diagnostic produced non-finite metrics: {nonfinite}")
    return payload


def _training_diagnostics(model: Policy, batch: AWRBatch, cfg: TrainConfig, update: int) -> dict[str, object]:
    """Collect O50 diagnostics plus O51's periodic arm-rejection signals."""
    metrics = _baseline_training_diagnostics(model, batch, cfg, update)
    if update % cfg.stability_every == 0:
        metrics.update(stability_diagnostics_log(model, batch, cfg, max_rows=cfg.layer_rms_batch_size))
    return metrics


def layer_activation_rms_log(
    model: Policy,
    batch: TrainBatch | AWRBatch,
    cfg: TrainConfig,
    *,
    max_rows: int,
) -> dict[str, float]:
    """Correct O50's unscaled temporal-MLP hook for O51's branch rule."""
    metrics = _baseline_layer_activation_rms_log(model, batch, cfg, max_rows=max_rows)
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


def _finalize_training(
    *,
    model: Policy,
    optimizer: SingleDeviceMuonWithAuxAdam,
    scheduler: LambdaLR,
    cfg: TrainConfig,
    stats: dict[str, FeatureStats],
    val_cache: list[TrainBatch],
    run_dir: Path,
    replay_dir: Path,
    uploader: BackgroundUploader | None,
    loader_wait_fractions: list[float],
    loader_state: dict[str, object],
    identity_masker_state: dict[str, object],
    update: int,
    actual_loss_positions: int,
    smoke: bool,
    smoke_eval_matchups: int,
    arm_guard: _ArmGuard,
) -> None:
    if update == cfg.max_steps and actual_loss_positions != cfg.target_positions:
        raise RuntimeError(
            f"O51 stopped at {actual_loss_positions} valid positions instead of D={cfg.target_positions}"
        )
    _baseline_finalize_training(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        stats=stats,
        val_cache=val_cache,
        run_dir=run_dir,
        replay_dir=replay_dir,
        uploader=uploader,
        loader_wait_fractions=loader_wait_fractions,
        loader_state=loader_state,
        identity_masker_state=identity_masker_state,
        update=update,
        actual_loss_positions=actual_loss_positions,
        smoke=smoke,
        smoke_eval_matchups=smoke_eval_matchups,
        arm_guard=arm_guard,
    )
    if not smoke and model_level(cfg.arch) == "large" and cfg.target_positions == D0 and cfg.tier_scale == 1:
        selection = data_selection(cfg)
        evidence_path = run_dir / "large-d0-evidence.json"
        evidence = {
            "completed": True,
            "experiment_id": _EXPERIMENT_ID,
            "model_level": "large",
            "target_positions": D0,
            "tier_scale": 1,
            "corpus_hash": selection.corpus_hash,
            "checkpoint_sha256": _checkpoint_sha256(run_dir / "final.pt"),
        }
        evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
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
    architecture = replace(Architecture(), **architecture_values)
    calibration = replace(AWRCalibration(), **calibration_values)
    cfg = replace(
        _FROZEN_RUNTIME_DEFAULT_CONFIG,
        arch=architecture,
        awr=calibration,
        **runtime,
    )
    if cfg.max_steps != values["max_steps"] or cfg.warmup_steps != values["warmup_steps"]:
        raise ValueError("checkpoint position schedule is not derivable from D and batch size")
    validate_config(cfg)
    return cfg


def _config_from_eval_state(values: dict[str, object]) -> TrainConfig:
    """Reject checkpoints from every earlier O51 data protocol."""
    return config_from_state(values)


def load_checkpoint(
    path: str,
    *,
    device: str = DEVICE,
) -> tuple[Policy, TrainConfig, dict[str, FeatureStats], dict[str, object]]:
    """Load an O51 checkpoint without weakening strict resume compatibility."""
    loaded = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(loaded, dict):
        raise TypeError("O51 checkpoint must contain a mapping")
    state = cast(dict[str, object], loaded)
    config_state = state.get("cfg")
    if not isinstance(config_state, dict):
        raise TypeError("O51 checkpoint has no configuration mapping")
    cfg = _config_from_eval_state(cast(dict[str, object], config_state))
    validate_config(cfg)
    raw_model_state = state.get("model")
    if not isinstance(raw_model_state, dict):
        raise TypeError("O51 checkpoint has no model state mapping")
    model_state = cast(dict[str, Tensor], raw_model_state)
    encoded = model_state.get("player_code_bytes")
    if not isinstance(encoded, Tensor) or not encoded.numel():
        raise ValueError("checkpoint has no embedded identity vocabulary")
    vocabulary = PlayerVocabulary(decode_player_codes(encoded.detach().cpu().numpy().tobytes()))
    model = Policy(cfg, vocabulary).to(device)
    model.load_state_dict(model_state)
    model.eval()
    stats = load_stats(cfg)
    return model, cfg, stats, state


def validate_production_config(cfg: TrainConfig) -> None:
    """O51's declared grids are treatments, not forbidden production overrides."""
    validate_config(cfg)
    if cfg.num_workers not in LOADER_WORKERS:
        raise ValueError(f"production num_workers must be one of {LOADER_WORKERS}")
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
            "prefetch_factor": PREFETCH_FACTOR,
            "replay_slots": DEFAULT_REPLAY_SLOTS,
            "windows_per_generation": WINDOWS_PER_GENERATION,
            "order": "keyed_physical_shards",
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
    model = Policy(cfg).to(DEVICE).train()
    counts = subsystem_parameter_counts(model)
    flops_per_update = approximate_training_flops_per_update(cfg, counts)
    optimizer = make_optimizer(model, cfg)
    scheduler = LambdaLR(optimizer, lr_schedule(cfg))
    trunk_fn, temporal_fn = _training_functions(model, cfg)
    batch = synthetic_awr_batch(cfg, torch.device(DEVICE))
    valid_prefixes = cfg.batch_size * _SUPERVISED_POSITIONS_PER_WINDOW
    for index in range(warmup_steps):
        train_step(
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
    timers: list[CudaPhaseTimer] = []
    started = time.monotonic()
    for index in range(measured_steps):
        timer = CudaPhaseTimer()
        timer.record("start")
        timer.record("h2d_end")
        train_step(
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
    elapsed = time.monotonic() - started
    update_s = elapsed / measured_steps
    device_name = torch.cuda.get_device_name()
    peak_flops = bf16_dense_peak_flops(device_name)
    phase_metrics = _mean_phase_metrics(timers)
    metrics = {
        "batch_size": float(cfg.batch_size),
        "measured_steps": float(measured_steps),
        "update_s": update_s,
        "samples_per_s": cfg.batch_size / update_s,
        "synthetic_mfu": 0.0
        if peak_flops is None
        else model_flops_utilization(flops_per_update, update_s, peak_flops),
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
    warmup_batches: int = 4_096,
    measured_batches: int = 1_000,
) -> dict[str, object]:
    """Measure the direct-source loader without running the model."""
    if warmup_batches < 1 or measured_batches < 1:
        raise ValueError("loader benchmark batch counts must be positive")
    validate_config(cfg)
    stats = load_stats(cfg)
    player_lookup = ReplayPlayerLookup(load_identity_sidecar(cfg).by_replay)
    loader = _make_train_loader(cfg, stats, player_lookup)
    loader_started = time.monotonic()
    iterator = iter(loader)
    worker_start_seconds = time.monotonic() - loader_started
    fill_started = time.monotonic()
    next(iterator)
    initial_fill_seconds = time.monotonic() - fill_started
    steady_warmup_started = time.monotonic()
    for warmup_batch in range(1, warmup_batches):
        next(iterator)
        if warmup_batches >= 1_024 and (warmup_batch + 1) % 512 == 0:
            print(
                json.dumps(
                    {
                        "event": "loader_warmup",
                        "completed_batches": warmup_batch + 1,
                        "generations_read": int(getattr(loader, "generations_read", 0)),
                        "raw_gib": int(getattr(loader, "raw_bytes_read", 0)) / 2**30,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    steady_warmup_seconds = time.monotonic() - steady_warmup_started

    batch_seconds: list[float] = []
    replay_frequencies: defaultdict[str, int] = defaultdict(int)
    rank_bucket_count = 256
    rank_observed = np.zeros(rank_bucket_count, dtype=np.int64)
    rank_expected = np.zeros(rank_bucket_count, dtype=np.float64)
    rank_pearson_denominator = np.zeros(rank_bucket_count, dtype=np.float64)
    bucket_sizes_by_active_count: dict[int, np.ndarray] = {}
    adjacent_repeats: list[int] = []
    adjacent_expected: list[float] = []
    adjacent_variances: list[float] = []
    prior_16: list[set[str]] = []
    repeats_within_16: list[int] = []
    prior_16_expected: list[float] = []
    selection_outside_active = False
    raw_bytes_at_start = int(getattr(loader, "raw_bytes_read", 0))
    generations_at_start = int(getattr(loader, "generations_read", 0))
    started = time.monotonic()
    for _batch_index in range(measured_batches):
        previous = prior_16[-1] if prior_16 else set()
        recent = set().union(*prior_16) if prior_16 else set()
        count_active = getattr(loader, "count_active_replay_ids", None)
        if callable(count_active):
            adjacent_candidates = int(count_active(previous))
            recent_candidates = int(count_active(recent))
        else:
            adjacent_candidates = len(previous)
            recent_candidates = len(recent)
        batch_started = time.monotonic()
        batch = next(iterator)
        batch_seconds.append(time.monotonic() - batch_started)
        if not isinstance(batch, AWRBatch) or batch.batch.replay_ids is None:
            raise TypeError("O51 loader benchmark requires replay-aware AWR batches")
        if len(batch.batch.replay_ids) != cfg.batch_size or len(set(batch.batch.replay_ids)) != cfg.batch_size:
            raise RuntimeError("O51 loader emitted a batch with repeated or missing replay IDs")
        current = set(batch.batch.replay_ids)
        active_count = int(getattr(loader, "sampled_identity_count", len(current)))
        sampled_ranks = tuple(getattr(loader, "sampled_identity_ranks", range(active_count)))
        if len(sampled_ranks) != cfg.batch_size or len(set(sampled_ranks)) != cfg.batch_size:
            raise RuntimeError("O51 loader did not report one distinct identity rank per batch row")
        selection_outside_active |= any(not 0 <= rank < active_count for rank in sampled_ranks)
        bucket_indices = np.asarray(sampled_ranks, dtype=np.int64) * rank_bucket_count // active_count
        rank_observed += np.bincount(bucket_indices, minlength=rank_bucket_count)
        bucket_sizes = bucket_sizes_by_active_count.get(active_count)
        if bucket_sizes is None:
            all_buckets = np.arange(active_count, dtype=np.int64) * rank_bucket_count // active_count
            bucket_sizes = np.bincount(all_buckets, minlength=rank_bucket_count)
            bucket_sizes_by_active_count[active_count] = bucket_sizes
        bucket_probability = bucket_sizes / active_count
        rank_expected += cfg.batch_size * bucket_probability
        if active_count > cfg.batch_size:
            finite_population_correction = (active_count - cfg.batch_size) / (active_count - 1)
            rank_pearson_denominator += cfg.batch_size * bucket_probability * finite_population_correction
        inclusion_probability = cfg.batch_size / active_count
        adjacent_repeats.append(len(current & previous))
        repeats_within_16.append(len(current & recent))
        adjacent_expected.append(adjacent_candidates * inclusion_probability)
        prior_16_expected.append(recent_candidates * inclusion_probability)
        if active_count > 1:
            marked_fraction = adjacent_candidates / active_count
            adjacent_variances.append(
                cfg.batch_size
                * marked_fraction
                * (1 - marked_fraction)
                * (active_count - cfg.batch_size)
                / (active_count - 1)
            )
        else:
            adjacent_variances.append(0.0)
        for replay_id in current:
            replay_frequencies[replay_id] += 1
        prior_16.append(current)
        if len(prior_16) > 16:
            prior_16.pop(0)
    elapsed = time.monotonic() - started
    windows = measured_batches * cfg.batch_size
    final_active = set(getattr(loader, "active_replay_ids", replay_frequencies))
    identity_universe = final_active | set(replay_frequencies)
    active_identities = int(getattr(loader, "active_identity_count", len(final_active)))
    observed_coverage = len(replay_frequencies) / len(identity_universe)
    expected_coverage = 1 - (1 - cfg.batch_size / active_identities) ** measured_batches
    valid_rank_buckets = rank_pearson_denominator > 0
    if valid_rank_buckets.any():
        rank_chi_square = float(
            np.sum(
                (rank_observed[valid_rank_buckets] - rank_expected[valid_rank_buckets]) ** 2
                / rank_pearson_denominator[valid_rank_buckets]
            )
        )
        rank_chi_square_dof = max(int(valid_rank_buckets.sum()) - 1, 1)
        rank_chi_square_z = (rank_chi_square - rank_chi_square_dof) / math.sqrt(2 * rank_chi_square_dof)
    else:
        rank_chi_square = 0.0
        rank_chi_square_dof = 0
        rank_chi_square_z = 0.0
    adjacent_variance = sum(adjacent_variances)
    adjacent_z = (
        (sum(adjacent_repeats) - sum(adjacent_expected)) / math.sqrt(adjacent_variance)
        if adjacent_variance > 0
        else 0.0
    )
    uniformity_passed = not selection_outside_active and abs(rank_chi_square_z) <= 6 and abs(adjacent_z) <= 6
    batch_mean = float(np.mean(batch_seconds))
    batch_p95 = float(np.percentile(batch_seconds, 95))
    batch_p99 = float(np.percentile(batch_seconds, 99))
    batch_cv = float(np.std(batch_seconds) / max(batch_mean, 1e-12))
    generations_read = int(getattr(loader, "generations_read", generations_at_start)) - generations_at_start
    turnover_per_batch = generations_read / measured_batches
    expected_turnover_per_batch = cfg.batch_size / WINDOWS_PER_GENERATION
    turnover_passed = generations_read == 0 or abs(turnover_per_batch / expected_turnover_per_batch - 1) <= 0.1
    stability_passed = (
        batch_p95 <= 2 * batch_mean and batch_p99 <= 3 * batch_mean and batch_cv <= 0.5 and turnover_passed
    )
    raw_bytes = int(getattr(loader, "raw_bytes_read", 0)) - raw_bytes_at_start
    pinned_batch_bytes = int(batch.target.numel() * batch.target.element_size())
    pinned_batch_bytes += int(batch.returns.numel() * batch.returns.element_size())
    pinned_batch_bytes += int(batch.eligible.numel() * batch.eligible.element_size())
    pinned_batch_bytes += sum(int(value.numel() * value.element_size()) for value in batch.context.features.values())
    pinned_batch_bytes += int(batch.context.ctx_pad.numel() * batch.context.ctx_pad.element_size())
    decoded_shard_bytes = int(getattr(loader, "max_decoded_shard_bytes", 0))
    host_memory = estimate_host_memory(
        central_buffer_bytes=int(getattr(loader, "buffer_bytes", 0)),
        decoded_shard_bytes=decoded_shard_bytes,
        replay_workspace_bytes=decoded_shard_bytes,
        pinned_batch_bytes=pinned_batch_bytes,
        validation_cache_bytes=4 * 2**30,
        compiler_and_process_bytes=64 * 2**30,
        workers=cfg.num_workers,
    )
    required_disk_bytes = int(getattr(loader, "required_disk_bytes", 0))
    disk_free_bytes = int(getattr(loader, "disk_free_bytes", 0))
    memory_passed = host_memory.peak_bytes <= 300 * 2**30
    metrics: dict[str, object] = {
        "fingerprint": preflight_fingerprint(cfg, data_selection(cfg)),
        "tier_scale": cfg.tier_scale,
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "replay_slots": int(getattr(loader, "replay_slots", active_identities)),
        "windows_per_generation": WINDOWS_PER_GENERATION,
        "loader_prefetch_factor": PREFETCH_FACTOR,
        "warmup_batches": warmup_batches,
        "worker_start_seconds": worker_start_seconds,
        "initial_fill_seconds": initial_fill_seconds,
        "steady_warmup_seconds": steady_warmup_seconds,
        "measured_batches": measured_batches,
        "measured_seconds": elapsed,
        "loader_only_windows_per_s": windows / elapsed,
        "batch_seconds_mean": batch_mean,
        "batch_seconds_p50": float(np.percentile(batch_seconds, 50)),
        "batch_seconds_p95": batch_p95,
        "batch_seconds_p99": batch_p99,
        "batch_seconds_cv": batch_cv,
        "loader_stability_passed": stability_passed,
        "distinct_replays": len(replay_frequencies),
        "active_replay_identities": active_identities,
        "eligible_identity_universe": len(identity_universe),
        "identity_coverage_fraction": observed_coverage,
        "expected_identity_coverage_fraction": expected_coverage,
        "identity_rank_bucket_count": rank_bucket_count,
        "identity_rank_chi_square": rank_chi_square,
        "identity_rank_chi_square_dof": rank_chi_square_dof,
        "identity_rank_chi_square_z": rank_chi_square_z,
        "adjacent_batch_repeats_mean": float(np.mean(adjacent_repeats)),
        "adjacent_batch_repeats_expected": float(np.mean(adjacent_expected)),
        "adjacent_batch_repeats_z": adjacent_z,
        "prior_16_batch_repeats_mean": float(np.mean(repeats_within_16)),
        "prior_16_batch_repeats_expected": float(np.mean(prior_16_expected)),
        "identity_uniformity_passed": uniformity_passed,
        "shuffle_passed": uniformity_passed,
        "within_batch_unique": True,
        "cooldown_batches": 0,
        "generations_read": generations_read,
        "generation_turnover_per_batch": turnover_per_batch,
        "expected_generation_turnover_per_batch": expected_turnover_per_batch,
        "steady_state_turnover_passed": turnover_passed,
        "raw_bytes_read": raw_bytes,
        "raw_bytes_per_window": raw_bytes / windows,
        "central_buffer_bytes": int(getattr(loader, "buffer_bytes", 0)),
        "max_decoded_shard_bytes": decoded_shard_bytes,
        "projected_host_peak_bytes": host_memory.peak_bytes,
        "memory_passed": memory_passed,
        "required_disk_bytes": required_disk_bytes,
        "disk_free_bytes": disk_free_bytes,
        "disk_requirement_passed": disk_free_bytes >= required_disk_bytes,
        "system/disk/required_bytes": required_disk_bytes,
        "system/disk/free_bytes": disk_free_bytes,
        "system/cgroup/projected_peak_gib": host_memory.peak_bytes / 2**30,
        "source_sample_counts": loader.source_sample_counts,
    }
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    close_loader = getattr(loader, "close", None)
    if callable(close_loader):
        close_loader()
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


@dataclass
class TrainArgs:
    cfg: TrainConfig = dataclass_field(default_factory=TrainConfig)
    level: Literal["base", "proxy", "mid", "large"] | None = None
    comment: str = ""
    resume: str | None = None
    resume_checkpoint: str = "latest.pt"
    resume_as: str | None = None
    resume_num_workers: int | None = None
    smoke: bool = False
    stop_after_update: int | None = None
    smoke_eval_matchups: int = 4
    eval_max_parallel: int | None = None
    preflight_report: Path | None = None
    promotion_evidence: Path | None = None
    large_d0_evidence: Path | None = None


@dataclass
class BenchmarkArgs:
    level: Literal["base", "proxy", "mid", "large"] = "mid"
    batch_size: Literal[128, 256, 512, 1024] = 512
    depth_alpha: float = 0.5
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
    warmup_batches: int = 4_096
    measured_batches: int = 1_000


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
    | Annotated[PlanArgs, tyro.conf.subcommand(name="sweep-plan")]
    | Annotated[LaunchArgs, tyro.conf.subcommand(name="sweep-launch")]
    | Annotated[RankArgs, tyro.conf.subcommand(name="sweep-rank")]
    | Annotated[AdjudicateArgs, tyro.conf.subcommand(name="sweep-adjudicate")]
    | Annotated[SelectArgs, tyro.conf.subcommand(name="sweep-select")]
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
    if args.resume is None and args.resume_num_workers is not None:
        raise SystemExit("--resume-num-workers requires --resume")
    if args.resume is not None:
        checkpoint = Path(args.resume_checkpoint)
        if checkpoint.is_absolute() or ".." in checkpoint.parts or checkpoint.suffix != ".pt":
            raise SystemExit("--resume-checkpoint must be a relative .pt object within the run")
        resume_state = load_for_resume(
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
        )
        if args.resume_as is not None:
            if Path(args.resume_as).name != args.resume_as or args.resume_as in ("", ".", "..", args.resume):
                raise SystemExit("--resume-as must be one new run-name component")
            destination_exists = (Path("runs") / args.resume_as).exists()
            if cfg.push_to_r2:
                destination_exists = destination_exists or _remote_run_exists(args.resume_as)
            if destination_exists:
                raise SystemExit(f"resume destination {args.resume_as!r} already exists")
            resume_state = {**resume_state, "wandb_id": None}
    if args.eval_max_parallel is not None:
        cfg = replace(cfg, eval_max_parallel=args.eval_max_parallel)
    validate_config(cfg)
    if not args.smoke:
        _require_launch_evidence(cfg, args)
    stats = load_stats(cfg)
    train(
        cfg,
        stats,
        comment=args.comment,
        resume_run=resume_run,
        resume_state=resume_state,
        smoke=args.smoke,
        stop_after_update=args.stop_after_update,
        smoke_eval_matchups=args.smoke_eval_matchups,
        prepared_data_factory=_prepare_training_data,
    )


def main(args: Command) -> None:
    if isinstance(args, PlanArgs):
        _plan(args)
        return
    if isinstance(args, LaunchArgs):
        launch(args)
        return
    if isinstance(args, RankArgs):
        rank(args)
        return
    if isinstance(args, AdjudicateArgs):
        adjudicate(args)
        return
    if isinstance(args, SelectArgs):
        select(args)
        return
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
        run = wandb.Api().run(args.run)
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
            "num_workers",
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
        checkpoint = _resolve_eval_checkpoint(args.checkpoint, args.run)
        eval_checkpoint(
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


if __name__ == "__main__":
    main(tyro.cli(cast(type[Command], Command)))
