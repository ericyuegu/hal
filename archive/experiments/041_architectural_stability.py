"""Architecturally stabilized scaled AWR-BC with projectile inputs.

This experiment trains AWR over the complete policy-world corpus. A learned
value head estimates the return from each trunk state, and the detached
``G_{t+1} - V(s_t)`` advantage weights the policy objective. It includes the
schema-v6 projectile block (``item{0..3}_*``) in every observation.

Compared with experiment 040, action logits come from normalized linear
readouts without a parallel trunk-logit skip. The temporal decoder normalizes
its input and scales both residual branches symmetrically. Same-frame group
conditioning is bounded and identity-initialized, and the critic follows a
detached normalized trunk state so value regression cannot move policy
features. Optimizer settings intentionally remain the 040 baseline.

The four item slots are ordered by ascending spawn id, so a slot keeps its item
until an OLDER item despawns and the remaining items shift down. A pooled set
encoder makes that churn invisible: one shared per-slot encoder, gated by the
slot's presence flag, summed over the slots. An empty slot adds the exact zero
vector and the live-item count stays implicit in the sum.

Run:
    uv run experiments/041_architectural_stability.py train
    uv run experiments/041_architectural_stability.py benchmark-train-step
    uv run experiments/041_architectural_stability.py eval --checkpoint runs/<run>/final.pt
    uv run experiments/041_architectural_stability.py eval --run <run> --checkpoint checkpoints/<step>.pt
"""

import contextlib
import functools
import hashlib
import itertools
import json
import math
import os
import time
from collections.abc import Callable
from collections.abc import Iterable
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from dataclasses import fields
from dataclasses import replace
from pathlib import Path
from typing import Annotated
from typing import Literal
from typing import TypedDict
from typing import cast

import melee
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
from hal.eval.self_play import DecodeTelemetry
from hal.eval.self_play import benchmark_checkpoint as benchmark_self_play
from hal.eval.self_play import canonical_context
from hal.eval.self_play import synthetic_context as build_synthetic_context
from hal.sim.rollout import covering_power_of_two
from hal.training import returns as returns_lib
from hal.training import scoring
from hal.training.checkpoints import BackgroundUploader
from hal.training.checkpoints import download_latest
from hal.training.checkpoints import load_for_resume
from hal.training.checkpoints import save_checkpoint
from hal.training.closed_loop import RecedingHorizon
from hal.training.dataloader import ResumableStreamingDataLoader
from hal.training.dataloader import make_loader
from hal.training.ego_stats import load_consolidated_mixture_stats
from hal.training.features import A_DIM
from hal.training.features import ACTION_CHANNELS
from hal.training.features import BASE_ITEMS_PROJECTION
from hal.training.features import BASE_PLAYER_PREFIXES
from hal.training.features import CAT_FEATURES
from hal.training.features import FLOAT_FEATURES
from hal.training.features import ITEM_COLUMNS
from hal.training.features import Context
from hal.training.features import ExtraColumns
from hal.training.features import FeatureProjection
from hal.training.features import TrainBatch
from hal.training.features import stack_actions
from hal.training.mfu import bf16_dense_peak_flops
from hal.training.mfu import bf16_peak_source
from hal.training.mfu import model_flops_utilization
from hal.training.muon import SingleDeviceMuonWithAuxAdam
from hal.training.runs import make_run_name
from hal.training.runs import setup_run_dir
from hal.training.system_metrics import HostMetricsSampler
from hal.training.trunk import Rotary
from hal.training.trunk import Trunk
from hal.training.trunk import TrunkConfig
from hal.training.trunk import apply_rotary_emb
from hal.wire import ITEM_SLOTS
from hal.wire import item_column

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_LN2 = math.log(2.0)
_N_CONT = 6
_PLAYER_PREFIXES = BASE_PLAYER_PREFIXES
_EXPERIMENT_ID = "041_architectural_stability_v3"
_RETURN_SUFFIX = "awr_return"
EGO_RETURN = f"ego_{_RETURN_SUFFIX}"
EGO_RETURN_VALID = f"{EGO_RETURN}_valid"
_INFERENCE_BUCKETS = (1, 2, 4, 8, 16, 32, 64)
_PRODUCTION_LOSS_POSITIONS = 2**35
_PRODUCTION_EVAL_MATCHUPS = 96
_N_NEAR = 6
EVAL_HORIZONS = (1, 2, 4, 6)
_TRAIN_METRICS_EVERY = 1
_TRAIN_PREFETCH_FACTOR = 4
_TRAIN_COMPILE_MODE = "reduce-overhead"
_TRUNK_ATTENTION_BACKEND = "varlen_flash"
_ADAM_UPDATE_CLIP_THRESHOLD = 1.0
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

GROUP_NAMES: tuple[str, ...] = ("buttons", "main_stick", "c_stick", "triggers")
GROUP_VOCABS: tuple[int, ...] = (
    scoring.N_BUTTON_COMBOS,
    scoring.STICK_CLUSTER_CENTERS_MAIN.shape[0],
    scoring.STICK_CLUSTER_CENTERS_C.shape[0],
    scoring.TRIGGER_CENTERS.shape[0] ** 2,
)
N_GROUPS = len(GROUP_NAMES)
BUTTONS_G, MAIN_G, C_G, TRIG_G = range(N_GROUPS)
GROUP_INDEX = {name: index for index, name in enumerate(GROUP_NAMES)}
GROUP_ORDER: tuple[str, ...] = ("c_stick", "main_stick", "triggers", "buttons")

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


@dataclass(frozen=True)
class Architecture:
    d_model: int = 1024
    n_layers: int = 16
    n_heads: int = 16
    attn_window: int = 0
    L_ctx: int = 128

    sample_chunk_length: int = 24
    head_offsets: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 9, 12, 16, 20, 24)
    temporal_d_model: int = 512
    temporal_layers: int = 4
    temporal_heads: int = 8
    temporal_ff_dim: int = 1536
    group_head_dim: int = 512
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
    value_hidden_dim: int = 512


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

    exec_horizon: int = 4
    inference_mode: str = "compiled"  # explicit "eager" is for debugging
    # Hardware-derived by default. An explicit power of two is a reproducibility
    # or memory-pressure override, not an architecture parameter.
    compiled_inference_bucket: int | None = None

    seed: int = 0
    eval_seed: int = 0
    batch_size: int = 512
    target_loss_positions: int = _PRODUCTION_LOSS_POSITIONS
    muon_lr: float = 0.028
    muon_weight_decay: float = 0.1
    adam_lr: float = 8.5e-4
    adam_weight_decay: float = 0.0071
    grad_clip: float = 1.0
    warmup_fraction: float = 0.03
    stable_fraction: float = 0.80
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
    val_every: int = 8192
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
    cache_limit_gb: int = 1792
    shuffle_block_size: int = 8192
    predownload: int = 8192
    download_retry: int = 8
    loader_timeout_s: float = 300.0
    val_split: str = "val"
    num_workers: int = 32
    push_to_r2: bool = True
    system_metrics_every: int = 25
    system_metrics_interval_s: float = 5.0
    process_metrics_interval_s: float = 30.0
    cache_metrics_interval_s: float = 30.0
    phase_timing_every: int = 0

    @property
    def max_steps(self) -> int:
        positions_per_update = self.batch_size * self.arch.L_ctx
        if self.target_loss_positions % positions_per_update:
            raise ValueError("target_loss_positions must be divisible by batch_size * L_ctx")
        return self.target_loss_positions // positions_per_update

    @property
    def warmup_steps(self) -> int:
        return int(self.warmup_fraction * self.max_steps)

    @property
    def stable_steps(self) -> int:
        return int(self.stable_fraction * self.max_steps)


def validate_config(cfg: TrainConfig) -> None:
    positive = {
        "d_model": cfg.arch.d_model,
        "n_layers": cfg.arch.n_layers,
        "n_heads": cfg.arch.n_heads,
        "L_ctx": cfg.arch.L_ctx,
        "sample_chunk_length": cfg.arch.sample_chunk_length,
        "temporal_d_model": cfg.arch.temporal_d_model,
        "temporal_layers": cfg.arch.temporal_layers,
        "temporal_heads": cfg.arch.temporal_heads,
        "temporal_ff_dim": cfg.arch.temporal_ff_dim,
        "group_head_dim": cfg.arch.group_head_dim,
        "action_embed_dim": cfg.arch.action_embed_dim,
        "offset_embed_dim": cfg.arch.offset_embed_dim,
        "item_type_dim": cfg.arch.item_type_dim,
        "item_state_dim": cfg.arch.item_state_dim,
        "item_hidden_dim": cfg.arch.item_hidden_dim,
        "item_dim": cfg.arch.item_dim,
        "value_hidden_dim": cfg.arch.value_hidden_dim,
        "batch_size": cfg.batch_size,
        "target_loss_positions": cfg.target_loss_positions,
        "layer_rms_batch_size": cfg.layer_rms_batch_size,
        "download_retry": cfg.download_retry,
    }
    for name, value in positive.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if cfg.max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {cfg.max_steps}")
    if cfg.arch.d_model % cfg.arch.n_heads or cfg.arch.temporal_d_model % cfg.arch.temporal_heads:
        raise ValueError("model dimensions must be divisible by their head counts")
    if (cfg.arch.temporal_d_model // cfg.arch.temporal_heads) % 2:
        raise ValueError("temporal head dimension must be even for rotary positions")
    offsets = tuple(cfg.arch.head_offsets)
    if offsets != tuple(sorted(set(offsets))) or not offsets or offsets[0] != 1:
        raise ValueError(f"head_offsets must be sorted, unique, and start at 1, got {offsets}")
    if offsets[-1] > cfg.arch.sample_chunk_length:
        raise ValueError("head_offsets extend beyond sample_chunk_length")
    if offsets[:_N_NEAR] != tuple(range(1, _N_NEAR + 1)):
        raise ValueError("the near bucket must be the dense offset prefix 1..6")
    if cfg.exec_horizon not in (4, 6):
        raise ValueError("exec_horizon must be 4 or 6")
    if cfg.inference_mode not in ("compiled", "eager"):
        raise ValueError("inference_mode must be 'compiled' or 'eager'")
    if cfg.compiled_inference_bucket is not None and (
        cfg.compiled_inference_bucket < 1 or cfg.compiled_inference_bucket & (cfg.compiled_inference_bucket - 1)
    ):
        raise ValueError("compiled_inference_bucket must be a positive power of two")
    if cfg.eval_max_parallel is not None and (
        not isinstance(cfg.eval_max_parallel, int)
        or isinstance(cfg.eval_max_parallel, bool)
        or cfg.eval_max_parallel < 1
    ):
        raise ValueError("eval_max_parallel must be a positive integer")
    if not set(cfg.source_names) <= _POLICY_WORLD_NAMES:
        raise ValueError(
            "projectile inputs need policy-world sources: no other decoder emits the item columns, "
            f"so the projectile block would never reach the model; got {sorted(cfg.source_names)}"
        )
    if not math.isfinite(cfg.awr.auxiliary_loss_weight) or cfg.awr.auxiliary_loss_weight < 0:
        raise ValueError("aux_loss_weight must be finite and non-negative")
    for name, value in (
        ("gradient_hist_every", cfg.gradient_hist_every),
        ("weight_hist_every", cfg.weight_hist_every),
        ("layer_rms_every", cfg.layer_rms_every),
        ("architecture_metrics_every", cfg.architecture_metrics_every),
        ("system_metrics_every", cfg.system_metrics_every),
        ("phase_timing_every", cfg.phase_timing_every),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    for name, value in (
        ("system_metrics_interval_s", cfg.system_metrics_interval_s),
        ("process_metrics_interval_s", cfg.process_metrics_interval_s),
        ("cache_metrics_interval_s", cfg.cache_metrics_interval_s),
        ("loader_timeout_s", cfg.loader_timeout_s),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive, got {value!r}")
    if cfg.amp_dtype not in ("bfloat16", "float32"):
        raise ValueError("amp_dtype must be bfloat16 or float32")
    if not isinstance(cfg.num_workers, int) or isinstance(cfg.num_workers, bool) or not 0 <= cfg.num_workers <= 32:
        raise ValueError(f"num_workers must be an integer in [0, 32], got {cfg.num_workers!r}")
    if not 0.0 < cfg.awr.gamma < 1.0:
        raise ValueError(f"awr_gamma must be in (0, 1), got {cfg.awr.gamma}")
    if not math.isfinite(cfg.awr.beta) or cfg.awr.beta <= 0:
        raise ValueError(f"awr_beta must be finite and positive, got {cfg.awr.beta}")
    if not math.isfinite(cfg.awr.weight_max) or cfg.awr.weight_max <= 1:
        raise ValueError(f"awr_weight_max must be finite and above 1, got {cfg.awr.weight_max}")
    if not math.isfinite(cfg.awr.value_loss_weight) or cfg.awr.value_loss_weight < 0:
        raise ValueError("awr_value_loss_weight must be finite and non-negative")
    if not math.isfinite(cfg.grad_clip) or cfg.grad_clip <= 0:
        raise ValueError(f"grad_clip must be finite and positive, got {cfg.grad_clip}")
    if not 0.0 <= cfg.warmup_fraction < cfg.stable_fraction < 1.0:
        raise ValueError("schedule fractions must satisfy 0 <= warmup < stable < 1")
    if not 0.0 < cfg.lr_floor_ratio <= 1.0:
        raise ValueError("lr_floor_ratio must be in (0, 1]")
    if cfg.lr_schedule_kind not in ("late-cosine", "cosine"):
        raise ValueError("lr_schedule_kind must be 'late-cosine' or 'cosine'")
    if not cfg.source_names or len(set(cfg.source_names)) != len(cfg.source_names):
        raise ValueError("source_names must be non-empty and unique")
    unknown = set(cfg.source_names) - streams.BY_NAME.keys()
    if unknown:
        raise ValueError(f"unknown source names: {sorted(unknown)}")
    if cfg.policy_world_schema_version != POLICY_WORLD_SCHEMA_VERSION:
        raise ValueError(
            f"policy_world_schema_version {cfg.policy_world_schema_version} != {POLICY_WORLD_SCHEMA_VERSION}"
        )


def validate_production_config(cfg: TrainConfig) -> None:
    """Require frozen settings apart from operations and schedule ablations."""
    expected = asdict(TrainConfig())
    actual = asdict(cfg)
    allowed_overrides = _PRODUCTION_OVERRIDE_FIELDS | _PRODUCTION_ABLATION_FIELDS
    unknown_overrides = allowed_overrides - actual.keys()
    if unknown_overrides:
        raise RuntimeError(f"production overrides are not config fields: {sorted(unknown_overrides)}")
    changed = {
        name: (actual[name], expected_value)
        for name, expected_value in expected.items()
        if name not in allowed_overrides and actual[name] != expected_value
    }
    if changed:
        details = ", ".join(
            f"{name}={value!r} (expected {expected_value!r})"
            for name, (value, expected_value) in sorted(changed.items())
        )
        raise ValueError(f"production config differs from the frozen treatment: {details}")


def synthetic_context(cfg: TrainConfig, batch_size: int, device: torch.device) -> Context:
    """Build the fixed base observation with projectile columns."""
    return build_synthetic_context(
        cfg,
        batch_size,
        device,
        context_length=cfg.arch.L_ctx,
        observation_bundle="base",
        items=True,
    )


def synthetic_awr_batch(cfg: TrainConfig, device: torch.device) -> AWRBatch:
    """Build one fully valid production-shaped batch without touching the corpus."""
    context = synthetic_context(cfg, cfg.batch_size, device)
    target = torch.zeros(cfg.batch_size, cfg.arch.sample_chunk_length, A_DIM, device=device)
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


class StructuredControllerCodec(nn.Module):
    """The sole raw<->categorical boundary and shared semantic action token."""

    main_centers: Tensor
    c_centers: Tensor
    trigger_centers: Tensor
    button_valid_for_trigger: Tensor

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.class_embeddings = nn.ModuleDict(
            {name: nn.Embedding(GROUP_VOCABS[GROUP_INDEX[name]], embed_dim) for name in GROUP_NAMES}
        )
        semantic_dims = {"buttons": 8, "main_stick": 2, "c_stick": 2, "triggers": 2}
        self.semantic_projections = nn.ModuleDict(
            {name: nn.Linear(width, embed_dim, bias=False) for name, width in semantic_dims.items()}
        )
        self.register_buffer("main_centers", scoring.STICK_CLUSTER_CENTERS_MAIN.clone())
        self.register_buffer("c_centers", scoring.STICK_CLUSTER_CENTERS_C.clone())
        self.register_buffer("trigger_centers", scoring.TRIGGER_CENTERS.clone())
        button_bits = scoring.combo_to_buttons(torch.arange(GROUP_VOCABS[BUTTONS_G]))
        pair = torch.arange(GROUP_VOCABS[TRIG_G])
        n_trigger = len(self.trigger_centers)
        left_full = pair.div(n_trigger, rounding_mode="floor") == n_trigger - 1
        right_full = pair.remainder(n_trigger) == n_trigger - 1
        left_click = button_bits[:, BUTTON_L_CH - _N_CONT].bool()
        right_click = button_bits[:, BUTTON_R_CH - _N_CONT].bool()
        valid = (~left_click[None, :] | left_full[:, None]) & (~right_click[None, :] | right_full[:, None])
        self.register_buffer("button_valid_for_trigger", valid)

    def _class_embedding(self, name: str) -> nn.Embedding:
        return cast(nn.Embedding, self.class_embeddings[name])

    def _semantic_projection(self, name: str) -> nn.Linear:
        return cast(nn.Linear, self.semantic_projections[name])

    @staticmethod
    def canonicalize(actions: Tensor) -> Tensor:
        if actions.shape[-1] != A_DIM:
            raise ValueError(f"controller actions must end in {A_DIM} channels, got {tuple(actions.shape)}")
        out = actions.clone()
        out[..., TRIGGER_L_CH] = torch.where(
            out[..., BUTTON_L_CH] > 0.5, torch.ones_like(out[..., TRIGGER_L_CH]), out[..., TRIGGER_L_CH]
        )
        out[..., TRIGGER_R_CH] = torch.where(
            out[..., BUTTON_R_CH] > 0.5, torch.ones_like(out[..., TRIGGER_R_CH]), out[..., TRIGGER_R_CH]
        )
        return out

    def quantize(self, actions: Tensor) -> Tensor:
        actions = self.canonicalize(actions)
        continuous, buttons_raw = actions[..., :_N_CONT], actions[..., _N_CONT:]
        buttons = scoring.buttons_to_combo(buttons_raw)
        main = scoring.nearest_cluster(continuous[..., 0:2], self.main_centers)
        c_stick = scoring.nearest_cluster(continuous[..., 2:4], self.c_centers)
        trigger_pair = scoring.nearest_center(continuous[..., 4:6], self.trigger_centers)
        triggers = trigger_pair[..., 0] * self.trigger_centers.shape[0] + trigger_pair[..., 1]
        return torch.stack((buttons, main, c_stick, triggers), dim=-1)

    def dequantize(self, indices: Tensor) -> Tensor:
        n_trigger = self.trigger_centers.shape[0]
        buttons = scoring.combo_to_buttons(indices[..., BUTTONS_G])
        main = scoring.cluster_to_xy(indices[..., MAIN_G], self.main_centers)
        c_stick = scoring.cluster_to_xy(indices[..., C_G], self.c_centers)
        trigger_l = scoring.center_to_value(indices[..., TRIG_G] // n_trigger, self.trigger_centers)
        trigger_r = scoring.center_to_value(indices[..., TRIG_G] % n_trigger, self.trigger_centers)
        return torch.cat((main, c_stick, torch.stack((trigger_l, trigger_r), dim=-1), buttons), dim=-1)

    def semantic_values(self, name: str, indices: Tensor) -> Tensor:
        if name == "buttons":
            return scoring.combo_to_buttons(indices).to(self._class_embedding(name).weight.dtype)
        if name == "main_stick":
            return scoring.cluster_to_xy(indices, self.main_centers)
        if name == "c_stick":
            return scoring.cluster_to_xy(indices, self.c_centers)
        if name == "triggers":
            n = self.trigger_centers.shape[0]
            return torch.stack(
                (
                    scoring.center_to_value(indices // n, self.trigger_centers),
                    scoring.center_to_value(indices % n, self.trigger_centers),
                ),
                dim=-1,
            )
        raise ValueError(f"unknown controller group {name!r}")

    def group_embedding(self, name: str, indices: Tensor) -> Tensor:
        class_embedding = self._class_embedding(name)
        semantic = self.semantic_values(name, indices).to(class_embedding.weight.dtype)
        value = class_embedding(indices) + self._semantic_projection(name)(semantic)
        return decoder_rmsnorm(value)

    def embed_groups(self, indices: Tensor) -> dict[str, Tensor]:
        return {name: self.group_embedding(name, indices[..., GROUP_INDEX[name]]) for name in GROUP_NAMES}

    def embed_frame(self, indices: Tensor, embedded: dict[str, Tensor] | None = None) -> Tensor:
        values = self.embed_groups(indices) if embedded is None else embedded
        return torch.cat([values[name] for name in GROUP_NAMES], dim=-1)

    def button_mask(self, trigger_indices: Tensor) -> Tensor:
        return ~self.button_valid_for_trigger[trigger_indices]


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


class LinearActionHead(nn.Module):
    """A scale-controlled linear readout from temporal group features."""

    def __init__(self, d_model: int, vocab: int) -> None:
        super().__init__()
        self.output = nn.Linear(d_model, vocab, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.output(action_rmsnorm(x))

    def forward_with_input(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return logits and the exact normalized tensor read by the output."""
        normalized = action_rmsnorm(x)
        return self.output(normalized), normalized


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
    """RoPE causal SDPA block with an exact cached one-token path."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.n_heads = cfg.arch.temporal_heads
        self.d_model = cfg.arch.temporal_d_model
        self.head_dim = self.d_model // self.n_heads
        self.scale = 1.0 / math.sqrt(2 * cfg.arch.temporal_layers)
        self.qkv = nn.Linear(self.d_model, 3 * self.d_model, bias=False)
        self.proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.rotary = Rotary(self.head_dim)
        self.mlp = SwiGLU(self.d_model, cfg.arch.temporal_ff_dim, self.d_model)

    def _qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, length, _ = x.shape
        q, k, v = self.qkv(decoder_rmsnorm(x)).split(self.d_model, dim=-1)
        shape = (batch, length, self.n_heads, self.head_dim)
        return q.view(shape), k.view(shape), v.view(shape)

    def forward(self, x: Tensor) -> Tensor:
        q, k, v = self._qkv(x)
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin).transpose(1, 2)
        k = apply_rotary_emb(k, cos, sin).transpose(1, 2)
        v = v.transpose(1, 2)
        # Equal attention chunks keep the tiny BMMs efficient. The large linear
        # and SwiGLU operations use the full batch.
        attended = torch.cat(
            [
                short_causal_attention(query, key, values)
                for query, key, values in zip(
                    q.split(TEMPORAL_ATTENTION_BATCH),
                    k.split(TEMPORAL_ATTENTION_BATCH),
                    v.split(TEMPORAL_ATTENTION_BATCH),
                    strict=True,
                )
            ],
            dim=0,
        )
        attended = attended.transpose(1, 2).contiguous().view_as(x)
        x = x + self.scale * self.proj(attended)
        return x + self.scale * self.mlp(decoder_rmsnorm(x))

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
        x = x + self.scale * self.mlp(decoder_rmsnorm(x))
        return x, (k, v)


def sample_categorical(
    logits: Tensor,
    *,
    argmax: bool,
    uniform: Tensor | None = None,
    gen: torch.Generator | None = None,
) -> Tensor:
    values = logits.float()
    if argmax:
        return values.argmax(dim=-1)
    probabilities = F.softmax(values, dim=-1)
    if uniform is None:
        return torch.multinomial(probabilities, 1, generator=gen).squeeze(-1)
    if uniform.shape != probabilities.shape[:-1]:
        raise ValueError(f"uniform shape {tuple(uniform.shape)} != batch shape {tuple(probabilities.shape[:-1])}")
    uniform = uniform.to(device=probabilities.device, dtype=probabilities.dtype)
    return (probabilities.cumsum(-1) < uniform[..., None]).sum(-1).clamp_max(probabilities.shape[-1] - 1)


class CausalTemporalDecoder(nn.Module):
    """Temporal action chain conditioned by concatenation."""

    def __init__(self, cfg: TrainConfig, codec: StructuredControllerCodec) -> None:
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
        for module in self.group_condition.values():
            condition = cast(nn.Linear, module)
            nn.init.zeros_(condition.weight)
            nn.init.zeros_(condition.bias)
        self.outputs = nn.ModuleDict(
            {name: LinearActionHead(self.d_model, GROUP_VOCABS[GROUP_INDEX[name]]) for name in GROUP_NAMES}
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
        self,
        previous: Tensor,
        offset: int,
        state_bias: Tensor,
        caches: list[tuple[Tensor, Tensor] | None],
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor] | None]]:
        """Advance the temporal chain by one selected frame offset."""
        offsets = torch.full((previous.shape[0],), offset, device=previous.device, dtype=torch.long)
        state = decoder_rmsnorm(state_bias + self._step_features(previous, offsets))
        next_caches: list[tuple[Tensor, Tensor] | None] = []
        for module, past in zip(self.blocks, caches, strict=True):
            block = cast(TemporalBlock, module)
            state, present = block.forward_step(state, past)
            next_caches.append(present)
        return decoder_rmsnorm(state), next_caches

    def teacher_forced_states(self, hidden: Tensor, observed: Tensor, targets: Tensor) -> Tensor:
        expected = (*hidden.shape[:2], len(self.head_offsets), N_GROUPS)
        if observed.shape != (*hidden.shape[:2], N_GROUPS) or targets.shape != expected:
            raise ValueError(
                f"expected observed {(*hidden.shape[:2], N_GROUPS)} and targets {expected}, got "
                f"{tuple(observed.shape)} and {tuple(targets.shape)}"
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
        scale = 0.5 * torch.tanh(raw_scale)
        shift = torch.tanh(raw_shift)
        return states * (1.0 + scale) + shift

    def _teacher_forced_outputs(
        self,
        hidden: Tensor,
        observed: Tensor,
        targets: Tensor,
    ) -> tuple[dict[str, Tensor], tuple[Tensor, Tensor, Tensor, Tensor]]:
        """Return group logits and the button tensors used for diagnostics."""
        states = self.teacher_forced_states(hidden, observed, targets)
        embedded = self.codec.embed_groups(targets)
        logits: dict[str, Tensor] = {}
        button_values: tuple[Tensor, Tensor, Tensor] | None = None
        for name in GROUP_NAMES:
            features = self.group_features(states, name, embedded)
            if name == "buttons":
                head = cast(LinearActionHead, self.outputs[name])
                combined_logits, head_input = head.forward_with_input(features)
                button_values = (features, head_input, combined_logits)
            else:
                combined_logits = self.outputs[name](features)
            logits[name] = combined_logits
        if button_values is None:
            raise RuntimeError("button head was not evaluated")
        button_mask = self.codec.button_mask(targets[..., TRIG_G])
        logits["buttons"] = logits["buttons"].masked_fill(button_mask, float("-inf"))
        return logits, (*button_values, button_mask)

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
        logits = self.teacher_forced_logits_by_group(hidden, observed, targets)
        return self.nll_from_logits(logits, targets)

    def teacher_forced_nll_with_diagnostics(
        self,
        hidden: Tensor,
        observed: Tensor,
        targets: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Return NLL plus a compact button-boundary stability signal."""
        logits, button_values = self._teacher_forced_outputs(hidden, observed, targets)
        features, head_input, raw_logits, button_mask = button_values
        feature_rms = features.detach().float().square().mean(dim=-1).sqrt()
        input_values = head_input.detach()
        raw_logits_values = raw_logits.detach()
        button_targets = targets[..., BUTTONS_G, None]
        target_logits = raw_logits_values.gather(-1, button_targets).squeeze(-1).float()
        legal_logits = raw_logits_values.masked_fill(button_mask, float("-inf"))
        competing_logits = legal_logits.scatter(-1, button_targets, float("-inf")).amax(dim=-1).float()
        margin = target_logits - competing_logits
        metrics = {
            "stability/button_pre_norm_rms_min": feature_rms.amin(),
            "stability/button_input_abs_p999": _sampled_quantile(input_values, 99.9, absolute=True),
            "stability/button_logit_abs_p999": _sampled_quantile(raw_logits_values, 99.9, absolute=True),
            "stability/button_margin_mean": margin.mean(),
        }
        return self.nll_from_logits(logits, targets), metrics

    def teacher_forced_logits(self, hidden: Tensor, observed: Tensor, targets: Tensor) -> list[dict[str, Tensor]]:
        values = self.teacher_forced_logits_by_group(hidden, observed, targets)
        return [
            {name: logits[..., depth, :] for name, logits in values.items()} for depth in range(len(self.head_offsets))
        ]

    def forced_stepwise_logits(self, hidden: Tensor, observed: Tensor, targets: Tensor) -> list[dict[str, Tensor]]:
        if targets.shape != (hidden.shape[0], len(self.head_offsets), N_GROUPS):
            raise ValueError("stepwise targets have the wrong shape")
        trunk = decoder_rmsnorm(hidden[:, -1])
        state_bias = self._state_bias(trunk)
        previous = observed
        caches: list[tuple[Tensor, Tensor] | None] = [None] * len(self.blocks)
        out: list[dict[str, Tensor]] = []
        for depth, offset in enumerate(self.head_offsets):
            state, caches = self._decode_step(previous, offset, state_bias, caches)
            target = targets[:, depth]
            embedded = self.codec.embed_groups(target)
            group_logits = {
                name: self.outputs[name](self.group_features(state, name, embedded)) for name in GROUP_NAMES
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
        trunk = decoder_rmsnorm(hidden[:, -1])
        state_bias = self._state_bias(trunk)
        previous = observed
        caches: list[tuple[Tensor, Tensor] | None] = [None] * len(self.blocks)
        frames: list[Tensor] = []
        for depth, offset in enumerate(offsets):
            state, caches = self._decode_step(previous, offset, state_bias, caches)
            embedded: dict[str, Tensor] = {}
            picks: dict[str, Tensor] = {}
            for name in GROUP_ORDER:
                logits = self.outputs[name](self.group_features(state, name, embedded))
                if name == "buttons":
                    logits = logits.masked_fill(self.codec.button_mask(picks["triggers"]), float("-inf"))
                group = GROUP_INDEX[name]
                uniform = None if uniforms is None else uniforms[depth, group]
                pick = sample_categorical(logits, argmax=argmax, uniform=uniform, gen=gen)
                picks[name] = pick
                embedded[name] = self.codec.group_embedding(name, pick)
            indices = torch.stack([picks[name] for name in GROUP_NAMES], dim=-1)
            frames.append(indices)
            previous = indices
        return torch.stack(frames, dim=1)

    def rollout_conditioned_logits(self, hidden: Tensor, observed: Tensor) -> tuple[list[dict[str, Tensor]], Tensor]:
        """Offline ancestral diagnostic across every selected offset.

        Unlike :meth:`sample_indices`, this intentionally includes the sparse
        tail.  It is used only by validation to measure exposure gaps and is not
        reachable from the closed-loop inference wrapper.
        """
        trunk = decoder_rmsnorm(hidden[:, -1])
        state_bias = self._state_bias(trunk)
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
                logits = self.outputs[name](self.group_features(state, name, embedded))
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


class GPT(nn.Module):
    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.L_chunk = cfg.arch.sample_chunk_length
        self.head_offsets = tuple(cfg.arch.head_offsets)
        self.codec = StructuredControllerCodec(cfg.arch.action_embed_dim)
        self.cat_specs = {**CAT_FEATURES, "action": (cfg.arch.action_vocab, cfg.arch.action_state_embed_dim)}
        self.cat_embeds = nn.ModuleDict(
            {name: nn.Embedding(vocab, dim) for name, (vocab, dim) in self.cat_specs.items()}
        )
        self.char_emb = nn.Embedding(cfg.arch.char_vocab, cfg.arch.char_dim)
        self.stage_emb = nn.Embedding(cfg.arch.stage_vocab, cfg.arch.stage_dim)
        per_player = len(FLOAT_FEATURES) * 2 + sum(dim for _, dim in self.cat_specs.values())
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
        self.observation_encoder = SwiGLU(d_in, cfg.arch.d_model, cfg.arch.d_model, output_bias=True)
        self.trunk = Trunk(
            TrunkConfig(
                d_model=cfg.arch.d_model,
                n_layers=cfg.arch.n_layers,
                n_heads=cfg.arch.n_heads,
                L_ctx=cfg.arch.L_ctx,
                attn_window=cfg.arch.attn_window,
                attention_backend=_TRUNK_ATTENTION_BACKEND,
            )
        )
        self.temporal = CausalTemporalDecoder(cfg, self.codec)
        # V(s_t) predicts G_{t+1}, the return aligned with the next action. Keep
        # it last so the same seed preserves every policy parameter's draw.
        # Closed-loop inference never reads it.
        self.value_head = SwiGLU(cfg.arch.d_model, cfg.arch.value_hidden_dim, 1, output_bias=True)

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
                f"the observation carries no {_ITEM_PROBE_COLUMN!r} column; training needs policy-world "
                "sources and closed-loop evaluation needs projectile routing"
            )
        zeros = torch.zeros_like(features[_ITEM_PROBE_COLUMN])
        slots: list[Tensor] = []
        presence: list[Tensor] = []
        for slot in range(ITEM_SLOTS):
            # The stored type is peppi's raw u16 item id, so the clamp lands every id at
            # or above the last row on that row, which is the unknown projectile.
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
        if action_indices is None:
            action_indices = self.codec.quantize(stack_actions(features))
        parts = [self._per_player_features(features, prefix) for prefix in _PLAYER_PREFIXES]
        parts.append(self.codec.embed_frame(action_indices))
        parts.append(self.char_emb(features["ego_character"].clamp(0, self.char_emb.num_embeddings - 1)))
        parts.append(self.char_emb(features["opp_character"].clamp(0, self.char_emb.num_embeddings - 1)))
        parts.append(self.stage_emb(features["stage"].clamp(0, self.stage_emb.num_embeddings - 1)))
        parts.append(self._item_features(features))
        return self.observation_encoder(torch.cat(parts, dim=-1))

    def forward(self, features: dict[str, Tensor], ctx_pad: Tensor, action_indices: Tensor | None = None) -> Tensor:
        return self.trunk(self.context_tokens(features, action_indices), ctx_pad)

    def forward_dense(
        self,
        features: dict[str, Tensor],
        ctx_pad: Tensor,
        action_indices: Tensor | None = None,
    ) -> Tensor:
        """Run the shared model weights through the dense inference trunk."""
        return self.trunk.forward_dense(self.context_tokens(features, action_indices), ctx_pad)


@dataclass(frozen=True, slots=True)
class AWRBatch:
    """A policy batch with return targets aligned to the next frame.

    Ineligible positions are padding or truncated returns. They keep policy
    weight one and take no value loss. Their return may be NaN, so select them
    with ``eligible`` rather than masking by multiplication.
    """

    batch: TrainBatch
    returns: Float[Tensor, "B L_ctx"]
    eligible: Bool[Tensor, "B L_ctx"]

    @property
    def context(self) -> Context:
        return self.batch.context

    @property
    def target(self) -> Tensor:
        return self.batch.target

    def to(self, device: str | torch.device) -> AWRBatch:
        target_device = torch.device(device)
        return AWRBatch(
            batch=self.batch.to(target_device),
            returns=self.returns.to(target_device, non_blocking=True),
            eligible=self.eligible.to(target_device, non_blocking=True),
        )

    def pin_memory(self) -> AWRBatch:
        return AWRBatch(
            batch=self.batch.pin_memory(),
            returns=self.returns.pin_memory(),
            eligible=self.eligible.pin_memory(),
        )

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        """Keep staged device storage alive until the compute stream is done."""
        self.batch.record_stream(stream)
        self.returns.record_stream(stream)
        self.eligible.record_stream(stream)


@jaxtyped(typechecker=beartype)
def prepared_targets(
    model: GPT, batch: TrainBatch | AWRBatch
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
    return full[:, :length], targets, valid


class DeviceBatchPrefetcher:
    """Fetch the next CPU batch during GPU compute and stage it for transfer."""

    def __init__(self, loader: Iterable[AWRBatch], cfg: TrainConfig, device: str | torch.device) -> None:
        self._loader = loader
        self._iterator = iter(loader)
        self._cfg = cfg
        self._device = torch.device(device)
        self._copy_stream = torch.cuda.Stream(device=self._device) if self._device.type == "cuda" else None
        self._staged: tuple[AWRBatch, AWRBatch, int] | None = None
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="device-batch-prefetch")
        self._future: Future[AWRBatch] | None = None
        self.preload()

    def _load_cpu_batch(self) -> AWRBatch:
        try:
            cpu_batch = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self._loader)
            cpu_batch = next(self._iterator)
        if not isinstance(cpu_batch, AWRBatch):
            raise TypeError(f"advantage loader yielded {type(cpu_batch).__name__}, expected AWRBatch")
        validate_batch_geometry(cpu_batch, self._cfg, self._cfg.batch_size)
        return cpu_batch

    def _stage(self, cpu_batch: AWRBatch) -> None:
        valid_prefixes = int((self._cfg.arch.L_ctx - cpu_batch.context.ctx_pad).sum())
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
        cpu_batch = future.result()
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


def microbatch_loss(
    model: GPT,
    batch: AWRBatch,
    cfg: TrainConfig,
    *,
    step: int,
    valid_prefixes: int,
    trunk_fn: Callable,
    temporal_fn: Callable,
    phase_timer: CudaPhaseTimer | None = None,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Compute the learned-value AWR policy and critic losses.

    Weighting stays outside the compiled policy functions, so crossing the
    warmup boundary does not trigger recompilation. Logged NLLs are unweighted.
    """
    if not isinstance(batch, AWRBatch):
        raise TypeError(f"advantage training needs an AWRBatch, got {type(batch).__name__}")
    history, targets, valid = prepared_targets(model, batch)
    if phase_timer is not None:
        phase_timer.record("target_prep_end")
    with amp_context(cfg, DEVICE):
        hidden = trunk_fn(batch.context.features, batch.context.ctx_pad, history)
        if phase_timer is not None:
            phase_timer.record("trunk_end")
        temporal_output = temporal_fn(hidden, history, targets)
        if isinstance(temporal_output, Tensor):
            dense_nll = temporal_output
            button_diagnostics: dict[str, Tensor] = {}
        else:
            dense_nll, button_diagnostics = temporal_output
        if phase_timer is not None:
            phase_timer.record("temporal_end")
    value_features = decoder_rmsnorm(hidden).detach()
    value = model.value_head(value_features.float()).squeeze(-1)
    value_loss, advantage, value_stats = value_objective(
        value,
        batch.returns,
        batch.eligible,
        beta=cfg.awr.beta,
        valid=valid,
    )
    active = step >= cfg.warmup_steps
    weights, stats = advantage_weights(
        advantage,
        batch.eligible,
        beta=cfg.awr.beta,
        weight_max=cfg.awr.weight_max,
        active=active,
        valid=valid,
    )
    button_loss = dense_nll[..., BUTTONS_G].float().mean(dim=-1)
    stats["weight_button_loss_correlation"] = masked_correlation(
        weights,
        button_loss,
        batch.eligible & valid,
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
        "aux/near_loss": near.detach() / _LN2,
        "aux/far_loss": far.detach() / _LN2,
        "aux/total_loss": loss.detach(),
        "value/loss": value_stats["value_loss"],
        "value/rmse": value_stats["value_rmse"],
        "awr/active": torch.ones_like(loss) if active else torch.zeros_like(loss),
        "awr/ess": stats["weight_ess"],
        "awr/max_weight": stats["weight_max"],
        "awr/clip_fraction": stats["weight_clip_frac"],
        "awr/button_loss_correlation": stats["weight_button_loss_correlation"],
        **button_diagnostics,
    }
    if phase_timer is not None:
        phase_timer.record("objective_end")
    return loss, nll_sum.detach(), extra


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
def val_metrics(model: GPT, batches: list[TrainBatch], cfg: TrainConfig) -> dict[str, float]:
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
                hidden = model.forward_dense(batch.context.features, batch.context.ctx_pad, history)
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
    horizon = cfg.exec_horizon
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


_UINT64_MASK = (1 << 64) - 1


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return value ^ (value >> 31)


class SlotGroupRandom:
    """Counter RNG keyed by slot, match generation, and controller group."""

    def __init__(self, seed: int) -> None:
        self.seed = seed & _UINT64_MASK
        self.generations: dict[int, int] = {}
        self.counters: dict[tuple[int, int, str], int] = {}
        self.slot_ids: tuple[int, ...] = ()
        self.device = torch.device("cpu")

    def begin(self, ctx: Context) -> None:
        if ctx.slot_ids is None:
            raise ValueError("slot-keyed sampling needs slot_ids")
        slot_ids = tuple(int(value) for value in ctx.slot_ids.tolist())
        if len(set(slot_ids)) != len(slot_ids):
            raise ValueError(f"slot_ids must be unique, got {slot_ids}")
        resets = (False,) * len(slot_ids) if ctx.reset is None else tuple(bool(v) for v in ctx.reset.tolist())
        for slot_id, reset in zip(slot_ids, resets, strict=True):
            if slot_id not in self.generations:
                self.generations[slot_id] = 0
            elif reset:
                self.generations[slot_id] += 1
            generation = self.generations[slot_id]
            if reset or not any(key[:2] == (slot_id, generation) for key in self.counters):
                for name in GROUP_NAMES:
                    self.counters[(slot_id, generation, name)] = 0
        self.slot_ids = slot_ids
        self.device = ctx.slot_ids.device

    def uniforms(self, group: str) -> Tensor:
        if group not in GROUP_INDEX:
            raise ValueError(f"unknown group {group!r}")
        values = []
        group_key = _splitmix64(GROUP_INDEX[group] + 1)
        for slot_id in self.slot_ids:
            generation = self.generations[slot_id]
            key = (slot_id, generation, group)
            counter = self.counters[key]
            mixed = self.seed ^ _splitmix64(slot_id) ^ _splitmix64(generation) ^ group_key ^ _splitmix64(counter)
            values.append(((_splitmix64(mixed) >> 11) + 0.5) / (1 << 53))
            self.counters[key] = counter + 1
        return torch.tensor(values, device=self.device)

    def state(self) -> tuple[tuple[int, int, str, int], ...]:
        return tuple(sorted((*key, value) for key, value in self.counters.items()))


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
        model: GPT,
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
        streams: SlotGroupRandom | None = None,
        argmax: bool = False,
        gen: torch.Generator | None = None,
    ) -> Tensor:
        if horizon not in EVAL_HORIZONS:
            raise ValueError(f"horizon must be one of {EVAL_HORIZONS}")
        rows = ctx.ctx_pad.shape[0]
        bucket = self._bucket(rows)
        padded = canonical_context(_pad_context(ctx, bucket), "base", items=True)
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


def make_policy(
    model: GPT,
    stats: dict[str, FeatureStats],
    cfg: TrainConfig,
    *,
    exec_horizon: int | None = None,
    decode_seed: int | None = None,
    inference: BF16Inference | None = None,
    telemetry: DecodeTelemetry | None = None,
    device: str = DEVICE,
) -> RecedingHorizon:
    horizon = cfg.exec_horizon if exec_horizon is None else exec_horizon
    if horizon not in EVAL_HORIZONS:
        raise ValueError(f"execution horizon must be one of {EVAL_HORIZONS}")
    engine = BF16Inference(model, cfg) if inference is None else inference
    random_streams = None if decode_seed is None else SlotGroupRandom(decode_seed)
    generator = None if decode_seed is None else torch.Generator(device=device).manual_seed(decode_seed)

    @torch.no_grad()
    def predict(ctx: Context, committed: np.ndarray | None) -> np.ndarray:
        if committed is not None:
            raise ValueError("experiment 040 does not condition on a committed RTC prefix")
        started = time.perf_counter()
        result = engine.decode(ctx, horizon, streams=random_streams, gen=generator).cpu().numpy()
        if telemetry is not None:
            telemetry.record(rows=ctx.ctx_pad.shape[0], horizon=horizon, seconds=time.perf_counter() - started)
        return result

    return RecedingHorizon(
        predict_chunk=predict,
        stats=stats,
        L_ctx=cfg.arch.L_ctx,
        L_chunk=horizon,
        s=horizon,
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
    exec_horizon: int
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
    model: GPT,
    *,
    n_matchups: int,
    exec_horizon: int,
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
        exec_horizon=exec_horizon,
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
    model: GPT,
    stats: dict[str, FeatureStats],
    cfg: TrainConfig,
    *,
    n_matchups: int,
    replay_dir: Path,
    exec_horizon: int | None = None,
    checkpoint_sha256: str = "unavailable",
    inference: BF16Inference | None = None,
) -> dict[str, float]:
    horizon = cfg.exec_horizon if exec_horizon is None else exec_horizon
    inference = BF16Inference(model, cfg) if inference is None else inference
    if inference.model is not model:
        raise ValueError("the supplied inference engine must own the evaluation model")
    protocol = _eval_protocol(
        cfg,
        model,
        n_matchups=n_matchups,
        exec_horizon=horizon,
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

    def factory() -> RecedingHorizon:
        return make_policy(
            model,
            stats,
            cfg,
            exec_horizon=horizon,
            decode_seed=protocol.seed + next(policy_index),
            inference=inference,
            telemetry=telemetry,
        )

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
    metrics["exec_horizon"] = float(horizon)
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
        "exec_horizon": "horizon",
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
        if step <= cfg.warmup_steps:
            return step / max(cfg.warmup_steps, 1)
        decay_start = cfg.stable_steps if cfg.lr_schedule_kind == "late-cosine" else cfg.warmup_steps
        if step <= decay_start:
            return 1.0
        progress = (step - decay_start) / max(cfg.max_steps - 1 - decay_start, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return cfg.lr_floor_ratio + (1.0 - cfg.lr_floor_ratio) * cosine

    return schedule


def make_optimizer(model: GPT, cfg: TrainConfig) -> SingleDeviceMuonWithAuxAdam:
    embedding_modules = [
        model.cat_embeds,
        model.char_emb,
        model.stage_emb,
        model.codec.class_embeddings,
        model.temporal.offset_embedding,
    ]
    # The item encoder's linears stay in the decayed bucket; only the tables here.
    embedding_modules += [model.item_type_emb, model.item_state_emb]
    embedding_ids = {id(parameter) for module in embedding_modules for parameter in module.parameters()}
    muon_modules = (model.trunk.blocks, model.temporal.blocks)
    muon = [
        parameter
        for module in muon_modules
        for parameter in module.parameters()
        if parameter.ndim >= 2 and id(parameter) not in embedding_ids
    ]
    muon_ids = {id(parameter) for parameter in muon}
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for parameter in model.parameters():
        if id(parameter) in muon_ids:
            continue
        (no_decay if parameter.ndim < 2 or id(parameter) in embedding_ids else decay).append(parameter)
    if len(muon) + len(decay) + len(no_decay) != sum(1 for _ in model.parameters()):
        raise RuntimeError("optimizer parameter partition is incomplete")
    adam = dict(
        betas=(0.9, 0.95),
        eps=1e-10,
        update_clip_threshold=_ADAM_UPDATE_CLIP_THRESHOLD,
        use_muon=False,
    )
    optimizer = SingleDeviceMuonWithAuxAdam(
        [
            dict(
                params=muon,
                lr=cfg.muon_lr,
                momentum=0.95,
                weight_decay=cfg.muon_weight_decay,
                use_muon=True,
            ),
            dict(params=decay, lr=cfg.adam_lr, weight_decay=cfg.adam_weight_decay, **adam),
            dict(params=no_decay, lr=cfg.adam_lr, weight_decay=0.0, **adam),
        ]
    )
    optimizer.set_adam_diagnostic_parameters(_button_adam_parameters(model))
    return optimizer


def _button_adam_parameters(model: GPT) -> dict[str, nn.Parameter]:
    """Return the AdamW action-path matrices monitored every update."""
    button_head = cast(LinearActionHead, model.temporal.outputs["buttons"])
    return {
        "buttons_output_weight": cast(nn.Parameter, button_head.output.weight),
        "buttons_condition_weight": cast(nn.Parameter, model.temporal.group_condition["buttons"].weight),
        "token_projection_weight": cast(nn.Parameter, model.temporal.token_projection.weight),
    }


def _button_gradient_abs_max(model: GPT) -> Tensor:
    """Return one pre-clipping action-path gradient guardrail."""
    maxima: list[Tensor] = []
    for name, parameter in _button_adam_parameters(model).items():
        if parameter.grad is None:
            raise RuntimeError(f"button parameter {name!r} has no gradient")
        maxima.append(parameter.grad.detach().float().abs().amax())
    return torch.stack(maxima).amax()


def _stable_adam_metrics(diagnostics: dict[str, Tensor]) -> dict[str, Tensor]:
    """Collapse per-tensor Adam telemetry into two actionable guardrails."""
    update_rms = [value for name, value in diagnostics.items() if name.endswith("/prospective_update_rms")]
    clip_factors = [value for name, value in diagnostics.items() if name.endswith("/update_clip_factor")]
    if not update_rms or not clip_factors:
        raise RuntimeError("StableAdamW action-path diagnostics are missing")
    return {
        "stability/action_update_rms_max": torch.stack(update_rms).amax(),
        "stability/action_update_clip_factor_min": torch.stack(clip_factors).amin(),
    }


def subsystem_parameter_counts(model: GPT) -> dict[str, int]:
    all_parameters = tuple(model.parameters())
    trunk_ids = {id(parameter) for parameter in model.trunk.parameters()}
    head_modules = model.temporal.outputs
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
    return counts


def approximate_training_flops_per_update(cfg: TrainConfig, parameter_counts: dict[str, int]) -> int:
    """Estimate forward-backward FLOPs from each subsystem's parameter uses."""
    shared_parameters = parameter_counts["trunk"] + parameter_counts["other"] + parameter_counts["value_head"]
    temporal_parameters = parameter_counts["temporal_decoder"] + parameter_counts["group_heads"]
    positions = cfg.batch_size * cfg.arch.L_ctx
    return 6 * positions * (shared_parameters + len(cfg.arch.head_offsets) * temporal_parameters)


def model_tag(cfg: TrainConfig) -> str:
    offsets = "-".join(map(str, cfg.arch.head_offsets))
    treatment = f"awr-v-near-b{cfg.awr.beta:g}-g{cfg.awr.gamma:g}-wu{cfg.warmup_steps}"
    return (
        f"stable041-d{cfg.arch.d_model}-L{cfg.arch.n_layers}-h{cfg.arch.n_heads}-Lc{cfg.arch.L_ctx}-"
        f"t{cfg.arch.temporal_d_model}x{cfg.arch.temporal_layers}-o{offsets}-s{cfg.exec_horizon}-"
        f"linear-head-no-skip-projectiles-mix-r1-{cfg.lr_schedule_kind}-{treatment}"
    )


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


def _residual_layers(model: GPT) -> tuple[tuple[str, nn.Module], ...]:
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
    model: GPT,
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
        hidden_base = model.forward_dense(diagnostic.context.features, diagnostic.context.ctx_pad, history)
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
        diagnostic.returns,
        diagnostic.eligible,
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
        head = cast(LinearActionHead, model.temporal.outputs[name])
        geometry, output_vectors[name] = _matrix_geometry_metrics(
            f"head_geometry/{name}_output",
            head.output.weight,
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


@torch.no_grad()
@torch.compiler.disable
def layer_activation_rms_log(
    model: GPT,
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
        handles.append(block.mlp.register_forward_hook(capture_branch(name, "mlp", block.scale)))
    try:
        diagnostic = _diagnostic_batch(batch, max_rows)
        history, targets, _ = prepared_targets(model, diagnostic)
        device = next(model.parameters()).device
        with amp_context(cfg, device):
            hidden = model.forward_dense(diagnostic.context.features, diagnostic.context.ctx_pad, history)
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
        extra=ITEM_COLUMNS,
        projection=BASE_ITEMS_PROJECTION,
    )


def source_mixture_weights(cfg: TrainConfig) -> tuple[float, ...]:
    """Return the natural MDS replay-count mixture."""
    return tuple(float(streams.POLICY_WORLD_V7_TRAIN_REPLAYS[name]) for name in cfg.source_names)


def validate_batch_geometry(
    batch: TrainBatch | AWRBatch, cfg: TrainConfig, expected_batch_size: int | None = None
) -> None:
    if batch.target.shape[1:] != (cfg.arch.sample_chunk_length, A_DIM):
        raise ValueError(
            f"target must be [B, {cfg.arch.sample_chunk_length}, {A_DIM}], got {tuple(batch.target.shape)}"
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
    model: GPT,
    optimizer: SingleDeviceMuonWithAuxAdam,
    scheduler: LambdaLR,
    cfg: TrainConfig,
    uploader: BackgroundUploader | None,
    milestone: bool,
    wandb_id: str | None,
    actual_loss_positions: int,
    loader_state: dict[str, object],
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


def load_stats(cfg: TrainConfig) -> dict[str, FeatureStats]:
    sources = tuple(streams.BY_NAME[name] for name in cfg.source_names)
    return load_consolidated_mixture_stats(
        [source.local_root / "stats.json" for source in sources],
        source_mixture_weights(cfg),
        expected_mds_schema_version=cfg.mds_schema_version,
    )


def _make_loaders(cfg: TrainConfig, stats: dict[str, FeatureStats]):
    common = loader_kwargs(cfg, stats)
    projection = common["projection"]
    if projection is not None:
        common["projection"] = replace(projection, columns=projection.columns | {EGO_RETURN, EGO_RETURN_VALID})
    replay_transform = functools.partial(
        returns_lib.label_replay,
        gamma=cfg.awr.gamma,
        damage_shaping=cfg.awr.damage_shaping,
        win_reward=cfg.awr.win_reward,
        stock_value=cfg.awr.stock_value,
        suffix=_RETURN_SUFFIX,
    )
    train_loader = make_loader(
        split="train",
        num_workers=cfg.num_workers,
        prefetch_factor=_TRAIN_PREFETCH_FACTOR,
        drop_last=True,
        predownload=cfg.predownload,
        download_retry=cfg.download_retry,
        timeout=cfg.loader_timeout_s,
        resumable=True,
        windows_per_replay=1,
        replay_format="policy-world",
        replay_transform=replay_transform,
        batch_transform=functools.partial(collate_awr_batch, L_ctx=cfg.arch.L_ctx),
        **common,
    )
    validation: _LoaderKwargs = {**common, "batch_size": cfg.val_batch_size}
    val_loader = make_loader(
        split=cfg.val_split,
        num_workers=0,
        replay_format="policy-world",
        shuffle=True,
        **validation,
    )
    if not isinstance(train_loader, ResumableStreamingDataLoader):
        raise TypeError("training loader must expose Mosaic deterministic resumption")
    return train_loader, cache_validation(val_loader, cfg.val_n_samples)


def _init_wandb(cfg: TrainConfig, run_name: str, resume_state: dict | None) -> None:
    """Start tracking and declare the experiment's logging semantics."""
    wandb.init(
        project="hal",
        name=run_name,
        id=None if resume_state is None else resume_state.get("wandb_id"),
        resume="allow" if resume_state is not None else None,
        tags=[
            "gpt",
            "temporal-mtp",
            "advantage-weighted-bc",
            "scaled",
            "041",
            "architecture-stability",
            "projectiles",
            "length-weighted-mix",
            cfg.lr_schedule_kind,
        ],
        config=asdict(cfg),
        settings=wandb.Settings(
            x_stats_sampling_interval=5.0,
            x_stats_track_process_tree=True,
        ),
    )
    if wandb.run is None:
        return
    wandb.define_metric("global_step")
    wandb.define_metric("*", step_metric="global_step")
    wandb.run.summary["nll_semantics"] = (
        "train/loss is weighted policy loss in bits; train/nll is its unweighted counterpart; "
        "aux/total_loss is the optimized policy-plus-value objective"
    )
    wandb.run.summary["layer_rms_semantics"] = (
        "activation=block input; residual_branch=block output-input; "
        "residual_ratio=residual_branch/activation; attention_branch and mlp_branch include depth scaling; "
        "gradient=parameter-gradient RMS"
    )
    wandb.run.summary["architecture/treatment"] = (
        "RMSNorm->linear action heads; no trunk-logit skip; normalized temporal input; symmetrically scaled temporal "
        "residuals; bounded identity-initialized group conditioning; detached normalized critic input"
    )
    wandb.run.summary["optimizer/adam_update_clip_threshold"] = _ADAM_UPDATE_CLIP_THRESHOLD
    wandb.run.summary["optimizer/lr_schedule"] = cfg.lr_schedule_kind
    wandb.run.summary["optimizer/update_clip_semantics"] = (
        "per-tensor StableAdamW factor=min(1, threshold/R_updated); stability metrics aggregate the monitored "
        "action-path tensors"
    )
    if cfg.wandb_log_code:
        log_wandb_code(wandb.run)


def _watch_gradients(model: nn.Module, cfg: TrainConfig) -> None:
    """Ask W&B to log per-parameter gradient histograms during training."""
    if wandb.run is not None and cfg.gradient_hist_every > 0:
        wandb.watch(
            model,
            log="gradients",
            log_freq=cfg.gradient_hist_every,
            log_graph=False,
        )


def _log_training_summary(
    cfg: TrainConfig,
    parameter_counts: dict[str, int],
    *,
    flops_per_update: int,
    device_name: str | None,
    peak_flops: float | None,
) -> None:
    """Record the fixed model and corpus accounting for this run."""
    if wandb.run is None:
        return
    for name, value in parameter_counts.items():
        wandb.run.summary[f"parameters/{name}"] = value

    unique_replays = sum(streams.POLICY_WORLD_V7_TRAIN_REPLAYS[name] for name in cfg.source_names)
    unique_frames = sum(streams.POLICY_WORLD_V7_TRAIN_FRAMES[name] for name in cfg.source_names)
    source_weights = source_mixture_weights(cfg)
    source_weight_total = sum(source_weights)
    wandb.run.summary["data/unique_replays"] = unique_replays
    wandb.run.summary["data/unique_frames"] = unique_frames
    wandb.run.summary["data/processed_loss_positions"] = cfg.target_loss_positions
    wandb.run.summary["data/effective_epochs"] = cfg.target_loss_positions / unique_frames
    wandb.run.summary["data/D_over_N"] = cfg.target_loss_positions / parameter_counts["total"]
    wandb.run.summary["data/nominal_loss_positions_per_update"] = cfg.batch_size * cfg.arch.L_ctx
    wandb.run.summary["data/train_prefetch_factor"] = _TRAIN_PREFETCH_FACTOR
    wandb.run.summary["training/approx_flops_per_update"] = flops_per_update
    wandb.run.summary["training/flops_formula"] = (
        "6*B*L_ctx*(N_trunk+N_other+N_value+n_offsets*(N_temporal+N_group_heads))"
    )
    if device_name is not None:
        wandb.run.summary["hardware/gpu_name"] = device_name
    if peak_flops is not None:
        wandb.run.summary["hardware/bf16_dense_peak_tflops"] = peak_flops / 1e12
        source = bf16_peak_source(device_name or "")
        if source is not None:
            wandb.run.summary["hardware/bf16_dense_peak_source"] = source
    wandb.run.summary["data/source_mixing"] = "repeat_1"
    for name, weight in zip(cfg.source_names, source_weights, strict=True):
        wandb.run.summary[f"data/source_sampling_share/{name}"] = weight / source_weight_total


def _training_functions(model: GPT, cfg: TrainConfig) -> tuple[Callable, Callable]:
    """Return eager or singly compiled trunk and temporal training functions."""
    trunk_fn: Callable = model.forward
    temporal_fn: Callable = model.temporal.teacher_forced_nll_with_diagnostics
    if DEVICE == "cuda" and cfg.compile_trunk:
        # Resolve FlexAttention before Dynamo sees the model. This entrypoint is
        # the sole compilation owner for the raw mask and attention operations.
        model.trunk.resolve_attention(DEVICE)
        if model.trunk.attn_path not in ("flex", "varlen_flash"):
            raise RuntimeError(
                f"compiled CUDA training requires a fused attention path, resolved {model.trunk.attn_path!r} instead"
            )
        trunk_fn = torch.compile(
            trunk_fn,
            dynamic=False,
            fullgraph=True,
            mode=_TRAIN_COMPILE_MODE,
        )
    if DEVICE == "cuda" and cfg.compile_temporal:
        temporal_fn = torch.compile(
            temporal_fn,
            dynamic=False,
            fullgraph=True,
            mode=_TRAIN_COMPILE_MODE,
        )
    return trunk_fn, temporal_fn


def _training_diagnostics(model: GPT, batch: AWRBatch, cfg: TrainConfig, update: int) -> dict[str, object]:
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
    model: GPT,
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


def benchmark_train_step(
    *,
    warmup_steps: int = 3,
    measured_steps: int = 20,
    profile_steps: int = 0,
    trace_path: str | None = None,
) -> dict[str, float]:
    """Benchmark the frozen production train step on a synthetic device batch."""
    if DEVICE != "cuda":
        raise RuntimeError("the production train-step benchmark requires CUDA")
    if warmup_steps < 1 or measured_steps < 1 or profile_steps < 0:
        raise ValueError("warmup_steps and measured_steps must be positive; profile_steps cannot be negative")
    if trace_path is not None and profile_steps == 0:
        raise ValueError("trace_path requires at least one profile step")

    cfg = replace(
        TrainConfig(),
        gradient_hist_every=0,
        weight_hist_every=0,
        layer_rms_every=0,
        phase_timing_every=0,
        push_to_r2=False,
        wandb_log_code=False,
    )
    validate_config(cfg)
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high" if cfg.allow_tf32 else "highest")
    model = GPT(cfg).to(DEVICE).train()
    optimizer = make_optimizer(model, cfg)
    muon_parameters = next(group["params"] for group in optimizer.param_groups if group["use_muon"])
    muon_shapes = {
        tuple(parameter.shape) if parameter.ndim != 4 else (len(parameter), parameter.numel() // len(parameter))
        for parameter in muon_parameters
    }
    scheduler = LambdaLR(optimizer, lr_schedule(cfg))
    trunk_fn, temporal_fn = _training_functions(model, cfg)
    batch = synthetic_awr_batch(cfg, torch.device(DEVICE))
    valid_prefixes = cfg.batch_size * cfg.arch.L_ctx

    print(
        f"[benchmark] compiling/warming {warmup_steps} production train steps "
        f"at batch={cfg.batch_size}, context={cfg.arch.L_ctx}",
        flush=True,
    )
    for index in range(warmup_steps):
        train_step(
            model,
            batch,
            cfg,
            step=index,
            update=index + 2,
            valid_prefixes=valid_prefixes,
            trunk_fn=trunk_fn,
            temporal_fn=temporal_fn,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    phase_timers: list[CudaPhaseTimer] = []
    wall_started = time.monotonic()
    for index in range(measured_steps):
        timer = CudaPhaseTimer()
        timer.record("start")
        timer.record("h2d_end")
        train_step(
            model,
            batch,
            cfg,
            step=warmup_steps + index,
            update=warmup_steps + index + 2,
            valid_prefixes=valid_prefixes,
            trunk_fn=trunk_fn,
            temporal_fn=temporal_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            phase_timer=timer,
        )
        phase_timers.append(timer)
    torch.cuda.synchronize()
    wall_seconds = time.monotonic() - wall_started

    phase_samples: dict[str, list[float]] = {}
    for timer in phase_timers:
        for name, value in timer.metrics().items():
            phase_samples.setdefault(name, []).append(value)
    metrics = {
        "batch_size": float(cfg.batch_size),
        "measured_steps": float(measured_steps),
        "muon_bucket_count": float(len(muon_shapes)),
        "muon_matrix_count": float(len(muon_parameters)),
        "update_s": wall_seconds / measured_steps,
        "samples_per_s": cfg.batch_size * measured_steps / wall_seconds,
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
        **{f"{name}_median": float(np.median(values)) for name, values in phase_samples.items()},
    }
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    if profile_steps:
        print(f"[benchmark] profiling {profile_steps} additional train steps", flush=True)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
        ) as profiler:
            for index in range(profile_steps):
                train_step(
                    model,
                    batch,
                    cfg,
                    step=warmup_steps + measured_steps + index,
                    update=warmup_steps + measured_steps + index + 2,
                    valid_prefixes=valid_prefixes,
                    trunk_fn=trunk_fn,
                    temporal_fn=temporal_fn,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )
        torch.cuda.synchronize()
        print(
            profiler.key_averages(group_by_input_shape=True).table(
                sort_by="self_cuda_time_total",
                row_limit=50,
            ),
            flush=True,
        )
        if trace_path is not None:
            trace = Path(trace_path)
            trace.parent.mkdir(parents=True, exist_ok=True)
            profiler.export_chrome_trace(str(trace))
            print(f"[benchmark] wrote GPU trace to {trace}", flush=True)
    return metrics


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


def _minimal_system_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Keep one host metric for each distinct resource constraint."""
    names = {
        "system/cgroup/current_gib": "system/memory_gb",
        "system/cgroup/usage_fraction": "system/memory_fraction",
        "system/process_tree/pss_gib": "system/process_memory_gb",
        "system/cache/allocated_gib": "system/cache_gb",
        "system/telemetry_errors": "system/telemetry_errors",
    }
    return {output: metrics[source] for source, output in names.items() if source in metrics}


def _finalize_training(
    *,
    model: GPT,
    optimizer: SingleDeviceMuonWithAuxAdam,
    scheduler: LambdaLR,
    cfg: TrainConfig,
    stats: dict[str, FeatureStats],
    val_cache: list[TrainBatch],
    run_dir: Path,
    replay_dir: Path,
    uploader: BackgroundUploader | None,
    eval_inference: BF16Inference | None,
    loader_wait_fractions: list[float],
    loader_state: dict[str, object],
    update: int,
    actual_loss_positions: int,
    smoke: bool,
    smoke_eval_matchups: int,
) -> None:
    """Save and evaluate the final model, then enforce the smoke loader gate."""
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
    )
    final_path = run_dir / ("smoke-final.pt" if smoke else "final.pt")
    _replace_link(snapshot, final_path)
    if uploader is not None:
        uploader.upload(snapshot, key=final_path.name)

    checkpoint_sha = _checkpoint_sha256(final_path)
    validation = _validation_wandb_metrics(val_metrics(model, val_cache, cfg), cfg)
    final_metrics = {f"val/{name}": value for name, value in validation.items()}
    final_matchups = smoke_eval_matchups if smoke else cfg.final_eval_n_matchups
    if final_matchups:
        if eval_inference is None:
            eval_inference = BF16Inference(model, cfg)
        final_eval = eval_vs_cpu(
            model,
            stats,
            cfg,
            n_matchups=final_matchups,
            replay_dir=replay_dir / ("smoke-final" if smoke else "final"),
            checkpoint_sha256=checkpoint_sha,
            inference=eval_inference,
        )
        if not smoke:
            require_complete_eval(final_eval, cfg.final_eval_n_matchups)
        final_metrics.update({f"eval/{name}": value for name, value in _eval_wandb_metrics(final_eval).items()})
    wandb.log({"global_step": update, **final_metrics})

    mean_wait = float(np.mean(loader_wait_fractions)) if loader_wait_fractions else 0.0
    p95_wait = float(np.percentile(loader_wait_fractions, 95)) if loader_wait_fractions else 0.0
    print(f"[loader] mean wait={100 * mean_wait:.2f}%, p95={100 * p95_wait:.2f}%", flush=True)
    if smoke and (mean_wait > 0.05 or p95_wait > 0.10):
        raise RuntimeError("smoke loader gate failed: require mean wait <=5% and p95 <=10%")


def train(
    cfg: TrainConfig,
    stats: dict[str, FeatureStats],
    *,
    comment: str = "",
    resume_run: str | None = None,
    resume_state: dict | None = None,
    smoke: bool = False,
    stop_after_update: int | None = None,
    smoke_eval_matchups: int = 4,
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
    run_dir, replay_dir = setup_run_dir(run_name)
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high" if cfg.allow_tf32 else "highest")
    model = GPT(cfg).to(DEVICE)
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
    if resume_state is not None:
        model.load_state_dict(resume_state["model"])
        optimizer.load_state_dict(resume_state["opt"])
        scheduler.load_state_dict(resume_state["sched"])
        start_step = int(resume_state["step"]) + 1
        actual_positions = int(resume_state.get("actual_loss_positions", start_step * cfg.batch_size * cfg.arch.L_ctx))
        if not 0 <= actual_positions <= start_step * cfg.batch_size * cfg.arch.L_ctx:
            raise ValueError(
                f"checkpoint actual_loss_positions={actual_positions} is invalid after {start_step} updates"
            )

    trunk_fn, temporal_fn = _training_functions(model, cfg)

    train_loader, val_cache = _make_loaders(cfg, stats)
    if resume_state is not None:
        loader_state = resume_state.get("loader")
        if not isinstance(loader_state, dict):
            raise ValueError("resume checkpoint does not contain Mosaic streaming loader state")
        train_loader.load_state_dict(loader_state)
    run_started = time.monotonic()
    batch_prefetcher = DeviceBatchPrefetcher(train_loader, cfg, DEVICE)
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
    eval_inference: BF16Inference | None = None
    model.train()
    try:
        for step in range(start_step, run_stop):
            update = step + 1
            if DEVICE == "cuda":
                torch.cuda.reset_peak_memory_stats()

            val_due = cfg.val_every > 0 and update % cfg.val_every == 0 and update < run_stop
            eval_due = cfg.eval_every > 0 and update % cfg.eval_every == 0 and update < run_stop
            ckpt_due = cfg.ckpt_every > 0 and update % cfg.ckpt_every == 0 and update < run_stop
            boundary_due = val_due or eval_due or ckpt_due
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
                    "data/loss_positions": actual_positions,
                    "data/valid_prefixes": window_valid_prefixes / window_updates,
                    "loader/wait_s": loader_wait_s,
                    "progress/elapsed_s": training_elapsed_wall_s,
                    "progress/remaining_s": projected_training_remaining_s,
                    "schedule/muon_lr": result.muon_lr,
                    "schedule/adam_lr": result.adam_lr,
                    **window_metric_values,
                    **window_diagnostics,
                    **_mean_phase_metrics(window_phase_timers),
                }
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
                wandb.log({"global_step": update, **log})
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
            if val_due or eval_due or ckpt_due:
                checkpoint_path = save_boundary_checkpoint(
                    run_dir,
                    update=update,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    cfg=cfg,
                    uploader=uploader,
                    milestone=ckpt_due,
                    wandb_id=None if wandb.run is None else wandb.run.id,
                    actual_loss_positions=actual_positions,
                    loader_state=train_loader.state_dict(),
                )
            boundary_metrics: dict[str, float] = {}
            if val_due:
                values = _validation_wandb_metrics(val_metrics(model, val_cache, cfg), cfg)
                boundary_metrics.update({f"val/{name}": value for name, value in values.items()})
            if eval_due:
                assert checkpoint_path is not None
                if eval_inference is None:
                    eval_inference = BF16Inference(model, cfg)
                values = eval_vs_cpu(
                    model,
                    stats,
                    cfg,
                    n_matchups=cfg.eval_n_matchups,
                    replay_dir=replay_dir / f"step_{update:07d}",
                    checkpoint_sha256=_checkpoint_sha256(checkpoint_path),
                    inference=eval_inference,
                )
                expected = cfg.eval_n_matchups
                if any(
                    int(values.get(name, 0.0)) != expected for name in ("scheduled_boots", "completed_boots", "boots")
                ):
                    print(f"[eval] warning: update {update} evaluation was incomplete: {values}", flush=True)
                boundary_metrics.update({f"eval/{name}": value for name, value in _eval_wandb_metrics(values).items()})
            if boundary_metrics:
                wandb.log({"global_step": update, **boundary_metrics})
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
            eval_inference=eval_inference,
            loader_wait_fractions=loader_wait_fractions,
            loader_state=train_loader.state_dict(),
            update=run_stop,
            actual_loss_positions=actual_positions,
            smoke=smoke,
            smoke_eval_matchups=smoke_eval_matchups,
        )
    finally:
        batch_prefetcher.close()
        host_metrics.close()
        if uploader is not None:
            uploader.upload_tree(replay_dir, base=run_dir)
            uploader.close()
        wandb.finish()


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


def config_from_state(values: dict) -> TrainConfig:
    """Restore a checkpoint written by the current experiment definition."""
    expected = {"experiment_id", "architecture", "awr_calibration", *_RUNTIME_CONFIG_FIELDS}
    missing = expected - values.keys()
    unexpected = values.keys() - expected
    if missing or unexpected:
        raise ValueError(f"checkpoint config mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    if values["experiment_id"] != _EXPERIMENT_ID:
        raise ValueError(f"checkpoint experiment_id {values['experiment_id']!r} != {_EXPERIMENT_ID!r}")
    architecture_values = values["architecture"]
    calibration_values = values["awr_calibration"]
    if set(architecture_values) != _ARCHITECTURE_FIELDS:
        raise ValueError("checkpoint architecture does not match the current architecture fields")
    if set(calibration_values) != _AWR_FIELDS:
        raise ValueError("checkpoint calibration does not match the current calibration fields")
    architecture = Architecture(**architecture_values)
    calibration = AWRCalibration(**calibration_values)
    runtime = {name: values[name] for name in _RUNTIME_CONFIG_FIELDS}
    return TrainConfig(arch=architecture, awr=calibration, **runtime)


def load_checkpoint(path: str, *, device: str = DEVICE) -> tuple[GPT, TrainConfig, dict[str, FeatureStats], dict]:
    state = torch.load(path, map_location=device, weights_only=False)
    cfg = config_from_state(state["cfg"])
    validate_config(cfg)
    model = GPT(cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    stats = load_stats(cfg)
    return model, cfg, stats, state


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


def eval_checkpoint(
    path: str,
    *,
    exec_horizon: int | None = None,
    n_matchups: int | None = None,
    eager: bool = False,
    max_parallel: int | None = None,
    output_name: str | None = None,
    upload_run: str | None = None,
    backfill_wandb: bool = False,
) -> dict[str, float]:
    model, cfg, stats, state = load_checkpoint(path)
    cfg = replace(
        cfg,
        inference_mode="eager" if eager else cfg.inference_mode,
        eval_max_parallel=cfg.eval_max_parallel if max_parallel is None else max_parallel,
    )
    validate_config(cfg)
    horizon = cfg.exec_horizon if exec_horizon is None else exec_horizon
    update = int(state["step"]) + 1
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
        exec_horizon=horizon,
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
class TrainArgs:
    cfg: TrainConfig = dataclass_field(default_factory=TrainConfig)
    comment: str = ""
    resume: str | None = None
    resume_checkpoint: str = "latest.pt"
    resume_as: str | None = None
    smoke: bool = False
    stop_after_update: int | None = None
    smoke_eval_matchups: int = 4
    eval_max_parallel: int | None = None


@dataclass
class EvalArgs:
    checkpoint: str
    run: str | None = None
    exec_horizon: int | None = None
    n_matchups: int | None = None
    eager: bool = False
    max_parallel: int | None = None
    output_name: str | None = None
    backfill_wandb: bool = False


@dataclass
class SelfPlayArgs:
    checkpoint: str
    matches: int = 12
    frames: int = 14_400
    eager: bool = False
    instant_match_restart: bool = False
    process_cohorts: int = 1
    cohort_sweep: bool = False


@dataclass
class BenchmarkTrainStepArgs:
    warmup_steps: int = 3
    measured_steps: int = 20
    profile_steps: int = 0
    trace_path: str | None = None


type Command = (
    Annotated[TrainArgs, tyro.conf.subcommand(name="train")]
    | Annotated[EvalArgs, tyro.conf.subcommand(name="eval")]
    | Annotated[SelfPlayArgs, tyro.conf.subcommand(name="self-play")]
    | Annotated[BenchmarkTrainStepArgs, tyro.conf.subcommand(name="benchmark-train-step")]
)


def main(args: Command) -> None:
    if isinstance(args, BenchmarkTrainStepArgs):
        benchmark_train_step(
            warmup_steps=args.warmup_steps,
            measured_steps=args.measured_steps,
            profile_steps=args.profile_steps,
            trace_path=args.trace_path,
        )
        return
    if isinstance(args, EvalArgs):
        checkpoint = _resolve_eval_checkpoint(args.checkpoint, args.run)
        eval_checkpoint(
            str(checkpoint),
            exec_horizon=args.exec_horizon,
            n_matchups=args.n_matchups,
            eager=args.eager,
            max_parallel=args.max_parallel,
            output_name=args.output_name,
            upload_run=args.run,
            backfill_wandb=args.backfill_wandb,
        )
        return
    if isinstance(args, SelfPlayArgs):
        cohorts = (1, 2, 3, 4) if args.cohort_sweep else (args.process_cohorts,)
        for cohort_count in cohorts:
            benchmark_self_play(
                args.checkpoint,
                load_checkpoint=load_checkpoint,
                make_inference=BF16Inference,
                make_policy=make_policy,
                n_matches=args.matches,
                max_frames=args.frames,
                eager=args.eager,
                instant_match_restart=args.instant_match_restart,
                process_cohorts=cohort_count,
            )
        return
    resume_run = resume_state = None
    cfg = args.cfg
    if args.resume is None and (args.resume_checkpoint != "latest.pt" or args.resume_as is not None):
        raise SystemExit("--resume-checkpoint and --resume-as require --resume")
    if args.resume is not None:
        checkpoint = Path(args.resume_checkpoint)
        if (
            checkpoint.is_absolute()
            or ".." in checkpoint.parts
            or checkpoint.suffix != ".pt"
            or args.resume_checkpoint in ("", ".")
        ):
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
        if args.resume_as is not None:
            if Path(args.resume_as).name != args.resume_as or args.resume_as in ("", ".", ".."):
                raise SystemExit("--resume-as must be one run-name component")
            if args.resume_as == args.resume:
                raise SystemExit("--resume-as must differ from the source run")
            destination_exists = (Path("runs") / args.resume_as).exists()
            if cfg.push_to_r2:
                destination_exists = destination_exists or _remote_run_exists(args.resume_as)
            if destination_exists:
                raise SystemExit(
                    f"resume destination {args.resume_as!r} already exists; continue it with --resume {args.resume_as}"
                )
            resume_state = {**resume_state, "wandb_id": None}
    if args.eval_max_parallel is not None:
        cfg = replace(cfg, eval_max_parallel=args.eval_max_parallel)
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
    )


if __name__ == "__main__":
    main(tyro.cli(cast(type[Command], Command)))
