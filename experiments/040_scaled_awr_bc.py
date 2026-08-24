"""Scaled light-AWR behavior cloning over the complete policy-world corpus.

Full replays are labeled before window sampling. The near-offset policy loss is
weighted by a fixed, globally calibrated ``min(exp((G - baseline) / beta),
w_max) / weight_norm``. There is no learned critic.

Run:
    uv run experiments/040_scaled_awr_bc.py
    uv run experiments/040_scaled_awr_bc.py --eval runs/<run>/final.pt
"""

from __future__ import annotations

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
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from dataclasses import fields
from dataclasses import replace
from pathlib import Path

import melee
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR

import wandb
from hal import streams
from hal.data.behavior import HITSTUN_ACTIONS
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
from hal.eval.self_play import synthetic_context
from hal.sim.rollout import covering_power_of_two
from hal.training import returns as returns_lib
from hal.training import scoring
from hal.training.checkpoints import BackgroundUploader
from hal.training.checkpoints import load_for_resume
from hal.training.checkpoints import save_checkpoint
from hal.training.closed_loop import RecedingHorizon
from hal.training.dataloader import make_loader
from hal.training.ego_stats import load_consolidated_mixture_stats
from hal.training.features import A_DIM
from hal.training.features import ACTION_CHANNELS
from hal.training.features import BASE_ACTION_PROJECTION
from hal.training.features import BASE_PLAYER_PREFIXES
from hal.training.features import CAT_FEATURES
from hal.training.features import FLOAT_FEATURES
from hal.training.features import SPATIAL_COLUMNS_LEAN
from hal.training.features import SPATIAL_MASKS
from hal.training.features import V6_PLAYER_COLUMNS
from hal.training.features import Context
from hal.training.features import TrainBatch
from hal.training.features import stack_actions
from hal.training.muon import SingleDeviceMuonWithAuxAdam
from hal.training.replay_reservoir import make_reservoir_loader
from hal.training.runs import make_run_name
from hal.training.runs import profile
from hal.training.runs import setup_run_dir
from hal.training.system_metrics import HostMetricsSampler
from hal.training.trunk import Rotary
from hal.training.trunk import Trunk
from hal.training.trunk import TrunkConfig
from hal.training.trunk import apply_rotary_emb

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_LN2 = math.log(2.0)
_N_CONT = 6
_PLAYER_PREFIXES = BASE_PLAYER_PREFIXES
_EXPERIMENT_ID = "040_scaled_awr_bc_v1"
_RETURN_SUFFIX = "awr_return"
EGO_RETURN = f"ego_{_RETURN_SUFFIX}"
EGO_RETURN_VALID = f"{EGO_RETURN}_valid"
_INFERENCE_BUCKETS = (1, 2, 4, 8, 16, 32)
_INFERENCE_PARITY_MAX_ABS = 1 / 16
_INFERENCE_PARITY_RELATIVE_RMS = 2e-2
_PRODUCTION_LOSS_POSITIONS = 2**35
_PRODUCTION_EVAL_MATCHUPS = 96
_AWR_CONSTANTS_CALIBRATED = True
_PRODUCTION_TREATMENT_FIELDS = frozenset(
    {
        "L_ctx",
        "action_embed_dim",
        "action_state_embed_dim",
        "action_vocab",
        "adam_lr",
        "adam_weight_decay",
        "allow_tf32",
        "amp_dtype",
        "attn_window",
        "aux_loss_weight",
        "awr_beta",
        "awr_damage_shaping",
        "awr_gamma",
        "awr_return_baseline",
        "awr_stock_value",
        "awr_weight_max",
        "awr_weight_norm",
        "awr_win_reward",
        "batch_size",
        "char_dim",
        "char_vocab",
        "ckpt_every",
        "d_model",
        "eval_every",
        "eval_max_frames",
        "eval_max_parallel",
        "eval_n_matchups",
        "eval_seed",
        "exec_horizon",
        "final_eval_n_matchups",
        "group_head_dim",
        "head_offsets",
        "inference_mode",
        "lr_floor_ratio",
        "mds_schema_version",
        "muon_lr",
        "muon_weight_decay",
        "n_heads",
        "n_layers",
        "n_near",
        "observation_bundle",
        "offset_embed_dim",
        "policy_world_schema_version",
        "reservoir_capacity",
        "sample_chunk_length",
        "seed",
        "shuffle_block_size",
        "source_names",
        "stable_fraction",
        "stage_dim",
        "stage_vocab",
        "target_loss_positions",
        "temporal_d_model",
        "temporal_ff_dim",
        "temporal_heads",
        "temporal_layers",
        "temporal_state_film",
        "val_batch_size",
        "val_every",
        "val_n_samples",
        "val_split",
        "warmup_fraction",
        "windows_per_replay",
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

_V6_FLOATS = tuple(V6_PLAYER_COLUMNS.floats)
_V6_CATS = {name: spec for name, spec in V6_PLAYER_COLUMNS.cats.items() if spec is not None}
_CHARACTER_LIVE = "character_live"
_MISC_AS = "misc_as"


@dataclass
class TrainConfig:
    d_model: int = 1024
    n_layers: int = 16
    n_heads: int = 16
    attn_window: int = 0
    L_ctx: int = 128

    sample_chunk_length: int = 24
    head_offsets: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 20, 24)
    n_near: int = 6
    temporal_d_model: int = 256
    temporal_layers: int = 4
    temporal_heads: int = 8
    temporal_ff_dim: int = 512
    group_head_dim: int = 512
    action_embed_dim: int = 16
    offset_embed_dim: int = 16
    # Ablation: FiLM the temporal-chain states on the trunk state. The FiLM layer
    # is zero-initialized, so the arm starts at the exact baseline function.
    # False reproduces 026's decoder behavior.
    temporal_state_film: bool = False
    aux_loss_weight: float = 0.5

    action_vocab: int = 1024
    action_state_embed_dim: int = 48
    char_vocab: int = 32
    char_dim: int = 8
    stage_vocab: int = 32
    stage_dim: int = 4
    observation_bundle: str = "base"  # or v6_lean

    exec_horizon: int = 4
    inference_mode: str = "compiled"  # explicit "eager" is for debugging
    # Hardware-derived by default. An explicit power of two is a reproducibility
    # or memory-pressure override, not an architecture parameter.
    compiled_inference_bucket: int | None = None

    seed: int = 0
    eval_seed: int = 0
    batch_size: int = 1024
    target_loss_positions: int = _PRODUCTION_LOSS_POSITIONS
    muon_lr: float = 0.040
    muon_weight_decay: float = 0.0013
    adam_lr: float = 8.5e-4
    adam_weight_decay: float = 0.0625
    warmup_fraction: float = 0.03
    stable_fraction: float = 0.80
    lr_floor_ratio: float = 1 / 170
    amp_dtype: str = "bfloat16"
    allow_tf32: bool = True
    compile_trunk: bool = True
    compile_temporal: bool = True

    wandb_log_code: bool = True
    wandb_hist_every: int = 4096
    layer_rms_every: int = 4096
    layer_rms_batch_size: int = 8
    val_every: int = 8192
    val_n_samples: int = 2048
    val_batch_size: int = 128
    ckpt_every: int = 2048
    eval_every: int = 32_768
    eval_max_frames: int = 7200
    eval_n_matchups: int = _PRODUCTION_EVAL_MATCHUPS
    final_eval_n_matchups: int = _PRODUCTION_EVAL_MATCHUPS
    eval_max_parallel: int | None = 32

    source_names: tuple[str, ...] = tuple(source.name for source in streams.POLICY_WORLD_V7_SOURCES)
    mds_schema_version: int = 7
    policy_world_schema_version: int = POLICY_WORLD_SCHEMA_VERSION
    cache_limit_gb: int = 1024
    shuffle_block_size: int = 2000
    predownload: int = 1024
    windows_per_replay: int = 2
    reservoir_capacity: int = 8192
    val_split: str = "val"
    num_workers: int = 16
    prefetch_factor: int = 2
    prefetch_batches: int = 4
    push_to_r2: bool = True
    system_metrics_every: int = 25
    system_metrics_interval_s: float = 5.0
    process_metrics_interval_s: float = 30.0
    cache_metrics_interval_s: float = 30.0
    phase_timing_every: int = 10

    awr_beta: float = 199.5
    awr_weight_max: float = 3.5
    awr_gamma: float = 0.99618
    awr_stock_value: float = 120.0
    awr_damage_shaping: float = 1.0
    awr_win_reward: float = 50.0
    # Frozen by notebooks/040_awr_constants.py from the checked-in 50k-replay
    # calibration artifact (seed 0, 2026-08-23).
    awr_return_baseline: float = -0.18709200966038386
    awr_weight_norm: float = 1.0201610817403675

    @property
    def max_steps(self) -> int:
        positions_per_update = self.batch_size * self.L_ctx
        if self.target_loss_positions % positions_per_update:
            raise ValueError("target_loss_positions must be divisible by batch_size * L_ctx")
        return self.target_loss_positions // positions_per_update

    @property
    def warmup_steps(self) -> int:
        return int(self.warmup_fraction * self.max_steps)

    @property
    def stable_steps(self) -> int:
        return int(self.stable_fraction * self.max_steps)

    @property
    def awr_warmup_steps(self) -> int:
        return self.warmup_steps


def validate_config(cfg: TrainConfig) -> None:
    positive = {
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "L_ctx": cfg.L_ctx,
        "sample_chunk_length": cfg.sample_chunk_length,
        "temporal_d_model": cfg.temporal_d_model,
        "temporal_layers": cfg.temporal_layers,
        "temporal_heads": cfg.temporal_heads,
        "temporal_ff_dim": cfg.temporal_ff_dim,
        "group_head_dim": cfg.group_head_dim,
        "action_embed_dim": cfg.action_embed_dim,
        "offset_embed_dim": cfg.offset_embed_dim,
        "batch_size": cfg.batch_size,
        "target_loss_positions": cfg.target_loss_positions,
        "n_near": cfg.n_near,
        "layer_rms_batch_size": cfg.layer_rms_batch_size,
    }
    for name, value in positive.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if cfg.max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {cfg.max_steps}")
    if cfg.d_model % cfg.n_heads or cfg.temporal_d_model % cfg.temporal_heads:
        raise ValueError("model dimensions must be divisible by their head counts")
    if (cfg.temporal_d_model // cfg.temporal_heads) % 2:
        raise ValueError("temporal head dimension must be even for rotary positions")
    offsets = tuple(cfg.head_offsets)
    if offsets != tuple(sorted(set(offsets))) or not offsets or offsets[0] != 1:
        raise ValueError(f"head_offsets must be sorted, unique, and start at 1, got {offsets}")
    if offsets[-1] > cfg.sample_chunk_length:
        raise ValueError("head_offsets extend beyond sample_chunk_length")
    if offsets[: cfg.n_near] != tuple(range(1, cfg.n_near + 1)) or cfg.n_near != 6:
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
        cfg.eval_max_parallel < 1 or cfg.eval_max_parallel & (cfg.eval_max_parallel - 1)
    ):
        raise ValueError("eval_max_parallel must be a positive power of two")
    if cfg.observation_bundle not in ("base", "v6_lean"):
        raise ValueError("observation_bundle must be 'base' or 'v6_lean'")
    if not math.isfinite(cfg.aux_loss_weight) or cfg.aux_loss_weight < 0:
        raise ValueError("aux_loss_weight must be finite and non-negative")
    if not isinstance(cfg.layer_rms_every, int) or isinstance(cfg.layer_rms_every, bool) or cfg.layer_rms_every < 0:
        raise ValueError(f"layer_rms_every must be a non-negative integer, got {cfg.layer_rms_every!r}")
    for name, value in (
        ("system_metrics_every", cfg.system_metrics_every),
        ("phase_timing_every", cfg.phase_timing_every),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    for name, value in (
        ("system_metrics_interval_s", cfg.system_metrics_interval_s),
        ("process_metrics_interval_s", cfg.process_metrics_interval_s),
        ("cache_metrics_interval_s", cfg.cache_metrics_interval_s),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive, got {value!r}")
    if cfg.amp_dtype not in ("bfloat16", "float32"):
        raise ValueError("amp_dtype must be bfloat16 or float32")
    if cfg.reservoir_capacity < 2 * cfg.batch_size:
        raise ValueError("reservoir_capacity must be at least twice the batch size")
    if not 0.0 < cfg.awr_gamma < 1.0:
        raise ValueError(f"awr_gamma must be in (0, 1), got {cfg.awr_gamma}")
    if not math.isfinite(cfg.awr_beta) or cfg.awr_beta <= 0:
        raise ValueError(f"awr_beta must be finite and positive, got {cfg.awr_beta}")
    if not math.isfinite(cfg.awr_weight_max) or cfg.awr_weight_max <= 1:
        raise ValueError(f"awr_weight_max must be finite and above 1, got {cfg.awr_weight_max}")
    if not math.isfinite(cfg.awr_return_baseline):
        raise ValueError("awr_return_baseline must be finite")
    if not math.isfinite(cfg.awr_weight_norm) or cfg.awr_weight_norm <= 0:
        raise ValueError("awr_weight_norm must be finite and positive")
    if not 0.0 <= cfg.warmup_fraction < cfg.stable_fraction < 1.0:
        raise ValueError("schedule fractions must satisfy 0 <= warmup < stable < 1")
    if not 0.0 < cfg.lr_floor_ratio <= 1.0:
        raise ValueError("lr_floor_ratio must be in (0, 1]")
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
    """Require frozen scientific settings while allowing operational tuning."""
    expected = asdict(TrainConfig())
    actual = asdict(cfg)
    changed = {
        name: (actual[name], expected_value)
        for name, expected_value in expected.items()
        if name in _PRODUCTION_TREATMENT_FIELDS and actual[name] != expected_value
    }
    if changed:
        details = ", ".join(
            f"{name}={value!r} (expected {expected_value!r})"
            for name, (value, expected_value) in sorted(changed.items())
        )
        raise ValueError(f"production config differs from the frozen treatment: {details}")


def _eval_parallelism(cfg: TrainConfig, n_matchups: int) -> int:
    # ``run_matches_vec`` accepts a power-of-two capacity and then limits the
    # active worker count to ``n_matchups``. Keep that capacity a valid bucket
    # when an ad hoc evaluation asks for, for example, 12 matchups.
    return covering_power_of_two(resolve_parallelism(n_matchups, cfg.eval_max_parallel))


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
                metrics[f"throughput/phase_{metric}_s"] = self._events[start].elapsed_time(self._events[end]) / 1000
        return metrics


def decoder_rmsnorm(x: Tensor) -> Tensor:
    return F.rms_norm(x, (x.shape[-1],), eps=1e-6)


class StructuredControllerCodec(nn.Module):
    """The sole raw<->categorical boundary and shared semantic action token."""

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
            return scoring.combo_to_buttons(indices).to(self.class_embeddings[name].weight.dtype)
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
        semantic = self.semantic_values(name, indices).to(self.class_embeddings[name].weight.dtype)
        value = self.class_embeddings[name](indices) + self.semantic_projections[name](semantic)
        return decoder_rmsnorm(value)

    def embed_groups(self, indices: Tensor) -> dict[str, Tensor]:
        return {name: self.group_embedding(name, indices[..., GROUP_INDEX[name]]) for name in GROUP_NAMES}

    def embed_frame(self, indices: Tensor, embedded: dict[str, Tensor] | None = None) -> Tensor:
        values = self.embed_groups(indices) if embedded is None else embedded
        return torch.cat([values[name] for name in GROUP_NAMES], dim=-1)

    def button_mask(self, trigger_indices: Tensor) -> Tensor:
        return ~self.button_valid_for_trigger[trigger_indices]


class NonlinearActionHead(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, vocab: int) -> None:
        super().__init__()
        self.up = nn.Linear(d_model, d_hidden, bias=False)
        self.down = nn.Linear(d_hidden, vocab)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.up(decoder_rmsnorm(x))))


TEMPORAL_SDPA_BATCH_LIMIT = 32_768


class TemporalBlock(nn.Module):
    """RoPE causal SDPA block with an exact cached one-token path."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.n_heads = cfg.temporal_heads
        self.d_model = cfg.temporal_d_model
        self.head_dim = self.d_model // self.n_heads
        self.scale = 1.0 / math.sqrt(2 * cfg.temporal_layers)
        self.qkv = nn.Linear(self.d_model, 3 * self.d_model, bias=False)
        self.proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.rotary = Rotary(self.head_dim)
        self.up = nn.Linear(self.d_model, cfg.temporal_ff_dim, bias=False)
        self.down = nn.Linear(cfg.temporal_ff_dim, self.d_model, bias=False)

    def _qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, length, _ = x.shape
        q, k, v = self.qkv(decoder_rmsnorm(x)).split(self.d_model, dim=-1)
        shape = (batch, length, self.n_heads, self.head_dim)
        return q.view(shape), k.view(shape), v.view(shape)

    def _forward_chunk(self, x: Tensor) -> Tensor:
        q, k, v = self._qkv(x)
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin).transpose(1, 2)
        k = apply_rotary_emb(k, cos, sin).transpose(1, 2)
        v = v.transpose(1, 2)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attended = attended.transpose(1, 2).contiguous().view_as(x)
        x = x + self.scale * self.proj(attended)
        return x + self.down(F.silu(self.up(decoder_rmsnorm(x))))

    def forward(self, x: Tensor) -> Tensor:
        # Flash SDPA's CUDA launch rejects a flattened batch dimension above
        # 65,535.  Training flattens batch and context positions, which is
        # 1024 * 128 for the production configuration.  Chunking that
        # independent dimension preserves the exact attention computation
        # while keeping every static launch safely below the CUDA grid limit.
        if x.shape[0] <= TEMPORAL_SDPA_BATCH_LIMIT:
            return self._forward_chunk(x)
        return torch.cat([self._forward_chunk(chunk) for chunk in x.split(TEMPORAL_SDPA_BATCH_LIMIT)], dim=0)

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
        x = x + self.down(F.silu(self.up(decoder_rmsnorm(x))))
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
    """Selected-offset temporal chain conditioned by concatenation, never cross-attention."""

    def __init__(self, cfg: TrainConfig, codec: StructuredControllerCodec) -> None:
        super().__init__()
        self.codec = codec
        self.head_offsets = tuple(cfg.head_offsets)
        self.d_model = cfg.temporal_d_model
        controller_width = N_GROUPS * cfg.action_embed_dim
        self.offset_embedding = nn.Embedding(cfg.sample_chunk_length + 1, cfg.offset_embed_dim)
        self.token_projection = nn.Linear(cfg.d_model + controller_width + cfg.offset_embed_dim, self.d_model)
        self.blocks = nn.ModuleList([TemporalBlock(cfg) for _ in range(cfg.temporal_layers)])
        self.group_condition = nn.ModuleDict(
            {
                name: nn.Linear(position * cfg.action_embed_dim, 2 * self.d_model)
                for position, name in enumerate(GROUP_ORDER)
                if position
            }
        )
        self.outputs = nn.ModuleDict(
            {
                name: NonlinearActionHead(self.d_model, cfg.group_head_dim, GROUP_VOCABS[GROUP_INDEX[name]])
                for name in GROUP_NAMES
            }
        )
        self.trunk_outputs = nn.ModuleDict(
            {name: nn.Linear(cfg.d_model, GROUP_VOCABS[GROUP_INDEX[name]], bias=False) for name in GROUP_NAMES}
        )
        self.trunk_width = cfg.d_model
        self.controller_width = controller_width
        # Ablation: FiLM the chain states on the trunk state. Zero initialization
        # makes the flag-on model START at the exact baseline function; training
        # moves it away. Created LAST, so with the flag off every other module
        # draws the same initialization.
        self.state_film: nn.Linear | None = None
        if cfg.temporal_state_film:
            self.state_film = nn.Linear(cfg.d_model, 2 * self.d_model)
            nn.init.zeros_(self.state_film.weight)
            nn.init.zeros_(self.state_film.bias)

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

    def _film_params(self, trunk: Tensor) -> tuple[Tensor, Tensor] | None:
        if self.state_film is None:
            return None
        scale, shift = self.state_film(trunk).chunk(2, dim=-1)
        return torch.tanh(scale), shift

    @staticmethod
    def _apply_film(states: Tensor, film: tuple[Tensor, Tensor] | None) -> Tensor:
        if film is None:
            return states
        scale, shift = film
        if states.ndim == scale.ndim + 1:
            scale = scale.unsqueeze(-2)
            shift = shift.unsqueeze(-2)
        return states * (1.0 + scale) + shift

    def _decode_step(
        self,
        previous: Tensor,
        offset: int,
        state_bias: Tensor,
        film: tuple[Tensor, Tensor] | None,
        caches: list[tuple[Tensor, Tensor] | None],
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor] | None]]:
        """Advance the temporal chain by one selected frame offset."""
        offsets = torch.full((previous.shape[0],), offset, device=previous.device, dtype=torch.long)
        state = state_bias + self._step_features(previous, offsets)
        next_caches: list[tuple[Tensor, Tensor] | None] = []
        for block, past in zip(self.blocks, caches, strict=True):
            state, present = block.forward_step(state, past)
            next_caches.append(present)
        state = self._apply_film(decoder_rmsnorm(state), film)
        return state, next_caches

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
        x = x.reshape(hidden.shape[0] * hidden.shape[1], len(self.head_offsets), self.d_model)
        for block in self.blocks:
            x = block(x)
        states = decoder_rmsnorm(x.view(*hidden.shape[:2], len(self.head_offsets), self.d_model))
        return self._apply_film(states, self._film_params(trunk))

    def group_features(self, states: Tensor, name: str, embedded: dict[str, Tensor]) -> Tensor:
        position = GROUP_ORDER.index(name)
        if position == 0:
            return states
        prefix = torch.cat([embedded[group] for group in GROUP_ORDER[:position]], dim=-1)
        scale, shift = self.group_condition[name](prefix).chunk(2, dim=-1)
        return states * (1.0 + torch.tanh(scale)) + shift

    def teacher_forced_logits_by_group(self, hidden: Tensor, observed: Tensor, targets: Tensor) -> dict[str, Tensor]:
        states = self.teacher_forced_states(hidden, observed, targets)
        embedded = self.codec.embed_groups(targets)
        trunk = decoder_rmsnorm(hidden)
        logits = {
            name: self.outputs[name](self.group_features(states, name, embedded))
            + self.trunk_outputs[name](trunk)[:, :, None]
            for name in GROUP_NAMES
        }
        logits["buttons"] = logits["buttons"].masked_fill(self.codec.button_mask(targets[..., TRIG_G]), float("-inf"))
        return logits

    def teacher_forced_nll(self, hidden: Tensor, observed: Tensor, targets: Tensor) -> Tensor:
        logits = self.teacher_forced_logits_by_group(hidden, observed, targets)
        losses = [
            F.cross_entropy(
                logits[name].float().reshape(-1, GROUP_VOCABS[group]),
                targets[..., group].reshape(-1),
                reduction="none",
            ).view(*targets.shape[:-1])
            for group, name in enumerate(GROUP_NAMES)
        ]
        return torch.stack(losses, dim=-1)

    def teacher_forced_logits(self, hidden: Tensor, observed: Tensor, targets: Tensor) -> list[dict[str, Tensor]]:
        values = self.teacher_forced_logits_by_group(hidden, observed, targets)
        return [
            {name: logits[..., depth, :] for name, logits in values.items()} for depth in range(len(self.head_offsets))
        ]

    def forced_stepwise_logits(self, hidden: Tensor, observed: Tensor, targets: Tensor) -> list[dict[str, Tensor]]:
        if targets.shape != (hidden.shape[0], len(self.head_offsets), N_GROUPS):
            raise ValueError("stepwise targets have the wrong shape")
        trunk = decoder_rmsnorm(hidden[:, -1])
        trunk_logits = {name: self.trunk_outputs[name](trunk) for name in GROUP_NAMES}
        state_bias = self._state_bias(trunk)
        film = self._film_params(trunk)
        previous = observed
        caches: list[tuple[Tensor, Tensor] | None] = [None] * len(self.blocks)
        out: list[dict[str, Tensor]] = []
        for depth, offset in enumerate(self.head_offsets):
            state, caches = self._decode_step(previous, offset, state_bias, film, caches)
            target = targets[:, depth]
            embedded = self.codec.embed_groups(target)
            group_logits = {
                name: self.outputs[name](self.group_features(state, name, embedded)) + trunk_logits[name]
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
        if offsets not in (self.head_offsets[:4], self.head_offsets[:6]):
            raise ValueError("live decode may compute only the dense four- or six-offset prefix")
        if uniforms is not None and uniforms.shape != (len(offsets), N_GROUPS, hidden.shape[0]):
            raise ValueError("uniform table must be [frames, groups, batch]")
        trunk = decoder_rmsnorm(hidden[:, -1])
        trunk_logits = {name: self.trunk_outputs[name](trunk) for name in GROUP_NAMES}
        state_bias = self._state_bias(trunk)
        film = self._film_params(trunk)
        previous = observed
        caches: list[tuple[Tensor, Tensor] | None] = [None] * len(self.blocks)
        frames: list[Tensor] = []
        for depth, offset in enumerate(offsets):
            state, caches = self._decode_step(previous, offset, state_bias, film, caches)
            embedded: dict[str, Tensor] = {}
            picks: dict[str, Tensor] = {}
            for name in GROUP_ORDER:
                logits = self.outputs[name](self.group_features(state, name, embedded)) + trunk_logits[name]
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
        trunk_logits = {name: self.trunk_outputs[name](trunk) for name in GROUP_NAMES}
        state_bias = self._state_bias(trunk)
        film = self._film_params(trunk)
        previous = observed
        caches: list[tuple[Tensor, Tensor] | None] = [None] * len(self.blocks)
        frames: list[Tensor] = []
        all_logits: list[dict[str, Tensor]] = []
        for offset in self.head_offsets:
            state, caches = self._decode_step(previous, offset, state_bias, film, caches)
            embedded: dict[str, Tensor] = {}
            picks: dict[str, Tensor] = {}
            frame_logits: dict[str, Tensor] = {}
            for name in GROUP_ORDER:
                logits = self.outputs[name](self.group_features(state, name, embedded)) + trunk_logits[name]
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
        self.L_chunk = cfg.sample_chunk_length
        self.head_offsets = tuple(cfg.head_offsets)
        self.codec = StructuredControllerCodec(cfg.action_embed_dim)
        self.cat_specs = {**CAT_FEATURES, "action": (cfg.action_vocab, cfg.action_state_embed_dim)}
        self.cat_embeds = nn.ModuleDict(
            {name: nn.Embedding(vocab, dim) for name, (vocab, dim) in self.cat_specs.items()}
        )
        self.v6_cat_embeds = nn.ModuleDict({name: nn.Embedding(vocab, dim) for name, (vocab, dim) in _V6_CATS.items()})
        self.char_emb = nn.Embedding(cfg.char_vocab, cfg.char_dim)
        self.stage_emb = nn.Embedding(cfg.stage_vocab, cfg.stage_dim)
        if cfg.observation_bundle == "base":
            per_player = len(FLOAT_FEATURES) * 2 + sum(dim for _, dim in self.cat_specs.values())
            d_in = (
                len(_PLAYER_PREFIXES) * per_player + N_GROUPS * cfg.action_embed_dim + 2 * cfg.char_dim + cfg.stage_dim
            )
        else:
            player_floats = FLOAT_FEATURES + _V6_FLOATS
            per_player = (
                len(player_floats) * 2
                + sum(dim for _, dim in self.cat_specs.values())
                + sum(dim for _, dim in _V6_CATS.values())
                + cfg.char_dim
            )
            d_in = (
                len(_PLAYER_PREFIXES) * per_player
                + N_GROUPS * cfg.action_embed_dim
                + cfg.stage_dim
                + len(SPATIAL_COLUMNS_LEAN)
            )
        self.ctx_proj = nn.Linear(d_in, cfg.d_model)
        self.trunk = Trunk(
            TrunkConfig(
                d_model=cfg.d_model,
                n_layers=cfg.n_layers,
                n_heads=cfg.n_heads,
                L_ctx=cfg.L_ctx,
                attn_window=cfg.attn_window,
            )
        )
        self.temporal = CausalTemporalDecoder(cfg, self.codec)
        hitstun = torch.zeros(cfg.action_vocab, dtype=torch.bool)
        hitstun[torch.tensor(sorted(HITSTUN_ACTIONS), dtype=torch.long)] = True
        # Values in misc_as are simply masked outside action ranges that use it.
        # Keeping a dense checkpointed table avoids evaluator/data-version drift.
        self.register_buffer("hitstun_action", hitstun)

    def _per_player_features(self, features: dict[str, Tensor], prefix: str) -> Tensor:
        ref = features[f"{prefix}_position_x"]
        batch, length = ref.shape
        floats = FLOAT_FEATURES if self.cfg.observation_bundle == "base" else FLOAT_FEATURES + _V6_FLOATS
        values: list[Tensor] = []
        masks: list[Tensor] = []
        for name in floats:
            value = features[f"{prefix}_{name}"]
            mask = features.get(f"{prefix}_{name}_mask", torch.zeros_like(ref))
            if name == _MISC_AS:
                action = features[f"{prefix}_action"].clamp(0, self.hitstun_action.shape[0] - 1)
                outside = ~self.hitstun_action[action]
                mask = torch.maximum(mask, outside.to(mask.dtype))
                value = value * (1.0 - mask)
            values.append(value[..., None])
            masks.append(mask[..., None])
        parts: list[Tensor] = values + masks
        for name, (vocab, _) in self.cat_specs.items():
            parts.append(self.cat_embeds[name](features[f"{prefix}_{name}"].clamp(0, vocab - 1)))
        if self.cfg.observation_bundle == "v6_lean":
            for name, (vocab, _) in _V6_CATS.items():
                parts.append(self.v6_cat_embeds[name](features[f"{prefix}_{name}"].clamp(0, vocab - 1)))
            parts.append(
                self.char_emb(features[f"{prefix}_{_CHARACTER_LIVE}"].clamp(0, self.char_emb.num_embeddings - 1))
            )
        return torch.cat(parts, dim=-1)

    def context_tokens(self, features: dict[str, Tensor], action_indices: Tensor | None = None) -> Tensor:
        if action_indices is None:
            action_indices = self.codec.quantize(stack_actions(features))
        parts = [self._per_player_features(features, prefix) for prefix in _PLAYER_PREFIXES]
        parts.append(self.codec.embed_frame(action_indices))
        if self.cfg.observation_bundle == "base":
            parts.append(self.char_emb(features["ego_character"].clamp(0, self.char_emb.num_embeddings - 1)))
            parts.append(self.char_emb(features["opp_character"].clamp(0, self.char_emb.num_embeddings - 1)))
        else:
            missing = [name for name in SPATIAL_COLUMNS_LEAN if name not in features]
            if missing:
                raise ValueError(f"v6_lean observation is missing spatial columns {missing}")
            parts.append(torch.stack([features[name] for name in SPATIAL_COLUMNS_LEAN], dim=-1))
        parts.append(self.stage_emb(features["stage"].clamp(0, self.stage_emb.num_embeddings - 1)))
        return self.ctx_proj(torch.cat(parts, dim=-1))

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


def prepared_targets(model: GPT, batch: TrainBatch | AWRBatch) -> tuple[Tensor, Tensor, Tensor]:
    """Quantize history+future exactly once, then align every selected offset."""
    if batch.target.shape[1] < model.L_chunk:
        raise ValueError(f"target contains {batch.target.shape[1]} frames, expected {model.L_chunk}")
    history = stack_actions(batch.context.features)
    if history.shape[1] != model.trunk.L_ctx:
        raise ValueError(f"context length {history.shape[1]} != {model.trunk.L_ctx}")
    full = model.codec.quantize(torch.cat((history, batch.target[:, : model.L_chunk]), dim=1))
    length = history.shape[1]
    targets = torch.stack([full[:, offset : offset + length] for offset in model.head_offsets], dim=2)
    valid = torch.arange(length, device=full.device)[None, :] >= batch.context.ctx_pad[:, None]
    return full[:, :length], targets, valid


@dataclass(frozen=True, slots=True)
class AWRBatch:
    """A policy batch with return targets aligned to the next frame.

    Ineligible positions are padding or truncated returns. They keep policy
    weight one and take no value loss. Their return may be NaN, so select them
    with ``eligible`` rather than masking by multiplication.
    """

    batch: TrainBatch
    returns: Tensor  # [B, L_ctx] float32
    eligible: Tensor  # [B, L_ctx] bool

    @property
    def context(self) -> Context:
        return self.batch.context

    @property
    def target(self) -> Tensor:
        return self.batch.target

    def to(self, device: str | torch.device) -> AWRBatch:
        return AWRBatch(
            batch=self.batch.to(device),
            returns=self.returns.to(device, non_blocking=True),
            eligible=self.eligible.to(device, non_blocking=True),
        )

    def pin_memory(self) -> AWRBatch:
        return AWRBatch(
            batch=self.batch.pin_memory(),
            returns=self.returns.pin_memory(),
            eligible=self.eligible.pin_memory(),
        )

    def valid_rows(self, valid: Tensor) -> tuple[Tensor, Tensor]:
        """Select the return rows used by the policy loss."""
        return self.returns[valid], self.eligible[valid]


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


def advantage_weights(
    return_target: Tensor,
    eligible: Tensor,
    *,
    baseline: float,
    beta: float,
    weight_max: float,
    weight_norm: float,
    active: bool = True,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute globally calibrated capped AWR weights; ineligible rows stay one."""
    if return_target.requires_grad:
        raise ValueError("AWR weights must come from detached returns")
    if return_target.shape != eligible.shape:
        raise ValueError(f"return {tuple(return_target.shape)} and eligibility {tuple(eligible.shape)} must align")
    if not math.isfinite(baseline) or not math.isfinite(beta) or beta <= 0:
        raise ValueError("baseline must be finite and beta must be finite and positive")
    if not math.isfinite(weight_max) or weight_max <= 1:
        raise ValueError("weight_max must be finite and above one")
    if not math.isfinite(weight_norm) or weight_norm <= 0:
        raise ValueError("weight_norm must be finite and positive")
    weights = torch.ones_like(return_target, dtype=torch.float32)
    zero = torch.zeros((), device=return_target.device)
    stats = {
        "advantage_mean": zero,
        "advantage_std": zero,
        "weight_ess": torch.ones_like(zero),
        "weight_clip_frac": zero,
        "weight_mean": torch.ones_like(zero),
        "weight_max": torch.ones_like(zero),
        "eligible_frac": eligible.float().mean(),
    }
    n_eligible = int(eligible.sum())
    if n_eligible == 0:
        return weights, stats

    eligible_return = return_target[eligible].float()
    eligible_advantage = eligible_return - baseline
    if not torch.isfinite(eligible_advantage).all():
        raise FloatingPointError("return or advantage contains a non-finite value on an eligible row")

    max_log_weight = math.log(weight_max)
    log_weights = (eligible_advantage / beta).clamp(max=max_log_weight)
    normalized_weights = torch.exp(log_weights) / weight_norm
    active_weights = normalized_weights if active else torch.ones_like(normalized_weights)
    if not torch.isfinite(normalized_weights).all():
        raise FloatingPointError("normalized AWR weight contains a non-finite value")
    weights[eligible] = active_weights
    stats_weights = active_weights.double()
    squared_sum = stats_weights.square().sum()
    ess = stats_weights.sum().square() / (n_eligible * squared_sum) if squared_sum > 0 else zero
    stats.update(
        advantage_mean=eligible_advantage.mean(),
        advantage_std=eligible_advantage.std(correction=0),
        weight_ess=ess.float(),
        weight_clip_frac=(log_weights >= max_log_weight).float().mean(),
        weight_mean=active_weights.mean(),
        weight_max=active_weights.max(),
    )
    return weights, stats


def temporal_objective_parts(
    nll: Tensor,
    weight: Tensor,
    *,
    valid_prefixes: int,
    aux_loss_weight: float,
    n_near: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return weighted near loss, unweighted far loss, and normalized total."""
    if nll.ndim != 3 or nll.shape[2] != N_GROUPS:
        raise ValueError(f"per-prefix NLL must be [n_valid, n_offsets, {N_GROUPS}], got {tuple(nll.shape)}")
    n_offsets = nll.shape[1]
    if not 0 < n_near < n_offsets:
        raise ValueError(f"n_near must split {n_offsets} offsets, got {n_near}")
    if valid_prefixes <= 0:
        raise ValueError("valid_prefixes must be positive")
    if weight.shape != (nll.shape[0],):
        raise ValueError("one weight is required per valid prefix")
    if weight.requires_grad:
        raise ValueError("objective weights must be detached")
    joint_nll = nll.float().sum(dim=-1)
    weights = weight.float()[:, None]
    near = (joint_nll[:, :n_near] * weights).sum() / (valid_prefixes * n_near)
    far = joint_nll[:, n_near:].sum() / (valid_prefixes * (n_offsets - n_near))
    total = (near + aux_loss_weight * far) / (1.0 + aux_loss_weight)
    return near, far, total


def advantage_weighted_objective(
    nll: Tensor,
    weight: Tensor,
    *,
    valid_prefixes: int,
    aux_loss_weight: float,
    n_near: int,
) -> Tensor:
    return temporal_objective_parts(
        nll,
        weight,
        valid_prefixes=valid_prefixes,
        aux_loss_weight=aux_loss_weight,
        n_near=n_near,
    )[2]


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
    """Compute the globally normalized near-weighted policy loss.

    Weighting stays outside the compiled policy functions, so crossing the
    warmup boundary does not trigger recompilation. Logged NLLs are unweighted.
    """
    if not isinstance(batch, AWRBatch):
        raise TypeError(f"advantage training needs an AWRBatch, got {type(batch).__name__}")
    with torch.profiler.record_function("train/target_prep"):
        history, targets, valid = prepared_targets(model, batch)
    if phase_timer is not None:
        phase_timer.record("target_prep_end")
    with amp_context(cfg, DEVICE):
        with torch.profiler.record_function("train/trunk"):
            hidden = trunk_fn(batch.context.features, batch.context.ctx_pad, history)
        if phase_timer is not None:
            phase_timer.record("trunk_end")
        with torch.profiler.record_function("train/temporal"):
            dense_nll = temporal_fn(hidden, history, targets)
        if phase_timer is not None:
            phase_timer.record("temporal_end")
    with torch.profiler.record_function("train/objective"):
        nll = dense_nll[valid]
        if nll.shape[0] != valid_prefixes:
            raise RuntimeError(f"GPU valid-prefix count {nll.shape[0]} != step normalizer {valid_prefixes}")
        returns, eligible = batch.valid_rows(valid)
        active = step >= cfg.awr_warmup_steps
        weights, stats = advantage_weights(
            returns.detach(),
            eligible,
            baseline=cfg.awr_return_baseline,
            beta=cfg.awr_beta,
            weight_max=cfg.awr_weight_max,
            weight_norm=cfg.awr_weight_norm,
            active=active,
        )
        near, far, loss = temporal_objective_parts(
            nll,
            weights,
            valid_prefixes=valid_prefixes,
            aux_loss_weight=cfg.aux_loss_weight,
            n_near=cfg.n_near,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"step {step}: non-finite advantage-weighted loss {loss}")
        extra = {
            "train/loss": loss.detach() / _LN2,
            "train/temporal_loss_near": near.detach() / _LN2,
            "train/temporal_loss_far": far.detach() / _LN2,
            "train/temporal_loss_total": loss.detach() / _LN2,
            "awr/active": torch.tensor(float(active)),
            **{f"train/{name}": value.detach() for name, value in stats.items()},
        }
    if phase_timer is not None:
        phase_timer.record("objective_end")
    return loss, nll.detach(), extra


def nll_mean_metrics(
    mean_nll: Tensor, offsets: tuple[int, ...], *, n_near: int = 6, aux_loss_weight: float = 0.5
) -> dict[str, float]:
    if mean_nll.shape != (len(offsets), N_GROUPS):
        raise ValueError(f"mean NLL has shape {tuple(mean_nll.shape)}")
    joint = mean_nll.sum(dim=-1) / _LN2
    if not 0 < n_near < len(offsets):
        raise ValueError(f"n_near must split {len(offsets)} offsets, got {n_near}")
    near = joint[:n_near].mean()
    far = joint[n_near:].mean()
    total = (near + aux_loss_weight * far) / (1.0 + aux_loss_weight)
    out = {
        "loss_unweighted": float(total),
        "temporal_loss_near_unweighted": float(near),
        "temporal_loss_far_unweighted": float(far),
        "temporal_loss_total_unweighted": float(total),
        "primary_nll": float(near),
        "auxiliary_nll": float(far),
    }
    for depth, offset in enumerate(offsets):
        out[f"nll_o{offset:02d}"] = float(joint[depth])
        for group, name in enumerate(GROUP_NAMES):
            out[f"nll_o{offset:02d}_{name}"] = float(mean_nll[depth, group] / _LN2)
    return out


def nll_metrics(
    nll: Tensor, offsets: tuple[int, ...], *, n_near: int = 6, aux_loss_weight: float = 0.5
) -> dict[str, float]:
    return nll_mean_metrics(nll.mean(dim=0), offsets, n_near=n_near, aux_loss_weight=aux_loss_weight)


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
                dense_nll = model.temporal.teacher_forced_nll(hidden, history, targets)
            row_valid = batch.context.ctx_pad < cfg.L_ctx
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
        n_near=cfg.n_near,
        aux_loss_weight=cfg.aux_loss_weight,
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
        if horizon not in (4, 6):
            raise ValueError("only the unrolled four- and six-frame decoders exist")
        rows = ctx.ctx_pad.shape[0]
        bucket = self._bucket(rows)
        padded = canonical_context(_pad_context(ctx, bucket), self.cfg.observation_bundle)
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


@torch.no_grad()
def decode_chunk(
    model: GPT,
    ctx: Context,
    n_frames: int,
    *,
    argmax: bool = False,
    gen: torch.Generator | None = None,
) -> Tensor:
    cfg = model.cfg
    return BF16Inference(model, replace(cfg, inference_mode="eager"), compiled=False).decode(
        ctx, n_frames, argmax=argmax, gen=gen
    )


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
    if horizon not in (4, 6):
        raise ValueError("execution horizon must be four or six")
    engine = BF16Inference(model, cfg) if inference is None else inference
    random_streams = None if decode_seed is None else SlotGroupRandom(decode_seed)
    generator = None if decode_seed is None else torch.Generator(device=device).manual_seed(decode_seed)

    @torch.no_grad()
    def predict(ctx: Context, committed: np.ndarray | None) -> np.ndarray:
        if committed is not None:
            raise ValueError("experiment 026 does not condition on a committed RTC prefix")
        started = time.perf_counter()
        result = engine.decode(ctx, horizon, streams=random_streams, gen=generator).cpu().numpy()
        if telemetry is not None:
            telemetry.record(rows=ctx.ctx_pad.shape[0], horizon=horizon, seconds=time.perf_counter() - started)
        return result

    v6 = cfg.observation_bundle == "v6_lean"
    return RecedingHorizon(
        predict_chunk=predict,
        stats=stats,
        L_ctx=cfg.L_ctx,
        L_chunk=horizon,
        s=horizon,
        d=0,
        device=device,
        float_dtype=next(model.parameters()).dtype,
        extra=V6_PLAYER_COLUMNS if v6 else None,
        projection=None if v6 else BASE_ACTION_PROJECTION,
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
    ego_port: int
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
        if step <= cfg.stable_steps:
            return 1.0
        progress = (step - cfg.stable_steps) / max(cfg.max_steps - 1 - cfg.stable_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return cfg.lr_floor_ratio + (1.0 - cfg.lr_floor_ratio) * cosine

    return schedule


def make_optimizer(model: GPT, cfg: TrainConfig) -> SingleDeviceMuonWithAuxAdam:
    muon = [parameter for parameter in model.trunk.blocks.parameters() if parameter.ndim >= 2]
    muon_ids = {id(parameter) for parameter in muon}
    embedding_modules = (
        model.cat_embeds,
        model.v6_cat_embeds,
        model.char_emb,
        model.stage_emb,
        model.codec.class_embeddings,
        model.temporal.offset_embedding,
    )
    embedding_ids = {id(parameter) for module in embedding_modules for parameter in module.parameters()}
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for parameter in model.parameters():
        if id(parameter) in muon_ids:
            continue
        (no_decay if parameter.ndim < 2 or id(parameter) in embedding_ids else decay).append(parameter)
    if len(muon) + len(decay) + len(no_decay) != sum(1 for _ in model.parameters()):
        raise RuntimeError("optimizer parameter partition is incomplete")
    adam = dict(betas=(0.9, 0.95), eps=1e-10, use_muon=False)
    return SingleDeviceMuonWithAuxAdam(
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


def subsystem_parameter_counts(model: GPT) -> dict[str, int]:
    all_parameters = tuple(model.parameters())
    trunk_ids = {id(parameter) for parameter in model.trunk.parameters()}
    head_modules = nn.ModuleList([model.temporal.outputs, model.temporal.trunk_outputs])
    head_ids = {id(parameter) for parameter in head_modules.parameters()}
    temporal_ids = {id(parameter) for parameter in model.temporal.parameters() if id(parameter) not in head_ids}
    other_ids = {id(parameter) for parameter in all_parameters} - trunk_ids - temporal_ids - head_ids
    partitions = {
        "trunk": trunk_ids,
        "temporal_decoder": temporal_ids,
        "group_heads": head_ids,
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


def model_tag(cfg: TrainConfig) -> str:
    offsets = "-".join(map(str, cfg.head_offsets))
    treatment = f"awr-near-b{cfg.awr_beta:g}-g{cfg.awr_gamma:g}-wu{cfg.awr_warmup_steps}"
    if cfg.temporal_state_film:
        treatment = f"{treatment}-film"
    return (
        f"mtp040-d{cfg.d_model}-L{cfg.n_layers}-h{cfg.n_heads}-Lc{cfg.L_ctx}-"
        f"t{cfg.temporal_d_model}x{cfg.temporal_layers}-o{offsets}-s{cfg.exec_horizon}-{cfg.observation_bundle}-"
        f"{treatment}"
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
        return "decoder/heads" if parts[1] in ("outputs", "trunk_outputs") else "decoder/other"
    return "trunk/input"


def _wandb_tensor_log(model: nn.Module, *, gradients: bool, sample_limit: int = 65_536) -> dict[str, object]:
    buckets: dict[str, list[Tensor]] = {}
    for name, parameter in model.named_parameters():
        value = parameter.grad if gradients else parameter
        if value is None:
            continue
        if value.is_sparse:
            value = value.coalesce().values()
        buckets.setdefault(_wandb_parameter_group(name), []).append(value.detach())

    payload: dict[str, object] = {}
    histogram_root = "gradients" if gradients else "weights"
    norm_root = "gradient_norm" if gradients else "weight_norm"
    for group, values in buckets.items():
        count = sum(value.numel() for value in values)
        stride = max(1, math.ceil(count / sample_limit))
        samples = torch.cat([value.reshape(-1)[::stride] for value in values])[:sample_limit]
        squared_norm = torch.stack([value.float().square().sum() for value in values]).sum()
        payload[f"{histogram_root}/{group}"] = wandb.Histogram(samples.float().cpu().numpy())
        payload[f"{norm_root}/{group}"] = float(squared_norm.sqrt())
    return payload


def wandb_gradient_log(model: nn.Module) -> dict[str, object]:
    return _wandb_tensor_log(model, gradients=True)


def wandb_weight_log(model: nn.Module) -> dict[str, object]:
    return _wandb_tensor_log(model, gradients=False)


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

    def capture(name: str):
        def hook(_module: nn.Module, inputs: tuple[object, ...], output: object) -> None:
            layer_input = inputs[0]
            if not isinstance(layer_input, Tensor) or not isinstance(output, Tensor):
                raise TypeError(f"{name} RMS hook expected tensor input and output")
            activation_rms = layer_input.detach().float().square().mean().sqrt()
            residual_rms = (output.detach().float() - layer_input.detach().float()).square().mean().sqrt()
            measurements[name] = (activation_rms, residual_rms)

        return hook

    handles = [layer.register_forward_hook(capture(name)) for name, layer in _residual_layers(model)]
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
    scalars = torch.stack([value for name in layer_names for value in measurements[name]]).double().cpu()
    payload: dict[str, float] = {}
    tiny = torch.finfo(torch.float64).tiny
    for index, name in enumerate(layer_names):
        activation_rms = float(scalars[2 * index])
        residual_rms = float(scalars[2 * index + 1])
        payload[f"activation_rms/{name}"] = activation_rms
        payload[f"residual_branch_rms/{name}"] = residual_rms
        payload[f"residual_ratio/{name}"] = residual_rms / max(activation_rms, tiny)
    nonfinite = {name: value for name, value in payload.items() if not math.isfinite(value)}
    if nonfinite:
        raise FloatingPointError(f"layer activation diagnostic produced non-finite metrics: {nonfinite}")
    return payload


@torch.no_grad()
def layer_gradient_rms_log(model: GPT) -> dict[str, float]:
    """Return parameter-gradient RMS for every monitored residual block."""
    names: list[str] = []
    values: list[Tensor] = []
    for name, layer in _residual_layers(model):
        gradients = [parameter.grad.detach() for parameter in layer.parameters() if parameter.grad is not None]
        if not gradients:
            raise RuntimeError(f"residual layer {name} has no parameter gradients")
        squared_sum = torch.stack([gradient.float().square().sum() for gradient in gradients]).sum()
        count = sum(gradient.numel() for gradient in gradients)
        names.append(name)
        values.append(torch.sqrt(squared_sum / count))
    cpu_values = torch.stack(values).double().cpu()
    payload = {f"gradient_rms/{name}": float(value) for name, value in zip(names, cpu_values, strict=True)}
    nonfinite = {name: value for name, value in payload.items() if not math.isfinite(value)}
    if nonfinite:
        raise FloatingPointError(f"layer gradient diagnostic produced non-finite metrics: {nonfinite}")
    return payload


def loader_kwargs(cfg: TrainConfig, stats: dict[str, FeatureStats]) -> dict:
    v6 = cfg.observation_bundle == "v6_lean"
    sources = tuple(streams.BY_NAME[name] for name in cfg.source_names)
    return dict(
        data_root=None,
        sources=sources,
        cache_limit=f"{cfg.cache_limit_gb}gb",
        shuffle_block_size=cfg.shuffle_block_size,
        shuffle_seed=cfg.seed,
        stats=stats,
        L_ctx=cfg.L_ctx,
        L_chunk=cfg.sample_chunk_length,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        schema_version=cfg.mds_schema_version,
        extra=V6_PLAYER_COLUMNS if v6 else None,
        projection=None if v6 else BASE_ACTION_PROJECTION,
    )


def validate_batch_geometry(
    batch: TrainBatch | AWRBatch, cfg: TrainConfig, expected_batch_size: int | None = None
) -> None:
    if batch.target.shape[1:] != (cfg.sample_chunk_length, A_DIM):
        raise ValueError(f"target must be [B, {cfg.sample_chunk_length}, {A_DIM}], got {tuple(batch.target.shape)}")
    batch_size = batch.target.shape[0]
    if expected_batch_size is not None and batch_size != expected_batch_size:
        raise ValueError(f"fixed training batch must contain {expected_batch_size} rows, got {batch_size}")
    if batch.context.ctx_pad.shape != (batch_size,):
        raise ValueError("ctx_pad shape does not match the batch")
    wrong = {
        name: tuple(value.shape)
        for name, value in batch.context.features.items()
        if value.shape[:2] != (batch_size, cfg.L_ctx)
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
        extra_state={"actual_loss_positions": actual_loss_positions},
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
        [streams.POLICY_WORLD_V7_TRAIN_REPLAYS[source.name] for source in sources],
        expected_mds_schema_version=cfg.mds_schema_version,
    )


def _make_loaders(cfg: TrainConfig, stats: dict[str, FeatureStats]):
    common = loader_kwargs(cfg, stats)
    projection = common["projection"]
    if projection is not None:
        common["projection"] = replace(projection, columns=projection.columns | {EGO_RETURN, EGO_RETURN_VALID})
    label_replay = functools.partial(
        returns_lib.label_replay,
        gamma=cfg.awr_gamma,
        damage_shaping=cfg.awr_damage_shaping,
        win_reward=cfg.awr_win_reward,
        stock_value=cfg.awr_stock_value,
        suffix=_RETURN_SUFFIX,
    )
    train_loader = make_reservoir_loader(
        split="train",
        num_workers=cfg.num_workers,
        prefetch_factor=cfg.prefetch_factor,
        predownload=cfg.predownload,
        windows_per_replay=cfg.windows_per_replay,
        reservoir_capacity=cfg.reservoir_capacity,
        prefetch_batches=cfg.prefetch_batches,
        replay_format="policy-world",
        replay_transform=label_replay,
        batch_transform=functools.partial(collate_awr_batch, L_ctx=cfg.L_ctx),
        **common,
    )
    validation = {**common, "batch_size": cfg.val_batch_size, "shuffle": True}
    val_loader = make_loader(
        split=cfg.val_split,
        num_workers=0,
        replay_format="policy-world",
        **validation,
    )
    return train_loader, cache_validation(val_loader, cfg.val_n_samples)


def _export_training_profile(profiler: torch.profiler.profile, profile_dir: Path) -> None:
    """Write one Chrome trace, CUDA memory view, and operator summary."""
    trace_path = profile_dir / "trace.json.gz"
    memory_path = profile_dir / "memory-timeline.html"
    operators_path = profile_dir / "operators.txt"
    profiler.export_chrome_trace(str(trace_path))
    profiler.export_memory_timeline(str(memory_path), device="cuda:0")
    operators_path.write_text(
        profiler.key_averages(group_by_input_shape=True).table(
            sort_by="self_cuda_time_total",
            row_limit=200,
        )
    )
    print(
        f"[profile] wrote {trace_path}, {memory_path}, and {operators_path}",
        flush=True,
    )


def _training_profiler(profile_dir: Path, *, enabled: bool) -> torch.profiler.profile | None:
    """Create a bounded post-compile CUDA profiler when requested."""
    if not enabled:
        return None
    if DEVICE != "cuda":
        raise ValueError("training profiling requires CUDA")
    profile_dir.mkdir(parents=True, exist_ok=True)
    return torch.profiler.profile(
        activities=(
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ),
        schedule=torch.profiler.schedule(wait=10, warmup=3, active=5, repeat=1),
        on_trace_ready=functools.partial(_export_training_profile, profile_dir=profile_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        post_processing_timeout_s=600,
    )


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
    profile_steps: int = 0,
) -> None:
    validate_config(cfg)
    if not smoke:
        validate_production_config(cfg)
        if stop_after_update is not None:
            raise ValueError("stop_after_update is a smoke-only control")
        if not _AWR_CONSTANTS_CALIBRATED:
            raise ValueError("run notebooks/040_awr_constants.py and freeze its constants before production")
    if stop_after_update is not None and not 1 <= stop_after_update <= cfg.max_steps:
        raise ValueError(f"stop_after_update must be in [1, {cfg.max_steps}], got {stop_after_update}")
    if not isinstance(profile_steps, int) or isinstance(profile_steps, bool) or profile_steps < 0:
        raise ValueError(f"profile_steps must be a non-negative integer, got {profile_steps!r}")
    if profile_steps and profile_steps < 20:
        raise ValueError("profile_steps must be at least 20 to exclude compilation from the active trace")
    if profile_steps and (smoke or stop_after_update is not None or resume_state is not None):
        raise ValueError("profile_steps requires a fresh production-config run")
    if smoke_eval_matchups < 0:
        raise ValueError("smoke_eval_matchups must be non-negative")
    run_stop = profile_steps or (cfg.max_steps if stop_after_update is None else stop_after_update)
    run_name = resume_run or make_run_name(Path(__file__).stem, model_tag(cfg), "policy-world-v7-35", comment)
    uploader = BackgroundUploader(run_name) if cfg.push_to_r2 else None
    wandb.init(
        project="hal",
        name=run_name,
        id=None if resume_state is None else resume_state.get("wandb_id"),
        resume="allow" if resume_state is not None else None,
        tags=["gpt", "temporal-mtp", "advantage-weighted-bc", "scaled", "040"],
        config=asdict(cfg),
        settings=wandb.Settings(
            x_stats_sampling_interval=1.0 if profile_steps else 5.0,
            x_stats_track_process_tree=True,
        ),
    )
    if wandb.run is not None:
        wandb.define_metric("eval/net_stock_lcb", step_metric="global_step")
        wandb.define_metric("eval/net_dmg_lcb", step_metric="global_step")
        wandb.run.summary["nll_semantics"] = "train/loss is weighted; *_unweighted and val metrics are unweighted"
        wandb.run.summary["layer_rms_semantics"] = (
            "activation=block input; residual_branch=block output-input; "
            "residual_ratio=residual_branch/activation; gradient=parameter-gradient RMS"
        )
        if cfg.wandb_log_code:
            log_wandb_code(wandb.run)
    run_dir, replay_dir = setup_run_dir(run_name)
    profile_dir = run_dir / "profile"
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high" if cfg.allow_tf32 else "highest")
    model = GPT(cfg).to(DEVICE)
    counts = subsystem_parameter_counts(model)
    if wandb.run is not None:
        for name, value in counts.items():
            wandb.run.summary[f"parameters/{name}"] = value
        unique_replays = sum(streams.POLICY_WORLD_V7_TRAIN_REPLAYS[name] for name in cfg.source_names)
        unique_frames = sum(streams.POLICY_WORLD_V7_TRAIN_FRAMES[name] for name in cfg.source_names)
        wandb.run.summary["data/unique_replays"] = unique_replays
        wandb.run.summary["data/unique_frames"] = unique_frames
        wandb.run.summary["data/processed_loss_positions"] = cfg.target_loss_positions
        wandb.run.summary["data/effective_epochs"] = cfg.target_loss_positions / unique_frames
        wandb.run.summary["data/D_over_N"] = cfg.target_loss_positions / counts["total"]
        wandb.run.summary["data/nominal_loss_positions_per_update"] = cfg.batch_size * cfg.L_ctx
        for name in cfg.source_names:
            wandb.run.summary[f"data/source_replay_share/{name}"] = (
                streams.POLICY_WORLD_V7_TRAIN_REPLAYS[name] / unique_replays
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
        actual_positions = int(resume_state.get("actual_loss_positions", start_step * cfg.batch_size * cfg.L_ctx))
        if not 0 <= actual_positions <= start_step * cfg.batch_size * cfg.L_ctx:
            raise ValueError(
                f"checkpoint actual_loss_positions={actual_positions} is invalid after {start_step} updates"
            )

    # Resolve the hardware-dependent backend before Dynamo sees the model. The shared trunk contains
    # raw mask and attention operations; this entrypoint is their one compilation owner.
    trunk_fn: Callable = model.forward
    temporal_fn: Callable = model.temporal.teacher_forced_nll
    if DEVICE == "cuda" and cfg.compile_trunk:
        model.trunk.resolve_attention(DEVICE)
        if model.trunk.attn_path != "flex":
            raise RuntimeError(
                f"compiled CUDA training requires FlexAttention, resolved {model.trunk.attn_path!r} instead"
            )
        trunk_fn = torch.compile(trunk_fn, dynamic=False, fullgraph=True)
    if DEVICE == "cuda" and cfg.compile_temporal:
        temporal_fn = torch.compile(temporal_fn, dynamic=False)

    train_loader, val_cache = _make_loaders(cfg, stats)
    source_counts = train_loader.source_sample_counts
    expected_counts = {name: streams.POLICY_WORLD_V7_TRAIN_REPLAYS[name] for name in cfg.source_names}
    if source_counts != expected_counts:
        raise RuntimeError(f"train stream counts changed: got {source_counts}, expected {expected_counts}")
    iterator = iter(train_loader)
    run_started = time.monotonic()
    loader_wait_fractions: list[float] = []
    cache_roots = tuple(streams.BY_NAME[name].local_root for name in cfg.source_names)
    host_metrics = HostMetricsSampler(
        cache_roots,
        interval_s=cfg.system_metrics_interval_s,
        process_interval_s=cfg.process_metrics_interval_s,
        cache_interval_s=cfg.cache_metrics_interval_s,
    )
    host_metrics.start()
    training_profiler = _training_profiler(profile_dir, enabled=bool(profile_steps))
    if training_profiler is not None:
        training_profiler.start()
    previous_update_started: float | None = None
    previous_wandb_log_s: float | None = None
    previous_checkpoint_s: float | None = None
    # CUDA compilation must remain on the training thread. Background compilation
    # deadlocked training on both H100 and L40S hosts.
    eval_inference: BF16Inference | None = None
    model.train()
    try:
        for step in range(start_step, run_stop):
            update_started = time.monotonic()
            previous_loop_s = None if previous_update_started is None else update_started - previous_update_started
            previous_update_started = update_started
            update = step + 1
            if DEVICE == "cuda":
                torch.cuda.reset_peak_memory_stats()
            loader_started = time.monotonic()
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            validate_batch_geometry(batch, cfg, cfg.batch_size)
            loader_wait = time.monotonic() - loader_started
            valid_prefixes = int((cfg.L_ctx - batch.context.ctx_pad).sum())
            if valid_prefixes <= 0:
                raise RuntimeError("training batch contains no valid context prefixes")

            phase_due = (
                DEVICE == "cuda"
                and cfg.phase_timing_every > 0
                and (update == 1 or update % cfg.phase_timing_every == 0)
            )
            phase_timer = CudaPhaseTimer() if phase_due else None
            optimizer.zero_grad()
            with profile("step") as stopwatch:
                if phase_timer is not None:
                    phase_timer.record("start")
                with torch.profiler.record_function("train/h2d"):
                    batch = batch.to(DEVICE)
                if phase_timer is not None:
                    phase_timer.record("h2d_end")
                loss, nll, step_metrics = microbatch_loss(
                    model,
                    batch,
                    cfg,
                    step=step,
                    valid_prefixes=valid_prefixes,
                    trunk_fn=trunk_fn,
                    temporal_fn=temporal_fn,
                    phase_timer=phase_timer,
                )
                with torch.profiler.record_function("train/backward"):
                    loss.backward()
                if phase_timer is not None:
                    phase_timer.record("backward_end")
                with torch.profiler.record_function("train/grad_norm"):
                    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError(f"update {update}: non-finite gradient norm {gradient_norm}")
                if phase_timer is not None:
                    phase_timer.record("grad_norm_end")
                histogram_payload = {}
                layer_rms_payload = {}
                with torch.profiler.record_function("train/diagnostics"):
                    if histogram_due(update, cfg.wandb_hist_every):
                        histogram_payload = {**wandb_gradient_log(model), **wandb_weight_log(model)}
                    if histogram_due(update, cfg.layer_rms_every):
                        layer_rms_payload = {
                            **layer_activation_rms_log(
                                model,
                                batch,
                                cfg,
                                max_rows=cfg.layer_rms_batch_size,
                            ),
                            **layer_gradient_rms_log(model),
                        }
                if phase_timer is not None:
                    phase_timer.record("diagnostics_end")
                muon_lr = next(group["lr"] for group in optimizer.param_groups if group["use_muon"])
                adam_lr = next(group["lr"] for group in optimizer.param_groups if not group["use_muon"])
                with torch.profiler.record_function("train/optimizer"):
                    optimizer.step()
                    scheduler.step()
                if phase_timer is not None:
                    phase_timer.record("optimizer_end")
                if DEVICE == "cuda":
                    torch.cuda.synchronize()
            phase_metrics = {} if phase_timer is None else phase_timer.metrics()
            actual_positions += valid_prefixes
            loader_wait_fractions.append(loader_wait / max(loader_wait + stopwatch.elapsed, 1e-12))
            metrics_started = time.monotonic()
            with torch.profiler.record_function("train/metrics_d2h"):
                metrics = nll_metrics(
                    nll.cpu(),
                    cfg.head_offsets,
                    n_near=cfg.n_near,
                    aux_loss_weight=cfg.aux_loss_weight,
                )
            metrics_d2h_s = time.monotonic() - metrics_started
            log_started = time.monotonic()
            log = {
                "global_step": update,
                "samples": update * cfg.batch_size,
                "data/actual_loss_positions": actual_positions,
                "data/actual_loss_positions_this_update": valid_prefixes,
                **{f"train/{name}": value for name, value in metrics.items()},
                "train/grad_norm": float(gradient_norm),
                "throughput/step_s": stopwatch.elapsed,
                "throughput/loader_wait_s": loader_wait,
                "throughput/samples_per_s": cfg.batch_size / stopwatch.elapsed,
                "throughput/samples_per_wall_s": cfg.batch_size / (stopwatch.elapsed + loader_wait),
                "throughput/prefixes_per_s": valid_prefixes / stopwatch.elapsed,
                "throughput/metrics_d2h_s": metrics_d2h_s,
                "lr/muon": muon_lr,
                "lr/adam": adam_lr,
                **{name: float(value) for name, value in step_metrics.items()},
                **histogram_payload,
                **layer_rms_payload,
                **phase_metrics,
            }
            if previous_loop_s is not None:
                log["throughput/previous_loop_s"] = previous_loop_s
                log["throughput/previous_samples_per_true_wall_s"] = cfg.batch_size / previous_loop_s
            if previous_wandb_log_s is not None:
                log["throughput/previous_wandb_log_s"] = previous_wandb_log_s
            if previous_checkpoint_s is not None:
                log["throughput/previous_checkpoint_save_s"] = previous_checkpoint_s
                previous_checkpoint_s = None
            if cfg.system_metrics_every > 0 and (update == 1 or update % cfg.system_metrics_every == 0):
                log.update(host_metrics.snapshot())
            if DEVICE == "cuda":
                log["hardware/peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 2**30
                log["hardware/peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30
            log["throughput/log_build_s"] = time.monotonic() - log_started
            log["throughput/pre_wandb_loop_s"] = time.monotonic() - update_started
            wandb_started = time.monotonic()
            with torch.profiler.record_function("train/wandb_log"):
                wandb.log(log)
            previous_wandb_log_s = time.monotonic() - wandb_started
            if update <= 10 or update % 50 == 0:
                print(
                    f"[t+{time.monotonic() - run_started:.0f}s] update {update}: "
                    f"{float(step_metrics['train/loss']):.3f} bits objective, "
                    f"{cfg.batch_size / stopwatch.elapsed:.0f} samples/s",
                    flush=True,
                )
            val_due = cfg.val_every > 0 and update % cfg.val_every == 0 and update < run_stop
            eval_due = cfg.eval_every > 0 and update % cfg.eval_every == 0 and update < run_stop
            ckpt_due = cfg.ckpt_every > 0 and update % cfg.ckpt_every == 0 and update < run_stop
            checkpoint_path: Path | None = None
            if val_due or eval_due or ckpt_due:
                checkpoint_started = time.monotonic()
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
                )
                previous_checkpoint_s = time.monotonic() - checkpoint_started
            if val_due:
                values = val_metrics(model, val_cache, cfg)
                wandb.log({"global_step": update, **{f"val/{name}": value for name, value in values.items()}})
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
                wandb.log({"global_step": update, **{f"eval/{name}": value for name, value in values.items()}})
            if training_profiler is not None:
                training_profiler.step()

        if profile_steps:
            print(
                f"[profile] completed {profile_steps} diagnostic updates; skipping checkpoint and evaluation",
                flush=True,
            )
            return

        final_snapshot = save_boundary_checkpoint(
            run_dir,
            update=run_stop,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            uploader=uploader,
            milestone=cfg.ckpt_every > 0 and run_stop % cfg.ckpt_every == 0,
            wandb_id=None if wandb.run is None else wandb.run.id,
            actual_loss_positions=actual_positions,
        )
        final_path = run_dir / ("smoke-final.pt" if smoke else "final.pt")
        _replace_link(final_snapshot, final_path)
        if uploader is not None:
            uploader.upload(final_snapshot, key=final_path.name)
        checkpoint_sha = _checkpoint_sha256(final_path)
        final_val = val_metrics(model, val_cache, cfg)
        wandb.log({"global_step": run_stop, **{f"val/{name}": value for name, value in final_val.items()}})
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
            wandb.log({"global_step": run_stop, **{f"eval/{name}": value for name, value in final_eval.items()}})
        mean_wait = float(np.mean(loader_wait_fractions)) if loader_wait_fractions else 0.0
        p95_wait = float(np.percentile(loader_wait_fractions, 95)) if loader_wait_fractions else 0.0
        print(f"[loader] mean wait={100 * mean_wait:.2f}%, p95={100 * p95_wait:.2f}%", flush=True)
        if smoke and (mean_wait > 0.05 or p95_wait > 0.10):
            raise RuntimeError("smoke loader gate failed: require mean wait <=5% and p95 <=10%")
    finally:
        try:
            if training_profiler is not None:
                training_profiler.stop()
        finally:
            host_metrics.close()
            if uploader is not None:
                uploader.upload_tree(replay_dir, base=run_dir)
                uploader.upload_tree(profile_dir, base=run_dir)
                uploader.close()
            wandb.finish()


_CHECKPOINT_ARCH_FIELDS = {
    "head_offsets",
    "n_near",
    "sample_chunk_length",
    "temporal_d_model",
    "temporal_layers",
    "temporal_heads",
    "temporal_ff_dim",
    "group_head_dim",
    "action_embed_dim",
    "offset_embed_dim",
    "observation_bundle",
    "target_loss_positions",
    "source_names",
}

_AWR_CHECKPOINT_FIELDS = {
    "experiment_id",
    "awr_beta",
    "awr_weight_max",
    "awr_gamma",
    "awr_stock_value",
    "awr_damage_shaping",
    "awr_win_reward",
    "awr_return_baseline",
    "awr_weight_norm",
    "temporal_state_film",
}


def _checkpoint_config(cfg: TrainConfig) -> dict[str, object]:
    return {"experiment_id": _EXPERIMENT_ID, **asdict(cfg)}


def config_from_state(values: dict) -> TrainConfig:
    """Restore only an explicitly identified experiment-040 checkpoint."""
    missing = (_CHECKPOINT_ARCH_FIELDS | _AWR_CHECKPOINT_FIELDS) - values.keys()
    if missing:
        raise ValueError(f"checkpoint is not experiment 040; missing {sorted(missing)}")
    if values["experiment_id"] != _EXPERIMENT_ID:
        raise ValueError(f"checkpoint experiment_id {values['experiment_id']!r} != required {_EXPERIMENT_ID!r}")
    known = {item.name for item in fields(TrainConfig)}
    return TrainConfig(**{name: value for name, value in values.items() if name in known})


def load_checkpoint(path: str, *, device: str = DEVICE) -> tuple[GPT, TrainConfig, dict[str, FeatureStats], dict]:
    state = torch.load(path, map_location=device, weights_only=False)
    cfg = config_from_state(state["cfg"])
    validate_config(cfg)
    model = GPT(cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    stats = load_stats(cfg)
    return model, cfg, stats, state


def eval_checkpoint(
    path: str,
    *,
    exec_horizon: int | None = None,
    n_matchups: int | None = None,
    eager: bool = False,
    max_parallel: int | None = None,
    output_name: str | None = None,
) -> dict[str, float]:
    model, cfg, stats, state = load_checkpoint(path)
    cfg = replace(
        cfg,
        inference_mode="eager" if eager else cfg.inference_mode,
        eval_max_parallel=cfg.eval_max_parallel if max_parallel is None else max_parallel,
    )
    validate_config(cfg)
    horizon = cfg.exec_horizon if exec_horizon is None else exec_horizon
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
    print(f"[eval] step={state['step']} horizon={horizon}: {values}", flush=True)
    return values


def _inference_parity_metrics(actual: Tensor, reference: Tensor) -> dict[str, float]:
    """Check compiled BF16 output without over-weighting relative error near zero."""
    error = actual.float() - reference.float()
    values = (
        torch.stack(
            (
                error.abs().max(),
                error.square().mean().sqrt(),
                reference.float().square().mean().sqrt(),
            )
        )
        .double()
        .cpu()
    )
    max_abs, error_rms, reference_rms = map(float, values)
    relative_rms = error_rms / max(reference_rms, torch.finfo(torch.float64).tiny)
    metrics = {
        "max_abs_error": max_abs,
        "rms_error": error_rms,
        "relative_rms_error": relative_rms,
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise AssertionError(f"compiled inference parity produced non-finite metrics: {metrics}")
    if max_abs > _INFERENCE_PARITY_MAX_ABS or relative_rms > _INFERENCE_PARITY_RELATIVE_RMS:
        raise AssertionError(
            "compiled inference parity exceeded BF16 tolerances: "
            f"{metrics}, max_abs <= {_INFERENCE_PARITY_MAX_ABS}, "
            f"relative_rms <= {_INFERENCE_PARITY_RELATIVE_RMS}"
        )
    return metrics


@torch.no_grad()
def run_benchmark(cfg: TrainConfig, *, iterations: int = 20) -> dict[str, float]:
    validate_config(cfg)
    device = torch.device(DEVICE)
    model = GPT(cfg).to(device).eval()
    ctx = synthetic_context(cfg, min(32, cfg.batch_size), device)
    eager = BF16Inference(model, replace(cfg, inference_mode="eager"), compiled=False)
    compiled = BF16Inference(model, cfg)

    def selected_context(rows: int) -> Context:
        return Context(
            features={name: value[:rows] for name, value in ctx.features.items()},
            ctx_pad=ctx.ctx_pad[:rows],
            slot_ids=ctx.slot_ids[:rows] if ctx.slot_ids is not None else None,
            reset=ctx.reset[:rows] if ctx.reset is not None else None,
        )

    def trunk_parity(rows: int) -> dict[str, float]:
        selected = selected_context(rows)
        bucket = compiled._bucket(rows)
        padded = canonical_context(_pad_context(selected, bucket), cfg.observation_bundle)
        observed = model.codec.quantize(stack_actions(padded.features))
        with amp_context(cfg, device):
            reference = model.forward_dense(padded.features, padded.ctx_pad, observed)
            actual = compiled._trunk(bucket)(padded.features, padded.ctx_pad, observed)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return _inference_parity_metrics(actual, reference)

    def measure(engine: BF16Inference, rows: int, horizon: int) -> float:
        selected = selected_context(rows)
        for _ in range(2):
            engine.decode(selected, horizon)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(iterations):
            engine.decode(selected, horizon)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - started) / iterations

    out: dict[str, float] = {}
    rows_to_check = tuple(sorted({1, min(4, ctx.ctx_pad.shape[0]), ctx.ctx_pad.shape[0]}))
    for horizon in (4, 6):
        out[f"compiled_b{compiled._bucket(1)}_s{horizon}_compile_s"] = compiled.prewarm(1, horizon)
    with torch.compiler.set_stance("fail_on_recompile"):
        for rows in rows_to_check:
            for name, value in trunk_parity(rows).items():
                out[f"compiled_dense_b{rows}_{name}"] = value
            for horizon in (4, 6):
                eager_s = measure(eager, rows, horizon)
                compiled_s = measure(compiled, rows, horizon)
                out[f"eager_b{rows}_s{horizon}_ms"] = eager_s * 1000
                out[f"compiled_b{rows}_s{horizon}_ms"] = compiled_s * 1000
                out[f"compiled_b{rows}_s{horizon}_executed_fps"] = rows * horizon / compiled_s
    print(json.dumps(out, indent=2, sort_keys=True), flush=True)
    return out


@dataclass
class Args:
    cfg: TrainConfig = dataclass_field(default_factory=TrainConfig)
    comment: str = ""
    resume: str | None = None
    eval: str | None = None
    eval_exec_horizon: int | None = None
    eval_n_matchups: int | None = None
    eval_eager: bool = False
    eval_max_parallel: int | None = None
    eval_output_name: str | None = None
    self_play_eval: str | None = None
    self_play_matches: int = 12
    self_play_frames: int = 14_400
    self_play_eager: bool = False
    self_play_instant_match_restart: bool = False
    self_play_process_cohorts: int = 1
    self_play_cohort_sweep: bool = False
    benchmark: bool = False
    benchmark_iterations: int = 20
    smoke: bool = False
    stop_after_update: int | None = None
    smoke_eval_matchups: int = 4
    profile_steps: int = 0


def main(args: Args) -> None:
    modes = {
        "--benchmark": args.benchmark,
        "--eval": args.eval is not None,
        "--self-play-eval": args.self_play_eval is not None,
        "--resume": args.resume is not None,
        "--profile-steps": args.profile_steps > 0,
    }
    selected_modes = [name for name, selected in modes.items() if selected]
    if len(selected_modes) > 1:
        raise SystemExit(f"pass only one mode, got {', '.join(selected_modes)}")

    if args.benchmark:
        run_benchmark(args.cfg, iterations=args.benchmark_iterations)
        return
    if args.eval is not None:
        eval_checkpoint(
            args.eval,
            exec_horizon=args.eval_exec_horizon,
            n_matchups=args.eval_n_matchups,
            eager=args.eval_eager,
            max_parallel=args.eval_max_parallel,
            output_name=args.eval_output_name,
        )
        return
    if args.self_play_eval is not None:
        cohorts = (1, 2, 3, 4) if args.self_play_cohort_sweep else (args.self_play_process_cohorts,)
        for cohort_count in cohorts:
            benchmark_self_play(
                args.self_play_eval,
                load_checkpoint=load_checkpoint,
                make_inference=BF16Inference,
                make_policy=make_policy,
                n_matches=args.self_play_matches,
                max_frames=args.self_play_frames,
                eager=args.self_play_eager,
                instant_match_restart=args.self_play_instant_match_restart,
                process_cohorts=cohort_count,
            )
        return
    resume_run = resume_state = None
    cfg = args.cfg
    if args.resume is not None:
        resume_state = load_for_resume(args.resume, Path("runs") / args.resume, device=DEVICE)
        if resume_state is None:
            raise SystemExit(f"no latest.pt for run {args.resume!r}")
        resume_run = args.resume
        cfg = config_from_state(resume_state["cfg"])
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
        profile_steps=args.profile_steps,
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
