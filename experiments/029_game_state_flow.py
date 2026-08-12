"""Sparse temporal controller with a training-only game-state flow expert.

The deployed policy is the experiment-026 architecture: a causal game-state
trunk and an autoregressive temporal MTP decoder. Training also asks a compact,
action-conditioned expert to predict structured game state at the same sparse
future offsets. The expert updates the shared trunk but is not constructed for
inference.

Run:
    uv run experiments/029_game_state_flow.py
    uv run experiments/029_game_state_flow.py --eval runs/<run>/final.pt
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import itertools
import json
import math
import time
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
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
from hal.eval.cross_stage import BOOTSTRAP_RESAMPLES
from hal.eval.cross_stage import PRIOR_SWEEP_SEED_STAGE
from hal.eval.cross_stage import MatchRow
from hal.eval.cross_stage import sweep_vs_cpu_prior_with_rows
from hal.eval.cross_stage import vs_cpu_metrics
from hal.eval.h2h import mirrored_configs
from hal.eval.h2h import run_h2h
from hal.eval.harness import DEFAULT_START_RETRIES
from hal.eval.harness import automatic_parallelism
from hal.eval.harness import default_session_cfg
from hal.eval.harness import resolve_parallelism
from hal.eval.harness import run_matches_vec
from hal.eval.harness import usable_cpus
from hal.eval.matchups import matchups_for_vs_cpu
from hal.eval.paired import summarize_paired
from hal.sim.process_vec import ProcessVecTelemetry
from hal.sim.rollout import covering_power_of_two
from hal.sim.session import Matchup
from hal.sim.session import PlayerSetup
from hal.sim.vec import VecMatch
from hal.training import scoring
from hal.training.checkpoints import BackgroundUploader
from hal.training.checkpoints import download_latest
from hal.training.checkpoints import load_for_resume
from hal.training.checkpoints import save_checkpoint
from hal.training.closed_loop import RecedingHorizon
from hal.training.dataloader import collate_windows
from hal.training.dataloader import make_loader
from hal.training.ego_stats import load_consolidated_stats
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
from hal.training.features import FeatureProjection
from hal.training.features import TrainBatch
from hal.training.features import preprocess
from hal.training.features import stack_actions
from hal.training.muon import SingleDeviceMuonWithAuxAdam
from hal.training.replay_reservoir import make_reservoir_loader
from hal.training.runs import make_run_name
from hal.training.runs import profile
from hal.training.runs import setup_run_dir
from hal.training.trunk import Rotary
from hal.training.trunk import Trunk
from hal.training.trunk import TrunkConfig
from hal.training.trunk import apply_rotary_emb

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_LN2 = math.log(2.0)
_N_CONT = 6
_PLAYER_PREFIXES = BASE_PLAYER_PREFIXES

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

STATE_ROLES: tuple[str, ...] = ("ego", "ego_nana", "opp_nana", "opp")
STATE_CONTINUOUS: tuple[str, ...] = ("position_x", "position_y", "percent", "shield", "hitlag_left", "direction")
STATE_CATEGORICAL: dict[str, int] = {
    "action": 1024,
    "stock": 5,
    "jumps_used": 9,
    "hurtbox_state": 4,
    "airborne": 2,
}
NANA_ROLES: frozenset[str] = frozenset(("ego_nana", "opp_nana"))
REFERENCE_026_RUN = (
    "260810-071709_026_temporal_mtp_mtp026-d384-L8-h6-Lc128-t128x2-"
    "o1-2-3-4-5-6-9-12-16-20-s4-base_ranked-anon-1_production-seed0-d384-b512"
)
REFERENCE_026_SHA256 = "22333d1d61d6b648c757f0f1f3e887925fbb12a08fdffd1cb4ae72d6d6f2ef88"


@dataclass
class TrainConfig:
    # Causal observation trunk.  The 5090 shakedown may change these together
    # to 256/4, but there is deliberately no intermediate width.
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 6
    attn_window: int = 0
    require_flex: bool = False
    L_ctx: int = 128

    decoder_arch_version: int = 3
    sample_chunk_length: int = 20
    head_offsets: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 9, 12, 16, 20)
    temporal_d_model: int = 128
    temporal_layers: int = 2
    temporal_heads: int = 4
    temporal_ff_dim: int = 256
    group_head_dim: int = 256
    action_embed_dim: int = 16
    offset_embed_dim: int = 16
    aux_loss_weight: float = 1.0
    group_order: tuple[str, ...] = GROUP_ORDER

    # Training-only action-conditioned game-state expert.
    state_offsets: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 9, 12, 16, 20)
    state_d_model: int = 128
    state_layers: int = 2
    state_heads: int = 4
    state_ff_dim: int = 512
    state_action_dim: int = 64
    state_time_dim: int = 128
    state_time_alpha: float = 1.5
    state_time_scale: float = 0.999
    state_loss_weight: float = 0.25
    state_loss_ramp_steps: int = 1000
    state_tail_weight: float = 0.25
    state_nana_weight: float = 0.2
    state_tail_endpoints: tuple[int, ...] = (6, 9, 12, 16, 20)

    action_vocab: int = 1024
    action_state_embed_dim: int = 48
    char_vocab: int = 32
    char_dim: int = 8
    stage_vocab: int = 32
    stage_dim: int = 4
    observation_bundle: str = "base"  # or v6_lean

    exec_horizon: int = 4
    final_diag_exec_horizon: int = 6
    decode_temp: float = 1.0
    inference_mode: str = "compiled"  # explicit "eager" is for debugging
    inference_buckets: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    # Hardware-derived by default. An explicit power of two is a reproducibility
    # or memory-pressure override, not an architecture parameter.
    compiled_inference_bucket: int | None = None

    seed: int = 0
    eval_seed: int = 0
    batch_size: int = 1024
    grad_accum_steps: int = 1
    muon_lr: float = 0.02
    adam_lr: float = 8.5e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: int = 16_384
    amp_dtype: str = "bfloat16"
    allow_tf32: bool = True
    compile_trunk: bool = True
    compile_temporal: bool = True

    wandb_log_code: bool = True
    wandb_grad_every: int = 1024
    gradient_diagnostic_batch_size: int = 64
    val_every: int = 1024
    val_n_samples: int = 1192
    val_batch_size: int = 128
    ckpt_every: int = 1024
    eval_every: int = 4096
    eval_max_frames: int = 7200
    eval_n_matchups: int = 32
    final_eval_n_matchups: int = 96
    final_diag_n_matchups: int = 32
    eval_max_parallel: int | None = None
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
        "state_d_model": cfg.state_d_model,
        "state_layers": cfg.state_layers,
        "state_heads": cfg.state_heads,
        "state_ff_dim": cfg.state_ff_dim,
        "state_action_dim": cfg.state_action_dim,
        "state_time_dim": cfg.state_time_dim,
        "state_loss_ramp_steps": cfg.state_loss_ramp_steps,
        "gradient_diagnostic_batch_size": cfg.gradient_diagnostic_batch_size,
        "batch_size": cfg.batch_size,
        "grad_accum_steps": cfg.grad_accum_steps,
        "max_steps": cfg.max_steps,
    }
    for name, value in positive.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if cfg.decoder_arch_version != 3:
        raise ValueError(f"unsupported decoder_arch_version={cfg.decoder_arch_version}")
    if cfg.d_model % cfg.n_heads or cfg.temporal_d_model % cfg.temporal_heads:
        raise ValueError("model dimensions must be divisible by their head counts")
    if (cfg.temporal_d_model // cfg.temporal_heads) % 2:
        raise ValueError("temporal head dimension must be even for rotary positions")
    offsets = tuple(cfg.head_offsets)
    if offsets != tuple(sorted(set(offsets))) or not offsets or offsets[0] != 1:
        raise ValueError(f"head_offsets must be sorted, unique, and start at 1, got {offsets}")
    if offsets[-1] > cfg.sample_chunk_length:
        raise ValueError("head_offsets extend beyond sample_chunk_length")
    if offsets[:6] != (1, 2, 3, 4, 5, 6):
        raise ValueError("the live four/six-frame decoders require a dense 1..6 prefix")
    if cfg.group_order != GROUP_ORDER:
        raise ValueError(f"group_order must be {GROUP_ORDER}, got {cfg.group_order}")
    if cfg.state_offsets != cfg.head_offsets:
        raise ValueError("state_offsets must exactly match the action head_offsets")
    if cfg.state_d_model % cfg.state_heads:
        raise ValueError("state_d_model must be divisible by state_heads")
    if cfg.state_time_dim % 2:
        raise ValueError("state_time_dim must be even")
    if cfg.state_tail_endpoints != (6, 9, 12, 16, 20):
        raise ValueError("state_tail_endpoints are frozen to (6,9,12,16,20)")
    if cfg.batch_size % cfg.grad_accum_steps:
        raise ValueError("batch_size must be divisible by grad_accum_steps")
    if cfg.exec_horizon not in (4, 6) or cfg.final_diag_exec_horizon != 6:
        raise ValueError("execution horizons are restricted to the unrolled four/six-frame decoders")
    if cfg.decode_temp != 1.0:
        raise ValueError("experiment 029 freezes sampling temperature at 1")
    if cfg.inference_mode not in ("compiled", "eager"):
        raise ValueError("inference_mode must be 'compiled' or 'eager'")
    if cfg.inference_buckets != (1, 2, 4, 8, 16, 32):
        raise ValueError("inference_buckets are frozen to (1,2,4,8,16,32)")
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
    if cfg.observation_bundle != "base":
        raise ValueError("experiment 029 is restricted to the base observation bundle")
    if not math.isfinite(cfg.aux_loss_weight) or cfg.aux_loss_weight < 0:
        raise ValueError("aux_loss_weight must be finite and non-negative")
    bounded = {
        "state_time_scale": cfg.state_time_scale,
        "state_loss_weight": cfg.state_loss_weight,
        "state_tail_weight": cfg.state_tail_weight,
        "state_nana_weight": cfg.state_nana_weight,
    }
    for name, value in bounded.items():
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be finite and in [0, 1], got {value!r}")
    if not math.isfinite(cfg.state_time_alpha) or cfg.state_time_alpha <= 0:
        raise ValueError("state_time_alpha must be finite and positive")
    if cfg.final_h2h_reference_run != REFERENCE_026_RUN:
        raise ValueError(f"final_h2h_reference_run must be pinned to {REFERENCE_026_RUN!r}")
    if cfg.final_h2h_reference_sha256.lower() != REFERENCE_026_SHA256:
        raise ValueError("final_h2h_reference_sha256 does not match the pinned experiment-026 checkpoint")
    if cfg.final_h2h_n_configs != 64:
        raise ValueError("final_h2h_n_configs is frozen to 64")
    if cfg.final_h2h_max_parallel != 32:
        raise ValueError("final_h2h_max_parallel is frozen to 32")
    if cfg.amp_dtype not in ("bfloat16", "float32"):
        raise ValueError("amp_dtype must be bfloat16 or float32")
    if cfg.reservoir_capacity < 2 * micro_batch_size(cfg):
        raise ValueError("reservoir_capacity must be at least twice the micro-batch size")


def micro_batch_size(cfg: TrainConfig) -> int:
    return cfg.batch_size // cfg.grad_accum_steps


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


def _planned_inference_programs(cfg: TrainConfig) -> tuple[tuple[int, int], ...]:
    scheduled = (
        (cfg.eval_n_matchups, cfg.exec_horizon),
        (cfg.final_eval_n_matchups, cfg.exec_horizon),
        (cfg.final_diag_n_matchups, cfg.final_diag_exec_horizon),
    )
    return tuple(sorted({(_eval_inference_bucket(cfg, n), horizon) for n, horizon in scheduled}))


def _planned_inference_buckets(cfg: TrainConfig) -> tuple[int, ...]:
    return tuple(sorted({bucket for bucket, _ in _planned_inference_programs(cfg)}))


def amp_context(cfg: TrainConfig, device: torch.device | str):
    if cfg.amp_dtype == "bfloat16" and torch.device(device).type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


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

    def _tokens(self, hidden: Tensor, previous: Tensor) -> Tensor:
        batch, length = hidden.shape[:2]
        horizon = len(self.head_offsets)
        action = self.codec.embed_frame(previous)
        offsets = torch.tensor(self.head_offsets, device=hidden.device)
        offset = self.offset_embedding(offsets).view(1, 1, horizon, -1).expand(batch, length, -1, -1)
        trunk = decoder_rmsnorm(hidden)[:, :, None].expand(-1, -1, horizon, -1)
        return self.token_projection(torch.cat((trunk, action, offset), dim=-1))

    def teacher_forced_states(self, hidden: Tensor, observed: Tensor, targets: Tensor) -> Tensor:
        expected = (*hidden.shape[:2], len(self.head_offsets), N_GROUPS)
        if observed.shape != (*hidden.shape[:2], N_GROUPS) or targets.shape != expected:
            raise ValueError(
                f"expected observed {(*hidden.shape[:2], N_GROUPS)} and targets {expected}, got "
                f"{tuple(observed.shape)} and {tuple(targets.shape)}"
            )
        previous = torch.cat((observed[:, :, None], targets[..., :-1, :]), dim=2)
        x = self._tokens(hidden, previous)
        x = x.reshape(hidden.shape[0] * hidden.shape[1], len(self.head_offsets), self.d_model)
        for block in self.blocks:
            x = block(x)
        return decoder_rmsnorm(x.view(*hidden.shape[:2], len(self.head_offsets), self.d_model))

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
        previous = observed
        caches: list[tuple[Tensor, Tensor] | None] = [None] * len(self.blocks)
        out: list[dict[str, Tensor]] = []
        for depth, offset in enumerate(self.head_offsets):
            offset_tensor = torch.full((hidden.shape[0],), offset, device=hidden.device, dtype=torch.long)
            state = self.token_projection(
                torch.cat((trunk, self.codec.embed_frame(previous), self.offset_embedding(offset_tensor)), dim=-1)
            )
            next_caches = []
            for block, past in zip(self.blocks, caches, strict=True):
                state, present = block.forward_step(state, past)
                next_caches.append(present)
            caches = next_caches
            state = decoder_rmsnorm(state)
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
        previous = observed
        caches: list[tuple[Tensor, Tensor] | None] = [None] * len(self.blocks)
        frames: list[Tensor] = []
        for depth, offset in enumerate(offsets):
            offset_tensor = torch.full((hidden.shape[0],), offset, device=hidden.device, dtype=torch.long)
            state = self.token_projection(
                torch.cat((trunk, self.codec.embed_frame(previous), self.offset_embedding(offset_tensor)), dim=-1)
            )
            next_caches = []
            for block, past in zip(self.blocks, caches, strict=True):
                state, present = block.forward_step(state, past)
                next_caches.append(present)
            caches = next_caches
            state = decoder_rmsnorm(state)
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
        previous = observed
        caches: list[tuple[Tensor, Tensor] | None] = [None] * len(self.blocks)
        frames: list[Tensor] = []
        all_logits: list[dict[str, Tensor]] = []
        for offset in self.head_offsets:
            offset_tensor = torch.full((hidden.shape[0],), offset, device=hidden.device, dtype=torch.long)
            state = self.token_projection(
                torch.cat((trunk, self.codec.embed_frame(previous), self.offset_embedding(offset_tensor)), dim=-1)
            )
            next_caches = []
            for block, past in zip(self.blocks, caches, strict=True):
                state, present = block.forward_step(state, past)
                next_caches.append(present)
            caches = next_caches
            state = decoder_rmsnorm(state)
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
        self.group_order = tuple(cfg.group_order)
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
                require_flex=cfg.require_flex,
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


def sinusoidal_time_embedding(t: Tensor, dim: int) -> Tensor:
    frequencies = torch.exp(
        -math.log(10_000.0) * torch.arange(dim // 2, device=t.device, dtype=torch.float32) / max(dim // 2 - 1, 1)
    )
    angles = t.float()[:, None] * frequencies[None]
    return torch.cat((angles.sin(), angles.cos()), dim=-1)


class StateExpertBlock(nn.Module):
    """Causal future-token attention followed by cross-attention to the policy trunk."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        d = cfg.state_d_model
        self.self_attention = nn.MultiheadAttention(d, cfg.state_heads, batch_first=True, bias=False)
        self.cross_attention = nn.MultiheadAttention(
            d, cfg.state_heads, batch_first=True, bias=False, kdim=cfg.d_model, vdim=cfg.d_model
        )
        self.up = nn.Linear(d, cfg.state_ff_dim, bias=False)
        self.down = nn.Linear(cfg.state_ff_dim, d, bias=False)

    def forward(self, x: Tensor, context: Tensor, context_pad: Tensor, causal_mask: Tensor) -> Tensor:
        normalized = decoder_rmsnorm(x)
        x = x + self.self_attention(normalized, normalized, normalized, attn_mask=causal_mask, need_weights=False)[0]
        x = (
            x
            + self.cross_attention(
                decoder_rmsnorm(x), context, context, key_padding_mask=context_pad, need_weights=False
            )[0]
        )
        return x + self.down(F.silu(self.up(decoder_rmsnorm(x))))


class GameStateExpert(nn.Module):
    """Training-only structured dynamics expert at the policy's sparse offsets."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.offsets = tuple(cfg.state_offsets)
        self.continuous_dim = len(STATE_ROLES) * len(STATE_CONTINUOUS)
        self.time_dim = cfg.state_time_dim
        self.action_in = nn.Linear(A_DIM, cfg.state_action_dim)
        self.action_prefix = nn.GRU(cfg.state_action_dim, cfg.state_action_dim, batch_first=True)
        token_in = self.continuous_dim + cfg.state_action_dim + cfg.state_time_dim
        self.token_in = nn.Linear(token_in, cfg.state_d_model)
        self.offset_embedding = nn.Embedding(cfg.sample_chunk_length + 1, cfg.state_d_model)
        self.blocks = nn.ModuleList([StateExpertBlock(cfg) for _ in range(cfg.state_layers)])
        self.continuous_out = nn.Linear(cfg.state_d_model, self.continuous_dim)
        self.categorical_out = nn.ModuleDict(
            {
                f"{role}_{name}": nn.Linear(cfg.state_d_model, vocab)
                for role in STATE_ROLES
                for name, vocab in STATE_CATEGORICAL.items()
            }
        )
        self.presence_out = nn.ModuleDict({role: nn.Linear(cfg.state_d_model, 2) for role in NANA_ROLES})
        nn.init.zeros_(self.continuous_out.weight)
        nn.init.zeros_(self.continuous_out.bias)

    def action_summaries(self, actions: Tensor) -> Tensor:
        if actions.shape[1:] != (max(self.offsets), A_DIM):
            raise ValueError(f"future actions must be [B, {max(self.offsets)}, {A_DIM}]")
        dense, _ = self.action_prefix(self.action_in(actions))
        indices = torch.tensor(self.offsets, device=actions.device) - 1
        return dense.index_select(1, indices)

    def _run(self, continuous: Tensor, actions: Tensor, t: Tensor, context: Tensor, ctx_pad: Tensor) -> Tensor:
        batch, horizon = continuous.shape[:2]
        action = self.action_summaries(actions)
        time = sinusoidal_time_embedding(t, self.time_dim).to(continuous.dtype)
        time = time[:, None].expand(-1, horizon, -1)
        x = self.token_in(torch.cat((continuous, action, time), dim=-1))
        offsets = torch.tensor(self.offsets, device=x.device)
        x = x + self.offset_embedding(offsets)[None]
        causal = torch.ones(horizon, horizon, device=x.device, dtype=torch.bool).triu(1)
        positions = torch.arange(context.shape[1], device=context.device)[None]
        context_pad = positions < ctx_pad[:, None]
        for block in self.blocks:
            x = block(x, context, context_pad, causal)
        return decoder_rmsnorm(x)

    def forward(
        self, noisy: Tensor, actions: Tensor, t: Tensor, context: Tensor, ctx_pad: Tensor
    ) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
        flow_states = self._run(noisy, actions, t, context, ctx_pad)
        # The CE stream never sees the noised target, which prevents target leakage at high flow times.
        categorical_states = self._run(torch.zeros_like(noisy), actions, t, context, ctx_pad)
        categorical = {name: head(categorical_states) for name, head in self.categorical_out.items()}
        presence = {name: head(categorical_states) for name, head in self.presence_out.items()}
        return self.continuous_out(flow_states), categorical, presence


class TrainingModel(nn.Module):
    """The full resumable training model; only ``policy`` is used after training."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        # Construct the complete policy first to preserve seeded experiment-026 initialization.
        self.policy = GPT(cfg)
        self.state_expert = GameStateExpert(cfg)


@dataclass(frozen=True, slots=True)
class StateBatch:
    batch: TrainBatch
    continuous: Tensor  # [B, offsets, roles, continuous fields]
    continuous_valid: Tensor
    categorical: dict[str, Tensor]  # role_field -> [B, offsets]
    categorical_valid: dict[str, Tensor]
    presence: dict[str, Tensor]  # Nana role -> [B, offsets]

    def to(self, device: str | torch.device) -> StateBatch:
        return StateBatch(
            batch=self.batch.to(device),
            continuous=self.continuous.to(device, non_blocking=True),
            continuous_valid=self.continuous_valid.to(device, non_blocking=True),
            categorical={name: value.to(device, non_blocking=True) for name, value in self.categorical.items()},
            categorical_valid={
                name: value.to(device, non_blocking=True) for name, value in self.categorical_valid.items()
            },
            presence={name: value.to(device, non_blocking=True) for name, value in self.presence.items()},
        )

    def pin_memory(self) -> StateBatch:
        return StateBatch(
            batch=self.batch.pin_memory(),
            continuous=self.continuous.pin_memory(),
            continuous_valid=self.continuous_valid.pin_memory(),
            categorical={name: value.pin_memory() for name, value in self.categorical.items()},
            categorical_valid={name: value.pin_memory() for name, value in self.categorical_valid.items()},
            presence={name: value.pin_memory() for name, value in self.presence.items()},
        )

    def take(self, count: int) -> StateBatch:
        context = self.batch.context
        return StateBatch(
            batch=TrainBatch(
                context=Context(
                    features={name: value[:count] for name, value in context.features.items()},
                    ctx_pad=context.ctx_pad[:count],
                    slot_ids=None if context.slot_ids is None else context.slot_ids[:count],
                    reset=None if context.reset is None else context.reset[:count],
                ),
                target=self.batch.target[:count],
                replay_ids=None if self.batch.replay_ids is None else self.batch.replay_ids[:count],
            ),
            continuous=self.continuous[:count],
            continuous_valid=self.continuous_valid[:count],
            categorical={name: value[:count] for name, value in self.categorical.items()},
            categorical_valid={name: value[:count] for name, value in self.categorical_valid.items()},
            presence={name: value[:count] for name, value in self.presence.items()},
        )


def collate_state_batch(
    windows: list[dict[str, np.ndarray]],
    batch: TrainBatch,
    *,
    stats: dict[str, FeatureStats],
    cfg: TrainConfig,
    projection: FeatureProjection,
) -> StateBatch:
    stacked = collate_windows(windows)
    features = preprocess(stacked, stats, projection=projection)
    indices = torch.tensor([cfg.L_ctx - 1 + offset for offset in cfg.state_offsets])
    continuous_rows: list[Tensor] = []
    valid_rows: list[Tensor] = []
    categorical: dict[str, Tensor] = {}
    categorical_valid: dict[str, Tensor] = {}
    presence: dict[str, Tensor] = {}
    for role in STATE_ROLES:
        role_valid = 1.0 - features.get(f"{role}_position_x_mask", torch.zeros_like(features[f"{role}_position_x"]))
        role_valid = role_valid.index_select(1, indices).bool()
        role_continuous: list[Tensor] = []
        role_continuous_valid: list[Tensor] = []
        for name in STATE_CONTINUOUS:
            value = features[f"{role}_{name}"].index_select(1, indices)
            mask = features.get(f"{role}_{name}_mask", torch.zeros_like(features[f"{role}_{name}"]))
            role_continuous.append(value)
            role_continuous_valid.append(~mask.index_select(1, indices).bool())
        continuous_rows.append(torch.stack(role_continuous, dim=-1))
        valid_rows.append(torch.stack(role_continuous_valid, dim=-1))
        for name in STATE_CATEGORICAL:
            key = f"{role}_{name}"
            categorical[key] = features[key].index_select(1, indices)
            categorical_valid[key] = role_valid
        if role in NANA_ROLES:
            presence[role] = role_valid.long()
    return StateBatch(
        batch=batch,
        continuous=torch.stack(continuous_rows, dim=2),
        continuous_valid=torch.stack(valid_rows, dim=2),
        categorical=categorical,
        categorical_valid=categorical_valid,
        presence=presence,
    )


def quantize(model: GPT, actions: Tensor) -> Tensor:
    return model.codec.quantize(actions)


def dequantize(model: GPT, indices: Tensor) -> Tensor:
    return model.codec.dequantize(indices)


def prepared_targets(model: GPT, batch: TrainBatch) -> tuple[Tensor, Tensor, Tensor]:
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


def chunk_targets(model: GPT, batch: TrainBatch) -> tuple[Tensor, Tensor]:
    _, targets, valid = prepared_targets(model, batch)
    return targets, valid


@dataclass(frozen=True, slots=True)
class ActionLoss:
    nll: Tensor  # [valid prefixes, selected offsets, groups]
    targets: Tensor


def action_loss(model: GPT, batch: TrainBatch) -> ActionLoss:
    history_indices, targets, valid = prepared_targets(model, batch)
    hidden = model(batch.context.features, batch.context.ctx_pad, history_indices)
    dense_nll = model.temporal.teacher_forced_nll(hidden, history_indices, targets)
    nll = dense_nll[valid]
    target_valid = targets[valid]
    if nll.numel() == 0:
        raise ValueError("batch contains no valid context prefixes")
    return ActionLoss(nll=nll, targets=target_valid)


def objective(parts: ActionLoss, aux_loss_weight: float = 1.0) -> Tensor:
    """Primary dense-four joint NLL plus the mean auxiliary joint NLL."""
    joint = parts.nll.sum(dim=-1)
    primary = joint[:, :4].mean()
    auxiliary = joint[:, 4:].mean()
    return primary + aux_loss_weight * auxiliary


@dataclass(frozen=True, slots=True)
class StateLoss:
    loss: Tensor
    continuous_by_offset: Tensor
    categorical_by_offset: Tensor
    active: Tensor


def sample_state_time(
    batch_size: int, cfg: TrainConfig, *, device: torch.device, gen: torch.Generator | None = None
) -> Tensor:
    uniform = torch.rand(batch_size, device=device, generator=gen)
    return cfg.state_time_scale * (1.0 - uniform.pow(1.0 / cfg.state_time_alpha))


def sample_tail_mask(
    batch_size: int, cfg: TrainConfig, *, device: torch.device, gen: torch.Generator | None = None
) -> Tensor:
    choices = torch.tensor(cfg.state_tail_endpoints, device=device)
    endpoint = choices[torch.randint(len(choices), (batch_size,), device=device, generator=gen)]
    offsets = torch.tensor(cfg.state_offsets, device=device)
    return offsets[None] <= endpoint[:, None]


def role_weight(role: str, cfg: TrainConfig) -> float:
    return cfg.state_nana_weight if role in NANA_ROLES else 1.0


def weighted_continuous_loss(squared: Tensor, valid: Tensor, cfg: TrainConfig) -> Tensor:
    if squared.shape != valid.shape or squared.shape[-2:] != (len(STATE_ROLES), len(STATE_CONTINUOUS)):
        raise ValueError("state continuous error and validity tensors have incompatible shapes")
    numerator = torch.zeros(squared.shape[:2], device=squared.device)
    denominator = torch.zeros_like(numerator)
    for role_index, role in enumerate(STATE_ROLES):
        weight = role_weight(role, cfg)
        for field_index in range(len(STATE_CONTINUOUS)):
            selected = valid[..., role_index, field_index].float()
            numerator += weight * squared[..., role_index, field_index] * selected
            denominator += selected
    return numerator / denominator.clamp_min(1.0)


def state_loss(
    model: TrainingModel,
    batch: StateBatch,
    hidden: Tensor,
    cfg: TrainConfig,
    *,
    gen: torch.Generator | None = None,
) -> StateLoss:
    target = batch.continuous
    flat_target = target.flatten(2)
    flat_valid = batch.continuous_valid.flatten(2)
    noise = torch.randn(flat_target.shape, device=flat_target.device, dtype=flat_target.dtype, generator=gen)
    t = sample_state_time(target.shape[0], cfg, device=target.device, gen=gen)
    noisy = (1.0 - t[:, None, None]) * noise + t[:, None, None] * flat_target
    # Missing Nana/field slots must not inject irreducible random inputs into
    # otherwise valid predictions. Their losses are masked below as well.
    noisy = noisy.masked_fill(~flat_valid, 0.0)
    velocity = flat_target - noise
    predicted, categorical_logits, presence_logits = model.state_expert(
        noisy, batch.batch.target[:, : cfg.sample_chunk_length], t, hidden, batch.batch.context.ctx_pad
    )
    squared = (predicted.float() - velocity.float()).square().view_as(target)

    continuous = weighted_continuous_loss(squared, batch.continuous_valid, cfg)

    categorical_numerator = torch.zeros_like(continuous)
    categorical_denominator = torch.zeros_like(continuous)
    for key, target_ids in batch.categorical.items():
        role, name = next(
            (role, key.removeprefix(f"{role}_"))
            for role in sorted(STATE_ROLES, key=len, reverse=True)
            if key.startswith(f"{role}_")
        )
        vocab = STATE_CATEGORICAL[name]
        valid = batch.categorical_valid[key].float()
        normalized = F.cross_entropy(
            categorical_logits[key].float().reshape(-1, vocab),
            target_ids.clamp(0, vocab - 1).reshape(-1),
            reduction="none",
        ).view_as(target_ids) / math.log(vocab)
        categorical_numerator += role_weight(role, cfg) * normalized * valid
        categorical_denominator += valid
    for role, target_ids in batch.presence.items():
        normalized = F.cross_entropy(
            presence_logits[role].float().reshape(-1, 2), target_ids.reshape(-1), reduction="none"
        ).view_as(target_ids) / math.log(2)
        categorical_numerator += cfg.state_nana_weight * normalized
        categorical_denominator += 1.0
    categorical = categorical_numerator / categorical_denominator.clamp_min(1.0)

    combined = 0.5 * continuous + 0.5 * categorical
    active = sample_tail_mask(target.shape[0], cfg, device=target.device, gen=gen)
    core = combined[:, :6].mean(dim=1)
    tail_active = active[:, 6:].float()
    tail = (combined[:, 6:] * tail_active).sum(dim=1) / tail_active.sum(dim=1).clamp_min(1.0)
    return StateLoss(
        loss=(core + cfg.state_tail_weight * tail).mean(),
        continuous_by_offset=continuous.detach().mean(dim=0),
        categorical_by_offset=categorical.detach().mean(dim=0),
        active=active.detach(),
    )


def state_weight(step: int, cfg: TrainConfig) -> float:
    return cfg.state_loss_weight * min(step / cfg.state_loss_ramp_steps, 1.0)


def nll_mean_metrics(mean_nll: Tensor, offsets: tuple[int, ...]) -> dict[str, float]:
    if mean_nll.shape != (len(offsets), N_GROUPS):
        raise ValueError(f"mean NLL has shape {tuple(mean_nll.shape)}")
    joint = mean_nll.sum(dim=-1) / _LN2
    out = {
        "loss": float(joint[:4].mean() + joint[4:].mean()),
        "primary_nll": float(joint[:4].mean()),
        "auxiliary_nll": float(joint[4:].mean()),
    }
    for depth, offset in enumerate(offsets):
        out[f"nll_o{offset:02d}"] = float(joint[depth])
        for group, name in enumerate(GROUP_NAMES):
            out[f"nll_o{offset:02d}_{name}"] = float(mean_nll[depth, group] / _LN2)
    return out


def nll_metrics(nll: Tensor, offsets: tuple[int, ...]) -> dict[str, float]:
    return nll_mean_metrics(nll.mean(dim=0), offsets)


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
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    nll_sum = torch.zeros(len(model.head_offsets), N_GROUPS, dtype=torch.float64)
    correct = torch.zeros_like(nll_sum)
    count = 0
    rollout_correct = torch.zeros_like(nll_sum)
    rollout_nll = torch.zeros_like(nll_sum)
    target_rows: list[Tensor] = []
    sampled_rows: list[Tensor] = []
    observed_rows: list[Tensor] = []
    quantization_squared = quantization_count = invalid_triggers = 0.0
    try:
        for cpu_batch in batches:
            batch = cpu_batch.to(device)
            history, targets, valid = prepared_targets(model, batch)
            with amp_context(cfg, device):
                hidden = model(batch.context.features, batch.context.ctx_pad, history)
                logits = model.temporal.teacher_forced_logits_by_group(hidden, history, targets)
                dense_nll = model.temporal.teacher_forced_nll(hidden, history, targets)
            selected_nll = dense_nll[valid]
            nll_sum += selected_nll.double().sum(dim=0).cpu()
            count += selected_nll.shape[0]
            for group, name in enumerate(GROUP_NAMES):
                correct[:, group] += (
                    (logits[name].argmax(dim=-1)[valid] == targets[..., group][valid]).double().sum(dim=0).cpu()
                )

            # Rollout-conditioned diagnostics use the last real context prefix;
            # temporal and within-frame prefixes are sampled greedily.
            last_observed = history[:, -1]
            with amp_context(cfg, device):
                rollout_logits, sampled_all = model.temporal.rollout_conditioned_logits(hidden, last_observed)
            sampled = sampled_all[:, :6]
            target_last = targets[:, -1]
            target_rows.append(target_last.cpu())
            sampled_rows.append(sampled.cpu())
            observed_rows.append(last_observed.cpu())
            for depth in range(len(model.head_offsets)):
                for group, name in enumerate(GROUP_NAMES):
                    expected = target_last[:, depth, group]
                    step_logits = rollout_logits[depth][name]
                    rollout_correct[depth, group] += (step_logits.argmax(-1) == expected).double().sum().cpu()
                    rollout_nll[depth, group] += (
                        F.cross_entropy(step_logits.float(), expected, reduction="sum").double().cpu()
                    )

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
    out = nll_mean_metrics(nll_sum / count, model.head_offsets)
    for depth, offset in enumerate(model.head_offsets):
        for group, name in enumerate(GROUP_NAMES):
            out[f"acc_o{offset:02d}_{name}"] = float(correct[depth, group] / count)
            rollout_count = sum(row.shape[0] for row in target_rows)
            roll_nll = float(rollout_nll[depth, group] / max(rollout_count, 1) / _LN2)
            out[f"rollout_nll_o{offset:02d}_{name}"] = roll_nll
            out[f"exposure_gap_o{offset:02d}_{name}"] = roll_nll - out[f"nll_o{offset:02d}_{name}"]
            out[f"rollout_acc_o{offset:02d}_{name}"] = float(rollout_correct[depth, group] / max(rollout_count, 1))
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
    return out


@torch.no_grad()
def combined_val_metrics(model: TrainingModel, batches: list[StateBatch], cfg: TrainConfig) -> dict[str, float]:
    action = val_metrics(model.policy, [batch.batch for batch in batches], cfg)
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(0)
    continuous = torch.zeros(len(cfg.state_offsets), dtype=torch.float64)
    categorical = torch.zeros_like(continuous)
    total = samples = 0
    try:
        for cpu_batch in batches:
            batch = cpu_batch.to(device)
            history, _, _ = prepared_targets(model.policy, batch.batch)
            with amp_context(cfg, device):
                hidden = model.policy(batch.batch.context.features, batch.batch.context.ctx_pad, history)
                parts = state_loss(model, batch, hidden, cfg, gen=generator)
            count = batch.batch.target.shape[0]
            total += float(parts.loss) * count
            continuous += parts.continuous_by_offset.double().cpu() * count
            categorical += parts.categorical_by_offset.double().cpu() * count
            samples += count
    finally:
        model.train(was_training)
    out = {f"action_{name}": value for name, value in action.items()}
    out["state_loss"] = total / max(samples, 1)
    for index, offset in enumerate(cfg.state_offsets):
        out[f"state_flow_mse_o{offset:02d}"] = float(continuous[index] / max(samples, 1))
        out[f"state_cat_nll_o{offset:02d}"] = float(categorical[index] / max(samples, 1))
    return out


def gradient_interaction(model: TrainingModel, batch: StateBatch, cfg: TrainConfig) -> dict[str, float]:
    """Compare action and state gradients on the shared trunk using fixed state noise."""
    device = next(model.parameters()).device
    selected = batch.take(min(cfg.gradient_diagnostic_batch_size, batch.batch.target.shape[0])).to(device)
    shared = tuple(model.policy.trunk.parameters())

    action_parts = action_loss(model.policy, selected.batch)
    action_objective = objective(action_parts, cfg.aux_loss_weight)
    action_grad = torch.autograd.grad(action_objective, shared, allow_unused=False)

    history, _, _ = prepared_targets(model.policy, selected.batch)
    hidden = model.policy(selected.batch.context.features, selected.batch.context.ctx_pad, history)
    generator = torch.Generator(device=device).manual_seed(0)
    state_objective = state_loss(model, selected, hidden, cfg, gen=generator).loss
    state_grad = torch.autograd.grad(state_objective, shared, allow_unused=False)

    action_norm_sq = sum(gradient.float().square().sum() for gradient in action_grad)
    state_norm_sq = sum(gradient.float().square().sum() for gradient in state_grad)
    dot = sum((left.float() * right.float()).sum() for left, right in zip(action_grad, state_grad, strict=True))
    action_norm = action_norm_sq.sqrt()
    state_norm = state_norm_sq.sqrt()
    cosine = dot / (action_norm * state_norm).clamp_min(1e-12)
    return {
        "action_trunk_grad_norm": float(action_norm),
        "state_trunk_grad_norm": float(state_norm),
        "trunk_grad_cosine": float(cosine),
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

    The background prewarm builds every bucket required by the scheduled evaluations.
    Runtime calls use the smallest compiled bucket that fits. Padding and slot-keyed
    random streams leave real rows unchanged.
    """

    def __init__(
        self,
        model: GPT,
        cfg: TrainConfig,
        *,
        compiled: bool | None = None,
        compile_mode: str = "default",
        compiled_buckets: tuple[int, ...] | None = None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.buckets = tuple(cfg.inference_buckets)
        chosen = _planned_inference_buckets(cfg) if compiled_buckets is None else compiled_buckets
        self.compiled_buckets = tuple(sorted(set(chosen)))
        if not self.compiled_buckets:
            raise ValueError("compiled_buckets must contain at least one bucket")
        if any(bucket < 1 or bucket & (bucket - 1) for bucket in self.compiled_buckets):
            raise ValueError(f"compiled_buckets must be positive powers of two, got {self.compiled_buckets}")
        requested = cfg.inference_mode == "compiled" if compiled is None else compiled
        self.compiled = bool(requested and next(model.parameters()).device.type == "cuda")
        self.compile_mode = compile_mode
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
            return next(bucket for bucket in self.buckets if bucket >= rows)
        except StopIteration:
            return covering_power_of_two(rows)

    def _trunk(self, bucket: int) -> Callable:
        if bucket not in self._trunks:

            def fn(features, pad, actions):
                return self.model(features, pad, actions)

            self._trunks[bucket] = torch.compile(fn, dynamic=False, mode=self.compile_mode) if self.compiled else fn
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
        padded = _pad_context(ctx, bucket)
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
class DecodeTelemetry:
    calls: int = 0
    rows: int = 0
    executed_frames: int = 0
    seconds: float = 0.0
    max_seconds: float = 0.0
    durations: list[float] = dataclass_field(default_factory=list)

    def record(self, *, rows: int, horizon: int, seconds: float) -> None:
        self.calls += 1
        self.rows += rows
        self.executed_frames += rows * horizon
        self.seconds += seconds
        self.max_seconds = max(self.max_seconds, seconds)
        self.durations.append(seconds)

    def metrics(self) -> dict[str, float]:
        durations = np.asarray(self.durations, dtype=np.float64)
        return {
            "decode_calls": float(self.calls),
            "decode_rows": float(self.rows),
            "decode_seconds": self.seconds,
            "decode_max_seconds": self.max_seconds,
            "decode_p50_ms": float(np.percentile(durations, 50) * 1000) if durations.size else 0.0,
            "decode_p95_ms": float(np.percentile(durations, 95) * 1000) if durations.size else 0.0,
            "decode_p99_ms": float(np.percentile(durations, 99) * 1000) if durations.size else 0.0,
            "decode_calls_over_100ms": float(np.count_nonzero(durations > 0.1)),
            "decode_mean_rows": self.rows / max(self.calls, 1),
            "decode_replans_per_s": self.calls / max(self.seconds, 1e-12),
            "decode_executed_frames_per_s": self.executed_frames / max(self.seconds, 1e-12),
        }


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
            raise ValueError("experiment 029 does not condition on a committed RTC prefix")
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
    started = time.perf_counter()
    try:
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
    metrics["exec_horizon"] = float(horizon)
    metrics.update(telemetry.metrics())
    _write_eval_evidence(replay_dir, rows, metrics, protocol)
    return metrics


def lr_schedule(cfg: TrainConfig):
    floor = 1e-5 / cfg.adam_lr

    def schedule(step: int) -> float:
        if step < cfg.warmup_steps:
            return step / max(cfg.warmup_steps, 1)
        progress = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return floor + (1.0 - floor) * cosine

    return schedule


def make_optimizer(model: TrainingModel, cfg: TrainConfig) -> SingleDeviceMuonWithAuxAdam:
    policy = model.policy
    muon = [parameter for parameter in policy.trunk.blocks.parameters() if parameter.ndim >= 2]
    muon_ids = {id(parameter) for parameter in muon}
    embedding_modules = (
        policy.cat_embeds,
        policy.v6_cat_embeds,
        policy.char_emb,
        policy.stage_emb,
        policy.codec.class_embeddings,
        policy.temporal.offset_embedding,
        model.state_expert.offset_embedding,
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
            dict(params=muon, lr=cfg.muon_lr, momentum=0.95, weight_decay=cfg.weight_decay, use_muon=True),
            dict(params=decay, lr=cfg.adam_lr, weight_decay=cfg.weight_decay, **adam),
            dict(params=no_decay, lr=cfg.adam_lr, weight_decay=0.0, **adam),
        ]
    )


def subsystem_parameter_counts(model: TrainingModel) -> dict[str, int]:
    policy = model.policy
    groups = {
        "trunk": policy.trunk,
        "observation": policy.ctx_proj,
        "codec": policy.codec,
        "temporal": policy.temporal.blocks,
        "heads": nn.ModuleList([policy.temporal.outputs, policy.temporal.trunk_outputs]),
        "state_expert": model.state_expert,
    }
    return {name: sum(parameter.numel() for parameter in module.parameters()) for name, module in groups.items()}


def model_tag(cfg: TrainConfig) -> str:
    offsets = "-".join(map(str, cfg.head_offsets))
    return (
        f"mtp029-stateflow-d{cfg.d_model}-L{cfg.n_layers}-h{cfg.n_heads}-Lc{cfg.L_ctx}-"
        f"t{cfg.temporal_d_model}x{cfg.temporal_layers}-o{offsets}-s{cfg.exec_horizon}-{cfg.observation_bundle}"
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


def loader_kwargs(cfg: TrainConfig, stats: dict[str, FeatureStats]) -> dict:
    v6 = cfg.observation_bundle == "v6_lean"
    return dict(
        data_root=cfg.data_root,
        remote=streams.remote_for_local(cfg.data_root),
        cache_limit=f"{cfg.cache_limit_gb}gb",
        shuffle_block_size=cfg.shuffle_block_size,
        stats=stats,
        L_ctx=cfg.L_ctx,
        L_chunk=cfg.sample_chunk_length,
        batch_size=micro_batch_size(cfg),
        seed=cfg.seed,
        schema_version=cfg.mds_schema_version,
        extra=V6_PLAYER_COLUMNS if v6 else None,
        projection=None if v6 else BASE_ACTION_PROJECTION,
    )


def validate_batch_geometry(batch: StateBatch, cfg: TrainConfig, expected_batch_size: int | None = None) -> None:
    train_batch = batch.batch
    if train_batch.target.shape[1:] != (cfg.sample_chunk_length, A_DIM):
        raise ValueError(
            f"target must be [B, {cfg.sample_chunk_length}, {A_DIM}], got {tuple(train_batch.target.shape)}"
        )
    batch_size = train_batch.target.shape[0]
    if expected_batch_size is not None and batch_size != expected_batch_size:
        raise ValueError(f"fixed training batch must contain {expected_batch_size} rows, got {batch_size}")
    if train_batch.context.ctx_pad.shape != (batch_size,):
        raise ValueError("ctx_pad shape does not match the batch")
    wrong = {
        name: tuple(value.shape)
        for name, value in train_batch.context.features.items()
        if value.shape[:2] != (batch_size, cfg.L_ctx)
    }
    if wrong:
        raise ValueError(f"context features have the wrong geometry: {wrong}")
    state_shape = (batch_size, len(cfg.state_offsets), len(STATE_ROLES), len(STATE_CONTINUOUS))
    if batch.continuous.shape != state_shape or batch.continuous_valid.shape != state_shape:
        raise ValueError(f"continuous state targets must both have shape {state_shape}")


def cache_validation(loader: Iterable[StateBatch], n_samples: int) -> list[StateBatch]:
    batches: list[StateBatch] = []
    count = 0
    for batch in loader:
        remaining = n_samples - count
        if remaining <= 0:
            break
        if batch.batch.target.shape[0] > remaining:
            batch = batch.take(remaining)
        batches.append(batch)
        count += batch.batch.target.shape[0]
    if count != n_samples:
        raise RuntimeError(f"validation yielded {count} samples, expected {n_samples}")
    return batches


def device_batches(
    cpu_batches: list[StateBatch], device: str | torch.device, copy_stream: torch.cuda.Stream | None
) -> Iterator[StateBatch]:
    # A one-ahead asynchronous copy keeps loader wait separate from GPU time.
    if not cpu_batches:
        return
    target = torch.device(device)
    if target.type != "cuda" or copy_stream is None:
        for batch in cpu_batches:
            yield batch.to(target)
        return
    compute_stream = torch.cuda.current_stream(target)
    with torch.cuda.stream(copy_stream):
        staged = cpu_batches[0].to(target)
    for index in range(len(cpu_batches)):
        compute_stream.wait_stream(copy_stream)
        ready = staged
        for tensor in (
            *ready.batch.context.features.values(),
            ready.batch.context.ctx_pad,
            ready.batch.target,
            ready.continuous,
            ready.continuous_valid,
            *ready.categorical.values(),
            *ready.categorical_valid.values(),
            *ready.presence.values(),
        ):
            tensor.record_stream(compute_stream)
        if index + 1 < len(cpu_batches):
            with torch.cuda.stream(copy_stream):
                staged = cpu_batches[index + 1].to(target)
        yield ready


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _make_loaders(cfg: TrainConfig, stats: dict[str, FeatureStats]):
    kwargs = loader_kwargs(cfg, stats)
    projection = kwargs["projection"]
    if projection is None:
        raise ValueError("experiment 029 requires an explicit base feature projection")
    batch_transform = functools.partial(
        collate_state_batch,
        stats=stats,
        cfg=cfg,
        projection=projection,
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
            batch_transform=batch_transform,
            **kwargs,
        )
    val_kwargs = {**kwargs, "batch_size": cfg.val_batch_size}
    val_loader = make_loader(
        split=cfg.val_split,
        num_workers=0,
        compact=cfg.compact_data,
        batch_transform=batch_transform,
        **val_kwargs,
    )
    return train_loader, cache_validation(val_loader, cfg.val_n_samples)


def resolve_h2h_reference(cfg: TrainConfig, run_dir: Path) -> Path:
    local = Path("runs") / cfg.final_h2h_reference_run / "final.pt"
    checkpoint = local.resolve() if local.is_file() else None
    if checkpoint is None:
        checkpoint = download_latest(
            cfg.final_h2h_reference_run,
            run_dir / "h2h_reference",
            name="final.pt",
        )
    if checkpoint is None:
        raise RuntimeError(f"no final.pt for pinned H2H reference {cfg.final_h2h_reference_run!r}")
    actual = _checkpoint_sha256(checkpoint)
    if actual != cfg.final_h2h_reference_sha256:
        raise RuntimeError(f"H2H reference SHA-256 mismatch: expected {cfg.final_h2h_reference_sha256}, got {actual}")
    print(f"[h2h] pinned reference: {checkpoint} ({actual})", flush=True)
    return checkpoint


def load_026_reference(path: Path, *, device: str = DEVICE) -> tuple[GPT, TrainConfig, dict[str, FeatureStats], dict]:
    state = torch.load(path, map_location=device, weights_only=False)
    known = {item.name for item in fields(TrainConfig)}
    cfg = TrainConfig(**{name: value for name, value in state["cfg"].items() if name in known})
    validate_config(cfg)
    model = GPT(cfg).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    stats = load_consolidated_stats(Path(cfg.data_root) / "stats.json")
    return model, cfg, stats, state


def final_h2h(
    model: GPT,
    stats: dict[str, FeatureStats],
    cfg: TrainConfig,
    *,
    run_dir: Path,
    reference: Path,
    uploader: BackgroundUploader | None,
    inference: BF16Inference | None = None,
) -> dict[str, object]:
    reference_model, reference_cfg, reference_stats, reference_state = load_026_reference(reference)
    protocol_fields = ("L_ctx", "head_offsets", "exec_horizon", "decode_temp", "observation_bundle")
    mismatches = {
        name: (getattr(cfg, name), getattr(reference_cfg, name))
        for name in protocol_fields
        if getattr(cfg, name) != getattr(reference_cfg, name)
    }
    if mismatches:
        raise RuntimeError(f"H2H protocol mismatch (029, 026): {mismatches}")

    self_label = "029-state-flow"
    reference_label = "026-reference"

    def build_self(seed: int) -> RecedingHorizon:
        return make_policy(model, stats, cfg, decode_seed=seed, inference=inference)

    def build_reference(seed: int) -> RecedingHorizon:
        return make_policy(reference_model, reference_stats, reference_cfg, decode_seed=seed)

    out_dir = run_dir / "h2h_final"

    def upload_orientation(_orientation: int) -> None:
        if uploader is not None:
            uploader.upload_tree(out_dir, base=run_dir)

    try:
        records = run_h2h(
            build_self,
            build_reference,
            name_a=self_label,
            name_b=reference_label,
            n_configs=cfg.final_h2h_n_configs,
            out_dir=out_dir,
            max_frames=cfg.eval_max_frames,
            max_parallel=cfg.final_h2h_max_parallel,
            seed=cfg.eval_seed,
            meta={
                "models": {
                    self_label: {
                        "experiment": str(Path(__file__)),
                        "checkpoint": str(run_dir / "final.pt"),
                        "step": cfg.max_steps,
                        "head_offsets": list(cfg.head_offsets),
                        "exec_horizon": cfg.exec_horizon,
                    },
                    reference_label: {
                        "experiment": "experiments/026_temporal_mtp.py",
                        "checkpoint": str(reference),
                        "checkpoint_sha256": cfg.final_h2h_reference_sha256,
                        "step": int(reference_state["step"]),
                        "head_offsets": list(reference_cfg.head_offsets),
                        "exec_horizon": reference_cfg.exec_horizon,
                    },
                }
            },
            on_orientation_done=upload_orientation,
        )
    finally:
        if uploader is not None:
            uploader.upload_tree(out_dir, base=run_dir)
    summary = summarize_paired(records, focal_model=self_label)
    print(summary.format_table(), flush=True)
    values = summary.as_dict()
    if wandb.run is not None:
        wandb.run.summary["h2h_final"] = values
    return values


def train(
    cfg: TrainConfig,
    stats: dict[str, FeatureStats],
    *,
    comment: str = "",
    resume_run: str | None = None,
    resume_state: dict | None = None,
) -> None:
    validate_config(cfg)
    run_name = resume_run or make_run_name(Path(__file__).stem, model_tag(cfg), cfg.data_root, comment)
    run_dir, replay_dir = setup_run_dir(run_name)
    # Resolve and authenticate the opponent before allocating the training
    # model or opening external logging/upload resources.
    h2h_reference = resolve_h2h_reference(cfg, run_dir)
    uploader = BackgroundUploader(run_name) if cfg.push_to_r2 else None
    wandb.init(
        project="hal",
        name=run_name,
        id=None if resume_state is None else resume_state.get("wandb_id"),
        resume="allow" if resume_state is not None else None,
        tags=["gpt", "temporal-mtp", "game-state-flow", "029"],
        config=asdict(cfg),
    )
    if wandb.run is not None:
        wandb.define_metric("eval/net_stock_lcb", step_metric="global_step")
        wandb.define_metric("eval/net_dmg_lcb", step_metric="global_step")
        if cfg.wandb_log_code:
            log_wandb_code(wandb.run)
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high" if cfg.allow_tf32 else "highest")
    model = TrainingModel(cfg).to(DEVICE)
    policy = model.policy
    counts = subsystem_parameter_counts(model)
    if wandb.run is not None:
        for name, value in counts.items():
            wandb.run.summary[f"parameters/{name}"] = value
        wandb.run.summary["parameters/total"] = sum(parameter.numel() for parameter in model.parameters())
    optimizer = make_optimizer(model, cfg)
    scheduler = LambdaLR(optimizer, lr_schedule(cfg))
    start_step = 0
    if resume_state is not None:
        model.load_state_dict(resume_state["model"])
        optimizer.load_state_dict(resume_state["opt"])
        scheduler.load_state_dict(resume_state["sched"])
        start_step = int(resume_state["step"]) + 1

    # Compilation stays in dedicated callables.  Evaluation therefore never
    # strips or mutates compiled methods and can build its own static buckets.
    def trunk_fn(features, pad, actions):
        return policy(features, pad, actions)

    temporal_fn: Callable = policy.temporal.teacher_forced_nll
    if DEVICE == "cuda" and cfg.compile_trunk:
        trunk_fn = torch.compile(trunk_fn, dynamic=False)
    if DEVICE == "cuda" and cfg.compile_temporal:
        temporal_fn = torch.compile(temporal_fn, dynamic=False)

    train_loader, val_cache = _make_loaders(cfg, stats)
    iterator = iter(train_loader)
    copy_stream = torch.cuda.Stream() if DEVICE == "cuda" else None
    run_started = time.monotonic()
    inference_prewarm: OverlappedInference | None = None
    inference_prewarm_started = False
    model.train()
    try:
        for step in range(start_step, cfg.max_steps):
            if DEVICE == "cuda":
                torch.cuda.reset_peak_memory_stats()
            loader_started = time.monotonic()
            cpu_batches: list[StateBatch] = []
            for _ in range(cfg.grad_accum_steps):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(train_loader)
                    batch = next(iterator)
                validate_batch_geometry(batch, cfg, micro_batch_size(cfg))
                cpu_batches.append(batch)
            loader_wait = time.monotonic() - loader_started
            valid_prefixes = sum(int((cfg.L_ctx - batch.batch.context.ctx_pad).sum()) for batch in cpu_batches)
            if valid_prefixes <= 0:
                raise RuntimeError("training accumulation contains no valid context prefixes")
            optimizer.zero_grad()
            nll_sum = torch.zeros(len(cfg.head_offsets), N_GROUPS, device=DEVICE)
            state_continuous_sum = torch.zeros(len(cfg.state_offsets), device=DEVICE)
            state_categorical_sum = torch.zeros_like(state_continuous_sum)
            state_samples = 0
            n_prefixes = 0
            with profile("step") as stopwatch:
                for batch in device_batches(cpu_batches, DEVICE, copy_stream):
                    train_batch = batch.batch
                    history, targets, valid = prepared_targets(policy, train_batch)
                    with amp_context(cfg, DEVICE):
                        hidden = trunk_fn(train_batch.context.features, train_batch.context.ctx_pad, history)
                        dense_nll = temporal_fn(hidden, history, targets)
                        parts = ActionLoss(nll=dense_nll[valid], targets=targets[valid])
                        joint_nll = parts.nll.sum(dim=-1)
                        primary = joint_nll[:, :4].sum() / (valid_prefixes * 4)
                        auxiliary = joint_nll[:, 4:].sum() / (valid_prefixes * (len(cfg.head_offsets) - 4))
                        state = state_loss(model, batch, hidden, cfg)
                        loss = (
                            primary
                            + cfg.aux_loss_weight * auxiliary
                            + state_weight(step, cfg) * state.loss / cfg.grad_accum_steps
                        )
                    loss.backward()
                    nll_sum += parts.nll.detach().sum(dim=0)
                    count = train_batch.target.shape[0]
                    state_continuous_sum += state.continuous_by_offset * count
                    state_categorical_sum += state.categorical_by_offset * count
                    state_samples += count
                    n_prefixes += parts.nll.shape[0]
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError(f"step {step}: non-finite gradient norm {gradient_norm}")
                optimizer.step()
                scheduler.step()
                if DEVICE == "cuda":
                    torch.cuda.synchronize()
            metrics = nll_mean_metrics((nll_sum / n_prefixes).cpu(), cfg.head_offsets)
            log = {
                "global_step": step,
                "samples": (step + 1) * cfg.batch_size,
                **{f"train/{name}": value for name, value in metrics.items()},
                "train/state_loss_weight": state_weight(step, cfg),
                "train/grad_norm": float(gradient_norm),
                "throughput/step_s": stopwatch.elapsed,
                "throughput/loader_wait_s": loader_wait,
                "throughput/samples_per_s": cfg.batch_size / stopwatch.elapsed,
                "throughput/samples_per_wall_s": cfg.batch_size / (stopwatch.elapsed + loader_wait),
                "throughput/prefixes_per_s": n_prefixes / stopwatch.elapsed,
                "lr/muon": next(group["lr"] for group in optimizer.param_groups if group["use_muon"]),
                "lr/adam": next(group["lr"] for group in optimizer.param_groups if not group["use_muon"]),
            }
            for index, offset in enumerate(cfg.state_offsets):
                log[f"train/state_flow_mse_o{offset:02d}"] = float(state_continuous_sum[index] / state_samples)
                log[f"train/state_cat_nll_o{offset:02d}"] = float(state_categorical_sum[index] / state_samples)
            if DEVICE == "cuda":
                log["hardware/peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 2**30
                log["hardware/peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30
            wandb.log(log)
            if step < 10 or step % 50 == 0:
                print(
                    f"[t+{time.monotonic() - run_started:.0f}s] step {step}: "
                    f"{metrics['loss']:.3f} bits objective, {cfg.batch_size / stopwatch.elapsed:.0f} samples/s",
                    flush=True,
                )
            # Let the first training step claim its compiled graphs before starting
            # the independent inference compile.  From here the CPU compiler and its
            # dedicated CUDA stream can overlap the many steps before evaluation.
            if not inference_prewarm_started:
                inference_prewarm = start_inference_prewarm(policy, cfg)
                inference_prewarm_started = True
            val_due = cfg.val_every > 0 and step > 0 and step % cfg.val_every == 0
            eval_due = cfg.eval_every > 0 and step > 0 and step % cfg.eval_every == 0
            ckpt_due = cfg.ckpt_every > 0 and step > 0 and step % cfg.ckpt_every == 0
            checkpoint_path = run_dir / "latest.pt"
            if val_due or eval_due or ckpt_due:
                save_checkpoint(
                    checkpoint_path,
                    step=step,
                    model=model,
                    opt=optimizer,
                    sched=scheduler,
                    cfg=asdict(cfg),
                    wandb_id=None if wandb.run is None else wandb.run.id,
                    uploader=uploader,
                )
            if val_due:
                values = combined_val_metrics(model, val_cache, cfg)
                values.update(gradient_interaction(model, val_cache[0], cfg))
                wandb.log({"global_step": step, **{f"val/{name}": value for name, value in values.items()}})
            if eval_due:
                eval_model, eval_inference = (
                    (policy, None) if inference_prewarm is None else inference_prewarm.prepare(policy)
                )
                values = eval_vs_cpu(
                    eval_model,
                    stats,
                    cfg,
                    n_matchups=cfg.eval_n_matchups,
                    replay_dir=replay_dir / f"step_{step:06d}",
                    checkpoint_sha256=_checkpoint_sha256(checkpoint_path),
                    inference=eval_inference,
                )
                if inference_prewarm is not None:
                    values.update(inference_prewarm.metrics())
                wandb.log({"global_step": step, **{f"eval/{name}": value for name, value in values.items()}})

        if not inference_prewarm_started:
            inference_prewarm = start_inference_prewarm(policy, cfg)
            inference_prewarm_started = True
        final_path = run_dir / "final.pt"
        save_checkpoint(
            final_path,
            step=cfg.max_steps,
            model=model,
            opt=optimizer,
            sched=scheduler,
            cfg=asdict(cfg),
            wandb_id=None if wandb.run is None else wandb.run.id,
            uploader=uploader,
        )
        checkpoint_sha = _checkpoint_sha256(final_path)
        final_val = combined_val_metrics(model, val_cache, cfg)
        wandb.log({"global_step": cfg.max_steps, **{f"val/{name}": value for name, value in final_val.items()}})
        eval_model, eval_inference = (policy, None) if inference_prewarm is None else inference_prewarm.prepare(policy)
        final_eval = eval_vs_cpu(
            eval_model,
            stats,
            cfg,
            n_matchups=cfg.final_eval_n_matchups,
            replay_dir=replay_dir / "final",
            checkpoint_sha256=checkpoint_sha,
            inference=eval_inference,
        )
        if inference_prewarm is not None:
            final_eval.update(inference_prewarm.metrics())
        wandb.log({"global_step": cfg.max_steps, **{f"eval/{name}": value for name, value in final_eval.items()}})
        stride6 = eval_vs_cpu(
            eval_model,
            stats,
            cfg,
            n_matchups=cfg.final_diag_n_matchups,
            replay_dir=replay_dir / "final_s6",
            exec_horizon=cfg.final_diag_exec_horizon,
            checkpoint_sha256=checkpoint_sha,
            inference=eval_inference,
        )
        if inference_prewarm is not None:
            stride6.update(inference_prewarm.metrics())
        wandb.log({"global_step": cfg.max_steps, **{f"eval_s6/{name}": value for name, value in stride6.items()}})
        model.state_expert.to("cpu")
        # The auxiliary expert and optimizer state are training-only. Release
        # them before loading the pinned 026 opponent for the final H2H games.
        del optimizer
        del scheduler
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        final_h2h(
            eval_model,
            stats,
            cfg,
            run_dir=run_dir,
            reference=h2h_reference,
            uploader=uploader,
            inference=eval_inference,
        )
    finally:
        if inference_prewarm is not None:
            inference_prewarm.close()
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
    "temporal_ff_dim",
    "group_head_dim",
    "action_embed_dim",
    "offset_embed_dim",
    "group_order",
    "observation_bundle",
    "state_offsets",
    "state_d_model",
    "state_layers",
    "state_heads",
    "state_ff_dim",
    "state_action_dim",
    "state_time_dim",
}


def config_from_state(values: dict) -> TrainConfig:
    missing = _CHECKPOINT_ARCH_FIELDS - values.keys()
    if missing:
        raise ValueError(f"checkpoint is not an experiment-029 architecture; missing {sorted(missing)}")
    known = {item.name for item in fields(TrainConfig)}
    return TrainConfig(**{name: value for name, value in values.items() if name in known})


def load_checkpoint(path: str, *, device: str = DEVICE) -> tuple[GPT, TrainConfig, dict[str, FeatureStats], dict]:
    state = torch.load(path, map_location=device, weights_only=False)
    cfg = config_from_state(state["cfg"])
    validate_config(cfg)
    model = GPT(cfg).to(device)
    prefix = "policy."
    policy_state = {
        name.removeprefix(prefix): value for name, value in state["model"].items() if name.startswith(prefix)
    }
    if not policy_state:
        raise ValueError("experiment-029 checkpoint contains no policy.* parameters")
    model.load_state_dict(policy_state, strict=True)
    model.eval()
    stats = load_consolidated_stats(Path(cfg.data_root) / "stats.json")
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
    print(f"[eval] step={state['step']} horizon={horizon}: {values}", flush=True)
    return values


def benchmark_self_play_checkpoint(
    path: str,
    *,
    n_matches: int = 12,
    max_frames: int = 14_400,
    eager: bool = False,
    instant_match_restart: bool = False,
) -> dict[str, float]:
    """Run one fixed wave with the checkpoint controlling both ports.

    By default, each Dolphin boot plays one normal head-to-head match, and
    ``max_frames`` is its cap.  Instant restart is available as a stress mode.
    The policy sees ``2 * n_matches`` slots while all matches are live.
    """
    if n_matches < 1:
        raise ValueError(f"n_matches must be >= 1, got {n_matches}")
    if max_frames < 2:
        raise ValueError(f"max_frames must be >= 2, got {max_frames}")
    model, cfg, stats, state = load_checkpoint(path)
    if eager:
        cfg = replace(cfg, inference_mode="eager")
    model.eval()
    horizon = cfg.exec_horizon
    real_rows = 2 * n_matches
    inference_bucket = covering_power_of_two(real_rows)
    # Match the background inference replica used during training. CUDA graphs
    # cannot safely own the trunk's mutable rotary cache across graph partitions.
    inference = BF16Inference(model, cfg, compile_mode="default", compiled_buckets=(inference_bucket,))
    telemetry = DecodeTelemetry()

    compile_started = time.perf_counter()
    context = synthetic_context(cfg, inference_bucket, next(model.parameters()).device)
    inference.decode(context, horizon)
    inference.decode(context, horizon)
    if next(model.parameters()).device.type == "cuda":
        torch.cuda.synchronize(next(model.parameters()).device)
    compile_seconds = time.perf_counter() - compile_started

    configs = mirrored_configs(n_matches)
    matches = [
        VecMatch(
            matchup=Matchup(
                stage=config.stage,
                players=(
                    PlayerSetup(port=1, character=config.character_port_1, cpu_level=0),
                    PlayerSetup(port=2, character=config.character_port_2, cpu_level=0),
                ),
            ),
            model_ports=(1, 2),
        )
        for config in configs
    ]
    checkpoint = Path(path).resolve()
    mode = "instant_restart" if instant_match_restart else "single_match"
    base_name = f"self_play_benchmark_{n_matches}x{max_frames}_{mode}"
    run_number = 1
    while True:
        suffix = "" if run_number == 1 else f"_run{run_number:02d}"
        out_dir = checkpoint.parent / f"{base_name}{suffix}"
        try:
            out_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            run_number += 1
    policy_index = itertools.count()

    def factory() -> RecedingHorizon:
        return make_policy(
            model,
            stats,
            cfg,
            decode_seed=cfg.eval_seed + next(policy_index),
            inference=inference,
            telemetry=telemetry,
        )

    rollout_started = time.perf_counter()
    process_telemetry = ProcessVecTelemetry()
    boots = run_matches_vec(
        default_session_cfg(out_dir / "replays", instant_match_restart=instant_match_restart),
        matches,
        factory,
        max_frames=max_frames,
        # Explicit overrides are powers of two. This covers all n_matches in
        # one wave while the active worker count remains exactly n_matches.
        max_parallel=covering_power_of_two(n_matches),
        start_retries=0,
        process_telemetry=process_telemetry,
    )
    rollout_seconds = time.perf_counter() - rollout_started
    completed_boots = sum(bool(boot) for boot in boots)
    captured_frames = sum(len(trajectory) for boot in boots for trajectory in boot)
    metrics = {
        "checkpoint_step": float(state["step"]),
        "workers": float(n_matches),
        "model_slots": float(real_rows),
        "max_frames_per_worker": float(max_frames),
        "completed_workers": float(completed_boots),
        "captured_frames": float(captured_frames),
        "inference_bucket": float(inference_bucket),
        "compile_seconds": compile_seconds,
        "rollout_seconds": rollout_seconds,
        "aggregate_fps": captured_frames / max(rollout_seconds, 1e-12),
        "wall_lockstep_fps": captured_frames / max(completed_boots, 1) / max(rollout_seconds, 1e-12),
        **telemetry.metrics(),
        **process_telemetry.metrics(),
    }
    payload = {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _checkpoint_sha256(checkpoint),
        "instant_match_restart": instant_match_restart,
        "metrics": metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return metrics


def synthetic_context(cfg: TrainConfig, batch_size: int, device: torch.device) -> Context:
    features: dict[str, Tensor] = {}
    floats = FLOAT_FEATURES if cfg.observation_bundle == "base" else FLOAT_FEATURES + _V6_FLOATS
    for prefix in _PLAYER_PREFIXES:
        for name in floats:
            features[f"{prefix}_{name}"] = torch.zeros(batch_size, cfg.L_ctx, device=device)
            features[f"{prefix}_{name}_mask"] = torch.zeros(batch_size, cfg.L_ctx, device=device)
        for name in CAT_FEATURES:
            features[f"{prefix}_{name}"] = torch.zeros(batch_size, cfg.L_ctx, dtype=torch.long, device=device)
        if cfg.observation_bundle == "v6_lean":
            for name in _V6_CATS:
                features[f"{prefix}_{name}"] = torch.zeros(batch_size, cfg.L_ctx, dtype=torch.long, device=device)
            features[f"{prefix}_{_CHARACTER_LIVE}"] = torch.zeros(
                batch_size, cfg.L_ctx, dtype=torch.long, device=device
            )
    for name in ACTION_CHANNELS:
        features[f"ego_{name}"] = torch.zeros(batch_size, cfg.L_ctx, device=device)
    features["ego_character"] = torch.zeros(batch_size, cfg.L_ctx, dtype=torch.long, device=device)
    features["opp_character"] = torch.zeros(batch_size, cfg.L_ctx, dtype=torch.long, device=device)
    features["stage"] = torch.zeros(batch_size, cfg.L_ctx, dtype=torch.long, device=device)
    if cfg.observation_bundle == "v6_lean":
        for name in SPATIAL_COLUMNS_LEAN:
            features[name] = torch.zeros(batch_size, cfg.L_ctx, device=device)
    return Context(
        features=features,
        ctx_pad=torch.zeros(batch_size, dtype=torch.long, device=device),
        slot_ids=torch.arange(batch_size, dtype=torch.long, device=device),
        reset=torch.ones(batch_size, dtype=torch.bool, device=device),
    )


class OverlappedInference:
    """Precompile a persistent inference replica while ordinary training continues.

    CUDA graph capture is process-global enough to conflict with the trainer's per-step
    synchronization, so the overlapped engine deliberately uses Inductor's ``default``
    mode (compiled kernels, no CUDA graphs).  Keeping a replica separate from the
    training model prevents the background no-grad forwards from racing with optimizer
    updates; later checkpoints are copied into the replica's existing storages.
    """

    def __init__(self, source: GPT, cfg: TrainConfig) -> None:
        if next(source.parameters()).device.type != "cuda" or cfg.inference_mode != "compiled":
            raise ValueError("overlapped inference requires compiled CUDA evaluation")
        self.cfg = cfg
        self.model = GPT(cfg).to(next(source.parameters()).device).eval()
        self.model.load_state_dict(source.state_dict())
        self.inference = BF16Inference(self.model, cfg, compile_mode="default")
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference-prewarm")
        self._future: Future[float] = self._executor.submit(self._prewarm)
        self._reported = False
        self.compile_seconds = 0.0
        self.first_wait_seconds = 0.0

    @torch.no_grad()
    def _prewarm(self) -> float:
        device = next(self.model.parameters()).device
        stream = torch.cuda.Stream(device=device)
        started = time.perf_counter()
        with torch.cuda.device(device), torch.cuda.stream(stream):
            for bucket, horizon in _planned_inference_programs(self.cfg):
                ctx = synthetic_context(self.cfg, bucket, device)
                # First call traces and compiles; the second proves the program can
                # replay before an evaluation depends on it.
                self.inference.decode(ctx, horizon)
                self.inference.decode(ctx, horizon)
        stream.synchronize()
        return time.perf_counter() - started

    @torch.no_grad()
    def prepare(self, source: GPT) -> tuple[GPT, BF16Inference]:
        """Wait for compilation if needed, then install the source's latest weights."""
        wait_started = time.perf_counter()
        compile_seconds = self._future.result()
        wait_seconds = time.perf_counter() - wait_started
        self.compile_seconds = compile_seconds
        if not self._reported:
            self.first_wait_seconds = wait_seconds
        self.model.load_state_dict(source.state_dict())
        torch.cuda.synchronize(next(self.model.parameters()).device)
        if not self._reported:
            print(
                f"[inference] background prewarm took {compile_seconds:.1f}s; "
                f"first evaluation waited {wait_seconds:.3f}s",
                flush=True,
            )
            self._reported = True
        return self.model, self.inference

    def metrics(self) -> dict[str, float]:
        return {
            "inference_prewarm_seconds": self.compile_seconds,
            "inference_prewarm_first_eval_wait_seconds": self.first_wait_seconds,
        }

    def close(self) -> None:
        self._executor.shutdown(wait=True)


def start_inference_prewarm(model: GPT, cfg: TrainConfig) -> OverlappedInference | None:
    if next(model.parameters()).device.type != "cuda" or cfg.inference_mode != "compiled":
        return None
    print("[inference] compiling the fixed inference programs in the background", flush=True)
    return OverlappedInference(model, cfg)


def run_benchmark(cfg: TrainConfig, *, iterations: int = 20) -> dict[str, float]:
    validate_config(cfg)
    device = torch.device(DEVICE)
    model = GPT(cfg).to(device).eval()
    ctx = synthetic_context(cfg, min(32, micro_batch_size(cfg)), device)
    eager = BF16Inference(model, replace(cfg, inference_mode="eager"), compiled=False)
    compiled = BF16Inference(model, cfg)

    def measure(engine: BF16Inference, rows: int, horizon: int) -> float:
        selected = Context(
            features={name: value[:rows] for name, value in ctx.features.items()},
            ctx_pad=ctx.ctx_pad[:rows],
            slot_ids=ctx.slot_ids[:rows] if ctx.slot_ids is not None else None,
            reset=ctx.reset[:rows] if ctx.reset is not None else None,
        )
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
    for rows in (1, min(32, ctx.ctx_pad.shape[0])):
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
    benchmark: bool = False
    benchmark_iterations: int = 20


def main(args: Args) -> None:
    if args.benchmark:
        if args.eval is not None or args.resume is not None or args.self_play_eval is not None:
            raise SystemExit("--benchmark cannot be combined with --eval, --self-play-eval, or --resume")
        run_benchmark(args.cfg, iterations=args.benchmark_iterations)
        return
    selected = sum(value is not None for value in (args.eval, args.self_play_eval, args.resume))
    if selected > 1:
        raise SystemExit("pass only one of --eval, --self-play-eval, or --resume")
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
        benchmark_self_play_checkpoint(
            args.self_play_eval,
            n_matches=args.self_play_matches,
            max_frames=args.self_play_frames,
            eager=args.self_play_eager,
            instant_match_restart=args.self_play_instant_match_restart,
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
    stats = load_consolidated_stats(Path(cfg.data_root) / "stats.json")
    train(cfg, stats, comment=args.comment, resume_run=resume_run, resume_state=resume_state)


if __name__ == "__main__":
    main(tyro.cli(Args))
