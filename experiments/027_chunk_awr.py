"""Chunk-scope AWR on the 026 temporal-MTP stack.

The policy is experiment 026 unchanged.  The treatment weights the dense 1..4
joint NLL — the macro-action the evaluator executes — by a fitted chunk
advantage.  Three small heads read the detached trunk state: V(s) fits the
Monte-Carlo return, Qhat(s, chunk) fits the sampled 4-step advantage, and
U(s, chunk) fits the squared residual (the calibrated confidence).  The actor
weight uses Qhat: regression to the state mean removes segment luck, which is
the failure that made the 020 Monte-Carlo dose null.

Arms: ``awr_mode=off`` (matched control), ``sample`` (weight by the raw 4-step
advantage; implemented for diagnostics), ``critic`` (weight by Qhat).

Run:
    uv run experiments/027_chunk_awr.py --cfg.awr-mode off
    uv run experiments/027_chunk_awr.py --cfg.awr-mode critic
    uv run experiments/027_chunk_awr.py --eval runs/<run>/final.pt
    uv run experiments/027_chunk_awr.py --audit val
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
from collections.abc import Mapping
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from dataclasses import fields
from dataclasses import replace
from pathlib import Path
from typing import Literal

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
from hal.data.policy_schema import unpack_player_state
from hal.eval.cross_stage import BOOTSTRAP_RESAMPLES
from hal.eval.cross_stage import PRIOR_SWEEP_SEED_STAGE
from hal.eval.cross_stage import MatchRow
from hal.eval.cross_stage import sweep_vs_cpu_prior_with_rows
from hal.eval.cross_stage import vs_cpu_metrics
from hal.eval.harness import DEFAULT_START_RETRIES
from hal.eval.harness import default_session_cfg
from hal.eval.matchups import matchups_for_vs_cpu
from hal.training import scoring
from hal.training.checkpoints import BackgroundUploader
from hal.training.checkpoints import load_for_resume
from hal.training.checkpoints import save_checkpoint
from hal.training.closed_loop import RecedingHorizon
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
from hal.training.features import stack_actions
from hal.training.muon import SingleDeviceMuonWithAuxAdam
from hal.training.replay_reservoir import make_reservoir_loader
from hal.training.returns import AWR_RETURN_SUFFIX
from hal.training.returns import AWR_REWARD_SUFFIX
from hal.training.returns import is_terminal
from hal.training.returns import replay_reward_columns
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

    # Chunk-scope AWR.  ``off`` is the matched control; ``sample`` weights by the
    # raw 4-step advantage; ``critic`` weights by the fitted chunk advantage.
    awr_mode: Literal["off", "sample", "critic"] = "off"
    awr_gamma: float = 0.99827
    awr_damage_shaping: float = 0.01
    awr_win_reward: float = 0.5
    awr_horizon: int = 4
    critic_fit: Literal["mc", "td"] = "mc"
    awr_beta: float | None = None
    awr_beta_grid: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4)
    awr_weight_max: float = 5.0
    value_loss_weight: float = 1.0
    qhat_loss_weight: float = 1.0
    variance_loss_weight: float = 1.0
    critic_hidden_dim: int = 256
    critic_action_embed_dim: int = 16
    critic_lr: float = 8.5e-4
    critic_weight_decay: float = 0.0
    critic_grad_clip: float = 1.0
    awr_warmup_steps: int = 2048
    gate_min_ess_frame: float = 0.2
    gate_min_ess_window: float = 0.2
    gate_max_clip_frac: float = 0.2
    gate_min_action_sensitivity: float = 1.05

    seed: int = 0
    eval_seed: int = 0
    batch_size: int = 512
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
    val_every: int = 1024
    val_n_samples: int = 1192
    val_batch_size: int = 128
    ckpt_every: int = 1024
    eval_every: int = 4096
    eval_max_frames: int = 7200
    eval_n_matchups: int = 32
    final_eval_n_matchups: int = 96
    final_diag_n_matchups: int = 32
    eval_max_parallel: int = 32

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
    if cfg.batch_size % cfg.grad_accum_steps:
        raise ValueError("batch_size must be divisible by grad_accum_steps")
    if cfg.exec_horizon not in (4, 6) or cfg.final_diag_exec_horizon != 6:
        raise ValueError("execution horizons are restricted to the unrolled four/six-frame decoders")
    if cfg.decode_temp != 1.0:
        raise ValueError("experiment 026 freezes sampling temperature at 1")
    if cfg.inference_mode not in ("compiled", "eager"):
        raise ValueError("inference_mode must be 'compiled' or 'eager'")
    if cfg.inference_buckets != (1, 2, 4, 8, 16, 32):
        raise ValueError("inference_buckets are frozen to (1,2,4,8,16,32)")
    if cfg.observation_bundle not in ("base", "v6_lean"):
        raise ValueError("observation_bundle must be 'base' or 'v6_lean'")
    if not math.isfinite(cfg.aux_loss_weight) or cfg.aux_loss_weight < 0:
        raise ValueError("aux_loss_weight must be finite and non-negative")
    if cfg.amp_dtype not in ("bfloat16", "float32"):
        raise ValueError("amp_dtype must be bfloat16 or float32")
    if cfg.reservoir_capacity < 2 * micro_batch_size(cfg):
        raise ValueError("reservoir_capacity must be at least twice the micro-batch size")
    if cfg.awr_mode not in ("off", "sample", "critic"):
        raise ValueError(f"awr_mode must be off, sample, or critic, got {cfg.awr_mode!r}")
    if cfg.awr_mode != "off":
        if cfg.grad_accum_steps != 1:
            raise ValueError("AWR weight normalization needs the whole optimizer batch; grad_accum_steps must be 1")
        if cfg.awr_horizon != cfg.exec_horizon:
            raise ValueError(f"awr_horizon={cfg.awr_horizon} must equal exec_horizon={cfg.exec_horizon}")
        if cfg.awr_horizon >= cfg.sample_chunk_length:
            raise ValueError("awr_horizon must leave at least one chunk frame beyond the reward slice")
        if not 0 < cfg.awr_warmup_steps < cfg.max_steps:
            raise ValueError(f"awr_warmup_steps={cfg.awr_warmup_steps} must be inside (0, max_steps)")
        if cfg.awr_beta is not None and (not math.isfinite(cfg.awr_beta) or cfg.awr_beta <= 0):
            raise ValueError(f"awr_beta must be a positive finite float or None, got {cfg.awr_beta!r}")
        grid = cfg.awr_beta_grid
        if grid != tuple(sorted(set(grid))) or not grid or any(b <= 0 for b in grid):
            raise ValueError(f"awr_beta_grid must be sorted, unique, and positive, got {grid}")
        if not 0 < cfg.awr_gamma < 1:
            raise ValueError(f"awr_gamma must be in (0, 1), got {cfg.awr_gamma}")
        if cfg.awr_weight_max <= 1:
            raise ValueError(f"awr_weight_max must exceed 1, got {cfg.awr_weight_max}")
        if cfg.critic_fit not in ("mc", "td"):
            raise ValueError(f"critic_fit must be mc or td, got {cfg.critic_fit!r}")
        for name, bound in (
            ("gate_min_ess_frame", cfg.gate_min_ess_frame),
            ("gate_min_ess_window", cfg.gate_min_ess_window),
            ("gate_max_clip_frac", cfg.gate_max_clip_frac),
        ):
            if not 0 <= bound <= 1:
                raise ValueError(f"{name} must be in [0, 1], got {bound}")


def micro_batch_size(cfg: TrainConfig) -> int:
    return cfg.batch_size // cfg.grad_accum_steps


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


@dataclass(frozen=True, slots=True)
class CriticOut:
    value: Tensor  # [B, L_ctx]
    qhat: Tensor  # [B, L_ctx]
    sigma2: Tensor  # [B, L_ctx]


class Critic(nn.Module):
    """V(s), Qhat(s, chunk), and the residual variance U(s, chunk).

    Every head reads the DETACHED trunk state: the critic must not become a second
    representation loss on the policy.  The chunk encoder owns its embeddings; the
    actor's codec tables stay policy-only."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.horizon = cfg.awr_horizon
        self.value = nn.Linear(cfg.d_model, 1)
        self.action_embeddings = nn.ModuleDict(
            {name: nn.Embedding(GROUP_VOCABS[GROUP_INDEX[name]], cfg.critic_action_embed_dim) for name in GROUP_NAMES}
        )
        chunk_width = cfg.awr_horizon * N_GROUPS * cfg.critic_action_embed_dim
        self.qhat_up = nn.Linear(cfg.d_model + chunk_width, cfg.critic_hidden_dim)
        self.qhat_down = nn.Linear(cfg.critic_hidden_dim, 1)
        self.variance_up = nn.Linear(cfg.d_model + chunk_width, cfg.critic_hidden_dim)
        self.variance_down = nn.Linear(cfg.critic_hidden_dim, 1)

    def embed_chunk(self, chunk: Tensor) -> Tensor:
        if chunk.shape[-2:] != (self.horizon, N_GROUPS):
            raise ValueError(f"chunk must end in {(self.horizon, N_GROUPS)}, got {tuple(chunk.shape)}")
        parts = [
            self.action_embeddings[name](chunk[..., GROUP_INDEX[name]]) for name in GROUP_NAMES
        ]  # each [..., horizon, e]
        return torch.cat(parts, dim=-1).flatten(-2)

    def forward(self, hidden: Tensor, chunk: Tensor) -> CriticOut:
        state = decoder_rmsnorm(hidden.detach())
        value = self.value(state).squeeze(-1).float()
        features = torch.cat((state, self.embed_chunk(chunk)), dim=-1)
        qhat = self.qhat_down(F.silu(self.qhat_up(features))).squeeze(-1).float()
        sigma2 = F.softplus(self.variance_down(F.silu(self.variance_up(features)))).squeeze(-1).float()
        return CriticOut(value=value, qhat=qhat, sigma2=sigma2)


@dataclass(frozen=True, slots=True)
class CriticLosses:
    value: Tensor
    qhat: Tensor
    variance: Tensor
    advantage: Tensor  # [B, L_ctx], detached
    eligible: Tensor  # [B, L_ctx]
    weight_source: Tensor  # [B, L_ctx], detached
    out: CriticOut
    value_rows: int


def critic_losses(
    model: GPT, hidden: Tensor, awr: AWRBatch, targets: Tensor, valid: Tensor, cfg: TrainConfig
) -> CriticLosses:
    """All critic terms for one batch.  Nothing here carries a policy gradient."""
    if model.critic is None:
        raise ValueError("critic_losses needs a model with awr_mode != 'off'")
    chunk = targets[..., : cfg.awr_horizon, :]
    out = model.critic(hidden, chunk)
    advantage, eligible = chunk_advantage(out.value, awr.rewards, valid, gamma=cfg.awr_gamma, horizon=cfg.awr_horizon)
    zero = hidden.new_zeros(())
    if cfg.critic_fit == "mc":
        value_mask = valid & awr.terminal[:, None]
        value_loss = F.mse_loss(out.value[value_mask], awr.returns[value_mask]) if value_mask.any() else zero
        value_rows = int(value_mask.sum())
    else:
        successor = valid[:, 1:] & valid[:, :-1]
        target = awr.rewards[:, : valid.shape[1] - 1] + cfg.awr_gamma * out.value[:, 1:].detach()
        value_loss = F.mse_loss(out.value[:, :-1][successor], target[successor]) if successor.any() else zero
        value_rows = int(successor.sum())
    qhat_loss = F.huber_loss(out.qhat[eligible], advantage[eligible], delta=1.0) if eligible.any() else zero
    residual = (advantage - out.qhat.detach()).square()
    variance_loss = F.mse_loss(out.sigma2[eligible], residual[eligible]) if eligible.any() else zero
    weight_source = out.qhat.detach() if cfg.awr_mode == "critic" else advantage
    return CriticLosses(
        value=value_loss,
        qhat=qhat_loss,
        variance=variance_loss,
        advantage=advantage,
        eligible=eligible,
        weight_source=weight_source,
        out=out,
        value_rows=value_rows,
    )


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
        # The critic is constructed LAST so every policy parameter takes the same
        # initialization draw as the awr_mode="off" control at the same seed.
        self.critic = Critic(cfg) if cfg.awr_mode != "off" else None

    def policy_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return [(name, parameter) for name, parameter in self.named_parameters() if not name.startswith("critic.")]

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


def objective(parts: ActionLoss, aux_loss_weight: float = 1.0, weight: Tensor | None = None) -> Tensor:
    """Primary dense-four joint NLL plus the mean auxiliary joint NLL.

    ``weight`` is the per-row AWR weight over the EXECUTED macro-action: one scalar
    multiplies the whole dense 1..4 joint NLL of its row.  The auxiliary offsets
    stay unweighted behavior cloning, so ``weight=None`` (or all ones) reproduces
    the control objective exactly."""
    joint = parts.nll.sum(dim=-1)
    if weight is None:
        primary = joint[:, :4].mean()
    else:
        if weight.shape != joint.shape[:1]:
            raise ValueError(f"weight must be per-row {tuple(joint.shape[:1])}, got {tuple(weight.shape)}")
        primary = (joint[:, :4] * weight[:, None].detach()).mean()
    auxiliary = joint[:, 4:].mean()
    return primary + aux_loss_weight * auxiliary


# %% AWR data path -------------------------------------------------------------

AWR_TERMINAL_COLUMN = "awr_terminal"
AWR_ANNOTATION_COLUMNS: tuple[str, ...] = (
    f"p1_{AWR_REWARD_SUFFIX}",
    f"p1_{AWR_RETURN_SUFFIX}",
    f"p2_{AWR_REWARD_SUFFIX}",
    f"p2_{AWR_RETURN_SUFFIX}",
    AWR_TERMINAL_COLUMN,
)
# Post-relabel columns the sampler must carry through windowing for the ego view.
AWR_PROJECTION = FeatureProjection(
    columns=BASE_ACTION_PROJECTION.columns
    | {f"ego_{AWR_RETURN_SUFFIX}", f"ego_{AWR_REWARD_SUFFIX}", AWR_TERMINAL_COLUMN},
    derive_spatial=BASE_ACTION_PROJECTION.derive_spatial,
)


def annotate_decoded_replay(
    sample: Mapping[str, np.ndarray], *, gamma: float, damage_shaping: float, win_reward: float
) -> dict[str, np.ndarray]:
    """Reward, return, and terminal columns for one decoded replay row."""
    out = replay_reward_columns(sample, gamma=gamma, damage_shaping=damage_shaping, win_reward=win_reward)
    frames = out[f"p1_{AWR_RETURN_SUFFIX}"].shape[0]
    out[AWR_TERMINAL_COLUMN] = np.full(frames, float(is_terminal(sample)), dtype=np.float32)
    return out


def annotate_compact_replay(
    compact: Mapping[str, object], *, gamma: float, damage_shaping: float, win_reward: float
) -> dict[str, np.ndarray]:
    """The same columns straight from one compact MDS row (no full decode)."""
    sample: dict[str, np.ndarray] = {}
    for port in ("p1", "p2"):
        sample[f"{port}_percent"] = np.asarray(compact[f"{port}_percent"], dtype=np.float32)
        sample[f"{port}_stock"] = unpack_player_state(np.asarray(compact[f"{port}_state"]))["stock"]
    return annotate_decoded_replay(sample, gamma=gamma, damage_shaping=damage_shaping, win_reward=win_reward)


def label_decoded_replay(sample: dict, *, gamma: float, damage_shaping: float, win_reward: float) -> dict:
    """``replay_transform`` for the decoded (validation) loader path."""
    return {
        **sample,
        **annotate_decoded_replay(sample, gamma=gamma, damage_shaping=damage_shaping, win_reward=win_reward),
    }


@dataclass(frozen=True, slots=True)
class AWRBatch:
    """One ``TrainBatch`` plus the ego reward slice, return labels, and terminal flag.

    ``returns[b, t]`` is ``G_{t+1}``: the discounted return that starts with the
    reward after the action the offset-1 head predicts.  ``rewards[b, k]`` is
    ``r_{k+1}``, extended ``awr_horizon`` frames past the context so every context
    position owns a full 4-step reward sum."""

    batch: TrainBatch
    returns: Tensor
    rewards: Tensor
    terminal: Tensor

    def to(self, device: torch.device | str) -> AWRBatch:
        return AWRBatch(
            batch=self.batch.to(device),
            returns=self.returns.to(device, non_blocking=True),
            rewards=self.rewards.to(device, non_blocking=True),
            terminal=self.terminal.to(device, non_blocking=True),
        )

    def pin_memory(self) -> AWRBatch:
        return AWRBatch(
            batch=self.batch.pin_memory(),
            returns=self.returns.pin_memory(),
            rewards=self.rewards.pin_memory(),
            terminal=self.terminal.pin_memory(),
        )

    def tensors(self) -> tuple[Tensor, ...]:
        return (
            *self.batch.context.features.values(),
            self.batch.context.ctx_pad,
            self.batch.target,
            self.returns,
            self.rewards,
            self.terminal,
        )


def collate_awr_batch(windows: list[dict], batch: TrainBatch, *, L_ctx: int, horizon: int) -> AWRBatch:
    """Slice the window labels onto the one-frame-shifted grid the heads use."""
    returns = np.stack([np.asarray(window[f"ego_{AWR_RETURN_SUFFIX}"]) for window in windows])
    rewards = np.stack([np.asarray(window[f"ego_{AWR_REWARD_SUFFIX}"]) for window in windows])
    terminal = np.stack([np.asarray(window[AWR_TERMINAL_COLUMN])[-1] for window in windows])
    if returns.shape[1] < L_ctx + horizon + 1:
        raise ValueError(f"window carries {returns.shape[1]} frames; need at least {L_ctx + horizon + 1}")
    return AWRBatch(
        batch=batch,
        returns=torch.from_numpy(returns[:, 1 : L_ctx + 1].astype(np.float32)),
        rewards=torch.from_numpy(rewards[:, 1 : L_ctx + horizon + 1].astype(np.float32)),
        terminal=torch.from_numpy(terminal > 0.5),
    )


# %% Advantage and weights -----------------------------------------------------


def chunk_advantage(
    value_grid: Tensor, rewards: Tensor, valid: Tensor, *, gamma: float, horizon: int
) -> tuple[Tensor, Tensor]:
    """4-step bootstrapped advantage and its eligibility mask, both detached.

    ``A_t = sum_k gamma^k r_{t+1+k} + gamma^H V(s_{t+H}) - V(s_t)``.  The last
    ``horizon`` context positions have no in-context bootstrap state and are not
    eligible; they keep behavior-cloning weight 1."""
    batch, length = value_grid.shape
    if rewards.shape != (batch, length + horizon):
        raise ValueError(f"rewards must be {(batch, length + horizon)}, got {tuple(rewards.shape)}")
    if valid.shape != (batch, length):
        raise ValueError(f"valid must be {(batch, length)}, got {tuple(valid.shape)}")
    value = value_grid.detach().float()
    reward = rewards.detach().float()
    discount = torch.tensor([gamma**k for k in range(horizon)], device=reward.device)
    reward_sum = sum(discount[k] * reward[:, k : k + length] for k in range(horizon))
    bootstrap = torch.cat([value[:, horizon:], value.new_zeros(batch, horizon)], dim=1)
    advantage = reward_sum + (gamma**horizon) * bootstrap - value
    positions = torch.arange(length, device=value.device)
    eligible = valid & (positions[None, :] < length - horizon)
    return advantage, eligible


def awr_weights(
    advantage: Tensor, eligible: Tensor, *, beta: float, weight_max: float
) -> tuple[Tensor, dict[str, float]]:
    """``w = exp(min(A / beta, log(w_max)))`` normalized to mean one over eligible rows.

    The clip happens on the LOG weight, before any exponentiation, and the mean-one
    normalization runs through logsumexp so a batch of large negative advantages can
    never underflow to ``0 / 0``.  Ineligible rows keep weight one and enter no
    statistic.  There is no second clip after normalization."""
    if advantage.requires_grad:
        raise ValueError("advantage must be detached before weighting")
    if advantage.shape != eligible.shape:
        raise ValueError(f"advantage {tuple(advantage.shape)} and eligible {tuple(eligible.shape)} differ")
    selected = advantage[eligible].float()
    weight = torch.ones_like(advantage, dtype=torch.float32)
    if selected.numel() == 0:
        return weight, {"ess_frame": 1.0, "w_raw_clip_frac": 0.0, "w_norm_max": 1.0, "w_norm_min": 1.0}
    if not torch.isfinite(selected).all():
        raise ValueError("advantage contains non-finite values")
    log_cap = math.log(weight_max)
    scaled = selected / beta
    q = torch.clamp(scaled, max=log_cap)
    log_mean = torch.logsumexp(q, dim=0) - math.log(selected.numel())
    normalized = torch.exp(q - log_mean)
    weight[eligible] = normalized
    stats = {
        "ess_frame": float(normalized.sum().square() / (normalized.numel() * normalized.square().sum())),
        "w_raw_clip_frac": float((scaled >= log_cap).float().mean()),
        "w_norm_max": float(normalized.max()),
        "w_norm_min": float(normalized.min()),
    }
    return weight, stats


def window_ess(raw_weight: Tensor, row_index: Tensor, n_rows: int) -> float:
    """ESS across per-window mean raw weights.

    Frames inside one replay window are correlated, so frame-level ESS alone is too
    optimistic; this is the second, stricter gate statistic."""
    if raw_weight.numel() == 0 or n_rows == 0:
        return 1.0
    sums = torch.zeros(n_rows, dtype=torch.float64)
    counts = torch.zeros(n_rows, dtype=torch.float64)
    sums.index_add_(0, row_index, raw_weight.double())
    counts.index_add_(0, row_index, torch.ones_like(raw_weight, dtype=torch.float64))
    present = counts > 0
    means = sums[present] / counts[present]
    if means.numel() == 0:
        return 1.0
    return float(means.sum().square() / (means.numel() * means.square().sum()))


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
    value_preds: list[Tensor] = []
    value_labels: list[Tensor] = []
    try:
        for cpu_item in batches:
            item = cpu_item.to(device)
            awr = item if isinstance(item, AWRBatch) else None
            batch = awr.batch if awr is not None else item
            history, targets, valid = prepared_targets(model, batch)
            if awr is not None and model.critic is not None:
                with amp_context(cfg, device):
                    critic_hidden = model(batch.context.features, batch.context.ctx_pad, history)
                    critic = critic_losses(model, critic_hidden, awr, targets, valid, cfg)
                mask = valid & awr.terminal[:, None]
                value_preds.append(critic.out.value[mask].float().cpu())
                value_labels.append(awr.returns[mask].float().cpu())
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
    if value_preds:
        preds = torch.cat(value_preds)
        labels = torch.cat(value_labels)
        out["value_mse"] = float((preds - labels).square().mean())
        out["value_bias"] = float((preds - labels).mean())
        out["value_corr"] = _correlation(preds, labels)
    return out


# %% Warm-up gate --------------------------------------------------------------


def beta_table(
    source: Tensor, rows: Tensor, n_rows: int, *, grid: tuple[float, ...], weight_max: float
) -> list[dict[str, float]]:
    """ESS and clip statistics for every candidate temperature, on held-out rows.

    ESS is scale-invariant, so the normalized weights serve both the frame-level
    and the window-level statistic."""
    table: list[dict[str, float]] = []
    eligible = torch.ones_like(source, dtype=torch.bool)
    for beta in grid:
        weight, stats = awr_weights(source, eligible, beta=beta, weight_max=weight_max)
        table.append(
            {
                "beta": float(beta),
                "ess_frame": stats["ess_frame"],
                "ess_window": window_ess(weight, rows, n_rows),
                "clip_frac": stats["w_raw_clip_frac"],
                "w_norm_max": stats["w_norm_max"],
            }
        )
    return table


def select_beta(table: list[dict[str, float]], cfg: TrainConfig) -> float | None:
    """The smallest temperature that passes both ESS floors and the clip ceiling."""
    for entry in table:
        if (
            entry["ess_frame"] >= cfg.gate_min_ess_frame
            and entry["ess_window"] >= cfg.gate_min_ess_window
            and entry["clip_frac"] <= cfg.gate_max_clip_frac
        ):
            return entry["beta"]
    return None


@dataclass(frozen=True, slots=True)
class GateResult:
    passed: bool
    beta: float | None
    value_corr: float
    qhat_corr: float
    action_sensitivity: float
    table: list[dict[str, float]]
    reasons: list[str]


def _correlation(a: Tensor, b: Tensor) -> float:
    if a.numel() < 2 or float(a.std()) == 0.0 or float(b.std()) == 0.0:
        return 0.0
    return float(torch.corrcoef(torch.stack((a, b)))[0, 1])


def run_activation_gate(
    model: GPT, batches: list[AWRBatch], cfg: TrainConfig, run_dir: Path, uploader: BackgroundUploader | None = None
) -> GateResult:
    """The pre-registered step-2048 gate, on held-out data.

    It verifies the critic before any weighted actor update: finite predictions,
    positive value and Qhat correlation, Qhat action sensitivity, and a passing
    temperature.  The artifact is written before the pass/fail decision so a failed
    run still leaves its evidence."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    values: list[Tensor] = []
    value_targets: list[Tensor] = []
    qhats: list[Tensor] = []
    advantages: list[Tensor] = []
    sources: list[Tensor] = []
    window_rows: list[Tensor] = []
    base_error = rolled_error = 0.0
    row_offset = 0
    try:
        with torch.no_grad():
            for cpu_batch in batches:
                awr = cpu_batch.to(device)
                history, targets, valid = prepared_targets(model, awr.batch)
                with amp_context(cfg, device):
                    hidden = model(awr.batch.context.features, awr.batch.context.ctx_pad, history)
                    losses = critic_losses(model, hidden, awr, targets, valid, cfg)
                    rolled = model.critic(hidden, targets[..., : cfg.awr_horizon, :].roll(1, dims=0))
                eligible = losses.eligible
                value_mask = valid & awr.terminal[:, None]
                values.append(losses.out.value[value_mask].cpu())
                value_targets.append(awr.returns[value_mask].cpu())
                qhats.append(losses.out.qhat[eligible].cpu())
                advantages.append(losses.advantage[eligible].cpu())
                sources.append(losses.weight_source[eligible].cpu())
                rows = torch.arange(valid.shape[0], device=device)[:, None].expand_as(valid)
                window_rows.append(rows[eligible].cpu() + row_offset)
                row_offset += valid.shape[0]
                base_error += float((losses.out.qhat[eligible] - losses.advantage[eligible]).square().sum())
                rolled_error += float((rolled.qhat[eligible] - losses.advantage[eligible]).square().sum())
    finally:
        model.train(was_training)
    source = torch.cat(sources)
    rows = torch.cat(window_rows)
    value_corr = _correlation(torch.cat(values), torch.cat(value_targets))
    qhat_corr = _correlation(torch.cat(qhats), torch.cat(advantages))
    sensitivity = rolled_error / max(base_error, 1e-12)
    grid = cfg.awr_beta_grid if cfg.awr_beta is None else (cfg.awr_beta,)
    table = beta_table(source, rows, row_offset, grid=grid, weight_max=cfg.awr_weight_max)
    beta = select_beta(table, cfg)
    reasons: list[str] = []
    if not torch.isfinite(source).all():
        reasons.append("non_finite_weight_source")
    if value_corr <= 0:
        reasons.append("value_correlation_not_positive")
    if cfg.awr_mode == "critic":
        if qhat_corr <= 0:
            reasons.append("qhat_correlation_not_positive")
        if sensitivity < cfg.gate_min_action_sensitivity:
            reasons.append("qhat_ignores_actions")
    if beta is None:
        reasons.append("no_temperature_passed_the_ess_and_clip_gates")
    result = GateResult(
        passed=not reasons,
        beta=beta if cfg.awr_beta is None else cfg.awr_beta,
        value_corr=value_corr,
        qhat_corr=qhat_corr,
        action_sensitivity=sensitivity,
        table=table,
        reasons=reasons,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact = run_dir / "awr_gate.json"
    artifact.write_text(json.dumps(asdict(result), sort_keys=True, default=float))
    if uploader is not None:
        uploader.upload(artifact)
    return result


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
    """Static-bucket trunk and unrolled dense-prefix decoders.

    CUDA builds compile one graph per (bucket, horizon).  Eager mode follows the
    identical padding and keyed-uniform path, which makes parity testable.
    """

    def __init__(self, model: GPT, cfg: TrainConfig, *, compiled: bool | None = None) -> None:
        self.model = model
        self.cfg = cfg
        self.buckets = tuple(cfg.inference_buckets)
        requested = cfg.inference_mode == "compiled" if compiled is None else compiled
        self.compiled = bool(requested and next(model.parameters()).device.type == "cuda")
        self._trunks: dict[int, Callable] = {}
        self._decoders: dict[tuple[int, int], Callable] = {}

    def _bucket(self, rows: int) -> int:
        try:
            return next(bucket for bucket in self.buckets if bucket >= rows)
        except StopIteration as exc:
            raise ValueError(f"inference batch {rows} exceeds largest bucket {self.buckets[-1]}") from exc

    def _trunk(self, bucket: int) -> Callable:
        if bucket not in self._trunks:

            def fn(features, pad, actions):
                return self.model(features, pad, actions)

            self._trunks[bucket] = torch.compile(fn, dynamic=False, mode="reduce-overhead") if self.compiled else fn
        return self._trunks[bucket]

    def _decoder(self, bucket: int, horizon: int) -> Callable:
        key = (bucket, horizon)
        if key not in self._decoders:
            offsets = self.model.head_offsets[:horizon]

            def fn(hidden, observed, uniforms):
                return self.model.temporal.sample_indices(hidden, observed, offsets, argmax=False, uniforms=uniforms)

            self._decoders[key] = torch.compile(fn, dynamic=False, mode="reduce-overhead") if self.compiled else fn
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
        if self.compiled:
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

    def record(self, *, rows: int, horizon: int, seconds: float) -> None:
        self.calls += 1
        self.rows += rows
        self.executed_frames += rows * horizon
        self.seconds += seconds

    def metrics(self) -> dict[str, float]:
        return {
            "decode_calls": float(self.calls),
            "decode_rows": float(self.rows),
            "decode_seconds": self.seconds,
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


@dataclass(frozen=True)
class DecodeSettings:
    """Frozen decode protocol: the h2h CLI reads and records these knobs."""

    temp: float = 1.0
    temps: tuple[float, ...] | None = None
    btn_support_min: int = 0
    min_p: float = 0.0
    click_trigger_fix: bool = False


def _decode_settings(model: GPT, cfg: TrainConfig) -> DecodeSettings:
    return DecodeSettings()


def make_policy(
    model: GPT,
    stats: dict[str, FeatureStats],
    cfg: TrainConfig,
    *,
    exec_horizon: int | None = None,
    decode_seed: int | None = None,
    decode_temp: float = 1.0,
    decode_temps: tuple[float, ...] | None = None,
    decode_btn_support_min: int = 0,
    decode_min_p: float = 0.0,
    decode_click_trigger_fix: bool = False,
    inference: BF16Inference | None = None,
    telemetry: DecodeTelemetry | None = None,
    device: str = DEVICE,
) -> RecedingHorizon:
    if decode_temp != 1.0:
        raise ValueError("experiment 027 freezes sampling temperature at 1")
    if decode_temps is not None or decode_btn_support_min != 0 or decode_min_p != 0.0 or decode_click_trigger_fix:
        raise ValueError("experiment 027 supports no decode knobs beyond the frozen defaults")
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
) -> EvalProtocol:
    pairs, egos, cpus, schedule_sha = assert_protocol_diversity(n_matchups)
    return EvalProtocol(
        n_matchups=n_matchups,
        max_parallel=min(n_matchups, cfg.eval_max_parallel, 32),
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
        checkpoint_sha256=checkpoint_sha256,
    )


def _write_eval_evidence(
    replay_dir: Path, rows: list[MatchRow], metrics: dict[str, float], protocol: EvalProtocol
) -> None:
    replay_dir.mkdir(parents=True, exist_ok=True)
    rows_payload = {
        "schema_version": 3,
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
) -> dict[str, float]:
    horizon = cfg.exec_horizon if exec_horizon is None else exec_horizon
    protocol = _eval_protocol(
        cfg, model, n_matchups=n_matchups, exec_horizon=horizon, checkpoint_sha256=checkpoint_sha256
    )
    if next(model.parameters()).device.type == "cuda" and protocol.inference_mode != "compiled":
        raise RuntimeError("official CUDA evaluation requires compiled BF16 inference")
    inference = BF16Inference(model, cfg)
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


def make_optimizer(model: GPT, cfg: TrainConfig) -> SingleDeviceMuonWithAuxAdam:
    """The POLICY optimizer.  Critic parameters train under their own AdamW so the
    policy optimizer state stays byte-identical to the control arm's."""
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
    for _, parameter in model.policy_parameters():
        if id(parameter) in muon_ids:
            continue
        (no_decay if parameter.ndim < 2 or id(parameter) in embedding_ids else decay).append(parameter)
    if len(muon) + len(decay) + len(no_decay) != len(model.policy_parameters()):
        raise RuntimeError("optimizer parameter partition is incomplete")
    adam = dict(betas=(0.9, 0.95), eps=1e-10, use_muon=False)
    return SingleDeviceMuonWithAuxAdam(
        [
            dict(params=muon, lr=cfg.muon_lr, momentum=0.95, weight_decay=cfg.weight_decay, use_muon=True),
            dict(params=decay, lr=cfg.adam_lr, weight_decay=cfg.weight_decay, **adam),
            dict(params=no_decay, lr=cfg.adam_lr, weight_decay=0.0, **adam),
        ]
    )


def make_critic_optimizer(model: GPT, cfg: TrainConfig) -> torch.optim.AdamW:
    if model.critic is None:
        raise ValueError("the control arm has no critic optimizer")
    return torch.optim.AdamW(
        model.critic.parameters(),
        lr=cfg.critic_lr,
        betas=(0.9, 0.95),
        eps=1e-10,
        weight_decay=cfg.critic_weight_decay,
    )


def subsystem_parameter_counts(model: GPT) -> dict[str, int]:
    groups = {
        "trunk": model.trunk,
        "observation": model.ctx_proj,
        "codec": model.codec,
        "temporal": model.temporal.blocks,
        "heads": nn.ModuleList([model.temporal.outputs, model.temporal.trunk_outputs]),
    }
    return {name: sum(parameter.numel() for parameter in module.parameters()) for name, module in groups.items()}


def model_tag(cfg: TrainConfig) -> str:
    return (
        f"awr027-{cfg.awr_mode}-d{cfg.d_model}-L{cfg.n_layers}-h{cfg.n_heads}-Lc{cfg.L_ctx}-"
        f"b{cfg.batch_size}-s{cfg.exec_horizon}-{cfg.observation_bundle}"
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


def validate_batch_geometry(batch: TrainBatch, cfg: TrainConfig, expected_batch_size: int | None = None) -> None:
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


def _slice_train_batch(batch: TrainBatch, rows: int) -> TrainBatch:
    return TrainBatch(
        context=Context(
            features={name: value[:rows] for name, value in batch.context.features.items()},
            ctx_pad=batch.context.ctx_pad[:rows],
        ),
        target=batch.target[:rows],
        replay_ids=None if batch.replay_ids is None else batch.replay_ids[:rows],
    )


def cache_validation(loader: Iterable, n_samples: int) -> list:
    batches: list = []
    count = 0
    for batch in loader:
        remaining = n_samples - count
        if remaining <= 0:
            break
        rows = (batch.batch if isinstance(batch, AWRBatch) else batch).target.shape[0]
        if rows > remaining:
            if isinstance(batch, AWRBatch):
                batch = AWRBatch(
                    batch=_slice_train_batch(batch.batch, remaining),
                    returns=batch.returns[:remaining],
                    rewards=batch.rewards[:remaining],
                    terminal=batch.terminal[:remaining],
                )
            else:
                batch = _slice_train_batch(batch, remaining)
            rows = remaining
        batches.append(batch)
        count += rows
    if count != n_samples:
        raise RuntimeError(f"validation yielded {count} samples, expected {n_samples}")
    return batches


def device_batches(cpu_batches: list, device: str | torch.device, copy_stream: torch.cuda.Stream | None) -> Iterator:
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
        if isinstance(ready, AWRBatch):
            tensors = ready.tensors()
        else:
            tensors = (*ready.context.features.values(), ready.context.ctx_pad, ready.target)
        for tensor in tensors:
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


def awr_seams(cfg: TrainConfig) -> dict[str, object]:
    """Loader callables for the AWR arms.  Partials keep them picklable for workers."""
    label_kwargs = dict(gamma=cfg.awr_gamma, damage_shaping=cfg.awr_damage_shaping, win_reward=cfg.awr_win_reward)
    return {
        "annotate_replay": functools.partial(annotate_compact_replay, **label_kwargs),
        "replay_transform": functools.partial(label_decoded_replay, **label_kwargs),
        "batch_transform": functools.partial(collate_awr_batch, L_ctx=cfg.L_ctx, horizon=cfg.awr_horizon),
    }


def _make_loaders(cfg: TrainConfig, stats: dict[str, FeatureStats]):
    kwargs = loader_kwargs(cfg, stats)
    awr_on = cfg.awr_mode != "off"
    seams = awr_seams(cfg) if awr_on else {}
    if awr_on and cfg.observation_bundle == "base":
        kwargs["projection"] = AWR_PROJECTION
    if cfg.compact_data:
        train_loader = make_reservoir_loader(
            split="train",
            num_workers=cfg.num_workers,
            prefetch_factor=cfg.prefetch_factor,
            predownload=cfg.predownload,
            windows_per_replay=cfg.windows_per_replay,
            reservoir_capacity=cfg.reservoir_capacity,
            prefetch_batches=cfg.prefetch_batches,
            **({k: seams[k] for k in ("annotate_replay", "batch_transform")} if awr_on else {}),
            **kwargs,
        )
    else:
        train_loader = make_loader(
            split="train",
            num_workers=cfg.num_workers,
            prefetch_factor=cfg.prefetch_factor,
            windows_per_replay=cfg.windows_per_replay,
            compact=False,
            **({k: seams[k] for k in ("replay_transform", "batch_transform")} if awr_on else {}),
            **kwargs,
        )
    val_kwargs = {**kwargs, "batch_size": cfg.val_batch_size}
    val_loader = make_loader(
        split=cfg.val_split,
        num_workers=0,
        compact=cfg.compact_data,
        **({k: seams[k] for k in ("replay_transform", "batch_transform")} if awr_on else {}),
        **val_kwargs,
    )
    return train_loader, cache_validation(val_loader, cfg.val_n_samples)


def _gate_log(gate: GateResult) -> dict[str, float]:
    out = {
        "passed": float(gate.passed),
        "beta": float(gate.beta or 0.0),
        "value_corr": gate.value_corr,
        "qhat_corr": gate.qhat_corr,
        "action_sensitivity": gate.action_sensitivity,
    }
    for entry in gate.table:
        out[f"ess_frame_b{entry['beta']:g}"] = entry["ess_frame"]
        out[f"ess_window_b{entry['beta']:g}"] = entry["ess_window"]
        out[f"clip_frac_b{entry['beta']:g}"] = entry["clip_frac"]
    return out


def _critic_log(critic: CriticLosses, cfg: TrainConfig) -> dict[str, float]:
    eligible = critic.eligible
    advantage = critic.advantage[eligible]
    out = {
        "value_loss": float(critic.value.detach()),
        "qhat_loss": float(critic.qhat.detach()),
        "variance_loss": float(critic.variance.detach()),
        "value_rows": float(critic.value_rows),
        "adv_eligible_frac": float(eligible.float().mean()),
    }
    if advantage.numel():
        out.update(
            {
                "adv_mean": float(advantage.mean()),
                "adv_std": float(advantage.std()),
                "qhat_std": float(critic.out.qhat.detach()[eligible].std()),
                "sigma2_mean": float(critic.out.sigma2.detach()[eligible].mean()),
                "sigma2_calibration": float(
                    critic.out.sigma2.detach()[eligible].mean() / advantage.square().mean().clamp_min(1e-12)
                ),
            }
        )
    return out


def _awr_checkpoint_extra(
    critic_optimizer: torch.optim.AdamW | None,
    critic_scheduler: LambdaLR | None,
    awr_active: bool,
    awr_beta: float | None,
) -> dict | None:
    if critic_optimizer is None:
        return None
    return {
        "critic_opt": critic_optimizer.state_dict(),
        "critic_sched": critic_scheduler.state_dict(),
        "awr_active": awr_active,
        "awr_beta": awr_beta,
    }


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
    uploader = BackgroundUploader(run_name) if cfg.push_to_r2 else None
    wandb.init(
        project="hal",
        name=run_name,
        id=None if resume_state is None else resume_state.get("wandb_id"),
        resume="allow" if resume_state is not None else None,
        tags=["gpt", "temporal-mtp", "chunk-awr", "027", cfg.awr_mode],
        config=asdict(cfg),
    )
    if wandb.run is not None:
        wandb.define_metric("eval/net_stock_lcb", step_metric="global_step")
        wandb.define_metric("eval/net_dmg_lcb", step_metric="global_step")
        if cfg.wandb_log_code:
            log_wandb_code(wandb.run)
    run_dir, replay_dir = setup_run_dir(run_name)
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high" if cfg.allow_tf32 else "highest")
    model = GPT(cfg).to(DEVICE)
    counts = subsystem_parameter_counts(model)
    if wandb.run is not None:
        for name, value in counts.items():
            wandb.run.summary[f"parameters/{name}"] = value
        wandb.run.summary["parameters/total"] = sum(parameter.numel() for parameter in model.parameters())
    optimizer = make_optimizer(model, cfg)
    scheduler = LambdaLR(optimizer, lr_schedule(cfg))
    awr_on = cfg.awr_mode != "off"
    critic_optimizer = make_critic_optimizer(model, cfg) if awr_on else None
    critic_scheduler = LambdaLR(critic_optimizer, lr_schedule(cfg)) if awr_on else None
    awr_active = False
    awr_beta = cfg.awr_beta
    start_step = 0
    if resume_state is not None:
        model.load_state_dict(resume_state["model"])
        optimizer.load_state_dict(resume_state["opt"])
        scheduler.load_state_dict(resume_state["sched"])
        start_step = int(resume_state["step"]) + 1
        if awr_on:
            critic_optimizer.load_state_dict(resume_state["critic_opt"])
            critic_scheduler.load_state_dict(resume_state["critic_sched"])
            awr_active = bool(resume_state["awr_active"])
            awr_beta = resume_state["awr_beta"]
            if awr_active and awr_beta is None:
                raise ValueError("resume state says AWR is active but carries no beta")

    # Compilation stays in dedicated callables.  Evaluation therefore never
    # strips or mutates compiled methods and can build its own static buckets.
    def trunk_fn(features, pad, actions):
        return model(features, pad, actions)

    temporal_fn: Callable = model.temporal.teacher_forced_nll
    if DEVICE == "cuda" and cfg.compile_trunk:
        trunk_fn = torch.compile(trunk_fn, dynamic=False)
    if DEVICE == "cuda" and cfg.compile_temporal:
        temporal_fn = torch.compile(temporal_fn, dynamic=False)

    train_loader, val_cache = _make_loaders(cfg, stats)
    iterator = iter(train_loader)
    copy_stream = torch.cuda.Stream() if DEVICE == "cuda" else None
    run_started = time.monotonic()
    model.train()
    try:
        for step in range(start_step, cfg.max_steps):
            if DEVICE == "cuda":
                torch.cuda.reset_peak_memory_stats()
            loader_started = time.monotonic()
            cpu_batches: list[TrainBatch] = []
            for _ in range(cfg.grad_accum_steps):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(train_loader)
                    batch = next(iterator)
                validate_batch_geometry(
                    batch.batch if isinstance(batch, AWRBatch) else batch, cfg, micro_batch_size(cfg)
                )
                cpu_batches.append(batch)
            loader_wait = time.monotonic() - loader_started
            valid_prefixes = sum(
                int((cfg.L_ctx - (item.batch if isinstance(item, AWRBatch) else item).context.ctx_pad).sum())
                for item in cpu_batches
            )
            if valid_prefixes <= 0:
                raise RuntimeError("training accumulation contains no valid context prefixes")
            if awr_on and not awr_active and step >= cfg.awr_warmup_steps:
                # The pre-registered activation gate: after the step-2047 update,
                # before the step-2048 batch, in the same process.
                gate = run_activation_gate(model, val_cache, cfg, run_dir, uploader)
                wandb.log({"global_step": step, **{f"gate/{k}": v for k, v in _gate_log(gate).items()}})
                if not gate.passed:
                    raise SystemExit(f"AWR activation gate failed at step {step}: {gate.reasons}")
                awr_beta = gate.beta
                awr_active = True
            optimizer.zero_grad()
            if critic_optimizer is not None:
                critic_optimizer.zero_grad()
            nll_sum = torch.zeros(len(cfg.head_offsets), N_GROUPS, device=DEVICE)
            n_prefixes = 0
            awr_log: dict[str, float] = {}
            with profile("step") as stopwatch:
                for item in device_batches(cpu_batches, DEVICE, copy_stream):
                    awr_batch = item if isinstance(item, AWRBatch) else None
                    batch = item.batch if awr_batch is not None else item
                    history, targets, valid = prepared_targets(model, batch)
                    with amp_context(cfg, DEVICE):
                        hidden = trunk_fn(batch.context.features, batch.context.ctx_pad, history)
                        dense_nll = temporal_fn(hidden, history, targets)
                        parts = ActionLoss(nll=dense_nll[valid], targets=targets[valid])
                        joint_nll = parts.nll.sum(dim=-1)
                        weight_flat = None
                        critic = None
                        if awr_batch is not None:
                            critic = critic_losses(model, hidden, awr_batch, targets, valid, cfg)
                            if awr_active:
                                weight_grid, weight_stats = awr_weights(
                                    critic.weight_source, critic.eligible, beta=awr_beta, weight_max=cfg.awr_weight_max
                                )
                                weight_flat = weight_grid[valid]
                                rows = torch.arange(valid.shape[0], device=valid.device)[:, None].expand_as(valid)
                                awr_log = {
                                    **weight_stats,
                                    "ess_window": window_ess(
                                        weight_grid[critic.eligible].cpu(),
                                        rows[critic.eligible].cpu(),
                                        valid.shape[0],
                                    ),
                                }
                        if weight_flat is None:
                            primary = joint_nll[:, :4].sum() / (valid_prefixes * 4)
                        else:
                            primary = (joint_nll[:, :4] * weight_flat[:, None]).sum() / (valid_prefixes * 4)
                        auxiliary = joint_nll[:, 4:].sum() / (valid_prefixes * (len(cfg.head_offsets) - 4))
                        loss = primary + cfg.aux_loss_weight * auxiliary
                        if critic is not None:
                            loss = loss + (
                                cfg.value_loss_weight * critic.value
                                + cfg.qhat_loss_weight * critic.qhat
                                + cfg.variance_loss_weight * critic.variance
                            )
                            awr_log.update(_critic_log(critic, cfg))
                    loss.backward()
                    nll_sum += parts.nll.detach().sum(dim=0)
                    n_prefixes += parts.nll.shape[0]
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    [parameter for _, parameter in model.policy_parameters()], float("inf")
                )
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError(f"step {step}: non-finite gradient norm {gradient_norm}")
                if critic_optimizer is not None:
                    critic_norm = torch.nn.utils.clip_grad_norm_(model.critic.parameters(), cfg.critic_grad_clip)
                    if not torch.isfinite(critic_norm):
                        raise FloatingPointError(f"step {step}: non-finite critic gradient norm {critic_norm}")
                    awr_log["grad_norm_critic"] = float(critic_norm)
                optimizer.step()
                scheduler.step()
                if critic_optimizer is not None:
                    critic_optimizer.step()
                    critic_scheduler.step()
                if DEVICE == "cuda":
                    torch.cuda.synchronize()
            metrics = nll_mean_metrics((nll_sum / n_prefixes).cpu(), cfg.head_offsets)
            log = {
                "global_step": step,
                "samples": (step + 1) * cfg.batch_size,
                **{f"train/{name}": value for name, value in metrics.items()},
                "train/grad_norm": float(gradient_norm),
                **({f"awr/{name}": value for name, value in awr_log.items()} if awr_on else {}),
                **({"awr/active": float(awr_active), "awr/beta": float(awr_beta or 0.0)} if awr_on else {}),
                "throughput/step_s": stopwatch.elapsed,
                "throughput/loader_wait_s": loader_wait,
                "throughput/samples_per_s": cfg.batch_size / stopwatch.elapsed,
                "throughput/samples_per_wall_s": cfg.batch_size / (stopwatch.elapsed + loader_wait),
                "throughput/prefixes_per_s": n_prefixes / stopwatch.elapsed,
                "lr/muon": next(group["lr"] for group in optimizer.param_groups if group["use_muon"]),
                "lr/adam": next(group["lr"] for group in optimizer.param_groups if not group["use_muon"]),
            }
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
                    extra=_awr_checkpoint_extra(critic_optimizer, critic_scheduler, awr_active, awr_beta),
                )
            if val_due:
                values = val_metrics(model, val_cache, cfg)
                wandb.log({"global_step": step, **{f"val/{name}": value for name, value in values.items()}})
            if eval_due:
                values = eval_vs_cpu(
                    model,
                    stats,
                    cfg,
                    n_matchups=cfg.eval_n_matchups,
                    replay_dir=replay_dir / f"step_{step:06d}",
                    checkpoint_sha256=_checkpoint_sha256(checkpoint_path),
                )
                wandb.log({"global_step": step, **{f"eval/{name}": value for name, value in values.items()}})

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
            extra=_awr_checkpoint_extra(critic_optimizer, critic_scheduler, awr_active, awr_beta),
        )
        checkpoint_sha = _checkpoint_sha256(final_path)
        final_val = val_metrics(model, val_cache, cfg)
        wandb.log({"global_step": cfg.max_steps, **{f"val/{name}": value for name, value in final_val.items()}})
        final_eval = eval_vs_cpu(
            model,
            stats,
            cfg,
            n_matchups=cfg.final_eval_n_matchups,
            replay_dir=replay_dir / "final",
            checkpoint_sha256=checkpoint_sha,
        )
        wandb.log({"global_step": cfg.max_steps, **{f"eval/{name}": value for name, value in final_eval.items()}})
        stride6 = eval_vs_cpu(
            model,
            stats,
            cfg,
            n_matchups=cfg.final_diag_n_matchups,
            replay_dir=replay_dir / "final_s6",
            exec_horizon=cfg.final_diag_exec_horizon,
            checkpoint_sha256=checkpoint_sha,
        )
        wandb.log({"global_step": cfg.max_steps, **{f"eval_s6/{name}": value for name, value in stride6.items()}})
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
    "temporal_ff_dim",
    "group_head_dim",
    "action_embed_dim",
    "offset_embed_dim",
    "group_order",
    "observation_bundle",
}


def config_from_state(values: dict) -> TrainConfig:
    missing = _CHECKPOINT_ARCH_FIELDS - values.keys()
    if missing:
        raise ValueError(f"checkpoint is not an experiment-026 architecture; missing {sorted(missing)}")
    known = {item.name for item in fields(TrainConfig)}
    return TrainConfig(**{name: value for name, value in values.items() if name in known})


def load_checkpoint(path: str, *, device: str = DEVICE) -> tuple[GPT, TrainConfig, dict[str, FeatureStats], dict]:
    state = torch.load(path, map_location=device, weights_only=False)
    cfg = config_from_state(state["cfg"])
    validate_config(cfg)
    model = GPT(cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    stats = load_consolidated_stats(Path(cfg.data_root) / "stats.json")
    return model, cfg, stats, state


_load_ckpt = load_checkpoint


def eval_checkpoint(
    path: str,
    *,
    exec_horizon: int | None = None,
    n_matchups: int | None = None,
    eager: bool = False,
) -> dict[str, float]:
    model, cfg, stats, state = load_checkpoint(path)
    if eager:
        cfg = replace(cfg, inference_mode="eager")
    horizon = cfg.exec_horizon if exec_horizon is None else exec_horizon
    replay_dir = Path(path).resolve().parent / ("eval_replays_s6" if horizon == 6 else "eval_replays")
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


def audit_from_annotations(annotations: list[dict[str, np.ndarray]], cfg: TrainConfig) -> dict:
    """Return, terminal, and surrogate-advantage statistics for the pre-flight audit.

    The true critic does not exist before training, so two surrogate value
    functions bracket it: the global mean return and the per-replay mean return.
    The 4-step advantage under each surrogate feeds the declared beta grid."""
    terminal = [bool(annotation[AWR_TERMINAL_COLUMN][0] > 0.5) for annotation in annotations]
    returns = [annotation[f"p1_{AWR_RETURN_SUFFIX}"] for annotation in annotations]
    rewards = [annotation[f"p1_{AWR_REWARD_SUFFIX}"] for annotation in annotations]
    flat_returns = np.concatenate(returns)
    quantiles = {f"p{q:02d}": float(np.percentile(flat_returns, q)) for q in (1, 5, 25, 50, 75, 95, 99)}
    global_mean = float(flat_returns.mean())
    horizon, gamma = cfg.awr_horizon, cfg.awr_gamma
    discount = np.array([gamma**k for k in range(horizon)], dtype=np.float64)
    out: dict[str, object] = {
        "replays": len(annotations),
        "frames": int(flat_returns.shape[0]),
        "terminal_frac": float(np.mean(terminal)),
        "return_mean": global_mean,
        "return_std": float(flat_returns.std()),
        "return_quantiles": quantiles,
    }
    for surrogate in ("global_mean", "replay_mean"):
        samples: list[np.ndarray] = []
        rows: list[np.ndarray] = []
        for index, (reward, ret) in enumerate(zip(rewards, returns, strict=True)):
            frames = reward.shape[0]
            if frames <= horizon + 1:
                continue
            value = global_mean if surrogate == "global_mean" else float(ret.mean())
            # rewards[t] here is r_t; position t uses r_{t+1..t+horizon}.
            window = np.lib.stride_tricks.sliding_window_view(reward[1:], horizon)[: frames - horizon - 1]
            advantage = window @ discount + (gamma**horizon - 1.0) * value
            keep = slice(0, None, 8)
            samples.append(advantage[keep].astype(np.float32))
            rows.append(np.full(samples[-1].shape[0], index, dtype=np.int64))
        flat = torch.from_numpy(np.concatenate(samples))
        row_index = torch.from_numpy(np.concatenate(rows))
        out[f"adv_std_{surrogate}"] = float(flat.std())
        out[f"beta_table_{surrogate}"] = beta_table(
            flat, row_index, len(annotations), grid=cfg.awr_beta_grid, weight_max=cfg.awr_weight_max
        )
    return out


def run_audit(cfg: TrainConfig, split: str, n_replays: int) -> dict:
    """Stream ``n_replays`` compact rows and print the return/advantage audit."""
    from streaming import StreamingDataset

    dataset = StreamingDataset(
        remote=f"{streams.remote_for_local(cfg.data_root)}/{split}"
        if streams.remote_for_local(cfg.data_root)
        else None,
        local=str(Path(cfg.data_root) / split),
        batch_size=1,
        shuffle=False,
    )
    annotations = []
    for index, compact in enumerate(dataset):
        if index >= n_replays:
            break
        annotations.append(
            annotate_compact_replay(
                compact,
                gamma=cfg.awr_gamma,
                damage_shaping=cfg.awr_damage_shaping,
                win_reward=cfg.awr_win_reward,
            )
        )
    report = audit_from_annotations(annotations, cfg)
    print(json.dumps(report, indent=2, sort_keys=True, default=float), flush=True)
    return report


def run_batch_hash_audit(cfg: TrainConfig, stats: dict[str, FeatureStats], n_batches: int) -> None:
    """Prove the annotation changes no sampled policy data.

    Hash the first ``n_batches`` training batches with the annotation on and off,
    after removing the AWR tensors, and require equal digests."""
    plain = replace(cfg, awr_mode="off")
    labeled = replace(cfg, awr_mode="critic", num_workers=0)
    plain = replace(plain, num_workers=0)

    def digest(loader_cfg: TrainConfig) -> list[str]:
        # Two loader stacks share one local dataset directory inside one process;
        # streaming tracks the directory in shared memory and must be cleaned
        # between instantiations.
        import gc

        from streaming.base.util import clean_stale_shared_memory

        gc.collect()
        clean_stale_shared_memory()
        loader, _ = _make_loaders(loader_cfg, stats)
        digests = []
        iterator = iter(loader)
        for _ in range(n_batches):
            item = next(iterator)
            batch = item.batch if isinstance(item, AWRBatch) else item
            hasher = hashlib.sha256()
            for name in sorted(batch.context.features):
                hasher.update(batch.context.features[name].numpy().tobytes())
            hasher.update(batch.context.ctx_pad.numpy().tobytes())
            hasher.update(batch.target.numpy().tobytes())
            digests.append(hasher.hexdigest())
        return digests

    plain_digests = digest(plain)
    labeled_digests = digest(labeled)
    if plain_digests != labeled_digests:
        raise AssertionError("annotated and plain loaders produced different policy batches")
    print(f"[audit] {n_batches} batches identical with and without annotation", flush=True)


@dataclass
class Args:
    cfg: TrainConfig = dataclass_field(default_factory=TrainConfig)
    comment: str = ""
    resume: str | None = None
    eval: str | None = None
    eval_exec_horizon: int | None = None
    eval_n_matchups: int | None = None
    eval_eager: bool = False
    benchmark: bool = False
    benchmark_iterations: int = 20
    audit: str | None = None
    audit_replays: int = 2000
    audit_batch_hash: bool = False
    audit_batches: int = 8


def main(args: Args) -> None:
    if args.audit is not None:
        run_audit(args.cfg, args.audit, args.audit_replays)
        return
    if args.audit_batch_hash:
        stats = load_consolidated_stats(Path(args.cfg.data_root) / "stats.json")
        run_batch_hash_audit(args.cfg, stats, args.audit_batches)
        return
    if args.benchmark:
        if args.eval is not None or args.resume is not None:
            raise SystemExit("--benchmark cannot be combined with --eval or --resume")
        run_benchmark(args.cfg, iterations=args.benchmark_iterations)
        return
    if args.eval is not None and args.resume is not None:
        raise SystemExit("pass only one of --eval or --resume")
    if args.eval is not None:
        eval_checkpoint(
            args.eval,
            exec_horizon=args.eval_exec_horizon,
            n_matchups=args.eval_n_matchups,
            eager=args.eval_eager,
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
