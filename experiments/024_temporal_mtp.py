"""Train a 20-frame causal multi-token action policy.

The causal trunk produces one state for each observed frame. A tiny causal
transformer models the next twenty controller frames. Teacher forcing shifts
the ground-truth controller sequence right, so all temporal positions train in
parallel without seeing their own target. A single causal cross-attention read
lets every action-prefix token retrieve any trunk state available at that
context prefix. Decode uses the same blocks one frame at a time with KV caches.
Each frame is factorized over controller groups in the fixed order C-stick,
triggers, buttons, main stick.

Run:
    uv run experiments/024_temporal_mtp.py
    uv run experiments/024_temporal_mtp.py --eval runs/<run>/final.pt
"""

# %%
import contextlib
import itertools
import math
import time
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
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
from torch.utils.checkpoint import checkpoint

import wandb
from hal import streams
from hal.data.feature_stats import FeatureStats
from hal.eval.cross_stage import sweep_vs_cpu_prior
from hal.eval.cross_stage import vs_cpu_metrics
from hal.eval.harness import default_session_cfg
from hal.training import scoring
from hal.training.checkpoints import BackgroundUploader
from hal.training.checkpoints import load_for_resume
from hal.training.checkpoints import save_checkpoint
from hal.training.closed_loop import RecedingHorizon
from hal.training.dataloader import make_loader
from hal.training.dataloader import make_replay_reservoir_loader
from hal.training.ego_stats import load_consolidated_stats
from hal.training.features import A_DIM
from hal.training.features import ACTION_CHANNELS
from hal.training.features import BASE_ACTION_PROJECTION
from hal.training.features import BASE_PLAYER_PREFIXES
from hal.training.features import CAT_FEATURES
from hal.training.features import FLOAT_FEATURES
from hal.training.features import Context
from hal.training.features import TrainBatch
from hal.training.features import stack_actions
from hal.training.muon import SingleDeviceMuonWithAuxAdam
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
_INPUT_PROJECTION = BASE_ACTION_PROJECTION

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
GROUP_ORDER: tuple[str, ...] = ("c_stick", "triggers", "buttons", "main_stick")

TRIGGER_L_CH = ACTION_CHANNELS.index("trigger_l")
TRIGGER_R_CH = ACTION_CHANNELS.index("trigger_r")
BUTTON_L_CH = ACTION_CHANNELS.index("button_l")
BUTTON_R_CH = ACTION_CHANNELS.index("button_r")


# %%
@dataclass
class TrainConfig:
    # Trunk.
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 4
    attn_window: int = 0
    require_flex: bool = False
    L_ctx: int = 256

    # Parallel-teacher-forced causal controller decoder.
    decoder_arch_version: int = 2
    L_chunk: int = 20
    temporal_d_model: int = 64
    temporal_layers: int = 1
    temporal_heads: int = 2
    temporal_ff_dim: int = 128
    # PyTorch fused SDPA cannot launch a CUDA batch axis above 65,535. The
    # temporal block flattens micro-batch x context prefixes, so bound each
    # fused launch while retaining parallel teacher forcing and FlashAttention.
    temporal_attn_chunk_sequences: int = 32_768
    group_head_dim: int = 64
    main_stick_embed_dim: int = 40
    c_stick_embed_dim: int = 8
    trigger_embed_dim: int = 8
    # One chunk for the default 128 x 256 x 20 teacher-forcing tensor. Keeping
    # the knob allows smaller chunks on lower-memory hardware.
    classifier_chunk_tokens: int = 1_048_576
    checkpoint_temporal: bool = False
    checkpoint_classifiers: bool = False

    # Observation categoricals. ``action_state_embed_dim`` is the in-game
    # animation/action state, not a controller-action embedding.
    action_vocab: int = 1024
    action_state_embed_dim: int = 48
    char_vocab: int = 32
    char_dim: int = 8
    stage_vocab: int = 32
    stage_dim: int = 4

    # Decode four planned frames before observing and replanning.
    exec_horizon: int = 4
    decode_temp: float = 1.0
    decode_click_trigger_fix: bool = True

    seed: int = 0
    # Effective batch. Whether the 128-example micro-batch can be increased is
    # decided by the measured 24 GiB peak after the cross-attention change.
    batch_size: int = 512
    grad_accum_steps: int = 4
    muon_lr: float = 0.02
    adam_lr: float = 8.5e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: int = 16_384
    amp_dtype: str = "bfloat16"
    allow_tf32: bool = True
    compile_trunk: bool = True
    compile_temporal: bool = True

    # Source is captured once. Full gradient distributions are captured after
    # accumulation, in optimizer-step units, so observability stays cheap.
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
    train_batch_prefetch: bool = True
    push_to_r2: bool = True


def validate_config(cfg: TrainConfig) -> None:
    positive = {
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "L_ctx": cfg.L_ctx,
        "decoder_arch_version": cfg.decoder_arch_version,
        "L_chunk": cfg.L_chunk,
        "temporal_d_model": cfg.temporal_d_model,
        "temporal_layers": cfg.temporal_layers,
        "temporal_heads": cfg.temporal_heads,
        "temporal_ff_dim": cfg.temporal_ff_dim,
        "temporal_attn_chunk_sequences": cfg.temporal_attn_chunk_sequences,
        "group_head_dim": cfg.group_head_dim,
        "main_stick_embed_dim": cfg.main_stick_embed_dim,
        "c_stick_embed_dim": cfg.c_stick_embed_dim,
        "trigger_embed_dim": cfg.trigger_embed_dim,
        "classifier_chunk_tokens": cfg.classifier_chunk_tokens,
        "action_vocab": cfg.action_vocab,
        "action_state_embed_dim": cfg.action_state_embed_dim,
        "char_vocab": cfg.char_vocab,
        "char_dim": cfg.char_dim,
        "stage_vocab": cfg.stage_vocab,
        "stage_dim": cfg.stage_dim,
        "batch_size": cfg.batch_size,
        "grad_accum_steps": cfg.grad_accum_steps,
        "max_steps": cfg.max_steps,
        "exec_horizon": cfg.exec_horizon,
        "val_n_samples": cfg.val_n_samples,
        "val_batch_size": cfg.val_batch_size,
    }
    for name, value in positive.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if cfg.L_chunk != 20:
        raise ValueError(f"this experiment requires L_chunk=20, got {cfg.L_chunk}")
    if cfg.decoder_arch_version != 2:
        raise ValueError(f"unsupported decoder_arch_version={cfg.decoder_arch_version}")
    if cfg.d_model % cfg.n_heads:
        raise ValueError("d_model must be divisible by n_heads")
    if cfg.temporal_d_model % cfg.temporal_heads:
        raise ValueError("temporal_d_model must be divisible by temporal_heads")
    if (cfg.temporal_d_model // cfg.temporal_heads) % 2:
        raise ValueError("temporal attention head dimension must be even for rotary positions")
    if cfg.temporal_attn_chunk_sequences > 65_535:
        raise ValueError("temporal_attn_chunk_sequences must not exceed CUDA's fused SDPA batch-axis limit (65,535)")
    if cfg.attn_window < 0:
        raise ValueError("attn_window must be non-negative")
    controller_width = 8 + cfg.main_stick_embed_dim + cfg.c_stick_embed_dim + cfg.trigger_embed_dim
    if controller_width != 64:
        raise ValueError(
            f"8 button bits + main/C/trigger embeddings must total 64 controller features; got {controller_width}"
        )
    if cfg.batch_size % cfg.grad_accum_steps:
        raise ValueError("batch_size must be divisible by grad_accum_steps")
    if not 1 <= cfg.exec_horizon <= cfg.L_chunk:
        raise ValueError("exec_horizon must be in [1, L_chunk]")
    if not math.isfinite(cfg.decode_temp) or cfg.decode_temp <= 0:
        raise ValueError("decode_temp must be finite and positive")
    if cfg.amp_dtype not in ("bfloat16", "float32"):
        raise ValueError("amp_dtype must be 'bfloat16' or 'float32'")
    for name in ("muon_lr", "adam_lr"):
        value = getattr(cfg, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(cfg.weight_decay) or cfg.weight_decay < 0:
        raise ValueError("weight_decay must be finite and non-negative")
    if cfg.warmup_steps < 0 or cfg.warmup_steps > cfg.max_steps:
        raise ValueError("warmup_steps must be in [0, max_steps]")
    if not isinstance(cfg.wandb_grad_every, int) or isinstance(cfg.wandb_grad_every, bool):
        raise ValueError("wandb_grad_every must be an integer")
    if cfg.wandb_grad_every < 0:
        raise ValueError("wandb_grad_every must be non-negative")
    if cfg.reservoir_capacity < 2 * micro_batch_size(cfg):
        raise ValueError("reservoir_capacity must be at least twice the micro-batch size")


def micro_batch_size(cfg: TrainConfig) -> int:
    return cfg.batch_size // cfg.grad_accum_steps


def amp_context(cfg: TrainConfig, device: torch.device | str):
    """Use the run's training precision consistently for validation and inference."""
    device_type = torch.device(device).type
    if cfg.amp_dtype == "bfloat16" and device_type == "cuda":
        return torch.autocast(device_type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


def model_tag(cfg: TrainConfig) -> str:
    attention = "full" if cfg.attn_window == 0 else f"swa{cfg.attn_window}"
    return (
        f"mtp20-d{cfg.d_model}-L{cfg.n_layers}-h{cfg.n_heads}-Lc{cfg.L_ctx}-"
        f"{attention}-ct{cfg.temporal_d_model}x{cfg.temporal_layers}-xattn-"
        f"gh{cfg.group_head_dim}-nlhead-v2-s{cfg.exec_horizon}"
    )


# %%
def quantize_groups(
    main_centers: Tensor,
    c_centers: Tensor,
    trigger_centers: Tensor,
    actions: Tensor,
) -> Tensor:
    """Convert native controller vectors to four categorical group indices."""
    continuous, buttons_raw = actions[..., :_N_CONT], actions[..., _N_CONT:]
    buttons = scoring.buttons_to_combo(buttons_raw)
    main = scoring.nearest_cluster(continuous[..., 0:2], main_centers)
    c_stick = scoring.nearest_cluster(continuous[..., 2:4], c_centers)
    trigger_pair = scoring.nearest_center(continuous[..., 4:6], trigger_centers)
    triggers = trigger_pair[..., 0] * trigger_centers.shape[0] + trigger_pair[..., 1]
    return torch.stack((buttons, main, c_stick, triggers), dim=-1)


def dequantize_groups(
    main_centers: Tensor,
    c_centers: Tensor,
    trigger_centers: Tensor,
    indices: Tensor,
) -> Tensor:
    """Convert four categorical group indices to native controller vectors."""
    n_trigger = trigger_centers.shape[0]
    buttons = scoring.combo_to_buttons(indices[..., BUTTONS_G])
    main = scoring.cluster_to_xy(indices[..., MAIN_G], main_centers)
    c_stick = scoring.cluster_to_xy(indices[..., C_G], c_centers)
    trigger_l = scoring.center_to_value(indices[..., TRIG_G] // n_trigger, trigger_centers)
    trigger_r = scoring.center_to_value(indices[..., TRIG_G] % n_trigger, trigger_centers)
    return torch.cat((main, c_stick, torch.stack((trigger_l, trigger_r), dim=-1), buttons), dim=-1)


def decoder_rmsnorm(x: Tensor) -> Tensor:
    """Fused eager RMSNorm used by the small, uncompiled temporal decoder."""
    return F.rms_norm(x, (x.shape[-1],), eps=1e-6)


class TemporalFeedForward(nn.Module):
    """Small nonlinear residual used after cross-attention."""

    def __init__(self, d_model: int, d_hidden: int) -> None:
        super().__init__()
        self.up = nn.Linear(d_model, d_hidden, bias=False)
        self.down = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.down(F.silu(self.up(decoder_rmsnorm(x))))


class NonlinearActionHead(nn.Module):
    """Tiny nonlinear classifier applied after causal group modulation."""

    def __init__(self, d_model: int, d_hidden: int, vocab: int) -> None:
        super().__init__()
        self.up = nn.Linear(d_model, d_hidden, bias=False)
        self.down = nn.Linear(d_hidden, vocab)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.up(decoder_rmsnorm(x))))


class TemporalBlock(nn.Module):
    """One RoPE causal temporal-attention block with an exact one-token KV path."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.n_heads = cfg.temporal_heads
        self.d_model = cfg.temporal_d_model
        self.head_dim = self.d_model // self.n_heads
        self.attn_chunk_sequences = cfg.temporal_attn_chunk_sequences
        self.attn_scale = 1.0 / math.sqrt(2 * cfg.temporal_layers)
        self.qkv = nn.Linear(self.d_model, 3 * self.d_model, bias=False)
        self.proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.rotary = Rotary(self.head_dim)
        self.up = nn.Linear(self.d_model, cfg.temporal_ff_dim, bias=False)
        self.down = nn.Linear(cfg.temporal_ff_dim, self.d_model, bias=False)

    def _qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, length, _ = x.shape
        q, k, v = self.qkv(decoder_rmsnorm(x)).split(self.d_model, dim=-1)
        return (
            q.view(batch, length, self.n_heads, self.head_dim),
            k.view(batch, length, self.n_heads, self.head_dim),
            v.view(batch, length, self.n_heads, self.head_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        q, k, v = self._qkv(x)
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin).transpose(1, 2)
        k = apply_rotary_emb(k, cos, sin).transpose(1, 2)
        v = v.transpose(1, 2)
        # A production micro-batch can contain 512 * 256 = 131,072 independent
        # action-prefix sequences. Fused CUDA SDPA maps this leading dimension
        # directly to a grid axis capped at 65,535 and otherwise fails with
        # cudaErrorInvalidValue. Chunk only that independent axis; no sequence
        # can attend across it, so this is mathematically identical to one call.
        attention_parts = [
            F.scaled_dot_product_attention(
                q[start : start + self.attn_chunk_sequences],
                k[start : start + self.attn_chunk_sequences],
                v[start : start + self.attn_chunk_sequences],
                is_causal=True,
            )
            for start in range(0, q.shape[0], self.attn_chunk_sequences)
        ]
        attention = attention_parts[0] if len(attention_parts) == 1 else torch.cat(attention_parts, dim=0)
        attention = attention.transpose(1, 2).contiguous().view_as(x)
        x = x + self.attn_scale * self.proj(attention)
        return x + self.down(F.silu(self.up(decoder_rmsnorm(x))))

    def forward_step(
        self,
        x: Tensor,
        past: tuple[Tensor, Tensor] | None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Advance one token. ``x`` is ``[B, d]`` and the cache is time-major in dim 2."""
        q, k, v = self._qkv(x[:, None, :])
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if past is not None:
            k = torch.cat((past[0], k), dim=2)
            v = torch.cat((past[1], v), dim=2)
        cos, sin = self.rotary.at(k.shape[2], x.device)
        q = apply_rotary_emb(q, cos[:, -1:], sin[:, -1:]).transpose(1, 2)
        k_rotated = apply_rotary_emb(k.transpose(1, 2), cos, sin).transpose(1, 2)
        attention = F.scaled_dot_product_attention(q, k_rotated, v)
        attention = attention.transpose(1, 2).contiguous().view_as(x)
        x = x + self.attn_scale * self.proj(attention)
        x = x + self.down(F.silu(self.up(decoder_rmsnorm(x))))
        return x, (k, v)


class CausalTrunkCrossAttention(nn.Module):
    """Let action-prefix tokens retrieve all causally available trunk states.

    During teacher forcing the twenty horizons are represented as grouped query
    heads. They share one trunk K/V projection and use SDPA's built-in causal
    mask along the context-time axis, avoiding a ``[B, L*H, L]`` materialized
    mask. Callers remove left padding before this path.
    """

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.n_heads = cfg.temporal_heads
        self.d_model = cfg.temporal_d_model
        self.head_dim = self.d_model // self.n_heads
        self.residual_scale = 1.0 / math.sqrt(2 * cfg.temporal_layers + 1)
        self.query = nn.Linear(self.d_model, self.d_model, bias=False)
        self.key_value = nn.Linear(cfg.d_model, 2 * self.d_model, bias=False)
        self.output = nn.Linear(self.d_model, self.d_model, bias=False)
        self.rotary = Rotary(self.head_dim)

    def project_memory(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        if hidden.ndim != 3:
            raise ValueError(f"trunk memory must be [B, L, d], got {tuple(hidden.shape)}")
        batch, length, _ = hidden.shape
        key, value = self.key_value(decoder_rmsnorm(hidden)).chunk(2, dim=-1)

        def heads(x: Tensor) -> Tensor:
            return x.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)

        return heads(key), heads(value)

    def rotate_memory_key(self, key: Tensor) -> Tensor:
        length = key.shape[2]
        cos, sin = self.rotary.at(length, key.device)
        return apply_rotary_emb(key.transpose(1, 2), cos, sin).transpose(1, 2)

    @staticmethod
    def memory_mask(ctx_pad: Tensor, length: int) -> Tensor:
        return torch.arange(length, device=ctx_pad.device)[None, :] >= ctx_pad[:, None]

    def forward(self, x: Tensor, key: Tensor, value: Tensor, valid_memory: Tensor | None = None) -> Tensor:
        """Cross-attend ``x[B, L, H, d]`` to causal trunk memory ``[B, heads, L, dh]``."""
        if x.ndim != 4 or key.shape != value.shape:
            raise ValueError(
                f"cross attention expects x [B, L, H, d] and matching K/V, got "
                f"{tuple(x.shape)}, {tuple(key.shape)}, {tuple(value.shape)}"
            )
        batch, length, horizon, _ = x.shape
        if key.shape[:3] != (batch, self.n_heads, length):
            raise ValueError(
                "teacher-forced cross attention requires query and memory to share "
                f"[B, L]; got {tuple(x.shape[:2])} and {(key.shape[0], key.shape[2])}"
            )
        # Group query heads by trunk-attention head. SDPA GQA then shares each
        # K/V head across all H horizons without repeating the K/V allocation.
        query = self.query(decoder_rmsnorm(x))
        query = query.view(batch, length, horizon, self.n_heads, self.head_dim)
        cos, sin = self.rotary.at(length, x.device)
        query = apply_rotary_emb(
            query.view(batch, length, horizon * self.n_heads, self.head_dim),
            cos,
            sin,
        ).view(batch, length, horizon, self.n_heads, self.head_dim)
        key = apply_rotary_emb(key.transpose(1, 2), cos, sin).transpose(1, 2)
        query = query.permute(0, 3, 2, 1, 4).reshape(batch, self.n_heads * horizon, length, self.head_dim)
        if valid_memory is not None and valid_memory.shape != (batch, length):
            raise ValueError(f"cross-attention memory mask must be {(batch, length)}, got {tuple(valid_memory.shape)}")
        attention_mask = None
        if valid_memory is not None:
            positions = torch.arange(length, device=x.device)
            causal = positions[:, None] >= positions[None, :]
            attention_mask = valid_memory[:, None, None, :] & causal[None, None, :, :]
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            is_causal=valid_memory is None,
            enable_gqa=horizon > 1,
        )
        attended = attended.view(batch, self.n_heads, horizon, length, self.head_dim)
        attended = attended.permute(0, 3, 2, 1, 4).contiguous().view_as(x)
        return x + self.residual_scale * self.output(attended)

    def forward_step(
        self,
        x: Tensor,
        key: Tensor,
        value: Tensor,
        valid_memory: Tensor,
    ) -> Tensor:
        """Read a cached trunk sequence for one autoregressive action token."""
        batch, length = key.shape[0], key.shape[2]
        if x.shape != (batch, self.d_model) or valid_memory.shape != (batch, length):
            raise ValueError(
                f"cross-attention step expects x {(batch, self.d_model)} and mask {(batch, length)}, "
                f"got {tuple(x.shape)} and {tuple(valid_memory.shape)}"
            )
        query = self.query(decoder_rmsnorm(x))
        query = query.view(batch, 1, self.n_heads, self.head_dim)
        cos, sin = self.rotary.at(length, x.device)
        query = apply_rotary_emb(query, cos[:, -1:], sin[:, -1:]).transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=valid_memory[:, None, None, :],
        )
        attended = attended.transpose(1, 2).contiguous().view_as(x)
        return x + self.residual_scale * self.output(attended)


class CausalTemporalDecoder(nn.Module):
    """Twenty-frame joint controller model with parallel teacher forcing.

    Temporal token ``k`` receives only the complete controller frame at ``k-1``.
    Its action state first attends causally over earlier action tokens, then
    cross-attends to trunk states no later than the context prefix being scored.
    There is deliberately no horizon-position embedding.
    """

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.L_chunk = cfg.L_chunk
        self.d_model = cfg.temporal_d_model
        self.classifier_chunk_tokens = cfg.classifier_chunk_tokens
        self.checkpoint_temporal = cfg.checkpoint_temporal
        self.checkpoint_classifiers = cfg.checkpoint_classifiers
        self.group_dims = {
            "buttons": 8,
            "main_stick": cfg.main_stick_embed_dim,
            "c_stick": cfg.c_stick_embed_dim,
            "triggers": cfg.trigger_embed_dim,
        }
        self.controller_embeddings = nn.ModuleDict(
            {
                "main_stick": nn.Embedding(GROUP_VOCABS[MAIN_G], cfg.main_stick_embed_dim),
                "c_stick": nn.Embedding(GROUP_VOCABS[C_G], cfg.c_stick_embed_dim),
                "triggers": nn.Embedding(GROUP_VOCABS[TRIG_G], cfg.trigger_embed_dim),
            }
        )
        self.condition_in = nn.Linear(cfg.d_model, self.d_model, bias=False)
        self.frame_projection = nn.Linear(sum(self.group_dims.values()), self.d_model, bias=False)
        self.bos = nn.Parameter(torch.zeros(self.d_model))
        self.blocks = nn.ModuleList([TemporalBlock(cfg) for _ in range(cfg.temporal_layers)])
        self.trunk_cross_attention = CausalTrunkCrossAttention(cfg)
        # With one temporal block, cross-attention used to feed the classifiers
        # directly. This residual lets retrieved trunk history interact
        # nonlinearly before any action group is scored.
        self.post_cross_ff = TemporalFeedForward(self.d_model, cfg.temporal_ff_dim)
        self.group_condition = nn.ModuleDict(
            {
                name: nn.Linear(
                    sum(self.group_dims[group] for group in GROUP_ORDER[:position]),
                    2 * self.d_model,
                )
                for position, name in enumerate(GROUP_ORDER)
                if position > 0
            }
        )
        self.outputs = nn.ModuleDict(
            {
                name: NonlinearActionHead(
                    self.d_model,
                    cfg.group_head_dim,
                    GROUP_VOCABS[GROUP_INDEX[name]],
                )
                for name in GROUP_NAMES
            }
        )
        # Full-width skip: action dynamics use a tiny temporal state, but no
        # classifier is forced to recover the current 256-wide trunk state from
        # that bottleneck. This term is shared across horizons; the temporal
        # logits add the causal future-action residual.
        self.trunk_outputs = nn.ModuleDict(
            {name: nn.Linear(cfg.d_model, GROUP_VOCABS[GROUP_INDEX[name]], bias=False) for name in GROUP_NAMES}
        )

    def group_embedding(self, name: str, indices: Tensor) -> Tensor:
        if name == "buttons":
            return scoring.combo_to_buttons(indices).to(dtype=self.bos.dtype)
        return self.controller_embeddings[name](indices)

    def embed_groups(self, indices: Tensor) -> dict[str, Tensor]:
        return {name: self.group_embedding(name, indices[..., GROUP_INDEX[name]]) for name in GROUP_NAMES}

    def frame_embedding(self, indices: Tensor, embedded: dict[str, Tensor] | None = None) -> Tensor:
        values = self.embed_groups(indices) if embedded is None else embedded
        return self.frame_projection(torch.cat([values[name] for name in GROUP_NAMES], dim=-1))

    def teacher_forced_states(
        self,
        hidden: Tensor,
        targets: Tensor,
        valid_memory: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        expected = (*hidden.shape[:2], self.L_chunk, N_GROUPS)
        if hidden.ndim != 3 or targets.shape != expected:
            raise ValueError(
                f"expected hidden [B, L, d] and targets [B, L, {self.L_chunk}, {N_GROUPS}], "
                f"got {tuple(hidden.shape)} and {tuple(targets.shape)}"
            )
        embedded = self.embed_groups(targets)
        previous_frames = self.frame_embedding(
            targets[..., :-1, :],
            {name: values[..., :-1, :] for name, values in embedded.items()},
        )
        batch, length = hidden.shape[:2]
        bos = self.bos.expand(batch, length, 1, self.d_model)
        condition = self.condition_in(hidden)[:, :, None, :]
        x = condition + torch.cat((bos, previous_frames), dim=2)
        key, value = self.trunk_cross_attention.project_memory(hidden)
        x = x.reshape(batch * length, self.L_chunk, self.d_model)
        for index, block in enumerate(self.blocks):
            if self.checkpoint_temporal and self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
            if index == 0:
                x = x.view(batch, length, self.L_chunk, self.d_model)
                if self.checkpoint_temporal and self.training and torch.is_grad_enabled():
                    x = checkpoint(
                        self.trunk_cross_attention,
                        x,
                        key,
                        value,
                        valid_memory,
                        use_reentrant=False,
                    )
                else:
                    x = self.trunk_cross_attention(x, key, value, valid_memory)
                x = self.post_cross_ff(x)
                x = x.reshape(batch * length, self.L_chunk, self.d_model)
        x = x.view(batch, length, self.L_chunk, self.d_model)
        return decoder_rmsnorm(x), embedded

    def group_features(
        self,
        states: Tensor,
        name: str,
        embedded: dict[str, Tensor],
    ) -> Tensor:
        position = GROUP_ORDER.index(name)
        earlier = GROUP_ORDER[:position]
        if not earlier:
            return states
        prefix = torch.cat([embedded[group] for group in earlier], dim=-1)
        scale, shift = self.group_condition[name](prefix).chunk(2, dim=-1)
        return states * (1.0 + torch.tanh(scale)) + shift

    def teacher_forced_logits_by_group(self, hidden: Tensor, targets: Tensor) -> dict[str, Tensor]:
        states, embedded = self.teacher_forced_states(hidden, targets)
        return {
            name: self.outputs[name](self.group_features(states, name, embedded))
            + self.trunk_outputs[name](decoder_rmsnorm(hidden))[:, :, None, :]
            for name in GROUP_NAMES
        }

    def teacher_forced_nll(
        self,
        hidden: Tensor,
        targets: Tensor,
        valid_memory: Tensor | None = None,
    ) -> Tensor:
        """Return ``[B, L, H, G]`` NLL while bounding classifier activations."""
        states, embedded = self.teacher_forced_states(hidden, targets, valid_memory)
        normalized_hidden = decoder_rmsnorm(hidden)
        prefix_index = (
            torch.arange(
                hidden.shape[0] * hidden.shape[1] * self.L_chunk,
                device=hidden.device,
            )
            // self.L_chunk
        )
        losses: list[Tensor] = []
        for name in GROUP_NAMES:
            group = GROUP_INDEX[name]
            features = self.group_features(states, name, embedded).reshape(-1, self.d_model)
            expected = targets[..., group].reshape(-1)
            output = self.outputs[name]
            trunk_logits = self.trunk_outputs[name](normalized_hidden).reshape(-1, GROUP_VOCABS[group])

            def score(
                feature_chunk: Tensor,
                trunk_chunk: Tensor,
                target_chunk: Tensor,
                output: nn.Module = output,
            ) -> Tensor:
                logits = output(feature_chunk) + trunk_chunk
                return F.cross_entropy(logits.float(), target_chunk, reduction="none")

            parts: list[Tensor] = []
            for start in range(0, features.shape[0], self.classifier_chunk_tokens):
                stop = start + self.classifier_chunk_tokens
                feature_chunk = features[start:stop]
                target_chunk = expected[start:stop]
                trunk_chunk = trunk_logits[prefix_index[start:stop]]
                if self.checkpoint_classifiers and self.training and torch.is_grad_enabled():
                    part = checkpoint(
                        score,
                        feature_chunk,
                        trunk_chunk,
                        target_chunk,
                        use_reentrant=False,
                    )
                else:
                    part = score(feature_chunk, trunk_chunk, target_chunk)
                parts.append(part)
            losses.append(torch.cat(parts).view(*hidden.shape[:2], self.L_chunk))
        return torch.stack(losses, dim=-1)

    def teacher_forced_logits(self, hidden: Tensor, targets: Tensor) -> list[dict[str, Tensor]]:
        """Compatibility/testing view of the parallel logits, split by horizon."""
        by_group = self.teacher_forced_logits_by_group(hidden, targets)
        return [{name: values[..., depth, :] for name, values in by_group.items()} for depth in range(self.L_chunk)]

    def forced_stepwise_logits(
        self,
        hidden: Tensor,
        targets: Tensor,
        ctx_pad: Tensor,
    ) -> list[dict[str, Tensor]]:
        """Reference the cached path while feeding exact prior/current targets."""
        if hidden.ndim != 3 or targets.shape != (hidden.shape[0], self.L_chunk, N_GROUPS):
            raise ValueError("forced stepwise decode requires hidden [B, L, d] and targets [B, H, G]")
        condition = self.condition_in(hidden[:, -1])
        trunk_condition = decoder_rmsnorm(hidden[:, -1])
        trunk_logits = {name: self.trunk_outputs[name](trunk_condition) for name in GROUP_NAMES}
        key, value = self.trunk_cross_attention.project_memory(hidden)
        key = self.trunk_cross_attention.rotate_memory_key(key)
        valid_memory = self.trunk_cross_attention.memory_mask(ctx_pad, hidden.shape[1])
        previous = self.bos.expand_as(condition)
        caches: list[tuple[Tensor, Tensor] | None] = [None] * len(self.blocks)
        out: list[dict[str, Tensor]] = []
        for depth in range(self.L_chunk):
            state = condition + previous
            next_caches: list[tuple[Tensor, Tensor]] = []
            for index, (block, past) in enumerate(zip(self.blocks, caches, strict=True)):
                state, present = block.forward_step(state, past)
                next_caches.append(present)
                if index == 0:
                    state = self.trunk_cross_attention.forward_step(state, key, value, valid_memory)
                    state = self.post_cross_ff(state)
            caches = next_caches
            state = decoder_rmsnorm(state)
            target = targets[:, depth]
            embedded = self.embed_groups(target)
            out.append(
                {
                    name: self.outputs[name](self.group_features(state, name, embedded)) + trunk_logits[name]
                    for name in GROUP_NAMES
                }
            )
            previous = self.frame_embedding(target, embedded)
        return out

    def sample_indices(
        self,
        hidden: Tensor,
        ctx_pad: Tensor,
        n_frames: int,
        *,
        temperature: float,
        argmax: bool,
        gen: torch.Generator | None,
    ) -> Tensor:
        if hidden.ndim != 3:
            raise ValueError(f"sample_indices requires full trunk memory [B, L, d], got {tuple(hidden.shape)}")
        condition = self.condition_in(hidden[:, -1])
        trunk_condition = decoder_rmsnorm(hidden[:, -1])
        trunk_logits = {name: self.trunk_outputs[name](trunk_condition) for name in GROUP_NAMES}
        key, value = self.trunk_cross_attention.project_memory(hidden)
        key = self.trunk_cross_attention.rotate_memory_key(key)
        valid_memory = self.trunk_cross_attention.memory_mask(ctx_pad, hidden.shape[1])
        previous = self.bos.expand_as(condition)
        caches: list[tuple[Tensor, Tensor] | None] = [None] * len(self.blocks)
        frames: list[Tensor] = []
        for _ in range(n_frames):
            state = condition + previous
            next_caches: list[tuple[Tensor, Tensor]] = []
            for index, (block, past) in enumerate(zip(self.blocks, caches, strict=True)):
                state, present = block.forward_step(state, past)
                next_caches.append(present)
                if index == 0:
                    state = self.trunk_cross_attention.forward_step(state, key, value, valid_memory)
                    state = self.post_cross_ff(state)
            caches = next_caches
            state = decoder_rmsnorm(state)
            embedded: dict[str, Tensor] = {}
            picks: dict[str, Tensor] = {}
            for name in GROUP_ORDER:
                features = self.group_features(state, name, embedded)
                logits = self.outputs[name](features) + trunk_logits[name]
                pick = sample_categorical(logits, temperature=temperature, argmax=argmax, gen=gen)
                picks[name] = pick
                embedded[name] = self.group_embedding(name, pick)
            indices = torch.stack([picks[name] for name in GROUP_NAMES], dim=-1)
            frames.append(indices)
            previous = self.frame_embedding(indices, embedded)
        return torch.stack(frames, dim=1)


class GPT(nn.Module):
    """Causal game-state trunk followed by a 20-frame temporal action chain."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.L_chunk = cfg.L_chunk
        self.group_order = GROUP_ORDER
        self.cat_specs = {**CAT_FEATURES, "action": (cfg.action_vocab, cfg.action_state_embed_dim)}
        self.cat_embeds = nn.ModuleDict(
            {name: nn.Embedding(vocab, dim) for name, (vocab, dim) in self.cat_specs.items()}
        )
        self.char_emb = nn.Embedding(cfg.char_vocab, cfg.char_dim)
        self.stage_emb = nn.Embedding(cfg.stage_vocab, cfg.stage_dim)
        per_player = len(FLOAT_FEATURES) * 2 + sum(dim for _, dim in self.cat_specs.values())
        d_in = len(_PLAYER_PREFIXES) * per_player + A_DIM + 2 * cfg.char_dim + cfg.stage_dim
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

        self.temporal = CausalTemporalDecoder(cfg)

        self.register_buffer("main_centers", scoring.STICK_CLUSTER_CENTERS_MAIN.clone())
        self.register_buffer("c_centers", scoring.STICK_CLUSTER_CENTERS_C.clone())
        self.register_buffer("trigger_centers", scoring.TRIGGER_CENTERS.clone())

    def _per_player_features(self, features: dict[str, Tensor], prefix: str) -> Tensor:
        ref = features[f"{prefix}_position_x"]
        batch, length = ref.shape
        parts: list[Tensor] = [features[f"{prefix}_{name}"][..., None] for name in FLOAT_FEATURES]
        for name in FLOAT_FEATURES:
            key = f"{prefix}_{name}_mask"
            mask = features.get(key)
            parts.append(
                mask[..., None]
                if mask is not None
                else torch.zeros(batch, length, 1, device=ref.device, dtype=ref.dtype)
            )
        for name, (vocab, _) in self.cat_specs.items():
            parts.append(self.cat_embeds[name](features[f"{prefix}_{name}"].clamp(0, vocab - 1)))
        return torch.cat(parts, dim=-1)

    def context_tokens(self, features: dict[str, Tensor]) -> Tensor:
        parts = [self._per_player_features(features, prefix) for prefix in _PLAYER_PREFIXES]
        parts.append(torch.stack([features[f"ego_{name}"] for name in ACTION_CHANNELS], dim=-1))
        parts.append(self.char_emb(features["ego_character"].clamp(0, self.char_emb.num_embeddings - 1)))
        parts.append(self.char_emb(features["opp_character"].clamp(0, self.char_emb.num_embeddings - 1)))
        parts.append(self.stage_emb(features["stage"].clamp(0, self.stage_emb.num_embeddings - 1)))
        return self.ctx_proj(torch.cat(parts, dim=-1))

    def forward(self, features: dict[str, Tensor], ctx_pad: Tensor) -> Tensor:
        return self.trunk(self.context_tokens(features), ctx_pad)

    def teacher_forced_logits(self, hidden: Tensor, targets: Tensor) -> list[dict[str, Tensor]]:
        """Return depth logits for targets shaped ``[..., H, N_GROUPS]``."""
        return self.temporal.teacher_forced_logits(hidden, targets)


def quantize(model: GPT, actions: Tensor) -> Tensor:
    return quantize_groups(model.main_centers, model.c_centers, model.trigger_centers, actions)


def dequantize(model: GPT, indices: Tensor) -> Tensor:
    return dequantize_groups(model.main_centers, model.c_centers, model.trigger_centers, indices)


def chunk_targets(model: GPT, batch: TrainBatch) -> tuple[Tensor, Tensor]:
    """Return next-20 frame targets at every context position and their valid mask."""
    if batch.target.shape[1] < model.L_chunk:
        raise ValueError(f"target contains {batch.target.shape[1]} frames, expected at least {model.L_chunk}")
    history = stack_actions(batch.context.features)
    if history.shape[1] != model.trunk.L_ctx:
        raise ValueError(f"context contains {history.shape[1]} frames, expected {model.trunk.L_ctx}")
    full = quantize(model, torch.cat((history, batch.target[:, : model.L_chunk]), dim=1))
    length = history.shape[1]
    targets = torch.stack([full[:, offset : offset + length] for offset in range(1, model.L_chunk + 1)], dim=2)
    positions = torch.arange(length, device=full.device)
    valid = positions[None, :] >= batch.context.ctx_pad[:, None]
    return targets, valid


@dataclass(frozen=True, slots=True)
class ActionLoss:
    nll: Tensor  # [N_valid, H, N_GROUPS], nats
    targets: Tensor  # [N_valid, H, N_GROUPS]


def _record_batch_stream(batch: TrainBatch, stream: torch.cuda.Stream) -> None:
    """Tell the CUDA allocator that the consumer stream owns these copies."""
    tensors = [*batch.context.features.values(), batch.context.ctx_pad, batch.target]
    if batch.context.slot_ids is not None:
        tensors.append(batch.context.slot_ids)
    if batch.context.reset is not None:
        tensors.append(batch.context.reset)
    for tensor in tensors:
        tensor.record_stream(stream)


def device_batches(
    cpu_batches: list[TrainBatch],
    device: str | torch.device,
    copy_stream: torch.cuda.Stream | None,
) -> Iterator[TrainBatch]:
    """Pipeline pinned host copies one micro-batch ahead of GPU compute."""
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
        _record_batch_stream(ready, compute_stream)
        if index + 1 < len(cpu_batches):
            with torch.cuda.stream(copy_stream):
                staged = cpu_batches[index + 1].to(target)
        yield ready


def action_loss(
    model: GPT,
    batch: TrainBatch,
    *,
    hidden: Tensor | None = None,
) -> ActionLoss:
    targets, valid = chunk_targets(model, batch)
    if hidden is None:
        hidden = model(batch.context.features, batch.context.ctx_pad)
    if hidden.shape[:2] != targets.shape[:2]:
        raise ValueError(
            f"hidden and target context axes must match, got {tuple(hidden.shape)} and {tuple(targets.shape)}"
        )
    # Keep the decoder's training geometry fixed for compilation. Cross-attention
    # combines the broadcast key-validity mask with its causal mask, so every
    # valid prefix sees the same history as the former variable-size sliced calls.
    positions = torch.arange(hidden.shape[1], device=hidden.device)
    valid_memory = positions[None, :] >= batch.context.ctx_pad[:, None]
    dense_nll = model.temporal.teacher_forced_nll(hidden, targets, valid_memory)
    nll = dense_nll[valid]
    target_valid = targets[valid]
    if nll.numel() == 0:
        raise ValueError("batch contains no valid context prefixes")
    if nll.shape != target_valid.shape:
        raise RuntimeError(f"NLL and target shapes differ: {tuple(nll.shape)} != {tuple(target_valid.shape)}")
    return ActionLoss(nll=nll, targets=target_valid)


def objective(parts: ActionLoss) -> Tensor:
    """Mean over valid prefix/horizon pairs of each frame's joint group NLL.

    The horizon mean is deliberate: summing the twenty conditionals would make
    the objective and its gradients twenty times larger than a per-frame NLL.
    """
    return parts.nll.sum(dim=-1).mean()


def sample_categorical(logits: Tensor, *, temperature: float, argmax: bool, gen: torch.Generator | None) -> Tensor:
    values = logits.float()
    if argmax:
        return values.argmax(dim=-1)
    return torch.multinomial(F.softmax(values / temperature, dim=-1), 1, generator=gen).squeeze(-1)


def apply_click_trigger_fix(action: Tensor) -> Tensor:
    action[..., TRIGGER_L_CH] = torch.where(
        action[..., BUTTON_L_CH] > 0.5,
        torch.ones_like(action[..., TRIGGER_L_CH]),
        action[..., TRIGGER_L_CH],
    )
    action[..., TRIGGER_R_CH] = torch.where(
        action[..., BUTTON_R_CH] > 0.5,
        torch.ones_like(action[..., TRIGGER_R_CH]),
        action[..., TRIGGER_R_CH],
    )
    return action


@torch.no_grad()
def sample_chunk_from_hidden(
    model: GPT,
    hidden: Tensor,
    ctx_pad: Tensor,
    n_frames: int,
    *,
    temperature: float = 1.0,
    argmax: bool = False,
    click_trigger_fix: bool = True,
    gen: torch.Generator | None = None,
) -> Tensor:
    if not 1 <= n_frames <= model.L_chunk:
        raise ValueError(f"n_frames must be in [1, {model.L_chunk}], got {n_frames}")
    if not argmax and (not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("temperature must be finite and positive")
    indices = model.temporal.sample_indices(
        hidden,
        ctx_pad,
        n_frames,
        temperature=temperature,
        argmax=argmax,
        gen=gen,
    )
    action = dequantize(model, indices)
    if click_trigger_fix:
        action = apply_click_trigger_fix(action)
    return action


@torch.no_grad()
def decode_chunk(
    model: GPT,
    ctx: Context,
    n_frames: int,
    *,
    temperature: float = 1.0,
    argmax: bool = False,
    click_trigger_fix: bool = True,
    gen: torch.Generator | None = None,
) -> Tensor:
    hidden = model(ctx.features, ctx.ctx_pad)
    return sample_chunk_from_hidden(
        model,
        hidden,
        ctx.ctx_pad,
        n_frames,
        temperature=temperature,
        argmax=argmax,
        click_trigger_fix=click_trigger_fix,
        gen=gen,
    )


def make_policy(
    model: GPT,
    stats: dict[str, FeatureStats],
    cfg: TrainConfig,
    *,
    exec_horizon: int | None = None,
    temperature: float | None = None,
    seed: int | None = None,
    device: str = DEVICE,
) -> RecedingHorizon:
    horizon = cfg.exec_horizon if exec_horizon is None else exec_horizon
    if not 1 <= horizon <= cfg.L_chunk:
        raise ValueError(f"execution horizon must be in [1, {cfg.L_chunk}]")
    temp = cfg.decode_temp if temperature is None else temperature
    generator = None if seed is None else torch.Generator(device=device).manual_seed(seed)

    @torch.no_grad()
    def predict(ctx: Context, committed: np.ndarray | None) -> np.ndarray:
        if committed is not None:
            raise ValueError("this policy does not use a committed RTC prefix")
        with amp_context(cfg, device):
            action = decode_chunk(
                model,
                ctx,
                horizon,
                temperature=temp,
                click_trigger_fix=cfg.decode_click_trigger_fix,
                gen=generator,
            )
        return action.cpu().numpy()

    return RecedingHorizon(
        predict_chunk=predict,
        stats=stats,
        L_ctx=cfg.L_ctx,
        L_chunk=horizon,
        s=horizon,
        d=0,
        device=device,
        float_dtype=next(model.parameters()).dtype,
        projection=_INPUT_PROJECTION,
    )


# %%
def nll_metrics(nll: Tensor) -> dict[str, float]:
    """Summarize an ``[N, H, G]`` teacher-forced NLL tensor in bits."""
    if nll.ndim != 3 or nll.shape[-1] != N_GROUPS:
        raise ValueError(f"NLL must be [N, H, {N_GROUPS}], got {tuple(nll.shape)}")
    return nll_mean_metrics(nll.mean(dim=0))


def nll_mean_metrics(mean_nll: Tensor) -> dict[str, float]:
    """Summarize a pre-aggregated ``[H, G]`` mean without retaining every prefix."""
    if mean_nll.shape != (20, N_GROUPS):
        raise ValueError(f"mean NLL must be [20, {N_GROUPS}], got {tuple(mean_nll.shape)}")
    per_horizon = mean_nll.sum(dim=-1) / _LN2
    metrics = {
        "loss": float(per_horizon.mean()),
        "nll_chunk": float(per_horizon.sum()),
        **{f"nll_h{depth + 1:02d}": float(value) for depth, value in enumerate(per_horizon)},
    }
    per_group = mean_nll.mean(dim=0) / _LN2
    metrics.update({f"nll_{name}": float(per_group[index]) for index, name in enumerate(GROUP_NAMES)})
    return metrics


@contextlib.contextmanager
def evaluation_mode(model: nn.Module) -> Iterator[None]:
    """Use the class's eager forward for every non-training batch shape."""
    was_training = model.training
    compiled = model.__dict__.pop("forward", None)
    temporal = getattr(model, "temporal", None)
    compiled_temporal = None if temporal is None else temporal.__dict__.pop("teacher_forced_nll", None)
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)
        if compiled is not None:
            model.forward = compiled
        if compiled_temporal is not None:
            assert temporal is not None
            temporal.teacher_forced_nll = compiled_temporal


@torch.no_grad()
def val_metrics(model: GPT, batches: list[TrainBatch], cfg: TrainConfig) -> dict[str, float]:
    device = next(model.parameters()).device
    with evaluation_mode(model):
        return _val_metrics_eval(model, batches, cfg, device)


def _val_metrics_eval(
    model: GPT,
    batches: list[TrainBatch],
    cfg: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    nll_sum = torch.zeros(model.L_chunk, N_GROUPS, dtype=torch.float64)
    n_prefixes = 0
    raw_exact = raw_groups = deploy_exact = deploy_groups = total_groups = 0
    for cpu_batch in batches:
        validate_batch_geometry(cpu_batch, cfg)
        batch = cpu_batch.to(device)
        with amp_context(cfg, device):
            hidden = model(batch.context.features, batch.context.ctx_pad)
            parts = action_loss(model, batch, hidden=hidden)
            raw = model.temporal.sample_indices(
                hidden,
                batch.context.ctx_pad,
                model.L_chunk,
                temperature=1.0,
                argmax=True,
                gen=None,
            )
        nll_sum += parts.nll.detach().double().sum(dim=0).cpu()
        n_prefixes += parts.nll.shape[0]
        target = quantize(model, batch.target[:, : model.L_chunk])
        raw_matches = raw == target
        raw_groups += int(raw_matches.sum())
        raw_exact += int(raw_matches.all(dim=-1).sum())
        deployed = raw
        if cfg.decode_click_trigger_fix:
            deployed = quantize(model, apply_click_trigger_fix(dequantize(model, raw)))
        deploy_matches = deployed == target
        deploy_groups += int(deploy_matches.sum())
        deploy_exact += int(deploy_matches.all(dim=-1).sum())
        total_groups += raw_matches.numel()
    if n_prefixes == 0:
        raise RuntimeError("validation contained no valid context prefixes")
    n_frames = total_groups // N_GROUPS
    return {
        **nll_mean_metrics(nll_sum / n_prefixes),
        "ancestral_group_acc": raw_groups / max(total_groups, 1),
        "ancestral_frame_acc": raw_exact / max(n_frames, 1),
        "deploy_group_acc": deploy_groups / max(total_groups, 1),
        "deploy_frame_acc": deploy_exact / max(n_frames, 1),
    }


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
    muon = [parameter for parameter in model.trunk.blocks.parameters() if parameter.ndim >= 2]
    muon_ids = {id(parameter) for parameter in muon}
    embedding_ids = {
        id(parameter)
        for module in (model.cat_embeds, model.char_emb, model.stage_emb, model.temporal.controller_embeddings)
        for parameter in module.parameters()
    }
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


def log_wandb_code(run: wandb.Run) -> None:
    """Capture the reproducibility-relevant source tree once per run."""
    root = Path(__file__).resolve().parents[1]
    source_dirs = {"docker", "experiments", "hal", "scripts", "tests"}
    source_files = {"Dockerfile", "pyproject.toml", "uv.lock"}

    def include(path: str, code_root: str) -> bool:
        try:
            relative = Path(path).resolve().relative_to(Path(code_root).resolve())
        except ValueError:
            return False
        if relative.as_posix() in source_files:
            return True
        return (
            bool(relative.parts)
            and relative.parts[0] in source_dirs
            and relative.suffix
            in {
                ".py",
                ".sh",
                ".toml",
                ".yaml",
                ".yml",
            }
        )

    run.log_code(root=str(root), include_fn=include)


def _gradient_group(name: str) -> str:
    """Keep W&B gradient panels useful without emitting one chart per tensor."""
    if name.startswith("ar."):
        name = name.removeprefix("ar.")
    parts = name.split(".")
    if len(parts) >= 3 and parts[:2] == ["trunk", "blocks"]:
        return f"trunk/block_{parts[2]}"
    if len(parts) >= 3 and parts[:2] == ["temporal", "blocks"]:
        return f"decoder/temporal_block_{parts[2]}"
    decoder_groups = {
        "condition_in": "decoder/input",
        "frame_projection": "decoder/input",
        "bos": "decoder/input",
        "controller_embeddings": "decoder/input",
        "trunk_cross_attention": "decoder/cross_attention",
        "post_cross_ff": "decoder/post_cross_ff",
        "group_condition": "decoder/group_condition",
        "outputs": "decoder/nonlinear_heads",
        "trunk_outputs": "decoder/trunk_skip",
    }
    if len(parts) >= 2 and parts[0] == "temporal":
        return decoder_groups.get(parts[1], "decoder/other")
    if len(parts) >= 3 and parts[:2] == ["flow", "blocks"]:
        return f"flow/block_{parts[2]}"
    if parts[0] == "flow":
        return "flow/other"
    return "trunk/input"


def wandb_gradient_log(model: nn.Module, *, sample_limit: int = 65_536) -> dict[str, object]:
    """Build sampled gradient histograms and exact grouped L2 norms.

    This is called only after a complete accumulation and only at the configured
    optimizer-step interval. Sampling bounds the device transfer while the norm
    still covers every gradient element.
    """
    buckets: dict[str, list[Tensor]] = {}
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        values = gradient.coalesce().values() if gradient.is_sparse else gradient
        buckets.setdefault(_gradient_group(name), []).append(values.detach())

    payload: dict[str, object] = {}
    for group, gradients in buckets.items():
        count = sum(gradient.numel() for gradient in gradients)
        stride = max(1, math.ceil(count / sample_limit))
        samples = torch.cat([gradient.reshape(-1)[::stride] for gradient in gradients])[:sample_limit]
        squared_norm = torch.stack([gradient.float().square().sum() for gradient in gradients]).sum()
        payload[f"gradients/{group}"] = wandb.Histogram(samples.float().cpu().numpy())
        payload[f"gradient_norm/{group}"] = float(squared_norm.sqrt())
    return payload


def gradient_log_due(step: int, start_step: int, every: int) -> bool:
    """Log the first accumulated gradient and then every N optimizer steps."""
    return every > 0 and (step == start_step or (step + 1) % every == 0)


def loader_kwargs(cfg: TrainConfig, stats: dict[str, FeatureStats]) -> dict:
    return dict(
        data_root=cfg.data_root,
        remote=streams.remote_for_local(cfg.data_root),
        cache_limit=f"{cfg.cache_limit_gb}gb",
        shuffle_block_size=cfg.shuffle_block_size,
        stats=stats,
        L_ctx=cfg.L_ctx,
        L_chunk=cfg.L_chunk,
        batch_size=micro_batch_size(cfg),
        seed=cfg.seed,
        schema_version=cfg.mds_schema_version,
        projection=_INPUT_PROJECTION,
    )


def validate_batch_geometry(
    batch: TrainBatch,
    cfg: TrainConfig,
    *,
    expected_batch_size: int | None = None,
) -> None:
    """Validate the fixed compile boundary and context/future window contract."""
    if batch.target.ndim != 3 or batch.target.shape[1:] != (cfg.L_chunk, A_DIM):
        raise ValueError(f"target must be [B, {cfg.L_chunk}, {A_DIM}], got {tuple(batch.target.shape)}")
    batch_size = batch.target.shape[0]
    if expected_batch_size is not None and batch_size != expected_batch_size:
        raise ValueError(f"fixed training batch must contain {expected_batch_size} rows, got {batch_size}")
    if batch.context.ctx_pad.shape != (batch_size,):
        raise ValueError(f"ctx_pad must have shape {(batch_size,)}, got {tuple(batch.context.ctx_pad.shape)}")
    if not batch.context.features:
        raise ValueError("context feature dictionary is empty")
    wrong = {
        name: tuple(value.shape)
        for name, value in batch.context.features.items()
        if value.shape[:2] != (batch_size, cfg.L_ctx)
    }
    if wrong:
        raise ValueError(f"context features must start with [B, L_ctx]; mismatches: {wrong}")
    pads = batch.context.ctx_pad
    if bool(((pads < 0) | (pads >= cfg.L_ctx)).any()):
        raise ValueError(f"ctx_pad must be in [0, {cfg.L_ctx}), got {pads.tolist()}")


def cache_validation(loader: Iterable[TrainBatch], n_samples: int) -> list[TrainBatch]:
    batches: list[TrainBatch] = []
    count = 0
    for batch in loader:
        if batch.target.shape[0] <= 0:
            raise RuntimeError("validation loader yielded an empty batch")
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


def eval_vs_cpu(
    model: GPT,
    stats: dict[str, FeatureStats],
    cfg: TrainConfig,
    *,
    n_matchups: int,
    replay_dir: Path,
) -> dict[str, float]:
    policies = itertools.count()
    with evaluation_mode(model):
        result = sweep_vs_cpu_prior(
            lambda: make_policy(model, stats, cfg, seed=cfg.seed + next(policies)),
            session_cfg=default_session_cfg(replay_dir, instant_match_restart=True),
            n_matchups=n_matchups,
            max_parallel=min(n_matchups, cfg.eval_max_parallel),
            max_frames=cfg.eval_max_frames,
        )
    return vs_cpu_metrics(result)


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
        tags=["gpt", "temporal-mtp", "chunk20"],
        config=asdict(cfg),
    )
    if cfg.wandb_log_code and wandb.run is not None:
        log_wandb_code(wandb.run)
    run_dir, replay_dir = setup_run_dir(run_name)
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high" if cfg.allow_tf32 else "highest")
    model = GPT(cfg).to(DEVICE)
    if cfg.compile_trunk and DEVICE == "cuda":
        model.forward = torch.compile(model.forward, dynamic=False)
    if cfg.compile_temporal and DEVICE == "cuda":
        model.temporal.teacher_forced_nll = torch.compile(
            model.temporal.teacher_forced_nll,
            dynamic=False,
        )
    optimizer = make_optimizer(model, cfg)
    scheduler = LambdaLR(optimizer, lr_schedule(cfg))
    start_step = 0
    if resume_state is not None:
        model.load_state_dict(resume_state["model"])
        optimizer.load_state_dict(resume_state["opt"])
        scheduler.load_state_dict(resume_state["sched"])
        start_step = int(resume_state["step"]) + 1

    kwargs = loader_kwargs(cfg, stats)
    if cfg.compact_data:
        train_loader = make_replay_reservoir_loader(
            split="train",
            num_workers=cfg.num_workers,
            prefetch_factor=cfg.prefetch_factor,
            predownload=cfg.predownload,
            windows_per_replay=cfg.windows_per_replay,
            reservoir_capacity=cfg.reservoir_capacity,
            batch_prefetch=cfg.train_batch_prefetch,
            batch_prefetch_depth=cfg.grad_accum_steps,
            **kwargs,
        )
    else:
        train_loader = make_loader(
            split="train",
            num_workers=cfg.num_workers,
            prefetch_factor=cfg.prefetch_factor,
            windows_per_replay=cfg.windows_per_replay,
            compact=False,
            **kwargs,
        )
    # Validation is deliberately eager and uses its own smaller batch geometry.
    # The training trunk compile is fixed-shape and must never see the sliced
    # final validation batch.
    val_kwargs = {**kwargs, "batch_size": cfg.val_batch_size}
    val_loader = make_loader(split=cfg.val_split, num_workers=0, compact=cfg.compact_data, **val_kwargs)
    val_cache = cache_validation(val_loader, cfg.val_n_samples)
    iterator = iter(train_loader)
    copy_stream = torch.cuda.Stream() if DEVICE == "cuda" else None
    run_started = time.monotonic()
    train_batches_seen = 0
    model.train()
    try:
        for step in range(start_step, cfg.max_steps):
            if DEVICE == "cuda":
                torch.cuda.reset_peak_memory_stats()
            with profile("step") as stopwatch:
                optimizer.zero_grad()
                cpu_batches: list[TrainBatch] = []
                loader_started = time.monotonic()
                for _ in range(cfg.grad_accum_steps):
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        iterator = iter(train_loader)
                        batch = next(iterator)
                    validate_batch_geometry(batch, cfg, expected_batch_size=micro_batch_size(cfg))
                    cpu_batches.append(batch)
                    train_batches_seen += 1
                loader_wait = time.monotonic() - loader_started
                valid_prefixes = sum(int((cfg.L_ctx - batch.context.ctx_pad).sum()) for batch in cpu_batches)
                if valid_prefixes <= 0:
                    raise RuntimeError("training accumulation contains no valid context prefixes")
                nll_sum = torch.zeros(cfg.L_chunk, N_GROUPS, device=DEVICE, dtype=torch.float32)
                n_prefixes = 0
                normalizer = valid_prefixes * cfg.L_chunk
                for batch in device_batches(cpu_batches, DEVICE, copy_stream):
                    with amp_context(cfg, DEVICE):
                        parts = action_loss(model, batch)
                        loss = parts.nll.sum() / normalizer
                    loss.backward()
                    nll_sum += parts.nll.detach().sum(dim=0)
                    n_prefixes += parts.nll.shape[0]
                if n_prefixes != valid_prefixes:
                    raise RuntimeError(f"decoded {n_prefixes} prefixes, expected {valid_prefixes}")
                gradients = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
                if not torch.isfinite(gradients):
                    raise FloatingPointError(f"step {step}: non-finite gradient norm {gradients}")
                gradient_log = (
                    wandb_gradient_log(model)
                    if wandb.run is not None and gradient_log_due(step, start_step, cfg.wandb_grad_every)
                    else {}
                )
                optimizer.step()
                scheduler.step()
                if DEVICE == "cuda":
                    torch.cuda.synchronize()
            # One small device-to-host transfer avoids one scalar synchronization
            # per horizon/group while constructing the logging dictionary.
            metrics = nll_mean_metrics((nll_sum / n_prefixes).cpu())
            log = {
                "global_step": step,
                "samples": (step + 1) * cfg.batch_size,
                **{f"train/{name}": value for name, value in metrics.items()},
                "train/grad_norm": float(gradients),
                **gradient_log,
                "lr/muon": next(group["lr"] for group in optimizer.param_groups if group["use_muon"]),
                "lr/adam": next(group["lr"] for group in optimizer.param_groups if not group["use_muon"]),
                "throughput/step_s": stopwatch.elapsed,
                "throughput/loader_wait_s": loader_wait,
                "throughput/samples_per_s": cfg.batch_size / stopwatch.elapsed,
                "throughput/prefixes_per_s": n_prefixes / stopwatch.elapsed,
                "data/train_batches_seen": train_batches_seen,
            }
            if DEVICE == "cuda":
                log["hardware/peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 2**30
                log["hardware/peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30
            wandb.log(log)
            if step == start_step:
                print(f"[model] attention path: {model.trunk.attn_path}, window={cfg.attn_window}", flush=True)
                if wandb.run is not None:
                    wandb.run.summary["model/attn_path"] = model.trunk.attn_path
                    wandb.run.summary["startup/compiled_step0_s"] = stopwatch.elapsed
            if step < 10 or step % 50 == 0:
                print(
                    f"[t+{time.monotonic() - run_started:.0f}s] step {step}: "
                    f"nll {metrics['loss']:.3f} bits/frame  dt={stopwatch.elapsed:.3f}s",
                    flush=True,
                )
            val_due = cfg.val_every > 0 and step > 0 and step % cfg.val_every == 0
            eval_due = cfg.eval_every > 0 and step > 0 and step % cfg.eval_every == 0
            periodic_ckpt_due = cfg.ckpt_every > 0 and step > 0 and step % cfg.ckpt_every == 0
            if periodic_ckpt_due or val_due or eval_due:
                save_checkpoint(
                    run_dir / "latest.pt",
                    step=step,
                    model=model,
                    opt=optimizer,
                    sched=scheduler,
                    cfg=asdict(cfg),
                    wandb_id=None if wandb.run is None else wandb.run.id,
                    uploader=uploader,
                )
            if val_due:
                values = val_metrics(model, val_cache, cfg)
                wandb.log({"global_step": step, **{f"val/{name}": value for name, value in values.items()}})
                print(f"[val] step {step}: {values}", flush=True)
            if eval_due:
                values = eval_vs_cpu(
                    model,
                    stats,
                    cfg,
                    n_matchups=cfg.eval_n_matchups,
                    replay_dir=replay_dir / f"step_{step:06d}",
                )
                wandb.log({"global_step": step, **{f"eval/{name}": value for name, value in values.items()}})

        # Save before either final evaluation loop: a simulator or driver crash
        # must not destroy the completed training result.
        save_checkpoint(
            run_dir / "final.pt",
            step=cfg.max_steps,
            model=model,
            opt=optimizer,
            sched=scheduler,
            cfg=asdict(cfg),
            wandb_id=None if wandb.run is None else wandb.run.id,
            uploader=uploader,
        )
        final_val = val_metrics(model, val_cache, cfg)
        wandb.log({"global_step": cfg.max_steps, **{f"val/{name}": value for name, value in final_val.items()}})
        final_eval = eval_vs_cpu(
            model,
            stats,
            cfg,
            n_matchups=cfg.final_eval_n_matchups,
            replay_dir=replay_dir / "final",
        )
        wandb.log({"global_step": cfg.max_steps, **{f"eval/{name}": value for name, value in final_eval.items()}})
    finally:
        if uploader is not None:
            uploader.upload_tree(replay_dir, base=run_dir)
            uploader.close()
        wandb.finish()


# %%
_CHECKPOINT_ARCH_FIELDS = {
    "decoder_arch_version",
    "temporal_d_model",
    "temporal_layers",
    "temporal_heads",
    "temporal_ff_dim",
    "group_head_dim",
    "main_stick_embed_dim",
    "c_stick_embed_dim",
    "trigger_embed_dim",
    "action_state_embed_dim",
}


def config_from_state(values: dict) -> TrainConfig:
    missing = _CHECKPOINT_ARCH_FIELDS - values.keys()
    if missing:
        raise ValueError(
            "checkpoint predates nonlinear causal decoder v2 and cannot be resumed or "
            f"loaded by experiment 024; missing config fields: {sorted(missing)}"
        )
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


def eval_checkpoint(
    path: str,
    *,
    exec_horizon: int | None = None,
    temperature: float | None = None,
    n_matchups: int | None = None,
) -> dict[str, float]:
    model, cfg, stats, state = load_checkpoint(path)
    if exec_horizon is not None:
        cfg = replace(cfg, exec_horizon=exec_horizon)
    if temperature is not None:
        cfg = replace(cfg, decode_temp=temperature)
    validate_config(cfg)
    replay_dir = Path(path).resolve().parent / "eval_replays"
    values = eval_vs_cpu(
        model,
        stats,
        cfg,
        n_matchups=cfg.final_eval_n_matchups if n_matchups is None else n_matchups,
        replay_dir=replay_dir,
    )
    print(f"[eval] step={state['step']} horizon={cfg.exec_horizon}: {values}", flush=True)
    return values


def synthetic_context(cfg: TrainConfig, batch_size: int, device: torch.device) -> Context:
    """A fixed-shape context for latency/throughput checks without touching the dataset."""
    features: dict[str, Tensor] = {}

    def float_zeros() -> Tensor:
        return torch.zeros(batch_size, cfg.L_ctx, device=device)

    def int_zeros() -> Tensor:
        return torch.zeros(batch_size, cfg.L_ctx, device=device, dtype=torch.long)

    for prefix in _PLAYER_PREFIXES:
        for name in FLOAT_FEATURES:
            features[f"{prefix}_{name}"] = float_zeros()
        for name in (*CAT_FEATURES, "action"):
            features[f"{prefix}_{name}"] = int_zeros()
    for name in ACTION_CHANNELS:
        features[f"ego_{name}"] = float_zeros()
    features["ego_character"] = int_zeros()
    features["opp_character"] = int_zeros()
    features["stage"] = int_zeros()
    return Context(features=features, ctx_pad=torch.zeros(batch_size, device=device, dtype=torch.long))


def measure_latency_ms(fn: Callable[[], object], *, warmup: int, iterations: int) -> np.ndarray:
    if warmup < 0 or iterations <= 0:
        raise ValueError("benchmark warmup must be non-negative and iterations must be positive")
    for _ in range(warmup):
        fn()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
        events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            events.append((start, end))
        torch.cuda.synchronize()
        return np.asarray([start.elapsed_time(end) for start, end in events], dtype=np.float64)
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1_000)
    return np.asarray(samples, dtype=np.float64)


def latency_summary(name: str, samples_ms: np.ndarray) -> dict[str, float | str]:
    return {
        "name": name,
        "median_ms": float(np.median(samples_ms)),
        "p95_ms": float(np.quantile(samples_ms, 0.95)),
        "min_ms": float(samples_ms.min()),
        "max_ms": float(samples_ms.max()),
    }


def run_benchmark(
    cfg: TrainConfig,
    *,
    warmup: int,
    iterations: int,
    train_warmup: int,
    train_iterations: int,
) -> None:
    """Benchmark the deployed four-frame path and an effective training step."""
    validate_config(cfg)
    device = torch.device(DEVICE)
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high" if cfg.allow_tf32 else "highest")

    inference_model = GPT(cfg).to(device).eval()
    if cfg.compile_trunk and device.type == "cuda":
        inference_model.forward = torch.compile(inference_model.forward, dynamic=False)
    context = synthetic_context(cfg, 1, device)
    with torch.no_grad(), amp_context(cfg, device):
        hidden = inference_model(context.features, context.ctx_pad)

    def trunk_only() -> Tensor:
        with torch.no_grad(), amp_context(cfg, device):
            return inference_model(context.features, context.ctx_pad)

    def decoder_only() -> Tensor:
        with torch.no_grad(), amp_context(cfg, device):
            return sample_chunk_from_hidden(
                inference_model,
                hidden,
                context.ctx_pad,
                cfg.exec_horizon,
                argmax=True,
                click_trigger_fix=cfg.decode_click_trigger_fix,
            )

    def full_decode() -> Tensor:
        with torch.no_grad(), amp_context(cfg, device):
            return decode_chunk(
                inference_model,
                context,
                cfg.exec_horizon,
                argmax=True,
                click_trigger_fix=cfg.decode_click_trigger_fix,
            )

    results = [
        latency_summary("trunk", measure_latency_ms(trunk_only, warmup=warmup, iterations=iterations)),
        latency_summary("decoder", measure_latency_ms(decoder_only, warmup=warmup, iterations=iterations)),
        latency_summary("full_decode", measure_latency_ms(full_decode, warmup=warmup, iterations=iterations)),
    ]
    full_p95 = float(results[-1]["p95_ms"])
    status = "target" if full_p95 <= 12.0 else "ceiling" if full_p95 <= 60.0 else "fail"
    print(
        {
            "device": str(device),
            "model": model_tag(cfg),
            "parameters": sum(parameter.numel() for parameter in inference_model.parameters()),
            "inference_status": status,
            "latency": results,
        },
        flush=True,
    )
    if full_p95 > 60.0 and device.type == "cuda":
        raise RuntimeError(f"four-frame inference p95 {full_p95:.2f} ms exceeds the 60 ms hard ceiling")
    if train_iterations <= 0 or device.type != "cuda":
        if train_iterations > 0:
            print({"name": "effective_train_step", "status": "skipped", "reason": "CUDA required"})
        return
    del inference_model, hidden
    if device.type == "cuda":
        torch.cuda.empty_cache()
    train_model = GPT(cfg).to(device).train()
    if cfg.compile_trunk and device.type == "cuda":
        train_model.forward = torch.compile(train_model.forward, dynamic=False)
    if cfg.compile_temporal and device.type == "cuda":
        train_model.temporal.teacher_forced_nll = torch.compile(
            train_model.temporal.teacher_forced_nll,
            dynamic=False,
        )
    optimizer = make_optimizer(train_model, cfg)
    micro = micro_batch_size(cfg)
    train_context = synthetic_context(cfg, micro, device)
    target = torch.zeros(micro, cfg.L_chunk, A_DIM, device=device)
    train_batch = TrainBatch(train_context, target)
    normalizer = cfg.batch_size * cfg.L_ctx * cfg.L_chunk

    def train_step() -> None:
        optimizer.zero_grad()
        for _ in range(cfg.grad_accum_steps):
            with amp_context(cfg, device):
                loss = action_loss(train_model, train_batch).nll.sum() / normalizer
            loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    train_samples = measure_latency_ms(train_step, warmup=train_warmup, iterations=train_iterations)
    train_result = latency_summary("effective_train_step", train_samples)
    train_result["throughput_status"] = "target" if float(train_result["median_ms"]) <= 600.0 else "regression"
    train_result["peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
    train_result["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30 if device.type == "cuda" else 0.0
    train_result["memory_status"] = "target" if float(train_result["peak_allocated_gb"]) <= 22.0 else "too_high"
    print(train_result, flush=True)


@dataclass
class Args:
    cfg: TrainConfig = field(default_factory=TrainConfig)
    eval: str | None = None
    eval_exec_horizon: int | None = None
    eval_temperature: float | None = None
    eval_n_matchups: int | None = None
    resume: str | None = None
    comment: str = ""
    benchmark: bool = False
    benchmark_warmup: int = 20
    benchmark_iterations: int = 100
    benchmark_train_warmup: int = 2
    benchmark_train_iterations: int = 5


def main(args: Args) -> None:
    if args.benchmark:
        if args.eval is not None or args.resume is not None:
            raise SystemExit("--benchmark uses --cfg and cannot be combined with --eval or --resume")
        run_benchmark(
            args.cfg,
            warmup=args.benchmark_warmup,
            iterations=args.benchmark_iterations,
            train_warmup=args.benchmark_train_warmup,
            train_iterations=args.benchmark_train_iterations,
        )
        return
    if args.eval is not None and args.resume is not None:
        raise SystemExit("pass only one of --eval or --resume")
    if args.eval is not None:
        eval_checkpoint(
            args.eval,
            exec_horizon=args.eval_exec_horizon,
            temperature=args.eval_temperature,
            n_matchups=args.eval_n_matchups,
        )
        return
    if args.resume is not None:
        state = load_for_resume(args.resume, Path("runs") / args.resume, device=DEVICE)
        if state is None:
            raise SystemExit(f"no latest.pt for run {args.resume!r}")
        cfg = config_from_state(state["cfg"])
        defaults = TrainConfig()
        cfg = replace(cfg, num_workers=defaults.num_workers, prefetch_factor=defaults.prefetch_factor)
        stats = load_consolidated_stats(Path(cfg.data_root) / "stats.json")
        train(cfg, stats, resume_run=args.resume, resume_state=state)
        return
    stats = load_consolidated_stats(Path(args.cfg.data_root) / "stats.json")
    train(args.cfg, stats, comment=args.comment or "causal-xattn-mtp20")


if __name__ == "__main__":
    main(tyro.cli(Args))
