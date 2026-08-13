"""Fast sparse-offset temporal controller model.

Experiment 024 remains the historical dense-20/cross-attention baseline.  This
experiment uses one structured controller codec everywhere, predicts ten
selected offsets, and only decodes the dense four- or six-frame prefix online.

Run:
    uv run experiments/026_temporal_mtp.py
    uv run experiments/026_temporal_mtp.py --eval runs/<run>/final.pt
"""

from __future__ import annotations

import contextlib
import hashlib
import itertools
import json
import math
import os
import platform
import threading
import time
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Mapping
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
from hal.eval.harness import DEFAULT_START_RETRIES
from hal.eval.harness import automatic_parallelism
from hal.eval.harness import default_session_cfg
from hal.eval.harness import resolve_parallelism
from hal.eval.harness import run_matches_vec
from hal.eval.harness import usable_cpus
from hal.eval.matchups import matchups_for_vs_cpu
from hal.sim.process_vec import ProcessVecTelemetry
from hal.sim.rollout import covering_power_of_two
from hal.sim.session import Matchup
from hal.sim.session import PlayerSetup
from hal.sim.vec import Slot
from hal.sim.vec import VecMatch
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
from hal.training.features import TrainBatch
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
from hal.wire import BUTTON_BITS
from hal.wire import libmelee_character_to_slp
from hal.wire import slp_stage_to_libmelee

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
    emit_all_masks: bool = False,
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
        emit_all_masks=emit_all_masks,
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
            dict(params=muon, lr=cfg.muon_lr, momentum=0.95, weight_decay=cfg.weight_decay, use_muon=True),
            dict(params=decay, lr=cfg.adam_lr, weight_decay=cfg.weight_decay, **adam),
            dict(params=no_decay, lr=cfg.adam_lr, weight_decay=0.0, **adam),
        ]
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
    offsets = "-".join(map(str, cfg.head_offsets))
    return (
        f"mtp026-d{cfg.d_model}-L{cfg.n_layers}-h{cfg.n_heads}-Lc{cfg.L_ctx}-"
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


def device_batches(
    cpu_batches: list[TrainBatch], device: str | torch.device, copy_stream: torch.cuda.Stream | None
) -> Iterator[TrainBatch]:
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
        for tensor in (*ready.context.features.values(), ready.context.ctx_pad, ready.target):
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
    if cfg.compact_data:
        train_loader = make_reservoir_loader(
            split="train",
            num_workers=cfg.num_workers,
            prefetch_factor=cfg.prefetch_factor,
            predownload=cfg.predownload,
            windows_per_replay=cfg.windows_per_replay,
            reservoir_capacity=cfg.reservoir_capacity,
            prefetch_batches=cfg.prefetch_batches,
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
    val_kwargs = {**kwargs, "batch_size": cfg.val_batch_size}
    val_loader = make_loader(split=cfg.val_split, num_workers=0, compact=cfg.compact_data, **val_kwargs)
    return train_loader, cache_validation(val_loader, cfg.val_n_samples)


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
        tags=["gpt", "temporal-mtp", "sparse-offset", "026"],
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
    start_step = 0
    if resume_state is not None:
        model.load_state_dict(resume_state["model"])
        optimizer.load_state_dict(resume_state["opt"])
        scheduler.load_state_dict(resume_state["sched"])
        start_step = int(resume_state["step"]) + 1

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
    # CUDA compilation must remain on the training thread. Background compilation
    # deadlocked training on both H100 and L40S hosts.
    eval_inference: BF16Inference | None = None
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
                validate_batch_geometry(batch, cfg, micro_batch_size(cfg))
                cpu_batches.append(batch)
            loader_wait = time.monotonic() - loader_started
            valid_prefixes = sum(int((cfg.L_ctx - batch.context.ctx_pad).sum()) for batch in cpu_batches)
            if valid_prefixes <= 0:
                raise RuntimeError("training accumulation contains no valid context prefixes")
            optimizer.zero_grad()
            nll_sum = torch.zeros(len(cfg.head_offsets), N_GROUPS, device=DEVICE)
            n_prefixes = 0
            with profile("step") as stopwatch:
                for batch in device_batches(cpu_batches, DEVICE, copy_stream):
                    history, targets, valid = prepared_targets(model, batch)
                    with amp_context(cfg, DEVICE):
                        hidden = trunk_fn(batch.context.features, batch.context.ctx_pad, history)
                        dense_nll = temporal_fn(hidden, history, targets)
                        parts = ActionLoss(nll=dense_nll[valid], targets=targets[valid])
                        joint_nll = parts.nll.sum(dim=-1)
                        primary = joint_nll[:, :4].sum() / (valid_prefixes * 4)
                        auxiliary = joint_nll[:, 4:].sum() / (valid_prefixes * (len(cfg.head_offsets) - 4))
                        loss = primary + cfg.aux_loss_weight * auxiliary
                    loss.backward()
                    nll_sum += parts.nll.detach().sum(dim=0)
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
                "train/grad_norm": float(gradient_norm),
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
                )
            if val_due:
                values = val_metrics(model, val_cache, cfg)
                wandb.log({"global_step": step, **{f"val/{name}": value for name, value in values.items()}})
            if eval_due:
                if eval_inference is None:
                    eval_inference = BF16Inference(model, cfg)
                values = eval_vs_cpu(
                    model,
                    stats,
                    cfg,
                    n_matchups=cfg.eval_n_matchups,
                    replay_dir=replay_dir / f"step_{step:06d}",
                    checkpoint_sha256=_checkpoint_sha256(checkpoint_path),
                    inference=eval_inference,
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
        )
        checkpoint_sha = _checkpoint_sha256(final_path)
        final_val = val_metrics(model, val_cache, cfg)
        wandb.log({"global_step": cfg.max_steps, **{f"val/{name}": value for name, value in final_val.items()}})
        if eval_inference is None:
            eval_inference = BF16Inference(model, cfg)
        final_eval = eval_vs_cpu(
            model,
            stats,
            cfg,
            n_matchups=cfg.final_eval_n_matchups,
            replay_dir=replay_dir / "final",
            checkpoint_sha256=checkpoint_sha,
            inference=eval_inference,
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
            inference=eval_inference,
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
    # CUDA graphs cannot safely own the trunk's mutable rotary cache across graph
    # partitions, so checkpoint evaluation uses ordinary compiled kernels.
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


def pack_slippi_actions(selected: Mapping[Slot, object], *, num_envs: int, action_dtype: np.dtype) -> np.ndarray:
    """Pack HAL controller values into SlippiVec's exact structured action wire."""
    expected = {Slot(match, port) for match in range(num_envs) for port in (1, 2)}
    if set(selected) != expected:
        raise ValueError(
            f"policy returned the wrong native slots: missing={sorted(expected - set(selected), key=repr)}, "
            f"extra={sorted(set(selected) - expected, key=repr)}"
        )
    actions = np.zeros((num_envs, 2), dtype=action_dtype)
    for slot, value in selected.items():
        analog = np.asarray(
            (
                value.main_x,
                value.main_y,
                value.c_x,
                value.c_y,
                value.trigger_l,
                value.trigger_r,
            ),
            dtype=np.float32,
        )
        buttons = int(value.buttons)
        if not np.isfinite(analog).all():
            raise ValueError(f"slot {slot} produced NaN/Inf controller analog values")
        if buttons < 0 or buttons > np.iinfo(np.uint16).max:
            raise ValueError(f"slot {slot} produced out-of-range button mask {buttons}")
        actions["analog"][slot.match, slot.port - 1] = analog
        actions["buttons"][slot.match, slot.port - 1] = buttons
    return actions


def _native_observation_map(observations: np.ndarray) -> dict[Slot, dict[str, float | int]]:
    if observations.ndim != 2 or observations.shape[1] != 2:
        raise ValueError(f"native observations must have shape (num_envs, 2), got {observations.shape}")
    if observations.dtype.names is None or "frame" not in observations.dtype.names:
        raise ValueError("native observations must be structured flat MDS rows containing frame")
    if observations[:, 0].tobytes() != observations[:, 1].tobytes():
        raise ValueError("the two policy slots in a native match received different match rows")
    return {
        Slot(match, port): {name: row[name].item() for name in observations.dtype.names}
        for match, row in enumerate(observations[:, 0])
        for port in (1, 2)
    }


def _first_array_difference(expected: np.ndarray, actual: np.ndarray) -> tuple[int, object, object] | None:
    if expected.shape != actual.shape:
        return min(expected.size, actual.size), expected.shape, actual.shape
    equal = expected == actual
    if expected.dtype.kind == "f" and actual.dtype.kind == "f":
        equal |= np.isnan(expected) & np.isnan(actual)
    differing = np.flatnonzero(~equal)
    if not differing.size:
        return None
    index = int(differing[0])
    return index, expected[index].item(), actual[index].item()


def _trim_replay_overlap(
    expected: np.ndarray, parsed: dict[str, np.ndarray]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Align a live reset-inclusive history with extract's in-game-only replay rows."""
    expected_frames = expected["frame"]
    actual_frames = parsed["frame"]
    if not len(expected_frames) or not len(actual_frames):
        return expected, parsed
    expected_at = {int(frame): index for index, frame in enumerate(expected_frames)}
    actual_at = {int(frame): index for index, frame in enumerate(actual_frames)}
    common = sorted(set(expected_at) & set(actual_at))
    if not common:
        return expected[:0], {name: value[:0] for name, value in parsed.items()}
    first, last = common[0], common[-1]
    expected_slice = slice(expected_at[first], expected_at[last] + 1)
    actual_slice = slice(actual_at[first], actual_at[last] + 1)
    return expected[expected_slice], {name: value[actual_slice] for name, value in parsed.items()}


def _validate_native_replay(path: Path, in_memory_rows: list[np.void]) -> dict[str, object]:
    """Parse a captured replay and exact-compare every schema-v5 field to live rows."""
    from hal.data.extract import extract_replay

    parsed = extract_replay(str(path))
    if parsed is None:
        return {
            "passed": False,
            "parsed_successfully": False,
            "frames": 0,
            "first_difference": {"event": "parse"},
        }
    if not in_memory_rows:
        return {
            "passed": False,
            "parsed_successfully": False,
            "frames": 0,
            "first_difference": {"event": "empty-history"},
        }
    expected = np.asarray(in_memory_rows, dtype=in_memory_rows[0].dtype)
    expected, parsed = _trim_replay_overlap(expected, parsed)
    if not len(expected) or not len(parsed["frame"]):
        return {
            "passed": False,
            "parsed_successfully": True,
            "frames": 0,
            "first_difference": {"event": "frame-overlap"},
        }
    first: dict[str, object] | None = None
    for field in expected.dtype.names or ():
        actual = parsed.get(field)
        if actual is None:
            first = {"event": "post/pre", "frame": None, "port": None, "field": field, "reason": "missing"}
            break
        difference = _first_array_difference(expected[field], actual)
        if difference is not None:
            index, wanted, got = difference
            port = int(field[1]) if field.startswith(("p1_", "p2_")) else None
            frame = int(expected["frame"][index]) if index < len(expected) else None
            first = {
                "event": "pre" if "button_" in field or "stick_" in field or "trigger_" in field else "post",
                "frame": frame,
                "port": port,
                "field": field,
                "expected": wanted,
                "actual": got,
            }
            break
    return {
        "passed": first is None,
        "parsed_successfully": True,
        "frames": int(len(expected)),
        "parsed_frames": int(len(parsed["frame"])),
        "first_difference": first,
    }


def _save_and_validate_native_replay(
    environment,
    *,
    env_index: int,
    path: Path,
    rows: list[np.void],
) -> dict[str, object]:
    environment.save_replay(path, env_index=env_index)
    live_path = path.with_suffix(".live.npy")
    np.save(live_path, np.asarray(rows, dtype=rows[0].dtype), allow_pickle=False)
    result = _validate_native_replay(path, rows)
    result |= {
        "path": str(path),
        "sha256": _checkpoint_sha256(path),
        "live_rows_path": str(live_path),
        "live_rows_sha256": _checkpoint_sha256(live_path),
        "env": env_index,
    }
    return result


def run_native_slippi_phase(
    environment,
    policy: RecedingHorizon,
    *,
    frames: int,
    output_dir: Path,
    phase: str,
    action_dtype: np.dtype,
    replay_stems: list[str] | None = None,
) -> dict[str, object]:
    """Drive one exact native phase, including terminal capture and selective reset."""
    if frames < 1:
        raise ValueError("native rollout frames must be positive")
    if replay_stems is not None and len(replay_stems) != environment.num_envs:
        raise ValueError("replay stems must have one entry per native environment")
    stems = replay_stems or [f"env-{index:03d}" for index in range(environment.num_envs)]
    output_dir.mkdir(parents=True, exist_ok=True)
    observations, _ = environment.reset()
    _native_observation_map(observations)
    histories: list[list[np.void]] = [[observations[i, 0].copy()] for i in range(environment.num_envs)]
    episode = [0] * environment.num_envs
    previous_frames = observations[:, 0]["frame"].astype(np.int64)
    non_neutral = np.zeros((environment.num_envs, 2), dtype=np.int64)
    policy_seconds = env_seconds = reset_seconds = replay_seconds = 0.0
    reset_count = 0
    replay_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for frame_index in range(frames):
        policy_started = time.perf_counter()
        selected = policy(frame_index, _native_observation_map(observations))
        actions = pack_slippi_actions(selected, num_envs=environment.num_envs, action_dtype=action_dtype)
        policy_seconds += time.perf_counter() - policy_started
        active = np.any(actions["analog"] != 0.0, axis=-1) | (actions["buttons"] != 0)
        non_neutral += active

        env_started = time.perf_counter()
        stepped, rewards, terminated, truncated, _ = environment.step(actions)
        env_seconds += time.perf_counter() - env_started
        _native_observation_map(stepped)
        rewards = np.asarray(rewards)
        terminated = np.asarray(terminated, dtype=bool)
        truncated = np.asarray(truncated, dtype=bool)
        expected_slots = 2 * environment.num_envs
        if (
            rewards.shape != (expected_slots,)
            or terminated.shape != (expected_slots,)
            or truncated.shape != (expected_slots,)
        ):
            raise ValueError("native reward and termination arrays must have one value per policy slot")
        if not np.array_equal(rewards[0::2], -rewards[1::2]):
            raise RuntimeError("native rewards were not antisymmetric within a match")
        if not np.array_equal(terminated[0::2], terminated[1::2]) or not np.array_equal(
            truncated[0::2], truncated[1::2]
        ):
            raise RuntimeError("native terminal flags disagreed across a match's two policy slots")
        current_frames = stepped[:, 0]["frame"].astype(np.int64)
        if not np.array_equal(current_frames, previous_frames + 1):
            raise RuntimeError(
                f"native observations did not advance monotonically: previous={previous_frames.tolist()} "
                f"current={current_frames.tolist()}"
            )
        for env_index in range(environment.num_envs):
            histories[env_index].append(stepped[env_index, 0].copy())

        done = terminated[0::2] | truncated[0::2]
        if done.any():
            for env_index in np.flatnonzero(done):
                replay_started = time.perf_counter()
                replay_rows.append(
                    _save_and_validate_native_replay(
                        environment,
                        env_index=int(env_index),
                        path=output_dir / f"{stems[env_index]}-episode-{episode[env_index]:03d}.slp",
                        rows=histories[env_index],
                    )
                )
                replay_seconds += time.perf_counter() - replay_started
                episode[env_index] += 1
            reset_started = time.perf_counter()
            reset_observations, _ = environment.reset(mask=done)
            reset_seconds += time.perf_counter() - reset_started
            for env_index in np.flatnonzero(~done):
                if reset_observations[env_index].tobytes() != stepped[env_index].tobytes():
                    raise RuntimeError(f"selective reset altered unrelated environment {env_index}")
            observations = reset_observations
            previous_frames = observations[:, 0]["frame"].astype(np.int64)
            for env_index in np.flatnonzero(done):
                histories[env_index] = [observations[env_index, 0].copy()]
            reset_count += int(done.sum())
        else:
            observations = stepped
            previous_frames = current_frames

    for env_index in range(environment.num_envs):
        replay_started = time.perf_counter()
        replay_rows.append(
            _save_and_validate_native_replay(
                environment,
                env_index=env_index,
                path=output_dir / f"{stems[env_index]}-episode-{episode[env_index]:03d}-partial.slp",
                rows=histories[env_index],
            )
        )
        replay_seconds += time.perf_counter() - replay_started
    wall_seconds = time.perf_counter() - started
    result = {
        "phase": phase,
        "frames_per_environment": frames,
        "aggregate_frames": frames * environment.num_envs,
        "wall_seconds": wall_seconds,
        "aggregate_fps": frames * environment.num_envs / max(wall_seconds, 1e-12),
        "per_environment_fps": frames / max(wall_seconds, 1e-12),
        "policy_seconds": policy_seconds,
        "environment_step_seconds": env_seconds,
        "reset_seconds": reset_seconds,
        "replay_capture_seconds": replay_seconds,
        "selective_resets": reset_count,
        "non_neutral_frames_by_slot": non_neutral.tolist(),
        "always_neutral_slots": [
            {"env": int(match), "port": int(port + 1)} for match, port in np.argwhere(non_neutral == 0)
        ],
        "replays": replay_rows,
    }
    result["replay_divergences"] = sum(not bool(row["passed"]) for row in replay_rows)
    result["replay_parse_failures"] = sum(not bool(row["parsed_successfully"]) for row in replay_rows)
    return result


class _NativeMatchupWave:
    """Present heterogeneous one-env Slippi hosts as one vector environment."""

    def __init__(
        self,
        matchups: list[tuple[melee.Character, melee.Character]],
        *,
        ciso: str | None,
        stage: int,
        seed: int,
        capture_replays: bool,
    ) -> None:
        from slippi_cuda import ACTION_DTYPE
        from slippi_cuda import SlippiVec

        if not matchups:
            raise ValueError("a native matchup wave cannot be empty")
        self.num_envs = len(matchups)
        self._matchups = matchups
        self._stage = slp_stage_to_libmelee(stage)
        self._action_dtype = ACTION_DTYPE
        self._executor = ThreadPoolExecutor(max_workers=self.num_envs, thread_name_prefix="slippi-matchup")
        self._environments = []
        self._last_observations: np.ndarray | None = None

        def construct(item: tuple[int, tuple[melee.Character, melee.Character]]):
            index, (p1, p2) = item

            def setup_character(character: melee.Character) -> int:
                # Synthetic StartMelee cannot select Sheik directly. Boot Zelda;
                # reset() performs a verified Down+B transform before publication.
                selected = melee.Character.ZELDA if character is melee.Character.SHEIK else character
                return libmelee_character_to_slp(selected)

            return SlippiVec(
                num_envs=1,
                ciso=ciso,
                stage=stage,
                p1_character=setup_character(p1),
                p2_character=setup_character(p2),
                seed=seed + index,
                capture_replays=capture_replays,
            )

        futures = [self._executor.submit(construct, item) for item in enumerate(matchups)]
        try:
            self._environments = [future.result() for future in futures]
        except Exception:
            self._executor.shutdown(wait=True, cancel_futures=True)
            for future in futures:
                if not future.cancelled():
                    with contextlib.suppress(Exception):
                        future.result().close(force=True)
            raise

    def _reset_one(self, index: int) -> np.ndarray:
        environment = self._environments[index]
        observations, _ = environment.reset()
        matchup = self._matchups[index]
        sheik_ports = [port for port, character in enumerate(matchup) if character is melee.Character.SHEIK]
        if not sheik_ports:
            return observations
        actions = np.zeros((1, 2), dtype=self._action_dtype)
        stable_sheik_frames = 0
        for transform_frame in range(360):
            transformed = all(
                int(observations[0, 0][f"p{port + 1}_character"]) == int(melee.Character.SHEIK.value)
                for port in sheik_ports
            )
            if transformed:
                stable_sheik_frames += 1
                actions[...] = np.zeros((), dtype=self._action_dtype)
                if stable_sheik_frames >= 60:
                    return observations
            else:
                stable_sheik_frames = 0
                actions[...] = np.zeros((), dtype=self._action_dtype)
                for port in sheik_ports:
                    actions["analog"][0, port, 1] = -1.0
                    actions["buttons"][0, port] = BUTTON_BITS["b"] if transform_frame % 2 == 0 else 0
            observations, *_ = environment.step(actions)
        found = [int(observations[0, 0][f"p{port + 1}_character"]) for port in sheik_ports]
        raise RuntimeError(f"native Zelda-to-Sheik transform did not stabilize: ports={sheik_ports}, found={found}")

    def reset(self, *, mask: np.ndarray | None = None):
        selected = np.ones(self.num_envs, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        if selected.shape != (self.num_envs,):
            raise ValueError(f"reset mask must have shape ({self.num_envs},)")
        if self._last_observations is None and not selected.all():
            raise RuntimeError("the first heterogeneous native reset must select every environment")
        pending = {
            index: self._executor.submit(self._reset_one, index) for index in range(self.num_envs) if selected[index]
        }
        observations = (
            np.empty((self.num_envs, 2), dtype=self._environments[0].single_observation_dtype)
            if self._last_observations is None
            else self._last_observations.copy()
        )
        for index, future in pending.items():
            observations[index] = future.result()[0]
            row = observations[index, 0]
            p1, p2 = self._matchups[index]
            observed = (int(row["p1_character"]), int(row["p2_character"]), int(row["stage"]))
            expected = (int(p1.value), int(p2.value), int(self._stage.value))
            if observed != expected:
                raise RuntimeError(f"native matchup {index} booted as {observed}, expected {expected}")
        self._last_observations = observations
        return observations, [{} for _ in range(2 * self.num_envs)]

    def step(self, actions):
        pending = [
            self._executor.submit(environment.step, actions[index : index + 1])
            for index, environment in enumerate(self._environments)
        ]
        results = [future.result() for future in pending]
        observations = np.concatenate([result[0] for result in results], axis=0)
        rewards = np.concatenate([result[1] for result in results])
        terminated = np.concatenate([result[2] for result in results])
        truncated = np.concatenate([result[3] for result in results])
        self._last_observations = observations
        return observations, rewards, terminated, truncated, [{} for _ in range(2 * self.num_envs)]

    def save_replay(self, path: Path, *, env_index: int) -> Path:
        return self._environments[env_index].save_replay(path)

    def close(self) -> None:
        pending = [self._executor.submit(environment.close) for environment in self._environments]
        for future in pending:
            with contextlib.suppress(Exception):
                future.result()
        self._executor.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _native_matchup_schedule(n_matchups: int) -> list[tuple[melee.Character, melee.Character]]:
    """Use the byte-identical matchup schedule from official closed-loop eval."""
    if n_matchups < 1:
        raise ValueError("native evaluation requires at least one matchup")
    return matchups_for_vs_cpu(n_matchups)


def _native_replay_stem(index: int, matchup: tuple[melee.Character, melee.Character]) -> str:
    p1, p2 = matchup
    return f"matchup-{index:03d}-{p1.name.lower()}-vs-{p2.name.lower()}"


def run_native_slippi_matchup_sweep(
    schedule: list[tuple[melee.Character, melee.Character]],
    policy_factory: Callable[[int], RecedingHorizon],
    *,
    max_parallel: int,
    frames: int,
    output_dir: Path,
    phase: str,
    ciso: str | None,
    stage: int,
    seed: int,
    action_dtype: np.dtype,
) -> dict[str, object]:
    """Run a prior-sampled character schedule in heterogeneous native waves."""
    if max_parallel < 1:
        raise ValueError("native maximum parallelism must be positive")
    results: list[dict[str, object]] = []
    matchup_activity: list[dict[str, object]] = []
    started = time.perf_counter()
    phase_dir = output_dir / phase
    for wave_start in range(0, len(schedule), max_parallel):
        wave = schedule[wave_start : wave_start + max_parallel]
        stems = [_native_replay_stem(wave_start + index, matchup) for index, matchup in enumerate(wave)]
        with _NativeMatchupWave(
            wave,
            ciso=ciso,
            stage=stage,
            seed=seed + wave_start,
            capture_replays=True,
        ) as environment:
            result = run_native_slippi_phase(
                environment,
                policy_factory(wave_start),
                frames=frames,
                output_dir=phase_dir,
                phase=phase,
                action_dtype=action_dtype,
                replay_stems=stems,
            )
        results.append(result)
        for local_index, matchup in enumerate(wave):
            p1, p2 = matchup
            matchup_activity.append(
                {
                    "index": wave_start + local_index,
                    "p1_character": p1.name,
                    "p2_character": p2.name,
                    "non_neutral_frames": result["non_neutral_frames_by_slot"][local_index],
                }
            )
    wall_seconds = time.perf_counter() - started
    aggregate_frames = frames * len(schedule)
    aggregate_fps = aggregate_frames / max(wall_seconds, 1e-12)
    replay_rows = [replay for result in results for replay in result["replays"]]
    return {
        "phase": phase,
        "matchups": len(schedule),
        "max_parallel": max_parallel,
        "frames_per_matchup": frames,
        "aggregate_frames": aggregate_frames,
        "wall_seconds": wall_seconds,
        "aggregate_fps": aggregate_fps,
        "per_active_environment_fps": aggregate_fps / min(max_parallel, len(schedule)),
        "policy_seconds": sum(float(result["policy_seconds"]) for result in results),
        "environment_step_seconds": sum(float(result["environment_step_seconds"]) for result in results),
        "reset_seconds": sum(float(result["reset_seconds"]) for result in results),
        "replay_capture_seconds": sum(float(result["replay_capture_seconds"]) for result in results),
        "selective_resets": sum(int(result["selective_resets"]) for result in results),
        "always_neutral_slots": [
            {"matchup": row["index"], "port": port + 1}
            for row in matchup_activity
            for port, count in enumerate(row["non_neutral_frames"])
            if count == 0
        ],
        "matchup_activity": matchup_activity,
        "replays": replay_rows,
        "replay_divergences": sum(not bool(row["passed"]) for row in replay_rows),
        "replay_parse_failures": sum(not bool(row["parsed_successfully"]) for row in replay_rows),
    }


def _native_env_kwargs(num_envs: int, ciso: str | None, *, capture_replays: bool) -> dict[str, object]:
    from slippi_cuda import Character
    from slippi_cuda import Stage

    return {
        "num_envs": num_envs,
        "ciso": ciso,
        "stage": Stage.FINAL_DESTINATION,
        "p1_character": Character.FOX,
        "p2_character": Character.FOX,
        "seed": 1,
        "capture_replays": capture_replays,
    }


def _native_engine_probe(num_envs: int, ciso: str | None, action_dtype: np.dtype) -> dict[str, object]:
    from slippi_cuda import SlippiVec

    warmup_frames, measured_frames = 64, 600
    with SlippiVec(**_native_env_kwargs(num_envs, ciso, capture_replays=False)) as environment:
        observations, _ = environment.reset()
        actions = np.zeros((num_envs, 2), dtype=action_dtype)
        for frame in range(warmup_frames + measured_frames):
            shoulder = (frame + np.arange(num_envs)[:, None]) % 3
            actions["analog"][..., 4] = (shoulder == 1) * (43.0 / 140.0)
            actions["analog"][..., 5] = (shoulder == 2) * (43.0 / 140.0)
            actions["buttons"] = np.where(
                shoulder == 1, BUTTON_BITS["l"], np.where(shoulder == 2, BUTTON_BITS["r"], 0)
            )
            if frame == warmup_frames:
                started = time.perf_counter()
            observations, _, terminated, truncated, _ = environment.step(actions)
            done = np.asarray(terminated)[0::2] | np.asarray(truncated)[0::2]
            if done.any():
                observations, _ = environment.reset(mask=done)
        elapsed = time.perf_counter() - started
    del observations
    return {
        "warmup_frames_per_environment": warmup_frames,
        "measured_frames_per_environment": measured_frames,
        "seconds": elapsed,
        "aggregate_fps": num_envs * measured_frames / elapsed,
        "per_environment_fps": measured_frames / elapsed,
    }


def _selective_reset_probe(num_envs: int, ciso: str | None, action_dtype: np.dtype) -> dict[str, object]:
    from slippi_cuda import SlippiVec

    with SlippiVec(**_native_env_kwargs(num_envs, ciso, capture_replays=False)) as environment:
        environment.reset()
        actions = np.zeros((num_envs, 2), dtype=action_dtype)
        before, *_ = environment.step(actions)
        mask = np.zeros(num_envs, dtype=bool)
        mask[0] = True
        after, _ = environment.reset(mask=mask)
    unchanged = [before[index].tobytes() == after[index].tobytes() for index in range(1, num_envs)]
    reset_dropped = int(after[0, 0]["frame"]) < int(before[0, 0]["frame"])
    if not all(unchanged) or not reset_dropped:
        raise RuntimeError("native selective-reset isolation probe failed")
    return {"reset_environment": 0, "unrelated_unchanged": unchanged, "reset_frame_dropped": reset_dropped}


class _GpuSampler:
    def __init__(self) -> None:
        self.samples: list[float] = []
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        def sample() -> None:
            try:
                import pynvml

                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
                while not self._stop.wait(0.1):
                    self.samples.append(float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu))
                pynvml.nvmlShutdown()
            except Exception as error:  # telemetry must not crash an otherwise valid rollout
                self.error = repr(error)

        self._thread = threading.Thread(target=sample, name="native-slippi-gpu-sampler", daemon=True)
        self._thread.start()

    def finish(self) -> dict[str, object]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        values = np.asarray(self.samples, dtype=np.float64)
        return {
            "samples": int(values.size),
            "mean_percent": float(values.mean()) if values.size else None,
            "p95_percent": float(np.percentile(values, 95)) if values.size else None,
            "max_percent": float(values.max()) if values.size else None,
            "error": self.error,
        }


def _hardware_evidence() -> dict[str, object]:
    model_name = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                model_name = line.partition(":")[2].strip()
                break
    except OSError:
        pass
    governors: dict[str, str] = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        path = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor")
        if path.is_file():
            governors[str(cpu)] = path.read_text().strip()
    cuda = None
    if torch.cuda.is_available():
        cuda = {
            "torch_cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "total_memory": torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory,
        }
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": model_name,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "cpu_governors": governors,
        "cuda": cuda,
    }


def _slippi_install_evidence() -> dict[str, object]:
    import importlib.metadata
    from urllib.parse import unquote
    from urllib.parse import urlparse

    from slippi_cuda.env import _default_host
    from slippi_cuda.env import _default_manifest

    distribution = importlib.metadata.distribution("slippi-cuda")
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    wheel_url = direct_url.get("url")
    wheel = Path(unquote(urlparse(wheel_url).path)) if isinstance(wheel_url, str) else None
    host = _default_host().resolve()
    manifest = _default_manifest().resolve()
    if os.environ.get("SC_ENV_HOST") or os.environ.get("SC_LIVE_MANIFEST"):
        raise RuntimeError("native evaluation refuses SC_ENV_HOST/SC_LIVE_MANIFEST source-tree overrides")
    if "site-packages" not in str(host) or host.parent.name != "_native":
        raise RuntimeError(f"native evaluation did not resolve the wheel-embedded host: {host}")
    return {
        "version": distribution.version,
        "direct_url": direct_url,
        "wheel": None if wheel is None else str(wheel),
        "wheel_sha256": None if wheel is None or not wheel.is_file() else _checkpoint_sha256(wheel),
        "host": str(host),
        "host_sha256": _checkpoint_sha256(host),
        "manifest": str(manifest),
        "manifest_sha256": _checkpoint_sha256(manifest),
        "source_overrides": False,
    }


def _profile_inference(inference: BF16Inference, cfg: TrainConfig, output_dir: Path, rows: int) -> dict[str, object]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    context = synthetic_context(cfg, rows, next(inference.model.parameters()).device)
    trace = output_dir / "inference_trace.json"
    table = output_dir / "inference_profile.txt"
    with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as profiler:
        for _ in range(6):
            inference.decode(context, cfg.exec_horizon)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(trace))
    sort_by = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
    table.write_text(profiler.key_averages().table(sort_by=sort_by, row_limit=40))
    return {"trace": str(trace), "table": str(table), "steps": 6, "real_rows": rows}


def native_slippi_checkpoint(
    path: str,
    *,
    num_envs: int = 6,
    frames: int = 3_000,
    n_matchups: int | None = None,
    ciso: str | None = None,
    output_name: str | None = None,
    replay_dir: str | None = None,
) -> dict[str, object]:
    """Run the official closed-loop matchup schedule through native self-play."""
    from slippi_cuda import ACTION_DTYPE
    from slippi_cuda import Stage

    if not torch.cuda.is_available():
        raise RuntimeError("native experiment-026 evaluation requires CUDA")
    if num_envs < 1 or frames < 64:
        raise ValueError("native evaluation requires at least one environment and 64 frames")
    checkpoint = Path(path).resolve()
    base_name = output_name or f"native_slippi_{num_envs}x{frames}"
    if Path(base_name).name != base_name or base_name in ("", ".", ".."):
        raise ValueError(f"native output name must be one directory name, got {base_name!r}")
    out_dir = checkpoint.parent / base_name
    suffix = 1
    while out_dir.exists():
        suffix += 1
        out_dir = checkpoint.parent / f"{base_name}_run{suffix:02d}"
    out_dir.mkdir(parents=True)
    replay_root = out_dir / "replays" if replay_dir is None else Path(replay_dir).expanduser().resolve()
    replay_root.mkdir(parents=True, exist_ok=False)
    model, cfg, stats, state = load_checkpoint(str(checkpoint))
    matchup_count = cfg.final_eval_n_matchups if n_matchups is None else n_matchups
    schedule = _native_matchup_schedule(matchup_count)
    pairs, p1_characters, p2_characters, schedule_sha = assert_protocol_diversity(matchup_count)
    schedule_rows = [
        {"index": index, "p1_character": p1.name, "p2_character": p2.name} for index, (p1, p2) in enumerate(schedule)
    ]
    payload: dict[str, object] = {
        "schema_version": 2,
        "configuration": {
            "checkpoint": str(checkpoint),
            "matchups": matchup_count,
            "oriented_pairs": pairs,
            "p1_characters": p1_characters,
            "p2_characters": p2_characters,
            "max_parallel_environments": num_envs,
            "max_policy_slots": 2 * num_envs,
            "stage": "BATTLEFIELD",
            "seed": cfg.eval_seed,
            "requested_frames_per_matchup": frames,
            "inference": "compiled-bfloat16",
            "compiled_bucket": 16,
            "replay_directory": str(replay_root),
        },
        "hashes": {
            "checkpoint_sha256": _checkpoint_sha256(checkpoint),
            "matchup_schedule_sha256": schedule_sha,
        },
        "matchup_schedule": schedule_rows,
        "hardware": _hardware_evidence(),
        "slippi_cuda": _slippi_install_evidence(),
        "failures": [],
        "phases": [],
    }

    def write_evidence() -> None:
        temporary = out_dir / "evidence.json.tmp"
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
        temporary.replace(out_dir / "evidence.json")

    write_evidence()
    sampler = _GpuSampler()
    sampler.start()
    try:
        cfg = replace(cfg, inference_mode="compiled", compiled_inference_bucket=16)
        validate_config(cfg)
        model.eval()
        inference = BF16Inference(model, cfg, compile_mode="default", compiled_buckets=(16,))
        compile_started = time.perf_counter()
        prewarm_context = synthetic_context(cfg, 16, next(model.parameters()).device)
        inference.decode(prewarm_context, cfg.exec_horizon)
        inference.decode(prewarm_context, cfg.exec_horizon)
        torch.cuda.synchronize()
        compile_seconds = time.perf_counter() - compile_started
        torch.cuda.reset_peak_memory_stats()
        payload["configuration"] |= {"checkpoint_step": int(state["step"]), "exec_horizon": cfg.exec_horizon}
        payload["timings"] = {"compile_seconds": compile_seconds}
        payload["profile"] = _profile_inference(inference, cfg, out_dir, 2 * num_envs)
        payload["engine_only"] = _native_engine_probe(num_envs, ciso, ACTION_DTYPE)
        payload["selective_reset_probe"] = _selective_reset_probe(num_envs, ciso, ACTION_DTYPE)
        write_evidence()

        targets = [("smoke", 64), ("measured", 600)]
        for phase_index, (phase, phase_frames) in enumerate(targets):
            telemetry = DecodeTelemetry()
            seed_base = cfg.eval_seed + phase_index * matchup_count

            def policy_factory(wave_start: int, *, seed_base: int = seed_base, telemetry=telemetry):
                return make_policy(
                    model,
                    stats,
                    cfg,
                    decode_seed=seed_base + wave_start,
                    inference=inference,
                    telemetry=telemetry,
                    emit_all_masks=True,
                )

            phase_result = run_native_slippi_matchup_sweep(
                schedule,
                policy_factory,
                max_parallel=num_envs,
                frames=phase_frames,
                output_dir=replay_root,
                phase=phase,
                ciso=ciso,
                stage=int(Stage.BATTLEFIELD),
                seed=cfg.eval_seed,
                action_dtype=ACTION_DTYPE,
            )
            phase_result["decode"] = telemetry.metrics()
            payload["phases"].append(phase_result)
            write_evidence()
        gate_failures = [
            f"{phase['phase']}: replay-parse-failures={phase['replay_parse_failures']}"
            for phase in payload["phases"]
            if phase["replay_parse_failures"]
        ]
        measured = payload["phases"][-1]
        if measured["always_neutral_slots"]:
            gate_failures.append(f"measured: always-neutral={measured['always_neutral_slots']}")
        if gate_failures:
            raise RuntimeError("native smoke/measured gate failed: " + "; ".join(gate_failures))
        if frames > 600:
            telemetry = DecodeTelemetry()
            phase_result = run_native_slippi_matchup_sweep(
                schedule,
                lambda wave_start: make_policy(
                    model,
                    stats,
                    cfg,
                    decode_seed=cfg.eval_seed + 2 * matchup_count + wave_start,
                    inference=inference,
                    telemetry=telemetry,
                    emit_all_masks=True,
                ),
                max_parallel=num_envs,
                frames=frames,
                output_dir=replay_root,
                phase="extended",
                ciso=ciso,
                stage=int(Stage.BATTLEFIELD),
                seed=cfg.eval_seed,
                action_dtype=ACTION_DTYPE,
            )
            phase_result["decode"] = telemetry.metrics()
            payload["phases"].append(phase_result)
            write_evidence()
            if phase_result["replay_parse_failures"] or phase_result["always_neutral_slots"]:
                raise RuntimeError("native extended gate failed")
        payload["cuda_memory"] = {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
    except Exception as error:
        payload["failures"].append({"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        if torch.cuda.is_available():
            payload["cuda_memory"] = {
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
        payload["gpu_utilization"] = sampler.finish()
        write_evidence()
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return payload


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
    slippi_eval: str | None = None
    slippi_num_envs: int = 6
    slippi_frames: int = 3_000
    slippi_matchups: int | None = None
    slippi_ciso: str | None = None
    slippi_output_name: str | None = None
    slippi_replay_dir: str | None = None
    benchmark: bool = False
    benchmark_iterations: int = 20


def main(args: Args) -> None:
    if args.benchmark:
        if any(value is not None for value in (args.eval, args.resume, args.self_play_eval, args.slippi_eval)):
            raise SystemExit(
                "--benchmark cannot be combined with --eval, --self-play-eval, --slippi-eval, or --resume"
            )
        run_benchmark(args.cfg, iterations=args.benchmark_iterations)
        return
    selected = sum(value is not None for value in (args.eval, args.self_play_eval, args.slippi_eval, args.resume))
    if selected > 1:
        raise SystemExit("pass only one of --eval, --self-play-eval, --slippi-eval, or --resume")
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
    if args.slippi_eval is not None:
        native_slippi_checkpoint(
            args.slippi_eval,
            num_envs=args.slippi_num_envs,
            frames=args.slippi_frames,
            n_matchups=args.slippi_matchups,
            ciso=args.slippi_ciso,
            output_name=args.slippi_output_name,
            replay_dir=args.slippi_replay_dir,
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
