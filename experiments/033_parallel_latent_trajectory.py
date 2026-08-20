"""One-pass parallel latent-trajectory policy on the experiment-026 trunk.

This is a direct, independently runnable copy of experiment 026.  The data,
observation trunk, optimizer, sparse future offsets, and receding-horizon
closed-loop protocol stay fixed.  Only the future-action decoder and its hard
best-of-K objective change: all offset/group slots are decoded non-causally in
one Flash-SDPA pass, with K folded into the batch dimension.

Run:
    uv run experiments/033_parallel_latent_trajectory.py --cfg.trajectory-modes 1
    uv run experiments/033_parallel_latent_trajectory.py --cfg.trajectory-modes 8
    uv run experiments/033_parallel_latent_trajectory.py --eval runs/<run>/final.pt \
        --eval-sampling-mode per-slot-k
"""

from __future__ import annotations

import contextlib
import hashlib
import itertools
import json
import math
import time
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
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
from hal.training.trunk import Trunk
from hal.training.trunk import TrunkConfig

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

    decoder_arch_version: int = 5
    sample_chunk_length: int = 20
    head_offsets: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 9, 12, 16, 20)
    trajectory_modes: int = 8
    temporal_d_model: int = 128
    temporal_layers: int = 2
    temporal_heads: int = 4
    temporal_ff_dim: int = 256
    group_head_dim: int = 256
    action_embed_dim: int = 16
    # Retained for direct config comparison/checkpoint readability with 026.
    # Parallel offset queries themselves are temporal_d_model-wide.
    offset_embed_dim: int = 16
    aux_loss_weight: float = 1.0
    group_order: tuple[str, ...] = GROUP_ORDER
    sampling_mode: str = "shared_k"
    attention_backend: str = "torch_sdpa_flash"
    diagnostic_histories: int = 128
    trajectory_samples_per_mode: int = 4
    joint_diagnostic_samples: int = 32

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
        "trajectory_modes": cfg.trajectory_modes,
        "temporal_d_model": cfg.temporal_d_model,
        "temporal_layers": cfg.temporal_layers,
        "temporal_heads": cfg.temporal_heads,
        "temporal_ff_dim": cfg.temporal_ff_dim,
        "group_head_dim": cfg.group_head_dim,
        "action_embed_dim": cfg.action_embed_dim,
        "offset_embed_dim": cfg.offset_embed_dim,
        "diagnostic_histories": cfg.diagnostic_histories,
        "trajectory_samples_per_mode": cfg.trajectory_samples_per_mode,
        "joint_diagnostic_samples": cfg.joint_diagnostic_samples,
        "batch_size": cfg.batch_size,
        "grad_accum_steps": cfg.grad_accum_steps,
        "max_steps": cfg.max_steps,
    }
    for name, value in positive.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if cfg.decoder_arch_version != 5:
        raise ValueError(f"unsupported decoder_arch_version={cfg.decoder_arch_version}")
    if cfg.d_model % cfg.n_heads or cfg.temporal_d_model % cfg.temporal_heads:
        raise ValueError("model dimensions must be divisible by their head counts")
    offsets = tuple(cfg.head_offsets)
    if offsets != tuple(sorted(set(offsets))) or not offsets or offsets[0] != 1:
        raise ValueError(f"head_offsets must be sorted, unique, and start at 1, got {offsets}")
    if offsets[-1] > cfg.sample_chunk_length:
        raise ValueError("head_offsets extend beyond sample_chunk_length")
    if offsets[:6] != (1, 2, 3, 4, 5, 6):
        raise ValueError("the live four/six-frame decoders require a dense 1..6 prefix")
    if cfg.group_order != GROUP_ORDER:
        raise ValueError(f"group_order must be {GROUP_ORDER}, got {cfg.group_order}")
    if cfg.trajectory_modes not in (1, 2, 4, 8, 16):
        raise ValueError("trajectory_modes must be one of 1, 2, 4, 8, or 16")
    if cfg.sampling_mode not in ("shared_k", "per_frame_k", "per_slot_k"):
        raise ValueError("sampling_mode must be shared_k, per_frame_k, or per_slot_k")
    if cfg.attention_backend != "torch_sdpa_flash":
        raise ValueError("experiment 033 requires the torch_sdpa_flash attention backend")
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
    if cfg.aux_loss_weight != 1.0:
        raise ValueError("hard full-trajectory best-of-K requires aux_loss_weight=1")
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


PARALLEL_SDPA_BATCH_LIMIT = 32_768


class ParallelBlock(nn.Module):
    """Full-attention Flash-SDPA block over the T*G trajectory slots."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.n_heads = cfg.temporal_heads
        self.d_model = cfg.temporal_d_model
        self.head_dim = self.d_model // self.n_heads
        self.scale = 1.0 / math.sqrt(2 * cfg.temporal_layers)
        self.qkv = nn.Linear(self.d_model, 3 * self.d_model, bias=False)
        self.proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.up = nn.Linear(self.d_model, cfg.temporal_ff_dim, bias=False)
        self.down = nn.Linear(cfg.temporal_ff_dim, self.d_model, bias=False)

    def _forward_chunk(self, x: Tensor) -> Tensor:
        batch, length, _ = x.shape
        q, k, v = self.qkv(decoder_rmsnorm(x)).split(self.d_model, dim=-1)
        shape = (batch, length, self.n_heads, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        # No mask and is_causal=False are deliberate.  On CUDA BF16 this is
        # PyTorch's fused FlashAttention SDPA path used elsewhere in the repo.
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        attended = attended.transpose(1, 2).contiguous().view_as(x)
        x = x + self.scale * self.proj(attended)
        return x + self.down(F.silu(self.up(decoder_rmsnorm(x))))

    def forward(self, x: Tensor) -> Tensor:
        # K is already part of this independent batch axis.  Chunking only that
        # axis cannot create cross-candidate attention and avoids CUDA grid
        # limits without a Python loop over modes or repeated decoder calls.
        if x.shape[0] <= PARALLEL_SDPA_BATCH_LIMIT:
            return self._forward_chunk(x)
        return torch.cat([self._forward_chunk(chunk) for chunk in x.split(PARALLEL_SDPA_BATCH_LIMIT)], dim=0)


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


class ParallelActionDecoder(nn.Module):
    """One non-causal decoder call for every K*T*G output query.

    Targets and observed actions never enter :meth:`forward`.  Each candidate
    is an independent batch item inside every block, while its T*G slots share
    full attention.
    """

    def __init__(self, cfg: TrainConfig, codec: StructuredControllerCodec) -> None:
        super().__init__()
        self.codec = codec
        self.head_offsets = tuple(cfg.head_offsets)
        self.K = cfg.trajectory_modes
        self.T = len(self.head_offsets)
        self.G = N_GROUPS
        self.d_model = cfg.temporal_d_model
        self.context_projection = nn.Linear(cfg.d_model, self.d_model)
        self.mode_embedding = nn.Embedding(self.K, self.d_model)
        self.offset_embedding = nn.Embedding(cfg.sample_chunk_length + 1, self.d_model)
        self.group_embedding = nn.Embedding(self.G, self.d_model)
        self.blocks = nn.ModuleList([ParallelBlock(cfg) for _ in range(cfg.temporal_layers)])
        self.outputs = nn.ModuleDict(
            {
                name: NonlinearActionHead(self.d_model, cfg.group_head_dim, GROUP_VOCABS[GROUP_INDEX[name]])
                for name in GROUP_NAMES
            }
        )
        self.register_buffer("offset_ids", torch.tensor(self.head_offsets), persistent=False)
        self.register_buffer("mode_ids", torch.arange(self.K), persistent=False)
        self.register_buffer("group_ids", torch.arange(self.G), persistent=False)

    def forward(self, hidden: Tensor) -> dict[str, Tensor]:
        prefix_shape = hidden.shape[:-1]
        context = self.context_projection(decoder_rmsnorm(hidden)).reshape(-1, 1, 1, 1, self.d_model)
        mode = self.mode_embedding(self.mode_ids)[None, :, None, None, :]
        offset = self.offset_embedding(self.offset_ids)[None, None, :, None, :]
        group = self.group_embedding(self.group_ids)[None, None, None, :, :]
        # The broadcast sum materializes exactly the decoder input once.  No
        # candidate loop or candidate-to-candidate attention exists.
        x = (context + mode + offset + group).reshape(-1, self.T * self.G, self.d_model)
        for block in self.blocks:
            x = block(x)
        x = decoder_rmsnorm(x).view(*prefix_shape, self.K, self.T, self.G, self.d_model)
        return {name: self.outputs[name](x[..., group_index, :]) for group_index, name in enumerate(GROUP_NAMES)}

    def teacher_forced_logits_by_group(
        self, hidden: Tensor, observed: Tensor | None = None, targets: Tensor | None = None
    ) -> dict[str, Tensor]:
        del observed, targets
        return self(hidden)

    def teacher_forced_nll(self, hidden: Tensor, observed: Tensor, targets: Tensor) -> Tensor:
        del observed
        return parallel_nll(self(hidden), targets)

    def teacher_forced_logits(self, hidden: Tensor, observed: Tensor, targets: Tensor) -> list[dict[str, Tensor]]:
        values = self.teacher_forced_logits_by_group(hidden, observed, targets)
        return [{name: logits[..., depth, :] for name, logits in values.items()} for depth in range(self.T)]

    def forced_stepwise_logits(self, hidden: Tensor, observed: Tensor, targets: Tensor) -> list[dict[str, Tensor]]:
        del observed, targets
        values = self(hidden[:, -1])
        return [{name: logits[..., depth, :] for name, logits in values.items()} for depth in range(self.T)]

    def sample_indices(
        self,
        hidden: Tensor,
        observed: Tensor,
        offsets: tuple[int, ...],
        *,
        argmax: bool,
        uniforms: Tensor | None = None,
        mode_uniforms: Tensor | None = None,
        sampling_mode: str = "shared_k",
        gen: torch.Generator | None = None,
    ) -> Tensor:
        del observed
        if offsets not in (self.head_offsets[:4], self.head_offsets[:6]):
            raise ValueError("live decode may execute only the dense four- or six-offset prefix")
        logits = self(hidden[:, -1])
        return sample_parallel_logits(
            logits,
            len(offsets),
            sampling_mode=sampling_mode,
            argmax=argmax,
            uniforms=uniforms,
            mode_uniforms=mode_uniforms,
            gen=gen,
        )


def parallel_nll(logits: dict[str, Tensor], targets: Tensor) -> Tensor:
    """Return per-factor NLL with shape ``[..., K, T, G]``."""
    prefix = targets.shape[:-2]
    horizon = targets.shape[-2]
    losses: list[Tensor] = []
    for group, name in enumerate(GROUP_NAMES):
        group_logits = logits[name]
        if group_logits.shape[:-3] != prefix or group_logits.shape[-2] != horizon:
            raise ValueError(f"{name} logits and targets have incompatible shapes")
        K = group_logits.shape[-3]
        selected = targets[..., group].unsqueeze(-2).expand(*prefix, K, horizon)
        losses.append(-group_logits.float().log_softmax(-1).gather(-1, selected[..., None]).squeeze(-1))
    return torch.stack(losses, dim=-1)


def best_of_k_loss(logits: dict[str, Tensor], targets: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Hard full-trajectory best-of-K with no loop over candidate modes."""
    per_factor = parallel_nll(logits, targets)
    loss_per_mode = per_factor.sum(dim=(-1, -2))
    best_loss, best_k = loss_per_mode.min(dim=-1)
    gather = best_k[..., None, None, None].expand(*best_k.shape, 1, targets.shape[-2], N_GROUPS)
    best_nll = per_factor.gather(-3, gather).squeeze(-3)
    return best_loss.mean(), best_k, loss_per_mode, best_nll


def mode_indices_from_uniforms(mode_uniforms: Tensor, K: int, sampling_mode: str) -> Tensor:
    """Map one [T,G,B] random table to [B,T,G] candidate indices."""
    if mode_uniforms.ndim != 3:
        raise ValueError("mode uniforms must be [frames, groups, batch]")
    raw = (mode_uniforms.permute(2, 0, 1).clamp(0, 1 - torch.finfo(mode_uniforms.dtype).eps) * K).long()
    if sampling_mode == "shared_k":
        return raw[:, :1, :1].expand_as(raw)
    if sampling_mode == "per_frame_k":
        return raw[:, :, :1].expand_as(raw)
    if sampling_mode == "per_slot_k":
        return raw
    raise ValueError(f"unknown sampling mode {sampling_mode!r}")


def sample_parallel_logits(
    logits: dict[str, Tensor],
    horizon: int,
    *,
    sampling_mode: str,
    argmax: bool,
    uniforms: Tensor | None = None,
    mode_uniforms: Tensor | None = None,
    gen: torch.Generator | None = None,
) -> Tensor:
    """Sample a trajectory with shared, per-frame, or per-slot uniform modes."""
    sample_logits = logits[GROUP_NAMES[0]]
    if sample_logits.ndim != 4:
        raise ValueError("sampling logits must be [batch, modes, offsets, vocab]")
    batch, K, T = sample_logits.shape[:3]
    if not 1 <= horizon <= T:
        raise ValueError("sampling horizon exceeds predicted offsets")
    expected = (horizon, N_GROUPS, batch)
    if uniforms is None:
        uniforms = torch.rand(expected, device=sample_logits.device, generator=gen)
    if mode_uniforms is None:
        mode_uniforms = torch.rand(expected, device=sample_logits.device, generator=gen)
    if uniforms.shape != expected or mode_uniforms.shape != expected:
        raise ValueError(f"uniform tables must both be {expected}")
    modes = mode_indices_from_uniforms(mode_uniforms, K, sampling_mode)
    picks: list[Tensor] = []
    for group, name in enumerate(GROUP_NAMES):
        # [B,K,T,V] -> [B,T,K,V], followed by one gather along K.
        candidates = logits[name][:, :, :horizon].permute(0, 2, 1, 3)
        index = modes[:, :, group, None, None].expand(batch, horizon, 1, candidates.shape[-1])
        selected = candidates.gather(2, index).squeeze(2)
        picks.append(
            sample_categorical(
                selected,
                argmax=argmax,
                uniform=uniforms[:, group].transpose(0, 1),
                gen=gen,
            )
        )
    return torch.stack(picks, dim=-1)


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
        self.temporal = ParallelActionDecoder(cfg, self.codec)
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
    best_k: Tensor  # [valid prefixes]
    best_loss: Tensor  # [valid prefixes], summed over offsets and groups
    loss_per_mode: Tensor  # [valid prefixes, K]


def action_loss(model: GPT, batch: TrainBatch) -> ActionLoss:
    history_indices, targets, valid = prepared_targets(model, batch)
    hidden = model(batch.context.features, batch.context.ctx_pad, history_indices)
    logits = model.temporal(hidden)
    _, dense_best_k, dense_loss_per_mode, dense_nll = best_of_k_loss(logits, targets)
    nll = dense_nll[valid]
    target_valid = targets[valid]
    if nll.numel() == 0:
        raise ValueError("batch contains no valid context prefixes")
    loss_per_mode = dense_loss_per_mode[valid]
    return ActionLoss(
        nll=nll,
        targets=target_valid,
        best_k=dense_best_k[valid],
        best_loss=loss_per_mode.min(dim=-1).values,
        loss_per_mode=loss_per_mode,
    )


def objective(parts: ActionLoss, aux_loss_weight: float = 1.0) -> Tensor:
    """Hard full-trajectory best-of-K, scaled to experiment-026 loss magnitude.

    The constant ``2/T`` does not change the winning mode.  It matches 026's
    primary-plus-auxiliary scale when per-offset NLL is stationary, retaining
    the optimizer/LR setup without changing the requested complete-trajectory
    winner selection.
    """
    if aux_loss_weight != 1.0:
        raise ValueError("experiment 033 does not reweight subsets of the trajectory")
    return parts.best_loss.mean() * (2.0 / parts.nll.shape[-2])


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


def mode_utilization_metrics(best_k: Tensor, K: int) -> dict[str, float]:
    counts = torch.bincount(best_k.reshape(-1).cpu(), minlength=K).double()
    usage = counts / counts.sum().clamp_min(1)
    nonzero = usage > 0
    entropy = -(usage[nonzero] * usage[nonzero].log()).sum()
    out = {
        "mode_entropy": float(entropy),
        "effective_modes": float(entropy.exp()),
        "inactive_modes": float((counts == 0).sum()),
        "modes_below_1pct": float((usage < 0.01).sum()),
        "fraction_modes_below_1pct": float((usage < 0.01).double().mean()),
    }
    out.update({f"mode_usage_k{k}": float(usage[k]) for k in range(K)})
    out.update({f"mode_win_count_k{k}": float(counts[k]) for k in range(K)})
    return out


def mode_utilization_histogram(metrics: dict[str, float], K: int) -> wandb.Histogram:
    counts = np.asarray([metrics[f"mode_win_count_k{k}"] for k in range(K)], dtype=np.float64)
    bins = np.arange(K + 1, dtype=np.float64) - 0.5
    return wandb.Histogram(np_histogram=(counts, bins))


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


def sample_trajectories_per_mode(logits: dict[str, Tensor], n_samples: int, *, gen: torch.Generator) -> Tensor:
    """Return independent action samples as [histories,K,samples,T,G]."""
    parts: list[Tensor] = []
    for name in GROUP_NAMES:
        values = logits[name].float()
        shape = values.shape[:3]
        picks = torch.multinomial(
            values.softmax(-1).reshape(-1, values.shape[-1]), n_samples, replacement=True, generator=gen
        )
        parts.append(picks.view(*shape, n_samples).permute(0, 1, 3, 2))
    return torch.stack(parts, dim=-1)


def mode_distance_metrics(samples: Tensor) -> dict[str, float]:
    if samples.ndim != 5 or samples.shape[2] < 2:
        raise ValueError("mode-distance samples must be [N,K,S,T,G] with S>=2")
    K, S = samples.shape[1:3]
    distance = (samples[:, :, :, None, None] != samples[:, None, None, :, :]).float().mean(dim=(-1, -2))
    mode = torch.arange(K)
    draw = torch.arange(S)
    within_mask = (mode[:, None, None, None] == mode[None, None, :, None]) & (
        draw[None, :, None, None] != draw[None, None, None, :]
    )
    between_mask = mode[:, None, None, None] != mode[None, None, :, None]
    within = distance[:, within_mask].mean()
    between = distance[:, between_mask.expand(K, S, K, S)].mean() if K > 1 else within
    return {
        "mode_distance_within": float(within),
        "mode_distance_between": float(between),
        "mode_distance_ratio": float(between / (within + 1e-8)),
    }


def repeated_sampler_samples(
    logits: dict[str, Tensor], n_samples: int, sampling_mode: str, *, gen: torch.Generator
) -> Tensor:
    batch = next(iter(logits.values())).shape[0]
    horizon = next(iter(logits.values())).shape[2]
    rows = []
    for _ in range(n_samples):
        uniforms = torch.rand(horizon, N_GROUPS, batch, generator=gen)
        mode_uniforms = torch.rand(horizon, N_GROUPS, batch, generator=gen)
        rows.append(
            sample_parallel_logits(
                logits,
                horizon,
                sampling_mode=sampling_mode,
                argmax=False,
                uniforms=uniforms,
                mode_uniforms=mode_uniforms,
            )
        )
    return torch.stack(rows, dim=1)


def _collision_dependence(collision: Tensor, slot_pairs: list[tuple[int, int, int, int]]) -> float:
    values = []
    for t1, g1, t2, g2 in slot_pairs:
        first = collision[..., t1, g1].float()
        second = collision[..., t2, g2].float()
        values.append((first * second).mean() - first.mean() * second.mean())
    return float(torch.stack(values).mean()) if values else 0.0


def sampler_dependence_metrics(samples: Tensor, offsets: tuple[int, ...]) -> dict[str, float]:
    """Simple collision dependence and temporal agreement over repeated draws."""
    n_samples = samples.shape[1]
    first, second = torch.triu_indices(n_samples, n_samples, offset=1)
    collision = samples[:, first] == samples[:, second]
    same_frame = [
        (t, g1, t, g2) for t in range(len(offsets)) for g1 in range(N_GROUPS) for g2 in range(g1 + 1, N_GROUPS)
    ]
    adjacent = [
        (t1, g, t2, g)
        for t1, o1 in enumerate(offsets)
        for t2, o2 in enumerate(offsets)
        for g in range(N_GROUPS)
        if o2 - o1 == 1
    ]
    separated = [
        (t1, g, t2, g)
        for t1, o1 in enumerate(offsets)
        for t2, o2 in enumerate(offsets)
        for g in range(N_GROUPS)
        if o2 - o1 == 4
    ]

    def temporal_agreement(pairs: list[tuple[int, int, int, int]]) -> float:
        values = [(samples[..., t1, g] == samples[..., t2, g]).float().mean() for t1, g, t2, _ in pairs]
        return float(torch.stack(values).mean()) if values else 0.0

    return {
        "same_frame_group_dependence": _collision_dependence(collision, same_frame),
        "adjacent_frame_dependence": _collision_dependence(collision, adjacent),
        "separated_frame_dependence": _collision_dependence(collision, separated),
        "adjacent_frame_agreement": temporal_agreement(adjacent),
        "separated_frame_agreement": temporal_agreement(separated),
    }


def empirical_marginal_tv(first: Tensor, second: Tensor) -> float:
    values = []
    for t in range(first.shape[-2]):
        for group, vocab in enumerate(GROUP_VOCABS):
            p = torch.bincount(first[..., t, group].reshape(-1), minlength=vocab).float()
            q = torch.bincount(second[..., t, group].reshape(-1), minlength=vocab).float()
            p /= p.sum().clamp_min(1)
            q /= q.sum().clamp_min(1)
            values.append(0.5 * (p - q).abs().sum())
    return float(torch.stack(values).mean())


@torch.no_grad()
def val_metrics(model: GPT, batches: list[TrainBatch], cfg: TrainConfig) -> dict[str, float]:
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    T, K = len(model.head_offsets), cfg.trajectory_modes
    nll_sum = torch.zeros(T, N_GROUPS, dtype=torch.float64)
    mixture_nll_sum = torch.zeros_like(nll_sum)
    correct = torch.zeros_like(nll_sum)
    mixture_correct = torch.zeros_like(nll_sum)
    mixture_entropy_sum = torch.zeros_like(nll_sum)
    mode_counts = torch.zeros(K, dtype=torch.long)
    count = 0
    target_rows: list[Tensor] = []
    sampled_rows: list[Tensor] = []
    observed_rows: list[Tensor] = []
    diagnostic_logits: dict[str, list[Tensor]] = {name: [] for name in GROUP_NAMES}
    diagnostic_count = 0
    quantization_squared = quantization_count = invalid_triggers = 0.0
    val_gen = torch.Generator(device=device).manual_seed(cfg.eval_seed)
    try:
        for cpu_batch in batches:
            batch = cpu_batch.to(device)
            history, targets, valid = prepared_targets(model, batch)
            with amp_context(cfg, device):
                hidden = model(batch.context.features, batch.context.ctx_pad, history)
                logits = model.temporal(hidden)
            _, best_k, _, best_nll = best_of_k_loss(logits, targets)
            selected_nll = best_nll[valid]
            selected_k = best_k[valid]
            selected_targets = targets[valid]
            n = selected_nll.shape[0]
            nll_sum += selected_nll.double().sum(dim=0).cpu()
            mode_counts += torch.bincount(selected_k.cpu(), minlength=K)
            count += n
            for group, name in enumerate(GROUP_NAMES):
                valid_logits = logits[name][valid]
                gather = selected_k[:, None, None, None].expand(n, 1, T, valid_logits.shape[-1])
                best_logits = valid_logits.gather(1, gather).squeeze(1)
                correct[:, group] += (best_logits.argmax(-1) == selected_targets[..., group]).double().sum(dim=0).cpu()
                mixture_logp = valid_logits.float().log_softmax(-1).logsumexp(dim=1) - math.log(K)
                mixture_nll_sum[:, group] += (
                    (-mixture_logp.gather(-1, selected_targets[..., group, None]).squeeze(-1))
                    .double()
                    .sum(dim=0)
                    .cpu()
                )
                mixture_correct[:, group] += (
                    (mixture_logp.argmax(-1) == selected_targets[..., group]).double().sum(dim=0).cpu()
                )
                mixture_entropy_sum[:, group] += (
                    (-(mixture_logp.exp() * mixture_logp).sum(dim=-1)).double().sum(dim=0).cpu()
                )

            last_logits = {name: values[:, -1] for name, values in logits.items()}
            sampled = sample_parallel_logits(
                last_logits,
                6,
                sampling_mode="shared_k",
                argmax=False,
                gen=val_gen,
            )
            target_rows.append(targets[:, -1].cpu())
            sampled_rows.append(sampled.cpu())
            observed_rows.append(history[:, -1].cpu())
            if diagnostic_count < cfg.diagnostic_histories:
                take = min(batch.target.shape[0], cfg.diagnostic_histories - diagnostic_count)
                for name in GROUP_NAMES:
                    diagnostic_logits[name].append(last_logits[name][:take].float().cpu())
                diagnostic_count += take

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
    for name, value in tuple(out.items()):
        out[f"best_mode/{name}"] = value
    out.update(mode_utilization_metrics(torch.repeat_interleave(torch.arange(K), mode_counts), K))
    out["bok_trajectory_nll"] = float((nll_sum.sum() / count) / _LN2)
    out["executed_offsets_nll"] = float((nll_sum[: cfg.exec_horizon].sum(-1).mean() / count) / _LN2)
    out["all_predicted_offsets_nll"] = float((nll_sum.sum(-1).mean() / count) / _LN2)
    for depth, offset in enumerate(model.head_offsets):
        for group, name in enumerate(GROUP_NAMES):
            out[f"acc_o{offset:02d}_{name}"] = float(correct[depth, group] / count)
            out[f"best_mode/acc_o{offset:02d}_{name}"] = out[f"acc_o{offset:02d}_{name}"]
            out[f"mixture_nll_o{offset:02d}_{name}"] = float(mixture_nll_sum[depth, group] / count / _LN2)
            out[f"mixture_acc_o{offset:02d}_{name}"] = float(mixture_correct[depth, group] / count)
            out[f"mixture_entropy_o{offset:02d}_{name}"] = float(mixture_entropy_sum[depth, group] / count / _LN2)

    target = torch.cat(target_rows)
    sampled = torch.cat(sampled_rows)
    observed = torch.cat(observed_rows)
    dense_target = target[:, :6]
    matches = sampled == dense_target
    out["exact_frame_acc"] = float(matches.all(dim=-1).float().mean())
    out["dense_four_sequence_acc"] = float(matches[:, :4].all(dim=-1).all(dim=-1).float().mean())
    behavior_metrics = {
        "exact_frame_acc": out["exact_frame_acc"],
        "dense_four_sequence_acc": out["dense_four_sequence_acc"],
        **_transition_metrics(dense_target, sampled, observed),
    }
    out.update({name: value for name, value in behavior_metrics.items() if name not in out})
    out.update({f"sampler_shared_k/{name}": value for name, value in behavior_metrics.items()})
    out["action_quantization_mse"] = quantization_squared / max(quantization_count, 1)
    out["invalid_trigger_count_raw"] = invalid_triggers
    out["invalid_trigger_count_sampled"] = float(
        (~model.codec.button_valid_for_trigger[sampled[..., TRIG_G], sampled[..., BUTTONS_G]]).sum()
    )

    fixed_logits = {name: torch.cat(parts) for name, parts in diagnostic_logits.items()}
    diagnostic_gen = torch.Generator().manual_seed(cfg.eval_seed + 33_000)
    per_mode = sample_trajectories_per_mode(fixed_logits, cfg.trajectory_samples_per_mode, gen=diagnostic_gen)
    out.update(mode_distance_metrics(per_mode))
    sampler_samples = {
        mode: repeated_sampler_samples(fixed_logits, cfg.joint_diagnostic_samples, mode, gen=diagnostic_gen)
        for mode in ("shared_k", "per_frame_k", "per_slot_k")
    }
    sampler_metrics = {
        mode: sampler_dependence_metrics(values, model.head_offsets) for mode, values in sampler_samples.items()
    }
    for mode, metrics in sampler_metrics.items():
        out.update({f"sampler_{mode}/{name}": value for name, value in metrics.items()})
    for name in sampler_metrics["shared_k"]:
        out[f"shared_vs_per_slot/{name}_delta"] = (
            sampler_metrics["shared_k"][name] - sampler_metrics["per_slot_k"][name]
        )
    out["sampler_marginal_tv_shared_vs_per_slot"] = empirical_marginal_tv(
        sampler_samples["shared_k"], sampler_samples["per_slot_k"]
    )
    return out


_UINT64_MASK = (1 << 64) - 1


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return value ^ (value >> 31)


class SlotGroupRandom:
    """Counter RNG keyed by slot, match generation, and sampling stream."""

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
                for name in (*GROUP_NAMES, "mode"):
                    self.counters[(slot_id, generation, name)] = 0
        self.slot_ids = slot_ids
        self.device = ctx.slot_ids.device

    def uniforms(self, group: str) -> Tensor:
        stream_index = {**GROUP_INDEX, "mode": N_GROUPS}
        if group not in stream_index:
            raise ValueError(f"unknown random stream {group!r}")
        values = []
        group_key = _splitmix64(stream_index[group] + 1)
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
    """Hardware-bucketed compiled trunk and one-pass parallel decoders.

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
        self.buckets = tuple(cfg.inference_buckets)
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

            def fn(hidden, observed, uniforms, mode_uniforms):
                return self.model.temporal.sample_indices(
                    hidden,
                    observed,
                    offsets,
                    argmax=False,
                    uniforms=uniforms,
                    mode_uniforms=mode_uniforms,
                    sampling_mode=self.cfg.sampling_mode,
                )

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
        padded = canonical_context(_pad_context(ctx, bucket), self.cfg.observation_bundle)
        observed = self.model.codec.quantize(stack_actions(padded.features))
        uniform_parts: list[Tensor] = []
        mode_uniform_parts: list[Tensor] = []
        if streams is not None:
            streams.begin(ctx)
        for _ in range(horizon):
            groups = []
            modes = []
            for name in GROUP_NAMES:
                if streams is None:
                    real = torch.rand(rows, device=ctx.ctx_pad.device, generator=gen)
                    mode_real = torch.rand(rows, device=ctx.ctx_pad.device, generator=gen)
                else:
                    real = streams.uniforms(name)
                    mode_real = streams.uniforms("mode")
                groups.append(F.pad(real, (0, bucket - rows), value=0.5))
                modes.append(F.pad(mode_real, (0, bucket - rows), value=0.5))
            uniform_parts.append(torch.stack(groups))
            mode_uniform_parts.append(torch.stack(modes))
        uniforms = torch.stack(uniform_parts)
        mode_uniforms = torch.stack(mode_uniform_parts)
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
                    hidden,
                    observed[:, -1],
                    self.model.head_offsets[:horizon],
                    argmax=True,
                    mode_uniforms=mode_uniforms,
                    sampling_mode=self.cfg.sampling_mode,
                )
            else:
                indices = self._decoder(bucket, horizon)(hidden, observed[:, -1], uniforms, mode_uniforms)
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
            raise ValueError("experiment 033 does not condition on a committed RTC prefix")
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
    sampling_mode: str
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
        sampling_mode=cfg.sampling_mode,
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
        model.temporal.mode_embedding,
        model.temporal.group_embedding,
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
        "trajectory_queries": nn.ModuleList(
            [
                model.temporal.context_projection,
                model.temporal.mode_embedding,
                model.temporal.offset_embedding,
                model.temporal.group_embedding,
            ]
        ),
        "trajectory_transformer": model.temporal.blocks,
        "heads": model.temporal.outputs,
    }
    return {name: sum(parameter.numel() for parameter in module.parameters()) for name, module in groups.items()}


def model_tag(cfg: TrainConfig) -> str:
    offsets = "-".join(map(str, cfg.head_offsets))
    return (
        f"plt033-k{cfg.trajectory_modes}-d{cfg.d_model}-L{cfg.n_layers}-h{cfg.n_heads}-Lc{cfg.L_ctx}-"
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
        tags=["gpt", "parallel-latent-trajectory", "best-of-k", "sparse-offset", "033"],
        config={
            **asdict(cfg),
            "K": cfg.trajectory_modes,
            "T": len(cfg.head_offsets),
            "G": N_GROUPS,
            "trajectory_decoder_width": cfg.temporal_d_model,
            "trajectory_decoder_depth": cfg.temporal_layers,
            "trajectory_decoder_heads": cfg.temporal_heads,
            "micro_batch_size": micro_batch_size(cfg),
            "decoder_causal": False,
            "mode_prior": "uniform",
            "evaluation_sampling_modes": (
                ["shared_k"] if cfg.trajectory_modes == 1 else ["shared_k", "per_frame_k", "per_slot_k"]
            ),
        },
    )
    if wandb.run is not None:
        for mode in ("shared_k", "per_frame_k", "per_slot_k"):
            wandb.define_metric(f"eval/{mode}/net_stock_lcb", step_metric="global_step")
            wandb.define_metric(f"eval/{mode}/net_dmg_lcb", step_metric="global_step")
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

    temporal_fn: Callable = model.temporal
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
            decoder_prefixes = sum(batch.target.shape[0] * cfg.L_ctx for batch in cpu_batches)
            if valid_prefixes <= 0:
                raise RuntimeError("training accumulation contains no valid context prefixes")
            optimizer.zero_grad()
            nll_sum = torch.zeros(len(cfg.head_offsets), N_GROUPS, device=DEVICE)
            mode_counts = torch.zeros(cfg.trajectory_modes, dtype=torch.long, device=DEVICE)
            n_prefixes = 0
            with profile("step") as stopwatch:
                for batch in device_batches(cpu_batches, DEVICE, copy_stream):
                    history, targets, valid = prepared_targets(model, batch)
                    with amp_context(cfg, DEVICE):
                        hidden = trunk_fn(batch.context.features, batch.context.ctx_pad, history)
                        logits = temporal_fn(hidden)
                        _, dense_best_k, dense_loss_per_mode, dense_nll = best_of_k_loss(logits, targets)
                        loss_per_mode = dense_loss_per_mode[valid]
                        parts = ActionLoss(
                            nll=dense_nll[valid],
                            targets=targets[valid],
                            best_k=dense_best_k[valid],
                            best_loss=loss_per_mode.min(dim=-1).values,
                            loss_per_mode=loss_per_mode,
                        )
                        loss = parts.best_loss.sum() * (2.0 / len(cfg.head_offsets)) / valid_prefixes
                    loss.backward()
                    nll_sum += parts.nll.detach().sum(dim=0)
                    mode_counts += torch.bincount(parts.best_k.detach(), minlength=cfg.trajectory_modes)
                    n_prefixes += parts.nll.shape[0]
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError(f"step {step}: non-finite gradient norm {gradient_norm}")
                optimizer.step()
                scheduler.step()
                if DEVICE == "cuda":
                    torch.cuda.synchronize()
            metrics = nll_mean_metrics((nll_sum / n_prefixes).cpu(), cfg.head_offsets)
            utilization = mode_utilization_metrics(
                torch.repeat_interleave(torch.arange(cfg.trajectory_modes), mode_counts.cpu()),
                cfg.trajectory_modes,
            )
            log = {
                "global_step": step,
                "samples": (step + 1) * cfg.batch_size,
                **{f"train/{name}": value for name, value in metrics.items()},
                **{f"train/{name}": value for name, value in utilization.items()},
                "train/bok_trajectory_nll": float(nll_sum.sum() / n_prefixes / _LN2),
                "train/grad_norm": float(gradient_norm),
                "throughput/step_s": stopwatch.elapsed,
                "throughput/train_steps_per_s": 1.0 / stopwatch.elapsed,
                "throughput/loader_wait_s": loader_wait,
                "throughput/samples_per_s": cfg.batch_size / stopwatch.elapsed,
                "throughput/examples_per_s": cfg.batch_size / stopwatch.elapsed,
                "throughput/samples_per_wall_s": cfg.batch_size / (stopwatch.elapsed + loader_wait),
                "throughput/prefixes_per_s": n_prefixes / stopwatch.elapsed,
                "throughput/predicted_action_slots_per_s": (
                    decoder_prefixes * cfg.trajectory_modes * len(cfg.head_offsets) * N_GROUPS / stopwatch.elapsed
                ),
                "throughput/valid_action_slots_per_s": (
                    n_prefixes * cfg.trajectory_modes * len(cfg.head_offsets) * N_GROUPS / stopwatch.elapsed
                ),
                "lr/muon": next(group["lr"] for group in optimizer.param_groups if group["use_muon"]),
                "lr/adam": next(group["lr"] for group in optimizer.param_groups if not group["use_muon"]),
            }
            if DEVICE == "cuda":
                log["hardware/peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 2**30
                log["hardware/peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30
                try:
                    log["hardware/gpu_utilization_pct"] = float(torch.cuda.utilization())
                except RuntimeError, OSError:
                    pass
            if step < 10 or step % 50 == 0:
                log["train/mode_utilization_histogram"] = wandb.Histogram(
                    np.repeat(np.arange(cfg.trajectory_modes), mode_counts.cpu().numpy()),
                    num_bins=cfg.trajectory_modes,
                )
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
                wandb.log(
                    {
                        "global_step": step,
                        **{f"val/{name}": value for name, value in values.items()},
                        "val/mode_utilization_histogram": mode_utilization_histogram(values, cfg.trajectory_modes),
                    }
                )
            if eval_due:
                shared_cfg = replace(cfg, sampling_mode="shared_k")
                if eval_inference is None:
                    eval_inference = BF16Inference(model, shared_cfg)
                values = eval_vs_cpu(
                    model,
                    stats,
                    shared_cfg,
                    n_matchups=cfg.eval_n_matchups,
                    replay_dir=replay_dir / f"step_{step:06d}",
                    checkpoint_sha256=_checkpoint_sha256(checkpoint_path),
                    inference=eval_inference,
                )
                wandb.log({"global_step": step, **{f"eval/shared_k/{name}": value for name, value in values.items()}})

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
        wandb.log(
            {
                "global_step": cfg.max_steps,
                **{f"val/{name}": value for name, value in final_val.items()},
                "val/mode_utilization_histogram": mode_utilization_histogram(final_val, cfg.trajectory_modes),
            }
        )
        final_modes = ("shared_k",) if cfg.trajectory_modes == 1 else ("shared_k", "per_frame_k", "per_slot_k")
        for mode in final_modes:
            mode_cfg = replace(cfg, sampling_mode=mode)
            # Periodic evaluation always builds a shared-k engine, regardless
            # of the checkpoint config's manual sampling-mode field.
            inference = eval_inference if mode == "shared_k" and eval_inference is not None else None
            if inference is None:
                inference = BF16Inference(model, mode_cfg)
            final_eval = eval_vs_cpu(
                model,
                stats,
                mode_cfg,
                n_matchups=cfg.final_eval_n_matchups,
                replay_dir=replay_dir / f"final_{mode}",
                checkpoint_sha256=checkpoint_sha,
                inference=inference,
            )
            wandb.log(
                {"global_step": cfg.max_steps, **{f"eval/{mode}/{name}": value for name, value in final_eval.items()}}
            )
        shared_cfg = replace(cfg, sampling_mode="shared_k")
        shared_inference = eval_inference or BF16Inference(model, shared_cfg)
        stride6 = eval_vs_cpu(
            model,
            stats,
            shared_cfg,
            n_matchups=cfg.final_diag_n_matchups,
            replay_dir=replay_dir / "final_s6",
            exec_horizon=cfg.final_diag_exec_horizon,
            checkpoint_sha256=checkpoint_sha,
            inference=shared_inference,
        )
        wandb.log({"global_step": cfg.max_steps, **{f"eval_s6/{name}": value for name, value in stride6.items()}})
    finally:
        if uploader is not None:
            uploader.upload_tree(replay_dir, base=run_dir)
            uploader.close()
        wandb.finish()


_CHECKPOINT_ARCH_FIELDS = {
    "decoder_arch_version",
    "trajectory_modes",
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
    "attention_backend",
}


def config_from_state(values: dict) -> TrainConfig:
    missing = _CHECKPOINT_ARCH_FIELDS - values.keys()
    if missing:
        raise ValueError(f"checkpoint is not an experiment-033 architecture; missing {sorted(missing)}")
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
    sampling_mode: str | None = None,
) -> dict[str, float]:
    model, cfg, stats, state = load_checkpoint(path)
    cfg = replace(
        cfg,
        inference_mode="eager" if eager else cfg.inference_mode,
        eval_max_parallel=cfg.eval_max_parallel if max_parallel is None else max_parallel,
        sampling_mode=cfg.sampling_mode if sampling_mode is None else sampling_mode,
    )
    validate_config(cfg)
    horizon = cfg.exec_horizon if exec_horizon is None else exec_horizon
    default_name = f"eval_{cfg.sampling_mode}_s6" if horizon == 6 else f"eval_{cfg.sampling_mode}"
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
    if state.get("wandb_id"):
        wandb.init(project="hal", id=state["wandb_id"], resume="allow", reinit=True)
        wandb.log(
            {
                "global_step": state["step"],
                **{f"eval/{cfg.sampling_mode}/{name}": value for name, value in values.items()},
            }
        )
        wandb.finish()
    print(f"[eval] step={state['step']} horizon={horizon} sampler={cfg.sampling_mode}: {values}", flush=True)
    return values


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
    eval_sampling_mode: str | None = None
    self_play_eval: str | None = None
    self_play_matches: int = 12
    self_play_frames: int = 14_400
    self_play_eager: bool = False
    self_play_instant_match_restart: bool = False
    self_play_process_cohorts: int = 1
    self_play_cohort_sweep: bool = False
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
            sampling_mode=args.eval_sampling_mode,
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
    stats = load_consolidated_stats(Path(cfg.data_root) / "stats.json")
    train(cfg, stats, comment=args.comment, resume_run=resume_run, resume_state=resume_state)


if __name__ == "__main__":
    main(tyro.cli(Args))
