"""Matched 2x2 future-action and controller-group factorization experiment.

The actor is the d256 experiment-036 actor. Two flags control only information
flow: selected future offsets are independent or autoregressive, and learned
within-frame group conditioning is off or autoregressive. Every cell keeps the
same modules, parameter shapes, optimizer ownership, data, and BC objective.

The detached value head always learns the experiment-036 Monte-Carlo returns.
``actor_weighting="uniform"`` is the initial D0-D3 matrix; ``"mc_awr"`` exists
only so D3 can later be compared with the exact 036 objective.

Run:
    uv run experiments/037_factorization_matrix.py --cell D0
    uv run experiments/037_factorization_matrix.py --cell D1
    uv run experiments/037_factorization_matrix.py --cell D2
    uv run experiments/037_factorization_matrix.py --cell D3
    uv run experiments/037_factorization_matrix.py --eval runs/<run>/final.pt --eval-exec-horizon 2
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
from streaming import StreamingDataset
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR

import wandb
from hal import streams
from hal.data.behavior import HITSTUN_ACTIONS
from hal.data.feature_stats import FeatureStats
from hal.data.policy_schema import decode_policy_replay
from hal.data.schema import check_schema_version
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_LN2 = math.log(2.0)
_N_CONT = 6
_PLAYER_PREFIXES = BASE_PLAYER_PREFIXES
_EXPERIMENT_ID = "037_factorization_matrix_v1"
_RETURN_SUFFIX = "awr_return"
EGO_RETURN = f"ego_{_RETURN_SUFFIX}"
EGO_RETURN_VALID = f"{EGO_RETURN}_valid"
_AUDIT_BETAS = (0.4, 0.8, 1.6, 3.2)
_INFERENCE_BUCKETS = (1, 2, 4, 8, 16, 32)
MATRIX_CELLS: dict[str, tuple[str, str]] = {
    "D0": ("independent", "independent"),
    "D1": ("independent", "autoregressive"),
    "D2": ("selected_ar", "independent"),
    "D3": ("selected_ar", "autoregressive"),
}
FINAL_EVAL_HORIZONS = (1, 2, 4, 6)

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
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 4
    attn_window: int = 0
    L_ctx: int = 128

    sample_chunk_length: int = 20
    head_offsets: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 9, 12, 16, 20)
    temporal_d_model: int = 128
    temporal_layers: int = 2
    temporal_heads: int = 4
    temporal_ff_dim: int = 256
    group_head_dim: int = 256
    action_embed_dim: int = 16
    offset_embed_dim: int = 16
    # Ablation: FiLM the temporal-chain states on the trunk state. The FiLM layer
    # is zero-initialized, so the arm starts at the exact baseline function.
    # False reproduces 026's decoder behavior.
    temporal_state_film: bool = False
    # Fixed 026/036 normalization: primary mean plus one times auxiliary mean.
    aux_loss_weight: float = 1.0

    # Cross-offset information flow in both teacher forcing and sampled decode.
    future_conditioning: Literal["independent", "selected_ar"] = "selected_ar"
    # Learned same-frame FiLM information; the legality mask stays fixed either way.
    group_conditioning: Literal["independent", "autoregressive"] = "autoregressive"
    # The production D0-D3 matrix is uniform BC. MC-AWR is for a later D3 control.
    actor_weighting: Literal["uniform", "mc_awr"] = "uniform"

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
    batch_size: int = 512
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
    val_every: int = 1024
    val_n_samples: int = 1192
    val_batch_size: int = 128
    ckpt_every: int = 1024
    eval_every: int = 0
    eval_max_frames: int = 7200
    eval_n_matchups: int = 32
    final_eval_n_matchups: int = 96
    eval_max_parallel: int | None = 32
    latency_iterations: int = 100

    data_root: str = "data/processed/ranked-anonymized-1/mds-policy-v7"
    mds_schema_version: int = 7
    cache_limit_gb: int = 160
    shuffle_block_size: int = 2000
    predownload: int = 512
    windows_per_replay: int = 4
    reservoir_capacity: int = 4096
    val_split: str = "val"
    num_workers: int = 16
    prefetch_factor: int = 2
    prefetch_batches: int = 4
    push_to_r2: bool = True

    # MC-AWR settings are inert when actor_weighting="uniform". They are kept so
    # a later D3 BC-versus-AWR comparison can use the exact same source and actor.
    advantage_scope: str = "primary"
    # Weight temperature and raw-weight cap: w = min(exp(A / beta), w_max), then
    # eligible weights are normalized to mean 1. The E5 design declares this dose
    # for the E5 reward; the --audit-returns beta sweep is the pre-run check.
    awr_beta: float = 0.8
    awr_weight_max: float = 5.0
    # Reward: stock +-1, damage_shaping per percent dealt/taken, win_reward extra
    # on the match-deciding stock. gamma follows the E5 doc (~400-frame
    # half-life); experiment 031 used 0.99**(1/4) ~= 0.99749 for the same dose.
    awr_gamma: float = 0.99827
    awr_damage_shaping: float = 0.01
    awr_win_reward: float = 0.5
    # Plain BC while the value head learns; weighting activates at this step.
    awr_warmup_steps: int = 2048
    awr_value_loss_weight: float = 1.0
    value_head_hidden_dim: int = 256


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
        "max_steps": cfg.max_steps,
    }
    for name, value in positive.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
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
    dense_prefix = tuple(range(1, cfg.exec_horizon + 1))
    if cfg.exec_horizon < 1 or cfg.exec_horizon > 6 or tuple(cfg.head_offsets[: cfg.exec_horizon]) != dense_prefix:
        raise ValueError("exec_horizon must be an available dense prefix from 1 through 6")
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
    if cfg.aux_loss_weight != 1.0:
        raise ValueError("experiment 037 fixes aux_loss_weight at 1.0")
    if cfg.temporal_state_film:
        raise ValueError("experiment 037 fixes temporal_state_film=False")
    if cfg.future_conditioning not in ("independent", "selected_ar"):
        raise ValueError(f"unknown future_conditioning {cfg.future_conditioning!r}")
    if cfg.group_conditioning not in ("independent", "autoregressive"):
        raise ValueError(f"unknown group_conditioning {cfg.group_conditioning!r}")
    if cfg.actor_weighting not in ("uniform", "mc_awr"):
        raise ValueError(f"unknown actor_weighting {cfg.actor_weighting!r}")
    if cfg.amp_dtype not in ("bfloat16", "float32"):
        raise ValueError("amp_dtype must be bfloat16 or float32")
    if cfg.reservoir_capacity < 2 * cfg.batch_size:
        raise ValueError("reservoir_capacity must be at least twice the batch size")
    if cfg.advantage_scope not in ("primary", "all"):
        raise ValueError(f"advantage_scope must be 'primary' or 'all', got {cfg.advantage_scope!r}")
    if not 0.0 < cfg.awr_gamma < 1.0:
        raise ValueError(f"awr_gamma must be in (0, 1), got {cfg.awr_gamma}")
    if not math.isfinite(cfg.awr_beta) or cfg.awr_beta <= 0:
        raise ValueError(f"awr_beta must be finite and positive, got {cfg.awr_beta}")
    if not math.isfinite(cfg.awr_weight_max) or cfg.awr_weight_max <= 1:
        raise ValueError(f"awr_weight_max must be finite and above 1, got {cfg.awr_weight_max}")
    if cfg.actor_weighting == "mc_awr" and not 0 <= cfg.awr_warmup_steps < cfg.max_steps:
        raise ValueError(f"awr_warmup_steps must be in [0, max_steps), got {cfg.awr_warmup_steps}")
    if not math.isfinite(cfg.awr_value_loss_weight) or cfg.awr_value_loss_weight < 0:
        raise ValueError(f"awr_value_loss_weight must be finite and >= 0, got {cfg.awr_value_loss_weight}")
    if cfg.value_head_hidden_dim <= 0:
        raise ValueError(f"value_head_hidden_dim must be positive, got {cfg.value_head_hidden_dim}")
    if cfg.latency_iterations < 0:
        raise ValueError("latency_iterations must be non-negative")


def config_for_cell(cell: Literal["D0", "D1", "D2", "D3"], cfg: TrainConfig | None = None) -> TrainConfig:
    future, group = MATRIX_CELLS[cell]
    return replace(
        TrainConfig() if cfg is None else cfg,
        future_conditioning=future,
        group_conditioning=group,
    )


def validate_production_config(cfg: TrainConfig) -> None:
    """Fail if a D0-D3 launch changes anything outside the two matrix axes."""
    validate_config(cfg)
    expected: dict[str, object] = {
        "d_model": 256,
        "n_layers": 8,
        "n_heads": 4,
        "attn_window": 0,
        "L_ctx": 128,
        "sample_chunk_length": 20,
        "head_offsets": (1, 2, 3, 4, 5, 6, 9, 12, 16, 20),
        "temporal_d_model": 128,
        "temporal_layers": 2,
        "temporal_heads": 4,
        "temporal_ff_dim": 256,
        "group_head_dim": 256,
        "action_embed_dim": 16,
        "offset_embed_dim": 16,
        "temporal_state_film": False,
        "aux_loss_weight": 1.0,
        "actor_weighting": "uniform",
        "action_vocab": 1024,
        "action_state_embed_dim": 48,
        "char_vocab": 32,
        "char_dim": 8,
        "stage_vocab": 32,
        "stage_dim": 4,
        "observation_bundle": "base",
        "exec_horizon": 4,
        "inference_mode": "compiled",
        "seed": 0,
        "eval_seed": 0,
        "batch_size": 512,
        "muon_lr": 0.02,
        "adam_lr": 8.5e-4,
        "weight_decay": 0.01,
        "warmup_steps": 500,
        "max_steps": 16_384,
        "amp_dtype": "bfloat16",
        "allow_tf32": True,
        "compile_trunk": True,
        "compile_temporal": True,
        "eval_every": 0,
        "data_root": "data/processed/ranked-anonymized-1/mds-policy-v7",
        "mds_schema_version": 7,
        "cache_limit_gb": 160,
        "shuffle_block_size": 2_000,
        "predownload": 512,
        "windows_per_replay": 4,
        "reservoir_capacity": 4_096,
        "num_workers": 16,
        "prefetch_factor": 2,
        "prefetch_batches": 4,
        "push_to_r2": True,
        "advantage_scope": "primary",
        "awr_beta": 0.8,
        "awr_weight_max": 5.0,
        "awr_gamma": 0.99827,
        "awr_damage_shaping": 0.01,
        "awr_win_reward": 0.5,
        "awr_warmup_steps": 2_048,
        "awr_value_loss_weight": 1.0,
        "value_head_hidden_dim": 256,
        "final_eval_n_matchups": 96,
        "eval_max_frames": 7_200,
        "eval_max_parallel": 32,
        "latency_iterations": 100,
    }
    mismatches = {name: (value, getattr(cfg, name)) for name, value in expected.items() if getattr(cfg, name) != value}
    if mismatches:
        details = ", ".join(f"{name}: expected {want!r}, got {got!r}" for name, (want, got) in mismatches.items())
        raise ValueError(f"production matrix configuration mismatch: {details}")


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

    def _forward_chunk(self, x: Tensor, *, self_only: bool = False) -> Tensor:
        q, k, v = self._qkv(x)
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin).transpose(1, 2)
        k = apply_rotary_emb(k, cos, sin).transpose(1, 2)
        v = v.transpose(1, 2)
        if self_only:
            mask = torch.eye(x.shape[1], dtype=torch.bool, device=x.device)
            attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attended = attended.transpose(1, 2).contiguous().view_as(x)
        x = x + self.scale * self.proj(attended)
        return x + self.down(F.silu(self.up(decoder_rmsnorm(x))))

    def forward(self, x: Tensor, *, self_only: bool = False) -> Tensor:
        # Flash SDPA's CUDA launch rejects a flattened batch dimension above
        # 65,535.  Training flattens batch and context positions, which is
        # 1024 * 128 for the production configuration.  Chunking that
        # independent dimension preserves the exact attention computation
        # while keeping every static launch safely below the CUDA grid limit.
        if x.shape[0] <= TEMPORAL_SDPA_BATCH_LIMIT:
            return self._forward_chunk(x, self_only=self_only)
        return torch.cat(
            [self._forward_chunk(chunk, self_only=self_only) for chunk in x.split(TEMPORAL_SDPA_BATCH_LIMIT)],
            dim=0,
        )

    def forward_step(
        self,
        x: Tensor,
        past: tuple[Tensor, Tensor] | None,
        *,
        self_only: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        q, k, v = self._qkv(x[:, None])
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if past is not None:
            k = torch.cat((past[0], k), dim=2)
            v = torch.cat((past[1], v), dim=2)
        cos, sin = self.rotary.at(k.shape[2], x.device)
        q = apply_rotary_emb(q, cos[:, -1:], sin[:, -1:]).transpose(1, 2)
        rotated_k = apply_rotary_emb(k.transpose(1, 2), cos, sin).transpose(1, 2)
        mask = None
        if self_only:
            mask = torch.zeros((1, k.shape[2]), dtype=torch.bool, device=x.device)
            mask[:, -1] = True
        attended = F.scaled_dot_product_attention(q, rotated_k, v, attn_mask=mask)
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
        self.future_conditioning = cfg.future_conditioning
        self.group_conditioning = cfg.group_conditioning
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
            state, present = block.forward_step(
                state,
                past,
                self_only=self.future_conditioning == "independent",
            )
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
        if self.future_conditioning == "selected_ar":
            previous = torch.cat((observed[:, :, None], targets[..., :-1, :]), dim=2)
        else:
            previous = observed[:, :, None].expand(-1, -1, len(self.head_offsets), -1)
        trunk = decoder_rmsnorm(hidden)
        offsets = torch.tensor(self.head_offsets, device=hidden.device)
        x = self._state_bias(trunk)[:, :, None] + self._step_features(previous, offsets)
        x = x.reshape(hidden.shape[0] * hidden.shape[1], len(self.head_offsets), self.d_model)
        for block in self.blocks:
            x = block(x, self_only=self.future_conditioning == "independent")
        states = decoder_rmsnorm(x.view(*hidden.shape[:2], len(self.head_offsets), self.d_model))
        return self._apply_film(states, self._film_params(trunk))

    def group_features(self, states: Tensor, name: str, embedded: dict[str, Tensor]) -> Tensor:
        position = GROUP_ORDER.index(name)
        if position == 0:
            return states
        parts = [embedded[group] for group in GROUP_ORDER[:position]]
        if self.group_conditioning == "independent":
            parts = [torch.zeros_like(value) for value in parts]
        prefix = torch.cat(parts, dim=-1)
        scale, shift = self.group_condition[name](prefix).chunk(2, dim=-1)
        return states * (1.0 + torch.tanh(scale)) + shift

    def teacher_forced_learned_logits_by_group(
        self,
        hidden: Tensor,
        observed: Tensor,
        targets: Tensor,
    ) -> dict[str, Tensor]:
        """Learned logits before the fixed trigger/button legality mask."""
        states = self.teacher_forced_states(hidden, observed, targets)
        embedded = self.codec.embed_groups(targets)
        trunk = decoder_rmsnorm(hidden)
        return {
            name: self.outputs[name](self.group_features(states, name, embedded))
            + self.trunk_outputs[name](trunk)[:, :, None]
            for name in GROUP_NAMES
        }

    def teacher_forced_logits_by_group(self, hidden: Tensor, observed: Tensor, targets: Tensor) -> dict[str, Tensor]:
        logits = self.teacher_forced_learned_logits_by_group(hidden, observed, targets)
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
            if self.future_conditioning == "selected_ar":
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
        if not 1 <= len(offsets) <= 6 or offsets != self.head_offsets[: len(offsets)]:
            raise ValueError("live decode requires an available dense prefix from one through six")
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
            if self.future_conditioning == "selected_ar":
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
            sampled = torch.stack([picks[name] for name in GROUP_NAMES], dim=-1)
            frames.append(sampled)
            if self.future_conditioning == "selected_ar":
                previous = sampled
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
        # V(s) for the advantage baseline. Created LAST, so every policy
        # parameter draws the same initialization a same-seed 026 model does.
        # Decode never reads it.
        self.value_head = NonlinearActionHead(cfg.d_model, cfg.value_head_hidden_dim, 1)

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


def actor_state_dict(model: GPT) -> dict[str, Tensor]:
    return {name: value for name, value in model.state_dict().items() if not name.startswith("value_head.")}


def load_036_actor_weights(model: GPT, state: dict) -> None:
    """Load the actor portion of a shape-compatible experiment-036 checkpoint."""
    if cell_for_config(model.cfg) != "D3":
        raise ValueError("experiment-036 actor weights are compatible only with the D3 factorization")
    values = state.get("model", state)
    expected = actor_state_dict(model)
    supplied = {name: value for name, value in values.items() if not name.startswith("value_head.")}
    if supplied.keys() != expected.keys():
        missing = sorted(expected.keys() - supplied.keys())
        unexpected = sorted(supplied.keys() - expected.keys())
        raise RuntimeError(f"036 actor state mismatch: missing={missing}, unexpected={unexpected}")
    for name, target in expected.items():
        if supplied[name].shape != target.shape:
            raise RuntimeError(
                f"036 actor shape mismatch for {name}: checkpoint {tuple(supplied[name].shape)} != "
                f"model {tuple(target.shape)}"
            )
    merged = model.state_dict()
    merged.update(supplied)
    model.load_state_dict(merged)


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
    advantage: Tensor, eligible: Tensor, *, beta: float, weight_max: float
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute capped AWR weights with mean one over eligible positions.

    Log-space normalization avoids underflow. The cap applies before
    normalization, so the final maximum may exceed ``weight_max``. Ineligible
    positions keep weight one.
    """
    if advantage.requires_grad:
        raise ValueError("AWR weights must come from a DETACHED advantage; the policy loss never trains V")
    if advantage.shape != eligible.shape:
        raise ValueError(f"advantage {tuple(advantage.shape)} and eligibility {tuple(eligible.shape)} must align")
    weights = torch.ones_like(advantage, dtype=torch.float32)
    zero = torch.zeros((), device=advantage.device)
    stats = {
        "advantage_mean": zero,
        "advantage_std": zero,
        "weight_ess": torch.ones_like(zero),
        "weight_clip_frac": zero,
        "weight_norm_max": torch.ones_like(zero),
        "eligible_frac": eligible.float().mean(),
    }
    n_eligible = int(eligible.sum())
    if n_eligible == 0:
        return weights, stats

    eligible_advantage = advantage[eligible].float()
    if not torch.isfinite(eligible_advantage).all():
        raise FloatingPointError("advantage contains a non-finite value on an eligible row")

    max_log_weight = math.log(weight_max)
    log_weights = (eligible_advantage / beta).clamp(max=max_log_weight)
    log_mean_weight = torch.logsumexp(log_weights, dim=0) - math.log(n_eligible)
    normalized_weights = torch.exp(log_weights - log_mean_weight)
    weights[eligible] = normalized_weights
    stats.update(
        advantage_mean=eligible_advantage.mean(),
        advantage_std=eligible_advantage.std(correction=0),
        weight_ess=normalized_weights.sum().square() / (n_eligible * normalized_weights.square().sum()),
        weight_clip_frac=(log_weights >= max_log_weight).float().mean(),
        weight_norm_max=normalized_weights.max(),
    )
    return weights, stats


def advantage_weighted_objective(
    nll: Tensor,
    weight: Tensor,
    *,
    scope: str,
    valid_prefixes: int,
    aux_loss_weight: float,
    n_primary: int = 4,
) -> Tensor:
    """026's objective with a per-position weight on the primary (or every) term.

    With every weight equal to 1 this reproduces 026's expression exactly:
    the dense-prefix mean plus ``aux_loss_weight`` times the auxiliary mean,
    both normalized by the whole optimizer batch's ``valid_prefixes``.
    """
    if scope not in ("primary", "all"):
        raise ValueError(f"advantage scope must be 'primary' or 'all', got {scope!r}")
    if nll.ndim != 3 or nll.shape[2] != N_GROUPS:
        raise ValueError(f"per-prefix NLL must be [n_valid, n_offsets, {N_GROUPS}], got {tuple(nll.shape)}")
    n_offsets = nll.shape[1]
    if n_offsets <= n_primary:
        raise ValueError(f"the objective needs offsets beyond the dense primary prefix, got {n_offsets}")
    if weight.shape != (nll.shape[0],):
        raise ValueError("one weight is required per valid prefix")
    if weight.requires_grad:
        raise ValueError("objective weights must be detached")
    joint_nll = nll.float().sum(dim=-1)
    weights = weight.float()[:, None]
    primary = (joint_nll[:, :n_primary] * weights).sum() / (valid_prefixes * n_primary)
    auxiliary_nll = joint_nll[:, n_primary:]
    if scope == "all":
        auxiliary_nll = auxiliary_nll * weights
    auxiliary = auxiliary_nll.sum() / (valid_prefixes * (n_offsets - n_primary))
    return primary + aux_loss_weight * auxiliary


def microbatch_loss(
    model: GPT,
    batch: AWRBatch,
    cfg: TrainConfig,
    *,
    step: int,
    valid_prefixes: int,
    trunk_fn: Callable,
    temporal_fn: Callable,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Compute the weighted policy loss and value loss for one batch.

    Weighting stays outside the compiled policy functions, so crossing the
    warmup boundary does not trigger recompilation. Logged NLLs are unweighted.
    """
    if not isinstance(batch, AWRBatch):
        raise TypeError(f"experiment 037 needs a return-labeled AWRBatch, got {type(batch).__name__}")
    history, targets, valid = prepared_targets(model, batch)
    with amp_context(cfg, DEVICE):
        hidden = trunk_fn(batch.context.features, batch.context.ctx_pad, history)
        dense_nll = temporal_fn(hidden, history, targets)
    nll = dense_nll[valid]
    if nll.shape[0] != valid_prefixes:
        raise RuntimeError(f"GPU valid-prefix count {nll.shape[0]} != step normalizer {valid_prefixes}")
    value = model.value_head(hidden.detach()[valid].float()).squeeze(-1)
    returns, eligible = batch.valid_rows(valid)
    if bool(eligible.any()):
        value_loss = F.mse_loss(value[eligible], returns[eligible])
    else:
        value_loss = torch.zeros((), device=nll.device)

    advantage = (returns - value).detach()
    weights, stats = advantage_weights(advantage, eligible, beta=cfg.awr_beta, weight_max=cfg.awr_weight_max)
    active = cfg.actor_weighting == "mc_awr" and step >= cfg.awr_warmup_steps
    if not active:
        weights.fill_(1.0)
    policy_loss = advantage_weighted_objective(
        nll,
        weights,
        scope=cfg.advantage_scope,
        valid_prefixes=valid_prefixes,
        aux_loss_weight=cfg.aux_loss_weight,
    )
    loss = policy_loss + cfg.awr_value_loss_weight * value_loss
    if not torch.isfinite(loss):
        raise FloatingPointError(f"step {step}: non-finite policy/value loss {loss}")
    extra = {
        "train/value_loss": value_loss.detach(),
        "train/weighted_objective": policy_loss.detach() / _LN2,
        "awr/active": torch.tensor(float(active)),
        **{f"train/{name}": value.detach() for name, value in stats.items()},
    }
    return loss, nll.detach(), extra


def nll_mean_metrics(mean_nll: Tensor, offsets: tuple[int, ...]) -> dict[str, float]:
    if mean_nll.shape != (len(offsets), N_GROUPS):
        raise ValueError(f"mean NLL has shape {tuple(mean_nll.shape)}")
    joint = mean_nll.sum(dim=-1) / _LN2
    out = {
        "loss": float(joint[:4].mean() + joint[4:].mean()),
        "primary_nll": float(joint[:4].mean()),
        "auxiliary_nll": float(joint[4:].mean()),
    }
    out["joint_nll"] = out["loss"]
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
    accuracy_values: list[float] = []
    for depth, offset in enumerate(model.head_offsets):
        for group, name in enumerate(GROUP_NAMES):
            out[f"acc_o{offset:02d}_{name}"] = float(correct[depth, group] / count)
            accuracy_values.append(out[f"acc_o{offset:02d}_{name}"])
            rollout_count = sum(row.shape[0] for row in target_rows)
            roll_nll = float(rollout_nll[depth, group] / max(rollout_count, 1) / _LN2)
            out[f"rollout_nll_o{offset:02d}_{name}"] = roll_nll
            out[f"exposure_gap_o{offset:02d}_{name}"] = roll_nll - out[f"nll_o{offset:02d}_{name}"]
            out[f"rollout_acc_o{offset:02d}_{name}"] = float(rollout_correct[depth, group] / max(rollout_count, 1))
        rollout_joint = sum(out[f"rollout_nll_o{offset:02d}_{name}"] for name in GROUP_NAMES)
        out[f"rollout_nll_o{offset:02d}"] = rollout_joint
        out[f"exposure_gap_o{offset:02d}"] = rollout_joint - out[f"nll_o{offset:02d}"]
    out["group_accuracy"] = sum(accuracy_values) / len(accuracy_values)
    out["rollout_conditioned_nll"] = sum(out[f"rollout_nll_o{offset:02d}"] for offset in model.head_offsets) / len(
        model.head_offsets
    )
    teacher_forced_all = sum(out[f"nll_o{offset:02d}"] for offset in model.head_offsets) / len(model.head_offsets)
    out["teacher_forced_nll"] = teacher_forced_all
    out["teacher_forced_vs_rollout_nll_gap"] = out["rollout_conditioned_nll"] - teacher_forced_all
    target = torch.cat(target_rows)
    sampled = torch.cat(sampled_rows)
    observed = torch.cat(observed_rows)
    dense_target = target[:, :6]
    matches = sampled == dense_target
    out["exact_frame_acc"] = float(matches.all(dim=-1).float().mean())
    out["dense_four_sequence_acc"] = float(matches[:, :4].all(dim=-1).all(dim=-1).float().mean())
    for horizon in FINAL_EVAL_HORIZONS:
        out[f"dense_prefix_h{horizon}_acc"] = float(matches[:, :horizon].all(dim=-1).all(dim=-1).float().mean())
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
    """Counter RNG keyed by evaluator slot, match generation, frame, and group."""

    def __init__(self, seed: int) -> None:
        self.seed = seed & _UINT64_MASK
        self.generations: dict[int, int] = {}
        self.next_frames: dict[tuple[int, int], int] = {}
        self.slot_ids: tuple[int, ...] = ()
        self.base_frames: tuple[int, ...] = ()
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
            self.next_frames.setdefault((slot_id, generation), 0)
        self.slot_ids = slot_ids
        self.base_frames = tuple(self.next_frames[(slot_id, self.generations[slot_id])] for slot_id in slot_ids)
        self.device = ctx.slot_ids.device

    def uniforms(self, frame: int, group: str) -> Tensor:
        if frame < 0:
            raise ValueError(f"frame offset must be non-negative, got {frame}")
        if group not in GROUP_INDEX:
            raise ValueError(f"unknown group {group!r}")
        values = []
        group_key = _splitmix64(GROUP_INDEX[group] + 1)
        for slot_id, base_frame in zip(self.slot_ids, self.base_frames, strict=True):
            generation = self.generations[slot_id]
            absolute_frame = base_frame + frame
            mixed = (
                self.seed ^ _splitmix64(slot_id) ^ _splitmix64(generation) ^ _splitmix64(absolute_frame) ^ group_key
            )
            values.append(((_splitmix64(mixed) >> 11) + 0.5) / (1 << 53))
        return torch.tensor(values, device=self.device)

    def advance(self, frames: int) -> None:
        if frames < 0:
            raise ValueError(f"frames must be non-negative, got {frames}")
        for slot_id in self.slot_ids:
            key = (slot_id, self.generations[slot_id])
            self.next_frames[key] += frames

    def state(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(sorted((*key, value) for key, value in self.next_frames.items()))


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
        if horizon not in FINAL_EVAL_HORIZONS:
            raise ValueError(f"horizon must be one of {FINAL_EVAL_HORIZONS}")
        rows = ctx.ctx_pad.shape[0]
        bucket = self._bucket(rows)
        padded = canonical_context(_pad_context(ctx, bucket), self.cfg.observation_bundle)
        observed = self.model.codec.quantize(stack_actions(padded.features))
        uniform_parts: list[Tensor] = []
        if streams is not None:
            streams.begin(ctx)
        for frame in range(horizon):
            groups = []
            for name in GROUP_NAMES:
                if streams is None:
                    real = torch.rand(rows, device=ctx.ctx_pad.device, generator=gen)
                else:
                    real = streams.uniforms(frame, name)
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
        if streams is not None:
            streams.advance(horizon)
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
    if horizon not in FINAL_EVAL_HORIZONS:
        raise ValueError(f"execution horizon must be one of {FINAL_EVAL_HORIZONS}")
    engine = BF16Inference(model, cfg) if inference is None else inference
    random_streams = None if decode_seed is None else SlotGroupRandom(decode_seed)
    generator = None if decode_seed is None else torch.Generator(device=device).manual_seed(decode_seed)

    @torch.no_grad()
    def predict(ctx: Context, committed: np.ndarray | None) -> np.ndarray:
        if committed is not None:
            raise ValueError("experiment 037 does not condition on a committed RTC prefix")
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
    future_conditioning: str
    group_conditioning: str
    actor_weighting: str
    sampling_temperature: float
    action_hygiene: str
    instant_legal_stage_restarts: bool
    protocol_sha256: str
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
    protocol = EvalProtocol(
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
        future_conditioning=cfg.future_conditioning,
        group_conditioning=cfg.group_conditioning,
        actor_weighting=cfg.actor_weighting,
        sampling_temperature=1.0,
        action_hygiene="structured_codec_trigger_button_mask_v1",
        instant_legal_stage_restarts=True,
        protocol_sha256="",
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
        start_retries=DEFAULT_START_RETRIES,
    )
    values = asdict(protocol)
    values.pop("protocol_sha256")
    digest = hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return replace(protocol, protocol_sha256=digest)


def _write_eval_evidence(
    replay_dir: Path, rows: list[MatchRow], metrics: dict[str, float], protocol: EvalProtocol
) -> None:
    replay_dir.mkdir(parents=True, exist_ok=True)
    rows_payload = {
        "schema_version": 7,
        "protocol": asdict(protocol),
        "rows": [row.as_dict() for row in rows],
    }
    metrics_payload = {"schema_version": 1, "protocol": asdict(protocol), "metrics": metrics}
    for path, payload in (
        (replay_dir / "match_rows.json", rows_payload),
        (replay_dir / "metrics.json", metrics_payload),
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
    metrics["scheduled_boots"] = float(protocol.n_matchups)
    metrics["crashes"] = float(round(metrics.get("crashed", 1.0) * protocol.n_matchups))
    metrics["eval_wall_seconds"] = time.perf_counter() - started
    metrics["exec_horizon"] = float(horizon)
    metrics.update(telemetry.metrics())
    _write_eval_evidence(replay_dir, rows, metrics, protocol)
    if wandb.run is not None:
        wandb.run.summary[f"eval_h{horizon}/protocol_sha256"] = protocol.protocol_sha256
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
        "value": model.value_head,
    }
    return {name: sum(parameter.numel() for parameter in module.parameters()) for name, module in groups.items()}


def parameter_counts(model: GPT) -> dict[str, int]:
    value = sum(parameter.numel() for parameter in model.value_head.parameters())
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    receiving_grad = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad and parameter.grad is not None
    )
    return {
        "total": total,
        "policy": total - value,
        "value": value,
        "trainable": trainable,
        "receiving_grad": receiving_grad,
    }


def approximate_training_flops(model: GPT, cfg: TrainConfig) -> int:
    """The audited blog estimate: 6 * trainable parameters * trunk positions."""
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return 6 * trainable * cfg.batch_size * cfg.max_steps * cfg.L_ctx


def decoder_mac_estimate(cfg: TrainConfig, horizon: int | None = None) -> dict[str, int]:
    """Nominal decoder MACs per trunk position, with mask-visible attention separate."""
    offsets = len(cfg.head_offsets) if horizon is None else horizon
    d = cfg.temporal_d_model
    controller = N_GROUPS * cfg.action_embed_dim
    projection = cfg.d_model * d + offsets * (controller + cfg.offset_embed_dim) * d
    block_linear = offsets * cfg.temporal_layers * (4 * d * d + 2 * d * cfg.temporal_ff_dim)
    dense_pairs = offsets * offsets
    visible_pairs = offsets if cfg.future_conditioning == "independent" else offsets * (offsets + 1) // 2
    attention_dense = cfg.temporal_layers * 2 * d * dense_pairs
    attention_visible = cfg.temporal_layers * 2 * d * visible_pairs
    group_film = offsets * sum(position * cfg.action_embed_dim * 2 * d for position in range(1, len(GROUP_ORDER)))
    output_heads = offsets * sum(d * cfg.group_head_dim + cfg.group_head_dim * vocab for vocab in GROUP_VOCABS)
    trunk_outputs = cfg.d_model * sum(GROUP_VOCABS)
    components = {
        "decoder_projection_macs_per_prefix": projection,
        "decoder_block_linear_macs_per_prefix": block_linear,
        "decoder_attention_dense_kernel_macs_per_prefix": attention_dense,
        "decoder_attention_visible_macs_per_prefix": attention_visible,
        "decoder_group_film_macs_per_prefix": group_film,
        "decoder_output_head_macs_per_prefix": output_heads,
        "decoder_trunk_output_macs_per_prefix": trunk_outputs,
    }
    components["decoder_nominal_macs_per_prefix"] = (
        projection + block_linear + attention_dense + group_film + output_heads + trunk_outputs
    )
    components["decoder_visible_macs_per_prefix"] = (
        projection + block_linear + attention_visible + group_film + output_heads + trunk_outputs
    )
    return components


def inference_flops_per_replan(model: GPT, cfg: TrainConfig, horizon: int) -> int:
    """Two FLOPs per MAC for one batch-one trunk plus one selected-prefix decode."""
    length = cfg.L_ctx
    width = cfg.d_model
    trunk_projection = length * model.ctx_proj.in_features * width
    trunk_linear = length * cfg.n_layers * 12 * width * width
    trunk_attention = cfg.n_layers * 2 * width * (length * (length + 1) // 2)
    decoder = decoder_mac_estimate(cfg, horizon)["decoder_nominal_macs_per_prefix"]
    return 2 * (trunk_projection + trunk_linear + trunk_attention + decoder)


def cell_for_config(cfg: TrainConfig) -> str:
    axes = (cfg.future_conditioning, cfg.group_conditioning)
    for cell, expected in MATRIX_CELLS.items():
        if axes == expected:
            return cell
    raise ValueError(f"factorization axes do not name a matrix cell: {axes}")


def production_run_name(cfg: TrainConfig) -> str:
    cell = cell_for_config(cfg)
    future = "future-ar" if cfg.future_conditioning == "selected_ar" else "future-independent"
    group = "group-ar" if cfg.group_conditioning == "autoregressive" else "group-independent"
    weighting = "bc" if cfg.actor_weighting == "uniform" else "mc-awr"
    return f"037-{cell}-{future}-{group}-{weighting}-seed{cfg.seed}"


def model_tag(cfg: TrainConfig) -> str:
    offsets = "-".join(map(str, cfg.head_offsets))
    treatment = f"f-{cfg.future_conditioning}-g-{cfg.group_conditioning}-w-{cfg.actor_weighting}"
    return (
        f"mtp037-d{cfg.d_model}-L{cfg.n_layers}-h{cfg.n_heads}-Lc{cfg.L_ctx}-"
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
        replay_format="policy",
        replay_transform=label_replay,
        batch_transform=functools.partial(collate_awr_batch, L_ctx=cfg.L_ctx),
        **common,
    )
    validation = {**common, "batch_size": cfg.val_batch_size}
    val_loader = make_loader(split=cfg.val_split, num_workers=0, compact=True, **validation)
    return train_loader, cache_validation(val_loader, cfg.val_n_samples)


def train(
    cfg: TrainConfig,
    stats: dict[str, FeatureStats],
    *,
    comment: str = "",
    requested_run_name: str | None = None,
    resume_run: str | None = None,
    resume_state: dict | None = None,
) -> None:
    validate_config(cfg)
    if requested_run_name is not None and resume_run is not None:
        raise ValueError("requested_run_name cannot be combined with resume_run")
    run_name = (
        requested_run_name or resume_run or make_run_name(Path(__file__).stem, model_tag(cfg), cfg.data_root, comment)
    )
    uploader = BackgroundUploader(run_name) if cfg.push_to_r2 else None
    wandb.init(
        project="hal",
        name=run_name,
        id=None if resume_state is None else resume_state.get("wandb_id"),
        resume="allow" if resume_state is not None else None,
        tags=["gpt", "temporal-mtp", "factorization-matrix", cell_for_config(cfg), "037"],
        config=asdict(cfg),
    )
    if wandb.run is not None:
        wandb.define_metric("eval/net_stock_lcb", step_metric="global_step")
        wandb.define_metric("eval/net_dmg_lcb", step_metric="global_step")
        wandb.run.summary["nll_semantics"] = (
            "four group losses sum to joint frame NLL; offsets 1-4 and 5+ are each mean-normalized"
        )
        wandb.run.summary["matrix_cell"] = cell_for_config(cfg)
        if cfg.wandb_log_code:
            log_wandb_code(wandb.run)
    run_dir, replay_dir = setup_run_dir(run_name)
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high" if cfg.allow_tf32 else "highest")
    model = GPT(cfg).to(DEVICE)
    counts = subsystem_parameter_counts(model)
    audit_counts = parameter_counts(model)
    if wandb.run is not None:
        for name, value in counts.items():
            wandb.run.summary[f"parameters/{name}"] = value
        for name, value in audit_counts.items():
            wandb.run.summary[f"parameters/{name}"] = value
        wandb.run.summary["training/examples"] = cfg.batch_size * cfg.max_steps
        wandb.run.summary["training/updates"] = cfg.max_steps
        wandb.run.summary["training/approx_flops_6nt"] = approximate_training_flops(model, cfg)
        for name, value in decoder_mac_estimate(cfg).items():
            wandb.run.summary[f"compute/{name}"] = value
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
    def eager_trunk(features, pad, actions):
        return model(features, pad, actions)

    trunk_fn: Callable = eager_trunk
    temporal_fn: Callable = model.temporal.teacher_forced_nll
    if DEVICE == "cuda" and cfg.compile_trunk:
        trunk_fn = torch.compile(trunk_fn, dynamic=False)
    if DEVICE == "cuda" and cfg.compile_temporal:
        temporal_fn = torch.compile(temporal_fn, dynamic=False)

    train_loader, val_cache = _make_loaders(cfg, stats)
    iterator = iter(train_loader)
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

            optimizer.zero_grad()
            with profile("step") as stopwatch:
                batch = batch.to(DEVICE)
                loss, nll, step_metrics = microbatch_loss(
                    model,
                    batch,
                    cfg,
                    step=step,
                    valid_prefixes=valid_prefixes,
                    trunk_fn=trunk_fn,
                    temporal_fn=temporal_fn,
                )
                loss.backward()
                if step == start_step and wandb.run is not None:
                    wandb.run.summary["parameters/receiving_grad"] = parameter_counts(model)["receiving_grad"]
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError(f"step {step}: non-finite gradient norm {gradient_norm}")
                optimizer.step()
                scheduler.step()
                if DEVICE == "cuda":
                    torch.cuda.synchronize()
            metrics = nll_metrics(nll.cpu(), cfg.head_offsets)
            log = {
                "global_step": step,
                "samples": (step + 1) * cfg.batch_size,
                **{f"train/{name}": value for name, value in metrics.items()},
                "train/grad_norm": float(gradient_norm),
                "throughput/step_s": stopwatch.elapsed,
                "throughput/loader_wait_s": loader_wait,
                "throughput/samples_per_s": cfg.batch_size / stopwatch.elapsed,
                "throughput/samples_per_wall_s": cfg.batch_size / (stopwatch.elapsed + loader_wait),
                "throughput/prefixes_per_s": valid_prefixes / stopwatch.elapsed,
                "lr/muon": next(group["lr"] for group in optimizer.param_groups if group["use_muon"]),
                "lr/adam": next(group["lr"] for group in optimizer.param_groups if not group["use_muon"]),
                **{name: float(value) for name, value in step_metrics.items()},
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
                    cfg=_checkpoint_config(cfg),
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
            cfg=_checkpoint_config(cfg),
            wandb_id=None if wandb.run is None else wandb.run.id,
            uploader=uploader,
        )
        checkpoint_sha = _checkpoint_sha256(final_path)
        final_val = val_metrics(model, val_cache, cfg)
        wandb.log({"global_step": cfg.max_steps, **{f"val/{name}": value for name, value in final_val.items()}})
        if eval_inference is None:
            eval_inference = BF16Inference(model, cfg)
        for horizon in FINAL_EVAL_HORIZONS:
            final_eval = eval_vs_cpu(
                model,
                stats,
                cfg,
                n_matchups=cfg.final_eval_n_matchups,
                replay_dir=replay_dir / f"final_h{horizon}",
                exec_horizon=horizon,
                checkpoint_sha256=checkpoint_sha,
                inference=eval_inference,
            )
            values = {f"eval_h{horizon}/{name}": value for name, value in final_eval.items()}
            if horizon == 4:
                values.update({f"eval/{name}": value for name, value in final_eval.items()})
            wandb.log({"global_step": cfg.max_steps, **values})
        if cfg.latency_iterations:
            latency = benchmark_model(model, cfg, iterations=cfg.latency_iterations, rows=1)
            wandb.log({"global_step": cfg.max_steps, **{f"latency/{name}": value for name, value in latency.items()}})
            if wandb.run is not None:
                wandb.run.summary["latency/hardware"] = torch.cuda.get_device_name() if DEVICE == "cuda" else "cpu"
                wandb.run.summary["latency/compile_mode"] = "default"
                wandb.run.summary["latency/precision"] = cfg.amp_dtype
                wandb.run.summary["latency/inference_batch"] = 1
                wandb.run.summary["latency/compiled_inference_bucket"] = 1
                wandb.run.summary["latency/torch_version"] = torch.__version__
                wandb.run.summary["latency/cuda_runtime"] = torch.version.cuda or "none"
    finally:
        if uploader is not None:
            uploader.upload_tree(replay_dir, base=run_dir)
            uploader.close()
        wandb.finish()


_CHECKPOINT_ARCH_FIELDS = {
    "d_model",
    "n_layers",
    "n_heads",
    "L_ctx",
    "head_offsets",
    "sample_chunk_length",
    "temporal_d_model",
    "temporal_layers",
    "temporal_heads",
    "temporal_ff_dim",
    "group_head_dim",
    "action_embed_dim",
    "offset_embed_dim",
    "observation_bundle",
    "future_conditioning",
    "group_conditioning",
    "actor_weighting",
}

_AWR_CHECKPOINT_FIELDS = {
    "experiment_id",
    "advantage_scope",
    "awr_beta",
    "awr_weight_max",
    "awr_gamma",
    "awr_damage_shaping",
    "awr_win_reward",
    "awr_warmup_steps",
    "awr_value_loss_weight",
    "value_head_hidden_dim",
    "temporal_state_film",
}


def _checkpoint_config(cfg: TrainConfig) -> dict[str, object]:
    return {"experiment_id": _EXPERIMENT_ID, **asdict(cfg)}


def config_from_state(values: dict) -> TrainConfig:
    """Restore only an explicitly identified experiment-037 checkpoint."""
    missing = (_CHECKPOINT_ARCH_FIELDS | _AWR_CHECKPOINT_FIELDS) - values.keys()
    if missing:
        raise ValueError(f"checkpoint is not experiment 037; missing {sorted(missing)}")
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
    default_name = f"eval_replays_h{horizon}"
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


def benchmark_model(model: GPT, cfg: TrainConfig, *, iterations: int, rows: int) -> dict[str, float]:
    """Measure full replan latency on one fixed runtime for every evaluation horizon."""
    if iterations < 1 or rows < 1:
        raise ValueError("latency iterations and rows must be positive")
    device = next(model.parameters()).device
    ctx = synthetic_context(cfg, rows, device)
    inference = BF16Inference(model, cfg, bucket=covering_power_of_two(rows))
    was_training = model.training
    model.eval()
    out: dict[str, float] = {}
    try:
        for horizon in FINAL_EVAL_HORIZONS:
            for _ in range(3):
                inference.decode(ctx, horizon)
            samples = []
            for _ in range(iterations):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                started = time.perf_counter()
                inference.decode(ctx, horizon)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                samples.append(1_000 * (time.perf_counter() - started))
            p50, p95 = np.percentile(np.asarray(samples), (50, 95))
            flops = inference_flops_per_replan(model, cfg, horizon)
            prefix = f"h{horizon}"
            out[f"{prefix}/replan_p50_ms"] = float(p50)
            out[f"{prefix}/replan_p95_ms"] = float(p95)
            out[f"{prefix}/decoder_calls_per_replan"] = float(horizon)
            out[f"{prefix}/inference_flops_per_replan"] = float(flops)
            out[f"{prefix}/amortized_flops_per_executed_frame"] = float(flops / horizon)
            out[f"{prefix}/amortized_ms_per_executed_frame"] = float(p50 / horizon)
            out[f"{prefix}/implementation_latency"] = float(
                cfg.future_conditioning == "independent" or cfg.group_conditioning == "independent"
            )
    finally:
        model.train(was_training)
    return out


def run_benchmark(cfg: TrainConfig, *, iterations: int = 20) -> dict[str, float]:
    validate_config(cfg)
    model = GPT(cfg).to(DEVICE)
    out = benchmark_model(model, cfg, iterations=iterations, rows=1)
    print(json.dumps(out, indent=2, sort_keys=True), flush=True)
    return out


def return_audit(cfg: TrainConfig, *, split: str = "train", max_replays: int = 512) -> dict[str, object]:
    """Preflight scan: return distribution, eligibility, and a beta sweep.

    The stream shuffles so the scan is a sample of the corpus, not the first
    shard. The two ports negate each other, so the pooled mean is zero by
    construction — the per-port means are the sign-sensitive check. The
    advantage is bracketed by two baselines the trained V should land between:
    the global mean return and each replay's own mean.
    """
    remote = streams.remote_for_local(cfg.data_root)
    dataset = StreamingDataset(
        remote=None if remote is None else f"{remote}/{split}",
        local=str(Path(cfg.data_root) / split),
        batch_size=1,
        shuffle=True,
        shuffle_seed=cfg.seed,
        cache_limit=f"{cfg.cache_limit_gb}gb" if remote else None,
        predownload=64 if remote else None,
    )
    per_port: dict[str, list[np.ndarray]] = {"p1": [], "p2": []}
    replays = truncated = 0
    decode_seconds = 0.0
    for compact in dataset:
        if replays >= max_replays:
            break
        replays += 1
        check_schema_version(
            {"schema_version": int(compact["source_schema_version"])}, expected=cfg.mds_schema_version
        )
        started = time.perf_counter()
        labeled = returns_lib.replay_returns(
            decode_policy_replay(compact),
            gamma=cfg.awr_gamma,
            damage_shaping=cfg.awr_damage_shaping,
            win_reward=cfg.awr_win_reward,
            suffix=_RETURN_SUFFIX,
        )
        decode_seconds += time.perf_counter() - started
        if not labeled[f"p1_{_RETURN_SUFFIX}_valid"].any():
            truncated += 1
            continue
        for port in ("p1", "p2"):
            per_port[port].append(labeled[f"{port}_{_RETURN_SUFFIX}"])
    per_replay = per_port["p1"] + per_port["p2"]
    if not per_replay:
        raise RuntimeError(f"audit found no terminal replays in the first {replays} rows of {split!r}")
    pooled = np.concatenate(per_replay)
    quantiles = np.percentile(pooled, (1, 5, 25, 50, 75, 95, 99))
    out: dict[str, object] = {
        "replays": replays,
        "truncated_frac": truncated / replays,
        "frames": int(pooled.size),
        "gamma": cfg.awr_gamma,
        "return_mean_p1": float(np.concatenate(per_port["p1"]).mean()),
        "return_mean_p2": float(np.concatenate(per_port["p2"]).mean()),
        "return_std": float(pooled.std()),
        "return_quantiles": {f"p{p:02d}": float(v) for p, v in zip((1, 5, 25, 50, 75, 95, 99), quantiles)},
        "decode_label_ms_per_replay": 1000.0 * decode_seconds / replays,
    }
    print({k: v for k, v in out.items() if k != "return_quantiles"}, flush=True)
    print("return quantiles:", out["return_quantiles"], flush=True)
    print(f"{'baseline':>16} {'beta':>5} {'ess':>6} {'clip%':>6} {'w_max':>7}", flush=True)
    baselines = {
        "global mean": pooled - pooled.mean(),
        "per-replay mean": np.concatenate([g - g.mean() for g in per_replay]),
    }
    for name, centered in baselines.items():
        advantage = torch.from_numpy(np.ascontiguousarray(centered)).float()
        eligible = torch.ones_like(advantage, dtype=torch.bool)
        for beta in _AUDIT_BETAS:
            _, stats = advantage_weights(advantage, eligible, beta=beta, weight_max=cfg.awr_weight_max)
            row = (
                f"{name:>16} {beta:>5g} {float(stats['weight_ess']):>6.3f} "
                f"{100 * float(stats['weight_clip_frac']):>5.1f}% {float(stats['weight_norm_max']):>7.2f}"
            )
            print(row, flush=True)
            out[f"{name}/beta{beta:g}"] = {k: float(v) for k, v in stats.items()}
    return out


@dataclass
class Args:
    cfg: TrainConfig = dataclass_field(default_factory=TrainConfig)
    cell: Literal["D0", "D1", "D2", "D3"] | None = None
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
    audit_returns: bool = False
    audit_split: str = "train"


def main(args: Args) -> None:
    modes = {
        "--audit-returns": args.audit_returns,
        "--benchmark": args.benchmark,
        "--eval": args.eval is not None,
        "--self-play-eval": args.self_play_eval is not None,
        "--resume": args.resume is not None,
    }
    selected_modes = [name for name, selected in modes.items() if selected]
    if len(selected_modes) > 1:
        raise SystemExit(f"pass only one mode, got {', '.join(selected_modes)}")

    if args.audit_returns:
        return_audit(args.cfg, split=args.audit_split)
        return
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
    cfg = config_for_cell(args.cell, args.cfg) if args.cell is not None else args.cfg
    requested_run_name = production_run_name(cfg) if args.cell is not None else None
    if args.cell is not None:
        validate_production_config(cfg)
    if args.resume is not None:
        resume_state = load_for_resume(args.resume, Path("runs") / args.resume, device=DEVICE)
        if resume_state is None:
            raise SystemExit(f"no latest.pt for run {args.resume!r}")
        resume_run = args.resume
        cfg = config_from_state(resume_state["cfg"])
        requested_run_name = None
        if args.cell is not None:
            validate_production_config(cfg)
            if cell_for_config(cfg) != args.cell:
                raise SystemExit(f"resume checkpoint is {cell_for_config(cfg)}, but --cell requested {args.cell}")
    stats = load_consolidated_stats(Path(cfg.data_root) / "stats.json")
    train(
        cfg,
        stats,
        comment=args.comment,
        requested_run_name=requested_run_name,
        resume_run=resume_run,
        resume_state=resume_state,
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
