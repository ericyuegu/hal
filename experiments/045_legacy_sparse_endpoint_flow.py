"""Legacy sparse categorical-endpoint flow with inference-time RTC.

The observation codec, causal trunk, optimizer, data, and closed-loop protocol
match experiment 038. The categorical action state is physically reduced to
O43's 57 legacy classes. Training is unconditioned. Evaluation compares clean
H4-D0 inference with hard-prefix RTC at H4-D1 and H1-D1.

Run:
    uv run experiments/045_legacy_sparse_endpoint_flow.py
    uv run experiments/045_legacy_sparse_endpoint_flow.py --eval runs/<run>/final.pt
    uv run experiments/045_legacy_sparse_endpoint_flow.py --benchmark
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
import wandb
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR

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
from hal.sim.process_vec import ProcessVecTelemetry
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
_EXPERIMENT_ID = "045_legacy_sparse_endpoint_flow_v1"
_INFERENCE_BUCKETS = (1, 2, 4, 8, 16, 32)
FINAL_EVAL_MODES = ((4, 0), (4, 1), (1, 1))
OFFLINE_NFES = (4, 8, 16)
FLOW_TIME_BUCKETS = 10

GROUP_NAMES: tuple[str, ...] = ("buttons", "main_stick", "c_stick", "triggers")
GROUP_VOCABS: tuple[int, ...] = (6, 37, 9, 5)
LEGACY_GROUP_VOCABS = GROUP_VOCABS
N_GROUPS = len(GROUP_NAMES)
FLOW_DIM = sum(GROUP_VOCABS)
BUTTONS_G, MAIN_G, C_G, TRIG_G = range(N_GROUPS)
GROUP_INDEX = {name: index for index, name in enumerate(GROUP_NAMES)}

TRIGGER_L_CH = ACTION_CHANNELS.index("trigger_l")
TRIGGER_R_CH = ACTION_CHANNELS.index("trigger_r")
BUTTON_L_CH = ACTION_CHANNELS.index("button_l")
BUTTON_R_CH = ACTION_CHANNELS.index("button_r")
_BUTTON_L_SEMANTIC_CH = ACTION_CHANNELS[6:].index("button_l")

_LEGACY_MAIN_SIGNED = (
    (0.0, 0.0),
    (0.35, 0.0),
    (-0.35, 0.0),
    (0.0, 0.35),
    (0.0, -0.35),
    (0.675, 0.0),
    (-0.675, 0.0),
    (0.0, 0.675),
    (0.0, -0.675),
    (1.0, 0.0),
    (0.0, 1.0),
    (-1.0, 0.0),
    (0.0, -1.0),
    (0.95, -0.3),
    (-0.95, -0.3),
    (0.95, 0.3),
    (-0.95, 0.3),
    (0.85, -0.5),
    (0.85, 0.5),
    (-0.85, -0.5),
    (-0.85, 0.5),
    (0.7, -0.7),
    (-0.7, -0.7),
    (0.7, 0.7),
    (-0.7, 0.7),
    (0.5, 0.5),
    (-0.5, 0.5),
    (0.5, -0.5),
    (-0.5, -0.5),
    (0.5, 0.85),
    (-0.5, 0.85),
    (0.5, -0.85),
    (-0.5, -0.85),
    (0.3, -0.95),
    (0.3, 0.95),
    (-0.3, -0.95),
    (-0.3, 0.95),
)
_LEGACY_C_SIGNED = (
    (0.0, 0.0),
    (1.0, 0.0),
    (-1.0, 0.0),
    (0.0, -1.0),
    (0.0, 1.0),
    (-0.7, -0.7),
    (0.7, -0.7),
    (0.7, 0.7),
    (-0.7, 0.7),
)
_LEGACY_TRIGGER_VALUES = (0.0, 0.35, 0.6, 0.85, 1.0)
LEGACY_MAIN_CENTERS = torch.tensor(_LEGACY_MAIN_SIGNED, dtype=torch.float32)
LEGACY_C_CENTERS = torch.tensor(_LEGACY_C_SIGNED, dtype=torch.float32)
LEGACY_TRIGGER_CENTERS = torch.tensor(_LEGACY_TRIGGER_VALUES, dtype=torch.float32)

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
    training_prefixes: int = 32
    codec_version: int = 2
    flow_d_model: int = 192
    flow_layers: int = 3
    flow_heads: int = 3
    flow_ff_dim: int = 512
    flow_time_embed_dim: int = 128
    flow_context_dim: int = 192
    flow_condition_dim: int = 192
    button_proj_dim: int = 64
    main_proj_dim: int = 48
    cstick_proj_dim: int = 16
    trigger_proj_dim: int = 24
    action_embed_dim: int = 16
    flow_epsilon: float = 1e-3
    flow_nfe: int = 4
    validation_solver_contexts: int = 128
    validation_diversity_contexts: int = 16
    validation_noise_samples: int = 8

    action_vocab: int = 1024
    action_state_embed_dim: int = 48
    char_vocab: int = 32
    char_dim: int = 8
    stage_vocab: int = 32
    stage_dim: int = 4
    observation_bundle: str = "base"  # or v6_lean

    execution_stride: int = 4
    committed_frames: int = 0
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
    compile_flow: bool = True

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


def dense_prefix_length(offsets: tuple[int, ...]) -> int:
    """Return the number of consecutive one-frame endpoints from offset one."""
    for index, offset in enumerate(offsets, start=1):
        if offset != index:
            return index - 1
    return len(offsets)


def runtime_prediction_frames(cfg: TrainConfig, execution_stride: int, committed_frames: int) -> int:
    """Validate runtime timing and return P = S + D."""
    dense_frames = dense_prefix_length(tuple(cfg.head_offsets))
    if not isinstance(execution_stride, int) or isinstance(execution_stride, bool) or execution_stride < 1:
        raise ValueError("execution_stride must be a positive integer")
    if not isinstance(committed_frames, int) or isinstance(committed_frames, bool) or committed_frames < 0:
        raise ValueError("committed_frames must be a non-negative integer")
    if execution_stride > dense_frames:
        raise ValueError(f"execution_stride must not exceed {dense_frames}")
    if committed_frames > dense_frames - execution_stride:
        raise ValueError(
            f"committed_frames={committed_frames} must satisfy D <= {dense_frames} - S="
            f"{dense_frames - execution_stride}"
        )
    return execution_stride + committed_frames


def validate_config(cfg: TrainConfig) -> None:
    positive = {
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "L_ctx": cfg.L_ctx,
        "sample_chunk_length": cfg.sample_chunk_length,
        "training_prefixes": cfg.training_prefixes,
        "flow_d_model": cfg.flow_d_model,
        "flow_layers": cfg.flow_layers,
        "flow_heads": cfg.flow_heads,
        "flow_ff_dim": cfg.flow_ff_dim,
        "flow_time_embed_dim": cfg.flow_time_embed_dim,
        "flow_context_dim": cfg.flow_context_dim,
        "flow_condition_dim": cfg.flow_condition_dim,
        "button_proj_dim": cfg.button_proj_dim,
        "main_proj_dim": cfg.main_proj_dim,
        "cstick_proj_dim": cfg.cstick_proj_dim,
        "trigger_proj_dim": cfg.trigger_proj_dim,
        "validation_solver_contexts": cfg.validation_solver_contexts,
        "validation_diversity_contexts": cfg.validation_diversity_contexts,
        "validation_noise_samples": cfg.validation_noise_samples,
        "action_embed_dim": cfg.action_embed_dim,
        "execution_stride": cfg.execution_stride,
        "batch_size": cfg.batch_size,
        "max_steps": cfg.max_steps,
    }
    for name, value in positive.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if not isinstance(cfg.committed_frames, int) or isinstance(cfg.committed_frames, bool):
        raise ValueError("committed_frames must be an integer")
    if cfg.d_model % cfg.n_heads or cfg.flow_d_model % cfg.flow_heads:
        raise ValueError("model dimensions must be divisible by their head counts")
    if cfg.codec_version != 2:
        raise ValueError(f"unsupported codec_version={cfg.codec_version}")
    if cfg.flow_d_model // cfg.flow_heads != 64:
        raise ValueError("experiment 045 fixes the flow expert head dimension at 64")
    if cfg.flow_time_embed_dim % 2:
        raise ValueError("flow_time_embed_dim must be even")
    offsets = tuple(cfg.head_offsets)
    if offsets != tuple(sorted(set(offsets))) or not offsets or offsets[0] != 1:
        raise ValueError(f"head_offsets must be sorted, unique, and start at 1, got {offsets}")
    if offsets[-1] > cfg.sample_chunk_length:
        raise ValueError("head_offsets extend beyond sample_chunk_length")
    if offsets[:6] != (1, 2, 3, 4, 5, 6):
        raise ValueError("the live four/six-frame decoders require a dense 1..6 prefix")
    runtime_prediction_frames(cfg, cfg.execution_stride, cfg.committed_frames)
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
    if cfg.training_prefixes > cfg.L_ctx:
        raise ValueError("training_prefixes cannot exceed the causal context length")
    if cfg.flow_nfe < 2:
        raise ValueError("flow_nfe must be at least two for the left-endpoint grid")
    if not 0.0 < cfg.flow_epsilon < 1.0:
        raise ValueError("flow_epsilon must be in (0, 1)")
    if cfg.validation_noise_samples < 2:
        raise ValueError("validation_noise_samples must be at least two")
    if cfg.amp_dtype not in ("bfloat16", "float32"):
        raise ValueError("amp_dtype must be bfloat16 or float32")
    if cfg.reservoir_capacity < 2 * cfg.batch_size:
        raise ValueError("reservoir_capacity must be at least twice the batch size")
    if cfg.latency_iterations < 0:
        raise ValueError("latency_iterations must be non-negative")
    if cfg.eval_every != 0:
        raise ValueError("experiment 045 permits closed-loop evaluation only once at the final checkpoint")


def validate_production_config(cfg: TrainConfig) -> None:
    """Fail if the primary experiment drifts from its declared comparison."""
    validate_config(cfg)
    sparse_offsets = (1, 2, 3, 4, 5, 6, 9, 12, 16, 20)
    dense_offsets = tuple(range(1, 17))
    timing = (cfg.head_offsets, cfg.execution_stride, cfg.committed_frames)
    if timing not in ((sparse_offsets, 4, 0), (dense_offsets, 12, 4)):
        raise ValueError(f"unsupported production endpoint/timing variant {timing}")
    expected: dict[str, object] = {
        "d_model": 256,
        "n_layers": 8,
        "n_heads": 4,
        "attn_window": 0,
        "L_ctx": 128,
        "sample_chunk_length": 20,
        "head_offsets": cfg.head_offsets,
        "training_prefixes": 32,
        "codec_version": 2,
        "flow_d_model": 192,
        "flow_layers": 3,
        "flow_heads": 3,
        "flow_ff_dim": 512,
        "flow_time_embed_dim": 128,
        "flow_context_dim": 192,
        "flow_condition_dim": 192,
        "button_proj_dim": 64,
        "main_proj_dim": 48,
        "cstick_proj_dim": 16,
        "trigger_proj_dim": 24,
        "action_embed_dim": 16,
        "flow_epsilon": 1e-3,
        "flow_nfe": 4,
        "validation_solver_contexts": 128,
        "validation_diversity_contexts": 16,
        "validation_noise_samples": 8,
        "action_vocab": 1024,
        "action_state_embed_dim": 48,
        "char_vocab": 32,
        "char_dim": 8,
        "stage_vocab": 32,
        "stage_dim": 4,
        "observation_bundle": "base",
        "execution_stride": cfg.execution_stride,
        "committed_frames": cfg.committed_frames,
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
        "compile_flow": True,
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
        "final_eval_n_matchups": 96,
        "eval_max_frames": 7_200,
        "eval_max_parallel": 32,
        "latency_iterations": 100,
    }
    mismatches = {name: (value, getattr(cfg, name)) for name, value in expected.items() if getattr(cfg, name) != value}
    if mismatches:
        details = ", ".join(f"{name}: expected {want!r}, got {got!r}" for name, (want, got) in mismatches.items())
        raise ValueError(f"production experiment configuration mismatch: {details}")


def _eval_parallelism(cfg: TrainConfig, n_matchups: int) -> int:
    # ``run_matches_vec`` accepts a power-of-two capacity and then limits the
    # active worker count to ``n_matchups``. Keep that capacity a valid bucket
    # when an ad hoc evaluation asks for, for example, 12 matchups. Never start
    # more Dolphin workers than physical CPUs: oversubscribed startup waves can
    # time out before reaching IN_GAME and leave no policy-quality evidence.
    requested = covering_power_of_two(resolve_parallelism(n_matchups, cfg.eval_max_parallel))
    available_cpus = usable_cpus()
    declared_cpus = cfg.num_workers if cfg.num_workers > 0 else available_cpus
    cpu_cap = 1 << (min(available_cpus, declared_cpus).bit_length() - 1)
    return min(requested, cpu_cap)


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


def legacy_button_classes(actions: Tensor) -> Tensor:
    """Apply ae29e3f's early-release reducer to action sequences."""
    if actions.ndim < 2 or actions.shape[-1] != A_DIM:
        raise ValueError(f"legacy button input must end in [length, {A_DIM}]")
    button = {name: ACTION_CHANNELS.index(f"button_{name}") for name in ("a", "b", "x", "y", "z", "l", "r")}
    pressed = torch.stack(
        (
            actions[..., button["a"]] > 0.5,
            actions[..., button["b"]] > 0.5,
            (actions[..., button["x"]] > 0.5) | (actions[..., button["y"]] > 0.5),
            actions[..., button["z"]] > 0.5,
            (actions[..., button["l"]] > 0.5) | (actions[..., button["r"]] > 0.5),
        ),
        dim=-1,
    )
    length = pressed.shape[-2]
    flat = pressed.reshape(-1, length, 5)
    time_axis = torch.arange(length, device=actions.device).expand(flat.shape[0], -1)
    previous = F.pad(flat[:, :-1], (0, 0, 1, 0))
    same = (flat == previous).all(-1)
    new = flat & ~previous
    selected = new.to(torch.int64).argmax(-1)
    none = torch.full_like(selected, GROUP_VOCABS[BUTTONS_G] - 1)
    base = torch.where(new.any(-1), selected, none)
    changed = ~same | ~flat.any(-1)
    anchor = torch.where(changed, time_axis, -1).cummax(-1).values
    output = base.gather(1, anchor)
    return output.reshape(pressed.shape[:-1])


class StructuredControllerCodec(nn.Module):
    """O43 codec version 2 with physically reduced trainable tables."""

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
        self.register_buffer("main_centers", LEGACY_MAIN_CENTERS.clone())
        self.register_buffer("c_centers", LEGACY_C_CENTERS.clone())
        self.register_buffer("trigger_centers", LEGACY_TRIGGER_CENTERS.clone())
        self.register_buffer(
            "button_valid_for_trigger",
            torch.ones(GROUP_VOCABS[TRIG_G], GROUP_VOCABS[BUTTONS_G], dtype=torch.bool),
        )
        self.register_buffer(
            "_main_quant_centers",
            torch.tensor(_LEGACY_MAIN_SIGNED, dtype=torch.float64) / 2.0 + 0.5,
            persistent=False,
        )
        self.register_buffer(
            "_c_quant_centers",
            torch.tensor(_LEGACY_C_SIGNED, dtype=torch.float64) / 2.0 + 0.5,
            persistent=False,
        )
        self.register_buffer(
            "_trigger_quant_centers",
            torch.tensor(_LEGACY_TRIGGER_VALUES, dtype=torch.float64),
            persistent=False,
        )

    @staticmethod
    def _nearest_stick(values: Tensor, centers: Tensor) -> Tensor:
        points = (values.to(torch.float64) / 2.0 + 0.5).to(torch.float32).to(torch.float64)
        return (points[..., None, :] - centers).square().sum(-1).argmin(-1)

    @staticmethod
    def canonicalize(actions: Tensor) -> Tensor:
        if actions.shape[-1] != A_DIM:
            raise ValueError(f"controller actions must end in {A_DIM} channels, got {tuple(actions.shape)}")
        out = actions.clone()
        fused = out[..., TRIGGER_L_CH : TRIGGER_R_CH + 1].amax(-1)
        out[..., TRIGGER_L_CH] = fused
        out[..., TRIGGER_R_CH] = 0
        return out

    def quantize(self, actions: Tensor) -> Tensor:
        if actions.shape[-1] != A_DIM:
            raise ValueError(f"controller actions must end in {A_DIM} channels, got {tuple(actions.shape)}")
        main = self._nearest_stick(actions[..., :2], self._main_quant_centers)
        c_stick = self._nearest_stick(actions[..., 2:4], self._c_quant_centers)
        fused = actions[..., TRIGGER_L_CH : TRIGGER_R_CH + 1].amax(-1)
        fused = fused.to(torch.float32).to(torch.float64)
        triggers = (fused[..., None] - self._trigger_quant_centers).square().argmin(-1)
        return torch.stack((legacy_button_classes(actions), main, c_stick, triggers), dim=-1)

    def dequantize(self, indices: Tensor) -> Tensor:
        dtype = self.class_embeddings["main_stick"].weight.dtype
        device = indices.device
        main = self.main_centers.to(dtype)[indices[..., MAIN_G]]
        c_stick = self.c_centers.to(dtype)[indices[..., C_G]]
        fused = self.trigger_centers.to(dtype)[indices[..., TRIG_G]]
        buttons = torch.zeros(*indices.shape[:-1], 8, dtype=dtype, device=device)
        classes = indices[..., BUTTONS_G]
        buttons[..., 0] = (classes == 0).to(dtype)
        buttons[..., 1] = (classes == 1).to(dtype)
        buttons[..., 2] = (classes == 2).to(dtype)
        buttons[..., 4] = (classes == 3).to(dtype)
        buttons[..., _BUTTON_L_SEMANTIC_CH] = (classes == 4).to(dtype)
        triggers = torch.stack((fused, torch.zeros_like(fused)), dim=-1)
        return torch.cat((main, c_stick, triggers, buttons), dim=-1)

    def semantic_values(self, name: str, indices: Tensor) -> Tensor:
        dtype = self.class_embeddings[name].weight.dtype
        device = indices.device
        if name == "buttons":
            values = torch.zeros(*indices.shape, 8, dtype=dtype, device=device)
            values[..., 0] = (indices == 0).to(dtype)
            values[..., 1] = (indices == 1).to(dtype)
            values[..., 2] = (indices == 2).to(dtype)
            values[..., 4] = (indices == 3).to(dtype)
            values[..., _BUTTON_L_SEMANTIC_CH] = (indices == 4).to(dtype)
            return values
        if name == "main_stick":
            return self.main_centers.to(dtype)[indices]
        if name == "c_stick":
            return self.c_centers.to(dtype)[indices]
        if name == "triggers":
            fused = self.trigger_centers.to(dtype)[indices]
            return torch.stack((fused, torch.zeros_like(fused)), dim=-1)
        raise ValueError(f"unknown controller group {name!r}")

    def group_embedding(self, name: str, indices: Tensor) -> Tensor:
        semantic = self.semantic_values(name, indices)
        value = self.class_embeddings[name](indices) + self.semantic_projections[name](semantic)
        return decoder_rmsnorm(value)

    def embed_groups(self, indices: Tensor) -> dict[str, Tensor]:
        return {name: self.group_embedding(name, indices[..., GROUP_INDEX[name]]) for name in GROUP_NAMES}

    def embed_frame(self, indices: Tensor, embedded: dict[str, Tensor] | None = None) -> Tensor:
        values = self.embed_groups(indices) if embedded is None else embedded
        return torch.cat([values[name] for name in GROUP_NAMES], dim=-1)

    def button_mask(self, trigger_indices: Tensor) -> Tensor:
        return ~self.button_valid_for_trigger[trigger_indices]


def sinusoidal_time_embedding(tau: Tensor, dim: int) -> Tensor:
    """Embed scalar flow time without mixing it into action-token content."""
    if dim % 2:
        raise ValueError("flow-time embedding dimension must be even")
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10_000.0) * torch.arange(half, device=tau.device, dtype=torch.float32) / max(half - 1, 1)
    )
    angles = tau.float()[:, None] * frequencies[None]
    return torch.cat((angles.sin(), angles.cos()), dim=-1)


class AdaLNZeroBlock(nn.Module):
    """Bidirectional SDPA and complete SwiGLU FFN with AdaLN-Zero."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.d_model = cfg.flow_d_model
        self.n_heads = cfg.flow_heads
        self.head_dim = self.d_model // self.n_heads
        self.qkv = nn.Linear(self.d_model, 3 * self.d_model, bias=False)
        self.attention_out = nn.Linear(self.d_model, self.d_model, bias=False)
        self.ff_gate = nn.Linear(self.d_model, cfg.flow_ff_dim, bias=False)
        self.ff_up = nn.Linear(self.d_model, cfg.flow_ff_dim, bias=False)
        self.ff_down = nn.Linear(cfg.flow_ff_dim, self.d_model, bias=False)
        self.modulation_activation = nn.SiLU()
        self.modulation_projection = nn.Linear(cfg.flow_condition_dim, 6 * self.d_model)
        nn.init.zeros_(self.modulation_projection.weight)
        nn.init.zeros_(self.modulation_projection.bias)

    def forward(self, x: Tensor, condition: Tensor) -> tuple[Tensor, Tensor]:
        modulation = self.modulation_projection(self.modulation_activation(condition))
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = modulation.chunk(6, dim=-1)
        normalized = decoder_rmsnorm(x)
        normalized = normalized * (1.0 + scale_a[:, None]) + shift_a[:, None]
        batch, length, _ = normalized.shape
        q, k, v = self.qkv(normalized).chunk(3, dim=-1)
        shape = (batch, length, self.n_heads, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        attended = F.scaled_dot_product_attention(q, k, v)
        attended = attended.transpose(1, 2).contiguous().view_as(x)
        x = x + gate_a[:, None] * self.attention_out(attended)

        normalized = decoder_rmsnorm(x)
        normalized = normalized * (1.0 + scale_m[:, None]) + shift_m[:, None]
        feed_forward = self.ff_down(F.silu(self.ff_gate(normalized)) * self.ff_up(normalized))
        x = x + gate_m[:, None] * feed_forward
        gate_means = torch.stack((gate_a.detach().abs().mean(), gate_m.detach().abs().mean()))
        return x, gate_means


class CategoricalEndpointFlow(nn.Module):
    """Parallel Gaussian-to-categorical-endpoint action expert."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.head_offsets = tuple(cfg.head_offsets)
        projection_dims = {
            "buttons": cfg.button_proj_dim,
            "main_stick": cfg.main_proj_dim,
            "c_stick": cfg.cstick_proj_dim,
            "triggers": cfg.trigger_proj_dim,
        }
        self.group_projections = nn.ModuleDict(
            {name: nn.Linear(GROUP_VOCABS[GROUP_INDEX[name]], projection_dims[name]) for name in GROUP_NAMES}
        )
        fused_width = sum(projection_dims.values())
        self.action_fusion = nn.Sequential(
            nn.Linear(fused_width, 256),
            nn.SiLU(),
            nn.Linear(256, cfg.flow_d_model),
        )
        self.offset_embedding = nn.Embedding(cfg.sample_chunk_length + 1, cfg.flow_d_model)
        self.context_projection = nn.Linear(cfg.d_model, cfg.flow_context_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.flow_time_embed_dim, cfg.flow_condition_dim),
            nn.SiLU(),
            nn.Linear(cfg.flow_condition_dim, cfg.flow_condition_dim),
        )
        self.condition_mlp = nn.Sequential(
            nn.Linear(cfg.flow_context_dim + cfg.flow_condition_dim, 2 * cfg.flow_condition_dim),
            nn.SiLU(),
            nn.Linear(2 * cfg.flow_condition_dim, cfg.flow_condition_dim),
        )
        self.blocks = nn.ModuleList([AdaLNZeroBlock(cfg) for _ in range(cfg.flow_layers)])
        self.output_heads = nn.ModuleDict(
            {name: nn.Linear(cfg.flow_d_model, GROUP_VOCABS[GROUP_INDEX[name]]) for name in GROUP_NAMES}
        )

    def condition(self, context_h: Tensor, tau: Tensor) -> Tensor:
        history = self.context_projection(decoder_rmsnorm(context_h))
        time = self.time_mlp(sinusoidal_time_embedding(tau, self.cfg.flow_time_embed_dim))
        return self.condition_mlp(torch.cat((history, time), dim=-1))

    def forward(self, noisy: dict[str, Tensor], context_h: Tensor, tau: Tensor) -> tuple[dict[str, Tensor], Tensor]:
        expected = (context_h.shape[0], len(self.head_offsets))
        for group, name in enumerate(GROUP_NAMES):
            if noisy[name].shape != (*expected, GROUP_VOCABS[group]):
                raise ValueError(
                    f"flow state {name} has shape {tuple(noisy[name].shape)}, expected vocab-shaped tokens"
                )
        if tau.shape != (context_h.shape[0],):
            raise ValueError(f"one flow time is required per prefix, got {tuple(tau.shape)}")
        projected = [self.group_projections[name](noisy[name]) for name in GROUP_NAMES]
        x = self.action_fusion(torch.cat(projected, dim=-1))
        offsets = torch.tensor(self.head_offsets, device=x.device)
        x = x + self.offset_embedding(offsets)[None]
        condition = self.condition(context_h, tau)
        gate_means = []
        for block in self.blocks:
            x, block_gates = block(x, condition)
            gate_means.append(block_gates)
        x = decoder_rmsnorm(x)
        logits = {name: self.output_heads[name](x) for name in GROUP_NAMES}
        return logits, torch.stack(gate_means)

    def solve(
        self,
        context_h: Tensor,
        noise: dict[str, Tensor],
        *,
        nfe: int | None = None,
        committed: Tensor | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Run the exact FP32 left-endpoint Euler convention."""
        evaluations = self.cfg.flow_nfe if nfe is None else nfe
        if evaluations < 2:
            raise ValueError("NFE must be at least two")
        state = {name: value.float() for name, value in noise.items()}
        committed_state = None if committed is None else categorical_endpoint(committed)
        committed_frames = 0 if committed is None else committed.shape[1]
        if committed is not None and committed.shape != (context_h.shape[0], committed_frames, N_GROUPS):
            raise ValueError(f"committed categorical prefix has invalid shape {tuple(committed.shape)}")
        final_logits: dict[str, Tensor] | None = None
        end_time = 1.0 - self.cfg.flow_epsilon
        step_size = end_time / (evaluations - 1)
        for index in range(evaluations):
            time_value = index * step_size
            tau = torch.full((context_h.shape[0],), time_value, device=context_h.device, dtype=torch.float32)
            if committed_state is not None:
                for name in GROUP_NAMES:
                    state[name][:, :committed_frames] = committed_state[name]
            logits, _ = self(state, context_h, tau)
            final_logits = logits
            if index == evaluations - 1:
                break
            for name in GROUP_NAMES:
                endpoint = logits[name].float().softmax(dim=-1)
                velocity = (endpoint - state[name]) / (1.0 - time_value)
                state[name] = state[name] + step_size * velocity
        if committed_state is not None:
            for name in GROUP_NAMES:
                state[name][:, :committed_frames] = committed_state[name]
        if final_logits is None:
            raise RuntimeError("flow solver did not evaluate the endpoint network")
        return final_logits, state

    @staticmethod
    def argmax_indices(logits: dict[str, Tensor]) -> Tensor:
        return torch.stack([logits[name].float().argmax(dim=-1) for name in GROUP_NAMES], dim=-1)


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
        self.flow = CategoricalEndpointFlow(cfg)
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


def stratified_prefix_indices(
    ctx_pad: Tensor,
    length: int,
    n_prefixes: int,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample one prefix per stratum and always include the final prefix."""
    if ctx_pad.ndim != 1:
        raise ValueError(f"ctx_pad must be one-dimensional, got {tuple(ctx_pad.shape)}")
    valid_counts = length - ctx_pad
    if bool((valid_counts < 1).any()):
        raise ValueError("every sequence needs at least one real causal prefix")
    if n_prefixes == 1:
        return torch.full_like(ctx_pad[:, None], length - 1)
    regions = n_prefixes - 1
    available = valid_counts - 1
    region_ids = torch.arange(regions, device=ctx_pad.device)
    lower = ctx_pad[:, None] + available[:, None] * region_ids[None] // regions
    upper = ctx_pad[:, None] + available[:, None] * (region_ids[None] + 1) // regions
    # Cold-start windows can contain fewer than 32 real frames. Their empty
    # strata deterministically reuse the nearest valid prefix, preserving 32
    # vectorized problems without dropping the deployment-like opening states.
    widths = (upper - lower).clamp_min(1)
    random = torch.rand(lower.shape, device=ctx_pad.device, generator=generator)
    sampled = lower + (random * widths).long()
    final = torch.full_like(ctx_pad[:, None], length - 1)
    return torch.cat((sampled, final), dim=1)


def prepared_targets(
    model: GPT,
    batch: TrainBatch,
    *,
    n_prefixes: int,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Quantize once, then gather sparse targets for stratified prefixes."""
    if batch.target.shape[1] < model.L_chunk:
        raise ValueError(f"target contains {batch.target.shape[1]} frames, expected {model.L_chunk}")
    history_actions = stack_actions(batch.context.features)
    if history_actions.shape[1] != model.trunk.L_ctx:
        raise ValueError(f"context length {history_actions.shape[1]} != {model.trunk.L_ctx}")
    full = model.codec.quantize(torch.cat((history_actions, batch.target[:, : model.L_chunk]), dim=1))
    length = history_actions.shape[1]
    prefixes = stratified_prefix_indices(
        batch.context.ctx_pad,
        length,
        n_prefixes,
        generator=generator,
    )
    offsets = torch.tensor(model.head_offsets, device=full.device)
    target_positions = prefixes[:, :, None] + offsets[None, None]
    batch_ids = torch.arange(full.shape[0], device=full.device)[:, None, None]
    targets = full[batch_ids, target_positions]
    return full[:, :length], targets, prefixes


def gather_prefix_context(hidden: Tensor, prefixes: Tensor) -> Tensor:
    return hidden.gather(1, prefixes[:, :, None].expand(-1, -1, hidden.shape[-1]))


def categorical_endpoint(targets: Tensor) -> dict[str, Tensor]:
    """Create named one-hot endpoint tensors for the four independent groups."""
    return {
        name: F.one_hot(targets[..., group], GROUP_VOCABS[group]).float() for group, name in enumerate(GROUP_NAMES)
    }


def gaussian_flow_state(
    rows: int,
    offsets: int,
    *,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> dict[str, Tensor]:
    return {
        name: torch.randn(rows, offsets, GROUP_VOCABS[group], device=device, generator=generator)
        for group, name in enumerate(GROUP_NAMES)
    }


def noisy_flow_state(
    targets: Tensor,
    tau: Tensor,
    *,
    generator: torch.Generator | None = None,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    clean = categorical_endpoint(targets)
    noise = gaussian_flow_state(
        targets.shape[0],
        targets.shape[1],
        device=targets.device,
        generator=generator,
    )
    noisy = {name: (1.0 - tau[:, None, None]) * noise[name] + tau[:, None, None] * clean[name] for name in GROUP_NAMES}
    return noisy, noise


def categorical_nll(logits: dict[str, Tensor], targets: Tensor) -> Tensor:
    """Return unreduced endpoint CE as [prefix, offset, group]."""
    losses = []
    for group, name in enumerate(GROUP_NAMES):
        loss = F.cross_entropy(
            logits[name].float().reshape(-1, GROUP_VOCABS[group]),
            targets[..., group].reshape(-1),
            reduction="none",
        ).view(*targets.shape[:-1])
        losses.append(loss)
    return torch.stack(losses, dim=-1)


def flow_metrics(
    nll: Tensor,
    targets: Tensor,
    logits: dict[str, Tensor],
    tau: Tensor,
    offsets: tuple[int, ...],
) -> dict[str, float]:
    """Aggregate group, offset, accuracy, and flow-time diagnostics."""
    if nll.shape != (*targets.shape[:-1], N_GROUPS):
        raise ValueError("NLL and categorical targets do not align")
    group_mean = nll.mean(dim=(0, 1))
    joint_by_offset = nll.sum(dim=-1).mean(dim=0) / _LN2
    out = {
        "flow/objective": float(group_mean.mean()),
        "loss_bits": float(group_mean.mean() / _LN2),
        "joint_nll_bits": float(group_mean.sum() / _LN2),
        "loss": float(joint_by_offset[:4].mean() + joint_by_offset[4:].mean()),
        "primary_nll": float(joint_by_offset[:4].mean()),
        "auxiliary_nll": float(joint_by_offset[4:].mean()),
        "group_accuracy": float(
            torch.stack(
                [
                    (logits[name].argmax(-1) == targets[..., group]).float().mean()
                    for group, name in enumerate(GROUP_NAMES)
                ]
            ).mean()
        ),
    }
    out["joint_nll"] = out["loss"]
    for group, name in enumerate(GROUP_NAMES):
        out[f"flow/loss_{name}"] = float(group_mean[group])
    for depth, offset in enumerate(offsets):
        out[f"flow/ce_o{offset:02d}"] = float(nll[:, depth].mean())
        out[f"nll_o{offset:02d}"] = float(nll[:, depth].sum(dim=-1).mean() / _LN2)
        accuracies = []
        for group, name in enumerate(GROUP_NAMES):
            accuracy = (logits[name][:, depth].argmax(-1) == targets[:, depth, group]).float().mean()
            out[f"flow/acc_o{offset:02d}_{name}"] = float(accuracy)
            out[f"nll_o{offset:02d}_{name}"] = float(nll[:, depth, group].mean() / _LN2)
            out[f"acc_o{offset:02d}_{name}"] = float(accuracy)
            accuracies.append(accuracy)
        out[f"flow/acc_o{offset:02d}"] = float(torch.stack(accuracies).mean())
    buckets = (tau.float() * FLOW_TIME_BUCKETS).long().clamp_max(FLOW_TIME_BUCKETS - 1)
    prefix_ce = nll.mean(dim=(1, 2))
    for bucket in range(FLOW_TIME_BUCKETS):
        selected = buckets == bucket
        if bool(selected.any()):
            bucket_name = f"{bucket / FLOW_TIME_BUCKETS:.1f}_{(bucket + 1) / FLOW_TIME_BUCKETS:.1f}"
            out[f"flow/time_{bucket_name}_ce"] = float(prefix_ce[selected].mean())
            out[f"flow/time_{bucket_name}_count"] = float(selected.sum())
    return out


def microbatch_loss(
    model: GPT,
    batch: TrainBatch,
    cfg: TrainConfig,
    *,
    trunk_fn: Callable,
    flow_fn: Callable,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Tensor], Tensor]:
    """Run the trunk once and one vectorized 32-prefix flow-expert call."""
    history, targets, prefixes = prepared_targets(
        model,
        batch,
        n_prefixes=cfg.training_prefixes,
        generator=generator,
    )
    with amp_context(cfg, DEVICE):
        hidden = trunk_fn(batch.context.features, batch.context.ctx_pad, history)
        context_h = gather_prefix_context(hidden, prefixes).flatten(0, 1)
        flat_targets = targets.flatten(0, 1)
        tau = torch.rand(context_h.shape[0], device=context_h.device, generator=generator)
        noisy, _ = noisy_flow_state(flat_targets, tau, generator=generator)
        logits, gate_means = flow_fn(noisy, context_h, tau)
        nll = categorical_nll(logits, flat_targets)
        loss = nll.mean(dim=(0, 1)).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite categorical endpoint loss {loss}")
    return (
        loss,
        nll.detach(),
        tau.detach(),
        flat_targets.detach(),
        {k: v.detach() for k, v in logits.items()},
        gate_means,
    )


def _mean_gradient_norm(parameters: Iterable[nn.Parameter]) -> float:
    squares = [
        parameter.grad.detach().float().square().sum() for parameter in parameters if parameter.grad is not None
    ]
    if not squares:
        return 0.0
    return float(torch.stack(squares).sum().sqrt())


def _transition_metrics(target: Tensor, prediction: Tensor, observed: Tensor) -> dict[str, float]:
    """Preserve the parent categorical transition metrics on dense offsets 1-6."""
    previous_target = torch.cat((observed[:, None], target[:, :-1]), dim=1)
    previous_prediction = torch.cat((observed[:, None], prediction[:, :-1]), dim=1)
    target_change = target != previous_target
    predicted_change = prediction != previous_prediction
    true_positive = (target_change & predicted_change).sum().float()
    precision = true_positive / predicted_change.sum().clamp_min(1)
    recall = true_positive / target_change.sum().clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {
        "hold_acc": float(((~target_change) & (prediction == target)).sum() / (~target_change).sum().clamp_min(1)),
        "transition_acc": float((target_change & (prediction == target)).sum() / target_change.sum().clamp_min(1)),
        "change_precision": float(precision),
        "change_recall": float(recall),
        "change_f1": float(f1),
        "target_transition_rate": float(target_change.float().mean()),
        "sampled_transition_rate": float(predicted_change.float().mean()),
        "copy_last_acc": float((target == previous_target).float().mean()),
    }


def _temporal_action_metrics(
    target: Tensor,
    prediction: Tensor,
    observed: Tensor,
    codec: StructuredControllerCodec,
) -> dict[str, float]:
    """Compare dense-prefix action persistence and switching against demonstrations."""
    dense_target = target[:, :6]
    dense_prediction = prediction[:, :6]
    out: dict[str, float] = {}
    for prefix, values in (("demo", dense_target), ("flow", dense_prediction)):
        previous = torch.cat((observed[:, None], values[:, :-1]), dim=1)
        out[f"temporal/{prefix}_main_stick_transition_rate"] = float(
            (values[..., MAIN_G] != previous[..., MAIN_G]).float().mean()
        )
        out[f"temporal/{prefix}_c_stick_transition_rate"] = float(
            (values[..., C_G] != previous[..., C_G]).float().mean()
        )
        out[f"temporal/{prefix}_trigger_transition_rate"] = float(
            (values[..., TRIG_G] != previous[..., TRIG_G]).float().mean()
        )
        buttons = codec.semantic_values("buttons", values[..., BUTTONS_G]).bool()
        previous_buttons = codec.semantic_values("buttons", previous[..., BUTTONS_G]).bool()
        press = buttons & ~previous_buttons
        release = ~buttons & previous_buttons
        hold_durations = []
        for sequence in buttons.transpose(1, 2).flatten(0, 1).cpu().tolist():
            duration = 0
            for pressed in sequence:
                if pressed:
                    duration += 1
                elif duration:
                    hold_durations.append(duration)
                    duration = 0
            if duration:
                hold_durations.append(duration)
        if not hold_durations:
            hold_durations = [0]
        out[f"temporal/{prefix}_button_hold_duration"] = float(np.mean(hold_durations))
        out[f"temporal/{prefix}_button_hold_duration_p50"] = float(np.percentile(hold_durations, 50))
        out[f"temporal/{prefix}_button_hold_duration_p95"] = float(np.percentile(hold_durations, 95))
        out[f"temporal/{prefix}_button_press_rate"] = float(press.float().mean())
        out[f"temporal/{prefix}_button_release_rate"] = float(release.float().mean())
    invalid = ~codec.button_valid_for_trigger[prediction[..., TRIG_G], prediction[..., BUTTONS_G]]
    out["temporal/invalid_trigger_button_rate_before_repair"] = float(invalid.float().mean())
    return out


def _pairwise_plan_hamming(plans: Tensor, horizon: int) -> float:
    """Mean pairwise categorical Hamming distance for [contexts, samples, T, G]."""
    sample_count = plans.shape[1]
    pair_ids = torch.combinations(torch.arange(sample_count, device=plans.device), r=2)
    distances = []
    for context_plans in plans:
        left = context_plans[pair_ids[:, 0], :horizon]
        right = context_plans[pair_ids[:, 1], :horizon]
        distances.append((left != right).float().mean())
    return float(torch.stack(distances).mean())


@torch.no_grad()
def val_metrics(model: GPT, batches: list[TrainBatch], cfg: TrainConfig) -> dict[str, float]:
    """Validation CE plus context-use, solver, diversity, and temporal diagnostics."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(cfg.eval_seed)
    nll_rows: list[Tensor] = []
    target_rows: list[Tensor] = []
    tau_rows: list[Tensor] = []
    logit_rows: dict[str, list[Tensor]] = {name: [] for name in GROUP_NAMES}
    shuffled_bucket_loss = torch.zeros(FLOW_TIME_BUCKETS, dtype=torch.float64)
    normal_bucket_loss = torch.zeros(FLOW_TIME_BUCKETS, dtype=torch.float64)
    bucket_count = torch.zeros(FLOW_TIME_BUCKETS, dtype=torch.long)
    solver_context: list[Tensor] = []
    solver_target: list[Tensor] = []
    solver_observed: list[Tensor] = []
    quantization_squared = quantization_count = raw_invalid_triggers = 0.0
    try:
        for cpu_batch in batches:
            batch = cpu_batch.to(device)
            history, targets, prefixes = prepared_targets(
                model,
                batch,
                n_prefixes=cfg.training_prefixes,
                generator=generator,
            )
            with amp_context(cfg, device):
                hidden = model(batch.context.features, batch.context.ctx_pad, history)
                context_h = gather_prefix_context(hidden, prefixes).flatten(0, 1)
                flat_targets = targets.flatten(0, 1)
                tau = torch.rand(context_h.shape[0], device=device, generator=generator)
                noisy, _ = noisy_flow_state(flat_targets, tau, generator=generator)
                logits, _ = model.flow(noisy, context_h, tau)
                shuffled_logits, _ = model.flow(noisy, context_h.roll(cfg.training_prefixes, dims=0), tau)
            nll = categorical_nll(logits, flat_targets)
            shuffled_nll = categorical_nll(shuffled_logits, flat_targets).mean(dim=(1, 2))
            prefix_nll = nll.mean(dim=(1, 2))
            buckets_for_tau = (tau * FLOW_TIME_BUCKETS).long().clamp_max(FLOW_TIME_BUCKETS - 1)
            for bucket in range(FLOW_TIME_BUCKETS):
                selected = buckets_for_tau == bucket
                normal_bucket_loss[bucket] += prefix_nll[selected].double().sum().cpu()
                shuffled_bucket_loss[bucket] += shuffled_nll[selected].double().sum().cpu()
                bucket_count[bucket] += int(selected.sum())
            nll_rows.append(nll.cpu())
            target_rows.append(flat_targets.cpu())
            tau_rows.append(tau.cpu())
            for name in GROUP_NAMES:
                logit_rows[name].append(logits[name].float().cpu())

            remaining = cfg.validation_solver_contexts - sum(value.shape[0] for value in solver_context)
            if remaining > 0:
                take = min(remaining, hidden.shape[0])
                solver_context.append(hidden[:take, -1])
                solver_target.append(targets[:take, -1])
                solver_observed.append(history[:take, -1])

            raw = torch.cat((stack_actions(batch.context.features), batch.target[:, : model.L_chunk]), dim=1)
            canonical = model.codec.canonicalize(raw)
            reconstructed = model.codec.dequantize(model.codec.quantize(raw))
            quantization_squared += float((canonical[..., :6] - reconstructed[..., :6]).square().sum())
            quantization_count += canonical[..., :6].numel()
            raw_invalid_triggers += float(
                (
                    ((raw[..., BUTTON_L_CH] > 0.5) & (raw[..., TRIGGER_L_CH] < 1.0))
                    | ((raw[..., BUTTON_R_CH] > 0.5) & (raw[..., TRIGGER_R_CH] < 1.0))
                ).sum()
            )
    finally:
        model.train(was_training)

    if not nll_rows or not solver_context:
        raise RuntimeError("validation contained no flow prefixes")
    all_nll = torch.cat(nll_rows)
    all_targets = torch.cat(target_rows)
    all_tau = torch.cat(tau_rows)
    all_logits = {name: torch.cat(values) for name, values in logit_rows.items()}
    out = flow_metrics(all_nll, all_targets, all_logits, all_tau, model.head_offsets)
    for bucket in range(FLOW_TIME_BUCKETS):
        if bucket_count[bucket] > 0:
            normal = normal_bucket_loss[bucket] / bucket_count[bucket]
            shuffled = shuffled_bucket_loss[bucket] / bucket_count[bucket]
            out[f"context/shuffled_delta_ce_t{bucket:02d}_{bucket + 1:02d}"] = float(shuffled - normal)
    out["context/shuffled_delta_ce"] = float(
        (shuffled_bucket_loss.sum() - normal_bucket_loss.sum()) / bucket_count.sum().clamp_min(1)
    )
    out["action_quantization_mse"] = quantization_squared / max(quantization_count, 1)
    out["invalid_trigger_count_raw"] = raw_invalid_triggers

    context_h = torch.cat(solver_context)
    solver_targets = torch.cat(solver_target)
    observed = torch.cat(solver_observed)
    shared_noise = gaussian_flow_state(
        context_h.shape[0],
        len(model.head_offsets),
        device=device,
        generator=generator,
    )
    primary_prediction: Tensor | None = None
    for nfe in OFFLINE_NFES:
        with amp_context(cfg, device):
            endpoint_logits, _ = model.flow.solve(context_h, shared_noise, nfe=nfe)
        endpoint_nll = categorical_nll(endpoint_logits, solver_targets)
        prediction = model.flow.argmax_indices(endpoint_logits)
        out[f"solver/nfe{nfe}_ce"] = float(endpoint_nll.mean())
        out[f"solver/nfe{nfe}_accuracy"] = float((prediction == solver_targets).float().mean())
        if nfe == cfg.flow_nfe:
            primary_prediction = prediction
    if primary_prediction is None:
        raise RuntimeError("primary NFE is absent from offline solver diagnostics")
    dense_targets = solver_targets[:, :6]
    dense_prediction = primary_prediction[:, :6]
    matches = dense_prediction == dense_targets
    out["exact_frame_acc"] = float(matches.all(dim=-1).float().mean())
    out["dense_four_sequence_acc"] = float(matches[:, :4].all(dim=-1).all(dim=-1).float().mean())
    for horizon in (1, 2, 4, 6):
        out[f"dense_prefix_h{horizon}_acc"] = float(matches[:, :horizon].all(dim=-1).all(dim=-1).float().mean())
    out.update(_transition_metrics(dense_targets, dense_prediction, observed))
    out.update(_temporal_action_metrics(solver_targets, primary_prediction, observed, model.codec))
    invalid_prediction = ~model.codec.button_valid_for_trigger[
        primary_prediction[..., TRIG_G], primary_prediction[..., BUTTONS_G]
    ]
    out["invalid_trigger_count_sampled"] = float(invalid_prediction.sum())

    contexts = context_h[: cfg.validation_diversity_contexts]
    repeated = contexts[:, None].expand(-1, cfg.validation_noise_samples, -1).flatten(0, 1)
    diversity_noise = gaussian_flow_state(
        repeated.shape[0],
        len(model.head_offsets),
        device=device,
        generator=generator,
    )
    with amp_context(cfg, device):
        diversity_logits, _ = model.flow.solve(repeated, diversity_noise, nfe=cfg.flow_nfe)
    plans = model.flow.argmax_indices(diversity_logits).view(
        contexts.shape[0], cfg.validation_noise_samples, len(model.head_offsets), N_GROUPS
    )
    unique_counts = [torch.unique(context_plan[:, :4].flatten(1), dim=0).shape[0] for context_plan in plans]
    out["noise/unique_first_four_plans"] = float(np.mean(unique_counts))
    out["noise/first_four_hamming"] = _pairwise_plan_hamming(plans, 4)
    out["noise/all_offsets_hamming"] = _pairwise_plan_hamming(plans, len(model.head_offsets))
    return out


_UINT64_MASK = (1 << 64) - 1


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return value ^ (value >> 31)


class SlotFlowRandom:
    """Gaussian plan RNG keyed by evaluator slot, generation, frame, and group."""

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

    def noise(self, offsets: int) -> dict[str, Tensor]:
        """Draw iid coordinates without depending on batch order or padding."""
        values: dict[str, list[Tensor]] = {name: [] for name in GROUP_NAMES}
        for slot_id, base_frame in zip(self.slot_ids, self.base_frames, strict=True):
            generation = self.generations[slot_id]
            for group, name in enumerate(GROUP_NAMES):
                mixed = (
                    self.seed
                    ^ _splitmix64(slot_id)
                    ^ _splitmix64(generation)
                    ^ _splitmix64(base_frame)
                    ^ _splitmix64(group + 1)
                )
                generator = torch.Generator().manual_seed(mixed % ((1 << 63) - 1))
                values[name].append(torch.randn(offsets, GROUP_VOCABS[group], generator=generator))
        return {name: torch.stack(group_values).to(self.device) for name, group_values in values.items()}

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
    """Hardware-bucketed compiled trunk plus four-NFE FP32 flow solver.

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
        self.raw_invalid_actions = 0
        self.raw_actions = 0

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

    def _decoder(self, bucket: int, committed_frames: int) -> Callable:
        key = (bucket, committed_frames)
        if key not in self._decoders:

            def fn(hidden, noise, committed):
                prefix = committed if committed_frames else None
                logits, _ = self.model.flow.solve(
                    hidden[:, -1],
                    noise,
                    nfe=self.cfg.flow_nfe,
                    committed=prefix,
                )
                indices = self.model.flow.argmax_indices(logits)
                if committed_frames:
                    indices[:, :committed_frames] = committed
                return indices

            self._decoders[key] = torch.compile(fn, dynamic=False, mode=self.compile_mode) if self.compiled else fn
        return self._decoders[key]

    @property
    def invalid_rate_before_repair(self) -> float:
        return self.raw_invalid_actions / max(self.raw_actions, 1)

    @torch.no_grad()
    def decode(
        self,
        ctx: Context,
        prediction_frames: int,
        *,
        execution_stride: int | None = None,
        committed: np.ndarray | None = None,
        committed_frames: int | None = None,
        streams: SlotFlowRandom | None = None,
        argmax: bool = True,
        gen: torch.Generator | None = None,
    ) -> Tensor:
        stride = prediction_frames if execution_stride is None else execution_stride
        delay = (0 if committed is None else committed.shape[1]) if committed_frames is None else committed_frames
        expected_prediction_frames = runtime_prediction_frames(self.cfg, stride, delay)
        if prediction_frames != expected_prediction_frames:
            raise ValueError(
                f"prediction_frames={prediction_frames} must equal execution_stride + committed_frames="
                f"{expected_prediction_frames}"
            )
        dense = tuple(range(1, prediction_frames + 1))
        if tuple(self.model.head_offsets[:prediction_frames]) != dense:
            raise ValueError(f"prediction frames require dense endpoint offsets {dense}")
        rows = ctx.ctx_pad.shape[0]
        bucket = self._bucket(rows)
        padded = canonical_context(_pad_context(ctx, bucket), self.cfg.observation_bundle)
        if not argmax:
            raise ValueError("experiment 045 closed-loop decoding uses final endpoint argmax")
        observed = self.model.codec.quantize(stack_actions(padded.features))
        clamped_frames = 0 if committed is None else delay
        if committed is None:
            committed_indices = torch.empty(bucket, 0, N_GROUPS, dtype=torch.long, device=ctx.ctx_pad.device)
        else:
            expected = (rows, delay, A_DIM)
            if committed.shape != expected:
                raise ValueError(f"committed raw actions have shape {committed.shape}, expected {expected}")
            committed_raw = torch.from_numpy(committed).to(ctx.ctx_pad.device)
            if bucket > rows:
                committed_raw = torch.cat(
                    (
                        committed_raw,
                        torch.zeros(
                            bucket - rows,
                            delay,
                            A_DIM,
                            dtype=committed_raw.dtype,
                            device=committed_raw.device,
                        ),
                    )
                )
            committed_indices = self.model.codec.quantize(committed_raw)
        if streams is not None:
            streams.begin(ctx)
            real_noise = streams.noise(len(self.model.head_offsets))
            noise = {
                name: torch.cat(
                    (
                        real_noise[name],
                        torch.zeros(
                            bucket - rows,
                            len(self.model.head_offsets),
                            GROUP_VOCABS[group],
                            device=ctx.ctx_pad.device,
                        ),
                    )
                )
                for group, name in enumerate(GROUP_NAMES)
            }
        else:
            noise = gaussian_flow_state(
                bucket,
                len(self.model.head_offsets),
                device=ctx.ctx_pad.device,
                generator=gen,
            )
        if self.uses_cuda_graphs:
            # The trunk and decoder are separate CUDA Graph trees.  Mark one
            # complete decode as a graph step so the next trunk replay may
            # safely reuse its managed output storage after the decoder has
            # consumed it.
            torch.compiler.cudagraph_mark_step_begin()
        with amp_context(self.cfg, ctx.ctx_pad.device):
            hidden = self._trunk(bucket)(padded.features, padded.ctx_pad, observed)
            indices = self._decoder(bucket, clamped_frames)(hidden, noise, committed_indices)
        if streams is not None:
            streams.advance(stride)
        indices = indices[:rows]
        invalid = ~self.model.codec.button_valid_for_trigger[indices[..., TRIG_G], indices[..., BUTTONS_G]]
        self.raw_invalid_actions += int(invalid.sum())
        self.raw_actions += invalid.numel()
        raw_actions = self.model.codec.canonicalize(self.model.codec.dequantize(indices[:, :prediction_frames]))
        if committed is not None:
            raw_actions[:, :delay] = torch.from_numpy(committed).to(raw_actions)
        return raw_actions


@torch.no_grad()
def decode_chunk(
    model: GPT,
    ctx: Context,
    n_frames: int,
    *,
    argmax: bool = True,
    gen: torch.Generator | None = None,
) -> Tensor:
    cfg = model.cfg
    return BF16Inference(model, replace(cfg, inference_mode="eager"), compiled=False).decode(
        ctx,
        n_frames,
        execution_stride=n_frames,
        argmax=argmax,
        gen=gen,
    )


def make_policy(
    model: GPT,
    stats: dict[str, FeatureStats],
    cfg: TrainConfig,
    *,
    execution_stride: int | None = None,
    committed_frames: int | None = None,
    decode_seed: int | None = None,
    inference: BF16Inference | None = None,
    telemetry: DecodeTelemetry | None = None,
    device: str = DEVICE,
) -> RecedingHorizon:
    stride = cfg.execution_stride if execution_stride is None else execution_stride
    delay = cfg.committed_frames if committed_frames is None else committed_frames
    prediction_frames = runtime_prediction_frames(cfg, stride, delay)
    dense = tuple(range(1, prediction_frames + 1))
    if tuple(cfg.head_offsets[:prediction_frames]) != dense:
        raise ValueError(f"(P,S,D)=({prediction_frames},{stride},{delay}) needs dense offsets {dense}")
    engine = BF16Inference(model, cfg) if inference is None else inference
    random_streams = None if decode_seed is None else SlotFlowRandom(decode_seed)
    generator = None if decode_seed is None else torch.Generator(device=device).manual_seed(decode_seed)

    @torch.no_grad()
    def predict(ctx: Context, committed: np.ndarray | None) -> np.ndarray:
        started = time.perf_counter()
        result = (
            engine.decode(
                ctx,
                prediction_frames,
                execution_stride=stride,
                committed=committed,
                committed_frames=delay,
                streams=random_streams,
                gen=generator,
            )
            .cpu()
            .numpy()
        )
        if telemetry is not None:
            telemetry.record(rows=ctx.ctx_pad.shape[0], horizon=stride, seconds=time.perf_counter() - started)
        return result

    v6 = cfg.observation_bundle == "v6_lean"
    return RecedingHorizon(
        predict_chunk=predict,
        stats=stats,
        L_ctx=cfg.L_ctx,
        L_chunk=prediction_frames,
        s=stride,
        d=delay,
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
    ego_port: Literal[1, 2]
    seed_stage: int
    matchup_schedule_sha256: str
    oriented_pairs: int
    ego_characters: int
    cpu_characters: int
    prediction_frames: int
    execution_stride: int
    committed_frames: int
    codec_version: int
    vocabularies: tuple[int, ...]
    hard_categorical_prefix_clamp: bool
    dtype: str
    inference_mode: str
    inference_compile_mode: str
    compiled_inference_bucket: int
    checkpoint_sha256: str
    future_offsets: tuple[int, ...]
    nfe: int
    flow_epsilon: float
    endpoint_decode: str
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
    execution_stride: int,
    committed_frames: int,
    checkpoint_sha256: str,
    inference_compile_mode: str = "reduce-overhead",
) -> EvalProtocol:
    runtime_prediction_frames(cfg, execution_stride, committed_frames)
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
        prediction_frames=execution_stride + committed_frames,
        execution_stride=execution_stride,
        committed_frames=committed_frames,
        codec_version=cfg.codec_version,
        vocabularies=GROUP_VOCABS,
        hard_categorical_prefix_clamp=committed_frames > 0,
        dtype=str(next(model.parameters()).dtype),
        inference_mode=cfg.inference_mode,
        inference_compile_mode=inference_compile_mode,
        compiled_inference_bucket=_eval_inference_bucket(cfg, n_matchups),
        checkpoint_sha256=checkpoint_sha256,
        future_offsets=cfg.head_offsets,
        nfe=cfg.flow_nfe,
        flow_epsilon=cfg.flow_epsilon,
        endpoint_decode="argmax",
        action_hygiene="legacy_codec_v2_fused_shoulders",
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
    execution_stride: int | None = None,
    committed_frames: int | None = None,
    checkpoint_sha256: str = "unavailable",
    inference: BF16Inference | None = None,
) -> dict[str, float]:
    stride = cfg.execution_stride if execution_stride is None else execution_stride
    delay = cfg.committed_frames if committed_frames is None else committed_frames
    prediction_frames = runtime_prediction_frames(cfg, stride, delay)
    inference = BF16Inference(model, cfg) if inference is None else inference
    if inference.model is not model:
        raise ValueError("the supplied inference engine must own the evaluation model")
    protocol = _eval_protocol(
        cfg,
        model,
        n_matchups=n_matchups,
        execution_stride=stride,
        committed_frames=delay,
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
            execution_stride=stride,
            committed_frames=delay,
            decode_seed=protocol.seed + next(policy_index),
            inference=inference,
            telemetry=telemetry,
        )

    was_training = model.training
    model.eval()
    started = time.perf_counter()
    process_telemetry = ProcessVecTelemetry()
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
            process_telemetry=process_telemetry,
        )
    finally:
        model.train(was_training)
    metrics = vs_cpu_metrics(results, seed=protocol.seed)
    metrics["scheduled_boots"] = float(protocol.n_matchups)
    metrics["crashes"] = float(round(metrics.get("crashed", 1.0) * protocol.n_matchups))
    metrics["eval_wall_seconds"] = time.perf_counter() - started
    metrics["prediction_frames"] = float(prediction_frames)
    metrics["execution_stride"] = float(stride)
    metrics["committed_frames"] = float(delay)
    metrics["codec_version"] = float(cfg.codec_version)
    metrics["hard_categorical_prefix_clamp"] = float(delay > 0)
    metrics["nfe"] = float(cfg.flow_nfe)
    metrics["invalid_trigger_button_rate_before_repair"] = inference.invalid_rate_before_repair
    decode_metrics = telemetry.metrics()
    metrics.update(decode_metrics)
    metrics["model_decode_wall_seconds"] = decode_metrics["decode_seconds"]
    metrics["model_decode_p50_ms"] = decode_metrics["decode_p50_ms"]
    metrics["model_decode_p95_ms"] = decode_metrics["decode_p95_ms"]
    metrics["model_decode_p99_ms"] = decode_metrics["decode_p99_ms"]
    metrics["amortized_decode_ms_per_executed_frame"] = decode_metrics["decode_p50_ms"] / stride
    metrics.update(process_telemetry.metrics())
    plan_calls = max(process_telemetry.plan_calls, 1)
    environment_seconds = (
        process_telemetry.control_wait_seconds
        + process_telemetry.request_read_seconds
        + process_telemetry.plan_write_seconds
        + process_telemetry.result_read_seconds
    )
    metrics["environment_step_ms_per_replan"] = 1_000.0 * environment_seconds / plan_calls
    metrics["control_loop_ms_per_replan"] = (
        1_000.0 * (environment_seconds + process_telemetry.policy_seconds) / plan_calls
    )
    _write_eval_evidence(replay_dir, rows, metrics, protocol)
    if wandb.run is not None:
        mode = f"h{stride}_d{delay}"
        wandb.run.summary[f"eval_{mode}/protocol_sha256"] = protocol.protocol_sha256
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
        model.flow.offset_embedding,
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
    groups: dict[str, nn.Module] = {
        "trunk": model.trunk,
        "observation": model.ctx_proj,
        "codec": model.codec,
        "flow_heads": model.flow.output_heads,
    }
    out = {name: sum(parameter.numel() for parameter in module.parameters()) for name, module in groups.items()}
    flow_total = sum(parameter.numel() for parameter in model.flow.parameters())
    out["flow_expert"] = flow_total - out["flow_heads"]
    return out


def parameter_counts(model: GPT) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    receiving_grad = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad and parameter.grad is not None
    )
    return {
        "total": total,
        "policy": total,
        "trainable": trainable,
        "receiving_grad": receiving_grad,
    }


def approximate_training_flops(model: GPT, cfg: TrainConfig) -> int:
    """The audited blog estimate: 6 * trainable parameters * trunk positions."""
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return 6 * trainable * cfg.batch_size * cfg.max_steps * cfg.L_ctx


def decoder_mac_estimate(cfg: TrainConfig) -> dict[str, int]:
    """Nominal MACs for one complete flow-expert evaluation."""
    tokens = len(cfg.head_offsets)
    d = cfg.flow_d_model
    projected = (cfg.button_proj_dim, cfg.main_proj_dim, cfg.cstick_proj_dim, cfg.trigger_proj_dim)
    group_projection = tokens * sum(vocab * width for vocab, width in zip(GROUP_VOCABS, projected, strict=True))
    fusion = tokens * (sum(projected) * 256 + 256 * d)
    conditioning = (
        cfg.d_model * cfg.flow_context_dim
        + cfg.flow_time_embed_dim * cfg.flow_condition_dim
        + cfg.flow_condition_dim**2
        + (cfg.flow_context_dim + cfg.flow_condition_dim) * 2 * cfg.flow_condition_dim
        + 2 * cfg.flow_condition_dim**2
    )
    block_linear = cfg.flow_layers * (tokens * (4 * d * d + 3 * d * cfg.flow_ff_dim) + cfg.flow_condition_dim * 6 * d)
    attention = cfg.flow_layers * 2 * d * tokens * tokens
    output_heads = tokens * d * sum(GROUP_VOCABS)
    components = {
        "flow_group_projection_macs_per_nfe": group_projection,
        "flow_fusion_macs_per_nfe": fusion,
        "flow_conditioning_macs_per_nfe": conditioning,
        "flow_block_linear_macs_per_nfe": block_linear,
        "flow_attention_macs_per_nfe": attention,
        "flow_output_head_macs_per_nfe": output_heads,
    }
    components["flow_macs_per_nfe"] = sum(components.values())
    return components


def inference_flops_per_replan(model: GPT, cfg: TrainConfig) -> int:
    """Two FLOPs per MAC for one trunk pass plus exactly ``flow_nfe`` expert calls."""
    length = cfg.L_ctx
    width = cfg.d_model
    trunk_projection = length * model.ctx_proj.in_features * width
    trunk_linear = length * cfg.n_layers * 12 * width * width
    trunk_attention = cfg.n_layers * 2 * width * (length * (length + 1) // 2)
    decoder = cfg.flow_nfe * decoder_mac_estimate(cfg)["flow_macs_per_nfe"]
    return 2 * (trunk_projection + trunk_linear + trunk_attention + decoder)


def evaluation_modes(cfg: TrainConfig) -> tuple[tuple[int, int], ...]:
    if cfg.head_offsets == tuple(range(1, 17)):
        return ((12, 4),)
    return FINAL_EVAL_MODES


def dense_16_config(cfg: TrainConfig) -> TrainConfig:
    """Configure the additional P16-D4 checkpoint (therefore S12)."""
    return replace(
        cfg,
        head_offsets=tuple(range(1, 17)),
        execution_stride=12,
        committed_frames=4,
    )


def production_run_name(cfg: TrainConfig) -> str:
    prediction_frames = cfg.execution_stride + cfg.committed_frames
    variant = "dense16" if cfg.head_offsets == tuple(range(1, 17)) else "sparse10"
    return (
        f"045-legacy-endpoint-flow-{variant}-nfe{cfg.flow_nfe}-"
        f"p{prediction_frames}-s{cfg.execution_stride}-d{cfg.committed_frames}-seed{cfg.seed}"
    )


def model_tag(cfg: TrainConfig) -> str:
    offsets = "-".join(map(str, cfg.head_offsets))
    return (
        f"flow045-legacy-v{cfg.codec_version}-d{cfg.d_model}-L{cfg.n_layers}-h{cfg.n_heads}-Lc{cfg.L_ctx}-"
        f"f{cfg.flow_d_model}x{cfg.flow_layers}-o{offsets}-nfe{cfg.flow_nfe}-"
        f"p{cfg.execution_stride + cfg.committed_frames}-s{cfg.execution_stride}-d{cfg.committed_frames}-"
        f"{cfg.observation_bundle}"
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


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _make_loaders(cfg: TrainConfig, stats: dict[str, FeatureStats]):
    common = loader_kwargs(cfg, stats)
    train_loader = make_reservoir_loader(
        split="train",
        num_workers=cfg.num_workers,
        prefetch_factor=cfg.prefetch_factor,
        predownload=cfg.predownload,
        windows_per_replay=cfg.windows_per_replay,
        reservoir_capacity=cfg.reservoir_capacity,
        prefetch_batches=cfg.prefetch_batches,
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
        tags=["gpt", "categorical-endpoint-flow", "legacy-codec", "inference-rtc", "045"],
        config=asdict(cfg),
    )
    if wandb.run is not None:
        for namespace in ("eval", *(f"eval_h{stride}_d{delay}" for stride, delay in evaluation_modes(cfg))):
            wandb.define_metric(f"{namespace}/net_stock_lcb", step_metric="global_step")
            wandb.define_metric(f"{namespace}/net_dmg_lcb", step_metric="global_step")
        wandb.run.summary["flow/objective_semantics"] = (
            "equal mean of buttons, main_stick, c_stick, and triggers endpoint CE"
        )
        wandb.run.summary["nll_semantics"] = (
            "train/loss retains 037's primary-plus-auxiliary reporting only; "
            "flow/objective is the optimized equal-group CE"
        )
        wandb.run.summary["training_prefixes_per_sequence"] = cfg.training_prefixes
        wandb.run.summary["flow_nfe"] = cfg.flow_nfe
        wandb.run.summary["flow_state_shape"] = f"{len(cfg.head_offsets)}x{FLOW_DIM}"
        wandb.run.summary["codec/version"] = cfg.codec_version
        wandb.run.summary["codec/group_vocabularies"] = list(GROUP_VOCABS)
        wandb.run.summary["rtc/training_conditioned"] = False
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
        wandb.run.summary["training/flow_prefix_problems"] = cfg.batch_size * cfg.training_prefixes * cfg.max_steps
        wandb.run.summary["training/approx_flow_expert_flops"] = (
            6 * decoder_mac_estimate(cfg)["flow_macs_per_nfe"] * cfg.batch_size * cfg.training_prefixes * cfg.max_steps
        )
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
    flow_fn: Callable = model.flow
    if DEVICE == "cuda" and cfg.compile_trunk:
        trunk_fn = torch.compile(trunk_fn, dynamic=False)
    if DEVICE == "cuda" and cfg.compile_flow:
        flow_fn = torch.compile(flow_fn, dynamic=False)

    train_loader, val_cache = _make_loaders(cfg, stats)
    iterator = iter(train_loader)
    run_started = time.monotonic()
    # CUDA compilation remains on the training thread; background compilation
    # deadlocked historical H100 and L40S jobs.
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
            optimizer.zero_grad()
            with profile("step") as stopwatch:
                batch = batch.to(DEVICE)
                loss, nll, tau, targets, logits, gate_means = microbatch_loss(
                    model,
                    batch,
                    cfg,
                    trunk_fn=trunk_fn,
                    flow_fn=flow_fn,
                )
                loss.backward()
                if step == start_step and wandb.run is not None:
                    wandb.run.summary["parameters/receiving_grad"] = parameter_counts(model)["receiving_grad"]
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
                trunk_gradient_norm = _mean_gradient_norm(model.trunk.parameters())
                expert_gradient_norm = _mean_gradient_norm(model.flow.parameters())
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError(f"step {step}: non-finite gradient norm {gradient_norm}")
                optimizer.step()
                scheduler.step()
                if DEVICE == "cuda":
                    torch.cuda.synchronize()
            metrics = flow_metrics(
                nll.cpu(),
                targets.cpu(),
                {name: value.cpu() for name, value in logits.items()},
                tau.cpu(),
                cfg.head_offsets,
            )
            metric_log = {
                (name if name.startswith("flow/") else f"train/{name}"): value for name, value in metrics.items()
            }
            log = {
                "global_step": step,
                "samples": (step + 1) * cfg.batch_size,
                **metric_log,
                "train/grad_norm": float(gradient_norm),
                "train/trunk_grad_norm": trunk_gradient_norm,
                "train/action_expert_grad_norm": expert_gradient_norm,
                "throughput/step_s": stopwatch.elapsed,
                "throughput/loader_wait_s": loader_wait,
                "throughput/samples_per_s": cfg.batch_size / stopwatch.elapsed,
                "throughput/samples_per_wall_s": cfg.batch_size / (stopwatch.elapsed + loader_wait),
                "throughput/prefixes_per_s": cfg.batch_size * cfg.training_prefixes / stopwatch.elapsed,
                "lr/muon": next(group["lr"] for group in optimizer.param_groups if group["use_muon"]),
                "lr/adam": next(group["lr"] for group in optimizer.param_groups if not group["use_muon"]),
            }
            for block in range(cfg.flow_layers):
                log[f"adaln/block_{block}/attention_gate_abs_mean"] = float(gate_means[block, 0])
                log[f"adaln/block_{block}/ffn_gate_abs_mean"] = float(gate_means[block, 1])
            if DEVICE == "cuda":
                log["hardware/peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 2**30
                log["hardware/peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30
            wandb.log(log)
            if step < 10 or step % 50 == 0:
                print(
                    f"[t+{time.monotonic() - run_started:.0f}s] step {step}: "
                    f"{metrics['flow/objective']:.3f} nats endpoint CE, "
                    f"{cfg.batch_size / stopwatch.elapsed:.0f} samples/s",
                    flush=True,
                )
            val_due = cfg.val_every > 0 and step > 0 and step % cfg.val_every == 0
            ckpt_due = cfg.ckpt_every > 0 and step > 0 and step % cfg.ckpt_every == 0
            checkpoint_path = run_dir / "latest.pt"
            if val_due or ckpt_due:
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
        eval_inference = BF16Inference(model, cfg)
        for stride, delay in evaluation_modes(cfg):
            mode = f"h{stride}_d{delay}"
            final_eval = eval_vs_cpu(
                model,
                stats,
                cfg,
                n_matchups=cfg.final_eval_n_matchups,
                replay_dir=replay_dir / f"final_{mode}",
                execution_stride=stride,
                committed_frames=delay,
                checkpoint_sha256=checkpoint_sha,
                inference=eval_inference,
            )
            values = {f"eval_{mode}/{name}": value for name, value in final_eval.items()}
            if (stride, delay) == (4, 0):
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
                wandb.run.summary["latency/nfe"] = cfg.flow_nfe
    finally:
        if uploader is not None:
            uploader.upload_tree(replay_dir, base=run_dir)
            uploader.close()
        wandb.finish()


_CHECKPOINT_ARCH_FIELDS = {
    "experiment_id",
    "d_model",
    "n_layers",
    "n_heads",
    "L_ctx",
    "head_offsets",
    "sample_chunk_length",
    "training_prefixes",
    "codec_version",
    "flow_d_model",
    "flow_layers",
    "flow_heads",
    "flow_ff_dim",
    "flow_time_embed_dim",
    "flow_context_dim",
    "flow_condition_dim",
    "button_proj_dim",
    "main_proj_dim",
    "cstick_proj_dim",
    "trigger_proj_dim",
    "action_embed_dim",
    "flow_epsilon",
    "flow_nfe",
    "observation_bundle",
}


def _checkpoint_config(cfg: TrainConfig) -> dict[str, object]:
    return {"experiment_id": _EXPERIMENT_ID, **asdict(cfg)}


def config_from_state(values: dict) -> TrainConfig:
    """Restore only an explicitly identified experiment-045 checkpoint."""
    missing = _CHECKPOINT_ARCH_FIELDS - values.keys()
    if missing:
        raise ValueError(f"checkpoint is not experiment 045; missing {sorted(missing)}")
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
    execution_stride: int | None = None,
    committed_frames: int | None = None,
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
    stride = cfg.execution_stride if execution_stride is None else execution_stride
    delay = cfg.committed_frames if committed_frames is None else committed_frames
    prediction_frames = runtime_prediction_frames(cfg, stride, delay)
    default_name = f"eval_replays_h{stride}_d{delay}"
    if output_name is not None and (Path(output_name).name != output_name or output_name in ("", ".", "..")):
        raise ValueError(f"evaluation output name must be one directory name, got {output_name!r}")
    replay_dir = Path(path).resolve().parent / (default_name if output_name is None else output_name)
    values = eval_vs_cpu(
        model,
        stats,
        cfg,
        n_matchups=cfg.final_eval_n_matchups if n_matchups is None else n_matchups,
        replay_dir=replay_dir,
        execution_stride=stride,
        committed_frames=delay,
        checkpoint_sha256=_checkpoint_sha256(Path(path)),
    )
    print(f"[eval] step={state['step']} (P,S,D)=({prediction_frames},{stride},{delay}): {values}", flush=True)
    return values


def benchmark_model(model: GPT, cfg: TrainConfig, *, iterations: int, rows: int) -> dict[str, float]:
    """Measure each production runtime mode through the full replan path."""
    if iterations < 1 or rows < 1:
        raise ValueError("latency iterations and rows must be positive")
    device = next(model.parameters()).device
    ctx = synthetic_context(cfg, rows, device)
    inference = BF16Inference(model, cfg, bucket=covering_power_of_two(rows))
    was_training = model.training
    model.eval()
    out: dict[str, float] = {}
    try:
        flops = inference_flops_per_replan(model, cfg)
        expert_flops = 2 * decoder_mac_estimate(cfg)["flow_macs_per_nfe"]
        for stride, delay in evaluation_modes(cfg):
            prediction_frames = runtime_prediction_frames(cfg, stride, delay)
            committed = None if delay == 0 else np.zeros((rows, delay, A_DIM), dtype=np.float32)
            for _ in range(3):
                inference.decode(
                    ctx,
                    prediction_frames,
                    execution_stride=stride,
                    committed=committed,
                )
            samples = []
            for _ in range(iterations):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                started = time.perf_counter()
                inference.decode(
                    ctx,
                    prediction_frames,
                    execution_stride=stride,
                    committed=committed,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                samples.append(1_000 * (time.perf_counter() - started))
            p50, p95, p99 = np.percentile(np.asarray(samples), (50, 95, 99))
            prefix = f"h{stride}_d{delay}"
            out[f"{prefix}/replan_p50_ms"] = float(p50)
            out[f"{prefix}/replan_p95_ms"] = float(p95)
            out[f"{prefix}/replan_p99_ms"] = float(p99)
            out[f"{prefix}/model_decode_p50_ms"] = float(p50)
            out[f"{prefix}/model_decode_p95_ms"] = float(p95)
            out[f"{prefix}/model_decode_p99_ms"] = float(p99)
            out[f"{prefix}/decoder_calls_per_replan"] = float(cfg.flow_nfe)
            out[f"{prefix}/nfe"] = float(cfg.flow_nfe)
            out[f"{prefix}/prediction_frames"] = float(prediction_frames)
            out[f"{prefix}/execution_stride"] = float(stride)
            out[f"{prefix}/committed_frames"] = float(delay)
            out[f"{prefix}/action_expert_flops_per_nfe"] = float(expert_flops)
            out[f"{prefix}/action_expert_flops_per_replan"] = float(expert_flops * cfg.flow_nfe)
            out[f"{prefix}/inference_flops_per_replan"] = float(flops)
            out[f"{prefix}/amortized_flops_per_executed_frame"] = float(flops / stride)
            out[f"{prefix}/amortized_ms_per_executed_frame"] = float(p50 / stride)
    finally:
        model.train(was_training)
    return out


def run_benchmark(cfg: TrainConfig, *, iterations: int = 20) -> dict[str, float]:
    validate_config(cfg)
    model = GPT(cfg).to(DEVICE)
    out = benchmark_model(model, cfg, iterations=iterations, rows=1)
    print(json.dumps(out, indent=2, sort_keys=True), flush=True)
    return out


@dataclass
class Args:
    cfg: TrainConfig = dataclass_field(default_factory=TrainConfig)
    comment: str = ""
    dense_16: bool = False
    resume: str | None = None
    eval: str | None = None
    eval_execution_stride: int | None = None
    eval_committed_frames: int | None = None
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


def main(args: Args) -> None:
    modes = {
        "--benchmark": args.benchmark,
        "--eval": args.eval is not None,
        "--self-play-eval": args.self_play_eval is not None,
        "--resume": args.resume is not None,
    }
    selected_modes = [name for name, selected in modes.items() if selected]
    if len(selected_modes) > 1:
        raise SystemExit(f"pass only one mode, got {', '.join(selected_modes)}")

    if args.benchmark:
        cfg = dense_16_config(args.cfg) if args.dense_16 else args.cfg
        run_benchmark(cfg, iterations=args.benchmark_iterations)
        return
    if args.eval is not None:
        eval_checkpoint(
            args.eval,
            execution_stride=args.eval_execution_stride,
            committed_frames=args.eval_committed_frames,
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
    cfg = dense_16_config(args.cfg) if args.dense_16 else args.cfg
    requested_run_name = production_run_name(cfg)
    validate_production_config(cfg)
    if args.resume is not None:
        resume_state = load_for_resume(args.resume, Path("runs") / args.resume, device=DEVICE)
        if resume_state is None:
            raise SystemExit(f"no latest.pt for run {args.resume!r}")
        resume_run = args.resume
        cfg = config_from_state(resume_state["cfg"])
        requested_run_name = None
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
