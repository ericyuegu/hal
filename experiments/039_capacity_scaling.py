"""Capacity, processed-exposure, and real-delay scaling from experiment 026.

Experiment 026 remains frozen. This file keeps its observation trunk, controller
codec, selected autoregressive loss heads, and evaluator. The scaled family
changes only the requested trunk geometry, slower-growing temporal geometry,
and the 36-frame dense internal temporal rollout needed for delayed control.

Run:
    uv run experiments/039_capacity_scaling.py --model-l 5 --phase prefix
    uv run experiments/039_capacity_scaling.py --model-l 5 --phase cooldown --target-d-exp 26
"""

from __future__ import annotations

import contextlib
import hashlib
import itertools
import json
import math
import tempfile
import time
from collections import deque
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
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
import wandb
from botocore.exceptions import ClientError
from torch import Tensor

from hal import r2
from hal import streams
from hal.data.behavior import HITSTUN_ACTIONS
from hal.data.feature_stats import FeatureStats
from hal.data.policy_schema import policy_replay_identity
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
from hal.eval.self_play import canonical_context
from hal.eval.self_play import synthetic_context
from hal.sim.inputs import ControllerInputs
from hal.sim.inputs import action_vec_to_controller
from hal.sim.process_vec import ProcessVecTelemetry
from hal.sim.rollout import ObservationRow
from hal.sim.rollout import PolicyRuntimeSpec
from hal.sim.rollout import covering_power_of_two
from hal.sim.vec import Slot
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
from hal.training.features import NEUTRAL_ACTION
from hal.training.features import SPATIAL_COLUMNS_LEAN
from hal.training.features import SPATIAL_MASKS
from hal.training.features import V6_PLAYER_COLUMNS
from hal.training.features import Context
from hal.training.features import TrainBatch
from hal.training.features import stack_actions
from hal.training.muon import SingleDeviceMuonWithAuxAdam
from hal.training.replay_reservoir import make_reservoir_loader
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
CAPACITY_LEVELS: tuple[int, ...] = (5, 7, 10, 13, 16, 18)
GRAD_ACCUM_BY_LEVEL: dict[int, int] = {5: 32, 7: 32, 10: 32, 13: 64, 16: 64, 18: 128}
BASELINE_026_GRAD_ACCUM = 32
PROCESSED_POSITION_EXPONENTS: tuple[int, ...] = (26, 27, 28, 29, 30)
EXACT_ISOFLOP_ENDPOINTS_BY_LEVEL: dict[int, tuple[int, ...]] = {
    5: (1_621_761_184,),
    7: (1_141_118_368, 3_803_727_888),
    10: (1_493_689_000,),
}
EXACT_ISOFLOP_PREFIX_SOURCE_BY_LEVEL: dict[int, int] = {
    5: 2**30,
    7: 2**30,
    10: 2**30,
}
DELAY_BUCKETS: tuple[int, ...] = (1, 2, 4, 6, 8, 10, 12, 14, 16)
HEAD_OFFSETS: tuple[int, ...] = tuple(range(1, 37))
FRAME_TIME_MS = 1000.0 / 60.0
LATENCY_ARTIFACT_SCHEMA = 3
LATENCY_START_BOUNDARY = "earliest_worker_observation_preprocessing"
LATENCY_END_BOUNDARY = "final_controller_pipe_ack"
WANDB_GROUP = "026-capacity-scaling-data-delay"

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
    model_family: str = "scaled"  # or 026-baseline
    # Scaled L=5 treatment. ``scaled_config`` changes these three together.
    d_model: int = 320
    n_layers: int = 5
    n_heads: int = 5
    attn_window: int = 0
    require_flex: bool = False
    L_ctx: int = 128

    decoder_arch_version: int = 4
    sample_chunk_length: int = 36
    head_offsets: tuple[int, ...] = HEAD_OFFSETS
    temporal_d_model: int = 128
    temporal_layers: int = 2
    temporal_heads: int = 4
    temporal_ff_dim: int = 256
    group_head_dim: int = 256
    action_embed_dim: int = 16
    offset_embed_dim: int = 16
    aux_loss_weight: float = 0.5
    group_order: tuple[str, ...] = GROUP_ORDER

    action_vocab: int = 1024
    action_state_embed_dim: int = 48
    char_vocab: int = 32
    char_dim: int = 8
    stage_vocab: int = 32
    stage_dim: int = 4
    observation_bundle: str = "base"  # or v6_lean

    control_delay: int = 1
    replan_interval: int = 1
    delay_buckets: tuple[int, ...] = DELAY_BUCKETS
    evaluation_delays: tuple[int, ...] = (1,)
    decode_temp: float = 1.0
    inference_mode: str = "compiled"  # explicit "eager" is for debugging
    inference_buckets: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    # A power-of-two capacity is required by vectorized inference. Keep an
    # explicit override available for reproducibility or memory pressure.
    compiled_inference_bucket: int | None = None

    seed: int = 0
    eval_seed: int = 0
    batch_size: int = 512
    grad_accum_steps: int = 1
    muon_lr: float = 0.040
    adam_lr: float = 8.5e-4
    muon_weight_decay: float = 0.1
    adam_weight_decay: float = 0.1
    adam_tau_scaling: str = "powerlines"  # or fixed
    adam_reference_weight_decay: float = 0.1
    adam_reference_positions: int = 2**28
    adam_reference_parameters: int = 7_000_000
    adam_weight_decay_endpoint: int = 2**30
    lr_floor_ratio: float = 1e-5 / 8.5e-4
    warmup_fraction: float = 0.03
    cooldown_fraction: float = 0.125
    phase: str = "prefix"  # or cooldown
    target_processed_positions: int = 2**30
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
    eval_every: int = 0
    eval_max_frames: int = 7200
    eval_n_matchups: int = 32
    final_eval_n_matchups: int = 96
    # L40S jobs request eight physical CPU cores. More Dolphin workers make
    # Slippstream startup unreliable even when the container can burst higher.
    eval_max_parallel: int | None = 8

    data_root: str = "data/processed/ranked-anonymized-1/mds-policy-v7"
    manifest_path: str = "data/processed/ranked-anonymized-1/mds-v7/manifest.jsonl"
    unique_data_divisor: int = 1
    compact_data: bool = True
    mds_schema_version: int = 7
    cache_limit_gb: int = 128
    shuffle_block_size: int = 2000
    predownload: int = 512
    windows_per_replay: int = 4
    reservoir_capacity: int = 1024
    val_split: str = "val"
    num_workers: int = 0
    prefetch_factor: int = 2
    prefetch_batches: int = 0
    push_to_r2: bool = True
    latency_iterations: int = 100


def scaled_config(level: int, cfg: TrainConfig | None = None) -> TrainConfig:
    """Apply one prescribed capacity level without changing policy semantics."""
    if level not in CAPACITY_LEVELS:
        raise ValueError(f"capacity level must be one of {CAPACITY_LEVELS}, got {level}")
    base = TrainConfig() if cfg is None else cfg
    d_model = 64 * level
    temporal_d_model = max(128, 64 * math.ceil(d_model / 256))
    return replace(
        base,
        model_family="scaled",
        decoder_arch_version=4,
        d_model=d_model,
        n_layers=level,
        n_heads=level,
        sample_chunk_length=36,
        head_offsets=HEAD_OFFSETS,
        temporal_d_model=temporal_d_model,
        temporal_layers=max(2, math.ceil(level / 4)),
        temporal_heads=temporal_d_model // 32,
        temporal_ff_dim=2 * temporal_d_model,
        group_head_dim=2 * temporal_d_model,
        grad_accum_steps=GRAD_ACCUM_BY_LEVEL[level],
    )


def baseline_026_config(cfg: TrainConfig | None = None) -> TrainConfig:
    """Keep the 026 width-256/L8 trunk as a distinct capacity treatment.

    The policy representation and trunk are the frozen 026 baseline. The study's
    common dense 1--36 decoder is attached so this treatment can participate in
    every required delay bucket; it is therefore not a raw old 026 checkpoint.
    """
    base = TrainConfig() if cfg is None else cfg
    return replace(
        base,
        model_family="026-baseline",
        decoder_arch_version=4,
        d_model=256,
        n_layers=8,
        n_heads=4,
        sample_chunk_length=36,
        head_offsets=HEAD_OFFSETS,
        temporal_d_model=128,
        temporal_layers=2,
        temporal_heads=4,
        temporal_ff_dim=256,
        group_head_dim=256,
        grad_accum_steps=BASELINE_026_GRAD_ACCUM,
    )


def cooldown_positions(target: int, fraction: float) -> int:
    value = target * fraction
    if not value.is_integer():
        raise ValueError(f"cooldown fraction {fraction} does not give an integer position count at D={target}")
    return int(value)


def branch_position(target: int, fraction: float) -> int:
    return target - cooldown_positions(target, fraction)


def _standard_endpoints() -> tuple[int, ...]:
    return tuple(2**exponent for exponent in PROCESSED_POSITION_EXPONENTS)


def _exact_isoflop_endpoints(cfg: TrainConfig) -> tuple[int, ...]:
    if cfg.model_family != "scaled":
        return ()
    return EXACT_ISOFLOP_ENDPOINTS_BY_LEVEL.get(cfg.n_layers, ())


def _is_exact_isoflop_endpoint(cfg: TrainConfig) -> bool:
    return cfg.target_processed_positions in _exact_isoflop_endpoints(cfg)


def _adam_weight_decay_endpoint(cfg: TrainConfig) -> int:
    if cfg.target_processed_positions in _standard_endpoints():
        return 2**30
    if _is_exact_isoflop_endpoint(cfg):
        return EXACT_ISOFLOP_PREFIX_SOURCE_BY_LEVEL[cfg.n_layers]
    return cfg.target_processed_positions


def decode_horizon(delay: int, replan_interval: int) -> int:
    """Last dense offset needed for frames t+d through t+d+R-1."""
    return delay + replan_interval - 1


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
        "target_processed_positions": cfg.target_processed_positions,
    }
    for name, value in positive.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    expected_decoder_version = 4
    if cfg.decoder_arch_version != expected_decoder_version:
        raise ValueError(
            f"model family {cfg.model_family!r} requires decoder_arch_version={expected_decoder_version}, "
            f"got {cfg.decoder_arch_version}"
        )
    if cfg.d_model % cfg.n_heads or cfg.temporal_d_model % cfg.temporal_heads:
        raise ValueError("model dimensions must be divisible by their head counts")
    if (cfg.temporal_d_model // cfg.temporal_heads) % 2:
        raise ValueError("temporal head dimension must be even for rotary positions")
    offsets = tuple(cfg.head_offsets)
    if offsets != tuple(sorted(set(offsets))) or not offsets or offsets[0] != 1:
        raise ValueError(f"head_offsets must be sorted, unique, and start at 1, got {offsets}")
    if offsets[-1] > cfg.sample_chunk_length:
        raise ValueError("head_offsets extend beyond sample_chunk_length")
    if cfg.model_family == "scaled":
        level = cfg.n_layers
        expected = scaled_config(level, cfg)
        geometry = (
            "d_model",
            "n_heads",
            "sample_chunk_length",
            "head_offsets",
            "temporal_d_model",
            "temporal_layers",
            "temporal_heads",
            "temporal_ff_dim",
            "group_head_dim",
        )
        mismatches = {
            name: (getattr(expected, name), getattr(cfg, name))
            for name in geometry
            if getattr(expected, name) != getattr(cfg, name)
        }
        if mismatches:
            raise ValueError(f"scaled model geometry is inconsistent at L={level}: {mismatches}")
    elif cfg.model_family != "026-baseline":
        raise ValueError("model_family must be 'scaled' or '026-baseline'")
    else:
        baseline = baseline_026_config(cfg)
        geometry = (
            "d_model",
            "n_layers",
            "n_heads",
            "sample_chunk_length",
            "head_offsets",
            "temporal_d_model",
            "temporal_layers",
            "temporal_heads",
            "temporal_ff_dim",
            "group_head_dim",
        )
        mismatches = {
            name: (getattr(baseline, name), getattr(cfg, name))
            for name in geometry
            if getattr(baseline, name) != getattr(cfg, name)
        }
        if mismatches:
            raise ValueError(f"026 baseline geometry is inconsistent: {mismatches}")
    if cfg.group_order != GROUP_ORDER:
        raise ValueError(f"group_order must be {GROUP_ORDER}, got {cfg.group_order}")
    if cfg.batch_size % cfg.grad_accum_steps:
        raise ValueError("batch_size must be divisible by grad_accum_steps")
    if cfg.control_delay < 1 or cfg.replan_interval < 1:
        raise ValueError("control delay and replanning interval must be positive")
    if decode_horizon(cfg.control_delay, cfg.replan_interval) > cfg.sample_chunk_length:
        raise ValueError("control delay plus replanning interval exceeds the dense decoder horizon")
    if cfg.delay_buckets != DELAY_BUCKETS:
        raise ValueError(f"delay_buckets are frozen to {DELAY_BUCKETS}")
    if not cfg.evaluation_delays or any(delay not in DELAY_BUCKETS for delay in cfg.evaluation_delays):
        raise ValueError(f"evaluation_delays must be a non-empty subset of {DELAY_BUCKETS}")
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
    if not math.isfinite(cfg.muon_weight_decay) or cfg.muon_weight_decay < 0:
        raise ValueError("muon_weight_decay must be finite and non-negative")
    if not math.isfinite(cfg.adam_weight_decay) or cfg.adam_weight_decay < 0:
        raise ValueError("adam_weight_decay must be finite and non-negative")
    if cfg.adam_tau_scaling not in ("powerlines", "fixed"):
        raise ValueError("adam_tau_scaling must be 'powerlines' or 'fixed'")
    if cfg.phase not in ("prefix", "cooldown"):
        raise ValueError("phase must be 'prefix' or 'cooldown'")
    if cfg.unique_data_divisor not in (1, 2, 4):
        raise ValueError("unique_data_divisor must be 1, 2, or 4")
    standard_endpoints = _standard_endpoints()
    if cfg.target_processed_positions not in standard_endpoints and not _is_exact_isoflop_endpoint(cfg):
        raise ValueError("target_processed_positions is not a study endpoint")
    expected_decay_endpoint = _adam_weight_decay_endpoint(cfg)
    if cfg.adam_weight_decay_endpoint != expected_decay_endpoint:
        raise ValueError(
            f"adam_weight_decay_endpoint must be {expected_decay_endpoint} for this endpoint, "
            f"got {cfg.adam_weight_decay_endpoint}"
        )
    for name, value in (("warmup_fraction", cfg.warmup_fraction), ("cooldown_fraction", cfg.cooldown_fraction)):
        if not math.isfinite(value) or not 0 < value < 1:
            raise ValueError(f"{name} must be finite and strictly between zero and one")
    cooldown_positions(cfg.target_processed_positions, cfg.cooldown_fraction)
    if not math.isfinite(cfg.lr_floor_ratio) or not 0 <= cfg.lr_floor_ratio <= 1:
        raise ValueError("lr_floor_ratio must be finite and between zero and one")
    if cfg.aux_loss_weight != 0.5:
        raise ValueError("aux_loss_weight is frozen to 0.5")
    if cfg.amp_dtype not in ("bfloat16", "float32"):
        raise ValueError("amp_dtype must be bfloat16 or float32")
    if cfg.reservoir_capacity < 2 * micro_batch_size(cfg):
        raise ValueError("reservoir_capacity must be at least twice the micro-batch size")
    if not cfg.compact_data:
        raise ValueError("capacity scaling requires the exactly resumable compact replay loader")
    if cfg.num_workers != 0 or cfg.prefetch_batches != 0:
        raise ValueError("exact branch resumption requires num_workers=0 and prefetch_batches=0")


def micro_batch_size(cfg: TrainConfig) -> int:
    return cfg.batch_size // cfg.grad_accum_steps


def _eval_parallelism(cfg: TrainConfig, n_matchups: int) -> int:
    # ``run_matches_vec`` accepts a power-of-two capacity and then limits the
    # active worker count to ``n_matchups``. Keep that capacity a valid bucket
    # when an ad hoc evaluation asks for, for example, 12 matchups.
    requested = covering_power_of_two(resolve_parallelism(n_matchups, cfg.eval_max_parallel))
    available = usable_cpus()
    cpu_cap = 1 << (available.bit_length() - 1)
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


def _planned_inference_programs(cfg: TrainConfig) -> tuple[tuple[int, int], ...]:
    scheduled = [(cfg.final_eval_n_matchups, decode_horizon(delay, delay)) for delay in cfg.evaluation_delays]
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
        expected = tuple(range(1, len(offsets) + 1))
        if not offsets or offsets != expected or offsets[-1] > self.head_offsets[-1]:
            raise ValueError(f"live decode requires one dense prefix of 1..{self.head_offsets[-1]}, got {offsets}")
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


def objective(parts: ActionLoss, aux_loss_weight: float = 0.5) -> Tensor:
    """Mean offsets 1-16 plus the configured mean weight on offsets 17-36."""
    joint = parts.nll.sum(dim=-1)
    primary = joint[:, :16].mean()
    auxiliary = joint[:, 16:].mean()
    return primary + aux_loss_weight * auxiliary


def nll_mean_metrics(mean_nll: Tensor, offsets: tuple[int, ...]) -> dict[str, float]:
    if mean_nll.shape != (len(offsets), N_GROUPS):
        raise ValueError(f"mean NLL has shape {tuple(mean_nll.shape)}")
    joint = mean_nll.sum(dim=-1) / _LN2
    out = {
        "loss": float(joint[:16].mean() + 0.5 * joint[16:].mean()),
        "primary_nll": float(joint[:16].mean()),
        "auxiliary_nll": float(joint[16:].mean()),
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
            offsets = tuple(range(1, horizon + 1))

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
        if not 1 <= horizon <= self.cfg.sample_chunk_length:
            raise ValueError(f"decode horizon must be in [1, {self.cfg.sample_chunk_length}], got {horizon}")
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
                    hidden, observed[:, -1], tuple(range(1, horizon + 1)), argmax=True
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


@dataclass(frozen=True, slots=True)
class HeldPlan:
    observation_frame: int
    actions: np.ndarray


@dataclass
class DelayTelemetry:
    latency_ms: list[float] = dataclass_field(default_factory=list)
    request_rows: int = 0
    request_seconds: float = 0.0
    deadline_misses: int = 0
    observation_ages: list[int] = dataclass_field(default_factory=list)
    neutral_actions: int = 0

    def record_request(self, *, rows: int, seconds: float, delay: int) -> None:
        milliseconds = 1_000 * seconds
        self.latency_ms.append(milliseconds)
        self.request_rows += rows
        self.request_seconds += seconds
        self.deadline_misses += rows * int(milliseconds > delay * FRAME_TIME_MS)

    def metrics(self) -> dict[str, float]:
        latency = np.asarray(self.latency_ms, dtype=np.float64)
        ages = np.asarray(self.observation_ages, dtype=np.float64)
        return {
            "latency_p50_ms": float(np.percentile(latency, 50)) if len(latency) else 0.0,
            "latency_p95_ms": float(np.percentile(latency, 95)) if len(latency) else 0.0,
            "sustained_inference_rows_per_s": self.request_rows / max(self.request_seconds, 1e-12),
            "deadline_misses": float(self.deadline_misses),
            "deadline_miss_fraction": self.deadline_misses / max(self.request_rows, 1),
            "observation_age_mean": float(ages.mean()) if len(ages) else 0.0,
            "observation_age_p50": float(np.percentile(ages, 50)) if len(ages) else 0.0,
            "observation_age_p95": float(np.percentile(ages, 95)) if len(ages) else 0.0,
            "neutral_actions": float(self.neutral_actions),
        }


class AbsoluteDelayPolicy:
    """Execute dense predictions by absolute game-frame target."""

    def __init__(
        self,
        predict: Callable[[Context], np.ndarray],
        stats: dict[str, FeatureStats],
        cfg: TrainConfig,
        *,
        delay: int,
        replan_interval: int,
        telemetry: DelayTelemetry | None,
        device: str,
        float_dtype: torch.dtype,
        inference_batch_rows: Callable[[int], int] | None = None,
    ) -> None:
        self.predict = predict
        self.delay = delay
        self.replan_interval = replan_interval
        self.horizon = decode_horizon(delay, replan_interval)
        self.context_frames = cfg.L_ctx
        self.telemetry = telemetry
        self.inference_batch_rows = (lambda rows: rows) if inference_batch_rows is None else inference_batch_rows
        v6 = cfg.observation_bundle == "v6_lean"
        self.context = RecedingHorizon(
            predict_chunk=lambda ctx, committed: np.empty((ctx.ctx_pad.shape[0], 1, A_DIM), dtype=np.float32),
            stats=stats,
            L_ctx=cfg.L_ctx,
            L_chunk=1,
            s=1,
            d=0,
            device=device,
            float_dtype=float_dtype,
            extra=V6_PLAYER_COLUMNS if v6 else None,
            projection=None if v6 else BASE_ACTION_PROJECTION,
        )
        self.last_observation: dict[Slot, int] = {}
        self.last_request: dict[Slot, int] = {}
        self.plans: dict[Slot, list[HeldPlan]] = {}
        self.last_plan_inferred = False
        self.last_inference_rows = 0
        self.last_inference_batch_rows = 0

    @property
    def runtime_spec(self) -> PolicyRuntimeSpec:
        """Request every game frame; inference itself remains on the R-frame clock.

        Returning one executed action per broker request is what lets an inference
        result be held until its absolute release frame without pausing or slowing
        the GPU. The model still decodes the common dense horizon in one request.
        """
        return PolicyRuntimeSpec(
            context_frames=self.context_frames,
            prediction_frames=1,
            execution_stride=1,
            committed_frames=0,
            action_dim=A_DIM,
        )

    def _observe_frame(self, slot: Slot, observation_frame: int, *, reset: bool) -> None:
        previous = self.last_observation.get(slot)
        if reset or (previous is not None and observation_frame < previous):
            self.last_request.pop(slot, None)
            self.plans.pop(slot, None)
        self.last_observation[slot] = observation_frame

    def _actions_for_live(self, live: list[Slot], call_started: float) -> dict[Slot, np.ndarray]:
        due = [
            slot
            for slot in live
            if slot not in self.last_request
            or self.last_observation[slot] - self.last_request[slot] >= self.replan_interval
        ]
        self.last_plan_inferred = bool(due)
        self.last_inference_rows = len(due)
        self.last_inference_batch_rows = self.inference_batch_rows(len(due)) if due else 0
        if due:
            batch = self.context._context(due)
            predicted = self.predict(batch)
            expected = (len(due), self.horizon, A_DIM)
            if predicted.shape != expected:
                raise ValueError(f"delay policy got plan shape {predicted.shape}, expected {expected}")
            for row, slot in enumerate(due):
                anchor = self.last_observation[slot]
                self.plans.setdefault(slot, []).append(HeldPlan(anchor, predicted[row]))
                self.last_request[slot] = anchor

        actions: dict[Slot, np.ndarray] = {}
        for slot in live:
            target_frame = self.last_observation[slot] + 1
            candidates = [
                plan
                for plan in self.plans.get(slot, ())
                if plan.observation_frame + self.delay <= target_frame
                and target_frame <= plan.observation_frame + self.horizon
            ]
            if candidates:
                plan = max(candidates, key=lambda item: item.observation_frame)
                offset = target_frame - plan.observation_frame
                action = plan.actions[offset - 1]
                if self.telemetry is not None:
                    self.telemetry.observation_ages.append(offset)
            else:
                action = NEUTRAL_ACTION
                if self.telemetry is not None:
                    self.telemetry.neutral_actions += 1
            actions[slot] = np.asarray(action, dtype=np.float32)
            self.plans[slot] = [
                plan for plan in self.plans.get(slot, ()) if plan.observation_frame + self.horizon >= target_frame
            ]
        if due and self.telemetry is not None:
            self.telemetry.record_request(
                rows=len(due),
                seconds=time.perf_counter() - call_started,
                delay=self.delay,
            )
        return actions

    def __call__(self, frame_index: int, obs: Mapping[Slot, dict]) -> Mapping[Slot, ControllerInputs]:
        del frame_index
        call_started = time.perf_counter()
        live = list(obs)
        for slot in live:
            self._observe_frame(slot, int(obs[slot]["id"]), reset=False)
        self.context._ingest(live, obs)
        actions = self._actions_for_live(live, call_started)
        for slot in live:
            self.context._push_ego(slot, actions[slot])
        return {slot: action_vec_to_controller(action) for slot, action in actions.items()}

    def plan_rows(self, rows: Mapping[Slot, Sequence[ObservationRow]]) -> Mapping[Slot, np.ndarray]:
        """Use worker-published absolute frames on the production IPC path."""
        call_started = time.perf_counter()
        live = list(rows)
        for slot in live:
            slot_rows = rows[slot]
            if len(slot_rows) != 1:
                raise ValueError(f"absolute-delay worker must publish one frame, got {len(slot_rows)} for {slot}")
            row = slot_rows[0]
            self._observe_frame(slot, row.frame_id, reset=row.reset)
            self.context._ingest_row(slot, row)
        actions = self._actions_for_live(live, call_started)
        return {slot: action[None] for slot, action in actions.items()}


def make_policy(
    model: GPT,
    stats: dict[str, FeatureStats],
    cfg: TrainConfig,
    *,
    delay: int | None = None,
    replan_interval: int | None = None,
    decode_seed: int | None = None,
    inference: BF16Inference | None = None,
    telemetry: DelayTelemetry | None = None,
    device: str = DEVICE,
) -> AbsoluteDelayPolicy:
    selected_delay = cfg.control_delay if delay is None else delay
    selected_interval = cfg.replan_interval if replan_interval is None else replan_interval
    horizon = decode_horizon(selected_delay, selected_interval)
    if horizon > cfg.sample_chunk_length:
        raise ValueError("requested delay plan exceeds the trained dense horizon")
    engine = BF16Inference(model, cfg) if inference is None else inference
    random_streams = None if decode_seed is None else SlotGroupRandom(decode_seed)
    generator = None if decode_seed is None else torch.Generator(device=device).manual_seed(decode_seed)

    @torch.no_grad()
    def predict(ctx: Context) -> np.ndarray:
        return engine.decode(ctx, horizon, streams=random_streams, gen=generator).cpu().numpy()

    return AbsoluteDelayPolicy(
        predict,
        stats,
        cfg,
        delay=selected_delay,
        replan_interval=selected_interval,
        telemetry=telemetry,
        device=device,
        float_dtype=next(model.parameters()).dtype,
        inference_batch_rows=engine._bucket,
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
    control_delay: int
    replan_interval: int
    decode_horizon: int
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
    control_delay: int,
    replan_interval: int,
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
        control_delay=control_delay,
        replan_interval=replan_interval,
        decode_horizon=decode_horizon(control_delay, replan_interval),
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
        (
            replay_dir / "metrics.json",
            {"schema_version": 1, "protocol": asdict(protocol), "metrics": metrics},
        ),
    ):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True))
        temporary.replace(path)


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


def _cluster_lcb(rows: list[MatchRow], values: np.ndarray, *, seed: int) -> float:
    if not rows:
        return 0.0
    by_boot: dict[tuple[int, int], list[float]] = {}
    for row, value in zip(rows, values, strict=True):
        by_boot.setdefault((row.stage, row.boot_index), []).append(float(value))
    boot_means = np.asarray([np.mean(items) for items in by_boot.values()], dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(boot_means), size=(BOOTSTRAP_RESAMPLES, len(boot_means)))
    return float(np.percentile(boot_means[indices].mean(axis=1), 5.0))


def closed_loop_difference_metrics(rows: list[MatchRow], *, seed: int) -> dict[str, float]:
    """Add raw episode differences and one-sided boot-cluster LCBs."""
    if not rows:
        return {
            "win_rate": 0.0,
            "mean_stock_difference": 0.0,
            "stock_lcb": 0.0,
            "mean_damage_difference": 0.0,
            "damage_lcb": 0.0,
        }
    stock = np.asarray([row.stocks_taken - row.stocks_lost for row in rows], dtype=np.float64)
    damage = np.asarray([row.damage_dealt - row.damage_taken for row in rows], dtype=np.float64)
    won = np.asarray([row.stocks_taken == 4 and row.stocks_lost < 4 for row in rows], dtype=np.float64)
    return {
        "win_rate": float(won.mean()),
        "mean_stock_difference": float(stock.mean()),
        "stock_lcb": _cluster_lcb(rows, stock, seed=seed),
        "mean_damage_difference": float(damage.mean()),
        "damage_lcb": _cluster_lcb(rows, damage, seed=seed + 1),
    }


def eval_vs_cpu(
    model: GPT,
    stats: dict[str, FeatureStats],
    cfg: TrainConfig,
    *,
    n_matchups: int,
    replay_dir: Path,
    delay: int | None = None,
    replan_interval: int | None = None,
    checkpoint_sha256: str = "unavailable",
    inference: BF16Inference | None = None,
    require_compiled_cuda: bool = True,
) -> dict[str, float]:
    selected_delay = cfg.control_delay if delay is None else delay
    selected_interval = cfg.replan_interval if replan_interval is None else replan_interval
    inference = BF16Inference(model, cfg) if inference is None else inference
    if inference.model is not model:
        raise ValueError("the supplied inference engine must own the evaluation model")
    protocol = _eval_protocol(
        cfg,
        model,
        n_matchups=n_matchups,
        control_delay=selected_delay,
        replan_interval=selected_interval,
        checkpoint_sha256=checkpoint_sha256,
        inference_compile_mode=inference.compile_mode,
    )
    if (
        require_compiled_cuda
        and next(model.parameters()).device.type == "cuda"
        and (protocol.inference_mode != "compiled" or not inference.compiled)
    ):
        raise RuntimeError("official CUDA evaluation requires compiled BF16 inference")
    telemetry = DelayTelemetry()
    process_telemetry = ProcessVecTelemetry()
    policy_index = itertools.count()

    def factory() -> AbsoluteDelayPolicy:
        return make_policy(
            model,
            stats,
            cfg,
            delay=selected_delay,
            replan_interval=selected_interval,
            decode_seed=protocol.seed + next(policy_index),
            inference=inference,
            telemetry=telemetry,
        )

    was_training = model.training
    model.eval()
    total_started = time.perf_counter()
    try:
        compile_seconds = inference.prewarm(protocol.max_parallel, protocol.decode_horizon)
        started = time.perf_counter()
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
    metrics["eval_wall_seconds"] = time.perf_counter() - started
    metrics["eval_total_wall_seconds"] = time.perf_counter() - total_started
    metrics["inference_compile_seconds"] = compile_seconds
    metrics["control_delay"] = float(selected_delay)
    metrics["replan_interval"] = float(selected_interval)
    metrics["decode_horizon"] = float(protocol.decode_horizon)
    metrics.update(telemetry.metrics())
    metrics.update(process_telemetry.metrics())
    broker_latency = np.asarray(process_telemetry.inference_latency_ms, dtype=np.float64)
    broker_rows = np.asarray(process_telemetry.inference_rows_by_call, dtype=np.int64)
    deadline = selected_delay * FRAME_TIME_MS
    misses = int(broker_rows[broker_latency > deadline].sum()) if len(broker_latency) else 0
    metrics.update(
        {
            "latency_p50_ms": float(np.percentile(broker_latency, 50)) if len(broker_latency) else 0.0,
            "latency_p95_ms": float(np.percentile(broker_latency, 95)) if len(broker_latency) else 0.0,
            "sustained_inference_rows_per_s": process_telemetry.inference_rows
            / max(float(broker_latency.sum()) / 1_000, 1e-12),
            "deadline_misses": float(misses),
            "deadline_miss_fraction": misses / max(process_telemetry.inference_rows, 1),
            "valid_latency_bucket": float(
                misses == 0
                and len(broker_latency) > 0
                and np.percentile(broker_latency, 95) <= deadline
                and process_telemetry.failed_workers == 0
            ),
        }
    )
    metrics.update(closed_loop_difference_metrics(rows, seed=protocol.seed))
    _write_eval_evidence(replay_dir, rows, metrics, protocol)
    return metrics


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
    decoder_ids = {id(parameter) for module in (model.codec, model.temporal) for parameter in module.parameters()}
    decoder = sum(parameter.numel() for parameter in model.parameters() if id(parameter) in decoder_ids)
    total = sum(parameter.numel() for parameter in model.parameters())
    return {"trunk": total - decoder, "decoder": decoder, "total": total}


@dataclass(frozen=True, slots=True)
class AdamScale:
    tokens_per_update: int
    d_over_n: float
    tau_ref: float
    tau: float
    weight_decay: float


def adam_scale(cfg: TrainConfig, total_parameters: int) -> AdamScale:
    """Compute fixed-prefix AdamW decay from the requested Power Lines rule."""
    tokens_per_update = cfg.batch_size * cfg.L_ctx
    tau_ref = tokens_per_update / (cfg.adam_lr * cfg.adam_reference_weight_decay * cfg.adam_reference_positions)
    d_over_n = cfg.adam_weight_decay_endpoint / total_parameters
    if cfg.adam_tau_scaling == "powerlines":
        reference_tpp = cfg.adam_reference_positions / cfg.adam_reference_parameters
        tau = tau_ref * (d_over_n / reference_tpp) ** -0.52
    else:
        tau = tau_ref
    weight_decay = tokens_per_update / (cfg.adam_lr * tau * cfg.adam_weight_decay_endpoint)
    return AdamScale(
        tokens_per_update=tokens_per_update,
        d_over_n=d_over_n,
        tau_ref=tau_ref,
        tau=tau,
        weight_decay=weight_decay,
    )


def achieved_adam_tau(cfg: TrainConfig, processed_positions: int) -> float:
    return cfg.batch_size * cfg.L_ctx / (cfg.adam_lr * cfg.adam_weight_decay * processed_positions)


def warmup_positions(cfg: TrainConfig) -> int:
    # The prefix is shared by every D endpoint, so its warmup cannot depend on
    # the branch target. Anchor the one common 3% warmup to the stated D_ref.
    return math.ceil(cfg.warmup_fraction * cfg.adam_reference_positions)


def lr_multiplier(cfg: TrainConfig, processed_positions: int) -> float:
    warmup = warmup_positions(cfg)
    if processed_positions < warmup:
        return processed_positions / warmup
    if cfg.phase == "prefix":
        return 1.0
    start = branch_position(cfg.target_processed_positions, cfg.cooldown_fraction)
    if processed_positions < start:
        raise ValueError(f"cooldown starts at {start} processed positions, got {processed_positions}")
    progress = min((processed_positions - start) / (cfg.target_processed_positions - start), 1.0)
    return 1.0 + progress * (cfg.lr_floor_ratio - 1.0)


class PositionLRScheduler:
    """Set both optimizer families from exact processed-position endpoints."""

    def __init__(self, optimizer: SingleDeviceMuonWithAuxAdam, cfg: TrainConfig) -> None:
        self.optimizer = optimizer
        self.cfg = cfg
        self.base_lrs = tuple(cfg.muon_lr if group["use_muon"] else cfg.adam_lr for group in optimizer.param_groups)
        self.processed_positions = 0

    def step(self, processed_positions: int) -> None:
        if processed_positions < self.processed_positions:
            raise ValueError("processed positions cannot move backward")
        factor = lr_multiplier(self.cfg, processed_positions)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            group["lr"] = base_lr * factor
        self.processed_positions = processed_positions

    def state_dict(self) -> dict[str, object]:
        return {
            "schema": 1,
            "phase": self.cfg.phase,
            "target_processed_positions": self.cfg.target_processed_positions,
            "processed_positions": self.processed_positions,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state.get("schema") != 1:
            raise ValueError(f"unsupported position-scheduler state {state.get('schema')!r}")
        if (
            state["phase"] != self.cfg.phase
            or state["target_processed_positions"] != self.cfg.target_processed_positions
        ):
            raise ValueError("position-scheduler configuration changed across resume")
        self.step(int(state["processed_positions"]))


def model_tag(cfg: TrainConfig) -> str:
    return (
        f"cap-{cfg.model_family}-L{cfg.n_layers}-d{cfg.d_model}-Lc{cfg.L_ctx}-"
        f"t{cfg.temporal_d_model}x{cfg.temporal_layers}-dense1to36-{cfg.observation_bundle}"
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


@dataclass(frozen=True, slots=True)
class DatasetAudit:
    replay_ids: frozenset[str]
    unique_replays: int
    episode_hash: str
    unique_loss_positions: int


def _ensure_manifest(path: Path) -> Path:
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    r2.client().download_file(
        r2.bucket(),
        "processed/ranked-anonymized-1/mds-v7/manifest.jsonl",
        str(path),
    )
    return path


def dataset_audit(cfg: TrainConfig) -> DatasetAudit:
    """Resolve deterministic nested train subsets in stable MDS row order."""
    entries = []
    with _ensure_manifest(Path(cfg.manifest_path)).open() as handle:
        for line in handle:
            row = json.loads(line)
            annotation = row["annotation"]
            if annotation["split"] != "train":
                continue
            entries.append(
                (
                    int(annotation["mds_row_idx"]),
                    policy_replay_identity(row["path"]),
                    int(annotation["frame_count_actual"]),
                )
            )
    entries.sort()
    keep = len(entries) // cfg.unique_data_divisor
    selected = entries[:keep]
    digest = hashlib.sha256()
    for _, replay_id, frames in selected:
        digest.update(f"{replay_id}:{frames}\n".encode())
    replay_ids = frozenset(replay_id for _, replay_id, _ in selected)
    if len(replay_ids) != len(selected):
        raise ValueError("stable policy replay identities are not unique")
    return DatasetAudit(
        replay_ids=replay_ids,
        unique_replays=len(replay_ids),
        episode_hash=digest.hexdigest(),
        unique_loss_positions=2 * sum(max(0, frames - 1) for _, _, frames in selected),
    )


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


def _load_eval_progress(
    run_dir: Path,
    run_name: str,
    cfg: TrainConfig,
    audit: DatasetAudit,
    checkpoint_sha256: str,
) -> set[int]:
    path = run_dir / "eval_progress.json"
    if not path.is_file() and cfg.push_to_r2:
        try:
            r2.client().download_file(r2.bucket(), f"runs/{run_name}/{path.name}", str(path))
        except ClientError as error:
            if error.response.get("Error", {}).get("Code") not in {"404", "NoSuchKey", "NotFound"}:
                raise
    if not path.is_file():
        return set()
    payload = json.loads(path.read_text())
    identity = (
        payload.get("target_processed_positions"),
        payload.get("model_family"),
        payload.get("n_layers"),
        payload.get("episode_hash"),
        payload.get("checkpoint_sha256"),
    )
    expected = (
        cfg.target_processed_positions,
        cfg.model_family,
        cfg.n_layers,
        audit.episode_hash,
        checkpoint_sha256,
    )
    if identity != expected:
        raise ValueError(f"evaluation progress belongs to a different endpoint: {identity} != {expected}")
    completed = {int(delay) for delay in payload.get("completed_delays", ())}
    if not completed.issubset(DELAY_BUCKETS):
        raise ValueError(f"evaluation progress has unknown delays {sorted(completed - set(DELAY_BUCKETS))}")
    return completed


def _write_eval_progress(
    run_dir: Path,
    cfg: TrainConfig,
    audit: DatasetAudit,
    checkpoint_sha256: str,
    completed: set[int],
    uploader: BackgroundUploader | None,
) -> None:
    path = run_dir / "eval_progress.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "target_processed_positions": cfg.target_processed_positions,
                "model_family": cfg.model_family,
                "n_layers": cfg.n_layers,
                "episode_hash": audit.episode_hash,
                "checkpoint_sha256": checkpoint_sha256,
                "completed_delays": sorted(completed),
            },
            sort_keys=True,
        )
    )
    temporary.replace(path)
    if uploader is not None:
        uploader.upload(path)


def _make_train_loader(
    cfg: TrainConfig,
    stats: dict[str, FeatureStats],
    audit: DatasetAudit,
    *,
    loader_workers: int = 0,
):
    kwargs = loader_kwargs(cfg, stats)
    if cfg.compact_data:
        return make_reservoir_loader(
            split="train",
            num_workers=loader_workers,
            prefetch_factor=cfg.prefetch_factor,
            predownload=cfg.predownload,
            windows_per_replay=cfg.windows_per_replay,
            reservoir_capacity=cfg.reservoir_capacity,
            prefetch_batches=cfg.prefetch_batches,
            replay_filter=audit.replay_ids.__contains__,
            **kwargs,
        )
    return make_loader(
        split="train",
        num_workers=loader_workers,
        prefetch_factor=cfg.prefetch_factor,
        windows_per_replay=cfg.windows_per_replay,
        compact=False,
        **kwargs,
    )


def _make_loaders(
    cfg: TrainConfig,
    stats: dict[str, FeatureStats],
    audit: DatasetAudit,
    *,
    loader_workers: int = 0,
):
    kwargs = loader_kwargs(cfg, stats)
    train_loader = _make_train_loader(cfg, stats, audit, loader_workers=loader_workers)
    val_kwargs = {**kwargs, "batch_size": cfg.val_batch_size}
    val_loader = make_loader(split=cfg.val_split, num_workers=0, compact=cfg.compact_data, **val_kwargs)
    return train_loader, cache_validation(val_loader, cfg.val_n_samples)


def run_training_smoke(
    cfg: TrainConfig,
    stats: dict[str, FeatureStats],
    audit: DatasetAudit,
    *,
    output_dir: Path,
) -> dict[str, object]:
    """Run one compiled real-data forward/backward/optimizer memory gate."""
    if DEVICE != "cuda":
        raise RuntimeError("the official training smoke requires CUDA")
    validate_config(cfg)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.set_float32_matmul_precision("high" if cfg.allow_tf32 else "highest")
    model = GPT(cfg).to(DEVICE).train()
    counts = subsystem_parameter_counts(model)
    scale = adam_scale(cfg, counts["total"])
    cfg = replace(cfg, adam_weight_decay=scale.weight_decay)
    optimizer = make_optimizer(model, cfg)
    loader = _make_train_loader(cfg, stats, audit)
    batch = next(iter(loader))
    validate_batch_geometry(batch, cfg, micro_batch_size(cfg))
    batch = batch.to(DEVICE)

    def trunk_fn(features, pad, actions):
        return model(features, pad, actions)

    temporal_fn: Callable = model.temporal.teacher_forced_nll
    if cfg.compile_trunk:
        trunk_fn = torch.compile(trunk_fn, dynamic=False)
    if cfg.compile_temporal:
        temporal_fn = torch.compile(temporal_fn, dynamic=False)
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad()
    started = time.perf_counter()
    history, targets, valid = prepared_targets(model, batch)
    with amp_context(cfg, DEVICE):
        hidden = trunk_fn(batch.context.features, batch.context.ctx_pad, history)
        dense_nll = temporal_fn(hidden, history, targets)
        parts = ActionLoss(nll=dense_nll[valid], targets=targets[valid])
        loss = objective(parts, cfg.aux_loss_weight)
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
    if not torch.isfinite(gradient_norm):
        raise FloatingPointError(f"training smoke produced non-finite gradient norm {gradient_norm}")
    optimizer.step()
    torch.cuda.synchronize()
    label = "026base" if cfg.model_family == "026-baseline" else f"L{cfg.n_layers}"
    payload: dict[str, object] = {
        "schema_version": 1,
        "model": label,
        "d_model": cfg.d_model,
        "parameters": counts,
        "effective_batch_size": cfg.batch_size,
        "grad_accum_steps": cfg.grad_accum_steps,
        "micro_batch_size": micro_batch_size(cfg),
        "loss_positions": int(valid.sum()),
        "objective_nats": float(loss.detach()),
        "gradient_norm": float(gradient_norm),
        "compiled_trunk": cfg.compile_trunk,
        "compiled_temporal": cfg.compile_temporal,
        "step_seconds_including_compile": time.perf_counter() - started,
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
        "cuda_device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda or "none",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{label}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return payload


def _valid_position_mask(batch: TrainBatch, cfg: TrainConfig) -> Tensor:
    positions = torch.arange(cfg.L_ctx, device=batch.context.ctx_pad.device)
    return positions[None, :] >= batch.context.ctx_pad[:, None]


type PendingBatch = tuple[TrainBatch, Tensor]


@dataclass(frozen=True, slots=True)
class LoadedUpdate:
    work: list[PendingBatch]
    service_seconds: float


class OrderedUpdateSource:
    """Load complete optimizer updates while keeping iterator rollover ordered."""

    def __init__(self, train_loader, cfg: TrainConfig) -> None:
        self._train_loader = train_loader
        self._cfg = cfg
        self._iterator = iter(train_loader)

    def load(self, count: int) -> list[LoadedUpdate]:
        loaded = []
        for _ in range(count):
            started = time.monotonic()
            work = []
            while len(work) < self._cfg.grad_accum_steps:
                try:
                    batch = next(self._iterator)
                except StopIteration:
                    self._iterator = iter(self._train_loader)
                    batch = next(self._iterator)
                validate_batch_geometry(batch, self._cfg, micro_batch_size(self._cfg))
                work.append((batch, _valid_position_mask(batch, self._cfg).cpu()))
            loaded.append(LoadedUpdate(work=work, service_seconds=time.monotonic() - started))
        return loaded


def _select_position_work(
    work: list[PendingBatch], maximum_positions: int
) -> tuple[list[PendingBatch], list[PendingBatch], int]:
    """Select an exact prefix of loss-bearing positions and retain its complement."""
    selected: list[PendingBatch] = []
    pending: list[PendingBatch] = []
    remaining = maximum_positions
    selected_count = 0
    for batch, available in work:
        available = available.to(dtype=torch.bool, device="cpu")
        count = int(available.sum())
        if remaining >= count:
            selected.append((batch, available))
            remaining -= count
            selected_count += count
            continue
        flat = available.flatten()
        indices = flat.nonzero(as_tuple=False).flatten()
        chosen_flat = torch.zeros_like(flat)
        if remaining:
            chosen_flat[indices[:remaining]] = True
        chosen = chosen_flat.view_as(available)
        leftover = available & ~chosen
        if chosen.any():
            selected.append((batch, chosen))
            selected_count += remaining
        if leftover.any():
            pending.append((batch, leftover))
        remaining = 0
    if remaining == 0:
        # A split batch contributes to both lists, so use identity to find all
        # untouched suffix batches without relying on that count.
        used_ids = {id(batch) for batch, _ in selected} | {id(batch) for batch, _ in pending}
        pending.extend((batch, mask) for batch, mask in work if id(batch) not in used_ids)
    return selected, pending, selected_count


def _restore_rng(state: dict) -> None:
    np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available() and state.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])


def _checkpoint_extra(
    train_loader,
    *,
    pending: list[PendingBatch],
    processed_positions: int,
    training_wall_seconds: float,
    update: int,
) -> dict[str, object]:
    return {
        "checkpoint_schema": 2,
        "processed_positions": processed_positions,
        "training_wall_seconds": training_wall_seconds,
        "update": update,
        "data_state": train_loader.state_dict(),
        "pending_batches": pending,
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _prefix_branch_targets(cfg: TrainConfig) -> tuple[int, ...]:
    if not _is_exact_isoflop_endpoint(cfg):
        return (
            _standard_endpoints()
            if cfg.target_processed_positions in _standard_endpoints()
            else (cfg.target_processed_positions,)
        )
    return tuple(target for target in _exact_isoflop_endpoints(cfg) if target <= cfg.target_processed_positions)


def _prefix_branch_positions(cfg: TrainConfig) -> tuple[int, ...]:
    return tuple(branch_position(target, cfg.cooldown_fraction) for target in _prefix_branch_targets(cfg))


def _training_stop(cfg: TrainConfig) -> int:
    return max(_prefix_branch_positions(cfg)) if cfg.phase == "prefix" else cfg.target_processed_positions


def run_name_for(cfg: TrainConfig, total_parameters: int) -> str:
    capacity = "026base" if cfg.model_family == "026-baseline" else f"L{cfg.n_layers}"
    width_label = f"d{cfg.d_model}"
    parameter_label = f"{round(total_parameters / 1_000_000)}M"
    unique_label = "U1" if cfg.unique_data_divisor == 1 else f"U1d{cfg.unique_data_divisor}"
    tau_label = "tauPL" if cfg.adam_tau_scaling == "powerlines" else "tauFixed"
    if cfg.phase == "prefix":
        exposure = f"prefix-{endpoint_label(max(_prefix_branch_targets(cfg)))}"
    else:
        exposure = endpoint_label(cfg.target_processed_positions)
    return f"cap-{capacity}-{width_label}-{parameter_label}-{unique_label}-{exposure}-{tau_label}"


def endpoint_label(target: int) -> str:
    if target > 0 and target & (target - 1) == 0:
        return f"D2p{target.bit_length() - 1}"
    return f"D{target}"


def branch_checkpoint_name(target: int) -> str:
    return f"branch_{endpoint_label(target)}.pt"


def _configs_match_for_branch(source: TrainConfig, target: TrainConfig) -> None:
    allowed = {"phase", "target_processed_positions", "evaluation_delays"}
    for field in fields(TrainConfig):
        if field.name in allowed:
            continue
        if getattr(source, field.name) != getattr(target, field.name):
            raise ValueError(
                f"branch changed {field.name}: source={getattr(source, field.name)!r}, "
                f"target={getattr(target, field.name)!r}"
            )
    expected = branch_position(target.target_processed_positions, target.cooldown_fraction)
    if source.phase != "prefix":
        raise ValueError("terminal cooldown must branch from a shared prefix checkpoint")
    if expected <= 0:
        raise ValueError("terminal cooldown branch must be positive")


def _configs_match_for_prefix_fork(
    source: TrainConfig,
    target: TrainConfig,
    processed_positions: int,
) -> None:
    if source.phase != "prefix" or target.phase != "prefix":
        raise ValueError("an exact prefix fork requires prefix source and target configurations")
    if not _is_exact_isoflop_endpoint(target):
        raise ValueError("an exact prefix fork target must be the model's registered IsoFLOP endpoint")
    expected_source_target = EXACT_ISOFLOP_PREFIX_SOURCE_BY_LEVEL.get(target.n_layers)
    if source.target_processed_positions != expected_source_target:
        raise ValueError(
            f"exact prefix fork for L={target.n_layers} requires source target {expected_source_target}, "
            f"got {source.target_processed_positions}"
        )
    allowed = {"target_processed_positions", "evaluation_delays"}
    for field in fields(TrainConfig):
        if field.name in allowed:
            continue
        if getattr(source, field.name) != getattr(target, field.name):
            raise ValueError(
                f"prefix fork changed {field.name}: source={getattr(source, field.name)!r}, "
                f"target={getattr(target, field.name)!r}"
            )
    if processed_positions not in _prefix_branch_positions(source):
        raise ValueError(f"prefix fork source position {processed_positions} is not a saved branch boundary")
    stop = branch_position(target.target_processed_positions, target.cooldown_fraction)
    if processed_positions >= stop:
        raise ValueError(f"prefix fork source position {processed_positions} must precede target branch {stop}")


def _validate_prefix_fork_state(state: dict, target: TrainConfig, total_parameters: int) -> None:
    required = {
        "cfg",
        "checkpoint_schema",
        "cuda_rng_state",
        "data_state",
        "model",
        "numpy_rng_state",
        "opt",
        "pending_batches",
        "processed_positions",
        "torch_rng_state",
        "update",
    }
    missing = sorted(required - state.keys())
    if missing:
        raise ValueError(f"prefix fork checkpoint is missing state: {missing}")
    if state["checkpoint_schema"] != 2:
        raise ValueError("checkpoint predates exact position/dataloader resumption")
    scale = adam_scale(target, total_parameters)
    resolved_target = replace(target, adam_weight_decay=scale.weight_decay)
    source = config_from_state(state["cfg"])
    _configs_match_for_prefix_fork(source, resolved_target, int(state["processed_positions"]))


def train(
    cfg: TrainConfig,
    stats: dict[str, FeatureStats],
    audit: DatasetAudit,
    *,
    requested_run_name: str,
    resume_state: dict | None = None,
    branch_state: dict | None = None,
    prefix_fork_state: dict | None = None,
    loader_workers: int = 0,
    loader_prefetch_updates: int = 1,
    throughput_probe_warmup: int | None = None,
    throughput_probe_updates: int | None = None,
    throughput_probe_eager: bool = False,
) -> dict[str, object] | None:
    """Train a shared stable prefix or one exact terminal-cooldown endpoint."""
    states = [state for state in (resume_state, branch_state, prefix_fork_state) if state is not None]
    if len(states) > 1:
        raise ValueError("a run cannot resume, branch, and fork at the same time")
    if loader_workers < 0:
        raise ValueError("loader_workers must be non-negative")
    if loader_prefetch_updates not in (1, 2):
        raise ValueError("loader_prefetch_updates must be 1 or 2")
    probe = throughput_probe_warmup is not None or throughput_probe_updates is not None
    if probe and (
        throughput_probe_warmup is None
        or throughput_probe_updates is None
        or throughput_probe_warmup < 0
        or throughput_probe_updates < 1
    ):
        raise ValueError("a throughput probe requires non-negative warmup and positive measured updates")
    validate_config(cfg)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.set_float32_matmul_precision("high" if cfg.allow_tf32 else "highest")
    model = GPT(cfg).to(DEVICE)
    if throughput_probe_eager:
        model.trunk.prefer_flex = False
    counts = subsystem_parameter_counts(model)
    scale = adam_scale(cfg, counts["total"])
    cfg = replace(cfg, adam_weight_decay=scale.weight_decay)
    validate_config(cfg)
    run_name = requested_run_name
    uploader = BackgroundUploader(run_name) if cfg.push_to_r2 and not probe else None
    state = states[0] if states else None
    resume_wandb_id = None if resume_state is None else resume_state.get("wandb_id")
    wandb.init(
        project="hal",
        group=WANDB_GROUP,
        name=run_name,
        id=None if resume_state is None else resume_state.get("wandb_id"),
        resume="allow" if resume_state is not None else None,
        tags=["gpt", "temporal-mtp", "dense-1-36", "026-capacity-scaling", cfg.phase],
        config={
            **asdict(cfg),
            "unique_replays": audit.unique_replays,
            "episode_hash": audit.episode_hash,
            "unique_loss_positions": audit.unique_loss_positions,
            "trunk_parameters": counts["trunk"],
            "decoder_parameters": counts["decoder"],
            "total_parameters": counts["total"],
            "D_over_N_weight_decay_endpoint": scale.d_over_n,
            "adam_tau": scale.tau,
            "adam_weight_decay": scale.weight_decay,
        },
        mode="disabled" if probe else None,
    )
    if wandb.run is not None:
        wandb.define_metric("processed_positions")
        wandb.define_metric("train/*", step_metric="processed_positions")
        wandb.define_metric("val/*", step_metric="processed_positions")
        for delay in DELAY_BUCKETS:
            wandb.define_metric(f"eval_d{delay}/*", step_metric="processed_positions")
        if cfg.wandb_log_code:
            log_wandb_code(wandb.run)
        for name, value in counts.items():
            wandb.run.summary[f"parameters/{name}"] = value
        wandb.run.summary["data/unique_replays"] = audit.unique_replays
        wandb.run.summary["data/episode_hash"] = audit.episode_hash
        wandb.run.summary["data/unique_loss_positions"] = audit.unique_loss_positions
        wandb.run.summary["optimizer/D_over_N"] = scale.d_over_n
        wandb.run.summary["optimizer/adam_tau"] = scale.tau
        wandb.run.summary["optimizer/adam_weight_decay"] = scale.weight_decay

    run_dir, replay_dir = setup_run_dir(run_name)
    optimizer = make_optimizer(model, cfg)
    scheduler = PositionLRScheduler(optimizer, cfg)
    processed_positions = 0
    prior_training_wall_seconds = 0.0
    update = 0
    pending: list[PendingBatch] = []
    if state is not None:
        source_cfg = config_from_state(state["cfg"])
        if branch_state is not None:
            _configs_match_for_branch(source_cfg, cfg)
        elif prefix_fork_state is not None:
            _validate_prefix_fork_state(prefix_fork_state, cfg, counts["total"])
        elif source_cfg != cfg:
            raise ValueError("run configuration changed across exact resume")
        if state.get("checkpoint_schema") != 2:
            raise ValueError("checkpoint predates exact position/dataloader resumption")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["opt"])
        processed_positions = int(state["processed_positions"])
        prior_training_wall_seconds = float(state.get("training_wall_seconds", 0.0))
        update = int(state["update"])
        pending = state["pending_batches"]
        if resume_state is not None:
            scheduler.load_state_dict(state["sched"])
        elif branch_state is not None:
            expected = branch_position(cfg.target_processed_positions, cfg.cooldown_fraction)
            if processed_positions != expected:
                raise ValueError(f"cooldown needs branch at {expected}, got {processed_positions}")
            scheduler.step(processed_positions)
        else:
            scheduler.step(processed_positions)

    def trunk_fn(features, pad, actions):
        return model(features, pad, actions)

    temporal_fn: Callable = model.temporal.teacher_forced_nll
    if DEVICE == "cuda" and cfg.compile_trunk and not throughput_probe_eager:
        trunk_fn = torch.compile(trunk_fn, dynamic=False)
    if DEVICE == "cuda" and cfg.compile_temporal and not throughput_probe_eager:
        temporal_fn = torch.compile(temporal_fn, dynamic=False)

    train_loader, val_cache = _make_loaders(cfg, stats, audit, loader_workers=loader_workers)
    if state is not None:
        train_loader.load_state_dict(state["data_state"])
    update_source = OrderedUpdateSource(train_loader, cfg)
    if state is not None:
        _restore_rng(state)
    copy_stream = torch.cuda.Stream() if DEVICE == "cuda" else None
    stop_position = _training_stop(cfg)
    branch_targets = {branch_position(target, cfg.cooldown_fraction): target for target in _prefix_branch_targets(cfg)}
    branches = set(branch_targets) if cfg.phase == "prefix" else set()
    completed_branches = {position for position in branches if position <= processed_positions}
    run_started = time.monotonic()
    model.train()
    prefetch_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="optimizer-update")
    prefetched: deque[Future[list[LoadedUpdate]]] = deque()
    probe_rows: list[dict[str, float]] = []
    probe_total = 0 if not probe else throughput_probe_warmup + throughput_probe_updates
    probe_completed = 0
    probe_start_positions = processed_positions

    def drain_prefetch() -> None:
        while prefetched:
            for loaded in prefetched.popleft().result():
                pending.extend(loaded.work)

    def save(path: Path) -> None:
        save_checkpoint(
            path,
            step=update,
            model=model,
            opt=optimizer,
            sched=scheduler,
            cfg=asdict(cfg),
            wandb_id=None if wandb.run is None else wandb.run.id,
            uploader=uploader,
            extra_state=_checkpoint_extra(
                train_loader,
                pending=pending,
                processed_positions=processed_positions,
                training_wall_seconds=prior_training_wall_seconds + time.monotonic() - run_started,
                update=update,
            ),
        )

    try:
        while processed_positions < stop_position and (not probe or probe_completed < probe_total):
            update_started = time.monotonic()
            if DEVICE == "cuda":
                torch.cuda.reset_peak_memory_stats()
            available = pending
            pending = []
            loader_service = 0.0
            uncovered_started = time.monotonic()
            if len(available) < cfg.grad_accum_steps:
                if prefetched:
                    loaded = prefetched.popleft().result()[0]
                else:
                    loaded = update_source.load(1)[0]
                loader_service = loaded.service_seconds
                available.extend(loaded.work)
            loader_wait = time.monotonic() - uncovered_started
            work = available[: cfg.grad_accum_steps]
            pending.extend(available[cfg.grad_accum_steps :])
            while len(work) < cfg.grad_accum_steps:
                loaded = update_source.load(1)[0]
                loader_service += loaded.service_seconds
                needed = cfg.grad_accum_steps - len(work)
                work.extend(loaded.work[:needed])
                pending.extend(loaded.work[needed:])

            upcoming = [stop_position]
            if cfg.phase == "prefix" and not probe:
                upcoming.extend(
                    position for position in branches - completed_branches if position > processed_positions
                )
            boundary = min(upcoming)
            selected, boundary_pending, valid_prefixes = _select_position_work(work, boundary - processed_positions)
            pending = boundary_pending + pending
            if valid_prefixes <= 0:
                raise RuntimeError("training update contains no loss-bearing positions")
            next_processed = processed_positions + valid_prefixes
            scheduler.step(next_processed)
            next_update = update + 1
            branch_due = not probe and cfg.phase == "prefix" and next_processed in branches - completed_branches
            val_due = not probe and cfg.val_every > 0 and next_update % cfg.val_every == 0
            ckpt_due = not probe and cfg.ckpt_every > 0 and next_update % cfg.ckpt_every == 0
            final_due = next_processed >= stop_position or (probe and probe_completed + 1 >= probe_total)
            boundary_due = branch_due or val_due or ckpt_due or final_due
            # An exact-D checkpoint can leave one partial batch pending. It must
            # not disable lookahead for every later update.
            if not boundary_due:
                while len(prefetched) < loader_prefetch_updates:
                    prefetched.append(prefetch_pool.submit(update_source.load, 1))
            loader_cache_size = len(prefetched)
            optimizer.zero_grad()
            nll_sum = torch.zeros(len(cfg.head_offsets), N_GROUPS, device=DEVICE)
            n_prefixes = 0
            cpu_batches = [batch for batch, _ in selected]
            cpu_masks = [mask for _, mask in selected]
            with profile("step") as stopwatch:
                for batch, position_mask in zip(
                    device_batches(cpu_batches, DEVICE, copy_stream), cpu_masks, strict=True
                ):
                    history, targets, valid = prepared_targets(model, batch)
                    selected_valid = valid & position_mask.to(valid.device)
                    with amp_context(cfg, DEVICE):
                        hidden = trunk_fn(batch.context.features, batch.context.ctx_pad, history)
                        dense_nll = temporal_fn(hidden, history, targets)
                        parts = ActionLoss(nll=dense_nll[selected_valid], targets=targets[selected_valid])
                        loss = objective(parts, cfg.aux_loss_weight) * (parts.nll.shape[0] / valid_prefixes)
                    loss.backward()
                    nll_sum += parts.nll.detach().sum(dim=0)
                    n_prefixes += parts.nll.shape[0]
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError(f"update {update}: non-finite gradient norm {gradient_norm}")
                optimizer.step()
                if DEVICE == "cuda":
                    torch.cuda.synchronize()
            processed_positions = next_processed
            update += 1
            if boundary_due:
                boundary_wait_started = time.monotonic()
                drain_prefetch()
                loader_wait += time.monotonic() - boundary_wait_started
            update_wall = time.monotonic() - update_started
            if probe and probe_completed >= throughput_probe_warmup:
                probe_rows.append(
                    {
                        "loader_service_s": loader_service,
                        "uncovered_wait_s": loader_wait,
                        "update_wall_s": update_wall,
                        "loss_positions": float(valid_prefixes),
                    }
                )
            probe_completed += 1
            metrics = nll_mean_metrics((nll_sum / n_prefixes).cpu(), cfg.head_offsets)
            log = {
                "global_step": update,
                "processed_positions": processed_positions,
                "samples": update * cfg.batch_size,
                "data/D_over_U_positions": processed_positions / max(audit.unique_loss_positions, 1),
                "scaling/D_over_N": processed_positions / counts["total"],
                "optimizer/adam_tau": achieved_adam_tau(cfg, processed_positions),
                "optimizer/adam_weight_decay": cfg.adam_weight_decay,
                **{f"train/{name}": value for name, value in metrics.items()},
                "train/grad_norm": float(gradient_norm),
                "throughput/step_s": stopwatch.elapsed,
                "throughput/loader_service_s": loader_service,
                "throughput/loader_uncovered_wait_s": loader_wait,
                "throughput/loader_wait_fraction": loader_wait / max(update_wall, 1e-12),
                "throughput/update_wall_s": update_wall,
                "throughput/loader_workers": loader_workers,
                "throughput/loader_cache_updates": loader_cache_size,
                "throughput/loader_cache_limit_gb": cfg.cache_limit_gb,
                "throughput/loss_positions_per_s": valid_prefixes / stopwatch.elapsed,
                "lr/muon": next(group["lr"] for group in optimizer.param_groups if group["use_muon"]),
                "lr/adam": next(group["lr"] for group in optimizer.param_groups if not group["use_muon"]),
            }
            if DEVICE == "cuda":
                log["hardware/peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 2**30
                log["hardware/peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30
            wandb.log(log)
            if update < 10 or update % 50 == 0:
                print(
                    f"[t+{time.monotonic() - run_started:.0f}s] update {update}: "
                    f"D={processed_positions:,}, {metrics['loss']:.3f} bits objective, "
                    f"{valid_prefixes / stopwatch.elapsed:,.0f} loss positions/s",
                    flush=True,
                )

            if not probe and cfg.phase == "prefix" and processed_positions in branches - completed_branches:
                target = branch_targets[processed_positions]
                save(run_dir / branch_checkpoint_name(target))
                completed_branches.add(processed_positions)
            if val_due or ckpt_due:
                save(run_dir / "latest.pt")
            if val_due:
                values = val_metrics(model, val_cache, cfg)
                wandb.log(
                    {
                        "global_step": update,
                        "processed_positions": processed_positions,
                        **{f"val/{name}": value for name, value in values.items()},
                    }
                )
                model.train()

        drain_prefetch()
        if probe:
            waits = np.array([row["uncovered_wait_s"] for row in probe_rows])
            walls = np.array([row["update_wall_s"] for row in probe_rows])
            positions = np.array([row["loss_positions"] for row in probe_rows])
            services = np.array([row["loader_service_s"] for row in probe_rows])
            measured_seconds = float(walls.sum())
            rate = float(positions.sum() / measured_seconds)
            p95_wait = float(np.quantile(waits, 0.95))
            p95_wait_fraction = float(np.quantile(waits / walls, 0.95))
            mean_wall = float(walls.mean())
            forecast_positions = max(cfg.target_processed_positions - probe_start_positions, 0)
            forecast_hours = forecast_positions / rate / 3600.0
            payload: dict[str, object] = {
                "schema_version": 1,
                "model": model_tag(cfg),
                "loader_workers": loader_workers,
                "loader_prefetch_updates": loader_prefetch_updates,
                "loader_cache_limit_gb": cfg.cache_limit_gb,
                "warmup_updates": throughput_probe_warmup,
                "measured_updates": throughput_probe_updates,
                "eager_training": throughput_probe_eager,
                "training_attention_path": "dense_sdpa" if throughput_probe_eager else "configured",
                "mean_loader_service_s": float(services.mean()),
                "std_loader_service_s": float(services.std()),
                "p95_uncovered_wait_s": p95_wait,
                "mean_update_wall_s": mean_wall,
                "std_update_wall_s": float(walls.std()),
                "uncovered_wait_fraction": float(waits.sum() / measured_seconds),
                "p95_uncovered_wait_fraction": p95_wait_fraction,
                "loss_positions_per_s": rate,
                "forecast_loss_positions": forecast_positions,
                "remaining_path_forecast_hours": forecast_hours,
                "loader_gate_pass": p95_wait <= 0.10 and p95_wait_fraction <= 0.10,
                "forecast_gate_pass": forecast_hours < 20.0,
                "cuda_device": torch.cuda.get_device_name() if DEVICE == "cuda" else "cpu",
            }
            probe_path = run_dir / "loader_probe.json"
            temporary = probe_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
            temporary.replace(probe_path)
            print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
            return payload
        save(run_dir / "latest.pt")
        final_path = run_dir / "final.pt"
        save(final_path)
        checkpoint_sha = _checkpoint_sha256(final_path)
        completed_eval_delays = (
            _load_eval_progress(run_dir, run_name, cfg, audit, checkpoint_sha) if cfg.phase == "cooldown" else set()
        )
        final_val = val_metrics(model, val_cache, cfg)
        wandb.log(
            {
                "global_step": update,
                "processed_positions": processed_positions,
                **{f"val/{name}": value for name, value in final_val.items()},
            }
        )
        if wandb.run is not None:
            incremental_wall_seconds = time.monotonic() - run_started
            cumulative_wall_seconds = prior_training_wall_seconds + incremental_wall_seconds
            wandb.run.summary["training/processed_positions"] = processed_positions
            wandb.run.summary["training/D_over_U_positions"] = processed_positions / max(
                audit.unique_loss_positions, 1
            )
            wandb.run.summary["training/D_over_N"] = processed_positions / counts["total"]
            wandb.run.summary["training/adam_tau"] = achieved_adam_tau(cfg, processed_positions)
            # Approximate one forward+backward parameter use for the trunk and
            # one per dense temporal token for decoder parameters.
            training_flops = 6 * processed_positions * (counts["trunk"] + cfg.sample_chunk_length * counts["decoder"])
            wandb.run.summary["training/flops"] = training_flops
            wandb.run.summary["training/flops_formula"] = "6*D*(N_trunk+36*N_decoder)"
            wandb.run.summary["training/wall_seconds"] = cumulative_wall_seconds
            wandb.run.summary["training/cumulative_wall_seconds"] = cumulative_wall_seconds
            wandb.run.summary["training/incremental_run_wall_seconds"] = incremental_wall_seconds
            wandb.run.summary["training/prefix_wall_seconds"] = prior_training_wall_seconds
            wandb.run.summary["val/nll"] = final_val["loss"]
        if cfg.phase == "cooldown":
            inference = BF16Inference(model, cfg)
            for delay in cfg.evaluation_delays:
                if delay in completed_eval_delays:
                    print(f"[eval] d={delay} already completed; preserving the once-per-endpoint result", flush=True)
                    continue
                values = eval_vs_cpu(
                    model,
                    stats,
                    cfg,
                    n_matchups=cfg.final_eval_n_matchups,
                    replay_dir=replay_dir / f"final_d{delay}",
                    delay=delay,
                    replan_interval=delay,
                    checkpoint_sha256=checkpoint_sha,
                    inference=inference,
                )
                wandb.log(
                    {
                        "global_step": update,
                        "processed_positions": processed_positions,
                        **{f"eval_d{delay}/{name}": value for name, value in values.items()},
                    }
                )
                require_complete_eval(values, cfg.final_eval_n_matchups)
                completed_eval_delays.add(delay)
                _write_eval_progress(
                    run_dir,
                    cfg,
                    audit,
                    checkpoint_sha,
                    completed_eval_delays,
                    uploader,
                )
                if uploader is not None:
                    uploader.upload_tree(replay_dir / f"final_d{delay}", base=run_dir)
                if wandb.run is not None:
                    for name, value in values.items():
                        wandb.run.summary[f"eval_d{delay}/{name}"] = value
    finally:
        prefetch_pool.shutdown(wait=True, cancel_futures=True)
        if uploader is not None:
            uploader.upload_tree(replay_dir, base=run_dir)
            uploader.close()
        wandb.finish()


_CHECKPOINT_ARCH_FIELDS = {
    "model_family",
    "d_model",
    "n_layers",
    "n_heads",
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
        raise ValueError(f"checkpoint is not an experiment-039 architecture; missing {sorted(missing)}")
    known = {item.name for item in fields(TrainConfig)}
    return TrainConfig(**{name: value for name, value in values.items() if name in known})


def load_checkpoint(path: str, *, device: str = DEVICE) -> tuple[GPT, TrainConfig, dict[str, FeatureStats], dict]:
    # Checkpoints also contain CPU dataloader/RNG state and optimizer tensors.
    # Loading the whole archive onto CUDA would both corrupt those CPU-only
    # states and unnecessarily duplicate optimizer memory during evaluation.
    state = torch.load(path, map_location="cpu", weights_only=False)
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
    delay: int | None = None,
    replan_interval: int | None = None,
    n_matchups: int | None = None,
    eager: bool = False,
    max_parallel: int | None = None,
    output_name: str | None = None,
) -> dict[str, float]:
    model, cfg, stats, state = load_checkpoint(path)
    selected_delay = cfg.control_delay if delay is None else delay
    selected_interval = selected_delay if replan_interval is None else replan_interval
    cfg = replace(
        cfg,
        inference_mode="eager" if eager else cfg.inference_mode,
        eval_max_parallel=cfg.eval_max_parallel if max_parallel is None else max_parallel,
        evaluation_delays=(selected_delay,),
    )
    validate_config(cfg)
    default_name = f"eval_replays_d{selected_delay}_r{selected_interval}"
    if output_name is not None and (Path(output_name).name != output_name or output_name in ("", ".", "..")):
        raise ValueError(f"evaluation output name must be one directory name, got {output_name!r}")
    replay_dir = Path(path).resolve().parent / (default_name if output_name is None else output_name)
    expected = cfg.final_eval_n_matchups if n_matchups is None else n_matchups
    values = eval_vs_cpu(
        model,
        stats,
        cfg,
        n_matchups=expected,
        replay_dir=replay_dir,
        delay=selected_delay,
        replan_interval=selected_interval,
        checkpoint_sha256=_checkpoint_sha256(Path(path)),
        require_compiled_cuda=not eager,
    )
    print(
        f"[eval] update={state['step']} d={selected_delay} R={selected_interval}: {values}",
        flush=True,
    )
    require_complete_eval(values, expected)
    return values


def _benchmark_post(t: int, side: int) -> dict:
    phase = 0.37 * t + 1.1 * side
    return {
        "position": {"x": 110.0 * np.cos(phase), "y": 40.0 * np.sin(0.21 * t) - 12.0 * side},
        "direction": -1.0 if (t + side) % 3 == 0 else 1.0,
        "percent": float((3 * t + 7 * side) % 180),
        "shield": 60.0 - (t % 41),
        "stock": 4 - (t // 137),
        "action": 14 + (t % 23),
        "jumps_used": t % 3,
        "airborne": (t + side) % 2,
        "hurtbox_state": t % 3,
        "hitlag_left": float(t % 5),
    }


def _benchmark_observation(t: int) -> dict:
    stage = int(melee.Stage.FINAL_DESTINATION.value)
    return {
        "id": 400 + t,
        "ports": {
            1: {"leader": {"post": _benchmark_post(t, 0)}, "follower": None},
            2: {"leader": {"post": _benchmark_post(t, 1)}, "follower": None},
        },
        "items": [],
        "stage": stage,
        "_matchup": {"stage": stage, "character": {1: 14, 2: 22}},
    }


def run_benchmark(
    cfg: TrainConfig,
    stats: dict[str, FeatureStats],
    *,
    iterations: int,
    output_dir: Path,
) -> dict[str, object]:
    """Measure compiled batch-8 latency through workers, IPC, and policy output."""
    validate_config(cfg)
    device = torch.device(DEVICE)
    if device.type != "cuda":
        raise RuntimeError("official deployment-path latency benchmarking requires CUDA")
    model = GPT(cfg).to(device).eval()
    counts = subsystem_parameter_counts(model)
    engine = BF16Inference(
        model,
        cfg,
        compiled_buckets=(8,),
        compile_mode="reduce-overhead",
    )
    slots = tuple(Slot(index, 1) for index in range(8))
    latency: dict[str, dict[str, float]] = {}
    label = "026base" if cfg.model_family == "026-baseline" else f"L{cfg.n_layers}"
    for delay in DELAY_BUCKETS:
        # Compile the exact batch/horizon program before timing the production
        # worker path. The warmup is not a latency sample.
        warm_telemetry = DelayTelemetry()
        warm_policy = make_policy(
            model,
            stats,
            cfg,
            delay=delay,
            replan_interval=delay,
            decode_seed=cfg.seed,
            inference=engine,
            telemetry=warm_telemetry,
        )
        frame = 0
        while len(warm_telemetry.latency_ms) < 3:
            observation = _benchmark_observation(frame)
            warm_policy(frame, {slot: observation for slot in slots})
            frame += 1
        torch.cuda.synchronize()

        policy_telemetry = DelayTelemetry()
        process_telemetry = ProcessVecTelemetry()
        policy_index = itertools.count()

        def factory(
            selected_delay: int = delay,
            selected_telemetry: DelayTelemetry = policy_telemetry,
            indices: Iterator[int] = policy_index,
        ) -> AbsoluteDelayPolicy:
            return make_policy(
                model,
                stats,
                cfg,
                delay=selected_delay,
                replan_interval=selected_delay,
                decode_seed=cfg.seed + 1 + next(indices),
                inference=engine,
                telemetry=selected_telemetry,
            )

        wall_started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix=f"039-latency-{label}-d{delay}-") as replay_directory:
            sweep_vs_cpu_prior_with_rows(
                factory,
                session_cfg=default_session_cfg(Path(replay_directory), instant_match_restart=True),
                n_matchups=8,
                max_parallel=8,
                max_frames=delay * (iterations + 3),
                cpu_level=9,
                ego_port=1,
                seed_stage=PRIOR_SWEEP_SEED_STAGE,
                start_retries=DEFAULT_START_RETRIES,
                process_telemetry=process_telemetry,
            )
        wall_seconds = time.perf_counter() - wall_started
        if len(process_telemetry.inference_latency_ms) < iterations:
            raise RuntimeError(
                f"production benchmark produced {len(process_telemetry.inference_latency_ms)} "
                f"inference calls at d={delay}, expected at least {iterations}"
            )
        samples = np.asarray(process_telemetry.inference_latency_ms[-iterations:], dtype=np.float64)
        row_samples = np.asarray(process_telemetry.inference_rows_by_call[-iterations:], dtype=np.int64)
        batch_samples = np.asarray(process_telemetry.inference_batch_rows_by_call[-iterations:], dtype=np.int64)
        budget = delay * FRAME_TIME_MS
        misses = int(row_samples[samples > budget].sum())
        rows = int(batch_samples.sum())
        batch_eight = (
            process_telemetry.failed_workers == 0
            and process_telemetry.timed_out_workers == 0
            and bool(len(batch_samples))
            and bool((batch_samples == 8).all())
        )
        values = {
            "latency_p50_ms": float(np.percentile(samples, 50)),
            "latency_p95_ms": float(np.percentile(samples, 95)),
            "sustained_inference_rows_per_s": rows / max(float(samples.sum()) / 1_000, 1e-12),
            "deadline_misses": float(misses),
            "deadline_miss_fraction": misses / max(int(row_samples.sum()), 1),
            "frame_budget_ms": budget,
            "wall_rows_per_s": rows / wall_seconds,
            "batch_eight_reliable": float(batch_eight),
            "valid_bucket": float(batch_eight and misses == 0 and np.percentile(samples, 95) <= budget),
            **process_telemetry.metrics(),
        }
        latency[str(delay)] = values
        print(f"[latency] {label} d={delay}: {values}", flush=True)

    native = next((delay for delay in DELAY_BUCKETS if latency[str(delay)]["valid_bucket"]), None)
    payload: dict[str, object] = {
        "schema_version": LATENCY_ARTIFACT_SCHEMA,
        "latency_start_boundary": LATENCY_START_BOUNDARY,
        "latency_end_boundary": LATENCY_END_BOUNDARY,
        "model_family": cfg.model_family,
        "L": cfg.n_layers,
        "d_model": cfg.d_model,
        "batch_size": 8,
        "frame_time_ms": FRAME_TIME_MS,
        "compile_mode": "reduce-overhead",
        "cuda_device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda or "none",
        "trunk_parameters": counts["trunk"],
        "decoder_parameters": counts["decoder"],
        "total_parameters": counts["total"],
        "native_delay": native,
        "latency": latency,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{label}.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(output)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return payload


def parameter_counts_for_config(cfg: TrainConfig) -> dict[str, int]:
    with torch.device("meta"):
        model = GPT(cfg)
    return subsystem_parameter_counts(model)


@dataclass
class Args:
    cfg: TrainConfig = dataclass_field(default_factory=TrainConfig)
    model_l: int = 5
    baseline_026: bool = False
    phase: str = "prefix"
    target_d_exp: int = 30
    target_positions: int | None = None
    unique_data_divisor: int = 1
    resume: str | None = None
    resume_checkpoint: str = "latest.pt"
    prefix_fork_from_run: str | None = None
    prefix_fork_checkpoint: str | None = None
    branch_from_run: str | None = None
    eval: str | None = None
    eval_delay: int | None = None
    eval_replan_interval: int | None = None
    eval_n_matchups: int | None = None
    eval_eager: bool = False
    eval_max_parallel: int | None = None
    eval_output_name: str | None = None
    benchmark: bool = False
    benchmark_iterations: int = 100
    benchmark_output_dir: str = "results/039_capacity_latency"
    training_smoke: bool = False
    training_smoke_output_dir: str = "results/039_training_smoke"
    audit_only: bool = False
    loader_workers: int = 8
    loader_prefetch_updates: int = 1
    throughput_probe_from_run: str | None = None
    throughput_probe_checkpoint: str = "latest.pt"
    throughput_probe_warmup: int = 32
    throughput_probe_updates: int = 256
    throughput_probe_eager: bool = False


def requested_config(args: Args) -> TrainConfig:
    target = 2**args.target_d_exp if args.target_positions is None else args.target_positions
    base = replace(
        args.cfg,
        phase=args.phase,
        target_processed_positions=target,
        unique_data_divisor=args.unique_data_divisor,
    )
    cfg = baseline_026_config(base) if args.baseline_026 else scaled_config(args.model_l, base)
    return replace(cfg, adam_weight_decay_endpoint=_adam_weight_decay_endpoint(cfg))


def main(args: Args) -> None:
    if args.loader_workers < 0:
        raise SystemExit("--loader-workers must be non-negative")
    if args.loader_prefetch_updates not in (1, 2):
        raise SystemExit("--loader-prefetch-updates must be 1 or 2")
    if args.throughput_probe_warmup < 0 or args.throughput_probe_updates < 1:
        raise SystemExit("throughput probe warmup must be non-negative and measured updates must be positive")
    if Path(args.throughput_probe_checkpoint).name != args.throughput_probe_checkpoint or not (
        args.throughput_probe_checkpoint.endswith(".pt")
    ):
        raise SystemExit("--throughput-probe-checkpoint must be one checkpoint filename ending in .pt")
    probe_mode = args.throughput_probe_from_run is not None
    has_prefix_fork = args.prefix_fork_from_run is not None or args.prefix_fork_checkpoint is not None
    if (args.prefix_fork_from_run is None) != (args.prefix_fork_checkpoint is None):
        raise SystemExit("--prefix-fork-from-run and --prefix-fork-checkpoint must be provided together")
    if has_prefix_fork and (
        args.resume is not None
        or args.branch_from_run is not None
        or args.eval is not None
        or args.benchmark
        or args.training_smoke
        or args.audit_only
        or probe_mode
    ):
        raise SystemExit("an exact prefix fork cannot be combined with another execution mode")
    if args.target_positions is not None and (args.resume is not None or args.eval is not None):
        raise SystemExit("--target-positions is only valid for fresh training, probes, smoke, or audit")
    if args.resume is None and args.resume_checkpoint != "latest.pt":
        raise SystemExit("--resume-checkpoint requires --resume")

    if args.eval is not None:
        if (
            args.benchmark
            or args.training_smoke
            or args.resume is not None
            or has_prefix_fork
            or args.audit_only
            or probe_mode
        ):
            raise SystemExit("--eval cannot be combined with benchmark, training smoke, resume, or audit")
        eval_checkpoint(
            args.eval,
            delay=args.eval_delay,
            replan_interval=args.eval_replan_interval,
            n_matchups=args.eval_n_matchups,
            eager=args.eval_eager,
            max_parallel=args.eval_max_parallel,
            output_name=args.eval_output_name,
        )
        return

    if probe_mode:
        if (
            args.resume is not None
            or has_prefix_fork
            or args.branch_from_run is not None
            or args.benchmark
            or args.training_smoke
            or args.audit_only
        ):
            raise SystemExit("--throughput-probe-from-run cannot be combined with another execution mode")
        probe_run = args.throughput_probe_from_run
        if probe_run is None:
            raise AssertionError("probe mode lost its source run")
        if args.target_positions is None:
            raise SystemExit("--throughput-probe-from-run requires the addition's --target-positions")
        probe_state = load_for_resume(
            probe_run,
            Path("runs") / probe_run,
            device="cpu",
            name=args.throughput_probe_checkpoint,
        )
        if probe_state is None:
            raise SystemExit(f"no {args.throughput_probe_checkpoint} for run {probe_run!r}")
        if args.phase != "prefix":
            raise SystemExit("an endpoint throughput probe requires --phase prefix")
        cfg = requested_config(args)
        _validate_prefix_fork_state(probe_state, cfg, parameter_counts_for_config(cfg)["total"])
        validate_config(cfg)
        stats = load_consolidated_stats(Path(cfg.data_root) / "stats.json")
        audit = dataset_audit(cfg)
        train(
            cfg,
            stats,
            audit,
            requested_run_name=(f"{probe_run}-loader-probe-w{args.loader_workers}-p{args.loader_prefetch_updates}"),
            prefix_fork_state=probe_state,
            loader_workers=args.loader_workers,
            loader_prefetch_updates=args.loader_prefetch_updates,
            throughput_probe_warmup=args.throughput_probe_warmup,
            throughput_probe_updates=args.throughput_probe_updates,
            throughput_probe_eager=args.throughput_probe_eager,
        )
        return

    if args.resume is not None:
        if (
            args.benchmark
            or args.training_smoke
            or args.audit_only
            or has_prefix_fork
            or args.branch_from_run is not None
        ):
            raise SystemExit("--resume cannot be combined with benchmark, training smoke, audit, or a fresh branch")
        if Path(args.resume_checkpoint).name != args.resume_checkpoint or not args.resume_checkpoint.endswith(".pt"):
            raise SystemExit("--resume-checkpoint must be one checkpoint filename ending in .pt")
        resume_state = load_for_resume(
            args.resume,
            Path("runs") / args.resume,
            device="cpu",
            name=args.resume_checkpoint,
        )
        if resume_state is None:
            raise SystemExit(f"no {args.resume_checkpoint} for run {args.resume!r}")
        cfg = config_from_state(resume_state["cfg"])
        requested_run_name = args.resume
        branch_state = None
        prefix_fork_state = None
    else:
        resume_state = None
        cfg = requested_config(args)
        validate_config(cfg)
        counts = parameter_counts_for_config(cfg)
        requested_run_name = run_name_for(cfg, counts["total"])
        branch_state = None
        prefix_fork_state = None
        if has_prefix_fork:
            if cfg.phase != "prefix" or args.target_positions is None:
                raise SystemExit("an exact prefix fork requires --phase prefix and --target-positions")
            prefix_source = args.prefix_fork_from_run
            prefix_checkpoint = args.prefix_fork_checkpoint
            if prefix_source is None or prefix_checkpoint is None:
                raise AssertionError("prefix-fork arguments lost after validation")
            if Path(prefix_source).name != prefix_source:
                raise SystemExit("--prefix-fork-from-run must be one run name")
            if Path(prefix_checkpoint).name != prefix_checkpoint or not prefix_checkpoint.endswith(".pt"):
                raise SystemExit("--prefix-fork-checkpoint must be one checkpoint filename ending in .pt")
            prefix_fork_state = load_for_resume(
                prefix_source,
                Path("runs") / prefix_source,
                device="cpu",
                name=prefix_checkpoint,
            )
            if prefix_fork_state is None:
                raise SystemExit(f"no {prefix_checkpoint} for prefix source {prefix_source!r}")
            _validate_prefix_fork_state(prefix_fork_state, cfg, counts["total"])
        elif cfg.phase == "cooldown":
            prefix_target = (
                2**30 if cfg.target_processed_positions in _standard_endpoints() else cfg.target_processed_positions
            )
            prefix_cfg = replace(cfg, phase="prefix", target_processed_positions=prefix_target)
            prefix_name = args.branch_from_run or run_name_for(prefix_cfg, counts["total"])
            checkpoint_name = branch_checkpoint_name(cfg.target_processed_positions)
            branch_state = load_for_resume(
                prefix_name,
                Path("runs") / prefix_name,
                device="cpu",
                name=checkpoint_name,
            )
            if branch_state is None:
                raise SystemExit(f"no {checkpoint_name} for shared-prefix run {prefix_name!r}")

    validate_config(cfg)
    stats = load_consolidated_stats(Path(cfg.data_root) / "stats.json")
    if args.benchmark:
        if args.training_smoke or args.resume is not None or cfg.phase != "prefix":
            raise SystemExit("--benchmark requires a fresh prefix configuration")
        run_benchmark(
            cfg,
            stats,
            iterations=args.benchmark_iterations,
            output_dir=Path(args.benchmark_output_dir),
        )
        return

    audit = dataset_audit(cfg)
    counts = parameter_counts_for_config(cfg)
    scale = adam_scale(cfg, counts["total"])
    if args.training_smoke:
        if cfg.phase != "prefix" or args.audit_only:
            raise SystemExit("--training-smoke requires a fresh prefix configuration")
        run_training_smoke(
            cfg,
            stats,
            audit,
            output_dir=Path(args.training_smoke_output_dir),
        )
        return
    if args.audit_only:
        print(
            json.dumps(
                {
                    "run_name": requested_run_name,
                    "config": asdict(cfg),
                    "parameters": counts,
                    "unique_replays": audit.unique_replays,
                    "episode_hash": audit.episode_hash,
                    "unique_loss_positions": audit.unique_loss_positions,
                    "adam_scale": asdict(scale),
                    "branch_positions": _prefix_branch_positions(cfg),
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return
    train(
        cfg,
        stats,
        audit,
        requested_run_name=requested_run_name,
        resume_state=resume_state,
        branch_state=branch_state,
        prefix_fork_state=prefix_fork_state,
        loader_workers=args.loader_workers,
        loader_prefetch_updates=args.loader_prefetch_updates,
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
