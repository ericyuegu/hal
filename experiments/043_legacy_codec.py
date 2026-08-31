"""Experiment 026 with the historical controller codec.

The trainable architecture and optimization match the ranked-anonymous-1 O26
baseline. Codec version 2 restores the old controller classes and early-release
reducer. The default treatment evaluates one frame at a time and gives offset 1
half of the action objective; both settings can be reverted for the forensic
ladder documented in ``docs/experiments/043_legacy_codec.md``.

Architecture arms B-D separately test zero-initialized FiLM, linear action
heads, and removal of the trunk-logit skip. Final evaluation runs at execution
horizons 1 and 4.

Run:
    uv run experiments/043_legacy_codec.py
    uv run experiments/043_legacy_codec.py --eval runs/<run>/final.pt
    uv run experiments/043_legacy_codec.py --eval final.pt --eval-run <run> --eval-backfill-wandb
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
from hal.eval.self_play import DecodeTelemetry
from hal.eval.self_play import benchmark_checkpoint as benchmark_self_play
from hal.eval.self_play import canonical_context
from hal.eval.self_play import synthetic_context
from hal.sim.rollout import covering_power_of_two
from hal.training import scoring
from hal.training.checkpoints import BackgroundUploader
from hal.training.checkpoints import download_latest
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
from hal.training.mfu import bf16_dense_peak_flops
from hal.training.mfu import bf16_peak_source
from hal.training.mfu import model_flops_utilization
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

# Historical class counts. The O26-sized tables and output layers remain in
# place so this treatment changes the codec without changing a trainable tensor.
LEGACY_GROUP_VOCABS: tuple[int, ...] = (6, 37, 9, 5)

# Signed equivalents of ae29e3f's [0, 1] controller centers, in historical
# class order. Quantization below recreates the old float32-data/float64-center
# arithmetic; decoding emits today's signed action wire.
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
_LIVE_HORIZONS = (1, 4, 6)

AblationArm = Literal["A", "B", "C", "D"]
_ABLATION_DESCRIPTIONS: dict[AblationArm, str] = {
    "A": "O43 baseline",
    "B": "zero-initialized FiLM projections",
    "C": "O42 normalized linear action heads",
    "D": "trunk-logit skip removed",
}
_ABLATION_TAGS: dict[AblationArm, str] = {
    "A": "",
    "B": "-ablB-film0",
    "C": "-ablC-linear-head",
    "D": "-ablD-no-trunk-skip",
}

LEGACY_MAIN_CENTERS = torch.tensor(_LEGACY_MAIN_SIGNED, dtype=torch.float32)
LEGACY_C_CENTERS = torch.tensor(_LEGACY_C_SIGNED, dtype=torch.float32)
LEGACY_TRIGGER_CENTERS = torch.tensor(_LEGACY_TRIGGER_VALUES, dtype=torch.float32)

TRIGGER_L_CH = ACTION_CHANNELS.index("trigger_l")
TRIGGER_R_CH = ACTION_CHANNELS.index("trigger_r")
BUTTON_L_CH = ACTION_CHANNELS.index("button_l")
BUTTON_R_CH = ACTION_CHANNELS.index("button_r")
_BUTTON_L_SEMANTIC_CH = ACTION_CHANNELS[6:].index("button_l")

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
    codec_version: int = 2
    ablation_arm: AblationArm = "A"
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
    # None restores O26's dense-four-plus-auxiliary objective. A numeric value
    # is the share of the action objective assigned to offset 1.
    next_frame_loss_share: float | None = 0.5
    group_order: tuple[str, ...] = GROUP_ORDER

    action_vocab: int = 1024
    action_state_embed_dim: int = 48
    char_vocab: int = 32
    char_dim: int = 8
    stage_vocab: int = 32
    stage_dim: int = 4
    observation_bundle: str = "base"  # or v6_lean

    exec_horizon: int = 1
    final_diag_exec_horizon: int = 4
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
    muon_lr: float = 0.02
    adam_lr: float = 8.5e-4
    muon_weight_decay: float = 0.01
    adam_weight_decay: float = 0.01
    lr_floor_ratio: float = 1e-5 / 8.5e-4
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
    final_diag_n_matchups: int = 96
    # Match the reference's 32-wide waves. The spawned evaluator admits at most
    # eight cold Dolphin startups at once, as in O41, avoiding a CPU thundering herd.
    eval_max_parallel: int | None = 32

    data_root: str = "data/processed/ranked-anonymized-1/mds-policy-v7"
    train_replay_paths: str | None = None
    replay_format: Literal["policy", "policy-world"] = "policy"
    val_data_root: str = "data/processed/ranked-anonymized-1/mds-policy-v7"
    val_replay_format: Literal["policy", "policy-world"] = "policy"
    compact_data: bool = True
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
    system_metrics_every: int = 25
    system_metrics_interval_s: float = 5.0
    process_metrics_interval_s: float = 30.0
    cache_metrics_interval_s: float = 30.0


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
    if cfg.codec_version != 2:
        raise ValueError(f"unsupported codec_version={cfg.codec_version}")
    if cfg.ablation_arm not in _ABLATION_DESCRIPTIONS:
        raise ValueError(f"unknown ablation_arm={cfg.ablation_arm!r}")
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
        raise ValueError("the live one/four/six-frame decoders require a dense 1..6 prefix")
    if cfg.group_order != GROUP_ORDER:
        raise ValueError(f"group_order must be {GROUP_ORDER}, got {cfg.group_order}")
    if cfg.batch_size % cfg.grad_accum_steps:
        raise ValueError("batch_size must be divisible by grad_accum_steps")
    if cfg.exec_horizon not in _LIVE_HORIZONS or cfg.final_diag_exec_horizon not in _LIVE_HORIZONS:
        raise ValueError(f"execution horizons must be one of {_LIVE_HORIZONS}")
    if cfg.decode_temp != 1.0:
        raise ValueError("experiment 043 freezes sampling temperature at 1")
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
    if cfg.replay_format not in ("policy", "policy-world"):
        raise ValueError("replay_format must be 'policy' or 'policy-world'")
    if cfg.train_replay_paths is not None and not cfg.compact_data:
        raise ValueError("train_replay_paths requires compact_data")
    if cfg.val_replay_format not in ("policy", "policy-world"):
        raise ValueError("val_replay_format must be 'policy' or 'policy-world'")
    if cfg.final_diag_n_matchups < 0:
        raise ValueError("final_diag_n_matchups must be non-negative")
    if not math.isfinite(cfg.muon_weight_decay) or cfg.muon_weight_decay < 0:
        raise ValueError("muon_weight_decay must be finite and non-negative")
    if not math.isfinite(cfg.adam_weight_decay) or cfg.adam_weight_decay < 0:
        raise ValueError("adam_weight_decay must be finite and non-negative")
    if not math.isfinite(cfg.lr_floor_ratio) or not 0 <= cfg.lr_floor_ratio <= 1:
        raise ValueError("lr_floor_ratio must be finite and between zero and one")
    if not math.isfinite(cfg.aux_loss_weight) or cfg.aux_loss_weight < 0:
        raise ValueError("aux_loss_weight must be finite and non-negative")
    if cfg.next_frame_loss_share is not None and (
        not math.isfinite(cfg.next_frame_loss_share) or not 0 <= cfg.next_frame_loss_share <= 1
    ):
        raise ValueError("next_frame_loss_share must be None or finite and between zero and one")
    if cfg.amp_dtype not in ("bfloat16", "float32"):
        raise ValueError("amp_dtype must be bfloat16 or float32")
    if cfg.reservoir_capacity < 2 * micro_batch_size(cfg):
        raise ValueError("reservoir_capacity must be at least twice the micro-batch size")
    if cfg.system_metrics_every < 0:
        raise ValueError("system_metrics_every must be non-negative")
    telemetry_intervals = (
        cfg.system_metrics_interval_s,
        cfg.process_metrics_interval_s,
        cfg.cache_metrics_interval_s,
    )
    if any(not math.isfinite(value) or value <= 0 for value in telemetry_intervals):
        raise ValueError("system telemetry intervals must be finite and positive")


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
    scheduled = [(cfg.final_eval_n_matchups, cfg.exec_horizon)]
    if cfg.eval_every > 0:
        scheduled.append((cfg.eval_n_matchups, cfg.exec_horizon))
    if cfg.final_diag_n_matchups > 0:
        scheduled.append((cfg.final_diag_n_matchups, cfg.final_diag_exec_horizon))
    return tuple(sorted({(_eval_inference_bucket(cfg, n), horizon) for n, horizon in scheduled}))


def _planned_inference_buckets(cfg: TrainConfig) -> tuple[int, ...]:
    return tuple(sorted({bucket for bucket, _ in _planned_inference_programs(cfg)}))


def amp_context(cfg: TrainConfig, device: torch.device | str):
    if cfg.amp_dtype == "bfloat16" and torch.device(device).type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def decoder_rmsnorm(x: Tensor) -> Tensor:
    return F.rms_norm(x, (x.shape[-1],), eps=1e-6)


def action_rmsnorm(x: Tensor) -> Tensor:
    """Normalize an action-head input with O42's near-zero stability epsilon."""
    return F.rms_norm(x, (x.shape[-1],), eps=1e-5)


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
    time = torch.arange(length, device=actions.device).expand(flat.shape[0], -1)

    # The original routine compares each raw set with the preceding raw set.
    # A changed set emits its lowest-index new button. A pure release emits
    # None. An unchanged set copies the previous reduced output.
    previous = F.pad(flat[:, :-1], (0, 0, 1, 0))
    same = (flat == previous).all(-1)
    new = flat & ~previous
    selected = new.to(torch.int64).argmax(-1)
    none = torch.full_like(selected, LEGACY_GROUP_VOCABS[BUTTONS_G] - 1)
    base = torch.where(new.any(-1), selected, none)
    changed = ~same | ~flat.any(-1)
    anchor = torch.where(changed, time, -1).cummax(-1).values
    output = base.gather(1, anchor)
    return output.reshape(pressed.shape[:-1])


class StructuredControllerCodec(nn.Module):
    """Exact legacy semantics inside O26-sized trainable tables."""

    main_centers: Tensor
    c_centers: Tensor
    trigger_centers: Tensor
    button_valid_for_trigger: Tensor
    _main_quant_centers: Tensor
    _c_quant_centers: Tensor
    _trigger_quant_centers: Tensor

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
        main_centers = torch.zeros_like(scoring.STICK_CLUSTER_CENTERS_MAIN)
        main_centers[: len(LEGACY_MAIN_CENTERS)] = LEGACY_MAIN_CENTERS
        trigger_centers = torch.zeros_like(scoring.TRIGGER_CENTERS)
        trigger_centers[: len(LEGACY_TRIGGER_CENTERS)] = LEGACY_TRIGGER_CENTERS
        self.register_buffer("main_centers", main_centers)
        self.register_buffer("c_centers", LEGACY_C_CENTERS.clone())
        self.register_buffer("trigger_centers", trigger_centers)
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
        main = self.main_centers[: LEGACY_GROUP_VOCABS[MAIN_G]].to(dtype)[indices[..., MAIN_G]]
        c_stick = self.c_centers.to(dtype)[indices[..., C_G]]
        fused = self.trigger_centers[: LEGACY_GROUP_VOCABS[TRIG_G]].to(dtype)[indices[..., TRIG_G]]
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
            return self.main_centers[: LEGACY_GROUP_VOCABS[MAIN_G]].to(dtype)[indices]
        if name == "c_stick":
            return self.c_centers.to(dtype)[indices]
        if name == "triggers":
            fused = self.trigger_centers[: LEGACY_GROUP_VOCABS[TRIG_G]].to(dtype)[indices]
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

    def mask_logits(self, name: str, logits: Tensor) -> Tensor:
        valid = LEGACY_GROUP_VOCABS[GROUP_INDEX[name]]
        invalid = torch.full_like(logits[..., valid:], -torch.inf)
        return torch.cat((logits[..., :valid], invalid), dim=-1)

    def button_mask(self, trigger_indices: Tensor) -> Tensor:
        return ~self.button_valid_for_trigger[trigger_indices]


class NonlinearActionHead(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, vocab: int) -> None:
        super().__init__()
        self.up = nn.Linear(d_model, d_hidden, bias=False)
        self.down = nn.Linear(d_hidden, vocab)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.up(decoder_rmsnorm(x))))


class LinearActionHead(nn.Module):
    """O42's scale-controlled linear action readout."""

    def __init__(self, d_model: int, vocab: int) -> None:
        super().__init__()
        self.output = nn.Linear(d_model, vocab, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.output(action_rmsnorm(x))


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
        # Build the complete A model before applying an arm. This keeps every
        # retained parameter and the caller's RNG state identical to A.
        if cfg.ablation_arm == "B":
            for condition in self.group_condition.values():
                nn.init.zeros_(condition.weight)
                nn.init.zeros_(condition.bias)
        elif cfg.ablation_arm == "C":
            with torch.random.fork_rng(devices=[]):
                self.outputs = nn.ModuleDict(
                    {name: LinearActionHead(self.d_model, GROUP_VOCABS[GROUP_INDEX[name]]) for name in GROUP_NAMES}
                )
        elif cfg.ablation_arm == "D":
            self.trunk_outputs = None

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

    def _group_logits(self, name: str, features: Tensor, trunk: Tensor | None) -> Tensor:
        """Apply one action head and the optional O43 trunk-logit skip."""
        logits = self.outputs[name](features)
        if self.trunk_outputs is not None:
            if trunk is None:
                raise ValueError("trunk features are required when the trunk-logit skip is enabled")
            trunk_logits = self.trunk_outputs[name](trunk)
            if trunk_logits.ndim + 1 == logits.ndim:
                trunk_logits = trunk_logits.unsqueeze(-2)
            logits = logits + trunk_logits
        return self.codec.mask_logits(name, logits)

    def teacher_forced_logits_by_group(self, hidden: Tensor, observed: Tensor, targets: Tensor) -> dict[str, Tensor]:
        states = self.teacher_forced_states(hidden, observed, targets)
        embedded = self.codec.embed_groups(targets)
        trunk = None if self.trunk_outputs is None else decoder_rmsnorm(hidden)
        logits = {
            name: self._group_logits(name, self.group_features(states, name, embedded), trunk) for name in GROUP_NAMES
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
                name: self._group_logits(name, self.group_features(state, name, embedded), trunk)
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
        allowed = tuple(self.head_offsets[:horizon] for horizon in _LIVE_HORIZONS)
        if offsets not in allowed:
            raise ValueError(f"live decode may compute only dense prefixes {_LIVE_HORIZONS}")
        if uniforms is not None and uniforms.shape != (len(offsets), N_GROUPS, hidden.shape[0]):
            raise ValueError("uniform table must be [frames, groups, batch]")
        trunk = decoder_rmsnorm(hidden[:, -1])
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
                logits = self._group_logits(name, self.group_features(state, name, embedded), trunk)
                if name == "buttons":
                    logits = logits.masked_fill(self.codec.button_mask(picks["triggers"]), float("-inf"))
                group = GROUP_INDEX[name]
                uniform = None if uniforms is None else uniforms[depth, group]
                # Training keeps O26's full output tensors, but closed loop only
                # samples real legacy classes. Besides avoiding meaningless work,
                # this prevents Inductor from lowering a mostly-masked 256-wide
                # button CDF to an unsupported split scan on L40S.
                valid_logits = logits[..., : LEGACY_GROUP_VOCABS[group]]
                pick = sample_categorical(valid_logits, argmax=argmax, uniform=uniform, gen=gen)
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
                logits = self._group_logits(name, self.group_features(state, name, embedded), trunk)
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


def _joint_objective(
    joint_nll: Tensor,
    valid_prefixes: int,
    *,
    aux_loss_weight: float,
    next_frame_loss_share: float | None,
) -> Tensor:
    """Compute one micro-batch's contribution to the action objective."""
    if next_frame_loss_share is None:
        primary = joint_nll[:, :4].sum() / (valid_prefixes * 4)
        auxiliary = joint_nll[:, 4:].sum() / (valid_prefixes * (joint_nll.shape[1] - 4))
        return primary + aux_loss_weight * auxiliary

    next_frame = joint_nll[:, 0].sum() / valid_prefixes
    remaining = joint_nll[:, 1:].sum() / (valid_prefixes * (joint_nll.shape[1] - 1))
    # O26's primary + auxiliary objective has total scale 2 when the auxiliary
    # weight is 1. Keep that scale while changing the relative allocation.
    return 2 * (next_frame_loss_share * next_frame + (1 - next_frame_loss_share) * remaining)


def objective(
    parts: ActionLoss,
    aux_loss_weight: float = 1.0,
    next_frame_loss_share: float | None = 0.5,
) -> Tensor:
    """Return the configured mean joint controller NLL."""
    joint_nll = parts.nll.sum(dim=-1)
    return _joint_objective(
        joint_nll,
        joint_nll.shape[0],
        aux_loss_weight=aux_loss_weight,
        next_frame_loss_share=next_frame_loss_share,
    )


def nll_mean_metrics(
    mean_nll: Tensor,
    offsets: tuple[int, ...],
    *,
    aux_loss_weight: float = 1.0,
    next_frame_loss_share: float | None = 0.5,
) -> dict[str, float]:
    if mean_nll.shape != (len(offsets), N_GROUPS):
        raise ValueError(f"mean NLL has shape {tuple(mean_nll.shape)}")
    joint = mean_nll.sum(dim=-1) / _LN2
    if next_frame_loss_share is None:
        loss = joint[:4].mean() + aux_loss_weight * joint[4:].mean()
    else:
        loss = 2 * (next_frame_loss_share * joint[0] + (1 - next_frame_loss_share) * joint[1:].mean())
    out = {
        "loss": float(loss),
        "primary_nll": float(joint[:4].mean()),
        "auxiliary_nll": float(joint[4:].mean()),
        "next_frame_nll": float(joint[0]),
        "remaining_nll": float(joint[1:].mean()),
    }
    for depth, offset in enumerate(offsets):
        out[f"nll_o{offset:02d}"] = float(joint[depth])
        for group, name in enumerate(GROUP_NAMES):
            out[f"nll_o{offset:02d}_{name}"] = float(mean_nll[depth, group] / _LN2)
    return out


def nll_metrics(
    nll: Tensor,
    offsets: tuple[int, ...],
    *,
    aux_loss_weight: float = 1.0,
    next_frame_loss_share: float | None = 0.5,
) -> dict[str, float]:
    return nll_mean_metrics(
        nll.mean(dim=0),
        offsets,
        aux_loss_weight=aux_loss_weight,
        next_frame_loss_share=next_frame_loss_share,
    )


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
    out = nll_mean_metrics(
        nll_sum / count,
        model.head_offsets,
        aux_loss_weight=cfg.aux_loss_weight,
        next_frame_loss_share=cfg.next_frame_loss_share,
    )
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
        if horizon not in _LIVE_HORIZONS:
            raise ValueError(f"execution horizon must be one of {_LIVE_HORIZONS}")
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
    if horizon not in _LIVE_HORIZONS:
        raise ValueError(f"execution horizon must be one of {_LIVE_HORIZONS}")
    engine = BF16Inference(model, cfg) if inference is None else inference
    random_streams = None if decode_seed is None else SlotGroupRandom(decode_seed)
    generator = None if decode_seed is None else torch.Generator(device=device).manual_seed(decode_seed)

    @torch.no_grad()
    def predict(ctx: Context, committed: np.ndarray | None) -> np.ndarray:
        if committed is not None:
            raise ValueError("experiment 043 does not condition on a committed RTC prefix")
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
    suite: str
    fixed_ego_character: int | None
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


def matchup_diversity(
    n_matchups: int,
    fixed_ego_character: melee.Character | None = None,
) -> tuple[int, int, int, str]:
    matchups = [(fixed_ego_character or scheduled_ego, cpu) for scheduled_ego, cpu in matchups_for_vs_cpu(n_matchups)]
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
    fixed_ego_character: melee.Character | None = None,
) -> EvalProtocol:
    if fixed_ego_character is None:
        pairs, egos, cpus, schedule_sha = assert_protocol_diversity(n_matchups)
    else:
        pairs, egos, cpus, schedule_sha = matchup_diversity(n_matchups, fixed_ego_character)
    return EvalProtocol(
        suite="char_matchup" if fixed_ego_character is None else "fox",
        fixed_ego_character=None if fixed_ego_character is None else int(fixed_ego_character.value),
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
    fixed_ego_character: melee.Character | None = None,
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
        fixed_ego_character=fixed_ego_character,
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
            fixed_ego_character=fixed_ego_character,
        )
    finally:
        model.train(was_training)
    metrics = vs_cpu_metrics(results, seed=protocol.seed)
    metrics["eval_wall_seconds"] = time.perf_counter() - started
    metrics["exec_horizon"] = float(horizon)
    metrics.update(telemetry.metrics())
    _write_eval_evidence(replay_dir, rows, metrics, protocol)
    return metrics


_EVAL_SUITES: tuple[tuple[str, melee.Character | None], ...] = (
    ("char_matchup", None),
    ("fox", melee.Character.FOX),
)


def eval_suites(
    model: GPT,
    stats: dict[str, FeatureStats],
    cfg: TrainConfig,
    *,
    n_matchups: int,
    replay_dir: Path,
    checkpoint_sha256: str,
    inference: BF16Inference,
    exec_horizon: int | None = None,
) -> dict[str, dict[str, float]]:
    """Run the fixed character schedule and Fox-only schedule sequentially."""
    return {
        name: eval_vs_cpu(
            model,
            stats,
            cfg,
            n_matchups=n_matchups,
            replay_dir=replay_dir / name,
            exec_horizon=exec_horizon,
            checkpoint_sha256=checkpoint_sha256,
            inference=inference,
            fixed_ego_character=fixed_ego_character,
        )
        for name, fixed_ego_character in _EVAL_SUITES
    }


def eval_suite_wandb_metrics(
    suites: dict[str, dict[str, float]],
    *,
    suffix: str = "",
) -> dict[str, float]:
    return {
        f"eval_{suite}{suffix}/{name}": value for suite, metrics in suites.items() for name, value in metrics.items()
    }


def lr_schedule(cfg: TrainConfig):
    def schedule(step: int) -> float:
        if step < cfg.warmup_steps:
            return step / max(cfg.warmup_steps, 1)
        progress = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
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
    parameters = tuple(model.parameters())
    trunk_ids = {id(parameter) for parameter in model.trunk.parameters()}
    head_ids = {id(parameter) for parameter in model.temporal.outputs.parameters()}
    temporal_ids = {id(parameter) for parameter in model.temporal.parameters() if id(parameter) not in head_ids}
    other_ids = {id(parameter) for parameter in parameters} - trunk_ids - temporal_ids - head_ids
    partitions = {
        "trunk": trunk_ids,
        "temporal_decoder": temporal_ids,
        "group_heads": head_ids,
        "other": other_ids,
    }
    counts = {
        name: sum(parameter.numel() for parameter in parameters if id(parameter) in parameter_ids)
        for name, parameter_ids in partitions.items()
    }
    counts["total"] = sum(parameter.numel() for parameter in parameters)
    if sum(value for name, value in counts.items() if name != "total") != counts["total"]:
        raise RuntimeError("parameter subsystem partition is incomplete")
    return counts


def approximate_training_flops_per_update(cfg: TrainConfig, parameter_counts: dict[str, int]) -> int:
    """Estimate forward-backward FLOPs using O42's parameter-use formula."""
    shared = parameter_counts["trunk"] + parameter_counts["other"]
    temporal = parameter_counts["temporal_decoder"] + parameter_counts["group_heads"]
    positions = cfg.batch_size * cfg.L_ctx
    return 6 * positions * (shared + len(cfg.head_offsets) * temporal)


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


def _init_wandb(cfg: TrainConfig, run_name: str, resume_state: dict | None) -> None:
    """Start W&B with the same metric conventions as O42."""
    wandb.init(
        project="hal",
        name=run_name,
        id=None if resume_state is None else resume_state.get("wandb_id"),
        resume="allow" if resume_state is not None else None,
        tags=["gpt", "temporal-mtp", "sparse-offset", "043", "legacy-codec", f"arm-{cfg.ablation_arm}"],
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
        "O26 BC objective: mean joint NLL at offsets 1-4 plus mean joint NLL at offsets 5,6,9,12,16,20"
    )
    wandb.run.summary["architecture/ablation_arm"] = cfg.ablation_arm
    wandb.run.summary["architecture/treatment"] = _ABLATION_DESCRIPTIONS[cfg.ablation_arm]
    wandb.run.summary["data/sampler"] = "O26 replay reservoir"
    wandb.run.summary["data/source"] = cfg.data_root
    wandb.run.summary["data/replay_format"] = cfg.replay_format
    wandb.run.summary["data/validation_source"] = cfg.val_data_root
    wandb.run.summary["data/validation_replay_format"] = cfg.val_replay_format
    wandb.run.summary["evaluation/suites"] = "char_matchup,fox"
    wandb.run.summary["evaluation/final_horizons"] = f"{cfg.exec_horizon},{cfg.final_diag_exec_horizon}"
    wandb.run.summary["training/updates"] = cfg.max_steps
    wandb.run.summary["data/nominal_samples"] = cfg.max_steps * cfg.batch_size
    wandb.run.summary["data/max_context_prefixes"] = cfg.max_steps * cfg.batch_size * cfg.L_ctx
    if cfg.wandb_log_code:
        log_wandb_code(wandb.run)


def _log_training_summary(
    cfg: TrainConfig,
    parameter_counts: dict[str, int],
    *,
    flops_per_update: int,
    device_name: str | None,
    peak_flops: float | None,
) -> None:
    if wandb.run is None:
        return
    for name, value in parameter_counts.items():
        wandb.run.summary[f"parameters/{name}"] = value
    wandb.run.summary["training/approx_flops_per_update"] = flops_per_update
    wandb.run.summary["training/flops_formula"] = "6*B*L_ctx*(N_trunk+N_other+n_offsets*(N_temporal+N_group_heads))"
    if device_name is not None:
        wandb.run.summary["hardware/gpu_name"] = device_name
    if peak_flops is not None:
        wandb.run.summary["hardware/bf16_dense_peak_tflops"] = peak_flops / 1e12
        source = bf16_peak_source(device_name or "")
        if source is not None:
            wandb.run.summary["hardware/bf16_dense_peak_source"] = source


def model_tag(cfg: TrainConfig) -> str:
    offsets = "-".join(map(str, cfg.head_offsets))
    loss_tag = (
        "legacy-loss" if cfg.next_frame_loss_share is None else f"o1w{round(100 * cfg.next_frame_loss_share):02d}"
    )
    baseline = (
        f"mtp043-legacy-v{cfg.codec_version}-d{cfg.d_model}-L{cfg.n_layers}-h{cfg.n_heads}-Lc{cfg.L_ctx}-"
        f"t{cfg.temporal_d_model}x{cfg.temporal_layers}-o{offsets}-s{cfg.exec_horizon}-{loss_tag}-"
        f"{cfg.observation_bundle}"
    )
    return baseline + _ABLATION_TAGS[cfg.ablation_arm]


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


class ReplayAllowlist:
    """Accept compact replay IDs derived from a materialization paths file."""

    def __init__(self, paths_file: str) -> None:
        path = Path(paths_file)
        if not path.is_file():
            raise FileNotFoundError(f"train replay paths file not found: {path}")
        source_paths = {line.strip() for line in path.read_text().splitlines() if line.strip()}
        if not source_paths:
            raise ValueError(f"train replay paths file is empty: {path}")
        self.replay_ids = frozenset(policy_replay_identity(source_path) for source_path in source_paths)

    def __call__(self, replay_id: str) -> bool:
        return replay_id in self.replay_ids


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
        replay_filter = None if cfg.train_replay_paths is None else ReplayAllowlist(cfg.train_replay_paths)
        train_loader = make_reservoir_loader(
            split="train",
            num_workers=cfg.num_workers,
            prefetch_factor=cfg.prefetch_factor,
            predownload=cfg.predownload,
            windows_per_replay=cfg.windows_per_replay,
            reservoir_capacity=cfg.reservoir_capacity,
            prefetch_batches=cfg.prefetch_batches,
            replay_format=cfg.replay_format,
            replay_filter=replay_filter,
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
    val_kwargs = {
        **kwargs,
        "data_root": cfg.val_data_root,
        "remote": streams.remote_for_local(cfg.val_data_root),
        "batch_size": cfg.val_batch_size,
    }
    val_loader = make_loader(
        split=cfg.val_split,
        num_workers=0,
        replay_format=cfg.val_replay_format if cfg.compact_data else None,
        **val_kwargs,
    )
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
    _init_wandb(cfg, run_name, resume_state)
    run_dir, replay_dir = setup_run_dir(run_name)
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high" if cfg.allow_tf32 else "highest")
    model = GPT(cfg).to(DEVICE)
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
    host_metrics = HostMetricsSampler(
        (Path(cfg.data_root),),
        interval_s=cfg.system_metrics_interval_s,
        process_interval_s=cfg.process_metrics_interval_s,
        cache_interval_s=cfg.cache_metrics_interval_s,
    )
    host_metrics.start()
    # CUDA compilation must remain on the training thread. Background compilation
    # deadlocked training on both H100 and L40S hosts.
    eval_inference: BF16Inference | None = None
    failure: BaseException | None = None
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
                        loss = _joint_objective(
                            joint_nll,
                            valid_prefixes,
                            aux_loss_weight=cfg.aux_loss_weight,
                            next_frame_loss_share=cfg.next_frame_loss_share,
                        )
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
            metrics = nll_mean_metrics(
                (nll_sum / n_prefixes).cpu(),
                cfg.head_offsets,
                aux_loss_weight=cfg.aux_loss_weight,
                next_frame_loss_share=cfg.next_frame_loss_share,
            )
            elapsed = time.monotonic() - run_started
            completed_since_start = step - start_step + 1
            remaining_updates = cfg.max_steps - (step + 1)
            projected_remaining = elapsed * remaining_updates / completed_since_start
            update_seconds = stopwatch.elapsed + loader_wait
            log = {
                "global_step": step,
                "samples": (step + 1) * cfg.batch_size,
                "data/samples": (step + 1) * cfg.batch_size,
                "data/valid_prefixes": n_prefixes,
                "loader/wait_s": loader_wait,
                "progress/update": step + 1,
                "progress/fraction": (step + 1) / cfg.max_steps,
                "progress/elapsed_s": elapsed,
                "progress/remaining_s": projected_remaining,
                **{f"train/{name}": value for name, value in metrics.items()},
                "train/grad_norm": float(gradient_norm),
                "throughput/step_s": stopwatch.elapsed,
                "throughput/loader_wait_s": loader_wait,
                "throughput/update_s": update_seconds,
                "throughput/samples_per_s": cfg.batch_size / stopwatch.elapsed,
                "throughput/samples_per_wall_s": cfg.batch_size / (stopwatch.elapsed + loader_wait),
                "throughput/prefixes_per_s": n_prefixes / stopwatch.elapsed,
                "lr/muon": next(group["lr"] for group in optimizer.param_groups if group["use_muon"]),
                "lr/adam": next(group["lr"] for group in optimizer.param_groups if not group["use_muon"]),
                "schedule/muon_lr": next(group["lr"] for group in optimizer.param_groups if group["use_muon"]),
                "schedule/adam_lr": next(group["lr"] for group in optimizer.param_groups if not group["use_muon"]),
            }
            if cfg.system_metrics_every > 0 and (step == start_step or (step + 1) % cfg.system_metrics_every == 0):
                log.update(_minimal_system_metrics(host_metrics.snapshot()))
            if DEVICE == "cuda":
                log["hardware/peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 2**30
                log["hardware/peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30
                log["system/gpu_memory_gb"] = log["hardware/peak_allocated_gb"]
            if peak_flops is not None:
                log["throughput/mfu"] = model_flops_utilization(
                    flops_per_update,
                    update_seconds,
                    peak_flops,
                )
            wandb.log(log)
            if step < 10 or step % 50 == 0:
                print(
                    f"[t+{time.monotonic() - run_started:.0f}s] step {step}: "
                    f"{metrics['loss']:.3f} bits objective, {cfg.batch_size / update_seconds:.0f} samples/s, "
                    f"projected training remaining {projected_remaining / 60:.1f}m",
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
                suites = eval_suites(
                    model,
                    stats,
                    cfg,
                    n_matchups=cfg.eval_n_matchups,
                    replay_dir=replay_dir / f"step_{step:06d}",
                    checkpoint_sha256=_checkpoint_sha256(checkpoint_path),
                    inference=eval_inference,
                )
                wandb.log({"global_step": step, **eval_suite_wandb_metrics(suites)})

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
        final_suites = eval_suites(
            model,
            stats,
            cfg,
            n_matchups=cfg.final_eval_n_matchups,
            replay_dir=replay_dir / "final",
            checkpoint_sha256=checkpoint_sha,
            inference=eval_inference,
        )
        wandb.log({"global_step": cfg.max_steps, **eval_suite_wandb_metrics(final_suites)})
        for values in final_suites.values():
            require_complete_eval(values, cfg.final_eval_n_matchups)
        if cfg.final_diag_n_matchups > 0:
            diagnostic_suites = eval_suites(
                model,
                stats,
                cfg,
                n_matchups=cfg.final_diag_n_matchups,
                replay_dir=replay_dir / f"final_s{cfg.final_diag_exec_horizon}",
                exec_horizon=cfg.final_diag_exec_horizon,
                checkpoint_sha256=checkpoint_sha,
                inference=eval_inference,
            )
            wandb.log(
                {
                    "global_step": cfg.max_steps,
                    **eval_suite_wandb_metrics(
                        diagnostic_suites,
                        suffix=f"_s{cfg.final_diag_exec_horizon}",
                    ),
                }
            )
            for values in diagnostic_suites.values():
                require_complete_eval(values, cfg.final_diag_n_matchups)
    except BaseException as error:
        failure = error
        if wandb.run is not None:
            wandb.run.summary["run/status"] = "failed"
            wandb.run.summary["run/failure_type"] = type(error).__name__
            wandb.run.summary["run/failure_message"] = str(error)[:2000]
        raise
    finally:
        host_metrics.close()
        if uploader is not None:
            uploader.upload_tree(replay_dir, base=run_dir)
            uploader.close()
        if wandb.run is not None and failure is None:
            wandb.run.summary["run/status"] = "finished"
        wandb.finish(exit_code=1 if failure is not None else 0)


_CHECKPOINT_ARCH_FIELDS = {
    "ablation_arm",
    "codec_version",
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
    values = dict(values)
    # O43 checkpoints written before the ablation matrix are Arm A by definition.
    values.setdefault("ablation_arm", "A")
    missing = _CHECKPOINT_ARCH_FIELDS - values.keys()
    if missing:
        raise ValueError(f"checkpoint is not an experiment-043 architecture; missing {sorted(missing)}")
    if "weight_decay" in values:
        values.setdefault("muon_weight_decay", values["weight_decay"])
        values.setdefault("adam_weight_decay", values["weight_decay"])
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


def _upload_eval_evidence(run_name: str, replay_dir: Path) -> None:
    uploader = BackgroundUploader(run_name)
    uploader.upload_tree(replay_dir, base=(Path("runs") / run_name).resolve())
    uploader.close()


def _backfill_eval_metrics(
    wandb_id: str,
    step: int,
    suites: dict[str, dict[str, float]],
    *,
    suffix: str = "",
) -> None:
    wandb.init(project="hal", id=wandb_id, resume="must")
    try:
        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")
        wandb.log(
            {
                "global_step": step,
                f"eval/backfilled{suffix}": 1,
                **eval_suite_wandb_metrics(suites, suffix=suffix),
            }
        )
        if wandb.run is not None:
            wandb.run.summary["run/status"] = "finished"
            wandb.run.summary[f"evaluation/backfilled_step{suffix}"] = step
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
) -> dict[str, dict[str, float]]:
    model, cfg, stats, state = load_checkpoint(path)
    cfg = replace(
        cfg,
        inference_mode="eager" if eager else cfg.inference_mode,
        eval_max_parallel=cfg.eval_max_parallel if max_parallel is None else max_parallel,
    )
    validate_config(cfg)
    horizon = cfg.exec_horizon if exec_horizon is None else exec_horizon
    step = int(state["step"])
    matchups = cfg.final_eval_n_matchups if n_matchups is None else n_matchups
    if upload_run is not None:
        default_name = f"eval_backfill_step_{step:07d}_s{horizon}"
    else:
        default_name = f"eval_replays_s{horizon}"
    if output_name is not None and (Path(output_name).name != output_name or output_name in ("", ".", "..")):
        raise ValueError(f"evaluation output name must be one directory name, got {output_name!r}")
    replay_dir = Path(path).resolve().parent / (default_name if output_name is None else output_name)
    inference = BF16Inference(model, cfg)
    suites = eval_suites(
        model,
        stats,
        cfg,
        n_matchups=matchups,
        replay_dir=replay_dir,
        checkpoint_sha256=_checkpoint_sha256(Path(path)),
        inference=inference,
        exec_horizon=horizon,
    )
    for values in suites.values():
        require_complete_eval(values, matchups)
    if upload_run is not None:
        _upload_eval_evidence(upload_run, replay_dir)
    if backfill_wandb:
        wandb_id = state.get("wandb_id")
        if not isinstance(wandb_id, str):
            raise RuntimeError("checkpoint has no W&B run id to backfill")
        suffix = "" if horizon == cfg.exec_horizon else f"_s{horizon}"
        _backfill_eval_metrics(wandb_id, step, suites, suffix=suffix)
    print(f"[eval] step={step} horizon={horizon}: {suites}", flush=True)
    return suites


def _resolve_eval_checkpoint(checkpoint: str, run: str | None) -> Path:
    if run is None:
        return Path(checkpoint)
    path = download_latest(run, Path("runs") / run, name=checkpoint)
    if path is None:
        raise SystemExit(f"no {checkpoint!r} for run {run!r}")
    return path


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
        for horizon in _LIVE_HORIZONS:
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
    eval_run: str | None = None
    eval_exec_horizon: int | None = None
    eval_n_matchups: int | None = None
    eval_eager: bool = False
    eval_max_parallel: int | None = None
    eval_output_name: str | None = None
    eval_backfill_wandb: bool = False
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
        checkpoint = _resolve_eval_checkpoint(args.eval, args.eval_run)
        eval_checkpoint(
            str(checkpoint),
            exec_horizon=args.eval_exec_horizon,
            n_matchups=args.eval_n_matchups,
            eager=args.eval_eager,
            max_parallel=args.eval_max_parallel,
            output_name=args.eval_output_name,
            upload_run=args.eval_run,
            backfill_wandb=args.eval_backfill_wandb,
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
