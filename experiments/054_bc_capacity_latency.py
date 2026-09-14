"""Experiment 054: behavior-cloning capacity versus buffered latency.

This standalone experiment scales O52's policy architecture along one joint
width-depth family.
It uses pure behavior cloning over nested prefixes of the all-44
policy-world-v8 corpus and evaluates each width with its measured buffered
deployment timing.

This file is deliberately standalone. It freezes O52's representation,
decoder, initialization, and identity conditioning without importing another
experiment.

The four item slots are ordered by ascending spawn id, so a slot keeps its item
until an OLDER item despawns and the remaining items shift down. A pooled set
encoder makes that churn invisible: one shared per-slot encoder, gated by the
slot's presence flag, summed over the slots. An empty slot adds the exact zero
vector and the live-item count stays implicit in the sum.

Run:
    uv run experiments/054_bc_capacity_latency.py train --width 256
    uv run experiments/054_bc_capacity_latency.py latency-preflight
    uv run experiments/054_bc_capacity_latency.py eval --checkpoint runs/<run>/final.pt
"""

from __future__ import annotations

import contextlib
import csv
import functools
import gc
import hashlib
import itertools
import json
import math
import os
import random
import re
import shlex
import threading
import time
from collections import defaultdict
from collections import deque
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from dataclasses import fields
from dataclasses import replace
from pathlib import Path
from typing import Annotated
from typing import Any
from typing import ClassVar
from typing import Final
from typing import Literal
from typing import cast

import melee
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from beartype import beartype
from jaxtyping import Bool
from jaxtyping import Float
from jaxtyping import Int
from jaxtyping import jaxtyped
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR

import wandb
from hal import r2
from hal import streams
from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.data.feature_stats import FeatureStats
from hal.data.policy_world_schema import POLICY_WORLD_SCHEMA_VERSION
from hal.eval.cross_stage import BOOTSTRAP_RESAMPLES
from hal.eval.cross_stage import FRAMES_PER_MINUTE
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
from hal.eval.policy_sampling import SlotGroupRng
from hal.eval.policy_sampling import sample_categorical
from hal.eval.self_play import DecodeTelemetry
from hal.eval.self_play import canonical_context
from hal.eval.self_play import synthetic_context as build_synthetic_context
from hal.paths import ISO_PATH
from hal.paths import NETPLAY_EMULATOR_PATH
from hal.sim.inputs import action_vec_to_controller
from hal.sim.netplay import tested_dolphin_version
from hal.sim.rollout import ObservationRow
from hal.sim.rollout import PolicyRuntimeSpec
from hal.sim.rollout import covering_power_of_two
from hal.sim.session import Matchup
from hal.sim.session import PlayerSetup
from hal.sim.session import Session
from hal.sim.vec import Slot
from hal.training import scoring
from hal.training.checkpoints import BackgroundUploader
from hal.training.checkpoints import advance_checkpoint_link
from hal.training.checkpoints import checkpoint_sha256
from hal.training.checkpoints import download_latest
from hal.training.checkpoints import load_for_resume
from hal.training.checkpoints import save_checkpoint
from hal.training.closed_loop import RecedingHorizon
from hal.training.controller_codec import BUTTON_LEFT_CHANNEL
from hal.training.controller_codec import BUTTON_RIGHT_CHANNEL
from hal.training.controller_codec import BUTTONS_GROUP
from hal.training.controller_codec import CONTROLLER_DECODE_ORDER
from hal.training.controller_codec import CONTROLLER_GROUP_COUNT
from hal.training.controller_codec import CONTROLLER_GROUP_INDEX
from hal.training.controller_codec import CONTROLLER_GROUP_NAMES
from hal.training.controller_codec import CONTROLLER_GROUP_VOCABS
from hal.training.controller_codec import TRIGGER_LEFT_CHANNEL
from hal.training.controller_codec import TRIGGER_RIGHT_CHANNEL
from hal.training.controller_codec import TRIGGERS_GROUP
from hal.training.controller_codec import DiscreteControllerCodec
from hal.training.dataloader import make_loader
from hal.training.dataloader import train_batch_from_columns
from hal.training.ego_stats import load_consolidated_mixture_stats
from hal.training.features import A_DIM
from hal.training.features import ACTION_CHANNELS
from hal.training.features import BASE_ITEMS_PROJECTION
from hal.training.features import BASE_PLAYER_PREFIXES
from hal.training.features import CAT_FEATURES
from hal.training.features import FLOAT_FEATURES
from hal.training.features import ITEM_CAT_VOCABS
from hal.training.features import ITEM_COLUMNS
from hal.training.features import ITEM_FLOATS
from hal.training.features import ITEM_PLAYER_COLUMNS
from hal.training.features import ITEM_PLAYER_PROJECTION
from hal.training.features import ITEM_PRESENCE_SUFFIX
from hal.training.features import ITEM_PROBE_COLUMN
from hal.training.features import NEUTRAL_ACTION
from hal.training.features import Context
from hal.training.features import FeatureProjection
from hal.training.features import TrainBatch
from hal.training.features import stack_actions
from hal.training.mfu import bf16_dense_peak_flops
from hal.training.mfu import bf16_peak_source
from hal.training.mfu import model_flops_utilization
from hal.training.physical_shard_loader import PREFETCH_FACTOR
from hal.training.physical_shard_loader import MDSStorageAdapter
from hal.training.physical_shard_loader import PhysicalShardReplayLoader
from hal.training.physical_shard_loader import PhysicalShardSelection
from hal.training.physical_shard_loader import SourceRowSelection
from hal.training.physical_shard_loader import build_shard_plan
from hal.training.player_identity import MASKED_PLAYER_ID
from hal.training.player_identity import PlayerIdentitySidecar
from hal.training.player_identity import PlayerVocabulary
from hal.training.player_identity import ReplayPlayerLookup
from hal.training.player_identity import decode_player_codes
from hal.training.player_identity import load_player_identity_artifact
from hal.training.player_identity import vocabulary_buffer
from hal.training.runs import make_run_name
from hal.training.runs import setup_run_dir
from hal.training.system_metrics import HostMetricsSampler
from hal.training.trunk import Rotary
from hal.training.trunk import Trunk
from hal.training.trunk import TrunkConfig
from hal.training.trunk import apply_rotary_emb
from hal.wire import ITEM_SLOTS
from hal.wire import item_column

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_EXPERIMENT_ID: Final[str] = "054_latency_capacity_iso_data_v1"
_LATENCY_EXPERIMENT_ID: Final[str] = _EXPERIMENT_ID
_STARTUP_LOG_INTERVAL_S: Final[float] = 60.0
LATENCY_BATCH_SIZE: Final[int] = 1
LATENCY_WARMUP_CALLS: Final[int] = 50
LATENCY_MEASURED_CALLS: Final[int] = 500
LATENCY_TRIALS: Final[int] = 1
LATENCY_ORDER_SEED: Final[int] = 54_3060


@contextlib.contextmanager
def _elapsed_heartbeat(message: str) -> Iterator[None]:
    stop = threading.Event()
    started = time.monotonic()

    def report() -> None:
        while not stop.wait(_STARTUP_LOG_INTERVAL_S):
            print(f"{message}; {time.monotonic() - started:.1f}s elapsed", flush=True)

    reporter = threading.Thread(target=report, name="startup-progress", daemon=True)
    reporter.start()
    try:
        yield
    finally:
        stop.set()
        reporter.join()


@dataclass(frozen=True, slots=True)
class Architecture:
    player_embed_dim: ClassVar[int] = 32
    activation_percentile_sample_size: ClassVar[int] = 65_536
    trunk_attention_backend: ClassVar[str] = "varlen_flash"
    trunk_reference_layers: ClassVar[int] = 8
    temporal_reference_layers: ClassVar[int] = 2
    trunk_reference_attention_scale: ClassVar[float] = 0.25
    temporal_reference_attention_scale: ClassVar[float] = 0.5

    d_model: int = 1024
    n_layers: int = 21
    n_heads: int = 16
    attn_window: int = 0
    L_ctx: int = 256

    sample_chunk_length: int = 20
    head_offsets: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 20)
    temporal_d_model: int = 512
    temporal_layers: int = 8
    temporal_heads: int = 8
    temporal_ff_dim: int = 1536
    group_head_dim: int = 512
    action_embed_dim: int = 32
    offset_embed_dim: int = 16
    action_vocab: int = 1024
    action_state_embed_dim: int = 48
    char_vocab: int = 32
    char_dim: int = 8
    stage_vocab: int = 32
    stage_dim: int = 4
    item_type_dim: int = 16
    item_state_dim: int = 4
    item_hidden_dim: int = 64
    item_dim: int = 32

    @classmethod
    def for_width(cls, width: int) -> Architecture:
        if width not in MODEL_WIDTHS:
            raise ValueError(f"width must be one of {MODEL_WIDTHS}, got {width}")
        return cls(
            d_model=width,
            n_layers=TRUNK_DEPTHS[width],
            n_heads=width // 64,
            temporal_d_model=width // 2,
            temporal_layers=TEMPORAL_DEPTHS[width],
            temporal_heads=width // 128,
            temporal_ff_dim=3 * width // 2,
            group_head_dim=width // 2,
        )

    @property
    def direct_loss_start(self) -> int:
        if self.L_ctx % 2:
            raise ValueError("context length must be even for suffix supervision")
        return self.L_ctx // 2

    @property
    def parameter_count_contract(self) -> dict[str, int]:
        try:
            return PARAMETER_COUNT_CONTRACTS[self.d_model]
        except KeyError as error:
            raise ValueError(f"no parameter contract for architecture {self}") from error


@dataclass(frozen=True, slots=True)
class TimingConfig:
    inference_delay_frames: int
    replan_interval_frames: int
    prediction_frames: int
    transport_delay_frames: int = 0
    timing_model: Literal["buffered"] = "buffered"

    def __post_init__(self) -> None:
        values = (
            self.inference_delay_frames,
            self.replan_interval_frames,
            self.prediction_frames,
            self.transport_delay_frames,
        )
        if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
            raise ValueError("timing values must be integers")
        if self.inference_delay_frames < 1 or self.replan_interval_frames < 1:
            raise ValueError("inference delay and replan interval must be positive")
        if self.prediction_frames < 1 or self.transport_delay_frames != 0:
            raise ValueError("prediction frames must be positive and transport delay must be zero")
        if self.inference_delay_frames + self.replan_interval_frames > self.prediction_frames:
            raise ValueError("inference_delay_frames + replan_interval_frames must not exceed prediction_frames")


WIDTHS: Final[tuple[int, ...]] = (256, 512, 768, 1024)
MODEL_WIDTHS: Final[tuple[int, ...]] = WIDTHS
TRUNK_DEPTHS: Final[dict[int, int]] = {
    256: 5,
    512: 11,
    768: 16,
    1024: 21,
}
TEMPORAL_DEPTHS: Final[dict[int, int]] = {256: 2, 512: 4, 768: 6, 1024: 8}
DEDUPLICATED_SOURCE_ROWS: Final[dict[str, tuple[int, ...]]] = {
    "professional-monotheon-policy-world-v8": (14_136, 14_139),
}
INITIAL_DELAYS: Final[dict[int, int]] = {
    256: 1,
    512: 2,
    768: 3,
    1024: 6,
}
DATA_EXPONENTS: Final[tuple[int, ...]] = (28, 29, 30, 31)
DATA_POSITIONS: Final[tuple[int, ...]] = tuple(2**exponent for exponent in DATA_EXPONENTS)
DATA_UPDATES: Final[tuple[int, ...]] = tuple(positions // (512 * 128) for positions in DATA_POSITIONS)
DATA_REPLAYS: Final[tuple[int, ...]] = tuple(positions // (32 * 128) for positions in DATA_POSITIONS)
DATA_PHASE_REPLAYS: Final[tuple[int, ...]] = tuple(
    cumulative - (0 if index == 0 else DATA_REPLAYS[index - 1]) for index, cumulative in enumerate(DATA_REPLAYS)
)
B200_REFERENCE_UPDATE_SECONDS: Final[dict[int, float]] = {
    256: 0.25,
    512: 0.30,
    768: 0.40,
    1024: 1.00,
}
TRAIN_MICROBATCH_SIZES: Final[dict[int, int]] = {256: 512, 512: 512, 768: 512, 1024: 256}
MODAL_TRAINING_DOLLARS_PER_HOUR: Final[float] = 8.781696
STUDY_WORKING_COST_DOLLARS: Final[int] = 225


def _derived_parameter_count_contract(width: int) -> dict[str, int]:
    trunk = 12 * width * width * TRUNK_DEPTHS[width]
    temporal = (width * width + 535 * width) // 2 + 12_144 + TEMPORAL_DEPTHS[width] * (5 * width * width // 2)
    group_heads = width * width + 1065 * width // 2 + 355
    inputs = 481 * width + 738_246
    return {
        "trunk": trunk,
        "temporal_decoder": temporal,
        "group_heads": group_heads,
        "inputs": inputs,
        "total": trunk + temporal + group_heads + inputs,
    }


PARAMETER_COUNT_CONTRACTS: Final[dict[int, dict[str, int]]] = {
    width: _derived_parameter_count_contract(width) for width in WIDTHS
}

POWERLINES_EXPONENT: Final[float] = 0.52
POWERLINES_REFERENCE_WIDTH: Final[int] = 512
POWERLINES_REFERENCE_POSITIONS: Final[int] = 2**30
POWERLINES_REFERENCE_WEIGHT_DECAY: Final[float] = 1e-4


def powerlines_weight_decay(positions: int, total_parameters: int) -> float:
    """Scale AdamW decay from the declared W512, D=2^30 anchor."""
    if positions < 1 or total_parameters < 1:
        raise ValueError("Power Lines positions and parameters must be positive")
    reference_parameters = PARAMETER_COUNT_CONTRACTS[POWERLINES_REFERENCE_WIDTH]["total"]
    tokens_per_parameter_ratio = (positions / total_parameters) / (
        POWERLINES_REFERENCE_POSITIONS / reference_parameters
    )
    return (
        POWERLINES_REFERENCE_WEIGHT_DECAY
        * (POWERLINES_REFERENCE_POSITIONS / positions)
        * tokens_per_parameter_ratio**POWERLINES_EXPONENT
    )


def timing_for_width(width: int) -> TimingConfig:
    try:
        delay = INITIAL_DELAYS[width]
    except KeyError as error:
        raise ValueError(f"width must be one of {MODEL_WIDTHS}, got {width}") from error
    return TimingConfig(delay, delay, 2 * delay)


@dataclass(frozen=True, slots=True)
class TrainConfig:
    reference_batch_size: ClassVar[int] = 512
    base_adam_betas: ClassVar[tuple[float, float]] = (0.9, 0.95)
    base_adam_eps: ClassVar[float] = 1e-12
    inference_buckets: ClassVar[tuple[int, ...]] = (1, 2, 4, 8, 16, 32, 64)
    train_metrics_every: ClassVar[int] = 25
    train_prefetch_factor: ClassVar[int] = 4
    train_compile_mode: ClassVar[str] = "reduce-overhead"
    raw_shard_materialization_threads: ClassVar[int] = 64
    materialization_threads_env: ClassVar[str] = "HAL_O54_MATERIALIZATION_THREADS"
    data_protocol: ClassVar[str] = "o54-all44-disjoint-nested-phases-v2"
    selection_sha256: ClassVar[str] = "75736da2e9d165781fb4ac34a79b8c99b49e96991f1ee0eab9f629d154805c23"
    mds_index_version: ClassVar[int] = 2
    mds_manifest_schema_sha256: ClassVar[str] = "405199de9494fe01350506734f0b2ec392fe79b0122d69cbcb5cae2afabc0d49"
    replay_slots: ClassVar[int] = 65_536
    windows_per_generation: ClassVar[int] = 8
    generations_per_replay: ClassVar[int] = 4
    replay_phase_block_batches: ClassVar[int] = 25
    minimum_replay_gap_batches: ClassVar[int] = 104
    reserved_disk_bytes: ClassVar[int] = 256 * 2**30
    player_sidecar_remote: ClassVar[str] = "s3://hal/processed/player-identity-v1/professional-code-v1.jsonl.gz"

    arch: Annotated[Architecture, tyro.conf.Suppress] = Architecture()
    timing: Annotated[TimingConfig, tyro.conf.Suppress] = dataclass_field(
        default_factory=lambda: timing_for_width(1024)
    )

    inference_mode: str = "eager"
    latency_manifest_sha256: str = ""
    # Hardware-derived by default. An explicit power of two is a reproducibility
    # or memory-pressure override, not an architecture parameter.
    compiled_inference_bucket: int | None = None

    seed: int = 0
    eval_seed: int = 0
    batch_size: int = 512
    adam_lr: float = 8e-4
    grad_clip: float = 1.0
    amp_dtype: str = "bfloat16"
    allow_tf32: bool = True
    compile_trunk: bool = True
    compile_temporal: bool = True

    wandb_log_code: bool = True
    val_every: int = 4096
    val_n_samples: int = 1024
    val_batch_size: int = 128
    ckpt_every: int = 2048
    eval_every: int = 8192
    eval_max_frames: int = 7200
    eval_n_matchups: int = 96
    final_eval_n_matchups: int = 96
    eval_max_parallel: int | None = 32
    automatic_evaluation: bool = True

    source_names: tuple[str, ...] = tuple(source.name for source in streams.POLICY_WORLD_V8_SOURCES)
    mds_schema_version: int = 7
    policy_world_schema_version: int = POLICY_WORLD_SCHEMA_VERSION
    download_retry: int = 8
    val_split: str = "val"
    num_workers: int = 24
    push_to_r2: bool = True
    system_metrics_every: int = 25
    system_metrics_interval_s: float = 5.0
    process_metrics_interval_s: float = 30.0
    cache_metrics_interval_s: float = 30.0
    phase_timing_every: int = 256
    identity_dropout: float = 0.10
    player_sidecar_local: str = "data/processed/player-identity-v1/professional-code-v1.jsonl.gz"
    player_sidecar_sha256: str = "54ccf8a2497fe240313117297ca2ea31158e08db2cc53c67e7aa46853a8dac1c"
    player_vocab_sha256: str = "c67c97c995ad033ea7f5b2223efce5b061394566439f091ff6e7aaa6a9d1cfd6"
    player_vocab_size: int = 21_181
    target_positions: int = DATA_POSITIONS[-1]
    depth_alpha: float = 0.5
    hidden_std_multiplier: float = 0.5
    readout_init: Literal["mup-normal"] = "mup-normal"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-12

    @property
    def prediction_frames(self) -> int:
        return self.timing.prediction_frames

    @property
    def delay_frames(self) -> int:
        return self.timing.inference_delay_frames

    @property
    def replan_interval_frames(self) -> int:
        return self.timing.replan_interval_frames

    @property
    def max_steps(self) -> int:
        positions_per_update = self.batch_size * (self.arch.L_ctx - self.arch.L_ctx // 2)
        updates, remainder = divmod(self.target_positions, positions_per_update)
        if remainder:
            raise ValueError("target_positions must end on an optimizer boundary")
        return updates

    @property
    def warmup_steps(self) -> int:
        return 512

    @property
    def adam_weight_decay(self) -> float:
        return powerlines_weight_decay(DATA_POSITIONS[-1], self.arch.parameter_count_contract["total"])

    @property
    def source_list_sha256(self) -> str:
        encoded = json.dumps(self.source_names, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @property
    def train_replays(self) -> int:
        return sum(
            streams.POLICY_WORLD_V8_TRAIN_REPLAYS[name] - len(DEDUPLICATED_SOURCE_ROWS.get(name, ()))
            for name in self.source_names
        )

    @property
    def train_frames(self) -> int:
        return sum(streams.POLICY_WORLD_V8_TRAIN_FRAMES[name] for name in self.source_names)

    def materialization_threads(self) -> int:
        value = os.environ.get(self.materialization_threads_env)
        if value is None:
            return self.raw_shard_materialization_threads
        try:
            threads = int(value)
        except ValueError as error:
            raise ValueError(f"{self.materialization_threads_env} must be an integer") from error
        if threads not in (32, 64, 96):
            raise ValueError(f"{self.materialization_threads_env} must be 32, 64, or 96")
        return threads


def validate_config(cfg: TrainConfig) -> None:
    positive = {
        "d_model": cfg.arch.d_model,
        "n_layers": cfg.arch.n_layers,
        "n_heads": cfg.arch.n_heads,
        "L_ctx": cfg.arch.L_ctx,
        "sample_chunk_length": cfg.arch.sample_chunk_length,
        "temporal_d_model": cfg.arch.temporal_d_model,
        "temporal_layers": cfg.arch.temporal_layers,
        "temporal_heads": cfg.arch.temporal_heads,
        "temporal_ff_dim": cfg.arch.temporal_ff_dim,
        "group_head_dim": cfg.arch.group_head_dim,
        "action_embed_dim": cfg.arch.action_embed_dim,
        "offset_embed_dim": cfg.arch.offset_embed_dim,
        "item_type_dim": cfg.arch.item_type_dim,
        "item_state_dim": cfg.arch.item_state_dim,
        "item_hidden_dim": cfg.arch.item_hidden_dim,
        "item_dim": cfg.arch.item_dim,
        "batch_size": cfg.batch_size,
        "max_steps": cfg.max_steps,
        "warmup_steps": cfg.warmup_steps,
        "target_positions": cfg.target_positions,
        "download_retry": cfg.download_retry,
    }
    for name, value in positive.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if cfg.arch.d_model % cfg.arch.n_heads or cfg.arch.temporal_d_model % cfg.arch.temporal_heads:
        raise ValueError("model dimensions must be divisible by their head counts")
    if (cfg.arch.temporal_d_model // cfg.arch.temporal_heads) % 2:
        raise ValueError("temporal head dimension must be even for rotary positions")
    offsets = tuple(cfg.arch.head_offsets)
    if offsets != tuple(sorted(set(offsets))) or not offsets or offsets[0] != 1:
        raise ValueError(f"head_offsets must be sorted, unique, and start at 1, got {offsets}")
    if offsets[-1] > cfg.arch.sample_chunk_length:
        raise ValueError("head_offsets extend beyond sample_chunk_length")
    if offsets != (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 20):
        raise ValueError("O54 uses the fixed dense-12 plus 16 and 20 offset set")
    expected_architecture = Architecture.for_width(cfg.arch.d_model)
    if cfg.arch != expected_architecture:
        raise ValueError(f"O54 requires the joint width-depth architecture {expected_architecture}, got {cfg.arch}")
    if not 1 <= cfg.delay_frames <= 6:
        raise ValueError("measured inference delay must be in [1, 6]")
    if (cfg.replan_interval_frames, cfg.prediction_frames) != (cfg.delay_frames, 2 * cfg.delay_frames):
        raise ValueError("buffered timing requires R=d and H=2d")
    if (cfg.eval_n_matchups, cfg.final_eval_n_matchups) != (96, 96):
        raise ValueError("gameplay evaluation is frozen to 96 matchups")
    if (
        cfg.batch_size != 512
        or cfg.arch.direct_loss_start != 128
        or cfg.replay_slots != 65_536
        or cfg.windows_per_generation * cfg.generations_per_replay != 32
        or DATA_UPDATES != (4096, 8192, 16_384, 32_768)
        or DATA_REPLAYS != (65_536, 131_072, 262_144, 524_288)
    ):
        raise ValueError("O54 nested-data geometry changed")
    microbatch_size = train_microbatch_size(cfg)
    if cfg.batch_size % microbatch_size:
        raise ValueError(f"training microbatch {microbatch_size} must divide optimizer batch {cfg.batch_size}")
    if cfg.inference_mode != "eager":
        raise ValueError("O54 requires the eager inference path measured by the latency preflight")
    if cfg.latency_manifest_sha256 and (
        len(cfg.latency_manifest_sha256) != 64
        or any(character not in "0123456789abcdef" for character in cfg.latency_manifest_sha256)
    ):
        raise ValueError("latency manifest SHA-256 is invalid")
    if cfg.compiled_inference_bucket is not None and (
        cfg.compiled_inference_bucket < 1 or cfg.compiled_inference_bucket & (cfg.compiled_inference_bucket - 1)
    ):
        raise ValueError("compiled_inference_bucket must be a positive power of two")
    if cfg.eval_max_parallel is not None and (
        not isinstance(cfg.eval_max_parallel, int)
        or isinstance(cfg.eval_max_parallel, bool)
        or cfg.eval_max_parallel < 1
    ):
        raise ValueError("eval_max_parallel must be a positive integer")
    policy_world_names = {source.name for source in streams.POLICY_WORLD_V8_SOURCES}
    if not set(cfg.source_names) <= policy_world_names:
        raise ValueError(
            "projectile inputs need policy-world sources: no other decoder emits the item columns, "
            f"so the projectile block would never reach the model; got {sorted(cfg.source_names)}"
        )
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
    if cfg.amp_dtype != "bfloat16":
        raise ValueError("O54 requires BF16 training and inference")
    if not isinstance(cfg.num_workers, int) or isinstance(cfg.num_workers, bool) or not 0 <= cfg.num_workers <= 48:
        raise ValueError(f"num_workers must be an integer in [0, 48], got {cfg.num_workers!r}")
    if not math.isfinite(cfg.grad_clip) or cfg.grad_clip <= 0:
        raise ValueError(f"grad_clip must be finite and positive, got {cfg.grad_clip}")
    if not 0.0 <= cfg.identity_dropout <= 1.0:
        raise ValueError("identity_dropout must be in [0, 1]")
    expected_identity = TrainConfig()
    if cfg.player_vocab_size != expected_identity.player_vocab_size:
        raise ValueError(f"player_vocab_size must be {expected_identity.player_vocab_size}")
    if (
        cfg.player_sidecar_sha256 != expected_identity.player_sidecar_sha256
        or cfg.player_vocab_sha256 != expected_identity.player_vocab_sha256
    ):
        raise ValueError("identity hashes differ from the frozen O49 artifacts")
    if not cfg.source_names or len(set(cfg.source_names)) != len(cfg.source_names):
        raise ValueError("source_names must be non-empty and unique")
    unknown = set(cfg.source_names) - streams.BY_NAME.keys()
    if unknown:
        raise ValueError(f"unknown source names: {sorted(unknown)}")
    if cfg.policy_world_schema_version != POLICY_WORLD_SCHEMA_VERSION:
        raise ValueError(
            f"policy_world_schema_version {cfg.policy_world_schema_version} != {POLICY_WORLD_SCHEMA_VERSION}"
        )
    expected_sources = tuple(source.name for source in streams.POLICY_WORLD_V8_SOURCES)
    if cfg.source_names != expected_sources:
        raise ValueError("O54 requires the frozen all-44 policy-world-v8 source order")
    if cfg.depth_alpha != 0.5 or cfg.hidden_std_multiplier != 0.5 or cfg.readout_init != "mup-normal":
        raise ValueError("O54 uses O52's selected depth and initialization parameterization")
    if (cfg.adam_beta1, cfg.adam_beta2, cfg.adam_eps) != (*cfg.base_adam_betas, cfg.base_adam_eps):
        raise ValueError("O54 uses the fixed Adam betas and epsilon")
    for name, value in (("adam_lr", cfg.adam_lr), ("adam_eps", cfg.adam_eps)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(cfg.adam_weight_decay) or cfg.adam_weight_decay <= 0:
        raise ValueError("derived AdamW weight decay must be finite and positive")


def proxy_config() -> TrainConfig:
    """Return the W256/L5 full nested-data proxy treatment."""
    return config_for_width(256)


def config_for_width(width: int, *, updates: int | None = None) -> TrainConfig:
    if updates is None:
        if width not in MODEL_WIDTHS:
            raise ValueError(f"width {width} has no architecture contract")
        updates = DATA_UPDATES[-1]
    if updates < 1:
        raise ValueError("updates must be positive")
    return TrainConfig(
        arch=Architecture.for_width(width),
        timing=timing_for_width(width),
        target_positions=updates * 512 * 128,
    )


def train_microbatch_size(cfg: TrainConfig) -> int:
    """Return the execution batch that preserves the 512-window optimizer batch."""
    try:
        return TRAIN_MICROBATCH_SIZES[cfg.arch.d_model]
    except KeyError as error:
        raise ValueError(f"width {cfg.arch.d_model} has no training microbatch contract") from error


def _training_compile_mode(cfg: TrainConfig) -> str:
    """Disable CUDA graphs when one optimizer step needs multiple backwards."""
    if train_microbatch_size(cfg) < cfg.batch_size:
        return "default"
    return cfg.train_compile_mode


def _training_uses_cuda_graphs(cfg: TrainConfig) -> bool:
    return _training_compile_mode(cfg) == "reduce-overhead"


def synthetic_context(cfg: TrainConfig, batch_size: int, device: torch.device) -> Context:
    """Build the fixed base observation with projectile columns."""
    context = build_synthetic_context(
        cfg,
        batch_size,
        device,
        context_length=cfg.arch.L_ctx,
        observation_bundle="base",
        items=True,
    )
    return Context(
        features={
            **context.features,
            "ego_player_id": torch.zeros(batch_size, cfg.arch.L_ctx, dtype=torch.long, device=device),
        },
        ctx_pad=context.ctx_pad,
        slot_ids=context.slot_ids,
        reset=context.reset,
    )


def synthetic_batch(cfg: TrainConfig, device: torch.device) -> TrainBatch:
    """Build one fully valid production-shaped batch without touching the corpus."""
    context = synthetic_context(cfg, cfg.batch_size, device)
    target = torch.zeros(cfg.batch_size, cfg.arch.sample_chunk_length, A_DIM, device=device)
    return TrainBatch(context=context, target=target)


def _eval_parallelism(cfg: TrainConfig, n_matchups: int, max_parallel: int | None = None) -> int:
    requested = cfg.eval_max_parallel if max_parallel is None else max_parallel
    return resolve_parallelism(n_matchups, requested)


def _eval_inference_bucket(cfg: TrainConfig, n_matchups: int, max_parallel: int | None = None) -> int:
    rows = _eval_parallelism(cfg, n_matchups, max_parallel)
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
    ("optimizer", "grad_norm_end", "optimizer_end"),
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
                metrics[f"profile/{metric}_s"] = self._events[start].elapsed_time(self._events[end]) / 1000
        return metrics


def decoder_rmsnorm(x: Tensor) -> Tensor:
    return F.rms_norm(x, (x.shape[-1],), eps=1e-6)


def action_rmsnorm(x: Tensor) -> Tensor:
    """Normalize an action-boundary tensor with a safer near-zero Jacobian."""
    return F.rms_norm(x, (x.shape[-1],), eps=1e-5)


class SwiGLU(nn.Module):
    """Gated MLP used by every nonlinear projection in the policy."""

    def __init__(self, d_input: int, d_hidden: int, d_output: int, *, output_bias: bool = False) -> None:
        super().__init__()
        self.up = nn.Linear(d_input, 2 * d_hidden, bias=False)
        self.down = nn.Linear(d_hidden, d_output, bias=output_bias)

    def activations(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return the activated gate, value branch, and their product."""
        gate_projection, value = self.up(x).chunk(2, dim=-1)
        gate = F.silu(gate_projection)
        return gate, value, gate * value

    def forward(self, x: Float[Tensor, "... d_input"]) -> Float[Tensor, "... d_output"]:
        _, _, product = self.activations(x)
        return self.down(product)


class NonlinearActionHead(nn.Module):
    """O26 RMSNorm-SiLU controller readout."""

    def __init__(self, d_model: int, d_hidden: int, vocab: int) -> None:
        super().__init__()
        self.up = nn.Linear(d_model, d_hidden, bias=False)
        self.down = nn.Linear(d_hidden, vocab)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.up(decoder_rmsnorm(x))))

    def forward_with_input(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return logits and the normalized tensor read by the hidden layer."""
        normalized = action_rmsnorm(x)
        return self.down(F.silu(self.up(normalized))), normalized


def _sampled_quantile(tensor: Tensor, percentile: float, *, absolute: bool = False) -> Tensor:
    """Estimate a quantile from a deterministic bounded-size flat sample."""
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"percentile must be in [0, 100], got {percentile}")
    values = tensor.detach().flatten()
    sample_size = Architecture.activation_percentile_sample_size
    stride = max((values.numel() + sample_size - 1) // sample_size, 1)
    if stride > 1:
        stride += 1
    sample = values[::stride].float()
    if absolute:
        sample = sample.abs()
    rank = min(max(math.ceil(percentile * sample.numel() / 100.0), 1), sample.numel())
    return torch.kthvalue(sample, rank).values


def short_causal_attention(
    query: Float[Tensor, "B H L D"],
    key: Float[Tensor, "B H L D"],
    value: Float[Tensor, "B H L D"],
) -> Float[Tensor, "B H L D"]:
    """Use explicit causal attention for the 14-token training sequence.

    On a B200, cuDNN flash SDPA used 149 ms per step for forward and
    backward. At this length, its launch and layout costs were more than the
    cost to materialize 121 scores per head. This implementation reduced the
    full train step by approximately 100 ms.
    """
    scores = query @ key.transpose(-2, -1)
    scores = scores.float() * (query.shape[-1] ** -0.5)
    causal = torch.ones(scores.shape[-2:], dtype=torch.bool, device=scores.device).tril()
    weights = F.softmax(scores.masked_fill(~causal, -torch.inf), dim=-1).to(query.dtype)
    return weights @ value


TEMPORAL_ATTENTION_BATCH = 16_384


@dataclass(frozen=True, slots=True)
class DepthRule:
    attention: float
    mlp: float


def depth_rule(stack: Literal["trunk", "temporal"], layers: int, alpha: float) -> DepthRule:
    """Return O51's residual multipliers for one stack."""
    if alpha != 0.5:
        raise ValueError("O52 fixes depth_alpha to 0.5")
    if layers < 1:
        raise ValueError("stack depth must be positive")
    if stack == "trunk":
        base_layers = Architecture.trunk_reference_layers
        base_attention = Architecture.trunk_reference_attention_scale
    elif stack == "temporal":
        base_layers = Architecture.temporal_reference_layers
        base_attention = Architecture.temporal_reference_attention_scale
    else:
        raise ValueError(f"unknown stack {stack!r}")
    branch = (layers / base_layers) ** -alpha
    return DepthRule(attention=base_attention * branch, mlp=branch)


class TemporalBlock(nn.Module):
    """RoPE causal block with O51 depth scaling on both branches."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.n_heads = cfg.arch.temporal_heads
        self.d_model = cfg.arch.temporal_d_model
        self.head_dim = self.d_model // self.n_heads
        rule = depth_rule("temporal", cfg.arch.temporal_layers, cfg.depth_alpha)
        self.scale = rule.attention
        self.mlp_scale = rule.mlp
        self.qkv = nn.Linear(self.d_model, 3 * self.d_model, bias=False)
        self.proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.rotary = Rotary(self.head_dim)
        self.up = nn.Linear(self.d_model, cfg.arch.temporal_ff_dim, bias=False)
        self.down = nn.Linear(cfg.arch.temporal_ff_dim, self.d_model, bias=False)

    def _qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, length, _ = x.shape
        q, k, v = self.qkv(decoder_rmsnorm(x)).split(self.d_model, dim=-1)
        shape = (batch, length, self.n_heads, self.head_dim)
        return q.view(shape), k.view(shape), v.view(shape)

    def forward(self, x: Tensor) -> Tensor:
        q, k, v = self._qkv(x)
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin).transpose(1, 2)
        k = apply_rotary_emb(k, cos, sin).transpose(1, 2)
        v = v.transpose(1, 2)
        # Equal attention chunks keep the tiny BMMs efficient. The large linear
        # and SwiGLU operations use the full batch.
        attended = torch.cat(
            [
                short_causal_attention(query, key, values)
                for query, key, values in zip(
                    q.split(TEMPORAL_ATTENTION_BATCH),
                    k.split(TEMPORAL_ATTENTION_BATCH),
                    v.split(TEMPORAL_ATTENTION_BATCH),
                    strict=True,
                )
            ],
            dim=0,
        )
        attended = attended.transpose(1, 2).contiguous().view_as(x)
        x = x + self.scale * self.proj(attended)
        return x + self.mlp_scale * self.down(F.silu(self.up(decoder_rmsnorm(x))))

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
        x = x + self.mlp_scale * self.down(F.silu(self.up(decoder_rmsnorm(x))))
        return x, (k, v)


class CausalTemporalDecoder(nn.Module):
    """Temporal action chain conditioned by concatenation."""

    def __init__(self, cfg: TrainConfig, codec: DiscreteControllerCodec) -> None:
        super().__init__()
        self.codec = codec
        self.head_offsets = tuple(cfg.arch.head_offsets)
        self.live_horizons = (cfg.prediction_frames,)
        self.d_model = cfg.arch.temporal_d_model
        controller_width = CONTROLLER_GROUP_COUNT * cfg.arch.action_embed_dim
        self.offset_embedding = nn.Embedding(cfg.arch.sample_chunk_length + 1, cfg.arch.offset_embed_dim)
        self.token_projection = nn.Linear(
            cfg.arch.d_model + controller_width + cfg.arch.offset_embed_dim, self.d_model
        )
        self.blocks = nn.ModuleList([TemporalBlock(cfg) for _ in range(cfg.arch.temporal_layers)])
        self.group_condition = nn.ModuleDict(
            {
                name: nn.Linear(position * cfg.arch.action_embed_dim, 2 * self.d_model)
                for position, name in enumerate(CONTROLLER_DECODE_ORDER)
                if position
            }
        )
        self.outputs = nn.ModuleDict(
            {
                name: NonlinearActionHead(
                    self.d_model, cfg.arch.group_head_dim, CONTROLLER_GROUP_VOCABS[CONTROLLER_GROUP_INDEX[name]]
                )
                for name in CONTROLLER_GROUP_NAMES
            }
        )
        self.trunk_outputs = nn.ModuleDict(
            {
                name: nn.Linear(cfg.arch.d_model, CONTROLLER_GROUP_VOCABS[CONTROLLER_GROUP_INDEX[name]], bias=False)
                for name in CONTROLLER_GROUP_NAMES
            }
        )
        self.trunk_width = cfg.arch.d_model
        self.controller_width = controller_width

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

    def _decode_step(
        self,
        previous: Tensor,
        offset: int,
        state_bias: Tensor,
        caches: list[tuple[Tensor, Tensor] | None],
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor] | None]]:
        """Advance the temporal chain by one selected frame offset."""
        offsets = torch.full((previous.shape[0],), offset, device=previous.device, dtype=torch.long)
        state = decoder_rmsnorm(state_bias + self._step_features(previous, offsets))
        next_caches: list[tuple[Tensor, Tensor] | None] = []
        for module, past in zip(self.blocks, caches, strict=True):
            block = cast(TemporalBlock, module)
            state, present = block.forward_step(state, past)
            next_caches.append(present)
        return decoder_rmsnorm(state), next_caches

    def teacher_forced_states(self, hidden: Tensor, observed: Tensor, targets: Tensor) -> Tensor:
        expected = (*hidden.shape[:2], len(self.head_offsets), CONTROLLER_GROUP_COUNT)
        if observed.shape != (*hidden.shape[:2], CONTROLLER_GROUP_COUNT) or targets.shape != expected:
            raise ValueError(
                f"expected observed {(*hidden.shape[:2], CONTROLLER_GROUP_COUNT)} and targets {expected}, got "
                f"{tuple(observed.shape)} and {tuple(targets.shape)}"
            )
        previous = torch.cat((observed[:, :, None], targets[..., :-1, :]), dim=2)
        trunk = decoder_rmsnorm(hidden)
        offsets = torch.tensor(self.head_offsets, device=hidden.device)
        x = self._state_bias(trunk)[:, :, None] + self._step_features(previous, offsets)
        x = decoder_rmsnorm(x)
        x = x.reshape(hidden.shape[0] * hidden.shape[1], len(self.head_offsets), self.d_model)
        for block in self.blocks:
            x = block(x)
        return decoder_rmsnorm(x.view(*hidden.shape[:2], len(self.head_offsets), self.d_model))

    def group_features(self, states: Tensor, name: str, embedded: dict[str, Tensor]) -> Tensor:
        position = CONTROLLER_DECODE_ORDER.index(name)
        if position == 0:
            return states
        prefix = torch.cat([embedded[group] for group in CONTROLLER_DECODE_ORDER[:position]], dim=-1)
        raw_scale, raw_shift = self.group_condition[name](prefix).chunk(2, dim=-1)
        scale = torch.tanh(raw_scale)
        shift = raw_shift
        return states * (1.0 + scale) + shift

    def _teacher_forced_outputs(
        self,
        hidden: Tensor,
        observed: Tensor,
        targets: Tensor,
    ) -> tuple[dict[str, Tensor], tuple[Tensor, Tensor, Tensor, Tensor]]:
        """Return group logits and the button tensors used for diagnostics."""
        states = self.teacher_forced_states(hidden, observed, targets)
        embedded = self.codec.embed_groups(targets)
        logits: dict[str, Tensor] = {}
        button_values: tuple[Tensor, Tensor, Tensor] | None = None
        for name in CONTROLLER_GROUP_NAMES:
            features = self.group_features(states, name, embedded)
            if name == "buttons":
                head = cast(NonlinearActionHead, self.outputs[name])
                combined_logits, head_input = head.forward_with_input(features)
                combined_logits = combined_logits + self.trunk_outputs[name](hidden)[..., None, :]
                button_values = (features, head_input, combined_logits)
            else:
                combined_logits = self.outputs[name](features) + self.trunk_outputs[name](hidden)[..., None, :]
            logits[name] = self._center(combined_logits)
        if button_values is None:
            raise RuntimeError("button head was not evaluated")
        button_mask = self.codec.button_mask(targets[..., TRIGGERS_GROUP])
        logits["buttons"] = logits["buttons"].masked_fill(button_mask, float("-inf"))
        return logits, (*button_values, button_mask)

    def teacher_forced_logits_by_group(self, hidden: Tensor, observed: Tensor, targets: Tensor) -> dict[str, Tensor]:
        logits, _ = self._teacher_forced_outputs(hidden, observed, targets)
        return logits

    @staticmethod
    def nll_from_logits(logits: dict[str, Tensor], targets: Tensor) -> Tensor:
        losses = [
            F.cross_entropy(
                logits[name].float().reshape(-1, CONTROLLER_GROUP_VOCABS[group]),
                targets[..., group].reshape(-1),
                reduction="none",
            ).view(*targets.shape[:-1])
            for group, name in enumerate(CONTROLLER_GROUP_NAMES)
        ]
        return torch.stack(losses, dim=-1)

    def teacher_forced_nll(self, hidden: Tensor, observed: Tensor, targets: Tensor) -> Tensor:
        logits = self.teacher_forced_logits_by_group(hidden, observed, targets)
        return self.nll_from_logits(logits, targets)

    def teacher_forced_nll_with_diagnostics(
        self,
        hidden: Tensor,
        observed: Tensor,
        targets: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Return NLL plus a compact button-boundary stability signal."""
        logits, button_values = self._teacher_forced_outputs(hidden, observed, targets)
        features, head_input, raw_logits, button_mask = button_values
        feature_rms = features.detach().float().square().mean(dim=-1).sqrt()
        input_values = head_input.detach()
        raw_logits_values = raw_logits.detach()
        button_targets = targets[..., BUTTONS_GROUP, None]
        target_logits = raw_logits_values.gather(-1, button_targets).squeeze(-1).float()
        legal_logits = raw_logits_values.masked_fill(button_mask, float("-inf"))
        competing_logits = legal_logits.scatter(-1, button_targets, float("-inf")).amax(dim=-1).float()
        margin = target_logits - competing_logits
        metrics = {
            "stability/button_pre_norm_rms_min": feature_rms.amin(),
            "stability/button_input_abs_p999": _sampled_quantile(input_values, 99.9, absolute=True),
            "stability/button_logit_abs_p999": _sampled_quantile(raw_logits_values, 99.9, absolute=True),
            "stability/button_margin_mean": margin.mean(),
        }
        return self.nll_from_logits(logits, targets), metrics

    def teacher_forced_logits(self, hidden: Tensor, observed: Tensor, targets: Tensor) -> list[dict[str, Tensor]]:
        values = self.teacher_forced_logits_by_group(hidden, observed, targets)
        return [
            {name: logits[..., depth, :] for name, logits in values.items()} for depth in range(len(self.head_offsets))
        ]

    def forced_stepwise_logits(self, hidden: Tensor, observed: Tensor, targets: Tensor) -> list[dict[str, Tensor]]:
        if targets.shape != (hidden.shape[0], len(self.head_offsets), CONTROLLER_GROUP_COUNT):
            raise ValueError("stepwise targets have the wrong shape")
        raw_trunk = hidden[:, -1]
        trunk = decoder_rmsnorm(raw_trunk)
        state_bias = self._state_bias(trunk)
        previous = observed
        caches: list[tuple[Tensor, Tensor] | None] = [None] * len(self.blocks)
        out: list[dict[str, Tensor]] = []
        for depth, offset in enumerate(self.head_offsets):
            state, caches = self._decode_step(previous, offset, state_bias, caches)
            target = targets[:, depth]
            embedded = self.codec.embed_groups(target)
            group_logits = {
                name: self._center(
                    self.outputs[name](self.group_features(state, name, embedded))
                    + self.trunk_outputs[name](raw_trunk)
                )
                for name in CONTROLLER_GROUP_NAMES
            }
            group_logits["buttons"] = group_logits["buttons"].masked_fill(
                self.codec.button_mask(target[:, TRIGGERS_GROUP]), float("-inf")
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
        allowed = tuple(self.head_offsets[:horizon] for horizon in self.live_horizons)
        if offsets not in allowed:
            raise ValueError(f"live decode offsets must select one of the dense prefixes {allowed}")
        if uniforms is not None and uniforms.shape != (len(offsets), CONTROLLER_GROUP_COUNT, hidden.shape[0]):
            raise ValueError("uniform table must be [frames, groups, batch]")
        raw_trunk = hidden[:, -1]
        trunk = decoder_rmsnorm(raw_trunk)
        state_bias = self._state_bias(trunk)
        previous = observed
        caches: list[tuple[Tensor, Tensor] | None] = [None] * len(self.blocks)
        frames: list[Tensor] = []
        for depth, offset in enumerate(offsets):
            state, caches = self._decode_step(previous, offset, state_bias, caches)
            embedded: dict[str, Tensor] = {}
            picks: dict[str, Tensor] = {}
            for name in CONTROLLER_DECODE_ORDER:
                logits = self._center(
                    self.outputs[name](self.group_features(state, name, embedded))
                    + self.trunk_outputs[name](raw_trunk)
                )
                if name == "buttons":
                    logits = logits.masked_fill(self.codec.button_mask(picks["triggers"]), float("-inf"))
                group = CONTROLLER_GROUP_INDEX[name]
                uniform = None if uniforms is None else uniforms[depth, group]
                pick = sample_categorical(logits, argmax=argmax, uniform=uniform, generator=gen)
                picks[name] = pick
                embedded[name] = self.codec.group_embedding(name, pick)
            indices = torch.stack([picks[name] for name in CONTROLLER_GROUP_NAMES], dim=-1)
            frames.append(indices)
            previous = indices
        return torch.stack(frames, dim=1)

    def rollout_conditioned_logits(self, hidden: Tensor, observed: Tensor) -> tuple[list[dict[str, Tensor]], Tensor]:
        """Offline ancestral diagnostic across every selected offset.

        Unlike :meth:`sample_indices`, this intentionally includes the sparse
        tail.  It is used only by validation to measure exposure gaps and is not
        reachable from the closed-loop inference wrapper.
        """
        raw_trunk = hidden[:, -1]
        trunk = decoder_rmsnorm(raw_trunk)
        state_bias = self._state_bias(trunk)
        previous = observed
        caches: list[tuple[Tensor, Tensor] | None] = [None] * len(self.blocks)
        frames: list[Tensor] = []
        all_logits: list[dict[str, Tensor]] = []
        for offset in self.head_offsets:
            state, caches = self._decode_step(previous, offset, state_bias, caches)
            embedded: dict[str, Tensor] = {}
            picks: dict[str, Tensor] = {}
            frame_logits: dict[str, Tensor] = {}
            for name in CONTROLLER_DECODE_ORDER:
                logits = self._center(
                    self.outputs[name](self.group_features(state, name, embedded))
                    + self.trunk_outputs[name](raw_trunk)
                )
                if name == "buttons":
                    logits = logits.masked_fill(self.codec.button_mask(picks["triggers"]), float("-inf"))
                pick = logits.argmax(dim=-1)
                frame_logits[name] = logits
                picks[name] = pick
                embedded[name] = self.codec.group_embedding(name, pick)
            previous = torch.stack([picks[name] for name in CONTROLLER_GROUP_NAMES], dim=-1)
            frames.append(previous)
            all_logits.append(frame_logits)
        return all_logits, torch.stack(frames, dim=1)

    @staticmethod
    def _center(logits: Tensor) -> Tensor:
        return center_class_logits(logits)


class GPT(nn.Module):
    def __init__(self, cfg: TrainConfig, vocabulary: PlayerVocabulary | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.L_chunk = cfg.arch.sample_chunk_length
        self.head_offsets = tuple(cfg.arch.head_offsets)
        self.codec = DiscreteControllerCodec(cfg.arch.action_embed_dim)
        self.cat_specs = {**CAT_FEATURES, "action": (cfg.arch.action_vocab, cfg.arch.action_state_embed_dim)}
        self.cat_embeds = nn.ModuleDict(
            {name: nn.Embedding(vocab, dim) for name, (vocab, dim) in self.cat_specs.items()}
        )
        self.char_emb = nn.Embedding(cfg.arch.char_vocab, cfg.arch.char_dim)
        self.stage_emb = nn.Embedding(cfg.arch.stage_vocab, cfg.arch.stage_dim)
        per_player = len(FLOAT_FEATURES) * 2 + sum(dim for _, dim in self.cat_specs.values())
        d_in = (
            len(BASE_PLAYER_PREFIXES) * per_player
            + CONTROLLER_GROUP_COUNT * cfg.arch.action_embed_dim
            + 2 * cfg.arch.char_dim
            + cfg.arch.stage_dim
        )
        self.item_type_emb = nn.Embedding(ITEM_CAT_VOCABS["type"], cfg.arch.item_type_dim)
        self.item_state_emb = nn.Embedding(ITEM_CAT_VOCABS["state"], cfg.arch.item_state_dim)
        slot_width = cfg.arch.item_type_dim + cfg.arch.item_state_dim + 2 * len(ITEM_FLOATS) + 1
        self.item_encoder = SwiGLU(slot_width, cfg.arch.item_hidden_dim, cfg.arch.item_dim)
        d_in += cfg.arch.item_dim
        self.observation_encoder = nn.Linear(d_in, cfg.arch.d_model)
        self.player_embedding = nn.Embedding(
            cfg.player_vocab_size,
            cfg.arch.player_embed_dim,
            padding_idx=MASKED_PLAYER_ID,
        )
        self.player_projection = nn.Linear(cfg.arch.player_embed_dim, cfg.arch.d_model, bias=False)
        code_payload = b"" if vocabulary is None else vocabulary_buffer(vocabulary)
        if vocabulary is not None and (
            vocabulary.size != cfg.player_vocab_size or vocabulary.sha256 != cfg.player_vocab_sha256
        ):
            raise ValueError("identity vocabulary does not match the frozen O52 contract")
        self.register_buffer("player_code_bytes", torch.from_numpy(np.frombuffer(code_payload, dtype=np.uint8).copy()))
        trunk_rule = depth_rule("trunk", cfg.arch.n_layers, cfg.depth_alpha)
        self.trunk = Trunk(
            TrunkConfig(
                d_model=cfg.arch.d_model,
                n_layers=cfg.arch.n_layers,
                n_heads=cfg.arch.n_heads,
                L_ctx=cfg.arch.L_ctx,
                attn_window=cfg.arch.attn_window,
                attention_backend=cfg.arch.trunk_attention_backend,
                attention_scale=trunk_rule.attention,
                mlp_scale=trunk_rule.mlp,
            )
        )
        self.temporal = CausalTemporalDecoder(cfg, self.codec)
        initialize_o51_parameters(self, cfg)

    def _per_player_features(self, features: dict[str, Tensor], prefix: str) -> Tensor:
        ref = features[f"{prefix}_position_x"]
        batch, length = ref.shape
        values: list[Tensor] = []
        masks: list[Tensor] = []
        for name in FLOAT_FEATURES:
            value = features[f"{prefix}_{name}"]
            mask = features.get(f"{prefix}_{name}_mask", torch.zeros_like(ref))
            values.append(value[..., None])
            masks.append(mask[..., None])
        parts: list[Tensor] = values + masks
        for name, (vocab, _) in self.cat_specs.items():
            parts.append(self.cat_embeds[name](features[f"{prefix}_{name}"].clamp(0, vocab - 1)))
        return torch.cat(parts, dim=-1)

    def _item_features(self, features: dict[str, Tensor]) -> Tensor:
        """Pool the four projectile slots into one permutation-invariant vector.

        One shared encoder reads each slot, the presence flag gates its output, and the
        gated outputs are summed. An empty slot therefore contributes the exact zero
        vector, the pooled value does not depend on WHICH slots the live items occupy,
        and the item count stays implicit in the sum.
        """
        if ITEM_PROBE_COLUMN not in features:
            raise ValueError(
                f"the observation carries no {ITEM_PROBE_COLUMN!r} column; training needs policy-world "
                "sources and closed-loop evaluation needs projectile routing"
            )
        zeros = torch.zeros_like(features[ITEM_PROBE_COLUMN])
        slots: list[Tensor] = []
        presence: list[Tensor] = []
        for slot in range(ITEM_SLOTS):
            # The stored type is peppi's raw u16 item id, so the clamp lands every id at
            # or above the last row on that row, which is the unknown projectile.
            type_ids = features[item_column(slot, "type")].clamp(0, self.item_type_emb.num_embeddings - 1)
            state_ids = features[item_column(slot, "state")].clamp(0, self.item_state_emb.num_embeddings - 1)
            masks = {name: features.get(f"{item_column(slot, name)}_mask", zeros) for name in ITEM_FLOATS}
            live = 1.0 - masks[ITEM_PRESENCE_SUFFIX]
            parts = [self.item_type_emb(type_ids), self.item_state_emb(state_ids)]
            parts += [features[item_column(slot, name)][..., None] for name in ITEM_FLOATS]
            parts += [masks[name][..., None] for name in ITEM_FLOATS]
            parts.append(live[..., None])
            slots.append(torch.cat(parts, dim=-1))
            presence.append(live)
        encoded = self.item_encoder(torch.stack(slots, dim=-2))
        return (encoded * torch.stack(presence, dim=-1)[..., None]).sum(dim=-2)

    def context_tokens(self, features: dict[str, Tensor], action_indices: Tensor | None = None) -> Tensor:
        if "opp_player_id" in features:
            raise ValueError("opponent identity must never enter the model")
        if "ego_player_id" not in features:
            raise KeyError("context is missing ego_player_id")
        if action_indices is None:
            action_indices = self.codec.quantize(stack_actions(features))
        parts = [self._per_player_features(features, prefix) for prefix in BASE_PLAYER_PREFIXES]
        parts.append(self.codec.embed_frame(action_indices))
        parts.append(self.char_emb(features["ego_character"].clamp(0, self.char_emb.num_embeddings - 1)))
        parts.append(self.char_emb(features["opp_character"].clamp(0, self.char_emb.num_embeddings - 1)))
        parts.append(self.stage_emb(features["stage"].clamp(0, self.stage_emb.num_embeddings - 1)))
        parts.append(self._item_features(features))
        observation = self.observation_encoder(torch.cat(parts, dim=-1))
        player_ids = features["ego_player_id"].clamp(0, self.player_embedding.num_embeddings - 1)
        return observation + self.player_projection(self.player_embedding(player_ids))

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


class IdentityMasker:
    """Checkpointable, per-window identity dropout independent of all other RNGs."""

    def __init__(self, seed: int, probability: float) -> None:
        self.probability = probability
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.forced = 0
        self.total = 0
        self.masked = 0

    def __call__(self, batch: TrainBatch) -> TrainBatch:
        features = dict(batch.context.features)
        player_id = features["ego_player_id"]
        drop = torch.rand(player_id.shape[0], generator=self.generator) < self.probability
        naturally_masked = player_id[:, 0] == MASKED_PLAYER_ID
        features["ego_player_id"] = player_id.masked_fill(drop[:, None], MASKED_PLAYER_ID)
        self.forced += int(drop.sum())
        self.masked += int((drop | naturally_masked).sum())
        self.total += len(drop)
        train_batch = TrainBatch(
            context=Context(features=features, ctx_pad=batch.context.ctx_pad),
            target=batch.target,
            replay_ids=batch.replay_ids,
        )
        return train_batch

    def state_dict(self) -> dict[str, object]:
        return {
            "generator": self.generator.get_state(),
            "forced": self.forced,
            "masked": self.masked,
            "total": self.total,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        generator_state = state["generator"]
        if not isinstance(generator_state, Tensor):
            raise TypeError("identity-mask generator state must be a tensor")
        self.generator.set_state(generator_state.detach().cpu())
        counts = (state["forced"], state["masked"], state["total"])
        if any(not isinstance(value, int) or isinstance(value, bool) for value in counts):
            raise TypeError("identity-mask counts must be integers")
        self.forced, self.masked, self.total = cast(tuple[int, int, int], counts)

    def metrics(self) -> dict[str, float]:
        denominator = max(self.total, 1)
        return {
            "data/id_dropout_fraction": self.forced / denominator,
            "data/id_masked_fraction": self.masked / denominator,
        }


@jaxtyped(typechecker=beartype)
def prepared_targets(
    model: GPT, batch: TrainBatch
) -> tuple[
    Int[Tensor, "B L_ctx n_groups"],
    Int[Tensor, "B L_ctx n_offsets n_groups"],
    Bool[Tensor, "B L_ctx"],
]:
    """Quantize history+future exactly once, then align every selected offset."""
    history = stack_actions(batch.context.features)
    full = model.codec.quantize(torch.cat((history, batch.target[:, : model.L_chunk]), dim=1))
    length = history.shape[1]
    targets = torch.stack([full[:, offset : offset + length] for offset in model.head_offsets], dim=2)
    valid = torch.arange(length, device=full.device)[None, :] >= batch.context.ctx_pad[:, None]
    suffix = slice(length // 2, None)
    return full[:, :length][:, suffix], targets[:, suffix], valid[:, suffix]


class DeviceBatchPrefetcher:
    """Keep a bounded serial CPU lookahead and stage one device batch."""

    def __init__(
        self,
        loader: Iterable[TrainBatch],
        cfg: TrainConfig,
        device: str | torch.device,
        identity_masker: IdentityMasker | None = None,
        *,
        iterator: Iterator[TrainBatch] | None = None,
        first_batch_future: Future[TrainBatch] | None = None,
    ) -> None:
        self._loader: Iterable[TrainBatch] | None = loader
        self._iterator: Iterator[TrainBatch] | None = iter(loader) if iterator is None else iterator
        self._cfg = cfg
        self._device = torch.device(device)
        self._identity_masker = identity_masker
        self._copy_stream = torch.cuda.Stream(device=self._device) if self._device.type == "cuda" else None
        self._staged: tuple[TrainBatch, TrainBatch, int] | None = None
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="device-batch-prefetch")
        self._futures: deque[Future[TrainBatch]] = deque()
        if first_batch_future is None:
            self.fill_lookahead(1)
        else:
            self._futures.append(first_batch_future)
        self.stage_next()

    def _load_cpu_batch(self) -> TrainBatch:
        if self._iterator is None or self._loader is None:
            raise RuntimeError("device batch prefetcher is closed")
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self._loader)
            return next(self._iterator)

    def _prepare_cpu_batch(self, cpu_batch: TrainBatch) -> TrainBatch:
        """Apply the parent-side transforms to an already-fetched batch."""
        if not isinstance(cpu_batch, TrainBatch):
            raise TypeError(f"training loader yielded {type(cpu_batch).__name__}, expected TrainBatch")
        if self._identity_masker is not None:
            cpu_batch = self._identity_masker(cpu_batch)
        validate_batch_geometry(cpu_batch, self._cfg, self._cfg.batch_size)
        return cpu_batch

    def _stage(self, cpu_batch: TrainBatch) -> None:
        start = self._cfg.arch.direct_loss_start
        suffix_pad = (cpu_batch.context.ctx_pad - start).clamp_min(0)
        valid_prefixes = int((self._cfg.arch.L_ctx - start - suffix_pad).sum())
        if valid_prefixes <= 0:
            raise RuntimeError("training batch contains no valid context prefixes")
        if self._copy_stream is None:
            device_batch = cpu_batch.to(self._device)
        else:
            with torch.cuda.stream(self._copy_stream):
                device_batch = cpu_batch.to(self._device)
        self._staged = (device_batch, cpu_batch, valid_prefixes)

    def fill_lookahead(self, batch_limit: int) -> None:
        """Submit up to four batches without crossing the next state boundary."""
        if batch_limit < 0:
            raise ValueError("lookahead batch limit must be non-negative")
        if self._staged is not None:
            raise RuntimeError("consume the staged batch before filling lookahead")
        if len(self._futures) > batch_limit:
            raise RuntimeError("queued batches cross the next state boundary")
        target = min(self._cfg.train_prefetch_factor, batch_limit)
        while len(self._futures) < target:
            self._futures.append(self._pool.submit(self._load_cpu_batch))

    def stage_next(self) -> float:
        """Stage the oldest queued batch and return its uncovered wait."""
        if self._staged is not None:
            raise RuntimeError("consume the staged batch before staging another")
        if not self._futures:
            raise RuntimeError("fill lookahead before staging another batch")
        started = time.monotonic()
        future = self._futures.popleft()
        cpu_batch = self._prepare_cpu_batch(future.result())
        self._stage(cpu_batch)
        return time.monotonic() - started

    def next(self) -> tuple[TrainBatch, int]:
        """Wait only for the uncovered tail of the staged transfer."""
        if self._staged is None:
            raise RuntimeError("preload a batch before consuming it")
        device_batch, cpu_batch, valid_prefixes = self._staged
        if self._copy_stream is not None:
            compute_stream = torch.cuda.current_stream(self._device)
            compute_stream.wait_stream(self._copy_stream)
            device_batch.record_stream(compute_stream)
        self._staged = None
        del cpu_batch
        return device_batch, valid_prefixes

    @property
    def queue_depth(self) -> int:
        """Return submitted CPU batches for legacy telemetry consumers."""
        return len(self._futures)

    @property
    def submitted_batches(self) -> int:
        return len(self._futures)

    @property
    def ready_batches(self) -> int:
        return sum(future.done() and not future.cancelled() and future.exception() is None for future in self._futures)

    @property
    def drained(self) -> bool:
        return self._staged is None and not self._futures

    def close(self) -> None:
        """Release the background loader thread and all phase-owned batches."""
        self._pool.shutdown(wait=True, cancel_futures=True)
        self._futures.clear()
        self._staged = None
        self._iterator = None
        self._loader = None


class _UpdateTimer:
    """Measure update work without checkpoint or logging gaps."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._started: float | None = None
        self._durations: list[float] = []

    def start(self) -> None:
        if self._started is not None:
            raise RuntimeError("an update timer is already active")
        self._started = self._clock()

    def finish(self) -> None:
        if self._started is None:
            raise RuntimeError("no update timer is active")
        duration = self._clock() - self._started
        if duration < 0:
            raise RuntimeError("update clock moved backwards")
        self._durations.append(duration)
        self._started = None

    def stats_and_reset(self, expected_updates: int) -> tuple[float, float]:
        if self._started is not None:
            raise RuntimeError("cannot flush an active update timer")
        if expected_updates < 1 or len(self._durations) != expected_updates:
            raise RuntimeError(f"update timer contains {len(self._durations)} updates, expected {expected_updates}")
        mean = sum(self._durations) / expected_updates
        p95 = float(np.percentile(self._durations, 95))
        self._durations.clear()
        return mean, p95


def offset_weights(offsets: tuple[int, ...], prediction_frames: int) -> tuple[float, ...]:
    """Give autoregressive offsets full weight and diagnostic offsets half weight."""
    if prediction_frames < 1:
        raise ValueError("prediction_frames must be positive")
    return tuple(1.0 if offset <= prediction_frames else 0.5 for offset in offsets)


@jaxtyped(typechecker=beartype)
def temporal_objective(
    nll: Float[Tensor, "*prefix n_offsets n_groups"],
    *,
    offsets: tuple[int, ...],
    prediction_frames: int,
    valid_prefixes: int,
    valid: Bool[Tensor, "*prefix"],
) -> Float[Tensor, ""]:
    """Return the offset-weighted behavior-cloning objective."""
    weights = nll.new_tensor(offset_weights(offsets, prediction_frames), dtype=torch.float32)
    joint_nll = torch.where(valid[..., None], nll.float().sum(dim=-1), 0)
    return (joint_nll * weights).sum() / (valid_prefixes * weights.sum())


def microbatch_loss(
    model: GPT,
    batch: TrainBatch,
    cfg: TrainConfig,
    *,
    step: int,
    valid_prefixes: int,
    trunk_fn: Callable,
    temporal_fn: Callable,
    phase_timer: CudaPhaseTimer | None = None,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Compute the pure behavior-cloning loss."""
    del step
    history, targets, valid = prepared_targets(model, batch)
    if phase_timer is not None:
        phase_timer.record("target_prep_end")
    with amp_context(cfg, DEVICE):
        hidden = trunk_fn(batch.context.features, batch.context.ctx_pad, None)
        if phase_timer is not None:
            phase_timer.record("trunk_end")
        hidden = hidden[:, cfg.arch.direct_loss_start :]
        temporal_output = temporal_fn(hidden, history, targets)
        if isinstance(temporal_output, Tensor):
            dense_nll = temporal_output
            button_diagnostics: dict[str, Tensor] = {}
        else:
            dense_nll, button_diagnostics = temporal_output
        if phase_timer is not None:
            phase_timer.record("temporal_end")
    loss = temporal_objective(
        dense_nll,
        offsets=cfg.arch.head_offsets,
        prediction_frames=cfg.prediction_frames,
        valid_prefixes=valid_prefixes,
        valid=valid,
    )
    nll_sum = torch.where(valid[..., None, None], dense_nll.float(), 0).sum(dim=(0, 1))
    extra = {
        "train/loss": scoring.nats_to_bits(loss.detach()),
        "train/objective": loss.detach(),
        **button_diagnostics,
    }
    if phase_timer is not None:
        phase_timer.record("objective_end")
    return loss, nll_sum.detach(), extra


def nll_mean_metrics(
    mean_nll: Tensor,
    offsets: tuple[int, ...],
    *,
    prediction_frames: int,
) -> dict[str, float]:
    if mean_nll.shape != (len(offsets), CONTROLLER_GROUP_COUNT):
        raise ValueError(f"mean NLL has shape {tuple(mean_nll.shape)}")
    joint = scoring.nats_to_bits(mean_nll.sum(dim=-1))
    weights = joint.new_tensor(offset_weights(offsets, prediction_frames))
    total = (joint * weights).sum() / weights.sum()
    out = {
        "loss_unweighted": float(joint.mean()),
        "loss_weighted": float(total),
    }
    for depth, offset in enumerate(offsets):
        out[f"nll_o{offset:02d}"] = float(joint[depth])
        for group, name in enumerate(CONTROLLER_GROUP_NAMES):
            out[f"nll_o{offset:02d}_{name}"] = float(scoring.nats_to_bits(mean_nll[depth, group]))
    return out


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
    nll_sum = torch.zeros(len(model.head_offsets), CONTROLLER_GROUP_COUNT, dtype=torch.float64)
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
                hidden = model.forward_dense(batch.context.features, batch.context.ctx_pad, None)
                hidden = hidden[:, cfg.arch.direct_loss_start :]
                logits = model.temporal.teacher_forced_logits_by_group(hidden, history, targets)
                dense_nll = model.temporal.nll_from_logits(logits, targets)
            row_valid = batch.context.ctx_pad < cfg.arch.L_ctx
            if not bool(row_valid.any()):
                continue
            selected_nll = dense_nll[:, -1][row_valid]
            nll_sum += selected_nll.double().sum(dim=0).cpu()
            count += selected_nll.shape[0]
            target_last = targets[:, -1][row_valid]
            for group, name in enumerate(CONTROLLER_GROUP_NAMES):
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
            sampled = sampled_all[:, : cfg.prediction_frames]
            target_rows.append(target_last.cpu())
            sampled_rows.append(sampled.cpu())
            observed_rows.append(last_observed.cpu())
            for depth in range(len(model.head_offsets)):
                compatible = model.codec.button_valid_for_trigger[
                    sampled_all[:, depth, TRIGGERS_GROUP], target_last[:, depth, BUTTONS_GROUP]
                ]
                button_incompatible[depth] += float((~compatible).sum())
                for group, name in enumerate(CONTROLLER_GROUP_NAMES):
                    expected = target_last[:, depth, group]
                    step_logits = rollout_logits[depth][name]
                    rollout_correct[depth, group] += (step_logits.argmax(-1) == expected).double().sum().cpu()
                    selected = compatible if group == BUTTONS_GROUP else torch.ones_like(compatible)
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
                    ((raw[..., BUTTON_LEFT_CHANNEL] > 0.5) & (raw[..., TRIGGER_LEFT_CHANNEL] < 1.0))
                    | ((raw[..., BUTTON_RIGHT_CHANNEL] > 0.5) & (raw[..., TRIGGER_RIGHT_CHANNEL] < 1.0))
                ).sum()
            )
    finally:
        model.train(was_training)
    if count == 0:
        raise RuntimeError("validation contained no valid prefixes")
    out = nll_mean_metrics(
        nll_sum / count,
        model.head_offsets,
        prediction_frames=cfg.prediction_frames,
    )
    for depth, offset in enumerate(model.head_offsets):
        for group, name in enumerate(CONTROLLER_GROUP_NAMES):
            out[f"acc_o{offset:02d}_{name}"] = float(correct[depth, group] / count)
            denominator = float(exposure_count[depth, group])
            if denominator <= 0:
                raise RuntimeError(f"validation has no compatible rollout rows for offset {offset} group {name}")
            roll_nll = float(scoring.nats_to_bits(rollout_nll[depth, group] / denominator))
            teacher_nll = float(scoring.nats_to_bits(teacher_exposure_nll[depth, group] / denominator))
            out[f"rollout_nll_o{offset:02d}_{name}"] = roll_nll
            out[f"exposure_gap_o{offset:02d}_{name}"] = roll_nll - teacher_nll
            out[f"rollout_acc_o{offset:02d}_{name}"] = float(rollout_correct[depth, group] / count)
        out[f"rollout_button_target_masked_rate_o{offset:02d}"] = float(button_incompatible[depth] / count)
    target = torch.cat(target_rows)
    sampled = torch.cat(sampled_rows)
    observed = torch.cat(observed_rows)
    dense_target = target[:, : cfg.prediction_frames]
    matches = sampled == dense_target
    out["exact_frame_acc"] = float(matches.all(dim=-1).float().mean())
    out["dense_four_sequence_acc"] = float(matches[:, :4].all(dim=-1).all(dim=-1).float().mean())
    out.update(_transition_metrics(dense_target, sampled, observed))
    out["action_quantization_mse"] = quantization_squared / max(quantization_count, 1)
    out["invalid_trigger_count_raw"] = invalid_triggers
    out["invalid_trigger_count_sampled"] = float(
        (~model.codec.button_valid_for_trigger[sampled[..., TRIGGERS_GROUP], sampled[..., BUTTONS_GROUP]]).sum()
    )
    nonfinite = {name: value for name, value in out.items() if not math.isfinite(value)}
    if nonfinite:
        raise FloatingPointError(f"validation produced non-finite metrics: {nonfinite}")
    return out


def _validation_wandb_metrics(values: dict[str, float], cfg: TrainConfig) -> dict[str, float]:
    """Reduce detailed validation evidence to orthogonal W&B signals."""
    horizon = cfg.prediction_frames
    rollout_nll = sum(values[f"rollout_nll_o{horizon:02d}_{name}"] for name in CONTROLLER_GROUP_NAMES)
    exposure_gap = sum(values[f"exposure_gap_o{horizon:02d}_{name}"] for name in CONTROLLER_GROUP_NAMES)
    return {
        "nll": values["loss_weighted"],
        "nll_unweighted": values["loss_unweighted"],
        "rollout_nll": rollout_nll,
        "exposure_gap": exposure_gap,
        "exact_frame_acc": values["exact_frame_acc"],
        "sequence_acc": values["dense_four_sequence_acc"],
        "change_f1": values["change_f1"],
        "sampled_transition_rate": values["sampled_transition_rate"],
    }


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


def _condition_ego_player(ctx: Context, player_id: int) -> Context:
    """Attach one runtime ego identity without specializing the compiled graph."""
    if "opp_player_id" in ctx.features:
        raise ValueError("opponent identity must never enter inference")
    if not isinstance(player_id, int) or isinstance(player_id, bool) or player_id < 0:
        raise ValueError(f"player_id must be a non-negative integer, got {player_id!r}")
    reference = ctx.features[next(iter(ctx.features))]
    features = dict(ctx.features)
    features["ego_player_id"] = torch.full(
        reference.shape[:2],
        player_id,
        dtype=torch.long,
        device=reference.device,
    )
    return replace(ctx, features=features)


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
        parameter = next(model.parameters())
        if parameter.device.type == "cuda" and parameter.dtype != torch.bfloat16:
            raise ValueError("O54 CUDA inference requires BF16 model parameters")
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
            return next(bucket for bucket in self.cfg.inference_buckets if bucket >= rows)
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
        streams: SlotGroupRng | None = None,
        argmax: bool = False,
        gen: torch.Generator | None = None,
    ) -> Tensor:
        if horizon != self.cfg.prediction_frames:
            raise ValueError(f"horizon must be {self.cfg.prediction_frames}")
        rows = ctx.ctx_pad.shape[0]
        bucket = self._bucket(rows)
        padded = _pad_context(ctx, bucket)
        if "ego_player_id" not in padded.features:
            padded = _condition_ego_player(padded, MASKED_PLAYER_ID)
        padded = canonical_context(padded, "base", items=True)
        observed = self.model.codec.quantize(stack_actions(padded.features))
        uniform_parts: list[Tensor] = []
        if streams is not None:
            streams.begin(ctx)
        for _ in range(horizon):
            groups = []
            for name in CONTROLLER_GROUP_NAMES:
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


@dataclass(frozen=True, slots=True)
class LatencyRow:
    width: int
    inference_delay_frames: int
    replan_interval_frames: int
    prediction_frames: int
    p99_seconds: float
    frame_period_seconds: float
    batch_size: int = LATENCY_BATCH_SIZE
    warmup_calls: int = LATENCY_WARMUP_CALLS
    measured_calls: int = LATENCY_MEASURED_CALLS

    def __post_init__(self) -> None:
        if self.width not in MODEL_WIDTHS:
            raise ValueError(f"unsupported latency width {self.width}")
        if not 1 <= self.inference_delay_frames <= 6:
            raise ValueError("latency delay must be in [1, 6]")
        if (self.replan_interval_frames, self.prediction_frames) != (
            self.inference_delay_frames,
            2 * self.inference_delay_frames,
        ):
            raise ValueError("latency row requires R=d and H=2d")
        if not math.isfinite(self.p99_seconds) or self.p99_seconds < 0:
            raise ValueError("p99 latency must be finite and non-negative")
        if not math.isfinite(self.frame_period_seconds) or self.frame_period_seconds <= 0:
            raise ValueError("frame period must be finite and positive")
        if self.batch_size < 1 or self.warmup_calls < 0 or self.measured_calls < 1:
            raise ValueError("latency measurement geometry is invalid")

    @property
    def timing(self) -> TimingConfig:
        return TimingConfig(
            self.inference_delay_frames,
            self.replan_interval_frames,
            self.prediction_frames,
        )

    @property
    def meets_deadline(self) -> bool:
        return self.p99_seconds < self.inference_delay_frames * self.frame_period_seconds


@dataclass(frozen=True, slots=True)
class LatencyProbe:
    width: int
    inference_delay_frames: int
    replan_interval_frames: int
    prediction_frames: int
    status: Literal["measured", "oom"]
    samples_seconds: tuple[float, ...]
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    frame_period_seconds: float = 1 / 60
    batch_size: int = LATENCY_BATCH_SIZE
    warmup_calls: int = LATENCY_WARMUP_CALLS
    measured_calls: int = LATENCY_MEASURED_CALLS
    trial_index: int = 0
    phase: Literal["search", "validation", "diagnostic"] = "search"

    def __post_init__(self) -> None:
        TimingConfig(
            self.inference_delay_frames,
            self.replan_interval_frames,
            self.prediction_frames,
        )
        if self.status == "measured":
            if len(self.samples_seconds) != self.measured_calls:
                raise ValueError("measured latency probe has the wrong sample count")
            if any(not math.isfinite(sample) or sample < 0 for sample in self.samples_seconds):
                raise ValueError("latency samples must be finite and non-negative")
        elif self.samples_seconds:
            raise ValueError("OOM latency probe cannot contain timing samples")
        if self.peak_allocated_bytes < 0 or self.peak_reserved_bytes < self.peak_allocated_bytes:
            raise ValueError("latency probe has invalid CUDA memory peaks")
        if not 0 <= self.trial_index < LATENCY_TRIALS:
            raise ValueError(f"latency trial index must be in [0, {LATENCY_TRIALS})")

    @property
    def deadline_seconds(self) -> float:
        return self.inference_delay_frames * self.frame_period_seconds

    def percentile_seconds(self, percentile: float) -> float | None:
        if not self.samples_seconds:
            return None
        return float(np.percentile(self.samples_seconds, percentile))

    @property
    def meets_deadline(self) -> bool:
        p99 = self.percentile_seconds(99)
        return p99 is not None and p99 < self.deadline_seconds


class LatencyProbeFailure(RuntimeError):
    """One width cannot satisfy the measured deployment contract."""


@dataclass(frozen=True, slots=True)
class DolphinLoadEvidence:
    executable_name: str
    executable_sha256: str
    graphics_backend: Literal["Vulkan"] = "Vulkan"
    internal_resolution_scale: int = 2
    emulation_speed: float = 0.0
    frame_pacing_hz: int = 60
    sessions: int = 1

    def __post_init__(self) -> None:
        if not self.executable_name or re.fullmatch(r"[0-9a-f]{64}", self.executable_sha256) is None:
            raise ValueError("Dolphin load evidence has no valid executable identity")
        if (
            self.graphics_backend != "Vulkan"
            or self.internal_resolution_scale != 2
            or self.emulation_speed != 0.0
            or self.frame_pacing_hz != 60
            or self.sessions != 1
        ):
            raise ValueError("latency preflight requires one real-time native-resolution Vulkan Dolphin session")


@dataclass(frozen=True, slots=True)
class LatencyTrialRow:
    trial_index: int
    width_order_index: int
    row: LatencyRow

    def __post_init__(self) -> None:
        if not 0 <= self.trial_index < LATENCY_TRIALS:
            raise ValueError(f"latency trial index must be in [0, {LATENCY_TRIALS})")
        if not 0 <= self.width_order_index < len(MODEL_WIDTHS):
            raise ValueError("latency width-order index is invalid")


class _DolphinLoad:
    def __init__(self, session: Session, evidence: DolphinLoadEvidence) -> None:
        self.session = session
        self.evidence = evidence
        self.stop = threading.Event()
        self.failure: Exception | None = None
        self.thread = threading.Thread(target=self._drive, name="o54-dolphin-load", daemon=True)

    def _drive(self) -> None:
        try:
            next_frame = time.perf_counter()
            while not self.stop.is_set():
                _frame, live = self.session.step({1: NEUTRAL_CONTROLLER_ACTION})
                if not live:
                    raise RuntimeError("Dolphin load session left active gameplay")
                next_frame += 1 / self.evidence.frame_pacing_hz
                self.stop.wait(max(0.0, next_frame - time.perf_counter()))
        except Exception as error:
            self.failure = error

    def raise_if_failed(self) -> None:
        if self.failure is not None:
            raise RuntimeError("production Dolphin load failed during the latency preflight") from self.failure


@contextlib.contextmanager
def production_dolphin_load() -> Iterator[_DolphinLoad]:
    """Keep one production-renderer Dolphin match active during timing."""
    executable = Path(NETPLAY_EMULATOR_PATH)
    iso = Path(ISO_PATH)
    if not executable.is_file() or not iso.is_file():
        raise RuntimeError(f"latency preflight needs Dolphin at {executable} and the game image at {iso}")
    with executable.open("rb") as source:
        executable_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    dolphin_version = tested_dolphin_version(str(executable))
    evidence = DolphinLoadEvidence(executable.name, executable_sha256)
    matchup = Matchup(
        stage=melee.Stage.FINAL_DESTINATION,
        players=(
            PlayerSetup(1, melee.Character.FOX),
            PlayerSetup(
                2,
                melee.Character.FOX,
                cpu_level=9,
            ),
        ),
    )
    with Session(
        iso,
        dolphin_path=executable,
        blocking_input=True,
        emulation_speed=0.0,
        use_exi_inputs=False,
        enable_ffw=False,
        disable_audio=True,
        polling_mode=True,
        instant_match_restart=True,
        gfx_backend="Vulkan",
        internal_resolution_scale=2,
        dolphin_version=dolphin_version,
    ) as session:
        session.start_match(matchup)
        load = _DolphinLoad(session, evidence)
        load.thread.start()
        try:
            yield load
            load.raise_if_failed()
        finally:
            load.stop.set()
            load.thread.join(timeout=10)
            if load.thread.is_alive():
                raise RuntimeError("production Dolphin load did not stop")


def _architecture_family(widths: Sequence[int]) -> list[dict[str, object]]:
    family: list[dict[str, object]] = []
    for width in widths:
        values: dict[str, object] = asdict(Architecture.for_width(width))
        values["head_offsets"] = list(cast(tuple[int, ...], values["head_offsets"]))
        family.append(values)
    return family


def _latency_architecture_family() -> list[dict[str, object]]:
    return _architecture_family(WIDTHS)


def _latency_manifest_payload(
    rows: Sequence[LatencyRow],
    trial_rows: Sequence[LatencyTrialRow],
    gpu_name: str,
    raw_probe_sha256: str,
    dolphin: DolphinLoadEvidence,
) -> dict[str, object]:
    return {
        "schema": 3,
        "experiment_id": _LATENCY_EXPERIMENT_ID,
        "gpu_name": gpu_name,
        "inference_mode": "eager",
        "raw_probe_sha256": raw_probe_sha256,
        "dolphin_load": asdict(dolphin),
        "architecture_family": _latency_architecture_family(),
        "rows": [asdict(row) for row in rows],
        "trial_rows": [asdict(row) for row in trial_rows],
    }


def _is_official_latency_row(row: LatencyRow) -> bool:
    return (row.batch_size, row.warmup_calls, row.measured_calls) == (
        LATENCY_BATCH_SIZE,
        LATENCY_WARMUP_CALLS,
        LATENCY_MEASURED_CALLS,
    )


def write_latency_manifest(
    path: Path,
    rows: Sequence[LatencyRow],
    trial_rows: Sequence[LatencyTrialRow],
    gpu_name: str,
    raw_probe_sha256: str,
    dolphin: DolphinLoadEvidence,
) -> str:
    if "RTX 3060" not in gpu_name:
        raise ValueError(f"latency manifest requires the target RTX 3060, got {gpu_name!r}")
    if tuple(row.width for row in rows) != WIDTHS:
        raise ValueError(f"latency manifest must cover widths {WIDTHS} in order")
    if any(not row.meets_deadline for row in rows):
        raise ValueError("latency manifest contains a row that misses its deadline")
    if any(not _is_official_latency_row(row) for row in rows):
        raise ValueError("latency manifest requires eager B1 with 50 warm-ups and 500 measurements")
    expected_trials = {(trial, width) for trial in range(LATENCY_TRIALS) for width in WIDTHS}
    actual_trials = {(trial.trial_index, trial.row.width) for trial in trial_rows}
    selected_delays = {row.width: row.inference_delay_frames for row in rows}
    if (
        len(trial_rows) != len(expected_trials)
        or actual_trials != expected_trials
        or any(not trial.row.meets_deadline or not _is_official_latency_row(trial.row) for trial in trial_rows)
        or any(trial.row.inference_delay_frames != selected_delays[trial.row.width] for trial in trial_rows)
    ):
        raise ValueError("latency manifest requires one passing B1 trial at every selected timing")
    if re.fullmatch(r"[0-9a-f]{64}", raw_probe_sha256) is None:
        raise ValueError("latency manifest has no valid raw-probe SHA-256")
    payload = _latency_manifest_payload(rows, trial_rows, gpu_name, raw_probe_sha256, dolphin)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    document = {**payload, "sha256": digest}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)
    return digest


def load_latency_manifest(path: Path) -> tuple[dict[int, LatencyRow], str]:
    document = json.loads(path.read_text())
    digest = document.pop("sha256", None)
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    actual = hashlib.sha256(encoded).hexdigest()
    if digest != actual:
        raise ValueError(f"latency manifest SHA-256 mismatch: {digest!r} != {actual}")
    if (
        document.get("schema") != 3
        or document.get("experiment_id") != _LATENCY_EXPERIMENT_ID
        or document.get("inference_mode") != "eager"
        or document.get("architecture_family") != _latency_architecture_family()
    ):
        raise ValueError("unsupported latency manifest")
    if "RTX 3060" not in str(document.get("gpu_name", "")):
        raise ValueError("latency manifest was not measured on the target RTX 3060")
    raw_rows = document.get("rows")
    raw_trial_rows = document.get("trial_rows")
    raw_probe_sha256 = document.get("raw_probe_sha256")
    raw_dolphin = document.get("dolphin_load")
    if (
        not isinstance(raw_rows, list)
        or not isinstance(raw_trial_rows, list)
        or not isinstance(raw_probe_sha256, str)
        or not isinstance(raw_dolphin, dict)
    ):
        raise ValueError("latency manifest has no rows")
    rows = tuple(LatencyRow(**row) for row in raw_rows)
    trial_rows = tuple(
        LatencyTrialRow(
            trial_index=row["trial_index"],
            width_order_index=row["width_order_index"],
            row=LatencyRow(**row["row"]),
        )
        for row in raw_trial_rows
    )
    dolphin = DolphinLoadEvidence(**raw_dolphin)
    if (
        tuple(row.width for row in rows) != WIDTHS
        or any(not row.meets_deadline for row in rows)
        or any(not _is_official_latency_row(row) for row in rows)
    ):
        raise ValueError("latency manifest does not contain the accepted width matrix")
    expected_payload = _latency_manifest_payload(rows, trial_rows, document["gpu_name"], raw_probe_sha256, dolphin)
    if document != expected_payload:
        raise ValueError("latency manifest contains unsupported fields")
    expected_trials = {(trial, width) for trial in range(LATENCY_TRIALS) for width in WIDTHS}
    if (
        {(trial.trial_index, trial.row.width) for trial in trial_rows} != expected_trials
        or len(trial_rows) != len(expected_trials)
        or any(not trial.row.meets_deadline for trial in trial_rows)
        or any(
            trial.row.inference_delay_frames != rows[WIDTHS.index(trial.row.width)].inference_delay_frames
            for trial in trial_rows
        )
    ):
        raise ValueError("latency manifest does not contain one accepted trial")
    probe_path = path.with_suffix(".probes.json")
    probe_document, probe_sha256 = load_latency_probe_report(probe_path)
    if probe_sha256 != raw_probe_sha256:
        raise ValueError("latency manifest raw-probe SHA-256 does not match its probe artifact")
    if probe_document.get("gpu_name") != document["gpu_name"] or probe_document.get("dolphin_load") != asdict(dolphin):
        raise ValueError("latency manifest and raw probes used different hardware load")
    raw_orders = probe_document.get("width_orders")
    raw_probes = probe_document.get("probes")
    if (
        not isinstance(raw_orders, list)
        or not all(isinstance(order, list) for order in raw_orders)
        or not isinstance(raw_probes, list)
        or not all(isinstance(probe, dict) for probe in raw_probes)
    ):
        raise ValueError("latency probe artifact has no trial evidence")
    typed_orders = cast(list[list[object]], raw_orders)
    if tuple(tuple(order) for order in typed_orders) != tuple(
        tuple(
            trial.row.width
            for trial in sorted(trial_rows, key=lambda item: item.width_order_index)
            if trial.trial_index == index
        )
        for index in range(LATENCY_TRIALS)
    ):
        raise ValueError("latency trial order differs from the raw probe artifact")
    probes = tuple(LatencyProbe(**probe) for probe in cast(list[dict], raw_probes))
    for trial in trial_rows:
        candidates = [
            probe
            for probe in probes
            if probe.trial_index == trial.trial_index
            and probe.width == trial.row.width
            and probe.inference_delay_frames == trial.row.inference_delay_frames
            and probe.status == "measured"
        ]
        if len(candidates) != 1 or candidates[0].percentile_seconds(99) != trial.row.p99_seconds:
            raise ValueError("accepted latency trial is not present in the raw probe artifact")
    for row in rows:
        if row.p99_seconds != max(trial.row.p99_seconds for trial in trial_rows if trial.row.width == row.width):
            raise ValueError("latency manifest p99 is not the worst accepted trial")
    return {row.width: row for row in rows}, actual


def write_latency_probe_report(
    path: Path,
    probes: Sequence[LatencyProbe],
    gpu_name: str,
    *,
    width_orders: Sequence[Sequence[int]] = (),
    dolphin: DolphinLoadEvidence | None = None,
) -> str:
    """Atomically persist every raw latency sample, including partial runs."""
    payload = {
        "schema": 3,
        "experiment_id": _LATENCY_EXPERIMENT_ID,
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "architecture_family": _latency_architecture_family(),
        "latency_order_seed": LATENCY_ORDER_SEED,
        "width_orders": [list(order) for order in width_orders],
        "dolphin_load": None if dolphin is None else asdict(dolphin),
        "probes": [asdict(probe) for probe in probes],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    document = {**payload, "sha256": digest}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)
    return digest


def load_latency_probe_report(path: Path) -> tuple[dict[str, object], str]:
    document = json.loads(path.read_text())
    if not isinstance(document, dict):
        raise ValueError("latency probe artifact must be a JSON object")
    digest = document.pop("sha256", None)
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    actual = hashlib.sha256(encoded).hexdigest()
    if digest != actual:
        raise ValueError(f"latency probe SHA-256 mismatch: {digest!r} != {actual}")
    if (
        document.get("schema") != 3
        or document.get("experiment_id") != _LATENCY_EXPERIMENT_ID
        or document.get("architecture_family") != _latency_architecture_family()
    ):
        raise ValueError("unsupported latency probe artifact")
    return document, actual


def _log_latency_probe(probe: LatencyProbe) -> None:
    allocated_gib = probe.peak_allocated_bytes / 2**30
    reserved_gib = probe.peak_reserved_bytes / 2**30
    prefix = (
        f"[latency] W{probe.width} L{TRUNK_DEPTHS[probe.width]} d={probe.inference_delay_frames} "
        f"R={probe.replan_interval_frames} H={probe.prediction_frames} B={probe.batch_size}"
    )
    if probe.status == "oom":
        print(
            f"{prefix} OOM peak_allocated={allocated_gib:.3f} GiB peak_reserved={reserved_gib:.3f} GiB",
            flush=True,
        )
        return
    p50 = probe.percentile_seconds(50)
    p95 = probe.percentile_seconds(95)
    p99 = probe.percentile_seconds(99)
    maximum = max(probe.samples_seconds)
    assert p50 is not None and p95 is not None and p99 is not None
    margin_ms = (probe.deadline_seconds - p99) * 1e3
    result = "PASS" if probe.meets_deadline else "MISS"
    print(
        f"{prefix} p50={p50 * 1e3:.3f} ms p95={p95 * 1e3:.3f} ms "
        f"p99={p99 * 1e3:.3f} ms max={maximum * 1e3:.3f} ms "
        f"deadline={probe.deadline_seconds * 1e3:.3f} ms margin={margin_ms:.3f} ms "
        f"peak_allocated={allocated_gib:.3f} GiB peak_reserved={reserved_gib:.3f} GiB {result}",
        flush=True,
    )


def latency_bucket_frontier(rows: Mapping[int, LatencyRow]) -> tuple[int, ...]:
    """Return the largest benchmarked width in each observed delay bucket."""
    frontier: dict[int, int] = {}
    for width in WIDTHS:
        try:
            row = rows[width]
        except KeyError as error:
            raise ValueError(f"latency rows do not cover width {width}") from error
        frontier[row.inference_delay_frames] = width
    return tuple(frontier[delay] for delay in sorted(frontier))


@dataclass(frozen=True, slots=True)
class StudyEndpoint:
    width: int
    updates: int
    roles: tuple[str, ...]


def study_endpoints(rows: Mapping[int, LatencyRow]) -> tuple[StudyEndpoint, ...]:
    """Return one D=2^31 trajectory for each declared architecture."""
    if set(rows) != set(WIDTHS):
        raise ValueError(f"study latency rows must cover exactly {WIDTHS}")
    return tuple(StudyEndpoint(width, DATA_UPDATES[-1], ("nested-D28-D31",)) for width in WIDTHS)


def study_launch_commands(
    rows: Mapping[int, LatencyRow],
    latency_manifest: Path,
) -> tuple[tuple[str, ...], ...]:
    """Build the exact detached Modal commands without submitting them."""
    commands = []
    for endpoint in study_endpoints(rows):
        role = "nested-d28-d31"
        app_name = f"o54-{role}-w{endpoint.width}"
        commands.append(
            (
                "uv",
                "run",
                "scripts/launch_modal.py",
                "--gpu",
                "B200",
                "--closed-loop-gpu",
                "RTX-PRO-6000",
                "--app-name",
                app_name,
                "--",
                "uv",
                "run",
                "experiments/054_bc_capacity_latency.py",
                "train",
                "--width",
                str(endpoint.width),
                "--updates",
                str(endpoint.updates),
                "--latency-manifest",
                str(latency_manifest),
                "--comment",
                role,
            )
        )
    return tuple(commands)


@torch.no_grad()
def benchmark_width_latency(
    width: int,
    *,
    delay: int | None = None,
    batch_size: int = LATENCY_BATCH_SIZE,
    frame_period_seconds: float = 1 / 60,
    warmup_calls: int = LATENCY_WARMUP_CALLS,
    measured_calls: int = LATENCY_MEASURED_CALLS,
    trial_index: int = 0,
    phase: Literal["search", "validation", "diagnostic"] = "search",
    on_probe: Callable[[LatencyProbe], None] | None = None,
) -> LatencyRow:
    if not torch.cuda.is_available():
        raise RuntimeError("latency preflight requires the target CUDA GPU")
    if batch_size < 1:
        raise ValueError("latency batch size must be positive")
    if delay is not None and not 1 <= delay <= 6:
        raise ValueError("latency delay must be in [1, 6]")
    device = torch.device("cuda")
    delays = range(1, 7) if delay is None else (delay,)
    for candidate_delay in delays:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        cfg = replace(
            config_for_width(width, updates=1),
            timing=TimingConfig(candidate_delay, candidate_delay, 2 * candidate_delay),
            inference_mode="eager",
            compile_trunk=False,
            compile_temporal=False,
        )
        model: GPT | None = None
        inference: BF16Inference | None = None
        context: Context | None = None
        try:
            model = GPT(cfg).to(device=device, dtype=torch.bfloat16).eval()
            actual_counts = subsystem_parameter_counts(model)
            if actual_counts != cfg.arch.parameter_count_contract:
                raise RuntimeError(
                    f"W{width} parameter contract changed: {actual_counts} != {cfg.arch.parameter_count_contract}"
                )
            inference = BF16Inference(model, cfg, compiled=False)
            context = synthetic_context(cfg, batch_size, device)
            generator = torch.Generator(device=device).manual_seed(cfg.eval_seed)
            for _ in range(warmup_calls):
                inference.decode(context, cfg.prediction_frames, gen=generator)
            torch.cuda.synchronize(device)
            samples = np.empty(measured_calls, dtype=np.float64)
            for index in range(measured_calls):
                started = time.perf_counter()
                inference.decode(context, cfg.prediction_frames, gen=generator)
                torch.cuda.synchronize(device)
                samples[index] = time.perf_counter() - started
            probe = LatencyProbe(
                width=width,
                inference_delay_frames=candidate_delay,
                replan_interval_frames=candidate_delay,
                prediction_frames=2 * candidate_delay,
                status="measured",
                samples_seconds=tuple(map(float, samples)),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                frame_period_seconds=frame_period_seconds,
                batch_size=batch_size,
                warmup_calls=warmup_calls,
                measured_calls=measured_calls,
                trial_index=trial_index,
                phase=phase,
            )
        except torch.cuda.OutOfMemoryError as error:
            probe = LatencyProbe(
                width=width,
                inference_delay_frames=candidate_delay,
                replan_interval_frames=candidate_delay,
                prediction_frames=2 * candidate_delay,
                status="oom",
                samples_seconds=(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                frame_period_seconds=frame_period_seconds,
                batch_size=batch_size,
                warmup_calls=warmup_calls,
                measured_calls=measured_calls,
                trial_index=trial_index,
                phase=phase,
            )
            _log_latency_probe(probe)
            if on_probe is not None:
                on_probe(probe)
            raise LatencyProbeFailure(
                f"B{batch_size} width {width} ran out of CUDA memory at d={candidate_delay}"
            ) from error
        finally:
            del inference, model, context
            gc.collect()
            torch.cuda.empty_cache()
        _log_latency_probe(probe)
        if on_probe is not None:
            on_probe(probe)
        p99 = probe.percentile_seconds(99)
        assert p99 is not None
        row = LatencyRow(
            width=width,
            inference_delay_frames=candidate_delay,
            replan_interval_frames=candidate_delay,
            prediction_frames=2 * candidate_delay,
            p99_seconds=p99,
            frame_period_seconds=frame_period_seconds,
            batch_size=batch_size,
            warmup_calls=warmup_calls,
            measured_calls=measured_calls,
        )
        if row.meets_deadline:
            return row
    raise LatencyProbeFailure(f"B{batch_size} width {width} cannot meet the buffered deadline with d <= 6")


def run_latency_preflight(path: Path) -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("latency preflight requires CUDA")
    gpu_name = torch.cuda.get_device_name()
    if "RTX 3060" not in gpu_name:
        raise RuntimeError(f"latency preflight requires the target RTX 3060, found {gpu_name!r}")
    report_path = path.with_suffix(".probes.json")
    probes: list[LatencyProbe] = []
    rng = np.random.default_rng(LATENCY_ORDER_SEED)
    width_orders = tuple(tuple(map(int, rng.permutation(WIDTHS))) for _ in range(LATENCY_TRIALS))

    print(f"[latency] raw probe report: {report_path}", flush=True)
    with production_dolphin_load() as dolphin_load:
        raw_probe_sha256 = write_latency_probe_report(
            report_path,
            probes,
            gpu_name,
            width_orders=width_orders,
            dolphin=dolphin_load.evidence,
        )

        def record(probe: LatencyProbe) -> None:
            nonlocal raw_probe_sha256
            probes.append(probe)
            raw_probe_sha256 = write_latency_probe_report(
                report_path,
                probes,
                gpu_name,
                width_orders=width_orders,
                dolphin=dolphin_load.evidence,
            )

        search_rows: dict[tuple[int, int], LatencyRow] = {}
        failures: list[str] = []
        for trial_index, order in enumerate(width_orders):
            print(f"[latency] trial {trial_index + 1}/{LATENCY_TRIALS} width order: {order}", flush=True)
            for width in order:
                dolphin_load.raise_if_failed()
                try:
                    search_rows[trial_index, width] = benchmark_width_latency(
                        width,
                        trial_index=trial_index,
                        phase="search",
                        on_probe=record,
                    )
                except LatencyProbeFailure as error:
                    failures.append(f"trial {trial_index + 1}: {error}")
        if failures:
            raise LatencyProbeFailure(f"latency preflight rejected trials: {'; '.join(failures)}")

        selected_delays = {
            width: max(search_rows[trial, width].inference_delay_frames for trial in range(LATENCY_TRIALS))
            for width in WIDTHS
        }
        accepted_trials: list[LatencyTrialRow] = []
        for trial_index, order in enumerate(width_orders):
            for width_order_index, width in enumerate(order):
                dolphin_load.raise_if_failed()
                row = search_rows[trial_index, width]
                selected_delay = selected_delays[width]
                if row.inference_delay_frames != selected_delay:
                    row = benchmark_width_latency(
                        width,
                        delay=selected_delay,
                        trial_index=trial_index,
                        phase="validation",
                        on_probe=record,
                    )
                accepted_trials.append(LatencyTrialRow(trial_index, width_order_index, row))
        rows = [
            LatencyRow(
                width=width,
                inference_delay_frames=selected_delays[width],
                replan_interval_frames=selected_delays[width],
                prediction_frames=2 * selected_delays[width],
                p99_seconds=max(trial.row.p99_seconds for trial in accepted_trials if trial.row.width == width),
                frame_period_seconds=1 / 60,
            )
            for width in WIDTHS
        ]
        dolphin_load.raise_if_failed()
    return write_latency_manifest(
        path,
        rows,
        accepted_trials,
        gpu_name,
        raw_probe_sha256,
        dolphin_load.evidence,
    )


def run_latency_diagnostic(path: Path, batch_sizes: Sequence[int]) -> None:
    """Measure non-official batch sizes without creating an accepted manifest."""
    if not torch.cuda.is_available():
        raise RuntimeError("latency diagnostic requires CUDA")
    if not batch_sizes or any(batch_size < 1 for batch_size in batch_sizes):
        raise ValueError("diagnostic batch sizes must be positive")
    if len(set(batch_sizes)) != len(batch_sizes):
        raise ValueError("diagnostic batch sizes must be unique")
    gpu_name = torch.cuda.get_device_name()
    if "RTX 3060" not in gpu_name:
        raise RuntimeError(f"latency diagnostic requires the target RTX 3060, found {gpu_name!r}")
    probes: list[LatencyProbe] = []

    def record(probe: LatencyProbe) -> None:
        probes.append(probe)
        write_latency_probe_report(path, probes, gpu_name)

    print(f"[latency] raw diagnostic report: {path}", flush=True)
    failures: list[str] = []
    for batch_size in batch_sizes:
        for width in WIDTHS:
            try:
                benchmark_width_latency(
                    width,
                    batch_size=batch_size,
                    phase="diagnostic",
                    on_probe=record,
                )
            except LatencyProbeFailure as error:
                failures.append(str(error))
    if failures:
        raise LatencyProbeFailure(f"latency diagnostic rejected configurations: {'; '.join(failures)}")


def _validate_deployment_timing(prediction_frames: int, delay_frames: int, replan_interval_frames: int) -> None:
    if not isinstance(delay_frames, int) or isinstance(delay_frames, bool) or delay_frames < 1:
        raise ValueError(f"delay_frames must be a positive integer, got {delay_frames!r}")
    if (
        not isinstance(replan_interval_frames, int)
        or isinstance(replan_interval_frames, bool)
        or replan_interval_frames < 1
    ):
        raise ValueError(f"replan_interval_frames must be a positive integer, got {replan_interval_frames!r}")
    if delay_frames + replan_interval_frames > prediction_frames:
        raise ValueError(
            "delay_frames + replan_interval_frames must not exceed prediction_frames: "
            f"{delay_frames} + {replan_interval_frames} > {prediction_frames}"
        )
    if replan_interval_frames != delay_frames or prediction_frames != 2 * delay_frames:
        raise ValueError("buffered timing requires R=d and H=2d")


@dataclass
class BufferedPolicy(RecedingHorizon):
    """Stage the current inference and execute the preceding plan slice."""

    inference_delay_frames: int = 1
    _staged: dict[Slot, np.ndarray] = dataclass_field(default_factory=dict)
    _executing: dict[Slot, list[np.ndarray]] = dataclass_field(default_factory=dict)
    _phases: dict[Slot, int] = dataclass_field(default_factory=dict)
    _inference_lock: threading.Lock = dataclass_field(default_factory=threading.Lock)
    last_plan_inferred: bool = False
    last_inference_rows: int = 0
    last_inference_batch_rows: int = 0
    neutral_actions: int = 0
    total_actions: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_deployment_timing(self.L_chunk, self.inference_delay_frames, self.s)

    @property
    def runtime_spec(self) -> PolicyRuntimeSpec:
        return PolicyRuntimeSpec(
            context_frames=self.L_ctx,
            prediction_frames=self.L_chunk,
            execution_stride=self.s,
            committed_frames=self.inference_delay_frames,
            action_dim=len(ACTION_CHANNELS),
        )

    def _infer(self, slots: list[Slot]) -> np.ndarray:
        if not self._inference_lock.acquire(blocking=False):
            raise RuntimeError("buffered policy cannot overlap model calls")
        try:
            plans = self.predict_chunk(self._context(slots), None)
        finally:
            self._inference_lock.release()
        if plans.shape[:2] != (len(slots), self.L_chunk):
            raise ValueError("predictor returned the wrong batch or prediction length")
        self.last_plan_inferred = True
        self.last_inference_rows = len(slots)
        self.last_inference_batch_rows = len(slots)
        return plans

    def _stage(self, slots: list[Slot]) -> None:
        plans = self._infer(slots)
        start = self.inference_delay_frames
        for row, slot in enumerate(slots):
            previous = self._staged.get(slot)
            if previous is None:
                previous = np.repeat(NEUTRAL_ACTION[None], self.s, axis=0)
            self._executing[slot] = [action.copy() for action in previous]
            self._staged[slot] = plans[row, start : start + self.s].astype(np.float32, copy=True)

    def __call__(self, frame_index: int, obs: Mapping[Slot, dict]):
        del frame_index
        live = list(obs)
        self._ingest(live, obs)
        for slot in live:
            if self._slots[slot].reset_pending:
                self._staged.pop(slot, None)
                self._executing.pop(slot, None)
                self._phases[slot] = 0
        due = [slot for slot in live if slot not in self._executing or self._phases.get(slot, 0) % self.s == 0]
        self.last_plan_inferred = False
        self.last_inference_rows = 0
        self.last_inference_batch_rows = 0
        if due:
            self._stage(due)
        actions: dict[Slot, np.ndarray] = {}
        for slot in live:
            queue = self._executing[slot]
            if not queue:
                raise RuntimeError(f"slot {slot} exhausted its buffered action slice")
            action = queue.pop(0)
            actions[slot] = action
            self._push_ego(slot, action)
            self._phases[slot] = self._phases.get(slot, 0) + 1
            self.total_actions += 1
            self.neutral_actions += int(np.array_equal(action, NEUTRAL_ACTION))
        return {slot: action_vec_to_controller(action) for slot, action in actions.items()}

    def plan_rows(self, rows: Mapping[Slot, Sequence[ObservationRow]]) -> Mapping[Slot, np.ndarray]:
        live = list(rows)
        if not live:
            return {}
        reset_slots: set[Slot] = set()
        for slot in live:
            if not rows[slot]:
                raise ValueError(f"slot {slot} requested a plan without observation rows")
            for row in rows[slot]:
                self._ingest_row(slot, row)
            if self._slots[slot].reset_pending:
                reset_slots.add(slot)
            if slot in reset_slots:
                self._staged.pop(slot, None)
        previous = {slot: self._staged.get(slot, np.repeat(NEUTRAL_ACTION[None], self.s, axis=0)) for slot in live}
        plans = self._infer(live)
        start = self.inference_delay_frames
        out: dict[Slot, np.ndarray] = {}
        for row, slot in enumerate(live):
            self._staged[slot] = plans[row, start : start + self.s].astype(np.float32, copy=True)
            returned = np.repeat(NEUTRAL_ACTION[None], self.L_chunk, axis=0)
            returned[: self.s] = previous[slot]
            out[slot] = returned
        return out


def make_policy(
    model: GPT,
    stats: dict[str, FeatureStats],
    cfg: TrainConfig,
    *,
    decode_seed: int | None = None,
    inference: BF16Inference | None = None,
    telemetry: DecodeTelemetry | None = None,
    ego_player_id: int = MASKED_PLAYER_ID,
    delay_frames: int | None = None,
    replan_interval_frames: int | None = None,
    device: str = DEVICE,
) -> BufferedPolicy:
    horizon = cfg.prediction_frames
    delay = cfg.delay_frames if delay_frames is None else delay_frames
    replan = cfg.replan_interval_frames if replan_interval_frames is None else replan_interval_frames
    _validate_deployment_timing(horizon, delay, replan)
    engine = BF16Inference(model, cfg) if inference is None else inference
    random_streams = None if decode_seed is None else SlotGroupRng(decode_seed, CONTROLLER_GROUP_NAMES)
    generator = None if decode_seed is None else torch.Generator(device=device).manual_seed(decode_seed)

    @torch.no_grad()
    def predict(ctx: Context, committed: np.ndarray | None) -> np.ndarray:
        if committed is not None:
            raise ValueError("O54 buffered inference never conditions on a committed prefix")
        started = time.perf_counter()
        result = (
            engine.decode(
                _condition_ego_player(ctx, ego_player_id),
                horizon,
                streams=random_streams,
                gen=generator,
            )
            .cpu()
            .numpy()
        )
        if telemetry is not None:
            telemetry.record(rows=ctx.ctx_pad.shape[0], horizon=horizon, seconds=time.perf_counter() - started)
        return result

    return BufferedPolicy(
        predict_chunk=predict,
        stats=stats,
        L_ctx=cfg.arch.L_ctx,
        L_chunk=horizon,
        s=replan,
        d=delay,
        inference_delay_frames=delay,
        device=device,
        float_dtype=next(model.parameters()).dtype,
        extra=ITEM_COLUMNS,
        projection=BASE_ITEMS_PROJECTION,
    )


@dataclass(frozen=True, slots=True)
class EvalProtocol:
    fixed_ego_character: int | None
    ego_player_id: int
    ego_player_code: str | None
    opponent_identity_conditioned: bool
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
    delay_frames: int
    replan_interval_frames: int
    transport_delay_frames: int
    timing_model: str
    dtype: str
    inference_mode: str
    inference_compile_mode: str
    inference_attention_backend: str
    compiled_inference_bucket: int
    checkpoint_sha256: str
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES
    start_retries: int = DEFAULT_START_RETRIES


def matchup_diversity(
    n_matchups: int,
    fixed_ego_character: melee.Character | None = None,
) -> tuple[int, int, int, str]:
    matchups = [
        (scheduled_ego if fixed_ego_character is None else fixed_ego_character, cpu)
        for scheduled_ego, cpu in matchups_for_vs_cpu(n_matchups)
    ]
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
    checkpoint_sha256: str,
    max_parallel: int | None = None,
    inference_mode: str | None = None,
    inference_compile_mode: str = "reduce-overhead",
    inference_attention_backend: str = "dense_sdpa",
    fixed_ego_character: melee.Character | None = None,
    ego_player_id: int = MASKED_PLAYER_ID,
    ego_player_code: str | None = None,
    delay_frames: int | None = None,
    replan_interval_frames: int | None = None,
) -> EvalProtocol:
    if fixed_ego_character is None:
        pairs, egos, cpus, schedule_sha = assert_protocol_diversity(n_matchups)
    else:
        pairs, egos, cpus, schedule_sha = matchup_diversity(n_matchups, fixed_ego_character)
    delay = cfg.delay_frames if delay_frames is None else delay_frames
    replan = cfg.replan_interval_frames if replan_interval_frames is None else replan_interval_frames
    _validate_deployment_timing(cfg.prediction_frames, delay, replan)
    return EvalProtocol(
        fixed_ego_character=None if fixed_ego_character is None else int(fixed_ego_character.value),
        ego_player_id=ego_player_id,
        ego_player_code=ego_player_code,
        opponent_identity_conditioned=False,
        n_matchups=n_matchups,
        allowed_cpus=usable_cpus(),
        hardware_wave_bucket=automatic_parallelism(),
        max_parallel=_eval_parallelism(cfg, n_matchups, max_parallel),
        max_frames=cfg.eval_max_frames,
        seed=cfg.eval_seed,
        cpu_level=9,
        ego_port=1,
        seed_stage=int(PRIOR_SWEEP_SEED_STAGE.value),
        matchup_schedule_sha256=schedule_sha,
        oriented_pairs=pairs,
        ego_characters=egos,
        cpu_characters=cpus,
        prediction_frames=cfg.prediction_frames,
        delay_frames=delay,
        replan_interval_frames=replan,
        transport_delay_frames=cfg.timing.transport_delay_frames,
        timing_model=cfg.timing.timing_model,
        dtype=str(next(model.parameters()).dtype),
        inference_mode=cfg.inference_mode if inference_mode is None else inference_mode,
        inference_compile_mode=inference_compile_mode,
        inference_attention_backend=inference_attention_backend,
        compiled_inference_bucket=_eval_inference_bucket(cfg, n_matchups, max_parallel),
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
    checkpoint_sha256: str = "unavailable",
    inference: BF16Inference | None = None,
    eager: bool = False,
    max_parallel: int | None = None,
    fixed_ego_character: melee.Character | None = None,
    ego_player_id: int = MASKED_PLAYER_ID,
    ego_player_code: str | None = None,
    delay_frames: int | None = None,
    replan_interval_frames: int | None = None,
) -> dict[str, float]:
    horizon = cfg.prediction_frames
    inference_mode = "eager" if eager else cfg.inference_mode
    inference = (
        BF16Inference(
            model,
            cfg,
            bucket=_eval_inference_bucket(cfg, n_matchups, max_parallel),
            compiled=inference_mode == "compiled",
        )
        if inference is None
        else inference
    )
    if inference.model is not model:
        raise ValueError("the supplied inference engine must own the evaluation model")
    protocol = _eval_protocol(
        cfg,
        model,
        n_matchups=n_matchups,
        checkpoint_sha256=checkpoint_sha256,
        max_parallel=max_parallel,
        inference_mode=inference_mode,
        inference_compile_mode=inference.compile_mode,
        inference_attention_backend=inference.attention_backend,
        fixed_ego_character=fixed_ego_character,
        ego_player_id=ego_player_id,
        ego_player_code=ego_player_code,
        delay_frames=delay_frames,
        replan_interval_frames=replan_interval_frames,
    )
    if protocol.inference_mode != "eager" or inference.compiled:
        raise RuntimeError("official O54 evaluation requires eager BF16 inference")
    telemetry = DecodeTelemetry()
    policy_index = itertools.count()
    policies: list[BufferedPolicy] = []

    def factory() -> RecedingHorizon:
        policy = make_policy(
            model,
            stats,
            cfg,
            decode_seed=protocol.seed + next(policy_index),
            inference=inference,
            telemetry=telemetry,
            ego_player_id=protocol.ego_player_id,
            delay_frames=protocol.delay_frames,
            replan_interval_frames=protocol.replan_interval_frames,
        )
        policies.append(policy)
        return policy

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
                fixed_ego_character=fixed_ego_character,
            )
    finally:
        model.train(was_training)
    metrics = vs_cpu_metrics(results, seed=protocol.seed)
    metrics["eval_wall_seconds"] = time.perf_counter() - started
    metrics["eval_total_wall_seconds"] = time.perf_counter() - total_started
    metrics["inference_compile_seconds"] = compile_seconds
    neutral = sum(policy.neutral_actions for policy in policies)
    actions = sum(policy.total_actions for policy in policies)
    metrics["neutral_action_fraction"] = neutral / max(actions, 1)
    metrics["prediction_frames"] = float(cfg.prediction_frames)
    metrics["delay_frames"] = float(protocol.delay_frames)
    metrics["replan_interval_frames"] = float(protocol.replan_interval_frames)
    metrics["ego_player_id"] = float(protocol.ego_player_id)
    if protocol.fixed_ego_character is not None:
        metrics["fixed_ego_character"] = float(protocol.fixed_ego_character)
    metrics.update(telemetry.metrics())
    _write_eval_evidence(replay_dir, rows, metrics, protocol)
    return metrics


def _eval_wandb_metrics(values: dict[str, float]) -> dict[str, float]:
    """Keep closed-loop quality, uncertainty, reliability, and latency."""
    names = {
        "stocks_taken_per_min": "stocks_taken_per_min",
        "stocks_lost_per_min": "stocks_lost_per_min",
        "damage_dealt_per_min": "damage_dealt_per_min",
        "damage_taken_per_min": "damage_taken_per_min",
        "net_stock_per_min": "net_stock_per_min",
        "net_dmg_per_min": "net_dmg_per_min",
        "net_stock_cluster_bootstrap_lcb": "net_stock_lcb",
        "net_dmg_cluster_bootstrap_lcb": "net_dmg_lcb",
        "boots": "boots",
        "crashed": "crashed",
        "dead_frame_frac": "dead_frame_fraction",
        "decode_p95_ms": "decode_p95_ms",
        "eval_total_wall_seconds": "wall_s",
        "neutral_action_fraction": "neutral_action_fraction",
        "prediction_frames": "prediction_frames",
        "delay_frames": "delay_frames",
        "replan_interval_frames": "replan_interval_frames",
    }
    return {output: values[source] for source, output in names.items() if source in values}


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


def closed_loop_evaluation_updates(final_update: int, every: int) -> tuple[int, ...]:
    """Return each periodic milestone plus the final update without duplicates."""
    if final_update <= 0 or every <= 0:
        raise ValueError("evaluation updates and cadence must be positive")
    updates = list(range(every, final_update + 1, every))
    if not updates or updates[-1] != final_update:
        updates.append(final_update)
    return tuple(updates)


def lr_schedule(cfg: TrainConfig):
    def schedule(step: int) -> float:
        update = step + 1
        if update <= cfg.warmup_steps:
            return update / cfg.warmup_steps
        return 1.0

    return schedule


def center_class_logits(logits: Tensor) -> Tensor:
    """Remove the softmax-invariant common mode from each class group."""
    return logits - logits.mean(dim=-1, keepdim=True)


def mup_readout_std(fan_in: int, base_fan_in: int) -> float:
    if fan_in < 1 or base_fan_in < 1:
        raise ValueError("readout fan-ins must be positive")
    return math.sqrt(base_fan_in) / fan_in


def _final_readouts(model: GPT) -> tuple[tuple[nn.Linear, int], ...]:
    action = tuple((cast(nn.Linear, model.temporal.outputs[name].down), 128) for name in CONTROLLER_GROUP_NAMES)
    trunk_skip = tuple((cast(nn.Linear, model.temporal.trunk_outputs[name]), 256) for name in CONTROLLER_GROUP_NAMES)
    return (*action, *trunk_skip)


def initialize_o51_parameters(model: GPT, cfg: TrainConfig) -> None:
    """Apply O51's selected hidden and μP-normal readout initialization."""
    final_modules = {id(module) for module, _ in _final_readouts(model)}
    for module in model.modules():
        if isinstance(module, nn.Linear):
            if id(module) in final_modules:
                continue
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=cfg.hidden_std_multiplier / math.sqrt(module.weight.shape[1]),
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=cfg.hidden_std_multiplier / math.sqrt(module.embedding_dim),
            )
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
    for module, base_fan_in in _final_readouts(model):
        nn.init.normal_(module.weight, mean=0.0, std=mup_readout_std(module.in_features, base_fan_in))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    nn.init.zeros_(model.player_projection.weight)


@dataclass(frozen=True, slots=True)
class OptimizerRole:
    lr_kind: Literal["hidden", "input", "output", "vector"]
    decay: bool
    fan_in_multiplier: float = 1.0


def _is_final_readout(name: str) -> bool:
    return (name.startswith("temporal.outputs.") and ".down." in name) or name.startswith("temporal.trunk_outputs.")


def _output_fan_in_multiplier(name: str, cfg: TrainConfig) -> float:
    if name.startswith("temporal.outputs."):
        return cfg.arch.group_head_dim / 128
    if name.startswith("temporal.trunk_outputs."):
        return cfg.arch.d_model / 256
    raise ValueError(f"{name!r} is not a final readout")


def optimizer_roles(model: GPT, cfg: TrainConfig) -> dict[str, OptimizerRole]:
    """Assign each parameter by its semantic role."""
    embedding_prefixes = (
        "codec.class_embeddings.",
        "cat_embeds.",
        "char_emb.",
        "stage_emb.",
        "item_type_emb.",
        "item_state_emb.",
        "player_embedding.",
        "temporal.offset_embedding.",
    )
    finite_prefixes = (
        "codec.semantic_projections.",
        "item_encoder.",
        "observation_encoder.",
        "player_projection.",
        "temporal.group_condition.",
    )
    roles: dict[str, OptimizerRole] = {}
    for name, parameter in model.named_parameters():
        if _is_final_readout(name):
            roles[name] = OptimizerRole(
                "output", parameter.ndim >= 2, fan_in_multiplier=_output_fan_in_multiplier(name, cfg)
            )
        elif name.startswith(("trunk.blocks.", "temporal.blocks.")):
            roles[name] = OptimizerRole("hidden", parameter.ndim >= 2)
        elif name == "temporal.token_projection.weight" or (
            name.startswith("temporal.outputs.") and name.endswith("up.weight")
        ):
            roles[name] = OptimizerRole("hidden", True)
        elif name.startswith(embedding_prefixes):
            roles[name] = OptimizerRole("input", False)
        elif name.startswith(finite_prefixes):
            roles[name] = OptimizerRole("input" if parameter.ndim >= 2 else "vector", parameter.ndim >= 2)
        elif name == "temporal.token_projection.bias":
            roles[name] = OptimizerRole("vector", False)
        else:
            raise RuntimeError(f"O54 has no optimizer role for {name} {tuple(parameter.shape)}")
    return roles


def _role_lr(role: OptimizerRole, cfg: TrainConfig) -> float:
    return cfg.adam_lr / role.fan_in_multiplier if role.lr_kind == "output" else cfg.adam_lr


def make_optimizer(model: GPT, cfg: TrainConfig) -> torch.optim.AdamW:
    """Build all-AdamW groups while preserving O50's semantic LR roles."""
    roles = optimizer_roles(model, cfg)
    named = dict(model.named_parameters())
    buckets: dict[tuple[object, ...], list[nn.Parameter]] = defaultdict(list)
    for name, role in roles.items():
        lr = _role_lr(role, cfg)
        key = (lr, cfg.adam_weight_decay if role.decay else 0.0)
        buckets[key].append(named[name])
    groups: list[dict[str, object]] = []
    for key, parameters in buckets.items():
        groups.append(
            {
                "params": parameters,
                "lr": key[0],
                "weight_decay": key[1],
            }
        )
    return torch.optim.AdamW(
        groups,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_eps,
        fused=DEVICE == "cuda",
    )


def _button_path_parameters(model: GPT) -> dict[str, nn.Parameter]:
    """Return the action-path matrices monitored before clipping."""
    button_head = cast(NonlinearActionHead, model.temporal.outputs["buttons"])
    return {
        "buttons_output_weight": cast(nn.Parameter, button_head.down.weight),
        "buttons_condition_weight": cast(nn.Parameter, model.temporal.group_condition["buttons"].weight),
        "token_projection_weight": cast(nn.Parameter, model.temporal.token_projection.weight),
    }


def _button_gradient_abs_max(model: GPT) -> Tensor:
    """Return one pre-clipping action-path gradient guardrail."""
    maxima: list[Tensor] = []
    for name, parameter in _button_path_parameters(model).items():
        if parameter.grad is None:
            raise RuntimeError(f"button parameter {name!r} has no gradient")
        maxima.append(parameter.grad.detach().float().abs().amax())
    return torch.stack(maxima).amax()


def training_microbatches(batch: TrainBatch, cfg: TrainConfig) -> tuple[TrainBatch, ...]:
    """Split only the execution batch; replay sampling and optimizer cadence stay fixed."""
    size = train_microbatch_size(cfg)
    batch_size = batch.target.shape[0]
    if batch_size != cfg.batch_size:
        raise ValueError(f"expected optimizer batch {cfg.batch_size}, got {batch_size}")
    if batch_size % size:
        raise ValueError(f"training microbatch {size} must divide optimizer batch {batch_size}")
    chunks: list[TrainBatch] = []
    for start in range(0, batch_size, size):
        stop = start + size
        context = batch.context
        chunks.append(
            TrainBatch(
                context=Context(
                    features={name: value[start:stop] for name, value in context.features.items()},
                    ctx_pad=context.ctx_pad[start:stop],
                    slot_ids=None if context.slot_ids is None else context.slot_ids[start:stop],
                    reset=None if context.reset is None else context.reset[start:stop],
                ),
                target=batch.target[start:stop],
                replay_ids=None if batch.replay_ids is None else batch.replay_ids[start:stop],
            )
        )
    return tuple(chunks)


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
        "inputs": other_ids,
    }
    counts = {
        name: sum(parameter.numel() for parameter in all_parameters if id(parameter) in parameter_ids)
        for name, parameter_ids in partitions.items()
    }
    counts["total"] = sum(parameter.numel() for parameter in all_parameters)
    if sum(value for name, value in counts.items() if name != "total") != counts["total"]:
        raise RuntimeError("parameter subsystem partition is incomplete")
    try:
        expected = model.cfg.arch.parameter_count_contract
    except ValueError as error:
        raise RuntimeError(str(error)) from error
    if counts != expected:
        raise RuntimeError(f"parameter contract changed: {counts} != {expected}")
    return counts


def approximate_training_flops_per_update(cfg: TrainConfig, parameter_counts: dict[str, int]) -> int:
    """Return ``6 * positions_per_update * N_eff``."""
    trunk_and_inputs = parameter_counts["trunk"] + parameter_counts["inputs"]
    temporal_and_heads = parameter_counts["temporal_decoder"] + parameter_counts["group_heads"]
    effective_parameters = 2 * trunk_and_inputs + len(cfg.arch.head_offsets) * temporal_and_heads
    positions = cfg.batch_size * (cfg.arch.L_ctx - cfg.arch.direct_loss_start)
    return 6 * positions * effective_parameters


def effective_parameter_count(parameter_counts: Mapping[str, int]) -> int:
    return 2 * (parameter_counts["trunk"] + parameter_counts["inputs"]) + 14 * (
        parameter_counts["temporal_decoder"] + parameter_counts["group_heads"]
    )


def training_compute(cfg: TrainConfig, parameter_counts: Mapping[str, int]) -> int:
    return 6 * cfg.target_positions * effective_parameter_count(parameter_counts)


@dataclass(frozen=True, slots=True)
class LearningCurveFit:
    """A monotone saturating power curve anchored at D=2^28."""

    response_at_d28: float
    asymptotic_gain: float
    exponent: float
    residual_sum_squares: float

    def predict(self, positions: float) -> float:
        if not math.isfinite(positions) or positions <= 0:
            raise ValueError("learning-curve positions must be finite and positive")
        ratio = positions / DATA_POSITIONS[0]
        return self.response_at_d28 + self.asymptotic_gain * (1 - ratio ** (-self.exponent))


def fit_gameplay_learning_curve(positions: Sequence[int], responses: Sequence[float]) -> LearningCurveFit:
    """Fit ``S(D)=S_28+A*(1-(D/2^28)^-alpha)`` with ``A>=0``."""
    from scipy.optimize import least_squares

    if len(positions) != len(responses) or len(responses) < 3:
        raise ValueError("a gameplay learning curve needs at least three paired observations")
    x = np.asarray(positions, dtype=np.float64)
    y = np.asarray(responses, dtype=np.float64)
    if np.any(x <= 0) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("gameplay learning-curve observations must be finite and positive in D")
    ratio = x / DATA_POSITIONS[0]

    def residuals(parameters: np.ndarray) -> np.ndarray:
        baseline, gain, exponent = parameters
        return baseline + gain * (1 - ratio**-exponent) - y

    initial_gain = max(float(y.max() - y[0]), 0.1)
    fit = least_squares(
        residuals,
        x0=np.asarray((y[0], initial_gain, 0.5)),
        bounds=((-np.inf, 0.0, 0.01), (np.inf, np.inf, 4.0)),
        max_nfev=10_000,
    )
    if not fit.success or not np.isfinite(fit.x).all():
        raise RuntimeError(f"gameplay learning-curve fit failed: {fit.message}")
    baseline, gain, exponent = map(float, fit.x)
    return LearningCurveFit(baseline, gain, exponent, float(np.square(fit.fun).sum()))


def _response_order(responses: Mapping[int, float]) -> tuple[int, ...]:
    if not responses or any(not math.isfinite(value) for value in responses.values()):
        raise ValueError("response ordering needs finite observations")
    return tuple(sorted(responses, key=lambda width: (-responses[width], width)))


@dataclass(frozen=True, slots=True)
class AnalysisEndpoint:
    run_id: str
    run_name: str
    width: int
    updates: int
    effective_parameters: int
    bc_parameters: int
    compute: int
    net_stock_per_min: float
    net_stock_lcb: float


def _validate_analysis_eval_protocol(protocol: Mapping[str, object], timing: LatencyRow) -> None:
    pairs, egos, cpus, schedule_sha256 = assert_protocol_diversity(96)
    expected: dict[str, object] = {
        "fixed_ego_character": None,
        "ego_player_id": MASKED_PLAYER_ID,
        "ego_player_code": None,
        "opponent_identity_conditioned": False,
        "n_matchups": 96,
        "max_parallel": 32,
        "max_frames": 7_200,
        "seed": 0,
        "cpu_level": 9,
        "ego_port": 1,
        "seed_stage": int(PRIOR_SWEEP_SEED_STAGE.value),
        "matchup_schedule_sha256": schedule_sha256,
        "oriented_pairs": pairs,
        "ego_characters": egos,
        "cpu_characters": cpus,
        "prediction_frames": timing.prediction_frames,
        "delay_frames": timing.inference_delay_frames,
        "replan_interval_frames": timing.replan_interval_frames,
        "transport_delay_frames": 0,
        "timing_model": "buffered",
        "dtype": "torch.bfloat16",
        "inference_mode": "eager",
        "inference_compile_mode": "default",
        "inference_attention_backend": "dense_sdpa",
        "compiled_inference_bucket": 32,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "start_retries": DEFAULT_START_RETRIES,
    }
    changed = {name: (protocol.get(name), value) for name, value in expected.items() if protocol.get(name) != value}
    checkpoint = protocol.get("checkpoint_sha256")
    if not isinstance(checkpoint, str) or re.fullmatch(r"[0-9a-f]{64}", checkpoint) is None:
        changed["checkpoint_sha256"] = (checkpoint, "64 lowercase hexadecimal characters")
    if changed:
        raise ValueError(f"evaluation endpoint changed the frozen O54 protocol: {changed}")


def load_analysis_match_rows(
    endpoints: Mapping[tuple[int, int], AnalysisEndpoint],
    latency_rows: Mapping[int, LatencyRow],
    output: Path,
) -> dict[tuple[int, int], tuple[MatchRow, ...]]:
    """Download and validate the common 96-block evaluation evidence."""
    destination_root = output / "match_rows"
    destination_root.mkdir(parents=True, exist_ok=True)
    client = r2.client()
    loaded: dict[tuple[int, int], tuple[MatchRow, ...]] = {}
    schedule: tuple[tuple[int, int], ...] | None = None
    schedule_sha256: str | None = None
    for key, endpoint in sorted(endpoints.items()):
        width, update = key
        destination = destination_root / f"w{width}-step-{update:07d}.json"
        object_key = f"runs/{endpoint.run_name}/checkpoints/eval96-step-{update:07d}/match_rows.json"
        client.download_file(r2.bucket(), object_key, str(destination))
        payload = json.loads(destination.read_text())
        protocol = payload.get("protocol") if isinstance(payload, dict) else None
        raw_rows = payload.get("rows") if isinstance(payload, dict) else None
        if payload.get("schema_version") != 6 or not isinstance(protocol, dict) or not isinstance(raw_rows, list):
            raise ValueError(f"W{width} update {update} has invalid match-row evidence")
        timing = latency_rows[width]
        try:
            _validate_analysis_eval_protocol(protocol, timing)
        except ValueError as error:
            raise ValueError(f"W{width} update {update} used the wrong evaluation protocol") from error
        rows = tuple(MatchRow.from_dict(row) for row in raw_rows)
        by_boot: dict[int, list[MatchRow]] = defaultdict(list)
        for row in rows:
            by_boot[row.boot_index].append(row)
        if tuple(sorted(by_boot)) != tuple(range(96)):
            raise ValueError(f"W{width} update {update} does not contain all 96 complete blocks")
        local_schedule = []
        for boot_index in range(96):
            pairs = {(row.ego_character, row.opp_character) for row in by_boot[boot_index]}
            if len(pairs) != 1:
                raise ValueError(f"W{width} update {update} block {boot_index} has mixed matchups")
            local_schedule.append(next(iter(pairs)))
        typed_schedule = tuple(local_schedule)
        local_sha = protocol.get("matchup_schedule_sha256")
        if schedule is None:
            schedule = typed_schedule
            schedule_sha256 = cast(str, local_sha)
        elif typed_schedule != schedule or local_sha != schedule_sha256:
            raise ValueError("evaluation endpoints do not use the same 96 matchup blocks")
        loaded[key] = rows
    return loaded


def _boot_net_stock_components(rows: Sequence[MatchRow]) -> tuple[np.ndarray, np.ndarray]:
    numerators = np.zeros(96, dtype=np.float64)
    active_minutes = np.zeros(96, dtype=np.float64)
    for row in rows:
        if row.active_frames <= 0:
            continue
        numerators[row.boot_index] += row.stocks_taken - row.stocks_lost
        active_minutes[row.boot_index] += row.active_frames / FRAMES_PER_MINUTE
    if np.any(active_minutes <= 0):
        raise ValueError("every evaluation block must contain active gameplay")
    return numerators, active_minutes


def bootstrap_learning_curve_analysis(
    endpoints: Mapping[tuple[int, int], AnalysisEndpoint],
    match_rows: Mapping[tuple[int, int], Sequence[MatchRow]],
    widths: Sequence[int],
    *,
    holdout_passed: bool,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 54,
) -> dict[str, object]:
    """Resample the same complete matchup blocks across every endpoint."""
    if resamples < 1:
        raise ValueError("bootstrap resamples must be positive")
    expected = {(width, update) for width in widths for update in DATA_UPDATES}
    if set(match_rows) != expected:
        raise ValueError("bootstrap match rows do not cover the full endpoint matrix")
    components = {key: _boot_net_stock_components(rows) for key, rows in match_rows.items()}
    for key, (numerator, minutes) in components.items():
        point = float(numerator.sum() / minutes.sum())
        logged = endpoints[key].net_stock_per_min
        if not math.isclose(point, logged, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"raw match rows disagree with the logged primary response at {key}")
    indices = np.random.default_rng(seed).integers(0, 96, size=(resamples, 96))
    responses = {
        key: numerator[indices].sum(axis=1) / minutes[indices].sum(axis=1)
        for key, (numerator, minutes) in components.items()
    }
    holdout_matches = 0
    failed_fits = 0
    winners = {width: 0 for width in widths}
    d34_predictions = {width: [] for width in widths}
    for sample in range(resamples):
        try:
            initial = {
                width: fit_gameplay_learning_curve(
                    DATA_POSITIONS[:3],
                    [responses[width, update][sample] for update in DATA_UPDATES[:3]],
                )
                for width in widths
            }
            predicted = {width: initial[width].predict(DATA_POSITIONS[-1]) for width in widths}
            observed = {width: responses[width, DATA_UPDATES[-1]][sample] for width in widths}
            holdout_matches += _response_order(predicted) == _response_order(observed)
            if holdout_passed:
                refits = {
                    width: fit_gameplay_learning_curve(
                        DATA_POSITIONS,
                        [responses[width, update][sample] for update in DATA_UPDATES],
                    )
                    for width in widths
                }
                projected = {width: refits[width].predict(2**34) for width in widths}
                winners[_response_order(projected)[0]] += 1
                for width, value in projected.items():
                    d34_predictions[width].append(value)
        except RuntimeError:
            failed_fits += 1
    completed = resamples - failed_fits
    if completed < 0.95 * resamples:
        raise RuntimeError(f"only {completed}/{resamples} block-bootstrap curve fits completed")
    report: dict[str, object] = {
        "unit": "complete matchup boot",
        "shared_resample_indices": True,
        "requested_resamples": resamples,
        "completed_resamples": completed,
        "holdout_order_match_fraction": holdout_matches / completed,
    }
    if holdout_passed:
        report["d34_winner_fraction"] = {str(width): winners[width] / completed for width in widths}
        report["d34_prediction_95_interval"] = {
            str(width): [
                float(np.percentile(d34_predictions[width], 2.5)),
                float(np.percentile(d34_predictions[width], 97.5)),
            ]
            for width in widths
        }
    return report


def _analysis_endpoints(runs: Iterable[Any], latency_sha256: str) -> dict[tuple[int, int], AnalysisEndpoint]:
    endpoints: dict[tuple[int, int], AnalysisEndpoint] = {}
    for run in runs:
        config = dict(run.config)
        if config.get("experiment_id") != _EXPERIMENT_ID:
            continue
        architecture = config.get("arch")
        if not isinstance(architecture, dict):
            raise ValueError(f"O54 run {run.id} has no architecture config")
        width = int(architecture["d_model"])
        if int(config["max_steps"]) != DATA_UPDATES[-1]:
            raise ValueError(f"O54 run {run.id} is not a complete D=2^31 trajectory")
        if config.get("latency_manifest_sha256") != latency_sha256:
            raise ValueError(f"O54 run {run.id} uses a different latency manifest")
        if run.state != "finished":
            continue
        history = run.scan_history(
            keys=[
                "global_step",
                "eval/checkpoint_step",
                "eval/boots",
                "eval/crashed",
                "eval/net_stock_per_min",
                "eval/net_stock_lcb",
            ],
            page_size=1_000,
        )
        for row in history:
            update = int(row.get("eval/checkpoint_step", row.get("global_step", -1)))
            if update not in DATA_UPDATES:
                continue
            if int(row.get("eval/boots", 0)) != 96 or float(row.get("eval/crashed", 1.0)) != 0.0:
                continue
            response = float(row.get("eval/net_stock_per_min", math.nan))
            lcb = float(row.get("eval/net_stock_lcb", math.nan))
            if not math.isfinite(response) or not math.isfinite(lcb):
                raise ValueError(f"O54 run {run.id} has an invalid evaluation at update {update}")
            counts = PARAMETER_COUNT_CONTRACTS[width]
            cfg = config_for_width(width, updates=update)
            endpoint = AnalysisEndpoint(
                run_id=str(run.id),
                run_name=str(run.name),
                width=width,
                updates=update,
                effective_parameters=effective_parameter_count(counts),
                bc_parameters=counts["total"],
                compute=training_compute(cfg, counts),
                net_stock_per_min=response,
                net_stock_lcb=lcb,
            )
            key = (width, update)
            if key in endpoints:
                raise ValueError(f"multiple complete O54 evaluations exist for W{width} at update {update}")
            endpoints[key] = endpoint
    return endpoints


def write_analysis_artifacts(
    endpoints: Mapping[tuple[int, int], AnalysisEndpoint],
    latency_rows: Mapping[int, LatencyRow],
    output: Path,
    *,
    match_rows: Mapping[tuple[int, int], Sequence[MatchRow]] | None = None,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, object]:
    """Test the registered D=2^31 holdout, then conditionally extrapolate to D=2^34."""
    widths = tuple(endpoint.width for endpoint in study_endpoints(latency_rows))
    expected = tuple((width, update) for width in widths for update in DATA_UPDATES)
    missing = [key for key in expected if key not in endpoints]
    if missing:
        raise ValueError(f"O54 analysis is missing complete endpoints: {missing}")
    output.mkdir(parents=True, exist_ok=True)
    ordered = [endpoints[key] for key in expected]
    with (output / "endpoints.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=tuple(asdict(ordered[0])))
        writer.writeheader()
        writer.writerows(asdict(endpoint) for endpoint in ordered)

    initial_fits = {
        width: fit_gameplay_learning_curve(
            DATA_POSITIONS[:3],
            [endpoints[width, update].net_stock_per_min for update in DATA_UPDATES[:3]],
        )
        for width in widths
    }
    predicted_d31 = {width: initial_fits[width].predict(DATA_POSITIONS[-1]) for width in widths}
    observed_d31 = {width: endpoints[width, DATA_UPDATES[-1]].net_stock_per_min for width in widths}
    predicted_order = _response_order(predicted_d31)
    observed_order = _response_order(observed_d31)
    holdout_passed = predicted_order == observed_order

    refits: dict[int, LearningCurveFit] = {}
    predicted_d34: dict[int, float] = {}
    if holdout_passed:
        refits = {
            width: fit_gameplay_learning_curve(
                DATA_POSITIONS,
                [endpoints[width, update].net_stock_per_min for update in DATA_UPDATES],
            )
            for width in widths
        }
        predicted_d34 = {width: refits[width].predict(2**34) for width in widths}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4.5))
    for exponent, update in zip(DATA_EXPONENTS, DATA_UPDATES, strict=True):
        axis.plot(
            [latency_rows[width].p99_seconds * 1e3 for width in widths],
            [endpoints[width, update].net_stock_per_min for width in widths],
            marker="o",
            label=f"D=2^{exponent}",
        )
    axis.set_xlabel("RTX 3060 eager B1 p99 latency (ms)")
    axis.set_ylabel("Mean net stocks/min")
    axis.set_title("O54 gameplay strength vs deployed latency")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "iso_data_latency.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4.5))
    fit_exponents = np.linspace(DATA_EXPONENTS[0], 34 if holdout_passed else DATA_EXPONENTS[-1], 200)
    for width in widths:
        axis.scatter(
            DATA_EXPONENTS,
            [endpoints[width, update].net_stock_per_min for update in DATA_UPDATES],
            label=f"W{width} observations",
        )
        fit = refits.get(width, initial_fits[width])
        axis.plot(fit_exponents, [fit.predict(2**value) for value in fit_exponents], label=f"W{width} fit")
    axis.axvline(31, color="black", linestyle="--", linewidth=1, alpha=0.6)
    axis.set_xlabel("log2 supervised positions D")
    axis.set_ylabel("Mean net stocks/min")
    axis.set_title("O54 gameplay learning curves")
    axis.grid(alpha=0.25)
    axis.legend(fontsize="small", ncol=2)
    figure.tight_layout()
    figure.savefig(output / "learning_curves.png", dpi=180)
    plt.close(figure)

    report: dict[str, object] = {
        "schema": 2,
        "primary_response": "mean net stocks/min",
        "learning_curve": "S(D)=S_28+A*(1-(D/2^28)^-alpha), A>=0, 0.01<=alpha<=4",
        "deployed_models": {
            str(width): {
                "bc_parameters": PARAMETER_COUNT_CONTRACTS[width]["total"],
                "effective_parameters": effective_parameter_count(PARAMETER_COUNT_CONTRACTS[width]),
                "latency_ms": latency_rows[width].p99_seconds * 1e3,
                "delay_frames": latency_rows[width].inference_delay_frames,
            }
            for width in widths
        },
        "observed_winner_by_data_scale": {
            f"2^{exponent}": _response_order({width: endpoints[width, update].net_stock_per_min for width in widths})[
                0
            ]
            for exponent, update in zip(DATA_EXPONENTS, DATA_UPDATES, strict=True)
        },
        "fit_through_d30": {
            str(width): {
                **asdict(initial_fits[width]),
                "predicted_d31": predicted_d31[width],
                "observed_d31": observed_d31[width],
            }
            for width in widths
        },
        "d31_holdout": {
            "predicted_order": list(predicted_order),
            "observed_order": list(observed_order),
            "passed": holdout_passed,
        },
        "d34": (
            {
                "reported": True,
                "predicted_net_stock_per_min": {str(width): predicted_d34[width] for width in widths},
                "winner_width": _response_order(predicted_d34)[0],
                "winner_bc_parameters": PARAMETER_COUNT_CONTRACTS[_response_order(predicted_d34)[0]]["total"],
                "winner_latency_ms": latency_rows[_response_order(predicted_d34)[0]].p99_seconds * 1e3,
                "refit_all_points": {str(width): asdict(refits[width]) for width in widths},
            }
            if holdout_passed
            else {
                "reported": False,
                "reason": "the through-D=2^30 curve fits did not predict the observed D=2^31 ordering",
            }
        ),
    }
    if match_rows is not None:
        report["block_bootstrap"] = bootstrap_learning_curve_analysis(
            endpoints,
            match_rows,
            widths,
            holdout_passed=holdout_passed,
            resamples=bootstrap_resamples,
        )
    (output / "analysis.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report


def model_tag(cfg: TrainConfig) -> str:
    offsets = "-".join(map(str, cfg.arch.head_offsets))
    return (
        f"bc054-d{cfg.arch.d_model}-L{cfg.arch.n_layers}-h{cfg.arch.n_heads}-Lc{cfg.arch.L_ctx}-"
        f"t{cfg.arch.temporal_d_model}x{cfg.arch.temporal_layers}-o{offsets}-"
        f"d{cfg.delay_frames}r{cfg.replan_interval_frames}h{cfg.prediction_frames}-"
        f"nonlinear-head-trunk-skip-projectiles-v8-all-adamw-"
        f"alr{cfg.adam_lr:g}-awd{cfg.adam_weight_decay:g}-wu{cfg.warmup_steps}"
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


def data_selection(cfg: TrainConfig) -> PhysicalShardSelection:
    """Return the frozen deduplicated all-44 policy-world-v8 replay order."""
    expected = tuple(source.name for source in streams.POLICY_WORLD_V8_SOURCES)
    if cfg.source_names != expected:
        raise ValueError("O54 selection requires the all-44 policy-world-v8 source order")
    sources = tuple(
        SourceRowSelection(
            name,
            streams.POLICY_WORLD_V8_TRAIN_REPLAYS[name],
            DEDUPLICATED_SOURCE_ROWS.get(name, ()),
        )
        for name in cfg.source_names
    )
    selection = PhysicalShardSelection.from_sources(sources)
    if selection.sha256 != cfg.selection_sha256:
        raise RuntimeError(f"policy-world-v8 selection hash changed: {selection.sha256} != {cfg.selection_sha256}")
    if selection.row_count != cfg.train_replays:
        raise RuntimeError(f"policy-world-v8 selection has {selection.row_count} rows, expected {cfg.train_replays}")
    return selection


def _proportional_source_row_counts(cfg: TrainConfig, target_replays: int) -> tuple[int, ...]:
    """Allocate an exact all-source prefix with largest-remainder rounding."""
    available = tuple(
        streams.POLICY_WORLD_V8_TRAIN_REPLAYS[name] - len(DEDUPLICATED_SOURCE_ROWS.get(name, ()))
        for name in cfg.source_names
    )
    total = sum(available)
    if not 1 <= target_replays <= total:
        raise ValueError(f"target replay count must be in [1, {total}], got {target_replays}")
    numerators = tuple(target_replays * count for count in available)
    counts = [numerator // total for numerator in numerators]
    remainder = target_replays - sum(counts)
    order = sorted(range(len(counts)), key=lambda index: (-(numerators[index] % total), index))
    for index in order[:remainder]:
        counts[index] += 1
    if sum(counts) != target_replays or any(count > limit for count, limit in zip(counts, available, strict=True)):
        raise RuntimeError("proportional replay allocation is invalid")
    return tuple(counts)


def _raw_prefix_stop(unique_rows: int, excluded_rows: tuple[int, ...]) -> int:
    """Convert a unique-row prefix length to its exclusive physical-row stop."""
    stop = unique_rows
    for row in excluded_rows:
        if row < stop:
            stop += 1
    return stop


def _source_unique_range(source: str, unique_start: int, unique_stop: int) -> SourceRowSelection:
    exclusions = DEDUPLICATED_SOURCE_ROWS.get(source, ())
    start = _raw_prefix_stop(unique_start, exclusions)
    stop = _raw_prefix_stop(unique_stop, exclusions)
    selected_exclusions = tuple(row for row in exclusions if start <= row < stop)
    return SourceRowSelection(source, stop, selected_exclusions, start)


def cumulative_data_selection(cfg: TrainConfig, data_exponent: int) -> PhysicalShardSelection:
    """Return the nested replay prefix exposed through one data endpoint."""
    try:
        target_replays = DATA_REPLAYS[DATA_EXPONENTS.index(data_exponent)]
    except ValueError as error:
        raise ValueError(f"data exponent must be one of {DATA_EXPONENTS}, got {data_exponent}") from error
    counts = _proportional_source_row_counts(cfg, target_replays)
    selection = PhysicalShardSelection.from_sources(
        tuple(_source_unique_range(name, 0, stop) for name, stop in zip(cfg.source_names, counts, strict=True))
    )
    if selection.row_count != target_replays:
        raise RuntimeError("nested cumulative selection has the wrong replay count")
    return selection


def phase_data_selection(cfg: TrainConfig, phase_index: int) -> PhysicalShardSelection:
    """Return only the new replay rows admitted during one nested-data phase."""
    if not 0 <= phase_index < len(DATA_EXPONENTS):
        raise ValueError(f"data phase index must be in [0, {len(DATA_EXPONENTS)}), got {phase_index}")
    stops = _proportional_source_row_counts(cfg, DATA_REPLAYS[phase_index])
    starts = (
        (0,) * len(cfg.source_names)
        if phase_index == 0
        else _proportional_source_row_counts(cfg, DATA_REPLAYS[phase_index - 1])
    )
    selection = PhysicalShardSelection.from_sources(
        tuple(
            _source_unique_range(name, start, stop)
            for name, start, stop in zip(cfg.source_names, starts, stops, strict=True)
        )
    )
    if selection.row_count != DATA_PHASE_REPLAYS[phase_index]:
        raise RuntimeError("nested phase selection has the wrong replay count")
    return selection


def data_phase_for_completed_updates(completed_updates: int) -> int:
    """Return the phase that owns the next update, or the terminal phase count."""
    if not 0 <= completed_updates <= DATA_UPDATES[-1]:
        raise ValueError(f"completed updates must be in [0, {DATA_UPDATES[-1]}]")
    return next(
        (index for index, endpoint in enumerate(DATA_UPDATES) if completed_updates < endpoint),
        len(DATA_UPDATES),
    )


_NESTED_LOADER_STATE_SCHEMA: Final[int] = 1
_TRAINING_RNG_STATE_SCHEMA: Final[int] = 1


def capture_training_rng_state() -> dict[str, object]:
    """Capture every process RNG that can affect an O54 update."""
    return {
        "schema": _TRAINING_RNG_STATE_SCHEMA,
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda": (
            tuple(state.cpu() for state in torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else None
        ),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_training_rng_state(state: object) -> None:
    """Restore a complete O54 process-RNG checkpoint."""
    if not isinstance(state, Mapping):
        raise ValueError("resume checkpoint has no complete O54 RNG state")
    typed_state = cast(Mapping[str, object], state)
    if set(typed_state) != {
        "schema",
        "torch_cpu",
        "torch_cuda",
        "numpy",
        "python",
    }:
        raise ValueError("resume checkpoint has no complete O54 RNG state")
    if typed_state["schema"] != _TRAINING_RNG_STATE_SCHEMA:
        raise ValueError("resume checkpoint has an unsupported O54 RNG-state schema")
    cpu = typed_state["torch_cpu"]
    cuda = typed_state["torch_cuda"]
    if not isinstance(cpu, Tensor):
        raise ValueError("resume checkpoint has an invalid Torch CPU RNG state")
    if torch.cuda.is_available():
        if (
            not isinstance(cuda, tuple)
            or len(cuda) != torch.cuda.device_count()
            or not all(isinstance(item, Tensor) for item in cuda)
        ):
            raise ValueError("resume checkpoint has no complete Torch CUDA RNG state")
        cuda_states = cast(tuple[Tensor, ...], cuda)
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_states])
    elif cuda is not None:
        raise ValueError("cannot exactly resume a CUDA checkpoint without CUDA")
    torch.set_rng_state(cpu.cpu())
    try:
        np.random.set_state(cast(tuple, typed_state["numpy"]))
        random.setstate(cast(tuple, typed_state["python"]))
    except (TypeError, ValueError) as error:
        raise ValueError("resume checkpoint has an invalid Python or NumPy RNG state") from error


def nested_loader_checkpoint_state(
    cfg: TrainConfig,
    *,
    update: int,
    phase_index: int,
    loader_state: dict[str, object],
) -> dict[str, object]:
    """Persist an exact phase-local loader state without admitting future rows."""
    expected_phase = data_phase_for_completed_updates(update - 1)
    if phase_index != expected_phase:
        raise ValueError(f"update {update} belongs to data phase {expected_phase}, got {phase_index}")
    phase_complete = update in DATA_UPDATES
    next_phase = phase_index + 1 if phase_complete else phase_index
    return {
        "schema": _NESTED_LOADER_STATE_SCHEMA,
        "completed_updates": update,
        "phase_index": next_phase,
        "phase_selection_sha256": (
            None if next_phase == len(DATA_UPDATES) else phase_data_selection(cfg, next_phase).sha256
        ),
        "physical_loader": None if phase_complete else loader_state,
    }


def resume_data_phase(
    cfg: TrainConfig,
    resume_state: dict[str, object] | None,
) -> tuple[int, dict[str, object] | None]:
    """Validate and return the phase-local physical-loader resume state."""
    if resume_state is None:
        return 0, None
    raw_step = resume_state["step"]
    if not isinstance(raw_step, int) or isinstance(raw_step, bool):
        raise ValueError("resume checkpoint has an invalid optimizer step")
    completed_updates = raw_step + 1
    expected_phase = data_phase_for_completed_updates(completed_updates)
    raw = resume_state.get("loader")
    if not isinstance(raw, Mapping):
        raise ValueError("resume checkpoint does not contain nested O54 replay-loader state")
    typed_raw = cast(Mapping[str, object], raw)
    if typed_raw.get("schema") != _NESTED_LOADER_STATE_SCHEMA:
        raise ValueError("resume checkpoint does not contain nested O54 replay-loader state")
    if typed_raw.get("completed_updates") != completed_updates or typed_raw.get("phase_index") != expected_phase:
        raise ValueError("resume checkpoint data phase does not match its optimizer update")
    expected_sha = None if expected_phase == len(DATA_UPDATES) else phase_data_selection(cfg, expected_phase).sha256
    if typed_raw.get("phase_selection_sha256") != expected_sha:
        raise ValueError("resume checkpoint data-phase selection changed")
    physical = typed_raw.get("physical_loader")
    if expected_phase == len(DATA_UPDATES):
        if physical is not None:
            raise ValueError("terminal replay-loader state must not contain a physical loader")
        return expected_phase, None
    if completed_updates in DATA_UPDATES:
        if physical is not None:
            raise ValueError("a completed data phase must not retain prefetched replay state")
        return expected_phase, None
    if not isinstance(physical, dict):
        raise ValueError("within-phase resume has no physical replay-loader state")
    return expected_phase, cast(dict[str, object], physical)


def source_mixture_weights(cfg: TrainConfig) -> tuple[float, ...]:
    """Return the deduplicated natural replay-count mixture."""
    return tuple(
        float(streams.POLICY_WORLD_V8_TRAIN_REPLAYS[name] - len(DEDUPLICATED_SOURCE_ROWS.get(name, ())))
        for name in cfg.source_names
    )


def source_manifest_sha256(cfg: TrainConfig) -> dict[str, str]:
    """Return the pinned manifests for the selected all-44 sources."""
    return {name: streams.POLICY_WORLD_V8_TRAIN_MANIFEST_SHA256[name] for name in cfg.source_names}


def source_replay_counts(cfg: TrainConfig) -> dict[str, int]:
    """Return the pinned replay counts for the selected all-44 sources."""
    return {name: streams.POLICY_WORLD_V8_TRAIN_REPLAYS[name] for name in cfg.source_names}


def validate_batch_geometry(batch: TrainBatch, cfg: TrainConfig, expected_batch_size: int | None = None) -> None:
    if batch.target.shape[1:] != (cfg.arch.sample_chunk_length, A_DIM):
        raise ValueError(
            f"target must be [B, {cfg.arch.sample_chunk_length}, {A_DIM}], got {tuple(batch.target.shape)}"
        )
    batch_size = batch.target.shape[0]
    if expected_batch_size is not None and batch_size != expected_batch_size:
        raise ValueError(f"fixed training batch must contain {expected_batch_size} rows, got {batch_size}")
    if batch.context.ctx_pad.shape != (batch_size,):
        raise ValueError("ctx_pad shape does not match the batch")
    wrong = {
        name: tuple(value.shape)
        for name, value in batch.context.features.items()
        if value.shape[:2] != (batch_size, cfg.arch.L_ctx)
    }
    if wrong:
        raise ValueError(f"context features have the wrong geometry: {wrong}")


def cache_validation(loader: Iterable[TrainBatch], n_samples: int) -> list[TrainBatch]:
    print(f"[validation] caching {n_samples:,} samples", flush=True)
    started = time.monotonic()
    last_progress_log = started
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
        now = time.monotonic()
        if count < n_samples and now - last_progress_log >= _STARTUP_LOG_INTERVAL_S:
            print(
                f"[validation] cached {count:,}/{n_samples:,} samples; {now - started:.1f}s elapsed",
                flush=True,
            )
            last_progress_log = now
    if count != n_samples:
        raise RuntimeError(f"validation yielded {count} samples, expected {n_samples}")
    print(f"[validation] cache complete: {count:,} samples in {time.monotonic() - started:.1f}s", flush=True)
    return batches


def save_boundary_checkpoint(
    run_dir: Path,
    *,
    update: int,
    model: GPT,
    optimizer: torch.optim.AdamW,
    scheduler: LambdaLR,
    cfg: TrainConfig,
    uploader: BackgroundUploader | None,
    milestone: bool,
    wandb_id: str | None,
    actual_loss_positions: int,
    loader_state: dict[str, object],
    identity_masker_state: dict[str, object] | None = None,
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
        extra_state={
            "actual_loss_positions": actual_loss_positions,
            "loader": loader_state,
            "identity_masker": (
                loader_state.get("identity_masker") if identity_masker_state is None else identity_masker_state
            ),
            "rng": capture_training_rng_state(),
        },
    )
    os.replace(temporary, snapshot)
    latest = run_dir / "latest.pt"
    advance_checkpoint_link(snapshot, latest)
    if uploader is not None:
        uploader.upload(snapshot, key="latest.pt")
        if milestone:
            uploader.upload(snapshot, key=f"checkpoints/step-{update:07d}.pt")
    return snapshot


def load_stats(cfg: TrainConfig) -> dict[str, FeatureStats]:
    sources = tuple(streams.BY_NAME[name] for name in cfg.source_names)
    return load_consolidated_mixture_stats(
        [source.local_root / "stats.json" for source in sources],
        source_mixture_weights(cfg),
        expected_mds_schema_version=cfg.mds_schema_version,
    )


def load_identity_sidecar(cfg: TrainConfig) -> PlayerIdentitySidecar:
    return load_player_identity_artifact(
        Path(cfg.player_sidecar_local),
        remote=cfg.player_sidecar_remote,
        expected_sha256=cfg.player_sidecar_sha256,
        expected_vocabulary_size=cfg.player_vocab_size,
        expected_vocabulary_sha256=cfg.player_vocab_sha256,
    )


def _collate_o54_batch(
    replay_ids: tuple[str, ...],
    columns: Mapping[str, np.ndarray],
    *,
    stats: dict[str, FeatureStats],
    projection: FeatureProjection,
    context_length: int,
) -> TrainBatch:
    batch = train_batch_from_columns(
        columns,
        stats=stats,
        L_ctx=context_length,
        extra=ITEM_PLAYER_COLUMNS,
        projection=projection,
    )
    return TrainBatch(batch.context, batch.target, replay_ids)


def _require_loader_disk(loader: PhysicalShardReplayLoader[TrainBatch]) -> None:
    required = loader.required_disk_bytes
    available = loader.disk_free_bytes
    if available < required:
        raise RuntimeError(
            "policy-world-v8 raw shards and the 256 GiB reserve do not fit: "
            f"required={required / 2**30:.1f} GiB, free={available / 2**30:.1f} GiB"
        )


def _make_train_loader(
    cfg: TrainConfig,
    stats: dict[str, FeatureStats],
    player_lookup: ReplayPlayerLookup,
    phase_index: int,
) -> PhysicalShardReplayLoader[TrainBatch]:
    selection = phase_data_selection(cfg, phase_index)
    adapter = MDSStorageAdapter(selection, download_retry=cfg.download_retry)
    adapter.validate_manifests(
        expected_sha256=source_manifest_sha256(cfg),
        expected_index_version=cfg.mds_index_version,
        expected_schema_sha256=cfg.mds_manifest_schema_sha256,
        expected_rows=source_replay_counts(cfg),
    )
    projection = ITEM_PLAYER_PROJECTION
    train_loader = PhysicalShardReplayLoader[TrainBatch](
        selection=selection,
        adapter=adapter,
        tasks=build_shard_plan(selection, adapter.manifests),
        data_protocol=cfg.data_protocol,
        source_manifest_sha256=source_manifest_sha256(cfg),
        batch_transform=functools.partial(
            _collate_o54_batch,
            stats=stats,
            projection=projection,
            context_length=cfg.arch.L_ctx,
        ),
        batch_size=cfg.batch_size,
        replay_slots=cfg.replay_slots,
        seed=cfg.seed,
        num_workers=cfg.num_workers,
        labels=player_lookup,
        projection=projection,
        context_length=cfg.arch.L_ctx,
        chunk_length=cfg.arch.sample_chunk_length,
        windows_per_generation=cfg.windows_per_generation,
        generations_per_replay=cfg.generations_per_replay,
        replay_phase_block_batches=cfg.replay_phase_block_batches,
        schema_version=cfg.mds_schema_version,
        reserved_disk_bytes=cfg.reserved_disk_bytes,
        pin_memory=torch.cuda.is_available(),
        materialization_threads=cfg.materialization_threads(),
    )
    try:
        _require_loader_disk(train_loader)
        if sum(train_loader.source_sample_counts.values()) != DATA_PHASE_REPLAYS[phase_index]:
            raise ValueError("physical-shard loader does not expose every selected nested-phase row")
        if train_loader.minimum_replay_gap_batches < cfg.minimum_replay_gap_batches:
            raise ValueError(
                f"replay ring is too small for the {cfg.minimum_replay_gap_batches}-batch reuse-gap contract"
            )
    except Exception:
        train_loader.close()
        raise
    return train_loader


def _make_loaders(
    cfg: TrainConfig,
    stats: dict[str, FeatureStats],
    phase_index: int,
    player_lookup: ReplayPlayerLookup | None = None,
) -> tuple[PhysicalShardReplayLoader[TrainBatch], list[TrainBatch]]:
    """Build physical-shard training and the unchanged generic validation cohort."""
    if player_lookup is None:
        player_lookup = ReplayPlayerLookup(load_identity_sidecar(cfg).by_replay)
    train_loader = _make_train_loader(cfg, stats, player_lookup, phase_index)
    try:
        val_loader = make_loader(
            data_root=None,
            split=cfg.val_split,
            stats=stats,
            L_ctx=cfg.arch.L_ctx,
            L_chunk=cfg.arch.sample_chunk_length,
            batch_size=cfg.val_batch_size,
            seed=cfg.seed,
            sources=tuple(streams.BY_NAME[name] for name in cfg.source_names),
            cache_limit="1792gb",
            shuffle_block_size=8192,
            shuffle_seed=cfg.seed,
            num_workers=0,
            schema_version=cfg.mds_schema_version,
            extra=ITEM_PLAYER_COLUMNS,
            projection=ITEM_PLAYER_PROJECTION,
            replay_format="policy-world",
            replay_labels=player_lookup,
            require_full_context=True,
            shuffle=True,
        )
        validation = cache_validation(val_loader, cfg.val_n_samples)
    except Exception:
        train_loader.close()
        raise
    return train_loader, validation


@dataclass(slots=True)
class PreparedTrainingData:
    phase_index: int
    loader: PhysicalShardReplayLoader[TrainBatch]
    validation: list[TrainBatch]
    iterator: Iterator[TrainBatch]
    first_batch_future: Future[TrainBatch]
    resources: ExitStack
    worker_start_seconds: float
    first_batch_seconds: list[float]


def _prepare_training_data(
    cfg: TrainConfig,
    stats: dict[str, FeatureStats],
    sidecar: PlayerIdentitySidecar,
    resume_state: dict[str, object] | None,
) -> PreparedTrainingData:
    """Start shard workers and the first batch before CUDA allocation."""
    phase_index, physical_resume_state = resume_data_phase(cfg, resume_state)
    if phase_index == len(DATA_UPDATES):
        raise ValueError("cannot resume a completed nested-data trajectory")
    train_loader, validation = _make_loaders(
        cfg,
        stats,
        phase_index,
        ReplayPlayerLookup(sidecar.by_replay),
    )
    try:
        if physical_resume_state is not None:
            train_loader.load_state_dict(physical_resume_state)
        worker_started = time.monotonic()
        train_iterator = iter(train_loader)
        worker_start_seconds = time.monotonic() - worker_started
    except Exception:
        train_loader.close()
        raise
    first_batch_seconds: list[float] = []

    def load_first_batch() -> TrainBatch:
        started = time.monotonic()
        batch = next(train_iterator)
        first_batch_seconds.append(time.monotonic() - started)
        return batch

    try:
        with ExitStack() as setup:
            executor = setup.enter_context(ThreadPoolExecutor(max_workers=1, thread_name_prefix="o50-first-batch"))
            first_batch = executor.submit(load_first_batch)
            resources = setup.pop_all()
    except Exception:
        train_loader.close()
        raise
    return PreparedTrainingData(
        phase_index,
        train_loader,
        validation,
        train_iterator,
        first_batch,
        resources,
        worker_start_seconds,
        first_batch_seconds,
    )


def _init_wandb(cfg: TrainConfig, run_name: str, resume_state: dict | None) -> None:
    """Start tracking and declare the experiment's logging semantics."""
    selection = data_selection(cfg)
    cumulative_selections = tuple(cumulative_data_selection(cfg, exponent) for exponent in DATA_EXPONENTS)
    phase_selections = tuple(phase_data_selection(cfg, index) for index in range(len(DATA_EXPONENTS)))
    wandb.init(
        project="hal",
        name=run_name,
        id=None if resume_state is None else resume_state.get("wandb_id"),
        resume="allow" if resume_state is not None else None,
        tags=[
            "gpt",
            "temporal-mtp",
            "behavior-cloning",
            "scaled",
            "054",
            "capacity-latency",
            "projectiles",
            "all-44-v8",
            "all-adamw",
            "powerlines-weight-decay",
            "constant-lr",
        ],
        config={
            "experiment_id": _EXPERIMENT_ID,
            **asdict(cfg),
            "max_steps": cfg.max_steps,
            "warmup_steps": cfg.warmup_steps,
            "adam_weight_decay": cfg.adam_weight_decay,
            "powerlines_exponent": POWERLINES_EXPONENT,
            "powerlines_reference_width": POWERLINES_REFERENCE_WIDTH,
            "powerlines_reference_positions": POWERLINES_REFERENCE_POSITIONS,
            "powerlines_reference_weight_decay": POWERLINES_REFERENCE_WEIGHT_DECAY,
            "data_protocol": cfg.data_protocol,
            "source_selection_sha256": selection.sha256,
            "nested_cumulative_selection_sha256": [item.sha256 for item in cumulative_selections],
            "nested_phase_selection_sha256": [item.sha256 for item in phase_selections],
            "train_microbatch_size": train_microbatch_size(cfg),
            "data_exponents": DATA_EXPONENTS,
            "data_endpoint_updates": DATA_UPDATES,
            "data_endpoint_replays": DATA_REPLAYS,
            "source_manifest_sha256": source_manifest_sha256(cfg),
            "mds_manifest_schema_sha256": cfg.mds_manifest_schema_sha256,
        },
        settings=wandb.Settings(
            mode="shared",
            x_label="training",
            x_primary=True,
            x_stats_sampling_interval=5.0,
            x_stats_track_process_tree=True,
        ),
    )
    if wandb.run is None:
        return
    wandb.define_metric("global_step")
    wandb.define_metric("*", step_metric="global_step")
    wandb.define_metric("eval/*", step_metric="global_step", summary="none")
    wandb.define_metric("eval/checkpoint_step", step_metric="global_step", summary="max")
    wandb.run.summary["nll_semantics"] = (
        "train/loss is the normalized offset-weighted behavior-cloning objective in bits"
    )
    wandb.run.summary["architecture/treatment"] = (
        f"O54 d{cfg.arch.d_model} L{cfg.arch.n_layers} pure BC over all-44 policy-world-v8 with "
        "O52 representation, initialization, centered logits, and AdamW"
    )
    wandb.run.summary["optimizer/adam_update_clip_threshold"] = None
    wandb.run.summary["optimizer/lr_schedule"] = "512-update linear warmup then constant"
    wandb.run.summary["optimizer/weight_decay_treatment"] = "fixed at the D=2^31 endpoint for the full trajectory"
    wandb.run.summary["optimizer/update_clip_semantics"] = "global pre-step gradient norm clipping only"
    wandb.run.summary["training/optimizer_batch_size"] = cfg.batch_size
    wandb.run.summary["training/microbatch_size"] = train_microbatch_size(cfg)
    wandb.run.summary["stability/microbatch_p999_aggregation"] = "maximum of per-microbatch p99.9 estimates"
    wandb.run.summary["data/nesting"] = (
        "each phase reads only its disjoint source-row range; cumulative replay unions are nested"
    )
    wandb.run.summary["data/representation_metadata"] = (
        "frozen O52 all-44 normalization statistics and player vocabulary; no future-phase replay enters a batch"
    )
    wandb.run.summary["evaluation/automatic"] = cfg.automatic_evaluation
    if cfg.wandb_log_code:
        log_wandb_code(wandb.run)


def _log_training_summary(
    cfg: TrainConfig,
    parameter_counts: dict[str, int],
    train_loader: PhysicalShardReplayLoader[TrainBatch],
    *,
    flops_per_update: int,
    device_name: str | None,
    peak_flops: float | None,
) -> None:
    """Record the fixed model and corpus accounting for this run."""
    if wandb.run is None:
        return
    for name, value in parameter_counts.items():
        wandb.run.summary[f"parameters/{name}"] = value

    unique_replays = DATA_REPLAYS[-1]
    source_weights = source_mixture_weights(cfg)
    source_weight_total = sum(source_weights)
    wandb.run.summary["data/unique_replays"] = unique_replays
    wandb.run.summary["data/reference_corpus_replays"] = cfg.train_replays
    wandb.run.summary["data/reference_corpus_frames"] = cfg.train_frames
    wandb.run.summary["data/source_list_sha256"] = cfg.source_list_sha256
    wandb.run.summary["data/reference_corpus_selection_sha256"] = cfg.selection_sha256
    wandb.run.summary["data/initial_phase_selection_sha256"] = train_loader.selection.sha256
    wandb.run.summary["data/mds_manifest_schema_sha256"] = cfg.mds_manifest_schema_sha256
    wandb.run.summary["data/loader_protocol"] = cfg.data_protocol
    supervised_positions = cfg.max_steps * cfg.batch_size * (cfg.arch.L_ctx - cfg.arch.direct_loss_start)
    wandb.run.summary["data/processed_loss_positions"] = supervised_positions
    wandb.run.summary["data/D_over_N"] = supervised_positions / parameter_counts["total"]
    wandb.run.summary["data/nominal_loss_positions_per_update"] = cfg.batch_size * (
        cfg.arch.L_ctx - cfg.arch.direct_loss_start
    )
    wandb.run.summary["data/cpu_lookahead_batches"] = cfg.train_prefetch_factor
    wandb.run.summary["data/loader_prefetch_factor"] = PREFETCH_FACTOR
    wandb.run.summary["data/raw_shard_materialization_threads"] = train_loader.materialization_threads
    wandb.run.summary["data/replay_slots"] = train_loader.replay_slots
    wandb.run.summary["data/generation_windows"] = cfg.windows_per_generation
    wandb.run.summary["data/generations_per_replay"] = cfg.generations_per_replay
    wandb.run.summary["data/epoch_semantics"] = "phase-local replay generations committed to ring / phase replays"
    wandb.run.summary["data/replay_phase_block_batches"] = cfg.replay_phase_block_batches
    wandb.run.summary["data/minimum_replay_gap_batches"] = train_loader.minimum_replay_gap_batches
    wandb.run.summary["system/disk/required_bytes"] = train_loader.required_disk_bytes
    wandb.run.summary["system/disk/free_bytes_at_start"] = train_loader.disk_free_bytes
    wandb.run.summary["system/disk/reserved_bytes"] = cfg.reserved_disk_bytes
    wandb.run.summary["training/approx_flops_per_update"] = flops_per_update
    wandb.run.summary["training/N_eff"] = effective_parameter_count(parameter_counts)
    wandb.run.summary["training/C"] = training_compute(cfg, parameter_counts)
    wandb.run.summary["training/flops_formula"] = "6*D*(2*(N_trunk+N_inputs)+14*(N_temporal+N_group_heads))"
    wandb.run.summary["optimizer/name"] = "AdamW"
    wandb.run.summary["optimizer/all_parameters_use_adamw"] = True
    wandb.run.summary["optimizer/implementation"] = "torch.optim.AdamW"
    wandb.run.summary["optimizer/fused"] = DEVICE == "cuda"
    wandb.run.summary["optimizer/adam_master_lr"] = cfg.adam_lr
    wandb.run.summary["optimizer/adam_input_lr"] = cfg.adam_lr
    wandb.run.summary["optimizer/adam_vector_lr"] = cfg.adam_lr
    wandb.run.summary["optimizer/adam_output_lr"] = _role_lr(OptimizerRole("output", True, 4), cfg)
    wandb.run.summary["optimizer/adam_betas"] = (cfg.adam_beta1, cfg.adam_beta2)
    wandb.run.summary["optimizer/adam_epsilon"] = cfg.adam_eps
    wandb.run.summary["optimizer/adam_weight_decay"] = cfg.adam_weight_decay
    wandb.run.summary["optimizer/weight_decay_scaling"] = "Power Lines D/N timescale, exponent 0.52"
    if device_name is not None:
        wandb.run.summary["hardware/gpu_name"] = device_name
    if peak_flops is not None:
        wandb.run.summary["hardware/bf16_dense_peak_tflops"] = peak_flops / 1e12
        source = bf16_peak_source(device_name or "")
        if source is not None:
            wandb.run.summary["hardware/bf16_dense_peak_source"] = source
    wandb.run.summary["data/source_mixing"] = "identity_uniform_physical_shards"
    for name, weight in zip(cfg.source_names, source_weights, strict=True):
        wandb.run.summary[f"data/source_sampling_share/{name}"] = weight / source_weight_total


def _training_functions(model: GPT, cfg: TrainConfig) -> tuple[Callable, Callable]:
    """Return eager or singly compiled trunk and temporal training functions."""
    trunk_fn: Callable = model.forward
    temporal_fn: Callable = model.temporal.teacher_forced_nll_with_diagnostics
    compile_mode = _training_compile_mode(cfg)
    graph_note = " (CUDA graphs disabled for gradient accumulation)" if compile_mode != cfg.train_compile_mode else ""
    if DEVICE == "cuda" and cfg.compile_trunk:
        # Resolve FlexAttention before Dynamo sees the model. This entrypoint is
        # the sole compilation owner for the raw mask and attention operations.
        model.trunk.resolve_attention(DEVICE)
        if model.trunk.attn_path not in ("flex", "varlen_flash"):
            raise RuntimeError(
                f"compiled CUDA training requires a fused attention path, resolved {model.trunk.attn_path!r} instead"
            )
        print(f"[compile] calling torch.compile for trunk (mode={compile_mode}){graph_note}", flush=True)
        trunk_fn = torch.compile(
            trunk_fn,
            dynamic=False,
            fullgraph=True,
            mode=compile_mode,
        )
    if DEVICE == "cuda" and cfg.compile_temporal:
        print(
            f"[compile] calling torch.compile for temporal model (mode={compile_mode}){graph_note}",
            flush=True,
        )
        temporal_fn = torch.compile(
            temporal_fn,
            dynamic=False,
            fullgraph=True,
            mode=compile_mode,
        )
    return trunk_fn, temporal_fn


@dataclass(frozen=True, slots=True)
class TrainStepResult:
    nll_sum: Tensor
    gradient_norm: Tensor
    metrics: dict[str, Tensor]
    adam_lr: float


@dataclass(slots=True)
class _TrainingMetricAccumulator:
    """Accumulate device metrics and download one payload per log window."""

    _sum: Tensor | None = None
    _metric_names: tuple[str, ...] = ()
    updates: int = 0
    valid_prefixes: int = 0

    def add(self, result: TrainStepResult, valid_prefixes: int) -> None:
        if valid_prefixes <= 0:
            raise ValueError("valid_prefixes must be positive")
        metric_names = tuple(result.metrics)
        if self._metric_names and metric_names != self._metric_names:
            raise RuntimeError(f"training metric names changed from {self._metric_names} to {metric_names}")
        scalar_metrics = torch.stack(
            [result.gradient_norm.detach().float(), *(result.metrics[name].detach().float() for name in metric_names)]
        )
        payload = torch.cat((result.nll_sum.detach().reshape(-1).float(), scalar_metrics))
        if self._sum is None:
            self._sum = payload
            self._metric_names = metric_names
        else:
            self._sum.add_(payload)
        self.updates += 1
        self.valid_prefixes += valid_prefixes

    def flush(self, cfg: TrainConfig, *, update: int) -> tuple[dict[str, float], int, int]:
        """Synchronize once, return window means, and reset the accumulator."""
        if self._sum is None or self.updates == 0 or self.valid_prefixes == 0:
            raise RuntimeError("cannot flush an empty training metric accumulator")
        payload = self._sum.cpu()
        if not torch.isfinite(payload).all():
            raise FloatingPointError(f"update {update}: accumulated training metrics contain a non-finite value")

        nll_values = len(cfg.arch.head_offsets) * CONTROLLER_GROUP_COUNT
        mean_nll = (
            payload[:nll_values].reshape(len(cfg.arch.head_offsets), CONTROLLER_GROUP_COUNT) / self.valid_prefixes
        )
        scalar_values = payload[nll_values:] / self.updates
        nll_metrics = nll_mean_metrics(
            mean_nll,
            cfg.arch.head_offsets,
            prediction_frames=cfg.prediction_frames,
        )
        values = {
            "train/nll": nll_metrics["loss_unweighted"],
            "optimizer/grad_norm": float(scalar_values[0]),
        }
        values.update({name: float(value) for name, value in zip(self._metric_names, scalar_values[1:], strict=True)})

        updates = self.updates
        valid_prefixes = self.valid_prefixes
        self._sum = None
        self._metric_names = ()
        self.updates = 0
        self.valid_prefixes = 0
        return values, updates, valid_prefixes


def _merge_microbatch_metrics(
    accumulated: dict[str, Tensor],
    current: dict[str, Tensor],
    *,
    batch_fraction: float,
) -> None:
    """Merge execution-microbatch diagnostics without changing the loss."""
    sum_names = {"train/loss", "train/objective"}
    min_names = {"stability/button_pre_norm_rms_min"}
    max_names = {
        "stability/button_input_abs_p999",
        "stability/button_logit_abs_p999",
    }
    mean_names = {"stability/button_margin_mean"}
    expected = sum_names | min_names | max_names | mean_names
    if set(current) != expected:
        raise RuntimeError(f"microbatch metrics changed: expected {sorted(expected)}, got {sorted(current)}")
    for name, value in current.items():
        detached = value.detach()
        if name not in accumulated:
            accumulated[name] = detached.mul(batch_fraction) if name in mean_names else detached.clone()
        elif name in sum_names:
            accumulated[name].add_(detached)
        elif name in min_names:
            accumulated[name] = torch.minimum(accumulated[name], detached)
        elif name in max_names:
            accumulated[name] = torch.maximum(accumulated[name], detached)
        else:
            accumulated[name].add_(detached, alpha=batch_fraction)


def _backward_training_batch(
    model: GPT,
    batch: TrainBatch,
    cfg: TrainConfig,
    *,
    step: int,
    valid_prefixes: int,
    trunk_fn: Callable,
    temporal_fn: Callable,
    phase_timer: CudaPhaseTimer | None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Backpropagate one optimizer batch as one or more execution microbatches."""
    microbatches = training_microbatches(batch, cfg)
    profile = phase_timer if len(microbatches) == 1 else None
    nll_sum: Tensor | None = None
    metrics: dict[str, Tensor] = {}
    for microbatch in microbatches:
        if DEVICE == "cuda" and (cfg.compile_trunk or cfg.compile_temporal) and _training_uses_cuda_graphs(cfg):
            torch.compiler.cudagraph_mark_step_begin()
        loss, microbatch_nll_sum, microbatch_metrics = microbatch_loss(
            model,
            microbatch,
            cfg,
            step=step,
            valid_prefixes=valid_prefixes,
            trunk_fn=trunk_fn,
            temporal_fn=temporal_fn,
            phase_timer=profile,
        )
        loss.backward()
        nll_sum = microbatch_nll_sum.clone() if nll_sum is None else nll_sum.add(microbatch_nll_sum)
        _merge_microbatch_metrics(
            metrics,
            microbatch_metrics,
            batch_fraction=microbatch.target.shape[0] / batch.target.shape[0],
        )
    if nll_sum is None:
        raise RuntimeError("optimizer batch contained no execution microbatches")
    return nll_sum, metrics


def train_step(
    model: GPT,
    batch: TrainBatch,
    cfg: TrainConfig,
    *,
    step: int,
    update: int,
    valid_prefixes: int,
    trunk_fn: Callable,
    temporal_fn: Callable,
    optimizer: torch.optim.AdamW,
    scheduler: LambdaLR,
    phase_timer: CudaPhaseTimer | None = None,
) -> TrainStepResult:
    """Run one complete optimization step on a device-resident batch."""
    optimizer.zero_grad()
    nll_sum, metrics = _backward_training_batch(
        model,
        batch,
        cfg,
        step=step,
        valid_prefixes=valid_prefixes,
        trunk_fn=trunk_fn,
        temporal_fn=temporal_fn,
        phase_timer=phase_timer,
    )
    if phase_timer is not None:
        phase_timer.record("backward_end")
    metrics["stability/action_grad_abs_max"] = _button_gradient_abs_max(model)
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    metrics["optimizer/clip_fraction"] = (gradient_norm > cfg.grad_clip).float()
    if phase_timer is not None:
        phase_timer.record("grad_norm_end")
    adam_lr = float(max(group["lr"] for group in optimizer.param_groups))
    optimizer.step()
    scheduler.step()
    if phase_timer is not None:
        phase_timer.record("optimizer_end")
    return TrainStepResult(nll_sum, gradient_norm, metrics, adam_lr)


def _cadence_in_window(first_update: int, last_update: int, every: int) -> bool:
    """Return whether update one or a cadence boundary is in the window."""
    if every <= 0:
        return False
    return first_update == 1 or last_update // every > (first_update - 1) // every


def _mean_phase_metrics(timers: list[CudaPhaseTimer]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for timer in timers:
        for name, value in timer.metrics().items():
            totals[name] = totals.get(name, 0.0) + value
    return {name: value / len(timers) for name, value in totals.items()} if timers else {}


def _minimal_system_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Keep one host metric for each distinct resource constraint."""
    names = {
        "system/cgroup/current_gib": "system/memory_gb",
        "system/cgroup/usage_fraction": "system/memory_fraction",
        "system/process_tree/pss_gib": "system/process_memory_gb",
        "system/cache/allocated_gib": "system/cache_gb",
        "system/network/read_mib_s": "system/network/read_mib_s",
        "system/telemetry_errors": "system/telemetry_errors",
    }
    return {output: metrics[source] for source, output in names.items() if source in metrics}


def spawn_closed_loop_evaluation(
    run_name: str,
    update: int,
    expected_checkpoint_sha256: str,
    n_matchups: int,
) -> str:
    """Ask the launcher to spawn its same-app L40S evaluator."""
    raw_fd = os.environ.get("HAL_MODAL_EVAL_FD")
    if raw_fd is None:
        raise RuntimeError("HAL_MODAL_EVAL_FD is required for automatic closed-loop evaluation")
    try:
        fd = int(raw_fd)
    except ValueError as e:
        raise RuntimeError("HAL_MODAL_EVAL_FD must be a file descriptor") from e
    request = {
        "run_name": run_name,
        "update": update,
        "expected_checkpoint_sha256": expected_checkpoint_sha256,
        "n_matchups": n_matchups,
    }
    payload = json.dumps(request, separators=(",", ":")).encode() + b"\n"
    while payload:
        payload = payload[os.write(fd, payload) :]
    response = bytearray()
    while b"\n" not in response:
        chunk = os.read(fd, 4096)
        if not chunk:
            break
        response.extend(chunk)
    line, _, _remainder = response.partition(b"\n")
    if not line:
        raise RuntimeError("the Modal evaluation broker closed without a response")
    try:
        response = json.loads(line)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise RuntimeError("the Modal evaluation broker returned invalid JSON") from e
    if not isinstance(response, dict):
        raise RuntimeError("the Modal evaluation broker returned a non-object response")
    error = response.get("error")
    if isinstance(error, str):
        raise RuntimeError(f"the Modal evaluation broker rejected the request: {error}")
    call_id = response.get("function_call_id")
    if not isinstance(call_id, str) or re.fullmatch(r"fc-[A-Za-z0-9]+", call_id) is None:
        raise RuntimeError("the Modal evaluation broker returned an invalid FunctionCall ID")
    return call_id


def _finalize_training(
    *,
    model: GPT,
    optimizer: torch.optim.AdamW,
    scheduler: LambdaLR,
    cfg: TrainConfig,
    stats: dict[str, FeatureStats],
    val_cache: list[TrainBatch],
    run_dir: Path,
    replay_dir: Path,
    uploader: BackgroundUploader | None,
    loader_wait_fractions: list[float],
    loader_state: dict[str, object],
    identity_masker_state: dict[str, object],
    update: int,
    actual_loss_positions: int,
    smoke: bool,
) -> None:
    """Save the final model and queue evaluation for a separate L40S worker."""
    snapshot = save_boundary_checkpoint(
        run_dir,
        update=update,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        uploader=uploader,
        milestone=cfg.ckpt_every > 0 and update % cfg.ckpt_every == 0,
        wandb_id=None if wandb.run is None else wandb.run.id,
        actual_loss_positions=actual_loss_positions,
        loader_state=loader_state,
        identity_masker_state=identity_masker_state,
    )
    final_path = run_dir / ("smoke-final.pt" if smoke else "final.pt")
    advance_checkpoint_link(snapshot, final_path)
    if uploader is not None:
        uploader.upload(snapshot, key=final_path.name)

    checkpoint_sha = checkpoint_sha256(final_path)
    validation = _validation_wandb_metrics(val_metrics(model, val_cache, cfg), cfg)
    final_metrics = {f"val/{name}": value for name, value in validation.items()}
    if not smoke and cfg.automatic_evaluation:
        if uploader is None:
            raise RuntimeError("production evaluation requires R2 checkpoint upload")
        uploader.wait()
        spawn_closed_loop_evaluation(run_dir.name, update, checkpoint_sha, cfg.final_eval_n_matchups)
    wandb.log({"global_step": update, **final_metrics})

    mean_wait = float(np.mean(loader_wait_fractions)) if loader_wait_fractions else 0.0
    p95_wait = float(np.percentile(loader_wait_fractions, 95)) if loader_wait_fractions else 0.0
    print(f"[loader] mean wait={100 * mean_wait:.2f}%, p95={100 * p95_wait:.2f}%", flush=True)
    if smoke and (mean_wait > 0.05 or p95_wait > 0.10):
        raise RuntimeError("smoke loader gate failed: require mean wait <=5% and p95 <=10%")


def _compile_synthetic_forward_backward(
    model: GPT,
    cfg: TrainConfig,
    *,
    step: int,
    trunk_fn: Callable,
    temporal_fn: Callable,
) -> None:
    """Compile production-shaped graphs without changing any training RNG."""
    compile_targets = []
    if DEVICE == "cuda" and cfg.compile_trunk:
        compile_targets.append("trunk")
    if DEVICE == "cuda" and cfg.compile_temporal:
        compile_targets.append("temporal model")
    targets = " and ".join(compile_targets)
    compile_started = None
    if compile_targets:
        print(f"[compile] starting synthetic forward/backward to trigger lazy compilation for {targets}", flush=True)
        compile_started = time.monotonic()
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if DEVICE == "cuda" else None
    try:
        heartbeat = (
            _elapsed_heartbeat(f"[compile] lazy compilation still running for {targets}")
            if compile_targets
            else contextlib.nullcontext()
        )
        with heartbeat:
            model.train()
            model.zero_grad(set_to_none=True)
            batch = synthetic_batch(cfg, torch.device(DEVICE))
            valid_prefixes = cfg.batch_size * (cfg.arch.L_ctx - cfg.arch.direct_loss_start)
            _backward_training_batch(
                model,
                batch,
                cfg,
                step=step,
                valid_prefixes=valid_prefixes,
                trunk_fn=trunk_fn,
                temporal_fn=temporal_fn,
                phase_timer=None,
            )
            nonfinite = [
                name
                for name, parameter in model.named_parameters()
                if parameter.grad is None or not torch.isfinite(parameter.grad).all()
            ]
            if nonfinite:
                raise FloatingPointError(f"finite-gradient smoke failed for parameters {nonfinite[:8]}")
            if DEVICE == "cuda":
                torch.cuda.synchronize()
    finally:
        model.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)
    if compile_started is not None:
        print(f"[compile] lazy compilation complete in {time.monotonic() - compile_started:.1f}s", flush=True)


def _loader_state_boundaries(cfg: TrainConfig, run_stop: int) -> tuple[int, ...]:
    """Return updates after which fetched CPU batches must be consumed."""
    boundaries = {run_stop}
    for cadence in (cfg.val_every, cfg.eval_every, cfg.ckpt_every):
        if cadence > 0:
            boundaries.update(range(cadence, run_stop, cadence))
    boundaries.update(update for update in scientific_checkpoint_updates(cfg) if update < run_stop)
    return tuple(sorted(boundaries))


def scientific_checkpoint_updates(cfg: TrainConfig) -> tuple[int, ...]:
    """Return every nested-data endpoint reached by this trajectory."""
    return tuple(update for update in DATA_UPDATES if update <= cfg.max_steps)


def train(
    cfg: TrainConfig,
    stats: dict[str, FeatureStats],
    *,
    comment: str = "",
    resume_run: str | None = None,
    resume_state: dict | None = None,
    smoke: bool = False,
    proxy: bool = False,
    stop_after_update: int | None = None,
) -> None:
    validate_config(cfg)
    if not smoke and not cfg.latency_manifest_sha256:
        raise ValueError("production training requires the measured latency manifest SHA-256")
    if not smoke and cfg.target_positions != DATA_POSITIONS[-1]:
        raise ValueError(f"production nested-data training must end at D=2^31, got {cfg.target_positions}")
    if not smoke and stop_after_update is not None:
        raise ValueError("stop_after_update is a smoke-only control")
    if stop_after_update is not None and not 1 <= stop_after_update <= cfg.max_steps:
        raise ValueError(f"stop_after_update must be in [1, {cfg.max_steps}], got {stop_after_update}")
    run_stop = cfg.max_steps if stop_after_update is None else stop_after_update
    run_name = resume_run or make_run_name(
        Path(__file__).stem,
        model_tag(cfg),
        "all-44-policy-world-v8",
        comment,
    )
    uploader = BackgroundUploader(run_name) if cfg.push_to_r2 else None
    _init_wandb(cfg, run_name, resume_state)
    run_dir, replay_dir = setup_run_dir(run_name)
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high" if cfg.allow_tf32 else "highest")
    sidecar = load_identity_sidecar(cfg)
    prepared_data = _prepare_training_data(cfg, stats, sidecar, resume_state)
    model_started = time.monotonic()
    print(f"[model] constructing GPT and moving parameters to {DEVICE}", flush=True)
    model = GPT(cfg, sidecar.vocabulary).to(DEVICE)
    print(f"[model] construction complete in {time.monotonic() - model_started:.1f}s", flush=True)
    counts = subsystem_parameter_counts(model)
    flops_per_update = approximate_training_flops_per_update(cfg, counts)
    device_name = torch.cuda.get_device_name() if DEVICE == "cuda" else None
    peak_flops = bf16_dense_peak_flops(device_name or "")
    _log_training_summary(
        cfg,
        counts,
        prepared_data.loader,
        flops_per_update=flops_per_update,
        device_name=device_name,
        peak_flops=peak_flops,
    )
    optimizer = make_optimizer(model, cfg)
    scheduler = LambdaLR(optimizer, lr_schedule(cfg))
    start_step = 0
    actual_positions = 0
    identity_masker = IdentityMasker(cfg.seed ^ 0x0541D, cfg.identity_dropout)
    if resume_state is not None:
        model.load_state_dict(resume_state["model"])
        optimizer.load_state_dict(resume_state["opt"])
        scheduler.load_state_dict(resume_state["sched"])
        identity_state = resume_state.get("identity_masker")
        if not isinstance(identity_state, dict):
            raise ValueError("resume checkpoint has no identity-mask RNG state")
        identity_masker.load_state_dict(identity_state)
        restore_training_rng_state(resume_state.get("rng"))
        start_step = int(resume_state["step"]) + 1
        positions_per_update = cfg.batch_size * (cfg.arch.L_ctx - cfg.arch.direct_loss_start)
        actual_positions = int(resume_state.get("actual_loss_positions", start_step * positions_per_update))
        if not 0 <= actual_positions <= start_step * positions_per_update:
            raise ValueError(
                f"checkpoint actual_loss_positions={actual_positions} is invalid after {start_step} updates"
            )

    trunk_fn, temporal_fn = _training_functions(model, cfg)
    phase_index = prepared_data.phase_index
    train_loader, val_cache = prepared_data.loader, prepared_data.validation
    _compile_synthetic_forward_backward(
        model,
        cfg,
        step=start_step,
        trunk_fn=trunk_fn,
        temporal_fn=temporal_fn,
    )
    run_started = time.monotonic()
    try:
        batch_prefetcher = DeviceBatchPrefetcher(
            train_loader,
            cfg,
            DEVICE,
            identity_masker,
            iterator=prepared_data.iterator,
            first_batch_future=prepared_data.first_batch_future,
        )
    finally:
        prepared_data.resources.close()
    if len(prepared_data.first_batch_seconds) != 1:
        raise RuntimeError("first physical-shard batch did not record its fill time")
    cold_fill_seconds = prepared_data.first_batch_seconds[0]
    print(
        f"[loader] workers started in {prepared_data.worker_start_seconds:.1f}s; "
        f"cold fill completed in {cold_fill_seconds:.1f}s",
        flush=True,
    )
    if wandb.run is not None:
        wandb.run.summary["loader/worker_start_s"] = prepared_data.worker_start_seconds
        wandb.run.summary["loader/cold_fill_s"] = cold_fill_seconds
    loader_wait_fractions: list[float] = []
    cache_roots = tuple(streams.BY_NAME[name].local_root for name in cfg.source_names)
    host_metrics = HostMetricsSampler(
        cache_roots,
        interval_s=cfg.system_metrics_interval_s,
        process_interval_s=cfg.process_metrics_interval_s,
        cache_interval_s=cfg.cache_metrics_interval_s,
    )
    host_metrics.start()
    metric_accumulator = _TrainingMetricAccumulator()
    window_loader_wait_seconds: list[float] = []
    window_loader_submitted_batches: list[int] = []
    window_loader_ready_batches: list[int] = []
    window_phase_timers: list[CudaPhaseTimer] = []
    window_peak_allocated_gb = 0.0
    update_timer = _UpdateTimer()
    # CUDA compilation must remain on the training thread. Background compilation
    # deadlocked training on both H100 and B200 hosts.
    model.train()
    evaluation_updates = frozenset(scientific_checkpoint_updates(cfg)) if cfg.automatic_evaluation else frozenset()
    loader_state_boundaries = _loader_state_boundaries(cfg, run_stop)
    try:
        for step in range(start_step, run_stop):
            update = step + 1
            update_timer.start()
            if DEVICE == "cuda":
                torch.cuda.reset_peak_memory_stats()

            val_due = cfg.val_every > 0 and update % cfg.val_every == 0 and update < run_stop
            eval_due = update in evaluation_updates and update < run_stop
            scientific_due = update in scientific_checkpoint_updates(cfg) and update < run_stop
            data_phase_due = update in DATA_UPDATES
            ckpt_due = (cfg.ckpt_every > 0 and update % cfg.ckpt_every == 0 and update < run_stop) or scientific_due
            boundary_due = val_due or eval_due or ckpt_due
            state_boundary_due = boundary_due or update == run_stop
            next_state_boundary = next(boundary for boundary in loader_state_boundaries if boundary >= update)

            phase_due = (
                DEVICE == "cuda"
                and cfg.phase_timing_every > 0
                and (update == 1 or update % cfg.phase_timing_every == 0)
            )
            phase_timer = CudaPhaseTimer() if phase_due else None
            if phase_timer is not None:
                phase_timer.record("start")
            batch, valid_prefixes = batch_prefetcher.next()
            if phase_timer is not None:
                phase_timer.record("h2d_end")
            lookahead_limit = 0 if state_boundary_due else next_state_boundary - update
            batch_prefetcher.fill_lookahead(lookahead_limit)
            window_loader_submitted_batches.append(batch_prefetcher.submitted_batches)
            window_loader_ready_batches.append(batch_prefetcher.ready_batches)
            result = train_step(
                model,
                batch,
                cfg,
                step=step,
                update=update,
                valid_prefixes=valid_prefixes,
                trunk_fn=trunk_fn,
                temporal_fn=temporal_fn,
                optimizer=optimizer,
                scheduler=scheduler,
                phase_timer=phase_timer,
            )
            loader_wait = batch_prefetcher.stage_next() if not state_boundary_due else 0.0
            actual_positions += valid_prefixes
            metric_accumulator.add(result, valid_prefixes)
            window_loader_wait_seconds.append(loader_wait)
            if phase_timer is not None:
                window_phase_timers.append(phase_timer)
            if DEVICE == "cuda":
                window_peak_allocated_gb = max(
                    window_peak_allocated_gb,
                    torch.cuda.max_memory_allocated() / 2**30,
                )

            metrics_due = update % cfg.train_metrics_every == 0 or update == run_stop
            if metrics_due:
                window_metric_values, window_updates, window_valid_prefixes = metric_accumulator.flush(
                    cfg,
                    update=update,
                )
            update_timer.finish()
            if metrics_due:
                if len(window_loader_wait_seconds) != window_updates:
                    raise RuntimeError("training telemetry window lost an update")
                if len(window_loader_submitted_batches) != window_updates:
                    raise RuntimeError("submitted-batch telemetry window lost an update")
                if len(window_loader_ready_batches) != window_updates:
                    raise RuntimeError("ready-batch telemetry window lost an update")
                loader_wait_s = sum(window_loader_wait_seconds) / window_updates
                loader_wait_p95_s = float(np.percentile(window_loader_wait_seconds, 95))
                submitted_batches = sum(window_loader_submitted_batches) / window_updates
                ready_batches = sum(window_loader_ready_batches) / window_updates
                training_elapsed_wall_s = time.monotonic() - run_started
                completed_updates = update - start_step
                projected_training_remaining_s = training_elapsed_wall_s * (run_stop - update) / completed_updates
                first_window_update = update - window_updates + 1
                log: dict[str, object] = {
                    "data/windows": update * cfg.batch_size,
                    "data/supervised_prefixes": actual_positions,
                    "data/future_targets": actual_positions * len(cfg.arch.head_offsets),
                    "data/dropped_windows": 0,
                    "data/phase_index": phase_index,
                    "data/admitted_unique_replays": DATA_REPLAYS[phase_index],
                    "loader/queue_depth": submitted_batches,
                    "loader/submitted_cpu_batches": submitted_batches,
                    "loader/ready_cpu_batches": ready_batches,
                    "loader/uncovered_wait_s": loader_wait_s,
                    "loader/uncovered_wait_p95_s": loader_wait_p95_s,
                    "progress/elapsed_s": training_elapsed_wall_s,
                    "progress/remaining_s": projected_training_remaining_s,
                    "schedule/adam_lr": result.adam_lr,
                    **window_metric_values,
                    **identity_masker.metrics(),
                    **_mean_phase_metrics(window_phase_timers),
                }
                loader_metrics = getattr(train_loader, "metrics", None)
                if isinstance(loader_metrics, dict):
                    log.update(loader_metrics)
                if _cadence_in_window(first_window_update, update, cfg.system_metrics_every):
                    log.update(_minimal_system_metrics(host_metrics.snapshot()))
                if DEVICE == "cuda":
                    log["system/gpu_memory_gb"] = window_peak_allocated_gb

                update_s, update_p95_s = update_timer.stats_and_reset(window_updates)
                samples_per_s = cfg.batch_size / update_s
                loader_wait_fraction = loader_wait_s / max(update_s, 1e-12)
                loader_wait_fractions.extend([loader_wait_fraction] * window_updates)
                log["throughput/update_s"] = update_s
                log["throughput/update_p95_s"] = update_p95_s
                log["throughput/samples_per_s"] = samples_per_s
                if peak_flops is not None:
                    log["throughput/mfu"] = model_flops_utilization(
                        flops_per_update,
                        update_s,
                        peak_flops,
                    )
                wandb.log({"global_step": update, **log})
                if update <= cfg.train_metrics_every or update % 50 == 0 or update == run_stop:
                    print(
                        f"[t+{time.monotonic() - run_started:.0f}s] update {update}: "
                        f"{window_metric_values['train/loss']:.3f} bits objective, "
                        f"{samples_per_s:.0f} samples/s, "
                        f"projected training remaining {projected_training_remaining_s / 60:.1f}m",
                        flush=True,
                    )
                window_loader_wait_seconds.clear()
                window_loader_submitted_batches.clear()
                window_loader_ready_batches.clear()
                window_phase_timers.clear()
                window_peak_allocated_gb = 0.0
            checkpoint_path: Path | None = None
            if boundary_due:
                if not batch_prefetcher.drained:
                    raise RuntimeError("CPU lookahead was not drained at a state boundary")
                checkpoint_path = save_boundary_checkpoint(
                    run_dir,
                    update=update,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    cfg=cfg,
                    uploader=uploader,
                    milestone=scientific_due,
                    wandb_id=None if wandb.run is None else wandb.run.id,
                    actual_loss_positions=actual_positions,
                    loader_state=nested_loader_checkpoint_state(
                        cfg,
                        update=update,
                        phase_index=phase_index,
                        loader_state=train_loader.state_dict(),
                    ),
                    identity_masker_state=identity_masker.state_dict(),
                )
            boundary_metrics: dict[str, float] = {}
            if val_due:
                values = _validation_wandb_metrics(val_metrics(model, val_cache, cfg), cfg)
                boundary_metrics.update({f"val/{name}": value for name, value in values.items()})
            if eval_due:
                assert checkpoint_path is not None
                if uploader is None:
                    raise RuntimeError("production evaluation requires R2 checkpoint upload")
                uploader.wait()
                spawn_closed_loop_evaluation(
                    run_name,
                    update,
                    checkpoint_sha256(checkpoint_path),
                    cfg.eval_n_matchups,
                )
            if boundary_metrics:
                wandb.log({"global_step": update, **boundary_metrics})
            transitioned_phase = data_phase_due and update < run_stop
            if transitioned_phase:
                batch_prefetcher.close()
                train_loader.close()
                phase_index += 1
                train_loader = _make_train_loader(
                    cfg,
                    stats,
                    ReplayPlayerLookup(sidecar.by_replay),
                    phase_index,
                )
                batch_prefetcher = DeviceBatchPrefetcher(train_loader, cfg, DEVICE, identity_masker)
                print(
                    f"[data] admitted phase {phase_index + 1}/{len(DATA_UPDATES)} "
                    f"selection={train_loader.selection.sha256}",
                    flush=True,
                )
            if update < run_stop and boundary_due and not transitioned_phase:
                next_state_boundary = next(boundary for boundary in loader_state_boundaries if boundary > update)
                batch_prefetcher.fill_lookahead(next_state_boundary - update)
                batch_prefetcher.stage_next()
        if not batch_prefetcher.drained:
            raise RuntimeError("CPU lookahead was not drained at the final update")
        _finalize_training(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            stats=stats,
            val_cache=val_cache,
            run_dir=run_dir,
            replay_dir=replay_dir,
            uploader=uploader,
            loader_wait_fractions=loader_wait_fractions,
            loader_state=nested_loader_checkpoint_state(
                cfg,
                update=run_stop,
                phase_index=phase_index,
                loader_state=train_loader.state_dict(),
            ),
            identity_masker_state=identity_masker.state_dict(),
            update=run_stop,
            actual_loss_positions=actual_positions,
            smoke=smoke,
        )
    finally:
        batch_prefetcher.close()
        train_loader.close()
        host_metrics.close()
        if uploader is not None:
            uploader.upload_tree(replay_dir, base=run_dir)
            uploader.close()
        wandb.finish()


def _checkpoint_config(cfg: TrainConfig) -> dict[str, object]:
    values = asdict(cfg)
    architecture = values.pop("arch")
    timing = values.pop("timing")
    return {
        "experiment_id": _EXPERIMENT_ID,
        "architecture": architecture,
        "timing": timing,
        **values,
        "max_steps": cfg.max_steps,
        "warmup_steps": cfg.warmup_steps,
    }


def config_from_state(values: dict) -> TrainConfig:
    """Restore a checkpoint written by the current experiment definition."""
    derived_fields = {"max_steps", "warmup_steps"}
    runtime_fields = {item.name for item in fields(TrainConfig)} - {"arch", "timing"}
    expected = {"experiment_id", "architecture", "timing", *runtime_fields, *derived_fields}
    missing = expected - values.keys()
    unexpected = values.keys() - expected
    if missing or unexpected:
        raise ValueError(f"checkpoint config mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    if values["experiment_id"] != _EXPERIMENT_ID:
        raise ValueError(f"checkpoint experiment_id {values['experiment_id']!r} != {_EXPERIMENT_ID!r}")
    architecture_values = values["architecture"]
    timing_values = values["timing"]
    if set(architecture_values) != {item.name for item in fields(Architecture)}:
        raise ValueError("checkpoint architecture does not match the current architecture fields")
    if set(timing_values) != {item.name for item in fields(TimingConfig)}:
        raise ValueError("checkpoint timing does not match the current timing fields")
    architecture = Architecture(**architecture_values)
    timing = TimingConfig(**timing_values)
    runtime = {name: values[name] for name in runtime_fields}
    cfg = TrainConfig(arch=architecture, timing=timing, **runtime)
    derived = {name: values[name] for name in derived_fields}
    expected_derived = {name: getattr(cfg, name) for name in derived_fields}
    if derived != expected_derived:
        raise ValueError(f"checkpoint derived schedule mismatch: {derived} != {expected_derived}")
    return cfg


def load_checkpoint(path: str, *, device: str = DEVICE) -> tuple[GPT, TrainConfig, dict[str, FeatureStats], dict]:
    state = torch.load(path, map_location=device, weights_only=False)
    cfg = config_from_state(state["cfg"])
    validate_config(cfg)
    encoded = state["model"].get("player_code_bytes")
    if not isinstance(encoded, Tensor) or not encoded.numel():
        raise ValueError("checkpoint has no embedded identity vocabulary")
    vocabulary = PlayerVocabulary(decode_player_codes(encoded.detach().cpu().numpy().tobytes()))
    model = GPT(cfg, vocabulary).to(device)
    model.load_state_dict(state["model"])
    if torch.device(device).type == "cuda":
        model.to(dtype=torch.bfloat16)
    model.eval()
    stats = load_stats(cfg)
    return model, cfg, stats, state


def _upload_eval_evidence(run_name: str, replay_dir: Path) -> None:
    uploader = BackgroundUploader(run_name)
    uploader.upload_tree(replay_dir, base=(Path("runs") / run_name).resolve())
    uploader.close()


def _log_shared_eval_metrics(
    wandb_id: str,
    update: int,
    values: dict[str, float],
    *,
    namespace: str = "eval",
) -> None:
    """Log one evaluation from a non-primary writer on the training run."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_/-]*", namespace):
        raise ValueError(f"invalid W&B metric namespace: {namespace!r}")
    run = wandb.init(
        project="hal",
        id=wandb_id,
        settings=wandb.Settings(
            mode="shared",
            x_label=f"{namespace.replace('/', '-')}-step-{update:07d}",
            x_primary=False,
            x_update_finish_state=False,
            x_disable_stats=True,
        ),
    )
    if run is None:
        raise RuntimeError("W&B shared run initialization returned no run")
    try:
        metrics = {f"{namespace}/{name}": value for name, value in _eval_wandb_metrics(values).items()}
        run.log({"global_step": update, "eval/checkpoint_step": update, **metrics})
    finally:
        run.finish()


def eval_checkpoint(
    path: str,
    *,
    n_matchups: int | None = None,
    eager: bool = False,
    max_parallel: int | None = None,
    output_name: str | None = None,
    upload_run: str | None = None,
    shared_wandb: bool = False,
    expected_checkpoint_sha256: str | None = None,
    player_code: str | None = None,
    fixed_ego_character: melee.Character | None = None,
    delay_frames: int | None = None,
    replan_interval_frames: int | None = None,
    wandb_namespace: str = "eval",
) -> dict[str, float]:
    actual_checkpoint_sha256 = checkpoint_sha256(Path(path))
    if expected_checkpoint_sha256 is not None and actual_checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError(
            f"checkpoint SHA-256 mismatch: expected {expected_checkpoint_sha256}, got {actual_checkpoint_sha256}"
        )
    model, cfg, stats, state = load_checkpoint(path)
    validate_config(cfg)
    if player_code is None:
        ego_player_id = MASKED_PLAYER_ID
        ego_player_code = None
    else:
        encoded = model.get_buffer("player_code_bytes")
        if not encoded.numel():
            raise ValueError("checkpoint has no embedded identity vocabulary")
        vocabulary = PlayerVocabulary(decode_player_codes(encoded.detach().cpu().numpy().tobytes()))
        ego_player_id = vocabulary.id_for_code(player_code)
        ego_player_code = player_code.strip()
    horizon = cfg.prediction_frames
    delay = cfg.delay_frames if delay_frames is None else delay_frames
    replan = cfg.replan_interval_frames if replan_interval_frames is None else replan_interval_frames
    _validate_deployment_timing(horizon, delay, replan)
    is_variant = any(
        value is not None for value in (player_code, fixed_ego_character, delay_frames, replan_interval_frames)
    )
    if is_variant and upload_run is not None and output_name is None:
        raise ValueError("evaluation overrides uploaded to a run require an explicit output_name")
    if is_variant and shared_wandb and wandb_namespace == "eval":
        raise ValueError("evaluation overrides require a distinct W&B namespace")
    update = int(state["step"]) + 1
    if upload_run is not None:
        default_name = f"eval_step_{update:07d}_s{horizon}"
    else:
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
        checkpoint_sha256=actual_checkpoint_sha256,
        eager=eager,
        max_parallel=max_parallel,
        fixed_ego_character=fixed_ego_character,
        ego_player_id=ego_player_id,
        ego_player_code=ego_player_code,
        delay_frames=delay,
        replan_interval_frames=replan,
    )
    require_complete_eval(values, cfg.final_eval_n_matchups if n_matchups is None else n_matchups)
    if upload_run is not None:
        _upload_eval_evidence(upload_run, replay_dir)
    if shared_wandb:
        wandb_id = state.get("wandb_id")
        if not isinstance(wandb_id, str):
            raise RuntimeError("checkpoint has no W&B run id for shared logging")
        _log_shared_eval_metrics(wandb_id, update, values, namespace=wandb_namespace)
    print(
        f"[eval] step={update} horizon={horizon} ego_player={ego_player_code!r} "
        f"fixed_ego_character={None if fixed_ego_character is None else fixed_ego_character.name}: {values}",
        flush=True,
    )
    return values


def _resolve_eval_checkpoint(checkpoint: str, run: str | None) -> Path:
    if run is None:
        return Path(checkpoint)
    path = download_latest(run, Path("runs") / run, name=checkpoint)
    if path is None:
        raise SystemExit(f"no {checkpoint!r} for run {run!r}")
    return path


def _remote_run_exists(run_name: str) -> bool:
    """Return whether R2 contains an object for a run name."""
    response = r2.client().list_objects_v2(Bucket=r2.bucket(), Prefix=f"runs/{run_name}/", MaxKeys=1)
    return bool(response.get("KeyCount", len(response.get("Contents", ()))))


@dataclass
class TrainArgs:
    cfg: TrainConfig = dataclass_field(default_factory=TrainConfig)
    width: int | None = None
    updates: int | None = None
    latency_manifest: Path = Path("docs/experiments/054_iso_data_latency_manifest.json")
    proxy: bool = False
    comment: str = ""
    resume: str | None = None
    resume_checkpoint: str = "latest.pt"
    resume_as: str | None = None
    resume_num_workers: int | None = None
    smoke: bool = False
    stop_after_update: int | None = None
    eval_max_parallel: int | None = None


@dataclass
class EvalArgs:
    checkpoint: str
    run: str | None = None
    n_matchups: int | None = None
    eager: bool = False
    max_parallel: int | None = None
    output_name: str | None = None
    shared_wandb: bool = False
    expected_checkpoint_sha256: str | None = None
    player_code: str | None = None
    fixed_ego_character: str | None = None
    delay_frames: int | None = None
    replan_interval_frames: int | None = None
    wandb_namespace: str = "eval"


@dataclass
class LatencyPreflightArgs:
    output: Path = Path("docs/experiments/054_iso_data_latency_manifest.json")


@dataclass
class LatencyDiagnosticArgs:
    output: Path = Path("runs/054_latency_b4.probes.json")
    batch_sizes: tuple[int, ...] = (4,)


@dataclass
class StudyPlanArgs:
    latency_manifest: Path = Path("docs/experiments/054_iso_data_latency_manifest.json")


@dataclass
class AnalyzeArgs:
    wandb_path: str = "ericyuegu/hal"
    latency_manifest: Path = Path("docs/experiments/054_iso_data_latency_manifest.json")
    output: Path = Path("results/054_bc_capacity_latency")


type Command = (
    Annotated[TrainArgs, tyro.conf.subcommand(name="train")]
    | Annotated[EvalArgs, tyro.conf.subcommand(name="eval")]
    | Annotated[LatencyPreflightArgs, tyro.conf.subcommand(name="latency-preflight")]
    | Annotated[LatencyDiagnosticArgs, tyro.conf.subcommand(name="latency-diagnostic")]
    | Annotated[StudyPlanArgs, tyro.conf.subcommand(name="study-plan")]
    | Annotated[AnalyzeArgs, tyro.conf.subcommand(name="analyze")]
)


def _parse_character(name: str | None) -> melee.Character | None:
    if name is None:
        return None
    normalized = name.strip().upper()
    try:
        return melee.Character[normalized]
    except KeyError as error:
        raise SystemExit(f"unknown fixed ego character: {name!r}") from error


def main(args: Command) -> None:
    if isinstance(args, AnalyzeArgs):
        latency_rows, latency_sha256 = load_latency_manifest(args.latency_manifest)
        api = wandb.Api(timeout=120)
        runs = api.runs(
            args.wandb_path,
            filters={
                "config.experiment_id": _EXPERIMENT_ID,
                "config.latency_manifest_sha256": latency_sha256,
            },
        )
        endpoints = _analysis_endpoints(runs, latency_sha256)
        expected = {(endpoint.width, update) for endpoint in study_endpoints(latency_rows) for update in DATA_UPDATES}
        if set(endpoints) != expected:
            missing = sorted(expected - set(endpoints))
            extra = sorted(set(endpoints) - expected)
            raise ValueError(f"O54 analysis endpoint mismatch: missing={missing}, extra={extra}")
        match_rows = load_analysis_match_rows(endpoints, latency_rows, args.output)
        report = write_analysis_artifacts(endpoints, latency_rows, args.output, match_rows=match_rows)
        print(json.dumps(report, sort_keys=True, indent=2), flush=True)
        return
    if isinstance(args, StudyPlanArgs):
        latency_rows, latency_sha256 = load_latency_manifest(args.latency_manifest)
        endpoints = study_endpoints(latency_rows)
        computes = tuple(
            training_compute(
                config_for_width(endpoint.width, updates=endpoint.updates),
                PARAMETER_COUNT_CONTRACTS[endpoint.width],
            )
            for endpoint in endpoints
        )
        total_compute = sum(computes)
        training_hours = sum(
            endpoint.updates * B200_REFERENCE_UPDATE_SECONDS[endpoint.width] / 3_600 for endpoint in endpoints
        )
        print(
            f"[study] latency_manifest_sha256={latency_sha256} runs={len(endpoints)} "
            f"compute={total_compute / 1e18:.3f} EFLOPs projected_training_hours={training_hours:.1f} "
            f"working_cost=${STUDY_WORKING_COST_DOLLARS} "
            f"warning=${2 * STUDY_WORKING_COST_DOLLARS} intervention=${3 * STUDY_WORKING_COST_DOLLARS}",
            flush=True,
        )
        print(
            "[study] projection uses conservative per-width B200 update times "
            "W256=0.25s W512=0.30s W768=0.40s W1024=1.00s; verify after launch",
            flush=True,
        )
        for endpoint, compute, command in zip(
            endpoints,
            computes,
            study_launch_commands(latency_rows, args.latency_manifest),
            strict=True,
        ):
            endpoint_hours = endpoint.updates * B200_REFERENCE_UPDATE_SECONDS[endpoint.width] / 3_600
            training_cost = endpoint_hours * MODAL_TRAINING_DOLLARS_PER_HOUR
            print(
                f"[study] W{endpoint.width}/L{TRUNK_DEPTHS[endpoint.width]} updates={endpoint.updates:,} "
                f"roles={','.join(endpoint.roles)} compute={compute / 1e18:.4f} EFLOPs "
                f"projected_training_hours={endpoint_hours:.2f} training_cost=${training_cost:.2f}",
                flush=True,
            )
            print(shlex.join(command), flush=True)
        return
    if isinstance(args, LatencyDiagnosticArgs):
        run_latency_diagnostic(args.output, args.batch_sizes)
        return
    if isinstance(args, LatencyPreflightArgs):
        digest = run_latency_preflight(args.output)
        print(f"[latency] wrote {args.output} sha256={digest}", flush=True)
        return
    if isinstance(args, EvalArgs):
        checkpoint = _resolve_eval_checkpoint(args.checkpoint, args.run)
        eval_checkpoint(
            str(checkpoint),
            n_matchups=args.n_matchups,
            eager=args.eager,
            max_parallel=args.max_parallel,
            output_name=args.output_name,
            upload_run=args.run,
            shared_wandb=args.shared_wandb,
            expected_checkpoint_sha256=args.expected_checkpoint_sha256,
            player_code=args.player_code,
            fixed_ego_character=_parse_character(args.fixed_ego_character),
            delay_frames=args.delay_frames,
            replan_interval_frames=args.replan_interval_frames,
            wandb_namespace=args.wandb_namespace,
        )
        return
    resume_run = resume_state = None
    if args.proxy and args.width is not None:
        raise SystemExit("--proxy and --width are mutually exclusive")
    cfg = args.cfg if args.width is None else config_for_width(args.width, updates=args.updates)
    if args.width is None and args.updates is not None:
        raise SystemExit("--updates requires --width")
    proxy_arch: Architecture | None = None
    proxy_timing: TimingConfig | None = None
    target_positions: int | None = None
    if args.resume is None and args.proxy:
        proxy = proxy_config()
        proxy_arch = proxy.arch
        proxy_timing = proxy.timing
        target_positions = proxy.target_positions
    if args.resume is None and (args.resume_checkpoint != "latest.pt" or args.resume_as is not None):
        raise SystemExit("--resume-checkpoint and --resume-as require --resume")
    if args.resume is None and args.resume_num_workers is not None:
        raise SystemExit("--resume-num-workers requires --resume")
    if args.resume is not None:
        checkpoint = Path(args.resume_checkpoint)
        if (
            checkpoint.is_absolute()
            or ".." in checkpoint.parts
            or checkpoint.suffix != ".pt"
            or args.resume_checkpoint in ("", ".")
        ):
            raise SystemExit("--resume-checkpoint must be a relative .pt object within the run")
        resume_state = load_for_resume(
            args.resume,
            Path("runs") / args.resume,
            device=DEVICE,
            name=args.resume_checkpoint,
        )
        if resume_state is None:
            raise SystemExit(f"no {args.resume_checkpoint!r} for run {args.resume!r}")
        resume_run = args.resume_as or args.resume
        cfg = config_from_state(resume_state["cfg"])
        proxy = proxy_config()
        if args.proxy != (cfg.arch == proxy.arch and cfg.target_positions == proxy.target_positions):
            raise SystemExit("--proxy must match the resumed checkpoint treatment")
        if args.resume_as is not None:
            if Path(args.resume_as).name != args.resume_as or args.resume_as in ("", ".", ".."):
                raise SystemExit("--resume-as must be one run-name component")
            if args.resume_as == args.resume:
                raise SystemExit("--resume-as must differ from the source run")
            destination_exists = (Path("runs") / args.resume_as).exists()
            if cfg.push_to_r2:
                destination_exists = destination_exists or _remote_run_exists(args.resume_as)
            if destination_exists:
                raise SystemExit(
                    f"resume destination {args.resume_as!r} already exists; continue it with --resume {args.resume_as}"
                )
            resume_state = {**resume_state, "wandb_id": None}
    cfg = replace(
        cfg,
        arch=cfg.arch if proxy_arch is None else proxy_arch,
        timing=cfg.timing if proxy_timing is None else proxy_timing,
        target_positions=cfg.target_positions if target_positions is None else target_positions,
        num_workers=cfg.num_workers if args.resume_num_workers is None else args.resume_num_workers,
        eval_max_parallel=cfg.eval_max_parallel if args.eval_max_parallel is None else args.eval_max_parallel,
    )
    if not args.smoke or args.latency_manifest.is_file():
        latency_rows, latency_sha256 = load_latency_manifest(args.latency_manifest)
        try:
            latency_row = latency_rows[cfg.arch.d_model]
        except KeyError as error:
            raise SystemExit(f"latency manifest has no width {cfg.arch.d_model}") from error
        if cfg.timing != latency_row.timing and args.resume is not None:
            raise SystemExit("resume checkpoint timing differs from the measured latency manifest")
        if args.resume is not None and cfg.latency_manifest_sha256 != latency_sha256:
            raise SystemExit("resume checkpoint uses a different measured latency manifest")
        cfg = replace(
            cfg,
            timing=latency_row.timing,
            latency_manifest_sha256=latency_sha256,
        )
    stats = load_stats(cfg)
    train(
        cfg,
        stats,
        comment=args.comment,
        resume_run=resume_run,
        resume_state=resume_state,
        smoke=args.smoke,
        proxy=args.proxy,
        stop_after_update=args.stop_after_update,
    )


if __name__ == "__main__":
    main(tyro.cli(cast(type[Command], Command)))
