"""Experiment 060: compute-optimal action-sequence training on two GPUs.

This experiment trains AWR over the complete policy-world corpus. A learned
value head estimates the return from each trunk state, and the detached
``G_{t+1} - V(s_t)`` advantage weights the policy objective. It includes the
schema-v6 projectile block (``item{0..3}_*``) in every observation.

This file is deliberately standalone. It retains the O41 detached-value light
AWR and O49 ego-identity contracts without importing another experiment. The
temporal decoder uses 4x MLP expansion and matched nonlinear decoder and
trunk-skip logit heads. Return conditioning is absent; return targets are used
only by the critic and AWR objective.

The four item slots are ordered by ascending spawn id, so a slot keeps its item
until an OLDER item despawns and the remaining items shift down. A pooled set
encoder makes that churn invisible: one shared per-slot encoder, gated by the
slot's presence flag, summed over the slots. An empty slot adds the exact zero
vector and the live-item count stays implicit in the sum.

The treatment uses a 1024-dimensional model with six trunk layers and eight
temporal-decoder layers. It predicts every offset from 1 through 30 with equal
loss coefficients. The policy samples 32 suffix prefixes per replay window, so
scaling-law data D counts those 32 positions. The critic and AWR normalizer still
use all 128 suffix prefixes and are accounted for separately in training FLOPs.

Two parameter-matched proxy arms test the depth allocation before production:
the prior 15M shape uses a 16-layer trunk and 4-layer 128-wide decoder; the
treatment uses a 6-layer trunk and 8-layer 256-wide decoder. All other experiment
inputs stay fixed. Gameplay evaluation selects between the proxy arms; offline
validation metrics are diagnostic only.

Training requires ``torchrun --standalone --nproc-per-node=2``. Each rank owns a
disjoint half of every source, a local batch of 256, 65,536 replay slots, twelve
loader workers, and independent checkpointed random streams. Rank zero alone
writes checkpoints, validation, W&B, and R2 artifacts.

Run:
    torchrun --standalone --nproc-per-node=2 \
        experiments/060_compute_optimal_action_sequence.py train
    torchrun --standalone --nproc-per-node=2 \
        experiments/060_compute_optimal_action_sequence.py train --resume <run>
    uv run experiments/060_compute_optimal_action_sequence.py eval --checkpoint runs/<run>/final.pt
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
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
from functools import partial
from pathlib import Path
from typing import Annotated
from typing import ClassVar
from typing import Final
from typing import Literal
from typing import cast

import melee
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import tyro
import wandb
from beartype import beartype
from jaxtyping import Bool
from jaxtyping import Float
from jaxtyping import Int
from jaxtyping import jaxtyped
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LambdaLR

from hal import r2
from hal import streams
from hal.data.feature_stats import FeatureStats
from hal.data.feature_stats import load_consolidated_mixture_stats
from hal.data.player_identity import PlayerIdentitySidecar
from hal.data.player_identity import ReplayPlayerLookup
from hal.data.player_identity import load_player_identity_artifact
from hal.data.policy_schema import unpack_player_stock
from hal.data.policy_world_schema import POLICY_WORLD_SCHEMA_VERSION
from hal.eval.action_trace import ActionTraceWriter
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
from hal.eval.policy import PolicyBatchAdapter
from hal.eval.scheduling import FrameTiming
from hal.inference.api import PredictionRequest
from hal.inference.api import RuntimeConfig
from hal.inference.benchmark import DecodeTelemetry
from hal.inference.warmup import canonical_context
from hal.inference.warmup import synthetic_context as build_synthetic_context
from hal.inference.window_policy import DecodedPlan
from hal.inference.window_policy import DenseWindowPredictionPolicy
from hal.inference.window_policy import WindowPolicy
from hal.models.action_sequence import ActionSequenceConfig
from hal.models.action_sequence import ActionSequenceTransformer
from hal.models.action_sequence import NonlinearActionHead
from hal.models.action_sequence import SwiGLU
from hal.models.action_sequence import activation_input_metrics
from hal.models.action_sequence import activation_output_metrics
from hal.models.action_sequence import decoder_rmsnorm
from hal.models.controller_codec import BUTTON_LEFT_CHANNEL
from hal.models.controller_codec import BUTTON_RIGHT_CHANNEL
from hal.models.controller_codec import BUTTONS_GROUP
from hal.models.controller_codec import CONTROLLER_GROUP_COUNT
from hal.models.controller_codec import CONTROLLER_GROUP_NAMES
from hal.models.controller_codec import TRIGGER_LEFT_CHANNEL
from hal.models.controller_codec import TRIGGER_RIGHT_CHANNEL
from hal.models.controller_codec import TRIGGERS_GROUP
from hal.models.controller_codec import MainStickLayout
from hal.models.sampling import validate_sampling_temperature
from hal.representation.features import ITEM_PLAYER_COLUMNS
from hal.representation.features import ITEM_PLAYER_PROJECTION
from hal.representation.features import Context
from hal.representation.features import FeatureProjection
from hal.representation.features import stack_actions
from hal.representation.player_identity import MASKED_PLAYER_ID
from hal.representation.player_identity import PlayerVocabulary
from hal.representation.player_identity import decode_player_codes
from hal.sim.process_vec import ProcessVecTelemetry
from hal.sim.rollout import covering_power_of_two
from hal.training import returns as returns_lib
from hal.training.batches import TrainBatch
from hal.training.buffered_mds_replay_loader import PREFETCH_FACTOR
from hal.training.buffered_mds_replay_loader import BufferedMDSReplayLoader
from hal.training.buffered_mds_replay_loader import MDSStorageAdapter
from hal.training.buffered_mds_replay_loader import PhysicalShardSelection
from hal.training.buffered_mds_replay_loader import SourceRowSelection
from hal.training.buffered_mds_replay_loader import build_shard_plan
from hal.training.checkpoints import BackgroundUploader
from hal.training.checkpoints import ResumeLineage
from hal.training.checkpoints import advance_checkpoint_link
from hal.training.checkpoints import checkpoint_resume_lineage
from hal.training.checkpoints import checkpoint_sha256
from hal.training.checkpoints import download_latest
from hal.training.checkpoints import load_for_resume
from hal.training.checkpoints import read_resume_lineage
from hal.training.checkpoints import save_checkpoint
from hal.training.checkpoints import validate_resume_provenance
from hal.training.mfu import bf16_dense_peak_flops
from hal.training.mfu import bf16_peak_source
from hal.training.mfu import model_flops_utilization
from hal.training.muon import SingleDeviceMuonWithAuxAdam
from hal.training.replay_windows import train_batch_from_columns
from hal.training.returns import ReturnCalibration
from hal.training.runs import make_run_name
from hal.training.runs import setup_run_dir
from hal.training.system_metrics import HostMetricsSampler
from hal.training.validation_replay_loader import make_validation_replay_loader
from hal.wire import ACTION_DIM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_EXPERIMENT_ID: Final[str] = "060_compute_optimal_action_sequence_v2"
_CHECKPOINT_FORMAT_VERSION: Final[int] = 1
_DISTRIBUTED_CHECKPOINT_VERSION: Final[int] = 1
_STARTUP_LOG_INTERVAL_S: Final[float] = 60.0
POLICY_PREFIXES_PER_WINDOW: Final[int] = 32
OFFSET_LOSS_WEIGHTS: Final[tuple[float, ...]] = (1 / 30,) * 30
COMPUTE_EQUIVALENT_PARAMETERS: Final[int] = 3_953_315_601
FLOPS_PER_SUPERVISED_POSITION: Final[int] = 23_719_893_606
SCALING_FIT_A: Final[float] = 89.11185023618282
SCALING_FIT_B: Final[float] = 2009.4388275435915
SCALING_FIT_ALPHA: Final[float] = 0.3680109101792296
SCALING_FIT_BETA: Final[float] = 0.49828476033099955
TARGET_UPDATES: Final[int] = 714_752
TARGET_POSITIONS: Final[int] = 11_710_496_768
WARMUP_UPDATES: Final[int] = 4_096
DECAY_START_UPDATE: Final[int] = 536_576
COOLDOWN_UPDATES: Final[int] = 178_176


RETURN_HORIZON: Final[int] = 60
RETURN_SCALE: Final[float] = 120.0
CALIBRATION_WINDOWS: Final[int] = 65_536


def compute_flops_per_supervised_position(compute_equivalent_parameters: int) -> int:
    """Apply the scaling chart's six-FLOP training convention."""
    if compute_equivalent_parameters <= 0:
        raise ValueError("compute-equivalent parameters must be positive")
    return 6 * compute_equivalent_parameters


def fitted_optimal_positions(compute_equivalent_parameters: int) -> float:
    """Return the O39 fit's optimal D for a fixed compute-equivalent model size."""
    if compute_equivalent_parameters <= 0:
        raise ValueError("compute-equivalent parameters must be positive")
    coefficient = SCALING_FIT_BETA * SCALING_FIT_B / (SCALING_FIT_ALPHA * SCALING_FIT_A)
    return coefficient ** (1 / SCALING_FIT_BETA) * compute_equivalent_parameters ** (
        SCALING_FIT_ALPHA / SCALING_FIT_BETA
    )


FITTED_OPTIMAL_POSITIONS: Final[float] = fitted_optimal_positions(COMPUTE_EQUIVALENT_PARAMETERS)


def checkpoint_rounded_updates(
    fitted_positions: float,
    *,
    positions_per_update: int,
    checkpoint_interval: int,
) -> int:
    """Round a fitted data optimum up to a complete checkpoint interval."""
    if not math.isfinite(fitted_positions) or fitted_positions <= 0:
        raise ValueError("fitted positions must be finite and positive")
    if positions_per_update <= 0 or checkpoint_interval <= 0:
        raise ValueError("update and checkpoint geometry must be positive")
    fitted_updates = math.ceil(fitted_positions / positions_per_update)
    return math.ceil(fitted_updates / checkpoint_interval) * checkpoint_interval


@dataclass(frozen=True, slots=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    def __post_init__(self) -> None:
        if not 0 <= self.rank < self.world_size:
            raise ValueError("distributed rank is outside the world")
        if self.local_rank < 0:
            raise ValueError("local rank must be non-negative")

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


def init_distributed(cfg: TrainConfig, *, backend: str | None = None) -> DistributedContext:
    """Initialize one process per GPU from torchrun's environment."""
    required = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError(
            f"O60 training requires torchrun --standalone --nproc-per-node=2; missing {', '.join(missing)}"
        )
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != cfg.world_size:
        raise ValueError(f"O60 requires world_size={cfg.world_size}, got {world_size}")
    selected_backend = "nccl" if backend is None else backend
    if selected_backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL training requires CUDA")
        if torch.cuda.device_count() != world_size:
            raise RuntimeError(f"O60 requires exactly {world_size} visible GPUs, got {torch.cuda.device_count()}")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    elif selected_backend == "gloo":
        device = torch.device("cpu")
    else:
        raise ValueError(f"unsupported distributed backend {selected_backend!r}")
    dist.init_process_group(backend=selected_backend, rank=rank, world_size=world_size)
    return DistributedContext(rank, local_rank, world_size, device)


def _all_reduce_sum(value: Tensor) -> Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def _all_reduce_mean(value: Tensor) -> Tensor:
    _all_reduce_sum(value)
    if dist.is_available() and dist.is_initialized():
        value.div_(dist.get_world_size())
    return value


def reduce_scalar(value: float, distributed: DistributedContext, *, maximum: bool = False) -> float:
    """Reduce one health or throughput signal across ranks."""
    tensor = torch.tensor(value, dtype=torch.float64, device=distributed.device)
    operation = dist.ReduceOp.MAX if maximum else dist.ReduceOp.SUM
    dist.all_reduce(tensor, op=operation)
    if not maximum:
        tensor.div_(distributed.world_size)
    return float(tensor.cpu())


def reduce_metric_dict(
    values: Mapping[str, float],
    distributed: DistributedContext,
    *,
    maximum: bool = False,
) -> dict[str, float]:
    """Reduce a stable scalar metric mapping in one collective."""
    names = tuple(sorted(values))
    tensor = torch.tensor([values[name] for name in names], dtype=torch.float64, device=distributed.device)
    operation = dist.ReduceOp.MAX if maximum else dist.ReduceOp.SUM
    dist.all_reduce(tensor, op=operation)
    if not maximum:
        tensor.div_(distributed.world_size)
    return {name: float(value) for name, value in zip(names, tensor.cpu(), strict=True)}


def _nats_to_bits(values: Tensor) -> Tensor:
    return values / math.log(2.0)


def conditioning_protocol() -> dict[str, object]:
    return {
        "version": 2,
        "horizon": RETURN_HORIZON,
        "gamma": 0.99855,
        "reward": "damage_opp-damage_ego+120*(stock_loss_opp-stock_loss_ego)+50*(last_stock_opp-last_stock_ego)",
        "scale": RETURN_SCALE,
        "alignment": "sum(k=1..60, gamma**(k-1)*r[t+k])",
        "availability": "full observed horizon or known terminal with zero rewards thereafter",
        "evaluation": "unconditioned",
        "modulation": "omitted",
        "dropout": "none",
        "calibration_windows": CALIBRATION_WINDOWS,
    }


def future_return_labels(sample: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    reward = returns_lib.frame_reward(
        sample,
        ego="p1",
        opp="p2",
        damage_shaping=1.0,
        win_reward=50.0,
        stock_value=120.0,
    )
    terminal_events = returns_lib.match_point_events(sample["p1_stock"]) + returns_lib.match_point_events(
        sample["p2_stock"]
    )
    terminal_indices = np.flatnonzero(terminal_events)
    terminal = bool(terminal_indices.size) or returns_lib.infer_terminal_replay(sample)
    if terminal_indices.size:
        reward[int(terminal_indices[0]) + 1 :] = 0.0
    frames = len(reward)
    values = np.zeros(frames, dtype=np.float64)
    for offset in range(1, min(RETURN_HORIZON, frames - 1) + 1):
        values[:-offset] += 0.99855 ** (offset - 1) * reward[offset:].astype(np.float64)
    available = (np.arange(frames) + RETURN_HORIZON < frames) | terminal
    values = np.where(available, values, np.nan).astype(np.float32)
    return {
        "p1_return60": values,
        "p2_return60": -values,
        "p1_return60_valid": available,
        "p2_return60_valid": available.copy(),
    }


@dataclass(frozen=True, slots=True)
class ReturnLabels:
    full_match: returns_lib.PolicyReturnLabels

    def __call__(self, compact: Mapping[str, object]) -> dict[str, np.ndarray]:
        labels = self.full_match(compact)
        frames = int(np.asarray(compact["num_frames"]).item())
        labels = {
            name: np.full(frames, value.item(), dtype=value.dtype) if value.shape == () else value
            for name, value in labels.items()
        }
        sample = {}
        for port in ("p1", "p2"):
            sample[f"{port}_percent"] = np.asarray(compact[f"{port}_percent"], dtype=np.float32)
            sample[f"{port}_stock"] = unpack_player_stock(np.asarray(compact[f"{port}_state"]))
        if "mc_terminated" in compact:
            sample["mc_terminated"] = np.asarray(compact["mc_terminated"])
        return {**labels, **future_return_labels(sample)}


@dataclass(frozen=True, slots=True)
class ReturnBatch:
    batch: TrainBatch
    returns: Tensor
    eligible: Tensor
    future_return: Tensor
    available: Tensor
    condition_present: Tensor

    @property
    def context(self) -> Context:
        return self.batch.context

    @property
    def target(self) -> Tensor:
        return self.batch.target

    def to(self, device: str | torch.device) -> ReturnBatch:
        return ReturnBatch(
            self.batch.to(device),
            self.returns.to(device, non_blocking=True),
            self.eligible.to(device, non_blocking=True),
            self.future_return.to(device, non_blocking=True),
            self.available.to(device, non_blocking=True),
            self.condition_present.to(device, non_blocking=True),
        )

    def pin_memory(self) -> ReturnBatch:
        return ReturnBatch(
            self.batch.pin_memory(),
            self.returns.pin_memory(),
            self.eligible.pin_memory(),
            self.future_return.pin_memory(),
            self.available.pin_memory(),
            self.condition_present.pin_memory(),
        )

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        self.batch.record_stream(stream)
        for tensor in (self.returns, self.eligible, self.future_return, self.available, self.condition_present):
            tensor.record_stream(stream)

    def slice(self, count: int) -> ReturnBatch:
        return self.row_slice(0, count)

    def row_slice(self, start: int, stop: int) -> ReturnBatch:
        if not 0 <= start < stop <= self.target.shape[0]:
            raise ValueError("return batch slice is outside the batch")
        rows = slice(start, stop)
        return ReturnBatch(
            TrainBatch(
                Context(
                    {name: value[rows] for name, value in self.context.features.items()},
                    self.context.ctx_pad[rows],
                ),
                self.target[rows],
                None if self.batch.replay_ids is None else self.batch.replay_ids[start:stop],
            ),
            self.returns[rows],
            self.eligible[rows],
            self.future_return[rows],
            self.available[rows],
            self.condition_present[rows],
        )


@contextlib.contextmanager
def _elapsed_heartbeat(message: str) -> Iterator[None]:
    stop = threading.Event()
    started = time.monotonic()
    reporter = threading.Thread(
        target=_report_startup_elapsed, args=(stop, started, message), name="startup-progress", daemon=True
    )
    reporter.start()
    try:
        yield
    finally:
        stop.set()
        reporter.join()


def _report_startup_elapsed(stop: threading.Event, started: float, message: str) -> None:
    while not stop.wait(_STARTUP_LOG_INTERVAL_S):
        print(f"{message}; {time.monotonic() - started:.1f}s elapsed", flush=True)


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
    n_layers: int = 6
    n_heads: int = 16
    attn_window: int = 0
    L_ctx: int = 256

    sample_chunk_length: int = 30
    head_offsets: tuple[int, ...] = tuple(range(1, 31))
    temporal_d_model: int = 1024
    temporal_layers: int = 8
    temporal_heads: int = 16
    temporal_ff_dim: int = 4096
    group_head_dim: int = 1024
    return_embed_dim: int = 0
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
    value_hidden_dim: int = 512
    main_stick_layout: MainStickLayout = "legacy65"

    @property
    def direct_loss_start(self) -> int:
        if self.L_ctx % 2:
            raise ValueError("context length must be even for suffix supervision")
        return self.L_ctx // 2

    @property
    def parameter_count_contract(self) -> dict[str, int]:
        if self == Architecture():
            return {
                "trunk": 75_497_472,
                "temporal_decoder": 106_466_320,
                "group_heads": 4_558_179,
                "trunk_skip_heads": 4_558_179,
                "value_head": 1_049_089,
                "return_conditioner": 0,
                "other": 1_230_790,
                "total": 193_360_029,
            }
        proxy_treatment = Architecture(
            d_model=256,
            n_layers=6,
            n_heads=4,
            sample_chunk_length=30,
            head_offsets=tuple(range(1, 31)),
            temporal_d_model=256,
            temporal_layers=8,
            temporal_heads=4,
            temporal_ff_dim=1408,
            group_head_dim=256,
            value_hidden_dim=128,
        )
        if self == proxy_treatment:
            return {
                "trunk": 4_718_592,
                "temporal_decoder": 8_341_264,
                "group_heads": 353_379,
                "trunk_skip_heads": 353_379,
                "value_head": 65_665,
                "return_conditioner": 0,
                "other": 861_382,
                "total": 14_693_661,
            }
        proxy_control = Architecture(
            d_model=256,
            n_layers=16,
            n_heads=4,
            sample_chunk_length=30,
            head_offsets=tuple(range(1, 31)),
            temporal_d_model=128,
            temporal_layers=4,
            temporal_heads=2,
            temporal_ff_dim=384,
            group_head_dim=128,
            value_hidden_dim=128,
        )
        if self == proxy_control:
            return {
                "trunk": 12_582_912,
                "temporal_decoder": 867_216,
                "group_heads": 111_331,
                "trunk_skip_heads": 176_867,
                "value_head": 65_665,
                "return_conditioner": 0,
                "other": 861_382,
                "total": 14_665_373,
            }
        raise ValueError(f"no parameter contract for architecture {self}")


@dataclass(frozen=True, slots=True)
class AWRCalibration:
    return_suffix: ClassVar[str] = "awr_return"
    near_offsets: ClassVar[int] = 6
    start_update: ClassVar[int] = 4097

    beta: float = 150.0
    weight_max: float = 10.0
    gamma: float = 0.99855
    stock_value: float = 120.0
    damage_shaping: float = 1.0
    win_reward: float = 50.0
    # Regressing the value error in beta units keeps the critic loss O(1)
    # despite the reward's roughly hundred-point scale.
    value_loss_weight: float = 1.0
    # Retained at 1.0 for checkpoint-format compatibility; offsets own the objective.
    auxiliary_loss_weight: float = 1.0

    @property
    def ego_return_column(self) -> str:
        return f"ego_{self.return_suffix}"

    @property
    def ego_return_valid_column(self) -> str:
        return f"{self.ego_return_column}_valid"


@dataclass(frozen=True, slots=True)
class TrainConfig:
    reference_batch_size: ClassVar[int] = 512
    reference_positions: ClassVar[int] = 2**30
    base_adam_betas: ClassVar[tuple[float, float]] = (0.9, 0.95)
    base_adam_eps: ClassVar[float] = 1e-12
    inference_buckets: ClassVar[tuple[int, ...]] = (1, 2, 4, 8, 16, 32, 64)
    train_metrics_every: ClassVar[int] = 25
    train_prefetch_factor: ClassVar[int] = 4
    raw_shard_materialization_threads: ClassVar[int] = 32
    materialization_threads_env: ClassVar[str] = "HAL_O60_MATERIALIZATION_THREADS"
    data_protocol: ClassVar[str] = "o60-rank-partitioned-replay-ring-v1"
    selection_sha256: ClassVar[str] = "2593361352b92e705be3fbeae1b4e9bb1a3c9f1787cd713014a7a95b7df62477"
    mds_index_version: ClassVar[int] = 2
    mds_manifest_schema_sha256: ClassVar[str] = "405199de9494fe01350506734f0b2ec392fe79b0122d69cbcb5cae2afabc0d49"
    replay_slots: ClassVar[int] = 65_536
    windows_per_generation: ClassVar[int] = 8
    replay_phase_block_batches: ClassVar[int] = 25
    minimum_replay_gap_batches: ClassVar[int] = 232
    reserved_disk_bytes: ClassVar[int] = 256 * 2**30
    player_sidecar_remote: ClassVar[str] = "s3://hal/processed/player-identity-v1/professional-code-v1.jsonl.gz"

    arch: Annotated[Architecture, tyro.conf.Suppress] = Architecture()
    awr: Annotated[AWRCalibration, tyro.conf.Suppress] = AWRCalibration()

    prediction_frames: int = 4
    delay_frames: int = 2
    replan_interval_frames: int = 2
    inference_mode: str = "compiled"  # explicit "eager" is for debugging
    # Hardware-derived by default. An explicit power of two is a reproducibility
    # or memory-pressure override, not an architecture parameter.
    compiled_inference_bucket: int | None = None

    seed: int = 0
    eval_seed: int = 0
    batch_size: int = 512
    local_batch_size: int = 256
    microbatch_size: int = 256
    world_size: int = 2
    optimizer: Literal["muon", "adamw"] = "muon"
    muon_lr: float = 0.007
    muon_lr_multiplier: Annotated[float, tyro.conf.Suppress] = 1.0
    muon_weight_decay: float = 1e-4
    adam_lr: float = 4.25e-4
    adam_weight_decay: float = 1e-4
    grad_clip: float = 1.0
    lr_floor_ratio: float = 1 / 170
    amp_dtype: str = "bfloat16"
    allow_tf32: bool = True
    compile_trunk: bool = True
    compile_temporal: bool = True
    train_compile_mode: Literal["reduce-overhead", "max-autotune"] = "reduce-overhead"

    wandb_log_code: bool = True
    val_every: int = 4096
    val_n_samples: int = 2048
    val_batch_size: int = 128
    ckpt_every: int = 2048
    eval_every: int = 8192
    eval_max_frames: int = 7200
    eval_n_matchups: int = 96
    final_eval_n_matchups: int = 96
    eval_max_parallel: int | None = 32

    source_names: tuple[str, ...] = tuple(source.name for source in streams.POLICY_WORLD_V8_SOURCES)
    mds_schema_version: int = 7
    policy_world_schema_version: int = POLICY_WORLD_SCHEMA_VERSION
    download_retry: int = 8
    val_split: str = "val"
    num_workers: int = 12
    push_to_r2: bool = True
    system_metrics_every: int = 25
    system_metrics_interval_s: float = 5.0
    process_metrics_interval_s: float = 30.0
    cache_metrics_interval_s: float = 30.0
    identity_dropout: float = 0.10
    return_conditioning: bool = False
    return_dropout: float = 0.0
    parent_run_name: Annotated[str | None, tyro.conf.Suppress] = None
    parent_checkpoint_name: Annotated[str | None, tyro.conf.Suppress] = None
    parent_checkpoint_sha256: Annotated[str | None, tyro.conf.Suppress] = None
    parent_wandb_id: Annotated[str | None, tyro.conf.Suppress] = None
    player_sidecar_local: str = "data/processed/player-identity-v1/professional-code-v1.jsonl.gz"
    player_sidecar_sha256: str = "54ccf8a2497fe240313117297ca2ea31158e08db2cc53c67e7aa46853a8dac1c"
    player_vocab_sha256: str = "c67c97c995ad033ea7f5b2223efce5b061394566439f091ff6e7aaa6a9d1cfd6"
    player_vocab_size: int = 21_181
    target_positions: int = TARGET_POSITIONS
    stable_updates: int = DECAY_START_UPDATE
    decay_start_update: int | None = DECAY_START_UPDATE
    decay_duration: int | None = COOLDOWN_UPDATES
    depth_alpha: float = 0.5
    hidden_std_multiplier: float = 0.5
    readout_init: Literal["mup-normal"] = "mup-normal"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-12

    @property
    def max_steps(self) -> int:
        updates, remainder = divmod(self.target_positions, self.supervised_positions_per_update)
        if remainder:
            raise ValueError("target_positions must end on an optimizer boundary")
        return updates

    @property
    def warmup_steps(self) -> int:
        return WARMUP_UPDATES

    @property
    def supervised_positions_per_update(self) -> int:
        return self.policy_prefixes_per_update

    @property
    def policy_prefixes_per_update(self) -> int:
        return self.batch_size * POLICY_PREFIXES_PER_WINDOW

    @property
    def value_prefixes_per_update(self) -> int:
        return self.batch_size * (self.arch.L_ctx - self.arch.direct_loss_start)

    @property
    def minimum_replay_frames(self) -> int:
        return self.arch.L_ctx + self.arch.sample_chunk_length + self.windows_per_generation - 1

    @property
    def source_list_sha256(self) -> str:
        encoded = json.dumps(self.source_names, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @property
    def train_replays(self) -> int:
        return sum(streams.POLICY_WORLD_V8_TRAIN_REPLAYS[name] for name in self.source_names)

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
    if compute_flops_per_supervised_position(COMPUTE_EQUIVALENT_PARAMETERS) != FLOPS_PER_SUPERVISED_POSITION:
        raise RuntimeError("O60 compute accounting constants disagree")
    if (
        checkpoint_rounded_updates(
            FITTED_OPTIMAL_POSITIONS,
            positions_per_update=cfg.supervised_positions_per_update,
            checkpoint_interval=cfg.ckpt_every,
        )
        != TARGET_UPDATES
    ):
        raise RuntimeError("O60 fitted optimum no longer rounds to its update boundary")
    if TARGET_UPDATES * cfg.supervised_positions_per_update != TARGET_POSITIONS:
        raise RuntimeError("O60 update and data budgets disagree")
    if DECAY_START_UPDATE + COOLDOWN_UPDATES != TARGET_UPDATES:
        raise RuntimeError("O60 decay boundary and cooldown do not end at the target update")
    if cfg.optimizer != "muon":
        raise ValueError("O60 requires Muon")
    if cfg.train_compile_mode not in ("reduce-overhead", "max-autotune"):
        raise ValueError("unsupported training compile mode")
    production = Architecture()
    proxies = (proxy_config().arch, proxy_control_config().arch)
    if cfg.arch not in (production, *proxies):
        raise ValueError("O60 architecture must be the production shape or one of its fixed proxy arms")
    if cfg.batch_size != 512 or cfg.seed != 0:
        raise ValueError("O60 freezes the global batch at 512 and seed at zero")
    if (cfg.world_size, cfg.local_batch_size) != (2, 256):
        raise ValueError("O60 freezes two ranks with a local batch of 256")
    if cfg.batch_size != cfg.world_size * cfg.local_batch_size:
        raise ValueError("global batch must equal world_size * local_batch_size")
    if cfg.microbatch_size not in (128, 256) or cfg.local_batch_size % cfg.microbatch_size:
        raise ValueError("microbatch_size must be 128 or 256 and divide the local batch")
    if cfg.num_workers != 12 or cfg.replay_slots != 65_536:
        raise ValueError("each O60 rank requires 12 workers and 65,536 replay slots")
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
        "value_hidden_dim": cfg.arch.value_hidden_dim,
        "batch_size": cfg.batch_size,
        "local_batch_size": cfg.local_batch_size,
        "microbatch_size": cfg.microbatch_size,
        "world_size": cfg.world_size,
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
    if offsets[: cfg.awr.near_offsets] != tuple(range(1, cfg.awr.near_offsets + 1)):
        raise ValueError("the near bucket must be the dense offset prefix 1..6")
    if offsets != tuple(range(1, 31)):
        raise ValueError("O60 prediction offsets are frozen")
    if len(OFFSET_LOSS_WEIGHTS) != len(offsets) or any(weight != 1 / 30 for weight in OFFSET_LOSS_WEIGHTS):
        raise ValueError("O60 requires equal 1/30 offset loss coefficients")
    if cfg.arch == Architecture() and cfg.minimum_replay_frames != 293:
        raise ValueError("eight distinct O60 windows require at least 293 replay frames")
    if (cfg.prediction_frames, cfg.delay_frames, cfg.replan_interval_frames) != (4, 2, 2):
        raise ValueError("evaluation protocol is frozen to prediction=4, delay=2, replan=2")
    if (cfg.eval_every, cfg.eval_n_matchups, cfg.final_eval_n_matchups) != (8192, 96, 96):
        raise ValueError("automatic evaluation is frozen to 96 matchups every 8,192 updates and at completion")
    if cfg.inference_mode not in ("compiled", "eager"):
        raise ValueError("inference_mode must be 'compiled' or 'eager'")
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
    if cfg.awr.auxiliary_loss_weight != 1.0:
        raise ValueError("auxiliary_loss_weight must remain 1.0 for checkpoint compatibility")
    if cfg.awr != AWRCalibration():
        raise ValueError("O60 freezes its value and AWR calibration")
    for name, value in (("system_metrics_every", cfg.system_metrics_every),):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    for name, value in (
        ("system_metrics_interval_s", cfg.system_metrics_interval_s),
        ("process_metrics_interval_s", cfg.process_metrics_interval_s),
        ("cache_metrics_interval_s", cfg.cache_metrics_interval_s),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive, got {value!r}")
    if cfg.amp_dtype not in ("bfloat16", "float32"):
        raise ValueError("amp_dtype must be bfloat16 or float32")
    if not isinstance(cfg.num_workers, int) or isinstance(cfg.num_workers, bool):
        raise ValueError(f"num_workers must be an integer, got {cfg.num_workers!r}")
    if not 0.0 < cfg.awr.gamma < 1.0:
        raise ValueError(f"awr_gamma must be in (0, 1), got {cfg.awr.gamma}")
    if not math.isfinite(cfg.awr.beta) or cfg.awr.beta <= 0:
        raise ValueError(f"awr_beta must be finite and positive, got {cfg.awr.beta}")
    if not math.isfinite(cfg.awr.weight_max) or cfg.awr.weight_max <= 1:
        raise ValueError(f"awr_weight_max must be finite and above 1, got {cfg.awr.weight_max}")
    if not math.isfinite(cfg.awr.value_loss_weight) or cfg.awr.value_loss_weight < 0:
        raise ValueError("awr_value_loss_weight must be finite and non-negative")
    if not math.isfinite(cfg.grad_clip) or cfg.grad_clip <= 0:
        raise ValueError(f"grad_clip must be finite and positive, got {cfg.grad_clip}")
    if not 0.0 <= cfg.identity_dropout <= 1.0:
        raise ValueError("identity_dropout must be in [0, 1]")
    if cfg.return_conditioning or cfg.return_dropout != 0.0 or cfg.arch.return_embed_dim != 0:
        raise ValueError("O60 omits return conditioning")
    expected_identity = TrainConfig()
    if cfg.player_vocab_size != expected_identity.player_vocab_size:
        raise ValueError(f"player_vocab_size must be {expected_identity.player_vocab_size}")
    if (
        cfg.player_sidecar_sha256 != expected_identity.player_sidecar_sha256
        or cfg.player_vocab_sha256 != expected_identity.player_vocab_sha256
    ):
        raise ValueError("identity hashes differ from the frozen O49 artifacts")
    if not 0.0 < cfg.lr_floor_ratio <= 1.0:
        raise ValueError("lr_floor_ratio must be in (0, 1]")
    if not cfg.source_names or len(set(cfg.source_names)) != len(cfg.source_names):
        raise ValueError("source_names must be non-empty and unique")
    unknown = set(cfg.source_names) - streams.BY_NAME.keys()
    if unknown:
        raise ValueError(f"unknown source names: {sorted(unknown)}")
    if cfg.policy_world_schema_version != POLICY_WORLD_SCHEMA_VERSION:
        raise ValueError(
            f"policy_world_schema_version {cfg.policy_world_schema_version} != {POLICY_WORLD_SCHEMA_VERSION}"
        )
    if tuple(cfg.source_names) != tuple(source.name for source in streams.POLICY_WORLD_V8_SOURCES):
        raise ValueError("O60 requires all 44 policy-world-v8 sources")
    if cfg.depth_alpha != 0.5 or cfg.hidden_std_multiplier != 0.5 or cfg.readout_init != "mup-normal":
        raise ValueError("O60 uses O50's selected depth and initialization parameterization")
    if (cfg.adam_beta1, cfg.adam_beta2, cfg.adam_eps) != (*cfg.base_adam_betas, cfg.base_adam_eps):
        raise ValueError("O60 scales the fixed base Adam betas and epsilon")
    for name, value in (
        ("muon_lr", cfg.muon_lr),
        ("muon_lr_multiplier", cfg.muon_lr_multiplier),
        ("adam_lr", cfg.adam_lr),
        ("adam_eps", cfg.adam_eps),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name, value in (
        ("muon_weight_decay", cfg.muon_weight_decay),
        ("adam_weight_decay", cfg.adam_weight_decay),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if (
        cfg.muon_lr != 0.007
        or cfg.muon_weight_decay != 1e-4
        or cfg.adam_lr != 4.25e-4
        or cfg.adam_weight_decay != 1e-4
        or cfg.grad_clip != 1.0
    ):
        raise ValueError("O60 optimizer hyperparameters are frozen")
    if cfg.muon_lr_multiplier != 1.0:
        raise ValueError("O60 freezes optimizer hyperparameters across continuations")
    if cfg.arch == Architecture() and (
        cfg.max_steps != TARGET_UPDATES
        or cfg.target_positions != TARGET_POSITIONS
        or cfg.warmup_steps != WARMUP_UPDATES
        or cfg.stable_updates != DECAY_START_UPDATE
        or cfg.decay_start_update != DECAY_START_UPDATE
        or cfg.decay_duration != COOLDOWN_UPDATES
    ):
        raise ValueError("O60 training boundaries are frozen")
    if cfg.arch != Architecture() and (cfg.decay_start_update is not None or cfg.decay_duration is not None):
        raise ValueError("the O60 proxy has no production cooldown")
    if any(
        value is not None
        for value in (
            cfg.parent_run_name,
            cfg.parent_checkpoint_name,
            cfg.parent_checkpoint_sha256,
            cfg.parent_wandb_id,
        )
    ):
        raise ValueError("O60 does not fork the schedule from a parent checkpoint")


def proxy_config() -> TrainConfig:
    """Return the 15M treatment proxy with a shallow trunk and deeper decoder."""
    return TrainConfig(
        arch=Architecture(
            d_model=256,
            n_layers=6,
            n_heads=4,
            sample_chunk_length=30,
            head_offsets=tuple(range(1, 31)),
            temporal_d_model=256,
            temporal_layers=8,
            temporal_heads=4,
            temporal_ff_dim=1408,
            group_head_dim=256,
            value_hidden_dim=128,
        ),
        target_positions=TrainConfig.reference_positions,
        stable_updates=TrainConfig.reference_positions // (512 * POLICY_PREFIXES_PER_WINDOW),
        decay_start_update=None,
        decay_duration=None,
    )


def proxy_control_config() -> TrainConfig:
    """Return the prior 15M allocation used as the proxy control."""
    return TrainConfig(
        arch=Architecture(
            d_model=256,
            n_layers=16,
            n_heads=4,
            sample_chunk_length=30,
            head_offsets=tuple(range(1, 31)),
            temporal_d_model=128,
            temporal_layers=4,
            temporal_heads=2,
            temporal_ff_dim=384,
            group_head_dim=128,
            value_hidden_dim=128,
        ),
        target_positions=TrainConfig.reference_positions,
        stable_updates=TrainConfig.reference_positions // (512 * POLICY_PREFIXES_PER_WINDOW),
        decay_start_update=None,
        decay_duration=None,
    )


ProxyArm = Literal["none", "control", "treatment"]


def proxy_config_for_arm(arm: ProxyArm) -> TrainConfig:
    """Resolve a command-line proxy arm without an implicit fallback."""
    if arm == "none":
        return TrainConfig()
    if arm == "control":
        return proxy_control_config()
    if arm == "treatment":
        return proxy_config()
    raise ValueError(f"unknown proxy arm {arm!r}")


def synthetic_context(cfg: TrainConfig, batch_size: int, device: torch.device) -> Context:
    """Build the fixed base observation with projectile columns."""
    context = build_synthetic_context(
        cfg.arch.L_ctx,
        batch_size,
        device,
        items=True,
    )
    return canonical_context(
        Context(
            features={
                **context.features,
                "ego_player_id": torch.zeros(batch_size, cfg.arch.L_ctx, dtype=torch.long, device=device),
            },
            ctx_pad=context.ctx_pad,
        ),
        items=True,
    )


def synthetic_awr_batch(cfg: TrainConfig, device: torch.device) -> ReturnBatch:
    """Build one fully valid production-shaped batch without touching the corpus."""
    context = synthetic_context(cfg, cfg.local_batch_size, device)
    target = torch.zeros(cfg.local_batch_size, cfg.arch.sample_chunk_length, ACTION_DIM, device=device)
    returns = torch.zeros(cfg.local_batch_size, cfg.arch.L_ctx, device=device)
    eligible = torch.ones(cfg.local_batch_size, cfg.arch.L_ctx, dtype=torch.bool, device=device)
    return ReturnBatch(
        TrainBatch(context=context, target=target),
        returns,
        eligible,
        returns.clone(),
        eligible.clone(),
        eligible.clone(),
    )


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


def model_config(cfg: TrainConfig) -> ActionSequenceConfig:
    """Translate saved experiment choices to the neural model contract."""
    return ActionSequenceConfig(
        **asdict(cfg.arch),
        player_vocab_size=cfg.player_vocab_size,
        player_vocab_sha256=cfg.player_vocab_sha256,
        depth_alpha=cfg.depth_alpha,
        hidden_std_multiplier=cfg.hidden_std_multiplier,
        return_conditioning=cfg.return_conditioning,
    )


def make_model(cfg: TrainConfig, vocabulary: PlayerVocabulary | None = None) -> ActionSequenceTransformer:
    return ActionSequenceTransformer(
        model_config(cfg),
        vocabulary,
        live_horizons=(cfg.prediction_frames,),
        training_diagnostics=cfg.optimizer == "adamw",
    )


def observe_calibration(calibration: ReturnCalibration, batch: ReturnBatch) -> None:
    if batch.batch.replay_ids is None:
        raise ValueError("calibration requires replay identities")
    calibration.observe(
        future_returns_BL=batch.future_return,
        available_BL=batch.available,
        replay_ids=batch.batch.replay_ids,
    )


class IdentityMasker:
    """Checkpointable, per-window identity dropout independent of all other RNGs."""

    def __init__(self, seed: int, probability: float) -> None:
        self.probability = probability
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.forced = 0
        self.total = 0
        self.masked = 0

    def __call__(self, batch: ReturnBatch) -> ReturnBatch:
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
            replay_ids=batch.batch.replay_ids,
        )
        return replace(batch, batch=train_batch)

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


class ReturnMasker:
    """Checkpointable per-position dropout, independent of prefix selection."""

    def __init__(self, seed: int, probability: float, *, enabled: bool = True) -> None:
        if not 0 <= probability <= 1 or not isinstance(enabled, bool):
            raise ValueError("invalid return dropout configuration")
        self.probability = probability
        self.enabled = enabled
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def __call__(self, batch: ReturnBatch) -> ReturnBatch:
        keep = torch.rand(batch.available.shape, generator=self.generator) >= self.probability
        return replace(batch, condition_present=batch.available & keep & self.enabled)

    def state_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "probability": self.probability,
            "enabled": self.enabled,
            "generator": self.generator.get_state(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if (
            set(state) != {"version", "probability", "enabled", "generator"}
            or state["version"] != 1
            or state["probability"] != self.probability
            or state["enabled"] != self.enabled
        ):
            raise ValueError("incompatible return masker state")
        generator = state["generator"]
        if not isinstance(generator, Tensor):
            raise TypeError("return masker RNG state must be a tensor")
        self.generator.set_state(generator.cpu())


class PrefixSampler:
    """Checkpointable uniform sampling without replacement on the device."""

    def __init__(self, seed: int, device: torch.device | str) -> None:
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)

    def sample(self, ctx_pad: Tensor, *, length: int, suffix_start: int, validated_on_cpu: bool = False) -> Tensor:
        if ctx_pad.ndim != 1:
            raise ValueError("ctx_pad must be one-dimensional")
        positions = torch.arange(suffix_start, length, device=ctx_pad.device)
        valid = positions[None, :] >= ctx_pad[:, None]
        if length - suffix_start < POLICY_PREFIXES_PER_WINDOW:
            raise ValueError("every O60 window must expose at least 32 real suffix positions")
        if not validated_on_cpu and not bool((valid.sum(dim=1) >= POLICY_PREFIXES_PER_WINDOW).all()):
            raise ValueError("every O60 window must expose at least 32 real suffix positions")
        draws = torch.rand(ctx_pad.shape[0], positions.numel(), device=ctx_pad.device, generator=self.generator)
        draws = draws.masked_fill(~valid, torch.inf)
        selected = draws.topk(POLICY_PREFIXES_PER_WINDOW, dim=1, largest=False).indices
        return positions[selected].sort(dim=1).values

    def state_dict(self) -> dict[str, object]:
        return {"generator": self.generator.get_state()}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        generator = state.get("generator")
        if not isinstance(generator, Tensor):
            raise TypeError("prefix RNG state must be a tensor")
        # CUDA generators also require their serialized ByteTensor on the CPU.
        self.generator.set_state(generator.cpu())


@jaxtyped(typechecker=beartype)
def prepared_targets(
    model: ActionSequenceTransformer, batch: TrainBatch | ReturnBatch
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
        loader: Iterable[ReturnBatch],
        cfg: TrainConfig,
        device: str | torch.device,
        identity_masker: IdentityMasker | None = None,
        *,
        return_masker: ReturnMasker | None = None,
        calibration: ReturnCalibration | None = None,
        iterator: Iterator[ReturnBatch] | None = None,
        first_batch_future: Future[ReturnBatch] | None = None,
    ) -> None:
        self._loader = loader
        self._iterator = iter(loader) if iterator is None else iterator
        self._cfg = cfg
        self._device = torch.device(device)
        self._identity_masker = identity_masker
        self._return_masker = return_masker
        self._calibration = calibration
        self._copy_stream = torch.cuda.Stream(device=self._device) if self._device.type == "cuda" else None
        self._staged: tuple[ReturnBatch, ReturnBatch, int] | None = None
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="device-batch-prefetch")
        self._futures: deque[Future[ReturnBatch]] = deque()
        if first_batch_future is None:
            self.fill_lookahead(1)
        else:
            self._futures.append(first_batch_future)
        self.stage_next()

    def _load_cpu_batch(self) -> ReturnBatch:
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self._loader)
            return next(self._iterator)

    def _prepare_cpu_batch(self, cpu_batch: ReturnBatch) -> ReturnBatch:
        """Apply the parent-side transforms to an already-fetched batch."""
        if not isinstance(cpu_batch, ReturnBatch):
            raise TypeError(f"advantage loader yielded {type(cpu_batch).__name__}, expected ReturnBatch")
        if self._identity_masker is not None:
            cpu_batch = self._identity_masker(cpu_batch)
        if self._return_masker is not None:
            cpu_batch = self._return_masker(cpu_batch)
        validate_batch_geometry(cpu_batch, self._cfg, self._cfg.local_batch_size)
        return cpu_batch

    def _stage(self, cpu_batch: ReturnBatch) -> None:
        start = self._cfg.arch.direct_loss_start
        if bool((cpu_batch.context.ctx_pad > start).any()):
            raise ValueError("O60 requires all 128 suffix positions for dense value and AWR normalization")
        valid_prefixes = self._cfg.local_batch_size * POLICY_PREFIXES_PER_WINDOW
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

    def next(self) -> tuple[ReturnBatch, int]:
        """Wait only for the uncovered tail of the staged transfer."""
        if self._staged is None:
            raise RuntimeError("preload a batch before consuming it")
        device_batch, cpu_batch, valid_prefixes = self._staged
        if self._copy_stream is not None:
            compute_stream = torch.cuda.current_stream(self._device)
            compute_stream.wait_stream(self._copy_stream)
            device_batch.record_stream(compute_stream)
        self._staged = None
        if self._calibration is not None:
            observe_calibration(self._calibration, cpu_batch)
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
        """Release the background loader thread."""
        self._pool.shutdown(wait=True, cancel_futures=True)


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


def collate_awr_batch(windows: list[dict], batch: TrainBatch, *, L_ctx: int) -> ReturnBatch:
    """Attach ``G_{t+1}`` and its validity mask to each context position."""
    batch = _canonical_training_batch(batch)
    next_frames = slice(1, L_ctx + 1)
    calibration = AWRCalibration()
    returns = np.stack([window[calibration.ego_return_column] for window in windows])[:, next_frames]
    eligible = np.stack([window[calibration.ego_return_valid_column] for window in windows])[:, next_frames]
    return ReturnBatch(
        batch=batch,
        returns=torch.from_numpy(np.ascontiguousarray(returns)),
        eligible=torch.from_numpy(np.ascontiguousarray(eligible)).bool(),
        future_return=torch.from_numpy(np.stack([window["ego_return60"][:L_ctx] for window in windows])),
        available=torch.from_numpy(np.stack([window["ego_return60_valid"][:L_ctx] for window in windows])).bool(),
        condition_present=torch.from_numpy(
            np.stack([window["ego_return60_valid"][:L_ctx] for window in windows])
        ).bool(),
    )


def _canonical_training_batch(batch: TrainBatch) -> TrainBatch:
    """Give compiled training and live inference the same fixed feature keys."""
    return TrainBatch(canonical_context(batch.context, items=True), batch.target, batch.replay_ids)


@jaxtyped(typechecker=beartype)
def advantage_weights(
    advantage: Float[Tensor, "*batch"],
    eligible: Bool[Tensor, "*batch"],
    *,
    beta: float,
    weight_max: float,
    active: bool = True,
    valid: Bool[Tensor, "*batch"] | None = None,
) -> tuple[Float[Tensor, "*batch"], dict[str, Float[Tensor, ""]]]:
    """Compute capped, batch-normalized weights from detached advantages."""
    if valid is None:
        valid = torch.ones_like(eligible)
    eligible = eligible & valid
    eligible_float = eligible.float()
    eligible_count = eligible_float.sum()
    eligible_denominator = eligible_count.clamp_min(1)
    valid_count = valid.float().sum().clamp_min(1)

    safe_advantage = torch.where(eligible, advantage, 0).float()

    max_log_weight = math.log(weight_max)
    log_weights = (safe_advantage / beta).clamp(max=max_log_weight)
    eligible_log_weights = log_weights.masked_fill(~eligible, -torch.inf)
    log_mean_weight = torch.logsumexp(eligible_log_weights.reshape(-1), dim=0) - eligible_count.clamp_min(1).log()
    log_mean_weight = torch.where(eligible_count > 0, log_mean_weight, torch.zeros_like(log_mean_weight))
    normalized_log_weights = torch.where(eligible, log_weights - log_mean_weight, 0)
    normalized_weights = torch.exp(normalized_log_weights)
    active_weights = normalized_weights if active else torch.ones_like(normalized_weights)
    weights = torch.where(eligible, active_weights, torch.ones_like(active_weights))

    has_eligible = eligible_count > 0
    advantage_scale = (safe_advantage.abs() * eligible_float).max().clamp_min(torch.finfo(torch.float32).tiny)
    scaled_advantage = safe_advantage / advantage_scale
    scaled_mean = (scaled_advantage * eligible_float).sum() / eligible_denominator
    scaled_variance = ((scaled_advantage - scaled_mean).square() * eligible_float).sum() / eligible_denominator
    advantage_mean = scaled_mean * advantage_scale
    advantage_std = scaled_variance.sqrt() * advantage_scale
    weight_sum = (active_weights * eligible_float).sum()
    squared_sum = (active_weights.square() * eligible_float).sum()
    raw_ess = weight_sum.square() / (eligible_count * squared_sum).clamp_min(torch.finfo(torch.float32).tiny)
    zero = torch.zeros((), device=advantage.device)
    stats = {
        "advantage_mean": torch.where(has_eligible, advantage_mean, zero),
        "advantage_std": torch.where(has_eligible, advantage_std, zero),
        "weight_ess": torch.where(has_eligible, raw_ess, torch.ones_like(zero)),
        "weight_clip_frac": torch.where(
            has_eligible,
            ((log_weights >= max_log_weight).float() * eligible_float).sum() / eligible_denominator,
            zero,
        ),
        "weight_mean": torch.where(has_eligible, weight_sum / eligible_denominator, torch.ones_like(zero)),
        "weight_max": torch.where(
            has_eligible,
            active_weights.masked_fill(~eligible, 0).max(),
            torch.ones_like(zero),
        ),
        "weight_min": torch.where(
            has_eligible,
            active_weights.masked_fill(~eligible, float("inf")).min(),
            torch.ones_like(zero),
        ),
        "eligible_frac": eligible_count / valid_count,
    }
    return weights, stats


def masked_correlation(x: Tensor, y: Tensor, selected: Tensor) -> Tensor:
    """Return a finite Pearson correlation over selected entries."""
    selected_float = selected.float()
    count = selected_float.sum()
    denominator = count.clamp_min(1)
    x_values = torch.where(selected, x.float(), 0)
    y_values = torch.where(selected, y.float(), 0)
    x_centered = torch.where(selected, x_values - x_values.sum() / denominator, 0)
    y_centered = torch.where(selected, y_values - y_values.sum() / denominator, 0)
    covariance = (x_centered * y_centered).sum()
    scale = (x_centered.square().sum() * y_centered.square().sum()).sqrt()
    correlation = covariance / scale.clamp_min(torch.finfo(torch.float32).tiny)
    return torch.where((count > 1) & (scale > 0), correlation, torch.zeros_like(correlation))


@jaxtyped(typechecker=beartype)
def value_objective(
    value: Float[Tensor, "B L_ctx"],
    return_target: Float[Tensor, "B L_ctx"],
    eligible: Bool[Tensor, "B L_ctx"],
    *,
    beta: float,
    valid: Bool[Tensor, "B L_ctx"],
) -> tuple[Float[Tensor, ""], Float[Tensor, "B L_ctx"], dict[str, Float[Tensor, ""]]]:
    """Fit ``V(s_t)`` to ``G_{t+1}`` and return a detached advantage."""
    selected = eligible & valid
    selected_float = selected.float()
    count = selected_float.sum().clamp_min(1)
    value_float = value.float()
    return_float = return_target.float()
    error = torch.where(selected, (value_float - return_float) / beta, 0)
    value_loss = error.square().sum() / count
    advantage = torch.where(selected, return_float - value_float.detach(), 0)
    stats = {
        "value_loss": value_loss.detach(),
        "value_rmse": value_loss.detach().sqrt() * beta,
        "value_mean": (value_float.detach() * selected_float).sum() / count,
        "return_mean": torch.where(selected, return_float, 0).sum() / count,
    }
    return value_loss, advantage, stats


@jaxtyped(typechecker=beartype)
def temporal_objective_parts(
    nll: Float[Tensor, "*prefix n_offsets n_groups"],
    weight: Float[Tensor, "*prefix"],
    *,
    valid_prefixes: int,
    valid: Bool[Tensor, "*prefix"],
) -> tuple[Float[Tensor, ""], Float[Tensor, ""], Float[Tensor, ""]]:
    """Apply the preregistered per-offset coefficients exactly once."""
    n_offsets = nll.shape[-2]
    if n_offsets != len(OFFSET_LOSS_WEIGHTS):
        raise ValueError(f"expected {len(OFFSET_LOSS_WEIGHTS)} offsets, got {n_offsets}")
    joint_nll = nll.float().sum(dim=-1)
    joint_nll = torch.where(valid[..., None], joint_nll, 0)
    awr_weights = weight.float()[..., None]
    near_offsets = AWRCalibration.near_offsets
    coefficients = torch.tensor(OFFSET_LOSS_WEIGHTS, device=nll.device)
    near = (joint_nll[..., :near_offsets] * awr_weights * coefficients[:near_offsets]).sum() / valid_prefixes
    far = (joint_nll[..., near_offsets:] * coefficients[near_offsets:]).sum() / valid_prefixes
    total = near + far
    return near, far, total


def projection_local_target_gradient_l1(
    nll: Tensor,
    weight: Tensor,
    *,
    offsets: tuple[int, ...],
    valid_prefixes: int,
    valid: Tensor,
) -> dict[str, Tensor]:
    """Attribute the exact policy gradient on each target logit."""
    near_offsets = AWRCalibration.near_offsets
    if len(offsets) != len(OFFSET_LOSS_WEIGHTS) or nll.shape[-2] != len(offsets):
        raise ValueError("gradient attribution requires all O60 offsets")
    offset_coefficients = torch.tensor(OFFSET_LOSS_WEIGHTS, device=nll.device)
    coefficients = offset_coefficients.expand(*weight.shape, -1).clone()
    coefficients[..., :near_offsets] *= weight.detach().float()[..., None]
    coefficients /= valid_prefixes
    target_derivative = -torch.expm1(-nll.detach().float())
    values = torch.where(valid[..., None, None], target_derivative * coefficients[..., None], 0).sum(dim=(0, 1))
    metrics: dict[str, Tensor] = {}
    for depth, offset in enumerate(offsets):
        for group, name in enumerate(CONTROLLER_GROUP_NAMES):
            metrics[f"diagnostics/projection_local/target_logit_grad_l1/o{offset:02d}/{name}"] = values[depth, group]
        metrics[f"diagnostics/projection_local/target_logit_grad_l1/by_offset/o{offset:02d}"] = values[depth].sum()
    for group, name in enumerate(CONTROLLER_GROUP_NAMES):
        metrics[f"diagnostics/projection_local/target_logit_grad_l1/by_action_group/{name}"] = values[:, group].sum()
    metrics["diagnostics/projection_local/target_logit_grad_l1/total"] = values.sum()
    return metrics


def value_with_activation_diagnostics(head: SwiGLU, inputs: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
    """Evaluate the value head and measure its existing projection path."""
    gate_output, value_output = head.up(inputs).chunk(2, dim=-1)
    down_input = F.silu(gate_output) * value_output
    output = head.down(down_input)
    prefix = "diagnostics/activations/value"
    metrics = {
        **activation_input_metrics(f"{prefix}/gate", inputs),
        **activation_output_metrics(f"{prefix}/gate", gate_output),
        **activation_input_metrics(f"{prefix}/value", inputs),
        **activation_output_metrics(f"{prefix}/value", value_output),
        **activation_input_metrics(f"{prefix}/down", down_input),
        **activation_output_metrics(f"{prefix}/down", output),
    }
    return output, metrics


def microbatch_loss(
    model: ActionSequenceTransformer,
    batch: ReturnBatch,
    cfg: TrainConfig,
    *,
    step: int,
    valid_prefixes: int,
    trunk_fn: Callable,
    temporal_fn: Callable,
    prefix_positions: Tensor,
    value_loss_scale: float = 1.0,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Compute the learned-value AWR policy and critic losses.

    Weighting stays outside the compiled policy functions, so crossing the
    warmup boundary does not trigger recompilation. Logged NLLs are unweighted.
    """
    if not isinstance(batch, ReturnBatch):
        raise TypeError(f"advantage training needs an ReturnBatch, got {type(batch).__name__}")
    dense_history, dense_targets, dense_valid = prepared_targets(model, batch)
    with amp_context(cfg, DEVICE):
        hidden = trunk_fn(batch.context.features, batch.context.ctx_pad, None)
        suffix_start = cfg.arch.direct_loss_start
        local_positions = prefix_positions - suffix_start
        batch_indices = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
        history = dense_history[batch_indices, local_positions]
        targets = dense_targets[batch_indices, local_positions]
        valid = dense_valid[batch_indices, local_positions]
        temporal_output = temporal_fn(
            hidden,
            batch.context.ctx_pad,
            prefix_positions,
            history,
            targets,
            batch.future_return[batch_indices, prefix_positions],
            batch.condition_present[batch_indices, prefix_positions],
        )
        if isinstance(temporal_output, Tensor):
            dense_nll = temporal_output
            button_diagnostics: dict[str, Tensor] = {}
        else:
            dense_nll, button_diagnostics = temporal_output
    value_hidden = hidden[:, suffix_start:]
    value_features = decoder_rmsnorm(value_hidden).detach()
    if cfg.optimizer == "adamw":
        value_output, value_diagnostics = value_with_activation_diagnostics(model.value_head, value_features.float())
    else:
        value_output = model.value_head(value_features.float())
        value_diagnostics = {}
    value = value_output.squeeze(-1)
    value_loss, advantage, value_stats = value_objective(
        value,
        batch.returns[:, suffix_start:],
        batch.eligible[:, suffix_start:],
        beta=cfg.awr.beta,
        valid=dense_valid,
    )
    active = step + 1 >= cfg.awr.start_update
    weights, stats = advantage_weights(
        advantage,
        batch.eligible[:, suffix_start:],
        beta=cfg.awr.beta,
        weight_max=cfg.awr.weight_max,
        active=active,
        valid=dense_valid,
    )
    sampled_weights = weights[batch_indices, local_positions]
    button_loss = dense_nll[..., BUTTONS_GROUP].float().mean(dim=-1)
    stats["weight_button_loss_correlation"] = masked_correlation(
        sampled_weights,
        button_loss,
        batch.eligible[:, suffix_start:][batch_indices, local_positions] & valid,
    )
    near, far, policy_loss = temporal_objective_parts(
        dense_nll,
        sampled_weights,
        valid_prefixes=valid_prefixes,
        valid=valid,
    )
    projection_attribution = (
        projection_local_target_gradient_l1(
            dense_nll,
            sampled_weights,
            offsets=cfg.arch.head_offsets,
            valid_prefixes=valid_prefixes,
            valid=valid,
        )
        if cfg.optimizer == "adamw"
        else {}
    )
    loss = policy_loss + value_loss_scale * cfg.awr.value_loss_weight * value_loss
    nll_sum = torch.where(valid[..., None, None], dense_nll.float(), 0).sum(dim=(0, 1))
    extra = {
        "train/loss": _nats_to_bits(policy_loss.detach()),
        "train/near_loss": _nats_to_bits(near.detach()),
        "train/far_nll": _nats_to_bits(far.detach()),
        "train/objective": loss.detach(),
        "value/loss": value_stats["value_loss"],
        "value/rmse": value_stats["value_rmse"],
        "awr/active": torch.ones_like(loss) if active else torch.zeros_like(loss),
        "awr/eligible_fraction": stats["eligible_frac"],
        "awr/weight_mean": stats["weight_mean"],
        "awr/weight_max": stats["weight_max"],
        "awr/ess_fraction": stats["weight_ess"],
        "awr/cap_fraction": stats["weight_clip_frac"],
        "awr/button_loss_correlation": stats["weight_button_loss_correlation"],
        **button_diagnostics,
        **value_diagnostics,
        **projection_attribution,
    }
    return loss, nll_sum.detach(), extra


def nll_mean_metrics(
    mean_nll: Tensor,
    offsets: tuple[int, ...],
) -> dict[str, float]:
    if mean_nll.shape != (len(offsets), CONTROLLER_GROUP_COUNT):
        raise ValueError(f"mean NLL has shape {tuple(mean_nll.shape)}")
    joint = _nats_to_bits(mean_nll.sum(dim=-1))
    if len(offsets) <= AWRCalibration.near_offsets:
        raise ValueError(
            f"the {AWRCalibration.near_offsets} near offsets must leave at least one far offset, got {len(offsets)}"
        )
    coefficients = torch.tensor(OFFSET_LOSS_WEIGHTS, device=mean_nll.device)
    near = (joint[: AWRCalibration.near_offsets] * coefficients[: AWRCalibration.near_offsets]).sum()
    far = (joint[AWRCalibration.near_offsets :] * coefficients[AWRCalibration.near_offsets :]).sum()
    total = near + far
    out = {
        "loss_unweighted": float(total),
        "temporal_loss_near_unweighted": float(near),
        "temporal_loss_far_unweighted": float(far),
    }
    for depth, offset in enumerate(offsets):
        out[f"nll_o{offset:02d}"] = float(joint[depth])
        for group, name in enumerate(CONTROLLER_GROUP_NAMES):
            out[f"nll_o{offset:02d}_{name}"] = float(_nats_to_bits(mean_nll[depth, group]))
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
def val_metrics(model: ActionSequenceTransformer, batches: list[ReturnBatch], cfg: TrainConfig) -> dict[str, float]:
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
                prefix_positions = torch.arange(cfg.arch.direct_loss_start, cfg.arch.L_ctx, device=device).expand(
                    hidden.shape[0], -1
                )
                logits = model.temporal.teacher_forced_logits_by_group(
                    hidden,
                    batch.context.ctx_pad,
                    prefix_positions,
                    history,
                    targets,
                    batch.future_return[:, cfg.arch.direct_loss_start :],
                    batch.available[:, cfg.arch.direct_loss_start :],
                )
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
                    hidden[row_valid],
                    last_observed,
                    batch.future_return[row_valid, -1],
                    batch.available[row_valid, -1],
                    ctx_pad=batch.context.ctx_pad[row_valid],
                )
            sampled = sampled_all[:, :6]
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
    )
    for depth, offset in enumerate(model.head_offsets):
        for group, name in enumerate(CONTROLLER_GROUP_NAMES):
            out[f"acc_o{offset:02d}_{name}"] = float(correct[depth, group] / count)
            denominator = float(exposure_count[depth, group])
            if denominator <= 0:
                raise RuntimeError(f"validation has no compatible rollout rows for offset {offset} group {name}")
            roll_nll = float(_nats_to_bits(rollout_nll[depth, group] / denominator))
            teacher_nll = float(_nats_to_bits(teacher_exposure_nll[depth, group] / denominator))
            out[f"rollout_nll_o{offset:02d}_{name}"] = roll_nll
            out[f"exposure_gap_o{offset:02d}_{name}"] = roll_nll - teacher_nll
            out[f"rollout_acc_o{offset:02d}_{name}"] = float(rollout_correct[depth, group] / count)
        out[f"rollout_button_target_masked_rate_o{offset:02d}"] = float(button_incompatible[depth] / count)
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
        "nll": values["loss_unweighted"],
        "near_nll": values["temporal_loss_near_unweighted"],
        "far_nll": values["temporal_loss_far_unweighted"],
        "rollout_nll": rollout_nll,
        "exposure_gap": exposure_gap,
        "exact_frame_acc": values["exact_frame_acc"],
        "sequence_acc": values["dense_four_sequence_acc"],
        "change_f1": values["change_f1"],
        "sampled_transition_rate": values["sampled_transition_rate"],
    }


def _offline_validation_metrics(
    model: ActionSequenceTransformer,
    batches: list[ReturnBatch],
    cfg: TrainConfig,
) -> dict[str, float]:
    """Permit validation-only FlexAttention graphs outside the training guard."""
    stance = (
        torch.compiler.set_stance("default")
        if DEVICE == "cuda" and (cfg.compile_trunk or cfg.compile_temporal)
        else contextlib.nullcontext()
    )
    with stance:
        return _validation_wandb_metrics(val_metrics(model, batches, cfg), cfg)


def _slice_train_batch(batch: TrainBatch, rows: int) -> TrainBatch:
    if rows < 1 or rows > batch.context.batch:
        raise ValueError(f"cannot take {rows} rows from a batch of {batch.context.batch}")
    context = batch.context
    return TrainBatch(
        context=Context(
            features={name: value[:rows].detach().cpu().clone() for name, value in context.features.items()},
            ctx_pad=context.ctx_pad[:rows].detach().cpu().clone(),
        ),
        target=batch.target[:rows].detach().cpu().clone(),
        replay_ids=None if batch.replay_ids is None else batch.replay_ids[:rows],
    )


def _train_batch_state(batch: TrainBatch) -> dict[str, object]:
    return {
        "features": {name: value.detach().cpu().contiguous() for name, value in batch.context.features.items()},
        "ctx_pad": batch.context.ctx_pad.detach().cpu().contiguous(),
        "target": batch.target.detach().cpu().contiguous(),
        "replay_ids": batch.replay_ids,
    }


def _train_batch_from_state(state: dict[str, object]) -> TrainBatch:
    if set(state) != {"features", "ctx_pad", "target", "replay_ids"}:
        raise ValueError("fixed diagnostic batch state has the wrong fields")
    features = state["features"]
    ctx_pad = state["ctx_pad"]
    target = state["target"]
    replay_ids = state["replay_ids"]
    if not isinstance(features, dict) or not all(
        isinstance(name, str) and isinstance(value, Tensor) for name, value in features.items()
    ):
        raise TypeError("fixed diagnostic features must be tensors keyed by name")
    if not isinstance(ctx_pad, Tensor) or not isinstance(target, Tensor):
        raise TypeError("fixed diagnostic context padding and target must be tensors")
    if replay_ids is not None and not (
        isinstance(replay_ids, tuple) and all(isinstance(replay_id, str) for replay_id in replay_ids)
    ):
        raise TypeError("fixed diagnostic replay ids must be a tuple of strings or None")
    cpu_features = {name: value.detach().cpu() for name, value in cast(dict[str, Tensor], features).items()}
    return TrainBatch(
        Context(cpu_features, ctx_pad.detach().cpu()),
        target.detach().cpu(),
        cast(tuple[str, ...] | None, replay_ids),
    )


def _train_batch_sha256(batch: TrainBatch) -> str:
    digest = hashlib.sha256()
    state = _train_batch_state(batch)
    features = cast(dict[str, Tensor], state["features"])
    tensors = [(f"feature/{name}", features[name]) for name in sorted(features)]
    tensors.extend((("ctx_pad", cast(Tensor, state["ctx_pad"])), ("target", cast(Tensor, state["target"]))))
    for name, tensor in tensors:
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    digest.update(json.dumps(state["replay_ids"], separators=(",", ":")).encode())
    return digest.hexdigest()


RETURN_BATCH_FIELDS = ("returns", "eligible", "future_return", "available", "condition_present")


def _return_batch_state(batch: ReturnBatch) -> dict[str, object]:
    return {
        "batch": _train_batch_state(batch.batch),
        **{name: getattr(batch, name).detach().cpu().contiguous() for name in RETURN_BATCH_FIELDS},
    }


def _return_batch_from_state(state: Mapping[str, object]) -> ReturnBatch:
    if set(state) != {"batch", *RETURN_BATCH_FIELDS} or not isinstance(state["batch"], dict):
        raise ValueError("incompatible return batch fields")
    tensors = [state[name] for name in RETURN_BATCH_FIELDS]
    if not all(isinstance(value, Tensor) for value in tensors):
        raise TypeError("return batch labels and masks must be tensors")
    return ReturnBatch(_train_batch_from_state(cast(dict[str, object], state["batch"])), *cast(list[Tensor], tensors))


def _return_batch_sha256(batch: ReturnBatch) -> str:
    digest = hashlib.sha256(_train_batch_sha256(batch.batch).encode())
    for name in RETURN_BATCH_FIELDS:
        tensor = getattr(batch, name).detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def make_window_policy(
    model: ActionSequenceTransformer,
    cfg: TrainConfig,
    *,
    bucket: int | None = None,
    compiled: bool | None = None,
    compile_mode: str = "default",
    compiled_buckets: tuple[int, ...] | None = None,
    temperature: float = 1.0,
    desired_return: float | None = None,
) -> WindowPolicy:
    """Translate 059 evaluation choices into the shared dense executor."""
    if bucket is not None and compiled_buckets is not None:
        raise ValueError("pass bucket or compiled_buckets, not both")
    use_compiled = cfg.inference_mode == "compiled" if compiled is None else compiled
    prepared = (bucket,) if bucket is not None else compiled_buckets
    if prepared is None:
        prepared = _planned_inference_buckets(cfg) if use_compiled else cfg.inference_buckets
    return WindowPolicy(
        model,
        context_frames=cfg.arch.L_ctx,
        prediction_frames=cfg.prediction_frames,
        prepared_buckets=prepared,
        compiled=use_compiled,
        amp_dtype=cfg.amp_dtype,
        return_conditioning=cfg.return_conditioning,
        compile_mode=compile_mode,
        temperature=temperature,
        desired_return=desired_return,
    )


def _validate_deployment_timing(prediction_frames: int, delay_frames: int, replan_interval_frames: int) -> None:
    if not isinstance(delay_frames, int) or isinstance(delay_frames, bool) or delay_frames < 0:
        raise ValueError(f"delay_frames must be a non-negative integer, got {delay_frames!r}")
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


@dataclass(frozen=True, slots=True)
class _ActionTraceRecorder:
    writer: ActionTraceWriter
    seed: int
    head_offsets: tuple[int, ...]
    temperature: float
    fixed_prefix_frames: int
    replan_interval_frames: int

    def __call__(self, requests: Sequence[PredictionRequest], decoded: DecodedPlan) -> None:
        self.writer.record_plan(
            decode_seed=self.seed,
            slot_ids=torch.tensor([request.stream_id for request in requests]),
            resets=torch.tensor([request.sequence == 0 for request in requests], dtype=torch.bool),
            indices=decoded.indices,
            logits=decoded.logits,
            uniforms=decoded.uniforms,
            head_offsets=self.head_offsets,
            temperature=self.temperature,
            delay_frames=self.fixed_prefix_frames,
            replan_interval_frames=self.replan_interval_frames,
        )


def make_policy(
    model: ActionSequenceTransformer,
    stats: dict[str, FeatureStats],
    cfg: TrainConfig,
    *,
    decode_seed: int | None = None,
    inference: WindowPolicy | None = None,
    telemetry: DecodeTelemetry | None = None,
    ego_player_id: int = MASKED_PLAYER_ID,
    delay_frames: int | None = None,
    replan_interval_frames: int | None = None,
    decode_temperature: float = 1.0,
    action_trace: ActionTraceWriter | None = None,
    max_batch_size: int | None = None,
    prewarm_executor: bool = True,
) -> PolicyBatchAdapter:
    """Build the official dense profile on the shared process/scheduler path."""
    horizon = cfg.prediction_frames
    prefix = cfg.delay_frames if delay_frames is None else delay_frames
    replan = cfg.replan_interval_frames if replan_interval_frames is None else replan_interval_frames
    _validate_deployment_timing(horizon, prefix, replan)
    temperature = validate_sampling_temperature(decode_temperature)
    engine = make_window_policy(model, cfg, temperature=temperature) if inference is None else inference
    if engine.temperature != temperature:
        raise ValueError(f"inference temperature {engine.temperature} does not match policy temperature {temperature}")
    if action_trace is not None and decode_seed is None:
        raise ValueError("action tracing requires a fixed decode_seed")
    if action_trace is not None and action_trace.group_vocabs != model.codec.group_vocabs:
        raise ValueError("action trace vocabulary does not match the model codec")
    seed = torch.seed() if decode_seed is None else decode_seed
    trace_sink = (
        None
        if action_trace is None
        else _ActionTraceRecorder(action_trace, seed, model.head_offsets[:horizon], temperature, prefix, replan)
    )
    policy = DenseWindowPredictionPolicy(
        engine, stats, seed=seed, ego_player_id=ego_player_id, telemetry=telemetry, trace_sink=trace_sink
    )
    capacity = max(engine.prepared_buckets) if max_batch_size is None else max_batch_size
    runtime = RuntimeConfig(capacity, (0,), replan_interval_frames=replan)
    policy.prepare_prediction(runtime, horizon, prefix, prewarm_executor=prewarm_executor)
    return PolicyBatchAdapter(
        policy,
        runtime,
        FrameTiming(0, 0, prefix, replan, horizon),
        desired_return=engine.desired_return,
        temperature=temperature,
        observed_actions=False,
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
    transport_semantics: Literal["conditioned_pending_actions_v1"]
    pending_prefix_conditioned: Literal[True]
    evaluation_protocol_version: Literal[3]
    forced_prefix_consumes_sampling_draws: Literal[False]
    dtype: str
    inference_mode: str
    inference_compile_mode: str
    inference_attention_backend: str
    compiled_inference_bucket: int
    checkpoint_sha256: str
    return_target: Literal["p90", "unconditioned"]
    desired_return: float | None
    return_calibration_sha256: str | None
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES
    start_retries: int = DEFAULT_START_RETRIES


@dataclass(slots=True)
class _EvalPolicyFactory:
    model: ActionSequenceTransformer
    stats: dict[str, FeatureStats]
    cfg: TrainConfig
    inference: WindowPolicy
    telemetry: DecodeTelemetry
    protocol: EvalProtocol
    policy_index: int = 0

    def __call__(self) -> PolicyBatchAdapter:
        seed = self.protocol.seed + self.policy_index
        self.policy_index += 1
        return make_policy(
            self.model,
            self.stats,
            self.cfg,
            decode_seed=seed,
            inference=self.inference,
            telemetry=self.telemetry,
            ego_player_id=self.protocol.ego_player_id,
            delay_frames=self.protocol.delay_frames,
            replan_interval_frames=self.protocol.replan_interval_frames,
            max_batch_size=self.protocol.max_parallel,
            prewarm_executor=False,
        )


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
    model: ActionSequenceTransformer,
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
    desired_return: float | None = None,
    return_calibration_sha256: str | None = None,
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
        transport_semantics="conditioned_pending_actions_v1",
        pending_prefix_conditioned=True,
        evaluation_protocol_version=3,
        forced_prefix_consumes_sampling_draws=False,
        dtype=str(next(model.parameters()).dtype),
        inference_mode=cfg.inference_mode if inference_mode is None else inference_mode,
        inference_compile_mode=inference_compile_mode,
        inference_attention_backend=inference_attention_backend,
        compiled_inference_bucket=_eval_inference_bucket(cfg, n_matchups, max_parallel),
        checkpoint_sha256=checkpoint_sha256,
        return_target="unconditioned" if desired_return is None else "p90",
        desired_return=desired_return,
        return_calibration_sha256=return_calibration_sha256,
    )


def _validate_eval_directory(replay_dir: Path, protocol: EvalProtocol) -> None:
    path = replay_dir / "match_rows.json"
    if not path.exists():
        return
    previous = json.loads(path.read_text())
    if (
        not isinstance(previous, dict)
        or previous.get("schema_version") != 7
        or previous.get("protocol") != asdict(protocol)
    ):
        raise ValueError("evaluation directory contains evidence from a different protocol")


def _write_eval_evidence(
    replay_dir: Path, rows: list[MatchRow], metrics: dict[str, float], protocol: EvalProtocol
) -> None:
    replay_dir.mkdir(parents=True, exist_ok=True)
    rows_payload = {
        "schema_version": 7,
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
    model: ActionSequenceTransformer,
    stats: dict[str, FeatureStats],
    cfg: TrainConfig,
    *,
    n_matchups: int,
    replay_dir: Path,
    checkpoint_sha256: str = "unavailable",
    inference: WindowPolicy | None = None,
    eager: bool = False,
    max_parallel: int | None = None,
    fixed_ego_character: melee.Character | None = None,
    ego_player_id: int = MASKED_PLAYER_ID,
    ego_player_code: str | None = None,
    delay_frames: int | None = None,
    replan_interval_frames: int | None = None,
    desired_return: float | None = None,
    return_calibration_sha256: str | None = None,
) -> dict[str, float]:
    horizon = cfg.prediction_frames
    inference_mode = "eager" if eager else cfg.inference_mode
    inference = (
        make_window_policy(
            model,
            cfg,
            bucket=_eval_inference_bucket(cfg, n_matchups, max_parallel),
            compiled=inference_mode == "compiled",
            desired_return=desired_return,
        )
        if inference is None
        else inference
    )
    if inference.desired_return != desired_return:
        raise ValueError("inference desired return differs from evaluation protocol")
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
        desired_return=desired_return,
        return_calibration_sha256=return_calibration_sha256,
    )
    _validate_eval_directory(replay_dir, protocol)
    if next(model.parameters()).device.type == "cuda" and (
        protocol.inference_mode != "compiled" or not inference.compiled
    ):
        raise RuntimeError("official CUDA evaluation requires compiled BF16 inference")
    telemetry = DecodeTelemetry()
    factory = _EvalPolicyFactory(model, stats, cfg, inference, telemetry, protocol)
    process_telemetry = ProcessVecTelemetry()

    was_training = model.training
    model.eval()
    total_started = time.perf_counter()
    try:
        compile_seconds = inference.prewarm(protocol.max_parallel, horizon, committed_frames=protocol.delay_frames)
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
                process_telemetry=process_telemetry,
            )
        sweep_seconds = time.perf_counter() - started
    finally:
        model.train(was_training)
    metrics = vs_cpu_metrics(results, seed=protocol.seed)
    metrics["eval_wall_seconds"] = sweep_seconds
    metrics["eval_total_wall_seconds"] = time.perf_counter() - total_started
    metrics["inference_compile_seconds"] = compile_seconds
    metrics["captured_emulator_frames"] = float(sum(row.total_frames for row in rows))
    metrics["emulator_fps"] = metrics["captured_emulator_frames"] / max(sweep_seconds, 1e-12)
    metrics["prediction_frames"] = float(cfg.prediction_frames)
    metrics["delay_frames"] = float(protocol.delay_frames)
    metrics["replan_interval_frames"] = float(protocol.replan_interval_frames)
    metrics["ego_player_id"] = float(protocol.ego_player_id)
    if protocol.fixed_ego_character is not None:
        metrics["fixed_ego_character"] = float(protocol.fixed_ego_character)
    decode_metrics = telemetry.metrics()
    decode_metrics["decode_predicted_actions_per_s"] = decode_metrics.pop("decode_executed_frames_per_s")
    metrics.update(decode_metrics)
    metrics.update(process_telemetry.metrics())
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
        "captured_emulator_frames": "captured_emulator_frames",
        "emulator_fps": "emulator_fps",
        "decode_predicted_actions_per_s": "decode_predicted_actions_per_s",
        "broker_inference_latency_p50_ms": "broker_inference_latency_p50_ms",
        "broker_inference_latency_p95_ms": "broker_inference_latency_p95_ms",
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


def _learning_rate_scale(
    step: int,
    *,
    warmup_steps: int,
    decay_start_update: int | None,
    decay_duration: int | None,
    lr_floor_ratio: float,
) -> float:
    update = step + 1
    if update <= warmup_steps:
        return update / warmup_steps
    if decay_start_update is None or update <= decay_start_update:
        return 1.0
    if decay_duration is None:
        raise RuntimeError("decay duration is missing")
    progress = min((update - decay_start_update) / decay_duration, 1.0)
    return 1.0 + progress * (lr_floor_ratio - 1.0)


def lr_schedule(cfg: TrainConfig) -> Callable[[int], float]:
    """Return warmup/stable/decay WSD independent of the stopping update."""
    return partial(
        _learning_rate_scale,
        warmup_steps=cfg.warmup_steps,
        decay_start_update=cfg.decay_start_update,
        decay_duration=cfg.decay_duration,
        lr_floor_ratio=cfg.lr_floor_ratio,
    )


class LearningRateScheduler(LambdaLR):
    def state_dict(self) -> dict[str, object]:
        # Torch 2.11 serializes a partial's attributes but omits a function's.
        # 060 format 1 stores schedule values in cfg and None in this record.
        # Remove this normalization when that checkpoint format is retired.
        state = super().state_dict()
        state["lr_lambdas"] = [None] * len(self.lr_lambdas)
        return state


def scaling_multipliers(cfg: TrainConfig) -> tuple[float, float]:
    return cfg.batch_size / cfg.reference_batch_size, cfg.target_positions / cfg.reference_positions


def scaled_adam_betas(cfg: TrainConfig) -> tuple[float, float]:
    batch_multiplier, duration_multiplier = scaling_multipliers(cfg)
    ratio = batch_multiplier / duration_multiplier
    betas = (1 - (1 - cfg.adam_beta1) * ratio, 1 - (1 - cfg.adam_beta2) * ratio)
    if any(not 0 <= beta < 1 for beta in betas):
        raise ValueError(f"scaled Adam betas are invalid: {betas}")
    return betas


def scaled_adam_epsilon(cfg: TrainConfig) -> float:
    batch_multiplier, duration_multiplier = scaling_multipliers(cfg)
    return cfg.adam_eps * math.sqrt(duration_multiplier / batch_multiplier)


@dataclass(frozen=True, slots=True)
class OptimizerRole:
    optimizer: Literal["muon", "adamw"]
    lr_kind: Literal["hidden", "input", "output", "vector"]
    decay: bool
    logical_splits: int = 1
    fan_in_multiplier: float = 1.0


def _is_final_readout(name: str) -> bool:
    return (
        (name.startswith("temporal.outputs.") and ".down." in name)
        or (name.startswith("temporal.trunk_outputs.") and ".down." in name)
        or name.startswith("value_head.down.")
    )


def _output_fan_in_multiplier(name: str, cfg: TrainConfig) -> float:
    if name.startswith("temporal.outputs."):
        return cfg.arch.group_head_dim / 128
    if name.startswith("temporal.trunk_outputs."):
        return cfg.arch.group_head_dim / 128
    if name.startswith("value_head.down."):
        return cfg.arch.value_hidden_dim / 128
    raise ValueError(f"{name!r} is not a final readout")


def optimizer_roles(model: ActionSequenceTransformer, cfg: TrainConfig) -> dict[str, OptimizerRole]:
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
                "adamw", "output", parameter.ndim >= 2, fan_in_multiplier=_output_fan_in_multiplier(name, cfg)
            )
        elif name.startswith("trunk.blocks."):
            splits = 3 if name.endswith("attn.c_attn.weight") else 1
            roles[name] = OptimizerRole(cfg.optimizer, "hidden", True, logical_splits=splits)
        elif name.startswith("temporal.blocks."):
            splits = 3 if name.endswith("qkv.weight") else 1
            roles[name] = OptimizerRole(cfg.optimizer, "hidden", True, logical_splits=splits)
        elif name.startswith("temporal.history_attention."):
            splits = 2 if name.endswith("key_value.weight") else 1
            roles[name] = OptimizerRole(cfg.optimizer, "hidden", True, logical_splits=splits)
        elif name == "temporal.token_projection.weight" or (
            name.startswith(("temporal.outputs.", "temporal.trunk_outputs.")) and name.endswith("up.weight")
        ):
            roles[name] = OptimizerRole(cfg.optimizer, "hidden", True)
        elif name == "value_head.up.weight":
            roles[name] = OptimizerRole(cfg.optimizer, "hidden", True, logical_splits=2)
        elif name.startswith(embedding_prefixes):
            roles[name] = OptimizerRole("adamw", "input", False)
        elif name.startswith((*finite_prefixes, "temporal.return_conditioner.")):
            roles[name] = OptimizerRole("adamw", "input" if parameter.ndim >= 2 else "vector", parameter.ndim >= 2)
        elif name == "temporal.token_projection.bias":
            roles[name] = OptimizerRole("adamw", "vector", False)
        else:
            raise RuntimeError(f"O60 has no optimizer role for {name} {tuple(parameter.shape)}")
    return roles


def _role_lr(role: OptimizerRole, cfg: TrainConfig) -> float:
    if role.optimizer == "muon":
        return cfg.muon_lr * cfg.muon_lr_multiplier
    batch_multiplier, duration_multiplier = scaling_multipliers(cfg)
    lr = cfg.adam_lr * math.sqrt(batch_multiplier / duration_multiplier)
    return lr / role.fan_in_multiplier if role.lr_kind == "output" else lr


def make_optimizer(model: ActionSequenceTransformer, cfg: TrainConfig) -> torch.optim.Optimizer:
    """Build the selected optimizer with O51's semantic learning-rate roles."""
    roles = optimizer_roles(model, cfg)
    named = dict(model.named_parameters())
    buckets: dict[tuple[object, ...], list[nn.Parameter]] = defaultdict(list)
    for name, role in roles.items():
        lr = _role_lr(role, cfg)
        key = (
            ("muon", lr, cfg.muon_weight_decay if role.decay else 0.0, role.logical_splits)
            if role.optimizer == "muon"
            else ("adamw", lr, cfg.adam_weight_decay if role.decay else 0.0)
        )
        buckets[key].append(named[name])
    groups: list[dict[str, object]] = []
    for key, parameters in buckets.items():
        if key[0] == "muon":
            groups.append(
                {
                    "params": parameters,
                    "lr": key[1],
                    "momentum": 0.95,
                    "weight_decay": key[2],
                    "use_muon": True,
                    "muon_scale_clamp_min_one": False,
                    "logical_splits": key[3],
                }
            )
        else:
            groups.append(
                {
                    "params": parameters,
                    "lr": key[1],
                    "betas": scaled_adam_betas(cfg),
                    "eps": scaled_adam_epsilon(cfg),
                    "weight_decay": key[2],
                    "update_clip_threshold": None,
                    "use_muon": False,
                }
            )
    if cfg.optimizer == "muon":
        return SingleDeviceMuonWithAuxAdam(groups)
    return torch.optim.AdamW(
        groups,
        betas=scaled_adam_betas(cfg),
        eps=scaled_adam_epsilon(cfg),
        fused=DEVICE == "cuda",
    )


def _button_path_parameters(model: ActionSequenceTransformer) -> dict[str, nn.Parameter]:
    """Return the action-path matrices monitored before clipping."""
    button_head = cast(NonlinearActionHead, model.temporal.outputs["buttons"])
    return {
        "buttons_output_weight": cast(nn.Parameter, button_head.down.weight),
        "buttons_condition_weight": cast(nn.Parameter, model.temporal.group_condition["buttons"].weight),
        "token_projection_weight": cast(nn.Parameter, model.temporal.token_projection.weight),
    }


def _button_gradient_abs_max(model: ActionSequenceTransformer) -> Tensor:
    """Return one pre-clipping action-path gradient guardrail."""
    maxima: list[Tensor] = []
    for name, parameter in _button_path_parameters(model).items():
        if parameter.grad is None:
            raise RuntimeError(f"button parameter {name!r} has no gradient")
        maxima.append(parameter.grad.detach().float().abs().amax())
    return torch.stack(maxima).amax()


def parameter_subsystems(model: ActionSequenceTransformer) -> dict[str, tuple[nn.Parameter, ...]]:
    """Partition every parameter by the model subsystem that owns it."""
    all_parameters = tuple(model.parameters())
    trunk_ids = {id(parameter) for parameter in model.trunk.parameters()}
    head_ids = {id(parameter) for parameter in model.temporal.outputs.parameters()}
    trunk_skip_ids = {id(parameter) for parameter in model.temporal.trunk_outputs.parameters()}
    conditioner_ids = {id(parameter) for parameter in model.temporal.return_conditioner.parameters()}
    temporal_ids = {
        id(parameter)
        for parameter in model.temporal.parameters()
        if id(parameter) not in head_ids | trunk_skip_ids | conditioner_ids
    }
    value_ids = {id(parameter) for parameter in model.value_head.parameters()}
    other_ids = (
        {id(parameter) for parameter in all_parameters}
        - trunk_ids
        - temporal_ids
        - head_ids
        - trunk_skip_ids
        - value_ids
        - conditioner_ids
    )
    partition_ids = {
        "trunk": trunk_ids,
        "temporal_decoder": temporal_ids,
        "group_heads": head_ids,
        "trunk_skip_heads": trunk_skip_ids,
        "value_head": value_ids,
        "return_conditioner": conditioner_ids,
        "other": other_ids,
    }
    partitions = {
        name: tuple(parameter for parameter in all_parameters if id(parameter) in parameter_ids)
        for name, parameter_ids in partition_ids.items()
    }
    if sum(len(parameters) for parameters in partitions.values()) != len(all_parameters):
        raise RuntimeError("parameter subsystem partition is incomplete")
    return partitions


def subsystem_parameter_counts(model: ActionSequenceTransformer) -> dict[str, int]:
    partitions = parameter_subsystems(model)
    counts = {name: sum(parameter.numel() for parameter in parameters) for name, parameters in partitions.items()}
    counts["total"] = sum(parameter.numel() for parameter in model.parameters())
    if sum(value for name, value in counts.items() if name != "total") != counts["total"]:
        raise RuntimeError("parameter subsystem partition is incomplete")
    try:
        expected = Architecture(
            **{item.name: getattr(model.cfg, item.name) for item in fields(Architecture)}
        ).parameter_count_contract
    except ValueError as error:
        raise RuntimeError(str(error)) from error
    if counts != expected:
        raise RuntimeError(f"parameter contract changed: {counts} != {expected}")
    return counts


def compute_equivalent_parameter_count(cfg: TrainConfig, parameter_counts: Mapping[str, int]) -> int:
    """Normalize subsystem parameter uses to one sampled policy position."""
    policy_positions = cfg.policy_prefixes_per_update
    parameter_uses = (
        cfg.batch_size * cfg.arch.L_ctx * (parameter_counts["trunk"] + parameter_counts["other"])
        + cfg.value_prefixes_per_update * parameter_counts["value_head"]
        + policy_positions * (parameter_counts["trunk_skip_heads"] + parameter_counts["return_conditioner"])
        + policy_positions
        * len(cfg.arch.head_offsets)
        * (parameter_counts["temporal_decoder"] + parameter_counts["group_heads"])
    )
    count, remainder = divmod(parameter_uses, policy_positions)
    if remainder:
        raise ValueError("subsystem parameter uses do not divide by policy positions")
    return count


def approximate_training_flops_per_update(cfg: TrainConfig, parameter_counts: dict[str, int]) -> int:
    """Return six times every parameter use in one optimizer update."""
    if cfg.arch == Architecture() and parameter_counts["total"] != 193_360_029:
        raise ValueError("production parameter count changed")
    return (
        compute_flops_per_supervised_position(compute_equivalent_parameter_count(cfg, parameter_counts))
        * cfg.supervised_positions_per_update
    )


def model_tag(cfg: TrainConfig) -> str:
    return (
        f"o60v2-d{cfg.arch.d_model}-L{cfg.arch.n_layers}-h{cfg.arch.n_heads}-c{cfg.arch.L_ctx}-"
        f"t{cfg.arch.temporal_d_model}x{cfg.arch.temporal_layers}-ff{cfg.arch.temporal_ff_dim}-"
        f"dual-nonlinear-head-rc{int(cfg.return_conditioning)}e{cfg.arch.return_embed_dim}-drop{cfg.return_dropout:g}-"
        f"{cfg.optimizer}-wsd-b{cfg.awr.beta:g}-wmax{cfg.awr.weight_max:g}-g{cfg.awr.gamma:g}"
    )


def _include_training_source(path: str, code_root: str) -> bool:
    try:
        relative = Path(path).resolve().relative_to(Path(code_root).resolve())
    except ValueError:
        return False
    return (
        bool(relative.parts)
        and relative.parts[0] in {"docker", "experiments", "hal", "notebooks", "scripts", "tests"}
        and relative.suffix in {".py", ".sh", ".toml", ".yaml", ".yml"}
    )


def log_wandb_code(run: wandb.Run) -> None:
    run.log_code(root=str(Path(__file__).resolve().parents[1]), include_fn=_include_training_source)


def data_selection(
    cfg: TrainConfig,
    *,
    rank: int | None = None,
    world_size: int | None = None,
) -> PhysicalShardSelection:
    """Return the full corpus or one deterministic per-source rank partition."""
    if tuple(cfg.source_names) != tuple(source.name for source in streams.POLICY_WORLD_V8_SOURCES):
        raise ValueError("O60 selection requires all policy-world-v8 sources in registry order")
    if (rank is None) != (world_size is None):
        raise ValueError("rank and world_size must be provided together")
    if rank is None:
        sources = tuple(
            SourceRowSelection(name, streams.POLICY_WORLD_V8_TRAIN_REPLAYS[name]) for name in cfg.source_names
        )
    else:
        assert world_size is not None
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("rank partition is outside the world")
        sources = tuple(
            SourceRowSelection(
                name,
                streams.POLICY_WORLD_V8_TRAIN_REPLAYS[name] * (rank + 1) // world_size,
                start=streams.POLICY_WORLD_V8_TRAIN_REPLAYS[name] * rank // world_size,
            )
            for name in cfg.source_names
        )
    selection = PhysicalShardSelection.from_sources(sources)
    if rank is None and selection.sha256 != cfg.selection_sha256:
        raise RuntimeError(f"policy-world-v8 selection hash changed: {selection.sha256} != {cfg.selection_sha256}")
    if rank is None and selection.row_count != cfg.train_replays:
        raise RuntimeError(f"policy-world-v8 selection has {selection.row_count} rows, expected {cfg.train_replays}")
    return selection


def data_partition_hashes(cfg: TrainConfig) -> tuple[str, ...]:
    """Return the ordered identities of all rank-local source partitions."""
    return tuple(data_selection(cfg, rank=rank, world_size=cfg.world_size).sha256 for rank in range(cfg.world_size))


def rank_seed(seed: int, rank: int) -> int:
    """Derive a stable, distinct random stream for one rank."""
    if seed < 0 or rank < 0:
        raise ValueError("seed and rank must be non-negative")
    return seed + rank * 1_000_003


def source_mixture_weights(cfg: TrainConfig) -> tuple[float, ...]:
    """Return the natural MDS replay-count mixture."""
    return tuple(float(streams.POLICY_WORLD_V8_TRAIN_REPLAYS[name]) for name in cfg.source_names)


def validate_batch_geometry(
    batch: TrainBatch | ReturnBatch, cfg: TrainConfig, expected_batch_size: int | None = None
) -> None:
    if isinstance(batch, ReturnBatch):
        expected = (batch.target.shape[0], cfg.arch.L_ctx)
        for name in RETURN_BATCH_FIELDS:
            value = getattr(batch, name)
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}")
        for name in ("eligible", "available", "condition_present"):
            if getattr(batch, name).dtype != torch.bool:
                raise ValueError(f"{name} must be boolean")
        if bool((batch.condition_present & ~batch.available).any()):
            raise ValueError("conditioning cannot be present for unavailable labels")
        if not bool(torch.isfinite(batch.future_return[batch.available]).all()):
            raise ValueError("available return labels must be finite")
    if batch.target.shape[1:] != (cfg.arch.sample_chunk_length, ACTION_DIM):
        raise ValueError(
            f"target must be [B, {cfg.arch.sample_chunk_length}, {ACTION_DIM}], got {tuple(batch.target.shape)}"
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


def cache_validation(loader: Iterable[ReturnBatch], n_samples: int) -> list[ReturnBatch]:
    print(f"[validation] caching {n_samples:,} samples", flush=True)
    started = time.monotonic()
    last_progress_log = started
    batches: list[ReturnBatch] = []
    count = 0
    for batch in loader:
        remaining = n_samples - count
        if remaining <= 0:
            break
        if batch.target.shape[0] > remaining:
            batch = batch.slice(remaining)
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


def distributed_training_contract(cfg: TrainConfig) -> dict[str, object]:
    """Return the persisted geometry that must match for an exact resume."""
    return {
        "version": _DISTRIBUTED_CHECKPOINT_VERSION,
        "world_size": cfg.world_size,
        "global_batch_size": cfg.batch_size,
        "local_batch_size": cfg.local_batch_size,
        "microbatch_size": cfg.microbatch_size,
        "partition_sha256": data_partition_hashes(cfg),
        "architecture": asdict(cfg.arch),
        "head_offsets": cfg.arch.head_offsets,
        "offset_loss_weights": OFFSET_LOSS_WEIGHTS,
    }


def _combine_calibration_states(
    rank_states: Sequence[Mapping[str, object]],
    *,
    world_size: int,
) -> dict[str, object]:
    if len(rank_states) != world_size or CALIBRATION_WINDOWS % world_size:
        raise ValueError("rank calibration geometry is incompatible")
    local_count = CALIBRATION_WINDOWS // world_size
    combined = ReturnCalibration(window_count=CALIBRATION_WINDOWS)
    for expected_rank, rank_state in enumerate(rank_states):
        if rank_state.get("rank") != expected_rank:
            raise ValueError("checkpoint rank states are not ordered")
        state = rank_state.get("return_calibration")
        if not isinstance(state, dict):
            raise ValueError("rank checkpoint lacks return calibration")
        calibration = ReturnCalibration(window_count=local_count)
        calibration.load_state_dict(cast(dict[str, object], state))
        combined.values.extend(calibration.values)
        combined.valid.extend(calibration.valid)
        combined.replay_ids.extend(calibration.replay_ids)
    return combined.state_dict()


def gather_rank_checkpoint_states(
    local_state: dict[str, object],
    distributed: DistributedContext,
) -> tuple[dict[str, object], ...] | None:
    """Gather ordered rank state on rank zero without replicating it elsewhere."""
    gathered: list[object] | None = [None] * distributed.world_size if distributed.is_primary else None
    dist.gather_object(local_state, gathered, dst=0)
    if gathered is None:
        return None
    if not all(isinstance(state, dict) for state in gathered):
        raise RuntimeError("distributed checkpoint gather returned invalid rank state")
    states = cast(list[dict[str, object]], gathered)
    if any(state.get("rank") != rank for rank, state in enumerate(states)):
        raise RuntimeError("distributed checkpoint gather returned out-of-order rank state")
    return tuple(states)


def rank_resume_state(
    state: Mapping[str, object],
    cfg: TrainConfig,
    distributed: DistributedContext,
) -> dict[str, object]:
    """Validate the distributed checkpoint contract and return this rank's state."""
    record = state.get("distributed")
    if not isinstance(record, dict):
        raise ValueError("checkpoint has no distributed state")
    record = cast(dict[str, object], record)
    contract = record.get("contract")
    if contract != distributed_training_contract(cfg):
        raise ValueError("checkpoint world, batch, partition, architecture, offsets, or weights changed")
    ranks = record.get("ranks")
    if not isinstance(ranks, list) or len(ranks) != distributed.world_size:
        raise ValueError("checkpoint does not contain every rank state")
    selected = ranks[distributed.rank]
    if not isinstance(selected, dict):
        raise ValueError("checkpoint rank states are not mappings")
    selected = cast(dict[str, object], selected)
    if selected.get("rank") != distributed.rank:
        raise ValueError("checkpoint rank states are not ordered")
    expected_fields = {
        "rank",
        "seed",
        "partition_sha256",
        "loader",
        "identity_masker",
        "return_masker",
        "prefix_sampler",
        "return_calibration",
        "rng",
    }
    if set(selected) != expected_fields:
        raise ValueError("checkpoint rank state fields changed")
    if selected["seed"] != rank_seed(cfg.seed, distributed.rank):
        raise ValueError("checkpoint rank seed changed")
    if selected["partition_sha256"] != data_partition_hashes(cfg)[distributed.rank]:
        raise ValueError("checkpoint rank partition changed")
    return selected


def save_boundary_checkpoint(
    run_dir: Path,
    *,
    update: int,
    model: ActionSequenceTransformer,
    calibration: ReturnCalibration,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    cfg: TrainConfig,
    uploader: BackgroundUploader | None,
    milestone: bool,
    wandb_id: str | None,
    actual_loss_positions: int,
    loader_state: dict[str, object],
    return_masker_state: dict[str, object],
    distributed: DistributedContext,
    identity_masker_state: dict[str, object] | None = None,
    prefix_sampler_state: dict[str, object] | None = None,
    resume_lineage: tuple[ResumeLineage, ...] = (),
) -> Path | None:
    """Save one immutable boundary snapshot, then atomically advance latest."""
    if identity_masker_state is None or prefix_sampler_state is None:
        raise ValueError("distributed checkpoints require explicit masker and sampler state")
    local_state: dict[str, object] = {
        "rank": distributed.rank,
        "seed": rank_seed(cfg.seed, distributed.rank),
        "partition_sha256": data_partition_hashes(cfg)[distributed.rank],
        "loader": loader_state,
        "identity_masker": identity_masker_state,
        "return_masker": return_masker_state,
        "prefix_sampler": prefix_sampler_state,
        "return_calibration": calibration.state_dict(),
        "rng": rng_state(),
    }
    rank_states = gather_rank_checkpoint_states(local_state, distributed)
    if not distributed.is_primary:
        return None
    assert rank_states is not None
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
            "actual_policy_prefixes": update * cfg.policy_prefixes_per_update,
            "actual_value_prefixes": update * cfg.value_prefixes_per_update,
            "schedule_phase": (
                "warmup"
                if update <= cfg.warmup_steps
                else "stable"
                if cfg.decay_start_update is None or update <= cfg.decay_start_update
                else "decay"
            ),
            "conditioning_protocol": conditioning_protocol(),
            "return_masker": rank_states[0]["return_masker"],
            "return_calibration": _combine_calibration_states(rank_states, world_size=cfg.world_size),
            "provenance": run_provenance(cfg),
            "distributed": {
                "contract": distributed_training_contract(cfg),
                "ranks": list(rank_states),
            },
            "resume_lineage": [entry.to_record() for entry in resume_lineage],
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


def rng_state() -> dict[str, object]:
    """Capture every process RNG used by the training path."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def restore_rng(state: Mapping[str, object]) -> None:
    random.setstate(cast(tuple, state["python"]))
    np.random.set_state(cast(tuple, state["numpy"]))
    torch.set_rng_state(cast(Tensor, state["torch"]).cpu())
    cuda = state["cuda"]
    if cuda is not None:
        if not torch.cuda.is_available() or not isinstance(cuda, Tensor):
            raise ValueError("resume CUDA RNG state is incompatible with this rank")
        torch.cuda.set_rng_state(cuda.cpu())


@functools.cache
def run_provenance(cfg: TrainConfig) -> dict[str, object]:
    """Resolve immutable code, data, artifact, optimizer, and environment identities."""
    root = Path(__file__).resolve().parents[1]
    git_sha = os.environ.get("HAL_GIT_SHA")
    if git_sha is None:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if re.fullmatch(r"[0-9a-f]{40}", git_sha) is None:
        raise RuntimeError(f"HAL_GIT_SHA must be a full lowercase Git SHA, got {git_sha!r}")
    statistics = {
        name: checkpoint_sha256(streams.BY_NAME[name].local_root / "stats.json") for name in cfg.source_names
    }
    compute_equivalent_parameters = compute_equivalent_parameter_count(cfg, cfg.arch.parameter_count_contract)
    return {
        "schema_version": 1,
        "git_sha": git_sha,
        "experiment_id": _EXPERIMENT_ID,
        "source_selection_sha256": data_selection(cfg).sha256,
        "source_partition_sha256": data_partition_hashes(cfg),
        "source_manifest_sha256": {
            name: streams.POLICY_WORLD_V8_TRAIN_MANIFEST_SHA256[name] for name in cfg.source_names
        },
        "source_statistics_sha256": statistics,
        "player_sidecar_sha256": cfg.player_sidecar_sha256,
        "player_vocab_sha256": cfg.player_vocab_sha256,
        "optimizer": {
            "muon_lr": cfg.muon_lr,
            "muon_momentum": 0.95,
            "muon_weight_decay": cfg.muon_weight_decay,
            "adam_lr": cfg.adam_lr,
            "adam_betas": scaled_adam_betas(cfg),
            "adam_epsilon": scaled_adam_epsilon(cfg),
            "adam_weight_decay": cfg.adam_weight_decay,
            "gradient_clip": cfg.grad_clip,
        },
        "compute": {
            "compute_equivalent_parameters": compute_equivalent_parameters,
            "flops_per_supervised_position": compute_flops_per_supervised_position(compute_equivalent_parameters),
            "fitted_optimal_positions": fitted_optimal_positions(compute_equivalent_parameters),
            "target_positions": cfg.target_positions,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "device": DEVICE,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        },
    }


def load_identity_sidecar(cfg: TrainConfig) -> PlayerIdentitySidecar:
    return load_player_identity_artifact(
        Path(cfg.player_sidecar_local),
        remote=cfg.player_sidecar_remote,
        expected_sha256=cfg.player_sidecar_sha256,
        expected_vocabulary_size=cfg.player_vocab_size,
        expected_vocabulary_sha256=cfg.player_vocab_sha256,
    )


def _collate_o60_batch(
    replay_ids: tuple[str, ...],
    columns: Mapping[str, np.ndarray],
    *,
    stats: dict[str, FeatureStats],
    projection: FeatureProjection,
    context_length: int,
    return_column: str,
    return_valid_column: str,
) -> ReturnBatch:
    batch = train_batch_from_columns(
        columns,
        stats=stats,
        L_ctx=context_length,
        extra=ITEM_PLAYER_COLUMNS,
        projection=projection,
    )
    batch = _canonical_training_batch(TrainBatch(batch.context, batch.target, replay_ids))
    next_frames = slice(1, context_length + 1)
    returns = columns[return_column][:, next_frames]
    eligible = columns[return_valid_column][:, next_frames]
    return ReturnBatch(
        batch,
        torch.from_numpy(np.ascontiguousarray(returns)),
        torch.from_numpy(np.ascontiguousarray(eligible)).bool(),
        torch.from_numpy(np.ascontiguousarray(columns["ego_return60"][:, :context_length])),
        torch.from_numpy(np.ascontiguousarray(columns["ego_return60_valid"][:, :context_length])).bool(),
        torch.from_numpy(np.ascontiguousarray(columns["ego_return60_valid"][:, :context_length])).bool(),
    )


def _require_loader_disk(loader: BufferedMDSReplayLoader[ReturnBatch]) -> None:
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
    distributed: DistributedContext,
) -> BufferedMDSReplayLoader[ReturnBatch]:
    selection = data_selection(cfg, rank=distributed.rank, world_size=distributed.world_size)
    adapter = MDSStorageAdapter(selection, download_retry=cfg.download_retry)
    adapter.validate_manifests(
        expected_sha256=streams.POLICY_WORLD_V8_TRAIN_MANIFEST_SHA256,
        expected_index_version=cfg.mds_index_version,
        expected_schema_sha256=cfg.mds_manifest_schema_sha256,
        expected_rows=streams.POLICY_WORLD_V8_TRAIN_REPLAYS,
    )
    projection = FeatureProjection(
        columns=ITEM_PLAYER_PROJECTION.columns
        | {cfg.awr.ego_return_column, cfg.awr.ego_return_valid_column, "ego_return60", "ego_return60_valid"},
    )
    train_loader = BufferedMDSReplayLoader[ReturnBatch](
        selection=selection,
        adapter=adapter,
        tasks=build_shard_plan(selection, adapter.manifests),
        data_protocol=cfg.data_protocol,
        source_manifest_sha256=streams.POLICY_WORLD_V8_TRAIN_MANIFEST_SHA256,
        batch_transform=functools.partial(
            _collate_o60_batch,
            stats=stats,
            projection=projection,
            context_length=cfg.arch.L_ctx,
            return_column=cfg.awr.ego_return_column,
            return_valid_column=cfg.awr.ego_return_valid_column,
        ),
        batch_size=cfg.local_batch_size,
        replay_slots=cfg.replay_slots,
        seed=rank_seed(cfg.seed, distributed.rank),
        num_workers=cfg.num_workers,
        labels=ReturnLabels(
            returns_lib.PolicyReturnLabels(
                player_lookup=player_lookup,
                gamma=cfg.awr.gamma,
                damage_shaping=cfg.awr.damage_shaping,
                win_reward=cfg.awr.win_reward,
                stock_value=cfg.awr.stock_value,
                suffix=cfg.awr.return_suffix,
            )
        ),
        projection=projection,
        context_length=cfg.arch.L_ctx,
        chunk_length=cfg.arch.sample_chunk_length,
        windows_per_generation=cfg.windows_per_generation,
        replay_phase_block_batches=cfg.replay_phase_block_batches,
        schema_version=cfg.mds_schema_version,
        reserved_disk_bytes=cfg.reserved_disk_bytes,
        pin_memory=torch.cuda.is_available(),
        materialization_threads=cfg.materialization_threads(),
    )
    try:
        _require_loader_disk(train_loader)
        if sum(train_loader.source_sample_counts.values()) != selection.row_count:
            raise ValueError("physical-shard loader does not expose every rank-selected training row")
        if train_loader.minimum_replay_gap_batches < cfg.minimum_replay_gap_batches:
            raise ValueError("replay ring is too small for the 232-batch reuse-gap contract")
    except Exception:
        train_loader.close()
        raise
    return train_loader


def _make_loaders(
    cfg: TrainConfig,
    stats: dict[str, FeatureStats],
    distributed: DistributedContext,
    player_lookup: ReplayPlayerLookup | None = None,
) -> tuple[BufferedMDSReplayLoader[ReturnBatch], list[ReturnBatch]]:
    """Build rank-local training and rank zero's fixed 059 validation cohort."""
    if player_lookup is None:
        player_lookup = ReplayPlayerLookup(load_identity_sidecar(cfg).by_replay)
    train_loader = _make_train_loader(cfg, stats, player_lookup, distributed)
    if not distributed.is_primary:
        return train_loader, []
    try:
        val_loader = make_validation_replay_loader(
            sources=tuple(streams.BY_NAME[name] for name in cfg.source_names),
            stats=stats,
            context_length=cfg.arch.L_ctx,
            chunk_length=cfg.arch.sample_chunk_length,
            batch_size=cfg.val_batch_size,
            seed=cfg.seed,
            cache_limit="1792gb",
            schema_version=cfg.mds_schema_version,
            extra=ITEM_PLAYER_COLUMNS,
            projection=FeatureProjection(
                columns=ITEM_PLAYER_PROJECTION.columns
                | {cfg.awr.ego_return_column, cfg.awr.ego_return_valid_column, "ego_return60", "ego_return60_valid"},
            ),
            batch_transform=functools.partial(collate_awr_batch, L_ctx=cfg.arch.L_ctx),
            replay_labels=ReturnLabels(
                returns_lib.PolicyReturnLabels(
                    player_lookup,
                    cfg.awr.gamma,
                    cfg.awr.damage_shaping,
                    cfg.awr.win_reward,
                    cfg.awr.stock_value,
                    cfg.awr.return_suffix,
                )
            ),
        )
        validation = cache_validation(val_loader, cfg.val_n_samples)
    except Exception:
        train_loader.close()
        raise
    return train_loader, validation


@dataclass(slots=True)
class PreparedTrainingData:
    loader: BufferedMDSReplayLoader[ReturnBatch]
    validation: list[ReturnBatch]
    iterator: Iterator[ReturnBatch]
    first_batch_future: Future[ReturnBatch]
    resources: ExitStack
    worker_start_seconds: float
    first_batch_seconds: list[float]


def _load_timed_first_batch(iterator: Iterator[ReturnBatch], elapsed_seconds: list[float]) -> ReturnBatch:
    started = time.monotonic()
    batch = next(iterator)
    elapsed_seconds.append(time.monotonic() - started)
    return batch


def _prepare_training_data(
    cfg: TrainConfig,
    stats: dict[str, FeatureStats],
    sidecar: PlayerIdentitySidecar,
    distributed: DistributedContext,
    resume_state: dict[str, object] | None,
) -> PreparedTrainingData:
    """Start shard workers and the first batch before CUDA allocation."""
    train_loader, validation = _make_loaders(cfg, stats, distributed, ReplayPlayerLookup(sidecar.by_replay))
    try:
        if resume_state is not None:
            loader_state = resume_state.get("loader")
            if not isinstance(loader_state, dict):
                raise ValueError("resume checkpoint does not contain O60 replay-loader state")
            train_loader.load_state_dict(cast(dict[str, object], loader_state))
        worker_started = time.monotonic()
        train_iterator = iter(train_loader)
        worker_start_seconds = time.monotonic() - worker_started
    except Exception:
        train_loader.close()
        raise
    first_batch_seconds: list[float] = []

    try:
        with ExitStack() as setup:
            executor = setup.enter_context(
                ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"o60-rank-{distributed.rank}-first-batch")
            )
            first_batch = executor.submit(_load_timed_first_batch, train_iterator, first_batch_seconds)
            resources = setup.pop_all()
    except Exception:
        train_loader.close()
        raise
    return PreparedTrainingData(
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
    wandb.init(
        project="hal",
        name=run_name,
        id=None if resume_state is None else resume_state.get("wandb_id"),
        resume="allow" if resume_state is not None else None,
        tags=[
            "gpt",
            "temporal-mtp",
            "advantage-weighted-bc",
            "return-unconditioned",
            "scaled",
            "060",
            "compute-optimal",
            "ddp",
            "projectiles",
            "natural-replay-count-mix",
            "wsd",
        ],
        config={
            **asdict(cfg),
            "experiment_id": _EXPERIMENT_ID,
            "checkpoint_format_version": _CHECKPOINT_FORMAT_VERSION,
            "conditioning_protocol": conditioning_protocol(),
            "max_steps": cfg.max_steps,
            "warmup_steps": cfg.warmup_steps,
            "data_protocol": cfg.data_protocol,
            "source_selection_sha256": selection.sha256,
            "source_partition_sha256": data_partition_hashes(cfg),
            "source_manifest_sha256": streams.POLICY_WORLD_V8_TRAIN_MANIFEST_SHA256,
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
        "train/loss is weighted policy loss in bits; train/nll is its unweighted counterpart; "
        "train/objective is the optimized policy-plus-value objective"
    )
    wandb.run.summary["architecture/treatment"] = (
        f"O60 d{cfg.arch.d_model} trunk L{cfg.arch.n_layers}, temporal L{cfg.arch.temporal_layers}, "
        "offsets 1..30 with equal weights, and 32 sampled policy positions per window"
    )
    wandb.run.summary["optimizer/name"] = "Muon+AdamW" if cfg.optimizer == "muon" else "AdamW"
    wandb.run.summary["optimizer/all_parameters_use_adamw"] = cfg.optimizer == "adamw"
    wandb.run.summary["optimizer/implementation"] = (
        "per-rank SingleDeviceMuonWithAuxAdam after DDP gradient averaging"
        if cfg.optimizer == "muon"
        else "per-rank torch.optim.AdamW after DDP gradient averaging"
    )
    wandb.run.summary["optimizer/fused"] = cfg.optimizer == "adamw" and DEVICE == "cuda"
    wandb.run.summary["optimizer/adam_update_clip_threshold"] = None
    wandb.run.summary["optimizer/lr_schedule"] = "warmup-stable-decay"
    wandb.run.summary["optimizer/update_clip_semantics"] = "global pre-step gradient norm clipping only"
    wandb.run.summary["diagnostics/activation_sample_limit"] = Architecture.activation_percentile_sample_size
    wandb.run.summary["diagnostics/target_logit_gradient_semantics"] = (
        "exact projection-local L1 from NLL and objective coefficient; not a shared-trunk decomposition"
    )
    if cfg.wandb_log_code:
        log_wandb_code(wandb.run)


def _log_training_summary(
    cfg: TrainConfig,
    parameter_counts: dict[str, int],
    train_loader: BufferedMDSReplayLoader[ReturnBatch],
    *,
    flops_per_update: int,
    device_name: str | None,
    peak_flops: float | None,
) -> None:
    """Record the fixed model and corpus accounting for this run."""
    if wandb.run is None:
        return
    compute_equivalent_parameters = compute_equivalent_parameter_count(cfg, parameter_counts)
    flops_per_supervised_position = compute_flops_per_supervised_position(compute_equivalent_parameters)
    for name, value in parameter_counts.items():
        wandb.run.summary[f"parameters/{name}"] = value

    unique_replays = cfg.train_replays
    unique_frames = cfg.train_frames
    source_weights = source_mixture_weights(cfg)
    source_weight_total = sum(source_weights)
    wandb.run.summary["data/unique_replays"] = unique_replays
    wandb.run.summary["data/unique_frames"] = unique_frames
    wandb.run.summary["data/source_list_sha256"] = cfg.source_list_sha256
    wandb.run.summary["data/source_selection_sha256"] = cfg.selection_sha256
    wandb.run.summary["data/mds_manifest_schema_sha256"] = cfg.mds_manifest_schema_sha256
    wandb.run.summary["data/loader_protocol"] = cfg.data_protocol
    policy_positions = cfg.max_steps * cfg.policy_prefixes_per_update
    value_positions = cfg.max_steps * cfg.value_prefixes_per_update
    wandb.run.summary["data/processed_loss_positions"] = policy_positions
    wandb.run.summary["data/D_semantics"] = "32 policy-loss positions sampled per replay window"
    wandb.run.summary["data/policy_supervised_prefixes"] = policy_positions
    wandb.run.summary["data/value_supervised_prefixes"] = value_positions
    wandb.run.summary["data/effective_epochs"] = policy_positions / unique_frames
    wandb.run.summary["data/D_over_N_eff"] = policy_positions / compute_equivalent_parameters
    wandb.run.summary["data/nominal_loss_positions_per_update"] = cfg.supervised_positions_per_update
    wandb.run.summary["data/cpu_lookahead_batches"] = cfg.train_prefetch_factor
    wandb.run.summary["data/loader_prefetch_factor"] = PREFETCH_FACTOR
    wandb.run.summary["data/raw_shard_materialization_threads"] = train_loader.materialization_threads
    wandb.run.summary["data/replay_slots"] = train_loader.replay_slots
    wandb.run.summary["data/aggregate_replay_slots"] = train_loader.replay_slots * cfg.world_size
    wandb.run.summary["data/source_partition_sha256"] = data_partition_hashes(cfg)
    wandb.run.summary["data/generation_windows"] = cfg.windows_per_generation
    wandb.run.summary["data/minimum_replay_frames"] = cfg.minimum_replay_frames
    wandb.run.summary["data/epoch_semantics"] = "replay generations committed to ring / unique train replays"
    wandb.run.summary["data/replay_phase_block_batches"] = cfg.replay_phase_block_batches
    wandb.run.summary["data/minimum_replay_gap_batches"] = train_loader.minimum_replay_gap_batches
    wandb.run.summary["system/disk/required_bytes"] = train_loader.required_disk_bytes
    wandb.run.summary["system/disk/free_bytes_at_start"] = train_loader.disk_free_bytes
    wandb.run.summary["system/disk/reserved_bytes"] = cfg.reserved_disk_bytes
    wandb.run.summary["training/compute_equivalent_parameters"] = compute_equivalent_parameters
    wandb.run.summary["training/flops_per_supervised_position"] = flops_per_supervised_position
    wandb.run.summary["training/flops_per_update"] = flops_per_update
    wandb.run.summary["training/total_flops"] = flops_per_update * cfg.max_steps
    wandb.run.summary["training/flops_formula"] = "6 * subsystem parameter uses per update"
    input_lr = cfg.adam_lr * math.sqrt(scaling_multipliers(cfg)[0] / scaling_multipliers(cfg)[1])
    if cfg.optimizer == "muon":
        wandb.run.summary["optimizer/muon_master_lr"] = cfg.muon_lr
        wandb.run.summary["optimizer/muon_lr_multiplier"] = cfg.muon_lr_multiplier
        wandb.run.summary["optimizer/muon_effective_lr"] = cfg.muon_lr * cfg.muon_lr_multiplier
        wandb.run.summary["optimizer/muon_momentum"] = 0.95
        wandb.run.summary["optimizer/muon_weight_decay"] = cfg.muon_weight_decay
        wandb.run.summary["optimizer/muon_scale_clamp_min_one"] = False
        wandb.run.summary["optimizer/muon_logical_splits"] = "QKV=3, SwiGLU=2"
    wandb.run.summary["optimizer/adam_master_lr"] = cfg.adam_lr
    wandb.run.summary["optimizer/adam_input_lr"] = input_lr
    wandb.run.summary["optimizer/adam_vector_lr"] = input_lr
    wandb.run.summary["optimizer/adam_action_output_lr"] = input_lr / 8
    wandb.run.summary["optimizer/adam_trunk_value_output_lr"] = input_lr / 4
    wandb.run.summary["optimizer/adam_betas"] = scaled_adam_betas(cfg)
    wandb.run.summary["optimizer/adam_epsilon"] = scaled_adam_epsilon(cfg)
    wandb.run.summary["optimizer/adam_weight_decay"] = cfg.adam_weight_decay
    if cfg.parent_wandb_id is not None:
        wandb.run.summary["lineage/parent_wandb_id"] = cfg.parent_wandb_id
        wandb.run.summary["lineage/parent_run_name"] = cfg.parent_run_name
        wandb.run.summary["lineage/parent_checkpoint_name"] = cfg.parent_checkpoint_name
        wandb.run.summary["lineage/parent_checkpoint_sha256"] = cfg.parent_checkpoint_sha256
        wandb.run.summary["lineage/parent_update"] = cfg.decay_start_update
    if device_name is not None:
        wandb.run.summary["hardware/gpu_name"] = device_name
    if peak_flops is not None:
        wandb.run.summary["hardware/bf16_dense_peak_tflops"] = peak_flops / 1e12
        source = bf16_peak_source(device_name or "")
        if source is not None:
            wandb.run.summary["hardware/bf16_dense_peak_source"] = source
    wandb.run.summary["data/source_mixing"] = "per-source contiguous two-rank partition"
    for name, weight in zip(cfg.source_names, source_weights, strict=True):
        wandb.run.summary[f"data/source_sampling_share/{name}"] = weight / source_weight_total


def _training_functions(
    model: ActionSequenceTransformer, cfg: TrainConfig, *, compile_mode: str | None = None
) -> tuple[Callable, Callable]:
    """Return eager or singly compiled trunk and temporal training functions."""
    trunk_fn: Callable = model.forward
    temporal_fn: Callable = model.temporal.teacher_forced_nll_with_diagnostics
    mode = cfg.train_compile_mode if compile_mode is None else compile_mode
    if DEVICE == "cuda" and cfg.compile_trunk:
        # Resolve FlexAttention before Dynamo sees the model. This entrypoint is
        # the sole compilation owner for the raw mask and attention operations.
        model.trunk.resolve_attention(DEVICE)
        if model.trunk.attn_path != "varlen_flash":
            raise RuntimeError(
                f"compiled CUDA training requires a fused attention path, resolved {model.trunk.attn_path!r} instead"
            )
        print(f"[compile] calling torch.compile for trunk (mode={mode})", flush=True)
        trunk_fn = torch.compile(
            trunk_fn,
            dynamic=False,
            fullgraph=True,
            mode=mode,
        )
    if DEVICE == "cuda" and cfg.compile_temporal:
        print(f"[compile] calling torch.compile for temporal model (mode={mode})", flush=True)
        temporal_fn = torch.compile(
            temporal_fn,
            dynamic=False,
            fullgraph=True,
            mode=mode,
        )
    return trunk_fn, temporal_fn


class _TrainingOwner(nn.Module):
    """Own the model while calling inner functions compiled before DDP wrapping."""

    def __init__(
        self,
        model: ActionSequenceTransformer,
        cfg: TrainConfig,
        trunk_fn: Callable,
        temporal_fn: Callable,
    ) -> None:
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.trunk_fn = trunk_fn
        self.temporal_fn = temporal_fn

    def forward(
        self,
        batch: ReturnBatch,
        *,
        step: int,
        valid_prefixes: int,
        prefix_positions: Tensor,
        value_loss_scale: float,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        return microbatch_loss(
            self.model,
            batch,
            self.cfg,
            step=step,
            valid_prefixes=valid_prefixes,
            trunk_fn=self.trunk_fn,
            temporal_fn=self.temporal_fn,
            prefix_positions=prefix_positions,
            value_loss_scale=value_loss_scale,
        )


def wrap_training_owner(
    model: ActionSequenceTransformer,
    cfg: TrainConfig,
    distributed: DistributedContext,
    trunk_fn: Callable,
    temporal_fn: Callable,
) -> DistributedDataParallel:
    """Wrap the owner after its inner trunk and decoder functions are compiled."""
    owner = _TrainingOwner(model, cfg, trunk_fn, temporal_fn)
    device_ids = [distributed.local_rank] if distributed.device.type == "cuda" else None
    output_device = distributed.local_rank if distributed.device.type == "cuda" else None
    return DistributedDataParallel(
        owner,
        device_ids=device_ids,
        output_device=output_device,
        broadcast_buffers=True,
        gradient_as_bucket_view=True,
        static_graph=True,
    )


@dataclass(frozen=True, slots=True)
class TrainStepResult:
    nll_sum: Tensor
    gradient_norm: Tensor
    metrics: dict[str, Tensor]
    muon_lr: float
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

    def flush(
        self,
        cfg: TrainConfig,
        *,
        update: int,
    ) -> tuple[dict[str, float], int]:
        """Synchronize once, return window means, and reset the accumulator."""
        if self._sum is None or self.updates == 0 or self.valid_prefixes == 0:
            raise RuntimeError("cannot flush an empty training metric accumulator")
        payload = self._sum
        _all_reduce_sum(payload)
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        global_valid_prefixes = self.valid_prefixes * world_size
        payload = payload.cpu()
        if not torch.isfinite(payload).all():
            raise FloatingPointError(f"update {update}: accumulated training metrics contain a non-finite value")

        nll_values = len(cfg.arch.head_offsets) * CONTROLLER_GROUP_COUNT
        mean_nll = (
            payload[:nll_values].reshape(len(cfg.arch.head_offsets), CONTROLLER_GROUP_COUNT) / global_valid_prefixes
        )
        scalar_values = payload[nll_values:] / (self.updates * world_size)
        nll_metrics = nll_mean_metrics(
            mean_nll,
            cfg.arch.head_offsets,
        )
        values = {
            "train/nll": nll_metrics["loss_unweighted"],
            "optimizer/grad_norm": float(scalar_values[0]),
        }
        values.update({name: float(value) for name, value in zip(self._metric_names, scalar_values[1:], strict=True)})

        updates = self.updates
        self._sum = None
        self._metric_names = ()
        self.updates = 0
        self.valid_prefixes = 0
        return values, updates


def train_step(
    model: ActionSequenceTransformer,
    training_owner: DistributedDataParallel,
    batch: ReturnBatch,
    cfg: TrainConfig,
    *,
    step: int,
    update: int,
    valid_prefixes: int,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    prefix_sampler: PrefixSampler,
    prefix_validated_on_cpu: bool = False,
) -> TrainStepResult:
    """Run one complete optimization step on a device-resident batch."""
    if DEVICE == "cuda" and (cfg.compile_trunk or cfg.compile_temporal):
        torch.compiler.cudagraph_mark_step_begin()
    optimizer.zero_grad()
    prefix_positions = prefix_sampler.sample(
        batch.context.ctx_pad,
        length=cfg.arch.L_ctx,
        suffix_start=cfg.arch.direct_loss_start,
        validated_on_cpu=prefix_validated_on_cpu,
    )
    nll_sum: Tensor | None = None
    metrics: dict[str, Tensor] = {}
    microbatches = cfg.local_batch_size // cfg.microbatch_size
    additive_metrics = {"train/loss", "train/near_loss", "train/far_nll", "train/objective"}
    for index, start in enumerate(range(0, cfg.local_batch_size, cfg.microbatch_size)):
        stop = start + cfg.microbatch_size
        sync = contextlib.nullcontext() if index + 1 == microbatches else training_owner.no_sync()
        with sync:
            loss, batch_nll, batch_metrics = training_owner(
                batch.row_slice(start, stop),
                step=step,
                valid_prefixes=valid_prefixes,
                prefix_positions=prefix_positions[start:stop],
                value_loss_scale=cfg.microbatch_size / cfg.local_batch_size,
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"update {update}: loss is non-finite on the local rank")
            loss.backward()
        nll_sum = batch_nll if nll_sum is None else nll_sum + batch_nll
        for name, value in batch_metrics.items():
            scale = 1.0 if name in additive_metrics or "target_logit_grad_l1" in name else 1 / microbatches
            metrics[name] = metrics.get(name, torch.zeros_like(value)) + scale * value
    assert nll_sum is not None
    metrics["stability/action_grad_abs_max"] = _button_gradient_abs_max(model)
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
    metrics["optimizer/clip_fraction"] = (gradient_norm > cfg.grad_clip).float()
    if cfg.optimizer == "muon":
        muon_lr = float(next(group["lr"] for group in optimizer.param_groups if group["use_muon"]))
        adam_lr = float(next(group["lr"] for group in optimizer.param_groups if not group["use_muon"]))
    else:
        muon_lr = 0.0
        adam_lr = float(max(group["lr"] for group in optimizer.param_groups))
    optimizer.step()
    scheduler.step()
    return TrainStepResult(nll_sum, gradient_norm, metrics, muon_lr, adam_lr)


def _cadence_in_window(first_update: int, last_update: int, every: int) -> bool:
    """Return whether update one or a cadence boundary is in the window."""
    if every <= 0:
        return False
    return first_update == 1 or last_update // every > (first_update - 1) // every


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
    *,
    return_target: Literal["p90", "unconditioned"] = "unconditioned",
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
        "return_target": return_target,
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
    model: ActionSequenceTransformer,
    calibration: ReturnCalibration,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    cfg: TrainConfig,
    stats: dict[str, FeatureStats],
    val_cache: list[ReturnBatch],
    run_dir: Path,
    replay_dir: Path,
    uploader: BackgroundUploader | None,
    loader_wait_fractions: list[float],
    loader_state: dict[str, object],
    identity_masker_state: dict[str, object],
    return_masker_state: dict[str, object],
    prefix_sampler_state: dict[str, object],
    distributed: DistributedContext,
    update: int,
    actual_loss_positions: int,
    smoke: bool,
    defer_gameplay_evaluations: bool,
    resume_lineage: tuple[ResumeLineage, ...] = (),
) -> dict[str, object]:
    """Save the final model and queue evaluation for a separate L40S worker."""
    wait_lists: list[object] | None = [None] * distributed.world_size if distributed.is_primary else None
    dist.gather_object(loader_wait_fractions, wait_lists, dst=0)
    snapshot = save_boundary_checkpoint(
        run_dir,
        update=update,
        model=model,
        calibration=calibration,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        uploader=uploader,
        milestone=cfg.ckpt_every > 0 and update % cfg.ckpt_every == 0,
        wandb_id=None if wandb.run is None else wandb.run.id,
        actual_loss_positions=actual_loss_positions,
        loader_state=loader_state,
        identity_masker_state=identity_masker_state,
        return_masker_state=return_masker_state,
        prefix_sampler_state=prefix_sampler_state,
        distributed=distributed,
        resume_lineage=resume_lineage,
    )
    failure: list[str | None] = [None]
    summary: list[object] = [None]
    if distributed.is_primary:
        try:
            assert snapshot is not None
            final_path = run_dir / ("smoke-final.pt" if smoke else "final.pt")
            advance_checkpoint_link(snapshot, final_path)
            if uploader is not None:
                uploader.upload(snapshot, key=final_path.name)

            checkpoint_sha = checkpoint_sha256(final_path)
            validation = _offline_validation_metrics(model, val_cache, cfg)
            final_metrics = {f"val/{name}": value for name, value in validation.items()}
            if not smoke and not defer_gameplay_evaluations:
                if uploader is None:
                    raise RuntimeError("production evaluation requires R2 checkpoint upload")
                uploader.wait()
                spawn_closed_loop_evaluation(run_dir.name, update, checkpoint_sha, cfg.final_eval_n_matchups)
            wandb.log({"global_step": update, **final_metrics})

            assert wait_lists is not None
            all_waits = [value for rank_waits in wait_lists for value in cast(list[float], rank_waits)]
            mean_wait = float(np.mean(all_waits)) if all_waits else 0.0
            p95_wait = float(np.percentile(all_waits, 95)) if all_waits else 0.0
            print(f"[loader] mean wait={100 * mean_wait:.2f}%, p95={100 * p95_wait:.2f}%", flush=True)
            if smoke and (mean_wait > 0.05 or p95_wait > 0.10):
                raise RuntimeError("smoke loader gate failed: require mean wait <=5% and p95 <=10%")
            summary[0] = {
                "checkpoint": str(final_path),
                "checkpoint_sha256": checkpoint_sha,
                "loader_wait_mean": mean_wait,
                "loader_wait_p95": p95_wait,
            }
        except Exception as error:
            failure[0] = f"rank-zero finalization failed: {type(error).__name__}: {error}"
    dist.broadcast_object_list(failure, src=0)
    if failure[0] is not None:
        raise RuntimeError(failure[0])
    dist.broadcast_object_list(summary, src=0)
    if not isinstance(summary[0], dict):
        raise RuntimeError("rank zero did not broadcast finalization metrics")
    return cast(dict[str, object], summary[0])


def write_smoke_qualification(
    path: Path,
    *,
    cfg: TrainConfig,
    finalization: Mapping[str, object],
    throughput_samples_per_second: Sequence[float],
    gpu_peak_gb: Sequence[float],
    host_peak_gb: Sequence[float],
) -> None:
    """Write the immutable production-shape smoke evidence used by qualification."""
    if path.exists():
        raise FileExistsError(f"immutable qualification output already exists: {path}")
    if not throughput_samples_per_second or not gpu_peak_gb or not host_peak_gb:
        raise RuntimeError("qualification smoke did not collect complete throughput and memory telemetry")
    gpu_total_bytes = [torch.cuda.get_device_properties(index).total_memory for index in range(cfg.world_size)]
    host_total_bytes = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    record = {
        "schema_version": 1,
        "experiment_id": _EXPERIMENT_ID,
        "git_sha": run_provenance(cfg)["git_sha"],
        "command": sys.argv,
        "resolved_config": asdict(cfg),
        "partition_sha256": data_partition_hashes(cfg),
        "updates": 512,
        "loss_and_gradients_finite": True,
        "both_gpus_work": True,
        "nccl_errors": 0,
        "recompilation_errors": 0,
        "throughput_samples_per_second": list(throughput_samples_per_second),
        "throughput_median_samples_per_second": float(np.median(throughput_samples_per_second)),
        "gpu_peak_gb": max(gpu_peak_gb),
        "gpu_total_bytes": gpu_total_bytes,
        "host_sampled_peak_gb": max(host_peak_gb),
        "host_total_bytes": host_total_bytes,
        **finalization,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def _compile_synthetic_forward_backward(
    model: ActionSequenceTransformer,
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
    cuda_rng_state = torch.cuda.get_rng_state() if DEVICE == "cuda" else None
    try:
        heartbeat = (
            _elapsed_heartbeat(f"[compile] lazy compilation still running for {targets}")
            if compile_targets
            else contextlib.nullcontext()
        )
        with heartbeat:
            model.train()
            model.zero_grad(set_to_none=True)
            if DEVICE == "cuda" and (cfg.compile_trunk or cfg.compile_temporal):
                torch.compiler.cudagraph_mark_step_begin()
            batch = synthetic_awr_batch(cfg, torch.device(DEVICE))
            valid_prefixes = cfg.local_batch_size * POLICY_PREFIXES_PER_WINDOW
            prefix_positions = _synthetic_prefix_positions(cfg, torch.device(DEVICE))
            loss, _nll, _metrics = microbatch_loss(
                model,
                batch,
                cfg,
                step=step,
                valid_prefixes=valid_prefixes,
                trunk_fn=trunk_fn,
                temporal_fn=temporal_fn,
                prefix_positions=prefix_positions,
            )
            loss.backward()
            if DEVICE == "cuda":
                torch.cuda.synchronize()
    finally:
        model.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state)
    if compile_started is not None:
        print(f"[compile] lazy compilation complete in {time.monotonic() - compile_started:.1f}s", flush=True)


def _synthetic_prefix_positions(cfg: TrainConfig, device: torch.device) -> Tensor:
    """Match the contiguous layout returned by the real prefix sampler."""
    positions = torch.arange(
        cfg.arch.direct_loss_start,
        cfg.arch.direct_loss_start + POLICY_PREFIXES_PER_WINDOW,
        device=device,
    )
    return positions.expand(cfg.local_batch_size, -1).contiguous()


def _loader_state_boundaries(cfg: TrainConfig, run_stop: int) -> tuple[int, ...]:
    """Return updates after which fetched CPU batches must be consumed."""
    boundaries = {run_stop}
    for cadence in (cfg.val_every, cfg.eval_every, cfg.ckpt_every):
        if cadence > 0:
            boundaries.update(range(cadence, run_stop, cadence))
    return tuple(sorted(boundaries))


def train(
    cfg: TrainConfig,
    stats: dict[str, FeatureStats],
    distributed: DistributedContext,
    *,
    comment: str = "",
    resume_run: str | None = None,
    resume_state: dict | None = None,
    source_transition: ResumeLineage | None = None,
    resume_checkpoint_sha256: str | None = None,
    smoke: bool = False,
    proxy: bool = False,
    defer_gameplay_evaluations: bool = False,
    stop_after_update: int | None = None,
    qualification_output: Path | None = None,
) -> None:
    validate_config(cfg)
    if qualification_output is not None and (not smoke or stop_after_update != 512):
        raise ValueError("qualification output requires a 512-update production-shape smoke run")
    lineage = checkpoint_resume_lineage(None if resume_state is None else resume_state.get("resume_lineage"))
    if source_transition is not None and resume_state is None:
        raise ValueError("a source transition requires a resume checkpoint")
    if lineage:
        assert resume_state is not None
        provenance = resume_state.get("provenance")
        if not isinstance(provenance, dict) or provenance.get("git_sha") != lineage[-1].new_source_sha:
            raise ValueError("resume lineage does not end at the checkpoint source")
    if source_transition is not None:
        if lineage and lineage[-1].new_source_sha != source_transition.old_source_sha:
            raise ValueError("resume lineage source transitions are not contiguous")
        lineage = (*lineage, source_transition)
    if smoke and proxy:
        raise ValueError("proxy and smoke modes are mutually exclusive")
    if stop_after_update is not None and stop_after_update < 1:
        raise ValueError("stop_after_update must be positive")
    if stop_after_update is not None:
        run_stop = stop_after_update
    elif cfg.decay_start_update is not None:
        if cfg.decay_duration is None:
            raise RuntimeError("validated decay branch has no duration")
        run_stop = cfg.decay_start_update + cfg.decay_duration
    elif resume_state is None and cfg.arch == Architecture():
        run_stop = cfg.stable_updates
    else:
        run_stop = cfg.max_steps
    names: list[object] = [
        resume_run or make_run_name(Path(__file__).stem, model_tag(cfg), "policy-world-v8", comment)
        if distributed.is_primary
        else None
    ]
    dist.broadcast_object_list(names, src=0)
    run_name = names[0]
    if not isinstance(run_name, str):
        raise RuntimeError("rank zero did not broadcast a run name")
    uploader = BackgroundUploader(run_name) if cfg.push_to_r2 and distributed.is_primary else None
    if distributed.is_primary:
        _init_wandb(cfg, run_name, resume_state)
    if cfg.parent_wandb_id is not None and wandb.run is not None and wandb.run.id == cfg.parent_wandb_id:
        raise RuntimeError("the continuation must use a new W&B run id")
    if distributed.is_primary:
        run_dir, replay_dir = setup_run_dir(run_name)
    else:
        run_dir = Path("runs") / run_name
        replay_dir = run_dir / "replays"
    dist.barrier()
    process_seed = rank_seed(cfg.seed, distributed.rank)
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    torch.set_float32_matmul_precision("high" if cfg.allow_tf32 else "highest")
    sidecar = load_identity_sidecar(cfg)
    local_resume = None if resume_state is None else rank_resume_state(resume_state, cfg, distributed)
    prepared_data = _prepare_training_data(cfg, stats, sidecar, distributed, local_resume)
    model_started = time.monotonic()
    print(
        f"[rank {distributed.rank}] constructing ActionSequenceTransformer and moving parameters "
        f"to {distributed.device}",
        flush=True,
    )
    model = make_model(cfg, sidecar.vocabulary).to(distributed.device)
    calibration = ReturnCalibration(window_count=CALIBRATION_WINDOWS // distributed.world_size)
    print(f"[model] construction complete in {time.monotonic() - model_started:.1f}s", flush=True)
    counts = subsystem_parameter_counts(model)
    flops_per_update = approximate_training_flops_per_update(cfg, counts)
    device_name = torch.cuda.get_device_name() if DEVICE == "cuda" else None
    peak_flops = bf16_dense_peak_flops(device_name or "")
    if distributed.is_primary:
        _log_training_summary(
            cfg,
            counts,
            prepared_data.loader,
            flops_per_update=flops_per_update,
            device_name=device_name,
            peak_flops=peak_flops,
        )
    optimizer = make_optimizer(model, cfg)
    scheduler = LearningRateScheduler(optimizer, lr_schedule(cfg))
    start_step = 0
    actual_positions = 0
    identity_masker = IdentityMasker(process_seed ^ 0x0501D, cfg.identity_dropout)
    return_masker = ReturnMasker(process_seed ^ 0x060C0D, cfg.return_dropout, enabled=cfg.return_conditioning)
    prefix_sampler = PrefixSampler(process_seed ^ 0x060A11, distributed.device)
    if resume_state is not None:
        assert local_resume is not None
        validate_conditioning_state(resume_state, cfg)
        return_masker.load_state_dict(cast(Mapping[str, object], local_resume["return_masker"]))
        calibration.load_state_dict(cast(dict[str, object], local_resume["return_calibration"]))
        model.load_state_dict(resume_state["model"])
        optimizer.load_state_dict(resume_state["opt"])
        scheduler.load_state_dict(resume_state["sched"])
        identity_state = local_resume.get("identity_masker")
        if not isinstance(identity_state, dict):
            raise ValueError("resume checkpoint has no identity-mask RNG state")
        identity_masker.load_state_dict(cast(dict[str, object], identity_state))
        prefix_state = local_resume.get("prefix_sampler")
        if not isinstance(prefix_state, dict):
            raise ValueError("resume checkpoint has no prefix-sampling RNG state")
        prefix_sampler.load_state_dict(cast(dict[str, object], prefix_state))
        rng = local_resume.get("rng")
        if not isinstance(rng, Mapping):
            raise ValueError("resume checkpoint has no global RNG state")
        validate_resume_provenance(
            resume_state.get("provenance"),
            run_provenance(cfg),
            transition=source_transition,
            parent_checkpoint_sha256=resume_checkpoint_sha256,
        )
        restore_rng(cast(Mapping[str, object], rng))
        start_step = int(resume_state["step"]) + 1
        positions_per_update = cfg.supervised_positions_per_update
        actual_positions = int(resume_state.get("actual_loss_positions", start_step * positions_per_update))
        if actual_positions != start_step * positions_per_update:
            raise ValueError(
                f"checkpoint actual_loss_positions={actual_positions} is invalid after {start_step} updates"
            )

    trunk_fn, temporal_fn = _training_functions(model, cfg)
    train_loader, val_cache = prepared_data.loader, prepared_data.validation
    _compile_synthetic_forward_backward(
        model,
        cfg,
        step=start_step,
        trunk_fn=trunk_fn,
        temporal_fn=temporal_fn,
    )
    training_owner = wrap_training_owner(model, cfg, distributed, trunk_fn, temporal_fn)
    run_started = time.monotonic()
    try:
        batch_prefetcher = DeviceBatchPrefetcher(
            train_loader,
            cfg,
            distributed.device,
            identity_masker,
            return_masker=return_masker,
            calibration=calibration,
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
    if distributed.is_primary and wandb.run is not None:
        wandb.run.summary["loader/worker_start_s"] = prepared_data.worker_start_seconds
        wandb.run.summary["loader/cold_fill_s"] = cold_fill_seconds
    loader_wait_fractions: list[float] = []
    throughput_samples_per_second: list[float] = []
    gpu_peak_gb: list[float] = []
    host_peak_gb: list[float] = []
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
    window_peak_allocated_gb = 0.0
    update_timer = _UpdateTimer()
    # CUDA compilation must remain on the training thread. Background compilation
    # deadlocked training on both H100 and B200 hosts.
    model.train()
    recompile_guard = contextlib.ExitStack()
    evaluation_updates = frozenset(closed_loop_evaluation_updates(run_stop, cfg.eval_every))
    loader_state_boundaries = _loader_state_boundaries(cfg, run_stop)
    try:
        if DEVICE == "cuda" and (cfg.compile_trunk or cfg.compile_temporal):
            recompile_guard.enter_context(torch.compiler.set_stance("fail_on_recompile"))
        for step in range(start_step, run_stop):
            update = step + 1
            update_timer.start()
            if DEVICE == "cuda":
                torch.cuda.reset_peak_memory_stats()

            val_due = cfg.val_every > 0 and update % cfg.val_every == 0 and update < run_stop
            eval_due = not defer_gameplay_evaluations and update in evaluation_updates and update < run_stop
            ckpt_due = cfg.ckpt_every > 0 and update % cfg.ckpt_every == 0 and update < run_stop
            boundary_due = val_due or eval_due or ckpt_due
            state_boundary_due = boundary_due or update == run_stop
            next_state_boundary = next(boundary for boundary in loader_state_boundaries if boundary >= update)

            batch, valid_prefixes = batch_prefetcher.next()
            lookahead_limit = 0 if state_boundary_due else next_state_boundary - update
            batch_prefetcher.fill_lookahead(lookahead_limit)
            window_loader_submitted_batches.append(batch_prefetcher.submitted_batches)
            window_loader_ready_batches.append(batch_prefetcher.ready_batches)
            metrics_due = update % cfg.train_metrics_every == 0 or update == run_stop
            result = train_step(
                model,
                training_owner,
                batch,
                cfg,
                step=step,
                update=update,
                valid_prefixes=valid_prefixes,
                optimizer=optimizer,
                scheduler=scheduler,
                prefix_sampler=prefix_sampler,
                prefix_validated_on_cpu=True,
            )
            loader_wait = batch_prefetcher.stage_next() if not state_boundary_due else 0.0
            actual_positions += cfg.supervised_positions_per_update
            metric_accumulator.add(result, valid_prefixes)
            window_loader_wait_seconds.append(loader_wait)
            if DEVICE == "cuda":
                window_peak_allocated_gb = max(
                    window_peak_allocated_gb,
                    torch.cuda.max_memory_reserved() / 2**30,
                )

            if metrics_due:
                window_metric_values, window_updates = metric_accumulator.flush(
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
                local_loader_wait_s = sum(window_loader_wait_seconds) / window_updates
                local_loader_wait_p95_s = float(np.percentile(window_loader_wait_seconds, 95))
                local_submitted_batches = sum(window_loader_submitted_batches) / window_updates
                local_ready_batches = sum(window_loader_ready_batches) / window_updates
                loader_wait_s = reduce_scalar(local_loader_wait_s, distributed)
                loader_wait_p95_s = reduce_scalar(local_loader_wait_p95_s, distributed, maximum=True)
                submitted_batches = reduce_scalar(local_submitted_batches, distributed)
                ready_batches = reduce_scalar(local_ready_batches, distributed)
                training_elapsed_wall_s = reduce_scalar(
                    time.monotonic() - run_started,
                    distributed,
                    maximum=True,
                )
                completed_updates = update - start_step
                projected_training_remaining_s = training_elapsed_wall_s * (run_stop - update) / completed_updates
                first_window_update = update - window_updates + 1
                identity_metrics = reduce_metric_dict(identity_masker.metrics(), distributed)
                log: dict[str, object] = {
                    "data/windows": update * cfg.batch_size,
                    "data/supervised_prefixes": actual_positions,
                    "data/value_supervised_prefixes": update * cfg.value_prefixes_per_update,
                    "data/future_targets": actual_positions * len(cfg.arch.head_offsets),
                    "data/dropped_windows": 0,
                    "loader/queue_depth": submitted_batches,
                    "loader/submitted_cpu_batches": submitted_batches,
                    "loader/ready_cpu_batches": ready_batches,
                    "loader/uncovered_wait_s": loader_wait_s,
                    "loader/uncovered_wait_p95_s": loader_wait_p95_s,
                    "progress/elapsed_s": training_elapsed_wall_s,
                    "progress/remaining_s": projected_training_remaining_s,
                    "schedule/muon_lr": result.muon_lr,
                    "schedule/adam_lr": result.adam_lr,
                    **window_metric_values,
                    **identity_metrics,
                }
                loader_metrics = getattr(train_loader, "metrics", None)
                if isinstance(loader_metrics, dict):
                    log.update(reduce_metric_dict(loader_metrics, distributed))
                if _cadence_in_window(first_window_update, update, cfg.system_metrics_every):
                    local_system_metrics = _minimal_system_metrics(host_metrics.snapshot())
                    memory_gb = local_system_metrics.get("system/memory_gb", 0.0)
                    log["system/memory_gb"] = reduce_scalar(memory_gb, distributed, maximum=True)
                if DEVICE == "cuda":
                    log["system/gpu_memory_gb"] = reduce_scalar(
                        window_peak_allocated_gb,
                        distributed,
                        maximum=True,
                    )

                local_update_s, local_update_p95_s = update_timer.stats_and_reset(window_updates)
                local_loader_wait_fraction = local_loader_wait_s / max(local_update_s, 1e-12)
                loader_wait_fractions.extend([local_loader_wait_fraction] * window_updates)
                update_s = reduce_scalar(local_update_s, distributed, maximum=True)
                update_p95_s = reduce_scalar(local_update_p95_s, distributed, maximum=True)
                samples_per_s = cfg.batch_size / update_s
                loader_wait_fraction = loader_wait_s / max(update_s, 1e-12)
                log["throughput/update_s"] = update_s
                log["throughput/update_p95_s"] = update_p95_s
                log["throughput/samples_per_s"] = samples_per_s
                log["throughput/windows_per_s"] = samples_per_s
                log["throughput/policy_prefixes_per_s"] = samples_per_s * POLICY_PREFIXES_PER_WINDOW
                log["throughput/value_prefixes_per_s"] = samples_per_s * 128
                log["loader/uncovered_wait_fraction"] = loader_wait_fraction
                if peak_flops is not None:
                    log["throughput/mfu"] = model_flops_utilization(
                        flops_per_update,
                        update_s,
                        peak_flops * distributed.world_size,
                    )
                if distributed.is_primary:
                    throughput_samples_per_second.append(samples_per_s)
                    gpu_peak_gb.append(cast(float, log["system/gpu_memory_gb"]))
                    if "system/memory_gb" in log:
                        host_peak_gb.append(cast(float, log["system/memory_gb"]))
                    wandb.log({"global_step": update, **log})
                if distributed.is_primary and (
                    update <= cfg.train_metrics_every or update % 50 == 0 or update == run_stop
                ):
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
                window_peak_allocated_gb = 0.0
            checkpoint_path: Path | None = None
            if boundary_due:
                if not batch_prefetcher.drained:
                    raise RuntimeError("CPU lookahead was not drained at a state boundary")
                checkpoint_path = save_boundary_checkpoint(
                    run_dir,
                    update=update,
                    model=model,
                    calibration=calibration,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    cfg=cfg,
                    uploader=uploader,
                    milestone=cfg.eval_every > 0 and update % cfg.eval_every == 0,
                    wandb_id=None if wandb.run is None else wandb.run.id,
                    actual_loss_positions=actual_positions,
                    loader_state=train_loader.state_dict(),
                    identity_masker_state=identity_masker.state_dict(),
                    return_masker_state=return_masker.state_dict(),
                    prefix_sampler_state=prefix_sampler.state_dict(),
                    distributed=distributed,
                    resume_lineage=lineage,
                )
            if boundary_due:
                boundary_failure: list[str | None] = [None]
                if distributed.is_primary:
                    try:
                        boundary_metrics: dict[str, float] = {}
                        if val_due:
                            values = _offline_validation_metrics(model, val_cache, cfg)
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
                    except Exception as error:
                        boundary_failure[0] = f"rank-zero boundary work failed: {type(error).__name__}: {error}"
                dist.broadcast_object_list(boundary_failure, src=0)
                if boundary_failure[0] is not None:
                    raise RuntimeError(boundary_failure[0])
            if update < run_stop and boundary_due:
                next_state_boundary = next(boundary for boundary in loader_state_boundaries if boundary > update)
                batch_prefetcher.fill_lookahead(next_state_boundary - update)
                batch_prefetcher.stage_next()
        if not batch_prefetcher.drained:
            raise RuntimeError("CPU lookahead was not drained at the final update")
        finalization = _finalize_training(
            model=model,
            calibration=calibration,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            stats=stats,
            val_cache=val_cache,
            run_dir=run_dir,
            replay_dir=replay_dir,
            uploader=uploader,
            loader_wait_fractions=loader_wait_fractions,
            loader_state=train_loader.state_dict(),
            identity_masker_state=identity_masker.state_dict(),
            return_masker_state=return_masker.state_dict(),
            prefix_sampler_state=prefix_sampler.state_dict(),
            distributed=distributed,
            resume_lineage=lineage,
            update=run_stop,
            actual_loss_positions=actual_positions,
            smoke=smoke,
            defer_gameplay_evaluations=defer_gameplay_evaluations,
        )
        if distributed.is_primary and qualification_output is not None:
            write_smoke_qualification(
                qualification_output,
                cfg=cfg,
                finalization=finalization,
                throughput_samples_per_second=throughput_samples_per_second,
                gpu_peak_gb=gpu_peak_gb,
                host_peak_gb=host_peak_gb,
            )
            if uploader is not None:
                uploader.upload(qualification_output, key="qualification/smoke.json")
    finally:
        recompile_guard.close()
        batch_prefetcher.close()
        train_loader.close()
        host_metrics.close()
        if uploader is not None:
            uploader.upload_tree(replay_dir, base=run_dir)
            uploader.close()
        if distributed.is_primary:
            wandb.finish()


def _checkpoint_config(cfg: TrainConfig) -> dict[str, object]:
    values = asdict(cfg)
    architecture = values.pop("arch")
    calibration = values.pop("awr")
    compute_equivalent_parameters = compute_equivalent_parameter_count(cfg, cfg.arch.parameter_count_contract)
    return {
        "experiment_id": _EXPERIMENT_ID,
        "checkpoint_format_version": _CHECKPOINT_FORMAT_VERSION,
        "architecture": architecture,
        "awr_calibration": calibration,
        **values,
        "max_steps": cfg.max_steps,
        "warmup_steps": cfg.warmup_steps,
        "offset_loss_weights": OFFSET_LOSS_WEIGHTS,
        "compute_equivalent_parameters": compute_equivalent_parameters,
        "flops_per_supervised_position": compute_flops_per_supervised_position(compute_equivalent_parameters),
    }


def config_from_state(values: dict) -> TrainConfig:
    """Restore a checkpoint written by the current experiment definition."""
    derived_fields = {
        "max_steps",
        "warmup_steps",
        "offset_loss_weights",
        "compute_equivalent_parameters",
        "flops_per_supervised_position",
    }
    runtime_fields = {item.name for item in fields(TrainConfig)} - {"arch", "awr"}
    expected = {
        "experiment_id",
        "checkpoint_format_version",
        "architecture",
        "awr_calibration",
        *runtime_fields,
        *derived_fields,
    }
    missing = expected - values.keys()
    unexpected = values.keys() - expected
    if missing or unexpected:
        raise ValueError(f"checkpoint config mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    if values["experiment_id"] != _EXPERIMENT_ID:
        raise ValueError(f"checkpoint experiment_id {values['experiment_id']!r} != {_EXPERIMENT_ID!r}")
    if values["checkpoint_format_version"] != _CHECKPOINT_FORMAT_VERSION:
        raise ValueError("checkpoint format version mismatch")
    architecture_values = values["architecture"]
    calibration_values = values["awr_calibration"]
    if set(architecture_values) != {item.name for item in fields(Architecture)}:
        raise ValueError("checkpoint architecture does not match the current architecture fields")
    if set(calibration_values) != {item.name for item in fields(AWRCalibration)}:
        raise ValueError("checkpoint calibration does not match the current calibration fields")
    architecture = Architecture(**architecture_values)
    calibration = AWRCalibration(**calibration_values)
    runtime = {name: values[name] for name in runtime_fields}
    cfg = TrainConfig(arch=architecture, awr=calibration, **runtime)
    compute_equivalent_parameters = compute_equivalent_parameter_count(cfg, cfg.arch.parameter_count_contract)
    derived = {name: values[name] for name in derived_fields}
    expected_derived = {
        "max_steps": cfg.max_steps,
        "warmup_steps": cfg.warmup_steps,
        "offset_loss_weights": OFFSET_LOSS_WEIGHTS,
        "compute_equivalent_parameters": compute_equivalent_parameters,
        "flops_per_supervised_position": compute_flops_per_supervised_position(compute_equivalent_parameters),
    }
    if derived != expected_derived:
        raise ValueError(f"checkpoint derived schedule mismatch: {derived} != {expected_derived}")
    return cfg


def validate_conditioning_state(state: Mapping[str, object], cfg: TrainConfig) -> None:
    if state.get("conditioning_protocol") != conditioning_protocol():
        raise ValueError("incompatible conditioning protocol")
    calibration = state.get("return_calibration")
    masker = state.get("return_masker")
    if not isinstance(calibration, dict) or not isinstance(masker, dict):
        raise ValueError("checkpoint lacks return calibration or masker state")
    ReturnCalibration(window_count=CALIBRATION_WINDOWS).load_state_dict(cast(dict[str, object], calibration))
    ReturnMasker(0, cfg.return_dropout, enabled=cfg.return_conditioning).load_state_dict(
        cast(Mapping[str, object], masker)
    )


def load_checkpoint(
    path: str, *, device: str = DEVICE
) -> tuple[ActionSequenceTransformer, TrainConfig, dict[str, FeatureStats], dict]:
    state = torch.load(path, map_location=device, weights_only=False)
    cfg = config_from_state(state["cfg"])
    validate_config(cfg)
    validate_conditioning_state(state, cfg)
    encoded = state["model"].get("player_code_bytes")
    if not isinstance(encoded, Tensor) or not encoded.numel():
        raise ValueError("checkpoint has no embedded identity vocabulary")
    vocabulary = PlayerVocabulary(decode_player_codes(encoded.detach().cpu().numpy().tobytes()))
    model = make_model(cfg, vocabulary).to(device)
    model.load_state_dict(state["model"])
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
    return_target: Literal["p90", "unconditioned"] = "unconditioned",
) -> dict[str, float]:
    actual_checkpoint_sha256 = checkpoint_sha256(Path(path))
    if expected_checkpoint_sha256 is not None and actual_checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError(
            f"checkpoint SHA-256 mismatch: expected {expected_checkpoint_sha256}, got {actual_checkpoint_sha256}"
        )
    model, cfg, stats, state = load_checkpoint(path)
    validate_config(cfg)
    if return_target not in ("p90", "unconditioned"):
        raise ValueError("unknown return target")
    calibration = ReturnCalibration(window_count=CALIBRATION_WINDOWS)
    calibration.load_state_dict(state["return_calibration"])
    desired_return = calibration.targets()[2] if return_target == "p90" else None
    calibration_hash = cast(str, calibration.state_dict()["sha256"])
    if return_target != "unconditioned":
        raise ValueError("O60 checkpoints do not support return-conditioned evaluation")
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
        desired_return=desired_return,
        return_calibration_sha256=calibration_hash,
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
    proxy_arm: ProxyArm = "none"
    comment: str = ""
    resume: str | None = None
    resume_checkpoint: str = "latest.pt"
    resume_source_transition: Path | None = None
    smoke: bool = False
    defer_gameplay_evaluations: bool = False
    stop_after_update: int | None = None
    qualification_output: Path | None = None
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
    return_target: Literal["p90", "unconditioned"] = "unconditioned"


type Command = (
    Annotated[TrainArgs, tyro.conf.subcommand(name="train")] | Annotated[EvalArgs, tyro.conf.subcommand(name="eval")]
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
            return_target=args.return_target,
        )
        return
    cfg = args.cfg
    source_transition = None
    resume_checkpoint_sha = None
    if args.resume is None and args.resume_source_transition is not None:
        raise SystemExit("--resume-source-transition requires --resume")
    if args.proxy_arm != "none":
        proxy = proxy_config_for_arm(args.proxy_arm)
        cfg = replace(
            cfg,
            arch=proxy.arch,
            target_positions=proxy.target_positions,
            stable_updates=proxy.stable_updates,
            decay_start_update=None,
            decay_duration=None,
        )
    cfg = replace(
        cfg,
        eval_max_parallel=cfg.eval_max_parallel if args.eval_max_parallel is None else args.eval_max_parallel,
    )
    distributed = init_distributed(cfg)
    try:
        resume_state: dict[str, object] | None = None
        if args.resume is not None:
            checkpoint = Path(args.resume_checkpoint)
            if (
                checkpoint.is_absolute()
                or ".." in checkpoint.parts
                or checkpoint.suffix != ".pt"
                or args.resume_checkpoint in ("", ".")
            ):
                raise SystemExit("--resume-checkpoint must be a relative .pt object within the run")
            if distributed.is_primary:
                resume_state = load_for_resume(
                    args.resume,
                    Path("runs") / args.resume,
                    device=str(distributed.device),
                    name=args.resume_checkpoint,
                )
            found: list[object] = [resume_state is not None]
            dist.broadcast_object_list(found, src=0)
            if not found[0]:
                raise SystemExit(f"no {args.resume_checkpoint!r} for run {args.resume!r}")
            dist.barrier()
            if not distributed.is_primary:
                resume_state = torch.load(
                    Path("runs") / args.resume / args.resume_checkpoint,
                    map_location=distributed.device,
                    weights_only=False,
                )
            assert resume_state is not None
            if args.resume_source_transition is not None:
                source_transition = read_resume_lineage(args.resume_source_transition)
                resume_checkpoint_sha = checkpoint_sha256(Path("runs") / args.resume / args.resume_checkpoint)
            cfg = config_from_state(cast(dict, resume_state["cfg"]))
            requested = proxy_config_for_arm(args.proxy_arm)
            if cfg.arch != requested.arch or cfg.target_positions != requested.target_positions:
                raise SystemExit("--proxy-arm must match the resumed checkpoint treatment")
        stats = load_stats(cfg)
        train(
            cfg,
            stats,
            distributed,
            comment=args.comment,
            resume_run=args.resume,
            resume_state=resume_state,
            source_transition=source_transition,
            resume_checkpoint_sha256=resume_checkpoint_sha,
            smoke=args.smoke,
            proxy=args.proxy_arm != "none",
            defer_gameplay_evaluations=args.defer_gameplay_evaluations,
            stop_after_update=args.stop_after_update,
            qualification_output=args.qualification_output,
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main(tyro.cli(cast(type[Command], Command)))
