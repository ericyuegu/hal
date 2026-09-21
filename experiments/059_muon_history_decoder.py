"""Experiment 059: wide history decoder with Muon and WSD.

This experiment trains AWR over the complete policy-world corpus. A learned
value head estimates the return from each trunk state, and the detached
``G_{t+1} - V(s_t)`` advantage weights the policy objective. It includes the
schema-v6 projectile block (``item{0..3}_*``) in every observation.

This file is deliberately standalone. It retains the O41 detached-value light
AWR and O49 ego-identity contracts without importing another experiment. The
temporal decoder uses 4x MLP expansion and matched nonlinear decoder and
trunk-skip logit heads. A 60-frame return modulates each decoder block's norms;
the trunk, critic, history memory, and trunk-skip heads remain return-independent.

The four item slots are ordered by ascending spawn id, so a slot keeps its item
until an OLDER item despawns and the remaining items shift down. A pooled set
encoder makes that churn invisible: one shared per-slot encoder, gated by the
slot's presence flag, summed over the slots. An empty slot adds the exact zero
vector and the live-item count stays implicit in the sum.

The policy samples 32 suffix prefixes per replay window. The critic and AWR
normalizer still use all 128 suffix prefixes. One history cross-attention block
after decoder block one shares a single projected 256-state memory per window.

Run:
    uv run experiments/059_muon_history_decoder.py train
    uv run experiments/059_muon_history_decoder.py train --resume <run>
    uv run experiments/059_muon_history_decoder.py train --resume <parent> \
        --resume-checkpoint checkpoints/step-0098304.pt --resume-as <child> \
        --decay-duration 32768
    uv run experiments/059_muon_history_decoder.py eval --checkpoint runs/<run>/final.pt
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import itertools
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
import triton
import triton.language as tl
import tyro
from beartype import beartype
from jaxtyping import Bool
from jaxtyping import Float
from jaxtyping import Int
from jaxtyping import jaxtyped
from torch import Tensor
from torch.nn.attention.flex_attention import flex_attention
from torch.optim.lr_scheduler import LambdaLR

import wandb
from hal import r2
from hal import streams
from hal.data.feature_stats import FeatureStats
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
from hal.eval.policy_sampling import SlotGroupRng
from hal.eval.policy_sampling import sample_categorical
from hal.eval.policy_sampling import validate_sampling_temperature
from hal.eval.self_play import DecodeTelemetry
from hal.eval.self_play import canonical_context
from hal.eval.self_play import synthetic_context as build_synthetic_context
from hal.sim.process_vec import ProcessVecTelemetry
from hal.sim.rollout import covering_power_of_two
from hal.training import returns as returns_lib
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
from hal.training.muon import SingleDeviceMuonWithAuxAdam
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
from hal.training.trunk import rmsnorm
from hal.wire import ITEM_SLOTS
from hal.wire import item_column

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_EXPERIMENT_ID: Final[str] = "059_muon_history_decoder_v5"
_CHECKPOINT_FORMAT_VERSION: Final[int] = 4
_STARTUP_LOG_INTERVAL_S: Final[float] = 60.0
POLICY_PREFIXES_PER_WINDOW: Final[int] = 32
OFFSET_LOSS_WEIGHTS: Final[tuple[float, ...]] = (
    0.50,
    0.07,
    0.07,
    0.07,
    0.07,
    0.07,
    0.015,
    0.015,
    0.015,
    0.015,
    0.015,
    0.015,
    0.015,
    0.015,
    0.015,
    0.015,
)


RETURN_HORIZON: Final[int] = 60
RETURN_SCALE: Final[float] = 120.0
CALIBRATION_WINDOWS: Final[int] = 65_536


def conditioning_protocol() -> dict[str, object]:
    return {
        "version": 1,
        "horizon": RETURN_HORIZON,
        "gamma": 0.99855,
        "reward": "damage_opp-damage_ego+120*(stock_loss_opp-stock_loss_ego)+50*(last_stock_opp-last_stock_ego)",
        "scale": RETURN_SCALE,
        "alignment": "sum(k=1..60, gamma**(k-1)*r[t+k])",
        "availability": "full observed horizon or known terminal with zero rewards thereafter",
        "evaluation": "positive p90 at every replan; separate unconditioned comparison",
        "modulation": "per-block RMSNorm affine, 128-wide SiLU, biased zero-initialized projections",
        "dropout": "independent CPU generator, per context position",
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
        return ReturnBatch(
            TrainBatch(
                Context(
                    {name: value[:count] for name, value in self.context.features.items()},
                    self.context.ctx_pad[:count],
                    None if self.context.slot_ids is None else self.context.slot_ids[:count],
                    None if self.context.reset is None else self.context.reset[:count],
                ),
                self.target[:count],
                None if self.batch.replay_ids is None else self.batch.replay_ids[:count],
            ),
            self.returns[:count],
            self.eligible[:count],
            self.future_return[:count],
            self.available[:count],
            self.condition_present[:count],
        )


class ReturnCalibration:
    def __init__(self) -> None:
        self.values: list[float] = []
        self.valid: list[bool] = []
        self.replay_ids: list[str] = []

    def observe(self, batch: ReturnBatch) -> None:
        remaining = CALIBRATION_WINDOWS - len(self.values)
        if remaining <= 0:
            return
        count = min(remaining, len(batch.future_return))
        if batch.batch.replay_ids is None:
            raise ValueError("calibration requires replay identities")
        valid = batch.available[:count, -1].detach().cpu()
        values = torch.where(valid, batch.future_return[:count, -1].detach().cpu(), 0.0)
        if not bool(torch.isfinite(values).all()):
            raise ValueError("nonfinite valid calibration return")
        self.values.extend(values.tolist())
        self.valid.extend(valid.tolist())
        self.replay_ids.extend(batch.batch.replay_ids[:count])

    def targets(self) -> tuple[float, float, float]:
        if len(self.values) != CALIBRATION_WINDOWS:
            raise ValueError("return calibration is incomplete")
        positive = np.asarray(self.values)[np.asarray(self.valid) & (np.asarray(self.values) > 0)]
        if not positive.size:
            raise ValueError("calibration has no strictly positive valid returns")
        median, p90 = np.quantile(positive, [0.5, 0.9], method="linear")
        return 0.0, float(median), float(p90)

    def state_dict(self) -> dict[str, object]:
        data = {"values": self.values.copy(), "valid": self.valid.copy(), "replay_ids": self.replay_ids.copy()}
        identity = hashlib.sha256(json.dumps(data, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        return {
            "version": 1,
            **data,
            "sha256": identity,
            "targets": self.targets() if len(self.values) == CALIBRATION_WINDOWS else None,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if set(state) != {"version", "values", "valid", "replay_ids", "sha256", "targets"} or state["version"] != 1:
            raise ValueError("incompatible return calibration")
        if not all(isinstance(state[name], list) for name in ("values", "valid", "replay_ids")):
            raise ValueError("calibration samples must be ordered lists")
        values = cast(list[float], state["values"])
        valid = cast(list[bool], state["valid"])
        replay_ids = cast(list[str], state["replay_ids"])
        if (
            any(type(value) is not float or not math.isfinite(value) for value in values)
            or any(type(value) is not bool for value in valid)
            or any(not isinstance(value, str) or not value for value in replay_ids)
        ):
            raise ValueError("invalid calibration sample types")
        self.values = values.copy()
        self.valid = list(cast(list[bool], state["valid"]))
        self.replay_ids = list(cast(list[str], state["replay_ids"]))
        if (
            len(self.values) != len(self.valid)
            or len(self.values) != len(self.replay_ids)
            or len(self.values) > CALIBRATION_WINDOWS
        ):
            raise ValueError("invalid calibration lengths")
        if self.state_dict() != state:
            raise ValueError("return calibration identity or targets changed")


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


@dataclass(frozen=True)
class Architecture:
    player_embed_dim: ClassVar[int] = 32
    activation_percentile_sample_size: ClassVar[int] = 65_536
    trunk_attention_backend: ClassVar[str] = "varlen_flash"
    trunk_reference_layers: ClassVar[int] = 8
    temporal_reference_layers: ClassVar[int] = 2
    trunk_reference_attention_scale: ClassVar[float] = 0.25
    temporal_reference_attention_scale: ClassVar[float] = 0.5

    d_model: int = 1024
    n_layers: int = 12
    n_heads: int = 16
    attn_window: int = 0
    L_ctx: int = 256

    sample_chunk_length: int = 28
    head_offsets: tuple[int, ...] = (*range(1, 13), 16, 20, 24, 28)
    temporal_d_model: int = 1024
    temporal_layers: int = 6
    temporal_heads: int = 16
    temporal_ff_dim: int = 4096
    group_head_dim: int = 1024
    return_embed_dim: int = 128
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

    @property
    def direct_loss_start(self) -> int:
        if self.L_ctx % 2:
            raise ValueError("context length must be even for suffix supervision")
        return self.L_ctx // 2

    @property
    def parameter_count_contract(self) -> dict[str, int]:
        if self == Architecture():
            return {
                "trunk": 150_994_944,
                "temporal_decoder": 81_300_464,
                "group_heads": 4_558_179,
                "trunk_skip_heads": 4_558_179,
                "value_head": 1_049_089,
                "return_conditioner": 3_170_560,
                "other": 1_230_790,
                "total": 246_862_205,
            }
        proxy = Architecture(
            d_model=256,
            n_layers=12,
            n_heads=4,
            temporal_d_model=256,
            temporal_layers=6,
            temporal_heads=4,
            temporal_ff_dim=1024,
            group_head_dim=256,
            value_hidden_dim=128,
        )
        if self == proxy:
            return {
                "trunk": 9_437_184,
                "temporal_decoder": 5_195_504,
                "group_heads": 353_379,
                "trunk_skip_heads": 353_379,
                "value_head": 65_665,
                "return_conditioner": 792_832,
                "other": 861_382,
                "total": 17_059_325,
            }
        raise ValueError(f"no parameter contract for architecture {self}")


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class TrainConfig:
    reference_batch_size: ClassVar[int] = 512
    reference_positions: ClassVar[int] = 2**30
    base_adam_betas: ClassVar[tuple[float, float]] = (0.9, 0.95)
    base_adam_eps: ClassVar[float] = 1e-12
    inference_buckets: ClassVar[tuple[int, ...]] = (1, 2, 4, 8, 16, 32, 64)
    train_metrics_every: ClassVar[int] = 25
    train_prefetch_factor: ClassVar[int] = 4
    raw_shard_materialization_threads: ClassVar[int] = 64
    materialization_threads_env: ClassVar[str] = "HAL_O59_MATERIALIZATION_THREADS"
    data_protocol: ClassVar[str] = "o59-replay-ring-v2"
    selection_sha256: ClassVar[str] = "2593361352b92e705be3fbeae1b4e9bb1a3c9f1787cd713014a7a95b7df62477"
    mds_index_version: ClassVar[int] = 2
    mds_manifest_schema_sha256: ClassVar[str] = "405199de9494fe01350506734f0b2ec392fe79b0122d69cbcb5cae2afabc0d49"
    replay_slots: ClassVar[int] = 131_072
    windows_per_generation: ClassVar[int] = 8
    replay_phase_block_batches: ClassVar[int] = 25
    minimum_replay_gap_batches: ClassVar[int] = 200
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
    num_workers: int = 24
    push_to_r2: bool = True
    system_metrics_every: int = 25
    system_metrics_interval_s: float = 5.0
    process_metrics_interval_s: float = 30.0
    cache_metrics_interval_s: float = 30.0
    identity_dropout: float = 0.10
    return_conditioning: bool = True
    return_dropout: float = 0.2
    parent_run_name: Annotated[str | None, tyro.conf.Suppress] = None
    parent_checkpoint_name: Annotated[str | None, tyro.conf.Suppress] = None
    parent_checkpoint_sha256: Annotated[str | None, tyro.conf.Suppress] = None
    parent_wandb_id: Annotated[str | None, tyro.conf.Suppress] = None
    player_sidecar_local: str = "data/processed/player-identity-v1/professional-code-v1.jsonl.gz"
    player_sidecar_sha256: str = "54ccf8a2497fe240313117297ca2ea31158e08db2cc53c67e7aa46853a8dac1c"
    player_vocab_sha256: str = "c67c97c995ad033ea7f5b2223efce5b061394566439f091ff6e7aaa6a9d1cfd6"
    player_vocab_size: int = 21_181
    target_positions: int = 8 * 2**30
    stable_updates: int = 98_304
    decay_start_update: int | None = None
    decay_duration: int | None = None
    depth_alpha: float = 0.5
    hidden_std_multiplier: float = 0.5
    readout_init: Literal["mup-normal"] = "mup-normal"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-12

    @property
    def max_steps(self) -> int:
        positions_per_update = self.batch_size * (self.arch.L_ctx - self.arch.L_ctx // 2)
        updates, remainder = divmod(self.target_positions, positions_per_update)
        if remainder:
            raise ValueError("target_positions must end on an optimizer boundary")
        return updates

    @property
    def warmup_steps(self) -> int:
        return 4096

    @property
    def policy_prefixes_per_update(self) -> int:
        return self.batch_size * 32

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
    if cfg.optimizer != "muon":
        raise ValueError("O59 requires Muon")
    if cfg.train_compile_mode not in ("reduce-overhead", "max-autotune"):
        raise ValueError("unsupported training compile mode")
    production = Architecture()
    proxy = proxy_config().arch
    if cfg.arch not in (production, proxy):
        raise ValueError("O59 architecture must be the production shape or its fixed proxy")
    if cfg.batch_size != 512 or cfg.seed != 0:
        raise ValueError("O59 freezes batch_size=512 and seed=0")
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
    if offsets != (*range(1, 13), 16, 20, 24, 28):
        raise ValueError("O59 prediction offsets are frozen")
    if cfg.arch == Architecture() and cfg.minimum_replay_frames != 291:
        raise ValueError("eight distinct O59 windows require at least 291 replay frames")
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
        raise ValueError("O59 freezes its value and AWR calibration")
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
    if not isinstance(cfg.num_workers, int) or isinstance(cfg.num_workers, bool) or not 0 <= cfg.num_workers <= 48:
        raise ValueError(f"num_workers must be an integer in [0, 48], got {cfg.num_workers!r}")
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
    if not isinstance(cfg.return_conditioning, bool) or not 0.0 <= cfg.return_dropout <= 1.0:
        raise ValueError("return conditioning requires a boolean enablement and dropout in [0, 1]")
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
        raise ValueError("O59 requires all 44 policy-world-v8 sources")
    if cfg.depth_alpha != 0.5 or cfg.hidden_std_multiplier != 0.5 or cfg.readout_init != "mup-normal":
        raise ValueError("O59 uses O50's selected depth and initialization parameterization")
    if (cfg.adam_beta1, cfg.adam_beta2, cfg.adam_eps) != (*cfg.base_adam_betas, cfg.base_adam_eps):
        raise ValueError("O59 scales the fixed base Adam betas and epsilon")
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
        raise ValueError("O59 optimizer hyperparameters are frozen")
    if cfg.muon_lr_multiplier != 1.0:
        raise ValueError("O59 freezes optimizer hyperparameters across continuations")
    if cfg.arch == Architecture() and (
        cfg.max_steps != 131_072 or cfg.warmup_steps != 4096 or cfg.stable_updates != 98_304
    ):
        raise ValueError("O59 training boundaries are frozen")
    if (cfg.decay_start_update is None) != (cfg.decay_duration is None):
        raise ValueError("decay_start_update and decay_duration must be set together")
    if cfg.decay_start_update is not None:
        if cfg.decay_start_update < cfg.warmup_steps:
            raise ValueError("a decay fork must start in the stable phase")
        if cfg.decay_duration is None or cfg.decay_duration <= 0:
            raise ValueError("decay_duration must be positive")
        lineage = (
            cfg.parent_run_name,
            cfg.parent_checkpoint_name,
            cfg.parent_checkpoint_sha256,
            cfg.parent_wandb_id,
        )
        if any(value is None for value in lineage):
            raise ValueError("a decay fork requires explicit parent checkpoint lineage")
    elif any(
        value is not None
        for value in (
            cfg.parent_run_name,
            cfg.parent_checkpoint_name,
            cfg.parent_checkpoint_sha256,
            cfg.parent_wandb_id,
        )
    ):
        raise ValueError("parent checkpoint lineage is only valid for a decay fork")


def proxy_config() -> TrainConfig:
    """Return the fixed-width proxy with the same 12/6 depth split."""
    return TrainConfig(
        arch=Architecture(
            d_model=256,
            n_layers=12,
            n_heads=4,
            temporal_d_model=256,
            temporal_layers=6,
            temporal_heads=4,
            temporal_ff_dim=1024,
            group_head_dim=256,
            value_hidden_dim=128,
        ),
        target_positions=TrainConfig.reference_positions,
    )


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


def synthetic_awr_batch(cfg: TrainConfig, device: torch.device) -> ReturnBatch:
    """Build one fully valid production-shaped batch without touching the corpus."""
    context = synthetic_context(cfg, cfg.batch_size, device)
    target = torch.zeros(cfg.batch_size, cfg.arch.sample_chunk_length, A_DIM, device=device)
    returns = torch.zeros(cfg.batch_size, cfg.arch.L_ctx, device=device)
    eligible = torch.ones(cfg.batch_size, cfg.arch.L_ctx, dtype=torch.bool, device=device)
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


def decoder_rmsnorm(x: Tensor) -> Tensor:
    return F.rms_norm(x, (x.shape[-1],), eps=1e-6)


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

    def __init__(self, d_model: int, d_hidden: int, vocab: int, *, norm_eps: float = 1e-6) -> None:
        super().__init__()
        self.norm_eps = norm_eps
        self.up = nn.Linear(d_model, d_hidden, bias=False)
        self.down = nn.Linear(d_hidden, vocab)

    def normalize(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.shape[-1],), eps=self.norm_eps)

    def project(self, x: Tensor) -> Tensor:
        """Apply the head MLP to an input prepared by its caller."""
        return self.down(F.silu(self.up(x)))

    def forward(self, x: Tensor) -> Tensor:
        return self.project(self.normalize(x))

    def forward_with_input(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return logits and the normalized tensor read by the hidden layer."""
        logits, normalized, _up_output, _down_input = self.projection_activations(x)
        return logits, normalized

    def projection_activations(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return the output and the tensors at both projection boundaries."""
        normalized = self.normalize(x)
        output, up_output, down_input = self.project_activations(normalized)
        return output, normalized, up_output, down_input

    def project_activations(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return the output and activations for an input prepared by its caller."""
        up_output = self.up(x)
        down_input = F.silu(up_output)
        return self.down(down_input), up_output, down_input


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


def _bounded_flat_sample(tensor: Tensor) -> Tensor:
    """Return a deterministic activation sample without retaining its source."""
    values = tensor.detach().flatten()
    sample_size = Architecture.activation_percentile_sample_size
    stride = max((values.numel() + sample_size - 1) // sample_size, 1)
    return values[::stride][:sample_size].float()


def _activation_input_metrics(prefix: str, tensor: Tensor) -> dict[str, Tensor]:
    values = _bounded_flat_sample(tensor)
    return {
        f"{prefix}/input_rms": values.square().mean().sqrt(),
        f"{prefix}/input_abs_max": values.abs().amax(),
    }


def _activation_output_metrics(prefix: str, tensor: Tensor) -> dict[str, Tensor]:
    """Measure a bounded output sample, ignoring masked infinite logits."""
    values = _bounded_flat_sample(tensor)
    legal = torch.isfinite(values)
    count = legal.sum().clamp_min(1)
    finite = values.masked_fill(~legal, 0)
    ordered = values.abs().masked_fill(~legal, torch.inf).sort().values

    def percentile(numerator: int, denominator: int) -> Tensor:
        rank = (count * numerator + denominator - 1).div(denominator, rounding_mode="floor").clamp_min(1)
        return ordered.gather(0, (rank - 1).reshape(1)).squeeze(0)

    return {
        f"{prefix}/output_rms": (finite.square().sum() / count).sqrt(),
        f"{prefix}/output_abs_p99": percentile(99, 100),
        f"{prefix}/output_abs_p999": percentile(999, 1000),
    }


def short_causal_attention(
    query: Float[Tensor, "B H L D"],
    key: Float[Tensor, "B H L D"],
    value: Float[Tensor, "B H L D"],
) -> Float[Tensor, "B H L D"]:
    """Use explicit causal attention for the 16-token training sequence.

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
        raise ValueError("O59 fixes depth_alpha to 0.5")
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


class ReturnConditioner(nn.Module):
    """Modulate decoder norms once per prefix; absence is exactly neutral."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.width = cfg.arch.temporal_d_model
        self.enabled = cfg.return_conditioning
        self.embedding = nn.Linear(1, cfg.arch.return_embed_dim, bias=True)
        self.projections = nn.ModuleList(
            nn.Linear(cfg.arch.return_embed_dim, 4 * self.width, bias=True) for _ in range(cfg.arch.temporal_layers)
        )
        nn.init.normal_(self.embedding.weight, std=cfg.hidden_std_multiplier)
        nn.init.zeros_(self.embedding.bias)
        for module in self.projections:
            projection = cast(nn.Linear, module)
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(self, return_value: Tensor, condition_present: Tensor) -> tuple[Tensor, ...]:
        if return_value.shape != condition_present.shape or condition_present.dtype != torch.bool:
            raise ValueError("return values and boolean presence must have the same shape")
        present = condition_present & self.enabled
        safe = torch.where(present, return_value, 0.0)
        embedded = F.silu(self.embedding((safe / RETURN_SCALE)[..., None]))
        return tuple(
            torch.where(
                present[..., None, None],
                projection(embedded).reshape(*return_value.shape, 4, self.width),
                0.0,
            )
            for projection in self.projections
        )


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

    def _qkv(self, x: Tensor, scale: Tensor, shift: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, length, _ = x.shape
        q, k, v = self.qkv((1 + scale) * decoder_rmsnorm(x) + shift).split(self.d_model, dim=-1)
        shape = (batch, length, self.n_heads, self.head_dim)
        return q.view(shape), k.view(shape), v.view(shape)

    def forward(self, x: Tensor, modulation: Tensor) -> Tensor:
        scale_attn, shift_attn, scale_mlp, shift_mlp = modulation.unbind(-2)
        q, k, v = self._qkv(x, scale_attn, shift_attn)
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
        return x + self.mlp_scale * self.down(F.silu(self.up((1 + scale_mlp) * decoder_rmsnorm(x) + shift_mlp)))

    def forward_step(
        self, x: Tensor, past: tuple[Tensor, Tensor] | None, modulation: Tensor
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        scale_attn, shift_attn, scale_mlp, shift_mlp = modulation.unbind(-2)
        q, k, v = self._qkv(x[:, None], scale_attn[:, None], shift_attn[:, None])
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
        x = x + self.mlp_scale * self.down(F.silu(self.up((1 + scale_mlp) * decoder_rmsnorm(x) + shift_mlp)))
        return x, (k, v)


class HistoryCrossAttention(nn.Module):
    """Attend packed prefix/offset queries to one shared trunk memory."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.n_heads = cfg.arch.temporal_heads
        self.d_model = cfg.arch.temporal_d_model
        self.head_dim = self.d_model // self.n_heads
        self.scale = depth_rule("temporal", cfg.arch.temporal_layers, cfg.depth_alpha).attention
        self.query = nn.Linear(self.d_model, self.d_model, bias=False)
        self.key_value = nn.Linear(cfg.arch.d_model, 2 * self.d_model, bias=False)
        self.output = nn.Linear(self.d_model, self.d_model, bias=False)
        self.rotary = Rotary(self.head_dim)

    def project_memory(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        if hidden.ndim != 3:
            raise ValueError(f"history memory must be [B, L, d], got {tuple(hidden.shape)}")
        batch, length, _ = hidden.shape
        key, value = self.key_value(decoder_rmsnorm(hidden)).chunk(2, dim=-1)
        shape = (batch, length, self.n_heads, self.head_dim)
        key = key.view(shape)
        value = value.view(shape).transpose(1, 2)
        cos, sin = self.rotary.at(length, key.device, key.dtype)
        return apply_rotary_emb(key, cos, sin).transpose(1, 2), value

    def forward_projected(
        self,
        x: Tensor,
        key: Tensor,
        value: Tensor,
        ctx_pad: Tensor,
        prefix_positions: Tensor,
        *,
        offsets_per_prefix: int,
    ) -> Tensor:
        """Apply causal history attention without replicating K/V by prefix."""
        batch, queries, _ = x.shape
        length = key.shape[2]
        if key.shape != (batch, self.n_heads, length, self.head_dim) or value.shape != key.shape:
            raise ValueError("history K/V shape mismatch")
        if ctx_pad.shape != (batch,) or prefix_positions.ndim != 2:
            raise ValueError("history padding and prefix positions have the wrong shape")
        if queries != prefix_positions.shape[1] * offsets_per_prefix:
            raise ValueError("history query packing does not match prefix positions")

        query = self.query(decoder_rmsnorm(x)).view(batch, queries, self.n_heads, self.head_dim)
        query_positions = prefix_positions.repeat_interleave(offsets_per_prefix, dim=1)
        cos, sin = self.rotary.at(length, query.device, query.dtype)
        query_cos = cos[0, :, 0][query_positions][:, :, None, :]
        query_sin = sin[0, :, 0][query_positions][:, :, None, :]
        query = apply_rotary_emb(query, query_cos, query_sin).transpose(1, 2)

        if query.device.type == "cuda":

            def score_mod(score: Tensor, b: Tensor, h: Tensor, q: Tensor, kv: Tensor) -> Tensor:
                del h
                allowed = (kv >= ctx_pad[b]) & (kv <= query_positions[b, q])
                return torch.where(allowed, score, -torch.inf)

            attended = cast(Tensor, flex_attention(query, key, value, score_mod=score_mod))
        else:
            memory = torch.arange(length, device=x.device)
            valid = (memory[None, None, :] >= ctx_pad[:, None, None]) & (
                memory[None, None, :] <= query_positions[:, :, None]
            )
            attended = F.scaled_dot_product_attention(query, key, value, attn_mask=valid[:, None])
        attended = attended.transpose(1, 2).contiguous().view_as(x)
        return x + self.scale * self.output(attended)


class CausalTemporalDecoder(nn.Module):
    """Temporal action chain conditioned by concatenation."""

    def __init__(self, cfg: TrainConfig, codec: DiscreteControllerCodec) -> None:
        super().__init__()
        self.codec = codec
        self.head_offsets = tuple(cfg.arch.head_offsets)
        self.live_horizons = (cfg.prediction_frames,)
        self.d_model = cfg.arch.temporal_d_model
        self.training_diagnostics = cfg.optimizer == "adamw"
        controller_width = CONTROLLER_GROUP_COUNT * cfg.arch.action_embed_dim
        self.offset_embedding = nn.Embedding(cfg.arch.sample_chunk_length + 1, cfg.arch.offset_embed_dim)
        self.token_projection = nn.Linear(
            cfg.arch.d_model + controller_width + cfg.arch.offset_embed_dim, self.d_model
        )
        self.blocks = nn.ModuleList([TemporalBlock(cfg) for _ in range(cfg.arch.temporal_layers)])
        self.history_attention = HistoryCrossAttention(cfg)
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
                    self.d_model,
                    cfg.arch.group_head_dim,
                    CONTROLLER_GROUP_VOCABS[CONTROLLER_GROUP_INDEX[name]],
                    # Keep the near-zero button Jacobian bounded in every path.
                    norm_eps=1e-5 if name == "buttons" else 1e-6,
                )
                for name in CONTROLLER_GROUP_NAMES
            }
        )
        self.trunk_outputs = nn.ModuleDict(
            {
                name: NonlinearActionHead(
                    cfg.arch.d_model,
                    cfg.arch.group_head_dim,
                    CONTROLLER_GROUP_VOCABS[CONTROLLER_GROUP_INDEX[name]],
                )
                for name in CONTROLLER_GROUP_NAMES
            }
        )
        self.trunk_width = cfg.arch.d_model
        self.controller_width = controller_width

    return_conditioner: ReturnConditioner

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

    def _step_features(self, previous: Tensor, offsets: Tensor, embedded: Tensor | None = None) -> Tensor:
        """The per-step share of the token projection: previous action and offset."""
        weight = self.token_projection.weight
        action_weight = weight[:, self.trunk_width : self.trunk_width + self.controller_width]
        offset_weight = weight[:, self.trunk_width + self.controller_width :]
        action = F.linear(self.codec.embed_frame(previous) if embedded is None else embedded, action_weight)
        return action + F.linear(self.offset_embedding(offsets), offset_weight)

    def _trunk_skip_logits(self, hidden: Tensor) -> dict[str, Tensor]:
        """Compute context-only logits once for all decoded offsets."""
        return {name: self.trunk_outputs[name](hidden) for name in CONTROLLER_GROUP_NAMES}

    def _decode_step(
        self,
        previous: Tensor,
        offset: int,
        state_bias: Tensor,
        caches: list[tuple[Tensor, Tensor] | None],
        history: tuple[Tensor, Tensor, Tensor, Tensor],
        conditioning: tuple[Tensor, ...],
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor] | None]]:
        """Advance the temporal chain by one selected frame offset."""
        offsets = torch.full((previous.shape[0],), offset, device=previous.device, dtype=torch.long)
        state = decoder_rmsnorm(state_bias + self._step_features(previous, offsets))
        next_caches: list[tuple[Tensor, Tensor] | None] = []
        for index, (module, past) in enumerate(zip(self.blocks, caches, strict=True)):
            block = cast(TemporalBlock, module)
            state, present = block.forward_step(state, past, conditioning[index])
            if index == 0:
                key, value, ctx_pad, prefix_positions = history
                state = self.history_attention.forward_projected(
                    state[:, None],
                    key,
                    value,
                    ctx_pad,
                    prefix_positions,
                    offsets_per_prefix=1,
                )[:, 0]
            next_caches.append(present)
        return decoder_rmsnorm(state), next_caches

    def teacher_forced_states(
        self,
        hidden: Tensor,
        ctx_pad: Tensor,
        prefix_positions: Tensor,
        observed: Tensor,
        targets: Tensor,
        return_value: Tensor,
        condition_present: Tensor,
    ) -> Tensor:
        prefix_shape = tuple(prefix_positions.shape)
        expected = (*prefix_shape, len(self.head_offsets), CONTROLLER_GROUP_COUNT)
        if observed.shape != (*prefix_shape, CONTROLLER_GROUP_COUNT) or targets.shape != expected:
            raise ValueError(
                f"expected observed {(*prefix_shape, CONTROLLER_GROUP_COUNT)} and targets {expected}, got "
                f"{tuple(observed.shape)} and {tuple(targets.shape)}"
            )
        batch_indices = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
        selected_hidden = hidden[batch_indices, prefix_positions]
        previous = torch.cat((observed[:, :, None], targets[..., :-1, :]), dim=2)
        trunk = decoder_rmsnorm(selected_hidden)
        offsets = torch.tensor(self.head_offsets, device=hidden.device)
        x = self._state_bias(trunk)[:, :, None] + self._step_features(previous, offsets)
        x = decoder_rmsnorm(x)
        x = x.reshape(hidden.shape[0] * prefix_positions.shape[1], len(self.head_offsets), self.d_model)
        conditioning = self.return_conditioner(return_value, condition_present)
        for index, block in enumerate(self.blocks):
            x = block(x, conditioning[index].reshape(-1, 1, 4, self.d_model))
            if index == 0:
                packed = x.view(hidden.shape[0], -1, self.d_model)
                key, value = self.history_attention.project_memory(hidden)
                packed = self.history_attention.forward_projected(
                    packed,
                    key,
                    value,
                    ctx_pad,
                    prefix_positions,
                    offsets_per_prefix=len(self.head_offsets),
                )
                x = packed.view_as(x)
        return decoder_rmsnorm(x.view(*prefix_shape, len(self.head_offsets), self.d_model))

    def group_features(self, states: Tensor, name: str, embedded: dict[str, Tensor]) -> Tensor:
        head = cast(NonlinearActionHead, self.outputs[name])
        normalized = head.normalize(states)
        position = CONTROLLER_DECODE_ORDER.index(name)
        if position == 0:
            return normalized
        prefix = torch.cat([embedded[group] for group in CONTROLLER_DECODE_ORDER[:position]], dim=-1)
        raw_scale, raw_shift = self.group_condition[name](prefix).chunk(2, dim=-1)
        scale = torch.tanh(raw_scale)
        shift = raw_shift
        return normalized * (1.0 + scale) + shift

    def _teacher_forced_outputs(
        self,
        hidden: Tensor,
        ctx_pad: Tensor,
        prefix_positions: Tensor,
        observed: Tensor,
        targets: Tensor,
        return_value: Tensor,
        condition_present: Tensor,
    ) -> tuple[
        dict[str, Tensor],
        tuple[Tensor, Tensor, Tensor],
        dict[str, tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]],
    ]:
        """Return group logits and tensors at the action projection boundaries."""
        states = self.teacher_forced_states(
            hidden, ctx_pad, prefix_positions, observed, targets, return_value, condition_present
        )
        batch_indices = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
        selected_hidden = hidden[batch_indices, prefix_positions]
        embedded = self.codec.embed_groups(targets)
        logits: dict[str, Tensor] = {}
        projection_values: dict[
            str,
            tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
        ] = {}
        button_values: tuple[Tensor, Tensor] | None = None
        for name in CONTROLLER_GROUP_NAMES:
            features = self.group_features(states, name, embedded)
            head = cast(NonlinearActionHead, self.outputs[name])
            head_output, up_output, down_input = head.project_activations(features)
            head_input = features
            trunk_head = cast(NonlinearActionHead, self.trunk_outputs[name])
            trunk_output, trunk_input, trunk_up_output, trunk_down_input = trunk_head.projection_activations(
                selected_hidden
            )
            combined_logits = head_output + trunk_output[..., None, :]
            projection_values[name] = (
                head_input,
                up_output,
                down_input,
                head_output,
                trunk_input,
                trunk_up_output,
                trunk_down_input,
                trunk_output,
            )
            if name == "buttons":
                button_values = (head_input, combined_logits)
            logits[name] = self._center(combined_logits)
        if button_values is None:
            raise RuntimeError("button head was not evaluated")
        button_mask = self.codec.button_mask(targets[..., TRIGGERS_GROUP])
        logits["buttons"] = logits["buttons"].masked_fill(button_mask, float("-inf"))
        return logits, (*button_values, button_mask), projection_values

    def teacher_forced_logits_by_group(
        self,
        hidden: Tensor,
        ctx_pad: Tensor,
        prefix_positions: Tensor,
        observed: Tensor,
        targets: Tensor,
        return_value: Tensor,
        condition_present: Tensor,
    ) -> dict[str, Tensor]:
        logits, _button_values, _projection_values = self._teacher_forced_outputs(
            hidden, ctx_pad, prefix_positions, observed, targets, return_value, condition_present
        )
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

    def teacher_forced_nll(
        self,
        hidden: Tensor,
        ctx_pad: Tensor,
        prefix_positions: Tensor,
        observed: Tensor,
        targets: Tensor,
        return_value: Tensor,
        condition_present: Tensor,
    ) -> Tensor:
        logits = self.teacher_forced_logits_by_group(
            hidden, ctx_pad, prefix_positions, observed, targets, return_value, condition_present
        )
        return self.nll_from_logits(logits, targets)

    def teacher_forced_nll_with_diagnostics(
        self,
        hidden: Tensor,
        ctx_pad: Tensor,
        prefix_positions: Tensor,
        observed: Tensor,
        targets: Tensor,
        return_value: Tensor,
        condition_present: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Return NLL plus a compact button-boundary stability signal."""
        logits, button_values, projection_values = self._teacher_forced_outputs(
            hidden, ctx_pad, prefix_positions, observed, targets, return_value, condition_present
        )
        metrics = self._button_metrics(*button_values, targets)
        if self.training_diagnostics:
            for name, values in projection_values.items():
                metrics.update(self._group_metrics(name, values, logits[name]))
        return self.nll_from_logits(logits, targets), metrics

    @staticmethod
    def _button_metrics(
        head_input: Tensor, raw_logits: Tensor, button_mask: Tensor, targets: Tensor
    ) -> dict[str, Tensor]:
        input_values = head_input.detach()
        raw_logits_values = raw_logits.detach()
        button_targets = targets[..., BUTTONS_GROUP, None]
        target_logits = raw_logits_values.gather(-1, button_targets).squeeze(-1).float()
        legal_logits = raw_logits_values.masked_fill(button_mask, float("-inf"))
        competing_logits = legal_logits.scatter(-1, button_targets, float("-inf")).amax(dim=-1).float()
        margin = target_logits - competing_logits
        return {
            "stability/button_input_abs_p999": _sampled_quantile(input_values, 99.9, absolute=True),
            "stability/button_logit_abs_p999": _sampled_quantile(raw_logits_values, 99.9, absolute=True),
            "stability/button_margin_mean": margin.mean(),
        }

    @staticmethod
    def _group_metrics(
        name: str, values: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor], logits: Tensor
    ) -> dict[str, Tensor]:
        (
            head_input,
            up_output,
            down_input,
            down_output,
            trunk_input,
            trunk_up_output,
            trunk_down_input,
            trunk_output,
        ) = values
        metrics: dict[str, Tensor] = {}
        prefix = f"diagnostics/activations/action/{name}"
        metrics.update(_activation_input_metrics(f"{prefix}/up", head_input))
        metrics.update(_activation_output_metrics(f"{prefix}/up", up_output))
        metrics.update(_activation_input_metrics(f"{prefix}/down", down_input))
        metrics.update(_activation_output_metrics(f"{prefix}/down", down_output))
        metrics.update(_activation_input_metrics(f"{prefix}/trunk_skip/up", trunk_input))
        metrics.update(_activation_output_metrics(f"{prefix}/trunk_skip/up", trunk_up_output))
        metrics.update(_activation_input_metrics(f"{prefix}/trunk_skip/down", trunk_down_input))
        metrics.update(_activation_output_metrics(f"{prefix}/trunk_skip/down", trunk_output))
        metrics.update(_activation_output_metrics(f"{prefix}/combined_centered_logits", logits))
        return metrics

    def teacher_forced_logits(
        self,
        hidden: Tensor,
        ctx_pad: Tensor,
        prefix_positions: Tensor,
        observed: Tensor,
        targets: Tensor,
        return_value: Tensor,
        condition_present: Tensor,
    ) -> list[dict[str, Tensor]]:
        values = self.teacher_forced_logits_by_group(
            hidden, ctx_pad, prefix_positions, observed, targets, return_value, condition_present
        )
        return [
            {name: logits[..., depth, :] for name, logits in values.items()} for depth in range(len(self.head_offsets))
        ]

    def _live_history(self, hidden: Tensor, ctx_pad: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if ctx_pad is None:
            ctx_pad = torch.zeros(hidden.shape[0], dtype=torch.long, device=hidden.device)
        prefix_positions = torch.full(
            (hidden.shape[0], 1), hidden.shape[1] - 1, dtype=torch.long, device=hidden.device
        )
        return (*self.history_attention.project_memory(hidden), ctx_pad, prefix_positions)

    def forced_stepwise_logits(
        self,
        hidden: Tensor,
        observed: Tensor,
        targets: Tensor,
        return_value: Tensor,
        condition_present: Tensor,
        *,
        ctx_pad: Tensor | None = None,
    ) -> list[dict[str, Tensor]]:
        if targets.shape != (hidden.shape[0], len(self.head_offsets), CONTROLLER_GROUP_COUNT):
            raise ValueError("stepwise targets have the wrong shape")
        raw_trunk = hidden[:, -1]
        trunk = decoder_rmsnorm(raw_trunk)
        state_bias = self._state_bias(trunk)
        trunk_logits = self._trunk_skip_logits(raw_trunk)
        previous = observed
        caches: list[tuple[Tensor, Tensor] | None] = [None] * len(self.blocks)
        history = self._live_history(hidden, ctx_pad)
        conditioning = self.return_conditioner(return_value, condition_present)
        out: list[dict[str, Tensor]] = []
        for depth, offset in enumerate(self.head_offsets):
            state, caches = self._decode_step(previous, offset, state_bias, caches, history, conditioning)
            target = targets[:, depth]
            embedded = self.codec.embed_groups(target)
            group_logits = {
                name: self._center(
                    cast(NonlinearActionHead, self.outputs[name]).project(self.group_features(state, name, embedded))
                    + trunk_logits[name]
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
        return_value: Tensor,
        condition_present: Tensor,
        *,
        argmax: bool,
        uniforms: Tensor | None = None,
        gen: torch.Generator | None = None,
        temperature: float = 1.0,
        ctx_pad: Tensor | None = None,
        forced_prefix: Tensor | None = None,
    ) -> Tensor:
        indices, _ = self._sample_indices_and_logits(
            hidden,
            observed,
            offsets,
            return_value,
            condition_present,
            argmax=argmax,
            uniforms=uniforms,
            gen=gen,
            temperature=temperature,
            capture_logits=False,
            ctx_pad=ctx_pad,
            forced_prefix=forced_prefix,
        )
        return indices

    def sample_indices_with_logits(
        self,
        hidden: Tensor,
        observed: Tensor,
        offsets: tuple[int, ...],
        return_value: Tensor,
        condition_present: Tensor,
        *,
        argmax: bool,
        uniforms: Tensor | None = None,
        gen: torch.Generator | None = None,
        temperature: float = 1.0,
        ctx_pad: Tensor | None = None,
        forced_prefix: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        """Sample once and return the exact conditional logits used by each draw."""
        return self._sample_indices_and_logits(
            hidden,
            observed,
            offsets,
            return_value,
            condition_present,
            argmax=argmax,
            uniforms=uniforms,
            gen=gen,
            temperature=temperature,
            capture_logits=True,
            ctx_pad=ctx_pad,
            forced_prefix=forced_prefix,
        )

    def _sample_indices_and_logits(
        self,
        hidden: Tensor,
        observed: Tensor,
        offsets: tuple[int, ...],
        return_value: Tensor,
        condition_present: Tensor,
        *,
        argmax: bool,
        uniforms: Tensor | None,
        gen: torch.Generator | None,
        temperature: float,
        capture_logits: bool,
        ctx_pad: Tensor | None,
        forced_prefix: Tensor | None,
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        temperature = validate_sampling_temperature(temperature)
        allowed = tuple(self.head_offsets[:horizon] for horizon in self.live_horizons)
        if offsets not in allowed:
            raise ValueError(f"live decode offsets must select one of the dense prefixes {allowed}")
        if uniforms is not None and uniforms.shape != (len(offsets), CONTROLLER_GROUP_COUNT, hidden.shape[0]):
            raise ValueError("uniform table must be [frames, groups, batch]")
        if forced_prefix is not None and (
            forced_prefix.ndim != 3
            or forced_prefix.shape[0] != hidden.shape[0]
            or forced_prefix.shape[2] != CONTROLLER_GROUP_COUNT
            or forced_prefix.shape[1] > len(offsets)
        ):
            raise ValueError("forced prefix must be [B, K, groups] with K no larger than the horizon")
        raw_trunk = hidden[:, -1]
        trunk = decoder_rmsnorm(raw_trunk)
        state_bias = self._state_bias(trunk)
        trunk_logits = self._trunk_skip_logits(raw_trunk)
        previous = observed
        caches: list[tuple[Tensor, Tensor] | None] = [None] * len(self.blocks)
        history = self._live_history(hidden, ctx_pad)
        conditioning = self.return_conditioner(return_value, condition_present)
        frames: list[Tensor] = []
        captured: dict[str, list[Tensor]] = {name: [] for name in CONTROLLER_GROUP_NAMES}
        for depth, offset in enumerate(offsets):
            state, caches = self._decode_step(previous, offset, state_bias, caches, history, conditioning)
            embedded: dict[str, Tensor] = {}
            picks: dict[str, Tensor] = {}
            for name in CONTROLLER_DECODE_ORDER:
                logits = self._center(
                    cast(NonlinearActionHead, self.outputs[name]).project(self.group_features(state, name, embedded))
                    + trunk_logits[name]
                )
                if name == "buttons":
                    logits = logits.masked_fill(self.codec.button_mask(picks["triggers"]), float("-inf"))
                if capture_logits:
                    captured[name].append(logits)
                group = CONTROLLER_GROUP_INDEX[name]
                uniform = None if uniforms is None else uniforms[depth, group]
                if forced_prefix is not None and depth < forced_prefix.shape[1]:
                    pick = forced_prefix[:, depth, group]
                else:
                    pick = sample_categorical(
                        logits,
                        argmax=argmax,
                        uniform=uniform,
                        generator=gen,
                        temperature=temperature,
                    )
                picks[name] = pick
                embedded[name] = self.codec.group_embedding(name, pick)
            indices = torch.stack([picks[name] for name in CONTROLLER_GROUP_NAMES], dim=-1)
            frames.append(indices)
            previous = indices
        logits_by_group = (
            tuple(torch.stack(captured[name], dim=1) for name in CONTROLLER_GROUP_NAMES) if capture_logits else ()
        )
        return torch.stack(frames, dim=1), logits_by_group

    def rollout_conditioned_logits(
        self,
        hidden: Tensor,
        observed: Tensor,
        return_value: Tensor,
        condition_present: Tensor,
        *,
        ctx_pad: Tensor | None = None,
    ) -> tuple[list[dict[str, Tensor]], Tensor]:
        """Offline ancestral diagnostic across every selected offset.

        Unlike :meth:`sample_indices`, this intentionally includes the sparse
        tail.  It is used only by validation to measure exposure gaps and is not
        reachable from the closed-loop inference wrapper.
        """
        raw_trunk = hidden[:, -1]
        trunk = decoder_rmsnorm(raw_trunk)
        state_bias = self._state_bias(trunk)
        trunk_logits = self._trunk_skip_logits(raw_trunk)
        previous = observed
        caches: list[tuple[Tensor, Tensor] | None] = [None] * len(self.blocks)
        history = self._live_history(hidden, ctx_pad)
        conditioning = self.return_conditioner(return_value, condition_present)
        frames: list[Tensor] = []
        all_logits: list[dict[str, Tensor]] = []
        for offset in self.head_offsets:
            state, caches = self._decode_step(previous, offset, state_bias, caches, history, conditioning)
            embedded: dict[str, Tensor] = {}
            picks: dict[str, Tensor] = {}
            frame_logits: dict[str, Tensor] = {}
            for name in CONTROLLER_DECODE_ORDER:
                logits = self._center(
                    cast(NonlinearActionHead, self.outputs[name]).project(self.group_features(state, name, embedded))
                    + trunk_logits[name]
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
            raise ValueError("identity vocabulary does not match the frozen O59 contract")
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
        # V(s_t) predicts G_{t+1}, the return aligned with the next action. Keep
        # it last so the same seed preserves every policy parameter's draw.
        # Closed-loop inference never reads it.
        self.value_head = SwiGLU(cfg.arch.d_model, cfg.arch.value_hidden_dim, 1, output_bias=True)
        initialize_o51_parameters(self, cfg)
        with torch.random.fork_rng(devices=[]):
            self.temporal.return_conditioner = ReturnConditioner(cfg)
        self.return_calibration = ReturnCalibration()

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
            raise ValueError("every O59 window must expose at least 32 real suffix positions")
        if not validated_on_cpu and not bool((valid.sum(dim=1) >= POLICY_PREFIXES_PER_WINDOW).all()):
            raise ValueError("every O59 window must expose at least 32 real suffix positions")
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
    model: GPT, batch: TrainBatch | ReturnBatch
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
        validate_batch_geometry(cpu_batch, self._cfg, self._cfg.batch_size)
        return cpu_batch

    def _stage(self, cpu_batch: ReturnBatch) -> None:
        start = self._cfg.arch.direct_loss_start
        if bool((cpu_batch.context.ctx_pad > start).any()):
            raise ValueError("O59 requires all 128 suffix positions for dense value and AWR normalization")
        valid_prefixes = self._cfg.batch_size * POLICY_PREFIXES_PER_WINDOW
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
            self._calibration.observe(cpu_batch)
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
        raise ValueError("gradient attribution requires all O59 offsets")
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
        **_activation_input_metrics(f"{prefix}/gate", inputs),
        **_activation_output_metrics(f"{prefix}/gate", gate_output),
        **_activation_input_metrics(f"{prefix}/value", inputs),
        **_activation_output_metrics(f"{prefix}/value", value_output),
        **_activation_input_metrics(f"{prefix}/down", down_input),
        **_activation_output_metrics(f"{prefix}/down", output),
    }
    return output, metrics


def microbatch_loss(
    model: GPT,
    batch: ReturnBatch,
    cfg: TrainConfig,
    *,
    step: int,
    valid_prefixes: int,
    trunk_fn: Callable,
    temporal_fn: Callable,
    prefix_positions: Tensor,
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
    loss = policy_loss + cfg.awr.value_loss_weight * value_loss
    nll_sum = torch.where(valid[..., None, None], dense_nll.float(), 0).sum(dim=(0, 1))
    extra = {
        "train/loss": scoring.nats_to_bits(policy_loss.detach()),
        "train/near_loss": scoring.nats_to_bits(near.detach()),
        "train/far_nll": scoring.nats_to_bits(far.detach()),
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
    joint = scoring.nats_to_bits(mean_nll.sum(dim=-1))
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
def val_metrics(model: GPT, batches: list[ReturnBatch], cfg: TrainConfig) -> dict[str, float]:
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
            roll_nll = float(scoring.nats_to_bits(rollout_nll[depth, group] / denominator))
            teacher_nll = float(scoring.nats_to_bits(teacher_exposure_nll[depth, group] / denominator))
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
    if batch.context.slot_ids is not None or batch.context.reset is not None:
        raise ValueError("the fixed diagnostic batch must not contain closed-loop metadata")
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
            # Keep one inert position visible. A fully masked synthetic row is
            # undefined for the decoder's attention kernels.
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
    observation_counts = None
    if ctx.slot_ids is not None:
        slot_ids = torch.cat(
            (ctx.slot_ids, torch.full((extra,), -1, dtype=ctx.slot_ids.dtype, device=ctx.slot_ids.device))
        )
    if ctx.reset is not None:
        reset = torch.cat((ctx.reset, torch.ones(extra, dtype=ctx.reset.dtype, device=ctx.reset.device)))
    if ctx.observation_counts is not None:
        observation_counts = torch.cat(
            (
                ctx.observation_counts,
                torch.zeros(extra, dtype=ctx.observation_counts.dtype, device=ctx.observation_counts.device),
            )
        )
    return Context(
        features=features,
        ctx_pad=ctx_pad,
        slot_ids=slot_ids,
        reset=reset,
        observation_counts=observation_counts,
    )


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


@dataclass(frozen=True, slots=True)
class DecodedPlan:
    actions: Tensor
    indices: Tensor
    logits: tuple[Tensor, ...]
    uniforms: Tensor


@dataclass(frozen=True, slots=True)
class _PreparedDecode:
    rows: int
    bucket: int
    ctx_pad: Tensor
    hidden: Tensor
    observed: Tensor
    uniforms: Tensor
    forced_prefix: Tensor
    return_value: Tensor
    condition_present: Tensor
    observation_counts: Tensor | None
    reference_hidden: Tensor | None = None


@dataclass(slots=True)
class _RollingTrunkState:
    slot_ids: tuple[int, ...]
    observation_counts: tuple[int, ...]
    keys: Tensor
    values: Tensor
    hidden: Tensor
    ctx_pad: Tensor


class _CacheComparison:
    """Device-side aggregates for rolling-cache comparisons against recomputation."""

    def __init__(self) -> None:
        self._totals: dict[str, Tensor] = {}
        self._maxima: dict[str, Tensor] = {}

    def _add(self, name: str, value: Tensor) -> None:
        value = value.detach()
        self._totals[name] = self._totals.get(name, torch.zeros_like(value)) + value

    def _maximum(self, name: str, value: Tensor) -> None:
        value = value.detach()
        self._maxima[name] = torch.maximum(self._maxima.get(name, torch.zeros_like(value)), value)

    def record(
        self,
        hidden: Tensor,
        reference_hidden: Tensor,
        logits: tuple[Tensor, ...],
        reference_logits: tuple[Tensor, ...],
        indices: Tensor,
        reference_indices: Tensor,
        ctx_pad: Tensor,
        observation_counts: Tensor,
        rows: int,
    ) -> None:
        batch, length, width = hidden.shape
        real_rows = torch.arange(batch, device=hidden.device) < rows
        positions = torch.arange(length, device=hidden.device)
        real_positions = positions[None, :] >= ctx_pad[:, None]
        error = (hidden.float() - reference_hidden.float()).abs()
        hidden_float = hidden.float()
        reference_float = reference_hidden.float()
        for phase, selected_rows in (
            ("pre_eviction", real_rows & (observation_counts <= length)),
            ("post_eviction", real_rows & (observation_counts > length)),
            ("all", real_rows),
        ):
            mask = selected_rows[:, None] & real_positions
            weight = mask[:, :, None]
            prefix = f"cache_compare/{phase}"
            self._add(f"{prefix}/hidden_error_sq", (error.square() * weight).sum())
            self._add(f"{prefix}/hidden_reference_sq", (reference_float.square() * weight).sum())
            self._add(f"{prefix}/hidden_dot", (hidden_float * reference_float * weight).sum())
            self._add(f"{prefix}/hidden_norm_sq", (hidden_float.square() * weight).sum())
            self._add(f"{prefix}/hidden_values", mask.sum() * width)
            self._maximum(f"{prefix}/hidden_abs_max", torch.where(weight, error, 0.0).max())

            row_weight = selected_rows[:, None]
            sampled_mismatch = (indices != reference_indices).float()
            self._add(f"{prefix}/sampled_disagreements", (sampled_mismatch * row_weight[:, :, None]).sum())
            self._add(
                f"{prefix}/sampled_values",
                row_weight.sum() * indices.shape[1] * indices.shape[2],
            )
            for name, values, reference_values in zip(CONTROLLER_GROUP_NAMES, logits, reference_logits, strict=True):
                values = values.float()
                reference_values = reference_values.float()
                finite = torch.isfinite(values) & torch.isfinite(reference_values)
                logit_error = torch.where(finite, values - reference_values, 0.0)
                selected = row_weight[:, :, None] & finite
                group_prefix = f"{prefix}/{name}"
                self._add(f"{group_prefix}/logit_error_sq", (logit_error.square() * selected).sum())
                self._add(f"{group_prefix}/logit_values", selected.sum())
                probabilities = values.softmax(dim=-1)
                reference_probabilities = reference_values.softmax(dim=-1)
                tv = 0.5 * (probabilities - reference_probabilities).abs().sum(dim=-1)
                kl = (
                    reference_probabilities
                    * (reference_probabilities.clamp_min(1e-12).log() - probabilities.clamp_min(1e-12).log())
                ).sum(dim=-1)
                self._add(f"{group_prefix}/tv_sum", (tv * row_weight).sum())
                self._add(f"{group_prefix}/kl_sum", (kl * row_weight).sum())
                self._add(f"{group_prefix}/distributions", row_weight.sum())
                disagreement = (values.argmax(-1) != reference_values.argmax(-1)).float()
                self._add(f"{group_prefix}/argmax_disagreements", (disagreement * row_weight).sum())

    def metrics(self) -> dict[str, float]:
        totals = {name: float(value.item()) for name, value in self._totals.items()}
        metrics = {name: float(value.item()) for name, value in self._maxima.items()}
        for phase in ("pre_eviction", "post_eviction", "all"):
            prefix = f"cache_compare/{phase}"
            hidden_values = totals.get(f"{prefix}/hidden_values", 0.0)
            if hidden_values:
                error_sq = totals[f"{prefix}/hidden_error_sq"]
                reference_sq = totals[f"{prefix}/hidden_reference_sq"]
                hidden_sq = totals[f"{prefix}/hidden_norm_sq"]
                metrics[f"{prefix}/hidden_rms"] = math.sqrt(error_sq / hidden_values)
                metrics[f"{prefix}/hidden_relative_rms"] = math.sqrt(error_sq / max(reference_sq, 1e-30))
                metrics[f"{prefix}/hidden_cosine"] = totals[f"{prefix}/hidden_dot"] / math.sqrt(
                    max(reference_sq * hidden_sq, 1e-30)
                )
            sampled_values = totals.get(f"{prefix}/sampled_values", 0.0)
            if sampled_values:
                metrics[f"{prefix}/sampled_disagreement"] = totals[f"{prefix}/sampled_disagreements"] / sampled_values
            for name in CONTROLLER_GROUP_NAMES:
                group_prefix = f"{prefix}/{name}"
                logit_values = totals.get(f"{group_prefix}/logit_values", 0.0)
                distributions = totals.get(f"{group_prefix}/distributions", 0.0)
                if logit_values:
                    metrics[f"{group_prefix}/logit_rms"] = math.sqrt(
                        totals[f"{group_prefix}/logit_error_sq"] / logit_values
                    )
                if distributions:
                    metrics[f"{group_prefix}/tv"] = totals[f"{group_prefix}/tv_sum"] / distributions
                    metrics[f"{group_prefix}/kl"] = totals[f"{group_prefix}/kl_sum"] / distributions
                    metrics[f"{group_prefix}/argmax_disagreement"] = (
                        totals[f"{group_prefix}/argmax_disagreements"] / distributions
                    )
        return metrics


@triton.jit
def _write_kv_slot_kernel(
    keys: tl.tensor,
    values: tl.tensor,
    key: tl.tensor,
    value: tl.tensor,
    active: tl.tensor,
    slots: tl.tensor,
    layer: tl.constexpr,
    batch: tl.constexpr,
    heads: tl.constexpr,
    length: tl.constexpr,
    head_dim: tl.constexpr,
    key_batch_stride: tl.constexpr,
    key_head_stride: tl.constexpr,
    value_batch_stride: tl.constexpr,
    value_head_stride: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offset = tl.program_id(1) * block_size + tl.arange(0, block_size)
    enabled = tl.load(active + row)
    slot = tl.load(slots + row)
    head = offset // head_dim
    channel = offset % head_dim
    mask = (offset < heads * head_dim) & enabled
    destination = ((layer * batch + row) * heads + head) * length * head_dim + slot * head_dim + channel
    next_key = tl.load(key + row * key_batch_stride + head * key_head_stride + channel, mask, other=0)
    next_value = tl.load(value + row * value_batch_stride + head * value_head_stride + channel, mask, other=0)
    tl.store(keys + destination, next_key, mask)
    tl.store(values + destination, next_value, mask)


@torch.library.triton_op("hal_o59::write_kv_slot", mutates_args={"keys", "values"})
def _write_kv_slot(
    keys: Tensor,
    values: Tensor,
    key: Tensor,
    value: Tensor,
    active: Tensor,
    slots: Tensor,
    layer: int,
) -> None:
    # PyTorch 2.11 turns native indexed assignments into full-cache temporaries.
    # Keep this operation until native writes pass the compiled storage tests and
    # generated-code inspection without those copies.
    if torch.__version__.split("+")[0] != "2.11.0" or triton.__version__ != "3.6.0":
        raise RuntimeError("O59 CUDA ring writes require validated PyTorch 2.11.0 and Triton 3.6.0")
    _, batch, heads, length, head_dim = keys.shape
    torch.library.wrap_triton(_write_kv_slot_kernel)[(batch, triton.cdiv(heads * head_dim, 256))](
        keys,
        values,
        key,
        value,
        active,
        slots,
        layer,
        batch,
        heads,
        length,
        head_dim,
        key.stride(0),
        key.stride(1),
        value.stride(0),
        value.stride(1),
        256,
    )


def _rolling_trunk_forward(
    model: GPT,
    tokens: Tensor,
    keys: Tensor,
    values: Tensor,
    hidden: Tensor,
    ctx_pad: Tensor,
    active_steps: Tensor,
    token_positions: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Mutate ring K/V slots; retain chronological final hidden states for the decoder."""
    batch, steps, width = tokens.shape
    layers, cache_batch, heads, length, head_dim = keys.shape
    if (
        cache_batch != batch
        or values.shape != keys.shape
        or hidden.shape != (batch, length, width)
        or ctx_pad.shape != (batch,)
        or active_steps.shape != (steps, batch)
        or token_positions.shape != (steps, batch)
        or layers != len(model.trunk.blocks)
        or not keys.is_contiguous()
        or not values.is_contiguous()
    ):
        raise ValueError("rolling trunk inputs have incompatible shapes")

    first_attention = cast(Any, model.trunk.blocks[0]).attn
    positions = torch.arange(length, device=tokens.device)
    rows = torch.arange(batch, device=tokens.device)
    for step in range(steps):
        active = active_steps[step]
        active_token = active[:, None, None]
        next_pad = torch.where(active, (ctx_pad - 1).clamp_min(0), ctx_pad)
        x = tokens[:, step : step + 1]
        inv_freq = first_attention.rotary.inv_freq
        if inv_freq.dtype != torch.float32:
            inv_freq = 1.0 / (
                first_attention.rotary.base ** (torch.arange(0, head_dim, 2, device=tokens.device).float() / head_dim)
            )
        frequencies = token_positions[step, :, None].float() * inv_freq[None]
        cos = frequencies.cos().to(tokens.dtype)[:, None, None]
        sin = frequencies.sin().to(tokens.dtype)[:, None, None]
        slots = token_positions[step].remainder(length)
        for layer, module in enumerate(model.trunk.blocks):
            block = cast(Any, module)
            attention = block.attn
            normalized = rmsnorm(x)
            query, key, value = attention.c_attn(normalized).split(attention.d_model, dim=-1)
            query = query.view(batch, 1, heads, head_dim)
            key = key.view(batch, 1, heads, head_dim)
            value = value.view(batch, 1, heads, head_dim).transpose(1, 2)
            query = apply_rotary_emb(query, cos, sin).transpose(1, 2)
            key = apply_rotary_emb(key, cos, sin).transpose(1, 2)
            if keys.dtype != key.dtype or values.dtype != value.dtype:
                raise ValueError("rolling K/V storage must match the attention dtype")
            # Only the incoming slot is written. Physical order does not affect
            # attention: keys already carry their absolute RoPE positions.
            if keys.is_cuda:
                _write_kv_slot(keys, values, key, value, active, slots, layer)
            else:
                keys[layer, rows, :, slots, :] = torch.where(
                    active[:, None, None], key[:, :, 0], keys[layer, rows, :, slots, :]
                )
                values[layer, rows, :, slots, :] = torch.where(
                    active[:, None, None], value[:, :, 0], values[layer, rows, :, slots, :]
                )
            valid = positions[None, :] < (length - next_pad)[:, None]
            attended = F.scaled_dot_product_attention(
                query,
                keys[layer],
                values[layer],
                attn_mask=valid[:, None, None],
            )
            attended = attended.transpose(1, 2).contiguous().view(batch, 1, width)
            candidate = x + block.attn_scale * attention.c_proj(attended)
            candidate = candidate + block.mlp_scale * block.mlp(rmsnorm(candidate))
            x = torch.where(active_token, candidate, x)
        final = rmsnorm(x)
        hidden = torch.where(
            active_token,
            torch.cat((hidden[:, 1:], final), dim=1),
            hidden,
        )
        ctx_pad = next_pad
    return keys, values, hidden, ctx_pad


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
        temperature: float = 1.0,
        desired_return: float | None = None,
        cache_mode: Literal["recompute", "rolling"] = "recompute",
        compare_recompute: bool = False,
    ) -> None:
        if desired_return is not None and (not math.isfinite(desired_return) or not cfg.return_conditioning):
            raise ValueError("desired return requires finite input and enabled conditioning")
        self.desired_return = desired_return
        self.model = model
        self.cfg = cfg
        if cache_mode not in ("recompute", "rolling"):
            raise ValueError(f"unknown cache mode {cache_mode!r}")
        if compare_recompute and cache_mode != "rolling":
            raise ValueError("recomputation comparison requires rolling cache mode")
        self.cache_mode = cache_mode
        self.compare_recompute = compare_recompute
        self.cache_comparison = _CacheComparison() if compare_recompute else None
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
        self.temperature = validate_sampling_temperature(temperature)
        self.attention_backend = "dense_sdpa" if cache_mode == "recompute" else "rolling_kv"
        self.compile_seconds = 0.0
        self._warmed: set[tuple[int, int, int]] = set()
        self._trunks: dict[int, Callable] = {}
        self._rolling_trunks: dict[tuple[int, int], Callable] = {}
        self._decoders: dict[tuple[int, int, int], Callable] = {}
        self._trace_decoders: dict[tuple[int, int, int], Callable] = {}
        self._rolling_state: _RollingTrunkState | None = None

    @property
    def uses_cuda_graphs(self) -> bool:
        return self.compiled and self.compile_mode == "reduce-overhead"

    @property
    def rolling_cache_bytes(self) -> int:
        if self._rolling_state is None:
            return 0
        state = self._rolling_state
        return sum(tensor.numel() * tensor.element_size() for tensor in (state.keys, state.values, state.hidden))

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

    def _rolling_trunk(self, bucket: int, steps: int) -> Callable:
        key = (bucket, steps)
        if key not in self._rolling_trunks:

            def forward(
                features: dict[str, Tensor],
                action_indices: Tensor,
                keys: Tensor,
                values: Tensor,
                hidden: Tensor,
                ctx_pad: Tensor,
                active_steps: Tensor,
                token_positions: Tensor,
            ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
                tokens = self.model.context_tokens(features, action_indices)
                return _rolling_trunk_forward(
                    self.model,
                    tokens,
                    keys,
                    values,
                    hidden,
                    ctx_pad,
                    active_steps,
                    token_positions,
                )

            self._rolling_trunks[key] = (
                torch.compile(forward, dynamic=False, fullgraph=True, mode=self.compile_mode)
                if self.compiled
                else forward
            )
        return self._rolling_trunks[key]

    def _initial_rolling_state(self, bucket: int, reference: Tensor) -> _RollingTrunkState:
        architecture = self.cfg.arch
        shape = (
            architecture.n_layers,
            bucket,
            architecture.n_heads,
            architecture.L_ctx,
            architecture.d_model // architecture.n_heads,
        )
        device_type = reference.device.type
        dtype = (
            torch.get_autocast_dtype(device_type)
            if torch.is_autocast_enabled(device_type)
            else next(self.model.parameters()).dtype
        )
        return _RollingTrunkState(
            slot_ids=tuple([-1] * bucket),
            observation_counts=tuple([0] * bucket),
            keys=torch.zeros(shape, dtype=dtype, device=reference.device),
            values=torch.zeros(shape, dtype=dtype, device=reference.device),
            hidden=torch.zeros(
                bucket,
                architecture.L_ctx,
                architecture.d_model,
                dtype=dtype,
                device=reference.device,
            ),
            ctx_pad=torch.full((bucket,), architecture.L_ctx, dtype=torch.long, device=reference.device),
        )

    def _align_rolling_state(
        self,
        slot_ids: tuple[int, ...],
        resets: tuple[bool, ...],
        observation_counts: tuple[int, ...],
        reference: Tensor,
    ) -> tuple[_RollingTrunkState, tuple[int, ...]]:
        bucket = len(slot_ids)
        state = self._rolling_state
        if state is None:
            state = self._initial_rolling_state(bucket, reference)
        elif state.keys.shape[1] != bucket:
            raise ValueError("rolling cache bucket changed after initialization")

        old_rows = {slot: row for row, slot in enumerate(state.slot_ids) if slot >= 0}
        source_rows: list[int] = []
        new_counts: list[int] = []
        for row, (slot, reset, count) in enumerate(zip(slot_ids, resets, observation_counts, strict=True)):
            if slot < 0:
                source_rows.append(row)
                new_counts.append(0)
                continue
            old_row = old_rows.get(slot)
            if old_row is None and not reset:
                raise ValueError(f"rolling cache has no state for continuing stream {slot}")
            source_rows.append(0 if old_row is None else old_row)
            previous = 0 if reset or old_row is None else state.observation_counts[old_row]
            delta = count - previous
            if delta <= 0 or delta > self.cfg.arch.L_ctx:
                raise ValueError(f"stream {slot} advanced by {delta} observations; expected 1..{self.cfg.arch.L_ctx}")
            new_counts.append(delta)

        if tuple(source_rows) != tuple(range(bucket)):
            index = torch.tensor(source_rows, dtype=torch.long, device=reference.device)
            keys = state.keys.index_select(1, index)
            values = state.values.index_select(1, index)
            hidden = state.hidden.index_select(0, index)
            ctx_pad = state.ctx_pad.index_select(0, index)
        else:
            keys, values, hidden, ctx_pad = state.keys, state.values, state.hidden, state.ctx_pad
        reset_tensor = torch.tensor(resets, dtype=torch.bool, device=reference.device)
        ctx_pad = torch.where(reset_tensor, self.cfg.arch.L_ctx, ctx_pad)
        return (
            _RollingTrunkState(
                slot_ids=slot_ids,
                observation_counts=observation_counts,
                keys=keys,
                values=values,
                hidden=hidden,
                ctx_pad=ctx_pad,
            ),
            tuple(new_counts),
        )

    def _rolling_hidden(self, ctx: Context, action_indices: Tensor) -> Tensor:
        if ctx.slot_ids is None or ctx.reset is None or ctx.observation_counts is None:
            raise ValueError("rolling inference requires slot ids, reset flags, and observation counts")
        slot_ids = tuple(int(value) for value in ctx.slot_ids.detach().cpu().tolist())
        resets = tuple(bool(value) for value in ctx.reset.detach().cpu().tolist())
        counts = tuple(int(value) for value in ctx.observation_counts.detach().cpu().tolist())
        if len({slot for slot in slot_ids if slot >= 0}) != sum(slot >= 0 for slot in slot_ids):
            raise ValueError("rolling inference received duplicate stream ids")
        expected_pad = tuple(
            self.cfg.arch.L_ctx - min(count, self.cfg.arch.L_ctx) if slot >= 0 else self.cfg.arch.L_ctx - 1
            for slot, count in zip(slot_ids, counts, strict=True)
        )
        actual_pad = tuple(int(value) for value in ctx.ctx_pad.detach().cpu().tolist())
        if actual_pad != expected_pad:
            raise ValueError(f"context padding does not match observation counts: {actual_pad} != {expected_pad}")

        state, new_counts = self._align_rolling_state(slot_ids, resets, counts, action_indices)
        steps = max(new_counts)
        suffix_features = {name: value[:, -steps:] for name, value in ctx.features.items()}
        suffix_actions = action_indices[:, -steps:]
        active_steps = torch.tensor(
            [[offset >= steps - count for count in new_counts] for offset in range(steps)],
            dtype=torch.bool,
            device=action_indices.device,
        )
        token_positions = torch.tensor(
            [[max(0, count - steps + offset) for count in counts] for offset in range(steps)],
            dtype=torch.long,
            device=action_indices.device,
        )
        keys, values, hidden, ctx_pad = self._rolling_trunk(len(slot_ids), steps)(
            suffix_features,
            suffix_actions,
            state.keys,
            state.values,
            state.hidden,
            state.ctx_pad,
            active_steps,
            token_positions,
        )
        self._rolling_state = _RollingTrunkState(
            slot_ids=slot_ids,
            observation_counts=counts,
            keys=keys,
            values=values,
            hidden=hidden,
            ctx_pad=ctx_pad,
        )
        return hidden

    def _decoder(self, bucket: int, horizon: int, committed_frames: int) -> Callable:
        key = (bucket, horizon, committed_frames)
        if key not in self._decoders:
            offsets = self.model.head_offsets[:horizon]

            def fn(
                hidden: Tensor,
                ctx_pad: Tensor,
                observed: Tensor,
                uniforms: Tensor,
                forced_prefix: Tensor,
                return_value: Tensor,
                condition_present: Tensor,
            ):
                return self.model.temporal.sample_indices(
                    hidden,
                    observed,
                    offsets,
                    return_value,
                    condition_present,
                    argmax=False,
                    uniforms=uniforms,
                    temperature=self.temperature,
                    ctx_pad=ctx_pad,
                    forced_prefix=forced_prefix,
                )

            self._decoders[key] = torch.compile(fn, dynamic=False, mode=self.compile_mode) if self.compiled else fn
        return self._decoders[key]

    def _trace_decoder(self, bucket: int, horizon: int, committed_frames: int) -> Callable:
        key = (bucket, horizon, committed_frames)
        if key not in self._trace_decoders:
            offsets = self.model.head_offsets[:horizon]

            def fn(
                hidden: Tensor,
                ctx_pad: Tensor,
                observed: Tensor,
                uniforms: Tensor,
                forced_prefix: Tensor,
                return_value: Tensor,
                condition_present: Tensor,
            ):
                return self.model.temporal.sample_indices_with_logits(
                    hidden,
                    observed,
                    offsets,
                    return_value,
                    condition_present,
                    argmax=False,
                    uniforms=uniforms,
                    temperature=self.temperature,
                    ctx_pad=ctx_pad,
                    forced_prefix=forced_prefix,
                )

            self._trace_decoders[key] = (
                torch.compile(fn, dynamic=False, mode=self.compile_mode) if self.compiled else fn
            )
        return self._trace_decoders[key]

    def _prepare_decode(
        self,
        ctx: Context,
        horizon: int,
        *,
        streams: SlotGroupRng | None,
        gen: torch.Generator | None,
        committed: Tensor | None,
    ) -> _PreparedDecode:
        if horizon != self.cfg.prediction_frames:
            raise ValueError(f"horizon must be {self.cfg.prediction_frames}")
        rows = ctx.ctx_pad.shape[0]
        if committed is not None and (
            committed.ndim != 3
            or committed.shape[0] != rows
            or committed.shape[2] != len(ACTION_CHANNELS)
            or committed.shape[1] > horizon
        ):
            raise ValueError("committed actions have the wrong shape")
        committed_frames = 0 if committed is None else committed.shape[1]
        bucket = self._bucket(rows)
        padded = _pad_context(ctx, bucket)
        if "ego_player_id" not in padded.features:
            padded = _condition_ego_player(padded, MASKED_PLAYER_ID)
        padded = canonical_context(padded, "base", items=True)
        observed = self.model.codec.quantize(stack_actions(padded.features))
        uniform_parts: list[Tensor] = []
        if streams is not None:
            streams.begin(ctx)
        for frame in range(horizon):
            groups = []
            for name in CONTROLLER_GROUP_NAMES:
                if frame < committed_frames:
                    real = torch.full((rows,), 0.5, device=ctx.ctx_pad.device)
                elif streams is None:
                    real = torch.rand(rows, device=ctx.ctx_pad.device, generator=gen)
                else:
                    real = streams.uniforms(name)
                groups.append(F.pad(real, (0, bucket - rows), value=0.5))
            uniform_parts.append(torch.stack(groups))
        uniforms = torch.stack(uniform_parts)
        if self.uses_cuda_graphs:
            # The trunk and decoder share one graph step so the next trunk replay
            # can reuse its output storage only after the decoder consumes it.
            torch.compiler.cudagraph_mark_step_begin()
        with amp_context(self.cfg, ctx.ctx_pad.device):
            reference_hidden = None
            if self.cache_mode == "rolling":
                hidden = self._rolling_hidden(padded, observed)
                if self.compare_recompute:
                    reference_hidden = self._trunk(bucket)(padded.features, padded.ctx_pad, observed)
            else:
                hidden = self._trunk(bucket)(padded.features, padded.ctx_pad, observed)
            if committed is None:
                forced_prefix = torch.empty(bucket, 0, CONTROLLER_GROUP_COUNT, dtype=torch.long, device=hidden.device)
            else:
                padded_committed = F.pad(committed, (0, 0, 0, 0, 0, bucket - rows))
                forced_prefix = self.model.codec.quantize(padded_committed)
        return_value = torch.full(
            (bucket,), 0.0 if self.desired_return is None else self.desired_return, device=hidden.device
        )
        condition_present = (torch.arange(bucket, device=hidden.device) < rows) & (self.desired_return is not None)
        return _PreparedDecode(
            rows,
            bucket,
            padded.ctx_pad,
            hidden,
            observed[:, -1],
            uniforms,
            forced_prefix,
            return_value,
            condition_present,
            padded.observation_counts,
            reference_hidden,
        )

    @torch.no_grad()
    def prewarm(self, rows: int, horizon: int, *, committed_frames: int) -> float:
        """Compile and replay the exact evaluation program before Dolphin starts."""
        if horizon != self.cfg.prediction_frames or not 0 <= committed_frames <= horizon:
            raise ValueError("invalid prewarm horizon or committed-prefix length")
        bucket = self._bucket(rows)
        key = (bucket, horizon, committed_frames)
        if key in self._warmed or not self.compiled:
            self._warmed.add(key)
            return 0.0
        device = next(self.model.parameters()).device
        started = time.perf_counter()
        context = synthetic_context(self.cfg, rows, device)
        neutral = torch.as_tensor(NEUTRAL_ACTION, device=device).expand(rows, committed_frames, -1)
        if self.cache_mode == "rolling":
            count = torch.ones(rows, dtype=torch.long, device=device)
            reset_context = replace(
                context,
                ctx_pad=torch.full_like(context.ctx_pad, self.cfg.arch.L_ctx - 1),
                observation_counts=count,
            )
            continuing_context = replace(
                reset_context,
                ctx_pad=torch.full_like(context.ctx_pad, self.cfg.arch.L_ctx - 3),
                reset=torch.zeros(rows, dtype=torch.bool, device=device),
                observation_counts=count + 2,
            )
            self.decode(reset_context, horizon, committed=neutral)
            self.decode(reset_context, horizon, committed=neutral)
            self.decode(continuing_context, horizon, committed=neutral)
            self.decode(reset_context, horizon, committed=neutral)
            self.decode(continuing_context, horizon, committed=neutral)
        else:
            self.decode(context, horizon, committed=neutral)
            self.decode(context, horizon, committed=neutral)
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
        committed: Tensor | None = None,
    ) -> Tensor:
        prepared = self._prepare_decode(ctx, horizon, streams=streams, gen=gen, committed=committed)
        with amp_context(self.cfg, ctx.ctx_pad.device):
            if argmax:
                if self.cache_comparison is not None:
                    raise ValueError("cache comparison requires sampled decoding")
                indices = self.model.temporal.sample_indices(
                    prepared.hidden,
                    prepared.observed,
                    self.model.head_offsets[:horizon],
                    prepared.return_value,
                    prepared.condition_present,
                    argmax=True,
                    temperature=self.temperature,
                    ctx_pad=prepared.ctx_pad,
                    forced_prefix=prepared.forced_prefix,
                )
            elif self.cache_comparison is None:
                indices = self._decoder(prepared.bucket, horizon, prepared.forced_prefix.shape[1])(
                    prepared.hidden,
                    prepared.ctx_pad,
                    prepared.observed,
                    prepared.uniforms,
                    prepared.forced_prefix,
                    prepared.return_value,
                    prepared.condition_present,
                )
            else:
                if prepared.reference_hidden is None or prepared.observation_counts is None:
                    raise RuntimeError("cache comparison is missing its recomputed reference")
                decoder = self._trace_decoder(prepared.bucket, horizon, prepared.forced_prefix.shape[1])
                indices, logits = decoder(
                    prepared.hidden,
                    prepared.ctx_pad,
                    prepared.observed,
                    prepared.uniforms,
                    prepared.forced_prefix,
                    prepared.return_value,
                    prepared.condition_present,
                )
                indices = indices.clone()
                logits = tuple(values.clone() for values in logits)
                reference_indices, reference_logits = decoder(
                    prepared.reference_hidden,
                    prepared.ctx_pad,
                    prepared.observed,
                    prepared.uniforms,
                    prepared.forced_prefix,
                    prepared.return_value,
                    prepared.condition_present,
                )
                self.cache_comparison.record(
                    prepared.hidden,
                    prepared.reference_hidden,
                    logits,
                    reference_logits,
                    indices,
                    reference_indices,
                    prepared.ctx_pad,
                    prepared.observation_counts,
                    prepared.rows,
                )
        return self.model.codec.dequantize(indices[: prepared.rows])

    @torch.no_grad()
    def decode_with_trace(
        self,
        ctx: Context,
        horizon: int,
        *,
        streams: SlotGroupRng,
        committed: Tensor | None = None,
    ) -> DecodedPlan:
        """Decode once and retain the exact logits and uniforms used to sample."""
        prepared = self._prepare_decode(ctx, horizon, streams=streams, gen=None, committed=committed)
        with amp_context(self.cfg, ctx.ctx_pad.device):
            indices, logits = self._trace_decoder(prepared.bucket, horizon, prepared.forced_prefix.shape[1])(
                prepared.hidden,
                prepared.ctx_pad,
                prepared.observed,
                prepared.uniforms,
                prepared.forced_prefix,
                prepared.return_value,
                prepared.condition_present,
            )
        real_indices = indices[: prepared.rows]
        return DecodedPlan(
            actions=self.model.codec.dequantize(real_indices),
            indices=real_indices,
            logits=tuple(values[: prepared.rows] for values in logits),
            uniforms=prepared.uniforms[:, :, : prepared.rows],
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
    decode_temperature: float = 1.0,
    action_trace: ActionTraceWriter | None = None,
    device: str = DEVICE,
) -> RecedingHorizon:
    horizon = cfg.prediction_frames
    delay = cfg.delay_frames if delay_frames is None else delay_frames
    replan = cfg.replan_interval_frames if replan_interval_frames is None else replan_interval_frames
    _validate_deployment_timing(horizon, delay, replan)
    temperature = validate_sampling_temperature(decode_temperature)
    engine = BF16Inference(model, cfg, temperature=temperature) if inference is None else inference
    if engine.temperature != temperature:
        raise ValueError(f"inference temperature {engine.temperature} does not match policy temperature {temperature}")
    random_streams = None if decode_seed is None else SlotGroupRng(decode_seed, CONTROLLER_GROUP_NAMES)
    generator = None if decode_seed is None else torch.Generator(device=device).manual_seed(decode_seed)
    if action_trace is not None and (decode_seed is None or random_streams is None):
        raise ValueError("action tracing requires a fixed decode_seed")

    @torch.no_grad()
    def predict(ctx: Context, committed: np.ndarray | None) -> np.ndarray:
        if committed is None:
            raise ValueError("O59 live decoding requires the committed-action prefix")
        started = time.perf_counter()
        conditioned = _condition_ego_player(ctx, ego_player_id)
        committed_tensor = torch.as_tensor(committed, device=device, dtype=next(model.parameters()).dtype)
        if action_trace is None:
            result = engine.decode(
                conditioned,
                horizon,
                streams=random_streams,
                gen=generator,
                committed=committed_tensor,
            )
        else:
            if ctx.slot_ids is None:
                raise ValueError("action tracing requires evaluation slot_ids")
            if decode_seed is None or random_streams is None:
                raise RuntimeError("validated action-trace sampling state is missing")
            decoded = engine.decode_with_trace(
                conditioned, horizon, streams=random_streams, committed=committed_tensor
            )
            action_trace.record_plan(
                decode_seed=decode_seed,
                slot_ids=ctx.slot_ids,
                resets=ctx.reset,
                indices=decoded.indices,
                logits=decoded.logits,
                uniforms=decoded.uniforms,
                head_offsets=model.head_offsets[:horizon],
                temperature=engine.temperature,
                delay_frames=delay,
                replan_interval_frames=replan,
            )
            result = decoded.actions
        result_array = result.cpu().numpy()
        if telemetry is not None:
            telemetry.record(rows=ctx.ctx_pad.shape[0], horizon=horizon, seconds=time.perf_counter() - started)
        return result_array

    return RecedingHorizon(
        predict_chunk=predict,
        stats=stats,
        L_ctx=cfg.arch.L_ctx,
        L_chunk=horizon,
        s=replan,
        d=delay,
        bootstrap_committed="neutral",
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
    transport_semantics: Literal["conditioned_pending_actions_v1"]
    pending_prefix_conditioned: Literal[True]
    evaluation_protocol_version: Literal[4]
    forced_prefix_consumes_sampling_draws: Literal[False]
    dtype: str
    inference_mode: str
    inference_compile_mode: str
    inference_attention_backend: str
    cache_mode: Literal["recompute", "rolling"]
    compare_recompute: bool
    compiled_inference_bucket: int
    checkpoint_sha256: str
    return_target: Literal["p90", "unconditioned"]
    desired_return: float | None
    return_calibration_sha256: str | None
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
    cache_mode: Literal["recompute", "rolling"] = "recompute",
    compare_recompute: bool = False,
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
        evaluation_protocol_version=4,
        forced_prefix_consumes_sampling_draws=False,
        dtype=str(next(model.parameters()).dtype),
        inference_mode=cfg.inference_mode if inference_mode is None else inference_mode,
        inference_compile_mode=inference_compile_mode,
        inference_attention_backend=inference_attention_backend,
        cache_mode=cache_mode,
        compare_recompute=compare_recompute,
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
        or previous.get("schema_version") != 8
        or previous.get("protocol") != asdict(protocol)
    ):
        raise ValueError("evaluation directory contains evidence from a different protocol")


def _write_eval_evidence(
    replay_dir: Path, rows: list[MatchRow], metrics: dict[str, float], protocol: EvalProtocol
) -> None:
    replay_dir.mkdir(parents=True, exist_ok=True)
    rows_payload = {
        "schema_version": 8,
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
    desired_return: float | None = None,
    return_calibration_sha256: str | None = None,
    cache_mode: Literal["recompute", "rolling"] = "recompute",
    compare_recompute: bool = False,
) -> dict[str, float]:
    horizon = cfg.prediction_frames
    inference_mode = "eager" if eager else cfg.inference_mode
    inference = (
        BF16Inference(
            model,
            cfg,
            bucket=_eval_inference_bucket(cfg, n_matchups, max_parallel),
            compiled=inference_mode == "compiled",
            desired_return=desired_return,
            cache_mode=cache_mode,
            compare_recompute=compare_recompute,
        )
        if inference is None
        else inference
    )
    if inference.desired_return != desired_return:
        raise ValueError("inference desired return differs from evaluation protocol")
    if inference.model is not model:
        raise ValueError("the supplied inference engine must own the evaluation model")
    if inference.cache_mode != cache_mode:
        raise ValueError("inference cache mode differs from evaluation protocol")
    if inference.compare_recompute != compare_recompute:
        raise ValueError("inference comparison mode differs from evaluation protocol")
    protocol = _eval_protocol(
        cfg,
        model,
        n_matchups=n_matchups,
        checkpoint_sha256=checkpoint_sha256,
        max_parallel=max_parallel,
        inference_mode=inference_mode,
        inference_compile_mode=inference.compile_mode,
        inference_attention_backend=inference.attention_backend,
        cache_mode=cache_mode,
        compare_recompute=compare_recompute,
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
    policy_index = itertools.count()
    process_telemetry = ProcessVecTelemetry()

    def factory() -> RecedingHorizon:
        return make_policy(
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

    was_training = model.training
    model.eval()
    total_started = time.perf_counter()
    try:
        compile_seconds = inference.prewarm(protocol.max_parallel, horizon, committed_frames=protocol.delay_frames)
        if next(model.parameters()).device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(next(model.parameters()).device)
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
    mean_decode_seconds = decode_metrics["decode_seconds"] / max(decode_metrics["decode_calls"], 1.0)
    decode_metrics["decode_stream_fps_capacity"] = protocol.replan_interval_frames / max(mean_decode_seconds, 1e-12)
    decode_metrics["decode_aggregate_emulator_fps_capacity"] = (
        decode_metrics["decode_rows"] * protocol.replan_interval_frames / max(decode_metrics["decode_seconds"], 1e-12)
    )
    decode_metrics["decode_p95_60fps_margin_ms"] = (
        1000.0 * protocol.replan_interval_frames / 60.0 - decode_metrics["decode_p95_ms"]
    )
    metrics.update(decode_metrics)
    metrics.update(process_telemetry.metrics())
    metrics["rolling_cache_bytes"] = float(inference.rolling_cache_bytes)
    if next(model.parameters()).device.type == "cuda":
        device = next(model.parameters()).device
        metrics["peak_cuda_allocated_bytes"] = float(torch.cuda.max_memory_allocated(device))
        metrics["peak_cuda_reserved_bytes"] = float(torch.cuda.max_memory_reserved(device))
    if inference.cache_comparison is not None:
        metrics.update(inference.cache_comparison.metrics())
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
        "decode_stream_fps_capacity": "decode_stream_fps_capacity",
        "decode_aggregate_emulator_fps_capacity": "decode_aggregate_emulator_fps_capacity",
        "decode_p95_60fps_margin_ms": "decode_p95_60fps_margin_ms",
        "rolling_cache_bytes": "rolling_cache_bytes",
        "peak_cuda_allocated_bytes": "peak_cuda_allocated_bytes",
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


def lr_schedule(cfg: TrainConfig):
    """Return warmup/stable/decay WSD independent of the stopping update."""

    def schedule(step: int) -> float:
        update = step + 1
        if update <= cfg.warmup_steps:
            return update / cfg.warmup_steps
        if cfg.decay_start_update is None or update <= cfg.decay_start_update:
            return 1.0
        if cfg.decay_duration is None:
            raise RuntimeError("decay duration is missing")
        progress = min((update - cfg.decay_start_update) / cfg.decay_duration, 1.0)
        return 1.0 + progress * (cfg.lr_floor_ratio - 1.0)

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
    trunk_skip = tuple(
        (cast(NonlinearActionHead, model.temporal.trunk_outputs[name]).down, 128) for name in CONTROLLER_GROUP_NAMES
    )
    return (*action, *trunk_skip, (model.value_head.down, 128))


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
            raise RuntimeError(f"O59 has no optimizer role for {name} {tuple(parameter.shape)}")
    return roles


def _role_lr(role: OptimizerRole, cfg: TrainConfig) -> float:
    if role.optimizer == "muon":
        return cfg.muon_lr * cfg.muon_lr_multiplier
    batch_multiplier, duration_multiplier = scaling_multipliers(cfg)
    lr = cfg.adam_lr * math.sqrt(batch_multiplier / duration_multiplier)
    return lr / role.fan_in_multiplier if role.lr_kind == "output" else lr


def make_optimizer(model: GPT, cfg: TrainConfig) -> torch.optim.Optimizer:
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


def parameter_subsystems(model: GPT) -> dict[str, tuple[nn.Parameter, ...]]:
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


def subsystem_parameter_counts(model: GPT) -> dict[str, int]:
    partitions = parameter_subsystems(model)
    counts = {name: sum(parameter.numel() for parameter in parameters) for name, parameters in partitions.items()}
    counts["total"] = sum(parameter.numel() for parameter in model.parameters())
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
    """Estimate forward-backward FLOPs from each subsystem's parameter uses."""
    full = cfg.arch.L_ctx
    suffix = full - cfg.arch.direct_loss_start
    policy_prefixes = POLICY_PREFIXES_PER_WINDOW
    trunk_and_inputs = parameter_counts["trunk"] + parameter_counts["other"]
    temporal_and_heads = parameter_counts["temporal_decoder"] + parameter_counts["group_heads"]
    parameter_uses = (
        full * trunk_and_inputs
        + suffix * parameter_counts["value_head"]
        + policy_prefixes * (parameter_counts["trunk_skip_heads"] + parameter_counts["return_conditioner"])
        + policy_prefixes * len(cfg.arch.head_offsets) * temporal_and_heads
    )
    return 6 * cfg.batch_size * parameter_uses


def model_tag(cfg: TrainConfig) -> str:
    return (
        f"o59v5-d{cfg.arch.d_model}-L{cfg.arch.n_layers}-h{cfg.arch.n_heads}-c{cfg.arch.L_ctx}-"
        f"t{cfg.arch.temporal_d_model}x{cfg.arch.temporal_layers}-ff{cfg.arch.temporal_ff_dim}-"
        f"dual-nonlinear-head-rc{int(cfg.return_conditioning)}e{cfg.arch.return_embed_dim}-drop{cfg.return_dropout:g}-"
        f"{cfg.optimizer}-wsd-b{cfg.awr.beta:g}-wmax{cfg.awr.weight_max:g}-g{cfg.awr.gamma:g}"
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
    """Return all policy-world-v8 train rows with their pinned identity."""
    if tuple(cfg.source_names) != tuple(source.name for source in streams.POLICY_WORLD_V8_SOURCES):
        raise ValueError("O59 selection requires all policy-world-v8 sources in registry order")
    sources = tuple(SourceRowSelection(name, streams.POLICY_WORLD_V8_TRAIN_REPLAYS[name]) for name in cfg.source_names)
    selection = PhysicalShardSelection.from_sources(sources)
    if selection.sha256 != cfg.selection_sha256:
        raise RuntimeError(f"policy-world-v8 selection hash changed: {selection.sha256} != {cfg.selection_sha256}")
    if selection.row_count != cfg.train_replays:
        raise RuntimeError(f"policy-world-v8 selection has {selection.row_count} rows, expected {cfg.train_replays}")
    return selection


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


def save_boundary_checkpoint(
    run_dir: Path,
    *,
    update: int,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    cfg: TrainConfig,
    uploader: BackgroundUploader | None,
    milestone: bool,
    wandb_id: str | None,
    actual_loss_positions: int,
    loader_state: dict[str, object],
    return_masker_state: dict[str, object],
    identity_masker_state: dict[str, object] | None = None,
    prefix_sampler_state: dict[str, object] | None = None,
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
            "actual_policy_prefixes": update * cfg.policy_prefixes_per_update,
            "actual_value_prefixes": update * cfg.value_prefixes_per_update,
            "schedule_phase": (
                "warmup"
                if update <= cfg.warmup_steps
                else "stable"
                if cfg.decay_start_update is None or update <= cfg.decay_start_update
                else "decay"
            ),
            "rng": rng_state(),
            "conditioning_protocol": conditioning_protocol(),
            "return_masker": return_masker_state,
            "return_calibration": model.return_calibration.state_dict(),
            "provenance": run_provenance(cfg),
            "loader": loader_state,
            "identity_masker": (
                loader_state.get("identity_masker") if identity_masker_state is None else identity_masker_state
            ),
            "prefix_sampler": prefix_sampler_state,
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
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: Mapping[str, object]) -> None:
    random.setstate(cast(tuple, state["python"]))
    np.random.set_state(cast(tuple, state["numpy"]))
    torch.set_rng_state(cast(Tensor, state["torch"]).cpu())
    cuda = cast(list[Tensor], state["cuda"])
    if len(cuda) != torch.cuda.device_count():
        raise ValueError("resume requires the same CUDA device count")
    if cuda:
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda])


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
    return {
        "schema_version": 1,
        "git_sha": git_sha,
        "experiment_id": _EXPERIMENT_ID,
        "source_selection_sha256": data_selection(cfg).sha256,
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


def validate_resume_provenance(stored: object, current: Mapping[str, object]) -> None:
    if not isinstance(stored, Mapping):
        raise ValueError("resume checkpoint has no O59 provenance record")
    typed_stored = cast(Mapping[str, object], stored)
    changed = [name for name, value in current.items() if typed_stored.get(name) != value]
    if changed:
        raise ValueError(f"resume provenance changed: {changed}")


def load_identity_sidecar(cfg: TrainConfig) -> PlayerIdentitySidecar:
    return load_player_identity_artifact(
        Path(cfg.player_sidecar_local),
        remote=cfg.player_sidecar_remote,
        expected_sha256=cfg.player_sidecar_sha256,
        expected_vocabulary_size=cfg.player_vocab_size,
        expected_vocabulary_sha256=cfg.player_vocab_sha256,
    )


def _collate_o59_batch(
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
    batch = TrainBatch(batch.context, batch.target, replay_ids)
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


def _require_loader_disk(loader: PhysicalShardReplayLoader[ReturnBatch]) -> None:
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
) -> PhysicalShardReplayLoader[ReturnBatch]:
    selection = data_selection(cfg)
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
        derive_spatial=ITEM_PLAYER_PROJECTION.derive_spatial,
    )
    train_loader = PhysicalShardReplayLoader[ReturnBatch](
        selection=selection,
        adapter=adapter,
        tasks=build_shard_plan(selection, adapter.manifests),
        data_protocol=cfg.data_protocol,
        source_manifest_sha256=streams.POLICY_WORLD_V8_TRAIN_MANIFEST_SHA256,
        batch_transform=functools.partial(
            _collate_o59_batch,
            stats=stats,
            projection=projection,
            context_length=cfg.arch.L_ctx,
            return_column=cfg.awr.ego_return_column,
            return_valid_column=cfg.awr.ego_return_valid_column,
        ),
        batch_size=cfg.batch_size,
        replay_slots=cfg.replay_slots,
        seed=cfg.seed,
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
        if sum(train_loader.source_sample_counts.values()) != cfg.train_replays:
            raise ValueError("physical-shard loader does not expose every policy-world-v8 training row")
        if train_loader.minimum_replay_gap_batches < cfg.minimum_replay_gap_batches:
            raise ValueError("replay ring is too small for the 200-batch reuse-gap contract")
    except Exception:
        train_loader.close()
        raise
    return train_loader


def _make_loaders(
    cfg: TrainConfig,
    stats: dict[str, FeatureStats],
    player_lookup: ReplayPlayerLookup | None = None,
) -> tuple[PhysicalShardReplayLoader[ReturnBatch], list[ReturnBatch]]:
    """Build physical-shard training and the unchanged generic validation cohort."""
    if player_lookup is None:
        player_lookup = ReplayPlayerLookup(load_identity_sidecar(cfg).by_replay)
    train_loader = _make_train_loader(cfg, stats, player_lookup)
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
            projection=FeatureProjection(
                columns=ITEM_PLAYER_PROJECTION.columns
                | {cfg.awr.ego_return_column, cfg.awr.ego_return_valid_column, "ego_return60", "ego_return60_valid"},
                derive_spatial=ITEM_PLAYER_PROJECTION.derive_spatial,
            ),
            batch_transform=functools.partial(collate_awr_batch, L_ctx=cfg.arch.L_ctx),
            replay_format="policy-world",
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
    loader: PhysicalShardReplayLoader[ReturnBatch]
    validation: list[ReturnBatch]
    iterator: Iterator[ReturnBatch]
    first_batch_future: Future[ReturnBatch]
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
    train_loader, validation = _make_loaders(cfg, stats, ReplayPlayerLookup(sidecar.by_replay))
    try:
        if resume_state is not None:
            loader_state = resume_state.get("loader")
            if not isinstance(loader_state, dict):
                raise ValueError("resume checkpoint does not contain O59 replay-loader state")
            train_loader.load_state_dict(cast(dict[str, object], loader_state))
        worker_started = time.monotonic()
        train_iterator = iter(train_loader)
        worker_start_seconds = time.monotonic() - worker_started
    except Exception:
        train_loader.close()
        raise
    first_batch_seconds: list[float] = []

    def load_first_batch() -> ReturnBatch:
        started = time.monotonic()
        batch = next(train_iterator)
        first_batch_seconds.append(time.monotonic() - started)
        return batch

    try:
        with ExitStack() as setup:
            executor = setup.enter_context(ThreadPoolExecutor(max_workers=1, thread_name_prefix="o59-first-batch"))
            first_batch = executor.submit(load_first_batch)
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
            "return-conditioned",
            "scaled",
            "059",
            "architecture-stability",
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
        f"O59 d{cfg.arch.d_model} L{cfg.arch.n_layers} history decoder with 32 sampled policy "
        "prefixes and dense 128-prefix value/AWR normalization"
    )
    wandb.run.summary["optimizer/name"] = "Muon+AdamW" if cfg.optimizer == "muon" else "AdamW"
    wandb.run.summary["optimizer/all_parameters_use_adamw"] = cfg.optimizer == "adamw"
    wandb.run.summary["optimizer/implementation"] = (
        "SingleDeviceMuonWithAuxAdam" if cfg.optimizer == "muon" else "torch.optim.AdamW"
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
    train_loader: PhysicalShardReplayLoader[ReturnBatch],
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
    wandb.run.summary["data/policy_supervised_prefixes"] = policy_positions
    wandb.run.summary["data/value_supervised_prefixes"] = value_positions
    wandb.run.summary["data/effective_epochs"] = value_positions / unique_frames
    wandb.run.summary["data/D_over_N"] = policy_positions / parameter_counts["total"]
    wandb.run.summary["data/nominal_loss_positions_per_update"] = cfg.policy_prefixes_per_update
    wandb.run.summary["data/cpu_lookahead_batches"] = cfg.train_prefetch_factor
    wandb.run.summary["data/loader_prefetch_factor"] = PREFETCH_FACTOR
    wandb.run.summary["data/raw_shard_materialization_threads"] = train_loader.materialization_threads
    wandb.run.summary["data/replay_slots"] = train_loader.replay_slots
    wandb.run.summary["data/generation_windows"] = cfg.windows_per_generation
    wandb.run.summary["data/minimum_replay_frames"] = cfg.minimum_replay_frames
    wandb.run.summary["data/epoch_semantics"] = "replay generations committed to ring / unique train replays"
    wandb.run.summary["data/replay_phase_block_batches"] = cfg.replay_phase_block_batches
    wandb.run.summary["data/minimum_replay_gap_batches"] = train_loader.minimum_replay_gap_batches
    wandb.run.summary["system/disk/required_bytes"] = train_loader.required_disk_bytes
    wandb.run.summary["system/disk/free_bytes_at_start"] = train_loader.disk_free_bytes
    wandb.run.summary["system/disk/reserved_bytes"] = cfg.reserved_disk_bytes
    wandb.run.summary["training/approx_flops_per_update"] = flops_per_update
    wandb.run.summary["training/flops_formula"] = (
        "6*B*(L_ctx*(N_trunk+N_other)+128*N_value+32*(N_trunk_skip_heads+N_return_conditioner)"
        "+32*n_offsets*(N_temporal+N_group_heads))"
    )
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
    wandb.run.summary["data/source_mixing"] = "identity_uniform_physical_shards"
    for name, weight in zip(cfg.source_names, source_weights, strict=True):
        wandb.run.summary[f"data/source_sampling_share/{name}"] = weight / source_weight_total


def _training_functions(model: GPT, cfg: TrainConfig, *, compile_mode: str | None = None) -> tuple[Callable, Callable]:
    """Return eager or singly compiled trunk and temporal training functions."""
    trunk_fn: Callable = model.forward
    temporal_fn: Callable = model.temporal.teacher_forced_nll_with_diagnostics
    mode = cfg.train_compile_mode if compile_mode is None else compile_mode
    if DEVICE == "cuda" and cfg.compile_trunk:
        # Resolve FlexAttention before Dynamo sees the model. This entrypoint is
        # the sole compilation owner for the raw mask and attention operations.
        model.trunk.resolve_attention(DEVICE)
        if model.trunk.attn_path not in ("flex", "varlen_flash"):
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
    model: GPT,
    batch: ReturnBatch,
    cfg: TrainConfig,
    *,
    step: int,
    update: int,
    valid_prefixes: int,
    trunk_fn: Callable,
    temporal_fn: Callable,
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
    loss, nll_sum, metrics = microbatch_loss(
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
    metrics["stability/action_grad_abs_max"] = _button_gradient_abs_max(model)
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
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
    return_target: Literal["p90", "unconditioned"] = "p90",
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
    model: GPT,
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
        return_masker_state=return_masker_state,
        prefix_sampler_state=prefix_sampler_state,
    )
    final_path = run_dir / ("smoke-final.pt" if smoke else "final.pt")
    advance_checkpoint_link(snapshot, final_path)
    if uploader is not None:
        uploader.upload(snapshot, key=final_path.name)

    checkpoint_sha = checkpoint_sha256(final_path)
    validation = _validation_wandb_metrics(val_metrics(model, val_cache, cfg), cfg)
    final_metrics = {f"val/{name}": value for name, value in validation.items()}
    if not smoke:
        if uploader is None:
            raise RuntimeError("production evaluation requires R2 checkpoint upload")
        uploader.wait()
        spawn_closed_loop_evaluation(run_dir.name, update, checkpoint_sha, cfg.final_eval_n_matchups)
        spawn_closed_loop_evaluation(
            run_dir.name, update, checkpoint_sha, cfg.final_eval_n_matchups, return_target="unconditioned"
        )
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
            if DEVICE == "cuda" and (cfg.compile_trunk or cfg.compile_temporal):
                torch.compiler.cudagraph_mark_step_begin()
            batch = synthetic_awr_batch(cfg, torch.device(DEVICE))
            valid_prefixes = cfg.batch_size * POLICY_PREFIXES_PER_WINDOW
            prefix_positions = torch.arange(
                cfg.arch.direct_loss_start,
                cfg.arch.direct_loss_start + POLICY_PREFIXES_PER_WINDOW,
                device=DEVICE,
            ).expand(cfg.batch_size, -1)
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
    return tuple(sorted(boundaries))


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
    run_name = resume_run or make_run_name(Path(__file__).stem, model_tag(cfg), "policy-world-v8", comment)
    uploader = BackgroundUploader(run_name) if cfg.push_to_r2 else None
    _init_wandb(cfg, run_name, resume_state)
    if cfg.parent_wandb_id is not None and wandb.run is not None and wandb.run.id == cfg.parent_wandb_id:
        raise RuntimeError("the continuation must use a new W&B run id")
    run_dir, replay_dir = setup_run_dir(run_name)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
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
    identity_masker = IdentityMasker(cfg.seed ^ 0x0501D, cfg.identity_dropout)
    return_masker = ReturnMasker(cfg.seed ^ 0x059C0D, cfg.return_dropout, enabled=cfg.return_conditioning)
    prefix_sampler = PrefixSampler(cfg.seed ^ 0x059A11, DEVICE)
    if resume_state is not None:
        validate_conditioning_state(resume_state, cfg)
        return_masker.load_state_dict(resume_state["return_masker"])
        model.return_calibration.load_state_dict(resume_state["return_calibration"])
        model.load_state_dict(resume_state["model"])
        optimizer.load_state_dict(resume_state["opt"])
        scheduler.load_state_dict(resume_state["sched"])
        identity_state = resume_state.get("identity_masker")
        if not isinstance(identity_state, dict):
            raise ValueError("resume checkpoint has no identity-mask RNG state")
        identity_masker.load_state_dict(identity_state)
        prefix_state = resume_state.get("prefix_sampler")
        if not isinstance(prefix_state, dict):
            raise ValueError("resume checkpoint has no prefix-sampling RNG state")
        prefix_sampler.load_state_dict(prefix_state)
        rng = resume_state.get("rng")
        if not isinstance(rng, Mapping):
            raise ValueError("resume checkpoint has no global RNG state")
        validate_resume_provenance(resume_state.get("provenance"), run_provenance(cfg))
        restore_rng(rng)
        start_step = int(resume_state["step"]) + 1
        positions_per_update = cfg.policy_prefixes_per_update
        actual_positions = int(
            resume_state.get(
                "actual_policy_prefixes",
                resume_state.get("actual_loss_positions", start_step * positions_per_update),
            )
        )
        if not 0 <= actual_positions <= start_step * positions_per_update:
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
    run_started = time.monotonic()
    try:
        batch_prefetcher = DeviceBatchPrefetcher(
            train_loader,
            cfg,
            DEVICE,
            identity_masker,
            return_masker=return_masker,
            calibration=model.return_calibration,
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
    window_peak_allocated_gb = 0.0
    update_timer = _UpdateTimer()
    # CUDA compilation must remain on the training thread. Background compilation
    # deadlocked training on both H100 and B200 hosts.
    model.train()
    evaluation_updates = frozenset(closed_loop_evaluation_updates(run_stop, cfg.eval_every))
    loader_state_boundaries = _loader_state_boundaries(cfg, run_stop)
    try:
        for step in range(start_step, run_stop):
            update = step + 1
            update_timer.start()
            if DEVICE == "cuda":
                torch.cuda.reset_peak_memory_stats()

            val_due = cfg.val_every > 0 and update % cfg.val_every == 0 and update < run_stop
            eval_due = update in evaluation_updates and update < run_stop
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
                batch,
                cfg,
                step=step,
                update=update,
                valid_prefixes=valid_prefixes,
                trunk_fn=trunk_fn,
                temporal_fn=temporal_fn,
                optimizer=optimizer,
                scheduler=scheduler,
                prefix_sampler=prefix_sampler,
                prefix_validated_on_cpu=True,
            )
            loader_wait = batch_prefetcher.stage_next() if not state_boundary_due else 0.0
            actual_positions += valid_prefixes
            metric_accumulator.add(result, valid_prefixes)
            window_loader_wait_seconds.append(loader_wait)
            if DEVICE == "cuda":
                window_peak_allocated_gb = max(
                    window_peak_allocated_gb,
                    torch.cuda.max_memory_allocated() / 2**30,
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
                    **identity_masker.metrics(),
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
                log["throughput/windows_per_s"] = samples_per_s
                log["throughput/policy_prefixes_per_s"] = samples_per_s * POLICY_PREFIXES_PER_WINDOW
                log["throughput/value_prefixes_per_s"] = samples_per_s * 128
                log["loader/uncovered_wait_fraction"] = loader_wait_fraction
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
                    milestone=cfg.eval_every > 0 and update % cfg.eval_every == 0,
                    wandb_id=None if wandb.run is None else wandb.run.id,
                    actual_loss_positions=actual_positions,
                    loader_state=train_loader.state_dict(),
                    identity_masker_state=identity_masker.state_dict(),
                    return_masker_state=return_masker.state_dict(),
                    prefix_sampler_state=prefix_sampler.state_dict(),
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
            if update < run_stop and boundary_due:
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
            loader_state=train_loader.state_dict(),
            identity_masker_state=identity_masker.state_dict(),
            return_masker_state=return_masker.state_dict(),
            prefix_sampler_state=prefix_sampler.state_dict(),
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
    calibration = values.pop("awr")
    return {
        "experiment_id": _EXPERIMENT_ID,
        "checkpoint_format_version": _CHECKPOINT_FORMAT_VERSION,
        "architecture": architecture,
        "awr_calibration": calibration,
        **values,
        "max_steps": cfg.max_steps,
        "warmup_steps": cfg.warmup_steps,
    }


def config_from_state(values: dict) -> TrainConfig:
    """Restore a checkpoint written by the current experiment definition."""
    derived_fields = {"max_steps", "warmup_steps"}
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
    derived = {name: values[name] for name in derived_fields}
    expected_derived = {name: getattr(cfg, name) for name in derived_fields}
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
    ReturnCalibration().load_state_dict(cast(dict[str, object], calibration))
    ReturnMasker(0, cfg.return_dropout, enabled=cfg.return_conditioning).load_state_dict(
        cast(Mapping[str, object], masker)
    )


def load_checkpoint(path: str, *, device: str = DEVICE) -> tuple[GPT, TrainConfig, dict[str, FeatureStats], dict]:
    state = torch.load(path, map_location=device, weights_only=False)
    cfg = config_from_state(state["cfg"])
    validate_config(cfg)
    validate_conditioning_state(state, cfg)
    encoded = state["model"].get("player_code_bytes")
    if not isinstance(encoded, Tensor) or not encoded.numel():
        raise ValueError("checkpoint has no embedded identity vocabulary")
    vocabulary = PlayerVocabulary(decode_player_codes(encoded.detach().cpu().numpy().tobytes()))
    model = GPT(cfg, vocabulary).to(device)
    model.load_state_dict(state["model"])
    model.return_calibration.load_state_dict(state["return_calibration"])
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
    return_target: Literal["p90", "unconditioned"] = "p90",
    cache_mode: Literal["recompute", "rolling"] = "recompute",
    compare_recompute: bool = False,
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
    desired_return = model.return_calibration.targets()[2] if return_target == "p90" else None
    calibration_hash = cast(str, model.return_calibration.state_dict()["sha256"])
    if return_target == "unconditioned" and (output_name is None or wandb_namespace == "eval"):
        raise ValueError("unconditioned comparisons require a separate output directory and W&B namespace")
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
    is_variant = (
        cache_mode != "recompute"
        or compare_recompute
        or any(value is not None for value in (player_code, fixed_ego_character, delay_frames, replan_interval_frames))
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
        cache_mode=cache_mode,
        compare_recompute=compare_recompute,
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
    proxy: bool = False
    comment: str = ""
    resume: str | None = None
    resume_checkpoint: str = "latest.pt"
    resume_as: str | None = None
    resume_num_workers: int | None = None
    decay_duration: int | None = None
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
    return_target: Literal["p90", "unconditioned"] = "p90"
    cache_mode: Literal["recompute", "rolling"] = "recompute"
    compare_recompute: bool = False


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
            cache_mode=args.cache_mode,
            compare_recompute=args.compare_recompute,
        )
        return
    resume_run = resume_state = None
    cfg = args.cfg
    proxy_arch: Architecture | None = None
    target_positions: int | None = None
    if args.resume is None and args.proxy:
        proxy = proxy_config()
        proxy_arch = proxy.arch
        target_positions = proxy.target_positions
    if args.resume is None and (args.resume_checkpoint != "latest.pt" or args.resume_as is not None):
        raise SystemExit("--resume-checkpoint and --resume-as require --resume")
    if args.resume is None and args.resume_num_workers is not None:
        raise SystemExit("--resume-num-workers requires --resume")
    if args.resume is None and args.decay_duration is not None:
        raise SystemExit("--decay-duration requires --resume and --resume-as")
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
            if args.decay_duration is not None:
                parent_update = int(resume_state["step"]) + 1
                if args.decay_duration <= 0:
                    raise SystemExit("--decay-duration must be positive")
                if args.resume_checkpoint == "latest.pt":
                    raise SystemExit("a decay fork requires an explicit immutable --resume-checkpoint")
                checkpoint_path = Path("runs") / args.resume / args.resume_checkpoint
                cfg = replace(
                    cfg,
                    decay_start_update=parent_update,
                    decay_duration=args.decay_duration,
                    parent_run_name=args.resume,
                    parent_checkpoint_name=args.resume_checkpoint,
                    parent_checkpoint_sha256=checkpoint_sha256(checkpoint_path),
                    parent_wandb_id=cast(str | None, resume_state.get("wandb_id")),
                )
            resume_state = {**resume_state, "wandb_id": None}
        elif args.decay_duration is not None:
            raise SystemExit("--decay-duration requires --resume-as to protect the stable parent")
    cfg = replace(
        cfg,
        arch=cfg.arch if proxy_arch is None else proxy_arch,
        target_positions=cfg.target_positions if target_positions is None else target_positions,
        num_workers=cfg.num_workers if args.resume_num_workers is None else args.resume_num_workers,
        eval_max_parallel=cfg.eval_max_parallel if args.eval_max_parallel is None else args.eval_max_parallel,
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
