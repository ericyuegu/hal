"""Experiment 045: one-way BYOL player-style representation learning.

The public artifact is the 128-dimensional pre-projector value returned by
``PlayerEncoder``. Metrics L2-normalize it. Projectors and the predictor are
training-only modules.

Run a local synthetic step:
    uv run experiments/045_player_identity.py --cpu-smoke

Run training:
    uv run experiments/045_player_identity.py --train
"""

from __future__ import annotations

import copy
import functools
import hashlib
import math
import random
import time
from collections import deque
from collections.abc import Mapping
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Final
from typing import Literal
from typing import cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from melee import Stage
from melee.stages import EDGE_POSITION
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from streaming import Stream
from streaming import StreamingDataLoader
from streaming import StreamingDataset
from torch import Tensor
from torch.optim import AdamW

import wandb
from hal import streams
from hal.data.feature_stats import FeatureStats
from hal.data.policy_schema import decode_policy_replay_slices
from hal.data.policy_world_schema import decode_policy_world_replay_slices
from hal.data.schema import Rank
from hal.data.streaming_compat import patch_streaming
from hal.training.checkpoints import BackgroundUploader
from hal.training.checkpoints import download_latest
from hal.training.dataloader import collate_windows
from hal.training.dataloader import relabel_ego
from hal.training.ego_stats import load_consolidated_stats
from hal.training.features import ACTION_CHANNELS
from hal.training.features import BASE_PLAYER_PREFIXES
from hal.training.features import CAT_FEATURES
from hal.training.features import FLOAT_FEATURES
from hal.training.features import Context
from hal.training.features import FeatureProjection
from hal.training.features import preprocess
from hal.training.runs import make_run_name
from hal.training.runs import setup_run_dir
from hal.training.trunk import Trunk
from hal.training.trunk import TrunkConfig
from hal.training.trunk import rmsnorm
from hal.training.trunk import varlen_flash_is_usable

patch_streaming()

WINDOW_LENGTH: Final[int] = 256
DESCRIPTOR_DIM: Final[int] = 80
ANONYMOUS_IO_BATCH: Final[int] = 64
ANONYMOUS_BATCHES_PER_UPDATE: Final[int] = 12
PROFESSIONAL_INPUT_BATCH: Final[int] = 28
PROFESSIONAL_QUEUE_LIMIT: Final[int] = 32
LOADER_WORKERS: Final[int] = 8
LOADER_PREFETCH_FACTOR: Final[int] = 2
HELDOUT_REPLAYS_PER_IDENTITY: Final[int] = 16
SEALED_IDENTITIES: Final[tuple[str, ...]] = (
    "cookbook",
    "solobattle",
    "rapm",
    "siddward",
    "gosu",
    "iliketurtles",
    "friend",
    "nicki",
    "trif",
    "mof",
)
DEVELOPMENT_IDENTITIES: Final[tuple[str, ...]] = tuple(
    identity for identity in streams.PROFESSIONAL_PLAYER_SLUGS if identity not in SEALED_IDENTITIES
)
IDENTITY_FEATURE_PROJECTION: Final[FeatureProjection] = FeatureProjection(
    columns=frozenset(
        {f"{prefix}_{name}" for prefix in BASE_PLAYER_PREFIXES for name in (*FLOAT_FEATURES, *CAT_FEATURES)}
        | {f"ego_{name}" for name in ACTION_CHANNELS}
    ),
    derive_spatial=False,
)


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Frozen configuration for the first complete O45 run."""

    window_length: int = WINDOW_LENGTH
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    representation_dim: int = 128
    projector_hidden_dim: int = 512
    logical_batch_size: int = 1024
    anonymous_pairs: int = 768
    professional_pairs: int = 256
    professional_identities_per_batch: int = 16
    professional_replays_per_identity: int = 16
    online_microbatch: int = 64
    target_microbatch: int = 128
    learning_rate: float = 3e-4
    minimum_learning_rate: float = 3e-5
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.05
    gradient_norm_limit: float = 1.0
    warmup_updates: int = 512
    max_updates: int = 8192
    ema_start: float = 0.996
    ema_end: float = 0.9995
    triplet_margin: float = 0.2
    triplet_weight: float = 0.25
    seed: int = 0
    allow_tf32: bool = True
    amp_dtype: str = "bfloat16"
    attention_backend: str = "varlen_flash"
    anonymous_data_root: str = "data/processed/ranked-anonymized-1/mds-policy-v7"
    professional_data_root: str = "data/processed/professional"
    checkpoint_every: int = 512
    wandb_project: str = "hal"
    wandb_mode: Literal["online", "offline", "disabled", "shared"] = "online"
    push_to_r2: bool = True

    def validate(self) -> None:
        """Reject configurations that would change the frozen logical protocol."""
        if self.window_length != WINDOW_LENGTH:
            raise ValueError(f"O45 fixes window_length at {WINDOW_LENGTH}")
        if self.logical_batch_size != self.anonymous_pairs + self.professional_pairs:
            raise ValueError("logical batch must equal anonymous_pairs + professional_pairs")
        expected_professional = self.professional_identities_per_batch * self.professional_replays_per_identity
        if self.professional_pairs != expected_professional:
            raise ValueError("professional batch must be identities_per_batch * replays_per_identity")
        if self.professional_identities_per_batch > len(DEVELOPMENT_IDENTITIES):
            raise ValueError("professional identity count exceeds the development identity pool")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.online_microbatch <= 0 or self.target_microbatch <= 0:
            raise ValueError("microbatch sizes must be positive")
        if self.max_updates <= 1 or not 0 <= self.warmup_updates < self.max_updates:
            raise ValueError("max_updates must exceed both one and warmup_updates")


@dataclass(frozen=True, slots=True)
class WindowMetadata:
    """Non-neural metadata used by pair and negative selection."""

    replay_id: str
    identity: str | None
    ego_character: int
    stage: int
    opponent_character: int
    descriptor: np.ndarray

    def __post_init__(self) -> None:
        if self.descriptor.shape != (DESCRIPTOR_DIM,):
            raise ValueError(f"descriptor must have shape {(DESCRIPTOR_DIM,)}, got {self.descriptor.shape}")


@dataclass(frozen=True, slots=True)
class PairMetadata:
    """Aligned metadata for one logical online/target batch."""

    online: tuple[WindowMetadata, ...]
    target: tuple[WindowMetadata, ...]
    professional_mask: Tensor

    def __post_init__(self) -> None:
        if len(self.online) != len(self.target) or self.professional_mask.shape != (len(self.online),):
            raise ValueError("pair metadata lengths do not match")
        if self.professional_mask.dtype != torch.bool:
            raise TypeError("professional_mask must be boolean")
        for index, is_professional in enumerate(self.professional_mask.tolist()):
            online_identity = self.online[index].identity
            target_identity = self.target[index].identity
            if is_professional and (online_identity is None or online_identity != target_identity):
                raise ValueError("professional positives must have one shared known identity")
            if not is_professional and (online_identity is not None or target_identity is not None):
                raise ValueError("anonymous pairs must not carry player identities")


@dataclass(frozen=True, slots=True)
class PairBatch:
    """One logical batch of aligned online and target views."""

    online: Context
    target: Context
    metadata: PairMetadata

    @property
    def batch_size(self) -> int:
        return len(self.metadata.online)

    def pin_memory(self) -> PairBatch:
        return PairBatch(self.online.pin_memory(), self.target.pin_memory(), self.metadata)


@dataclass(frozen=True, slots=True)
class PreparedPair:
    """Two anonymous views prepared from one replay in a loader worker."""

    online: dict[str, np.ndarray]
    target: dict[str, np.ndarray]
    online_metadata: WindowMetadata
    target_metadata: WindowMetadata


@dataclass(frozen=True, slots=True)
class ProfessionalCandidates:
    """Eligible professional windows from one stratified MDS input batch."""

    context: Context | None
    metadata: tuple[WindowMetadata, ...]
    raw_count: int

    def pin_memory(self) -> ProfessionalCandidates:
        context = None if self.context is None else self.context.pin_memory()
        return ProfessionalCandidates(context, self.metadata, self.raw_count)


@dataclass(frozen=True, slots=True)
class ProfessionalExample:
    """One queued professional replay window."""

    context: Context
    metadata: WindowMetadata


@dataclass(frozen=True, slots=True)
class HeldoutCache:
    """Fixed development-identity gallery and query windows."""

    gallery: tuple[ProfessionalExample, ...]
    query: tuple[ProfessionalExample, ...]


@dataclass(frozen=True, slots=True)
class LoaderMetrics:
    """Time spent waiting for the two input pipelines."""

    anonymous_wait_s: float
    professional_wait_s: float
    cache_gib: float


@dataclass(frozen=True, slots=True)
class StepMetrics:
    loss: float
    byol_loss: float
    triplet_loss: float
    valid_triplets: int
    gradient_norm: float
    ema_tau: float
    learning_rate: float


def adaptive_minimum_separation(num_frames: int, window_length: int = WINDOW_LENGTH) -> int:
    """Return ``min(1024, floor((T - window_length) / 3))``."""
    if num_frames < window_length:
        raise ValueError(f"replay has {num_frames} frames, fewer than window length {window_length}")
    return min(1024, (num_frames - window_length) // 3)


def sample_anonymous_anchor(
    num_frames: int,
    rng: np.random.Generator,
    window_length: int = WINDOW_LENGTH,
) -> int:
    """Sample the anchor start uniformly over every valid replay start."""
    adaptive_minimum_separation(num_frames, window_length)
    return int(rng.integers(num_frames - window_length + 1))


def sample_distant_start(
    num_frames: int,
    anchor: int,
    rng: np.random.Generator,
    window_length: int = WINDOW_LENGTH,
) -> int:
    """Sample a valid distant start with probability proportional to distance squared."""
    separation = adaptive_minimum_separation(num_frames, window_length)
    final_start = num_frames - window_length
    if not 0 <= anchor <= final_start:
        raise ValueError(f"anchor {anchor} is outside [0, {final_start}]")
    candidates = np.arange(final_start + 1, dtype=np.int64)
    distance = np.abs(candidates - anchor)
    eligible = distance >= separation
    weights = np.square(distance, dtype=np.float64) * eligible
    if weights.sum() == 0:
        return anchor
    return int(rng.choice(candidates, p=weights / weights.sum()))


def sample_anonymous_starts(
    num_frames: int,
    rng: np.random.Generator,
    window_length: int = WINDOW_LENGTH,
) -> tuple[int, int]:
    """Sample a uniform anchor and a squared-distance partner.

    The returned order already includes the independent 0.5 online/target role
    exchange.
    """
    anchor = sample_anonymous_anchor(num_frames, rng, window_length)
    partner = sample_distant_start(num_frames, anchor, rng, window_length)
    return (partner, anchor) if rng.random() < 0.5 else (anchor, partner)


def professional_split(identity: str, replay_id: str) -> str:
    """Assign a professional replay to train, gallery, or query by stable hash."""
    if identity in SEALED_IDENTITIES:
        return "sealed"
    if identity not in DEVELOPMENT_IDENTITIES:
        raise ValueError(f"unknown professional identity {identity!r}")
    digest = hashlib.blake2b(
        f"{identity}\0{replay_id}".encode(),
        digest_size=8,
        person=b"hal-o45-split",
    ).digest()
    bucket = int.from_bytes(digest, "little") % 100
    if bucket < 80:
        return "train"
    return "gallery" if bucket < 90 else "query"


def professional_ego_side(row: Mapping[str, object]) -> str | None:
    """Return the only Rank.PRO side, or ``None`` unless exactly one exists."""
    pro_value = int(Rank.PRO)
    sides = []
    for side in ("p1", "p2"):
        value = np.asarray(row[f"{side}_rank"])
        if value.size != 1:
            raise ValueError(f"{side}_rank must be scalar")
        if int(value.item()) == pro_value:
            sides.append(side)
    return sides[0] if len(sides) == 1 else None


def _stage_width(stage_id: int) -> float:
    try:
        width = 2.0 * float(EDGE_POSITION[Stage(stage_id)])
    except (KeyError, ValueError) as error:
        raise ValueError(f"stage {stage_id} has no known width") from error
    if width <= 0:
        raise ValueError(f"stage {stage_id} has non-positive width {width}")
    return width


def game_state_descriptor(window: Mapping[str, np.ndarray]) -> np.ndarray:
    """Build the fixed 8-bin, 80-value safe game-state descriptor."""
    required = (
        "stage",
        "ego_position_x",
        "opp_position_x",
        "ego_position_y",
        "opp_position_y",
        "ego_percent",
        "opp_percent",
        "ego_stock",
        "opp_stock",
    )
    missing = [name for name in required if name not in window]
    if missing:
        raise KeyError(f"descriptor input is missing {missing}")
    if len(np.asarray(window["ego_position_x"])) != WINDOW_LENGTH:
        raise ValueError(f"descriptor requires exactly {WINDOW_LENGTH} frames")
    stage_values = np.asarray(window["stage"])
    if stage_values.shape != (WINDOW_LENGTH,) or not np.all(stage_values == stage_values[0]):
        raise ValueError("stage must be constant over the window")
    width = _stage_width(int(stage_values[0]))
    ego_x = np.nan_to_num(np.asarray(window["ego_position_x"], dtype=np.float32)) / width
    opponent_x = np.nan_to_num(np.asarray(window["opp_position_x"], dtype=np.float32)) / width
    columns = np.stack(
        (
            ego_x,
            opponent_x,
            np.nan_to_num(np.asarray(window["ego_position_y"], dtype=np.float32)) / 100.0,
            np.nan_to_num(np.asarray(window["opp_position_y"], dtype=np.float32)) / 100.0,
            np.nan_to_num(np.asarray(window["ego_percent"], dtype=np.float32)) / 100.0,
            np.nan_to_num(np.asarray(window["opp_percent"], dtype=np.float32)) / 100.0,
            np.nan_to_num(np.asarray(window["ego_stock"], dtype=np.float32)),
            np.nan_to_num(np.asarray(window["opp_stock"], dtype=np.float32)),
            opponent_x - ego_x,
            (
                np.nan_to_num(np.asarray(window["opp_position_y"], dtype=np.float32))
                - np.nan_to_num(np.asarray(window["ego_position_y"], dtype=np.float32))
            )
            / 100.0,
        ),
        axis=1,
    )
    return columns.reshape(8, 32, 10).mean(axis=1).reshape(-1).astype(np.float32)


def professional_derangement(metadata: Sequence[WindowMetadata]) -> np.ndarray:
    """Return the maximum-score one-to-one cross-replay pairing."""
    count = len(metadata)
    if count < 2:
        raise ValueError("a professional derangement needs at least two windows")
    replay_ids = [item.replay_id for item in metadata]
    if len(set(replay_ids)) != count:
        raise ValueError("professional windows for one identity must use distinct replays")
    descriptors = np.stack([item.descriptor for item in metadata])
    state_distance = np.linalg.norm(descriptors[:, None] - descriptors[None, :], axis=-1)
    score = state_distance
    characters = np.asarray([item.ego_character for item in metadata])
    stages = np.asarray([item.stage for item in metadata])
    opponents = np.asarray([item.opponent_character for item in metadata])
    score += 8.0 * (characters[:, None] != characters[None, :])
    score += 4.0 * (stages[:, None] != stages[None, :])
    score += 2.0 * (opponents[:, None] != opponents[None, :])
    np.fill_diagonal(score, -1e9)
    rows, columns = linear_sum_assignment(-score)
    pairing = np.empty(count, dtype=np.int64)
    pairing[rows] = columns
    if np.any(pairing == np.arange(count)):
        raise RuntimeError("assignment did not produce a derangement")
    return pairing


def _linear_gelu_norm(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, out_dim),
        nn.LayerNorm(out_dim),
    )


class PlayerEncoder(nn.Module):
    """Shared causal trunk and public pre-projector player representation."""

    def __init__(self, cfg: TrainConfig, *, prefer_flex: bool = True) -> None:
        super().__init__()
        self.cat_embeds = nn.ModuleDict(
            {name: nn.Embedding(vocab, dim) for name, (vocab, dim) in CAT_FEATURES.items()}
        )
        per_player = 2 * len(FLOAT_FEATURES) + sum(dim for _, dim in CAT_FEATURES.values())
        input_dim = len(BASE_PLAYER_PREFIXES) * per_player + len(ACTION_CHANNELS)
        self.input_projection = nn.Linear(input_dim, cfg.d_model)
        self.trunk = Trunk(
            TrunkConfig(
                d_model=cfg.d_model,
                n_layers=cfg.n_layers,
                n_heads=cfg.n_heads,
                L_ctx=cfg.window_length,
                attn_window=0,
                attention_backend=cfg.attention_backend,
            ),
            prefer_flex=prefer_flex,
        )
        self.representation = nn.Linear(cfg.d_model, cfg.representation_dim)

    def _player_features(self, features: Mapping[str, Tensor], prefix: str) -> Tensor:
        reference = features[f"{prefix}_position_x"]
        batch, length = reference.shape
        parts = [features[f"{prefix}_{name}"][..., None] for name in FLOAT_FEATURES]
        for name in FLOAT_FEATURES:
            mask_name = f"{prefix}_{name}_mask"
            mask = features.get(mask_name)
            if mask is None:
                mask = torch.zeros(batch, length, device=reference.device, dtype=reference.dtype)
            parts.append(mask[..., None])
        for name, (vocab, _) in CAT_FEATURES.items():
            parts.append(self.cat_embeds[name](features[f"{prefix}_{name}"].clamp(0, vocab - 1)))
        return torch.cat(parts, dim=-1)

    def forward(self, context: Context) -> Tensor:
        """Return RMS-normalized pre-projector ``z`` without L2 normalization."""
        if context.ctx_pad.any():
            raise ValueError("O45 windows are complete and cannot contain left padding")
        parts = [self._player_features(context.features, prefix) for prefix in BASE_PLAYER_PREFIXES]
        parts.append(torch.stack([context.features[f"ego_{name}"] for name in ACTION_CHANNELS], dim=-1))
        hidden = self.trunk(self.input_projection(torch.cat(parts, dim=-1)), context.ctx_pad)
        return rmsnorm(self.representation(hidden.mean(dim=1)))

    def normalized(self, context: Context) -> Tensor:
        """Return the L2-normalized public representation used by metrics."""
        return F.normalize(self(context), dim=-1)


class BYOL(nn.Module):
    """One-way BYOL model with an EMA encoder and projector."""

    def __init__(self, cfg: TrainConfig, *, prefer_flex: bool = True) -> None:
        super().__init__()
        self.online_encoder = PlayerEncoder(cfg, prefer_flex=prefer_flex)
        self.online_projector = _linear_gelu_norm(
            cfg.representation_dim,
            cfg.projector_hidden_dim,
            cfg.representation_dim,
        )
        self.predictor = _linear_gelu_norm(
            cfg.representation_dim,
            cfg.projector_hidden_dim,
            cfg.representation_dim,
        )
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        for parameter in self.target_parameters():
            parameter.requires_grad_(False)

    def target_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.target_encoder.parameters()) + tuple(self.target_projector.parameters())

    def online(self, context: Context) -> tuple[Tensor, Tensor]:
        z = self.online_encoder(context)
        prediction = self.predictor(self.online_projector(z))
        return F.normalize(prediction, dim=-1), F.normalize(z, dim=-1)

    @torch.no_grad()
    def target(self, context: Context) -> tuple[Tensor, Tensor]:
        z = self.target_encoder(context)
        projected = self.target_projector(z)
        return F.normalize(projected, dim=-1), F.normalize(z, dim=-1)

    @torch.no_grad()
    def update_target(self, tau: float) -> None:
        if not 0 <= tau <= 1:
            raise ValueError(f"EMA tau must be in [0, 1], got {tau}")
        online_modules = (self.online_encoder, self.online_projector)
        target_modules = (self.target_encoder, self.target_projector)
        for online_module, target_module in zip(online_modules, target_modules, strict=True):
            for online, target in zip(online_module.parameters(), target_module.parameters(), strict=True):
                target.lerp_(online, 1.0 - tau)
            for online, target in zip(online_module.buffers(), target_module.buffers(), strict=True):
                if target.dtype.is_floating_point:
                    target.lerp_(online, 1.0 - tau)
                else:
                    target.copy_(online)

    def export_encoder(self) -> PlayerEncoder:
        """Return an independent encoder with no projector or predictor."""
        return copy.deepcopy(self.online_encoder).eval()


def one_way_byol_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Return one loss per aligned pair. The target is always detached."""
    if prediction.shape != target.shape:
        raise ValueError(f"BYOL shapes differ: {prediction.shape} and {target.shape}")
    return 2.0 - 2.0 * (F.normalize(prediction, dim=-1) * F.normalize(target.detach(), dim=-1)).sum(dim=-1)


def ema_tau(step: int, cfg: TrainConfig) -> float:
    """Cosine interpolation from 0.996 to 0.9995, including both endpoints."""
    if not 0 <= step < cfg.max_updates:
        raise ValueError(f"EMA step {step} is outside [0, {cfg.max_updates})")
    fraction = step / (cfg.max_updates - 1)
    blend = 0.5 - 0.5 * math.cos(math.pi * fraction)
    return cfg.ema_start + (cfg.ema_end - cfg.ema_start) * blend


def learning_rate(step: int, cfg: TrainConfig) -> float:
    """Linear warmup, then cosine decay to the fixed minimum rate."""
    if not 0 <= step < cfg.max_updates:
        raise ValueError(f"learning-rate step {step} is outside [0, {cfg.max_updates})")
    if step < cfg.warmup_updates:
        return cfg.learning_rate * (step + 1) / cfg.warmup_updates
    denominator = max(1, cfg.max_updates - cfg.warmup_updates - 1)
    progress = (step - cfg.warmup_updates) / denominator
    blend = 0.5 + 0.5 * math.cos(math.pi * progress)
    return cfg.minimum_learning_rate + (cfg.learning_rate - cfg.minimum_learning_rate) * blend


def make_optimizer(model: BYOL, cfg: TrainConfig) -> AdamW:
    return AdamW(
        (*model.online_encoder.parameters(), *model.online_projector.parameters(), *model.predictor.parameters()),
        lr=learning_rate(0, cfg),
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
    )


def _context_slice(context: Context, start: int, stop: int, device: torch.device) -> Context:
    return Context(
        features={name: value[start:stop].to(device) for name, value in context.features.items()},
        ctx_pad=context.ctx_pad[start:stop].to(device),
    )


def select_professional_negatives(
    anchor_z: Tensor,
    candidate_z: Tensor,
    anchors: Sequence[WindowMetadata],
    candidates: Sequence[WindowMetadata],
    *,
    descriptor_neighbors: int = 16,
) -> Tensor:
    """Select safe batch-hard negatives and return candidate indices or -1."""
    if (
        anchor_z.ndim != 2
        or candidate_z.ndim != 2
        or anchor_z.shape[1] != candidate_z.shape[1]
        or anchor_z.shape[0] != len(anchors)
        or candidate_z.shape[0] != len(candidates)
    ):
        raise ValueError("professional embeddings and metadata are not aligned")
    selected = torch.full((len(anchors),), -1, dtype=torch.long, device=anchor_z.device)
    candidate_embeddings = F.normalize(candidate_z.detach(), dim=-1)
    for anchor_index, anchor in enumerate(anchors):
        if anchor.identity is None:
            raise ValueError("negative selection accepts professional anchors only")
        same_character = [
            index
            for index, candidate in enumerate(candidates)
            if candidate.identity is not None
            and candidate.identity != anchor.identity
            and candidate.ego_character == anchor.ego_character
        ]
        groups = (
            [
                index
                for index in same_character
                if candidates[index].stage == anchor.stage
                and candidates[index].opponent_character == anchor.opponent_character
            ],
            [index for index in same_character if candidates[index].stage == anchor.stage],
            same_character,
        )
        pool = next((group for group in groups if group), None)
        if pool is None:
            continue
        distances = np.asarray([np.linalg.norm(anchor.descriptor - candidates[index].descriptor) for index in pool])
        nearest_order = np.argsort(distances, kind="stable")[:descriptor_neighbors]
        nearest = torch.tensor([pool[index] for index in nearest_order], device=anchor_z.device)
        similarity = candidate_embeddings[nearest] @ F.normalize(anchor_z[anchor_index].detach(), dim=-1)
        selected[anchor_index] = nearest[similarity.argmax()]
    return selected


def professional_triplet_loss(
    anchor_z: Tensor,
    positive_z: Tensor,
    anchors: Sequence[WindowMetadata],
    candidates: Sequence[WindowMetadata],
    margin: float = 0.2,
) -> tuple[Tensor, Tensor]:
    """Return valid cosine-margin losses and their selected negative indices."""
    negatives = select_professional_negatives(anchor_z, positive_z, anchors, candidates)
    valid = negatives >= 0
    if not valid.any():
        return anchor_z.new_empty(0), negatives
    anchors_normalized = F.normalize(anchor_z[valid], dim=-1)
    positives_normalized = F.normalize(positive_z[valid].detach(), dim=-1)
    negatives_normalized = F.normalize(positive_z[negatives[valid]].detach(), dim=-1)
    positive_similarity = (anchors_normalized * positives_normalized).sum(dim=-1)
    negative_similarity = (anchors_normalized * negatives_normalized).sum(dim=-1)
    return F.relu(margin + negative_similarity - positive_similarity), negatives


def train_step(
    model: BYOL,
    batch: PairBatch,
    optimizer: AdamW,
    cfg: TrainConfig,
    step: int,
    device: torch.device,
) -> StepMetrics:
    """Apply one microbatched logical update and then update the EMA branch."""
    if batch.batch_size != cfg.logical_batch_size:
        raise ValueError(f"logical batch has {batch.batch_size} pairs, expected {cfg.logical_batch_size}")
    for group in optimizer.param_groups:
        group["lr"] = learning_rate(step, cfg)
    amp = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" and cfg.amp_dtype == "bfloat16"
        else nullcontext()
    )
    target_projected: list[Tensor] = []
    target_z: list[Tensor] = []
    with torch.no_grad():
        for start in range(0, batch.batch_size, cfg.target_microbatch):
            context = _context_slice(batch.target, start, start + cfg.target_microbatch, device)
            with amp:
                projected, z = model.target(context)
            target_projected.append(projected.float())
            target_z.append(z.float())
    target_projected_tensor = torch.cat(target_projected)
    target_z_tensor = torch.cat(target_z)

    optimizer.zero_grad(set_to_none=True)
    byol_sum = torch.zeros((), device=device)
    professional_indices: list[int] = []
    professional_z: list[Tensor] = []
    for start in range(0, batch.batch_size, cfg.online_microbatch):
        stop = min(start + cfg.online_microbatch, batch.batch_size)
        context = _context_slice(batch.online, start, stop, device)
        with amp:
            prediction, z = model.online(context)
            losses = one_way_byol_loss(prediction, target_projected_tensor[start:stop])
            scaled = losses.sum() / batch.batch_size
        local_mask = batch.metadata.professional_mask[start:stop]
        keep_graph = bool(local_mask.any())
        scaled.backward(retain_graph=keep_graph)
        byol_sum += losses.detach().sum()
        if keep_graph:
            local_indices = torch.where(local_mask)[0]
            professional_indices.extend((local_indices + start).tolist())
            professional_z.append(z[local_indices])

    if professional_indices:
        online_professional_z = torch.cat(professional_z).float()
        index = torch.tensor(professional_indices, device=device)
        target_professional_z = target_z_tensor[index]
        anchors = [batch.metadata.online[item] for item in professional_indices]
        candidates = [batch.metadata.target[item] for item in professional_indices]
        triplets, _ = professional_triplet_loss(
            online_professional_z,
            target_professional_z,
            anchors,
            candidates,
            cfg.triplet_margin,
        )
        if triplets.numel():
            (cfg.triplet_weight * triplets.mean()).backward()
        triplet_mean = float(triplets.detach().mean()) if triplets.numel() else 0.0
    else:
        triplets = torch.empty(0, device=device)
        triplet_mean = 0.0

    parameters = (
        *model.online_encoder.parameters(),
        *model.online_projector.parameters(),
        *model.predictor.parameters(),
    )
    gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, cfg.gradient_norm_limit)
    if not torch.isfinite(gradient_norm):
        raise FloatingPointError(f"step {step}: gradient norm is {gradient_norm}")
    optimizer.step()
    tau = ema_tau(step, cfg)
    model.update_target(tau)
    byol_mean = float(byol_sum / batch.batch_size)
    return StepMetrics(
        loss=byol_mean + cfg.triplet_weight * triplet_mean,
        byol_loss=byol_mean,
        triplet_loss=triplet_mean,
        valid_triplets=int(triplets.numel()),
        gradient_norm=float(gradient_norm),
        ema_tau=tau,
        learning_rate=float(optimizer.param_groups[0]["lr"]),
    )


def _average_precision(relevant: np.ndarray) -> float:
    positions = np.flatnonzero(relevant) + 1
    if not len(positions):
        return 0.0
    return float(np.mean(np.arange(1, len(positions) + 1) / positions))


def retrieval_metrics(
    gallery_z: np.ndarray,
    gallery_identities: Sequence[str],
    gallery_replays: Sequence[str],
    query_z: np.ndarray,
    query_identities: Sequence[str],
    query_replays: Sequence[str],
) -> dict[str, float]:
    """Compute cross-replay Recall@1/5, MRR, and mAP."""
    gallery = _normalize_numpy(gallery_z)
    queries = _normalize_numpy(query_z)
    scores = queries @ gallery.T
    reciprocal_ranks: list[float] = []
    average_precisions: list[float] = []
    recalls = {1: [], 5: []}
    for row, (identity, replay_id) in enumerate(zip(query_identities, query_replays, strict=True)):
        allowed = np.asarray(gallery_replays) != replay_id
        relevant = (np.asarray(gallery_identities) == identity) & allowed
        if not relevant.any():
            continue
        order = np.flatnonzero(allowed)[np.argsort(-scores[row, allowed], kind="stable")]
        ranked_relevant = np.asarray(gallery_identities)[order] == identity
        first = int(np.flatnonzero(ranked_relevant)[0]) + 1
        reciprocal_ranks.append(1.0 / first)
        average_precisions.append(_average_precision(ranked_relevant))
        for cutoff in recalls:
            recalls[cutoff].append(float(ranked_relevant[:cutoff].any()))
    if not reciprocal_ranks:
        raise ValueError("retrieval evaluation has no cross-replay positives")
    return {
        "recall_at_1": float(np.mean(recalls[1])),
        "recall_at_5": float(np.mean(recalls[5])),
        "mrr": float(np.mean(reciprocal_ranks)),
        "map": float(np.mean(average_precisions)),
        "queries": float(len(reciprocal_ranks)),
    }


def knn_identification(
    gallery_z: np.ndarray,
    gallery_identities: Sequence[str],
    query_z: np.ndarray,
    query_identities: Sequence[str],
    k_values: Sequence[int] = (1, 5, 15),
) -> dict[str, float]:
    """Compute cosine kNN identification with similarity-weighted voting."""
    scores = _normalize_numpy(query_z) @ _normalize_numpy(gallery_z).T
    gallery_labels = np.asarray(gallery_identities)
    query_labels = np.asarray(query_identities)
    output: dict[str, float] = {}
    for k in k_values:
        if not 1 <= k <= len(gallery_labels):
            raise ValueError(f"k={k} is outside gallery size {len(gallery_labels)}")
        neighbors = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        predictions = []
        for row, selected in enumerate(neighbors):
            votes: dict[str, float] = {}
            for index in selected:
                label = str(gallery_labels[index])
                votes[label] = votes.get(label, 0.0) + float(scores[row, index])
            predictions.append(max(sorted(votes), key=votes.__getitem__))
        output[f"knn_{k}"] = float(np.mean(np.asarray(predictions) == query_labels))
    return output


def linear_probe_metrics(
    train_z: np.ndarray,
    train_identities: Sequence[str],
    query_z: np.ndarray,
    query_identities: Sequence[str],
) -> dict[str, float]:
    """Fit the fixed C=1 multinomial probe and report query accuracy."""
    probe = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    probe.fit(_normalize_numpy(train_z), np.asarray(train_identities))
    return {"linear_probe": float(probe.score(_normalize_numpy(query_z), np.asarray(query_identities)))}


def prototype_identification(
    support_z: np.ndarray,
    support_identities: Sequence[str],
    support_replays: Sequence[str],
    query_z: np.ndarray,
    query_identities: Sequence[str],
    shots: Sequence[int] = (1, 4, 16),
) -> dict[str, float]:
    """Compute deterministic nearest-prototype identification."""
    support = _normalize_numpy(support_z)
    query = _normalize_numpy(query_z)
    labels = np.asarray(support_identities)
    identities = sorted(set(support_identities))
    output: dict[str, float] = {}
    for shot_count in shots:
        prototypes = []
        retained = []
        for identity in identities:
            indices = np.flatnonzero(labels == identity)
            indices = indices[np.argsort(np.asarray(support_replays)[indices], kind="stable")]
            if len(indices) < shot_count:
                continue
            prototypes.append(_normalize_numpy(support[indices[:shot_count]].mean(axis=0, keepdims=True))[0])
            retained.append(identity)
        if not prototypes:
            output[f"prototype_{shot_count}"] = float("nan")
            continue
        predictions = np.asarray(retained)[np.argmax(query @ np.stack(prototypes).T, axis=1)]
        eligible = np.isin(np.asarray(query_identities), retained)
        output[f"prototype_{shot_count}"] = float(
            np.mean(predictions[eligible] == np.asarray(query_identities)[eligible])
        )
    return output


def distance_distributions(z: np.ndarray, identities: Sequence[str]) -> dict[str, np.ndarray]:
    """Return upper-triangle same- and different-player cosine distances."""
    normalized = _normalize_numpy(z)
    distance = 1.0 - normalized @ normalized.T
    row, column = np.triu_indices(len(z), k=1)
    same = np.asarray(identities)[row] == np.asarray(identities)[column]
    return {"same_player": distance[row[same], column[same]], "different_player": distance[row[~same], column[~same]]}


def select_nuisance_triplets(
    metadata: Sequence[WindowMetadata],
    query_indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select nuisance-crossing positives and state-matched safe comparisons.

    Positives are from another replay of the query identity. The lexicographic
    preference is different ego character, opponent character, stage, then
    descriptor distance. Negatives are the nearest descriptor from a different
    identity with the query's ego character.
    """
    selected_queries: list[int] = []
    positives: list[int] = []
    negatives: list[int] = []
    for query_index in query_indices:
        query = metadata[query_index]
        if query.identity is None:
            raise ValueError("nuisance evaluation requires known-player queries")
        positive_pool = [
            index
            for index, candidate in enumerate(metadata)
            if candidate.identity == query.identity and candidate.replay_id != query.replay_id
        ]
        negative_pool = [
            index
            for index, candidate in enumerate(metadata)
            if candidate.identity is not None
            and candidate.identity != query.identity
            and candidate.ego_character == query.ego_character
        ]
        if not positive_pool or not negative_pool:
            continue

        positive_scores = []
        for index in positive_pool:
            candidate = metadata[index]
            score = (
                candidate.ego_character != query.ego_character,
                candidate.opponent_character != query.opponent_character,
                candidate.stage != query.stage,
                float(np.linalg.norm(candidate.descriptor - query.descriptor)),
            )
            positive_scores.append((score, -index, index))
        positive = max(positive_scores)[2]
        negative_distances = [
            (float(np.linalg.norm(metadata[index].descriptor - query.descriptor)), index) for index in negative_pool
        ]
        negative = min(negative_distances)[1]
        selected_queries.append(query_index)
        positives.append(positive)
        negatives.append(negative)
    return (
        np.asarray(selected_queries, dtype=np.int64),
        np.asarray(positives, dtype=np.int64),
        np.asarray(negatives, dtype=np.int64),
    )


def sealed_replay_split(identity: str, replay_id: str) -> str:
    """Deterministically assign a sealed player's replay to support or query."""
    if identity not in SEALED_IDENTITIES:
        raise ValueError(f"{identity!r} is not a sealed identity")
    digest = hashlib.blake2b(
        f"{identity}\0{replay_id}".encode(),
        digest_size=8,
        person=b"hal-o45-sealed",
    ).digest()
    return "support" if int.from_bytes(digest, "little") % 2 == 0 else "query"


def nuisance_controlled_metrics(
    query_z: np.ndarray,
    positive_z: np.ndarray,
    negative_z: np.ndarray,
    query_metadata: Sequence[WindowMetadata],
    positive_metadata: Sequence[WindowMetadata],
    *,
    high_state_quantile: float = 0.75,
) -> dict[str, float]:
    """Compute the primary gap, AUC, triplet accuracy, and nuisance subsets."""
    if not (len(query_z) == len(positive_z) == len(negative_z) == len(query_metadata) == len(positive_metadata)):
        raise ValueError("nuisance-controlled arrays are not aligned")
    query = _normalize_numpy(query_z)
    positive = _normalize_numpy(positive_z)
    negative = _normalize_numpy(negative_z)
    same_distance = 1.0 - np.sum(query * positive, axis=1)
    different_distance = 1.0 - np.sum(query * negative, axis=1)
    labels = np.concatenate((np.ones(len(query)), np.zeros(len(query))))
    scores = -np.concatenate((same_distance, different_distance))
    output = {
        "distance_gap": float(different_distance.mean() - same_distance.mean()),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "triplet_accuracy": float(np.mean(same_distance < different_distance)),
        "coverage": float(len(query)),
    }
    state_distance = np.asarray(
        [np.linalg.norm(left.descriptor - right.descriptor) for left, right in zip(query_metadata, positive_metadata)]
    )
    threshold = float(np.quantile(state_distance, high_state_quantile))
    subsets = {
        "character_crossing": np.asarray(
            [left.ego_character != right.ego_character for left, right in zip(query_metadata, positive_metadata)]
        ),
        "stage_crossing": np.asarray(
            [left.stage != right.stage for left, right in zip(query_metadata, positive_metadata)]
        ),
        "opponent_crossing": np.asarray(
            [
                left.opponent_character != right.opponent_character
                for left, right in zip(query_metadata, positive_metadata)
            ]
        ),
        "high_state_distance": state_distance >= threshold,
    }
    for name, mask in subsets.items():
        output[f"{name}/coverage"] = float(mask.sum())
        output[f"{name}/distance_gap"] = (
            float(different_distance[mask].mean() - same_distance[mask].mean()) if mask.any() else float("nan")
        )
        output[f"{name}/triplet_accuracy"] = (
            float(np.mean(same_distance[mask] < different_distance[mask])) if mask.any() else float("nan")
        )
    return output


def collapse_diagnostics(z: np.ndarray) -> dict[str, float]:
    """Return normalized covariance effective rank and mean coordinate std."""
    normalized = _normalize_numpy(z)
    covariance = np.cov(normalized, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    probabilities = eigenvalues / max(float(eigenvalues.sum()), np.finfo(np.float64).eps)
    positive = probabilities > 0
    effective_rank = math.exp(float(-(probabilities[positive] * np.log(probabilities[positive])).sum()))
    return {
        "effective_rank": effective_rank,
        "mean_coordinate_std": float(normalized.std(axis=0).mean()),
    }


def bootstrap_lower_bound(
    values: np.ndarray,
    *,
    chance: float = 0.0,
    seed: int = 0,
    samples: int = 10_000,
) -> float:
    """Return the one-sided 95% bootstrap lower bound for a mean minus chance."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("bootstrap values must be a non-empty vector")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1) - chance
    return float(np.quantile(means, 0.025))


def _normalize_numpy(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, np.finfo(values.dtype).eps)


def _window_metadata(
    window: Mapping[str, np.ndarray],
    replay_id: str,
    identity: str | None,
) -> WindowMetadata:
    return WindowMetadata(
        replay_id=replay_id,
        identity=identity,
        ego_character=int(np.asarray(window["ego_character"])[0]),
        stage=int(np.asarray(window["stage"])[0]),
        opponent_character=int(np.asarray(window["opp_character"])[0]),
        descriptor=game_state_descriptor(window),
    )


def _array_columns(window: Mapping[str, np.ndarray | int]) -> dict[str, np.ndarray]:
    """Drop compact-schema scalar bookkeeping before frame-window collation."""
    return {name: value for name, value in window.items() if isinstance(value, np.ndarray)}


def _context_from_windows(
    windows: Sequence[Mapping[str, np.ndarray]],
    stats: Mapping[str, FeatureStats],
) -> Context:
    stacked = collate_windows([dict(window) for window in windows])
    features = preprocess(stacked, dict(stats), projection=IDENTITY_FEATURE_PROJECTION)
    return Context(features=features, ctx_pad=torch.zeros(len(windows), dtype=torch.long))


def _stable_replay_rng(seed: int, epoch: int, replay_id: str, source: str) -> np.random.Generator:
    """Return replay-local randomness that is exact after an MDS cursor resume."""
    digest = hashlib.blake2b(
        f"{seed}\0{epoch}\0{source}\0{replay_id}".encode(),
        digest_size=16,
        person=b"hal-o45-window",
    ).digest()
    return np.random.default_rng(int.from_bytes(digest, "little"))


def _streaming_epoch(dataset: StreamingDataset) -> int:
    return max(0, dataset.next_epoch - 1)


class AnonymousPairDataset(StreamingDataset):
    """MDS dataset that prepares both anonymous views in its loader worker."""

    def __init__(self, *args: Any, seed: int, **kwargs: Any) -> None:
        self._window_seed = seed
        super().__init__(*args, **kwargs)

    def get_item(self, sample_id: int, retry: int = 7) -> PreparedPair:
        row = super().get_item(sample_id, retry)
        replay_id = str(row["replay_id"])
        rng = _stable_replay_rng(self._window_seed, _streaming_epoch(self), replay_id, "anonymous")
        online_start, target_start = sample_anonymous_starts(int(row["num_frames"]), rng)
        decoded = decode_policy_replay_slices(
            row,
            (
                (online_start, online_start + WINDOW_LENGTH),
                (target_start, target_start + WINDOW_LENGTH),
            ),
        )
        ego_side = "p1" if rng.random() < 0.5 else "p2"
        online = relabel_ego(_array_columns(decoded[0]), ego_side)
        target = relabel_ego(_array_columns(decoded[1]), ego_side)
        return PreparedPair(
            online,
            target,
            _window_metadata(online, replay_id, None),
            _window_metadata(target, replay_id, None),
        )


class SourceTaggedStreamingDataset(StreamingDataset):
    """Attach a professional source slug to each row."""

    def __init__(self, *args: Any, source_slugs: Sequence[str], **kwargs: Any) -> None:
        self._source_slugs = tuple(source_slugs)
        super().__init__(*args, **kwargs)
        if len(self._source_slugs) != self.num_streams:
            raise ValueError("source slug count must match MDS stream count")

    def get_item(self, sample_id: int, retry: int = 7) -> dict[str, Any]:
        shard_id, _ = self.spanner[sample_id]
        row = dict(super().get_item(sample_id, retry))
        row["_o45_source"] = self._source_slugs[int(self.stream_per_shard[shard_id])]
        row["_o45_epoch"] = _streaming_epoch(self)
        return row


class O45StreamingDataLoader(StreamingDataLoader):
    """Count raw MDS rows for O45's custom worker-collated batch types."""

    def _get_batch_size(self, batch: object) -> int:
        if isinstance(batch, PairBatch):
            return batch.batch_size
        if isinstance(batch, ProfessionalCandidates):
            return batch.raw_count
        return super()._get_batch_size(batch)

    def close(self) -> None:
        """Stop persistent workers when a pipeline is no longer needed."""
        if self._iterator is not None:
            cast(Any, self._iterator)._shutdown_workers()
            self._iterator = None


def _collate_anonymous(
    pairs: Sequence[PreparedPair],
    *,
    stats: Mapping[str, FeatureStats],
) -> PairBatch:
    online = [pair.online for pair in pairs]
    target = [pair.target for pair in pairs]
    mask = torch.zeros(len(pairs), dtype=torch.bool)
    return PairBatch(
        _context_from_windows(online, stats),
        _context_from_windows(target, stats),
        PairMetadata(
            tuple(pair.online_metadata for pair in pairs),
            tuple(pair.target_metadata for pair in pairs),
            mask,
        ),
    )


def _collate_professional(
    rows: Sequence[Mapping[str, Any]],
    *,
    stats: Mapping[str, FeatureStats],
    seed: int,
    allowed_splits: tuple[str, ...],
) -> ProfessionalCandidates:
    windows: list[dict[str, np.ndarray]] = []
    metadata: list[WindowMetadata] = []
    for row in rows:
        identity = str(row["_o45_source"])
        replay_id = str(row["replay_id"])
        if professional_split(identity, replay_id) not in allowed_splits:
            continue
        ego_side = professional_ego_side(row)
        frames = int(row["num_frames"])
        if ego_side is None or frames < WINDOW_LENGTH:
            continue
        rng = _stable_replay_rng(seed, int(row["_o45_epoch"]), replay_id, identity)
        start = int(rng.integers(frames - WINDOW_LENGTH + 1))
        decoded = decode_policy_world_replay_slices(row, ((start, start + WINDOW_LENGTH),))[0]
        window = relabel_ego(_array_columns(decoded), ego_side)
        windows.append(window)
        metadata.append(_window_metadata(window, replay_id, identity))
    context = _context_from_windows(windows, stats) if windows else None
    return ProfessionalCandidates(context, tuple(metadata), len(rows))


def _professional_streams(cfg: TrainConfig) -> list[Stream]:
    proportion = 1.0 / len(DEVELOPMENT_IDENTITIES)
    selected = []
    for identity in DEVELOPMENT_IDENTITIES:
        source = streams.PROFESSIONAL_POLICY_WORLD_V7[identity]
        selected.append(
            Stream(
                remote=source.remote,
                local=str(Path(cfg.professional_data_root) / identity / "mds-policy-world-v7"),
                split="train",
                proportion=proportion,
                keep_zip=False,
            )
        )
    return selected


def make_anonymous_loader(
    cfg: TrainConfig,
    stats: Mapping[str, FeatureStats],
    *,
    num_workers: int = LOADER_WORKERS,
) -> O45StreamingDataLoader:
    """Build the normal shuffled anonymous MDS pipeline."""
    source = streams.RANKED_ANONYMIZED_1_POLICY_V7
    remote, _ = source.for_split("train")
    dataset = AnonymousPairDataset(
        seed=cfg.seed,
        remote=remote,
        local=str(Path(cfg.anonymous_data_root) / "train"),
        batch_size=ANONYMOUS_IO_BATCH,
        shuffle=True,
        shuffle_algo="py1e",
        shuffle_seed=cfg.seed,
        shuffle_block_size=8192,
        cache_limit=None,
        keep_zip=False,
    )
    return O45StreamingDataLoader(
        dataset,
        batch_size=ANONYMOUS_IO_BATCH,
        num_workers=num_workers,
        collate_fn=functools.partial(_collate_anonymous, stats=dict(stats)),
        persistent_workers=num_workers > 0,
        prefetch_factor=LOADER_PREFETCH_FACTOR if num_workers > 0 else None,
        pin_memory=True,
        in_order=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(cfg.seed),
    )


def make_professional_loader(
    cfg: TrainConfig,
    stats: Mapping[str, FeatureStats],
    *,
    allowed_splits: tuple[str, ...] = ("train",),
    num_workers: int = LOADER_WORKERS,
    shuffle: bool = True,
) -> O45StreamingDataLoader:
    """Build one equally mixed, stratified professional MDS pipeline."""
    dataset = SourceTaggedStreamingDataset(
        streams=_professional_streams(cfg),
        source_slugs=DEVELOPMENT_IDENTITIES,
        batch_size=PROFESSIONAL_INPUT_BATCH,
        batching_method="stratified",
        shuffle=shuffle,
        shuffle_algo="py1e",
        shuffle_seed=cfg.seed,
        shuffle_block_size=65536,
        cache_limit=None,
        keep_zip=False,
    )
    return O45StreamingDataLoader(
        dataset,
        batch_size=PROFESSIONAL_INPUT_BATCH,
        num_workers=num_workers,
        collate_fn=functools.partial(
            _collate_professional,
            stats=dict(stats),
            seed=cfg.seed,
            allowed_splits=allowed_splits,
        ),
        persistent_workers=num_workers > 0,
        prefetch_factor=LOADER_PREFETCH_FACTOR if num_workers > 0 else None,
        pin_memory=True,
        in_order=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(cfg.seed + 1),
    )


def _concat_contexts(contexts: Sequence[Context]) -> Context:
    if not contexts:
        raise ValueError("cannot concatenate an empty context sequence")
    names = contexts[0].features.keys()
    if any(context.features.keys() != names for context in contexts[1:]):
        raise ValueError("context feature sets do not match")
    return Context(
        features={name: torch.cat([context.features[name] for context in contexts]) for name in names},
        ctx_pad=torch.cat([context.ctx_pad for context in contexts]),
    )


def _context_row(context: Context, index: int) -> Context:
    return Context(
        features={name: value[index : index + 1].clone() for name, value in context.features.items()},
        ctx_pad=context.ctx_pad[index : index + 1].clone(),
    )


def _candidate_examples(batch: ProfessionalCandidates) -> tuple[ProfessionalExample, ...]:
    if batch.context is None:
        return ()
    if batch.context.batch != len(batch.metadata):
        raise ValueError("professional contexts and metadata are not aligned")
    return tuple(
        ProfessionalExample(_context_row(batch.context, index), metadata)
        for index, metadata in enumerate(batch.metadata)
    )


class PairLoader:
    """Join two MDS pipelines and own only experiment-level sampling state."""

    def __init__(
        self,
        cfg: TrainConfig,
        anonymous_loader: O45StreamingDataLoader,
        professional_loader: O45StreamingDataLoader,
        *,
        identities: Sequence[str] = DEVELOPMENT_IDENTITIES,
        anonymous_batches_per_update: int = ANONYMOUS_BATCHES_PER_UPDATE,
    ) -> None:
        self.cfg = cfg
        self.anonymous_loader = anonymous_loader
        self.professional_loader = professional_loader
        self.identities = tuple(identities)
        self.anonymous_batches_per_update = anonymous_batches_per_update
        self.rng = np.random.default_rng(cfg.seed)
        self.queues = {identity: deque[ProfessionalExample]() for identity in self.identities}
        self._anonymous_iterator = None
        self._professional_iterator = None
        self.last_metrics = LoaderMetrics(0.0, 0.0, self.cache_gib)

    @property
    def cache_gib(self) -> float:
        datasets = (
            cast(StreamingDataset, self.anonymous_loader.dataset),
            cast(StreamingDataset, self.professional_loader.dataset),
        )
        return sum(dataset.cache_usage for dataset in datasets) / 2**30

    def _next_anonymous(self) -> PairBatch:
        if self._anonymous_iterator is None:
            self._anonymous_iterator = iter(self.anonymous_loader)
        try:
            return next(self._anonymous_iterator)
        except StopIteration:
            self._anonymous_iterator = iter(self.anonymous_loader)
            return next(self._anonymous_iterator)

    def _next_professional(self) -> ProfessionalCandidates:
        if self._professional_iterator is None:
            self._professional_iterator = iter(self.professional_loader)
        try:
            return next(self._professional_iterator)
        except StopIteration:
            self._professional_iterator = iter(self.professional_loader)
            return next(self._professional_iterator)

    def _add_candidates(self, batch: ProfessionalCandidates) -> None:
        for example in _candidate_examples(batch):
            identity = example.metadata.identity
            if identity not in self.queues:
                raise ValueError(f"professional pipeline returned unexpected identity {identity!r}")
            queue = self.queues[identity]
            replay_id = example.metadata.replay_id
            if len(queue) >= PROFESSIONAL_QUEUE_LIMIT:
                continue
            if any(item.metadata.replay_id == replay_id for item in queue):
                continue
            queue.append(example)

    def _fill_professional_queues(self) -> None:
        required = self.cfg.professional_replays_per_identity
        attempts = 0
        while any(len(self.queues[identity]) < required for identity in self.identities):
            attempts += 1
            if attempts > 10_000:
                missing = {identity: len(queue) for identity, queue in self.queues.items() if len(queue) < required}
                raise RuntimeError(f"professional stream could not fill replay queues: {missing}")
            self._add_candidates(self._next_professional())

    def sample(self) -> PairBatch:
        """Assemble one exact anonymous/professional logical batch."""
        anonymous_started = time.monotonic()
        anonymous = [self._next_anonymous() for _ in range(self.anonymous_batches_per_update)]
        anonymous_wait = time.monotonic() - anonymous_started

        professional_started = time.monotonic()
        self._fill_professional_queues()
        professional_wait = time.monotonic() - professional_started
        selected = self.rng.choice(
            self.identities,
            self.cfg.professional_identities_per_batch,
            replace=False,
        )
        professional: list[ProfessionalExample] = []
        target_professional: list[ProfessionalExample] = []
        for identity_value in selected:
            identity = str(identity_value)
            examples = [self.queues[identity].popleft() for _ in range(self.cfg.professional_replays_per_identity)]
            metadata = [example.metadata for example in examples]
            pairing = professional_derangement(metadata)
            professional.extend(examples)
            target_professional.extend(examples[index] for index in pairing)

        online_contexts = [batch.online for batch in anonymous]
        online_contexts.extend(example.context for example in professional)
        target_contexts = [batch.target for batch in anonymous]
        target_contexts.extend(example.context for example in target_professional)
        online_metadata = tuple(item for batch in anonymous for item in batch.metadata.online)
        online_metadata += tuple(example.metadata for example in professional)
        target_metadata = tuple(item for batch in anonymous for item in batch.metadata.target)
        target_metadata += tuple(example.metadata for example in target_professional)
        mask = torch.zeros(len(online_metadata), dtype=torch.bool)
        mask[self.cfg.anonymous_pairs :] = True
        result = PairBatch(
            _concat_contexts(online_contexts),
            _concat_contexts(target_contexts),
            PairMetadata(online_metadata, target_metadata, mask),
        )
        if result.batch_size != self.cfg.logical_batch_size:
            raise RuntimeError(
                f"pair loader assembled {result.batch_size} pairs, expected {self.cfg.logical_batch_size}"
            )
        self.last_metrics = LoaderMetrics(anonymous_wait, professional_wait, self.cache_gib)
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": 1,
            "anonymous_mds": self.anonymous_loader.state_dict(),
            "professional_mds": self.professional_loader.state_dict(),
            "professional_queues": {identity: tuple(queue) for identity, queue in self.queues.items()},
            "rng": copy.deepcopy(self.rng.bit_generator.state),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") != 1:
            raise ValueError(f"unsupported pair loader schema {state.get('schema')!r}")
        if self._anonymous_iterator is not None or self._professional_iterator is not None:
            raise RuntimeError("pair loader state must be restored before iteration starts")
        anonymous_state = state.get("anonymous_mds")
        professional_state = state.get("professional_mds")
        if not isinstance(anonymous_state, dict) or not isinstance(professional_state, dict):
            raise TypeError("pair loader state must contain both MDS cursors")
        self.anonymous_loader.load_state_dict(anonymous_state)
        self.professional_loader.load_state_dict(professional_state)
        saved_queues = state.get("professional_queues")
        if not isinstance(saved_queues, dict) or set(saved_queues) != set(self.identities):
            raise ValueError("checkpoint professional queue identities do not match")
        restored: dict[str, deque[ProfessionalExample]] = {}
        for identity in self.identities:
            values = tuple(saved_queues[identity])
            replay_ids = [example.metadata.replay_id for example in values]
            if len(values) > PROFESSIONAL_QUEUE_LIMIT or len(replay_ids) != len(set(replay_ids)):
                raise ValueError(f"checkpoint queue for {identity} is invalid")
            restored[identity] = deque(values)
        self.queues = restored
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])

    def close(self) -> None:
        self.anonymous_loader.close()
        self.professional_loader.close()


def capture_rng_state() -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def save_checkpoint(
    path: Path,
    *,
    model: BYOL,
    optimizer: AdamW,
    cfg: TrainConfig,
    stats: Mapping[str, FeatureStats],
    step: int,
    pair_loader: PairLoader | None,
    wandb_id: str | None,
    uploader: BackgroundUploader | None = None,
) -> None:
    """Save every model, optimizer, schedule, RNG, and split state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "schema": 2,
        "online_encoder": model.online_encoder.state_dict(),
        "online_projector": model.online_projector.state_dict(),
        "predictor": model.predictor.state_dict(),
        "ema_encoder": model.target_encoder.state_dict(),
        "ema_projector": model.target_projector.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": {"name": "o45-explicit", "next_step": step + 1},
        "ema_schedule_position": step,
        "rng": capture_rng_state(),
        "pair_loader": None if pair_loader is None else pair_loader.state_dict(),
        "professional_split": {
            "sealed": SEALED_IDENTITIES,
            "development": DEVELOPMENT_IDENTITIES,
            "buckets": (80, 90, 100),
        },
        "config": asdict(cfg),
        "feature_statistics": {name: asdict(value) for name, value in stats.items()},
        "step": step,
        "wandb_id": wandb_id,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)
    print(f"[ckpt] saved {path}", flush=True)
    if uploader is not None:
        uploader.upload(path)


def load_checkpoint(
    path: Path,
    *,
    device: torch.device,
) -> tuple[BYOL, AdamW, TrainConfig, dict[str, FeatureStats], dict[str, Any]]:
    state = torch.load(path, map_location=device, weights_only=False)
    if state.get("schema") != 2:
        raise ValueError(f"unsupported O45 checkpoint schema {state.get('schema')!r}")
    cfg = TrainConfig(**state["config"])
    cfg.validate()
    model = BYOL(cfg, prefer_flex=device.type == "cuda").to(device)
    model.online_encoder.load_state_dict(state["online_encoder"])
    model.online_projector.load_state_dict(state["online_projector"])
    model.predictor.load_state_dict(state["predictor"])
    model.target_encoder.load_state_dict(state["ema_encoder"])
    model.target_projector.load_state_dict(state["ema_projector"])
    optimizer = make_optimizer(model, cfg)
    optimizer.load_state_dict(state["optimizer"])
    stats = {name: FeatureStats(**values) for name, values in state["feature_statistics"].items()}
    restore_rng_state(state["rng"])
    return model, optimizer, cfg, stats, state


def build_heldout_cache(
    cfg: TrainConfig,
    stats: Mapping[str, FeatureStats],
    *,
    num_workers: int = LOADER_WORKERS,
) -> HeldoutCache:
    """Collect fixed 16-replay gallery and query sets for each development identity."""
    loader = make_professional_loader(
        cfg,
        stats,
        allowed_splits=("gallery", "query"),
        num_workers=num_workers,
        shuffle=False,
    )
    selected = {split: {identity: [] for identity in DEVELOPMENT_IDENTITIES} for split in ("gallery", "query")}
    replay_ids = {split: {identity: set() for identity in DEVELOPMENT_IDENTITIES} for split in ("gallery", "query")}
    iterator = iter(loader)
    batches = 0
    try:
        while any(
            len(selected[split][identity]) < HELDOUT_REPLAYS_PER_IDENTITY
            for split in ("gallery", "query")
            for identity in DEVELOPMENT_IDENTITIES
        ):
            batches += 1
            if batches > 10_000:
                missing = {
                    f"{identity}/{split}": len(selected[split][identity])
                    for split in ("gallery", "query")
                    for identity in DEVELOPMENT_IDENTITIES
                    if len(selected[split][identity]) < HELDOUT_REPLAYS_PER_IDENTITY
                }
                raise RuntimeError(f"could not build the fixed O45 held-out cache: {missing}")
            try:
                batch = next(iterator)
            except StopIteration as error:
                raise RuntimeError("professional MDS epoch ended before the held-out cache was complete") from error
            for example in _candidate_examples(batch):
                identity = example.metadata.identity
                if identity is None:
                    raise RuntimeError("held-out professional window has no identity")
                split = professional_split(identity, example.metadata.replay_id)
                identity_replays = replay_ids[split][identity]
                identity_examples = selected[split][identity]
                if len(identity_examples) >= HELDOUT_REPLAYS_PER_IDENTITY:
                    continue
                if example.metadata.replay_id in identity_replays:
                    continue
                identity_replays.add(example.metadata.replay_id)
                identity_examples.append(example)
    finally:
        loader.close()
    gallery = tuple(example for identity in DEVELOPMENT_IDENTITIES for example in selected["gallery"][identity])
    query = tuple(example for identity in DEVELOPMENT_IDENTITIES for example in selected["query"][identity])
    return HeldoutCache(gallery, query)


def _encode_examples(
    encoder: PlayerEncoder,
    examples: Sequence[ProfessionalExample],
    *,
    device: torch.device,
    microbatch: int,
) -> np.ndarray:
    output = []
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    with torch.no_grad(), amp:
        for start in range(0, len(examples), microbatch):
            context = _concat_contexts([example.context for example in examples[start : start + microbatch]])
            output.append(encoder.normalized(context.to(device)).float().cpu().numpy())
    return np.concatenate(output)


def heldout_metrics(
    model: BYOL,
    cache: HeldoutCache,
    cfg: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    """Run the existing development-identity metrics on the fixed cache."""
    was_training = model.online_encoder.training
    model.online_encoder.eval()
    try:
        gallery_z = _encode_examples(
            model.online_encoder,
            cache.gallery,
            device=device,
            microbatch=cfg.target_microbatch,
        )
        query_z = _encode_examples(
            model.online_encoder,
            cache.query,
            device=device,
            microbatch=cfg.target_microbatch,
        )
    finally:
        model.online_encoder.train(was_training)

    gallery_metadata = [example.metadata for example in cache.gallery]
    query_metadata = [example.metadata for example in cache.query]
    gallery_identities = [str(item.identity) for item in gallery_metadata]
    query_identities = [str(item.identity) for item in query_metadata]
    gallery_replays = [item.replay_id for item in gallery_metadata]
    query_replays = [item.replay_id for item in query_metadata]
    values = {
        f"retrieval/{name}": value
        for name, value in retrieval_metrics(
            gallery_z,
            gallery_identities,
            gallery_replays,
            query_z,
            query_identities,
            query_replays,
        ).items()
    }
    values.update(
        {
            f"identification/{name}": value
            for name, value in knn_identification(
                gallery_z,
                gallery_identities,
                query_z,
                query_identities,
            ).items()
        }
    )
    values.update(
        {
            f"identification/{name}": value
            for name, value in linear_probe_metrics(
                gallery_z,
                gallery_identities,
                query_z,
                query_identities,
            ).items()
        }
    )
    values.update(
        {
            f"identification/{name}": value
            for name, value in prototype_identification(
                gallery_z,
                gallery_identities,
                gallery_replays,
                query_z,
                query_identities,
            ).items()
        }
    )

    all_z = np.concatenate((gallery_z, query_z))
    all_metadata = [*gallery_metadata, *query_metadata]
    query_indices = range(len(gallery_metadata), len(all_metadata))
    nuisance_query, nuisance_positive, nuisance_negative = select_nuisance_triplets(all_metadata, query_indices)
    nuisance = nuisance_controlled_metrics(
        all_z[nuisance_query],
        all_z[nuisance_positive],
        all_z[nuisance_negative],
        [all_metadata[index] for index in nuisance_query],
        [all_metadata[index] for index in nuisance_positive],
    )
    values.update({f"nuisance/{name}": value for name, value in nuisance.items()})
    normalized = _normalize_numpy(all_z)
    same_distance = 1.0 - np.sum(normalized[nuisance_query] * normalized[nuisance_positive], axis=1)
    different_distance = 1.0 - np.sum(normalized[nuisance_query] * normalized[nuisance_negative], axis=1)
    values["nuisance/distance_gap_lower_95"] = bootstrap_lower_bound(different_distance - same_distance)
    values["nuisance/triplet_accuracy_lower_95"] = bootstrap_lower_bound(
        (same_distance < different_distance).astype(np.float64),
        chance=0.5,
    )
    values.update({f"collapse/{name}": value for name, value in collapse_diagnostics(all_z).items()})
    distributions = distance_distributions(all_z, gallery_identities + query_identities)
    values["distance/same_player_mean"] = float(distributions["same_player"].mean())
    values["distance/different_player_mean"] = float(distributions["different_player"].mean())
    return values


def synthetic_context(cfg: TrainConfig, batch_size: int, seed: int = 0) -> Context:
    """Build a complete model-shaped context for CPU and CUDA smoke tests."""
    generator = torch.Generator().manual_seed(seed)
    features: dict[str, Tensor] = {}
    for prefix in BASE_PLAYER_PREFIXES:
        for name in FLOAT_FEATURES:
            features[f"{prefix}_{name}"] = torch.randn(
                batch_size,
                cfg.window_length,
                generator=generator,
            )
        for name, (vocab, _) in CAT_FEATURES.items():
            features[f"{prefix}_{name}"] = torch.randint(
                vocab,
                (batch_size, cfg.window_length),
                generator=generator,
            )
    for name in ACTION_CHANNELS:
        features[f"ego_{name}"] = torch.rand(batch_size, cfg.window_length, generator=generator)
    features["stage"] = torch.zeros(batch_size, cfg.window_length, dtype=torch.long)
    features["ego_character"] = torch.zeros(batch_size, cfg.window_length, dtype=torch.long)
    features["opp_character"] = torch.zeros(batch_size, cfg.window_length, dtype=torch.long)
    return Context(features, torch.zeros(batch_size, dtype=torch.long))


def synthetic_pair_batch(cfg: TrainConfig) -> PairBatch:
    online = synthetic_context(cfg, cfg.logical_batch_size, seed=1)
    target = synthetic_context(cfg, cfg.logical_batch_size, seed=2)
    metadata: list[WindowMetadata] = []
    target_metadata: list[WindowMetadata] = []
    for index in range(cfg.logical_batch_size):
        professional = index >= cfg.anonymous_pairs
        identity_index = (index - cfg.anonymous_pairs) // cfg.professional_replays_per_identity
        identity = DEVELOPMENT_IDENTITIES[identity_index] if professional else None
        character = (index - cfg.anonymous_pairs) % cfg.professional_replays_per_identity if professional else 0
        descriptor = np.full(DESCRIPTOR_DIM, index / cfg.logical_batch_size, dtype=np.float32)
        metadata.append(WindowMetadata(f"online-{index}", identity, character, 2, index % 5, descriptor))
        target_metadata.append(WindowMetadata(f"target-{index}", identity, character, 2, index % 5, descriptor + 0.01))
    mask = torch.arange(cfg.logical_batch_size) >= cfg.anonymous_pairs
    return PairBatch(online, target, PairMetadata(tuple(metadata), tuple(target_metadata), mask))


def train(cfg: TrainConfig, *, resume: str | None = None) -> None:
    """Train O45 from compact anonymous and professional replay streams."""
    cfg.validate()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("the full O45 run requires CUDA; use --cpu-smoke for a local contract test")
    if cfg.attention_backend != "varlen_flash":
        raise ValueError("CUDA O45 training requires attention_backend='varlen_flash'")
    if not varlen_flash_is_usable("cuda", cfg.window_length, cfg.n_heads, cfg.d_model // cfg.n_heads):
        raise RuntimeError("native variable-length FlashAttention is unavailable for the O45 geometry")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.set_float32_matmul_precision("high" if cfg.allow_tf32 else "highest")
    stats = load_consolidated_stats(Path(cfg.anonymous_data_root) / "stats.json")
    run_name = resume or make_run_name(
        Path(__file__).stem,
        f"byol-d{cfg.d_model}-L{cfg.n_layers}-H{cfg.n_heads}-z{cfg.representation_dim}-s{cfg.seed}",
        cfg.anonymous_data_root,
    )
    run_dir, _ = setup_run_dir(run_name)
    if resume is None:
        model = BYOL(cfg).to(device)
        optimizer = make_optimizer(model, cfg)
        start_step = 0
        resume_state = None
    else:
        checkpoint_path = download_latest(resume, run_dir)
        if checkpoint_path is None:
            raise FileNotFoundError(f"R2 has no latest checkpoint for run {resume!r}")
        model, optimizer, loaded_cfg, stats, resume_state = load_checkpoint(checkpoint_path, device=device)
        if loaded_cfg != cfg:
            raise ValueError("resume configuration differs from the requested configuration")
        start_step = int(resume_state["step"]) + 1
    heldout = build_heldout_cache(cfg, stats)
    pair_loader = PairLoader(
        cfg,
        make_anonymous_loader(cfg, stats),
        make_professional_loader(cfg, stats),
    )
    if resume_state is not None and resume_state["pair_loader"] is not None:
        pair_loader.load_state_dict(resume_state["pair_loader"])
    uploader = BackgroundUploader(run_name) if cfg.push_to_r2 else None
    wandb.init(
        project=cfg.wandb_project,
        name=run_name,
        id=None if resume_state is None else resume_state.get("wandb_id"),
        resume=None if resume_state is None else "must",
        mode=cfg.wandb_mode,
        config=asdict(cfg),
    )
    try:
        if start_step == 0:
            initial_eval = heldout_metrics(model, heldout, cfg, device)
            wandb.log({**{f"heldout/{name}": value for name, value in initial_eval.items()}, "global_step": 0})
            print(
                f"update 0: heldout gap={initial_eval['nuisance/distance_gap']:.4f} "
                f"rank={initial_eval['collapse/effective_rank']:.1f}",
                flush=True,
            )
        for step in range(start_step, cfg.max_updates):
            started = time.monotonic()
            batch = pair_loader.sample()
            metrics = train_step(model, batch, optimizer, cfg, step, device)
            completed_updates = step + 1
            values = {f"train/{name}": value for name, value in asdict(metrics).items()}
            values.update(
                {
                    "global_step": completed_updates,
                    "throughput/complete_update_s": time.monotonic() - started,
                    "throughput/anonymous_wait_s": pair_loader.last_metrics.anonymous_wait_s,
                    "throughput/professional_wait_s": pair_loader.last_metrics.professional_wait_s,
                    "system/cache_gib": pair_loader.last_metrics.cache_gib,
                }
            )
            if completed_updates in (512, 1024, cfg.max_updates):
                evaluation = heldout_metrics(model, heldout, cfg, device)
                values.update({f"heldout/{name}": value for name, value in evaluation.items()})
            wandb.log(values)
            if completed_updates in (1, 512, 1024) or completed_updates % 50 == 0:
                print(
                    f"update {completed_updates}: loss={metrics.loss:.4f} byol={metrics.byol_loss:.4f} "
                    f"triplet={metrics.triplet_loss:.4f} valid={metrics.valid_triplets}",
                    flush=True,
                )
            if completed_updates % cfg.checkpoint_every == 0:
                save_checkpoint(
                    run_dir / "latest.pt",
                    model=model,
                    optimizer=optimizer,
                    cfg=cfg,
                    stats=stats,
                    step=step,
                    pair_loader=pair_loader,
                    wandb_id=None if wandb.run is None else wandb.run.id,
                    uploader=uploader,
                )
        save_checkpoint(
            run_dir / "final.pt",
            model=model,
            optimizer=optimizer,
            cfg=cfg,
            stats=stats,
            step=cfg.max_updates - 1,
            pair_loader=pair_loader,
            wandb_id=None if wandb.run is None else wandb.run.id,
            uploader=uploader,
        )
    finally:
        pair_loader.close()
        if uploader is not None:
            uploader.close()
        wandb.finish()


@dataclass(frozen=True, slots=True)
class Args:
    cfg: TrainConfig = field(default_factory=TrainConfig)
    train: bool = False
    cpu_smoke: bool = False
    resume: str | None = None


def main(args: Args) -> None:
    if args.train == args.cpu_smoke:
        raise SystemExit("select exactly one of --train and --cpu-smoke")
    if args.train:
        train(args.cfg, resume=args.resume)
        return
    smoke_cfg = TrainConfig(
        d_model=32,
        n_layers=1,
        n_heads=4,
        logical_batch_size=4,
        anonymous_pairs=2,
        professional_pairs=2,
        professional_identities_per_batch=1,
        professional_replays_per_identity=2,
        online_microbatch=2,
        target_microbatch=2,
        warmup_updates=1,
        max_updates=2,
        attention_backend="dense_sdpa",
    )
    model = BYOL(smoke_cfg, prefer_flex=False)
    optimizer = make_optimizer(model, smoke_cfg)
    metrics = train_step(model, synthetic_pair_batch(smoke_cfg), optimizer, smoke_cfg, 0, torch.device("cpu"))
    print(asdict(metrics))


if __name__ == "__main__":
    main(tyro.cli(Args))
