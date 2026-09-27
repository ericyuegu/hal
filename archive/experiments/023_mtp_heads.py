"""Train the normalized independent-head MTP baseline."""

# %%
import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import concurrent.futures
import contextlib
import functools
import hashlib
import itertools
import json
import math
import subprocess
import sys
import time
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Mapping
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from dataclasses import replace
from pathlib import Path
from typing import Literal
from typing import TypeVar
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
from hal import streams
from hal.data.feature_stats import FeatureStats
from hal.eval.cross_stage import BOOTSTRAP_RESAMPLES
from hal.eval.cross_stage import PRIOR_SWEEP_SEED_STAGE
from hal.eval.cross_stage import MatchRow
from hal.eval.cross_stage import sweep_vs_cpu_prior_with_rows
from hal.eval.cross_stage import vs_cpu_metrics
from hal.eval.h2h import run_h2h
from hal.eval.harness import DEFAULT_START_RETRIES
from hal.eval.harness import default_session_cfg
from hal.eval.matchups import matchups_for_vs_cpu
from hal.eval.paired import summarize_paired
from hal.training import scoring
from hal.training.checkpoints import BackgroundUploader
from hal.training.checkpoints import download_latest
from hal.training.checkpoints import load_for_resume
from hal.training.checkpoints import save_checkpoint
from hal.training.closed_loop import RecedingHorizon
from hal.training.dataloader import VAL_L_CHUNK
from hal.training.dataloader import make_loader
from hal.training.ego_stats import load_consolidated_stats
from hal.training.features import A_DIM
from hal.training.features import ACTION_CHANNELS
from hal.training.features import BASE_ACTION_PROJECTION
from hal.training.features import BASE_PLAYER_PREFIXES
from hal.training.features import CAT_FEATURES
from hal.training.features import FLOAT_FEATURES
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
from hal.wire import mask_value

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_LN2 = math.log(2.0)

# Action-vector channel split (A_DIM=14): [0:6] sticks+triggers (continuous), [6:14] buttons {0,1}.
_N_CONT = 6
_N_BUTTONS = A_DIM - _N_CONT

# Per-frame input: all four players' gamestate concatenated in the feature dim.
_PLAYER_PREFIXES = BASE_PLAYER_PREFIXES
_INPUT_PROJECTION = BASE_ACTION_PROJECTION

# Output groups (fixed order; the canonical order of every per-group tensor and of the class-index
# columns quantize_groups emits) + their discrete vocab sizes from the scoring discretizers.
_GROUP_NAMES: tuple[str, ...] = ("buttons", "main_stick", "c_stick", "triggers")
_GROUP_VOCABS: tuple[int, ...] = (
    scoring.N_BUTTON_COMBOS,  # 256
    scoring.STICK_CLUSTER_CENTERS_MAIN.shape[0],  # 65
    scoring.STICK_CLUSTER_CENTERS_C.shape[0],  # 9
    scoring.TRIGGER_CENTERS.shape[0] ** 2,  # 25 (joint L*5 + R)
)
N_GROUPS = len(_GROUP_NAMES)
_BUTTONS_G, _MAIN_G, _C_G, _TRIG_G = range(N_GROUPS)
_GROUP_INDEX: dict[str, int] = {name: g for g, name in enumerate(_GROUP_NAMES)}
_GROUP_OFFSETS: tuple[int, ...] = tuple(itertools.accumulate((0,) + _GROUP_VOCABS))[:N_GROUPS]
A_VOCAB = sum(_GROUP_VOCABS)

_BUTTON_COUNTS_VERSION = 1

# Action-vector channels for the click=>trigger hygiene fix (digital L/R click => analog trigger = 1.0).
_TRIGGER_L_CH = ACTION_CHANNELS.index("trigger_l")
_TRIGGER_R_CH = ACTION_CHANNELS.index("trigger_r")
_BUTTON_L_CH = ACTION_CHANNELS.index("button_l")
_BUTTON_R_CH = ACTION_CHANNELS.index("button_r")


# %%
@dataclass
class TrainConfig:
    # GPT backbone
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 4
    # Zero uses full causal attention. Positive values enable the SWA ablation.
    attn_window: int = 0
    # Fail instead of using the slower dense attention path.
    require_flex: bool = False
    # Offset 1 is deployed. Other offsets are auxiliary targets.
    head_offsets: tuple[int, ...] = (1, 5, 9, 13)
    # Select independent linear, state-only MLP, or within-frame factored outputs.
    head_mode: Literal["linear", "state_mlp", "factored_mlp"] = "linear"
    # Set the MLP width as a multiple of d_model.
    action_mlp_ratio: int = 2
    # Set the feature width of each ancestor action embedding.
    action_condition_dim: int = 32
    # List the within-frame conditional order.
    action_group_order: tuple[str, ...] = ("c_stick", "triggers", "buttons", "main_stick")
    # Use private random streams for ancestor-sampled validation.
    factorization_diag_seed: int = 0
    # Set the number of ancestor samples per validation frame.
    factorization_diag_samples: int = 1
    # Total weight of the mean auxiliary-head loss. This stays fixed if head_offsets changes.
    aux_loss_weight: float = 1.0
    # Drop the complete ego action history for a training sample.
    history_dropout_p: float = 0.0
    # Weight action transitions inside each group loss. One disables this weighting.
    transition_loss_weight: float = 1.0
    # Character and stage embeddings use raw libmelee IDs.
    char_vocab: int = 32
    char_dim: int = 12
    stage_vocab: int = 32
    stage_dim: int = 4
    # Ranked v7 contains action states through 525. Older checkpoints used 512 rows.
    action_vocab: int = 1024
    # Closed-loop sampling temperature.
    decode_temp: float = 1.0
    # Group order: buttons, main stick, C-stick, triggers.
    decode_temps: tuple[float, float, float, float] | None = None
    decode_btn_support_min: int = 0
    decode_min_p: float = 0.0
    decode_click_trigger_fix: bool = False
    # Replan after this many contiguous output heads. One means per-frame control.
    exec_horizon: int = 1
    seed: int = 0
    L_ctx: int = 256
    # Effective batch size. The default processes 131,072 tokens in one forward pass.
    batch_size: int = 512
    grad_accum_steps: int = 1
    # Muon trains block matrices. AdamW trains the other parameters.
    muon_lr: float = 0.02
    adam_lr: float = 8.5e-4
    weight_decay: float = 0.01
    # Apply AdamW weight decay to output weights.
    head_weight_decay: bool = True
    # Shared warmup/cosine schedule and training duration.
    warmup_steps: int = 500
    max_steps: int = 16384
    amp_dtype: str = "bfloat16"  # "bfloat16" | "float32"
    allow_tf32: bool = True
    # Compile training forwards. Validation and diagnostics run eagerly.
    compile_trunk: bool = True
    # eval cadence
    val_every: int = 1024
    # Fixed validation examples. This must not change with the training batch geometry.
    val_n_samples: int = 1192
    # Examples used for per-head shared-trunk gradient comparisons.
    gradient_diagnostic_batch_size: int = 64
    # Validation-only rarity threshold for buttons.
    diagnostic_rare_button_count: int = 100
    # Closed-loop evaluation cadence and per-boot frame budget.
    eval_every: int = 4096
    eval_max_frames: int = 7200
    # These sample counts do not depend on host concurrency.
    eval_n_matchups: int = 32
    final_eval_n_matchups: int = 96
    eval_seed: int = 0
    # Maximum concurrent Dolphin boots. Matchup count is a separate fixed sample size.
    eval_max_parallel: int = 32
    # Overlap evaluation only when it has a separate GPU.
    eval_overlap_training: bool = False
    # Cast floating model parameters to FP16 once when closed-loop evaluation loads them.
    eval_fp16: bool = True
    # If an eval is still running at the next boundary, the trainer waits up to this bound and
    # then kills the worker.
    eval_timeout_seconds: float = 2700.0
    # Run the final mirrored H2H before the cloud instance exits. None disables it.
    final_h2h_reference_run: str | None = None
    final_h2h_reference_sha256: str | None = None
    final_h2h_reference_experiment: str = "experiments/023_mtp_heads.py"
    final_h2h_reference_label: str = "023-e0"
    final_h2h_self_label: str = "023-challenger"
    final_h2h_n_configs: int = 64
    # checkpointing
    ckpt_every: int = 2048
    # data
    data_root: str = "data/processed/ranked-anonymized-1/mds-policy-v7"
    compact_data: bool = True
    # The loader rejects rows from another MDS schema.
    mds_schema_version: int = 7
    # Full-dataset button counts used by decode support masking.
    button_combo_counts_path: str | None = None
    # Streaming dataset cache and shuffle geometry.
    cache_limit_gb: int = 128
    shuffle_block_size: int = 2000
    predownload: int = 512
    # Each replay supplies four windows, but no batch contains the same replay twice.
    windows_per_replay: int = 4
    reservoir_capacity: int = 4096
    val_split: str = "val"
    num_workers: int = 16
    prefetch_factor: int = 2
    # Prepare this many complete batches while the GPU trains.
    prefetch_batches: int = 1


def _model_tag(cfg: TrainConfig) -> str:
    offs = ".".join(str(o) for o in cfg.head_offsets)
    attention = "full" if cfg.attn_window == 0 else f"swa{cfg.attn_window}"
    if cfg.head_mode == "linear":
        head = "linear"
    elif cfg.head_mode == "state_mlp":
        head = f"state-mlp-r{cfg.action_mlp_ratio}"
    else:
        order = ".".join(name.replace("_stick", "") for name in cfg.action_group_order)
        head = f"factored-mlp-r{cfg.action_mlp_ratio}-c{cfg.action_condition_dim}-{order}"
    return (
        f"gpt-d{cfg.d_model}-L{cfg.n_layers}-h{cfg.n_heads}-Lc{cfg.L_ctx}-a{cfg.action_vocab}-"
        f"{attention}-recompute-o{offs}-{head}"
    )


def _micro_batch(cfg: TrainConfig) -> int:
    """Samples per forward/backward. ``batch_size`` is the effective batch, split into
    ``grad_accum_steps`` equal micro-batches; ``validate_config`` pins that they divide."""
    return cfg.batch_size // cfg.grad_accum_steps


def _eval_max_parallel(cfg: TrainConfig, n_matchups: int) -> int:
    """Concurrent Dolphin boots per wave; never changes the fixed statistical sample size."""
    return min(n_matchups, cfg.eval_max_parallel)


def _load_button_combo_counts(cfg: TrainConfig) -> Tensor | None:
    """Load and validate the dataset-scoped 256-way button support artifact, when configured."""
    if cfg.button_combo_counts_path is None:
        return None
    path = Path(cfg.button_combo_counts_path)
    data = json.loads(path.read_text())
    required = {"version", "data_root", "total_frames", "counts"}
    if set(data) != required:
        raise ValueError(f"{path}: expected exactly keys {sorted(required)}, got {sorted(data)}")
    if data["version"] != _BUTTON_COUNTS_VERSION:
        raise ValueError(
            f"{path}: button-count version {data['version']} != supported version {_BUTTON_COUNTS_VERSION}"
        )
    artifact_root = Path(data["data_root"]).resolve()
    cfg_root = Path(cfg.data_root).resolve()
    if artifact_root != cfg_root:
        raise ValueError(f"{path}: data_root {artifact_root} does not match configured data_root {cfg_root}")
    counts = data["counts"]
    if not isinstance(counts, list) or len(counts) != scoring.N_BUTTON_COMBOS:
        raise ValueError(f"{path}: counts must contain exactly {scoring.N_BUTTON_COMBOS} entries")
    if any(not isinstance(c, int) or c < 0 for c in counts):
        raise ValueError(f"{path}: every button count must be a non-negative integer")
    if not isinstance(data["total_frames"], int) or data["total_frames"] <= 0:
        raise ValueError(f"{path}: total_frames must be a positive integer")
    if sum(counts) != data["total_frames"]:
        raise ValueError(f"{path}: counts sum to {sum(counts)}, expected total_frames={data['total_frames']}")
    return torch.tensor(counts, dtype=torch.long)


# %%
@jaxtyped(typechecker=beartype)
def quantize_groups(
    main_centers: Float[Tensor, "n_main 2"],
    c_centers: Float[Tensor, "n_c 2"],
    trig_centers: Float[Tensor, " n_trig"],
    actions: Float[Tensor, "*batch d_action"],
) -> Int[Tensor, "*batch n_groups"]:
    """Raw ``A_DIM`` action vec → the four group class indices, in order
    ``(buttons, main_stick, c_stick, triggers)``. Inverse: ``dequantize_groups``."""
    cont, btn = actions[..., :_N_CONT], actions[..., _N_CONT:]
    buttons = scoring.buttons_to_combo(btn)
    main = scoring.nearest_cluster(cont[..., 0:2], main_centers)
    c = scoring.nearest_cluster(cont[..., 2:4], c_centers)
    trig = scoring.nearest_center(cont[..., 4:6], trig_centers)  # [*batch, 2]
    triggers = trig[..., 0] * trig_centers.shape[0] + trig[..., 1]
    return torch.stack([buttons, main, c, triggers], dim=-1)


@jaxtyped(typechecker=beartype)
def dequantize_groups(
    main_centers: Float[Tensor, "n_main 2"],
    c_centers: Float[Tensor, "n_c 2"],
    trig_centers: Float[Tensor, " n_trig"],
    idx: Int[Tensor, "*batch n_groups"],
) -> Float[Tensor, "*batch d_action"]:
    """Inverse of ``quantize_groups``: group class indices → raw ``A_DIM`` action vec
    (``[-1,1]`` sticks, ``[0,1]`` triggers, ``{0,1}`` buttons)."""
    n_trig = trig_centers.shape[0]
    btn = scoring.combo_to_buttons(idx[..., _BUTTONS_G])
    main = scoring.cluster_to_xy(idx[..., _MAIN_G], main_centers)
    c = scoring.cluster_to_xy(idx[..., _C_G], c_centers)
    tl = scoring.center_to_value(idx[..., _TRIG_G] // n_trig, trig_centers)
    tr = scoring.center_to_value(idx[..., _TRIG_G] % n_trig, trig_centers)
    trig = torch.stack([tl, tr], dim=-1)
    return torch.cat([main, c, trig, btn], dim=-1)


# %%
class IndependentHead(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, A_VOCAB)

    def logits(self, h: Tensor) -> dict[str, Tensor]:
        flat_logits = self.proj(h)
        return {
            name: flat_logits[..., start : start + vocab]
            for name, start, vocab in zip(_GROUP_NAMES, _GROUP_OFFSETS, _GROUP_VOCABS, strict=True)
        }


class StateMLPAdapter(nn.Module):
    def __init__(self, d_model: int, ratio: int, n_output_heads: int) -> None:
        super().__init__()
        hidden = ratio * d_model
        self.state_proj = nn.Linear(d_model, hidden)
        self.residual_projs = nn.ModuleList(
            [
                nn.ModuleDict({name: nn.Linear(hidden, vocab) for name, vocab in zip(_GROUP_NAMES, _GROUP_VOCABS)})
                for _ in range(n_output_heads)
            ]
        )
        for output_head in self.residual_projs:
            for module in cast(nn.ModuleDict, output_head).values():
                projection = cast(nn.Linear, module)
                nn.init.zeros_(projection.weight)
                nn.init.zeros_(projection.bias)

    def preactivation(self, h: Tensor) -> Tensor:
        return self.state_proj(h)

    def features(self, h: Tensor) -> Tensor:
        return F.silu(self.preactivation(h))

    def add(self, head_index: int, base: dict[str, Tensor], features: Tensor) -> dict[str, Tensor]:
        output_head = cast(nn.ModuleDict, self.residual_projs[head_index])
        return {name: logits + cast(nn.Linear, output_head[name])(features) for name, logits in base.items()}


class FactoredMLPAdapter(StateMLPAdapter):
    def __init__(
        self,
        d_model: int,
        ratio: int,
        n_output_heads: int,
        condition_dim: int,
        group_order: tuple[str, ...],
    ) -> None:
        super().__init__(d_model, ratio, n_output_heads)
        hidden = ratio * d_model
        self.group_order = group_order
        self.condition_dim = condition_dim
        self.action_embeddings = nn.ModuleDict(
            {name: nn.Embedding(_GROUP_VOCABS[_GROUP_INDEX[name]], condition_dim) for name in group_order[:-1]}
        )
        self.condition_projs = nn.ModuleDict(
            {
                name: nn.Linear(position * condition_dim, hidden, bias=False)
                for position, name in enumerate(group_order)
                if position > 0
            }
        )
        for projection in self.condition_projs.values():
            nn.init.zeros_(cast(nn.Linear, projection).weight)

    @staticmethod
    def _check_indices(name: str, indices: Tensor) -> None:
        vocab = _GROUP_VOCABS[_GROUP_INDEX[name]]
        valid = (indices >= 0).all() & (indices < vocab).all()
        message = f"{name} class index is out of range"
        if indices.device.type == "cuda":
            torch._assert_async(valid, message)
        else:
            torch._assert(valid, message)

    def _condition(
        self,
        group: str,
        prefix: dict[str, Tensor],
        embedded_prefix: dict[str, Tensor] | None,
    ) -> Tensor | None:
        position = self.group_order.index(group)
        earlier = self.group_order[:position]
        if tuple(prefix) != tuple(earlier):
            raise ValueError(f"{group} needs prefix {earlier}, got {tuple(prefix)}")
        if not earlier:
            return None
        if embedded_prefix is None:
            for name in earlier:
                self._check_indices(name, prefix[name])
            embedded_prefix = {name: self.action_embeddings[name](prefix[name]) for name in earlier}
        if tuple(embedded_prefix) != tuple(earlier):
            raise ValueError(f"{group} needs embedded prefix {earlier}, got {tuple(embedded_prefix)}")
        return torch.cat([embedded_prefix[name] for name in earlier], dim=-1)

    def group_logits(
        self,
        head_index: int,
        group: str,
        base: Tensor,
        state_preactivation: Tensor,
        prefix: dict[str, Tensor],
        embedded_prefix: dict[str, Tensor] | None = None,
    ) -> Tensor:
        condition = self._condition(group, prefix, embedded_prefix)
        preactivation = state_preactivation
        if condition is not None:
            preactivation = preactivation + self.condition_projs[group](condition)
        projection = cast(nn.Linear, cast(nn.ModuleDict, self.residual_projs[head_index])[group])
        return base + projection(F.silu(preactivation))

    def teacher_forced(
        self,
        head_index: int,
        base: dict[str, Tensor],
        state_preactivation: Tensor,
        target_indices: Tensor,
    ) -> dict[str, Tensor]:
        if target_indices.shape[-1] != N_GROUPS:
            raise ValueError(f"expected {N_GROUPS} action groups, got shape {tuple(target_indices.shape)}")
        target_by_name = {name: target_indices[..., _GROUP_INDEX[name]] for name in self.group_order}
        for name, indices in target_by_name.items():
            self._check_indices(name, indices)
        prefix: dict[str, Tensor] = {}
        embedded_prefix: dict[str, Tensor] = {}
        logits: dict[str, Tensor] = {}
        for name in self.group_order:
            logits[name] = self.group_logits(
                head_index,
                name,
                base[name],
                state_preactivation,
                prefix,
                embedded_prefix,
            )
            prefix[name] = target_by_name[name]
            if name != self.group_order[-1]:
                embedded_prefix[name] = self.action_embeddings[name](prefix[name])
        return logits


class GPT(nn.Module):
    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        if not cfg.decode_temp > 0:
            raise ValueError(f"decode_temp must be > 0, got {cfg.decode_temp}")
        if not 0.0 <= cfg.history_dropout_p <= 1.0:
            raise ValueError(f"history_dropout_p must be in [0, 1], got {cfg.history_dropout_p}")
        self.history_dropout_p = cfg.history_dropout_p
        offs = tuple(cfg.head_offsets)
        if not offs:
            raise ValueError("head_offsets must be non-empty")
        if any(o < 1 for o in offs):
            raise ValueError(f"head_offsets must all be >= 1, got {offs}")
        if len(set(offs)) != len(offs):
            raise ValueError(f"head_offsets must be unique, got {offs}")
        if 1 not in offs:
            raise ValueError(f"head_offsets must contain 1 (closed-loop next-frame decode), got {offs}")
        self.head_offsets = offs
        self.primary_head_idx = offs.index(1)

        # Share one embedding table for each game-state feature across players.
        self.cat_specs = {**CAT_FEATURES, "action": (cfg.action_vocab, CAT_FEATURES["action"][1])}
        self.cat_embeds = nn.ModuleDict(
            {name: nn.Embedding(vocab, dim) for name, (vocab, dim) in self.cat_specs.items()}
        )
        self.char_emb = nn.Embedding(cfg.char_vocab, cfg.char_dim)
        self.stage_emb = nn.Embedding(cfg.stage_vocab, cfg.stage_dim)
        per_player = len(FLOAT_FEATURES) * 2 + sum(dim for _, dim in CAT_FEATURES.values())
        d_in = len(_PLAYER_PREFIXES) * per_player + A_DIM + 2 * cfg.char_dim + cfg.stage_dim

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
        self.heads = nn.ModuleList([IndependentHead(cfg.d_model) for _ in offs])
        if cfg.head_mode == "linear":
            self.head_adapter = None
        elif cfg.head_mode == "state_mlp":
            self.head_adapter = StateMLPAdapter(cfg.d_model, cfg.action_mlp_ratio, len(offs))
        else:
            self.head_adapter = FactoredMLPAdapter(
                cfg.d_model,
                cfg.action_mlp_ratio,
                len(offs),
                cfg.action_condition_dim,
                cfg.action_group_order,
            )

        self.register_buffer("main_centers", scoring.STICK_CLUSTER_CENTERS_MAIN.clone())
        self.register_buffer("c_centers", scoring.STICK_CLUSTER_CENTERS_C.clone())
        self.register_buffer("trig_centers", scoring.TRIGGER_CENTERS.clone())
        # A negative count means the full-dataset count is unavailable.
        self.register_buffer("button_combo_counts", torch.full((scoring.N_BUTTON_COMBOS,), -1, dtype=torch.long))
        self._btn_support_dead_cache: dict[tuple[int, torch.device], Tensor] = {}

    def group_logits(self, h: Tensor, head_index: int, target_indices: Tensor | None = None) -> dict[str, Tensor]:
        targets = None if target_indices is None else (target_indices,)
        return self.group_logits_many(h, (head_index,), targets)[0]

    def group_logits_many(
        self,
        h: Tensor,
        head_indices: tuple[int, ...],
        target_indices: tuple[Tensor, ...] | None = None,
    ) -> list[dict[str, Tensor]]:
        base = [cast(IndependentHead, self.heads[index]).logits(h) for index in head_indices]
        if self.head_adapter is None:
            return base
        preactivation = self.head_adapter.preactivation(h)
        if isinstance(self.head_adapter, FactoredMLPAdapter):
            if target_indices is None or len(target_indices) != len(head_indices):
                raise ValueError("factored logits need one target-index tensor per output head")
            return [
                self.head_adapter.teacher_forced(head_index, values, preactivation, targets)
                for head_index, values, targets in zip(head_indices, base, target_indices, strict=True)
            ]
        features = F.silu(preactivation)
        return [
            self.head_adapter.add(head_index, values, features)
            for head_index, values in zip(head_indices, base, strict=True)
        ]

    def base_and_group_logits(
        self,
        h: Tensor,
        target_indices: tuple[Tensor, ...] | None = None,
    ) -> tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]:
        base = [cast(IndependentHead, head).logits(h) for head in self.heads]
        if self.head_adapter is None:
            return base, base
        preactivation = self.head_adapter.preactivation(h)
        if isinstance(self.head_adapter, FactoredMLPAdapter):
            if target_indices is None or len(target_indices) != len(base):
                raise ValueError("factored logits need one target-index tensor per output head")
            logits = [
                self.head_adapter.teacher_forced(index, values, preactivation, targets)
                for index, (values, targets) in enumerate(zip(base, target_indices, strict=True))
            ]
            return base, logits
        features = F.silu(preactivation)
        logits = [self.head_adapter.add(index, values, features) for index, values in enumerate(base)]
        return base, logits

    def all_group_logits(self, h: Tensor, target_indices: tuple[Tensor, ...] | None = None) -> list[dict[str, Tensor]]:
        return self.base_and_group_logits(h, target_indices)[1]

    def _per_player_features(self, features: dict[str, Tensor], prefix: str) -> Tensor:
        ref = features[f"{prefix}_position_x"]
        B, L = ref.shape
        device = ref.device
        parts: list[Tensor] = [features[f"{prefix}_{feat}"][..., None] for feat in FLOAT_FEATURES]
        for feat in FLOAT_FEATURES:
            mk = f"{prefix}_{feat}_mask"
            # An absent sidecar reads as zeros, in the dtype of the block it is concatenated with
            # (fp16 under the eval cast, fp32 in training).
            parts.append(
                features[mk][..., None] if mk in features else torch.zeros(B, L, 1, device=device, dtype=ref.dtype)
            )
        for name, (vocab, _) in self.cat_specs.items():
            parts.append(self.cat_embeds[name](features[f"{prefix}_{name}"].clamp(0, vocab - 1)))
        return torch.cat(parts, dim=-1)

    def _context_tokens(self, features: dict[str, Tensor]) -> Float[Tensor, "B L_ctx d_model"]:
        parts = [self._per_player_features(features, p) for p in _PLAYER_PREFIXES]
        # Build a new controller-history tensor so target features stay unchanged.
        ego_hist = torch.cat([features[f"ego_{ch}"][..., None] for ch in ACTION_CHANNELS], dim=-1)
        if self.training and self.history_dropout_p > 0.0:
            keep = (torch.rand(ego_hist.shape[0], device=ego_hist.device) >= self.history_dropout_p).to(ego_hist.dtype)
            ego_hist = ego_hist * keep[:, None, None]
        parts.append(ego_hist)
        parts.append(self.char_emb(features["ego_character"].clamp(0, self.char_emb.num_embeddings - 1)))
        parts.append(self.char_emb(features["opp_character"].clamp(0, self.char_emb.num_embeddings - 1)))
        parts.append(self.stage_emb(features["stage"].clamp(0, self.stage_emb.num_embeddings - 1)))
        return self.ctx_proj(torch.cat(parts, dim=-1))

    def forward(self, features: dict[str, Tensor], ctx_pad: Int[Tensor, " B"]) -> Float[Tensor, "B L_ctx d_model"]:
        return self.trunk(self._context_tokens(features), ctx_pad)


# %%
def _quantize(model: GPT, actions: Tensor) -> Tensor:
    return quantize_groups(model.main_centers, model.c_centers, model.trig_centers, actions)


def _dequantize(model: GPT, idx: Tensor) -> Tensor:
    return dequantize_groups(model.main_centers, model.c_centers, model.trig_centers, idx)


def _multi_offset_targets(ctx: Context, target: Tensor, offsets: tuple[int, ...]) -> tuple[dict[int, Tensor], Tensor]:
    """Return target actions at each offset and the shared non-padding mask."""
    a_full = torch.cat([stack_actions(ctx.features), target], dim=1)  # [B, L_ctx + max_off, A_DIM]
    L_ctx = a_full.size(1) - target.size(1)
    pos = torch.arange(L_ctx, device=a_full.device)
    valid = pos[None, :] >= ctx.ctx_pad[:, None]  # [B, L_ctx], shared by all offsets
    targets = {o: a_full[:, o : o + L_ctx] for o in offsets}  # each [B, L_ctx, A_DIM]
    return targets, valid


def group_nll(logits: dict[str, Tensor], tgt_idx: Tensor, valid: Tensor) -> dict[str, Tensor]:
    """Return one categorical NLL vector for each action group."""
    flat_valid = valid.reshape(-1)
    out: dict[str, Tensor] = {}
    for g, name in enumerate(_GROUP_NAMES):
        lg = logits[name].reshape(-1, logits[name].shape[-1])[flat_valid]
        out[name] = F.cross_entropy(lg, tgt_idx[..., g].reshape(-1)[flat_valid], reduction="none")
    return out


@dataclass(frozen=True, slots=True)
class LossParts:
    nll: dict[tuple[int, str], Tensor]
    transition: dict[tuple[int, str], Tensor]
    valid: Bool[Tensor, "B L_ctx"]


def action_loss(model: GPT, batch: TrainBatch) -> LossParts:
    """Return aligned NLL and transition vectors for every output head."""
    ctx = batch.context
    max_offset = max(model.head_offsets)
    if batch.target.size(1) < max_offset:
        raise ValueError(
            f"target chunk has {batch.target.size(1)} frames, but max(head_offsets)={max_offset}; "
            "the target horizon must cover every prediction head"
        )
    h = model(ctx.features, ctx.ctx_pad)  # [B, L_ctx, d_model]
    a_full = torch.cat([stack_actions(ctx.features), batch.target], dim=1)  # [B, L_ctx + max_off, A_DIM]
    q_full = _quantize(model, a_full)  # [B, L_ctx + max_off, n_groups]
    L_ctx = a_full.size(1) - batch.target.size(1)
    pos = torch.arange(L_ctx, device=a_full.device)
    valid = pos[None, :] >= ctx.ctx_pad[:, None]  # [B, L_ctx], shared by all offsets
    flat_valid = valid.reshape(-1)
    bnd_full = scoring.transition_mask(q_full)  # [B, L_ctx + max_off - 1, n_groups]; pos t = (q[t+1] != q[t])
    nll: dict[tuple[int, str], Tensor] = {}
    trans: dict[tuple[int, str], Tensor] = {}
    target_indices = tuple(q_full[:, o : o + L_ctx] for o in model.head_offsets)
    all_logits = model.all_group_logits(h, target_indices)
    for hi, o in enumerate(model.head_offsets):
        tgt_idx = target_indices[hi]
        logits = {name: lg.float() for name, lg in all_logits[hi].items()}
        bnd_o = bnd_full[:, o - 1 : o - 1 + L_ctx]  # transition at i iff q[i+o] != q[i+o-1]
        gnll = group_nll(logits, tgt_idx, valid)
        for g, name in enumerate(_GROUP_NAMES):
            nll[(o, name)] = gnll[name]
            trans[(o, name)] = bnd_o[..., g].reshape(-1)[flat_valid]
    return LossParts(nll=nll, transition=trans, valid=valid)


def _btn_support_dead(model: GPT, min_count: int, device: torch.device) -> Tensor:
    """``[N_BUTTON_COMBOS]`` bool mask, True on button combos with fewer than ``min_count`` train
    frames according to the checkpoint's dataset-scoped counts. Cached per ``(min_count, device)`` so
    closed-loop decode builds it once, never per frame."""
    counts = model.button_combo_counts
    if bool((counts < 0).any()):
        raise ValueError(
            "decode_btn_support_min > 0 requires a validated dataset-scoped button-count artifact; "
            "set TrainConfig.button_combo_counts_path when training a new checkpoint"
        )
    dead = model._btn_support_dead_cache.get((min_count, device))
    if dead is None:
        counts = counts.to(device)
        dead = counts < min_count
        if bool(dead.all()):
            raise ValueError(
                f"btn_support_min={min_count} masks every button combo (max train count is {int(counts.max())})"
            )
        model._btn_support_dead_cache[(min_count, device)] = dead
    return dead


def _resolve_decode_args(
    temp: float,
    temps: tuple[float, float, float, float] | None,
    btn_support_min: int,
    min_p: float,
    argmax: bool,
) -> tuple[float, ...]:
    """Validate the shared decode knobs and resolve the per-group temperatures (buttons, main_stick, c_stick,
    triggers): ``temps`` overrides the scalar ``temp`` when given. Fails loud on a wrong-length ``temps``, a
    negative support floor, a min-p outside [0, 1], or a non-positive sampling temperature."""
    if temps is not None and len(temps) != N_GROUPS:
        raise ValueError(f"decode temps must have {N_GROUPS} entries (one per group), got {len(temps)}")
    if btn_support_min < 0:
        raise ValueError(f"decode btn_support_min must be >= 0, got {btn_support_min}")
    if not math.isfinite(min_p) or not 0.0 <= min_p <= 1.0:
        raise ValueError(f"decode min_p must be in [0, 1], got {min_p}")
    group_temps = (temp,) * N_GROUPS if temps is None else temps
    if not argmax and any(not math.isfinite(t) or t <= 0 for t in group_temps):
        raise ValueError(f"decode temperatures must be > 0, got {group_temps}")
    return group_temps


def _sample_group_indices(
    logits: dict[str, Tensor],
    *,
    group_temps: tuple[float, ...],
    btn_dead: Tensor | None,
    min_p: float,
    argmax: bool,
    gen: torch.Generator | None,
    uniforms: Callable[[str], Tensor] | None = None,
) -> Int[Tensor, "B n_groups"]:
    picks: list[Tensor] = []
    for group, name in enumerate(_GROUP_NAMES):
        group_logits = logits[name].float()
        if btn_dead is not None and name == "buttons":
            group_logits = group_logits.masked_fill(btn_dead, float("-inf"))
        pick = _sample_categorical(
            group_logits,
            temperature=group_temps[group],
            min_p=min_p,
            argmax=argmax,
            gen=gen,
            uniform=None if uniforms is None else uniforms(name),
        )
        picks.append(pick)
    return torch.stack(picks, dim=-1)


def _sample_categorical(
    logits: Tensor,
    *,
    temperature: float,
    min_p: float,
    argmax: bool,
    gen: torch.Generator | None,
    uniform: Tensor | None = None,
) -> Tensor:
    if argmax:
        return logits.argmax(-1)
    probs = F.softmax(logits / temperature, dim=-1)
    if min_p > 0:
        probs = probs * (probs >= min_p * probs.amax(dim=-1, keepdim=True))
        probs = probs / probs.sum(dim=-1, keepdim=True)
    if uniform is None:
        return torch.multinomial(probs, 1, generator=gen).squeeze(-1)
    if uniform.shape != probs.shape[:-1]:
        raise ValueError(f"uniform shape {tuple(uniform.shape)} does not match logits {tuple(probs.shape)}")
    uniform = uniform.to(device=probs.device, dtype=probs.dtype)
    return (probs.cumsum(-1) < uniform.unsqueeze(-1)).sum(-1).clamp_max(probs.shape[-1] - 1)


def _sample_action(
    model: GPT,
    head_index: int,
    h: Tensor,
    *,
    group_temps: tuple[float, ...],
    btn_support_min: int,
    min_p: float,
    click_trigger_fix: bool,
    argmax: bool,
    gen: torch.Generator | None,
) -> Float[Tensor, "B d_action"]:
    if isinstance(model.head_adapter, FactoredMLPAdapter):
        offset = model.head_offsets[head_index]
        return chunk_from_hidden(
            model,
            h,
            (offset,),
            group_temps=group_temps,
            btn_support_min=btn_support_min,
            min_p=min_p,
            click_trigger_fix=click_trigger_fix,
            argmax=argmax,
            gen=gen,
        )[:, 0]
    return _sample_action_from_logits(
        model,
        model.group_logits(h, head_index),
        group_temps=group_temps,
        btn_support_min=btn_support_min,
        min_p=min_p,
        click_trigger_fix=click_trigger_fix,
        argmax=argmax,
        gen=gen,
    )


def _sample_action_from_logits(
    model: GPT,
    logits: dict[str, Tensor],
    *,
    group_temps: tuple[float, ...],
    btn_support_min: int,
    min_p: float,
    click_trigger_fix: bool,
    argmax: bool,
    gen: torch.Generator | None,
    uniforms: Callable[[str], Tensor] | None = None,
) -> Float[Tensor, "B d_action"]:
    device = next(iter(logits.values())).device
    dead = _btn_support_dead(model, btn_support_min, device) if btn_support_min >= 1 else None
    idx = _sample_group_indices(
        logits,
        group_temps=group_temps,
        btn_dead=dead,
        min_p=min_p,
        argmax=argmax,
        gen=gen,
        uniforms=uniforms,
    )
    return _indices_to_action(model, idx, click_trigger_fix)


@torch.no_grad()
def decode(
    model: GPT,
    ctx: Context,
    *,
    temp: float = 1.0,
    temps: tuple[float, float, float, float] | None = None,
    btn_support_min: int = 0,
    min_p: float = 0.0,
    click_trigger_fix: bool = False,
    argmax: bool = False,
    gen: torch.Generator | None = None,
) -> Float[Tensor, "B 1 d_action"]:
    group_temps = _resolve_decode_args(temp, temps, btn_support_min, min_p, argmax)
    h = model(ctx.features, ctx.ctx_pad)[:, -1]  # [B, d_model]
    a = chunk_from_hidden(
        model,
        h,
        (1,),
        group_temps=group_temps,
        btn_support_min=btn_support_min,
        min_p=min_p,
        click_trigger_fix=click_trigger_fix,
        argmax=argmax,
        gen=gen,
    )
    return a


@torch.no_grad()
def decode_chunk(
    model: GPT,
    ctx: Context,
    offsets: tuple[int, ...],
    *,
    temp: float = 1.0,
    temps: tuple[float, float, float, float] | None = None,
    btn_support_min: int = 0,
    min_p: float = 0.0,
    click_trigger_fix: bool = False,
    argmax: bool = False,
    gen: torch.Generator | None = None,
) -> Float[Tensor, "B s d_action"]:
    """Sample the requested offset heads after one backbone forward."""
    group_temps = _resolve_decode_args(temp, temps, btn_support_min, min_p, argmax)
    h = model(ctx.features, ctx.ctx_pad)[:, -1]  # [B, d_model]
    return chunk_from_hidden(
        model,
        h,
        offsets,
        group_temps=group_temps,
        btn_support_min=btn_support_min,
        min_p=min_p,
        click_trigger_fix=click_trigger_fix,
        argmax=argmax,
        gen=gen,
    )


@torch.no_grad()
def chunk_from_hidden(
    model: GPT,
    h: Float[Tensor, "B d_model"],
    offsets: tuple[int, ...],
    *,
    group_temps: tuple[float, ...],
    btn_support_min: int = 0,
    min_p: float = 0.0,
    click_trigger_fix: bool = False,
    argmax: bool = False,
    gen: torch.Generator | None = None,
    uniforms: Callable[[str], Tensor] | None = None,
) -> Float[Tensor, "B s d_action"]:
    """Sample requested offset heads from an existing hidden state."""
    head_indices = tuple(model.head_offsets.index(offset) for offset in offsets)
    if isinstance(model.head_adapter, FactoredMLPAdapter):
        base_by_head = [cast(IndependentHead, model.heads[index]).logits(h) for index in head_indices]
        state_preactivation = model.head_adapter.preactivation(h)
        dead = _btn_support_dead(model, btn_support_min, h.device) if btn_support_min >= 1 else None
        sampled_actions: list[Tensor] = []
        for head_index, base in zip(head_indices, base_by_head, strict=True):
            prefix: dict[str, Tensor] = {}
            embedded_prefix: dict[str, Tensor] = {}
            picks: dict[str, Tensor] = {}
            for name in model.head_adapter.group_order:
                logits = model.head_adapter.group_logits(
                    head_index,
                    name,
                    base[name],
                    state_preactivation,
                    prefix,
                    embedded_prefix,
                ).float()
                if dead is not None and name == "buttons":
                    logits = logits.masked_fill(dead, float("-inf"))
                group = _GROUP_INDEX[name]
                pick = _sample_categorical(
                    logits,
                    temperature=group_temps[group],
                    min_p=min_p,
                    argmax=argmax,
                    gen=gen,
                    uniform=None if uniforms is None else uniforms(name),
                )
                prefix[name] = pick
                if name != model.head_adapter.group_order[-1]:
                    embedded_prefix[name] = model.head_adapter.action_embeddings[name](pick)
                picks[name] = pick
            indices = torch.stack([picks[name] for name in _GROUP_NAMES], dim=-1)
            sampled_actions.append(_indices_to_action(model, indices, click_trigger_fix))
        return torch.stack(sampled_actions, dim=1)

    logits_by_head = model.group_logits_many(h, head_indices)
    actions: list[Tensor] = []
    for logits in logits_by_head:
        actions.append(
            _sample_action_from_logits(
                model,
                logits,
                group_temps=group_temps,
                btn_support_min=btn_support_min,
                min_p=min_p,
                click_trigger_fix=click_trigger_fix,
                argmax=argmax,
                gen=gen,
                uniforms=uniforms,
            )
        )
    return torch.stack(actions, dim=1)  # [B, s, A_DIM]


def _indices_to_action(model: GPT, indices: Tensor, click_trigger_fix: bool) -> Tensor:
    action = _dequantize(model, indices)
    if click_trigger_fix:
        action[..., _TRIGGER_L_CH] = torch.where(
            action[..., _BUTTON_L_CH] > 0.5,
            1.0,
            action[..., _TRIGGER_L_CH],
        )
        action[..., _TRIGGER_R_CH] = torch.where(
            action[..., _BUTTON_R_CH] > 0.5,
            1.0,
            action[..., _TRIGGER_R_CH],
        )
    return action


def _exec_horizon_offsets(head_offsets: tuple[int, ...], s: int) -> tuple[int, ...]:
    if s < 1:
        raise ValueError(f"exec_horizon must be >= 1, got {s}")
    required = tuple(range(1, s + 1))
    missing = [o for o in required if o not in head_offsets]
    if missing:
        raise ValueError(
            f"exec_horizon={s} needs head_offsets to contain the contiguous prefix {required}; "
            f"missing {missing} in {head_offsets}"
        )
    return required


def validate_config(cfg: TrainConfig, *, has_button_combo_counts: bool) -> None:
    positive_ints = {
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "L_ctx": cfg.L_ctx,
        "batch_size": cfg.batch_size,
        "grad_accum_steps": cfg.grad_accum_steps,
        "max_steps": cfg.max_steps,
        "exec_horizon": cfg.exec_horizon,
        "char_vocab": cfg.char_vocab,
        "char_dim": cfg.char_dim,
        "stage_vocab": cfg.stage_vocab,
        "stage_dim": cfg.stage_dim,
        "action_vocab": cfg.action_vocab,
        "action_mlp_ratio": cfg.action_mlp_ratio,
        "action_condition_dim": cfg.action_condition_dim,
        "factorization_diag_samples": cfg.factorization_diag_samples,
        "val_n_samples": cfg.val_n_samples,
        "gradient_diagnostic_batch_size": cfg.gradient_diagnostic_batch_size,
        "diagnostic_rare_button_count": cfg.diagnostic_rare_button_count,
        "eval_n_matchups": cfg.eval_n_matchups,
        "eval_max_parallel": cfg.eval_max_parallel,
        "eval_max_frames": cfg.eval_max_frames,
        "windows_per_replay": cfg.windows_per_replay,
        "predownload": cfg.predownload,
        "reservoir_capacity": cfg.reservoir_capacity,
        "shuffle_block_size": cfg.shuffle_block_size,
        "prefetch_factor": cfg.prefetch_factor,
        "cache_limit_gb": cfg.cache_limit_gb,
        "mds_schema_version": cfg.mds_schema_version,
    }
    for name, value in positive_ints.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if not isinstance(cfg.prefetch_batches, int) or isinstance(cfg.prefetch_batches, bool):
        raise ValueError(f"prefetch_batches must be an integer, got {cfg.prefetch_batches!r}")
    if cfg.prefetch_batches < 0:
        raise ValueError(f"prefetch_batches must be non-negative, got {cfg.prefetch_batches}")
    if cfg.batch_size % cfg.grad_accum_steps:
        raise ValueError(
            f"batch_size={cfg.batch_size} is the EFFECTIVE batch and must divide into "
            f"grad_accum_steps={cfg.grad_accum_steps} equal micro-batches"
        )
    if cfg.compact_data and cfg.reservoir_capacity < 2 * _micro_batch(cfg):
        raise ValueError(
            f"reservoir_capacity={cfg.reservoir_capacity} must be at least twice the micro-batch "
            f"size {_micro_batch(cfg)} to enforce one-batch replay cooldown"
        )
    if not isinstance(cfg.attn_window, int) or isinstance(cfg.attn_window, bool) or cfg.attn_window < 0:
        raise ValueError(f"attn_window must be a non-negative integer (0 = full context), got {cfg.attn_window!r}")
    if cfg.final_h2h_reference_run is not None:
        if not cfg.final_h2h_reference_run:
            raise ValueError("final_h2h_reference_run must be a run name or None, not an empty string")
        if cfg.final_h2h_n_configs < 1:
            raise ValueError(f"final_h2h_n_configs must be >= 1, got {cfg.final_h2h_n_configs}")
        if cfg.final_h2h_self_label == cfg.final_h2h_reference_label:
            raise ValueError(f"h2h labels must differ, got {cfg.final_h2h_self_label!r} twice")
        expected_sha = cfg.final_h2h_reference_sha256
        if expected_sha is not None and (
            len(expected_sha) != 64 or any(character not in "0123456789abcdef" for character in expected_sha.lower())
        ):
            raise ValueError("final_h2h_reference_sha256 must be a 64-character hexadecimal digest or None")
        exp = cfg.final_h2h_reference_experiment
        if exp.endswith(".py") and not Path(exp).exists():
            raise ValueError(f"final_h2h_reference_experiment does not exist: {exp}")
    elif cfg.final_h2h_reference_sha256 is not None:
        raise ValueError("final_h2h_reference_sha256 requires final_h2h_reference_run")
    if cfg.d_model % cfg.n_heads != 0:
        raise ValueError(f"d_model={cfg.d_model} must be divisible by n_heads={cfg.n_heads}")
    if cfg.head_mode not in ("linear", "state_mlp", "factored_mlp"):
        raise ValueError(f"head_mode must be 'linear', 'state_mlp', or 'factored_mlp', got {cfg.head_mode!r}")
    order = tuple(cfg.action_group_order)
    if len(order) != N_GROUPS or set(order) != set(_GROUP_NAMES):
        raise ValueError(f"action_group_order must be a permutation of {_GROUP_NAMES}, got {order}")
    head_dim = cfg.d_model // cfg.n_heads
    if head_dim % 2:
        raise ValueError(f"rotary attention head_dim=d_model/n_heads={head_dim} must be even")
    offsets = tuple(cfg.head_offsets)
    if not offsets or any(not isinstance(o, int) or isinstance(o, bool) or o < 1 for o in offsets):
        raise ValueError(f"head_offsets must be non-empty positive integers, got {offsets}")
    if len(set(offsets)) != len(offsets):
        raise ValueError(f"head_offsets must be unique, got {offsets}")
    if 1 not in offsets:
        raise ValueError(f"head_offsets must contain 1 (the deployed next-frame head), got {offsets}")
    if max(offsets) > VAL_L_CHUNK:
        raise ValueError(f"max(head_offsets)={max(offsets)} exceeds frozen VAL_L_CHUNK={VAL_L_CHUNK}")
    _exec_horizon_offsets(offsets, cfg.exec_horizon)
    finite_nonnegative = {
        "aux_loss_weight": cfg.aux_loss_weight,
        "weight_decay": cfg.weight_decay,
    }
    for name, value in finite_nonnegative.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
    finite_positive = {
        "muon_lr": cfg.muon_lr,
        "adam_lr": cfg.adam_lr,
        "eval_timeout_seconds": cfg.eval_timeout_seconds,
    }
    for name, value in finite_positive.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    if not math.isfinite(cfg.transition_loss_weight) or cfg.transition_loss_weight <= 0:
        raise ValueError(f"transition_loss_weight must be finite and > 0, got {cfg.transition_loss_weight!r}")
    if not math.isfinite(cfg.history_dropout_p) or not 0.0 <= cfg.history_dropout_p <= 1.0:
        raise ValueError(f"history_dropout_p must be in [0, 1], got {cfg.history_dropout_p!r}")
    if not isinstance(cfg.warmup_steps, int) or isinstance(cfg.warmup_steps, bool) or cfg.warmup_steps < 0:
        raise ValueError(f"warmup_steps must be a non-negative integer, got {cfg.warmup_steps!r}")
    if cfg.warmup_steps > cfg.max_steps:
        raise ValueError(f"warmup_steps={cfg.warmup_steps} exceeds max_steps={cfg.max_steps}")
    # final_eval_n_matchups = 0 skips the end-of-training closed-loop eval (an eval-incapable box).
    for name in ("val_every", "eval_every", "ckpt_every", "num_workers", "final_eval_n_matchups"):
        value = getattr(cfg, name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    for name in ("seed", "eval_seed", "factorization_diag_seed"):
        value = getattr(cfg, name)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer, got {value!r}")
    if not isinstance(cfg.decode_btn_support_min, int) or isinstance(cfg.decode_btn_support_min, bool):
        raise ValueError(f"decode_btn_support_min must be a non-negative integer, got {cfg.decode_btn_support_min!r}")
    if not cfg.data_root or not cfg.val_split:
        raise ValueError("data_root and val_split must be non-empty")
    if cfg.amp_dtype not in ("bfloat16", "float32"):
        raise ValueError(f"amp_dtype must be 'bfloat16' or 'float32', got {cfg.amp_dtype!r}")
    _resolve_decode_args(
        cfg.decode_temp,
        cfg.decode_temps,
        cfg.decode_btn_support_min,
        cfg.decode_min_p,
        argmax=False,
    )
    if cfg.decode_btn_support_min > 0 and not has_button_combo_counts:
        raise ValueError(
            "decode_btn_support_min > 0 requires button_combo_counts_path for a fresh run or embedded "
            "dataset-scoped counts in a resumed checkpoint"
        )


def _load_model_state(model: GPT, state_dict: dict[str, Tensor]) -> None:
    """Load an E0 checkpoint and allow an older missing count buffer."""
    if any(key.startswith("blocks.") for key in state_dict):
        raise RuntimeError(
            "this checkpoint stores the transformer under 'blocks.*' — it is from 016/019/020/021, "
            "whose trunk is an inline copy. 023 holds the shared trunk under 'trunk.blocks.*', so "
            "the two are not interchangeable; evaluate that checkpoint with its own experiment file"
        )
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if missing - {"button_combo_counts"} or unexpected:
        raise RuntimeError(f"checkpoint/model mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")


@dataclass(frozen=True, slots=True)
class DecodeSettings:
    temp: float
    temps: tuple[float, float, float, float] | None
    btn_support_min: int
    min_p: float
    click_trigger_fix: bool


_T = TypeVar("_T")


@dataclass
class DecodeTelemetry:
    policy_calls: int = 0
    slot_frames: int = 0
    model_forwards: int = 0
    model_rows: int = 0
    _cpu_forward_seconds: list[float] = field(default_factory=list)
    _cuda_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = field(default_factory=list)

    def record_policy_call(self, slots: int) -> None:
        self.policy_calls += 1
        self.slot_frames += slots

    def model_forward(self, fn: Callable[[], _T], *, rows: int, device: torch.device) -> _T:
        self.model_forwards += 1
        self.model_rows += rows
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = fn()
            end.record()
            self._cuda_events.append((start, end))
            return result
        started_at = time.perf_counter()
        result = fn()
        self._cpu_forward_seconds.append(time.perf_counter() - started_at)
        return result

    def metrics(self) -> dict[str, float]:
        forward_seconds = list(self._cpu_forward_seconds)
        if self._cuda_events:
            torch.cuda.synchronize()
            forward_seconds.extend(start.elapsed_time(end) / 1_000 for start, end in self._cuda_events)
        latency_ms = np.asarray(forward_seconds, dtype=np.float64) * 1_000
        total_seconds = float(latency_ms.sum() / 1_000)
        return {
            "decode_policy_calls": float(self.policy_calls),
            "decode_slot_frames": float(self.slot_frames),
            "decode_model_forwards": float(self.model_forwards),
            "decode_model_rows": float(self.model_rows),
            "decode_model_forward_seconds": total_seconds,
            "decode_model_forward_mean_ms": float(latency_ms.mean()) if latency_ms.size else 0.0,
            "decode_model_forward_median_ms": float(np.median(latency_ms)) if latency_ms.size else 0.0,
            "decode_model_forward_p95_ms": float(np.quantile(latency_ms, 0.95)) if latency_ms.size else 0.0,
            "decode_model_forward_ms_per_row": (1_000 * total_seconds / self.model_rows if self.model_rows else 0.0),
        }


def _decode_settings(
    model: GPT,
    cfg: TrainConfig,
    *,
    temp: float | None = None,
    temps: tuple[float, float, float, float] | None = None,
    btn_support_min: int | None = None,
    min_p: float | None = None,
    click_trigger_fix: bool | None = None,
) -> DecodeSettings:
    settings = DecodeSettings(
        temp=cfg.decode_temp if temp is None else temp,
        temps=cfg.decode_temps if temps is None else temps,
        btn_support_min=cfg.decode_btn_support_min if btn_support_min is None else btn_support_min,
        min_p=cfg.decode_min_p if min_p is None else min_p,
        click_trigger_fix=cfg.decode_click_trigger_fix if click_trigger_fix is None else click_trigger_fix,
    )
    _resolve_decode_args(settings.temp, settings.temps, settings.btn_support_min, settings.min_p, argmax=False)
    if settings.btn_support_min > 0 and bool((model.button_combo_counts < 0).any()):
        raise ValueError("button-support masking requested, but this checkpoint has no dataset-scoped counts")
    return settings


_UINT64_MASK = (1 << 64) - 1


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return value ^ (value >> 31)


class SlotGroupRandom:
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
        resets = (False,) * len(slot_ids) if ctx.reset is None else tuple(bool(value) for value in ctx.reset.tolist())
        for slot_id, reset in zip(slot_ids, resets, strict=True):
            if slot_id not in self.generations:
                self.generations[slot_id] = 0
            elif reset:
                self.generations[slot_id] += 1
            generation = self.generations[slot_id]
            if reset or not any(key[0] == slot_id and key[1] == generation for key in self.counters):
                for group in _GROUP_NAMES:
                    self.counters[(slot_id, generation, group)] = 0
        self.slot_ids = slot_ids
        self.device = ctx.slot_ids.device

    def select(self, ctx: Context) -> None:
        if ctx.slot_ids is None:
            raise ValueError("slot-keyed sampling needs slot_ids")
        slot_ids = tuple(int(value) for value in ctx.slot_ids.tolist())
        if len(set(slot_ids)) != len(slot_ids):
            raise ValueError(f"slot_ids must be unique, got {slot_ids}")
        missing = [slot_id for slot_id in slot_ids if slot_id not in self.generations]
        if missing:
            raise ValueError(f"slot IDs were not observed before sampling: {missing}")
        self.slot_ids = slot_ids
        self.device = ctx.slot_ids.device

    def uniforms(self, group: str) -> Tensor:
        if group not in _GROUP_INDEX:
            raise ValueError(f"unknown action group {group!r}")
        if not self.slot_ids:
            raise RuntimeError("begin must be called before sampling")
        values: list[float] = []
        group_key = _splitmix64(_GROUP_INDEX[group] + 1)
        for slot_id in self.slot_ids:
            generation = self.generations[slot_id]
            key = (slot_id, generation, group)
            counter = self.counters[key]
            mixed = self.seed
            mixed ^= _splitmix64(slot_id & _UINT64_MASK)
            mixed ^= _splitmix64(generation)
            mixed ^= group_key
            mixed ^= _splitmix64(counter)
            random_bits = _splitmix64(mixed)
            values.append(((random_bits >> 11) + 0.5) / (1 << 53))
            self.counters[key] = counter + 1
        return torch.tensor(values, device=self.device)

    def state(self) -> tuple[tuple[int, int, str, int], ...]:
        return tuple(sorted((*key, count) for key, count in self.counters.items()))


def make_policy(
    model: GPT,
    stats: dict[str, FeatureStats],
    cfg: TrainConfig,
    *,
    device: str = DEVICE,
    exec_horizon: int | None = None,
    decode_temp: float | None = None,
    decode_temps: tuple[float, float, float, float] | None = None,
    decode_btn_support_min: int | None = None,
    decode_min_p: float | None = None,
    decode_click_trigger_fix: bool | None = None,
    decode_seed: int | None = None,
    telemetry: DecodeTelemetry | None = None,
) -> RecedingHorizon:
    """Build one closed-loop policy with optional decode overrides."""
    s = cfg.exec_horizon if exec_horizon is None else exec_horizon
    offsets = _exec_horizon_offsets(model.head_offsets, s)
    settings = _decode_settings(
        model,
        cfg,
        temp=decode_temp,
        temps=decode_temps,
        btn_support_min=decode_btn_support_min,
        min_p=decode_min_p,
        click_trigger_fix=decode_click_trigger_fix,
    )
    model_device = next(model.parameters()).device
    gen = None if decode_seed is None else torch.Generator(device=model_device).manual_seed(decode_seed)
    streams = None if decode_seed is None else SlotGroupRandom(decode_seed)

    @torch.no_grad()
    def predict_chunk(ctx: Context, committed: np.ndarray | None) -> np.ndarray:
        assert committed is None, "receding-horizon policy does not condition on a committed prefix"
        run = functools.partial(model, ctx.features, ctx.ctx_pad)
        hidden_states = (
            run()
            if telemetry is None
            else telemetry.model_forward(run, rows=ctx.ctx_pad.shape[0], device=model_device)
        )
        h = hidden_states[:, -1]
        keyed_uniforms = None
        if streams is not None and ctx.slot_ids is not None:
            streams.begin(ctx)
            keyed_uniforms = streams.uniforms
        chunk = chunk_from_hidden(
            model,
            h,
            offsets,
            group_temps=settings.temps or (settings.temp,) * N_GROUPS,
            btn_support_min=settings.btn_support_min,
            min_p=settings.min_p,
            click_trigger_fix=settings.click_trigger_fix,
            gen=gen if keyed_uniforms is None else None,
            uniforms=keyed_uniforms,
        )
        return chunk.cpu().numpy()

    return RecedingHorizon(
        predict_chunk=predict_chunk,
        stats=stats,
        L_ctx=cfg.L_ctx,
        L_chunk=s,
        s=s,
        d=0,
        device=device,
        float_dtype=next(model.parameters()).dtype,
        projection=_INPUT_PROJECTION,
    )


# %%
def lr_schedule(cfg: TrainConfig):
    floor = 0.01

    def fn(step: int) -> float:
        if step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)
        progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    return fn


def _output_head_parameters(model: GPT) -> tuple[nn.Parameter, ...]:
    modules: tuple[nn.Module, ...] = (
        (model.heads,) if model.head_adapter is None else (model.heads, model.head_adapter)
    )
    return tuple(parameter for module in modules for parameter in module.parameters())


def parameter_counts(model: GPT) -> dict[str, int]:
    adapter = model.head_adapter
    counts = {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trunk": sum(parameter.numel() for parameter in model.trunk.parameters()),
        "classifier": sum(parameter.numel() for parameter in model.heads.parameters()),
        "state_projection": 0,
        "residual_output": 0,
        "condition_projection": 0,
        "action_embedding": 0,
    }
    if adapter is not None:
        counts["state_projection"] = sum(parameter.numel() for parameter in adapter.state_proj.parameters())
        counts["residual_output"] = sum(parameter.numel() for parameter in adapter.residual_projs.parameters())
    if isinstance(adapter, FactoredMLPAdapter):
        counts["condition_projection"] = sum(parameter.numel() for parameter in adapter.condition_projs.parameters())
        counts["action_embedding"] = sum(parameter.numel() for parameter in adapter.action_embeddings.parameters())
    return counts


def make_optimizer(model: GPT, cfg: TrainConfig) -> SingleDeviceMuonWithAuxAdam:
    """Use Muon for transformer matrices and AdamW for all other parameters."""
    muon_params = [p for p in model.trunk.blocks.parameters() if p.ndim >= 2]
    muon_ids = {id(p) for p in muon_params}
    embed_ids = {id(p) for m in (model.cat_embeds, model.char_emb, model.stage_emb) for p in m.parameters()}
    # Optionally exclude output-head weights from weight decay.
    head_ids = set() if cfg.head_weight_decay else {id(p) for p in _output_head_parameters(model)}

    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for p in model.parameters():
        if id(p) in muon_ids:
            continue
        # AdamW: no weight decay on embeddings, 1D params (biases), or the head weights when disabled.
        (no_decay if id(p) in embed_ids or p.ndim < 2 or id(p) in head_ids else decay).append(p)

    n_assigned = len(muon_params) + len(decay) + len(no_decay)
    n_total = sum(1 for _ in model.parameters())
    if n_assigned != n_total:
        raise RuntimeError(f"optimizer param partition covers {n_assigned}/{n_total} params")

    adam = dict(betas=(0.9, 0.95), eps=1e-10, use_muon=False)
    param_groups = [
        dict(params=muon_params, lr=cfg.muon_lr, momentum=0.95, weight_decay=cfg.weight_decay, use_muon=True),
        dict(params=decay, lr=cfg.adam_lr, weight_decay=cfg.weight_decay, **adam),
        dict(params=no_decay, lr=cfg.adam_lr, weight_decay=0.0, **adam),
    ]
    return SingleDeviceMuonWithAuxAdam(param_groups)


def nll_breakdown(comps: dict[str, Tensor]) -> dict[str, float]:
    """Per-group NLL (bits) + ``total`` bits/frame, from one head's per-group ``[n_valid]`` nats. Flat
    keys (``buttons``/``main_stick``/``c_stick``/``triggers``/``total``) so callers land in one W&B section."""
    out = {name: (c.mean().item() / _LN2) for name, c in comps.items()}
    out["total"] = sum(c.mean() for c in comps.values()).item() / _LN2
    return out


def _weighted_mean(nll: Tensor, is_trans: Tensor, weight: float) -> Tensor:
    """Return the mean NLL after optional transition upweighting."""
    if weight == 1.0:
        return nll.mean()
    w = torch.where(is_trans, weight, 1.0).to(nll.dtype)
    return (w * nll).sum() / w.sum()


def _offset_objective(
    nll: dict[tuple[int, str], Tensor],
    trans: dict[tuple[int, str], Tensor],
    offset: int,
    transition_weight: float,
) -> Tensor:
    """Return one offset's action loss."""
    return torch.stack(
        [
            _weighted_mean(
                nll[(offset, name)],
                trans[(offset, name)],
                transition_weight,
            )
            for name in _GROUP_NAMES
        ]
    ).sum()


def objective(
    nll: dict[tuple[int, str], Tensor],
    trans: dict[tuple[int, str], Tensor],
    aux_weight: float,
    transition_weight: float,
) -> Tensor:
    """Return the primary loss plus a fixed-total mean auxiliary loss."""
    primary = _offset_objective(nll, trans, 1, transition_weight)
    aux_offsets = sorted({offset for offset, _ in nll if offset != 1})
    if not aux_offsets or aux_weight == 0.0:
        return primary
    auxiliary = torch.stack(
        [_offset_objective(nll, trans, offset, transition_weight) for offset in aux_offsets]
    ).mean()
    return primary + aux_weight * auxiliary


def _finite_gradient_norm(model: nn.Module, objective_value: Tensor, step: int) -> Tensor:
    if not bool(torch.isfinite(objective_value).item()):
        raise FloatingPointError(f"step {step}: loss is not finite; optimizer step was skipped")
    try:
        return torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float("inf"),
            error_if_nonfinite=True,
        )
    except RuntimeError as error:
        raise FloatingPointError(f"step {step}: gradients are not finite; optimizer step was skipped") from error


def _parameter_gradient_norm(parameters: Iterable[nn.Parameter]) -> float:
    norms = [parameter.grad.detach().float().norm() for parameter in parameters if parameter.grad is not None]
    if not norms:
        return 0.0
    return torch.stack(norms).norm().item()


def _parameter_norm(parameters: Iterable[nn.Parameter]) -> float:
    values = [parameter.detach().float().norm() for parameter in parameters]
    return torch.stack(values).norm().item() if values else 0.0


def _replay_overlap(previous: frozenset[str] | None, current: set[str]) -> int | None:
    if previous is None or not current:
        return None
    return len(previous.intersection(current))


def _slice_batch(batch: TrainBatch, n: int) -> TrainBatch:
    return TrainBatch(
        context=Context(
            features={name: value[:n] for name, value in batch.context.features.items()},
            ctx_pad=batch.context.ctx_pad[:n],
        ),
        target=batch.target[:n],
        replay_ids=None if batch.replay_ids is None else batch.replay_ids[:n],
    )


def _gradient_dot(a: tuple[Tensor, ...], b: tuple[Tensor, ...]) -> Tensor:
    return torch.stack([torch.dot(x.reshape(-1), y.reshape(-1)) for x, y in zip(a, b, strict=True)]).sum()


def _gradient_cosine(a: tuple[Tensor, ...], b: tuple[Tensor, ...], norm_a: Tensor, norm_b: Tensor) -> Tensor:
    denom = (norm_a * norm_b).clamp_min(torch.finfo(norm_a.dtype).tiny)
    return (_gradient_dot(a, b) / denom).clamp(-1.0, 1.0)


@contextlib.contextmanager
def _evaluation_mode(model: nn.Module) -> Iterator[None]:
    """Temporarily enter eval mode, on the UNCOMPILED forward, and restore both on every exit.

    Every non-training forward in this file goes through here, and ``compile_trunk`` installs the
    compiled forward as an INSTANCE attribute, so removing that attribute for the duration falls
    back to the class method. Validation and the gradient diagnostic each carry their own batch
    shape, and a compiled graph that meets one of them dies inside FlexAttention with a CUDA illegal
    memory access. Training keeps the compiled graph — and the same one, so its cache survives."""
    was_training = model.training
    compiled = model.__dict__.pop("forward", None)
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)
        if compiled is not None:
            model.forward = compiled


def gradient_diagnostics(model: GPT, batch: TrainBatch, cfg: TrainConfig) -> dict[str, float]:
    """Exact shared-trunk gradient norms/cosines for each horizon on a fixed val subset.

    Output-head parameters are deliberately excluded: the question is whether each
    task asks the representation shared with the deployed head to move in an aligned
    direction. ``autograd.grad`` leaves ``parameter.grad`` untouched, and eval mode
    disables history dropout, so this cannot perturb the optimizer or RNG stream.
    """
    with _evaluation_mode(model):
        return _gradient_diagnostics_eval(model, batch, cfg)


def _representation_parameters(model: GPT) -> tuple[nn.Parameter, ...]:
    modules = (model.cat_embeds, model.char_emb, model.stage_emb, model.ctx_proj, model.trunk)
    parameters = tuple(parameter for module in modules for parameter in module.parameters())
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise RuntimeError("shared representation contains a parameter more than once")
    return parameters


def _gradient_diagnostics_eval(model: GPT, batch: TrainBatch, cfg: TrainConfig) -> dict[str, float]:
    diagnostic_batch = _slice_batch(batch, min(cfg.gradient_diagnostic_batch_size, batch.context.batch))
    parts = action_loss(model, diagnostic_batch)
    losses = {
        offset: _offset_objective(parts.nll, parts.transition, offset, cfg.transition_loss_weight)
        for offset in model.head_offsets
    }
    representation = _representation_parameters(model)
    gradients: dict[int, tuple[Tensor, ...]] = {}
    for i, offset in enumerate(model.head_offsets):
        gradients[offset] = tuple(
            gradient.detach()
            for gradient in torch.autograd.grad(
                losses[offset],
                representation,
                retain_graph=model.head_adapter is not None or i + 1 < len(model.head_offsets),
            )
        )

    norms = {offset: _gradient_dot(gradient, gradient).sqrt() for offset, gradient in gradients.items()}
    out = {f"grad/head_{offset}_norm": norm.item() for offset, norm in norms.items()}
    for i, left in enumerate(model.head_offsets):
        for right in model.head_offsets[i + 1 :]:
            out[f"grad/cos_{left}_{right}"] = _gradient_cosine(
                gradients[left], gradients[right], norms[left], norms[right]
            ).item()

    aux_offsets = tuple(offset for offset in model.head_offsets if offset != 1)
    if aux_offsets:
        # This is the auxiliary direction used by the current objective.
        weighted_aux = tuple(
            cfg.aux_loss_weight
            * sum((gradients[offset][pi] for offset in aux_offsets), start=torch.zeros_like(p))
            / len(aux_offsets)
            for pi, p in enumerate(representation)
        )
        aux_norm = _gradient_dot(weighted_aux, weighted_aux).sqrt()
        primary = gradients[1]
        out["grad/weighted_aux_norm"] = aux_norm.item()
        out["grad/weighted_aux_to_primary_norm"] = (aux_norm / norms[1].clamp_min(norms[1].new_tensor(1e-30))).item()
        out["grad/cos_1_weighted_aux"] = _gradient_cosine(primary, weighted_aux, norms[1], aux_norm).item()
        conflicting = sum(
            int(((p != 0) & (a != 0) & ((p > 0) != (a > 0))).sum()) for p, a in zip(primary, weighted_aux)
        )
        comparable = sum(int(((p != 0) & (a != 0)).sum()) for p, a in zip(primary, weighted_aux))
        out["grad/sign_conflict_frac_1_weighted_aux"] = conflicting / comparable if comparable else 0.0
    if model.head_adapter is not None:
        auxiliary = torch.stack([losses[offset] for offset in aux_offsets]).mean() if aux_offsets else 0.0
        total = losses[1] + cfg.aux_loss_weight * auxiliary
        state_parameters = tuple(model.head_adapter.state_proj.parameters())
        residual_parameters = tuple(model.head_adapter.residual_projs.parameters())
        condition_parameters: tuple[nn.Parameter, ...] = ()
        embedding_parameters: tuple[nn.Parameter, ...] = ()
        if isinstance(model.head_adapter, FactoredMLPAdapter):
            condition_parameters = tuple(model.head_adapter.condition_projs.parameters())
            embedding_parameters = tuple(model.head_adapter.action_embeddings.parameters())
        adapter_parameters = state_parameters + residual_parameters + condition_parameters + embedding_parameters
        adapter_gradients = tuple(gradient.detach() for gradient in torch.autograd.grad(total, adapter_parameters))
        state_end = len(state_parameters)
        residual_end = state_end + len(residual_parameters)
        condition_end = residual_end + len(condition_parameters)
        state_gradients = adapter_gradients[:state_end]
        residual_gradients = adapter_gradients[state_end:residual_end]
        out["grad/state_mlp_state_norm"] = _gradient_dot(state_gradients, state_gradients).sqrt().item()
        out["grad/state_mlp_residual_norm"] = _gradient_dot(residual_gradients, residual_gradients).sqrt().item()
        if isinstance(model.head_adapter, FactoredMLPAdapter):
            condition_gradients = adapter_gradients[residual_end:condition_end]
            embedding_gradients = adapter_gradients[condition_end:]
            out["grad/factored_condition_norm"] = (
                _gradient_dot(
                    condition_gradients,
                    condition_gradients,
                )
                .sqrt()
                .item()
            )
            out["grad/factored_embedding_norm"] = (
                _gradient_dot(
                    embedding_gradients,
                    embedding_gradients,
                )
                .sqrt()
                .item()
            )
    return out


def _offset_total_bits(comps: dict[tuple[int, str], Tensor], o: int) -> float:
    """Total bits/frame summed over the four groups for one head offset ``o``."""
    return sum(comps[(o, name)].mean() for name in _GROUP_NAMES).item() / _LN2


def _masked_mean_bits(nats: Tensor, mask: Tensor) -> float:
    """Mean of per-position NLL (nats) over the masked subset, in bits; 0.0 when the subset is empty."""
    return (nats[mask].mean().item() / _LN2) if bool(mask.any()) else 0.0


def _bool_mean(values: Tensor) -> float:
    return values.float().mean().item() if values.numel() else 0.0


def _group_kl_bits(logits_p: dict[str, Tensor], logits_q: dict[str, Tensor]) -> Tensor:
    """Return the sum of the four group KL divergences in bits."""
    first = logits_p[_GROUP_NAMES[0]]
    total = torch.zeros(first.shape[:-1], device=first.device)
    for name in _GROUP_NAMES:
        logp = F.log_softmax(logits_p[name], dim=-1)
        logq = F.log_softmax(logits_q[name], dim=-1)
        total = total + (logp.exp() * (logp - logq)).sum(-1)
    return total / _LN2


def _factorization_generators(seed: int, device: torch.device) -> dict[str, torch.Generator]:
    return {
        name: torch.Generator(device=device).manual_seed(
            _splitmix64((seed & _UINT64_MASK) ^ _splitmix64(_GROUP_INDEX[name] + 1)) % (2**63 - 1)
        )
        for name in _GROUP_NAMES
    }


def _ancestor_sampled_logits(
    model: GPT,
    hidden: Tensor,
    head_index: int,
    generators: dict[str, torch.Generator],
) -> tuple[dict[str, Tensor], Tensor]:
    adapter = model.head_adapter
    if not isinstance(adapter, FactoredMLPAdapter):
        raise ValueError("ancestor sampling needs a factored MLP head")
    base = cast(IndependentHead, model.heads[head_index]).logits(hidden)
    state_preactivation = adapter.preactivation(hidden)
    prefix: dict[str, Tensor] = {}
    embedded_prefix: dict[str, Tensor] = {}
    logits: dict[str, Tensor] = {}
    picks: dict[str, Tensor] = {}
    for name in adapter.group_order:
        group_logits = adapter.group_logits(
            head_index,
            name,
            base[name],
            state_preactivation,
            prefix,
            embedded_prefix,
        ).float()
        pick = _sample_categorical(
            group_logits,
            temperature=1.0,
            min_p=0.0,
            argmax=False,
            gen=generators[name],
        )
        logits[name] = group_logits
        prefix[name] = pick
        if name != adapter.group_order[-1]:
            embedded_prefix[name] = adapter.action_embeddings[name](pick)
        picks[name] = pick
    return logits, torch.stack([picks[name] for name in _GROUP_NAMES], dim=-1)


def _pooled_mean(parts: list[tuple[float, int]]) -> float:
    """The count-weighted mean of per-batch ``(mean, count)`` pairs — the mean over the pooled
    positions, up to summation order. One batch needs no combination at all, so its own number is
    returned untouched: multiplying by a count and dividing it out again would only round it."""
    if len(parts) == 1:
        return parts[0][0]
    count = sum(count for _, count in parts)
    return sum(value * count for value, count in parts) / count if count else 0.0


class _MeanAccumulator:
    """Validation metrics, collected one batch at a time.

    A 128-batch pass over 1024-frame windows scores 8.4M positions, so holding a per-position tensor
    for every metric until the end of the pass would cost gigabytes of VRAM. Each batch instead
    reports its own mean and the number of positions behind it, and nothing per-frame survives the
    loop iteration."""

    def __init__(self) -> None:
        self._parts: dict[str, list[tuple[float, int]]] = {}

    def add(self, key: str, value: float, count: int) -> None:
        """``value`` is this batch's mean over ``count`` positions. A count of 0 registers the key
        (the metric exists; its subset was empty in this batch) and moves the pooled mean nowhere."""
        self._parts.setdefault(key, []).append((value, count))

    def update(self, values: dict[str, float], count: int) -> None:
        for key, value in values.items():
            self.add(key, value, count)

    def means(self) -> dict[str, float]:
        return {key: _pooled_mean(parts) for key, parts in self._parts.items()}


def _hidden_pair(model: GPT, ctx: Context) -> tuple[Tensor, Tensor]:
    """The only two distinct trunk forwards a val batch needs: the state, and its history-ablated
    twin (the copycat probe, with the ego's own controller history zeroed). Both metric families
    read the pair inside one loop iteration, so a pass costs two forwards per batch, not five."""
    ablated_features = dict(ctx.features)
    for ch in ACTION_CHANNELS:
        ablated_features[f"ego_{ch}"] = torch.zeros_like(ablated_features[f"ego_{ch}"])
    return model(ctx.features, ctx.ctx_pad), model(ablated_features, ctx.ctx_pad)


@torch.no_grad()
def val_metrics(model: GPT, val_cache: list[TrainBatch], cfg: TrainConfig) -> dict[str, float]:
    """Return offline action metrics from the fixed validation set."""
    with _evaluation_mode(model):
        return _val_metrics_eval(model, val_cache, cfg)


def _val_metrics_eval(model: GPT, val_cache: list[TrainBatch], cfg: TrainConfig) -> dict[str, float]:
    acc = _MeanAccumulator()
    change_predictions: dict[tuple[int, str], list[Tensor]] = {}
    change_targets: dict[tuple[int, str], list[Tensor]] = {}
    counts_available = bool((model.button_combo_counts >= 0).all())
    rare_mask = model.button_combo_counts < cfg.diagnostic_rare_button_count
    unseen_mask = model.button_combo_counts == 0
    combo_bits = scoring.combo_to_buttons(torch.arange(scoring.N_BUTTON_COMBOS, device=model.main_centers.device))
    device = next(model.parameters()).device
    factorization_generators = (
        _factorization_generators(cfg.factorization_diag_seed, device)
        if isinstance(model.head_adapter, FactoredMLPAdapter)
        else None
    )
    for cached in val_cache:
        batch = cached.to(device)
        ctx = batch.context
        h, h_ablated = _hidden_pair(model, ctx)
        _, valid = _multi_offset_targets(ctx, batch.target[:, : max(model.head_offsets)], model.head_offsets)
        flat_valid = valid.reshape(-1)
        n_valid = int(flat_valid.sum())
        full_actions = torch.cat([stack_actions(ctx.features), batch.target], dim=1)
        full_indices = _quantize(model, full_actions)
        context_length = full_actions.shape[1] - batch.target.shape[1]
        cur_idx = full_indices[:, :context_length]
        comps: dict[tuple[int, str], Tensor] = {}
        ablated: dict[str, Tensor] = {}
        target_indices = tuple(full_indices[:, offset : offset + context_length] for offset in model.head_offsets)
        base_logits, all_logits = model.base_and_group_logits(h, target_indices)
        for hi, o in enumerate(model.head_offsets):
            tgt_idx = target_indices[hi]
            logits = {name: lg.float() for name, lg in all_logits[hi].items()}
            comps.update({(o, name): c for name, c in group_nll(logits, tgt_idx, valid).items()})
            group_predictions: dict[str, Tensor] = {}
            group_targets: dict[str, Tensor] = {}
            for g, name in enumerate(_GROUP_NAMES):
                predicted = logits[name].argmax(-1).reshape(-1)[flat_valid]
                target_group = tgt_idx[..., g].reshape(-1)[flat_valid]
                group_predictions[name] = predicted
                group_targets[name] = target_group
                acc.add(f"acc_off{o}_{name}", _bool_mean(predicted == target_group), n_valid)
                if model.head_adapter is not None:
                    base = base_logits[hi][name].float().reshape(-1, _GROUP_VOCABS[g])[flat_valid]
                    residual = logits[name].reshape(-1, _GROUP_VOCABS[g])[flat_valid] - base
                    n_logits = base.numel()
                    acc.add(f"_base_sq_off{o}_{name}", base.square().mean().item(), n_logits)
                    acc.add(f"_residual_sq_off{o}_{name}", residual.square().mean().item(), n_logits)
            predicted_frame = torch.stack(
                [group_predictions[name] for name in _GROUP_NAMES],
                dim=-1,
            )
            target_frame = torch.stack([group_targets[name] for name in _GROUP_NAMES], dim=-1)
            acc.add(
                f"exact_frame_acc_off{o}", (predicted_frame == target_frame).all(-1).float().mean().item(), n_valid
            )
            previous_idx = full_indices[:, o - 1 : o - 1 + context_length]
            true_change = tgt_idx != previous_idx
            for g, name in enumerate(_GROUP_NAMES):
                trans = true_change[..., g].reshape(-1)[flat_valid]
                predicted = group_predictions[name]
                target_group = group_targets[name]
                correct = predicted == target_group
                predicted_change_full = all_logits[hi][name].argmax(-1) != previous_idx[..., g]
                predicted_change = predicted_change_full.reshape(-1)[flat_valid]
                acc.add(f"pred_change_rate_off{o}_{name}", _bool_mean(predicted_change), n_valid)
                acc.add(f"pred_persistence_off{o}_{name}", _bool_mean(~predicted_change), n_valid)
                acc.add(f"nll_off{o}_{name}_trans", _masked_mean_bits(comps[(o, name)], trans), int(trans.sum()))
                acc.add(f"nll_off{o}_{name}_hold", _masked_mean_bits(comps[(o, name)], ~trans), int((~trans).sum()))
                acc.add(f"acc_off{o}_{name}_trans", _bool_mean(correct[trans]), int(trans.sum()))
                acc.add(f"acc_off{o}_{name}_hold", _bool_mean(correct[~trans]), int((~trans).sum()))
                key = (o, name)
                change_predictions.setdefault(key, []).append(predicted_change_full & valid)
                change_targets.setdefault(key, []).append(true_change[..., g] & valid)
            if factorization_generators is not None:
                hidden_valid = h.reshape(-1, h.shape[-1])[flat_valid]
                target_valid = tgt_idx.reshape(-1, N_GROUPS)[flat_valid]
                for _ in range(cfg.factorization_diag_samples):
                    ancestor_logits, sampled = _ancestor_sampled_logits(
                        model,
                        hidden_valid,
                        hi,
                        factorization_generators,
                    )
                    for g, name in enumerate(_GROUP_NAMES):
                        ancestor_nll = F.cross_entropy(ancestor_logits[name], target_valid[:, g])
                        ancestor_acc = (ancestor_logits[name].argmax(-1) == target_valid[:, g]).float().mean()
                        acc.add(f"ancestor_nll_off{o}_{name}", ancestor_nll.item() / _LN2, n_valid)
                        acc.add(f"ancestor_acc_off{o}_{name}", ancestor_acc.item(), n_valid)
                    exact = (sampled == target_valid).all(-1).float().mean().item()
                    acc.add(f"ancestor_exact_frame_acc_off{o}", exact, n_valid)
            if o != 1:
                continue
            ablated_logits = {
                name: lg.float() for name, lg in model.group_logits(h_ablated, model.primary_head_idx, tgt_idx).items()
            }
            kl_bits = _group_kl_bits(logits, ablated_logits).reshape(-1)[flat_valid]  # KL(full ‖ ablated), bits
            acc.add("ablate_hist_kl", kl_bits.mean().item(), n_valid)
            ablated = group_nll(ablated_logits, tgt_idx, valid)
            for g, name in enumerate(_GROUP_NAMES):
                trans = true_change[..., g].reshape(-1)[flat_valid]
                predicted = group_predictions[name]
                target_group = group_targets[name]
                correct = predicted == target_group
                predicted_change = predicted != cur_idx[..., g].reshape(-1)[flat_valid]
                acc.add(f"acc_{name}", _bool_mean(correct), n_valid)
                acc.add(f"pred_change_rate_{name}", _bool_mean(predicted_change), n_valid)
                acc.add(f"pred_persistence_{name}", _bool_mean(~predicted_change), n_valid)
                acc.add(f"nll_{name}_trans", _masked_mean_bits(comps[(1, name)], trans), int(trans.sum()))
                acc.add(f"nll_{name}_hold", _masked_mean_bits(comps[(1, name)], ~trans), int((~trans).sum()))
                acc.add(f"acc_{name}_trans", _bool_mean(correct[trans]), int(trans.sum()))
                acc.add(f"acc_{name}_hold", _bool_mean(correct[~trans]), int((~trans).sum()))
            btn_logits = logits["buttons"]
            combo_probs = F.softmax(btn_logits.reshape(-1, scoring.N_BUTTON_COMBOS)[flat_valid], dim=-1)
            onehot = F.one_hot(tgt_idx[..., _BUTTONS_G].reshape(-1)[flat_valid], scoring.N_BUTTON_COMBOS).to(
                combo_probs.dtype
            )
            acc.add("brier_buttons", (combo_probs - onehot).pow(2).sum(-1).mean().item(), n_valid)
            marginal_btn_probs = combo_probs @ combo_bits.to(combo_probs.dtype)
            tgt_btn = _dequantize(model, tgt_idx)[..., _N_CONT:].reshape(-1, _N_BUTTONS)[flat_valid]
            logloss, brier = scoring.bernoulli_scores_from_probs(marginal_btn_probs, tgt_btn)
            acc.add("btn_logloss", logloss.item(), n_valid)
            acc.add("btn_brier", brier.item(), n_valid)
            if counts_available:
                acc.add("btn_rare_mass", combo_probs[:, rare_mask].sum(-1).mean().item(), n_valid)
                acc.add("btn_unseen_mass", combo_probs[:, unseen_mask].sum(-1).mean().item(), n_valid)
        primary = nll_breakdown({name: comps[(1, name)] for name in _GROUP_NAMES})
        acc.add("loss", primary["total"], n_valid)  # offset-1 total bits/frame (deployed policy)
        acc.update({f"nll_{name}": primary[name] for name in _GROUP_NAMES}, n_valid)
        acc.add(
            "cont_discrete_bits",
            (comps[(1, "main_stick")].mean() + comps[(1, "c_stick")].mean() + comps[(1, "triggers")].mean()).item()
            / _LN2,
            n_valid,
        )
        for o in model.head_offsets:
            acc.add(f"nll_off{o}", _offset_total_bits(comps, o), n_valid)
            for name in _GROUP_NAMES:
                acc.add(f"nll_off{o}_{name}", comps[(o, name)].mean().item() / _LN2, n_valid)
        ablate_total = 0.0
        for name in _GROUP_NAMES:
            d = (ablated[name].mean() - comps[(1, name)].mean()).item() / _LN2  # positive ⇒ history helps
            acc.add(f"ablate_hist_dnll_{name}", d, n_valid)
            ablate_total += d
        acc.add("ablate_hist_dnll", ablate_total, n_valid)

    out = acc.means()
    if model.head_adapter is not None:
        for offset in model.head_offsets:
            for name in _GROUP_NAMES:
                base_sq = out.pop(f"_base_sq_off{offset}_{name}")
                residual_sq = out.pop(f"_residual_sq_off{offset}_{name}")
                out[f"residual_logit_rms_ratio_off{offset}_{name}"] = math.sqrt(residual_sq) / max(
                    math.sqrt(base_sq), 1e-30
                )
    if factorization_generators is not None:
        out["factorization_diag_seed"] = float(cfg.factorization_diag_seed)
        out["factorization_diag_samples"] = float(cfg.factorization_diag_samples)
        for offset in model.head_offsets:
            for name in _GROUP_NAMES:
                out[f"ancestor_nll_gap_off{offset}_{name}"] = (
                    out[f"ancestor_nll_off{offset}_{name}"] - out[f"nll_off{offset}_{name}"]
                )
    out["btn_counts_available"] = float(counts_available)
    if counts_available:
        out["btn_rare_count_threshold"] = float(cfg.diagnostic_rare_button_count)
    for offset in model.head_offsets:
        for name in _GROUP_NAMES:
            key = (offset, name)
            out[f"changeF1_off{offset}_{name}"] = scoring.change_event_prf(
                torch.cat(change_predictions[key]),
                torch.cat(change_targets[key]),
            )[2]
    out["changeF1_buttons"] = out["changeF1_off1_buttons"]
    return out


@dataclass(frozen=True, slots=True)
class EvalProtocol:
    n_matchups: int
    max_parallel: int
    max_frames: int
    seed: int
    cpu_level: int
    ego_port: int
    seed_stage: int
    matchup_schedule_sha256: str
    instant_match_restart: bool
    stage_policy: str
    completion_policy: str
    active_frame_policy: str
    uncertainty_policy: str
    start_retries: int
    exec_horizon: int
    decode_temp: float
    decode_temps: tuple[float, float, float, float] | None
    decode_btn_support_min: int
    decode_min_p: float
    decode_click_trigger_fix: bool
    model_dtype: str


def _eval_protocol(
    cfg: TrainConfig,
    *,
    settings: DecodeSettings,
    exec_horizon: int,
    default_n_matchups: int,
    model_dtype: str,
    n_matchups: int | None = None,
    max_parallel: int | None = None,
    max_frames: int | None = None,
    seed: int | None = None,
) -> EvalProtocol:
    n = default_n_matchups if n_matchups is None else n_matchups
    frames = cfg.eval_max_frames if max_frames is None else max_frames
    resolved_seed = cfg.eval_seed if seed is None else seed
    if n <= 0:
        raise ValueError(f"n_matchups must be > 0, got {n}")
    parallel = _eval_max_parallel(cfg, n) if max_parallel is None else max_parallel
    if not 1 <= parallel <= n:
        raise ValueError(f"max_parallel must be in 1..{n}, got {parallel}")
    if frames <= 0:
        raise ValueError(f"max_frames must be > 0, got {frames}")
    matchups = matchups_for_vs_cpu(n)
    schedule = [[int(ego.value), int(opp.value)] for ego, opp in matchups]
    schedule_sha256 = hashlib.sha256(json.dumps(schedule, separators=(",", ":")).encode()).hexdigest()
    return EvalProtocol(
        n_matchups=n,
        max_parallel=parallel,
        max_frames=frames,
        seed=resolved_seed,
        cpu_level=9,
        ego_port=1,
        seed_stage=int(PRIOR_SWEEP_SEED_STAGE.value),
        matchup_schedule_sha256=schedule_sha256,
        instant_match_restart=True,
        stage_policy="battlefield_then_random_legal",
        completion_policy="finish_in_flight_wave",
        active_frame_policy="frame_id_gte_0_exclude_zero_active",
        uncertainty_policy=f"bootstrap_boots_{BOOTSTRAP_RESAMPLES}",
        start_retries=DEFAULT_START_RETRIES,
        exec_horizon=exec_horizon,
        decode_temp=settings.temp,
        decode_temps=settings.temps,
        decode_btn_support_min=settings.btn_support_min,
        decode_min_p=settings.min_p,
        decode_click_trigger_fix=settings.click_trigger_fix,
        model_dtype=model_dtype,
    )


def _write_match_rows(path: Path, rows: list[MatchRow], protocol: EvalProtocol) -> None:
    """Atomically persist exact trajectory-derived rows plus pairing protocol."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "protocol": asdict(protocol),
        "rows": [row.as_dict() for row in rows],
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True))
    tmp.replace(path)


def _queue_periodic_eval_evidence(
    uploader: BackgroundUploader,
    *,
    run_dir: Path,
    replay_dir: Path,
    result_path: Path,
    log_path: Path,
) -> tuple[int, int]:
    replay_count = uploader.upload_tree(replay_dir, base=run_dir, pattern="*.slp")
    evidence = (
        replay_dir / "match_rows.json",
        replay_dir / "metrics.json",
        result_path,
        log_path,
    )
    evidence_count = 0
    for path in evidence:
        if path.is_file():
            evidence_count += uploader.upload(path, key=str(path.relative_to(run_dir)))
    return replay_count, evidence_count


def _run_eval_sweep(
    policy_factory: Callable[[], RecedingHorizon],
    *,
    protocol: EvalProtocol,
    replay_dir: Path | None,
    rows_path: Path | None,
    telemetry: DecodeTelemetry | None = None,
) -> dict[str, float]:
    def tracked_factory() -> Callable:
        policy = policy_factory()

        def tracked(frame_index: int, obs: dict) -> dict:
            if telemetry is not None:
                telemetry.record_policy_call(len(obs))
            return policy(frame_index, obs)

        return tracked

    started_at = time.perf_counter()
    results, rows = sweep_vs_cpu_prior_with_rows(
        tracked_factory,
        session_cfg=default_session_cfg(replay_dir, instant_match_restart=protocol.instant_match_restart),
        n_matchups=protocol.n_matchups,
        max_parallel=protocol.max_parallel,
        max_frames=protocol.max_frames,
        cpu_level=protocol.cpu_level,
        ego_port=protocol.ego_port,
        seed_stage=melee.Stage(protocol.seed_stage),
        start_retries=protocol.start_retries,
    )
    metrics = vs_cpu_metrics(results, seed=protocol.seed)
    metrics["eval_wall_seconds"] = time.perf_counter() - started_at
    if telemetry is not None:
        metrics.update(telemetry.metrics())
    if rows_path is not None:
        _write_match_rows(rows_path, rows, protocol)
        metrics_path = rows_path.with_name("metrics.json")
        tmp = metrics_path.with_suffix(metrics_path.suffix + ".tmp")
        tmp.write_text(json.dumps(metrics, sort_keys=True))
        tmp.replace(metrics_path)
    return metrics


def eval_vs_cpu(
    model: GPT,
    stats: dict[str, FeatureStats],
    cfg: TrainConfig,
    *,
    max_frames: int,
    replay_dir: Path | None = None,
    n_matchups: int | None = None,
    eval_seed: int | None = None,
    rows_path: Path | None = None,
) -> dict[str, float]:
    """In-training closed-loop eval vs lvl-9 CPU over prior-sampled char matchups.

    A fixed ``n_matchups`` controls statistical coverage; host CPU count controls only
    how many of those boots execute concurrently. Each policy wave gets an explicit,
    deterministic sampling seed. Reduced to a flat metric dict."""
    settings = _decode_settings(model, cfg)
    protocol = _eval_protocol(
        cfg,
        settings=settings,
        exec_horizon=cfg.exec_horizon,
        default_n_matchups=cfg.eval_n_matchups,
        model_dtype=str(next(model.parameters()).dtype),
        n_matchups=n_matchups,
        max_frames=max_frames,
        seed=eval_seed,
    )
    policy_index = itertools.count()
    telemetry = DecodeTelemetry()

    def policy_factory() -> RecedingHorizon:
        return make_policy(
            model,
            stats,
            cfg,
            exec_horizon=protocol.exec_horizon,
            decode_temp=settings.temp,
            decode_temps=settings.temps,
            decode_btn_support_min=settings.btn_support_min,
            decode_min_p=settings.min_p,
            decode_click_trigger_fix=settings.click_trigger_fix,
            decode_seed=protocol.seed + next(policy_index),
            telemetry=telemetry,
        )

    with _evaluation_mode(model):
        return _run_eval_sweep(
            policy_factory,
            protocol=protocol,
            replay_dir=replay_dir,
            rows_path=rows_path,
            telemetry=telemetry,
        )


# %%
def _loader_kwargs(cfg: TrainConfig, stats: dict[str, FeatureStats]) -> dict:
    """Loader arguments shared by the train and val splits. The loader yields MICRO-batches: one
    optimizer step consumes ``grad_accum_steps`` of them. ``schema_version`` declares which MDS
    materialization this run reads, so a stale (or newer) dataset raises on the first row."""
    return dict(
        data_root=cfg.data_root,
        remote=streams.remote_for_local(cfg.data_root),
        cache_limit=f"{cfg.cache_limit_gb}gb",
        shuffle_block_size=cfg.shuffle_block_size,
        stats=stats,
        L_ctx=cfg.L_ctx,
        L_chunk=max(cfg.head_offsets),  # target horizon must cover the farthest auxiliary head
        batch_size=_micro_batch(cfg),
        seed=cfg.seed,
        schema_version=cfg.mds_schema_version,
        projection=_INPUT_PROJECTION,
    )


def _start_data_loading(
    train_loader: Iterable[TrainBatch],
    val_loader: Iterable[TrainBatch],
    val_n_samples: int,
) -> tuple[
    Iterator[TrainBatch],
    concurrent.futures.ThreadPoolExecutor,
    concurrent.futures.Future[tuple[list[TrainBatch], float]],
    float,
]:
    train_iterator = iter(train_loader)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="val-cache")
    val_started_at = time.monotonic()

    def read_validation() -> tuple[list[TrainBatch], float]:
        batches: list[TrainBatch] = []
        samples = 0
        for batch in val_loader:
            batch_size = batch.target.shape[0]
            if batch_size <= 0:
                raise RuntimeError("validation loader yielded an empty batch")
            remaining = val_n_samples - samples
            if remaining <= 0:
                break
            batches.append(batch if batch_size <= remaining else _slice_batch(batch, remaining))
            samples += min(batch_size, remaining)
            if samples == val_n_samples:
                break
        if samples != val_n_samples:
            raise RuntimeError(f"validation loader yielded {samples} samples, expected {val_n_samples}")
        return batches, time.monotonic()

    future = pool.submit(read_validation)
    return train_iterator, pool, future, val_started_at


def _fetch_h2h_reference(cfg: TrainConfig, run_dir: Path) -> Path:
    """Download the reference final.pt at train START, so a bad reference fails in
    minute one, not after the full training run."""
    assert cfg.final_h2h_reference_run is not None
    ref_ckpt = download_latest(cfg.final_h2h_reference_run, run_dir / "h2h_reference", name="final.pt")
    if ref_ckpt is None:
        raise RuntimeError(f"no final.pt on R2 for reference run {cfg.final_h2h_reference_run!r}")
    with ref_ckpt.open("rb") as checkpoint_file:
        actual_sha = hashlib.file_digest(checkpoint_file, "sha256").hexdigest()
    expected_sha = cfg.final_h2h_reference_sha256
    if expected_sha is not None and actual_sha != expected_sha.lower():
        raise RuntimeError(f"reference checkpoint SHA-256 mismatch: expected {expected_sha.lower()}, got {actual_sha}")
    print(f"[h2h] reference checkpoint SHA-256: {actual_sha}", flush=True)
    return ref_ckpt


def _require_matched_h2h_protocol(self_protocol: Mapping[str, object], ref_protocol: Mapping[str, object]) -> None:
    fields = ("L_ctx", "model_dtype", "decode_settings")
    mismatches = {
        field: (self_protocol.get(field), ref_protocol.get(field))
        for field in fields
        if self_protocol.get(field) != ref_protocol.get(field)
    }
    if mismatches:
        raise RuntimeError(f"h2h protocol mismatch (challenger, reference): {mismatches}")


def _final_h2h(
    cfg: TrainConfig,
    model: GPT,
    stats: dict[str, FeatureStats],
    run_dir: Path,
    uploader: BackgroundUploader,
    ref_ckpt: Path,
) -> None:
    """Run mirrored H2H and upload each completed orientation."""
    from hal.scripts.h2h import ModelArgs
    from hal.scripts.h2h import load_policy_builder

    build_ref, ref_protocol = load_policy_builder(
        ModelArgs(
            name=cfg.final_h2h_reference_label,
            checkpoint=str(ref_ckpt),
            experiment=cfg.final_h2h_reference_experiment,
        )
    )
    self_dtype = str(next(model.parameters()).dtype)
    self_protocol = {
        "name": cfg.final_h2h_self_label,
        "experiment": str(Path(__file__)),
        "checkpoint": str(run_dir / "final.pt"),
        "step": cfg.max_steps,
        "L_ctx": cfg.L_ctx,
        "model_dtype": self_dtype,
        "decode_settings": asdict(_decode_settings(model, cfg)),
        "exec_horizon": cfg.exec_horizon,
        "head_offsets": list(cfg.head_offsets),
    }
    _require_matched_h2h_protocol(self_protocol, ref_protocol)

    def build_self(seed: int) -> RecedingHorizon:
        return make_policy(model, stats, cfg, decode_seed=seed)

    out_dir = run_dir / "h2h_final"

    def upload_orientation(_orientation: int) -> None:
        uploader.upload_tree(out_dir, base=run_dir)

    try:
        with _evaluation_mode(model):
            records = run_h2h(
                build_self,
                build_ref,
                name_a=cfg.final_h2h_self_label,
                name_b=cfg.final_h2h_reference_label,
                n_configs=cfg.final_h2h_n_configs,
                out_dir=out_dir,
                max_frames=cfg.eval_max_frames,
                max_parallel=_eval_max_parallel(cfg, cfg.final_h2h_n_configs),
                seed=cfg.eval_seed,
                meta={
                    "models": {
                        cfg.final_h2h_self_label: self_protocol,
                        cfg.final_h2h_reference_label: ref_protocol,
                    }
                },
                on_orientation_done=upload_orientation,
            )
    finally:
        uploader.upload_tree(out_dir, base=run_dir)
    summary = summarize_paired(records, focal_model=cfg.final_h2h_self_label)
    if wandb.run is not None:
        wandb.run.summary["h2h_final"] = summary.as_dict()
    print(summary.format_table(), flush=True)


def train(
    cfg: TrainConfig,
    stats: dict[str, FeatureStats],
    *,
    comment: str = "",
    resume_run: str | None = None,
    resume_state: dict | None = None,
) -> None:
    resumed_counts = resume_state is not None and "button_combo_counts" in resume_state["model"]
    button_combo_counts = None if resumed_counts else _load_button_combo_counts(cfg)
    validate_config(cfg, has_button_combo_counts=button_combo_counts is not None or resumed_counts)
    run_name = resume_run or make_run_name(Path(__file__).stem, _model_tag(cfg), cfg.data_root, comment)
    uploader = BackgroundUploader(run_name)
    wandb.init(
        project="hal",
        name=run_name,
        id=resume_state["wandb_id"] if resume_state else None,
        resume="allow" if resume_state else None,
        tags=["gpt", f"d{cfg.d_model}", f"L{cfg.n_layers}"],
        config=asdict(cfg),
    )
    # W&B's own step is a free-running monotonic timestamp; we plot everything against the training
    # step logged as data (``global_step``). This lets an async eval that *finishes* late be logged
    # at its *origin* step without violating step monotonicity.
    wandb.define_metric("global_step")
    wandb.define_metric("*", step_metric="global_step")
    ckpt_dir, replay_dir = setup_run_dir(run_name)
    # Fetch the h2h reference NOW: a bad reference must fail in minute one, not at hour three.
    h2h_ref_ckpt = _fetch_h2h_reference(cfg, ckpt_dir) if cfg.final_h2h_reference_run else None

    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high" if cfg.allow_tf32 else "highest")
    autocast = (
        torch.autocast(DEVICE, dtype=torch.bfloat16)
        if cfg.amp_dtype == "bfloat16" and DEVICE == "cuda"
        else contextlib.nullcontext()
    )
    start_step = resume_state["step"] + 1 if resume_state else 0
    model = GPT(cfg).to(DEVICE)
    if button_combo_counts is not None:
        model.button_combo_counts.copy_(button_combo_counts.to(DEVICE))
    if cfg.compile_trunk:
        # The BOUND method, not the module: reassigning ``model.trunk`` would rename every trunk key
        # in the state dict (``trunk._orig_mod.blocks…``) and break resume and eval.
        model.forward = torch.compile(model.forward, dynamic=False)
    counts = parameter_counts(model)
    n_params = counts["total"]
    if wandb.run is not None:
        wandb.run.summary["model/num_params"] = n_params
        for name, count in counts.items():
            wandb.run.summary[f"model/num_params_{name}"] = count
    print(
        f"[model] {_model_tag(cfg)}  num_params={n_params / 1e6:.2f}M  "
        f"batch={cfg.batch_size} (micro {_micro_batch(cfg)} x accum {cfg.grad_accum_steps})",
        flush=True,
    )
    loader_kwargs = _loader_kwargs(cfg, stats)
    if cfg.compact_data:
        train_loader = make_reservoir_loader(
            split="train",
            num_workers=cfg.num_workers,
            prefetch_factor=cfg.prefetch_factor,
            predownload=cfg.predownload,
            windows_per_replay=cfg.windows_per_replay,
            reservoir_capacity=cfg.reservoir_capacity,
            prefetch_batches=cfg.prefetch_batches,
            **loader_kwargs,
        )
    else:
        train_loader = make_loader(
            split="train",
            num_workers=cfg.num_workers,
            prefetch_factor=cfg.prefetch_factor,
            predownload=cfg.predownload,
            windows_per_replay=cfg.windows_per_replay,
            **loader_kwargs,
        )
    # Val uses the FROZEN wider chunk (VAL_L_CHUNK) so its window geometry — hence its NLL — is
    # comparable across experiments regardless of the train-time L_chunk. This makes val loss NOT
    # comparable to pre-freeze 012 runs; that break is the intended freeze. The val path slices the
    # wider target back to max(head_offsets) frames.
    val_loader = make_loader(
        split=cfg.val_split,
        num_workers=0,
        compact=cfg.compact_data,
        **{**loader_kwargs, "L_chunk": VAL_L_CHUNK},
    )

    opt = make_optimizer(model, cfg)
    sched = LambdaLR(opt, lr_schedule(cfg))
    if resume_state is not None:
        _load_model_state(model, resume_state["model"])
        opt.load_state_dict(resume_state["opt"])
        sched.load_state_dict(resume_state["sched"])
        print(f"[resume] {run_name}: continuing from step {start_step}", flush=True)

    print("[val] building cached val set while training prefetch starts…", flush=True)
    train_iter_t0 = time.monotonic()
    it, val_pool, val_future, val_started_at = _start_data_loading(train_loader, val_loader, cfg.val_n_samples)
    train_iterator_s = val_started_at - train_iter_t0
    val_cache: list[TrainBatch] | None = None

    def _resolve_val_cache(*, wait: bool) -> list[TrainBatch] | None:
        nonlocal val_cache
        if val_cache is not None:
            return val_cache
        if not wait and not val_future.done():
            return None
        try:
            val_cache, finished_at = val_future.result()
        finally:
            val_pool.shutdown(wait=False)
        if not val_cache:
            raise RuntimeError("val loader yielded zero batches")
        val_cache_s = finished_at - val_started_at
        print(
            f"[val] cached {len(val_cache)} batches "
            f"({sum(b.target.shape[0] for b in val_cache)} samples) in {val_cache_s:.1f}s",
            flush=True,
        )
        if wandb.run is not None:
            wandb.run.summary["startup/train_iterator_s"] = train_iterator_s
            wandb.run.summary["startup/val_cache_s"] = val_cache_s
        return val_cache

    def _wandb_id() -> str | None:
        return wandb.run.id if wandb.run is not None else None

    def _eval_and_upload(step_tag: str, *, n_matchups: int) -> dict[str, float]:
        """Synchronous closed-loop eval on the live model + .slp upload (the final eval).
        Returns the flat metric dict."""
        sub = replay_dir / step_tag
        rows_path = sub / "match_rows.json"
        metrics = eval_vs_cpu(
            model,
            stats,
            cfg,
            max_frames=cfg.eval_max_frames,
            replay_dir=sub,
            n_matchups=n_matchups,
            rows_path=rows_path,
        )
        n = uploader.upload_tree(sub, base=ckpt_dir, pattern="*.slp")
        uploader.upload(rows_path, key=str(rows_path.relative_to(ckpt_dir)))
        metrics_path = sub / "metrics.json"
        uploader.upload(metrics_path, key=str(metrics_path.relative_to(ckpt_dir)))
        print(f"[eval] queued {n} .slp + rows + metrics for R2 ({step_tag})", flush=True)
        return metrics

    def _val_log_dict() -> dict[str, object]:
        """Flat ``val/*`` metric dict (one W&B section). Merged into the per-step log; no wandb.log
        here. One pass uses two trunk forwards per batch."""
        cache = _resolve_val_cache(wait=True)
        assert cache is not None
        vm = val_metrics(model, cache, cfg)
        out: dict[str, object] = {f"val/{k}": v for k, v in vm.items()}
        out.update(gradient_diagnostics(model, cache[0].to(DEVICE), cfg))
        return out

    def _log_eval(step: int, metrics: dict[str, float]) -> None:
        """Sole eval-logging site: plot ``eval/*`` at the eval's origin ``global_step``."""
        wandb.log({**{f"eval/{k}": v for k, v in metrics.items()}, "global_step": step})
        print(f"[t+{time.monotonic() - run_t0:.0f}s] step {step}: closed_loop {metrics}", flush=True)

    def _save(name: str, step: int) -> None:
        save_checkpoint(
            ckpt_dir / name,
            step=step,
            model=model,
            opt=opt,
            sched=sched,
            cfg=asdict(cfg),
            wandb_id=_wandb_id(),
            uploader=uploader,
        )

    # At most one async eval in flight. The worker is a separate process (own GPU/CUDA + GIL) that
    # evals the just-saved checkpoint and writes a metrics JSON; the trainer drains it between steps.
    pending_eval: dict | None = None

    def _drain_eval(*, wait: bool) -> None:
        """Reap the in-flight eval. ``wait`` blocks (bounded by ``eval_timeout_seconds``) for the
        result; otherwise just polls. A worker over budget is killed. On success: log + upload .slp."""
        nonlocal pending_eval
        if pending_eval is None:
            return
        proc: subprocess.Popen = pending_eval["proc"]
        if wait:
            try:
                proc.wait(timeout=max(0.0, cfg.eval_timeout_seconds - (time.monotonic() - pending_eval["t0"])))
            except subprocess.TimeoutExpired:
                pass
        rc = proc.poll()
        timed_out = False
        if rc is None:
            if not wait and (time.monotonic() - pending_eval["t0"]) <= cfg.eval_timeout_seconds:
                return  # still running, within budget — re-check next iteration
            proc.kill()
            proc.wait()
            rc = proc.returncode
            timed_out = True
            print(
                f"[eval] step {pending_eval['step']} timed out (>{cfg.eval_timeout_seconds:.0f}s); "
                f"killed. see {pending_eval['log']}",
                flush=True,
            )
        pending_eval["log_f"].close()
        step, result = pending_eval["step"], pending_eval["result"]
        replay = pending_eval["replay"]
        if not timed_out and rc == 0 and result.is_file():
            data = json.loads(result.read_text())
            _log_eval(data["step"], data["metrics"])
        elif not timed_out:
            print(f"[eval] worker for step {step} failed (rc={rc}); see {pending_eval['log']}", flush=True)
        n, uploaded = _queue_periodic_eval_evidence(
            uploader,
            run_dir=ckpt_dir,
            replay_dir=replay,
            result_path=result,
            log_path=pending_eval["log"],
        )
        print(f"[eval] queued {n} .slp + {uploaded} evidence files for R2 (step {step})", flush=True)
        pending_eval = None

    def _launch_eval(step: int) -> None:
        """Save the checkpoint and spawn a background eval worker for it. Waits out any prior eval
        first (bounded), so only one runs at a time."""
        nonlocal pending_eval
        _drain_eval(wait=True)
        _save(f"step_{step:06d}.pt", step)
        result = ckpt_dir / "eval_results" / f"step_{step:06d}.json"
        log = ckpt_dir / "eval_logs" / f"step_{step:06d}.log"
        result.parent.mkdir(parents=True, exist_ok=True)
        log.parent.mkdir(parents=True, exist_ok=True)
        log_f = open(log, "w")  # noqa: SIM115 — spans the worker's lifetime; closed in _drain_eval
        proc = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--eval-worker",
                str(ckpt_dir / f"step_{step:06d}.pt"),
                "--eval-worker-step",
                str(step),
                "--eval-worker-result",
                str(result),
                "--eval-worker-replay",
                str(replay_dir / f"step_{step:06d}"),
            ],
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )
        pending_eval = {
            "step": step,
            "proc": proc,
            "result": result,
            "replay": replay_dir / f"step_{step:06d}",
            "log": log,
            "log_f": log_f,
            "t0": time.monotonic(),
        }
        print(f"[t+{time.monotonic() - run_t0:.0f}s] step {step}: launched async eval (pid {proc.pid})", flush=True)
        if not cfg.eval_overlap_training:
            # The worker and trainer otherwise inherit the same CUDA device. Waiting here makes
            # the subprocess asynchronous with respect to process setup/logging, but exclusive
            # with respect to GPU execution; this avoids the measured ~6 FPS/boot contention mode.
            _drain_eval(wait=True)

    model.train()
    run_t0 = time.monotonic()
    train_batches_seen = 0
    previous_step_replay_ids: frozenset[str] | None = None
    for step in range(start_step, cfg.max_steps):
        _resolve_val_cache(wait=False)
        with profile("step") as sw:
            opt.zero_grad()
            comps_acc: dict[tuple[int, str], list[Tensor]] = {}
            obj_acc: Tensor | None = None
            loader_wait = 0.0
            replay_ids: set[str] = set()
            finished_epoch_stats: dict[str, int] | None = None
            for _ in range(cfg.grad_accum_steps):
                wait_start = time.monotonic()
                try:
                    batch = next(it)
                except StopIteration:
                    finished_epoch_stats = getattr(train_loader, "last_epoch_stats", None)
                    it = iter(train_loader)
                    batch = next(it)
                train_batches_seen += 1
                loader_wait += time.monotonic() - wait_start
                if batch.replay_ids is not None:
                    replay_ids.update(batch.replay_ids)
                batch = batch.to(DEVICE)
                with autocast:
                    parts = action_loss(model, batch)
                    obj = objective(
                        parts.nll,
                        parts.transition,
                        cfg.aux_loss_weight,
                        cfg.transition_loss_weight,
                    )
                    loss = obj / cfg.grad_accum_steps
                loss.backward()
                obj_acc = obj.detach() if obj_acc is None else obj_acc + obj.detach()
                for k, v in parts.nll.items():
                    comps_acc.setdefault(k, []).append(v.detach())
            assert obj_acc is not None
            grad_norm = _finite_gradient_norm(model, obj_acc / cfg.grad_accum_steps, step)
            adapter_gradient_log: dict[str, float] = {}
            if model.head_adapter is not None:
                adapter_gradient_log = {
                    "train/gnorm_state_projection": _parameter_gradient_norm(
                        model.head_adapter.state_proj.parameters()
                    ),
                    "train/gnorm_residual_output": _parameter_gradient_norm(
                        model.head_adapter.residual_projs.parameters()
                    ),
                }
                if isinstance(model.head_adapter, FactoredMLPAdapter):
                    adapter_gradient_log.update(
                        {
                            "train/gnorm_condition_projection": _parameter_gradient_norm(
                                model.head_adapter.condition_projs.parameters()
                            ),
                            "train/gnorm_action_embedding": _parameter_gradient_norm(
                                model.head_adapter.action_embeddings.parameters()
                            ),
                            "train/norm_condition_projection": _parameter_norm(
                                model.head_adapter.condition_projs.parameters()
                            ),
                            "train/norm_action_embedding": _parameter_norm(
                                model.head_adapter.action_embeddings.parameters()
                            ),
                        }
                    )
            opt.step()
            sched.step()
            if DEVICE == "cuda":
                torch.cuda.synchronize()
        objective_bits = (obj_acc / cfg.grad_accum_steps).item() / _LN2  # the actual backprop objective, bits
        comps_cat = {k: torch.cat(v) for k, v in comps_acc.items()}
        primary = nll_breakdown({name: comps_cat[(1, name)] for name in _GROUP_NAMES})
        aux_offsets = [o for o in cfg.head_offsets if o != 1]
        aux_loss = (
            sum(_offset_total_bits(comps_cat, o) for o in aux_offsets) / len(aux_offsets) if aux_offsets else 0.0
        )
        expected_batches = (step - start_step + 1) * cfg.grad_accum_steps
        if train_batches_seen != expected_batches:
            raise RuntimeError(
                f"step {step}: consumed {train_batches_seen} train batches, expected {expected_batches}"
            )
        sps = cfg.batch_size / sw.elapsed  # batch_size is the effective batch: one step's samples
        samples = (step + 1) * cfg.batch_size
        log: dict[str, object] = {
            "global_step": step,
            "samples": samples,
            "tokens": samples * cfg.L_ctx,
            "train/loss": primary["total"],  # offset-1 head (deployed), UNWEIGHTED, so the arms compare
            **{f"train/nll_{name}": primary[name] for name in _GROUP_NAMES},
            "train/aux_loss": aux_loss,  # mean total bits/frame over the auxiliary (offset != 1) heads
            "train/objective": objective_bits,
            "lr/muon": next(g["lr"] for g in opt.param_groups if g["use_muon"]),
            "lr/adam": next(g["lr"] for g in opt.param_groups if not g["use_muon"]),
            "train/gnorm": grad_norm.item(),
            "throughput/step_s": sw.elapsed,
            "throughput/loader_wait_s": loader_wait,
            "throughput/samples_per_s": sps,
            "throughput/tokens_per_s": sps * cfg.L_ctx,
            "data/train_batches_seen": train_batches_seen,
            **adapter_gradient_log,
        }
        if DEVICE == "cuda":
            log["hardware/peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 2**30
            log["hardware/peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30
        if replay_ids:
            log["data/distinct_replays"] = len(replay_ids)
            replay_overlap = _replay_overlap(previous_step_replay_ids, replay_ids)
            if replay_overlap is not None:
                log["data/replays_reused_from_previous_step"] = replay_overlap
            previous_step_replay_ids = frozenset(replay_ids)
        if finished_epoch_stats is not None:
            log.update({f"data/epoch_{name}": value for name, value in finished_epoch_stats.items()})
        if step == start_step:
            # The trunk resolves flex-vs-dense at its first forward, so the answer exists only now.
            if wandb.run is not None:
                wandb.run.summary["model/attn_path"] = model.trunk.attn_path
                wandb.run.summary["startup/compiled_step0_s"] = sw.elapsed
                wandb.run.summary["startup/step0_loader_wait_s"] = loader_wait
            print(f"[model] attention path: {model.trunk.attn_path}, window={cfg.attn_window}", flush=True)
        if step < 20 or step % 50 == 0:
            print(
                f"[t+{time.monotonic() - run_t0:.0f}s] step {step}: loss {primary['total']:.4f} "
                f"step_dt={sw.elapsed * 1000:.0f}ms ({sps:.1f} samples/s)",
                flush=True,
            )
        if cfg.ckpt_every > 0 and step > 0 and step % cfg.ckpt_every == 0:
            _save("latest.pt", step)
        if cfg.val_every > 0 and step > 0 and step % cfg.val_every == 0:
            vm = _val_log_dict()
            log.update(vm)
            print(
                f"[t+{time.monotonic() - run_t0:.0f}s] step {step}: "
                f"action_nll {vm['val/loss']:.3f} btn_logloss {vm['val/btn_logloss']:.3f}",
                flush=True,
            )
        wandb.log(log)
        _drain_eval(wait=False)
        if cfg.eval_every > 0 and step > 0 and step % cfg.eval_every == 0:
            _launch_eval(step)

    close_train_iterator = getattr(it, "close", None)
    if close_train_iterator is not None:
        close_train_iterator()
    _drain_eval(wait=True)  # finish the last async eval before the final pass
    vm_final = _val_log_dict()
    wandb.log({**vm_final, "global_step": cfg.max_steps})
    print(f"[final] action_nll {vm_final['val/loss']:.3f}", flush=True)
    # Save BEFORE the closed-loop eval. A box that cannot boot Dolphin (the H100's glibc is older
    # than the build's floor) otherwise crashes at the finish line and the weights are lost.
    _save("final.pt", cfg.max_steps)
    # Periodic and manual evaluation load the saved checkpoint through _load_ckpt, which applies
    # the configured decode dtype. Match that protocol for the live final evaluator and H2H model.
    _prepare_final_decode_model(model, cfg)
    if cfg.final_eval_n_matchups > 0:
        _log_eval(cfg.max_steps, _eval_and_upload("final", n_matchups=cfg.final_eval_n_matchups))
    else:
        print("[final] closed-loop eval skipped (final_eval_n_matchups=0)", flush=True)
    try:
        if cfg.final_h2h_reference_run:
            assert h2h_ref_ckpt is not None
            _final_h2h(cfg, model, stats, ckpt_dir, uploader, ref_ckpt=h2h_ref_ckpt)
    finally:
        # Drain pending uploads even when the h2h raises: the box can self-destruct after exit,
        # so everything already written must reach R2.
        uploader.close()


# %%
def _cfg_from_state(saved: dict) -> TrainConfig:
    """Load known configuration fields and ignore removed host settings."""
    known = {f.name for f in fields(TrainConfig)}
    dropped = sorted(set(saved) - known)
    if dropped:
        print(f"[ckpt] dropping {len(dropped)} stale cfg key(s) not on current TrainConfig: {dropped}", flush=True)
    values = {k: v for k, v in saved.items() if k in known}
    for name in ("head_offsets", "action_group_order", "decode_temps"):
        if values.get(name) is not None:
            values[name] = tuple(values[name])
    # Older checkpoints used the 512-row table.
    values.setdefault("action_vocab", 512)
    values.setdefault("compact_data", False)
    return TrainConfig(**values)


def _load_ckpt(ckpt_path: str) -> tuple[GPT, TrainConfig, dict[str, FeatureStats], dict]:
    """The one door every eval path (``--eval``, the async eval worker, ``hal.scripts.h2h``) loads a
    checkpoint through, so the decode-speed settings live here and no entry point can forget them."""
    # train() sets this per step; eval never did, and the default ("highest") costs the closed-loop
    # forward 1.09x for precision the sampler throws away.
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = _cfg_from_state(state["cfg"])
    torch.set_float32_matmul_precision("high" if cfg.allow_tf32 else "highest")
    embedded_counts = "button_combo_counts" in state["model"]
    button_combo_counts = None if embedded_counts else _load_button_combo_counts(cfg)
    validate_config(cfg, has_button_combo_counts=embedded_counts or button_combo_counts is not None)
    model = GPT(cfg).to(DEVICE)
    if button_combo_counts is not None:
        model.button_combo_counts.copy_(button_combo_counts.to(DEVICE))
    _load_model_state(model, state["model"])
    model.eval()
    if cfg.eval_fp16 and DEVICE == "cuda":
        _halve_for_decode(model)
    stats = load_consolidated_stats(Path(cfg.data_root) / "stats.json")
    return model, cfg, stats, state


def _halve_for_decode(model: GPT) -> None:
    """Cast the model's float parameters to fp16, and put the quantization grids back to fp32.

    The grids are the decode's OUTPUT scale: a stick center sits on the 1/80 grid and every stored
    value must reproduce its exact byte through the pipe, which fp16's ~5e-4 of relative slack
    cannot promise. The trunk and the heads have no such contract — their logits are cast back to
    fp32 before the softmax."""
    grids = {name: getattr(model, name) for name in ("main_centers", "c_centers", "trig_centers")}
    model.half()
    for name, grid in grids.items():
        setattr(model, name, grid)  # the ORIGINAL tensor: a fp32->fp16->fp32 round trip loses 2e-4


def _prepare_final_decode_model(model: GPT, cfg: TrainConfig) -> None:
    if cfg.eval_fp16 and DEVICE == "cuda":
        _halve_for_decode(model)


def eval_ckpt(
    ckpt_path: str,
    *,
    eval_output_dir: str | None = None,
    eval_exec_horizon: int | None = None,
    decode_temp: float | None = None,
    decode_temps: tuple[float, float, float, float] | None = None,
    decode_btn_support_min: int | None = None,
    decode_min_p: float | None = None,
    decode_click_trigger_fix: bool | None = None,
    eval_n_matchups: int | None = None,
    eval_max_parallel: int | None = None,
    eval_max_frames: int | None = None,
    eval_seed: int | None = None,
    wandb_run_id: str | None = None,
    wandb_project: str = "hal",
    wandb_entity: str | None = None,
    wandb_label: str | None = None,
) -> dict[str, float]:
    """Load a checkpoint and run the closed-loop evaluation."""
    model, cfg, stats, state = _load_ckpt(ckpt_path)
    exec_horizon = cfg.exec_horizon if eval_exec_horizon is None else eval_exec_horizon
    settings = _decode_settings(
        model,
        cfg,
        temp=decode_temp,
        temps=decode_temps,
        btn_support_min=decode_btn_support_min,
        min_p=decode_min_p,
        click_trigger_fix=decode_click_trigger_fix,
    )
    protocol = _eval_protocol(
        cfg,
        settings=settings,
        exec_horizon=exec_horizon,
        default_n_matchups=cfg.final_eval_n_matchups,
        model_dtype=str(next(model.parameters()).dtype),
        n_matchups=eval_n_matchups,
        max_parallel=eval_max_parallel,
        max_frames=eval_max_frames,
        seed=eval_seed,
    )
    _exec_horizon_offsets(model.head_offsets, exec_horizon)
    print(
        f"[eval] loaded {ckpt_path}  step={state['step']}  device={DEVICE}  exec_horizon={exec_horizon}  "
        f"temp={settings.temp}  temps={settings.temps}  btn_support_min={settings.btn_support_min}  "
        f"min_p={settings.min_p}  click_trigger_fix={settings.click_trigger_fix}",
        flush=True,
    )
    replay_dir = (
        Path(eval_output_dir).resolve()
        if eval_output_dir is not None
        else Path(ckpt_path).resolve().parent / "eval_replays"
    )
    replay_dir.mkdir(parents=True, exist_ok=True)
    policy_index = itertools.count()
    telemetry = DecodeTelemetry()

    def policy_factory() -> RecedingHorizon:
        return make_policy(
            model,
            stats,
            cfg,
            exec_horizon=protocol.exec_horizon,
            decode_temp=settings.temp,
            decode_temps=settings.temps,
            decode_btn_support_min=settings.btn_support_min,
            decode_min_p=settings.min_p,
            decode_click_trigger_fix=settings.click_trigger_fix,
            decode_seed=protocol.seed + next(policy_index),
            telemetry=telemetry,
        )

    print(
        f"\n[eval] ===== vs-cpu, {protocol.n_matchups} prior-sampled matchups, "
        f"max_parallel={protocol.max_parallel} (instant-restart, seed={protocol.seed}) =====",
        flush=True,
    )
    metrics = _run_eval_sweep(
        policy_factory,
        protocol=protocol,
        replay_dir=replay_dir,
        rows_path=replay_dir / "match_rows.json",
        telemetry=telemetry,
    )
    if wandb_run_id is not None:
        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            id=wandb_run_id,
            resume="allow",
            config={
                "manual_eval_checkpoint": str(Path(ckpt_path).resolve()),
                "manual_eval_label": wandb_label or "",
            },
        )
        wandb.define_metric("global_step")
        wandb.define_metric("eval/*", step_metric="global_step")
        protocol_log = {
            f"eval_protocol/{key}": value
            for key, value in asdict(protocol).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        labeled_log = (
            {f"eval_manual/{wandb_label}/{key}": value for key, value in metrics.items()} if wandb_label else {}
        )
        wandb.log(
            {
                **{f"eval/{key}": value for key, value in metrics.items()},
                **labeled_log,
                **protocol_log,
                "global_step": state["step"],
            }
        )
        run.summary["eval/last_checkpoint"] = str(Path(ckpt_path).resolve())
        run.summary["eval/last_label"] = wandb_label or "manual"
        run.summary["eval/model_dtype"] = protocol.model_dtype
        wandb.finish()
    print(f"  {metrics}", flush=True)
    return metrics


# %%
def run_eval_worker(ckpt_path: str, step: int, result_path: str, replay_dir: str) -> None:
    """One-shot closed-loop eval for the async path: load a checkpoint, sweep vs CPU, and write the
    flat metric dict to ``result_path`` (atomically) with the .slp recordings under ``replay_dir``.
    Touches neither W&B nor R2 — the launching trainer is the sole writer/uploader."""
    model, cfg, stats, _ = _load_ckpt(ckpt_path)
    replay_path = Path(replay_dir)
    metrics = eval_vs_cpu(
        model,
        stats,
        cfg,
        max_frames=cfg.eval_max_frames,
        replay_dir=replay_path,
        rows_path=replay_path / "match_rows.json",
    )
    out = Path(result_path)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps({"step": step, "metrics": metrics}))
    tmp.replace(out)  # atomic rename: the trainer never reads a partial file
    print(f"[eval-worker] step {step}: {metrics}", flush=True)


# %%
@dataclass
class Args:
    """Top-level CLI surface. Pass TrainConfig fields as kebab-case flags, e.g. ``--cfg.d-model 512``."""

    cfg: TrainConfig = field(default_factory=TrainConfig)
    eval: str | None = None  # ckpt path; closed-loop eval instead of train
    eval_run: str | None = None  # R2 run name; download eval_checkpoint_name, evaluate, and upload evidence
    eval_checkpoint_name: str = "final.pt"
    eval_output_dir: str | None = None
    eval_exec_horizon: int | None = None  # override execution horizon s for --eval (chunked decode; 1=per-frame)
    eval_temp: float | None = None  # override decode temperature for --eval
    eval_temps: tuple[float, float, float, float] | None = None  # per-group temps (buttons, main, c, triggers)
    eval_btn_support_min: int | None = None  # mask button combos with < this many train frames (0=off)
    eval_min_p: float | None = None  # min-p nucleus: keep classes with p >= min_p * p_max
    eval_click_trigger_fix: bool | None = None  # force trigger_l/r to 1.0 on a digital L/R click
    eval_n_matchups: int | None = None  # manual --eval override; default is cfg.final_eval_n_matchups (96)
    eval_max_parallel: int | None = None  # manual --eval override; default is saved in the checkpoint
    eval_max_frames: int | None = None  # manual --eval override; default is checkpoint cfg.eval_max_frames
    eval_seed: int | None = None  # manual --eval sampling/bootstrap seed override
    wandb_run_id: str | None = None  # resume an existing run and log this manual eval to it
    wandb_project: str = "hal"
    wandb_entity: str | None = None
    wandb_label: str | None = None
    resume: str | None = None  # run_name to resume; pulls latest.pt (local, else R2)
    comment: str = ""
    # internal: one-shot async-eval worker (the trainer spawns this; not for manual use).
    eval_worker: str | None = None  # ckpt path
    eval_worker_step: int = 0
    eval_worker_result: str | None = None
    eval_worker_replay: str | None = None


def main(args: Args) -> None:
    if args.eval_worker is not None:
        assert args.eval_worker_result is not None and args.eval_worker_replay is not None
        run_eval_worker(args.eval_worker, args.eval_worker_step, args.eval_worker_result, args.eval_worker_replay)
        return
    selected_modes = sum(value is not None for value in (args.eval, args.eval_run, args.resume))
    if selected_modes > 1:
        raise SystemExit("pass only one of --eval, --eval-run, or --resume")
    if args.eval is not None or args.eval_run is not None:
        eval_path = args.eval
        output_dir = args.eval_output_dir
        eval_label = args.wandb_label
        uploader: BackgroundUploader | None = None
        upload_base: Path | None = None
        if args.eval_run is not None:
            if Path(args.eval_run).name != args.eval_run:
                raise SystemExit("--eval-run must be one run-name path segment")
            if Path(args.eval_checkpoint_name).name != args.eval_checkpoint_name:
                raise SystemExit("--eval-checkpoint-name must be one file name")
            label = eval_label or f"manual-{Path(args.eval_checkpoint_name).stem}"
            if Path(label).name != label:
                raise SystemExit("--wandb-label must be one path segment with --eval-run")
            run_dir = Path("runs") / args.eval_run
            checkpoint = download_latest(
                args.eval_run,
                run_dir / "manual_checkpoints",
                name=args.eval_checkpoint_name,
            )
            if checkpoint is None:
                raise SystemExit(f"no {args.eval_checkpoint_name} for run {args.eval_run!r}")
            eval_path = str(checkpoint)
            output_path = Path(output_dir) if output_dir is not None else run_dir / "manual_evals" / label
            output_path = output_path.resolve()
            upload_base = run_dir.resolve()
            if not output_path.is_relative_to(upload_base):
                raise SystemExit("--eval-output-dir must be inside runs/<eval-run> so evidence can upload")
            output_dir = str(output_path)
            eval_label = label
            uploader = BackgroundUploader(args.eval_run)
        assert eval_path is not None
        try:
            eval_ckpt(
                eval_path,
                eval_output_dir=output_dir,
                eval_exec_horizon=args.eval_exec_horizon,
                decode_temp=args.eval_temp,
                decode_temps=args.eval_temps,
                decode_btn_support_min=args.eval_btn_support_min,
                decode_min_p=args.eval_min_p,
                decode_click_trigger_fix=args.eval_click_trigger_fix,
                eval_n_matchups=args.eval_n_matchups,
                eval_max_parallel=args.eval_max_parallel,
                eval_max_frames=args.eval_max_frames,
                eval_seed=args.eval_seed,
                wandb_run_id=args.wandb_run_id,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                wandb_label=eval_label,
            )
        finally:
            if uploader is not None:
                assert output_dir is not None and upload_base is not None
                count = uploader.upload_tree(Path(output_dir), base=upload_base)
                print(f"[eval] queued {count} evidence files for R2", flush=True)
                uploader.close()
        return
    if args.resume is not None:
        state = load_for_resume(args.resume, Path("runs") / args.resume, device=DEVICE)
        if state is None:
            raise SystemExit(f"no latest.pt for run {args.resume!r} (local or R2)")
        # Only pure host-scaling knobs (worker/prefetch counts) follow the current code; the
        # model-identity knobs MUST come from the checkpoint so a resume can't silently change them.
        d = TrainConfig()
        cfg = replace(_cfg_from_state(state["cfg"]), num_workers=d.num_workers, prefetch_factor=d.prefetch_factor)
        stats = load_consolidated_stats(Path(cfg.data_root) / "stats.json")
        train(cfg, stats, resume_run=args.resume, resume_state=state)
        return
    cfg = args.cfg
    stats = load_consolidated_stats(Path(cfg.data_root) / "stats.json")
    comment = args.comment or f"gpt-{cfg.max_steps // 1000}k-b{cfg.batch_size}"
    train(cfg, stats, comment=comment)


if __name__ == "__main__":
    main(tyro.cli(Args))
