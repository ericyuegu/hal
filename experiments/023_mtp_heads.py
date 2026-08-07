"""Train the normalized independent-head MTP baseline."""

# %%
import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import concurrent.futures
import contextlib
import functools
import itertools
import json
import math
import subprocess
import sys
import time
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from dataclasses import replace
from pathlib import Path
from typing import Literal

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
from hal.eval.cross_stage import MatchRow
from hal.eval.cross_stage import sweep_vs_cpu_prior_with_rows
from hal.eval.cross_stage import vs_cpu_metrics
from hal.eval.h2h import run_h2h
from hal.eval.harness import default_session_cfg
from hal.eval.harness import usable_cpus
from hal.eval.paired import summarize_paired
from hal.training import scoring
from hal.training.checkpoints import BackgroundUploader
from hal.training.checkpoints import download_latest
from hal.training.checkpoints import load_for_resume
from hal.training.checkpoints import save_checkpoint
from hal.training.closed_loop import RecedingHorizon
from hal.training.dataloader import VAL_L_CHUNK
from hal.training.dataloader import make_loader
from hal.training.dataloader import make_replay_reservoir_loader
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
    # Total weight of the mean auxiliary-head loss. This stays fixed if head_offsets changes.
    aux_loss_weight: float = 1.0
    # Drop the complete ego action history for a training sample.
    history_dropout_p: float = 0.0
    # Weight action transitions inside each group loss. One disables this weighting.
    transition_loss_weight: float = 1.0
    # Matchup conditioning (schema v4). char/stage embeddings are indexed by the RAW libmelee id
    # (characters 0-26 dense; stages sparse in 0-26), so the vocab must exceed the max id, not the
    # number of included categories; out-of-range ids clamp to the last row.
    char_vocab: int = 32
    char_dim: int = 12
    stage_vocab: int = 32
    stage_dim: int = 4
    # Ranked v7 contains action states through 525. Older checkpoints used 512 rows.
    action_vocab: int = 1024
    # closed-loop sampling temperature. Greedy argmax collapses the policy to a do-nothing fixed
    # point in closed loop, so deployed play always samples.
    decode_temp: float = 1.0
    # Decode-time hygiene (all default to current behavior). decode_temps overrides the single decode_temp
    # per group in _GROUP_NAMES order (buttons, main_stick, c_stick, triggers); None -> decode_temp for all.
    # decode_btn_support_min >= 1 masks button combos with fewer than that many train frames
    # according to the configured dataset-scoped count artifact to -inf before softmax/argmax.
    # decode_min_p > 0 keeps only classes with
    # p >= decode_min_p * p_max per group, then renormalizes. decode_click_trigger_fix forces trigger_l/r to
    # 1.0 wherever the sampled combo sets the digital L/R bit (the only train-supported click joint).
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
    # torch.compile the model's forward for TRAINING. About 40% of the step's GPU time is unfused
    # elementwise work and autocast casts, against 26% in the matmuls, which is what a fused graph
    # takes back: measured 406 -> 231 ms per step (1.76x) at this geometry on a 3060, over a 30-step
    # bench whose loss trajectory tracks eager to 1e-6 relative. A compiled graph that then meets a
    # validation or gradient-diagnostic shape dies inside FlexAttention with a CUDA illegal memory
    # access, so every non-training forward runs eager — see ``_evaluation_mode``, which is the one
    # door they all go through.
    compile_trunk: bool = True
    # eval cadence
    val_every: int = 1024
    # Maximum validation batches. The current v7 validation split is smaller than this cap.
    val_n_batches: int = 32
    # Examples used for per-head shared-trunk gradient comparisons.
    gradient_diagnostic_batch_size: int = 64
    # Validation-only rarity threshold. The metric is emitted only when the checkpoint embeds validated
    # full-dataset button counts; it never falls back to the old reference-sample table.
    diagnostic_rare_button_count: int = 100
    # Closed-loop evaluation cadence and per-boot frame budget.
    eval_every: int = 4096
    eval_max_frames: int = 7200
    # These sample counts do not depend on host concurrency.
    eval_n_matchups: int = 32
    final_eval_n_matchups: int = 96
    eval_seed: int = 0
    # Closed-loop eval parallelism scales with the box: max_parallel = round(this * cpu_count).
    # Each parallel slot is one Dolphin boot that (via instant-restart) plays many prior-sampled
    # matches back-to-back. Profile {0.5, 1, 2, 4} for the best eval throughput vs trainer impact.
    eval_parallel_per_cpu: float = 1.0
    # Closed-loop eval runs in a background subprocess. By default the trainer waits for that
    # subprocess before resuming, so evaluator and trainer never contend for the same CUDA device.
    # Set true only when the evaluator has an actually separate GPU (or intentionally accepts
    # same-device contention).
    eval_overlap_training: bool = False
    # Opt-in decode ablation for SWA models. Full causal attention must rebuild the raw window.
    eval_incremental_kv: bool = False
    # Cast the model's float parameters to fp16 for closed-loop decode. The decode is launch-bound,
    # not precision-bound: fp16 weights + fp16 context measured 1.7x the fp32 forward, and the
    # sampled action is unchanged (the stick/trigger centers and every logit stay fp32). Autocast is
    # NOT the same thing and measured SLOWER — this casts the weights once, at load.
    eval_fp16: bool = True
    # If an eval is still running at the next boundary, the trainer waits up to this bound and
    # then kills the worker.
    eval_timeout_seconds: float = 2700.0
    # Final in-process mirrored h2h vs a reference run. The cloud box can self-destruct after
    # training, so the sweep runs inside train() and its records/replays upload before exit.
    # None disables the sweep.
    # The reference is the BC arm of THIS experiment: the geometry moved, so no older run is a
    # valid control.
    final_h2h_reference_run: str | None = None
    final_h2h_reference_experiment: str = "experiments/023_mtp_heads.py"
    final_h2h_reference_label: str = "023-e0"
    final_h2h_self_label: str = "023-challenger"
    final_h2h_n_configs: int = 64
    # checkpointing
    ckpt_every: int = 2048
    # data
    data_root: str = "data/processed/ranked-anonymized-1/mds-policy-v7"
    compact_data: bool = True
    # MDS materialization this run reads. The dataloader's per-row guard rejects any other version
    # — never silent.
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


def _model_tag(cfg: TrainConfig) -> str:
    offs = ".".join(str(o) for o in cfg.head_offsets)
    attention = "full" if cfg.attn_window == 0 else f"swa{cfg.attn_window}"
    decode = "kv" if cfg.eval_incremental_kv else "recompute"
    return (
        f"gpt-d{cfg.d_model}-L{cfg.n_layers}-h{cfg.n_heads}-Lc{cfg.L_ctx}-a{cfg.action_vocab}-"
        f"{attention}-{decode}-o{offs}-linear"
    )


def _micro_batch(cfg: TrainConfig) -> int:
    """Samples per forward/backward. ``batch_size`` is the effective batch, split into
    ``grad_accum_steps`` equal micro-batches; ``validate_config`` pins that they divide."""
    return cfg.batch_size // cfg.grad_accum_steps


def _eval_max_parallel(cfg: TrainConfig, n_matchups: int) -> int:
    """Concurrent Dolphin boots per wave; never changes the fixed statistical sample size."""
    return min(n_matchups, max(1, round(cfg.eval_parallel_per_cpu * usable_cpus())))


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

    def sample(
        self,
        h: Tensor,
        *,
        group_temps: tuple[float, ...],
        btn_dead: Tensor | None,
        min_p: float,
        argmax: bool,
        gen: torch.Generator | None,
    ) -> Int[Tensor, "B n_groups"]:
        """Sample each action group from its own logit slice."""
        logits = self.logits(h)
        picks: list[Tensor] = []
        for g, name in enumerate(_GROUP_NAMES):
            lg = logits[name].float()
            if btn_dead is not None and name == "buttons":
                lg = lg.masked_fill(btn_dead, float("-inf"))
            if argmax:
                pick = lg.argmax(-1)
            else:
                probs = F.softmax(lg / group_temps[g], dim=-1)
                if min_p > 0:
                    probs = probs * (probs >= min_p * probs.amax(dim=-1, keepdim=True))
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                pick = torch.multinomial(probs, 1, generator=gen).squeeze(-1)
            picks.append(pick)
        return torch.stack(picks, dim=-1)


class GPT(nn.Module):
    """Causal frame model with one independent linear head per offset."""

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

        # Gamestate categoricals: one table per feature name, shared across the four players.
        self.cat_specs = {**CAT_FEATURES, "action": (cfg.action_vocab, CAT_FEATURES["action"][1])}
        self.cat_embeds = nn.ModuleDict(
            {name: nn.Embedding(vocab, dim) for name, (vocab, dim) in self.cat_specs.items()}
        )
        self.char_emb = nn.Embedding(cfg.char_vocab, cfg.char_dim)
        self.stage_emb = nn.Embedding(cfg.stage_vocab, cfg.stage_dim)
        per_player = len(FLOAT_FEATURES) * 2 + sum(dim for _, dim in CAT_FEATURES.values())  # float+mask+cat
        d_in = len(_PLAYER_PREFIXES) * per_player + A_DIM + 2 * cfg.char_dim + cfg.stage_dim  # 374

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

        # Stick/trigger center grids (registered so they move with .to() and serialize).
        self.register_buffer("main_centers", scoring.STICK_CLUSTER_CENTERS_MAIN.clone())
        self.register_buffer("c_centers", scoring.STICK_CLUSTER_CENTERS_C.clone())
        self.register_buffer("trig_centers", scoring.TRIGGER_CENTERS.clone())
        # -1 means unavailable. A validated full-dataset artifact populates this at train start, and the
        # buffer then travels with checkpoints so later eval never depends on a mutable sidecar file.
        self.register_buffer("button_combo_counts", torch.full((scoring.N_BUTTON_COMBOS,), -1, dtype=torch.long))
        self._btn_support_dead_cache: dict[tuple[int, torch.device], Tensor] = {}

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
        # Ego controller history slice, assembled into a FRESH tensor (never mutating `features`, so the
        # targets built from stack_actions(features) stay intact). Per-sample history dropout (train only):
        # draw a Bernoulli keep-mask via the module RNG (probability history_dropout_p of dropping) and zero
        # the whole slice for dropped samples, forcing the trunk off copying its own recent inputs.
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
        """Backbone hidden (one rmsnorm'd vector per frame); callers apply the per-offset heads."""
        return self.trunk(self._context_tokens(features), ctx_pad)

    @torch.no_grad()
    def forward_incremental(
        self,
        features: dict[str, Tensor],
        past: list[tuple[Tensor, Tensor] | None],
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        """Encode one current token using per-layer rolling KV state. The trunk answers with the
        whole (one-token) sequence, so the last position is taken here."""
        h, new_past = self.trunk.forward_incremental(self._context_tokens(features), past)
        return h[:, -1], new_past


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
    for hi, o in enumerate(model.head_offsets):
        tgt_idx = q_full[:, o : o + L_ctx]  # target frame i+o
        logits = {name: lg.float() for name, lg in model.heads[hi].logits(h).items()}
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


def _sample_action(
    model: GPT,
    head: IndependentHead,
    h: Tensor,
    *,
    group_temps: tuple[float, ...],
    btn_support_min: int,
    min_p: float,
    click_trigger_fix: bool,
    argmax: bool,
    gen: torch.Generator | None,
) -> Float[Tensor, "B d_action"]:
    """Sample one action vector from an independent joint output head."""
    dead = _btn_support_dead(model, btn_support_min, h.device) if btn_support_min >= 1 else None
    idx = head.sample(h, group_temps=group_temps, btn_dead=dead, min_p=min_p, argmax=argmax, gen=gen)
    a = _dequantize(model, idx)  # [B, A_DIM]
    if click_trigger_fix:
        a[..., _TRIGGER_L_CH] = torch.where(a[..., _BUTTON_L_CH] > 0.5, 1.0, a[..., _TRIGGER_L_CH])
        a[..., _TRIGGER_R_CH] = torch.where(a[..., _BUTTON_R_CH] > 0.5, 1.0, a[..., _TRIGGER_R_CH])
    return a


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
    """Sample the next action from the offset-1 head."""
    group_temps = _resolve_decode_args(temp, temps, btn_support_min, min_p, argmax)
    h = model(ctx.features, ctx.ctx_pad)[:, -1]  # [B, d_model]
    a = _sample_action(
        model,
        model.heads[model.primary_head_idx],
        h,
        group_temps=group_temps,
        btn_support_min=btn_support_min,
        min_p=min_p,
        click_trigger_fix=click_trigger_fix,
        argmax=argmax,
        gen=gen,
    )
    return a[:, None, :]  # [B, 1, A_DIM]


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
    """Chunked receding-horizon decode: from ONE backbone forward's last hidden state, sample the action at
    each offset in ``offsets`` (the contiguous execution horizon 1..s) via that offset's own head, stacked in
    offset order → ``[B, s, A_DIM]``. One forward per s frames — the deploy-time saving of chunked execution.
    Each offset uses an independent head. The offsets do not condition on each other. The sampling
    options match ``decode``."""
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
) -> Float[Tensor, "B s d_action"]:
    """The head + sampling half of a chunked decode, from a hidden state the caller already has.

    Split out because the incremental decoder produces that hidden state one frame at a time, in its
    own forward, and must not run the backbone a second time to sample from it."""
    actions: list[Tensor] = []
    for o in offsets:
        actions.append(
            _sample_action(
                model,
                model.heads[model.head_offsets.index(o)],
                h,
                group_temps=group_temps,
                btn_support_min=btn_support_min,
                min_p=min_p,
                click_trigger_fix=click_trigger_fix,
                argmax=argmax,
                gen=gen,
            )
        )
    return torch.stack(actions, dim=1)  # [B, s, A_DIM]


def _exec_horizon_offsets(head_offsets: tuple[int, ...], s: int) -> tuple[int, ...]:
    """The contiguous execution-horizon offsets ``(1, ..., s)`` decoded per chunk. Fails loud if ``s < 1`` or
    ``head_offsets`` is missing any of them (e.g. offsets (1,5,9,13) with s=2 has no offset-2 head)."""
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
    """Fail before W&B, loader construction, or Dolphin startup on invalid experiment geometry."""
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
        "val_n_batches": cfg.val_n_batches,
        "gradient_diagnostic_batch_size": cfg.gradient_diagnostic_batch_size,
        "diagnostic_rare_button_count": cfg.diagnostic_rare_button_count,
        "eval_n_matchups": cfg.eval_n_matchups,
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
    _validate_incremental_decode(cfg)
    if cfg.final_h2h_reference_run is not None:
        if not cfg.final_h2h_reference_run:
            raise ValueError("final_h2h_reference_run must be a run name or None, not an empty string")
        if cfg.final_h2h_n_configs < 1:
            raise ValueError(f"final_h2h_n_configs must be >= 1, got {cfg.final_h2h_n_configs}")
        if cfg.final_h2h_self_label == cfg.final_h2h_reference_label:
            raise ValueError(f"h2h labels must differ, got {cfg.final_h2h_self_label!r} twice")
        exp = cfg.final_h2h_reference_experiment
        if exp.endswith(".py") and not Path(exp).exists():
            raise ValueError(f"final_h2h_reference_experiment does not exist: {exp}")
    if cfg.d_model % cfg.n_heads != 0:
        raise ValueError(f"d_model={cfg.d_model} must be divisible by n_heads={cfg.n_heads}")
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
        "eval_parallel_per_cpu": cfg.eval_parallel_per_cpu,
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
    for name in ("seed", "eval_seed"):
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


def _validate_incremental_decode(cfg: TrainConfig) -> None:
    if cfg.eval_incremental_kv and cfg.attn_window == 0:
        raise ValueError(
            "eval_incremental_kv needs attn_window > 0: at full attention the rolling KV cache drops "
            "history the full forward keeps, so the decode is silently wrong past L_ctx frames. A "
            "full-context arm must say so: pass --cfg.no-eval-incremental-kv"
        )
    if cfg.eval_incremental_kv and cfg.attn_window > 0:
        receptive_field = 1 + cfg.n_layers * (cfg.attn_window - 1)
        # Strict inequality is intentional. The full rolling-window builder masks finite
        # differences at its oldest retained row because that row's predecessor was truncated;
        # the streaming token originally saw that predecessor. If the receptive field lands
        # exactly on the left edge, that one boundary-feature difference can still reach the
        # newest output. One extra raw row puts the truncation boundary out of reach.
        if receptive_field >= cfg.L_ctx:
            raise ValueError(
                "eval_incremental_kv receptive field is not equivalent to the training rolling window: "
                f"1 + n_layers * (attn_window - 1) = {receptive_field} reaches or exceeds L_ctx={cfg.L_ctx}. "
                "Increase L_ctx, reduce attn_window/layers, or pass --cfg.no-eval-incremental-kv."
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

    # Per-slot incremental state: the rolling KV cache and the hidden state of that slot's newest
    # frame. Slots are kept apart because instant-restart boundaries land on different frames.
    kv_cache: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    hidden: dict[int, torch.Tensor] = {}

    @torch.no_grad()
    def predict_chunk(ctx: Context, committed: np.ndarray | None) -> np.ndarray:
        assert committed is None, "receding-horizon policy does not condition on a committed prefix"
        chunk = (
            decode(
                model,
                ctx,
                temp=settings.temp,
                temps=settings.temps,
                btn_support_min=settings.btn_support_min,
                min_p=settings.min_p,
                click_trigger_fix=settings.click_trigger_fix,
                gen=gen,
            )
            if s == 1
            else decode_chunk(
                model,
                ctx,
                offsets,
                temp=settings.temp,
                temps=settings.temps,
                btn_support_min=settings.btn_support_min,
                min_p=settings.min_p,
                click_trigger_fix=settings.click_trigger_fix,
                gen=gen,
            )
        )
        return chunk.cpu().numpy()

    n_layers = len(model.trunk.blocks)

    @torch.no_grad()
    def encode_frame(ctx: Context) -> None:
        """Take this frame into every live slot's KV cache and keep the hidden state it produces.

        Called EVERY frame, so the cache never misses one while a chunk is being executed. Slots are
        batched by cache LENGTH, not by absolute frame: the cache is a rolling window and the trunk
        re-applies RoPE over its own positions, so two slots holding the same number of frames share
        one forward. Every slot saturates at ``attn_window``, so a wave settles into one forward per
        frame however far apart its matches drift."""
        if ctx.slot_ids is None or ctx.reset is None:
            raise ValueError("incremental closed-loop decode requires slot_ids and reset metadata")
        ids = [int(sid) for sid in ctx.slot_ids.tolist()]
        for sid, reset in zip(ids, ctx.reset.tolist(), strict=True):
            if reset:
                kv_cache.pop(sid, None)
                hidden.pop(sid, None)
        groups: dict[int, list[int]] = {}
        for row, sid in enumerate(ids):
            cached = kv_cache.get(sid)
            groups.setdefault(0 if cached is None else cached[0][0].size(2), []).append(row)
        for rows in groups.values():
            index = torch.tensor(rows, device=model_device, dtype=torch.long)
            features = {name: value.index_select(0, index) for name, value in ctx.features.items()}
            caches = [kv_cache.get(ids[row]) for row in rows]
            past: list[tuple[torch.Tensor, torch.Tensor] | None] = (
                [None] * n_layers
                if caches[0] is None
                else [
                    (
                        torch.cat([c[layer][0] for c in caches if c is not None], 0),
                        torch.cat([c[layer][1] for c in caches if c is not None], 0),
                    )
                    for layer in range(n_layers)
                ]
            )
            h, new = model.forward_incremental(features, past)
            for j, row in enumerate(rows):
                kv_cache[ids[row]] = [(k[j : j + 1], v[j : j + 1]) for k, v in new]
                hidden[ids[row]] = h[j]

    @torch.no_grad()
    def predict_incremental(ctx: Context, committed: np.ndarray | None) -> np.ndarray:
        """Sample a chunk from the hidden states ``encode_frame`` already computed — heads only, no
        second backbone forward. The heads 1..s all read the same state, so an execution horizon
        above 1 costs one extra head per frame it covers."""
        assert committed is None, "receding-horizon policy does not condition on a committed prefix"
        if ctx.slot_ids is None:
            raise ValueError("incremental closed-loop decode requires slot_ids metadata")
        h = torch.stack([hidden[int(sid)] for sid in ctx.slot_ids.tolist()])
        chunk = chunk_from_hidden(
            model,
            h,
            offsets,
            group_temps=settings.temps or (settings.temp,) * N_GROUPS,
            btn_support_min=settings.btn_support_min,
            min_p=settings.min_p,
            click_trigger_fix=settings.click_trigger_fix,
            gen=gen,
        )
        return chunk.cpu().numpy()

    incremental = cfg.eval_incremental_kv
    return RecedingHorizon(
        predict_chunk=predict_chunk,
        predict_incremental=predict_incremental if incremental else None,
        encode_frame=encode_frame if incremental else None,
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
    """Linear warmup → cosine to a small floor. The returned multiplier scales every param group's
    base lr uniformly, so the Muon and AdamW groups share one schedule shape."""
    floor = 0.01

    def fn(step: int) -> float:
        if step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)
        progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    return fn


def make_optimizer(model: GPT, cfg: TrainConfig) -> SingleDeviceMuonWithAuxAdam:
    """Muon for the transformer blocks' hidden weight matrices (attn + MLP); AdamW for everything
    else — input projection, output head, embeddings, biases — split by weight-decay eligibility.
    Exactly two LRs (``cfg.muon_lr`` / ``cfg.adam_lr``); the partition asserts full coverage so no
    parameter can silently escape an optimizer. The Muon sweep names the trunk's BLOCKS, not the
    trunk, so a parameter added beside the block stack later lands in AdamW and not silently here."""
    muon_params = [p for p in model.trunk.blocks.parameters() if p.ndim >= 2]
    muon_ids = {id(p) for p in muon_params}
    embed_ids = {id(p) for m in (model.cat_embeds, model.char_emb, model.stage_emb) for p in m.parameters()}
    # Optionally exclude output-head weights from weight decay.
    head_ids = set() if cfg.head_weight_decay else {id(p) for p in model.heads.parameters()}

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


def _slice_batch(batch: TrainBatch, n: int) -> TrainBatch:
    return TrainBatch(
        context=Context(
            features={name: value[:n] for name, value in batch.context.features.items()},
            ctx_pad=batch.context.ctx_pad[:n],
        ),
        target=batch.target[:n],
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


def _gradient_diagnostics_eval(model: GPT, batch: TrainBatch, cfg: TrainConfig) -> dict[str, float]:
    diagnostic_batch = _slice_batch(batch, min(cfg.gradient_diagnostic_batch_size, batch.context.batch))
    parts = action_loss(model, diagnostic_batch)
    losses = {
        offset: _offset_objective(parts.nll, parts.transition, offset, cfg.transition_loss_weight)
        for offset in model.head_offsets
    }
    # Exclude output heads. This measures what each horizon asks of the shared trunk.
    trunk = tuple(parameter for name, parameter in model.named_parameters() if not name.startswith("heads."))
    gradients: dict[int, tuple[Tensor, ...]] = {}
    for i, offset in enumerate(model.head_offsets):
        gradients[offset] = tuple(
            gradient.detach()
            for gradient in torch.autograd.grad(
                losses[offset],
                trunk,
                retain_graph=i + 1 < len(model.head_offsets),
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
            for pi, p in enumerate(trunk)
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
    # Change F1 needs the full frame masks because it allows one frame of timing error.
    btn_pred_change: list[Tensor] = []
    btn_true_change: list[Tensor] = []
    counts_available = bool((model.button_combo_counts >= 0).all())
    rare_mask = model.button_combo_counts < cfg.diagnostic_rare_button_count
    unseen_mask = model.button_combo_counts == 0
    combo_bits = scoring.combo_to_buttons(torch.arange(scoring.N_BUTTON_COMBOS, device=model.main_centers.device))
    device = next(model.parameters()).device
    for cached in val_cache:
        batch = cached.to(device)
        ctx = batch.context
        h, h_ablated = _hidden_pair(model, ctx)
        targets, valid = _multi_offset_targets(ctx, batch.target[:, : max(model.head_offsets)], model.head_offsets)
        flat_valid = valid.reshape(-1)
        n_valid = int(flat_valid.sum())
        cur_idx = _quantize(model, stack_actions(ctx.features))  # [B, L_ctx, n_groups] current frames
        comps: dict[tuple[int, str], Tensor] = {}
        ablated: dict[str, Tensor] = {}
        for hi, o in enumerate(model.head_offsets):
            tgt_idx = _quantize(model, targets[o])
            logits = {name: lg.float() for name, lg in model.heads[hi].logits(h).items()}
            comps.update({(o, name): c for name, c in group_nll(logits, tgt_idx, valid).items()})
            if o != 1:
                continue
            # The deployed head drives the button / transition / ablation stats.
            true_change = scoring.transition_mask(torch.cat([cur_idx, tgt_idx[:, -1:]], dim=1))  # [B,L_ctx,n_grp]
            ablated_logits = {
                name: lg.float() for name, lg in model.heads[model.primary_head_idx].logits(h_ablated).items()
            }
            kl_bits = _group_kl_bits(logits, ablated_logits).reshape(-1)[flat_valid]  # KL(full ‖ ablated), bits
            acc.add("ablate_hist_kl", kl_bits.mean().item(), n_valid)
            ablated = group_nll(ablated_logits, tgt_idx, valid)
            for g, name in enumerate(_GROUP_NAMES):
                trans = true_change[..., g].reshape(-1)[flat_valid]
                predicted = logits[name].argmax(-1).reshape(-1)[flat_valid]
                target_group = tgt_idx[..., g].reshape(-1)[flat_valid]
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
            pred_change = (btn_logits.argmax(-1) != cur_idx[..., _BUTTONS_G]) & valid
            btn_pred_change.append(pred_change)
            btn_true_change.append(true_change[..., _BUTTONS_G] & valid)
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
        acc.update({f"nll_off{o}": _offset_total_bits(comps, o) for o in model.head_offsets}, n_valid)
        ablate_total = 0.0
        for name in _GROUP_NAMES:
            d = (ablated[name].mean() - comps[(1, name)].mean()).item() / _LN2  # positive ⇒ history helps
            acc.add(f"ablate_hist_dnll_{name}", d, n_valid)
            ablate_total += d
        acc.add("ablate_hist_dnll", ablate_total, n_valid)

    out = acc.means()
    out["btn_counts_available"] = float(counts_available)
    if counts_available:
        out["btn_rare_count_threshold"] = float(cfg.diagnostic_rare_button_count)
    out["changeF1_buttons"] = scoring.change_event_prf(torch.cat(btn_pred_change), torch.cat(btn_true_change))[2]
    return out


@dataclass(frozen=True, slots=True)
class EvalProtocol:
    n_matchups: int
    max_parallel: int
    max_frames: int
    seed: int
    exec_horizon: int
    decode_temp: float
    decode_temps: tuple[float, float, float, float] | None
    decode_btn_support_min: int
    decode_min_p: float
    decode_click_trigger_fix: bool
    model_dtype: str
    eval_incremental_kv: bool


def _eval_protocol(
    cfg: TrainConfig,
    *,
    settings: DecodeSettings,
    exec_horizon: int,
    default_n_matchups: int,
    model_dtype: str,
    n_matchups: int | None = None,
    max_frames: int | None = None,
    seed: int | None = None,
) -> EvalProtocol:
    n = default_n_matchups if n_matchups is None else n_matchups
    frames = cfg.eval_max_frames if max_frames is None else max_frames
    resolved_seed = cfg.eval_seed if seed is None else seed
    if n <= 0:
        raise ValueError(f"n_matchups must be > 0, got {n}")
    if frames <= 0:
        raise ValueError(f"max_frames must be > 0, got {frames}")
    return EvalProtocol(
        n_matchups=n,
        max_parallel=_eval_max_parallel(cfg, n),
        max_frames=frames,
        seed=resolved_seed,
        exec_horizon=exec_horizon,
        decode_temp=settings.temp,
        decode_temps=settings.temps,
        decode_btn_support_min=settings.btn_support_min,
        decode_min_p=settings.min_p,
        decode_click_trigger_fix=settings.click_trigger_fix,
        model_dtype=model_dtype,
        eval_incremental_kv=cfg.eval_incremental_kv,
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


def _run_eval_sweep(
    policy_factory: Callable[[], RecedingHorizon],
    *,
    protocol: EvalProtocol,
    replay_dir: Path | None,
    rows_path: Path | None,
) -> dict[str, float]:
    results, rows = sweep_vs_cpu_prior_with_rows(
        policy_factory,
        session_cfg=default_session_cfg(replay_dir, instant_match_restart=True),
        n_matchups=protocol.n_matchups,
        max_parallel=protocol.max_parallel,
        max_frames=protocol.max_frames,
    )
    if rows_path is not None:
        _write_match_rows(rows_path, rows, protocol)
    return vs_cpu_metrics(results, seed=protocol.seed)


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
        )

    with _evaluation_mode(model):
        return _run_eval_sweep(
            policy_factory,
            protocol=protocol,
            replay_dir=replay_dir,
            rows_path=rows_path,
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
    val_n_batches: int,
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
        return list(itertools.islice(val_loader, val_n_batches)), time.monotonic()

    future = pool.submit(read_validation)
    return train_iterator, pool, future, val_started_at


def _fetch_h2h_reference(cfg: TrainConfig, run_dir: Path) -> Path:
    """Download the reference final.pt at train START, so a bad reference fails in
    minute one, not after the full training run."""
    assert cfg.final_h2h_reference_run is not None
    ref_ckpt = download_latest(cfg.final_h2h_reference_run, run_dir / "h2h_reference", name="final.pt")
    if ref_ckpt is None:
        raise RuntimeError(f"no final.pt on R2 for reference run {cfg.final_h2h_reference_run!r}")
    return ref_ckpt


def _final_h2h(
    cfg: TrainConfig,
    model: GPT,
    stats: dict[str, FeatureStats],
    run_dir: Path,
    uploader: BackgroundUploader,
    ref_ckpt: Path,
) -> None:
    """Mirrored h2h vs the reference run, in-process after training.

    Records and replays land in ``run_dir/h2h_final``. Evidence uploads after EACH
    orientation and again in the ``finally`` block, so a kill mid-sweep costs at most
    one orientation and a tripwire raise costs nothing."""
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
    if ref_protocol["model_dtype"] != self_dtype:
        raise RuntimeError(
            f"h2h decode dtype mismatch: {cfg.final_h2h_self_label} uses {self_dtype}, "
            f"but {cfg.final_h2h_reference_label} uses {ref_protocol['model_dtype']}"
        )
    self_protocol = {
        "name": cfg.final_h2h_self_label,
        "experiment": str(Path(__file__)),
        "checkpoint": str(run_dir / "final.pt"),
        "step": cfg.max_steps,
        "L_ctx": cfg.L_ctx,
        "model_dtype": self_dtype,
        "eval_incremental_kv": cfg.eval_incremental_kv,
        "decode_settings": asdict(_decode_settings(model, cfg)),
        "exec_horizon": cfg.exec_horizon,
        "head_offsets": list(cfg.head_offsets),
    }

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
    n_params = sum(p.numel() for p in model.parameters())
    if wandb.run is not None:
        wandb.run.summary["model/num_params"] = n_params
    print(
        f"[model] {_model_tag(cfg)}  num_params={n_params / 1e6:.2f}M  "
        f"batch={cfg.batch_size} (micro {_micro_batch(cfg)} x accum {cfg.grad_accum_steps})",
        flush=True,
    )
    loader_kwargs = _loader_kwargs(cfg, stats)
    if cfg.compact_data:
        train_loader = make_replay_reservoir_loader(
            split="train",
            num_workers=cfg.num_workers,
            prefetch_factor=cfg.prefetch_factor,
            predownload=cfg.predownload,
            windows_per_replay=cfg.windows_per_replay,
            reservoir_capacity=cfg.reservoir_capacity,
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
    it, val_pool, val_future, val_started_at = _start_data_loading(train_loader, val_loader, cfg.val_n_batches)
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
        print(f"[eval] queued {n} .slp + matchup rows for R2 ({step_tag})", flush=True)
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
        if rc is None:
            if not wait and (time.monotonic() - pending_eval["t0"]) <= cfg.eval_timeout_seconds:
                return  # still running, within budget — re-check next iteration
            proc.kill()
            proc.wait()
            print(
                f"[eval] step {pending_eval['step']} timed out (>{cfg.eval_timeout_seconds:.0f}s); "
                f"killed. see {pending_eval['log']}",
                flush=True,
            )
        else:
            step, result = pending_eval["step"], pending_eval["result"]
            if rc == 0 and result.is_file():
                data = json.loads(result.read_text())
                _log_eval(data["step"], data["metrics"])
                n = uploader.upload_tree(pending_eval["replay"], base=ckpt_dir, pattern="*.slp")
                rows_path = pending_eval["replay"] / "match_rows.json"
                if rows_path.is_file():
                    uploader.upload(rows_path, key=str(rows_path.relative_to(ckpt_dir)))
                print(f"[eval] queued {n} .slp + matchup rows for R2 (step {step})", flush=True)
            else:
                print(f"[eval] worker for step {step} failed (rc={rc}); see {pending_eval['log']}", flush=True)
        pending_eval["log_f"].close()
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
            "data/train_batches_seen": train_batches_seen,
        }
        if replay_ids:
            log["data/distinct_replays"] = len(replay_ids)
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
    """Rebuild a ``TrainConfig`` from a checkpoint's saved cfg dict, tolerating
    schema drift in *eval/host* knobs across code versions: keys no longer on
    ``TrainConfig`` (e.g. the old ``eval_max_parallel``/``eval_replicas``, replaced
    by ``eval_parallel_per_cpu``) are dropped and new fields take their defaults, so
    past checkpoints still load. Model-identity fields (``d_model``, ``head_offsets``,
    …) are unaffected — they're always present and reconstruct exactly."""
    known = {f.name for f in fields(TrainConfig)}
    dropped = sorted(set(saved) - known)
    if dropped:
        print(f"[ckpt] dropping {len(dropped)} stale cfg key(s) not on current TrainConfig: {dropped}", flush=True)
    values = {k: v for k, v in saved.items() if k in known}
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
    eval_incremental_kv: bool | None = None,
    eval_exec_horizon: int | None = None,
    decode_temp: float | None = None,
    decode_temps: tuple[float, float, float, float] | None = None,
    decode_btn_support_min: int | None = None,
    decode_min_p: float | None = None,
    decode_click_trigger_fix: bool | None = None,
    eval_n_matchups: int | None = None,
    eval_max_frames: int | None = None,
    eval_seed: int | None = None,
    wandb_run_id: str | None = None,
    wandb_project: str = "hal",
    wandb_entity: str | None = None,
    wandb_label: str | None = None,
) -> dict[str, float]:
    """Load a checkpoint and run the closed-loop evaluation."""
    model, cfg, stats, state = _load_ckpt(ckpt_path)
    if eval_incremental_kv is not None:
        cfg = replace(cfg, eval_incremental_kv=eval_incremental_kv)
        _validate_incremental_decode(cfg)
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
        max_frames=eval_max_frames,
        seed=eval_seed,
    )
    _exec_horizon_offsets(model.head_offsets, exec_horizon)
    print(
        f"[eval] loaded {ckpt_path}  step={state['step']}  device={DEVICE}  exec_horizon={exec_horizon}  "
        f"incremental_kv={cfg.eval_incremental_kv}  "
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
    eval_decode: Literal["checkpoint", "recompute", "kv"] = "checkpoint"
    eval_exec_horizon: int | None = None  # override execution horizon s for --eval (chunked decode; 1=per-frame)
    eval_temp: float | None = None  # override decode temperature for --eval
    eval_temps: tuple[float, float, float, float] | None = None  # per-group temps (buttons, main, c, triggers)
    eval_btn_support_min: int | None = None  # mask button combos with < this many train frames (0=off)
    eval_min_p: float | None = None  # min-p nucleus: keep classes with p >= min_p * p_max
    eval_click_trigger_fix: bool | None = None  # force trigger_l/r to 1.0 on a digital L/R click
    eval_n_matchups: int | None = None  # manual --eval override; default is cfg.final_eval_n_matchups (96)
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
    if args.eval is not None and args.eval_run is not None:
        raise SystemExit("pass one of --eval or --eval-run, not both")
    if args.eval is not None or args.eval_run is not None:
        eval_incremental_kv = None if args.eval_decode == "checkpoint" else args.eval_decode == "kv"
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
                eval_incremental_kv=eval_incremental_kv,
                eval_exec_horizon=args.eval_exec_horizon,
                decode_temp=args.eval_temp,
                decode_temps=args.eval_temps,
                decode_btn_support_min=args.eval_btn_support_min,
                decode_min_p=args.eval_min_p,
                decode_click_trigger_fix=args.eval_click_trigger_fix,
                eval_n_matchups=args.eval_n_matchups,
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
