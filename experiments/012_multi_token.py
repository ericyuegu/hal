"""A nanoGPT/GPT-2-style causal decoder over per-frame tokens with multi-token
(multi-frame) auxiliary output heads. One token per frame concatenates all four
players' gamestate (ego, ego-nana, opp-nana, opp — float + mask + categorical
embeddings, nana masked when absent) with the ego's own controller history and
the matchup char/stage embeddings, projected to ``d_model``.

Each frame's hidden state feeds one INDEPENDENT output head per future offset in
``cfg.head_offsets`` (default ``(1, 5, 9, 13)``): head ``o`` predicts the action
``o`` frames ahead. Every head jointly emits the concatenation of the action
groups' vocabs (buttons 256 + main-stick 65 + c-stick 9 + triggers 25 = 355
logits), each group's slice softmax-sampled independently (groups conditionally
independent given context; the autoregressive-groups variant lives in
010_ar_groups.py). The far-horizon heads are an AUXILIARY training signal à la
Meta/DeepSeek multi-token prediction — they shape the trunk but are never used at
inference: closed-loop play decodes ONLY the offset-1 head (``head_offsets`` must
contain ``1``), so the deployed policy is byte-for-byte the 011 next-token policy.
Chunked execution (``exec_horizon`` s>1, deploy-time only) instead runs the contiguous
heads 1..s from a single backbone forward and executes all s actions before replanning
— s× cheaper closed-loop play; s=1 is the default and reproduces the per-frame policy.

Run:
    uv run experiments/012_multi_token.py
    uv run experiments/012_multi_token.py --cfg.head-offsets 1 5 9 13
    uv run experiments/012_multi_token.py --cfg.history-dropout-p 0.25
    uv run experiments/012_multi_token.py --cfg.transition-loss-weight 5.0
    uv run experiments/012_multi_token.py --eval <ckpt> --eval-temp 0.7
    uv run experiments/012_multi_token.py --eval <ckpt> --eval-exec-horizon 2
"""

# %%
import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import contextlib
import itertools
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from dataclasses import replace
from pathlib import Path

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
from hal.eval.cross_stage import sweep_vs_cpu_prior
from hal.eval.cross_stage import vs_cpu_metrics
from hal.eval.harness import default_session_cfg
from hal.training import scoring
from hal.training.checkpoints import BackgroundUploader
from hal.training.checkpoints import load_for_resume
from hal.training.checkpoints import save_checkpoint
from hal.training.closed_loop import RecedingHorizon
from hal.training.dataloader import VAL_L_CHUNK
from hal.training.dataloader import make_loader
from hal.training.ego_stats import load_consolidated_stats
from hal.training.features import A_DIM
from hal.training.features import ACTION_CHANNELS
from hal.training.features import CAT_FEATURES
from hal.training.features import FLOAT_FEATURES
from hal.training.features import Context
from hal.training.features import TrainBatch
from hal.training.features import stack_actions
from hal.training.muon import SingleDeviceMuonWithAuxAdam
from hal.training.runs import make_run_name
from hal.training.runs import profile
from hal.training.runs import setup_run_dir

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_LN2 = math.log(2.0)

# Action-vector channel split (A_DIM=14): [0:6] sticks+triggers (continuous), [6:14] buttons {0,1}.
_N_CONT = 6
_N_BUTTONS = A_DIM - _N_CONT

# Per-frame input: all four players' gamestate concatenated in the feature dim.
_PLAYER_PREFIXES: tuple[str, ...] = ("ego", "ego_nana", "opp_nana", "opp")

# Output groups (fixed order) + their discrete vocab sizes from the scoring discretizers.
_GROUP_NAMES: tuple[str, ...] = ("buttons", "main_stick", "c_stick", "triggers")
_GROUP_VOCABS: tuple[int, ...] = (
    scoring.N_BUTTON_COMBOS,  # 256
    scoring.STICK_CLUSTER_CENTERS_MAIN.shape[0],  # 65
    scoring.STICK_CLUSTER_CENTERS_C.shape[0],  # 9
    scoring.TRIGGER_CENTERS.shape[0] ** 2,  # 25 (joint L*5 + R)
)
N_GROUPS = len(_GROUP_NAMES)
_BUTTONS_G, _MAIN_G, _C_G, _TRIG_G = range(N_GROUPS)
_GROUP_OFFSETS: tuple[int, ...] = tuple(itertools.accumulate((0,) + _GROUP_VOCABS))[:N_GROUPS]  # (0,256,321,330)
A_VOCAB = sum(_GROUP_VOCABS)  # 355

# Decode-time button-combo support mask: train frequencies as a tensor (built once), plus a per
# (min_count, device) cache of the "dead" mask so closed-loop decode never rebuilds it per frame.
_BTN_COMBO_COUNTS: Tensor = torch.tensor(scoring.BTN_COMBO_COUNTS, dtype=torch.long)
_BTN_SUPPORT_DEAD_CACHE: dict[tuple[int, torch.device], Tensor] = {}

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
    # Multi-token (multi-frame) auxiliary output heads: one independent head per future-frame offset;
    # head o predicts the action o frames ahead. MUST contain 1 — closed-loop decodes only the offset-1
    # head, so the far-horizon heads are a training-only signal. Spread-out offsets beat contiguous.
    head_offsets: tuple[int, ...] = (1, 5, 9, 13)
    aux_loss_weight: float = 1.0  # weight on the non-primary (offset != 1) heads' NLL in the objective
    # Per-sample ego-history input dropout (train only): with probability p, zero a sample's ego
    # controller-history slice of the context token so the trunk cannot lean on copying its own recent
    # inputs. Lives purely in the model's input assembly (targets untouched). 0.0 = current 012 behavior.
    history_dropout_p: float = 0.0
    # Upweight transition-target positions (the predicted group id differs from the frame before it) in the
    # BACKPROP objective only; every logged/val NLL stays unweighted so arms compare. Per (offset, group) the
    # plain mean becomes sum(w·nll)/sum(w) with w=λ on transitions else 1. λ=1.0 reduces exactly to the mean.
    transition_loss_weight: float = 1.0
    # Matchup conditioning (schema v4). char/stage embeddings are indexed by the RAW libmelee id
    # (characters 0-26 dense; stages sparse in 0-26), so the vocab must exceed the max id, not the
    # number of included categories; out-of-range ids clamp to the last row.
    char_vocab: int = 32
    char_dim: int = 12
    stage_vocab: int = 32
    stage_dim: int = 4
    # closed-loop sampling temperature. Greedy argmax collapses the policy to a do-nothing fixed
    # point in closed loop, so deployed play always samples; argmax stays for the recon metric.
    decode_temp: float = 1.0
    # Decode-time hygiene (all default to current behavior). decode_temps overrides the single decode_temp
    # per group in _GROUP_NAMES order (buttons, main_stick, c_stick, triggers); None -> decode_temp for all.
    # decode_btn_support_min >= 1 masks button combos with fewer than that many train frames
    # (scoring.BTN_COMBO_COUNTS) to -inf before softmax/argmax. decode_min_p > 0 keeps only classes with
    # p >= decode_min_p * p_max per group, then renormalizes. decode_click_trigger_fix forces trigger_l/r to
    # 1.0 wherever the sampled combo sets the digital L/R bit (the only train-supported click joint).
    decode_temps: tuple[float, float, float, float] | None = None
    decode_btn_support_min: int = 0
    decode_min_p: float = 0.0
    decode_click_trigger_fix: bool = False
    # Chunked execution (deploy-time only): replan every s frames, executing the contiguous heads 1..s from
    # ONE backbone forward (head_offsets must contain 1..s). s=1 = per-frame decode (current 012). Training
    # is unaffected — closed-loop deployment only; eval can override via --eval-exec-horizon.
    exec_horizon: int = 1
    seed: int = 0
    L_ctx: int = 256
    # optimization
    batch_size: int = 1024
    grad_accum_steps: int = 1
    # Two LRs: Muon for the blocks' hidden matrices, AdamW for the input proj / head / embeddings / biases.
    muon_lr: float = 0.02
    adam_lr: float = 8.5e-4
    weight_decay: float = 0.01
    # When False, keep the output-head (heads) weights out of weight decay (route them to the no-decay
    # AdamW group). Default True leaves the heads in the decayed group — exactly the current behavior.
    head_weight_decay: bool = True
    warmup_steps: int = 500
    max_steps: int = 2**15
    amp_dtype: str = "bfloat16"  # "bfloat16" | "float32"
    allow_tf32: bool = True
    # eval cadence
    val_every: int = 1024
    val_n_batches: int = 16
    eval_every: int = 2048
    eval_max_frames: int = 7200
    # Closed-loop eval parallelism scales with the box: max_parallel = round(this * cpu_count).
    # Each parallel slot is one Dolphin boot that (via instant-restart) plays many prior-sampled
    # matches back-to-back. Profile {0.5, 1, 2, 4} for the best eval throughput vs trainer impact.
    eval_parallel_per_cpu: float = 1.0
    # Closed-loop eval always runs in a background subprocess (training keeps using the GPU); the
    # trainer drains its result between steps. If a prior eval is still running at the next boundary
    # the trainer waits for it, bounded by eval_timeout_seconds (then kills the worker).
    eval_timeout_seconds: float = 900.0
    # checkpointing
    ckpt_every: int = 2048
    # data (v4 MDS carries the stage + p{1,2}_character + nana columns)
    data_root: str = "data/processed/ranked-anonymized-1/mds"
    cache_limit_gb: int = 440
    shuffle_block_size: int = 2000
    # Each replay deserialized off disk yields this many non-overlapping windows,
    # amortizing the whole-replay read (the disk bottleneck) over K samples. Train
    # only; val stays 1/replay so its loss stays comparable across runs. Small (4)
    # keeps same-replay windows per batch low.
    windows_per_replay: int = 4
    val_split: str = "val"
    num_workers: int = 16
    prefetch_factor: int = 4


def _model_tag(cfg: TrainConfig) -> str:
    offs = ".".join(str(o) for o in cfg.head_offsets)
    return f"gpt-d{cfg.d_model}-L{cfg.n_layers}-h{cfg.n_heads}-Lc{cfg.L_ctx}-o{offs}"


def _eval_max_parallel(cfg: TrainConfig) -> int:
    """Concurrent Dolphin boots per eval wave, scaled to the host's CPU count."""
    return max(1, round(cfg.eval_parallel_per_cpu * (os.cpu_count() or 1)))


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
# --- GPT backbone (nanoGPT-style: rotary, RMSNorm, causal SDPA) ---------------
class Rotary(nn.Module):
    inv_freq: Tensor
    seq_len_cached: int | None
    cos_cached: Tensor | None
    sin_cached: Tensor | None

    def __init__(self, dim: int, base: int = 10000) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[Tensor, "B L n_heads head_dim"]
    ) -> tuple[
        Float[Tensor, "1 L 1 half_dim"],
        Float[Tensor, "1 L 1 half_dim"],
    ]:
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        assert self.cos_cached is not None and self.sin_cached is not None
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]


@jaxtyped(typechecker=beartype)
def apply_rotary_emb(
    x: Float[Tensor, "B L n_heads head_dim"],
    cos: Float[Tensor, "1 L 1 half_dim"],
    sin: Float[Tensor, "1 L 1 half_dim"],
) -> Float[Tensor, "B L n_heads head_dim"]:
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], 3)


@jaxtyped(typechecker=beartype)
def rmsnorm(x0: Float[Tensor, "... d"], eps: float = 1e-6) -> Float[Tensor, "... d"]:
    x = x0.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.type_as(x0)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.d_model = cfg.d_model
        self.head_dim = cfg.d_model // cfg.n_heads
        self.c_attn = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.c_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.rotary = Rotary(self.head_dim)

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "B L d_model"], mask: Bool[Tensor, "B 1 L L"]) -> Float[Tensor, "B L d_model"]:
        B, L, _ = x.shape
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        q = q.view(B, L, self.n_heads, self.head_dim)
        k = k.view(B, L, self.n_heads, self.head_dim)
        v = v.view(B, L, self.n_heads, self.head_dim)
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask)
        y = y.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=False)
        self.c_proj = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=False)

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "B L d_model"]) -> Float[Tensor, "B L d_model"]:
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.attn = CausalSelfAttention(cfg)
        self.mlp = MLP(cfg)
        self.attn_scale = 1 / (2 * cfg.n_layers) ** 0.5

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "B L d_model"], mask: Bool[Tensor, "B 1 L L"]) -> Float[Tensor, "B L d_model"]:
        x = x + self.attn_scale * self.attn(rmsnorm(x), mask)
        x = x + self.mlp(rmsnorm(x))
        return x


# %%
class GPT(nn.Module):
    """Causal GPT over per-frame tokens with multi-token auxiliary heads. ``hidden[i]`` (causal) feeds
    one independent head per offset in ``cfg.head_offsets``; head ``o`` predicts the action ``o`` frames
    ahead via a joint ``A_VOCAB`` logit vector (the concatenation of the four group vocabs). Closed-loop
    decode uses only the offset-1 head (``primary_head_idx``); the rest are an auxiliary training signal."""

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
        self.L_ctx = cfg.L_ctx

        # Gamestate categoricals: one table per feature name, shared across the four players.
        self.cat_embeds = nn.ModuleDict(
            {name: nn.Embedding(vocab, dim) for name, (vocab, dim) in CAT_FEATURES.items()}
        )
        self.char_emb = nn.Embedding(cfg.char_vocab, cfg.char_dim)
        self.stage_emb = nn.Embedding(cfg.stage_vocab, cfg.stage_dim)
        per_player = len(FLOAT_FEATURES) * 2 + sum(dim for _, dim in CAT_FEATURES.values())  # float+mask+cat
        d_in = len(_PLAYER_PREFIXES) * per_player + A_DIM + 2 * cfg.char_dim + cfg.stage_dim

        self.ctx_proj = nn.Linear(d_in, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        # One independent joint head per future-frame offset (order matches self.head_offsets).
        self.heads = nn.ModuleList([nn.Linear(cfg.d_model, A_VOCAB) for _ in offs])

        # Stick/trigger center grids (registered so they move with .to() and serialize).
        self.register_buffer("main_centers", scoring.STICK_CLUSTER_CENTERS_MAIN.clone())
        self.register_buffer("c_centers", scoring.STICK_CLUSTER_CENTERS_C.clone())
        self.register_buffer("trig_centers", scoring.TRIGGER_CENTERS.clone())

    def _per_player_features(self, features: dict[str, Tensor], prefix: str) -> Tensor:
        ref = features[f"{prefix}_position_x"]
        B, L = ref.shape
        device = ref.device
        parts: list[Tensor] = [features[f"{prefix}_{feat}"][..., None] for feat in FLOAT_FEATURES]
        for feat in FLOAT_FEATURES:
            mk = f"{prefix}_{feat}_mask"
            parts.append(features[mk][..., None] if mk in features else torch.zeros(B, L, 1, device=device))
        for name, (vocab, _) in CAT_FEATURES.items():
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

    def _attn_mask(self, ctx_pad: Int[Tensor, " B"], L: int, device: torch.device) -> Bool[Tensor, "B 1 L L"]:
        """Causal mask that also hides each sample's left-padded cold-start prefix (key < ctx_pad).
        A padded query keeps its diagonal so its row is never fully masked (SDPA would NaN)."""
        idx = torch.arange(L, device=device)
        causal = idx[:, None] >= idx[None, :]
        key_real = idx[None, :] >= ctx_pad[:, None]
        diag = torch.eye(L, dtype=torch.bool, device=device)
        return (causal[None] & (key_real[:, None, :] | diag[None]))[:, None]

    def forward(self, features: dict[str, Tensor], ctx_pad: Int[Tensor, " B"]) -> Float[Tensor, "B L_ctx d_model"]:
        """Backbone hidden (one rmsnorm'd vector per frame); callers apply the per-offset heads."""
        x = self._context_tokens(features)
        mask = self._attn_mask(ctx_pad, x.size(1), x.device)
        for block in self.blocks:
            x = block(x, mask)
        return rmsnorm(x)


# %%
def _quantize(model: GPT, actions: Tensor) -> Tensor:
    return quantize_groups(model.main_centers, model.c_centers, model.trig_centers, actions)


def _dequantize(model: GPT, idx: Tensor) -> Tensor:
    return dequantize_groups(model.main_centers, model.c_centers, model.trig_centers, idx)


def _multi_offset_targets(ctx: Context, target: Tensor, offsets: tuple[int, ...]) -> tuple[dict[int, Tensor], Tensor]:
    """Per context position ``i`` and offset ``o``, the action ``o`` frames ahead + a shared validity
    mask. The ego controller history already lives in ``ctx.features``, so ``a_full = [history | target]``
    spans frames ``0 .. L_ctx + max(offsets) - 1`` and offset ``o``'s target is ``a_full[:, o : o + L_ctx]``
    (offset 1 recovers 011's next-frame target). A position is valid iff it is a real (non-pad) context
    frame; ``i >= ctx_pad`` then guarantees the target frame ``i + o`` is real too (``o >= 1``)."""
    a_full = torch.cat([stack_actions(ctx.features), target], dim=1)  # [B, L_ctx + max_off, A_DIM]
    L_ctx = a_full.size(1) - target.size(1)
    pos = torch.arange(L_ctx, device=a_full.device)
    valid = pos[None, :] >= ctx.ctx_pad[:, None]  # [B, L_ctx], shared by all offsets
    targets = {o: a_full[:, o : o + L_ctx] for o in offsets}  # each [B, L_ctx, A_DIM]
    return targets, valid


def group_nll(logits: Tensor, tgt_idx: Tensor, valid: Tensor) -> dict[str, Tensor]:
    """Per-group categorical NLL (nats) over the VALID positions only. Returns ``{name: [n_valid]}``
    1D tensors (same ordering across groups) so callers reduce once for exact sample weighting."""
    flat_valid = valid.reshape(-1)
    out: dict[str, Tensor] = {}
    for g, name in enumerate(_GROUP_NAMES):
        lo = _GROUP_OFFSETS[g]
        lg = logits[..., lo : lo + _GROUP_VOCABS[g]].reshape(-1, _GROUP_VOCABS[g])[flat_valid]
        out[name] = F.cross_entropy(lg, tgt_idx[..., g].reshape(-1)[flat_valid], reduction="none")
    return out


def action_loss(model: GPT, batch: TrainBatch) -> tuple[dict[tuple[int, str], Tensor], dict[tuple[int, str], Tensor]]:
    """Dense multi-token NLL + aligned transition flags. Every valid context position predicts the action at
    each head offset; one shared backbone forward, one head each. ``a_full = [history | target]`` is quantized
    ONCE ([B, L_ctx+max_off, n_groups]) and its per-frame boundary mask computed once, both sliced per offset:
    for head ``o`` position ``i``'s target frame is ``i+o`` and it is a transition iff ``q[i+o] != q[i+o-1]``.
    Returns ``(nll, trans)`` both keyed ``(offset, group_name)`` → ``[n_valid]`` (nats; bool), aligned position
    for position so the objective can upweight transitions without touching the logged NLL."""
    ctx = batch.context
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
        logits = model.heads[hi](h).float()  # [B, L_ctx, A_VOCAB]
        tgt_idx = q_full[:, o : o + L_ctx]  # target frame i+o
        bnd_o = bnd_full[:, o - 1 : o - 1 + L_ctx]  # transition at i iff q[i+o] != q[i+o-1]
        gnll = group_nll(logits, tgt_idx, valid)
        for g, name in enumerate(_GROUP_NAMES):
            nll[(o, name)] = gnll[name]
            trans[(o, name)] = bnd_o[..., g].reshape(-1)[flat_valid]
    return nll, trans


def _btn_support_dead(min_count: int, device: torch.device) -> Tensor:
    """``[N_BUTTON_COMBOS]`` bool mask, True on button combos with fewer than ``min_count`` train
    frames (``BTN_COMBO_COUNTS``) — the classes to mask to -inf before softmax/argmax. Cached per
    ``(min_count, device)`` so closed-loop decode builds it once, never per frame."""
    dead = _BTN_SUPPORT_DEAD_CACHE.get((min_count, device))
    if dead is None:
        dead = _BTN_COMBO_COUNTS.to(device) < min_count
        if bool(dead.all()):
            raise ValueError(
                f"btn_support_min={min_count} masks every button combo "
                f"(max train count is {int(_BTN_COMBO_COUNTS.max())})"
            )
        _BTN_SUPPORT_DEAD_CACHE[(min_count, device)] = dead
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
    if not 0.0 <= min_p <= 1.0:
        raise ValueError(f"decode min_p must be in [0, 1], got {min_p}")
    group_temps = (temp,) * N_GROUPS if temps is None else temps
    if not argmax and any(t <= 0 for t in group_temps):
        raise ValueError(f"decode temperatures must be > 0, got {group_temps}")
    return group_temps


def _sample_action_from_logits(
    model: GPT,
    logits: Tensor,
    *,
    group_temps: tuple[float, ...],
    btn_support_min: int,
    min_p: float,
    click_trigger_fix: bool,
    argmax: bool,
    gen: torch.Generator | None,
) -> Float[Tensor, "B d_action"]:
    """Sample one action vector from a single head's ``[B, A_VOCAB]`` joint logits, applying the decode
    hygiene in order support-mask -> per-group temperature -> min-p -> per-group sample (or ``argmax``),
    then dequantize + the click=>trigger fix. Shared by the offset-1 ``decode`` and the chunked
    ``decode_chunk`` so the per-group sampler never forks between the two deploy paths."""
    if btn_support_min >= 1:
        dead = _btn_support_dead(btn_support_min, logits.device)
        logits = logits.clone()
        logits[:, : scoring.N_BUTTON_COMBOS].masked_fill_(dead, float("-inf"))
    picks: list[Tensor] = []
    for g in range(N_GROUPS):
        lo = _GROUP_OFFSETS[g]
        lg = logits[:, lo : lo + _GROUP_VOCABS[g]]
        if argmax:
            picks.append(lg.argmax(-1))
            continue
        probs = F.softmax(lg / group_temps[g], dim=-1)
        if min_p > 0:
            probs = probs * (probs >= min_p * probs.amax(dim=-1, keepdim=True))
            probs = probs / probs.sum(dim=-1, keepdim=True)
        picks.append(torch.multinomial(probs, 1, generator=gen).squeeze(-1))
    idx = torch.stack(picks, dim=-1)  # [B, N_GROUPS]
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
    """One next-frame action per sample from the LAST context position, in raw action ranges, via the
    offset-1 (primary) head only. Each group's logit slice is sampled (per-group ``temp``-scaled softmax,
    optional min-p nucleus) or taken greedily (``argmax``, for the recon metric) independently. Decode-time
    hygiene applied in order support-mask -> per-group temperature -> min-p -> sample: ``btn_support_min``
    >= 1 masks button combos with fewer than that many train frames to -inf; ``temps`` overrides ``temp``
    per group (buttons, main_stick, c_stick, triggers); ``min_p`` > 0 keeps only classes with
    ``p >= min_p * p_max`` then renormalizes; ``click_trigger_fix`` forces trigger_l/r to 1.0 wherever the
    sampled combo sets the digital L/R bit. ``argmax`` ignores ``temps``/``min_p`` but respects the mask."""
    group_temps = _resolve_decode_args(temp, temps, btn_support_min, min_p, argmax)
    h = model(ctx.features, ctx.ctx_pad)[:, -1]  # [B, d_model]
    logits = model.heads[model.primary_head_idx](h).float()  # [B, A_VOCAB]
    a = _sample_action_from_logits(
        model,
        logits,
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
    Same per-group sampling + hygiene as ``decode``; ``offsets == (1,)`` is byte-identical to ``decode``."""
    group_temps = _resolve_decode_args(temp, temps, btn_support_min, min_p, argmax)
    h = model(ctx.features, ctx.ctx_pad)[:, -1]  # [B, d_model]
    actions: list[Tensor] = []
    for o in offsets:
        logits = model.heads[model.head_offsets.index(o)](h).float()  # [B, A_VOCAB]
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
) -> RecedingHorizon:
    """Fresh closed-loop policy for one eval wave. Replan every ``s`` frames (``exec_horizon``, defaulting to
    ``cfg.exec_horizon``): s=1 decodes the offset-1 head per frame (byte-identical to prior 012); s>1 decodes
    the contiguous heads 1..s from one backbone forward and executes all s before replanning. Each decode-
    hygiene knob falls back to its ``cfg`` field when the override is ``None`` so an eval can A/B without a
    retrain."""
    s = cfg.exec_horizon if exec_horizon is None else exec_horizon
    offsets = _exec_horizon_offsets(model.head_offsets, s)
    temp = cfg.decode_temp if decode_temp is None else decode_temp
    temps = cfg.decode_temps if decode_temps is None else decode_temps
    btn_support_min = cfg.decode_btn_support_min if decode_btn_support_min is None else decode_btn_support_min
    min_p = cfg.decode_min_p if decode_min_p is None else decode_min_p
    click_fix = cfg.decode_click_trigger_fix if decode_click_trigger_fix is None else decode_click_trigger_fix

    @torch.no_grad()
    def predict_chunk(ctx: Context, committed: np.ndarray | None) -> np.ndarray:
        assert committed is None, "receding-horizon policy does not condition on a committed prefix"
        chunk = (
            decode(
                model,
                ctx,
                temp=temp,
                temps=temps,
                btn_support_min=btn_support_min,
                min_p=min_p,
                click_trigger_fix=click_fix,
            )
            if s == 1
            else decode_chunk(
                model,
                ctx,
                offsets,
                temp=temp,
                temps=temps,
                btn_support_min=btn_support_min,
                min_p=min_p,
                click_trigger_fix=click_fix,
            )
        )
        return chunk.cpu().numpy()

    return RecedingHorizon(
        predict_chunk=predict_chunk, stats=stats, L_ctx=cfg.L_ctx, L_chunk=s, s=s, d=0, device=device
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
    parameter can silently escape an optimizer."""
    muon_params = [p for p in model.blocks.parameters() if p.ndim >= 2]
    muon_ids = {id(p) for p in muon_params}
    embed_ids = {id(p) for m in (model.cat_embeds, model.char_emb, model.stage_emb) for p in m.parameters()}
    # Optionally exclude the output-head weights from weight decay (cfg.head_weight_decay=False).
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
    """Mean per-position NLL (nats) with transition positions upweighted by ``weight``. ``weight == 1.0`` is
    exactly the plain mean (current 012); otherwise ``sum(w·nll)/sum(w)`` with ``w = weight`` on transitions
    (``is_trans``) else 1 — the rare press/release/stick-move frames dominate the objective."""
    if weight == 1.0:
        return nll.mean()
    w = torch.where(is_trans, weight, 1.0).to(nll.dtype)
    return (w * nll).sum() / w.sum()


def objective(
    nll: dict[tuple[int, str], Tensor],
    trans: dict[tuple[int, str], Tensor],
    aux_weight: float,
    transition_weight: float,
) -> Tensor:
    """Weighted-sum multi-token training objective (nats): the offset-1 (primary) head's per-group NLL at
    weight 1, every auxiliary head (offset != 1) at ``aux_weight``; within each (offset, group) the per-group
    reduction upweights transition targets by ``transition_weight`` (1.0 = plain mean, i.e. current 012)."""
    terms = [
        (1.0 if o == 1 else aux_weight) * _weighted_mean(c, trans[(o, name)], transition_weight)
        for (o, name), c in nll.items()
    ]
    return torch.stack(terms).sum()


def _offset_total_bits(comps: dict[tuple[int, str], Tensor], o: int) -> float:
    """Total bits/frame summed over the four groups for one head offset ``o``."""
    return sum(comps[(o, name)].mean() for name in _GROUP_NAMES).item() / _LN2


def _masked_mean_bits(nats: Tensor, mask: Tensor) -> float:
    """Mean of per-position NLL (nats) over the masked subset, in bits; 0.0 when the subset is empty."""
    return (nats[mask].mean().item() / _LN2) if bool(mask.any()) else 0.0


def _group_kl_bits(logits_p: Tensor, logits_q: Tensor) -> Tensor:
    """Summed-over-groups KL(p‖q) in bits per position: for each group's logit slice, p=softmax(p_slice),
    q=softmax(q_slice); ``[..., A_VOCAB]`` logits → ``[...]`` bits summed over the four group softmaxes."""
    total = torch.zeros(logits_p.shape[:-1], device=logits_p.device)
    for g in range(N_GROUPS):
        lo = _GROUP_OFFSETS[g]
        sl = slice(lo, lo + _GROUP_VOCABS[g])
        logp = F.log_softmax(logits_p[..., sl], dim=-1)
        logq = F.log_softmax(logits_q[..., sl], dim=-1)
        total = total + (logp.exp() * (logp - logq)).sum(-1)
    return total / _LN2


@torch.no_grad()
def val_metrics(model: GPT, val_cache: list[TrainBatch], cfg: TrainConfig) -> dict[str, float]:
    """Dense multi-token proper-scoring metrics over the cached val batches. Per-offset NLL (``nll_off{o}``)
    tracks how predictability decays with horizon. The offset-1 (deployed) head additionally drives button
    proper-scoring, transition-vs-hold NLL splits, ±1-tolerant change-event F1, a copycat history-ablation
    probe, and rare-combo dead mass. Per-element tensors are concatenated then reduced once (exactly
    sample-weighted)."""
    was_training = model.training
    model.eval()
    dead_combos = (~scoring.btn_combo_support(100)).to(DEVICE)  # button combos with < 100 train frames
    comps_cat: dict[tuple[int, str], list[Tensor]] = {}
    ablated_cat: dict[str, list[Tensor]] = {}
    trans_cat: dict[str, list[Tensor]] = {}
    pred_change_cat: dict[str, list[Tensor]] = {}
    true_change_cat: dict[str, list[Tensor]] = {}
    kl_bits: list[Tensor] = []
    dead_mass: list[Tensor] = []
    btn_probs: list[Tensor] = []
    btn_tgts: list[Tensor] = []
    multipress: list[Tensor] = []
    for batch in val_cache:
        ctx = batch.context
        h = model(ctx.features, ctx.ctx_pad)
        # Copycat probe: a second backbone forward with the ego's own controller history zeroed.
        ablated_features = dict(ctx.features)
        for ch in ACTION_CHANNELS:
            ablated_features[f"ego_{ch}"] = torch.zeros_like(ablated_features[f"ego_{ch}"])
        h_ablated = model(ablated_features, ctx.ctx_pad)
        targets, valid = _multi_offset_targets(ctx, batch.target[:, : max(model.head_offsets)], model.head_offsets)
        flat_valid = valid.reshape(-1)
        cur_idx = _quantize(model, stack_actions(ctx.features))  # [B, L_ctx, n_groups] current frames
        for hi, o in enumerate(model.head_offsets):
            logits = model.heads[hi](h).float()
            tgt_idx = _quantize(model, targets[o])
            for name, c in group_nll(logits, tgt_idx, valid).items():
                comps_cat.setdefault((o, name), []).append(c)
            if o == 1:  # deployed head drives button / transition / ablation / dead-mass stats
                true_change = scoring.transition_mask(torch.cat([cur_idx, tgt_idx[:, -1:]], dim=1))  # [B,L_ctx,n_grp]
                ablated_logits = model.heads[model.primary_head_idx](h_ablated).float()
                kl_bits.append(_group_kl_bits(logits, ablated_logits).reshape(-1)[flat_valid])
                for name, c in group_nll(ablated_logits, tgt_idx, valid).items():
                    ablated_cat.setdefault(name, []).append(c)
                for g, name in enumerate(_GROUP_NAMES):
                    lo = _GROUP_OFFSETS[g]
                    pred_id = logits[..., lo : lo + _GROUP_VOCABS[g]].argmax(-1)  # [B, L_ctx] argmax next-frame id
                    tc = true_change[..., g]
                    trans_cat.setdefault(name, []).append(tc.reshape(-1)[flat_valid])
                    pred_change_cat.setdefault(name, []).append((pred_id != cur_idx[..., g]) & valid)
                    true_change_cat.setdefault(name, []).append(tc & valid)
                btn_logits = logits[..., : scoring.N_BUTTON_COMBOS].reshape(-1, scoring.N_BUTTON_COMBOS)[flat_valid]
                btn_probs.append(scoring.combo_marginal_probs(btn_logits))
                dead_mass.append(F.softmax(btn_logits, dim=-1)[:, dead_combos].sum(-1))
                tgt_btn = _dequantize(model, tgt_idx)[..., _N_CONT:].reshape(-1, _N_BUTTONS)[flat_valid]
                btn_tgts.append(tgt_btn)
                multipress.append((tgt_btn > 0.5).sum(-1) >= 2)
    comps = {k: torch.cat(v) for k, v in comps_cat.items()}
    ablated = {k: torch.cat(v) for k, v in ablated_cat.items()}
    primary = nll_breakdown({name: comps[(1, name)] for name in _GROUP_NAMES})
    logloss, brier = scoring.bernoulli_scores_from_probs(torch.cat(btn_probs), torch.cat(btn_tgts))
    out = {
        "loss": primary["total"],  # offset-1 total bits/frame (deployed policy); per-group below
        **{f"nll_{name}": primary[name] for name in _GROUP_NAMES},
        "cont_discrete_bits": (
            comps[(1, "main_stick")].mean() + comps[(1, "c_stick")].mean() + comps[(1, "triggers")].mean()
        ).item()
        / _LN2,
        "btn_logloss": logloss.item(),
        "btn_brier": brier.item(),
        "btn_multipress": torch.cat(multipress).float().mean().item(),
        "dead_mass": torch.cat(dead_mass).mean().item(),  # softmax mass on <100-count button combos
        "ablate_hist_kl": torch.cat(kl_bits).mean().item(),  # KL(full ‖ history-ablated), bits
        **{f"nll_off{o}": _offset_total_bits(comps, o) for o in model.head_offsets},
    }
    ablate_total = 0.0
    for name in _GROUP_NAMES:
        trans = torch.cat(trans_cat[name])
        out[f"nll_{name}_trans"] = _masked_mean_bits(comps[(1, name)], trans)
        out[f"nll_{name}_hold"] = _masked_mean_bits(comps[(1, name)], ~trans)
        out[f"trans_rate_{name}"] = trans.float().mean().item()
        out[f"changeF1_{name}"] = scoring.change_event_prf(
            torch.cat(pred_change_cat[name]), torch.cat(true_change_cat[name])
        )[2]
        d = (ablated[name].mean() - comps[(1, name)].mean()).item() / _LN2  # positive ⇒ history helps
        out[f"ablate_hist_dnll_{name}"] = d
        ablate_total += d
    out["ablate_hist_dnll"] = ablate_total
    if was_training:
        model.train()
    return out


@torch.no_grad()
def recon_metrics(
    model: GPT, val_cache: list[TrainBatch], *, argmax: bool, temp: float = 1.0, gen: torch.Generator | None = None
) -> dict[str, float]:
    """Sample-space reconstruction proxy: decode the next action and score it vs ground truth.
    Buttons → acc + F1 @ decode; continuous → MAE. ``argmax`` is the deterministic controller proxy."""
    was_training = model.training
    model.eval()
    tp = fp = fn = btn_correct = btn_total = 0
    cont_abs_err = 0.0
    cont_count = 0
    for batch in val_cache:
        pred = decode(model, batch.context, temp=temp, argmax=argmax, gen=gen)
        # decode yields one frame ([B, 1, A_DIM]); score it against the next-frame target only. Comparing
        # against the full VAL_L_CHUNK-wide target would silently broadcast [B,1,·] vs [B,L_chunk,·].
        tgt = batch.target[:, :1]
        pb = pred[..., _N_CONT:] > 0.5
        tb = tgt[..., _N_CONT:] > 0.5
        tp += int((pb & tb).sum())
        fp += int((pb & ~tb).sum())
        fn += int((~pb & tb).sum())
        btn_correct += int((pb == tb).sum())
        btn_total += pb.numel()
        cont_abs_err += float((pred[..., :_N_CONT] - tgt[..., :_N_CONT]).abs().sum())
        cont_count += tgt[..., :_N_CONT].numel()
    if was_training:
        model.train()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "recon_button_acc": btn_correct / btn_total,
        "recon_button_f1": f1,
        "recon_cont_mae": cont_abs_err / cont_count,
    }


def eval_vs_cpu(
    model: GPT, stats: dict[str, FeatureStats], cfg: TrainConfig, *, max_frames: int, replay_dir: Path | None = None
) -> dict[str, float]:
    """In-training closed-loop eval vs lvl-9 CPU over prior-sampled char matchups.

    ``max_parallel`` Dolphin boots (one per distinct prior-sampled matchup, scaled to
    the box's CPUs) each play many matches back-to-back via instant-restart on random
    legal stages — broad, prior-weighted coverage that navigates the flaky stage-select
    menu only once per boot. Reduced to a flat metric dict."""
    was_training = model.training
    model.eval()
    n = _eval_max_parallel(cfg)
    try:
        results = sweep_vs_cpu_prior(
            lambda: make_policy(model, stats, cfg),
            session_cfg=default_session_cfg(replay_dir, instant_match_restart=True),
            n_matchups=n,
            max_parallel=n,
            max_frames=max_frames,
        )
    finally:
        if was_training:
            model.train()
    return vs_cpu_metrics(results)


# %%
def train(
    cfg: TrainConfig,
    stats: dict[str, FeatureStats],
    *,
    comment: str = "",
    resume_run: str | None = None,
    resume_state: dict | None = None,
) -> None:
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

    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high" if cfg.allow_tf32 else "highest")
    if cfg.amp_dtype not in ("bfloat16", "float32"):
        raise ValueError(f"amp_dtype must be 'bfloat16' or 'float32', got {cfg.amp_dtype!r}")
    autocast = (
        torch.autocast(DEVICE, dtype=torch.bfloat16)
        if cfg.amp_dtype == "bfloat16" and DEVICE == "cuda"
        else contextlib.nullcontext()
    )
    start_step = resume_state["step"] + 1 if resume_state else 0
    model = GPT(cfg).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    if wandb.run is not None:
        wandb.run.summary["model/num_params"] = n_params
    print(f"[model] {_model_tag(cfg)}  num_params={n_params / 1e6:.2f}M", flush=True)
    loader_kwargs = dict(
        data_root=cfg.data_root,
        remote=streams.remote_for_local(cfg.data_root),
        cache_limit=f"{cfg.cache_limit_gb}gb",
        shuffle_block_size=cfg.shuffle_block_size,
        stats=stats,
        L_ctx=cfg.L_ctx,
        L_chunk=max(cfg.head_offsets),  # target horizon must cover the farthest auxiliary head
        batch_size=cfg.batch_size,
        seed=cfg.seed,
    )
    train_loader = make_loader(
        split="train",
        num_workers=cfg.num_workers,
        prefetch_factor=cfg.prefetch_factor,
        windows_per_replay=cfg.windows_per_replay,
        **loader_kwargs,
    )
    # The frozen val chunk must cover the farthest auxiliary head so every offset has a val target.
    if max(cfg.head_offsets) > VAL_L_CHUNK:
        raise ValueError(f"max(head_offsets)={max(cfg.head_offsets)} exceeds frozen VAL_L_CHUNK={VAL_L_CHUNK}")
    # Val uses the FROZEN wider chunk (VAL_L_CHUNK) so its window geometry — hence its NLL — is
    # comparable across experiments regardless of the train-time L_chunk. This makes val loss NOT
    # comparable to pre-freeze 012 runs; that break is the intended freeze. The val path slices the
    # wider target back to max(head_offsets) frames.
    val_loader = make_loader(split=cfg.val_split, num_workers=0, **{**loader_kwargs, "L_chunk": VAL_L_CHUNK})

    opt = make_optimizer(model, cfg)
    sched = LambdaLR(opt, lr_schedule(cfg))
    if resume_state is not None:
        model.load_state_dict(resume_state["model"])
        opt.load_state_dict(resume_state["opt"])
        sched.load_state_dict(resume_state["sched"])
        print(f"[resume] {run_name}: continuing from step {start_step}", flush=True)

    print("[val] building cached val set…", flush=True)
    val_t0 = time.monotonic()
    val_cache = [b.to(DEVICE) for b in itertools.islice(val_loader, cfg.val_n_batches)]
    if not val_cache:
        raise RuntimeError("val loader yielded zero batches")
    print(
        f"[val] cached {len(val_cache)} batches "
        f"({sum(b.target.shape[0] for b in val_cache)} samples) in {time.monotonic() - val_t0:.1f}s",
        flush=True,
    )

    def _wandb_id() -> str | None:
        return wandb.run.id if wandb.run is not None else None

    def _eval_and_upload(step_tag: str) -> dict[str, float]:
        """Synchronous closed-loop eval on the live model + .slp upload (the final eval).
        Returns the flat metric dict."""
        sub = replay_dir / step_tag
        metrics = eval_vs_cpu(model, stats, cfg, max_frames=cfg.eval_max_frames, replay_dir=sub)
        n = uploader.upload_tree(sub, base=ckpt_dir, pattern="*.slp")
        print(f"[eval] queued {n} .slp for R2 ({step_tag})", flush=True)
        return metrics

    def _val_log_dict() -> dict[str, float]:
        """Flat ``val/*`` metric dict (one W&B section). Merged into the per-step log; no wandb.log here."""
        vm = val_metrics(model, val_cache, cfg)
        gen = torch.Generator(device=DEVICE).manual_seed(0)
        recon = {"argmax": recon_metrics(model, val_cache, argmax=True)}
        recon["sample"] = recon_metrics(model, val_cache, argmax=False, temp=cfg.decode_temp, gen=gen)
        out = {f"val/{k}": v for k, v in vm.items()}
        for tag, rm in recon.items():
            out[f"val/recon_{tag}_acc"] = rm["recon_button_acc"]
            out[f"val/recon_{tag}_f1"] = rm["recon_button_f1"]
            out[f"val/recon_{tag}_mae"] = rm["recon_cont_mae"]
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
                print(f"[eval] queued {n} .slp for R2 (step {step})", flush=True)
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

    model.train()
    it = iter(train_loader)
    run_t0 = time.monotonic()
    for step in range(start_step, cfg.max_steps):
        with profile("step") as sw:
            opt.zero_grad()
            comps_acc: dict[tuple[int, str], list[Tensor]] = {}
            obj_acc: Tensor | None = None
            for _ in range(cfg.grad_accum_steps):
                try:
                    batch = next(it).to(DEVICE)
                except StopIteration:
                    it = iter(train_loader)
                    batch = next(it).to(DEVICE)
                with autocast:
                    comps, trans = action_loss(model, batch)
                    obj = objective(comps, trans, cfg.aux_loss_weight, cfg.transition_loss_weight)
                    loss = obj / cfg.grad_accum_steps
                loss.backward()
                obj_acc = obj.detach() if obj_acc is None else obj_acc + obj.detach()
                for k, v in comps.items():
                    comps_acc.setdefault(k, []).append(v.detach())
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))  # measure only
            opt.step()
            sched.step()
            if DEVICE == "cuda":
                torch.cuda.synchronize()
        assert obj_acc is not None  # grad_accum_steps >= 1
        objective_bits = (obj_acc / cfg.grad_accum_steps).item() / _LN2  # the actual backprop objective, bits
        comps_cat = {k: torch.cat(v) for k, v in comps_acc.items()}
        primary = nll_breakdown({name: comps_cat[(1, name)] for name in _GROUP_NAMES})
        aux_offsets = [o for o in cfg.head_offsets if o != 1]
        aux_loss = (
            sum(_offset_total_bits(comps_cat, o) for o in aux_offsets) / len(aux_offsets) if aux_offsets else 0.0
        )
        sps = cfg.batch_size * cfg.grad_accum_steps / sw.elapsed
        samples = (step + 1) * cfg.batch_size * cfg.grad_accum_steps
        log = {
            "global_step": step,
            "samples": samples,
            "tokens": samples * cfg.L_ctx,
            "train/loss": primary["total"],  # offset-1 head (deployed); aux heads tracked separately
            **{f"train/nll_{name}": primary[name] for name in _GROUP_NAMES},
            "train/aux_loss": aux_loss,  # mean total bits/frame over the auxiliary (offset != 1) heads
            "train/objective": objective_bits,  # weighted backprop objective (transition-upweighted), bits
            "lr/muon": next(g["lr"] for g in opt.param_groups if g["use_muon"]),
            "lr/adam": next(g["lr"] for g in opt.param_groups if not g["use_muon"]),
            "train/gnorm": grad_norm.item(),
            "throughput/step_s": sw.elapsed,
            "throughput/samples_per_s": sps,
        }
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
    _log_eval(cfg.max_steps, _eval_and_upload("final"))
    _save("final.pt", cfg.max_steps)
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
    return TrainConfig(**{k: v for k, v in saved.items() if k in known})


def _load_ckpt(ckpt_path: str) -> tuple[GPT, TrainConfig, dict[str, FeatureStats], dict]:
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = _cfg_from_state(state["cfg"])
    model = GPT(cfg).to(DEVICE)
    model.load_state_dict(state["model"])
    model.eval()
    stats = load_consolidated_stats(Path(cfg.data_root) / "stats.json")
    return model, cfg, stats, state


def eval_ckpt(
    ckpt_path: str,
    *,
    eval_exec_horizon: int | None = None,
    decode_temp: float | None = None,
    decode_temps: tuple[float, float, float, float] | None = None,
    decode_btn_support_min: int | None = None,
    decode_min_p: float | None = None,
    decode_click_trigger_fix: bool | None = None,
) -> None:
    """Load a checkpoint and run the prior-distribution vs-CPU sweep, printing the pooled metrics. Each
    override (execution horizon, temp, per-group temps, button-support floor, min-p, click=>trigger fix)
    replaces the trained cfg for this eval only (test-time sweep); ``None`` keeps the trained value."""
    model, cfg, stats, state = _load_ckpt(ckpt_path)
    exec_horizon = cfg.exec_horizon if eval_exec_horizon is None else eval_exec_horizon
    temp = cfg.decode_temp if decode_temp is None else decode_temp
    temps = cfg.decode_temps if decode_temps is None else decode_temps
    btn_support_min = cfg.decode_btn_support_min if decode_btn_support_min is None else decode_btn_support_min
    min_p = cfg.decode_min_p if decode_min_p is None else decode_min_p
    click_fix = cfg.decode_click_trigger_fix if decode_click_trigger_fix is None else decode_click_trigger_fix
    print(
        f"[eval] loaded {ckpt_path}  step={state['step']}  device={DEVICE}  exec_horizon={exec_horizon}  "
        f"temp={temp}  temps={temps}  btn_support_min={btn_support_min}  min_p={min_p}  click_trigger_fix={click_fix}",
        flush=True,
    )
    replay_dir = Path(ckpt_path).resolve().parent / "eval_replays"
    replay_dir.mkdir(parents=True, exist_ok=True)
    session_cfg = default_session_cfg(replay_dir, instant_match_restart=True)
    n = _eval_max_parallel(cfg)

    def policy_factory() -> RecedingHorizon:
        return make_policy(
            model,
            stats,
            cfg,
            exec_horizon=eval_exec_horizon,
            decode_temp=decode_temp,
            decode_temps=decode_temps,
            decode_btn_support_min=decode_btn_support_min,
            decode_min_p=decode_min_p,
            decode_click_trigger_fix=decode_click_trigger_fix,
        )

    print(f"\n[eval] ===== vs-cpu, {n} prior-sampled matchups (instant-restart) =====", flush=True)
    results = sweep_vs_cpu_prior(
        policy_factory,
        session_cfg=session_cfg,
        n_matchups=n,
        max_parallel=n,
        max_frames=15_000,
    )
    print(f"  {vs_cpu_metrics(results)}", flush=True)


# %%
def run_eval_worker(ckpt_path: str, step: int, result_path: str, replay_dir: str) -> None:
    """One-shot closed-loop eval for the async path: load a checkpoint, sweep vs CPU, and write the
    flat metric dict to ``result_path`` (atomically) with the .slp recordings under ``replay_dir``.
    Touches neither W&B nor R2 — the launching trainer is the sole writer/uploader."""
    model, cfg, stats, _ = _load_ckpt(ckpt_path)
    metrics = eval_vs_cpu(model, stats, cfg, max_frames=cfg.eval_max_frames, replay_dir=Path(replay_dir))
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
    eval_exec_horizon: int | None = None  # override execution horizon s for --eval (chunked decode; 1=per-frame)
    eval_temp: float | None = None  # override decode temperature for --eval
    eval_temps: tuple[float, float, float, float] | None = None  # per-group temps (buttons, main, c, triggers)
    eval_btn_support_min: int | None = None  # mask button combos with < this many train frames (0=off)
    eval_min_p: float | None = None  # min-p nucleus: keep classes with p >= min_p * p_max
    eval_click_trigger_fix: bool | None = None  # force trigger_l/r to 1.0 on a digital L/R click
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
    if args.eval is not None:
        eval_ckpt(
            args.eval,
            eval_exec_horizon=args.eval_exec_horizon,
            decode_temp=args.eval_temp,
            decode_temps=args.eval_temps,
            decode_btn_support_min=args.eval_btn_support_min,
            decode_min_p=args.eval_min_p,
            decode_click_trigger_fix=args.eval_click_trigger_fix,
        )
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
    auto_comment = f"gpt-{cfg.max_steps // 1000}k-b{cfg.batch_size}"
    train(cfg, stats, comment=args.comment or auto_comment)


if __name__ == "__main__":
    main(tyro.cli(Args))
