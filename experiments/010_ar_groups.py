"""GPT next-token action policy with an autoregressive-groups head (009 variant).

Identical to 009 (GPT next-token backbone, 4-player input incl. nana) except the
output head: instead of one joint linear emitting every group's vocab
independently, the four groups are predicted AUTOREGRESSIVELY within each frame.
Group ``g`` conditions on the realized classes of groups ``0..g-1`` (their
embeddings concatenated onto the backbone hidden), so group ``g``'s linear has a
GROWING input dim ``d_model + g·d_group`` — a fixed-in-dim linear from the hidden
alone cannot be autoregressive. Order: buttons → main-stick → c-stick → triggers
(buttons first so the trigger analog can condition on the digital L/R click).

Training teacher-forces all four groups in one parallel forward (the conditioning
classes are the targets); decode samples them sequentially (4 steps per frame).

Run:
    uv run experiments/010_ar_groups.py
    uv run experiments/010_ar_groups.py --eval <ckpt>
    uv run experiments/010_ar_groups.py --eval <ckpt> --eval-temp 0.7
"""

# %%
import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import contextlib
import itertools
import math
import time
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from pathlib import Path

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
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import wandb
from hal import streams
from hal.data.stats import FeatureStats
from hal.eval.cross_stage import sweep_self_play
from hal.eval.cross_stage import sweep_vs_cpu
from hal.eval.cross_stage import vs_cpu_metrics
from hal.eval.harness import default_session_cfg
from hal.training import scoring
from hal.training.checkpoints import BackgroundUploader
from hal.training.checkpoints import load_for_resume
from hal.training.checkpoints import save_checkpoint
from hal.training.closed_loop import RecedingHorizon
from hal.training.dataloader import make_loader
from hal.training.features import A_DIM
from hal.training.features import ACTION_CHANNELS
from hal.training.features import FLOAT_FEATURES
from hal.training.features import PLAYER_CAT_FEATURES
from hal.training.features import Context
from hal.training.features import TrainBatch
from hal.training.features import stack_actions
from hal.training.runs import make_run_name
from hal.training.runs import profile
from hal.training.runs import setup_run_dir
from hal.training.stats import load_consolidated_stats

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_LN2 = math.log(2.0)
L_CHUNK = 1  # next-token: predict one frame ahead, replan every frame

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


# %%
@dataclass
class TrainConfig:
    # GPT backbone
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 8
    d_group: int = 32  # embedding dim for a realized group class fed to later groups' heads (AR conditioning)
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
    seed: int = 0
    L_ctx: int = 256
    # optimization
    batch_size: int = 128
    grad_accum_steps: int = 1
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: int = 2**15
    amp_dtype: str = "bfloat16"  # "bfloat16" | "float32"
    allow_tf32: bool = True
    # eval cadence
    val_every: int = 1024
    val_n_batches: int = 16
    eval_every: int = 2048
    eval_max_frames: int = 7200
    eval_replicas: int = 16
    eval_max_parallel: int = 8
    # checkpointing
    ckpt_every: int = 2048
    push_to_r2: bool = True
    # data (v4 MDS carries the stage + p{1,2}_character + nana columns)
    data_root: str = "data/processed/ranked-anonymized-1/mds"
    cache_limit_gb: int = 440
    shuffle_block_size: int = 2000
    val_split: str = "val"
    num_workers: int = 8
    prefetch_factor: int = 4


def _model_tag(cfg: TrainConfig) -> str:
    return f"gpt-ar-d{cfg.d_model}-L{cfg.n_layers}-h{cfg.n_heads}-dg{cfg.d_group}-Lc{cfg.L_ctx}"


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
    """Causal GPT over per-frame tokens + an autoregressive-groups head. ``hidden[i]`` (causal)
    predicts the next frame's action: the four groups are decoded in order, each conditioned on the
    realized classes of the earlier ones (``head_logits`` teacher-forced, ``head_decode`` sequential)."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        if not cfg.decode_temp > 0:
            raise ValueError(f"decode_temp must be > 0, got {cfg.decode_temp}")
        self.L_ctx = cfg.L_ctx

        # Gamestate categoricals: one table per feature name, shared across the four players.
        self.cat_embeds = nn.ModuleDict(
            {name: nn.Embedding(vocab, dim) for name, (vocab, dim) in PLAYER_CAT_FEATURES.items()}
        )
        self.char_emb = nn.Embedding(cfg.char_vocab, cfg.char_dim)
        self.stage_emb = nn.Embedding(cfg.stage_vocab, cfg.stage_dim)
        per_player = len(FLOAT_FEATURES) * 2 + sum(dim for _, dim in PLAYER_CAT_FEATURES.values())  # float+mask+cat
        d_in = len(_PLAYER_PREFIXES) * per_player + A_DIM + 2 * cfg.char_dim + cfg.stage_dim

        self.ctx_proj = nn.Linear(d_in, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])

        # Autoregressive-groups head: group g's input is the hidden plus the embedded realized classes
        # of groups 0..g-1, so its linear's in-dim grows by d_group per earlier group.
        self.group_in = nn.ModuleList([nn.Embedding(_GROUP_VOCABS[g], cfg.d_group) for g in range(N_GROUPS - 1)])
        self.group_out = nn.ModuleList(
            [nn.Linear(cfg.d_model + g * cfg.d_group, _GROUP_VOCABS[g]) for g in range(N_GROUPS)]
        )

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
        for name, (vocab, _) in PLAYER_CAT_FEATURES.items():
            parts.append(self.cat_embeds[name](features[f"{prefix}_{name}"].clamp(0, vocab - 1)))
        return torch.cat(parts, dim=-1)

    def _context_tokens(self, features: dict[str, Tensor]) -> Float[Tensor, "B L_ctx d_model"]:
        parts = [self._per_player_features(features, p) for p in _PLAYER_PREFIXES]
        parts.append(torch.cat([features[f"ego_{ch}"][..., None] for ch in ACTION_CHANNELS], dim=-1))
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
        """Backbone hidden (one vector per frame); the AR head turns it into per-group logits."""
        x = self._context_tokens(features)
        mask = self._attn_mask(ctx_pad, x.size(1), x.device)
        for block in self.blocks:
            x = block(x, mask)
        return rmsnorm(x)

    def head_logits(self, h: Tensor, tgt_idx: Tensor) -> list[Tensor]:
        """Teacher-forced per-group logits ``[..., vocab_g]``. Group g's head sees the hidden plus the
        embedded REALIZED (target) classes of groups 0..g-1, so one parallel forward yields all four."""
        cond = [h]
        out: list[Tensor] = []
        for g in range(N_GROUPS):
            out.append(self.group_out[g](torch.cat(cond, dim=-1)).float())
            if g < N_GROUPS - 1:
                cond.append(self.group_in[g](tgt_idx[..., g]).to(h.dtype))
        return out

    def head_decode(self, h: Tensor, *, temp: float, argmax: bool, gen: torch.Generator | None) -> Tensor:
        """Sequentially decode the four groups for one frame (``h`` ``[B, d_model]``); group g
        conditions on the SAMPLED classes of groups 0..g-1. Returns class indices ``[B, N_GROUPS]``."""
        cond = [h]
        picks: list[Tensor] = []
        for g in range(N_GROUPS):
            lg = self.group_out[g](torch.cat(cond, dim=-1)).float()
            c = (
                lg.argmax(-1)
                if argmax
                else torch.multinomial(F.softmax(lg / temp, dim=-1), 1, generator=gen).squeeze(-1)
            )
            picks.append(c)
            if g < N_GROUPS - 1:
                cond.append(self.group_in[g](c).to(h.dtype))
        return torch.stack(picks, dim=-1)


# %%
def _quantize(model: GPT, actions: Tensor) -> Tensor:
    return quantize_groups(model.main_centers, model.c_centers, model.trig_centers, actions)


def _dequantize(model: GPT, idx: Tensor) -> Tensor:
    return dequantize_groups(model.main_centers, model.c_centers, model.trig_centers, idx)


def _next_action_targets(ctx: Context, target: Tensor) -> tuple[Tensor, Tensor]:
    """Per context position ``i``, the next frame's action + a validity mask. The ego controller
    history already lives in ``ctx.features``, so ``a_full = [history | target]`` and position
    ``i``'s leak-free target is ``a_full[i+1]`` (last position recovers ``target``)."""
    a_full = torch.cat([stack_actions(ctx.features), target], dim=1)  # [B, L_ctx+1, A_DIM]
    nxt = a_full[:, 1:]  # [B, L_ctx, A_DIM]
    pos = torch.arange(nxt.size(1), device=nxt.device)
    valid = pos[None, :] >= ctx.ctx_pad[:, None]
    return nxt, valid


def group_nll(logits: list[Tensor], tgt_idx: Tensor, valid: Tensor) -> dict[str, Tensor]:
    """Per-group categorical NLL (nats) over the VALID positions only. Returns ``{name: [n_valid]}``
    1D tensors (same ordering across groups) so callers reduce once for exact sample weighting."""
    flat_valid = valid.reshape(-1)
    out: dict[str, Tensor] = {}
    for g, name in enumerate(_GROUP_NAMES):
        lg = logits[g].reshape(-1, _GROUP_VOCABS[g])[flat_valid]
        out[name] = F.cross_entropy(lg, tgt_idx[..., g].reshape(-1)[flat_valid], reduction="none")
    return out


def action_loss(model: GPT, batch: TrainBatch) -> dict[str, Tensor]:
    """Dense next-token NLL: every valid context position predicts its next frame's action."""
    ctx = batch.context
    nxt, valid = _next_action_targets(ctx, batch.target)
    tgt_idx = _quantize(model, nxt)
    logits = model.head_logits(model(ctx.features, ctx.ctx_pad), tgt_idx)
    return group_nll(logits, tgt_idx, valid)


@torch.no_grad()
def decode(
    model: GPT, ctx: Context, *, temp: float = 1.0, argmax: bool = False, gen: torch.Generator | None = None
) -> Float[Tensor, "B 1 d_action"]:
    """One next-frame action per sample from the LAST context position, in raw action ranges.
    The four groups are decoded autoregressively (each conditions on the realized earlier ones),
    sampling (``temp``-scaled softmax) or greedily (``argmax``, for the recon metric)."""
    h = model(ctx.features, ctx.ctx_pad)[:, -1]  # [B, d_model]
    idx = model.head_decode(h, temp=temp, argmax=argmax, gen=gen)  # [B, N_GROUPS]
    return _dequantize(model, idx)[:, None, :]


def make_policy(
    model: GPT,
    stats: dict[str, FeatureStats],
    cfg: TrainConfig,
    *,
    device: str = DEVICE,
    decode_temp: float | None = None,
) -> RecedingHorizon:
    """Fresh closed-loop policy for one eval wave: replan every frame, decode the next action, sample."""
    temp = cfg.decode_temp if decode_temp is None else decode_temp

    @torch.no_grad()
    def predict_chunk(ctx: Context, committed: np.ndarray | None) -> np.ndarray:
        assert committed is None, "next-token policy does not condition on a committed prefix"
        return decode(model, ctx, temp=temp).cpu().numpy()

    return RecedingHorizon(
        predict_chunk=predict_chunk, stats=stats, L_ctx=cfg.L_ctx, L_chunk=L_CHUNK, s=1, d=0, device=device
    )


# %%
def lr_schedule(cfg: TrainConfig):
    """Linear warmup → cosine to floor."""
    floor = 1e-5 / cfg.lr

    def fn(step: int) -> float:
        if step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)
        progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    return fn


def nll_breakdown(comps: dict[str, Tensor]) -> dict[str, float]:
    """Per-group modality NLL (bits) + total bits/frame, from the per-group ``[n_valid]`` nats."""
    out = {f"modality/{name}": (c.mean().item() / _LN2) for name, c in comps.items()}
    out["total"] = sum(c.mean() for c in comps.values()).item() / _LN2
    return out


@torch.no_grad()
def val_metrics(model: GPT, val_cache: list[TrainBatch], cfg: TrainConfig) -> dict[str, float]:
    """Dense next-token proper-scoring metrics over the cached val batches. Per-element tensors are
    concatenated then reduced once, so the means are exactly sample-weighted."""
    was_training = model.training
    model.eval()
    comps_cat: dict[str, list[Tensor]] = {}
    btn_probs: list[Tensor] = []
    btn_tgts: list[Tensor] = []
    multipress: list[Tensor] = []
    for batch in val_cache:
        ctx = batch.context
        nxt, valid = _next_action_targets(ctx, batch.target)
        tgt_idx = _quantize(model, nxt)
        logits = model.head_logits(model(ctx.features, ctx.ctx_pad), tgt_idx)
        for k, v in group_nll(logits, tgt_idx, valid).items():
            comps_cat.setdefault(k, []).append(v)
        flat_valid = valid.reshape(-1)
        btn_logits = logits[_BUTTONS_G].reshape(-1, scoring.N_BUTTON_COMBOS)[flat_valid]
        btn_probs.append(scoring.combo_marginal_probs(btn_logits))
        tgt_btn = _dequantize(model, tgt_idx)[..., _N_CONT:].reshape(-1, _N_BUTTONS)[flat_valid]
        btn_tgts.append(tgt_btn)
        multipress.append((tgt_btn > 0.5).sum(-1) >= 2)
    comps = {k: torch.cat(v) for k, v in comps_cat.items()}
    out = {f"loss/{k}": v for k, v in nll_breakdown(comps).items()}
    out["action_nll_bits_per_frame"] = sum(c.mean() for c in comps.values()).item() / _LN2
    out["cont_discrete_bits"] = (
        comps["main_stick"].mean() + comps["c_stick"].mean() + comps["triggers"].mean()
    ).item() / _LN2
    logloss, brier = scoring.bernoulli_scores_from_probs(torch.cat(btn_probs), torch.cat(btn_tgts))
    out["buttons/logloss_bits"] = logloss.item()
    out["buttons/brier"] = brier.item()
    out["buttons/multipress_rate"] = torch.cat(multipress).float().mean().item()
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
        tgt = batch.target
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
    """In-training closed-loop eval on FD vs lvl-9 CPU, reduced to a flat metric dict."""
    was_training = model.training
    model.eval()
    try:
        results = sweep_vs_cpu(
            lambda: make_policy(model, stats, cfg),
            session_cfg=default_session_cfg(replay_dir),
            stages=(melee.Stage.FINAL_DESTINATION,),
            replicas=cfg.eval_replicas,
            max_parallel=cfg.eval_max_parallel,
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
    run_name = resume_run or make_run_name(_model_tag(cfg), cfg.data_root, comment)
    uploader = BackgroundUploader(run_name) if cfg.push_to_r2 else None
    wandb.init(
        project="hal",
        name=run_name,
        id=resume_state["wandb_id"] if resume_state else None,
        resume="allow" if resume_state else None,
        tags=["gpt-ar", f"d{cfg.d_model}", f"L{cfg.n_layers}"],
        config=asdict(cfg),
    )
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
        L_chunk=L_CHUNK,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
    )
    train_loader = make_loader(
        split="train", num_workers=cfg.num_workers, prefetch_factor=cfg.prefetch_factor, **loader_kwargs
    )
    val_loader = make_loader(split=cfg.val_split, num_workers=0, **loader_kwargs)

    opt = AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=cfg.weight_decay)
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
        sub = replay_dir / step_tag
        metrics = eval_vs_cpu(model, stats, cfg, max_frames=cfg.eval_max_frames, replay_dir=sub)
        if uploader is not None:
            n = uploader.upload_tree(sub, base=ckpt_dir, pattern="*.slp")
            print(f"[eval] queued {n} .slp for R2 ({step_tag})", flush=True)
        return metrics

    def _log_val(step: int) -> dict[str, float]:
        vm = val_metrics(model, val_cache, cfg)
        gen = torch.Generator(device=DEVICE).manual_seed(0)
        rm_arg = recon_metrics(model, val_cache, argmax=True)
        rm_smp = recon_metrics(model, val_cache, argmax=False, temp=cfg.decode_temp, gen=gen)
        wandb.log(
            {
                **{f"val/{k}": v for k, v in vm.items()},
                **{f"val/argmax/{k}": v for k, v in rm_arg.items()},
                **{f"val/sample/{k}": v for k, v in rm_smp.items()},
            },
            step=step,
        )
        return vm

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

    model.train()
    it = iter(train_loader)
    run_t0 = time.monotonic()
    for step in range(start_step, cfg.max_steps):
        with profile("step") as sw:
            opt.zero_grad()
            loss_val = 0.0
            comps_acc: dict[str, list[Tensor]] = {}
            for _ in range(cfg.grad_accum_steps):
                try:
                    batch = next(it).to(DEVICE)
                except StopIteration:
                    it = iter(train_loader)
                    batch = next(it).to(DEVICE)
                with autocast:
                    comps = action_loss(model, batch)
                    loss = sum(comps.values()).mean() / cfg.grad_accum_steps
                loss.backward()
                loss_val += loss.item()
                for k, v in comps.items():
                    comps_acc.setdefault(k, []).append(v.detach())
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))  # measure only
            opt.step()
            sched.step()
            if DEVICE == "cuda":
                torch.cuda.synchronize()
        breakdown = nll_breakdown({k: torch.cat(v) for k, v in comps_acc.items()})
        sps = cfg.batch_size * cfg.grad_accum_steps / sw.elapsed
        samples = (step + 1) * cfg.batch_size * cfg.grad_accum_steps
        wandb.log(
            {
                "train/loss": loss_val,
                **{f"train/loss/{k}": v for k, v in breakdown.items()},
                "train/lr": opt.param_groups[0]["lr"],
                "train/gnorm": grad_norm.item(),
                "throughput/step_s": sw.elapsed,
                "throughput/samples_per_s": sps,
                "samples": samples,
                "tokens": samples * cfg.L_ctx,
            },
            step=step,
        )
        if step < 20 or step % 50 == 0:
            print(
                f"[t+{time.monotonic() - run_t0:.0f}s] step {step}: loss {loss_val:.4f} "
                f"step_dt={sw.elapsed * 1000:.0f}ms ({sps:.1f} samples/s)",
                flush=True,
            )
        if cfg.ckpt_every > 0 and step > 0 and step % cfg.ckpt_every == 0:
            _save("latest.pt", step)
        if cfg.val_every > 0 and step > 0 and step % cfg.val_every == 0:
            vm = _log_val(step)
            print(
                f"[t+{time.monotonic() - run_t0:.0f}s] step {step}: "
                f"action_nll {vm['action_nll_bits_per_frame']:.3f} btn_logloss {vm['buttons/logloss_bits']:.3f}",
                flush=True,
            )
        if cfg.eval_every > 0 and step > 0 and step % cfg.eval_every == 0:
            _save(f"step_{step:06d}.pt", step)
            metrics = _eval_and_upload(f"step_{step:06d}")
            wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=step)
            print(f"[t+{time.monotonic() - run_t0:.0f}s] step {step}: closed_loop {metrics}", flush=True)

    vm_final = _log_val(cfg.max_steps)
    print(f"[final] action_nll {vm_final['action_nll_bits_per_frame']:.3f}", flush=True)
    metrics_final = _eval_and_upload("final")
    wandb.log({f"eval/{k}": v for k, v in metrics_final.items()}, step=cfg.max_steps)
    print(f"[final] closed_loop {metrics_final}", flush=True)
    _save("final.pt", cfg.max_steps)
    if uploader is not None:
        uploader.close()


# %%
def _load_ckpt(ckpt_path: str) -> tuple[GPT, TrainConfig, dict[str, FeatureStats], dict]:
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = TrainConfig(**state["cfg"])
    model = GPT(cfg).to(DEVICE)
    model.load_state_dict(state["model"])
    model.eval()
    stats = load_consolidated_stats(Path(cfg.data_root) / "stats.json")
    return model, cfg, stats, state


def eval_ckpt(ckpt_path: str, *, decode_temp: float | None = None) -> None:
    """Load a checkpoint, sweep stages vs CPU + self-play, print summaries. ``decode_temp`` overrides
    the trained cfg for this eval only (test-time temperature sweep)."""
    from hal.policy import INCLUDED_STAGES

    model, cfg, stats, state = _load_ckpt(ckpt_path)
    temp = cfg.decode_temp if decode_temp is None else decode_temp
    print(f"[eval] loaded {ckpt_path}  step={state['step']}  device={DEVICE}  temp={temp}", flush=True)
    replay_dir = Path(ckpt_path).resolve().parent / "eval_replays"
    replay_dir.mkdir(parents=True, exist_ok=True)
    session_cfg = default_session_cfg(replay_dir)
    stages = tuple(s for s in INCLUDED_STAGES if s is not melee.Stage.FOUNTAIN_OF_DREAMS)

    def policy_factory() -> RecedingHorizon:
        return make_policy(model, stats, cfg, decode_temp=decode_temp)

    print("\n[eval] ============== vs-cpu ==============", flush=True)
    for stage, r, s in sweep_vs_cpu(
        policy_factory,
        session_cfg=session_cfg,
        stages=stages,
        replicas=cfg.eval_replicas,
        max_parallel=cfg.eval_max_parallel,
        max_frames=15_000,
    ):
        print(f"  {stage.name:18s} r{r} {s.as_dict() if s else 'CRASHED'}", flush=True)
    print("\n[eval] ============== self-play ==============", flush=True)
    for stage, r, s in sweep_self_play(
        policy_factory,
        session_cfg=session_cfg,
        stages=stages,
        replicas=cfg.eval_replicas,
        max_parallel=cfg.eval_max_parallel,
        max_frames=15_000,
    ):
        print(f"  {stage.name:18s} r{r} {s.as_dict() if s else 'CRASHED'}", flush=True)


# %%
@dataclass
class Args:
    """Top-level CLI surface. Pass TrainConfig fields as kebab-case flags, e.g. ``--cfg.d-model 512``."""

    cfg: TrainConfig = field(default_factory=TrainConfig)
    eval: str | None = None  # ckpt path; closed-loop eval instead of train
    eval_temp: float | None = None  # override decode temperature for --eval
    resume: str | None = None  # run_name to resume; pulls latest.pt (local, else R2)
    comment: str = ""


def main(args: Args) -> None:
    if args.eval is not None:
        eval_ckpt(args.eval, decode_temp=args.eval_temp)
        return
    if args.resume is not None:
        state = load_for_resume(args.resume, Path("runs") / args.resume, device=DEVICE)
        if state is None:
            raise SystemExit(f"no latest.pt for run {args.resume!r} (local or R2)")
        # Only pure host-scaling knobs (worker/prefetch counts) follow the current code; the
        # model-identity knobs MUST come from the checkpoint so a resume can't silently change them.
        d = TrainConfig()
        cfg = replace(TrainConfig(**state["cfg"]), num_workers=d.num_workers, prefetch_factor=d.prefetch_factor)
        stats = load_consolidated_stats(Path(cfg.data_root) / "stats.json")
        train(cfg, stats, resume_run=args.resume, resume_state=state)
        return
    cfg = args.cfg
    stats = load_consolidated_stats(Path(cfg.data_root) / "stats.json")
    auto_comment = f"gpt-ar-{cfg.max_steps // 1000}k-b{cfg.batch_size}"
    train(cfg, stats, comment=args.comment or auto_comment)


if __name__ == "__main__":
    main(tyro.cli(Args))
