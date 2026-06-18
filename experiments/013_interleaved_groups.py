"""A nanoGPT/GPT-2-style causal decoder that INTERLEAVES the four controller
action groups as their own tokens in the main transformer sequence, so the trunk
autoregressively factorizes them.

Where 011/012 emit one token per frame and predict a joint logit vector split into
four group slices sampled INDEPENDENTLY (conditional independence given context),
and where 010 factorizes the groups in a shallow growing-input HEAD, 013 expands
the sequence itself. Each frame contributes ``P = 5`` tokens:

    state_t, a^c-stick_t, a^trigger_t, a^buttons_t, a^main-stick_t, state_{t+1}, ...

The full transformer depth then learns the proper conditionals
``P(c | s) · P(trig | s,c) · P(btn | s,c,trig) · P(main | s,c,trig,btn)`` — a clean
AR factorization, and no concat-feature-dims output head. The state token is
GAMESTATE-ONLY (all four players + matchup char/stage); the ego action flows in via
the preceding group tokens (concatenating frame-t's action into state-t would leak
the target). Prediction sites per frame are token offsets 0..3 (state→g0, g0→g1,
g1→g2, g2→g3); offset 4 (g3) would predict the next state → no action loss.

Next-frame-only (no multi-frame auxiliary heads): the action a_t supervised at
frame t is the leak-free NEXT action (``a_full[t+1]``, exactly 010's target shift),
which is both the teacher-forced group tokens at frame t and the 4-site target.
Closed-loop play decodes one frame at a time, AR-decoding the 4 groups with 4
sequential trunk passes (RecedingHorizon is agnostic to internal pass count).

Run:
    uv run experiments/013_interleaved_groups.py
    uv run experiments/013_interleaved_groups.py --eval <ckpt> --eval-temp 0.7
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
from hal.data.stats import FeatureStats
from hal.eval.cross_stage import sweep_vs_cpu_prior
from hal.eval.cross_stage import vs_cpu_metrics
from hal.eval.harness import default_session_cfg
from hal.training import scoring
from hal.training.checkpoints import BackgroundUploader
from hal.training.checkpoints import load_for_resume
from hal.training.checkpoints import save_checkpoint
from hal.training.closed_loop import RecedingHorizon
from hal.training.dataloader import make_loader
from hal.training.features import A_DIM
from hal.training.features import CAT_FEATURES
from hal.training.features import FLOAT_FEATURES
from hal.training.features import Context
from hal.training.features import TrainBatch
from hal.training.features import stack_actions
from hal.training.muon import SingleDeviceMuonWithAuxAdam
from hal.training.runs import make_run_name
from hal.training.runs import profile
from hal.training.runs import setup_run_dir
from hal.training.stats import load_consolidated_stats

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_LN2 = math.log(2.0)
# Closed-loop decodes one frame at a time, replan every frame.
CLOSED_LOOP_L_CHUNK = 1

# Action-vector channel split (A_DIM=14): [0:6] sticks+triggers (continuous), [6:14] buttons {0,1}.
_N_CONT = 6
_N_BUTTONS = A_DIM - _N_CONT

# Per-frame input: all four players' gamestate concatenated in the feature dim.
_PLAYER_PREFIXES: tuple[str, ...] = ("ego", "ego_nana", "opp_nana", "opp")

# --- group / token bookkeeping -----------------------------------------------
# quantize_groups emits class indices in this NATIVE order (matches the discretizer stacking).
_NATIVE_ORDER: tuple[str, ...] = ("buttons", "main_stick", "c_stick", "triggers")
# Autoregressive DECODE order for the interleaved tokens: sparser/simpler groups first, the
# dominant main_stick conditioned on everything. g0..g3 below.
_GROUP_NAMES: tuple[str, ...] = ("c_stick", "triggers", "buttons", "main_stick")
N_GROUPS = len(_GROUP_NAMES)
# Permutations between the two orders (this particular pairing is its own inverse, but derive both
# from the name lists so the order can change without silent breakage).
_AR_PERM: list[int] = [_NATIVE_ORDER.index(n) for n in _GROUP_NAMES]  # ar_idx = native_idx[..., _AR_PERM]
_NATIVE_PERM: list[int] = [_GROUP_NAMES.index(n) for n in _NATIVE_ORDER]  # native_idx = ar_idx[..., _NATIVE_PERM]
_VOCAB_BY_NAME: dict[str, int] = {
    "buttons": scoring.N_BUTTON_COMBOS,  # 256
    "main_stick": scoring.STICK_CLUSTER_CENTERS_MAIN.shape[0],  # 65
    "c_stick": scoring.STICK_CLUSTER_CENTERS_C.shape[0],  # 9
    "triggers": scoring.TRIGGER_CENTERS.shape[0] ** 2,  # 25 (joint L*5 + R)
}
_GROUP_VOCABS: tuple[int, ...] = tuple(_VOCAB_BY_NAME[n] for n in _GROUP_NAMES)  # AR order: (9, 25, 256, 65)
_BUTTONS_AR_IDX = _GROUP_NAMES.index("buttons")  # 2: the AR slot / prediction site that emits buttons
# One state token + N_GROUPS action tokens per frame.
P = 1 + N_GROUPS  # 5

# Native indices inside the quantize/dequantize codec (decoupled from the AR order above).
_NATIVE_BUTTONS, _NATIVE_MAIN, _NATIVE_C, _NATIVE_TRIG = 0, 1, 2, 3


# %%
@dataclass
class TrainConfig:
    # GPT backbone
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 4
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
    # FRAMES; the interleaved token sequence is P * L_ctx long. Set to 51 (=> 255 tokens) so the
    # token budget — hence FLOPs/step — matches the d256/L8 joint-marginal baseline at its 256-frame
    # (256-token) window: an iso-FLOP / iso-step / iso-backbone head-to-head where the ONLY arch
    # change is the interleaved-AR token structure. The cost of the 5 tokens/frame is paid in context
    # (51 vs 256 frames), not in compute or capacity, so neither model is starved nor needs grad-accum.
    L_ctx: int = 51
    # optimization. Matched token budget lets the whole effective batch fit one forward — no grad-accum.
    # The matched-context variant (--cfg.L-ctx 256) is 5x longer, so there shrink the micro-batch and
    # raise grad-accum to keep effective batch 512 (e.g. --cfg.batch-size 256 --cfg.grad-accum-steps 2).
    batch_size: int = 512
    grad_accum_steps: int = 1
    # Two LRs: Muon for the blocks' hidden matrices, AdamW for the input proj / heads / embeddings / biases.
    muon_lr: float = 0.02
    adam_lr: float = 8.5e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: int = 2**14
    amp_dtype: str = "bfloat16"  # "bfloat16" | "float32"
    allow_tf32: bool = True
    # eval cadence
    val_every: int = 1024
    val_n_batches: int = 16
    eval_every: int = 4096
    eval_max_frames: int = 1800
    # Closed-loop eval parallelism scales with the box: max_parallel = round(this * cpu_count).
    # Each parallel slot is one Dolphin boot that (via instant-restart) plays many prior-sampled
    # matches back-to-back. The vast boxes are big (32 vCPU); 2.0 (~64 boots) collects lots of matches
    # per wave — they handle the oversubscription fine.
    eval_parallel_per_cpu: float = 2.0
    # Closed-loop eval runs SYNCHRONOUSLY at each eval boundary: training pauses and the eval gets the
    # GPU (and the box) uncontended. Async/subprocess eval starved + timed out under the disk-bound
    # trainer (every in-training eval in the 012 run hit the cap), so we pay the pause instead — each
    # eval's wall-time is bounded by eval_max_frames per boot, not a timeout.
    # checkpointing
    ckpt_every: int = 2048
    # data (v4 MDS carries the stage + p{1,2}_character + nana columns)
    data_root: str = "data/processed/ranked-anonymized-1/mds"
    cache_limit_gb: int = 440
    shuffle_block_size: int = 2000
    # Each replay deserialized off disk yields this many non-overlapping windows,
    # amortizing the whole-replay read (the disk bottleneck) over K samples. Train
    # only; val stays 1/replay so its loss stays comparable across runs.
    windows_per_replay: int = 4
    val_split: str = "val"
    num_workers: int = 16
    prefetch_factor: int = 4


def _model_tag(cfg: TrainConfig) -> str:
    return f"gpt-il-d{cfg.d_model}-L{cfg.n_layers}-h{cfg.n_heads}-Lc{cfg.L_ctx}"


def _eval_max_parallel(cfg: TrainConfig) -> int:
    """Concurrent Dolphin boots per eval wave: ``eval_parallel_per_cpu * cpu_count``, but RAM-capped.
    Each headless Dolphin needs a few GB; some vast hosts expose 256 vCPUs to the container, so the
    raw cpu scaling asks for 512 boots that overrun RAM and ALL fail to reach IN_GAME (-> crashed).
    Cap at ~one boot per 6 GB of total RAM so the wave fits (eval is synchronous, so the trainer's
    memory is idle and the box's RAM is the real limit)."""
    raw = round(cfg.eval_parallel_per_cpu * (os.cpu_count() or 1))
    total_gb = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9
    ram_cap = max(4, int(total_gb // 6))
    return max(1, min(raw, ram_cap))


# %%
@jaxtyped(typechecker=beartype)
def quantize_groups(
    main_centers: Float[Tensor, "n_main 2"],
    c_centers: Float[Tensor, "n_c 2"],
    trig_centers: Float[Tensor, " n_trig"],
    actions: Float[Tensor, "*batch d_action"],
) -> Int[Tensor, "*batch n_groups"]:
    """Raw ``A_DIM`` action vec → the four group class indices, in NATIVE order
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
    """Inverse of ``quantize_groups`` (NATIVE-order indices) → raw ``A_DIM`` action vec
    (``[-1,1]`` sticks, ``[0,1]`` triggers, ``{0,1}`` buttons)."""
    n_trig = trig_centers.shape[0]
    btn = scoring.combo_to_buttons(idx[..., _NATIVE_BUTTONS])
    main = scoring.cluster_to_xy(idx[..., _NATIVE_MAIN], main_centers)
    c = scoring.cluster_to_xy(idx[..., _NATIVE_C], c_centers)
    tl = scoring.center_to_value(idx[..., _NATIVE_TRIG] // n_trig, trig_centers)
    tr = scoring.center_to_value(idx[..., _NATIVE_TRIG] % n_trig, trig_centers)
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
    """Causal GPT over an INTERLEAVED token sequence: per frame, one gamestate token followed by the
    four action-group tokens (AR order ``c_stick, triggers, buttons, main_stick``). Plain causal
    attention over the ``P * L_ctx`` tokens factorizes the groups autoregressively; the four
    prediction sites per frame (token offsets 0..3) feed the four per-group heads."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        if not cfg.decode_temp > 0:
            raise ValueError(f"decode_temp must be > 0, got {cfg.decode_temp}")
        self.L_ctx = cfg.L_ctx

        # Gamestate categoricals: one table per feature name, shared across the four players.
        self.cat_embeds = nn.ModuleDict(
            {name: nn.Embedding(vocab, dim) for name, (vocab, dim) in CAT_FEATURES.items()}
        )
        self.char_emb = nn.Embedding(cfg.char_vocab, cfg.char_dim)
        self.stage_emb = nn.Embedding(cfg.stage_vocab, cfg.stage_dim)
        per_player = len(FLOAT_FEATURES) * 2 + sum(dim for _, dim in CAT_FEATURES.values())  # float+mask+cat
        # State token is GAMESTATE-ONLY (no ego-action concat): the action flows via the group tokens.
        d_state_in = len(_PLAYER_PREFIXES) * per_player + 2 * cfg.char_dim + cfg.stage_dim

        self.state_proj = nn.Linear(d_state_in, cfg.d_model)
        # Group-token input embeddings (vocab_g → d_model) and a learned per-role/intra-frame embedding
        # (P rows: state, g0, g1, g2, g3) added to every token so the trunk knows each token's role.
        self.group_emb = nn.ModuleList([nn.Embedding(_GROUP_VOCABS[g], cfg.d_model) for g in range(N_GROUPS)])
        self.role_emb = nn.Embedding(P, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        # One output head per group (d_model → vocab_g); head g reads prediction-site g.
        self.heads = nn.ModuleList([nn.Linear(cfg.d_model, _GROUP_VOCABS[g]) for g in range(N_GROUPS)])

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

    def _state_tokens(self, features: dict[str, Tensor]) -> Float[Tensor, "B L_ctx d_model"]:
        parts = [self._per_player_features(features, p) for p in _PLAYER_PREFIXES]
        parts.append(self.char_emb(features["ego_character"].clamp(0, self.char_emb.num_embeddings - 1)))
        parts.append(self.char_emb(features["opp_character"].clamp(0, self.char_emb.num_embeddings - 1)))
        parts.append(self.stage_emb(features["stage"].clamp(0, self.stage_emb.num_embeddings - 1)))
        return self.state_proj(torch.cat(parts, dim=-1))

    def _token_grid(self, features: dict[str, Tensor], g_idx: Int[Tensor, "B L_ctx n_groups"]) -> Tensor:
        """Interleave gamestate + group tokens → ``[B, L_ctx, P, d_model]`` (role embedding added).
        ``g_idx`` is the AR-order group class index per frame (teacher-forced action tokens)."""
        state = self._state_tokens(features)  # [B, L_ctx, d_model]
        g_emb = torch.stack([self.group_emb[g](g_idx[..., g]) for g in range(N_GROUPS)], dim=2)  # [B,L,4,d]
        tok = torch.cat([state.unsqueeze(2), g_emb.to(state.dtype)], dim=2)  # [B, L_ctx, P, d_model]
        return tok + self.role_emb.weight.to(state.dtype)

    def _attn_mask(
        self, ctx_pad_frames: Int[Tensor, " B"], L_tok: int, device: torch.device
    ) -> Bool[Tensor, "B 1 L L"]:
        """Causal mask over the interleaved sequence that also hides each sample's left-padded
        cold-start prefix (key < ctx_pad, scaled to tokens). A padded query keeps its diagonal so its
        row is never fully masked (SDPA would NaN)."""
        ctx_pad = ctx_pad_frames * P
        idx = torch.arange(L_tok, device=device)
        causal = idx[:, None] >= idx[None, :]
        key_real = idx[None, :] >= ctx_pad[:, None]
        diag = torch.eye(L_tok, dtype=torch.bool, device=device)
        return (causal[None] & (key_real[:, None, :] | diag[None]))[:, None]

    def trunk(self, x: Float[Tensor, "B L_tok d_model"], ctx_pad_frames: Int[Tensor, " B"]) -> Tensor:
        mask = self._attn_mask(ctx_pad_frames, x.size(1), x.device)
        for block in self.blocks:
            x = block(x, mask)
        return rmsnorm(x)

    def forward(self, features: dict[str, Tensor], ctx_pad: Int[Tensor, " B"], g_idx: Tensor) -> Tensor:
        """Hidden over the interleaved sequence → ``[B, P * L_ctx, d_model]``. ``g_idx`` are the
        AR-order teacher-forced group tokens (leak-free: each frame's groups encode its own action,
        causally after that frame's state token)."""
        tok = self._token_grid(features, g_idx)
        B, L, _, d = tok.shape
        return self.trunk(tok.reshape(B, P * L, d), ctx_pad)


# %%
def _quantize(model: GPT, actions: Tensor) -> Tensor:
    """Raw actions → NATIVE-order group indices."""
    return quantize_groups(model.main_centers, model.c_centers, model.trig_centers, actions)


def _dequantize(model: GPT, idx_native: Tensor) -> Tensor:
    """NATIVE-order group indices → raw actions."""
    return dequantize_groups(model.main_centers, model.c_centers, model.trig_centers, idx_native)


def _ar_group_idx(model: GPT, actions: Tensor) -> Tensor:
    """Raw actions → AR-order group class indices ``[..., n_groups]``."""
    return _quantize(model, actions)[..., _AR_PERM]


def _next_action_targets(ctx: Context, target: Tensor) -> tuple[Tensor, Tensor]:
    """Per context position ``i``, the next frame's action + a validity mask. The ego controller
    history (``a_{i-1}`` at position ``i`` — the ``(post_i, pre_i)`` alignment) lives in
    ``ctx.features``, so ``a_full = [history | target]`` and position ``i``'s leak-free action is
    ``a_full[i+1]`` (the last position uses ``target``). This action is BOTH the teacher-forced group
    tokens at frame ``i`` and the 4-site target."""
    a_full = torch.cat([stack_actions(ctx.features), target], dim=1)  # [B, L_ctx+L_chunk, A_DIM]
    nxt = a_full[:, 1 : 1 + ctx.features["ego_position_x"].size(1)]  # [B, L_ctx, A_DIM]
    pos = torch.arange(nxt.size(1), device=nxt.device)
    valid = pos[None, :] >= ctx.ctx_pad[:, None]
    return nxt, valid


def group_nll(logits: list[Tensor], tgt_idx: Tensor, valid: Tensor) -> dict[str, Tensor]:
    """Per-group categorical NLL (nats) over the VALID positions only. ``logits``/``tgt_idx`` are in
    AR order. Returns ``{name: [n_valid]}`` 1D tensors so callers reduce once for exact weighting."""
    flat_valid = valid.reshape(-1)
    out: dict[str, Tensor] = {}
    for g, name in enumerate(_GROUP_NAMES):
        lg = logits[g].reshape(-1, _GROUP_VOCABS[g])[flat_valid]
        out[name] = F.cross_entropy(lg, tgt_idx[..., g].reshape(-1)[flat_valid], reduction="none")
    return out


def _site_logits(model: GPT, hidden: Tensor, n_frames: int) -> list[Tensor]:
    """From the interleaved hidden ``[B, P*L, d]``, gather the 4 prediction sites per frame (token
    offsets 0..3) and apply the matching per-group head → ``[ [B, L, vocab_g] for g ]`` (AR order)."""
    B = hidden.size(0)
    h = hidden.reshape(B, n_frames, P, -1)[:, :, :N_GROUPS, :]  # [B, L, 4, d]
    return [model.heads[g](h[:, :, g, :]).float() for g in range(N_GROUPS)]


def action_loss(model: GPT, batch: TrainBatch) -> dict[str, Tensor]:
    """Dense interleaved-AR NLL: every valid context position predicts its next frame's action,
    factorized over the four groups. One trunk forward; per-group CE at the four sites."""
    ctx = batch.context
    nxt, valid = _next_action_targets(ctx, batch.target)
    g_idx = _ar_group_idx(model, nxt)  # [B, L_ctx, n_groups] — teacher-forced tokens AND targets
    hidden = model(ctx.features, ctx.ctx_pad, g_idx)
    logits = _site_logits(model, hidden, valid.size(1))
    return group_nll(logits, g_idx, valid)


@torch.no_grad()
def decode(
    model: GPT, ctx: Context, *, temp: float = 1.0, argmax: bool = False, gen: torch.Generator | None = None
) -> Float[Tensor, "B 1 d_action"]:
    """One next-frame action per sample, in raw action ranges. Frames ``0..L-2`` use the realized
    history as teacher-forced group tokens; the last frame's four groups are decoded autoregressively
    (each conditions on the sampled earlier ones) via four sequential trunk passes."""
    feats, ctx_pad = ctx.features, ctx.ctx_pad
    acts = stack_actions(feats)  # [B, L, A_DIM]; acts[i] = a_{i-1}
    B, L, _ = acts.shape
    # Frame i's action token = a_i = acts[i+1] for i<L-1; last frame is a placeholder we overwrite.
    seq_actions = torch.cat([acts[:, 1:], torch.zeros_like(acts[:, :1])], dim=1)  # [B, L, A_DIM]
    g_idx = _ar_group_idx(model, seq_actions)  # [B, L, n_groups]
    tok = model._token_grid(feats, g_idx)  # [B, L, P, d]
    d = tok.size(-1)
    site_base = (L - 1) * P
    picks: list[Tensor] = []
    for g in range(N_GROUPS):
        hidden = model.trunk(tok.reshape(B, P * L, d), ctx_pad)
        lg = model.heads[g](hidden[:, site_base + g, :]).float()  # [B, vocab_g]
        if argmax:
            c = lg.argmax(-1)
        else:
            c = torch.multinomial(F.softmax(lg / temp, dim=-1), 1, generator=gen).squeeze(-1)
        picks.append(c)
        if g < N_GROUPS - 1:  # feed the sampled group token forward (role offset 1+g)
            tok[:, -1, 1 + g, :] = model.group_emb[g](c).to(tok.dtype) + model.role_emb.weight[1 + g].to(tok.dtype)
    idx_native = torch.stack(picks, dim=-1)[..., _NATIVE_PERM]  # [B, n_groups] AR → native
    return _dequantize(model, idx_native)[:, None, :]


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
        predict_chunk=predict_chunk, stats=stats, L_ctx=cfg.L_ctx, L_chunk=CLOSED_LOOP_L_CHUNK, s=1, d=0, device=device
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
    else — state projection, output heads, embeddings, biases — split by weight-decay eligibility.
    Exactly two LRs (``cfg.muon_lr`` / ``cfg.adam_lr``); the partition asserts full coverage so no
    parameter can silently escape an optimizer."""
    muon_params = [p for p in model.blocks.parameters() if p.ndim >= 2]
    muon_ids = {id(p) for p in muon_params}
    embed_modules = (model.cat_embeds, model.char_emb, model.stage_emb, model.group_emb, model.role_emb)
    embed_ids = {id(p) for m in embed_modules for p in m.parameters()}

    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for p in model.parameters():
        if id(p) in muon_ids:
            continue
        # AdamW: no weight decay on embeddings or 1D params (biases); decay the remaining matrices.
        (no_decay if id(p) in embed_ids or p.ndim < 2 else decay).append(p)

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
    """Per-group NLL (bits) + ``total`` bits/frame, from the per-group ``[n_valid]`` nats. Flat keys
    (``c_stick``/``triggers``/``buttons``/``main_stick``/``total``) so callers land in one W&B section."""
    out = {name: (c.mean().item() / _LN2) for name, c in comps.items()}
    out["total"] = sum(c.mean() for c in comps.values()).item() / _LN2
    return out


def objective(comps: dict[str, Tensor]) -> Tensor:
    """Interleaved-AR training objective (nats): the summed per-group mean NLL = the joint next-frame
    NLL under the AR factorization."""
    return torch.stack([c.mean() for c in comps.values()]).sum()


@torch.no_grad()
def val_metrics(model: GPT, val_cache: list[TrainBatch], cfg: TrainConfig) -> dict[str, float]:
    """Dense next-frame proper-scoring metrics over the cached val batches. Per-element tensors are
    concatenated then reduced once (exactly sample-weighted)."""
    was_training = model.training
    model.eval()
    comps_cat: dict[str, list[Tensor]] = {name: [] for name in _GROUP_NAMES}
    btn_probs: list[Tensor] = []
    btn_tgts: list[Tensor] = []
    multipress: list[Tensor] = []
    for batch in val_cache:
        ctx = batch.context
        nxt, valid = _next_action_targets(ctx, batch.target)
        g_idx = _ar_group_idx(model, nxt)
        hidden = model(ctx.features, ctx.ctx_pad, g_idx)
        logits = _site_logits(model, hidden, valid.size(1))
        for name, c in group_nll(logits, g_idx, valid).items():
            comps_cat[name].append(c)
        flat_valid = valid.reshape(-1)
        btn_logits = logits[_BUTTONS_AR_IDX].reshape(-1, scoring.N_BUTTON_COMBOS)[flat_valid]
        btn_probs.append(scoring.combo_marginal_probs(btn_logits))
        tgt_btn = nxt[..., _N_CONT:].reshape(-1, _N_BUTTONS)[flat_valid]
        btn_tgts.append(tgt_btn)
        multipress.append((tgt_btn > 0.5).sum(-1) >= 2)
    comps = {name: torch.cat(v) for name, v in comps_cat.items()}
    bits = nll_breakdown(comps)
    logloss, brier = scoring.bernoulli_scores_from_probs(torch.cat(btn_probs), torch.cat(btn_tgts))
    out = {
        "loss": bits["total"],  # next-frame total bits/frame; per-group below
        **{f"nll_{name}": bits[name] for name in _GROUP_NAMES},
        "cont_discrete_bits": (comps["c_stick"].mean() + comps["triggers"].mean() + comps["main_stick"].mean()).item()
        / _LN2,
        "btn_logloss": logloss.item(),
        "btn_brier": brier.item(),
        "btn_multipress": torch.cat(multipress).float().mean().item(),
    }
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
        tags=["gpt", "interleaved", f"d{cfg.d_model}", f"L{cfg.n_layers}"],
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
        L_chunk=CLOSED_LOOP_L_CHUNK,  # one future frame supplies the last position's next-action target
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
    val_loader = make_loader(split=cfg.val_split, num_workers=0, **loader_kwargs)

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

    def _eval_step(step: int) -> None:
        """Synchronous closed-loop eval at a training boundary: save the checkpoint, run the eval on
        the live model (training is paused, so the GPU is uncontended), log + upload .slp."""
        _save(f"step_{step:06d}.pt", step)
        tag = f"step_{step:06d}"
        print(f"[t+{time.monotonic() - run_t0:.0f}s] step {step}: running synchronous eval…", flush=True)
        _log_eval(step, _eval_and_upload(tag))

    model.train()
    it = iter(train_loader)
    run_t0 = time.monotonic()
    for step in range(start_step, cfg.max_steps):
        with profile("step") as sw:
            opt.zero_grad()
            comps_acc: dict[str, list[Tensor]] = {}
            for _ in range(cfg.grad_accum_steps):
                try:
                    batch = next(it).to(DEVICE)
                except StopIteration:
                    it = iter(train_loader)
                    batch = next(it).to(DEVICE)
                with autocast:
                    comps = action_loss(model, batch)
                    loss = objective(comps) / cfg.grad_accum_steps
                loss.backward()
                for k, v in comps.items():
                    comps_acc.setdefault(k, []).append(v.detach())
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))  # measure only
            opt.step()
            sched.step()
            if DEVICE == "cuda":
                torch.cuda.synchronize()
        comps_cat = {k: torch.cat(v) for k, v in comps_acc.items()}
        bits = nll_breakdown(comps_cat)
        sps = cfg.batch_size * cfg.grad_accum_steps / sw.elapsed
        samples = (step + 1) * cfg.batch_size * cfg.grad_accum_steps
        log = {
            "global_step": step,
            "samples": samples,
            "tokens": samples * cfg.L_ctx * P,
            "train/loss": bits["total"],
            **{f"train/nll_{name}": bits[name] for name in _GROUP_NAMES},
            "lr/muon": next(g["lr"] for g in opt.param_groups if g["use_muon"]),
            "lr/adam": next(g["lr"] for g in opt.param_groups if not g["use_muon"]),
            "train/gnorm": grad_norm.item(),
            "throughput/step_s": sw.elapsed,
            "throughput/samples_per_s": sps,
        }
        if step < 20 or step % 50 == 0:
            print(
                f"[t+{time.monotonic() - run_t0:.0f}s] step {step}: loss {bits['total']:.4f} "
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
        if cfg.eval_every > 0 and step > 0 and step % cfg.eval_every == 0:
            _eval_step(step)

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
    past checkpoints still load. Model-identity fields (``d_model``, …) are
    unaffected — they're always present and reconstruct exactly."""
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


def eval_ckpt(ckpt_path: str, *, decode_temp: float | None = None) -> None:
    """Load a checkpoint and run the prior-distribution vs-CPU sweep, printing the
    pooled metrics. ``decode_temp`` overrides the trained cfg for this eval only
    (test-time temperature sweep)."""
    model, cfg, stats, state = _load_ckpt(ckpt_path)
    temp = cfg.decode_temp if decode_temp is None else decode_temp
    print(f"[eval] loaded {ckpt_path}  step={state['step']}  device={DEVICE}  temp={temp}", flush=True)
    replay_dir = Path(ckpt_path).resolve().parent / "eval_replays"
    replay_dir.mkdir(parents=True, exist_ok=True)
    session_cfg = default_session_cfg(replay_dir, instant_match_restart=True)
    n = _eval_max_parallel(cfg)

    def policy_factory() -> RecedingHorizon:
        return make_policy(model, stats, cfg, decode_temp=decode_temp)

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
        # Operational knobs (host scaling + eval/val cadence) follow the CURRENT code so a resume can
        # retune them; the model-identity knobs MUST come from the checkpoint so a resume can't silently
        # change them. Eval cadence is operational (it never touches the weights), so refresh it too —
        # this is what lets a resume turn closed-loop eval back on / change its parallelism.
        d = TrainConfig()
        cfg = replace(
            _cfg_from_state(state["cfg"]),
            num_workers=d.num_workers,
            prefetch_factor=d.prefetch_factor,
            eval_every=d.eval_every,
            eval_max_frames=d.eval_max_frames,
            eval_parallel_per_cpu=d.eval_parallel_per_cpu,
            val_every=d.val_every,
        )
        stats = load_consolidated_stats(Path(cfg.data_root) / "stats.json")
        train(cfg, stats, resume_run=args.resume, resume_state=state)
        return
    cfg = args.cfg
    stats = load_consolidated_stats(Path(cfg.data_root) / "stats.json")
    auto_comment = f"gpt-il-{cfg.max_steps // 1000}k-b{cfg.batch_size}x{cfg.grad_accum_steps}"
    train(cfg, stats, comment=args.comment or auto_comment)


if __name__ == "__main__":
    main(tyro.cli(Args))
